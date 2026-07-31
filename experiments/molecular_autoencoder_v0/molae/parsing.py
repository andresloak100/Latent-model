"""Parse and clean protein structures from mmCIF/PDB into atom arrays.

Design decisions (recorded per structure so filtering is auditable):
  * First model only (matters for NMR ensembles).
  * By default a single polymer (peptide) chain is selected and the others
    are dropped and recorded. ``multi_chain=True`` keeps every peptide
    chain meeting the length floor (protein-protein complexes); res_pos is
    then GLOBAL across kept chains, which the direct decoder requires --
    per-chain numbering would alias chain B's residue 0 onto chain A's.
  * Waters, ligands, ions, and hydrogens are removed.
  * Alternate conformations are collapsed to a single conformer (gemmi's
    ``remove_alternative_conformations`` keeps one altloc per atom).
  * Only the 20 standard amino acids are kept; non-standard residues are
    dropped and recorded (this prototype does not model modified residues).
  * Only heavy atoms whose names are in the fixed vocabulary are kept.
  * Missing residues (gaps in author numbering) and residues with an
    incomplete backbone are counted and recorded, not silently ignored.

Topology (bonds) is *perceived* from the deposited coordinates using
covalent radii, restricted to intra-residue and consecutive-residue pairs so
no spurious long-range bonds are created. This keeps the pipeline free of any
external monomer library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gemmi

from . import constants as C


class AllResiduesFiltered(ValueError):
    """Raised when a peptide chain exists but nothing survives filtering."""


@dataclass
class ParsedStructure:
    pdb_id: str
    chain_id: str
    # Per-atom arrays (length N).
    element_idx: np.ndarray      # int64  (N,)
    residue_idx: np.ndarray      # int64  (N,)  standard-AA vocab index
    atom_name_idx: np.ndarray    # int64  (N,)
    slot_idx: np.ndarray         # int64  (N,)  decoder slot within the group
    res_pos: np.ndarray          # int64  (N,)  GLOBAL group index (residues, then ligands)
    chain_idx: np.ndarray        # int64  (N,)  0-based index of the atom's chain
    res_seq: np.ndarray          # int64  (N,)  author seq id
    coords: np.ndarray           # float32 (N, 3)
    element_symbol: list         # list[str] length N (for bond perception)
    atom_name: list              # list[str] length N
    bonds: np.ndarray            # int64 (M, 2) atom-index pairs
    sequence: str                # one-letter chain sequence
    record: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_residues(self) -> int:
        """PROTEIN residues. res_pos also numbers ligand groups, so deriving
        this from res_pos.max() would inflate it and break the size filters."""
        if "n_residues" in self.record:
            return int(self.record["n_residues"])
        return int(self.res_pos.max()) + 1 if self.n_atoms else 0

    @property
    def n_groups(self) -> int:
        """Decoder groups = protein residues + ligand chunks."""
        return int(self.res_pos.max()) + 1 if self.n_atoms else 0


def perceive_bonds(
    coords: np.ndarray,
    element_symbol: list,
    res_pos: np.ndarray,
    chain_idx: Optional[np.ndarray] = None,
    unrestricted_chains=(),
) -> np.ndarray:
    """Return (M, 2) array of bonded atom-index pairs (i < j).

    A pair is bonded when it lies within the same residue or in consecutive
    residues *and* its distance is below the sum of covalent radii plus a
    tolerance. Restricting to (near-)neighbour residues avoids spurious bonds
    across chain gaps or between packed side chains.

    ``unrestricted_chains`` lifts the residue-adjacency restriction for the
    listed chains, allowing any intra-chain pair within covalent range. A
    polymer is a chain of small residues, so adjacency is a good proxy for
    "could be bonded"; an arbitrary molecule is NOT, and one large enough to be
    split across several groups would otherwise silently lose every bond
    between non-adjacent chunks. Non-polymer chains hold exactly one molecule,
    so lifting the restriction there cannot connect separate species.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    radii = np.array(
        [C.COVALENT_RADII.get(e.upper(), C.DEFAULT_COVALENT_RADIUS) for e in element_symbol],
        dtype=np.float32,
    )
    # Candidate pairs: only atoms in the same or consecutive residues.
    ia, ib = np.triu_indices(n, k=1)
    dres = np.abs(res_pos[ia] - res_pos[ib])
    keep = dres <= 1
    if chain_idx is not None:
        if len(unrestricted_chains):
            free = np.isin(chain_idx, np.asarray(list(unrestricted_chains)))
            keep = keep | (free[ia] & free[ib])
        # res_pos is GLOBAL across chains, so the last residue of chain k and
        # the first of chain k+1 are numerically adjacent. Without this guard
        # they would be bonded across a chain break.
        keep = keep & (chain_idx[ia] == chain_idx[ib])
    ia, ib = ia[keep], ib[keep]
    if ia.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    dist = np.linalg.norm(coords[ia] - coords[ib], axis=1)
    cutoff = radii[ia] + radii[ib] + C.BOND_TOLERANCE
    bonded = dist < cutoff
    pairs = np.stack([ia[bonded], ib[bonded]], axis=1).astype(np.int64)
    if pairs.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    # Sort for determinism.
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _count_raw_stats(structure: gemmi.Structure) -> dict:
    """Count waters/hetero/altloc/hydrogen atoms in model 0 before cleaning."""
    n_models = len(structure)
    model = structure[0]
    n_water = n_hetero = n_altloc = n_hydrogen = 0
    for chain in model:
        for res in chain:
            is_water = res.is_water()
            for atom in res:
                if atom.element == gemmi.Element("H"):
                    n_hydrogen += 1
                if atom.has_altloc():
                    n_altloc += 1
                if is_water:
                    n_water += 1
                elif res.het_flag == "H":
                    n_hetero += 1
    return {
        "n_models": n_models,
        "n_water_atoms": n_water,
        "n_hetero_atoms": n_hetero,
        "n_altloc_atoms": n_altloc,
        "n_hydrogen_atoms": n_hydrogen,
    }


def parse_structure(
    path: str,
    pdb_id: Optional[str] = None,
    min_chain_len: int = 8,
    model_index: int = 0,
    multi_chain: bool = False,
    keep_ligands: bool = False,
    exclude_ligands=C.CRYSTALLIZATION_ADDITIVES,
    min_ligand_atoms: int = 1,
    keep_modified_residues: bool = False,
) -> Optional[ParsedStructure]:
    """Parse one structure file into a cleaned :class:`ParsedStructure`.

    Returns ``None`` only if no usable peptide chain is found. All filtering
    counts are stored on ``.record``.

    ``model_index`` selects which deposited model to keep (default 0, the
    historical behaviour). NMR entries deposit many models of the *same*
    molecule, so iterating this over ``record["n_models"]`` yields a
    conformational ensemble with identical atom composition and no MD compute
    — which is the distribution the latent-diffusion stage will actually run
    on. Raises ``IndexError`` for an out-of-range model.

    ``keep_ligands=True`` additionally keeps non-water hetero groups (ligands,
    cofactors, ions). A ligand has no residue index and no fixed atom-name
    vocabulary, so its atoms are addressed by ORDINAL within their group via
    ``slot_idx``; each ligand becomes one or more groups of at most
    ``C.N_SLOTS`` atoms, numbered after the protein residues. Ligand atoms get
    ``residue_idx = LIG`` and ``atom_name_idx = UNK``. Waters are always
    dropped, as are ``exclude_ligands`` (crystallisation additives by default).
    """
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if pdb_id is None:
        pdb_id = st.name or "UNKNOWN"

    raw = _count_raw_stats(st)

    if not 0 <= model_index < len(st):
        raise IndexError(
            f"{pdb_id}: model_index {model_index} out of range (n_models={len(st)})")
    # Keep exactly one model (NMR ensembles deposit many).
    for i in range(len(st) - 1, -1, -1):
        if i != model_index:
            del st[i]

    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if keep_ligands:
        # Waters are always noise; ligands are the point of this mode.
        st.remove_waters()
    else:
        st.remove_ligands_and_waters()
    st.remove_empty_chains()

    model = st[0]

    # Enumerate all chains (peptide and non-peptide) so the audit trail records
    # everything that was present and which chains were dropped.
    chain_lengths = {}          # peptide chains only (selection candidates)
    all_chains = {}             # name -> {"type", "length"} for every chain
    for chain in model:
        poly = chain.get_polymer()
        ptype = poly.check_polymer_type()
        is_peptide = ptype in (
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        )
        all_chains[chain.name] = {"polymer_type": str(ptype), "length": len(poly)}
        if is_peptide:
            chain_lengths[chain.name] = len(poly)

    if not chain_lengths:
        return None

    # Which chains to keep. Single-chain (default) preserves the historical
    # behaviour exactly; multi_chain keeps every peptide chain meeting the
    # minimum length, which is what protein-protein complexes need.
    eligible = [c.name for c in model
                if c.name in chain_lengths and chain_lengths[c.name] >= min_chain_len]
    if multi_chain:
        kept_chains = eligible or [max(chain_lengths, key=chain_lengths.get)]
    else:
        kept_chains = [eligible[0]] if eligible else [max(chain_lengths, key=chain_lengths.get)]
    selected = kept_chains[0] if len(kept_chains) == 1 else ",".join(kept_chains)

    element_idx, residue_idx, atom_name_idx, slot_idx = [], [], [], []
    res_pos, res_seq, coords, element_symbol, atom_name = [], [], [], [], []
    chain_idx_list, seq_one, per_chain_seq = [], [], []
    dropped_nonstandard = {}
    dropped_unknown_atoms = 0
    dropped_oversized_residues = 0
    incomplete_backbone = 0
    missing_residue_gaps = 0
    n_modified = 0

    # Every residue belonging to ANY polymer, so the hetero pass cannot re-add
    # one as a free-floating "ligand". Modified residues (MSE, HYP, D-amino
    # acids) carry het_flag "H" *while sitting inside the backbone*, so without
    # this they get a synthetic chain_idx and lose both peptide bonds -- the
    # chain guard in perceive_bonds then correctly, and uselessly, refuses to
    # bond them to the protein they are part of.
    polymer_residues = set()
    for chain in model:
        for res in chain.get_polymer():
            polymer_residues.add((chain.name, res.seqid.num, res.name))

    # res_pos is GLOBAL across kept chains. That matters for more than
    # bookkeeping: the direct decoder indexes its output slot bank by
    # (res_pos, atom_name), so per-chain numbering would make chain B's
    # residue 0 alias onto chain A's residue 0 and silently share coordinates.
    pos = 0
    for ci, cname in enumerate(kept_chains):
        poly = model[cname].get_polymer()
        prev_seqid = None
        chain_seq = []
        for res in poly:
            resname = res.name
            seqid = res.seqid.num
            # Count numbering gaps (missing/unmodelled residues) within a chain.
            if prev_seqid is not None and seqid - prev_seqid > 1:
                missing_residue_gaps += (seqid - prev_seqid - 1)
            prev_seqid = seqid

            modified = resname not in C.STANDARD_AA_SET
            if modified:
                dropped_nonstandard[resname] = dropped_nonstandard.get(resname, 0) + 1
                if not keep_modified_residues:
                    continue
                n_modified += 1

            kept_names = set()
            res_atoms = []
            for atom in res:
                aname = atom.name
                # A modified residue's extra atoms (SME's methyl, HYP's
                # hydroxyl) are not in the protein vocabulary; dropping them
                # would silently truncate the residue, so it takes ordinal
                # slots instead and keeps every atom.
                if aname not in C.ATOM_NAME_TO_IDX and not modified:
                    dropped_unknown_atoms += 1
                    continue
                esym = atom.element.name.upper()
                res_atoms.append((aname, esym, atom.pos))
                kept_names.add(aname)

            if not res_atoms:
                continue
            if not set(C.BACKBONE_ATOMS).issubset(kept_names):
                incomplete_backbone += 1
            if modified and len(res_atoms) > C.N_SLOTS:
                # Cannot address more atoms than the group has slots.
                dropped_oversized_residues += 1
                continue

            # Named slots need every atom to be in the vocabulary AND unique;
            # decided per RESIDUE so the two schemes never mix inside a group.
            named = (not modified
                     or (all(a in C.ATOM_NAME_TO_IDX for a, _, _ in res_atoms)
                         and len({a for a, _, _ in res_atoms}) == len(res_atoms)))
            for k, (aname, esym, p) in enumerate(res_atoms):
                element_idx.append(C.ELEMENT_TO_IDX.get(esym, C.UNK_ELEMENT_IDX))
                residue_idx.append(C.RESIDUE_TO_IDX.get(resname, C.UNK_RESIDUE_IDX))
                atom_name_idx.append(C.ATOM_NAME_TO_IDX.get(aname, C.UNK_ATOM_IDX))
                # For a standard residue the decoder slot IS the atom name, so
                # the gather is unchanged from the protein-only pipeline.
                slot_idx.append(C.ATOM_NAME_TO_IDX[aname] if named else k)
                res_pos.append(pos)
                res_seq.append(seqid)
                # The polymer's chain, NOT a synthetic one: a modified residue
                # is covalently in the backbone and must keep its peptide bonds.
                chain_idx_list.append(ci)
                coords.append([p.x, p.y, p.z])
                element_symbol.append(esym)
                atom_name.append(aname)
            one = C.THREE_TO_ONE.get(resname, "X")
            seq_one.append(one)
            chain_seq.append(one)
            pos += 1
        per_chain_seq.append("".join(chain_seq))

    # --- Ligands / cofactors / ions ---------------------------------------
    # Each kept hetero group becomes one or more decoder groups of at most
    # C.N_SLOTS atoms, numbered after the protein residues. Every ligand gets
    # its own chain_idx so perceive_bonds cannot invent a covalent bond to the
    # protein's last residue or to a neighbouring ligand -- res_pos is global
    # and therefore numerically adjacent across that boundary.
    n_protein_residues = pos          # before ligand groups extend res_pos
    ligands_kept, ligands_dropped = [], {}
    n_ligand_atoms = 0
    next_chain = len(kept_chains)
    if keep_ligands:
        for chain in model:
            for res in chain:
                if res.is_water() or res.het_flag != "H":
                    continue
                if (chain.name, res.seqid.num, res.name) in polymer_residues:
                    continue          # a modified residue, handled in-chain above
                if res.name in exclude_ligands:
                    ligands_dropped[res.name] = ligands_dropped.get(res.name, 0) + 1
                    continue
                lig_atoms = [(a.name, a.element.name.upper(), a.pos) for a in res]
                if len(lig_atoms) < min_ligand_atoms:
                    ligands_dropped[res.name] = ligands_dropped.get(res.name, 0) + 1
                    continue
                # A hetero group can still be a *known* chemical species -- a
                # bound amino acid or peptide ligand (1PIN deposits ALA and PRO
                # this way). Labelling those "LIG/UNK" would throw away
                # chemistry we have a vocabulary for. Decide per GROUP, never
                # per atom: mixing name-derived and ordinal slots inside one
                # group could collide two atoms onto the same decoder slot.
                named = (res.name in C.STANDARD_AA_SET
                         and all(a in C.ATOM_NAME_TO_IDX for a, _, _ in lig_atoms)
                         and len({a for a, _, _ in lig_atoms}) == len(lig_atoms))
                for k, (aname, esym, p) in enumerate(lig_atoms):
                    # Ordinal addressing: slot k of the group, chunking every
                    # N_SLOTS atoms into the next group so a large ligand can
                    # never overflow into a neighbouring group's slot bank.
                    if not named and k and k % C.N_SLOTS == 0:
                        pos += 1
                    element_idx.append(C.ELEMENT_TO_IDX.get(esym, C.UNK_ELEMENT_IDX))
                    residue_idx.append(C.RESIDUE_TO_IDX[res.name] if named
                                       else C.LIGAND_RESIDUE_IDX)
                    atom_name_idx.append(C.ATOM_NAME_TO_IDX[aname] if named
                                         else C.UNK_ATOM_IDX)
                    slot_idx.append(C.ATOM_NAME_TO_IDX[aname] if named
                                    else k % C.N_SLOTS)
                    res_pos.append(pos)
                    res_seq.append(res.seqid.num)
                    chain_idx_list.append(next_chain)
                    coords.append([p.x, p.y, p.z])
                    element_symbol.append(esym)
                    atom_name.append(aname)
                pos += 1
                next_chain += 1
                n_ligand_atoms += len(lig_atoms)
                ligands_kept.append({"name": res.name, "n_atoms": len(lig_atoms),
                                     "chain": chain.name, "seq_id": res.seqid.num})

    if not coords:
        # A peptide chain existed but every residue/atom was filtered out
        # (non-standard residues, unknown atom names). Distinguish this from
        # the no-peptide-chain case so the manifest is accurate.
        raise AllResiduesFiltered(
            f"{pdb_id}: peptide chain(s) {selected} had no usable standard-AA heavy atoms"
        )

    coords = np.asarray(coords, dtype=np.float32)
    res_pos = np.asarray(res_pos, dtype=np.int64)
    chain_idx = np.asarray(chain_idx_list, dtype=np.int64)
    # Ligand chains hold one molecule each, so intra-chain bonding is not
    # restricted to adjacent groups -- a compound larger than N_SLOTS spans
    # several groups and its topology must survive that split intact.
    ligand_chains = tuple(range(len(kept_chains), next_chain))
    bonds = perceive_bonds(coords, element_symbol, res_pos, chain_idx,
                           unrestricted_chains=ligand_chains)

    record = {
        "pdb_id": pdb_id,
        "chain_id": selected,
        "model_index": int(model_index),
        "n_atoms": int(coords.shape[0]),
        # n_residues counts PROTEIN residues only. res_pos also numbers ligand
        # groups, so max(res_pos)+1 would silently inflate this and corrupt the
        # size filters in prepare_dataset.
        "n_residues": int(n_protein_residues),
        "n_groups": int(res_pos.max()) + 1,
        "n_ligand_atoms": int(n_ligand_atoms),
        "ligands_kept": ligands_kept,
        "ligands_dropped": ligands_dropped,
        "chains_present": dict(chain_lengths),
        "all_chains": all_chains,
        "dropped_non_peptide_chains": [n for n in all_chains if n not in chain_lengths],
        "multi_chain_entry": len(chain_lengths) > 1,
        "kept_chains": list(kept_chains),
        "n_kept_chains": len(kept_chains),
        "per_chain_sequence": list(per_chain_seq),
        "dropped_nonstandard_residues": dropped_nonstandard,
        "n_modified_residues_kept": int(n_modified),
        "dropped_oversized_residues": int(dropped_oversized_residues),
        "dropped_unknown_atoms": dropped_unknown_atoms,
        "residues_incomplete_backbone": incomplete_backbone,
        "missing_residue_gaps": missing_residue_gaps,
        "n_bonds": int(bonds.shape[0]),
        **raw,
    }

    return ParsedStructure(
        pdb_id=pdb_id,
        chain_id=selected,
        element_idx=np.asarray(element_idx, dtype=np.int64),
        residue_idx=np.asarray(residue_idx, dtype=np.int64),
        atom_name_idx=np.asarray(atom_name_idx, dtype=np.int64),
        slot_idx=np.asarray(slot_idx, dtype=np.int64),
        res_pos=res_pos,
        chain_idx=chain_idx,
        res_seq=np.asarray(res_seq, dtype=np.int64),
        coords=coords,
        element_symbol=element_symbol,
        atom_name=atom_name,
        bonds=bonds,
        sequence="".join(seq_one),
        record=record,
    )


def to_npz_dict(ps: ParsedStructure) -> dict:
    """Serialisable dict for ``np.savez`` (arrays + json-friendly metadata)."""
    return {
        "element_idx": ps.element_idx,
        "residue_idx": ps.residue_idx,
        "atom_name_idx": ps.atom_name_idx,
        "slot_idx": ps.slot_idx,
        "res_pos": ps.res_pos,
        "chain_idx": ps.chain_idx,
        "res_seq": ps.res_seq,
        "coords": ps.coords,
        "bonds": ps.bonds,
        "atom_name": np.array(ps.atom_name),
        "element_symbol": np.array(ps.element_symbol),
        "pdb_id": ps.pdb_id,
        "chain_id": ps.chain_id,
        "sequence": ps.sequence,
    }
