"""Parse and clean protein structures from mmCIF/PDB into atom arrays.

Design decisions (recorded per structure so filtering is auditable):
  * First model only (matters for NMR ensembles).
  * A single polymer (peptide) chain is selected; other chains are dropped
    and recorded.
  * Waters, ligands, ions, and hydrogens are removed.
  * Alternate conformations are collapsed (highest occupancy kept by gemmi).
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


@dataclass
class ParsedStructure:
    pdb_id: str
    chain_id: str
    # Per-atom arrays (length N).
    element_idx: np.ndarray      # int64  (N,)
    residue_idx: np.ndarray      # int64  (N,)  standard-AA vocab index
    atom_name_idx: np.ndarray    # int64  (N,)
    res_pos: np.ndarray          # int64  (N,)  0..L-1 position within chain
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
        return int(self.res_pos.max()) + 1 if self.n_atoms else 0


def perceive_bonds(
    coords: np.ndarray,
    element_symbol: list,
    res_pos: np.ndarray,
) -> np.ndarray:
    """Return (M, 2) array of bonded atom-index pairs (i < j).

    A pair is bonded when it lies within the same residue or in consecutive
    residues *and* its distance is below the sum of covalent radii plus a
    tolerance. Restricting to (near-)neighbour residues avoids spurious bonds
    across chain gaps or between packed side chains.
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
) -> Optional[ParsedStructure]:
    """Parse one structure file into a cleaned :class:`ParsedStructure`.

    Returns ``None`` only if no usable peptide chain is found. All filtering
    counts are stored on ``.record``.
    """
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if pdb_id is None:
        pdb_id = st.name or "UNKNOWN"

    raw = _count_raw_stats(st)

    # Keep only the first model (NMR ensembles -> model 1).
    while len(st) > 1:
        del st[len(st) - 1]

    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_empty_chains()

    model = st[0]

    # Enumerate peptide chains and their lengths.
    chain_lengths = {}
    for chain in model:
        poly = chain.get_polymer()
        ptype = poly.check_polymer_type()
        is_peptide = ptype in (
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        )
        if is_peptide:
            chain_lengths[chain.name] = len(poly)

    if not chain_lengths:
        return None

    # Select the first peptide chain meeting the minimum length.
    selected = None
    for chain in model:
        name = chain.name
        if name in chain_lengths and chain_lengths[name] >= min_chain_len:
            selected = name
            break
    if selected is None:  # fall back to the longest peptide chain
        selected = max(chain_lengths, key=chain_lengths.get)

    chain = model[selected]
    poly = chain.get_polymer()

    element_idx, residue_idx, atom_name_idx = [], [], []
    res_pos, res_seq, coords, element_symbol, atom_name = [], [], [], [], []
    seq_one = []
    dropped_nonstandard = {}
    dropped_unknown_atoms = 0
    incomplete_backbone = 0
    prev_seqid = None
    missing_residue_gaps = 0

    pos = 0
    for res in poly:
        resname = res.name
        seqid = res.seqid.num
        # Count numbering gaps (missing/unmodelled residues) within the chain.
        if prev_seqid is not None and seqid - prev_seqid > 1:
            missing_residue_gaps += (seqid - prev_seqid - 1)
        prev_seqid = seqid

        if resname not in C.STANDARD_AA_SET:
            dropped_nonstandard[resname] = dropped_nonstandard.get(resname, 0) + 1
            continue

        kept_names = set()
        res_atoms = []
        for atom in res:
            aname = atom.name
            if aname not in C.ATOM_NAME_TO_IDX:
                dropped_unknown_atoms += 1
                continue
            esym = atom.element.name.upper()
            res_atoms.append((aname, esym, atom.pos))
            kept_names.add(aname)

        if not res_atoms:
            continue
        if not set(C.BACKBONE_ATOMS).issubset(kept_names):
            incomplete_backbone += 1

        for aname, esym, p in res_atoms:
            element_idx.append(C.ELEMENT_TO_IDX.get(esym, C.UNK_ELEMENT_IDX))
            residue_idx.append(C.RESIDUE_TO_IDX[resname])
            atom_name_idx.append(C.ATOM_NAME_TO_IDX[aname])
            res_pos.append(pos)
            res_seq.append(seqid)
            coords.append([p.x, p.y, p.z])
            element_symbol.append(esym)
            atom_name.append(aname)
        seq_one.append(C.THREE_TO_ONE[resname])
        pos += 1

    if not coords:
        return None

    coords = np.asarray(coords, dtype=np.float32)
    res_pos = np.asarray(res_pos, dtype=np.int64)
    bonds = perceive_bonds(coords, element_symbol, res_pos)

    record = {
        "pdb_id": pdb_id,
        "chain_id": selected,
        "n_atoms": int(coords.shape[0]),
        "n_residues": int(res_pos.max()) + 1,
        "chains_present": dict(chain_lengths),
        "multi_chain_entry": len(chain_lengths) > 1,
        "dropped_nonstandard_residues": dropped_nonstandard,
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
        res_pos=res_pos,
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
        "res_pos": ps.res_pos,
        "res_seq": ps.res_seq,
        "coords": ps.coords,
        "bonds": ps.bonds,
        "atom_name": np.array(ps.atom_name),
        "element_symbol": np.array(ps.element_symbol),
        "pdb_id": ps.pdb_id,
        "chain_id": ps.chain_id,
        "sequence": ps.sequence,
    }
