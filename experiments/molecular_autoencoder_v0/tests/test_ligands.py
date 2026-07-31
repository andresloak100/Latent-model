"""Ligand / non-polymer support.

The direct decoder addresses each atom as ``res_pos * N_SLOTS + slot_idx``. For
a protein residue the slot IS the atom name, so everything here must leave the
protein-only path bit-identical -- that is the first and most important test.
A ligand has neither a residue index nor a vocabulary atom name, so its atoms
are addressed by ORDINAL within their group instead, which is what lets them
share the decoder at all.
"""

import numpy as np
import torch

from molae import constants as C
from molae.parsing import parse_structure, perceive_bonds
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import ModelConfig
from molae.model_direct import DirectAutoencoder
from molae.synthetic import make_synthetic_ala


# --- fixture: a tiny PDB with a peptide chain and a hetero group -----------

def _atom_line(record, serial, name, resname, chain, resseq, xyz, elem, occ=1.00):
    """A column-exact PDB ATOM/HETATM record.

    The columns are not negotiable: 13-16 atom name, 17 altLoc, 18-20 resName,
    22 chainID, 23-26 resSeq, 31-54 xyz, 77-78 element. Getting 17 wrong makes
    gemmi read the whole file as alternate conformations.
    """
    x, y, z = xyz
    return (f"{record:<6s}{serial:5d} {name:<4s}{'':1s}{resname:>3s} "
            f"{chain:1s}{resseq:4d}{'':4s}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{0.00:6.2f}{'':10s}{elem:>2s}")


def _pdb_text(n_res=10, ligand_atoms=5, ligand_name="LIG", water=True,
              ligand_atom_names=None):
    """Minimal but gemmi-parseable PDB: one poly-ALA chain plus a HETATM group."""
    lines, serial = [], 1
    for r in range(1, n_res + 1):
        base = np.array([3.8 * r, 0.0, 0.0])
        for aname, elem, off in [("N", "N", (0.0, 0.0, 0.0)), ("CA", "C", (1.4, 0.2, 0.0)),
                                 ("C", "C", (2.5, -0.4, 0.0)), ("O", "O", (2.6, -1.6, 0.0)),
                                 ("CB", "C", (1.5, 1.7, 0.0))]:
            # Element symbol is right-justified in cols 13-14, so a short name
            # is written as " N  " / " CA ".
            lines.append(_atom_line("ATOM", serial, f" {aname:<3s}"[:4], "ALA",
                                    "A", r, base + np.array(off), elem))
            serial += 1
    names = ligand_atom_names or [f"C{k + 1}" for k in range(ligand_atoms)]
    for k, nm in enumerate(names):
        lines.append(_atom_line("HETATM", serial, f" {nm:<3s}"[:4],
                                ligand_name, "B", 501,
                                (5.0 + 1.4 * k, 8.0, 0.0), nm[0]))
        serial += 1
    if water:
        lines.append(_atom_line("HETATM", serial, " O  ", "HOH", "B", 601,
                                (20.0, 20.0, 0.0), "O"))
    lines.append("END")
    return "\n".join(lines) + "\n"


def _write(tmp_path, **kw):
    p = tmp_path / "test.pdb"
    p.write_text(_pdb_text(**kw))
    return str(p)


# --- the protein path must not move ---------------------------------------

def test_protein_slot_idx_is_atom_name_idx(tmp_path):
    ps = parse_structure(_write(tmp_path), pdb_id="TEST", keep_ligands=False)
    assert np.array_equal(ps.slot_idx, ps.atom_name_idx)


def test_gather_identical_to_legacy_on_protein_only():
    """New group/slot gather == old res_pos*N_ATOM_NAMES+atom_name_idx."""
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(n_res=6)),
                        sample_from_arrays(make_synthetic_ala(n_res=4))])
    legacy = batch["res_pos"] * C.N_ATOM_NAMES + batch["atom_name_idx"]
    new = batch["res_pos"] * C.N_SLOTS + batch["slot_idx"]
    assert torch.equal(legacy, new)


def test_npz_without_slot_idx_still_loads():
    """Every .npz written before ligand support lacks slot_idx."""
    d = dict(make_synthetic_ala(n_res=5))
    d.pop("slot_idx", None)
    s = sample_from_arrays(d)
    assert torch.equal(s["slot_idx"], s["atom_name_idx"])


# --- ligands ---------------------------------------------------------------

def test_ligands_dropped_by_default(tmp_path):
    ps = parse_structure(_write(tmp_path), pdb_id="TEST")
    assert not (ps.residue_idx == C.LIGAND_RESIDUE_IDX).any()
    assert ps.record["n_ligand_atoms"] == 0


def test_ligand_kept_with_ordinal_slots(tmp_path):
    ps = parse_structure(_write(tmp_path, ligand_atoms=5), pdb_id="TEST",
                         keep_ligands=True)
    lig = ps.residue_idx == C.LIGAND_RESIDUE_IDX
    assert lig.sum() == 5
    # Ligand atoms carry no vocabulary name, and are addressed by ordinal.
    assert (ps.atom_name_idx[lig] == C.UNK_ATOM_IDX).all()
    assert np.array_equal(np.sort(ps.slot_idx[lig]), np.arange(5))
    # One group, numbered after the protein residues.
    assert len(set(ps.res_pos[lig].tolist())) == 1
    assert ps.res_pos[lig][0] == ps.record["n_residues"]


def test_water_always_dropped(tmp_path):
    ps = parse_structure(_write(tmp_path, water=True), pdb_id="TEST",
                         keep_ligands=True)
    assert "O" not in [n for n, r in zip(ps.atom_name, ps.residue_idx)
                       if r == C.LIGAND_RESIDUE_IDX and n == "O"]
    assert ps.record["n_ligand_atoms"] == 5      # the water is not counted


def test_crystallization_additive_excluded(tmp_path):
    ps = parse_structure(_write(tmp_path, ligand_name="GOL"), pdb_id="TEST",
                         keep_ligands=True)
    assert ps.record["n_ligand_atoms"] == 0
    assert "GOL" in ps.record["ligands_dropped"]


def test_peg_oligomer_excluded(tmp_path):
    """Real entries deposit these (1PIN keeps a 17-atom 1PG); they are pure
    crystallisation agents, not chemistry we want to learn."""
    ps = parse_structure(_write(tmp_path, ligand_name="1PG"), pdb_id="TEST",
                         keep_ligands=True)
    assert ps.record["n_ligand_atoms"] == 0
    assert "1PG" in ps.record["ligands_dropped"]


def test_bound_amino_acid_keeps_its_chemistry(tmp_path):
    """A hetero group can still be a species we have a vocabulary for -- 1PIN
    deposits bound ALA/PRO this way. Labelling those LIG/UNK would discard
    chemistry for no reason."""
    ps = parse_structure(
        _write(tmp_path, ligand_name="ALA",
               ligand_atom_names=["N", "CA", "C", "O", "CB"]),
        pdb_id="TEST", keep_ligands=True)
    het = ps.res_pos >= ps.record["n_residues"]
    assert het.sum() == 5
    # Real residue identity and real atom names, not LIG/UNK.
    assert (ps.residue_idx[het] == C.RESIDUE_TO_IDX["ALA"]).all()
    assert (ps.atom_name_idx[het] != C.UNK_ATOM_IDX).all()
    assert np.array_equal(ps.slot_idx[het], ps.atom_name_idx[het])


def test_named_and_ordinal_slots_never_collide(tmp_path):
    """The named/ordinal choice is per GROUP; mixing them inside one group
    could land two atoms on the same decoder slot."""
    for name, anames in [("ALA", ["N", "CA", "C", "O", "CB"]),   # named path
                         ("LIG", None)]:                          # ordinal path
        ps = parse_structure(_write(tmp_path, ligand_name=name,
                                    ligand_atom_names=anames),
                             pdb_id="TEST", keep_ligands=True)
        addr = ps.res_pos * C.N_SLOTS + ps.slot_idx
        assert len(set(addr.tolist())) == len(addr), name


def test_duplicate_atom_names_do_not_alias_a_slot(tmp_path):
    """Two atoms sharing a name in one group would land on the same decoder
    slot. gemmi already collapses such duplicates on read, so this asserts the
    end-to-end property (addresses stay unique) rather than the code path --
    the uniqueness guard in parse_structure is belt-and-braces behind gemmi.
    """
    ps = parse_structure(
        _write(tmp_path, ligand_name="ALA", ligand_atom_names=["CB", "CB", "CA"]),
        pdb_id="TEST", keep_ligands=True)
    het = ps.res_pos >= ps.record["n_residues"]
    names = [n for n, h in zip(ps.atom_name, het) if h]
    assert len(names) == len(set(names)), "duplicate atom name survived"
    addr = ps.res_pos * C.N_SLOTS + ps.slot_idx
    assert len(set(addr.tolist())) == len(addr)


def test_hetero_group_with_unknown_atom_names_uses_ordinal(tmp_path):
    """The named path requires EVERY atom to be in the vocabulary; one unknown
    name sends the whole group to ordinal slots."""
    ps = parse_structure(
        _write(tmp_path, ligand_name="ALA", ligand_atom_names=["N", "CA", "ZZ9"]),
        pdb_id="TEST", keep_ligands=True)
    het = ps.res_pos >= ps.record["n_residues"]
    assert (ps.residue_idx[het] == C.LIGAND_RESIDUE_IDX).all()
    assert np.array_equal(np.sort(ps.slot_idx[het]), np.arange(int(het.sum())))


def test_min_ligand_atoms_filter(tmp_path):
    ps = parse_structure(_write(tmp_path, ligand_atoms=3), pdb_id="TEST",
                         keep_ligands=True, min_ligand_atoms=6)
    assert ps.record["n_ligand_atoms"] == 0


def test_n_residues_counts_protein_only(tmp_path):
    """res_pos numbers ligand groups too; the size filters must not see them."""
    ps = parse_structure(_write(tmp_path, n_res=10, ligand_atoms=5),
                         pdb_id="TEST", keep_ligands=True)
    assert ps.n_residues == 10
    assert ps.n_groups == 11          # 10 residues + 1 ligand group


def test_large_ligand_chunks_without_slot_collision(tmp_path):
    """A ligand bigger than N_SLOTS must not overflow the next group's bank."""
    n_lig = C.N_SLOTS + 7
    ps = parse_structure(_write(tmp_path, ligand_atoms=n_lig), pdb_id="TEST",
                         keep_ligands=True)
    lig = ps.residue_idx == C.LIGAND_RESIDUE_IDX
    assert lig.sum() == n_lig
    assert len(set(ps.res_pos[lig].tolist())) == 2      # split across 2 groups
    # The decoder address must be unique for every atom in the structure.
    addr = ps.res_pos * C.N_SLOTS + ps.slot_idx
    assert len(set(addr.tolist())) == len(addr)


def test_large_compound_keeps_bonds_across_group_chunks(tmp_path):
    """A molecule bigger than N_SLOTS spans several groups; its topology must
    not be silently truncated at the chunk boundary."""
    n_lig = C.N_SLOTS + 7
    ps = parse_structure(_write(tmp_path, ligand_atoms=n_lig), pdb_id="TEST",
                         keep_ligands=True)
    lig = np.where(ps.residue_idx == C.LIGAND_RESIDUE_IDX)[0]
    ligset = set(lig.tolist())
    lig_bonds = [(int(i), int(j)) for i, j in ps.bonds
                 if int(i) in ligset and int(j) in ligset]
    # The fixture is a 1.4 A chain, so every consecutive pair is bonded --
    # including the pair straddling the chunk boundary.
    assert len(lig_bonds) == n_lig - 1
    groups = ps.res_pos[lig]
    assert any(groups[i] != groups[i + 1] for i in range(len(lig) - 1))


def test_no_spurious_protein_ligand_bonds(tmp_path):
    """res_pos is global, so the ligand group is numerically adjacent to the
    last residue; only the chain guard stops a bond being perceived."""
    ps = parse_structure(_write(tmp_path), pdb_id="TEST", keep_ligands=True)
    lig = set(np.where(ps.residue_idx == C.LIGAND_RESIDUE_IDX)[0].tolist())
    for i, j in ps.bonds:
        assert (int(i) in lig) == (int(j) in lig), "bond crosses protein/ligand"


def test_ligand_gets_its_own_chain_idx(tmp_path):
    ps = parse_structure(_write(tmp_path), pdb_id="TEST", keep_ligands=True)
    lig = ps.residue_idx == C.LIGAND_RESIDUE_IDX
    assert set(ps.chain_idx[lig].tolist()).isdisjoint(set(ps.chain_idx[~lig].tolist()))


def test_two_ligands_are_not_bonded_to_each_other():
    """Distinct hetero groups get distinct chain_idx, so even if placed close
    together no bond is perceived between them."""
    coords = np.array([[0., 0., 0.], [1.4, 0., 0.],      # ligand 1
                       [2.8, 0., 0.], [4.2, 0., 0.]],    # ligand 2, 1.4 A away
                      dtype=np.float32)
    res_pos = np.array([0, 0, 1, 1], dtype=np.int64)
    chain = np.array([0, 0, 1, 1], dtype=np.int64)
    bonds = perceive_bonds(coords, ["C"] * 4, res_pos, chain)
    assert all(chain[i] == chain[j] for i, j in bonds)


# --- the model consumes it -------------------------------------------------

def test_direct_decoder_reconstructs_a_ligand_batch(tmp_path):
    ps = parse_structure(_write(tmp_path, ligand_atoms=8), pdb_id="TEST",
                         keep_ligands=True)
    from molae.parsing import to_npz_dict
    batch = collate_fn([sample_from_arrays(to_npz_dict(ps))])
    cfg = ModelConfig(d_model=32, n_heads=2, latent_dim=4, enc_self_layers=1,
                      dec_self_layers=1, encoder_type="direct", use_slot_emb=True)
    model = DirectAutoencoder(cfg)
    preds, z = model(batch)
    assert preds.shape == batch["coords"].shape
    assert torch.isfinite(preds).all()


def test_slot_embedding_separates_ligand_atoms():
    """Without it every ligand atom of the same element is identical input."""
    from molae.model import AtomFeaturizer
    batch = {
        "element_idx": torch.tensor([[2, 2, 2]]),
        "residue_idx": torch.full((1, 3), C.LIGAND_RESIDUE_IDX),
        "atom_name_idx": torch.full((1, 3), C.UNK_ATOM_IDX),
        "slot_idx": torch.tensor([[0, 1, 2]]),
        "res_pos": torch.zeros(1, 3, dtype=torch.long),
        "chain_idx": torch.zeros(1, 3, dtype=torch.long),
    }
    off = AtomFeaturizer(16, 64, use_coords=False, use_slot_emb=False)(batch)
    assert torch.allclose(off[0, 0], off[0, 1])       # indistinguishable
    on = AtomFeaturizer(16, 64, use_coords=False, use_slot_emb=True)(batch)
    assert not torch.allclose(on[0, 0], on[0, 1])


# --- checkpoint compatibility ---------------------------------------------

def test_grown_vocabulary_checkpoint_loads_and_preserves_rows():
    """Milestone-1 checkpoints predate the ligand element/residue symbols."""
    import torch.nn as nn
    from molae.utils import grow_embedding_rows

    model = nn.ModuleDict({"e": nn.Embedding(C.N_ELEMENTS, 8)})
    old = {"e.weight": torch.arange(16 * 8, dtype=torch.float32).view(16, 8)}
    grown = grow_embedding_rows(old, model)
    assert grown and grown[0][1] == (16, 8) and grown[0][2] == (C.N_ELEMENTS, 8)
    model.load_state_dict(old)
    # Existing rows survive verbatim; only the appended ones are new.
    assert torch.equal(model["e"].weight[:16],
                       torch.arange(16 * 8, dtype=torch.float32).view(16, 8))


def test_grow_leaves_mismatched_tensors_alone():
    """A genuinely wrong shape must still raise, not be silently padded."""
    import torch.nn as nn
    from molae.utils import grow_embedding_rows

    model = nn.ModuleDict({"e": nn.Embedding(C.N_ELEMENTS, 8)})
    bad = {"e.weight": torch.zeros(4, 5)}     # wrong width
    assert grow_embedding_rows(bad, model) == []


# --- covalent anchoring ----------------------------------------------------

def test_covalent_ligand_linkage_is_perceived():
    """N-linked glycans bond to ASN and heme C to the CXXCH cysteines. Those
    pairs are in different chains by construction, so the chain guard used to
    sever them and the ligand floated free of the residue holding it."""
    coords = np.array([[0., 0., 0.], [1.45, 0., 0.]], dtype=np.float32)
    b = perceive_bonds(coords, ["N", "C"], np.array([0, 1]), np.array([0, 1]),
                       unrestricted_chains=(1,))
    assert len(b) == 1


def test_noncovalent_contact_is_not_a_bond():
    """A binding-pocket contact must not become a bond. H-bond range."""
    coords = np.array([[0., 0., 0.], [2.9, 0., 0.]], dtype=np.float32)
    b = perceive_bonds(coords, ["N", "C"], np.array([0, 1]), np.array([0, 1]),
                       unrestricted_chains=(1,))
    assert len(b) == 0


def test_metal_coordination_is_not_a_bond():
    """Zn-His sits at 2.0-2.4 A; the metal radii are set so it stays unbonded."""
    coords = np.array([[0., 0., 0.], [2.15, 0., 0.]], dtype=np.float32)
    b = perceive_bonds(coords, ["N", "ZN"], np.array([0, 1]), np.array([0, 1]),
                       unrestricted_chains=(1,))
    assert len(b) == 0


def test_glycan_tree_stays_connected():
    """An N-glycan is a TREE of separately-deposited monosaccharides, each its
    own hetero residue and so its own chain. Admitting only ligand-to-polymer
    pairs anchored the first sugar and left the rest floating free."""
    coords = np.array([[0., 0, 0], [1.45, 0, 0], [2.90, 0, 0], [4.35, 0, 0]],
                      dtype=np.float32)                 # ASN-NAG-NAG-BMA
    b = perceive_bonds(coords, ["N", "C", "C", "C"], np.array([0, 1, 2, 3]),
                       np.array([0, 1, 2, 3]), unrestricted_chains=(1, 2, 3))
    assert len(b) == 3, b.tolist()      # one anchor + two glycosidic links


def test_ligands_at_packing_distance_do_not_bond():
    """The distance gate is what keeps two co-bound species separate; nothing
    physical sits under 1.9 A without a bond, but 3.4 A packing is common."""
    coords = np.array([[0., 0., 0.], [3.4, 0., 0.]], dtype=np.float32)
    b = perceive_bonds(coords, ["C", "C"], np.array([0, 1]), np.array([1, 2]),
                       unrestricted_chains=(1, 2))
    assert len(b) == 0


# --- modelling artifacts ---------------------------------------------------

def test_sub_angstrom_pair_is_not_a_bond():
    """Nothing real is bonded below ~0.9 A. A shorter pair is one atom
    deposited twice; bonding it fuses two copies into an over-valent blob
    (6S2M has C1-C1 at 0.17 A)."""
    coords = np.array([[0., 0., 0.], [0.17, 0., 0.]], dtype=np.float32)
    b = perceive_bonds(coords, ["C", "C"], np.array([0, 1]), np.array([1, 2]),
                       unrestricted_chains=(1, 2))
    assert len(b) == 0


def _overlapping_ligands_pdb(tmp_path, occ_a=0.35, occ_b=0.43, names_b=None):
    """Two DIFFERENTLY-NAMED partial-occupancy ligands in one density blob,
    as 6S2M deposits PLM (0.35) and VCA (0.43) 0.3 A apart."""
    lines, serial = [], 1
    for r in range(1, 11):
        base = np.array([3.8 * r, 0.0, 0.0])
        for aname, elem, off in [("N", "N", (0., 0., 0.)), ("CA", "C", (1.4, .2, 0.)),
                                 ("C", "C", (2.5, -.4, 0.)), ("O", "O", (2.6, -1.6, 0.)),
                                 ("CB", "C", (1.5, 1.7, 0.))]:
            lines.append(_atom_line("ATOM", serial, f" {aname:<3s}"[:4], "ALA",
                                    "A", r, base + np.array(off), elem))
            serial += 1
    default = [f"C{k + 1}" for k in range(6)]
    for name, occ, dx, nms in [("PLM", occ_a, 0.0, default),
                               ("VCA", occ_b, 0.3, names_b or default)]:
        for k, nm in enumerate(nms):
            lines.append(_atom_line("HETATM", serial, f" {nm:<3s}"[:4], name,
                                    "B", 501 if name == "PLM" else 502,
                                    (50.0 + 1.5 * k + dx, 8.0, 0.0), nm[0], occ=occ))
            serial += 1
    lines.append("END")
    p = tmp_path / "dup.pdb"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_overlapping_partial_occupancy_ligands_collapse_to_one(tmp_path):
    ps = parse_structure(_overlapping_ligands_pdb(tmp_path), pdb_id="T",
                         keep_ligands=True)
    assert ps.record["n_duplicate_ligands_dropped"] == 1
    assert len(ps.record["ligands_kept"]) == 1


def test_the_better_supported_copy_is_the_one_kept(tmp_path):
    """Deposition order would keep PLM (0.35); occupancy ordering keeps VCA."""
    ps = parse_structure(_overlapping_ligands_pdb(tmp_path, 0.35, 0.43),
                         pdb_id="T", keep_ligands=True)
    assert ps.record["ligands_kept"][0]["name"] == "VCA"
    # and it flips when the occupancies do
    ps = parse_structure(_overlapping_ligands_pdb(tmp_path, 0.80, 0.43),
                         pdb_id="T", keep_ligands=True)
    assert ps.record["ligands_kept"][0]["name"] == "PLM"


def test_nested_ligand_with_disjoint_names_is_kept(tmp_path):
    """The case the overlap rule must NOT catch: a small ligand genuinely
    nested inside a larger one (an ion chelated at the centre of a ring).
    Centroids nearly coincide, but they share no atom names."""
    ps = parse_structure(
        _overlapping_ligands_pdb(tmp_path, 1.0, 1.0, names_b=["N1", "N2", "N3",
                                                              "N4", "N5", "N6"]),
        pdb_id="T", keep_ligands=True)
    assert ps.record["n_duplicate_ligands_dropped"] == 0
    assert len(ps.record["ligands_kept"]) == 2


def test_same_names_at_one_site_are_duplicates_at_any_occupancy(tmp_path):
    """Full occupancy does not make two overlapping same-named copies real."""
    ps = parse_structure(_overlapping_ligands_pdb(tmp_path, 1.0, 1.0),
                         pdb_id="T", keep_ligands=True)
    assert ps.record["n_duplicate_ligands_dropped"] == 1


def test_shortcut_bond_is_pruned():
    """1H4X deposits a phosphoserine whose CB...P 1,3 distance falls inside
    the C/P cutoff, bonding CB straight to P and pushing P to valence 5. The
    real path is CB-OG-P."""
    cb = np.array([0., 0., 0.]); og = np.array([1.43, 0., 0.])
    p = og + np.array([0.55, 1.50, 0.]) / np.linalg.norm([0.55, 1.50, 0.]) * 1.60
    coords = np.stack([cb, og, p]).astype(np.float32)
    b = perceive_bonds(coords, ["C", "O", "P"], np.zeros(3, dtype=np.int64))
    assert len(b) == 2
    assert [0, 2] not in b.tolist()


def test_three_membered_ring_survives_shortcut_pruning():
    """Every pair in cyclopropane is a real bond AND a 1,3 via the third, so
    the pruning must key on the length ratio, not on 1,3-ness alone."""
    r = 1.51 / np.sqrt(3)
    s = np.sqrt(3) / 2
    ring = np.array([[r, 0, 0], [-r / 2, r * s, 0], [-r / 2, -r * s, 0]],
                    dtype=np.float32)
    b = perceive_bonds(ring, ["C"] * 3, np.zeros(3, dtype=np.int64))
    assert len(b) == 3


def test_repeated_atom_names_within_one_group_are_collapsed(tmp_path):
    """A PDB atom name is unique within a residue by definition, so repeats
    mean two copies were deposited into ONE residue -- 5IVT carries two C28,
    two C10, two C27. altloc removal does not touch these."""
    ps = parse_structure(
        _write(tmp_path, ligand_atom_names=["C1", "C2", "C1", "C3", "C2"]),
        pdb_id="T", keep_ligands=True)
    lig = ps.residue_idx == C.LIGAND_RESIDUE_IDX
    # gemmi already collapses same-named atoms on read, so this asserts the
    # end-to-end property; the guard in parse_structure is belt-and-braces
    # behind it, for readers that do not.
    names = [n for n, k in zip(ps.atom_name, lig) if k]
    assert len(names) == len(set(names)) == 3   # C1, C2, C3


def _alt_conformation_pdb(tmp_path):
    """5IVT's shape: ONE ligand deposited as two alternate conformations. Same
    atom names, but the copies are DISPLACED -- centroids 5.78 A apart -- while
    individual atoms still overlap at ~1.0 A. A centroid test cannot see this.
    """
    lines, serial = [], 1
    for r in range(1, 11):
        base = np.array([3.8 * r, 0.0, 0.0])
        for aname, elem, off in [("N", "N", (0., 0., 0.)), ("CA", "C", (1.4, .2, 0.)),
                                 ("C", "C", (2.5, -.4, 0.)), ("O", "O", (2.6, -1.6, 0.)),
                                 ("CB", "C", (1.5, 1.7, 0.))]:
            lines.append(_atom_line("ATOM", serial, f" {aname:<3s}"[:4], "ALA",
                                    "A", r, base + np.array(off), elem))
            serial += 1
    # Copy A runs along +x; copy B is the same molecule shifted, so the two
    # centroids are far apart but their near ends interpenetrate.
    for seq, shift in [(501, 0.0), (502, 11.0)]:
        for k in range(12):
            lines.append(_atom_line("HETATM", serial, f" C{k + 1:<2d}"[:4], "LIG",
                                    "B", seq, (50.0 + 1.0 * k + shift, 8.0, 0.0),
                                    "C", occ=0.5))
            serial += 1
    lines.append("END")
    p = tmp_path / "alt.pdb"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_displaced_alternate_conformations_are_collapsed(tmp_path):
    """The 5IVT case: centroids far apart, atoms overlapping. Interpenetration
    catches it where centroid coincidence cannot."""
    ps = parse_structure(_alt_conformation_pdb(tmp_path), pdb_id="T",
                         keep_ligands=True)
    cens = [np.mean([ps.coords[i] for i in range(ps.n_atoms)
                     if ps.res_pos[i] == g], axis=0)
            for g in sorted(set(ps.res_pos[ps.residue_idx == C.LIGAND_RESIDUE_IDX].tolist()))]
    assert ps.record["n_duplicate_ligands_dropped"] == 1
    assert len(ps.record["ligands_kept"]) == 1


def test_same_ligand_at_two_distant_sites_is_kept(tmp_path):
    """The case interpenetration must NOT catch: two real copies of one ligand
    bound at two different sites. Identical names, but far apart."""
    lines, serial = [], 1
    for r in range(1, 11):
        base = np.array([3.8 * r, 0.0, 0.0])
        for aname, elem, off in [("N", "N", (0., 0., 0.)), ("CA", "C", (1.4, .2, 0.)),
                                 ("C", "C", (2.5, -.4, 0.)), ("O", "O", (2.6, -1.6, 0.)),
                                 ("CB", "C", (1.5, 1.7, 0.))]:
            lines.append(_atom_line("ATOM", serial, f" {aname:<3s}"[:4], "ALA",
                                    "A", r, base + np.array(off), elem))
            serial += 1
    for seq, shift in [(501, 0.0), (502, 40.0)]:      # 40 A apart: distinct sites
        for k in range(6):
            lines.append(_atom_line("HETATM", serial, f" C{k + 1:<2d}"[:4], "LIG",
                                    "B", seq, (50.0 + 1.5 * k + shift, 8.0, 0.0), "C"))
            serial += 1
    lines.append("END")
    f = tmp_path / "twosites.pdb"; f.write_text("\n".join(lines) + "\n")
    ps = parse_structure(str(f), pdb_id="T", keep_ligands=True)
    assert ps.record["n_duplicate_ligands_dropped"] == 0
    assert len(ps.record["ligands_kept"]) == 2


def test_equal_occupancy_duplicates_are_still_collapsed(tmp_path):
    """5IVT's two copies sit at EQUAL occupancy, which the occupancy rule
    cannot separate -- name overlap is what identifies them."""
    ps = parse_structure(_overlapping_ligands_pdb(tmp_path, 0.5, 0.5),
                         pdb_id="T", keep_ligands=True)
    # same-named atoms in both copies, centroids overlapping
    assert ps.record["n_duplicate_ligands_dropped"] == 1
