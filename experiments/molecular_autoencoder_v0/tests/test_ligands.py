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

def _atom_line(record, serial, name, resname, chain, resseq, xyz, elem):
    """A column-exact PDB ATOM/HETATM record.

    The columns are not negotiable: 13-16 atom name, 17 altLoc, 18-20 resName,
    22 chainID, 23-26 resSeq, 31-54 xyz, 77-78 element. Getting 17 wrong makes
    gemmi read the whole file as alternate conformations.
    """
    x, y, z = xyz
    return (f"{record:<6s}{serial:5d} {name:<4s}{'':1s}{resname:>3s} "
            f"{chain:1s}{resseq:4d}{'':4s}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}{'':10s}{elem:>2s}")


def _pdb_text(n_res=10, ligand_atoms=5, ligand_name="LIG", water=True):
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
    for k in range(ligand_atoms):
        lines.append(_atom_line("HETATM", serial, f" {'C' + str(k + 1):<3s}"[:4],
                                ligand_name, "B", 501,
                                (5.0 + 1.4 * k, 8.0, 0.0), "C"))
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
