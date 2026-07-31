"""Binding-pocket cropping.

The two load-bearing invariants are whole-residue selection and contiguous
group renumbering. Both are decoder requirements: slots are addressed by
``res_pos * N_SLOTS + slot_idx``, so a partial residue trains the decoder to
emit slots that never appear, and a gap in the numbering silently inflates the
slot bank. Neither shows up as an error -- they show up as wasted capacity.
"""

import numpy as np
import pytest

from molae import constants as C
from molae.pocket import pocket_residues, crop_to_pocket, pocket_stats


def _complex(n_res=20, atoms_per_res=5, ligand_atoms=6, spacing=4.0):
    """A protein line along +x with a ligand parked next to residue 10."""
    coords, res_pos, resid, slot, elems, names = [], [], [], [], [], []
    for r in range(n_res):
        for a in range(atoms_per_res):
            coords.append([r * spacing, a * 0.4, 0.0])
            res_pos.append(r)
            resid.append(C.RESIDUE_TO_IDX["ALA"])
            slot.append(2 + a)
            elems.append("C")
            names.append(f"A{a}")
    lig_centre = np.array([10 * spacing, 3.0, 0.0])
    for k in range(ligand_atoms):
        coords.append(lig_centre + np.array([k * 0.5, 0.0, 0.0]))
        res_pos.append(n_res)
        resid.append(C.LIGAND_RESIDUE_IDX)
        slot.append(k)
        elems.append("C")
        names.append(f"C{k}")
    coords = np.array(coords, dtype=np.float32)
    n = len(coords)
    # chain bonds within each residue plus consecutive-residue links
    bonds = []
    for i in range(n - 1):
        if res_pos[i] == res_pos[i + 1] or (res_pos[i + 1] - res_pos[i] == 1):
            bonds.append((i, i + 1))
    return {
        "coords": coords,
        "res_pos": np.array(res_pos, dtype=np.int64),
        "residue_idx": np.array(resid, dtype=np.int64),
        "slot_idx": np.array(slot, dtype=np.int64),
        "atom_name_idx": np.array(slot, dtype=np.int64),
        "element_idx": np.ones(n, dtype=np.int64) * 2,
        "chain_idx": np.zeros(n, dtype=np.int64),
        "res_seq": np.array(res_pos, dtype=np.int64),
        "bonds": np.array(bonds, dtype=np.int64),
        "element_symbol": np.array(elems),
        "atom_name": np.array(names),
        "pdb_id": "TEST",
        "chain_id": "A",
        "sequence": "A" * n_res,
    }


def test_ligand_is_always_kept_whole():
    """Cropping around the ligand must never clip the ligand itself."""
    d = _complex()
    out = crop_to_pocket(d, radius=6.0)
    s = pocket_stats(d, out)
    assert s["ligand_complete"]
    assert s["ligand_atoms_after"] == 6


def test_only_nearby_residues_survive():
    d = _complex(n_res=20, spacing=4.0)
    out = crop_to_pocket(d, radius=6.0)
    # ligand sits at x=40 (residue 10); radius 6 A reaches residues 9-11
    kept = sorted(set(out["res_pos"].tolist()))
    assert len(kept) < 20
    assert out["coords"].shape[0] < d["coords"].shape[0]


def test_selection_is_by_whole_residue():
    """A partial residue would leave a group whose atoms are a subset of its
    slot bank -- representable, but it trains the decoder on slots that never
    appear."""
    d = _complex(atoms_per_res=5)
    out = crop_to_pocket(d, radius=6.0)
    counts = {}
    for g in out["res_pos"].tolist():
        counts[g] = counts.get(g, 0) + 1
    protein_groups = [g for g in counts
                      if (out["residue_idx"][out["res_pos"] == g][0]
                          != C.LIGAND_RESIDUE_IDX)]
    assert all(counts[g] == 5 for g in protein_groups)


def test_group_numbering_is_contiguous_from_zero():
    """A gap inflates the slot bank, which max(res_pos)+1 sizes."""
    d = _complex()
    out = crop_to_pocket(d, radius=6.0)
    groups = sorted(set(out["res_pos"].tolist()))
    assert groups == list(range(len(groups)))


def test_decoder_addresses_stay_unique_after_cropping():
    d = _complex()
    out = crop_to_pocket(d, radius=8.0)
    addr = out["res_pos"] * C.N_SLOTS + out["slot_idx"]
    assert len(set(addr.tolist())) == len(addr)


def test_bond_indices_are_remapped_not_stale():
    """The silent failure: keeping old indices after subsetting points bonds
    at whatever atom now occupies that slot."""
    d = _complex()
    out = crop_to_pocket(d, radius=6.0)
    n = len(out["coords"])
    b = out["bonds"]
    assert b.size == 0 or (b.min() >= 0 and b.max() < n)
    # every retained bond still joins atoms that are actually close
    for i, j in b:
        assert np.linalg.norm(out["coords"][i] - out["coords"][j]) < 5.0


def test_severed_bonds_are_counted_not_hidden():
    """Cropping cuts the backbone at the boundary by construction. That is
    expected, but a crop that shreds the chain should be visible."""
    d = _complex()
    out = crop_to_pocket(d, radius=6.0)
    assert out["n_severed_bonds"] >= 1
    assert pocket_stats(d, out)["n_severed_bonds"] == out["n_severed_bonds"]


def test_larger_radius_keeps_more():
    d = _complex()
    small = crop_to_pocket(d, radius=5.0)
    large = crop_to_pocket(d, radius=15.0)
    assert len(large["coords"]) > len(small["coords"])


def test_radius_spanning_everything_is_a_no_op_on_content():
    d = _complex()
    out = crop_to_pocket(d, radius=1e6)
    assert len(out["coords"]) == len(d["coords"])
    assert out["n_severed_bonds"] == 0


def test_per_atom_arrays_are_all_subset_together():
    """A stale per-atom array would silently misalign identity from geometry."""
    d = _complex()
    out = crop_to_pocket(d, radius=6.0)
    n = len(out["coords"])
    for key in ("res_pos", "residue_idx", "slot_idx", "atom_name_idx",
                "element_idx", "chain_idx", "res_seq", "element_symbol",
                "atom_name"):
        assert len(out[key]) == n, key


def test_scalar_metadata_survives():
    d = _complex()
    out = crop_to_pocket(d, radius=6.0)
    assert out["pdb_id"] == "TEST" and out["chain_id"] == "A"


def test_no_ligand_is_an_error_not_a_silent_empty_crop():
    d = _complex(ligand_atoms=0)
    d["residue_idx"][:] = C.RESIDUE_TO_IDX["ALA"]
    with pytest.raises(ValueError, match="no ligand"):
        crop_to_pocket(d, radius=6.0)


def test_compression_is_reported():
    d = _complex(n_res=60)
    out = crop_to_pocket(d, radius=6.0)
    s = pocket_stats(d, out)
    assert s["compression"] > 3.0
    assert s["atoms_after"] < s["atoms_before"]
