"""Crop a complex to its binding pocket.

MISATO's complexes have a median of 2,542 heavy atoms, and 38% exceed our
3,000-atom cap. Cropping to the residues that actually contact the ligand cuts
that to a few hundred, which is the fallback if reconstruction quality turns
out to degrade with size.

It is worth having regardless of that. Binding lives in the interface: a
global RMSD averaged over a 2,500-atom complex can look excellent while the
few dozen atoms that determine whether the ligand binds are wrong. Cropping
makes the interface the whole measurement rather than 2% of it.

Two invariants the decoder depends on, both enforced here rather than assumed:

  * WHOLE RESIDUES ONLY. Slots are addressed by (group, slot); keeping half a
    residue would leave a group whose atoms are a subset of its slot bank,
    which is representable but trains the decoder to emit slots that never
    appear. Selection is therefore by residue, not by atom.
  * CONTIGUOUS GROUP NUMBERING FROM 0. The gather is
    ``res_pos * N_SLOTS + slot_idx`` against a bank sized by max(res_pos)+1,
    so a gap in the numbering silently inflates the bank and wastes capacity
    on groups that do not exist.

Cropping severs peptide bonds at the boundary by construction. That is
expected -- the result is a fragment -- but it is counted, not hidden, so a
crop that shreds the backbone is visible rather than showing up later as an
unexplained bond-loss term.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def pocket_residues(coords, res_pos, ligand_mask, radius=10.0):
    """Group ids with any atom within ``radius`` of any ligand atom.

    Whole groups, so a residue is in or out as a unit. The ligand's own groups
    are always included.
    """
    lig = np.asarray(ligand_mask, dtype=bool)
    if not lig.any():
        raise ValueError("no ligand atoms: nothing to crop around")
    lig_xyz = coords[lig]
    # Distance from every atom to its nearest ligand atom.
    d = np.linalg.norm(coords[:, None, :] - lig_xyz[None, :, :], axis=-1).min(axis=1)
    near = d <= radius
    keep = set(res_pos[near].tolist()) | set(res_pos[lig].tolist())
    return np.array(sorted(keep), dtype=np.int64)


def crop_to_pocket(d, ligand_mask=None, radius=10.0):
    """Return a new parsed-array dict containing only the pocket.

    ``ligand_mask`` defaults to atoms whose residue is LIG. Bonds whose partner
    was cropped away are dropped and counted in ``n_severed_bonds``.
    """
    coords = np.asarray(d["coords"], dtype=np.float32)
    res_pos = np.asarray(d["res_pos"], dtype=np.int64)
    resid = np.asarray(d["residue_idx"], dtype=np.int64)
    if ligand_mask is None:
        ligand_mask = resid == C.LIGAND_RESIDUE_IDX

    keep_groups = pocket_residues(coords, res_pos, ligand_mask, radius)
    atom_keep = np.isin(res_pos, keep_groups)
    old_to_new = -np.ones(len(coords), dtype=np.int64)
    old_to_new[atom_keep] = np.arange(int(atom_keep.sum()))

    # Contiguous renumbering: group ids must run 0..G-1 with no gaps.
    remap = {int(g): i for i, g in enumerate(keep_groups)}
    new_res_pos = np.array([remap[int(g)] for g in res_pos[atom_keep]], dtype=np.int64)

    bonds = np.asarray(d["bonds"]).reshape(-1, 2)
    if len(bonds):
        both = atom_keep[bonds[:, 0]] & atom_keep[bonds[:, 1]]
        severed = int((atom_keep[bonds[:, 0]] ^ atom_keep[bonds[:, 1]]).sum())
        new_bonds = old_to_new[bonds[both]]
    else:
        severed = 0
        new_bonds = np.zeros((0, 2), dtype=np.int64)

    out = {}
    for key, val in d.items():
        arr = np.asarray(val) if not isinstance(val, (str, int, float)) else val
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and len(arr) == len(coords):
            out[key] = arr[atom_keep]
        else:
            out[key] = val
    out["coords"] = coords[atom_keep]
    out["res_pos"] = new_res_pos
    out["bonds"] = new_bonds
    out["n_severed_bonds"] = severed
    out["n_pocket_groups"] = len(keep_groups)
    return out


def pocket_stats(before, after):
    """What the crop cost and kept, for the manifest."""
    nb = len(np.asarray(before["coords"]))
    na = len(np.asarray(after["coords"]))
    lig_b = int((np.asarray(before["residue_idx"]) == C.LIGAND_RESIDUE_IDX).sum())
    lig_a = int((np.asarray(after["residue_idx"]) == C.LIGAND_RESIDUE_IDX).sum())
    return {
        "atoms_before": nb,
        "atoms_after": na,
        "compression": round(nb / max(na, 1), 2),
        "ligand_atoms_before": lig_b,
        "ligand_atoms_after": lig_a,
        "ligand_complete": lig_a == lig_b,
        "n_severed_bonds": int(after.get("n_severed_bonds", 0)),
        "n_groups": int(after.get("n_pocket_groups", 0)),
    }
