"""Torch dataset + collate for variable-size proteins.

Each processed structure is stored as an ``.npz``. Coordinates are centred to
their centroid on load (removing global translation). Batching pads to the
largest structure and produces an atom mask; per-sample topology (bonds,
chirality centres, a bonded-pair mask) is carried as a list because sizes
vary and the losses/metrics loop per sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import constants as C
from .metrics import _CA, _N, _Cc, _CB


def _chirality_centers(atom_name_idx, res_pos):
    n_res = int(res_pos.max()) + 1 if len(res_pos) else 0
    centers = []
    for r in range(n_res):
        sel = np.where(res_pos == r)[0]
        names = {int(atom_name_idx[i]): i for i in sel}
        if all(k in names for k in (_CA, _N, _Cc, _CB)):
            centers.append([names[_CA], names[_N], names[_Cc], names[_CB]])
    return np.array(centers, dtype=np.int64) if centers else np.zeros((0, 4), dtype=np.int64)


class ProteinStructureDataset(Dataset):
    def __init__(self, npz_paths):
        self.paths = [Path(p) for p in npz_paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i], allow_pickle=True)
        return sample_from_arrays(d)


def sample_from_arrays(d) -> dict:
    """Build a collate-ready sample from a dict/npz of parsed arrays.

    Coordinates are centred to their centroid (removes global translation).
    Works with both ``np.load`` objects and plain dicts (e.g. the synthetic
    molecule), so tests can reuse the exact training-time preprocessing.
    """
    coords = np.asarray(d["coords"]).astype(np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)  # centre
    n = coords.shape[0]
    bonds = np.asarray(d["bonds"]).astype(np.int64).reshape(-1, 2)
    atom_name_idx = np.asarray(d["atom_name_idx"]).astype(np.int64)
    res_pos = np.asarray(d["res_pos"]).astype(np.int64)
    chain_idx = (np.asarray(d["chain_idx"]).astype(np.int64)
                 if "chain_idx" in d else np.zeros(len(res_pos), dtype=np.int64))
    # Decoder slot within the group. Absent from every .npz written before
    # ligand support existed, and for a protein residue the slot IS the atom
    # name -- so this default reproduces the old gather exactly.
    slot_idx = (np.asarray(d["slot_idx"]).astype(np.int64)
                if "slot_idx" in d else atom_name_idx)
    res_seq = np.asarray(d["res_seq"]).astype(np.int64) if "res_seq" in d else res_pos

    # NOTE: no dense (N, N) bonded mask is built. It cost 34 MB/structure at
    # 3000 atoms and 382 MB at 10,000 -- per sample, in a dataloader worker --
    # which is what blocked complex-scale training. The sparse `bonds` array is
    # carried instead and the clash loss tests membership from it.
    return {
        "element_idx": torch.from_numpy(np.asarray(d["element_idx"]).astype(np.int64)),
        "residue_idx": torch.from_numpy(np.asarray(d["residue_idx"]).astype(np.int64)),
        "atom_name_idx": torch.from_numpy(atom_name_idx),
        "slot_idx": torch.from_numpy(slot_idx),
        "res_pos": torch.from_numpy(res_pos),
        "chain_idx": torch.from_numpy(chain_idx),
        "res_seq": torch.from_numpy(res_seq),
        "coords": torch.from_numpy(coords),
        "bonds": torch.from_numpy(bonds),
        "chirality_centers": torch.from_numpy(_chirality_centers(atom_name_idx, res_pos)),
        "n_atoms": n,
        "pdb_id": str(d["pdb_id"]),
        "chain_id": str(d["chain_id"]),
        "element_symbol": [str(x) for x in d["element_symbol"]],
    }


def collate_fn(samples):
    B = len(samples)
    nmax = max(s["n_atoms"] for s in samples)

    def pad_int(key, pad_val=0):
        out = torch.full((B, nmax), pad_val, dtype=torch.long)
        for i, s in enumerate(samples):
            n = s["n_atoms"]
            out[i, :n] = s[key]
        return out

    coords = torch.zeros(B, nmax, 3, dtype=torch.float32)
    mask = torch.zeros(B, nmax, dtype=torch.float32)
    for i, s in enumerate(samples):
        n = s["n_atoms"]
        coords[i, :n] = s["coords"]
        mask[i, :n] = 1.0

    return {
        "element_idx": pad_int("element_idx", C.PAD_ELEMENT_IDX),
        "residue_idx": pad_int("residue_idx", C.PAD_RESIDUE_IDX),
        "atom_name_idx": pad_int("atom_name_idx", C.PAD_ATOM_IDX),
        "slot_idx": pad_int("slot_idx", C.PAD_ATOM_IDX),
        "res_pos": pad_int("res_pos", 0),
        "chain_idx": pad_int("chain_idx", 0),
        "coords": coords,
        "mask": mask,
        "n_atoms": torch.tensor([s["n_atoms"] for s in samples], dtype=torch.long),
        "bonds": [s["bonds"] for s in samples],
        "chirality_centers": [s["chirality_centers"] for s in samples],
        "pdb_id": [s["pdb_id"] for s in samples],
        "chain_id": [s["chain_id"] for s in samples],
        "element_symbol": [s["element_symbol"] for s in samples],
    }
