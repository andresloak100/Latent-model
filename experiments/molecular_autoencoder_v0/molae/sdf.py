"""Minimal V2000 molfile / SDF reader — the input with no residues at all.

Deliberately dependency-free. An .sdf gives elements, coordinates, bonds,
bond orders and formal charges, and nothing else: no residue index, no chain,
no atom-name vocabulary. That is exactly the input the atom-only path has to
handle, so it is the honest end-to-end test of it.

Residue-shaped fields are still emitted, filled with neutral values, because
``collate_fn`` expects them. They are inert by construction when
``model.atom_addressing == "graph"``, and a test asserts the decoder's output
does not change when they are overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import constants as C
from .graph_identity import graph_atom_features


def parse_sdf_block(text: str) -> dict:
    """Parse one V2000 molfile block -> arrays. Raises ValueError if malformed."""
    lines = text.splitlines()
    if len(lines) < 4:
        raise ValueError("molfile too short")
    counts = lines[3]
    try:
        n_atoms, n_bonds = int(counts[0:3]), int(counts[3:6])
    except ValueError as e:
        raise ValueError(f"bad counts line: {counts!r}") from e

    coords, symbols = [], []
    for ln in lines[4:4 + n_atoms]:
        coords.append([float(ln[0:10]), float(ln[10:20]), float(ln[20:30])])
        symbols.append(ln[31:34].strip())

    bonds, bond_types = [], []
    for ln in lines[4 + n_atoms:4 + n_atoms + n_bonds]:
        a, b, t = int(ln[0:3]) - 1, int(ln[3:6]) - 1, int(ln[6:9])
        bonds.append([a, b])
        bond_types.append(t)

    charges = np.zeros(n_atoms, dtype=np.int64)
    for ln in lines[4 + n_atoms + n_bonds:]:
        if ln.startswith("M  CHG"):
            f = ln.split()
            for k in range(int(f[2])):
                charges[int(f[3 + 2 * k]) - 1] = int(f[4 + 2 * k])
        elif ln.startswith("M  END"):
            break

    return {
        "coords": np.asarray(coords, dtype=np.float32),
        "element_symbol": symbols,
        "bonds": np.asarray(bonds, dtype=np.int64).reshape(-1, 2),
        "bond_types": np.asarray(bond_types, dtype=np.int64),
        "formal_charge": charges,
    }


def sample_from_sdf(text: str, name: str = "SDF") -> dict:
    """A molfile block -> the sample dict the pipeline consumes."""
    raw = parse_sdf_block(text)
    n = len(raw["element_symbol"])
    if n == 0:
        raise ValueError("no atoms")
    elements = np.array([C.ELEMENT_TO_IDX.get(s.capitalize(), C.UNK_ELEMENT_IDX)
                         for s in raw["element_symbol"]], dtype=np.int64)
    g = graph_atom_features(elements, raw["formal_charge"],
                            raw["bonds"], raw["bond_types"])

    coords = raw["coords"] - raw["coords"].mean(axis=0, keepdims=True)
    out = {
        "n_atoms": n,
        "pdb_id": name,
        "chain_id": "_",
        "element_symbol": list(raw["element_symbol"]),
        "coords": torch.from_numpy(coords),
        "bonds": torch.from_numpy(raw["bonds"]),
        "chirality_centers": torch.zeros((0, 4), dtype=torch.long),
        # Residue-shaped fields, neutral. Inert under graph addressing.
        "res_pos": torch.zeros(n, dtype=torch.long),
        "res_seq": torch.zeros(n, dtype=torch.long),
        "chain_idx": torch.zeros(n, dtype=torch.long),
        "residue_idx": torch.full((n,), C.LIGAND_RESIDUE_IDX, dtype=torch.long),
        "atom_name_idx": torch.full((n,), C.UNK_ATOM_IDX, dtype=torch.long),
        "slot_idx": torch.zeros(n, dtype=torch.long),
    }
    for k, v in g.items():
        out[k] = torch.from_numpy(np.asarray(v, dtype=np.int64))
    out["element_idx"] = torch.from_numpy(elements)
    return out


def read_sdf(path) -> list:
    """All records in an .sdf (blocks separated by ``$$$$``)."""
    text = Path(path).read_text()
    out = []
    for i, block in enumerate(text.split("$$$$")):
        if block.strip():
            out.append(sample_from_sdf(block, name=f"{Path(path).stem}_{i}"))
    return out
