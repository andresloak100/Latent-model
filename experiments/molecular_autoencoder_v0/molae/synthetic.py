"""A tiny synthetic molecule with fully known geometry, for unit tests.

Builds a chain of ``n_res`` alanine residues. Each residue uses a fixed,
chemically plausible local frame (N, CA, C, O, CB) with hand-verified bond
lengths; residues are spaced 10 A apart so bond perception yields exactly
four intra-residue bonds per residue and no accidental cross-residue bonds.
The signed chirality volume of every residue is positive (an "L" convention),
so reflecting z flips it -- used to test the chirality metric.
"""

from __future__ import annotations

import numpy as np

from . import constants as C

# One residue in its local frame (angstrom). Verified: |N-CA|=1.458,
# |CA-C|=1.523, |C-O|=1.230, |CA-CB|=1.523, and N/C are > cutoff from CB.
_LOCAL = {
    "N":  np.array([0.000, 0.000, 0.000]),
    "CA": np.array([1.458, 0.000, 0.000]),
    "C":  np.array([2.008, 1.420, 0.000]),
    "O":  np.array([2.008, 2.650, 0.000]),
    "CB": np.array([1.658, -0.800, -1.280]),
}
_ELEMENTS = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C"}
_ORDER = ["N", "CA", "C", "O", "CB"]


def make_synthetic_ala(n_res: int = 3, spacing: float = 10.0) -> dict:
    coords, element_idx, residue_idx, atom_name_idx = [], [], [], []
    res_pos, res_seq, element_symbol, atom_name = [], [], [], []
    for r in range(n_res):
        offset = np.array([spacing * r, 0.0, 0.0])
        for aname in _ORDER:
            coords.append(_LOCAL[aname] + offset)
            esym = _ELEMENTS[aname]
            element_idx.append(C.ELEMENT_TO_IDX[esym])
            residue_idx.append(C.RESIDUE_TO_IDX["ALA"])
            atom_name_idx.append(C.ATOM_NAME_TO_IDX[aname])
            res_pos.append(r)
            res_seq.append(r + 1)
            element_symbol.append(esym)
            atom_name.append(aname)

    coords = np.asarray(coords, dtype=np.float32)
    res_pos = np.asarray(res_pos, dtype=np.int64)
    from .parsing import perceive_bonds
    bonds = perceive_bonds(coords, element_symbol, res_pos)
    return {
        "coords": coords,
        "element_idx": np.asarray(element_idx, dtype=np.int64),
        "residue_idx": np.asarray(residue_idx, dtype=np.int64),
        "atom_name_idx": np.asarray(atom_name_idx, dtype=np.int64),
        "res_pos": res_pos,
        "res_seq": np.asarray(res_seq, dtype=np.int64),
        "bonds": bonds,
        "element_symbol": np.array(element_symbol),
        "atom_name": np.array(atom_name),
        "pdb_id": "SYNTH",
        "chain_id": "A",
        "sequence": "A" * n_res,
    }
