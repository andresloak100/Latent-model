"""Minimal PDB writer for saving example reconstructions.

Writes heavy-atom records only. Element/residue/atom names come from the
parsed vocabulary indices. This is for qualitative inspection (e.g. in PyMOL);
it does not write connectivity or headers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import constants as C

_IDX_TO_RES = {v: k for k, v in C.RESIDUE_TO_IDX.items()}
_IDX_TO_ATOM = {v: k for k, v in C.ATOM_NAME_TO_IDX.items()}


def write_pdb(path, coords, residue_idx, atom_name_idx, res_seq, element_symbol,
              chain_id="A"):
    coords = np.asarray(coords)
    lines = []
    for i in range(coords.shape[0]):
        aname = _IDX_TO_ATOM.get(int(atom_name_idx[i]), "X")
        rname = _IDX_TO_RES.get(int(residue_idx[i]), "UNK")
        elem = str(element_symbol[i])
        x, y, z = coords[i]
        # Standard PDB ATOM record columns.
        aname_field = (" " + aname) if len(aname) < 4 else aname
        lines.append(
            f"ATOM  {i+1:>5} {aname_field:<4} {rname:>3} {chain_id:>1}"
            f"{int(res_seq[i]):>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{1.00:>6.2f}{0.00:>6.2f}          {elem:>2}"
        )
    lines.append("END")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return path
