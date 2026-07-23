"""Parsing/filtering tests using a small inline PDB (no network)."""

import numpy as np

from molae.parsing import parse_structure, perceive_bonds
from molae import constants as C


# Two ALA residues + one water + one hydrogen + one alternate conformation.
# Filtering must drop the water, the hydrogen, and collapse the altloc.
MINI_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.008   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.008   2.650   0.000  1.00  0.00           O
ATOM      5  CB AALA A   1       1.658  -0.800  -1.280  0.60  0.00           C
ATOM      6  CB BALA A   1       1.700  -0.850  -1.300  0.40  0.00           C
ATOM      7  HA  ALA A   1       1.400   0.900   0.500  1.00  0.00           H
ATOM      8  N   ALA A   2       3.330   0.100   0.200  1.00  0.00           N
ATOM      9  CA  ALA A   2       4.788   0.100   0.200  1.00  0.00           C
ATOM     10  C   ALA A   2       5.338   1.520   0.200  1.00  0.00           C
ATOM     11  O   ALA A   2       5.338   2.750   0.200  1.00  0.00           O
ATOM     12  CB  ALA A   2       4.988  -0.700  -1.080  1.00  0.00           C
HETATM   13  O   HOH A 101       9.000   9.000   9.000  1.00  0.00           O
END
"""


def _write(tmp_path):
    p = tmp_path / "mini.pdb"
    p.write_text(MINI_PDB)
    return str(p)


def test_filters_water_hydrogen_altloc(tmp_path):
    ps = parse_structure(_write(tmp_path), pdb_id="MINI")
    assert ps is not None
    # 2 residues x 5 heavy atoms = 10 (water/H removed, altloc collapsed to one CB)
    assert ps.n_residues == 2
    assert ps.n_atoms == 10
    # no hydrogens survived
    assert "H" not in [e.upper() for e in ps.element_symbol]
    # exactly one CB per residue (altloc collapsed)
    cb = C.ATOM_NAME_TO_IDX["CB"]
    assert int((ps.atom_name_idx == cb).sum()) == 2


def test_records_filtering_counts(tmp_path):
    ps = parse_structure(_write(tmp_path), pdb_id="MINI")
    rec = ps.record
    assert rec["n_water_atoms"] == 1
    assert rec["n_hydrogen_atoms"] == 1
    assert rec["n_altloc_atoms"] >= 1
    assert rec["pdb_id"] == "MINI"


def test_sequence_and_bonds(tmp_path):
    ps = parse_structure(_write(tmp_path), pdb_id="MINI")
    assert ps.sequence == "AA"
    # every bond connects atoms in the same or consecutive residues
    for a, b in ps.bonds:
        assert abs(int(ps.res_pos[a]) - int(ps.res_pos[b])) <= 1
    # includes the peptide bond C(res0)-N(res1)
    assert ps.bonds.shape[0] >= 8


def test_perceive_bonds_symmetry_free():
    coords = np.array([[0, 0, 0], [1.5, 0, 0], [10, 0, 0]], dtype=np.float32)
    res_pos = np.array([0, 0, 1], dtype=np.int64)
    bonds = perceive_bonds(coords, ["C", "C", "C"], res_pos)
    assert bonds.shape[0] == 1        # only the close pair bonds
    assert tuple(bonds[0]) == (0, 1)
