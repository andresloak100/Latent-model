import numpy as np

from molae.synthetic import make_synthetic_ala
from molae import constants as C


def test_atom_and_bond_counts():
    m = make_synthetic_ala(n_res=3)
    assert m["coords"].shape == (15, 3)          # 5 atoms x 3 residues
    assert m["bonds"].shape[0] == 12             # 4 intra-residue bonds x 3


def test_known_bond_lengths():
    m = make_synthetic_ala(n_res=2)
    lengths = {}
    for a, b in m["bonds"]:
        na, nb = m["atom_name"][a], m["atom_name"][b]
        d = float(np.linalg.norm(m["coords"][a] - m["coords"][b]))
        lengths[frozenset((na, nb))] = d
    assert abs(lengths[frozenset(("N", "CA"))] - 1.458) < 1e-2
    assert abs(lengths[frozenset(("CA", "C"))] - 1.523) < 1e-2
    assert abs(lengths[frozenset(("C", "O"))] - 1.230) < 1e-2
    assert abs(lengths[frozenset(("CA", "CB"))] - 1.523) < 1e-2


def test_no_cross_residue_bonds():
    # residues are spaced 10 A apart, so no bond should cross residues
    m = make_synthetic_ala(n_res=3)
    rp = m["res_pos"]
    for a, b in m["bonds"]:
        assert rp[a] == rp[b]


def test_all_atoms_in_vocab():
    m = make_synthetic_ala(n_res=2)
    assert set(m["atom_name"]).issubset(set(C.ATOM_NAMES))
    assert m["residue_idx"].min() >= 2  # standard-AA region of the vocab
