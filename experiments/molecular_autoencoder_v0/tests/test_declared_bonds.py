"""Bonds the depositor declared in mmCIF struct_conn.

This exists because of the largest defect found in this pipeline, and one that
no amount of the debugging that preceded it could have surfaced: we perceived
ZERO disulfide bonds. 1BPI deposits three at ~2.0 A and we found none, because
the ``|delta res_pos| <= 1`` adjacency rule excludes cysteines 21-50 residues
apart. Thirty were missing across the 47-structure cache, and every result in
the project -- Milestone 1 included -- had a bond loss that never restrained a
disulfide.

A valence invariant cannot catch this: missing bonds lower degrees, they never
raise them. Nor could a geometric heuristic, because the pair looks exactly
like any other 2.0 A contact. The mmCIF states it outright, which is the
lesson: read the file rather than infer what it already says.
"""

import numpy as np
import pytest

from molae.parsing import declared_bonds, merge_bonds


class _Addr:
    def __init__(self, chain, seqid, atom):
        self.chain_name = chain
        self.atom_name = atom
        self.res_id = type("R", (), {"seqid": type("S", (), {"num": seqid})()})()


class _Conn:
    def __init__(self, kind, p1, p2):
        self.type = kind
        self.partner1 = p1
        self.partner2 = p2


class _Struct:
    def __init__(self, conns):
        self.connections = conns


ADDR = {("A", 5, "SG"): 0, ("A", 55, "SG"): 1, ("A", 14, "SG"): 2,
        ("B", 501, "C1"): 3, ("A", 30, "ND2"): 4, ("A", 40, "NE2"): 5}


def _conn(kind, a, b):
    return _Conn(kind, _Addr(*a), _Addr(*b))


def test_disulfide_is_read():
    st = _Struct([_conn("ConnectionType.Disulf", ("A", 5, "SG"), ("A", 55, "SG"))])
    b = declared_bonds(st, ADDR)
    assert b.tolist() == [[0, 1]]


def test_covalent_link_is_read():
    """A glycan anchored to ASN -- stated, not guessed."""
    st = _Struct([_conn("ConnectionType.Covale", ("A", 30, "ND2"), ("B", 501, "C1"))])
    assert declared_bonds(st, ADDR).tolist() == [[3, 4]]


def test_metal_coordination_is_ignored():
    """Coordination is not a covalent bond; perceiving it would create
    topology the model should not learn."""
    st = _Struct([_conn("ConnectionType.MetalC", ("A", 5, "SG"), ("A", 55, "SG"))])
    assert declared_bonds(st, ADDR).size == 0


def test_hydrogen_bond_is_ignored():
    st = _Struct([_conn("ConnectionType.Hydrog", ("A", 30, "ND2"), ("A", 40, "NE2"))])
    assert declared_bonds(st, ADDR).size == 0


def test_partner_filtered_out_is_skipped():
    """A dropped chain, excluded ligand or collapsed duplicate leaves a
    dangling partner; it must be skipped, not crash or mis-index."""
    st = _Struct([_conn("ConnectionType.Disulf", ("A", 5, "SG"), ("Z", 999, "SG"))])
    assert declared_bonds(st, ADDR).size == 0


def test_self_bond_is_skipped():
    st = _Struct([_conn("ConnectionType.Disulf", ("A", 5, "SG"), ("A", 5, "SG"))])
    assert declared_bonds(st, ADDR).size == 0


def test_duplicate_declarations_collapse():
    st = _Struct([_conn("ConnectionType.Disulf", ("A", 5, "SG"), ("A", 55, "SG")),
                  _conn("ConnectionType.Disulf", ("A", 55, "SG"), ("A", 5, "SG"))])
    assert declared_bonds(st, ADDR).tolist() == [[0, 1]]


def test_merge_is_a_deduplicated_sorted_union():
    perceived = np.array([[0, 1], [2, 3]], dtype=np.int64)
    stated = np.array([[2, 3], [4, 5]], dtype=np.int64)
    out = merge_bonds(perceived, stated)
    assert out.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert (out[:, 0] < out[:, 1]).all()


def test_merge_handles_empty_sides():
    empty = np.zeros((0, 2), dtype=np.int64)
    some = np.array([[0, 1]], dtype=np.int64)
    assert merge_bonds(some, empty).tolist() == [[0, 1]]
    assert merge_bonds(empty, some).tolist() == [[0, 1]]
    assert merge_bonds(empty, empty).size == 0


def test_declared_bonds_bypass_the_geometric_filters():
    """A depositor's statement is not a guess and is not second-guessed: it is
    merged after the distance floor and shortcut pruning, not through them."""
    perceived = np.zeros((0, 2), dtype=np.int64)
    stated = np.array([[0, 1]], dtype=np.int64)
    assert merge_bonds(perceived, stated).tolist() == [[0, 1]]
