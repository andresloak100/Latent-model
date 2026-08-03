"""Angle-loss triples: vectorised, cached, and degenerate-input safe.

py-spy caught the training thread inside the original Python-loop build, GIL
held and the GPU at 2-4% utilisation. These pin the properties of the
replacement: same answer, no loop, and no degenerate triples from malformed
bond lists.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molae.losses import angle_loss_single, angle_triples, bond_loss_single  # noqa: E402


def _chain(n):
    return torch.stack([torch.arange(n - 1), torch.arange(1, n)], 1)


def test_triples_are_real_angles():
    n = 60
    bonds = _chain(n)
    tri = angle_triples(bonds, n)
    edges = {(int(a), int(b)) for a, b in bonds.tolist()}
    edges |= {(b, a) for a, b in edges}
    for i, j, k in tri.tolist():
        assert (i, j) in edges and (j, k) in edges
        assert i != k and i != j and j != k


def test_chain_has_exactly_one_angle_per_interior_atom():
    n = 10
    assert angle_triples(_chain(n), n).shape[0] == n - 2


def test_precomputed_triples_give_the_same_loss():
    n = 40
    bonds = _chain(n)
    torch.manual_seed(0)
    pred, target = torch.randn(n, 3), torch.randn(n, 3)
    a = angle_loss_single(pred, target, bonds)
    b = angle_loss_single(pred, target, bonds, triples=angle_triples(bonds, n))
    assert torch.allclose(a, b, atol=1e-7)


def test_self_loops_and_duplicate_bonds_produce_no_degenerate_triples():
    """A self-loop makes an atom its own neighbour; a duplicate lists one twice."""
    bonds = torch.tensor([[0, 1], [1, 2], [1, 2], [3, 3], [2, 3]])
    tri = angle_triples(bonds, 4)
    for i, j, k in tri.tolist():
        assert i != j and j != k and i != k


def test_fan_out_is_capped_per_centre():
    """One atom bonded to many must not dominate the angle set."""
    n = 12
    hub = torch.stack([torch.zeros(n - 1, dtype=torch.long),
                       torch.arange(1, n)], 1)
    tri = angle_triples(hub, n, max_per_centre=4)
    assert tri.shape[0] == 6                    # C(4,2), not C(11,2)=55


def test_max_angles_truncates():
    n = 500
    assert angle_triples(_chain(n), n, max_angles=50).shape[0] == 50


def test_angle_catches_what_bond_length_cannot():
    """The reason the term exists: identical bond lengths, one joint straightened."""
    target = torch.tensor([[0., 0, 0], [1.5, 0, 0], [2.3, 1.3, 0], [3.8, 1.3, 0]])
    pred = torch.tensor([[0., 0, 0], [1.5, 0, 0], [3.026, 0, 0], [4.526, 0, 0]])
    bonds = torch.tensor([[0, 1], [1, 2], [2, 3]])
    assert float(bond_loss_single(pred, target, bonds)) < 1e-4
    assert float(angle_loss_single(pred, target, bonds)) > 0.1


def test_empty_and_tiny_bond_lists_return_zero():
    for bonds in (torch.zeros((0, 2), dtype=torch.long),
                  torch.tensor([[0, 1]])):
        pred = torch.randn(4, 3)
        assert float(angle_loss_single(pred, pred, bonds)) == 0.0
