"""Loss-term correctness tests."""

import numpy as np
import torch

from molae.synthetic import make_synthetic_ala
from molae.losses import (
    coord_loss_single, bond_loss_single, clash_loss_single,
    chirality_loss_single, distance_loss_single,
)


def _synth_tensors():
    m = make_synthetic_ala(n_res=3)
    coords = torch.tensor(m["coords"], dtype=torch.float32)
    bonds = torch.tensor(m["bonds"], dtype=torch.long)
    return m, coords, bonds


def test_coord_loss_zero_on_identity():
    _, coords, _ = _synth_tensors()
    assert coord_loss_single(coords, coords).item() < 1e-6


def test_coord_loss_invariant_to_rotation():
    _, coords, _ = _synth_tensors()
    theta = 0.7
    R = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]], dtype=torch.float32)
    rotated = coords @ R.T + torch.tensor([3.0, 1.0, -2.0])
    assert coord_loss_single(rotated, coords).item() < 1e-4


def test_bond_loss_zero_on_identity_positive_when_stretched():
    _, coords, bonds = _synth_tensors()
    assert bond_loss_single(coords, coords, bonds).item() < 1e-8
    stretched = coords * 1.2
    assert bond_loss_single(stretched, coords, bonds).item() > 0.0


def test_clash_loss_zero_when_separated():
    _, coords, _ = _synth_tensors()
    n = coords.shape[0]
    bonded = torch.zeros(n, n)
    # atoms are >=1.2 A apart within residues and 10 A across -> no clashes at 1.0
    assert clash_loss_single(coords, bonded, clash_dist=1.0).item() < 1e-6


def test_clash_loss_positive_when_overlapping():
    coords = torch.tensor([[0, 0, 0], [0.3, 0, 0], [5, 0, 0]], dtype=torch.float32)
    bonded = torch.zeros(3, 3)
    assert clash_loss_single(coords, bonded, clash_dist=1.0).item() > 0.0


def test_chirality_loss_detects_reflection():
    m, coords, _ = _synth_tensors()
    from molae.dataset import _chirality_centers
    centers = torch.tensor(_chirality_centers(m["atom_name_idx"], m["res_pos"]), dtype=torch.long)
    assert centers.shape[0] == 3
    # |signed volume| of the synthetic residue exceeds the margin, so a correct
    # chirality gives zero hinge loss.
    assert chirality_loss_single(coords, coords, centers).item() < 1e-6
    reflected = coords.clone()
    reflected[:, 2] *= -1.0
    # reflected has opposite signed volume -> hinge penalty exceeds the margin
    assert chirality_loss_single(reflected, coords, centers).item() > 1.0


def test_distance_loss_zero_on_identity():
    _, coords, _ = _synth_tensors()
    assert distance_loss_single(coords, coords).item() < 1e-6
