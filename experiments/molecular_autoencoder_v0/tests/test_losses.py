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


def test_distance_loss_single_atom_is_zero_not_nan():
    """Regression: n<2 has no atom pairs -> must return 0, not NaN."""
    one = torch.zeros(1, 3)
    v = distance_loss_single(one, one)
    assert torch.isfinite(v).all() and v.item() == 0.0


def test_clash_loss_matches_the_dense_mask_it_replaced():
    """Sparse bond list must give bit-identical results to the old (N,N) mask.

    The dense mask cost 34 MB/structure at 3000 atoms and 382 MB at 10,000 --
    per sample, in a dataloader worker -- which blocked complex-scale training.
    Replacing it is only safe if the loss value is unchanged, including through
    the max_atoms subsample path where bonds must be re-indexed.
    """
    import torch.nn.functional as F
    from molae.losses import clash_loss_single

    def dense_reference(pred, bonds, clash_dist=1.5, max_atoms=1200):
        n = pred.shape[0]
        bm = torch.zeros(n, n)
        for a, b in bonds.tolist():
            bm[a, b] = 1.0
            bm[b, a] = 1.0
        if n > max_atoms:
            idx = torch.linspace(0, n - 1, max_atoms).long()
            pred, bm = pred[idx], bm[idx][:, idx]
        d = torch.cdist(pred, pred)
        iu = torch.triu_indices(pred.shape[0], pred.shape[0], offset=1)
        dv, b = d[iu[0], iu[1]], bm[iu[0], iu[1]]
        return (F.relu(clash_dist - dv) * (1.0 - b)).sum() / (1.0 - b).sum().clamp_min(1.0)

    torch.manual_seed(0)
    for n, nb in [(40, 30), (200, 190), (1500, 1400), (60, 0)]:
        pred = torch.randn(n, 3) * 3.0
        if nb:
            a, b = torch.randint(0, n, (nb,)), torch.randint(0, n, (nb,))
            keep = a != b
            bonds = torch.stack([a[keep], b[keep]], 1)
        else:
            bonds = torch.zeros(0, 2, dtype=torch.long)
        assert torch.allclose(dense_reference(pred, bonds), clash_loss_single(pred, bonds),
                              atol=1e-6), f"mismatch at n={n}, bonds={nb}"


def test_no_dense_pair_matrix_in_a_sample():
    """No per-sample tensor may be quadratic in atom count.

    Guards the property directly: a 600-atom structure must not carry anything
    of size ~600^2. Without this, a future change could reintroduce the dense
    mask and only be noticed when complexes OOM.
    """
    from molae.dataset import sample_from_arrays
    from molae.synthetic import make_synthetic_ala
    s = sample_from_arrays(make_synthetic_ala(n_res=40))
    n = int(s["n_atoms"])
    for k, v in s.items():
        if torch.is_tensor(v):
            assert v.numel() < n * n / 4, f"{k} is quadratic in atom count: {tuple(v.shape)}"
