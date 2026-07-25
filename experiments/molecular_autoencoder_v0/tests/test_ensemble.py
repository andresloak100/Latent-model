"""Multi-model (NMR ensemble) parsing + the latent-suitability measurements."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from molae.synthetic import make_synthetic_ala  # noqa: E402
from latent_suitability import spearman, smoothness  # noqa: E402


def test_spearman_matches_known_cases():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    # monotone but non-linear -> rank correlation is still exactly 1
    assert spearman(a, a ** 3) == pytest.approx(1.0)


def test_spearman_handles_constant_input():
    a = np.array([1.0, 2.0, 3.0])
    assert np.isnan(spearman(a, np.zeros(3)))


def test_smoothness_is_one_for_an_isometric_latent():
    """If the latent is a copy of the coordinates, rho must be exactly 1.

    Pins the direction of the metric: latent distance tracking structural
    distance is the property that makes a diffusion path in latent space
    correspond to a physical path in structure space.

    Uses isotropic scalings of one shape, the case where Kabsch provably cannot
    interfere: for centred X, the optimal rotation between X and sX is exactly
    the identity, so Kabsch RMSD is |s_i - s_j|*||X||/sqrt(N) while the latent
    distance is |s_i - s_j|*||X||_F. Both are proportional to |s_i - s_j|, so
    the ranks must agree exactly.

    (A directional displacement would NOT give 1.0 -- Kabsch absorbs part of it
    as rotation while the raw latent norm does not, which is precisely why the
    real script superimposes ensemble members before encoding.)
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 3))
    base -= base.mean(axis=0, keepdims=True)
    trues = [s * base for s in (1.0, 1.4, 2.1, 3.3, 5.0)]
    zs = [torch.from_numpy(t).unsqueeze(0) for t in trues]
    rho, n_pairs = smoothness(zs, trues)
    assert n_pairs == 10
    assert rho == pytest.approx(1.0)


def test_smoothness_needs_enough_pairs():
    trues = [np.zeros((4, 3)), np.ones((4, 3))]
    zs = [torch.zeros(1, 4, 3), torch.ones(1, 4, 3)]
    rho, n_pairs = smoothness(zs, trues)
    assert np.isnan(rho) and n_pairs == 0


NMR_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_nmr"


@pytest.mark.skipif(not (NMR_DIR / "1D3Z.cif").exists(),
                    reason="NMR ensembles not downloaded (see scripts/fetch_nmr_ids.py)")
def test_bond_lengths_are_conformer_invariant_on_real_ensembles():
    """The assumption the interpolation test rests on, checked on real data.

    Using an endpoint's true coordinates as the bond-length reference for an
    interpolated midpoint is only valid if bond lengths barely move between
    conformers. Measured here on a genuine deposited NMR ensemble rather than
    on synthetic noise, which would only be testing the noise model.
    """
    from latent_suitability import load_ensemble
    structs = load_ensemble(str(NMR_DIR / "1D3Z.cif"), "1D3Z",
                            max_models=10, max_atoms=2000)
    assert len(structs) >= 5
    bonds = structs[0].bonds
    lengths = np.stack([
        np.linalg.norm(s.coords[bonds[:, 0]] - s.coords[bonds[:, 1]], axis=1)
        for s in structs
    ])
    spread = float(np.std(lengths, axis=0).mean())
    # Must be far below the ~0.16 A bond error the codec reports, or the
    # reference would be swamped by conformer-to-conformer variation.
    assert spread < 0.03, f"bond lengths vary {spread:.4f} A across conformers"


@pytest.mark.skipif(not (NMR_DIR / "1D3Z.cif").exists(),
                    reason="NMR ensembles not downloaded")
def test_ensemble_members_are_superimposed_and_still_differ():
    """Superimposition must remove frame differences without flattening the
    conformational signal the whole experiment depends on."""
    from latent_suitability import load_ensemble
    from molae.alignment import kabsch_rmsd_numpy
    structs = load_ensemble(str(NMR_DIR / "1D3Z.cif"), "1D3Z",
                            max_models=6, max_atoms=2000)
    ref = structs[0].coords.astype(np.float64)
    for s in structs[1:]:
        c = s.coords.astype(np.float64)
        # Already in model 0's frame: raw deviation == Kabsch-aligned deviation.
        raw = float(np.sqrt(((c - c.mean(0) - (ref - ref.mean(0))) ** 2).sum(1).mean()))
        assert raw == pytest.approx(kabsch_rmsd_numpy(c, ref), abs=1e-6)
    # ...but the conformers are still genuinely distinct.
    assert kabsch_rmsd_numpy(structs[1].coords.astype(np.float64), ref) > 0.1


def test_parse_structure_rejects_out_of_range_model():
    import gemmi
    from molae.parsing import parse_structure
    st = gemmi.Structure()
    st.add_model(gemmi.Model("1"))
    path = Path(__file__).parent / "_tmp_one_model.pdb"
    st.write_pdb(str(path))
    try:
        with pytest.raises(IndexError):
            parse_structure(str(path), pdb_id="TEST", model_index=7)
    finally:
        path.unlink(missing_ok=True)
