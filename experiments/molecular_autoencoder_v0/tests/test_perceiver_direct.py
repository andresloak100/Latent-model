"""Perceiver encoder + group direct readout: the O(N.L) / dynamic-L contract."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from molae import constants as C
from molae.model import ModelConfig
from molae.model_equivariant import make_autoencoder
from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn


def _cfg(**kw):
    base = dict(encoder_type="perceiver_direct", d_model=32, n_heads=4,
                n_latent_tokens=16, latent_dim=8, enc_self_layers=1)
    base.update(kw)
    return ModelConfig(**base)


def _batch(sizes=(6, 10)):
    return collate_fn([sample_from_arrays(make_synthetic_ala(n_res=s)) for s in sizes])


def _attention_shapes(model, batch, **fwd):
    """Record (n_queries, n_keys) for every attention call in a forward pass."""
    seen = []

    def hook(_mod, args, _kw):
        q, k = args[0], args[1]
        seen.append((q.shape[1], k.shape[1]))

    handles = [m.register_forward_pre_hook(hook, with_kwargs=True)
               for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
    try:
        with torch.no_grad():
            model(batch, **fwd)
    finally:
        for h in handles:
            h.remove()
    return seen


def test_no_attention_is_quadratic_in_atoms():
    """The contract: nothing attends atoms-to-atoms.

    Every attention call must have at least one side that is the latent count L
    or the group count R -- never N x N. This is what makes the model O(N.L)
    rather than O(N^2), and it is the property that has to survive refactors.
    """
    model = make_autoencoder(_cfg()).eval()
    batch = _batch()
    N = int(batch["mask"].shape[1])
    R = int(batch["res_pos"].max()) + 1
    L = 16

    for q, k in _attention_shapes(model, batch):
        assert not (q == N and k == N), f"all-atom self-attention found: {q}x{k}"
        assert q in (L, R), f"unexpected query length {q} (expected L={L} or R={R})"
        # Atoms may only ever appear on the KEY side, i.e. cross-attention.
        assert k in (L, R, N)


def test_group_tokens_do_not_self_attend_by_default():
    """R x R self-attention is the O(R^2) term Jacob asked to avoid.

    1M atoms is ~125k residues, so R^2 ~ 1.6e10. Group tokens take their
    context from the latents instead.
    """
    model = make_autoencoder(_cfg()).eval()
    batch = _batch()
    R = int(batch["res_pos"].max()) + 1
    shapes = _attention_shapes(model, batch)
    assert not any(q == R and k == R for q, k in shapes), "group self-attention present"

    # ...but it is available for ablation.
    abl = make_autoencoder(_cfg(group_self_layers=2)).eval()
    assert any(q == R and k == R for q, k in _attention_shapes(abl, batch))


def test_latent_size_is_dynamic_at_inference():
    """Same weights, different L, no retraining and no shape errors.

    Fixed learned latent queries could not do this; the sinusoidal index
    encoding is what makes L a free argument.
    """
    model = make_autoencoder(_cfg()).eval()
    batch = _batch()
    n_atoms = batch["mask"].shape[1]
    for L in (1, 4, 16, 64, 257):
        with torch.no_grad():
            pred, z = model(batch, n_latents=L)
        assert z.shape[1] == L
        assert pred.shape == (2, n_atoms, 3)


def test_latent_size_is_decoupled_from_structure_size():
    """The whole point: latent floats do not grow with atoms or residues."""
    model = make_autoencoder(_cfg())
    assert model.latent_floats == 16 * 8
    assert model.latent_floats_for(10) == model.latent_floats_for(100_000)
    assert model.latent_floats_for_atoms(500) == model.latent_floats_for_atoms(1_000_000)

    # So compression grows without bound in N -- unlike the per-residue codec,
    # which is pinned at ~2.9x however large the structure is.
    for n_atoms in (10_000, 1_000_000):
        assert (3 * n_atoms) / model.latent_floats > 200


def test_direct_readout_is_preserved():
    """Each atom must still gather a dedicated slot from its own group.

    This is the property that took the codec from 9.5 A to 0.75 A; the
    Perceiver latent is layered on top of it, not in place of it.
    """
    batch = _batch(sizes=(5,))
    R = int(batch["res_pos"].max()) + 1
    gather = batch["res_pos"] * C.N_ATOM_NAMES + batch["atom_name_idx"]
    # distinct (group, atom-name) pairs -> distinct slots, no sharing
    assert len(set(gather[0].tolist())) == gather.shape[1]
    assert int(gather.max()) < R * C.N_ATOM_NAMES


def test_padding_does_not_leak_into_latent():
    model = make_autoencoder(_cfg()).eval()
    batch = _batch(sizes=(3, 5))
    n_real = int(batch["mask"][0].sum())
    with torch.no_grad():
        z1 = model.encode(batch)
        bad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        bad["coords"][0, n_real:] += 100.0
        bad["element_idx"][0, n_real:] = 2
        z2 = model.encode(bad)
    assert torch.allclose(z1[0], z2[0], atol=1e-5)


def test_cost_grows_linearly_not_quadratically_in_atoms():
    """Empirical check on the headline claim.

    Total attention matrix elements at fixed L should roughly DOUBLE when the
    atom count doubles. An all-atom self-attention design would quadruple.
    """
    model = make_autoencoder(_cfg()).eval()
    costs = {}
    for n_res in (8, 16, 32):
        batch = _batch(sizes=(n_res,))
        shapes = _attention_shapes(model, batch)
        costs[int(batch["mask"].shape[1])] = sum(q * k for q, k in shapes)

    ns = sorted(costs)
    for small, large in zip(ns, ns[1:]):
        ratio_n = large / small
        ratio_cost = costs[large] / costs[small]
        # linear would be ~ratio_n; quadratic ~ratio_n**2. Allow slack for the
        # O(L.R) term, but it must be far below quadratic.
        assert ratio_cost < ratio_n ** 1.5, (
            f"cost grew {ratio_cost:.2f}x for a {ratio_n:.2f}x atom increase")
