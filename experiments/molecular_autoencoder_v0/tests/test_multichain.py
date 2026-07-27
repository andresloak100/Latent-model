"""Multi-chain parsing and the decoder-slot aliasing it would otherwise cause."""

import numpy as np
import pytest
import torch

from molae import constants as C
from molae.parsing import perceive_bonds
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import ModelConfig
from molae.model_equivariant import make_autoencoder
from molae.synthetic import make_synthetic_ala


def _two_chain_arrays(n_res=4, gap=40.0):
    """Two copies of the synthetic peptide, offset in space, as separate chains.

    res_pos is GLOBAL across the chains, which is what the parser now emits.
    """
    d = make_synthetic_ala(n_res=n_res)
    base = {k: np.asarray(v) for k, v in d.items()}
    n = len(base["res_pos"])
    out = {}
    for k in ("element_idx", "residue_idx", "atom_name_idx", "res_seq"):
        src = base[k] if k in base else base["res_pos"]
        out[k] = np.concatenate([src, src])
    out["res_pos"] = np.concatenate([base["res_pos"], base["res_pos"] + n_res])
    out["chain_idx"] = np.concatenate([np.zeros(n, np.int64), np.ones(n, np.int64)])
    out["coords"] = np.concatenate([base["coords"], base["coords"] + gap]).astype(np.float32)
    out["element_symbol"] = list(base["element_symbol"]) * 2
    out["atom_name"] = list(base["atom_name"]) * 2
    out["bonds"] = np.concatenate([base["bonds"], base["bonds"] + n])
    out["pdb_id"] = "TWOCH"
    out["chain_id"] = "A,B"
    out["sequence"] = "A" * (2 * n_res)
    return out, n


def test_global_res_pos_prevents_decoder_slot_aliasing():
    """The bug this design avoids.

    The direct decoder indexes its slot bank by (res_pos, atom_name). With
    PER-CHAIN numbering, chain B residue 0 and chain A residue 0 would gather
    the SAME slot and be forced to identical coordinates. Global res_pos gives
    every residue its own slots.
    """
    arrays, n = _two_chain_arrays()
    batch = collate_fn([sample_from_arrays(arrays)])
    cfg = ModelConfig(encoder_type="direct", d_model=32, n_heads=2, latent_dim=4,
                      enc_self_layers=1, dec_self_layers=1, max_chains=4)
    model = make_autoencoder(cfg).eval()
    with torch.no_grad():
        pred, _ = model(batch)

    # Chain A's atoms and chain B's atoms must not be forced to coincide.
    a, b = pred[0, :n], pred[0, n:]
    assert not torch.allclose(a, b, atol=1e-4), "chains collapsed onto the same slots"

    # And the gather indices themselves must be disjoint between chains.
    n_slots = C.N_ATOM_NAMES
    gi = (batch["res_pos"] * n_slots + batch["atom_name_idx"])[0]
    assert set(gi[:n].tolist()).isdisjoint(set(gi[n:].tolist()))


def test_per_chain_numbering_would_alias():
    """Control: shows the failure is real, not hypothetical."""
    arrays, n = _two_chain_arrays()
    arrays["res_pos"] = np.concatenate([arrays["res_pos"][:n], arrays["res_pos"][:n]])
    batch = collate_fn([sample_from_arrays(arrays)])
    n_slots = C.N_ATOM_NAMES
    gi = (batch["res_pos"] * n_slots + batch["atom_name_idx"])[0]
    assert set(gi[:n].tolist()) == set(gi[n:].tolist()), (
        "with per-chain numbering the two chains share every slot -- the bug")


def test_bond_perception_does_not_bridge_chains():
    """Global res_pos makes chain ends numerically ADJACENT, so the last residue
    of chain A and the first of chain B look like consecutive residues to the
    covalent-radius rule. If two such atoms happen to pack within bonding
    distance — which happens constantly at a real protein-protein interface —
    the parser would invent a covalent bond across the interface.

    Minimal explicit case: four carbons, global residues 0/0/1/1, chains 0/0/1/1,
    with one cross-chain pair placed at 1.4 A (a C-C bond length).
    """
    coords = np.array([[0.0, 0, 0], [1.4, 0, 0],      # chain 0, residue 0
                       [2.8, 0, 0], [4.2, 0, 0]],     # chain 1, residue 1
                      dtype=np.float64)
    res_pos = np.array([0, 0, 1, 1], dtype=np.int64)
    chain = np.array([0, 0, 1, 1], dtype=np.int64)
    elems = ["C", "C", "C", "C"]

    without = perceive_bonds(coords, elems, res_pos)
    with_guard = perceive_bonds(coords, elems, res_pos, chain)
    cross_without = sum(1 for i, j in without if chain[i] != chain[j])
    cross_with = sum(1 for i, j in with_guard if chain[i] != chain[j])
    assert cross_without > 0, "test needs a case that would bridge chains"
    assert cross_with == 0
    # Intra-chain bonds must survive the guard untouched.
    assert sum(1 for i, j in with_guard if chain[i] == chain[j]) == \
           sum(1 for i, j in without if chain[i] == chain[j])


def test_chain_embedding_is_optional_and_used():
    cfg1 = ModelConfig(encoder_type="direct", d_model=32, n_heads=2, latent_dim=4,
                       enc_self_layers=1, dec_self_layers=1)
    cfgN = ModelConfig(encoder_type="direct", d_model=32, n_heads=2, latent_dim=4,
                       enc_self_layers=1, dec_self_layers=1, max_chains=4)
    m1, mN = make_autoencoder(cfg1), make_autoencoder(cfgN)
    # max_chains=1 must not create the module, so old checkpoints still load.
    extra = set(mN.state_dict()) - set(m1.state_dict())
    assert extra == {"encoder.feat.chain_emb.weight"}

    # And when present it must actually change the encoding.
    arrays, n = _two_chain_arrays()
    batch = collate_fn([sample_from_arrays(arrays)])
    mN.eval()
    with torch.no_grad():
        z1 = mN.encode(batch)
        flipped = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        flipped["chain_idx"] = 1 - flipped["chain_idx"]
        z2 = mN.encode(flipped)
    assert not torch.allclose(z1, z2, atol=1e-6)


def test_single_chain_path_is_unchanged():
    """Regression: default sample_from_arrays with no chain_idx still works and
    reports one chain, so every existing dataset and checkpoint is unaffected."""
    d = make_synthetic_ala(n_res=3)
    s = sample_from_arrays(d)
    assert "chain_idx" in s
    assert int(s["chain_idx"].sum()) == 0
    batch = collate_fn([s])
    cfg = ModelConfig(encoder_type="direct", d_model=32, n_heads=2, latent_dim=4,
                      enc_self_layers=1, dec_self_layers=1)
    with torch.no_grad():
        pred, _ = make_autoencoder(cfg).eval()(batch)
    assert pred.shape == (1, int(batch["n_atoms"][0]), 3)
