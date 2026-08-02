"""Chain awareness in the direct decoder.

The decoder used to see a single sequential ``res_pos`` spanning the whole
assembly, with no marker at chain boundaries -- an adjacency prior that is
true along a chain and false between them. These tests pin the index
arithmetic that fixes it, and pin the default OFF so every earlier result
stays reproducible.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molae.model import ModelConfig  # noqa: E402
from molae.model_direct import DirectAutoencoder, residue_chain_index  # noqa: E402
from molae.dataset import collate_fn  # noqa: E402
from molae import constants as C  # noqa: E402


def _sample(n_res_per_chain=(3, 4), atoms_per_res=4):
    """Two chains laid out contiguously, as the parser emits them."""
    res_pos, chain_idx = [], []
    r = 0
    for ci, nres in enumerate(n_res_per_chain):
        for _ in range(nres):
            res_pos += [r] * atoms_per_res
            chain_idx += [ci] * atoms_per_res
            r += 1
    n = len(res_pos)
    return {
        "n_atoms": n,
        "pdb_id": "TEST",
        "chain_id": "A",
        "element_symbol": ["C"] * n,
        "coords": torch.randn(n, 3),
        "res_pos": torch.tensor(res_pos),
        "chain_idx": torch.tensor(chain_idx),
        "residue_idx": torch.zeros(n, dtype=torch.long),
        "element_idx": torch.zeros(n, dtype=torch.long),
        "atom_name_idx": torch.zeros(n, dtype=torch.long),
        "slot_idx": torch.arange(n) % atoms_per_res,
        "res_seq": torch.tensor(res_pos),
        "bonds": torch.zeros((0, 2), dtype=torch.long),
        "chirality_centers": torch.zeros((0, 4), dtype=torch.long),
    }


def test_within_chain_position_restarts_at_each_chain():
    res_pos = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    chain = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    rchain, within = residue_chain_index(res_pos, chain, R=4, max_chains=4)
    assert rchain[0].tolist() == [0, 0, 1, 1]
    # Residues 2 and 3 are chain 1's first and second, not the assembly's 3rd/4th.
    assert within[0].tolist() == [0, 1, 0, 1]


def test_single_chain_within_equals_global():
    res_pos = torch.tensor([[0, 1, 2, 3, 4]])
    chain = torch.zeros(1, 5, dtype=torch.long)
    rchain, within = residue_chain_index(res_pos, chain, R=5, max_chains=4)
    assert rchain[0].tolist() == [0] * 5
    assert within[0].tolist() == [0, 1, 2, 3, 4]


def test_chain_ids_beyond_max_chains_are_clamped_not_out_of_range():
    res_pos = torch.tensor([[0, 1]])
    chain = torch.tensor([[0, 99]])
    rchain, within = residue_chain_index(res_pos, chain, R=2, max_chains=4)
    assert int(rchain.max()) == 3
    assert int(within.min()) >= 0


def test_padding_residues_cannot_corrupt_a_real_chain_offset():
    """R is the batch max; a short structure's tail residues are never real."""
    res_pos = torch.tensor([[0, 0, 1, 1]])
    chain = torch.tensor([[0, 0, 1, 1]])
    rchain, within = residue_chain_index(res_pos, chain, R=6, max_chains=4)
    # Residues 0 and 1 are real and keep the offsets they would have alone.
    assert within[0, 0] == 0 and within[0, 1] == 0
    assert rchain[0, 0] == 0 and rchain[0, 1] == 1


def _cfg(chain_aware):
    return ModelConfig(d_model=32, n_heads=2, latent_dim=4, enc_self_layers=1,
                       dec_self_layers=1, max_res_pos=64, max_chains=4,
                       use_slot_emb=True, encoder_type="direct",
                       dec_chain_aware=chain_aware)


def test_flag_defaults_off_so_existing_results_reproduce():
    assert ModelConfig().dec_chain_aware is False
    m = DirectAutoencoder(_cfg(False))
    assert not hasattr(m.decoder, "chain_emb")


def test_chain_aware_model_runs_and_adds_only_embeddings():
    plain, aware = DirectAutoencoder(_cfg(False)), DirectAutoencoder(_cfg(True))
    extra = aware.num_parameters() - plain.num_parameters()
    d = _cfg(True).d_model
    assert extra == d * (4 + 64)          # chain table + within-chain table

    batch = collate_fn([_sample()])
    out, z = aware(batch)
    assert out.shape == batch["coords"].shape
    assert torch.isfinite(out).all()


def test_chain_awareness_changes_the_output():
    """A flag that cannot change the prediction cannot be tested by an A/B."""
    torch.manual_seed(0)
    m = DirectAutoencoder(_cfg(True)).eval()
    batch = collate_fn([_sample()])
    with torch.no_grad():
        a, _ = m(batch)
        # Same atoms, same residues -- relabelled as ONE chain instead of two.
        merged = dict(batch)
        merged["chain_idx"] = torch.zeros_like(batch["chain_idx"])
        b, _ = m(merged)
    assert not torch.allclose(a, b, atol=1e-6)


def test_plain_decoder_ignores_chain_labels():
    """The control: without the flag, chain identity provably does not reach it."""
    torch.manual_seed(0)
    m = DirectAutoencoder(_cfg(False)).eval()
    batch = collate_fn([_sample()])
    with torch.no_grad():
        a, _ = m(batch)
        merged = dict(batch)
        merged["chain_idx"] = torch.zeros_like(batch["chain_idx"])
        b, _ = m(merged)
    # The ENCODER still sees chain_idx, so this holds only for the decoder.
    z = m.encode(batch)
    with torch.no_grad():
        d1 = m.decode(z, batch)
        d2 = m.decode(z, merged)
    assert torch.allclose(d1, d2, atol=1e-6)
    assert a.shape == b.shape
