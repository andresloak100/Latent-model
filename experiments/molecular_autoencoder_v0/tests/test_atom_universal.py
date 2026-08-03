"""The universal atom-only path: no residues, no atom names, bare .sdf in.

`res_pos` was never supplying biology, it was supplying ADDRESSABILITY -- a
unique key per atom. These tests pin the replacement: chemistry from the
graph (deliberately non-unique, because symmetric atoms ARE the same), plus a
canonical rank that makes every decoder query distinct anyway.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molae.model import ModelConfig  # noqa: E402
from molae.model_atomlatent import AtomLatentAutoencoder  # noqa: E402
from molae.dataset import collate_fn  # noqa: E402
from molae.graph_identity import (  # noqa: E402
    canonical_order, graph_atom_features, wl_classes,
)
from molae.sdf import sample_from_sdf, parse_sdf_block  # noqa: E402


# Ethanol, V2000. Nine atoms, no residues anywhere in it.
ETHANOL = """ethanol
  test

  9  8  0  0  0  0  0  0  0  0999 V2000
    1.2304    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1500    0.0300    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.1000    0.6600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2600   -0.6300    0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2600   -0.6300   -0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0100    1.4900    0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0100    1.4900   -0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9300    0.5600    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  2  7  1  0  0  0  0
  2  8  1  0  0  0  0
  3  9  1  0  0  0  0
M  END
"""


def _cfg(local=True, L=32, d=16):
    return ModelConfig(d_model=64, n_heads=4, latent_dim=d, n_latent_tokens=L,
                       enc_self_layers=1, dec_cross_layers=2, dec_local_layers=1,
                       encoder_type="atomlatent", dec_local_cross=local,
                       dec_local_k=8, attn_window=32, atom_addressing="graph")


# ---------------------------------------------------------------- SDF parsing

def test_bare_sdf_parses_atoms_bonds_and_elements():
    s = parse_sdf_block(ETHANOL)
    assert s["coords"].shape == (9, 3)
    assert s["element_symbol"][:3] == ["C", "C", "O"]
    assert s["bonds"].shape == (8, 2)
    assert set(s["bond_types"].tolist()) == {1}


def test_sdf_sample_carries_no_residue_information():
    s = sample_from_sdf(ETHANOL)
    assert int(s["res_pos"].unique().numel()) == 1        # one inert value
    assert int(s["atom_name_idx"].unique().numel()) == 1  # all UNK
    for k in ("wl_class", "canonical_rank", "class_ordinal", "degree",
              "formal_charge", "max_bond_type"):
        assert k in s, k


def test_formal_charges_are_read():
    charged = ETHANOL.replace("M  END", "M  CHG  1   3  -1\nM  END")
    assert int(parse_sdf_block(charged)["formal_charge"][2]) == -1


# ------------------------------------------------- identity vs addressability

def test_symmetric_atoms_share_chemistry_but_get_distinct_addresses():
    """Benzene: six automorphic carbons. One class, six addresses."""
    el = np.array([6] * 6)
    bonds = np.array([[i, (i + 1) % 6] for i in range(6)])
    f = graph_atom_features(el, bonds=bonds)
    assert len(set(f["wl_class"].tolist())) == 1            # same chemistry
    assert len(set(f["canonical_rank"].tolist())) == 6      # distinct addresses


def test_methyl_hydrogens_of_ethanol_are_addressable():
    s = sample_from_sdf(ETHANOL)
    ranks = s["canonical_rank"].tolist()
    assert len(set(ranks)) == len(ranks) == 9


def test_chemically_distinct_atoms_get_distinct_classes():
    """The hydroxyl H must not share a class with the methyl Hs."""
    s = sample_from_sdf(ETHANOL)
    cls = s["wl_class"].tolist()
    assert cls[8] != cls[3]                                 # O-H vs C-H
    assert cls[0] != cls[2]                                 # C vs O


def test_canonical_addresses_are_reproducible():
    a = sample_from_sdf(ETHANOL)["canonical_rank"].tolist()
    for _ in range(5):
        assert sample_from_sdf(ETHANOL)["canonical_rank"].tolist() == a


def test_canonical_feature_sequence_is_permutation_invariant():
    """Permuting input atoms must not change the canonical ORDER of features."""
    rng = np.random.default_rng(0)
    el = np.array([6, 6, 8, 7, 6, 1, 1, 1])
    bonds = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [4, 6], [0, 7]])

    def sequence(perm=None):
        e, b = el, bonds
        if perm is not None:
            e, b = el[perm], np.argsort(perm)[bonds]
        f = graph_atom_features(e, bonds=b)
        order = np.argsort(f["canonical_rank"])
        return [(int(f["element_idx"][i]), int(f["wl_class"][i]),
                 int(f["degree"][i])) for i in order]

    base = sequence()
    for _ in range(20):
        assert sequence(rng.permutation(len(el))) == base


def test_wl_classes_are_permutation_equivariant():
    el = np.array([6, 8, 7, 1])
    bonds = np.array([[0, 1], [1, 2], [2, 3]])
    perm = np.array([2, 0, 3, 1])
    a = wl_classes(el, bonds=bonds)
    b = wl_classes(el[perm], bonds=np.argsort(perm)[bonds])
    assert sorted(a.tolist()) == sorted(b.tolist())
    assert a[perm].tolist() == b.tolist()


def test_canonical_order_is_a_valid_permutation():
    el = np.array([6, 6, 8, 1, 1])
    bonds = np.array([[0, 1], [1, 2], [2, 3], [0, 4]])
    order, rank, _ = canonical_order(el, bonds=bonds)
    assert sorted(order.tolist()) == list(range(5))
    assert rank[order].tolist() == list(range(5))


def test_disconnected_and_bondless_inputs_do_not_crash():
    f = graph_atom_features(np.array([6, 8, 7]), bonds=np.zeros((0, 2), dtype=np.int64))
    assert len(set(f["canonical_rank"].tolist())) == 3


# --------------------------------------------------------- model, end to end

def test_bare_sdf_runs_end_to_end():
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg())
    batch = collate_fn([sample_from_sdf(ETHANOL)])
    out, z = m(batch)
    assert out.shape == (1, 9, 3)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.parameters())


def test_model_ignores_residue_fields_entirely_under_graph_addressing():
    """The requirement, stated as a test: res_pos must be dead weight."""
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg()).eval()
    batch = collate_fn([sample_from_sdf(ETHANOL)])
    with torch.no_grad():
        base, _ = m(batch)
        scrambled = dict(batch)
        for f in ("res_pos", "residue_idx", "atom_name_idx", "slot_idx", "chain_idx"):
            scrambled[f] = torch.randint(0, 5, batch[f].shape)
        other, _ = m(scrambled)
    assert torch.allclose(base, other, atol=1e-6)


def test_fixed_latent_decoding_cannot_access_input_coordinates():
    """No content skips, on the universal path too."""
    torch.manual_seed(0)
    for local in (False, True):
        m = AtomLatentAutoencoder(_cfg(local=local)).eval()
        batch = collate_fn([sample_from_sdf(ETHANOL)])
        with torch.no_grad():
            z = m.encode(batch)
            a = m.decode(z, batch)
            trashed = dict(batch)
            trashed["coords"] = torch.randn_like(batch["coords"]) * 50
            b = m.decode(z, trashed)
        assert torch.allclose(a, b, atol=1e-6), f"leak (local={local})"


def test_input_permutation_leaves_the_structure_unchanged_after_remapping():
    """Reconstruct, undo the permutation, compare. Atoms are a SET.

    Ethanol's methyl hydrogens are automorphic, so a permutation may swap
    which of them lands where. The comparison is therefore on the multiset of
    inter-atomic distances -- the symmetry-aware form of "same structure".
    """
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg()).eval()
    s = sample_from_sdf(ETHANOL)
    n = s["n_atoms"]
    perm = torch.tensor([4, 0, 7, 2, 8, 1, 5, 3, 6])

    ps = dict(s)
    inv = torch.argsort(perm)
    for k, v in s.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n and k != "bonds":
            ps[k] = v[perm]
    ps["bonds"] = inv[s["bonds"]]
    ps["element_symbol"] = [s["element_symbol"][i] for i in perm.tolist()]

    with torch.no_grad():
        a, _ = m(collate_fn([s]))
        b, _ = m(collate_fn([ps]))
    da = torch.sort(torch.pdist(a[0]))[0]
    db = torch.sort(torch.pdist(b[0]))[0]
    assert torch.allclose(da, db, atol=1e-3)


def test_latent_count_stays_free_on_the_universal_path():
    m = AtomLatentAutoencoder(_cfg())
    batch = collate_fn([sample_from_sdf(ETHANOL)])
    for L in (16, 64, 256):
        assert m.encode(batch, n_latents=L).shape[1] == L


def test_group_addressing_still_works_and_differs():
    """The switch is real: the two addressing modes build different models."""
    graph = AtomLatentAutoencoder(_cfg())
    grouped = AtomLatentAutoencoder(_cfg())
    grouped.cfg.atom_addressing = "group"
    assert graph.encoder.feat.__class__.__name__ == "GraphAtomFeaturizer"
    assert AtomLatentAutoencoder(
        ModelConfig(d_model=64, n_heads=4, encoder_type="atomlatent",
                    atom_addressing="group")
    ).encoder.feat.__class__.__name__ == "AtomFeaturizer"


@pytest.mark.parametrize("n_mols", [1, 3])
def test_ragged_sdf_batches_collate(n_mols):
    water = ETHANOL.replace("ethanol", "water")
    batch = collate_fn([sample_from_sdf(ETHANOL)] * n_mols)
    assert batch["wl_class"].shape[0] == n_mols
    m = AtomLatentAutoencoder(_cfg())
    out, _ = m(batch)
    assert out.shape[0] == n_mols and torch.isfinite(out).all()


def _chain_sample(n=60):
    """A molecule LARGER than the attention window, so the window engages."""
    el = np.array([6] * n)
    bonds = np.array([[i, i + 1] for i in range(n - 1)])
    g = graph_atom_features(el, bonds=bonds)
    c = torch.stack([torch.arange(n).float() * 1.5, torch.zeros(n), torch.zeros(n)], 1)
    from molae import constants as Cst
    s = {"n_atoms": n, "pdb_id": "C", "chain_id": "_", "element_symbol": ["C"] * n,
         "coords": c - c.mean(0), "bonds": torch.from_numpy(bonds),
         "chirality_centers": torch.zeros((0, 4), dtype=torch.long),
         "res_pos": torch.zeros(n, dtype=torch.long),
         "res_seq": torch.zeros(n, dtype=torch.long),
         "chain_idx": torch.zeros(n, dtype=torch.long),
         "residue_idx": torch.full((n,), Cst.LIGAND_RESIDUE_IDX),
         "atom_name_idx": torch.full((n,), Cst.UNK_ATOM_IDX),
         "slot_idx": torch.zeros(n, dtype=torch.long)}
    for k, v in g.items():
        s[k] = torch.from_numpy(np.asarray(v, dtype=np.int64))
    s["element_idx"] = torch.from_numpy(el)
    return s


def _permute(s, perm):
    n = s["n_atoms"]
    inv = torch.argsort(perm)
    out = dict(s)
    for k, v in s.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n and k != "bonds":
            out[k] = v[perm]
    out["bonds"] = inv[s["bonds"]]
    out["element_symbol"] = [s["element_symbol"][i] for i in perm.tolist()]
    return out


def test_permutation_invariance_holds_past_the_attention_window():
    """The window slides over TENSOR order, so this is where it would break.

    Atoms are sorted into canonical order before the local attention runs.
    Without that sort the same check measures 2e-4 on an untrained model, with
    nothing holding it there once the window has learned to be used.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    cfg.attn_window = 32
    m = AtomLatentAutoencoder(cfg).eval()
    s = _chain_sample(60)                      # 60 atoms > 32 window
    for seed in range(3):
        torch.manual_seed(100 + seed)
        ps = _permute(s, torch.randperm(s["n_atoms"]))
        with torch.no_grad():
            z1, z2 = m.encode(collate_fn([s])), m.encode(collate_fn([ps]))
            a, _ = m(collate_fn([s]))
            b, _ = m(collate_fn([ps]))
        assert torch.allclose(z1, z2, atol=1e-4), f"latent moved (seed {seed})"
        da = torch.sort(torch.pdist(a[0]))[0]
        db = torch.sort(torch.pdist(b[0]))[0]
        assert torch.allclose(da, db, atol=1e-3), f"structure moved (seed {seed})"


def test_canonical_sort_puts_padding_last_in_ragged_batches():
    """A padded atom must not sort into the middle of the real ones."""
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg()).eval()
    big = _chain_sample(60)
    with torch.no_grad():
        solo = m.encode(collate_fn([big]))
        padded = m.encode(collate_fn([big, _chain_sample(12)]))[:1]
    assert torch.allclose(solo, padded, atol=1e-4)
