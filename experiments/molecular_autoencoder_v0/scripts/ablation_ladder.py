#!/usr/bin/env python3
"""Ablation ladder: WHERE does the whole-protein pipeline break?

Two facts now sit side by side:
  * matched task (fixed 80-atom aligned backbones, flat MLP): the AE BEATS PCA.
  * full pipeline (variable-size all-atom proteins, our Perceiver):
    every arm loses to a 0-float mean-shape baseline.

Something between those two settings destroys the model. Three things differ:
  (a) the decoder -- a flat MLP that emits a fixed-length vector, versus
      attention over per-atom identity queries;
  (b) variable structure size (padding, no fixed output slot per atom);
  (c) no canonical alignment (targets live in arbitrary rigid frames, with
      only the Kabsch-in-loss making the objective invariant).

This script isolates (a) -- the cheapest and most likely culprit -- by running
both decoders on IDENTICAL data (the same aligned 80-atom cohort), with the
same latent budget, split, and epoch count. Rungs (b) and (c) need the full
pipeline on GPU; this rung runs on CPU in minutes.

  rung 1  flat MLP            (known: beats PCA)
  rung 2  attention decoder   <- if this collapses, the decoder is the culprit

Usage:
    python scripts/ablation_ladder.py --config configs/recipe_proteinae.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))   # reuse helpers from matched_task_pca.py

from molae.config import ExperimentConfig  # noqa: E402
from molae import constants as C  # noqa: E402
from molae.model import ModelConfig, MolecularAutoencoder  # noqa: E402
from molae.baselines import evaluate_baselines  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy  # noqa: E402
from molae import utils  # noqa: E402
from matched_task_pca import load_cohort, train_ae, rmsd_rows  # noqa: E402


def cohort_to_batch(X, n_res, device):
    """Turn flat aligned backbone vectors into the batch dict the Perceiver wants.

    Identity is deliberately uninformative and identical across structures
    (all ALA): the only thing distinguishing atoms is (res_pos, atom_name),
    exactly the positional information the flat MLP gets from its fixed output
    ordering. So the two rungs are given the same information.
    """
    B = X.shape[0]
    n_atoms = 4 * n_res
    coords = torch.tensor(X.reshape(B, n_atoms, 3), dtype=torch.float32, device=device)
    res_pos = torch.arange(n_res, device=device).repeat_interleave(4).unsqueeze(0).expand(B, -1)
    atom_names = [C.ATOM_NAME_TO_IDX[a] for a in C.BACKBONE_ATOMS]
    atom_name_idx = torch.tensor(atom_names, device=device).repeat(n_res).unsqueeze(0).expand(B, -1)
    elements = [C.ELEMENT_TO_IDX[e] for e in ("N", "C", "C", "O")]
    element_idx = torch.tensor(elements, device=device).repeat(n_res).unsqueeze(0).expand(B, -1)
    residue_idx = torch.full((B, n_atoms), C.RESIDUE_TO_IDX["ALA"], device=device)
    return {
        "coords": coords,
        "mask": torch.ones(B, n_atoms, device=device),
        "res_pos": res_pos.contiguous(),
        "atom_name_idx": atom_name_idx.contiguous(),
        "element_idx": element_idx.contiguous(),
        "residue_idx": residue_idx,
        "n_atoms": torch.full((B,), n_atoms, dtype=torch.long, device=device),
    }


def train_attention_ae(X_train, X_val, latent, n_res, epochs, seed=0, d_model=128, lr=1e-3):
    """Rung 2: our Perceiver encoder/decoder on the fixed-size aligned cohort.

    Uses a plain masked-MSE objective on the already-aligned coordinates (no
    Kabsch in the loss) so the ONLY difference from rung 1 is the architecture.
    """
    utils.set_seed(seed)
    device = torch.device("cpu")
    n_atoms = 4 * n_res

    n_dev = max(8, int(0.15 * X_train.shape[0]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_train.shape[0])
    X_dev, X_tr = X_train[perm[:n_dev]], X_train[perm[n_dev:]]

    cfg = ModelConfig(d_model=d_model, n_heads=4, n_latent_tokens=1, latent_dim=latent,
                      enc_self_layers=2, dec_self_layers=2, coord_scale=10.0)
    model = MolecularAutoencoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    b_tr = cohort_to_batch(X_tr, n_res, device)
    b_dev = cohort_to_batch(X_dev, n_res, device)
    b_val = cohort_to_batch(X_val, n_res, device)

    def mse(batch, idx=None):
        sub = batch if idx is None else {
            k: (v[idx] if torch.is_tensor(v) and v.shape[0] == batch["coords"].shape[0] else v)
            for k, v in batch.items()}
        preds, _ = model(sub)
        return ((preds - sub["coords"]) ** 2).mean(), preds

    best_dev, best_state = float("inf"), None
    N = b_tr["coords"].shape[0]
    for ep in range(epochs):
        model.train()
        order = torch.randperm(N)
        for i in range(0, N, 64):
            idx = order[i:i + 64]
            loss, _ = mse(b_tr, idx)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                d, _ = mse(b_dev)
            if d.item() < best_dev:
                best_dev = d.item()
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, pt = mse(b_tr)
        _, pv = mse(b_val)
    return (rmsd_rows(pt.numpy().reshape(pt.shape[0], -1), X_tr, n_atoms),
            rmsd_rows(pv.numpy().reshape(pv.shape[0], -1), X_val, n_atoms),
            model.num_parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/recipe_proteinae.yaml")
    ap.add_argument("--n-res", type=int, default=20)
    ap.add_argument("--latents", default="4,8,16")
    ap.add_argument("--epochs", type=int, default=800)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(ROOT / args.config)
    n_atoms = 4 * args.n_res
    X_train, X_val = load_cohort(cfg, args.n_res)
    print(f"[ladder] cohort {args.n_res} res / {n_atoms} atoms; "
          f"train {X_train.shape[0]}  val {X_val.shape[0]}")
    if X_train.shape[0] < 10:
        raise SystemExit("cohort too small — prepare a multi-structure dataset first")

    ks = [int(k) for k in args.latents.split(",")]
    pca = evaluate_baselines(X_train, X_val, args.n_res, ks)
    mean_shape = pca.get("mean_shape", {}).get("mean_backbone_rmsd", float("nan"))

    print(f"\n{'latent':>7} | {'PCA':>7} | {'rung1 MLP':>10} | {'rung2 attn':>11} | culprit?")
    print("-" * 64)
    rows = []
    for k in ks:
        p = pca.get(f"pca_k{k}", {}).get("mean_backbone_rmsd", float("nan"))
        _, mlp_val, _ = train_ae(X_train, X_val, k, n_atoms, args.epochs)
        _, att_val, _ = train_attention_ae(X_train, X_val, k, args.n_res, args.epochs)
        flag = "YES — decoder" if att_val > mlp_val + 0.5 else "no"
        print(f"{k:>7} | {p:>7.3f} | {mlp_val:>10.3f} | {att_val:>11.3f} | {flag}")
        rows.append({"latent": k, "pca": p, "rung1_mlp": mlp_val,
                     "rung2_attention": att_val})

    print(f"\nmean-shape baseline: {mean_shape:.3f} A")
    worse = [r for r in rows if r["rung2_attention"] > r["rung1_mlp"] + 0.5]
    print("\nVERDICT:")
    if len(worse) == len(rows):
        print("  The attention decoder collapses on data the MLP handles fine.")
        print("  => The DECODER (identity-query cross-attention) is the bottleneck,")
        print("     not variable size and not the missing canonical alignment.")
        print("     Next: fix the decoder (local frames / direct positional readout)")
        print("     before touching anything else in the pipeline.")
    elif not worse:
        print("  Both decoders handle the fixed aligned task. The break must come")
        print("  from variable size or the missing canonical alignment — run the")
        print("  remaining rungs (whole proteins, with and without pre-alignment).")
    else:
        print(f"  Mixed: attention lags at {len(worse)}/{len(rows)} budgets — the")
        print("  decoder is a partial contributor, worth combining with rung 3/4.")

    utils.save_json({"n_res": args.n_res, "mean_shape": mean_shape, "rows": rows},
                    ROOT / "outputs" / "ablation_ladder.json")
    print(f"\n[ladder] wrote {ROOT/'outputs'/'ablation_ladder.json'}")


if __name__ == "__main__":
    main()
