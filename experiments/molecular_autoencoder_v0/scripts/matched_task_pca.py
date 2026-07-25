#!/usr/bin/env python3
"""Matched-task head-to-head: can a neural autoencoder beat PCA at PCA's own task?

Every comparison so far has been indirect. The learned models compress WHOLE
variable-size proteins and are then scored on a fixed 20-residue/80-atom
backbone cohort, while PCA compresses exactly that cohort. So "PCA wins" might
mean "the neural model is worse" or merely "the neural model was solving a
harder problem".

This removes the confound: both methods get the SAME input (the fixed aligned
80-atom backbone cohort, 240 numbers), the SAME latent budget, and the SAME
leak-free train/val split.

  * AE >= PCA  -> the concept is sound; our full-protein pipeline is the problem.
  * AE <  PCA  -> on aligned fixed-size fragments the variation is close to
                  linear, where PCA is near-optimal for L2 at a given rank, and
                  "beat PCA at fragment reconstruction" is the wrong bar for the
                  project.

CPU-only and fast (240-dim vectors), no GPU needed.

Usage:
    python scripts/matched_task_pca.py --config configs/recipe_proteinae.yaml
    python scripts/matched_task_pca.py --config <cfg> --latents 2,4,8,16 --epochs 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae.baselines import build_backbone_cohort, evaluate_baselines  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy  # noqa: E402
from molae import utils  # noqa: E402


def load_cohort(cfg, n_res):
    """Aligned fixed-size backbone cohorts for train and val (leak-free)."""
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    processed = ROOT / cfg.data.processed_dir

    def dicts(keys):
        out = []
        for k in keys:
            p = processed / f"{k}.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            c = d["coords"].astype(np.float64)
            out.append({"coords": c - c.mean(0, keepdims=True),
                        "atom_name_idx": d["atom_name_idx"],
                        "res_pos": d["res_pos"], "pdb_id": str(d["pdb_id"])})
        return out

    X_train, _, ref = build_backbone_cohort(dicts(splits["train"]), n_res)
    X_val, _, _ = build_backbone_cohort(dicts(splits["val"]), n_res, ref=ref)
    return X_train, X_val


class MLPAutoencoder(nn.Module):
    """Plain nonlinear autoencoder on the flattened aligned backbone."""

    def __init__(self, dim, latent, hidden=512, depth=3):
        super().__init__()
        enc, d = [], dim
        for _ in range(depth):
            enc += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        enc += [nn.Linear(hidden, latent)]
        dec, d = [], latent
        for _ in range(depth):
            dec += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        dec += [nn.Linear(hidden, dim)]
        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(self.enc(x))


def rmsd_rows(recon, truth, n_atoms):
    return float(np.mean([
        kabsch_rmsd_numpy(recon[i].reshape(n_atoms, 3), truth[i].reshape(n_atoms, 3))
        for i in range(recon.shape[0])
    ]))


def train_ae(X_train, X_val, latent, n_atoms, epochs, seed=0, hidden=512, depth=3, lr=1e-3):
    """Train an MLP autoencoder. Checkpoint selection uses a DEV split carved
    out of train -- never the val set, which PCA also never sees. Selecting on
    val would give the AE a peek at the test set that PCA does not get, and
    would bias this comparison in the AE's favour.
    """
    utils.set_seed(seed)
    n_dev = max(8, int(0.15 * X_train.shape[0]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_train.shape[0])
    dev_idx, tr_idx = perm[:n_dev], perm[n_dev:]
    X_dev, X_tr = X_train[dev_idx], X_train[tr_idx]

    mu, sd = X_tr.mean(0, keepdims=True), X_tr.std() + 1e-8
    xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32)
    xd = torch.tensor((X_dev - mu) / sd, dtype=torch.float32)
    xv = torch.tensor((X_val - mu) / sd, dtype=torch.float32)
    model = MLPAutoencoder(xt.shape[1], latent, hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_dev, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(xt.shape[0])
        for i in range(0, xt.shape[0], 64):
            b = xt[order[i:i + 64]]
            loss = ((model(b) - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                d = ((model(xd) - xd) ** 2).mean().item()      # DEV, not val
            if d < best_dev:
                best_dev = d
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rec_t = (model(xt).numpy() * sd) + mu
        rec_v = (model(xv).numpy() * sd) + mu
    return (rmsd_rows(rec_t, X_tr, n_atoms),
            rmsd_rows(rec_v, X_val, n_atoms),
            sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/recipe_proteinae.yaml")
    ap.add_argument("--n-res", type=int, default=20)
    ap.add_argument("--latents", default="2,4,8,16")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=512)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(ROOT / args.config)
    n_atoms = 4 * args.n_res
    X_train, X_val = load_cohort(cfg, args.n_res)
    print(f"[matched] cohort: {args.n_res} residues / {n_atoms} atoms "
          f"({X_train.shape[1]} numbers)")
    print(f"[matched] train {X_train.shape[0]}  val {X_val.shape[0]}  (leak-free split)")
    if X_train.shape[0] < 10 or X_val.shape[0] < 5:
        raise SystemExit("cohort too small — need a prepared multi-structure dataset")

    ks = [int(k) for k in args.latents.split(",")]
    # PCA is fit on the FULL train set (it needs no checkpoint selection); the
    # AE fits on train-minus-dev and selects on dev. If anything this now
    # slightly favours PCA, which is the safe direction for this comparison.
    pca = evaluate_baselines(X_train, X_val, args.n_res, ks)
    mean_shape = pca.get("mean_shape", {}).get("mean_backbone_rmsd", float("nan"))

    print(f"\n{'latent':>7} | {'PCA (val)':>10} | {'AE (val)':>9} | {'AE (train)':>10} | winner")
    print("-" * 62)
    rows = []
    for k in ks:
        p = pca.get(f"pca_k{k}", {}).get("mean_backbone_rmsd", float("nan"))
        a_tr, a_va, nparam = train_ae(X_train, X_val, k, n_atoms, args.epochs, hidden=args.hidden)
        win = "AE" if a_va < p else "PCA"
        print(f"{k:>7} | {p:>10.3f} | {a_va:>9.3f} | {a_tr:>10.3f} | {win}")
        rows.append({"latent": k, "pca_val": p, "ae_val": a_va,
                     "ae_train": a_tr, "ae_params": nparam, "winner": win})

    print(f"\nmean-shape baseline (0 floats): {mean_shape:.3f} A")
    ae_wins = sum(r["winner"] == "AE" for r in rows)
    print("\nVERDICT:")
    if ae_wins == len(rows):
        print("  The neural AE beats PCA at every matched budget -> the concept is")
        print("  sound and the full-protein pipeline is what needs fixing.")
    elif ae_wins == 0:
        print("  PCA wins at every matched budget, on identical inputs and splits.")
        print("  On aligned fixed-size fragments the variation is near-linear, where")
        print("  PCA is near-optimal -> 'beat PCA at fragment reconstruction' is the")
        print("  wrong bar. Re-frame the target task (e.g. conformational ensembles")
        print("  of one molecule, which is what latent MD actually needs).")
    else:
        print(f"  Mixed: AE wins at {ae_wins}/{len(rows)} budgets — advantage is")
        print("  budget-dependent, worth probing where the crossover sits.")

    out = ROOT / "outputs" / "matched_task_pca.json"
    utils.save_json({"n_res": args.n_res, "n_atoms": n_atoms,
                     "n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]),
                     "mean_shape": mean_shape, "rows": rows}, out)
    print(f"\n[matched] wrote {out}")


if __name__ == "__main__":
    main()
