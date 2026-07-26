#!/usr/bin/env python3
"""Fine-tune the codec to RESOLVE conformers — a test of why the floor exists.

The hypothesis
--------------
The in-domain gate found a ~0.9 A all-atom reconstruction floor, and the latent
ladder showed it does not move from latent 4 to latent 16. Flat-in-capacity is
the signature of an OBJECTIVE problem, not a capacity one.

And the objective explains it directly. The codec is trained to reconstruct
single structures. If two conformers of the same protein differ by 0.3 A,
mapping both to the SAME latent costs almost nothing: the coordinate loss is
dominated by getting the fold right, and the conformational difference is a
rounding error inside it. The codec is never asked to tell conformers apart, so
it averages them away. That is precisely the failure the resolution ratio
measures, and precisely what breaks Step 3, where the whole signal IS the
difference between conformers.

The intervention
----------------
Add a margin term over pairs of conformers of the same protein. For a pair
(A, B), the reconstruction of A must sit closer to A than to B, by at least the
true separation between them:

    d_AA = RMSD(decode(z_A), A)
    d_AB = RMSD(decode(z_A), B)
    m    = RMSD(A, B)                     # the true conformational separation
    loss = relu(margin_frac * m - (d_AB - d_AA))

A codec that collapses the pair has d_AB == d_AA and pays the full margin. Note
this is deliberately NOT the evaluation metric: we score with the resolution
ratio and recon-vs-spread, so a model cannot trivially Goodhart its way to a
good score by inflating reconstruction spread.

Sampling
--------
Uniformly over GROUPS, not over pairs. The train census is brutally skewed:
792 pairs exist but 7HG0 (34 members, 561 pairs) and 1BQ9 (13 members, 78)
supply ~81% of them. Sampling pairs uniformly would teach the model one
protein's crystal forms. Sampling groups uniformly gives all 60 proteins equal
weight.

Standard reconstruction batches are interleaved (``--recon-every``) so the
margin term cannot destroy general reconstruction quality — catastrophic
forgetting would show up as the val-group recon getting worse, which is the
thing we are trying to improve.

What the result means
---------------------
floor drops on HELD-OUT val groups  -> the floor is objective-limited. Invest
                                       in conformational training data (more
                                       crystal groups, MD) before Step 3.
floor unmoved                       -> the floor is architectural, and a
                                       different fix is needed.

Only 60 train groups exist, so this is a hypothesis test, not a production fix.
A negative is informative; a partial improvement justifies getting more data.

Usage:
    python scripts/conformer_finetune.py --config configs/direct_d8.yaml \
        --checkpoint <direct_d8 final.pt> --steps 4000
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
sys.path.insert(0, str(HERE))

from molae.config import ExperimentConfig  # noqa: E402
from molae.dataset import ProteinStructureDataset, sample_from_arrays, collate_fn  # noqa: E402
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.losses import LossComputer  # noqa: E402
from molae.alignment import aligned_rmsd_torch, kabsch_rmsd_numpy  # noqa: E402
from molae import utils  # noqa: E402
from xray_ensemble_gate import load_groups, run_group  # noqa: E402
from latent_suitability import ensemble_resolution  # noqa: E402


def _one(d, device):
    batch = collate_fn([sample_from_arrays(d)])
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def margin_loss(model, group, i, j, device, margin_frac=1.0):
    """relu(margin_frac * RMSD(A,B) - (RMSD(predA,B) - RMSD(predA,A))).

    Computed on the group's common atom subset so structures with different
    resolved loops stay comparable. Differentiable through the Kabsch align.
    """
    (_, dA), idxA = group["members"][i], group["index_maps"][i]
    (_, dB), idxB = group["members"][j], group["index_maps"][j]
    gA, gB = _one(dA, device), _one(dB, device)

    predA = model.decode(model.encode(gA), gA)[0]
    tA = gA["coords"][0]
    tB = gB["coords"][0]

    iA = torch.as_tensor(idxA, device=device, dtype=torch.long)
    iB = torch.as_tensor(idxB, device=device, dtype=torch.long)
    pA_c, tA_c, tB_c = predA[iA][None], tA[iA][None], tB[iB][None]
    ones = torch.ones(1, pA_c.shape[1], device=device)

    d_AA = aligned_rmsd_torch(pA_c, tA_c, ones)[0]
    d_AB = aligned_rmsd_torch(pA_c, tB_c, ones)[0]
    with torch.no_grad():                       # the true separation is a target
        m = aligned_rmsd_torch(tA_c, tB_c, ones)[0]
    loss = torch.relu(margin_frac * m - (d_AB - d_AA))
    return loss, float(m), float(d_AA.detach()), float(d_AB.detach())


@torch.no_grad()
def score_groups(model, groups, device):
    """Held-out recon / spread / resolution over val groups."""
    model.eval()
    recon, ratios, spreads = [], [], []
    for g in groups:
        _, preds, trues = run_group(model, g, device)
        recon.extend(kabsch_rmsd_numpy(p, t) for p, t in zip(preds, trues))
        r, _, st = ensemble_resolution(preds, trues)
        if not np.isnan(r):
            ratios.append(r)
            spreads.append(st)
    model.train()
    return (float(np.mean(recon)) if recon else float("nan"),
            float(np.mean(ratios)) if ratios else float("nan"),
            float(np.mean(spreads)) if spreads else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="outputs/conformer_finetune")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4, help="lower than training: this is a fine-tune")
    ap.add_argument("--margin-weight", type=float, default=1.0)
    ap.add_argument("--margin-frac", type=float, default=1.0)
    ap.add_argument("--recon-every", type=int, default=2,
                    help="interleave a standard reconstruction batch every N steps")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--min-members", type=int, default=2)
    ap.add_argument("--min-common-atoms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    utils.set_seed(args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = ROOT / cfg.data.processed_dir
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    train_groups = load_groups(processed, args.min_members, args.min_common_atoms,
                           set(splits["train"]))
    val_groups = load_groups(processed, 3, args.min_common_atoms, set(splits["val"]))
    if not train_groups:
        raise SystemExit("no same-sequence train groups; nothing to fine-tune on")
    sizes = sorted(len(g["members"]) for g in train_groups)
    print(f"[finetune] {len(train_groups)} train groups (members: min {sizes[0]}, "
          f"median {sizes[len(sizes)//2]}, max {sizes[-1]}), {len(val_groups)} val groups")
    print("[finetune] sampling groups UNIFORMLY (the pair distribution is ~81% two proteins)")

    model = make_autoencoder(cfg.model).to(device)
    utils.load_checkpoint(args.checkpoint, model, None)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = LossComputer(cfg.loss, clash_dist=cfg.train.clash_dist)

    train_paths = [processed / f"{k}.npz" for k in splits["train"]]
    recon_ds = ProteinStructureDataset([p for p in train_paths if p.exists()])

    r0, ratio0, spread0 = score_groups(model, val_groups, device)
    print(f"[finetune] BEFORE: val-group recon {r0:.3f} A  resolution {ratio0:.3f}  "
          f"(true spread {spread0:.3f} A)")

    rng = np.random.default_rng(args.seed)
    log = [{"step": 0, "val_recon": r0, "val_resolution": ratio0}]
    hist = []
    for step in range(1, args.steps + 1):
        gi = int(rng.integers(len(train_groups)))          # uniform over GROUPS
        g = train_groups[gi]
        i, j = rng.choice(len(g["members"]), size=2, replace=False)
        opt.zero_grad()
        ml, m, d_aa, d_ab = margin_loss(model, g, int(i), int(j), device, args.margin_frac)
        total = args.margin_weight * ml

        if step % args.recon_every == 0:                   # keep general quality
            k = int(rng.integers(len(recon_ds)))
            batch = collate_fn([recon_ds[k]])
            gb = {kk: (v.to(device) if torch.is_tensor(v) else v)
                  for kk, v in batch.items()}
            preds, _ = model(gb)
            rl, _ = loss_fn(preds, gb)
            total = total + rl

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        hist.append((float(ml), m, d_aa, d_ab))

        if step % args.eval_every == 0 or step == args.steps:
            r, ratio, _ = score_groups(model, val_groups, device)
            mh = np.mean([h[0] for h in hist[-args.eval_every:]])
            print(f"  step {step:5d}  margin {mh:.4f}  ->  val-group recon {r:.3f} A "
                  f"(from {r0:.3f})  resolution {ratio:.3f} (from {ratio0:.3f})")
            log.append({"step": step, "margin": float(mh),
                        "val_recon": r, "val_resolution": ratio})

    utils.save_checkpoint(out_dir / "final.pt", model, opt, args.steps, extra={"log": log})
    rN, ratioN = log[-1]["val_recon"], log[-1]["val_resolution"]
    utils.save_json({
        "n_train_groups": len(train_groups), "n_val_groups": len(val_groups),
        "steps": args.steps, "margin_weight": args.margin_weight,
        "before": {"val_recon": r0, "val_resolution": ratio0},
        "after": {"val_recon": rN, "val_resolution": ratioN},
        "true_spread": spread0, "log": log,
    }, out_dir / "conformer_finetune.json")

    print(f"\n=== conformer-margin fine-tune ===")
    print(f"val-group recon      {r0:.3f} -> {rN:.3f} A   (true spread {spread0:.3f} A)")
    print(f"val-group resolution {ratio0:.3f} -> {ratioN:.3f}")
    print("\nfloor DROPS on held-out groups -> objective-limited; get more")
    print("  conformational data before Step 3.")
    print("floor UNMOVED -> architectural; a different fix is needed.")
    print("Recon must not get WORSE -- that would be catastrophic forgetting,")
    print("not a resolution gain.")


if __name__ == "__main__":
    main()
