#!/usr/bin/env python3
"""Diagnose *why* a trained autoencoder fails to generalise.

A model that ties the mean-shape baseline is doing one of three very different
things; this script tells them apart.

  1. UNDERFITTING          train RMSD is also poor -> optimisation/capacity
                           problem, not a generalisation problem.
  2. MEMORISING            train RMSD good, held-out poor -> classic overfit;
                           needs more data / regularisation.
  3. LATENT COLLAPSE       the decoder ignores z and models the marginal. The
                           tell: decoding structure A with structure B's latent
                           changes almost nothing. For diffusion autoencoders
                           this is the dominant failure mode, and no amount of
                           data fixes it -- the conditioning path must change.

For flow-matching models it also separates latent quality from sampling noise:
`reconstruct()` integrates the ODE from fresh noise, so the output is a *sample*
from p(x|z), not a deterministic decode. If fixing the noise sharply improves
RMSD, the latent is fine and the stochasticity is what's hurting.

Usage:
    python scripts/diagnose_latent.py --config configs/recipe_proteinae.yaml
    python scripts/diagnose_latent.py --config <cfg> --ckpt path/to/final.pt
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

from molae.config import ExperimentConfig  # noqa: E402
from molae.dataset import ProteinStructureDataset, collate_fn  # noqa: E402
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.alignment import aligned_rmsd_torch  # noqa: E402
from molae import utils  # noqa: E402


def load_split(cfg, split, limit):
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    processed = ROOT / cfg.data.processed_dir
    keys = list(splits[split])[:limit]
    paths = [processed / f"{k}.npz" for k in keys if (processed / f"{k}.npz").exists()]
    return ProteinStructureDataset(paths)


@torch.no_grad()
def reconstruct(model, batch, fixed_noise=False, seed=0):
    """Reconstruct, optionally with the ODE started from a fixed noise draw."""
    if hasattr(model, "reconstruct"):                       # flow-matching
        if fixed_noise:
            torch.manual_seed(seed)
        return model.reconstruct(batch)
    preds, _ = model(batch)
    return preds


@torch.no_grad()
def mean_rmsd(model, ds, device, fixed_noise=False, max_n=40):
    vals = []
    for i in range(min(len(ds), max_n)):
        b = collate_fn([ds[i]])
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        pred = reconstruct(model, b, fixed_noise=fixed_noise, seed=i)
        vals.append(aligned_rmsd_torch(pred, b["coords"], b["mask"]).item())
    return float(np.mean(vals)), float(np.std(vals))


@torch.no_grad()
def latent_swap_test(model, ds, device, group=6):
    """Decode each structure using a *different* structure's latent.

    Structures are collated into a padded group so all latents share a shape,
    then the latent tensor is rolled by one along the batch dimension: every
    structure keeps its own identity/topology but receives a neighbour's
    latent. If the latent carries the geometry, this should badly damage the
    reconstruction. If the output barely moves, the decoder is ignoring z
    (latent collapse) and no amount of extra data will help.
    """
    own, foreign, drift = [], [], []
    for start in range(0, min(len(ds), 4 * group), group):
        items = [ds[i] for i in range(start, min(start + group, len(ds)))]
        if len(items) < 2:
            continue
        b = collate_fn(items)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        enc = model.encode(b)
        z = enc[0] if isinstance(enc, tuple) else enc
        z_foreign = torch.roll(z, shifts=1, dims=0)      # neighbour's latent

        flow = hasattr(model, "reconstruct")
        torch.manual_seed(0)
        p_own = _decode_flow(model, z, b) if flow else model.decode(z, b)
        torch.manual_seed(0)
        p_for = _decode_flow(model, z_foreign, b) if flow else model.decode(z_foreign, b)

        own.append(aligned_rmsd_torch(p_own, b["coords"], b["mask"]).mean().item())
        foreign.append(aligned_rmsd_torch(p_for, b["coords"], b["mask"]).mean().item())
        drift.append(aligned_rmsd_torch(p_for, p_own, b["mask"]).mean().item())
    if not own:
        return None
    return {"own_latent_rmsd": float(np.mean(own)),
            "foreign_latent_rmsd": float(np.mean(foreign)),
            "output_change_when_latent_swapped": float(np.mean(drift)),
            "n_groups": len(own)}


@torch.no_grad()
def _decode_flow(model, z, batch, n_steps=None):
    """Run the flow ODE with an explicitly supplied latent."""
    from molae.flow import zero_com
    mask = batch["mask"]
    n_steps = n_steps or model.cfg.flow_steps
    latent_mask = None
    if getattr(model, "per_residue", False):
        from molae.model_perresidue import residue_mask
        latent_mask = ~residue_mask(batch["res_pos"], mask, z.shape[1]).bool()
    x = zero_com(torch.randn(*mask.shape, 3, device=mask.device), mask)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((mask.shape[0],), k * dt, device=mask.device)
        v = model._decode_v(x, t, z, batch, latent_mask)
        x = zero_com(x + v * dt, mask)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-structures", type=int, default=40)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(args.device)
    utils.set_seed(cfg.train.seed)

    model = make_autoencoder(cfg.model).to(device)
    ckpt = Path(args.ckpt) if args.ckpt else (ROOT / cfg.train.out_dir / "final.pt")
    if not ckpt.exists():
        ckpt = (ROOT / cfg.train.out_dir / "latest.pt")
    utils.load_checkpoint(ckpt, model, None, map_location=device)
    model.eval()
    print(f"[diag] {cfg.name}  ckpt={ckpt.name}  params={model.num_parameters():,}")

    train_ds = load_split(cfg, "train", args.max_structures)
    val_ds = load_split(cfg, "val", args.max_structures)

    tr, tr_s = mean_rmsd(model, train_ds, device, max_n=args.max_structures)
    va, va_s = mean_rmsd(model, val_ds, device, max_n=args.max_structures)
    print(f"\n1) FIT")
    print(f"   train  RMSD : {tr:6.3f} A  (sd {tr_s:.2f}, n={min(len(train_ds), args.max_structures)})")
    print(f"   held-out RMSD: {va:6.3f} A  (sd {va_s:.2f}, n={min(len(val_ds), args.max_structures)})")
    print(f"   gap          : {va - tr:6.3f} A")

    if hasattr(model, "reconstruct"):
        fa, _ = mean_rmsd(model, val_ds, device, fixed_noise=True, max_n=args.max_structures)
        print(f"\n2) SAMPLING NOISE (flow models)")
        print(f"   held-out, random noise: {va:6.3f} A")
        print(f"   held-out, fixed  noise: {fa:6.3f} A   (delta {va - fa:+.3f})")

    sw = latent_swap_test(model, val_ds, device)
    print(f"\n3) LATENT USAGE (swap test)")
    if sw is None:
        print("   no equal-size pairs found in the sampled val subset")
    else:
        print(f"   decode with own latent    : {sw['own_latent_rmsd']:6.3f} A")
        print(f"   decode with foreign latent: {sw['foreign_latent_rmsd']:6.3f} A")
        print(f"   output change on swap     : {sw['output_change_when_latent_swapped']:6.3f} A"
              f"   (n={sw['n_groups']} groups)")
        delta = sw["foreign_latent_rmsd"] - sw["own_latent_rmsd"]
        if sw["output_change_when_latent_swapped"] < 0.5:
            verdict = "LATENT COLLAPSE — decoder ignores z; more data will not help"
        elif delta < 0.5:
            verdict = "latent barely informative — weak conditioning"
        else:
            verdict = "latent IS used — failure is elsewhere (fit or capacity)"
        print(f"   verdict: {verdict}")

    print("\n4) READING")
    if tr > 4.0:
        print("   train RMSD is poor too -> UNDERFITTING (optimisation/capacity),")
        print("   not a generalisation problem. Train longer / larger model first.")
    elif va - tr > 2.0:
        print("   fits train, fails held-out -> MEMORISATION.")
    else:
        print("   train and held-out are similar -> the model learned a global")
        print("   average rather than structure-specific geometry.")


if __name__ == "__main__":
    main()
