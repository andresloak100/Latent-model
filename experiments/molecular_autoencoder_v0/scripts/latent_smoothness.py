#!/usr/bin/env python3
"""Does the codec's latent vary SMOOTHLY across conformers?

The gating question for the temporal stage, and it is a property of the CODEC
— measurable with no diffusion model and no MD data.

Why it matters
--------------
Latent diffusion assumes the latent is a space where nearby points are nearby
structures. A codec trained on static PDB has only ever seen one conformer per
entry, so nothing in its training forced that property. It can reconstruct
every conformer beautifully while mapping neighbouring ones to distant latents
— and reconstruction RMSD cannot see this at all. If the latent is jagged,
the fix belongs in the codec (temporal context, conformer-pair training), and
finding that out AFTER training a diffusion model on top would be expensive.

Method
------
NMR entries deposit ~20 models of one molecule with identical atom
composition: real conformational ensembles, free, no MD. For each entry we
compare the PAIRWISE latent distance matrix against the PAIRWISE structural
(Kabsch-aligned) RMSD matrix, by Spearman correlation.

Two controls, both of which this measurement is worthless without:

  * RANDOM-INIT CONTROL. A random projection preserves distances in
    expectation (Johnson-Lindenstrauss), so an UNTRAINED encoder already
    scores a high correlation. An earlier version of this measurement reported
    rho 0.71 and only the control revealed that essentially all of it was
    free. The number that means anything is trained MINUS random.

  * KABSCH ALIGNMENT. The encoder is not rotation-invariant, so two
    identical conformers in different deposition frames have different
    latents. Structural distance is measured after superposition; without it
    the structural matrix measures frame choice, not conformation.

Ensembles are unordered, so no consecutive-frame drift is computed here —
that requires a genuine trajectory (MISATO), and `molae.trajectory.drift_ratio`
is the measure for it.

Usage:
    python scripts/latent_smoothness.py \
        --config outputs/cluster/ladder_direct3m_n2272/config.yaml \
        --checkpoint outputs/cluster/ladder_direct3m_n2272/final.pt \
        --entries 1L2Y 1FSD 1E0L
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
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy  # noqa: E402
from molae.trajectory import frames_from_nmr, check_consistent, encode_segment  # noqa: E402
from molae import utils  # noqa: E402


def spearman(a, b):
    """Rank correlation, without a scipy dependency."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def pairwise(latents, coords_list):
    """Flattened upper-triangular latent and structural distance vectors."""
    T = len(coords_list)
    lat, struc = [], []
    for i in range(T):
        for j in range(i + 1, T):
            lat.append(float(np.linalg.norm(latents[i] - latents[j])))
            # Kabsch-aligned: the encoder is not rotation-invariant, so
            # unaligned RMSD would measure deposition frame, not conformation.
            struc.append(float(kabsch_rmsd_numpy(coords_list[i], coords_list[j])))
    return np.array(lat), np.array(struc)


@torch.no_grad()
def reconstruct(model, frames, device):
    from molae.dataset import sample_from_arrays, collate_fn
    out = []
    for f in frames:
        b = collate_fn([sample_from_arrays(f)])
        gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        pred = model.decode(model.encode(gb), gb)
        n = int(gb["mask"][0].sum().item())
        out.append(pred[0, :n].cpu().numpy().astype(np.float64))
    return out


def resolution_ratio(preds, trues):
    """Spread of RECONSTRUCTED conformers over spread of TRUE conformers.

    rho is a RANK correlation and is blind to collapse: a decoder that maps
    every conformer to the same mean structure, preserving only their order,
    scores rho = 1.0. This ratio catches it -- a collapsed codec scores ~0
    because its reconstructions barely differ from each other.

    ~1 means the ensemble's conformational spread survives the round trip.
    Well below 1 means conformers are being averaged away, which is fatal for
    a temporal model however good the per-frame RMSD looks.
    """
    def spread(xs):
        d = [kabsch_rmsd_numpy(xs[i], xs[j])
             for i in range(len(xs)) for j in range(i + 1, len(xs))]
        return float(np.mean(d)) if d else float("nan")
    sp, st = spread(preds), spread(trues)
    return sp / st if st > 1e-9 else float("nan"), sp, st


def analyse(model, frames, device):
    z = encode_segment(model, frames, device).numpy()          # (T, R, d)
    z = z.reshape(len(frames), -1)
    coords = [np.asarray(f["coords"], dtype=np.float64) for f in frames]
    lat, struc = pairwise(z, coords)
    preds = reconstruct(model, frames, device)
    ratio, sp, st = resolution_ratio(preds, coords)
    recon = float(np.mean([kabsch_rmsd_numpy(p, c) for p, c in zip(preds, coords)]))
    return {
        "rho": spearman(lat, struc),
        "resolution_ratio": ratio,
        "spread_reconstructions": sp,
        "spread_true": st,
        "recon_rmsd": recon,
        "mean_struct_rmsd": float(struc.mean()),
        "struct_spread": float(struc.std()),
        "mean_latent_dist": float(lat.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--entries", nargs="+", required=True)
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--out", default="outputs/latent_smoothness.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(args.device)

    trained = make_autoencoder(cfg.model).to(device)
    utils.load_checkpoint(args.checkpoint, trained, None)
    trained.eval()

    # The control: identical architecture, never trained.
    torch.manual_seed(0)
    control = make_autoencoder(cfg.model).to(device)
    control.eval()

    rows = []
    for pid in args.entries:
        path = ROOT / args.raw_dir / f"{pid}.cif"
        if not path.exists():
            print(f"  {pid}: not cached, skipping")
            continue
        frames = frames_from_nmr(path, pdb_id=pid, max_frames=args.max_frames)
        if len(frames) < 4:
            print(f"  {pid}: only {len(frames)} models, skipping")
            continue
        n_atoms = check_consistent(frames)
        t = analyse(trained, frames, device)
        c = analyse(control, frames, device)
        row = {"pdb_id": pid, "n_models": len(frames), "n_atoms": n_atoms,
               "rho_trained": t["rho"], "rho_random_init": c["rho"],
               "rho_above_control": t["rho"] - c["rho"],
               "mean_struct_rmsd": t["mean_struct_rmsd"],
               "struct_spread": t["struct_spread"],
               "recon_rmsd": t["recon_rmsd"],
               "resolution_ratio": t["resolution_ratio"],
               "spread_true": t["spread_true"],
               "spread_reconstructions": t["spread_reconstructions"]}
        rows.append(row)
        print(f"  {pid}: {len(frames)} models, {n_atoms} atoms")
        print(f"      rho trained {t['rho']:.3f}  random {c['rho']:.3f}  "
              f"gain {row['rho_above_control']:+.3f}")
        print(f"      recon {t['recon_rmsd']:.2f} A  |  true spread "
              f"{t['spread_true']:.2f} A -> reconstructed spread "
              f"{t['spread_reconstructions']:.2f} A  |  RESOLUTION "
              f"{t['resolution_ratio']:.3f}")

    if not rows:
        raise SystemExit("no usable ensembles")

    summary = {
        "n_entries": len(rows),
        "mean_rho_trained": float(np.mean([r["rho_trained"] for r in rows])),
        "mean_rho_random_init": float(np.mean([r["rho_random_init"] for r in rows])),
        "mean_gain_over_control": float(np.mean([r["rho_above_control"] for r in rows])),
        "mean_resolution_ratio": float(np.mean([r["resolution_ratio"] for r in rows])),
        "mean_recon_rmsd": float(np.mean([r["recon_rmsd"] for r in rows])),
        "entries": rows,
    }
    utils.save_json(summary, ROOT / args.out)

    print("\n=== latent smoothness across conformers ===")
    print(f"trained rho      : {summary['mean_rho_trained']:.3f}")
    print(f"random-init rho  : {summary['mean_rho_random_init']:.3f}   <- free, "
          f"Johnson-Lindenstrauss")
    print(f"GAIN over control: {summary['mean_gain_over_control']:+.3f}   <- the "
          f"only number that means anything")
    print(f"\nresolution ratio : {summary['mean_resolution_ratio']:.3f}   <- "
          f"conformer spread surviving the round trip")
    print(f"reconstruction   : {summary['mean_recon_rmsd']:.3f} A")
    print("\nReading it:")
    print("  gain near 0    -> the latent metric tracks structure no better than")
    print("     a random projection. Training never organised the space along")
    print("     conformational coordinates, and diffusing over it is a bet.")
    print("     The fix is in the CODEC (temporal context, conformer pairs).")
    print("  gain clearly >0 -> training DID organise the latent by")
    print("     conformation; stacking per-frame latents is a sound basis for")
    print("     the temporal stage.")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
