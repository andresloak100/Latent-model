#!/usr/bin/env python3
"""Is the per-timestep deviation low-dimensional enough for a global latent?

The design is one shared global latent per timestep carrying the dynamic
state, with atom identity, bonding and reference geometry supplied as static
per-atom conditioning. That only works if the thing the global latent must
carry -- the DEVIATION of the system from its reference at time t -- actually
lives in few dimensions.

There is good reason to expect it does: PCA of MD, tICA and Markov-state
modelling all find a handful of slow collective modes dominating. But
"expected" is not "measured on our corpus at our system size", and this is the
premise the whole architecture rests on, so measure it before building.

What is measured, per trajectory:

  displacement spectrum   PCA over frames of (x_t - x_ref). The number of
      components needed for 90/95/99% of the variance IS the required latent
      width, up to the encoder's ability to find those directions.
  frame-to-frame          the same for consecutive-frame deltas rather than
      deviation from a reference -- the quantity a stepping model predicts.
  locality                what fraction of the total displacement variance
      sits in the top-1% most mobile atoms. High means the motion is LOCAL,
      not collective, and a global latent is the wrong instrument for it --
      which is the argument for the sparse local event channel.

Reading it:

  90% variance in << 1024 modes   -> a fixed global latent is well-founded;
      the width tells you how big it needs to be
  90% variance needs ~3N modes    -> the deviation is not collective at all
      and the global-latent premise is wrong for this corpus
  high locality fraction          -> collective modes miss the events that
      matter, and the sparse channel is load-bearing rather than optional

Usage:
    python scripts/trajectory_dimensionality.py --traj-dir data/misato_frames \\
        --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae import utils  # noqa: E402


def modes_for(spectrum, frac):
    """How many PCA components carry `frac` of the total variance."""
    if spectrum.sum() <= 0:
        return 0
    c = np.cumsum(spectrum) / spectrum.sum()
    return int(np.searchsorted(c, frac) + 1)


def spectrum_of(X):
    """PCA variance spectrum of (T, 3N) displacements, without forming 3N x 3N."""
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return np.zeros(0)
    # T frames is far smaller than 3N coords, so the T x T Gram matrix carries
    # the whole spectrum at a fraction of the cost.
    s = np.linalg.svd(X, compute_uv=False)
    return (s ** 2) / max(X.shape[0] - 1, 1)


def analyse(frames, name=""):
    """frames: (T, N, 3), already aligned. Returns the dimensionality report."""
    T, N, _ = frames.shape
    ref = frames[0]
    dev = (frames - ref).reshape(T, -1)
    delta = np.diff(frames, axis=0).reshape(T - 1, -1)

    sd, sdel = spectrum_of(dev), spectrum_of(delta)
    per_atom_var = ((frames - frames.mean(0)) ** 2).sum(-1).mean(0)   # (N,)
    top1 = max(1, N // 100)
    mobile = np.sort(per_atom_var)[::-1][:top1].sum() / max(per_atom_var.sum(), 1e-12)

    return {
        "name": name, "n_frames": T, "n_atoms": N,
        "dev_modes_90": modes_for(sd, 0.90), "dev_modes_95": modes_for(sd, 0.95),
        "dev_modes_99": modes_for(sd, 0.99),
        "delta_modes_90": modes_for(sdel, 0.90),
        "delta_modes_99": modes_for(sdel, 0.99),
        "max_possible_modes": min(T - 1, 3 * N),
        "top1pct_atom_variance_frac": float(mobile),
        "rmsf_mean": float(np.sqrt(per_atom_var).mean()),
    }


def load_frames(path):
    """(T, N, 3) from an .npz or .npy of trajectory coordinates."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    d = np.load(path, allow_pickle=True)
    for k in ("frames", "coords", "trajectory", "xyz"):
        if k in d:
            a = np.asarray(d[k], dtype=np.float64)
            if a.ndim == 3:
                return a
    raise ValueError(f"{path.name}: no (T, N, 3) array found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="outputs/trajectory_dimensionality.json")
    args = ap.parse_args()

    d = Path(args.traj_dir)
    if not d.is_absolute():
        d = ROOT / d
    paths = sorted(d.glob(args.pattern))[:args.limit]
    if not paths:
        raise SystemExit(f"no trajectories matching {args.pattern} in {d}")

    rows = []
    for p in paths:
        try:
            frames = load_frames(p)
        except Exception as e:
            print(f"  {p.name}: skipped ({e})")
            continue
        rows.append(analyse(frames, p.stem))
        r = rows[-1]
        print(f"  {r['name'][:28]:28s} T={r['n_frames']:4d} N={r['n_atoms']:6d}  "
              f"dev90={r['dev_modes_90']:4d}  delta90={r['delta_modes_90']:4d}  "
              f"top1%var={r['top1pct_atom_variance_frac']:.2f}")

    if not rows:
        raise SystemExit("nothing analysed")

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"\n=== over {len(rows)} trajectories ===")
    print(f"  atoms (mean)                     {mean('n_atoms'):.0f}")
    print(f"  frames (mean)                    {mean('n_frames'):.0f}")
    print("\nDEVIATION FROM REFERENCE  (what a global latent must carry)")
    print(f"  modes for 90% variance           {mean('dev_modes_90'):.1f}")
    print(f"  modes for 95% variance           {mean('dev_modes_95'):.1f}")
    print(f"  modes for 99% variance           {mean('dev_modes_99'):.1f}")
    print(f"  ceiling (min(T-1, 3N))           {mean('max_possible_modes'):.0f}")
    print("\nFRAME-TO-FRAME DELTA      (what a stepping model predicts)")
    print(f"  modes for 90% variance           {mean('delta_modes_90'):.1f}")
    print(f"  modes for 99% variance           {mean('delta_modes_99'):.1f}")
    print("\nLOCALITY                  (is the sparse event channel load-bearing?)")
    print(f"  variance in top 1% of atoms      {mean('top1pct_atom_variance_frac'):.3f}")
    print(f"  mean RMSF (A)                    {mean('rmsf_mean'):.3f}")
    print("\nCAVEAT: modes are bounded by T-1, so a short trajectory cannot show")
    print("high dimensionality even if it is there. Compare dev_modes_90 against")
    print("the ceiling: near it means the trajectory is too short to answer.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    utils.save_json({"rows": rows}, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
