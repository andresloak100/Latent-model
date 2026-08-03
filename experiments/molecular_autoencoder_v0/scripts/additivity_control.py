"""Does the multi-molecule mode-count discount compound in K?

The per-molecule token budget in ROADMAP §7 assumes the dynamic modes of
independent molecules ADD. `dev_modes_90` (a 90%-variance count) is sub-additive
across independent blocks even with zero physical sharing, so the question is
whether that discount stays mild or compounds toward K=231 (1M atoms).

This computes on the POOLED SPECTRUM directly (independent blocks -> eigenvalue
union), i.e. the infinite-T limit -- removing the T-1 ceiling that would
otherwise cap a finite-T measurement at ~99 modes by K=2 and disguise truncation
as drift. Real block spectra are the measured MISATO deviation spectra (built by
scripts/build_traj_probe.py + trajectory_dimensionality.py's spectrum_of).

Two metrics, two block regimes:
  - dev_modes_90: modes for 90% of pooled variance (the width-relevant count)
  - PR = (sum L)^2 / sum L^2: participation ratio (effective dimension)
  - homogeneous  = one real spectrum replicated K times (identical blocks)
  - heterogeneous = K real spectra sampled with replacement (the mixed-box case)

Finding: dev_modes_90 heterogeneous ratio flattens at ~0.77 (does not compound);
PR is exactly additive for identical blocks (ratio 1.000) but collapses to ~0.29
for heterogeneous blocks, so it is NOT the cleaner additivity metric for a real
mixed box.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from trajectory_dimensionality import spectrum_of, modes_for, load_frames


def PR(s):
    s = np.asarray(s, float)
    return float((s.sum() ** 2) / np.square(s).sum())


def d90(s):
    return modes_for(np.sort(np.asarray(s, float))[::-1], 0.90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,231")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    specs = []
    for p in sorted(glob.glob(f"{args.traj_dir}/*.npz")):
        fr = load_frames(Path(p))
        s = spectrum_of((fr - fr[0]).reshape(fr.shape[0], -1))
        specs.append(s[s > 0])
    Ks = [int(k) for k in args.ks.split(",")]
    print(f"loaded {len(specs)} real spectra; block dev90 "
          f"{min(d90(s) for s in specs)}-{max(d90(s) for s in specs)}")

    med = sorted(specs, key=d90)[len(specs) // 2]
    print(f"\nrepresentative block: dev90={d90(med)} PR={PR(med):.1f}")
    print("\nHOMOGENEOUS (identical block x K) -- reference")
    print("  K     dev90_ratio   PR_ratio")
    for K in Ks:
        pool = np.tile(med, K)
        print(f"  {K:4d}   {d90(pool)/(K*d90(med)):.3f}        {PR(pool)/(K*PR(med)):.3f}")

    print("\nHETEROGENEOUS (K real spectra, w/ replacement, avg over seeds)")
    print("  K     dev90_pool   dev90_ratio   PR_ratio")
    for K in Ks:
        dr, pr, ab = [], [], []
        for seed in range(args.seeds):
            idx = np.random.RandomState(seed).randint(0, len(specs), size=K)
            chosen = [specs[i] for i in idx]
            pool = np.concatenate(chosen)
            dr.append(d90(pool) / sum(d90(s) for s in chosen))
            pr.append(PR(pool) / sum(PR(s) for s in chosen))
            ab.append(d90(pool))
        print(f"  {K:4d}   {np.mean(ab):9.0f}    {np.mean(dr):.3f}        {np.mean(pr):.3f}")


if __name__ == "__main__":
    main()
