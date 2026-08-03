"""Does the multi-molecule mode-count discount compound in K, and which metric
sizes the latent?

The per-molecule token budget in ROADMAP §7 assumes the dynamic modes of
independent molecules ADD. Three candidate metrics, all computed on the POOLED
SPECTRUM directly (independent blocks -> eigenvalue union = infinite-T limit,
ceiling-free; a finite-T measurement would cap pooled counts at ~99 by K=2 and
disguise truncation as drift). Real block spectra are the measured MISATO
deviation spectra (scripts/build_traj_probe.py + spectrum_of).

  dev_modes_90 : modes for 90% of pooled variance   (relative; threshold shifts)
  PR           : (sum L)^2 / sum L^2                 (relative; scale-dominated)
  modes_abs(t) : min m with residual_var / N <= t    (ABSOLUTE; sizes width)

Findings (see §7): under HETEROGENEITY every POOLED criterion is sub-additive --
dev90 ~0.77, PR ~0.29, modes_abs ~0.74 (0.5A) / ~0.34 (1.0A) at K=231 -- because
pooling mixes molecule scales. The artifact-free sizing quantity is the
PER-MOLECULE SUM sum_k modes_abs(block_k) (additive by construction; = per-molecule
fidelity), not any pooled count. dev90's discount does not compound (flat ~0.77).
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


def modes_abs(spec, N, tau):
    """min modes so residual per-atom variance <= tau (A^2). Absolute, so a rigid
    molecule needs few modes and a floppy one many -- no normalisation artifact."""
    s = np.sort(np.asarray(spec, float))[::-1]
    tot = s.sum()
    if tot / N <= tau:
        return 0
    return int(np.searchsorted(np.cumsum(s), tot - tau * N, side="left") + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,231")
    ap.add_argument("--taus", default="0.25,1.0")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    specs, Ns = [], []
    for p in sorted(glob.glob(f"{args.traj_dir}/*.npz")):
        fr = load_frames(Path(p))
        s = spectrum_of((fr - fr[0]).reshape(fr.shape[0], -1))
        specs.append(s[s > 0])
        Ns.append(fr.shape[1])
    Ns = np.array(Ns)
    Ks = [int(k) for k in args.ks.split(",")]
    taus = [float(t) for t in args.taus.split(",")]
    vpn = np.array([specs[i].sum() / Ns[i] for i in range(len(specs))])
    print(f"loaded {len(specs)} spectra; per-atom variance V/N mean={vpn.mean():.2f} "
          f"range {vpn.min():.2f}-{vpn.max():.2f} A^2")
    for tau in taus:
        ma = [modes_abs(specs[i], Ns[i], tau) for i in range(len(specs))]
        print(f"  modes_abs(tau={tau}): per-block mean={np.mean(ma):.0f} range {min(ma)}-{max(ma)}")

    med = sorted(range(len(specs)), key=lambda i: d90(specs[i]))[len(specs) // 2]

    def het(K, fn):
        vals = []
        for seed in range(args.seeds):
            idx = np.random.RandomState(seed).randint(0, len(specs), size=K)
            vals.append(fn(idx))
        return float(np.nanmean(vals))

    print("\nHETEROGENEOUS pooled/sum ratios (the mixed-box case)")
    print("  K     dev90   PR      mabs.25  mabs1.0   |  sum_mabs.25(width, additive)")
    for K in Ks:
        r_d = het(K, lambda ix: d90(np.concatenate([specs[i] for i in ix])) / sum(d90(specs[i]) for i in ix))
        r_p = het(K, lambda ix: PR(np.concatenate([specs[i] for i in ix])) / sum(PR(specs[i]) for i in ix))
        def rma(ix, tau):
            den = sum(modes_abs(specs[i], Ns[i], tau) for i in ix)
            return modes_abs(np.concatenate([specs[i] for i in ix]), Ns[ix].sum(), tau) / den if den else np.nan
        r_a = het(K, lambda ix: rma(ix, 0.25))
        r_b = het(K, lambda ix: rma(ix, 1.0))
        sum25 = het(K, lambda ix: sum(modes_abs(specs[i], Ns[i], 0.25) for i in ix))
        print(f"  {K:4d}  {r_d:.3f}   {r_p:.3f}   {r_a:.3f}    {r_b:.3f}    |  {sum25:8.0f}")

    # correctness gate: exact duplication must be EXACTLY K*m at small K (the
    # per-block code path that the width requirement also uses). Any drift at
    # larger K is real overshoot-slack (pooled = box-average sheds whole modes
    # once accumulated over-satisfaction exceeds a mode), not a bug -- itself a
    # demonstration that a pooled absolute criterion is unsound even for
    # identical blocks.
    j = sorted(range(len(specs)), key=lambda i: Ns[i])[len(specs) // 2]
    m = modes_abs(specs[j], Ns[j], 0.25)
    for K in (1, 2):
        assert modes_abs(np.tile(specs[j], K), Ns[j] * K, 0.25) == K * m, "duplication self-check failed"
    print(f"\n[selfcheck] exact duplication == K*m at K=1,2 (block m={m}); "
          f"K=231 ratio {modes_abs(np.tile(specs[j],231),Ns[j]*231,0.25)/(231*m):.3f} (overshoot-slack, real)")

    print("\nWIDTH REQUIREMENT = per-molecule-fidelity sum (additive, pool-free):")
    print("  RMSD(A)  tau(A^2)  scalars(~231 x mean)   vs 2 tok/mol (7,392)")
    for rmsd in (0.40, 0.50, 0.60, 0.75, 1.00):
        tau = rmsd ** 2
        per = np.mean([modes_abs(specs[i], Ns[i], tau) for i in range(len(specs))])
        req = 231 * per
        print(f"   {rmsd:.2f}    {tau:.3f}      {req:8.0f}          "
              f"{'BELOW (2 tok ok)' if req < 7392 else 'above (needs 4)'}")
    lo, hi = 0.40, 1.00
    for _ in range(40):
        mid = (lo + hi) / 2
        req = 231 * np.mean([modes_abs(specs[i], Ns[i], mid ** 2) for i in range(len(specs))])
        hi, lo = (mid, lo) if req < 7392 else (hi, mid)
    print(f"  crossover: 2 tok/mol suffices at >= ~{hi:.2f} A RMSD")


if __name__ == "__main__":
    main()
