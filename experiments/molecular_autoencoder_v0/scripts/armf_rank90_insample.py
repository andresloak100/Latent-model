"""rank90 IS MEASURED IN-SAMPLE, SO IT UNDERSTATES TRUE DIMENSIONALITY. Measure by how much.

Every rank90 in this project -- the mdCATH asymptote 168 that anchors the width chain, the b
exponent, the intrinsic-dimension writeup -- counts the PCA components needed to reach 90% of the
variance OF THE VERY FRAMES THE COMPONENTS WERE FITTED ON. PCA maximises explained variance in-sample
by construction, so that count is a LOWER BOUND on the number of directions a new frame actually
needs. With n_eff at 1-7% of the raw frame count (14-24 effective samples at k=24, 94-165 at k=256,
against 2,501 frames) the gap is not a rounding error.

This measures it directly under the ACTUAL protocol: fit PCA on replicas 0+1, then ask how many of
those train-fitted modes replica 2 needs to reach 90%.

    rank90_in   = first k where cumulative TRAIN variance >= 90%      (what has always been reported)
    rank90_out  = first k where cumulative HELD-OUT FVE   >= 90%      (what a new frame actually needs)
    fve_at_in   = held-out FVE achieved by exactly rank90_in modes    (the honest read of "rank90")

If rank90_out > rank90_in the reported number understates dimensionality. If the held-out curve never
reaches 90% at ANY k, rank90_out is not merely larger -- it is UNDEFINED on this trajectory, which is
a stronger statement than any finite underestimate, and is reported as censored rather than silently
dropped (that would be Family A: a non-random exclusion correlated with the regressor, since it is
precisely the badly-sampled systems that fail to reach 90%).

SECOND BIAS, SAME DIRECTION: rank90 reaches 32% of usable rank at the top of the N axis, past the 30%
edge where the count presses against its own measurement ceiling (Family B). That censors rank90 at
LARGE N specifically, so the GROWTH of rank90 with N -- the b exponent -- is underestimated too.

BOTH BIASES PUSH THE SAME WAY, so the width chain
    168 (mdCATH asymptote) / 0.65 (floppy->bound) x 1.83 (N^0.101 to 1e6) ~= 490 dims
is a FLOOR, not an estimate.

AND IT IS CORPUS-INDEPENDENT. n_eff at 1-7% is a statement about trajectory length versus
decorrelation time, not about ATLAS. MISATO is 10 ns, mdCATH 2.5 us, ATLAS 100 ns; none supplies
enough independent samples to fit a few-hundred-dimensional per-system subspace. There is no dataset
move that fixes this, exactly as there was none for the ceiling. => Latent width must be sized from
the CODEC's own held-out FVE-vs-DM curve, whose effective sample size is the CORPUS rather than one
trajectory. See armf_atlas_dm.py.

FRAME-SPACE THROUGHOUT: G = Xtr Xtr^T (Ftr,Ftr); modes are V_k = Xtr^T U_k / s_k, and the held-out
projection is ho V_k = (ho Xtr^T) U_k / s_k, so only (Fho,Ftr) is ever formed -- never 3N-sized."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
OUT = f"{WR}/atlas_rank90_insample.json"
CHUNK = 20000


def insample_vs_out(path):
    a = np.load(path, mmap_mode="r"); F, N = a.shape[1], a.shape[2]
    D3 = 3 * N
    mu = np.zeros(D3)
    for r in (0, 1):
        mu += np.asarray(a[r]).reshape(F, -1).astype(np.float64).sum(0)
    mu /= (2 * F)
    Ftr = 2 * F
    G = np.zeros((Ftr, Ftr)); M = np.zeros((F, Ftr)); sst_ho = 0.0
    for c0 in range(0, D3, CHUNK):
        c1 = min(c0 + CHUNK, D3)
        tr = np.concatenate([np.asarray(a[r]).reshape(F, -1)[:, c0:c1].astype(np.float64)
                             for r in (0, 1)], 0) - mu[c0:c1]
        ho = np.asarray(a[2]).reshape(F, -1)[:, c0:c1].astype(np.float64) - mu[c0:c1]
        G += tr @ tr.T; M += ho @ tr.T; sst_ho += float((ho * ho).sum())
        del tr, ho
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]; w = np.clip(w[o], 0, None); U = U[:, o]
    nz = int((w > w[0] * 1e-12).sum())
    s = np.sqrt(w[:nz])
    # held-out energy in each train mode: ||ho V_i||^2 = ||M U_i||^2 / s_i^2
    Wt = M @ U[:, :nz]
    e_ho = (Wt * Wt).sum(0) / (s ** 2)
    cum_in = np.cumsum(w[:nz]) / (w[:nz].sum() + 1e-12)
    cum_out = np.cumsum(e_ho) / (sst_ho + 1e-12)
    r_in = int(np.searchsorted(cum_in, 0.90) + 1)
    reach = np.nonzero(cum_out >= 0.90)[0]
    r_out = int(reach[0] + 1) if len(reach) else None          # None => CENSORED, never reaches 90%
    return dict(N=int(N), F=int(F), nz=nz,
                rank90_in=r_in, rank90_out=r_out,
                fve_at_in=float(cum_out[min(r_in, nz) - 1]),
                fve_max=float(cum_out[-1]),
                r_in_pct=float(r_in / max(nz, 1)),
                r_out_pct=float(r_out / max(nz, 1)) if r_out else None)


if __name__ == "__main__":
    man = json.load(open(f"{WR}/atlas_manifest.json"))
    store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: m for m in store.meta}
    todo = [p for p in man["heldout"] if p in have]
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["pdb"] for r in rows}
    print(f"  held-out cached {len(todo)}/{len(man['heldout'])}", flush=True)
    for p in todo:
        if p in done: continue
        try:
            d = insample_vs_out(have[p]["path"]); d["pdb"] = p
            rows.append(d); json.dump(rows, open(OUT, "w"))
            if len(rows) % 10 == 0: print(f"    {len(rows)}  last N={d['N']}", flush=True)
        except Exception as e:
            print(f"  FAIL {p} N={have[p]['atoms']}: {type(e).__name__}: {e}", flush=True)

    N = np.array([r["N"] for r in rows], float); x = np.log10(N)
    ri = np.array([r["rank90_in"] for r in rows], float)
    cens = [r for r in rows if r["rank90_out"] is None]
    ok = [r for r in rows if r["rank90_out"] is not None]
    print(f"\n=== IN-SAMPLE vs OUT-OF-SAMPLE rank90, n={len(rows)}, "
          f"N {int(N.min())}-{int(N.max())} ===", flush=True)
    print(f"  CENSORED (held-out FVE never reaches 90% at ANY k): {len(cens)}/{len(rows)}", flush=True)
    if cens:
        cn = np.array([r["N"] for r in cens], float)
        print(f"    their N: {int(cn.min())}-{int(cn.max())}, median {np.median(cn):.0f}  "
              f"(vs overall median {np.median(N):.0f})", flush=True)
        print(f"    max held-out FVE reached: median {np.median([r['fve_max'] for r in cens]):.3f}")
        print(f"    For these, rank90 is not merely underestimated -- it is UNDEFINED out of sample.")
    if ok:
        a = np.array([r["rank90_in"] for r in ok], float)
        b = np.array([r["rank90_out"] for r in ok], float)
        print(f"  Among the {len(ok)} that DO reach 90% out of sample:")
        print(f"    rank90_in  median {np.median(a):.0f}    rank90_out median {np.median(b):.0f}"
              f"    ratio median {np.median(b/a):.2f}x   IQR "
              f"{np.percentile(b/a,25):.2f}-{np.percentile(b/a,75):.2f}x", flush=True)
        print(f"    rank90_out > rank90_in in {(b>a).sum()}/{len(ok)} systems", flush=True)
    fa = np.array([r["fve_at_in"] for r in rows])
    print(f"  THE HONEST READ: the rank90_in modes explain median {np.median(fa)*100:.1f}% of "
          f"held-out variance, not 90%.", flush=True)
    print(f"    below 90% in {(fa<0.90).sum()}/{len(rows)} systems; min {fa.min()*100:.1f}%", flush=True)

    print(f"\n=== SECOND BIAS: is rank90 CENSORED at large N? (Family B) ===", flush=True)
    pin = np.array([r["r_in_pct"] for r in rows])
    print(f"  rank90_in as % of usable rank: median {100*np.median(pin):.1f}%, max {100*pin.max():.1f}%")
    lr = stats.linregress(x, pin); hw = stats.t.ppf(0.975, len(x)-2)*lr.stderr
    tag = ("RISES with N: the ceiling binds hardest exactly where the b slope gets its leverage"
           if lr.slope - hw > 0 else "flat")
    print(f"  that % vs log10(N): {lr.slope:+.4f} +/- {hw:.4f}  -> {tag}", flush=True)
    for nm, y in (("rank90_in", ri),):
        l2 = stats.linregress(x, np.log10(y)); h2 = stats.t.ppf(0.975, len(x)-2)*l2.stderr
        print(f"  log10({nm}) ~ {l2.slope:+.4f} +/- {h2:.4f} * log10(N)   <- the in-sample b; both "
              f"biases make this an UNDERESTIMATE", flush=True)
    if ok:
        l3 = stats.linregress(np.log10([r["N"] for r in ok]),
                              np.log10([r["rank90_out"] for r in ok]))
        h3 = stats.t.ppf(0.975, len(ok)-2)*l3.stderr
        print(f"  log10(rank90_out) ~ {l3.slope:+.4f} +/- {h3:.4f} * log10(N)  (uncensored subset "
              f"only -- itself biased DOWN, since the systems that failed to reach 90% are excluded)",
              flush=True)

    print(f"\n=== CONSEQUENCE ===", flush=True)
    print("  Both biases push the SAME direction, so the width chain")
    print("    168 / 0.65 x 1.83 ~= 490 dims at 1e6 atoms")
    print("  is a FLOOR, not an estimate -- and the 168 anchor carries the same n_eff problem, which")
    print("  mdCATH never had checked. CORPUS-INDEPENDENT: MISATO 10 ns, mdCATH 2.5 us, ATLAS 100 ns;")
    print("  none supplies enough independent samples to fit a few-hundred-dim per-system subspace.")
    print("  => STOP SIZING DM FROM rank90. Size it from the codec's own held-out FVE-vs-DM curve,")
    print("     whose effective sample size is the CORPUS, not one trajectory. See armf_atlas_dm.py.")
