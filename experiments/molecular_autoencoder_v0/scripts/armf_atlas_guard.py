"""RE-MEASURE the PCA-ceiling guard under the NEW protocol.

The -0.1096 +/- 0.1311 on record is STALE: it evaluated on held-out frames from the SAME trajectory.
PCA is now fitted on replicas 0+1 and evaluated on replica 2 -- a different and harder test, so the
ceiling should DROP and the N-slope may move in either direction. Neither old number may be quoted.

The guard does NOT gate the run: the PRIMARY comparison (codec vs ANM) is two zero-shot methods
scored on the same held-out frames -- no ceiling, no denominator, immune to this. A failed guard
damages only the SECONDARY oracle-fraction, which is then reported WITH this slope attached and the
bias direction stated (a ceiling that degrades with N makes the high-N oracle-fraction an UPPER
BOUND, not a flattering artifact).

Also reports, per k: rank validity per system (h varies), and the s_k/s_1 conditioning ratio so a
high-k ceiling cannot silently be fitting numerical noise."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ceiling_chunked
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
KS = [24, 256]

man = json.load(open(f"{WR}/atlas_manifest.json"))
store = AtlasStore(f"{WR}/atlas_cache")
have = {m["pdb"]: i for i, m in enumerate(store.meta)}
ho = [p for p in man["heldout"] if p in have]
print(f"  held-out cached {len(ho)}/{len(man['heldout'])}", flush=True)

rows = []
for k, p in enumerate(ho):
    d = sysdata(store, have[p])
    if d is None:
        continue
    c = ceiling_chunked(d, KS)
    rec = {"pdb": d["pdb"], "N": d["N"]}
    for kk in KS:
        rec[str(kk)] = c[kk]
        rec[f"valid{kk}"] = c[f"valid{kk}"]
        rec[f"cond{kk}"] = c[f"cond{kk}"]
    rows.append(rec)
    json.dump(rows, open(f"{WR}/atlas_guard.json", "w"))
    if (k + 1) % 10 == 0:
        print(f"    {k+1}/{len(ho)}  last N={d['N']}", flush=True)

store.conservation_report()
A = np.array([[r["N"]] + [r[str(k)] for k in KS] for r in rows])
x = np.log10(A[:, 0]); span = x.max() - x.min()
print(f"\n=== GUARD, NEW PROTOCOL (fit replicas 0+1 -> evaluate replica 2), n={len(A)} ===", flush=True)
for i, k in enumerate(KS):
    y = A[:, 1 + i]
    lr = stats.linregress(x, y); hw = stats.t.ppf(0.975, len(A) - 2) * lr.stderr
    flat = abs(lr.slope) < hw
    nv = sum(1 for r in rows if not r[f"valid{k}"])
    print(f"  PCA-{k:<4} median {np.median(y):.4f}   slope {lr.slope:+.4f} +/- {hw:.4f}   "
          f"p={lr.pvalue:.3f}   -> {'FLAT' if flat else '*** DEGRADES WITH N ***'}", flush=True)
    print(f"           FAMILY C: permits up to {abs(lr.slope)+hw:+.4f}, i.e. "
          f"{(abs(lr.slope)+hw)*span:+.3f} FVE across the range;  rank-void {nv}/{len(rows)};  "
          f"median s_k/s_1 {np.median([r[f'cond{k}'] for r in rows]):.2e}", flush=True)
print(f"  realised log-N span {span:.2f} decades"
      + ("" if span >= 1.2 else "   <-- UNDER 1.2 decades: guard UNTESTED"), flush=True)
print("\n  PRIMARY (codec vs ANM) is ceiling-free and unaffected either way.", flush=True)
