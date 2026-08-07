"""Scale-up read-out: density-matched pairs, powered slope with the pre-registered asymmetry, and
steps-to-plateau vs DM as well as vs N. Reads whichever result JSONs exist."""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
AVAIL = {(700,1000):18,(1000,2000):410,(2000,4000):1970,(4000,8000):3935,(8000,16000):1717,(16000,32000):416}
PAIRS = [("A", (1000,2000), (16000,32000)), ("B", (2000,4000), (8000,16000))]


def boot_ci(x, n=5000):
    if len(x) < 2: return (float('nan'), float('nan'))
    b = [np.median(np.random.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def read(fn, key, ratio_key, gap_key):
    if not os.path.exists(fn): print(f"  [{fn.split('/')[-1]} not present yet]"); return
    d = json.load(open(fn))
    for arm, v in d.items():
        rows = [r for r in v["rows"] if not r["void"] and np.isfinite(r.get(ratio_key, float('nan')))]
        if len(rows) < 4: print(f"\n  {key}={arm}: {len(rows)} valid rows -- too few"); continue
        N = np.array([r["N"] for r in rows], float); rt = np.array([r[ratio_key] for r in rows], float)
        gp = np.array([r[gap_key] for r in rows], float)
        lr = stats.linregress(np.log10(N), rt); hw = stats.t.ppf(0.975, len(N)-2)*lr.stderr
        rng = np.log10(N.max()) - np.log10(N.min()); base = float(np.median(rt))
        s13 = base*0.3/rng; s20 = base*1.0/rng
        print(f"\n  === {key}={arm}  (held-out n={len(rows)}) ===")
        print(f"    slope {lr.slope:+.4f} +/- {hw:.4f}   R^2 {lr.rvalue**2:.3f}   p={lr.pvalue:.3f}")
        print(f"    thresholds at this baseline: 1.3x -> slope {s13:+.4f} | 2.0x -> slope {s20:+.4f}")
        excl = lr.slope + hw < s20; conf = lr.slope + hw < s13
        print(f"    INDEX-ADDRESSING (>2x): {'EXCLUDED' if excl else 'NOT excluded'}   "
              f"ARBITRARY-L SURVIVES (<1.3x): {'CONFIRMED' if conf else 'not confirmable at this n'}")
        try: EFFECT[(key, float(arm))] = bool(lr.slope - hw > 0)      # is there a positive N effect at all?
        except Exception: pass
        gl = stats.linregress(np.log10(N), gp); ghw = stats.t.ppf(0.975, len(N)-2)*gl.stderr
        print(f"    absolute gap slope {gl.slope:+.4f} +/- {ghw:.4f} (must agree with the ratio)")
        print(f"    DENSITY-MATCHED PAIRS (availability held ~fixed, N varies):")
        for nm, b1, b2 in PAIRS:
            r1 = [r[ratio_key] for r in rows if tuple(r["bucket"]) == b1]
            r2 = [r[ratio_key] for r in rows if tuple(r["bucket"]) == b2]
            if not r1 or not r2: continue
            c1, c2 = boot_ci(r1), boot_ci(r2)
            u = stats.mannwhitneyu(r1, r2, alternative="two-sided") if min(len(r1), len(r2)) > 1 else None
            print(f"      pair {nm}: avail {AVAIL[b1]} vs {AVAIL[b2]} | "
                  f"N {int(np.median([r['N'] for r in rows if tuple(r['bucket'])==b1]))} vs "
                  f"{int(np.median([r['N'] for r in rows if tuple(r['bucket'])==b2]))} | "
                  f"ratio {np.median(r1):.3f}[{c1[0]:.2f},{c1[1]:.2f}] (n={len(r1)}) vs "
                  f"{np.median(r2):.3f}[{c2[0]:.2f},{c2[1]:.2f}] (n={len(r2)})"
                  + (f"  p={u.pvalue:.3f}" if u else ""))
        st = {b: v["g8"][str(b)]["steps"] for b in AVAIL if str(b) in v["g8"]}
        print(f"    steps-to-plateau by bucket: " + " ".join(f"{b[0]//1000}k:{s}" for b, s in st.items() if s))


EFFECT = {}


print("=== L SWEEP (DM=64) ===")
read(f"{WR}/phase1_rows_v2.json", "L", "ratio", "gap")
print("\n=== DM SWEEP (L=1) ===")
read(f"{WR}/phase1_dm_rows.json", "DM", "ratio_anm", "gap_anm")
for fn, key in [(f"{WR}/phase1_dm_rows.json", "DM"), (f"{WR}/phase1_rows_v2.json", "L")]:
    if not os.path.exists(fn): continue
    d = json.load(open(fn)); xs, ys = [], []
    for arm, v in d.items():
        s = [g["steps"] for g in v["g8"].values() if g["steps"]]
        if s: xs.append(float(arm)); ys.append(np.median(s))
    if len(xs) > 2:
        lr = stats.linregress(np.log10(xs), np.log10(ys))
        print(f"\n  steps-to-plateau vs {key}: log10(steps) ~ {lr.slope:+.3f}*log10({key})  R^2 {lr.rvalue**2:.3f}"
              f"   [{' '.join(f'{key}={int(a)}:{int(b)}' for a,b in zip(xs,ys))}]")
        print(f"    -> objective-4: convergence cost scales with {key} " +
              ("YES" if abs(lr.slope) > 0.2 else "NO (flat)"))


# ---------------- THREE-WAY READ (pre-registered; label explicitly) ----------------
print("\n=== THREE-WAY READ ===")
l1 = [v for (k, a), v in EFFECT.items() if a == 1.0]
lhi = [v for (k, a), v in EFFECT.items() if k == "L" and a in (12.0, 24.0)]
if not l1 or not lhi:
    print("  insufficient arms reported yet (need L=1 and L=12/24) -- no label")
else:
    e1 = any(l1); ehi = any(lhi)
    if ehi and not e1:
        print("  OUTCOME A: N effect at L=12/24 but NOT at L=1 -> SLOT ASSIGNMENT / ROUTING.")
        print("    Fix: addressing mechanism. NEXT RUN IS MECHANISTIC, not more systems.")
    elif e1:
        # INBOX 002: the "NOT capacity" inference below is RETIRED. rank90's CI-spans-0 is SUPERSEDED
        # (b = +0.101 +/- 0.021, and b >= 0.93 out of sample) and rank90 is an IN-SAMPLE FLOOR, so
        # capacity must be ruled out EMPIRICALLY by the DM sweep before any pooling/broadcast
        # redesign is proposed. The old text asserted the opposite and is kept only as a record.
        print("  OUTCOME B: N effect at L=1 as well.")
        print("    *** The rank90/TICA basis for EXCLUDING CAPACITY is RETIRED (INBOX 002). Capacity is")
        print("    a LIVE hypothesis: rule it out with the DM sweep (FVE still rising at DM=512 =>")
        print("    information-limited) BEFORE localising to the pooling/broadcast pathway. ***")
        print("    If capacity IS ruled out, this localises to the POOLING/BROADCAST PATHWAY: encoder")
        print("    aggregating N tokens into a fixed code, or decoder broadcasting one code to N atoms.")
        print("    Fix: aggregation architecture (hierarchical pooling / deeper cross-attention / relative-position")
        print("    conditioning). NEXT RUN IS MECHANISTIC, not more systems.")
    else:
        print("  OUTCOME C: no N effect anywhere -> arbitrary-L holds, SUBJECT TO THE STATED POWER LIMIT")
        print("    (failure mode excluded at >2x; success mode <1.3x NOT certified at this n).")
        print("    Pre-committed: do NOT spend 565 sampled systems to certify <1.3x now. Excluding")
        print("    index-addressing is enough to proceed; schedule certification only if something downstream")
        print("    depends on the tighter bound.")
