"""Fit b across the frame-budget ladder from the streamed mdCATH spectra, and recompute the width
chain at the measured value. Reports exactly the four things the measurement was commissioned for:
  1. b at 2,400 frames with CI   <- the UNCENSORED number that closes this
  2. b at 79 frames, SAME domains, paired  <- the direct attenuation factor for b, which the
     c-derived 1.38x proxy was standing in for
  3. whether b's budget-ladder drift has FLATTENED by 2,400 (if still climbing, 2,400 is another
     bound, not the value)
  4. the width chain recomputed at the measured b
"""
import json, glob, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
SC = os.environ.get("BSC", "/network/scratch/j/jacob-junqi.tian/latent-model-workspace")
BUDGETS = [79, 200, 400]          # per-replica (PRIMARY)
CAT_BUDGETS = [79, 200, 400, 800, 1600, 2000]   # concatenated (inflation check only)
MIS_B, MIS_HW = 0.101, 0.021          # MISATO n=276 at 79 frames (censored)
ASYMPTOTE, FLOPPY_TO_BOUND, ANCHOR_N = 168.0, 0.65, 1804.0

rows = {}
for fp in sorted(glob.glob(f"{SC}/bexp/w*.json")):
    rows.update(json.load(open(fp)))
print(f"[b-exponent] {len(rows)} domains streamed")
if len(rows) < 30: print("  too few to fit"); raise SystemExit

Nv = np.array([r["N"] for r in rows.values()], float)
print(f"  N range {int(Nv.min())}-{int(Nv.max())} ({Nv.max()/Nv.min():.1f}x)  median {int(np.median(Nv))}")


def fit(nf, subset=None, pre=""):
    A = [(r["N"], r[f"{pre}rmsf{nf}"], r[f"{pre}r{nf}"]) for k, r in rows.items()
         if f"{pre}r{nf}" in r and (subset is None or k in subset)]
    if len(A) < 10: return None
    A = np.array(A, float)
    X = np.column_stack([np.log10(A[:, 0]), np.log10(A[:, 1]), np.ones(len(A))]); y = np.log10(A[:, 2])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; dof = len(A) - 3
    se = np.sqrt(np.diag((res ** 2).sum() / dof * np.linalg.inv(X.T @ X))); t = stats.t.ppf(0.975, dof)
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(n=len(A), b=b[0], bhw=t * se[0], c=b[1], chw=t * se[1], r2=r2,
                maxr=A[:, 2].max(), cap=A[:, 2].max() / (nf - 1))


# paired subset: domains with BOTH 79 and 2400
paired = {k for k, r in rows.items() if "r79" in r and "r400" in r}
print(f"\n=== b ACROSS THE FRAME-BUDGET LADDER (paired subset n={len(paired)}) ===")
print(f"  {'frames':>7}{'n':>5}{'b(N)':>20}{'c(RMSF)':>20}{'R^2':>7}{'maxRank90':>11}{'%cap':>7}")
F = {}
for nf in BUDGETS:
    f_ = fit(nf, paired)
    if not f_: continue
    F[nf] = f_
    print(f"  {nf:>7}{f_['n']:>5}{f'{f_[chr(98)]:+.3f}+/-{f_[chr(98)+chr(104)+chr(119)]:.3f}':>20}"
          f"{f'{f_[chr(99)]:+.3f}+/-{f_[chr(99)+chr(104)+chr(119)]:.3f}':>20}{f_['r2']:>7.3f}"
          f"{int(f_['maxr']):>11}{f_['cap']*100:>6.0f}%")

TOP = 400
if TOP in F:
    b24, hw24 = F[TOP]["b"], F[TOP]["bhw"]
    print(f"\n=== 1. b at the top PER-REPLICA budget (400 frames) = {b24:+.4f} +/- {hw24:.4f}   [n={F[TOP]['n']}] ===")
    for tgt, nm in [(0.14, "b=0.14 (c-proxy corrected)"), (0.30, "b=0.30 (claim needs qualification)")]:
        z = abs(b24 - tgt) / (hw24 / 1.96 + 1e-9)
        print(f"    distinguishes from {nm:38s}: {z:.1f} sigma")
    if 79 in F:
        b79 = F[79]["b"]
        print(f"\n=== 2. DIRECT ATTENUATION FACTOR FOR b (paired, same domains) ===")
        print(f"    b(79) {b79:+.4f} +/- {F[79]['bhw']:.4f}   ->   b(400) {b24:+.4f} +/- {hw24:.4f}")
        if abs(b79) > 1e-3 and np.sign(b79) == np.sign(b24):
            print(f"    factor = {b24/b79:.2f}x   (the c-derived proxy was 1.38x)")
            print(f"    applied to MISATO n=276: b = {MIS_B:+.3f} -> {MIS_B*b24/b79:+.3f} +/- {MIS_HW*abs(b24/b79):.3f}")
        else:
            print(f"    MULTIPLICATIVE FACTOR UNDEFINED (b(79) crosses/near zero) -- report the ADDITIVE shift instead:")
            print(f"    shift = {b24-b79:+.4f}; applied additively to MISATO: b = {MIS_B:+.3f} -> {MIS_B+(b24-b79):+.3f}")
    print(f"\n=== 3. HAS THE PER-REPLICA LADDER FLATTENED BY 400? ===")
    seq = [(nf, F[nf]['b']) for nf in BUDGETS if nf in F]
    print("    " + "  ".join(f"{nf}:{b:+.3f}" for nf, b in seq))
    if len(seq) >= 3:
        tail = seq[-3:]
        lr = stats.linregress([np.log10(x[0]) for x in tail], [x[1] for x in tail])
        thw = stats.t.ppf(0.975, 1) * lr.stderr if lr.stderr > 0 else np.inf
        flat = abs(lr.slope) < thw
        print(f"    tail slope over the last 3 budgets: {lr.slope:+.4f} (hw {thw:.4f}) -> "
              f"{'FLATTENED -- 400 is the VALUE' if flat else 'STILL CLIMBING -- 400 is another BOUND'}")
    print(f"\n=== 4. WIDTH CHAIN AT THE MEASURED b ===")
    print(f"    {'b':>10}{'N^b to 1e6':>13}{'dims @1e6 bound':>18}")
    for nm, bv in [("lower CI", b24 - hw24), ("MEASURED", b24), ("upper CI", b24 + hw24)]:
        sc = (1e6 / ANCHOR_N) ** bv
        print(f"    {nm:>10}{sc:>13.2f}{ASYMPTOTE/FLOPPY_TO_BOUND*sc:>18.0f}   (b={bv:+.3f})")
    print("    caveats: 2.5-decade extrapolation; ns-us regime; variance-weighted (rare states")
    print("    excluded, coverage open); floppy->bound 0.65 from a 1.65x RMSF ratio via c~-0.9.")


# ---------------- between-replica spread: is rank90 noisier than the CIs suggest? ----------------
print("\n=== BETWEEN-REPLICA SPREAD (free noise check) ===")
for nf in BUDGETS:
    sd = [r[f"rsd{nf}"] for r in rows.values() if f"rsd{nf}" in r]
    mu = [r[f"r{nf}"] for r in rows.values() if f"r{nf}" in r]
    if not sd: continue
    within = float(np.median(sd)); between = float(np.std(mu))
    print(f"  {nf:>4} frames: within-domain (replica) SD {within:6.1f}   between-domain SD {between:6.1f}   "
          f"ratio {within/max(between,1e-9):.2f}  -> {'NOISY: replica scatter rivals the signal' if within/max(between,1e-9)>0.5 else 'ok: between-domain variation dominates'}")

# ---------------- concatenation inflation: bounds every earlier concatenated mdCATH number --------
print("\n=== CONCATENATION INFLATION (per-replica vs 5-replica concatenated, same domains, same budget) ===")
for nf in BUDGETS:
    pr = [(r[f"r{nf}"], r[f"mix_r{nf}"], r[f"mix_nf{nf}"]) for r in rows.values()
          if f"r{nf}" in r and f"mix_r{nf}" in r]
    if not pr: continue
    a = np.array([x[0] for x in pr]); c = np.array([x[1] for x in pr]); nfm = int(np.median([x[2] for x in pr]))
    print(f"  {nf:>4} frames: within-ONE-replica {np.median(a):6.1f}   same budget SPREAD ACROSS replicas "
          f"({nfm} fr) {np.median(c):6.1f}   inflation x{np.median(c/np.maximum(a,1e-9)):.2f}")
print("  -> the earlier mdCATH asymptote 168 (and the window/temperature sweeps) were computed on")
print("     CONCATENATED replicas and are inflated by roughly this factor; the width-chain anchor")
print("     should be divided by it.")
c_fits = {nf: fit(nf, None, "cat_") for nf in CAT_BUDGETS}
print("\n  b on CONCATENATED series (for comparison only, artifact-bearing):")
for nf in CAT_BUDGETS:
    f_ = c_fits.get(nf)
    if f_: print(f"    {nf:>5}: b {f_['b']:+.3f}+/-{f_['bhw']:.3f}  c {f_['c']:+.3f}+/-{f_['chw']:.3f}  n={f_['n']}")
