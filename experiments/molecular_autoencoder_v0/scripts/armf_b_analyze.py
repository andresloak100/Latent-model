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
# PRIMARY = CONCATENATED at 2,000. Censoring is the error that DEMONSTRABLY flips b's sign
# (-0.057 -> +0.246 across the ladder); concatenation is a measured ~3-5% median inflation that
# tests as INTERCEPT-ONLY. Trading a quantified small bias for a sign-inverting artifact would be
# the wrong trade. Rank usage: per-replica 400 puts the rank90 tail near ~50% of available rank
# (past the 30% void line); concatenated 2,000 sits at ~10% median, ~19-24% at the tail.
CAT_BUDGETS = [79, 200, 400, 800, 1600, 2000]   # PRIMARY ladder
JOINS = [1, 2, 3, 5]; MATCHNF = 400
BUDGETS = [79, 200, 400]                        # per-replica: ARTIFACT CHECK
MIS_B, MIS_HW = 0.101, 0.021          # MISATO n=276 at 79 frames (censored)
# WIDTH-CHAIN CONSTANTS -- EVERYTHING DERIVED FROM THESE IS A FLOOR (INBOX 002). ASYMPTOTE=168 is an
# IN-SAMPLE rank90; measured out of sample on ATLAS such modes cover a median 74.7% of held-out
# variance, not 90%, and the shortfall grows with N. b is separately censored-low. Same direction.
ASYMPTOTE, FLOPPY_TO_BOUND, ANCHOR_N = 168.0, 0.65, 1804.0

rows = {}
for fp in sorted(glob.glob(f"{SC}/bexp/w*.json")):
    rows.update(json.load(open(fp)))
print(f"[b-exponent] {len(rows)} domains streamed")
_f = {}
for fp in sorted(glob.glob(f"{SC}/bexp/fail_w*.json")): _f.update(json.load(open(fp)))
if _f:
    fn = np.array([v["projN"] for v in _f.values()], float); kn = np.array([r["N"] for r in rows.values()], float)
    print(f"  FAMILY A CHECK -- {len(_f)} domains FAILED (a filter, not a non-event): "
          f"failed projN median {np.median(fn):.0f} vs kept median {np.median(kn):.0f}; "
          f"{'*** N-CORRELATED FAILURE -- biases b ***' if abs(np.median(fn)-np.median(kn))/max(np.median(kn),1)>0.25 else 'no large N skew'}")
else:
    print("  FAMILY A CHECK -- 0 download/parse failures (no hidden filter)")
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
paired = {k for k, r in rows.items() if "cat_r79" in r and "cat_r2000" in r}
# The paired LADDER subset requires 2,000 frames, which drops SHORT-REPLICA domains -- short replicas
# correlate with unfolding, i.e. FLOPPY. That is a non-random, mobility-correlated exclusion, so it is
# quantified here and the ladder is reported as DIAGNOSTIC. The PRIMARY join-sweep fit below excludes
# NOTHING: it keeps every domain that has the given join count.
_drop = [r for k, r in rows.items() if k not in paired]
if _drop:
    _kept = [r for k, r in rows.items() if k in paired]
    _mk = np.median([r["j1_rmsf"] for r in _kept if "j1_rmsf" in r] or [np.nan])
    _md = np.median([r["j1_rmsf"] for r in _drop if "j1_rmsf" in r] or [np.nan])
    print(f"  LADDER paired subset drops {len(_drop)}/{len(rows)} domains (short replicas). "
          f"medRMSF kept {_mk:.2f} vs dropped {_md:.2f} -- exclusion is mobility-correlated, ladder is DIAGNOSTIC ONLY.")
print(f"\n=== PRIMARY: b ACROSS THE CONCATENATED LADDER (paired subset n={len(paired)}) ===")
print(f"  {'frames':>7}{'n':>5}{'b(N)':>20}{'c(RMSF)':>20}{'R^2':>7}{'maxRank90':>11}{'%cap':>7}")
F = {}
for nf in CAT_BUDGETS:
    f_ = fit(nf, paired, "cat_")
    if not f_: continue
    F[nf] = f_
    print(f"  {nf:>7}{f_['n']:>5}{f'{f_[chr(98)]:+.3f}+/-{f_[chr(98)+chr(104)+chr(119)]:.3f}':>20}"
          f"{f'{f_[chr(99)]:+.3f}+/-{f_[chr(99)+chr(104)+chr(119)]:.3f}':>20}{f_['r2']:>7.3f}"
          f"{int(f_['maxr']):>11}{f_['cap']*100:>6.0f}%")

TOP = 2000
if TOP in F:
    b24, hw24 = F[TOP]["b"], F[TOP]["bhw"]
    print(f"\n=== 1. UNCENSORED b (concatenated, 2,000 frames) = {b24:+.4f} +/- {hw24:.4f}   [n={F[TOP]['n']}] ===")
    for tgt, nm in [(0.14, "b=0.14 (c-proxy corrected)"), (0.30, "b=0.30 (claim needs qualification)")]:
        z = abs(b24 - tgt) / (hw24 / 1.96 + 1e-9)
        print(f"    distinguishes from {nm:38s}: {z:.1f} sigma")
    if 79 in F:
        b79 = F[79]["b"]
        print(f"\n=== 2. DIRECT ATTENUATION FACTOR FOR b (paired, same domains) ===")
        print(f"    b(79) {b79:+.4f} +/- {F[79]['bhw']:.4f}   ->   b(2000) {b24:+.4f} +/- {hw24:.4f}")
        if abs(b79) > 1e-3 and np.sign(b79) == np.sign(b24):
            print(f"    factor = {b24/b79:.2f}x   (the c-derived proxy was 1.38x)")
            print(f"    applied to MISATO n=276: b = {MIS_B:+.3f} -> {MIS_B*b24/b79:+.3f} +/- {MIS_HW*abs(b24/b79):.3f}")
        else:
            print(f"    MULTIPLICATIVE FACTOR UNDEFINED (b(79) crosses/near zero) -- report the ADDITIVE shift instead:")
            print(f"    shift = {b24-b79:+.4f}; applied additively to MISATO: b = {MIS_B:+.3f} -> {MIS_B+(b24-b79):+.3f}")
    print(f"\n=== 3. HAS THE LADDER FLATTENED BY 2,000? ===")
    seq = [(nf, F[nf]['b']) for nf in CAT_BUDGETS if nf in F]
    print("    " + "  ".join(f"{nf}:{b:+.3f}" for nf, b in seq))
    if len(seq) >= 3:
        tail = seq[-3:]
        lr = stats.linregress([np.log10(x[0]) for x in tail], [x[1] for x in tail])
        thw = stats.t.ppf(0.975, 1) * lr.stderr if lr.stderr > 0 else np.inf
        flat = abs(lr.slope) < thw
        print(f"    tail slope over the last 3 budgets: {lr.slope:+.4f} (hw {thw:.4f}) -> "
              f"{'FLATTENED -- 2,000 is the VALUE' if flat else 'STILL CLIMBING -- 2,000 is another BOUND'}")
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


# ---------------- DOES CONCATENATION BIAS THE SLOPE OR ONLY THE INTERCEPT? (at full n) ----------
print("\n=== DECIDING TEST: per-domain inflation factor vs N ===")
print("  uncorrelated -> intercept-only shift, b unaffected, concatenated ladder is correct as primary")
print("  correlated   -> it biases the slope; report b BOTH ways and state the divergence")
for nf in BUDGETS:
    P = [(r["N"], r[f"mix_r{nf}"] / max(r[f"r{nf}"], 1e-9)) for r in rows.values()
         if f"r{nf}" in r and f"mix_r{nf}" in r]
    if len(P) < 20: continue
    A = np.array(P, float)
    lr = stats.linregress(np.log10(A[:, 0]), A[:, 1]); ci = stats.t.ppf(0.975, len(A) - 2) * lr.stderr
    pr = stats.pearsonr(np.log10(A[:, 0]), A[:, 1])
    biased = (lr.slope - ci > 0) or (lr.slope + ci < 0)
    # what would this slope do to b, over the actual N range?
    rng = np.log10(A[:, 0].max()) - np.log10(A[:, 0].min())
    med = np.median(A[:, 1])
    db_pt = np.log10((med + lr.slope * rng) / max(med, 1e-9)) / max(rng, 1e-9)
    db_hi = np.log10((med + (lr.slope + ci) * rng) / max(med, 1e-9)) / max(rng, 1e-9)
    print(f"  {nf:>4} frames (n={len(A)}): median inflation x{med:.3f}   "
          f"slope {lr.slope:+.4f} +/- {ci:.4f}   r={pr[0]:+.3f} p={pr[1]:.3f}")
    print(f"       -> {'CORRELATED: biases the slope' if biased else 'UNCORRELATED: intercept-only'};"
          f"  implied bias on b: point {db_pt:+.4f}, at CI upper {db_hi:+.4f}")
print("\n  ARTIFACT-CHECK ladder (per-replica, censored at the top -- reported, not primary):")
for nf in BUDGETS:
    f_ = fit(nf, None, "")
    if f_: print(f"    {nf:>4}: b {f_['b']:+.3f}+/-{f_['bhw']:.3f}  c {f_['c']:+.3f}+/-{f_['chw']:.3f}  "
                 f"n={f_['n']}  maxRank90 {int(f_['maxr'])} = {f_['cap']*100:.0f}% of rank")


# ---------------- REPLICA-COUNT SWEEP: measure the optimum instead of arguing it ----------------
print("\n=== REPLICA-COUNT SWEEP: censoring vs concatenation move OPPOSITE ways; find the optimum ===")
print(f"  {'joins':>6}{'n':>5}{'totalFrames':>13}{'usable':>8}{'rank90':>9}{'%ofRank':>9}"
      f"{'inflation':>11}{'b (CI)':>20}{'c (CI)':>20}")
JF = {}
for J in JOINS:
    A = [(r["N"], r[f"j{J}_rmsf"], r[f"j{J}_r"], r[f"j{J}_nf"], r[f"j{J}_pct"],
          r.get(f"j{J}_mix", np.nan), r.get(f"j1_mix", np.nan))
         for r in rows.values() if f"j{J}_r" in r]
    if len(A) < 20: continue
    A = np.array(A, float)
    X = np.column_stack([np.log10(A[:, 0]), np.log10(A[:, 1]), np.ones(len(A))]); y = np.log10(A[:, 2])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; dof = len(A) - 3
    se = np.sqrt(np.diag((res ** 2).sum() / dof * np.linalg.inv(X.T @ X))); tc = stats.t.ppf(0.975, dof)
    infl = np.nanmedian(A[:, 5] / np.maximum(A[:, 6], 1e-9)) if J > 1 else 1.0
    JF[J] = dict(b=b[0], hw=tc * se[0], c=b[1], chw=tc * se[1], n=len(A),
                 nf=np.median(A[:, 3]), pct=np.median(A[:, 4]), infl=infl)
    print(f"  {J:>6}{len(A):>5}{np.median(A[:,3]):>13.0f}{np.median(A[:,3])-1:>8.0f}"
          f"{np.median(A[:,2]):>9.0f}{np.median(A[:,4])*100:>8.0f}%{infl:>11.3f}"
          f"{f'{b[0]:+.3f}+/-{tc*se[0]:.3f}':>20}{f'{b[1]:+.3f}+/-{tc*se[1]:.3f}':>20}")

if len(JF) >= 3:
    ks = sorted(JF); bs = np.array([JF[k]["b"] for k in ks]); hw = np.array([JF[k]["hw"] for k in ks])
    stable = [k for k in ks if k >= 2]
    if len(stable) >= 2:
        bb = np.array([JF[k]["b"] for k in stable]); hh = np.array([JF[k]["hw"] for k in stable])
        spread = bb.max() - bb.min(); typ = float(np.mean(hh))
        print(f"\n  b across joins 2-5: " + " ".join(f"J{k}:{JF[k]['b']:+.3f}" for k in stable))
        print(f"  FAMILY C: a 'stable' verdict here is a NULL -- it permits a b-shift of up to "
              f"{spread+typ:.3f} across joins, which moves the 1e6 width chain by "
              f"x{(1e6/ANCHOR_N)**(spread+typ):.2f}")
        print(f"  spread {spread:.3f} vs typical CI half-width {typ:.3f}  -> "
              f"{'STABLE: the join choice does not matter, and that stability IS the answer' if spread < typ else 'MOVES WITH JOIN COUNT: concatenation slope-bias detected DIRECTLY'}")
        lr = stats.linregress([np.log10(k) for k in stable], bb)
        print(f"  b vs log10(joins): slope {lr.slope:+.4f}  (p={lr.pvalue:.3f})")
    # pick primary from the measured curve
    print("\n  RANK-USAGE DISTRIBUTION per join (censoring is system-dependent -- the median hides it):")
    for J in ks:
        P = np.array([r[f"j{J}_pct"] for r in rows.values() if f"j{J}_pct" in r])
        print(f"    J={J}: median {np.median(P)*100:>5.1f}%  p90 {np.percentile(P,90)*100:>5.1f}%  "
              f"max {P.max()*100:>5.1f}%   frac over 30% void line: {(P>0.30).mean()*100:>4.0f}%")
    ok = [k for k in ks if np.percentile([r[f"j{k}_pct"] for r in rows.values() if f"j{k}_pct" in r], 90) < 0.30]
    if ok:
        pick = min(ok, key=lambda k: JF[k]["infl"])
        print(f"\n  PRIMARY BY MEASUREMENT: J={pick}  (rank usage {JF[pick]['pct']*100:.0f}% < 30% void line, "
              f"lowest inflation x{JF[pick]['infl']:.3f} among uncensored joins)")
        print(f"    b = {JF[pick]['b']:+.4f} +/- {JF[pick]['hw']:.4f}   [n={JF[pick]['n']}]")
        sc = (1e6 / ANCHOR_N) ** JF[pick]["b"]
        print(f"    width chain at this b: >= {ASYMPTOTE/FLOPPY_TO_BOUND*sc:.0f} dims @1e6-atom bound "
              f"complex -- A FLOOR (in-sample anchor + censored b). Do not size DM from it; use the\n"
              f"    codec saturation curve (armf_atlas_dm.py). See ROADMAP 'WIDTH CHAIN IS A FLOOR'.")
    else:
        print("\n  NO JOIN COUNT CLEARS THE 30% RANK LINE -- every option is censored; report as such.")


# ---------------- ITEM 2: RANK-USAGE THRESHOLD SWEEP, AT FIXED JOIN ----------------
# DIAGNOSTIC SUBSETTING, NOT AN EXCLUSION RULE. The primary fit keeps every domain. Here we fit b on
# progressively less-censored subsets to see whether b DRIFTS -- drift measures the censoring bias
# directly and extrapolates to a corrected b. Axes are crossed one at a time: this holds JOIN fixed
# and varies THRESHOLD; the sweep above holds THRESHOLD fixed (none) and varies JOIN.
print("\n=== ITEM 2: b vs RANK-USAGE THRESHOLD (join fixed) -- measures censoring bias directly ===")
for JFIX in [j for j in JOINS if j >= 3]:
    have = [(k, r) for k, r in rows.items() if f"j{JFIX}_r" in r]
    if len(have) < 40: continue
    print(f"  --- join J={JFIX} (n available {len(have)}) ---")
    print(f"    {'thresh':>8}{'nKept':>7}{'%kept':>7}{'N range':>16}{'medRMSF':>9}{'b (CI)':>20}{'c (CI)':>20}")
    prev = None
    for th in [0.40, 0.30, 0.20, 0.15, 0.10]:
        sub = [(k, r) for k, r in have if r[f"j{JFIX}_pct"] < th]
        if len(sub) < 25: 
            print(f"    {th:>8.2f}{len(sub):>7}   too few to fit"); continue
        A = np.array([[r["N"], r[f"j{JFIX}_rmsf"], r[f"j{JFIX}_r"]] for _, r in sub], float)
        X = np.column_stack([np.log10(A[:, 0]), np.log10(A[:, 1]), np.ones(len(A))]); y = np.log10(A[:, 2])
        bb, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ bb; dof = len(A) - 3
        se = np.sqrt(np.diag((res ** 2).sum() / dof * np.linalg.inv(X.T @ X))); tc = stats.t.ppf(0.975, dof)
        _rng = np.log10(A[:, 0].max()) - np.log10(A[:, 0].min())
        _flag = "  <-- FAMILY D: N range collapsed, b unidentifiable" if _rng < 0.5 else ""
        print(f"    {th:>8.2f}{len(A):>7}{len(A)/len(have)*100:>6.0f}%"
              f"{f'{int(A[:,0].min())}-{int(A[:,0].max())}':>16}{np.median(A[:,1]):>9.2f}"
              f"{f'{bb[0]:+.3f}+/-{tc*se[0]:.3f}':>20}{f'{bb[1]:+.3f}+/-{tc*se[1]:.3f}':>20}{_flag}")
        prev = (th, bb[0])
    print("    read: b STABLE across thresholds -> censoring is not biasing b.")
    print("          b DRIFTS UPWARD as the threshold tightens -> that IS the censoring bias, measured;")
    print("          extrapolate the trend to threshold->0 for a corrected b.")
    print("    NOTE: tightening the threshold preferentially keeps FLOPPY/low-rank systems, so the N")
    print("          range and mobility range shrink too -- both are printed so the loss of leverage is visible.")
