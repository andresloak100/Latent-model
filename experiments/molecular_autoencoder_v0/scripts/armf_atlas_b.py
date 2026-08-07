"""b ON ATLAS -- the atom-count exponent of intrinsic dimensionality, measured on the right corpus.

b decides whether "one global latent, width independent of atom count" holds at 1e6 atoms. It was
known only as a CENSORED LOWER BOUND (+0.101 +/- 0.021, MISATO, 79 frames), and mdCATH's 28 domains
could not measure it (CI spanned zero at every frame budget).

WHY ATLAS INSTEAD OF THE STAGED mdCATH STREAM (which is CANCELLED, not merely deprioritised):
    N range        56x                          vs mdCATH 11x
    systems        825 cached (1,938 available) vs 28
    frames         7,503                        vs 2,000
    rank position  ~500/5,002 = 10% (clean)     vs ~500/1,600 = 31% (EDGE)
And it costs nothing: the cache is already built for the learning curve. Measuring the exponent that
carries the 1e6-atom claim on the corpus with 5x the N range and 30x the systems is not a close call.

MACHINERY CARRIED OVER UNCHANGED, with ATLAS REPLICAS as the join unit:
 - JOIN SWEEP (1, 2, 3 replicas). Concatenation inflation grows with joins while censoring shrinks
   with total frames, so the optimum is MEASURED, not argued. Per join: total frames, rank90 as % of
   usable rank (censoring axis), budget-matched inflation vs 1 replica (concatenation axis), and b.
   Selection uses the p90 of rank usage, not the median: censoring is strongly system-dependent, and
   a median-clean join can still be censored for the RIGID tail, which is where the slope's leverage
   lives.
 - THRESHOLD SWEEP at fixed join (rank usage < 10/15/20/30/40%). DIAGNOSTIC SUBSETTING, NEVER an
   exclusion rule: the primary fit keeps every system. Drift as the threshold tightens measures the
   censoring bias directly. Retained N range and median RMSF are printed, and a collapsed N range is
   flagged (Family D: b becomes unidentifiable while noise mimics drift).
 - MOBILITY CONTROL: log10(rank90) ~ b*log10(N) + c*log10(RMSF). The naive N-exponent is a MEDIATED
   CONFOUND -- bigger proteins are more rigid (corr(logN, logRMSF) = -0.31) and rigidity raises
   rank90 (c ~ -0.9 to -1.35), so an uncontrolled fit reports the mobility channel as a size effect.
 - CONSERVATION OF n, and every failure logged with its N so an exclusion cannot be silent.
"""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
OUT = f"{WR}/atlas_b.json"
JOINS = [1, 2, 3]
MATCHNF = 2000                 # fixed budget for the inflation axis, so joins are comparable
THRESHOLDS = [0.40, 0.30, 0.20, 0.15, 0.10]
# WIDTH-CHAIN CONSTANTS -- EVERY NUMBER DERIVED FROM THESE IS A FLOOR (INBOX 002).
# ASYMPTOTE=168 is an IN-SAMPLE rank90: PCA scored on the very frames it was fitted to, which it
# explains optimally by construction. Measured out of sample on ATLAS, in-sample rank90 modes cover a
# median of 74.7% of held-out variance rather than 90%, and the shortfall GROWS with N (2.26x -> 7.70x,
# undefined at N=33,377). FLOPPY_TO_BOUND=0.65 is derived through c on the same under-sampled spectra.
# b is separately censored-low. So the chain floors the requirement; it does not estimate it, and it
# must never be printed as though it did.
ANCHOR_N, ASYMPTOTE, FLOPPY_TO_BOUND = 1804.0, 168.0, 0.65


def spectra(path, joins, matchnf):
    """rank90 and RMSF per join count, plus the budget-matched inflation term. Column-chunked so
    nothing of size (F, 3N) is ever materialised."""
    a = np.load(path, mmap_mode="r")
    R, F, N = a.shape[0], a.shape[1], a.shape[2]
    out = {"N": int(N), "F": int(F), "R": int(R)}
    for J in joins:
        if R < J: continue
        nf = J * F
        G = np.zeros((nf, nf)); ssq = 0.0
        for c0 in range(0, 3 * N, 20000):
            c1 = min(c0 + 20000, 3 * N)
            X = np.concatenate([np.asarray(a[r]).reshape(F, -1)[:, c0:c1] for r in range(J)], 0).astype(np.float64)
            X -= X.mean(0)
            G += X @ X.T; ssq += float((X ** 2).sum())
            del X
        w = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
        cum = np.cumsum(w) / (w.sum() + 1e-12)
        r90 = int(np.searchsorted(cum, 0.90) + 1)
        out[f"j{J}_r"] = r90; out[f"j{J}_nf"] = nf
        out[f"j{J}_pct"] = float(r90 / max(nf - 1, 1))
        out[f"j{J}_rmsf"] = float(np.sqrt(ssq / (nf * N)))
        # ---- INBOX 011 / Q2: b measured OUT OF SAMPLE, both ways ----
        # rank90 above is IN-SAMPLE: the basis is fitted and scored on the same frames, which it
        # explains optimally by construction. On ATLAS the in-sample rank90 modes cover a median
        # 77.1% of held-out variance, not 90%, and the exponent roughly DOUBLES out of sample -- so an
        # in-sample b is not the quantity the architecture question needs. And per 011 the
        # out-of-sample count must be reported BOTH in the train basis's own ORDER (which charges the
        # basis for ordering worse at large N) and ORDERING-FREE (which does not). The codec's decoder
        # is learned and structure-conditioned, so it is not locked to a fixed component order the way
        # a PCA basis is -- but it must also allocate ZERO-SHOT, without seeing the trajectory, so the
        # ordering-free number is a LOWER bound on what it owes and the ordered one an UPPER bound.
        if J < R:                                        # a replica remains to hold out
            hoR = J                                      # first unused replica
            M = np.zeros((F, nf)); ss_f = np.zeros(F)
            mu = np.zeros(3 * N)
            for c0 in range(0, 3 * N, 20000):
                c1 = min(c0 + 20000, 3 * N)
                X = np.concatenate([np.asarray(a[r]).reshape(F, -1)[:, c0:c1]
                                    for r in range(J)], 0).astype(np.float64)
                m = X.mean(0); X -= m
                ho = np.asarray(a[hoR]).reshape(F, -1)[:, c0:c1].astype(np.float64) - m
                M += ho @ X.T; ss_f += (ho * ho).sum(1)
                del X, ho
            wv, Uv = np.linalg.eigh(G); ov = np.argsort(wv)[::-1]
            wv = np.clip(wv[ov], 0, None); Uv = Uv[:, ov]
            nzv = int((wv > wv[0] * 1e-12).sum()); sv = np.sqrt(wv[:nzv])
            Wt = M @ Uv[:, :nzv]
            eh = (Wt * Wt).sum(0) / (sv ** 2)
            sst = float(ss_f.sum())
            cO = np.cumsum(eh) / (sst + 1e-12)           # TRAIN ORDER
            hit = np.nonzero(cO >= 0.90)[0]
            out[f"j{J}_rout"] = int(hit[0] + 1) if len(hit) else None
            # ordering-free, CROSS-FIT on alternating ~300-frame blocks (tau_int ~600 of 2,501, so a
            # contiguous split would put a slow drift entirely in one half)
            idx = np.arange(F); blk = idx // 300
            A_ = idx[blk % 2 == 0]; B_ = idx[blk % 2 == 1]
            if len(A_) > 1 and len(B_) > 1:
                eA = (Wt[A_] ** 2).sum(0) / (sv ** 2)
                eB = (Wt[B_] ** 2).sum(0) / (sv ** 2)
                cB = np.cumsum(eB[np.argsort(eA)[::-1]]) / (float(ss_f[B_].sum()) + 1e-12)
                hb = np.nonzero(cB >= 0.90)[0]
                out[f"j{J}_rsort"] = int(hb[0] + 1) if len(hb) else None
            out[f"j{J}_holdout_rep"] = hoR
        # BUDGET-MATCHED inflation: same total frames, spread across J replicas vs taken from one.
        take = matchnf // J
        if take >= 8 and take <= F:
            Gm = np.zeros((take * J, take * J))
            for c0 in range(0, 3 * N, 20000):
                c1 = min(c0 + 20000, 3 * N)
                Xm = np.concatenate([np.asarray(a[r][:take]).reshape(take, -1)[:, c0:c1] for r in range(J)], 0).astype(np.float64)
                Xm -= Xm.mean(0); Gm += Xm @ Xm.T; del Xm
            wm = np.clip(np.linalg.eigvalsh(Gm)[::-1], 0, None)
            out[f"j{J}_mix"] = int(np.searchsorted(np.cumsum(wm) / (wm.sum() + 1e-12), 0.90) + 1)
    return out


def fit(rows, key, sub=None, which="r"):
    """log10(rank90) ~ b*log10(N) + c*log10(RMSF). Mobility control is mandatory: the naive
    N-exponent is a mediated confound.

    `which` selects the rank90 variant: "r" in-sample, "rout" out-of-sample in TRAIN ORDER,
    "rsort" out-of-sample ORDERING-FREE (cross-fit). All three are reported -- on ATLAS the
    in-sample and out-of-sample exponents differ by roughly 2x, so an in-sample b answers a
    different question from the one the architecture claim needs."""
    A = [(r["N"], r[f"{key}_rmsf"], r[f"{key}_{which}"]) for r in rows
         if r.get(f"{key}_{which}") and f"{key}_rmsf" in r and (sub is None or r["pdb"] in sub)]
    if len(A) < 8: return None
    A = np.array(A, float)
    X = np.column_stack([np.log10(A[:, 0]), np.log10(np.maximum(A[:, 1], 1e-6)), np.ones(len(A))])
    y = np.log10(np.maximum(A[:, 2], 1))
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; dof = len(A) - 3
    se = np.sqrt(np.diag((res ** 2).sum() / dof * np.linalg.inv(X.T @ X))); t = stats.t.ppf(0.975, dof)
    return dict(n=len(A), b=float(b[0]), bhw=float(t * se[0]), c=float(b[1]), chw=float(t * se[1]),
                r2=float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                nrange=float(np.log10(A[:, 0].max() / A[:, 0].min())))


if __name__ == "__main__":
    store = AtlasStore(f"{WR}/atlas_cache")
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["pdb"] for r in rows}
    for i, m in enumerate(store.meta):
        if m["pdb"] in done: continue
        try:
            rec = spectra(m["path"], JOINS, MATCHNF); rec["pdb"] = m["pdb"]
            rows.append(rec); json.dump(rows, open(OUT, "w"))
            if len(rows) % 10 == 0:
                print(f"    {len(rows)}/{len(store.meta)}  last N={rec['N']}", flush=True)
        except Exception as e:
            print(f"  FAIL {m['pdb']} N={m['atoms']}: {type(e).__name__}", flush=True)

    N = np.array([r["N"] for r in rows], float)
    print(f"\n[atlas-b] {len(rows)} systems, N {int(N.min())}-{int(N.max())} "
          f"({N.max()/N.min():.0f}x, {np.log10(N.max()/N.min()):.2f} decades)", flush=True)

    print("\n=== REPLICA-JOIN SWEEP: censoring and concatenation move OPPOSITE ways ===", flush=True)
    print(f"  {'joins':>6}{'n':>5}{'frames':>9}{'rank90':>9}{'%rank med':>11}{'%rank p90':>11}"
          f"{'inflation':>11}{'b (CI)':>20}{'c (CI)':>20}")
    JF = {}
    for J in JOINS:
        f_ = fit(rows, f"j{J}")
        if not f_: continue
        P = np.array([r[f"j{J}_pct"] for r in rows if f"j{J}_pct" in r])
        inf = np.median([r[f"j{J}_mix"] / max(r["j1_mix"], 1) for r in rows
                         if f"j{J}_mix" in r and "j1_mix" in r]) if J > 1 else 1.0
        JF[J] = dict(pct=float(np.median(P)), p90=float(np.percentile(P, 90)), infl=float(inf), **f_)
        print(f"  {J:>6}{f_['n']:>5}{int(np.median([r[f'j{J}_nf'] for r in rows if f'j{J}_nf' in r])):>9}"
              f"{int(np.median([r[f'j{J}_r'] for r in rows if f'j{J}_r' in r])):>9}"
              f"{np.median(P)*100:>10.1f}%{np.percentile(P,90)*100:>10.1f}%{inf:>11.3f}"
              f"{f'{f_[chr(98)]:+.3f}+/-{f_[chr(98)+chr(104)+chr(119)]:.3f}':>20}"
              f"{f'{f_[chr(99)]:+.3f}+/-{f_[chr(99)+chr(104)+chr(119)]:.3f}':>20}", flush=True)

    ok = [J for J in JF if JF[J]["p90"] < 0.30]
    if ok:
        pick = min(ok, key=lambda J: JF[J]["infl"])
        print(f"\n  PRIMARY BY MEASUREMENT: J={pick} (p90 rank usage {JF[pick]['p90']*100:.1f}% < 30%, "
              f"lowest inflation x{JF[pick]['infl']:.3f} among uncensored joins)", flush=True)
        bb, hw = JF[pick]["b"], JF[pick]["bhw"]
        print(f"    b = {bb:+.4f} +/- {hw:.4f}   c = {JF[pick]['c']:+.4f} +/- {JF[pick]['chw']:.4f}   "
              f"R^2 {JF[pick]['r2']:.3f}   n={JF[pick]['n']}   N range {JF[pick]['nrange']:.2f} decades", flush=True)
        # ---- Q2/011: the SAME fit on all three rank90 variants ----
        print(f"\n  === b IN-SAMPLE vs OUT-OF-SAMPLE, ORDERED vs ORDERING-FREE (J={pick}) ===", flush=True)
        print(f"  On ATLAS the in-sample and out-of-sample rank90 exponents differ by roughly 2x, so an")
        print(f"  in-sample b answers a different question from the architecture claim. And per 011 the")
        print(f"  out-of-sample count in TRAIN ORDER also charges the basis for ordering worse at large")
        print(f"  N -- a defect the codec's learned, structure-conditioned decoder does not share.")
        VB = {}
        for wk, lab in (("r", "in-sample"), ("rout", "out-of-sample, TRAIN ORDER"),
                        ("rsort", "out-of-sample, ORDERING-FREE (cross-fit)")):
            fv = fit(rows, f"j{pick}", which=wk)
            if not fv:
                print(f"    {lab:44s}  n<8, not fitted", flush=True); continue
            VB[wk] = fv
            print(f"    {lab:44s}  b = {fv['b']:+.4f} +/- {fv['bhw']:.4f}   "
                  f"c = {fv['c']:+.4f} +/- {fv['chw']:.4f}   R^2 {fv['r2']:.3f}   n={fv['n']}", flush=True)
        if "rout" in VB and "rsort" in VB:
            lo, hi = VB["rsort"]["b"], VB["rout"]["b"]
            print(f"\n    THE INTERVAL THE CODEC OWES: [{min(lo,hi):+.4f}, {max(lo,hi):+.4f}].")
            print(f"    ORDERING-FREE is a LOWER bound -- it grants a perfect ordering chosen with")
            print(f"    knowledge of the held-out trajectory, which the codec must instead produce")
            print(f"    ZERO-SHOT from structure alone. TRAIN ORDER is an UPPER bound -- it locks the")
            print(f"    allocation to a fixed global component order the learned decoder is not bound")
            print(f"    to. Neither endpoint may be quoted alone as 'the' exponent.", flush=True)
        if "r" in VB and "rsort" in VB:
            print(f"    in-sample -> ordering-free out-of-sample: {VB['r']['b']:+.4f} -> "
                  f"{VB['rsort']['b']:+.4f} ({VB['rsort']['b']-VB['r']['b']:+.4f})", flush=True)
        nA = sum(1 for r in rows if r.get(f"j{pick}_rout") is None and f"j{pick}_r" in r)
        if nA:
            NA = [r["N"] for r in rows if r.get(f"j{pick}_rout") is None and f"j{pick}_r" in r]
            NK = [r["N"] for r in rows if r.get(f"j{pick}_rout")]
            print(f"    FAMILY A: {nA} systems never reach 90% out of sample; dropped median N "
                  f"{np.median(NA):.0f} vs kept {np.median(NK):.0f}"
                  + ("   <-- DROPS AT HIGH N: every out-of-sample b above is biased LOW"
                     if np.median(NA) > np.median(NK) else ""), flush=True)

        print(f"\n  === WIDTH CHAIN AT THE MEASURED b -- ALL VALUES ARE FLOORS ===", flush=True)
        for nm, v in (("lower CI", bb - hw), ("MEASURED", bb), ("upper CI", bb + hw)):
            sc = (1e6 / ANCHOR_N) ** v
            if v <= 0:
                print(f"    {nm:>9}  b={v:+.4f}  -> b <= 0: the chain is UNDEFINED here (a "
                      f"non-positive exponent means dimensionality does not grow with N, so there "
                      f"is no extrapolation to make). Not '0 dims'.", flush=True)
                continue
            print(f"    {nm:>9}  b={v:+.4f}  N^b to 1e6 = {sc:>7.2f}  -> "
                  f">= {ASYMPTOTE/FLOPPY_TO_BOUND*sc:>7.0f} dims for a 1e6-atom bound complex "
                  f"(LOWER BOUND)", flush=True)
        print("    caveats: ns-us regime; variance-weighted (rare states excluded); floppy->bound 0.65", flush=True)
        print("    AND: the 168 anchor is an IN-SAMPLE rank90 (its modes cover a median 74.7% of", flush=True)
        print("    held-out variance, not 90%, with the shortfall GROWING with N), while b is", flush=True)
        print("    censored-low. Two biases, same direction => these are FLOORS, not estimates.", flush=True)
        print("    DO NOT SIZE DM FROM THEM -- size it from the codec's own held-out FVE-vs-DM", flush=True)
        print("    saturation curve (armf_atlas_dm.py). See ROADMAP 'WIDTH CHAIN IS A FLOOR'.", flush=True)
        bs = [JF[J]["b"] for J in sorted(JF)]
        if len(bs) >= 2:
            spread = max(bs) - min(bs); typ = float(np.mean([JF[J]["bhw"] for J in JF]))
            print(f"\n  b across joins: " + " ".join(f"J{J}:{JF[J]['b']:+.3f}" for J in sorted(JF)))
            print(f"    spread {spread:.3f} vs typical CI half-width {typ:.3f} -> "
                  f"{'STABLE: join choice does not matter, and that stability IS the answer' if spread < typ else 'MOVES WITH JOIN COUNT: concatenation slope-bias detected DIRECTLY'}", flush=True)
    else:
        print("\n  NO join clears the 30% p90 rank line -- every option is censored; report as such.", flush=True)
        pick = max(JF) if JF else None

    if pick:
        print(f"\n=== THRESHOLD SWEEP at fixed join J={pick} (DIAGNOSTIC subsetting, NOT exclusion) ===", flush=True)
        print(f"  {'thresh':>8}{'nKept':>7}{'%kept':>7}{'N range(dec)':>14}{'medRMSF':>10}{'b (CI)':>20}{'c (CI)':>20}")
        have = [r for r in rows if f"j{pick}_pct" in r]
        for th in THRESHOLDS:
            sub = {r["pdb"] for r in have if r[f"j{pick}_pct"] < th}
            f_ = fit(rows, f"j{pick}", sub)
            if not f_:
                print(f"  {th:>8.2f}{len(sub):>7}   too few to fit"); continue
            flag = "  <-- FAMILY D: N range collapsed, b unidentifiable" if f_["nrange"] < 0.5 else ""
            rm = np.median([r[f"j{pick}_rmsf"] for r in have if r["pdb"] in sub])
            print(f"  {th:>8.2f}{f_['n']:>7}{f_['n']/len(have)*100:>6.0f}%{f_['nrange']:>14.2f}{rm:>10.3f}"
                  f"{f'{f_[chr(98)]:+.3f}+/-{f_[chr(98)+chr(104)+chr(119)]:.3f}':>20}"
                  f"{f'{f_[chr(99)]:+.3f}+/-{f_[chr(99)+chr(104)+chr(119)]:.3f}':>20}{flag}", flush=True)
        print("  read: b STABLE across thresholds -> censoring is not biasing b. b DRIFTS UPWARD as the", flush=True)
        print("  threshold tightens -> that IS the censoring bias, measured; extrapolate to threshold->0.", flush=True)
        print("  NOTE: tightening preferentially retains FLOPPY/low-rank systems, shrinking the N and", flush=True)
        print("  mobility ranges -- both printed so the loss of leverage is visible.", flush=True)
