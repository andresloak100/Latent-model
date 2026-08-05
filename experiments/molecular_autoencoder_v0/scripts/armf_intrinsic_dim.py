"""Intrinsic dimensionality vs N -- the physics measurement behind objective 1, and its censoring.

rank90 (PCA components for 90% of train-frame displacement variance) rose 29 -> 57 as N rose
959 -> 23,895 on MISATO: sublinear growth, ~N^0.21, i.e. 2x dimensionality for 25x atoms.
Extrapolated to 1e6 atoms that is rank90 ~125 -- a latent budget in the low hundreds covering a
million atoms. But rank90=57 against a ~79-frame cap is 72% of available rank, so the MISATO
exponent is CENSORED and is a LOWER BOUND.

mdCATH has 500 frames/replica -> 400 train frames, so rank90 can run to ~400 uncensored, over
N = 711-7,524 all-atom (28 local domains, no download needed). Decisive design: measure rank90 on
the SAME domains at 79 frames (censored, MISATO-matched) and at 400 frames (uncensored). The gap
between those two exponents IS the censoring correction.

Protocol matched to the MISATO run: ALL-ATOM, Kabsch-aligned displacement from frame 0, PCA on
train frames. mdCATH 320K (closest to MISATO's ~300K), replica 0. Reports exponent with 95% CI."""
import h5py, numpy as np, glob, time
from scipy import stats
MDC = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data/*.h5"
NFR = [79, 200, 400]                                             # frame budgets: MISATO-matched, mid, uncensored
np.random.seed(0)


def kabsch_traj(traj):
    ref = traj[0]; rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def rank_at(disp, nf, frac=0.90):                                # PCA components for `frac` of variance, nf frames
    d = disp[:nf].reshape(nf, -1); mu = d.mean(0)
    S = np.linalg.svd(d - mu, compute_uv=False); cum = np.cumsum(S**2) / ((S**2).sum() + 1e-12)
    return int(np.searchsorted(cum, frac) + 1)


def fit_exponent(N, R):                                          # log-log slope + 95% CI
    x = np.log10(np.asarray(N, float)); y = np.log10(np.asarray(R, float))
    r = stats.linregress(x, y); tcrit = stats.t.ppf(0.975, len(x)-2)
    return r.slope, tcrit*r.stderr, r.rvalue**2


print("[intrinsic-dim] mdCATH 320K rep0, ALL-ATOM, Kabsch displacement. rank90 vs frame budget.", flush=True)
fs = sorted(glob.glob(MDC)); rows = []; t0 = time.time()
for fp in fs:
    try:
        ff = h5py.File(fp, "r"); dom = list(ff.keys())[0]; g = ff[dom]
        temps = sorted([k for k in g.keys() if k.isdigit()], key=int)
        co = np.array(g[temps[0]]["0"]["coords"]).astype(np.float64)   # 320K, replica 0, (500,N,3)
        N = co.shape[1]; ff.close()
        al = kabsch_traj(co); disp = al - al[0]
        rr = {nf: rank_at(disp, nf) for nf in NFR}
        mag = np.linalg.norm(disp, axis=-1); rmsf = float(np.median(np.sqrt((mag**2).mean(0))))
        st = np.linalg.norm(np.diff(al, axis=0), axis=-1)              # (T-1,N) consecutive steps
        # N-ROBUST wrap test: a PBC wrap is a DETACHED population at ~box scale, not a large max.
        # max-over-atoms grows with N by extreme-value statistics alone and would drop big systems.
        p999 = np.percentile(st, 99.9); frac_out = float((st.max(0) > 3*p999).mean())
        cons = float(st.max())
        rows.append(dict(dom=dom, N=N, rmsf=rmsf, cons=cons, frac_out=frac_out, p999=float(p999),
                         **{f"r{nf}": rr[nf] for nf in NFR}))
        print(f"  {dom:10s} N={N:>6}  rank90 @79={rr[79]:>3} @200={rr[200]:>3} @400={rr[400]:>3}   "
              f"medRMSF {rmsf:.2f}  step_p99.9 {p999:.1f}A max {cons:.1f}A  detached {frac_out*100:.2f}%", flush=True)
    except Exception as e:
        print(f"  {fp.split('/')[-1]}: skipped ({type(e).__name__})", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

# PBC screen on mdCATH (same test as the MISATO screen)
clean = [r for r in rows if r["frac_out"] <= 0.001]        # detached-population test, not a max threshold
print(f"\n  PBC screen (N-robust): {len(rows)-len(clean)}/{len(rows)} domains show a DETACHED step population "
      f"(>0.1% of atoms above 3x the system's own p99.9); using {len(clean)} clean domains for the fit", flush=True)

print("\n=== INTRINSIC-DIMENSIONALITY EXPONENT  rank90 ~ N^b  (95% CI) ===", flush=True)
N = [r["N"] for r in clean]
for nf in NFR:
    R = [r[f"r{nf}"] for r in clean]
    b, ci, r2 = fit_exponent(N, R)
    cap = max(R)/ (nf-1) * 100
    tag = "CENSORED (MISATO-matched)" if nf == 79 else ("UNCENSORED" if nf == 400 else "mid")
    print(f"  {nf:>3} frames [{tag:26s}]: b = {b:.3f} +/- {ci:.3f}   R^2 {r2:.3f}   "
          f"rank90 range {min(R)}-{max(R)}  (max is {cap:.0f}% of available rank)", flush=True)

print("\n=== 1M-ATOM EXTRAPOLATION (median N anchor) ===", flush=True)
for nf in NFR:
    R = [r[f"r{nf}"] for r in clean]; b, ci, _ = fit_exponent(N, R)
    a = np.median(np.log10(R) - b*np.log10(N))
    lo = 10**(a + (b-ci)*6); hi = 10**(a + (b+ci)*6); mid = 10**(a + b*6)
    print(f"  {nf:>3} frames: rank90(1e6 atoms) ~ {mid:.0f}   [95% CI {lo:.0f} - {hi:.0f}]", flush=True)
print("\n  read: if the UNCENSORED (400-frame) exponent ~= the censored one, the MISATO ~N^0.21 stands and", flush=True)
print("  the 1M-atom latent budget in the low hundreds is on real ground. If it is STEEPER once the cap is", flush=True)
print("  lifted, the MISATO number is an artifact of the 79-frame cap and the budget must be re-planned.", flush=True)
