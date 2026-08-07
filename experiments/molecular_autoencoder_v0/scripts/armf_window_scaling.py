"""rank90 vs OBSERVATION WINDOW -- does intrinsic dimensionality saturate with simulated time?
This is objective 3 (multi-millisecond) asked as a measurement instead of an assumption.

  SATURATES        -> finite intrinsic dimensionality; a fixed latent budget covers arbitrarily
                      long trajectories, and long-timescale modelling is an ENGINEERING problem.
  GROWS UNBOUNDED  -> capacity must grow with simulated time; multi-millisecond has a STRUCTURAL
                      problem no architecture fixes.

mdCATH, 28 domains, ALL-ATOM, 320K, 5 replicas concatenated (2,500 frames). PCA only, no training.
Per domain: rank90(T) on a dense T grid, then three competing models fitted in LINEAR space and
selected by AIC:
    saturating   rank90 = A*T/(K+T)        -> report asymptote A (finite budget) with CI
    power-law    rank90 = a*T^c            -> report exponent c with CI
    logarithmic  rank90 = a + b*ln(T)      -> unbounded but slowly
Then: do the still-climbing domains differ systematically from the saturating ones (mobility, size,
secondary structure)? Mann-Whitney U on each.

EFFICIENCY: rank90(T) for every window from ONE Gram matrix. For window T the centered Gram is
C_T G[:T,:T] C_T with C_T = I - 11^T/T, whose eigenvalues are the squared singular values of the
centered frame block -- so the 3N dimension is touched once, not once per window. Verified against
a direct SVD in-script."""
import h5py, numpy as np, glob, time, json
from scipy import stats
from scipy.optimize import curve_fit
MDC = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data/*.h5"
OUT = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/window_scaling.json"
TGRID = [50, 79, 120, 200, 320, 500, 800, 1200, 1800, 2400]; np.random.seed(0)


def kabsch_to(traj, ref):
    rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def rank_from_gram(G, T, frac=0.90):                             # rank90 of first T frames, from the Gram
    g = G[:T, :T]; m = g.mean(1, keepdims=True)
    gc = g - m - m.T + g.mean()                                  # double-centering = centering the frames
    w = np.linalg.eigvalsh(gc); w = np.clip(w[::-1], 0, None)
    cum = np.cumsum(w) / (w.sum() + 1e-12)
    return int(np.searchsorted(cum, frac) + 1)


f_sat = lambda T, A, K: A * T / (K + T)
f_pow = lambda T, a, c: a * np.power(T, c)
f_log = lambda T, a, b: a + b * np.log(T)


def aic(y, yh, k): n = len(y); rss = float(((y - yh) ** 2).sum()); return n * np.log(rss / n + 1e-12) + 2 * k, rss


def fit_all(T, R):
    T = np.asarray(T, float); R = np.asarray(R, float); out = {}
    for name, fn, p0, bnds in [
            ("saturating", f_sat, [R.max() * 1.5, T.mean()], ([1e-3, 1e-3], [1e6, 1e9])),
            ("power",      f_pow, [R[0] / T[0] ** 0.5, 0.5],  ([1e-9, -2.0], [1e6, 3.0])),
            ("log",        f_log, [R[0], (R[-1] - R[0]) / np.log(T[-1] / T[0])], ([-1e6, -1e6], [1e6, 1e6]))]:
        try:
            p, cov = curve_fit(fn, T, R, p0=p0, bounds=bnds, maxfev=200000)
            a_, rss = aic(R, fn(T, *p), len(p)); se = np.sqrt(np.diag(cov))
            out[name] = dict(p=[float(x) for x in p], se=[float(x) for x in se], aic=float(a_),
                             r2=float(1 - rss / (((R - R.mean()) ** 2).sum() + 1e-12)))
        except Exception:
            out[name] = dict(p=[float('nan')] * 2, se=[float('nan')] * 2, aic=float('inf'), r2=float('nan'))
    return out


print("[window-scaling] mdCATH 320K x5 replicas (2500 frames), ALL-ATOM. rank90(T), PCA only.", flush=True)
rows = []; t0 = time.time(); checked = False
for fp in sorted(glob.glob(MDC)):
    try:
        ff = h5py.File(fp, "r"); dom = list(ff.keys())[0]; g = ff[dom]
        T0 = sorted([k for k in g.keys() if k.isdigit()], key=int)[0]
        reps = sorted(g[T0].keys(), key=lambda x: int(x))
        ref = np.array(g[T0][reps[0]]["coords"][0]).astype(np.float64)
        ds = np.array(g[T0][reps[0]]["dssp"])[0]                 # secondary structure, frame 0
        ss = [x.decode() if isinstance(x, bytes) else str(x) for x in ds]
        helix = float(np.mean([c in "HGI" for c in ss])); sheet = float(np.mean([c in "EB" for c in ss]))
        chunks = [kabsch_to(np.array(g[T0][r]["coords"]).astype(np.float64), ref) - ref for r in reps]
        ff.close()
        disp = np.concatenate(chunks, 0); N = disp.shape[1]; F = len(disp)
        mag = np.linalg.norm(disp, axis=-1); rmsf = float(np.median(np.sqrt((mag ** 2).mean(0))))
        X = disp.reshape(F, -1); G = X @ X.T                     # ONE Gram; all windows come from it
        tg = [t for t in TGRID if t <= F]; R = [rank_from_gram(G, t) for t in tg]
        if not checked:                                          # correctness check vs direct SVD
            t_ = tg[3]; x = X[:t_] - X[:t_].mean(0); S = np.linalg.svd(x, compute_uv=False)
            cum = np.cumsum(S ** 2) / (S ** 2).sum(); direct = int(np.searchsorted(cum, 0.90) + 1)
            print(f"  [check] Gram-path rank90={R[3]} vs direct-SVD rank90={direct} at T={t_} -> "
                  f"{'MATCH' if direct == R[3] else 'MISMATCH'}", flush=True); checked = True
        fits = fit_all(tg, R); best = min(fits, key=lambda k: fits[k]["aic"])
        rows.append(dict(dom=dom, N=N, rmsf=rmsf, helix=helix, sheet=sheet, T=tg, R=R, fits=fits, best=best))
        extra = (f"A={fits['saturating']['p'][0]:.0f}+/-{1.96*fits['saturating']['se'][0]:.0f}" if best == "saturating"
                 else f"c={fits['power']['p'][1]:.2f}+/-{1.96*fits['power']['se'][1]:.2f}" if best == "power"
                 else f"b={fits['log']['p'][1]:.0f}")
        print(f"  {dom:10s} N={N:>6} RMSF {rmsf:>5.2f} h/e {helix:.2f}/{sheet:.2f}  rank90 {R[0]:>3}->{R[-1]:>3}  "
              f"BEST={best:10s} {extra}", flush=True)
    except Exception as e:
        print(f"  {fp.split('/')[-1]}: skipped ({type(e).__name__}: {e})", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
json.dump(rows, open(OUT, "w"))

print("\n=== CLASSIFICATION (AIC-selected model per domain) ===", flush=True)
for k in ["saturating", "power", "log"]:
    sel = [r for r in rows if r["best"] == k]
    print(f"  {k:11s}: {len(sel):>3}/{len(rows)} domains", flush=True)
sat = [r for r in rows if r["best"] == "saturating"]
if sat:
    A = np.array([r["fits"]["saturating"]["p"][0] for r in sat]); Rl = np.array([r["R"][-1] for r in sat])
    print(f"  saturating asymptote A: median {np.median(A):.0f}  range {A.min():.0f}-{A.max():.0f}   "
          f"(observed rank90 at T=2400 is {np.median(Rl/A)*100:.0f}% of A -- how close to the asymptote we are)", flush=True)
    print(f"  *** A IS A FLOOR (INBOX 002). This median is the ~168 that ANCHORED the width chain, and", flush=True)
    print(f"  it is fitted to IN-SAMPLE rank90, which understates dimensionality -- out of sample the", flush=True)
    print(f"  same modes cover 77.1% of held-out variance, not 90%. The SATURATION-IN-TIME finding", flush=True)
    print(f"  stands (that is what this file measures); what is retired is using A to SIZE DM. ***", flush=True)
pw = [r for r in rows if r["best"] == "power"]
if pw:
    C = np.array([r["fits"]["power"]["p"][1] for r in pw])
    print(f"  power-law exponent c: median {np.median(C):.2f}  range {C.min():.2f}-{C.max():.2f}", flush=True)

print("\n=== DO STILL-CLIMBING DOMAINS DIFFER SYSTEMATICALLY? (Mann-Whitney U) ===", flush=True)
climb = [r for r in rows if r["best"] != "saturating"]; conv = [r for r in rows if r["best"] == "saturating"]
print(f"  climbing n={len(climb)}  vs  saturating n={len(conv)}", flush=True)
if climb and conv:
    for key, lab in [("rmsf", "mobility medRMSF"), ("N", "size N"), ("helix", "helix fraction"), ("sheet", "sheet fraction")]:
        a = [r[key] for r in climb]; b = [r[key] for r in conv]
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {lab:20s} climbing med {np.median(a):>8.2f}  saturating med {np.median(b):>8.2f}   "
              f"p={u.pvalue:.3f} {'*SIGNIFICANT*' if u.pvalue < 0.05 else ''}", flush=True)
print("\n  read: saturating dominant -> finite dimensionality, fixed latent budget covers long", flush=True)
print("  trajectories, objective 3 is engineering. power/log dominant -> capacity must grow with", flush=True)
print("  simulated time and multi-ms is structural. NOTE: 2,500 frames of 320K mdCATH is ns-us, so", flush=True)
print("  this bounds the ns-us regime only; it cannot see ms-scale rare events by construction.", flush=True)
