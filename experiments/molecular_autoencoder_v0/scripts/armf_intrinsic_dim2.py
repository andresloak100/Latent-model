"""Intrinsic dimensionality, take 2: CONVERGENCE + MOBILITY CONTROL.

Take 1 (armf_intrinsic_dim.py) found: (a) censoring is severe -- rank90 doubles/triples from 79 to
400 frames; (b) the N-exponent on mdCATH is uninformative (R^2 ~0.1) because rank90 is driven
INVERSELY by mobility (floppy domain -> one dominant motion -> LOW rank; stable domain -> diffuse
thermal motion -> HIGH rank), and mdCATH spans medRMSF 1.25-11.4 A; (c) rank90 is still climbing at
400 frames, so it is measuring frame budget, not dimensionality.

Two fixes, both available without any download:
 1. MORE FRAMES: concatenate all 5 replicas at 320K -> 2500 frames/domain. Sweep rank90 over
    nf = 79..2400 and test CONVERGENCE per domain (is the curve flattening?). Concatenating
    replicas samples the accessible conformational ENSEMBLE, which is the right object for a latent
    budget anyway.
 2. MOBILITY CONTROL: multiple regression log10(rank90) ~ b*log10(N) + c*log10(medRMSF), so the
    N-exponent is estimated at fixed mobility. Report partial b with 95% CI.

Only converged domains (rank90 within 10% between the top two frame budgets) enter the exponent fit;
unconverged ones are reported separately, not silently pooled."""
import h5py, numpy as np, glob, time
from scipy import stats
MDC = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data/*.h5"
NFR = [79, 200, 400, 800, 1600, 2400]; np.random.seed(0)


def kabsch_to(traj, ref):
    rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def rank_at(d, nf, frac=0.90):
    x = d[:nf]; mu = x.mean(0); S = np.linalg.svd(x - mu, compute_uv=False)
    cum = np.cumsum(S**2) / ((S**2).sum() + 1e-12); return int(np.searchsorted(cum, frac) + 1)


print("[intrinsic-dim-2] mdCATH 320K, ALL 5 replicas concatenated (2500 frames), ALL-ATOM.", flush=True)
print(f"  {'domain':10s}{'N':>7}{'medRMSF':>9}  " + "".join(f"{('r@'+str(n)):>7}" for n in NFR) + "   converged?", flush=True)
rows = []; t0 = time.time()
for fp in sorted(glob.glob(MDC)):
    try:
        ff = h5py.File(fp, "r"); dom = list(ff.keys())[0]; g = ff[dom]
        T0 = sorted([k for k in g.keys() if k.isdigit()], key=int)[0]
        reps = sorted(g[T0].keys(), key=lambda x: int(x))
        ref = np.array(g[T0][reps[0]]["coords"][0]).astype(np.float64)
        chunks = []
        for r in reps:
            co = np.array(g[T0][r]["coords"]).astype(np.float64)
            chunks.append(kabsch_to(co, ref) - ref)
        ff.close()
        disp = np.concatenate(chunks, 0); N = disp.shape[1]; F = len(disp)
        mag = np.linalg.norm(disp, axis=-1); rmsf = float(np.median(np.sqrt((mag**2).mean(0))))
        d2 = disp.reshape(F, -1)
        rr = {nf: rank_at(d2, min(nf, F)) for nf in NFR}
        conv = abs(rr[NFR[-1]] - rr[NFR[-2]]) / max(rr[NFR[-1]], 1) < 0.10
        rows.append(dict(dom=dom, N=N, rmsf=rmsf, conv=conv, F=F, **{f"r{nf}": rr[nf] for nf in NFR}))
        print(f"  {dom:10s}{N:>7}{rmsf:>9.2f}  " + "".join(f"{rr[nf]:>7}" for nf in NFR) +
              f"   {'YES' if conv else 'no (still climbing)'}", flush=True)
    except Exception as e:
        print(f"  {fp.split('/')[-1]}: skipped ({type(e).__name__}: {e})", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

R = np.array([r[f"r{NFR[-1]}"] for r in rows], float); Nv = np.array([r["N"] for r in rows], float)
Mv = np.array([r["rmsf"] for r in rows], float); cv = np.array([r["conv"] for r in rows])
print(f"\n=== CONVERGENCE: {cv.sum()}/{len(rows)} domains converged by {NFR[-1]} frames "
      f"(rank90 within 10% of the {NFR[-2]}-frame value) ===", flush=True)
print(f"  rank90 @{NFR[-1]}: min {int(R.min())} median {int(np.median(R))} max {int(R.max())} "
      f"({R.max()/(NFR[-1]-1)*100:.0f}% of available rank at the top)", flush=True)

print("\n=== EXPONENT, MOBILITY-CONTROLLED: log10(rank90) ~ b*log10(N) + c*log10(medRMSF) ===", flush=True)
for tag, msk in [("ALL domains", np.ones(len(rows), bool)), ("CONVERGED only", cv)]:
    if msk.sum() < 4: print(f"  {tag}: n={msk.sum()}, too few to fit", flush=True); continue
    X = np.column_stack([np.log10(Nv[msk]), np.log10(Mv[msk]), np.ones(msk.sum())])
    y = np.log10(R[msk]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; dof = msk.sum() - 3
    s2 = (resid**2).sum() / dof; cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
    t = stats.t.ppf(0.975, dof); r2 = 1 - (resid**2).sum()/((y-y.mean())**2).sum()
    print(f"  {tag:16s} n={msk.sum():>3}  b(N) = {beta[0]:+.3f} +/- {t*se[0]:.3f}   "
          f"c(RMSF) = {beta[1]:+.3f} +/- {t*se[1]:.3f}   R^2 {r2:.3f}", flush=True)
    # naive (uncontrolled) for contrast
    rr_ = stats.linregress(np.log10(Nv[msk]), y)
    print(f"  {'':16s}      naive b(N) without mobility control = {rr_.slope:+.3f}  (R^2 {rr_.rvalue**2:.3f})", flush=True)
    if dof > 0 and abs(beta[0]) > t*se[0]:
        a = np.median(y - beta[0]*np.log10(Nv[msk]) - beta[1]*np.log10(Mv[msk]))
        m0 = np.log10(np.median(Mv[msk]))
        print(f"  {'':16s}      -> rank90(1e6 atoms, median mobility) ~ {10**(a+beta[0]*6+beta[1]*m0):.0f}"
              f"  [95% CI {10**(a+(beta[0]-t*se[0])*6+beta[1]*m0):.0f} - {10**(a+(beta[0]+t*se[0])*6+beta[1]*m0):.0f}]"
              f"  -- A FLOOR, NOT A REQUIREMENT (INBOX 002)", flush=True)
        print(f"  {'':16s}         in-sample rank90 understates dimensionality and b is censored at", flush=True)
        print(f"  {'':16s}         large N, so both inputs bias this DOWN. It is not a DM budget.", flush=True)
    else:
        print(f"  {'':16s}      -> b(N) NOT significantly different from 0; no extrapolation is licensed", flush=True)
print("\n  read: converged + significant b -> a latent budget can be planned. Unconverged or b~0 ->", flush=True)
print("  rank90 is measuring frame budget / mobility, not dimensionality, and the 1M claim is unsupported.", flush=True)
