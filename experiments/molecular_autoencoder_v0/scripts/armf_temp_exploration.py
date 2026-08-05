"""Does EXPLORATION add dimensions? rank90 asymptote A vs temperature, with a folding control.

The 320K window result (21/28 saturating) may be saturation WITHIN THE SAMPLED BASIN: 2,500 frames
x ~1 ns = ~2.5 us cannot contain ms-scale rare barrier crossings. mdCATH ships 5 temperatures
(320/348/379/413/450K) which accelerate barrier crossing, so A(T) as a CURVE tests whether exploring
more landscape raises the asymptote -- and, via Arrhenius effective time, supports extrapolation.

CONFOUND, controlled mandatorily: at 450K many domains UNFOLD, and unfolding inflates rank90 for
reasons unrelated to barrier crossing. Per domain per temperature we report CA-RMSD from the common
reference, radius of gyration, and fraction of native contacts retained (Q). A domain whose Rg or Q
departs materially from its own 320K value is DENATURED and is analysed separately, never pooled.

PRE-REGISTERED READ:
  A(T) flat, folded throughout          -> saturation robust to exploration; ms concern drops.
  A(T) rises with effective rate, folded-> exploration genuinely adds dimensions; ms concern real,
                                           and the A(T) trend gives a first estimate of A at ms.
  A rises only where unfolding markers move -> denaturation artifact; drop and read the rest.
  If heating makes the 21 SATURATING domains climb, "saturating" was an artifact of insufficient
  exploration at 320K -- that is the headline.

All temperatures share ONE reference (320K replica 0, frame 0) so displacements, RMSD, Rg and Q are
comparable across T. rank90(T_window) from one Gram per (domain, temperature). Resumable: results
are checkpointed per domain, so preemption on `long` does not lose completed work."""
import h5py, numpy as np, glob, time, json, os
from scipy import stats
from scipy.optimize import curve_fit
MDC = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data/*.h5"
OUT = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/temp_exploration.json"
TGRID = [50, 79, 120, 200, 320, 500, 800, 1200, 1800, 2400]
RGAS = 0.0019872041                                              # kcal/mol/K
EAS = [5.0, 10.0, 15.0]; TREF = 320.0; TSIM_US = 2.5             # 2500 frames x ~1 ns
np.random.seed(0)


def kabsch_to(traj, ref):
    rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def rank_from_gram(G, T, frac=0.90):
    g = G[:T, :T]; m = g.mean(1, keepdims=True)
    gc = g - m - m.T + g.mean(); w = np.clip(np.linalg.eigvalsh(gc)[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(w) / (w.sum() + 1e-12), frac) + 1)


f_sat = lambda T, A, K: A * T / (K + T)
f_pow = lambda T, a, c: a * np.power(T, c)
f_log = lambda T, a, b: a + b * np.log(T)


def fit_all(T, R):
    T = np.asarray(T, float); R = np.asarray(R, float); out = {}
    for nm, fn, p0, bd in [("saturating", f_sat, [R.max()*1.5, T.mean()], ([1e-3, 1e-3], [1e7, 1e9])),
                           ("power", f_pow, [R[0]/T[0]**0.5, 0.5], ([1e-9, -2.0], [1e6, 3.0])),
                           ("log", f_log, [R[0], (R[-1]-R[0])/np.log(T[-1]/T[0])], ([-1e6, -1e6], [1e6, 1e6]))]:
        try:
            p, cov = curve_fit(fn, T, R, p0=p0, bounds=bd, maxfev=200000)
            rss = float(((R - fn(T, *p))**2).sum()); a_ = len(R)*np.log(rss/len(R) + 1e-12) + 2*len(p)
            out[nm] = dict(p=[float(x) for x in p], se=[float(x) for x in np.sqrt(np.diag(cov))], aic=float(a_))
        except Exception:
            out[nm] = dict(p=[float('nan')]*2, se=[float('nan')]*2, aic=float('inf'))
    return out


def fold_metrics(ca, ref_ca, pairs, dref):
    al = kabsch_to(ca, ref_ca)
    rmsd = float(np.sqrt(((al - ref_ca)**2).sum(-1).mean(-1)).mean())
    c = ca - ca.mean(1, keepdims=True); rg = float(np.sqrt((c**2).sum(-1).mean(-1)).mean())
    d = np.linalg.norm(ca[:, pairs[:, 0], :] - ca[:, pairs[:, 1], :], axis=-1)
    q = float((d < 1.2 * dref[None, :]).mean())
    return rmsd, rg, q


done = json.load(open(OUT)) if os.path.exists(OUT) else {}
print(f"[temp-exploration] mdCATH 5 temperatures x 5 replicas, ALL-ATOM. resuming with {len(done)} done.", flush=True)
t0 = time.time()
for fp in sorted(glob.glob(MDC)):
    ff = h5py.File(fp, "r"); dom = list(ff.keys())[0]
    if dom in done: ff.close(); continue
    try:
        g = ff[dom]; temps = sorted([k for k in g.keys() if k.isdigit()], key=int)
        pdb = g["pdb"][()]; pdb = pdb.decode() if isinstance(pdb, bytes) else str(pdb)
        names = [ln[12:16].strip() for ln in pdb.splitlines() if ln.startswith(("ATOM", "HETATM"))]
        ref_all = np.array(g[temps[0]]["0"]["coords"][0]).astype(np.float64); N = len(ref_all)
        names = names[:N]; ca_idx = np.array([i for i, nm in enumerate(names) if nm == "CA"])
        if len(ca_idx) < 10: raise RuntimeError(f"CA parse failed ({len(ca_idx)} found, N={N})")
        ref_ca = ref_all[ca_idx]
        D = np.linalg.norm(ref_ca[:, None] - ref_ca[None], axis=-1)
        ii, jj = np.where((D < 8.0) & (np.abs(np.arange(len(ref_ca))[:, None] - np.arange(len(ref_ca))[None]) > 3))
        k = ii < jj; pairs = np.stack([ii[k], jj[k]], 1); dref = D[pairs[:, 0], pairs[:, 1]]
        rec = dict(N=int(N), nca=int(len(ca_idx)), temps={})
        for T in temps:
            reps = sorted(g[T].keys(), key=lambda x: int(x))
            chunks = [kabsch_to(np.array(g[T][r]["coords"]).astype(np.float64), ref_all) - ref_all for r in reps]
            disp = np.concatenate(chunks, 0); F = len(disp)
            ca = np.concatenate([np.array(g[T][r]["coords"]).astype(np.float64)[:, ca_idx, :] for r in reps], 0)
            rmsd, rg, q = fold_metrics(ca, ref_ca, pairs, dref)
            mag = np.linalg.norm(disp, axis=-1); rmsf = float(np.median(np.sqrt((mag**2).mean(0))))
            X = disp.reshape(F, -1); G = X @ X.T
            tg = [t for t in TGRID if t <= F]; R = [rank_from_gram(G, t) for t in tg]
            fits = fit_all(tg, R); best = min(fits, key=lambda z: fits[z]["aic"])
            rec["temps"][T] = dict(R=R, T=tg, fits=fits, best=best, rmsd=rmsd, rg=rg, q=q, rmsf=rmsf)
            del disp, X, G, chunks, ca
        done[dom] = rec; json.dump(done, open(OUT, "w"))
        s = "  ".join(f"{T}K:A={rec['temps'][T]['fits']['saturating']['p'][0]:.0f},Q={rec['temps'][T]['q']:.2f}" for T in temps)
        print(f"  {dom:10s} N={N:>6} nCA={len(ca_idx):>4}  {s}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  {dom}: skipped ({type(e).__name__}: {e})", flush=True)
    finally:
        ff.close()

# ---------------- analysis ----------------
rows = done; temps = sorted({t for r in rows.values() for t in r["temps"]}, key=int)
print(f"\n=== EFFECTIVE TIME (Arrhenius, t_eff = {TSIM_US} us x k(T)/k({TREF:.0f}K)) ===", flush=True)
print(f"  {'T(K)':>6}" + "".join(f"{('Ea='+str(e)):>16}" for e in EAS), flush=True)
for T in temps:
    Tf = float(T); cells = []
    for Ea in EAS:
        f = np.exp(Ea/RGAS * (1/TREF - 1/Tf)); te = TSIM_US*f
        cells.append(f"{f:>7.0f}x = {te*1e-3:>5.2f}ms" if te >= 1000 else f"{f:>7.0f}x = {te:>5.0f}us")
    print(f"  {T:>6}" + "".join(f"{c:>16}" for c in cells), flush=True)
print("  (assumed barrier heights stated explicitly; this is what licenses any ms-scale claim)", flush=True)

print("\n=== FOLDING CONTROL (vs each domain's own 320K value) ===", flush=True)
for T in temps:
    rg_r, q_r, den = [], [], 0
    for d, r in rows.items():
        if T not in r["temps"] or temps[0] not in r["temps"]: continue
        b = r["temps"][temps[0]]; c = r["temps"][T]
        rr = c["rg"]/b["rg"]; qq = c["q"]/max(b["q"], 1e-9); rg_r.append(rr); q_r.append(qq)
        if rr > 1.10 or qq < 0.85: den += 1
    if rg_r:
        print(f"  {T}K: Rg/Rg320 median {np.median(rg_r):.3f}  Q/Q320 median {np.median(q_r):.3f}  "
              f"DENATURED {den}/{len(rg_r)} (Rg>1.10x or Q<0.85x)", flush=True)

print("\n=== A(T): does exploration add dimensions? (FOLDED domains only, denatured shown separately) ===", flush=True)
base = temps[0]
def folded(r, T):
    if T not in r["temps"] or base not in r["temps"]: return False
    b = r["temps"][base]; c = r["temps"][T]
    return (c["rg"]/b["rg"] <= 1.10) and (c["q"]/max(b["q"], 1e-9) >= 0.85)
climb0 = {d for d, r in rows.items() if base in r["temps"] and r["temps"][base]["best"] != "saturating"}
for grp, sel in [("ALL", set(rows)), ("saturating@320K", set(rows)-climb0), ("climbing@320K", climb0)]:
    print(f"  -- {grp} (n={len(sel)}) --", flush=True)
    for T in temps:
        A_f = [rows[d]["temps"][T]["fits"]["saturating"]["p"][0] for d in sel if folded(rows[d], T)]
        A_d = [rows[d]["temps"][T]["fits"]["saturating"]["p"][0] for d in sel if T in rows[d]["temps"] and not folded(rows[d], T)]
        r90 = [rows[d]["temps"][T]["R"][-1] for d in sel if folded(rows[d], T)]
        nsat = sum(1 for d in sel if folded(rows[d], T) and rows[d]["temps"][T]["best"] == "saturating")
        print(f"    {T}K  folded n={len(A_f):>3} A_med {np.median(A_f) if A_f else float('nan'):>7.0f}  "
              f"rank90@2400 med {np.median(r90) if r90 else float('nan'):>6.0f}  saturating {nsat}/{len(A_f)}   "
              f"| denatured n={len(A_d):>3} A_med {np.median(A_d) if A_d else float('nan'):>7.0f}", flush=True)
    # trend of A vs effective log-rate, folded only
    xs, ys = [], []
    for d in sel:
        for T in temps:
            if folded(rows[d], T):
                xs.append(np.log10(np.exp(10.0/RGAS*(1/TREF - 1/float(T))))); ys.append(np.log10(max(rows[d]["temps"][T]["fits"]["saturating"]["p"][0], 1e-6)))
    if len(xs) > 4:
        lr = stats.linregress(xs, ys); ci = stats.t.ppf(0.975, len(xs)-2)*lr.stderr
        print(f"    A vs log10(effective time), Ea=10: slope {lr.slope:+.3f} +/- {ci:.3f}  R^2 {lr.rvalue**2:.3f}  "
              f"{'RISES' if lr.slope-ci > 0 else ('FLAT' if abs(lr.slope) < ci else 'unclear')}", flush=True)
        if lr.slope - ci > 0:
            a = np.median(np.array(ys) - lr.slope*np.array(xs))
            xm = np.log10(1000.0/TSIM_US)     # effective time to reach 1 ms
            print(f"    -> extrapolated A at 1 ms effective time ~ {10**(a+lr.slope*xm):.0f}", flush=True)
print("\n  read: A flat + folded -> saturation robust, ms concern drops. A rises with effective time +", flush=True)
print("  folded -> exploration adds dimensions, ms concern REAL. A rises only where Rg/Q move ->", flush=True)
print("  denaturation artifact. If saturating@320K domains start climbing when heated, '21/28", flush=True)
print("  saturating' was an artifact of insufficient exploration and THAT is the headline.", flush=True)
