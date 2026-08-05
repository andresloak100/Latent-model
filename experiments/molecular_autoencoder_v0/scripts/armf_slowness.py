"""Slowness-sensitive dimensionality: closes the variance-weighting gap left by rank90.

rank90 counts directions carrying >=10% of VARIANCE, so a rare brief excursion into a new basin --
exactly the ms-scale event of concern -- contributes little variance and barely moves it. Two
metrics that are sensitive to what PCA discards, on the SAME window and temperature sweeps:

 1. TICA dimensionality: time-lagged independent component analysis ranks directions by SLOWNESS
    (autocorrelation at lag tau) instead of variance. Reported as the number of ICs needed for 90%
    of the KINETIC VARIANCE (sum of squared IC autocorrelations, Noe/Clementi). A slow rare
    transition is a leading TICA component even when it is a negligible PCA component.
 2. STATE COUNT: distinct conformational states by leader clustering at a fixed RMSD cutoff --
    "new states" directly, not a variance proxy.

READ: if TICA dimensionality and state count stay flat under heating the way rank90 did, the ms
concern is genuinely reduced rather than merely unmeasured. If they RISE while rank90 stays flat,
that is the rare-state effect and it quantifies exactly what variance-weighted PCA was blind to.

EFFICIENCY: one Gram per (domain, temperature) serves all three metrics -- PCA basis (eigh of the
centered Gram), TICA (on the PCA-projected coordinates, which is exact since TICA is invariant
under invertible linear maps), and all pairwise RMSDs (||Xi-Xj||^2 = Gii + Gjj - 2Gij).
CENSORING GUARD, learned from rank90: TICA is capped by the number of PCA components retained (m);
if the TICA dimension approaches m the measurement is censored, so m is reported alongside."""
import h5py, numpy as np, glob, json, os, time, warnings
warnings.filterwarnings("ignore")
from scipy import linalg, stats
MDC = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data/*.h5"
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
OUT = f"{WR}/slowness.json"; TEMPJSON = f"{WR}/temp_exploration.json"
WINDOWS = [200, 400, 800, 1600, 2400]; TAU = 10; CUTOFFS = [1.5, 2.0, 3.0]
np.random.seed(0)


def kabsch_to(traj, ref):
    rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def pca_coords(G, W):                                            # PCA-projected coords Y (W x r) from Gram
    g = G[:W, :W]; m = g.mean(1, keepdims=True); gc = g - m - m.T + g.mean()
    w, V = np.linalg.eigh(gc); idx = np.argsort(w)[::-1]; w = np.clip(w[idx], 0, None); V = V[:, idx]
    r = int((w > w[0] * 1e-10).sum()); return V[:, :r] * np.sqrt(w[:r])[None, :], w[:r]


def tica_dim(Y, ev, tau=TAU, frac=0.90):
    W = Y.shape[0]
    cum = np.cumsum(ev) / (ev.sum() + 1e-12)
    m = int(np.searchsorted(cum, 0.95) + 1)                      # 95% variance basis ...
    m = max(2, min(m, W // 10))                                  # ... capped so C0 stays conditioned (>=10 samples/dim)
    Z = Y[:, :m]
    C0 = Z.T @ Z / (W - 1) + 1e-8 * np.eye(m)
    A = Z[:-tau].T @ Z[tau:]; Ct = (A + A.T) / (2 * (W - tau - 1))
    try:
        lam = linalg.eigh(Ct, C0, eigvals_only=True)
    except Exception:
        return float('nan'), m
    lam = np.clip(np.sort(np.abs(lam))[::-1], 0, 1)              # |autocorrelation| per IC, slowest first
    kv = lam ** 2; c = np.cumsum(kv) / (kv.sum() + 1e-12)
    return int(np.searchsorted(c, frac) + 1), m


def state_count(G, W, natoms, cutoffs=CUTOFFS):                  # leader clustering on pairwise RMSD
    g = G[:W, :W]; dd = np.diag(g)
    D2 = np.clip(dd[:, None] + dd[None, :] - 2 * g, 0, None); R = np.sqrt(D2 / natoms)
    out = {}
    for c in cutoffs:
        un = np.ones(W, bool); k = 0
        while un.any():
            i = int(np.argmax(un)); k += 1; un &= ~(R[i] < c); un[i] = False
        out[c] = k
    return out


done = json.load(open(OUT)) if os.path.exists(OUT) else {}
print(f"[slowness] mdCATH 5 temps x 5 replicas; TICA dim + state count + rank90. resuming {len(done)}", flush=True)
t0 = time.time()
for fp in sorted(glob.glob(MDC)):
    ff = h5py.File(fp, "r"); dom = list(ff.keys())[0]
    if dom in done: ff.close(); continue
    try:
        g = ff[dom]; temps = sorted([k for k in g.keys() if k.isdigit()], key=int)
        ref = np.array(g[temps[0]]["0"]["coords"][0]).astype(np.float64); N = len(ref)
        rec = {"N": int(N), "temps": {}}
        for T in temps:
            reps = sorted(g[T].keys(), key=lambda x: int(x))
            X = np.concatenate([kabsch_to(np.array(g[T][r]["coords"]).astype(np.float64), ref) - ref
                                for r in reps], 0).reshape(-1, 3 * N)
            G = X @ X.T; F = len(X); e = {}
            for W in [w for w in WINDOWS if w <= F]:
                Y, ev = pca_coords(G, W)
                td, m = tica_dim(Y, ev)
                cum = np.cumsum(ev) / (ev.sum() + 1e-12); r90 = int(np.searchsorted(cum, 0.90) + 1)
                e[W] = dict(tica=td, tica_basis=m, rank90=r90, states=state_count(G, W, N))
            rec["temps"][T] = e; del X, G
        done[dom] = rec; json.dump(done, open(OUT, "w"))
        w = WINDOWS[-1]
        s = "  ".join(f"{T}K:tica={rec['temps'][T][w]['tica']},st={rec['temps'][T][w]['states'][2.0]}" for T in temps)
        print(f"  {dom:10s} N={N:>6} {s}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as ex:
        print(f"  {dom}: skipped ({type(ex).__name__}: {ex})", flush=True)
    finally:
        ff.close()

# ---------------- analysis ----------------
tj = json.load(open(TEMPJSON)) if os.path.exists(TEMPJSON) else {}
def folded(dm, T):
    r = tj.get(dm)
    if not r or T not in r["temps"] or "320" not in r["temps"]: return False
    b = r["temps"]["320"]; c = r["temps"][T]
    return c["rg"] / b["rg"] <= 1.10 and c["q"] / max(b["q"], 1e-9) >= 0.85
temps = ["320", "348", "379", "413", "450"]; W = WINDOWS[-1]

print("\n=== 1. WINDOW SWEEP at 320K (does slowness-dimensionality saturate like rank90?) ===", flush=True)
print(f"  {'window':>8}{'TICA dim':>11}{'TICAbasis':>11}{'rank90':>9}{'states@2A':>11}{'states@1.5A':>13}", flush=True)
for w in WINDOWS:
    td = [d["temps"]["320"][str(w)]["tica"] if str(w) in d["temps"]["320"] else d["temps"]["320"][w]["tica"]
          for d in done.values() if "320" in d["temps"]]
    tb = [list(d["temps"]["320"].values())[WINDOWS.index(w)]["tica_basis"] for d in done.values() if "320" in d["temps"]]
    r9 = [list(d["temps"]["320"].values())[WINDOWS.index(w)]["rank90"] for d in done.values() if "320" in d["temps"]]
    s2 = [list(d["temps"]["320"].values())[WINDOWS.index(w)]["states"]["2.0"] for d in done.values() if "320" in d["temps"]]
    s15 = [list(d["temps"]["320"].values())[WINDOWS.index(w)]["states"]["1.5"] for d in done.values() if "320" in d["temps"]]
    print(f"  {w:>8}{np.median(td):>11.0f}{np.median(tb):>11.0f}{np.median(r9):>9.0f}{np.median(s2):>11.0f}{np.median(s15):>13.0f}", flush=True)

print("\n=== 2. TEMPERATURE SWEEP, PAIRED within-domain, FOLDED only (vs each domain's own 320K) ===", flush=True)
print(f"  {'T':>5}{'nFolded':>9}{'TICA ratio':>13}{'rank90 ratio':>14}{'states@2A ratio':>17}", flush=True)
R = 0.0019872041; xs = {k: ([], []) for k in ["tica", "rank90", "states"]}
for T in temps[1:]:
    tr_, rr_, sr_, n = [], [], [], 0
    for dm, d in done.items():
        if T not in d["temps"] or "320" not in d["temps"] or not folded(dm, T): continue
        a = list(d["temps"]["320"].values())[-1]; b = list(d["temps"][T].values())[-1]
        lt = np.log10(np.exp(10.0 / R * (1 / 320.0 - 1 / float(T))))
        tr_.append(b["tica"] / max(a["tica"], 1)); rr_.append(b["rank90"] / max(a["rank90"], 1))
        sr_.append(b["states"]["2.0"] / max(a["states"]["2.0"], 1)); n += 1
        xs["tica"][0].append(lt); xs["tica"][1].append(np.log10(tr_[-1]))
        xs["rank90"][0].append(lt); xs["rank90"][1].append(np.log10(rr_[-1]))
        xs["states"][0].append(lt); xs["states"][1].append(np.log10(sr_[-1]))
    if n: print(f"  {T:>5}{n:>9}{np.median(tr_):>13.3f}{np.median(rr_):>14.3f}{np.median(sr_):>17.3f}", flush=True)
print("\n  paired slope vs log10(effective time), Ea=10:", flush=True)
for k in ["tica", "rank90", "states"]:
    x, y = xs[k]
    if len(x) < 4: continue
    lr = stats.linregress(x, y); ci = stats.t.ppf(0.975, len(x) - 2) * lr.stderr
    v = "RISES" if lr.slope - ci > 0 else ("FALLS" if lr.slope + ci < 0 else "FLAT")
    ms = np.log10(1000 / 2.5)
    print(f"    {k:>7}: slope {lr.slope:+.3f} +/- {ci:.3f}  R^2 {lr.rvalue**2:.3f}  -> {v}   "
          f"[at 1ms effective: {10**((lr.slope-ci)*ms):.2f}x - {10**((lr.slope+ci)*ms):.2f}x]", flush=True)
print("\n  read: TICA + states FLAT like rank90 -> ms concern genuinely reduced, not just unmeasured.", flush=True)
print("  TICA/states RISE while rank90 flat -> that IS the rare-state effect PCA was blind to.", flush=True)
