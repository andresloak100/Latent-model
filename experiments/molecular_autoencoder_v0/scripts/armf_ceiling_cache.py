"""CPU-only ceiling precompute + cache. Ceilings do not change between runs, so computing them
inside a GPU job wastes GPU-node wall-clock (~104 min = 90s x 69 systems on the last run) on a
borrowed account. Same pattern as the 2.6 GB data cache: build once on CPU, load instantly after.

Three ceilings per system, keyed by DOMAIN NAME so any train/held-out split can reuse them:

 1. PCA-k  -- unconstrained per-system ceiling. RANK-VALID only while k < 30% of usable rank
              (79 train frames -> k <= 23); above that it is a frame artifact and is flagged, not
              silently reported.
 2. ANM-k  -- all-atom, structure-predicted, zero-parameter. Dense eigh gives ALL modes at once so
              every k is free. Affordable to N ~ 6000 (~90-285 s); ABSENT above that, which is
              exactly where the interesting buckets live (medN 9,270 and 22,755).
 3. CG-ANM-k -- THE FIX for (2). Coarse-grain to residue centroids (sequential runs of
              atoms_residue), build the Hessian on ~n_res beads, eigendecompose densely (n_res~1500
              -> 4500^2, seconds instead of impossible), then expand modes back to all atoms
              piecewise-constant per residue and ORTHONORMALISE (QR) in all-atom space so the
              projection FVE is meaningful.
              CG-ANM IS A BASELINE, NOT A CEILING. It is approximate, and it must NEVER be compared
              against the all-atom ANM numbers from the lower buckets -- report it under its own
              label and compare CG-ANM to CG-ANM only. Both are computed where they overlap
              (N <= 6000) precisely so the approximation gap is visible rather than assumed."""
import h5py, numpy as np, pickle, json, os, time, warnings
from scipy import stats
warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
DATA = f"{WR}/phase1_data_v2.pkl"; OUT = f"{WR}/ceiling_cache.json"
KS = [1, 12, 16, 24, 64, 128, 256, 512]; ANM_MAXN = 6000; RANKVOID = 0.30 * 79
np.random.seed(0)


def hessian(xyz, cutoff):
    n = len(xyz); tree = cKDTree(xyz); pr = tree.query_pairs(cutoff, output_type='ndarray')
    if not len(pr): return None
    v = xyz[pr[:, 1]] - xyz[pr[:, 0]]; u = v / (np.linalg.norm(v, axis=1)[:, None] + 1e-9)
    rows, cols, vals = [], [], []
    for (i, j), uu in zip(pr, u):
        b = np.outer(uu, uu)
        for A, B, s in [(i, j, -1), (j, i, -1), (i, i, 1), (j, j, 1)]:
            for a in range(3):
                for c in range(3): rows.append(3*A+a); cols.append(3*B+c); vals.append(s*b[a, c])
    return coo_matrix((vals, (rows, cols)), shape=(3*n, 3*n)).tocsr()


def modes_dense(H, kmax):
    w, V = np.linalg.eigh(H.toarray()); V = V[:, np.argsort(w)][:, 6:6+kmax]; return V


def fve_basis(ho, M, sst):                                       # M assumed orthonormal (3N x k)
    return 1 - float(((ho - ho @ M @ M.T) ** 2).sum()) / (sst + 1e-12)


f = h5py.File(MD, "r")
data = pickle.load(open(DATA, "rb"))
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
# held-out first (that is what any run needs), then the rest if time allows
order = [d for i, d in enumerate(data) if i % 4 == 0] + [d for i, d in enumerate(data) if i % 4 != 0]
print(f"[ceiling-cache] {len(data)} systems ({sum(1 for i in range(len(data)) if i%4==0)} held-out first); "
      f"resuming with {len(done)} cached", flush=True)
t0 = time.time()
for d in order:
    dom = d["dom"]
    if dom in done: continue
    try:
        N = d["N"]; h = d["h"]; X = d["dc"].astype(np.float64).reshape(d["T"], -1)
        tr, ho = X[:h], X[h:]; sst = float((ho ** 2).sum())
        rec = {"N": int(N), "bucket": list(d["bucket"]), "pca": {}, "pca_valid": {}, "anm": {}, "cganm": {}}
        _, _, Vt = np.linalg.svd(tr, full_matrices=False)
        for k in KS:
            rec["pca_valid"][str(k)] = bool(k <= RANKVOID)
            rec["pca"][str(k)] = (fve_basis(ho, Vt[:k].T, sst) if k <= min(len(Vt), h-1) else float('nan'))
        ref = d["ref"].astype(np.float64)
        # (2) all-atom ANM where affordable
        if N <= ANM_MAXN:
            H = hessian(ref, 10.0)
            if H is not None:
                M = modes_dense(H, max(KS))
                for k in KS:
                    rec["anm"][str(k)] = fve_basis(ho, M[:, :k], sst) if M.shape[1] >= k else float('nan')
        for k in KS: rec["anm"].setdefault(str(k), float('nan'))
        # (3) CG-ANM: residue beads -> modes -> expand piecewise-constant -> orthonormalise
        g = f[dom]; res = np.array(g["atoms_residue"])[:N]
        rid = np.concatenate([[0], np.cumsum(res[1:] != res[:-1])]); nb = int(rid.max()) + 1
        if nb >= 12:
            cen = np.stack([ref[rid == r].mean(0) for r in range(nb)])
            Hc = hessian(cen, 12.0)
            if Hc is not None:
                kmax = min(max(KS), 3*nb - 6)
                Mc = modes_dense(Hc, kmax)                        # (3nb, kmax)
                E = np.zeros((3*N, Mc.shape[1]))                  # expand: atom inherits its residue's mode
                for r in range(nb):
                    idx = np.where(rid == r)[0]
                    for a in range(3): E[3*idx + a, :] = Mc[3*r + a, :]
                Q, _ = np.linalg.qr(E)                            # orthonormalise IN ALL-ATOM SPACE
                for k in KS:
                    rec["cganm"][str(k)] = fve_basis(ho, Q[:, :k], sst) if Q.shape[1] >= k else float('nan')
                rec["nres"] = int(nb)
        for k in KS: rec["cganm"].setdefault(str(k), float('nan'))
        done[dom] = rec; json.dump(done, open(OUT, "w"))
        print(f"  {dom:8s} N={N:>6} nres={rec.get('nres','-'):>5}  "
              f"PCA24 {rec['pca']['24']:.3f}  ANM64 {rec['anm']['64']:.3f}  CG-ANM64 {rec['cganm']['64']:.3f}  "
              f"({time.time()-t0:.0f}s, {len(done)} done)", flush=True)
    except Exception as e:
        print(f"  {dom}: skipped ({type(e).__name__}: {e})", flush=True)

# agreement check where BOTH exist -- makes the CG approximation gap visible instead of assumed
both = [(v["anm"][str(k)], v["cganm"][str(k)], k) for v in done.values() for k in KS
        if np.isfinite(v["anm"].get(str(k), float('nan'))) and np.isfinite(v["cganm"].get(str(k), float('nan')))]
print(f"\n=== CG-ANM vs all-atom ANM where both exist (N<={ANM_MAXN}) -- the approximation gap ===", flush=True)
for k in KS:
    p = [(a, c) for a, c, kk in both if kk == k]
    if not p: continue
    a = np.array([x[0] for x in p]); c = np.array([x[1] for x in p])
    print(f"  k={k:>4}: ANM {np.median(a):.3f}  CG-ANM {np.median(c):.3f}  median gap {np.median(a-c):+.3f}  (n={len(p)})", flush=True)

# GAP AS A FUNCTION OF N -- the validation must span the whole overlap, not one size point. CG-ANM is
# applied up to N=22,755, a ~9x extrapolation beyond where it was first checked (N~2,600). If the gap
# GROWS with N the baseline is itself N-dependent and would contaminate the high-N comparison it exists
# to support -- which would be a confound, not a detail.
print("\n=== CG-ANM approximation gap vs N (k=64) ===", flush=True)
BINS = [(700,1500),(1500,2500),(2500,4000),(4000,6000)]
xs, ys = [], []
for lo, hi in BINS:
    g = [(v["anm"]["64"] - v["cganm"]["64"], v["N"]) for v in done.values()
         if lo <= v["N"] < hi and np.isfinite(v["anm"].get("64", float("nan"))) and np.isfinite(v["cganm"].get("64", float("nan")))]
    if not g: continue
    gap = np.array([x[0] for x in g]); nn_ = np.array([x[1] for x in g])
    xs += list(np.log10(nn_)); ys += list(gap)
    print(f"  N in [{lo},{hi}): n={len(g):>3}  medN={int(np.median(nn_)):>6}  median gap {np.median(gap):+.4f}  "
          f"IQR [{np.percentile(gap,25):+.4f},{np.percentile(gap,75):+.4f}]", flush=True)
if len(xs) > 4:
    lr = stats.linregress(xs, ys); ci = stats.t.ppf(0.975, len(xs)-2)*lr.stderr
    verdict = "GROWS WITH N -- CG-ANM baseline is N-dependent, CONFOUND" if lr.slope-ci > 0 else (
              "SHRINKS with N" if lr.slope+ci < 0 else "FLAT in N -- safe to extrapolate")
    print(f"  gap vs log10(N): slope {lr.slope:+.4f} +/- {ci:.4f}  R^2 {lr.rvalue**2:.3f}  -> {verdict}", flush=True)
    print(f"  extrapolated gap at N=22,755: {lr.intercept + lr.slope*np.log10(22755):+.4f} "
          f"(vs measured median {np.median(ys):+.4f} in the overlap)", flush=True)
print("\n  CG-ANM is a BASELINE, not a ceiling. Compare CG-ANM to CG-ANM across buckets; never mix it", flush=True)
print("  with all-atom ANM numbers from the lower buckets.", flush=True)
