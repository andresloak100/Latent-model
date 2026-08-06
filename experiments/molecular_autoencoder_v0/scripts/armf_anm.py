"""Sparse all-atom ANM: the ZERO-SHOT PEER baseline, computable across the full ATLAS N range.

WHY THIS IS NOT A REPEAT OF THE CG-ANM FAILURE. CG-ANM was an APPROXIMATION -- a different basis
(QR-orthonormalised rigid-residue modes) whose bias was measured at -0.059 median and, fatally, was
N-DEPENDENT (slope +0.110 +/- 0.039). Sparse eigsh on the same Hessian is the IDENTICAL eigenproblem
solved without materialising the dense matrix, so it agrees with dense to numerical precision BY
CONSTRUCTION and cannot carry an N-dependent bias. Different category entirely.
VALIDATED against dense on the overlap: max relative eigenvalue error 1.8e-10 to 5.0e-09, subspace
overlap 1.000000 -- and already 2.3x faster at N=2,449 (15.4 s vs 35.2 s).

TWO POLICIES THIS FILE ENFORCES.

1. RETRY WITH ESCALATION BEFORE EXCLUDING. Observed convergence time varies 45 s vs 205 s at matched
   N -- that is shift-invert convergence, which is TUNABLE. A timeout that preferentially kills
   slow-converging systems is an N-CORRELATED EXCLUSION if convergence difficulty tracks size, which
   would void the very comparison it feeds. So: attempt 1 at default, attempt 2 with adjusted
   sigma/maxiter, attempt 3 with a larger Krylov subspace -- and only then record a Family A
   exclusion. Retrying costs minutes against a bias that would void the result.

2. THE CUTOFF IS SELECTED ON TRAINING SYSTEMS ONLY. The peer baseline should be the BEST zero-shot
   method, not an arbitrary one: 10 A on all-atom gives ~142 pairs/atom, which over-connects and
   washes out local flexibility, while the all-atom literature sits at 5-7 A (~22 pairs/atom at 5 A,
   and 6.5x cheaper). If ANM-10 is a WEAK version of ANM then the codec beating it is a hollow win --
   and that comparison carries the thesis. Protocol: sweep {5, 7, 10} A on TRAINING systems, pick the
   single best cutoff by mean FVE ON TRAINING SYSTEMS, apply that one cutoff to held-out systems.
   Selecting on train and applying to held-out keeps it a fair baseline rather than an oracle; all
   three training curves are reported so the choice is visible.
   ANM-10 is RETAINED as the labelled CONTINUITY baseline for comparison against earlier results.
   The two are NEVER pooled."""
import time, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

CUTOFF_SWEEP = (5.0, 7.0, 10.0)
CONTINUITY_CUTOFF = 10.0          # the cutoff every earlier ANM number used; never pooled with a new one


def hessian(ref, cutoff):
    """All-atom ANM Hessian, sparse CSR. Vectorised outer products; identical maths to the dense path."""
    n = len(ref); pr = cKDTree(ref).query_pairs(cutoff, output_type='ndarray')
    if len(pr) == 0: return None
    d = ref[pr[:, 1]] - ref[pr[:, 0]]; L = np.linalg.norm(d, axis=1); u = d / (L[:, None] + 1e-12)
    B = u[:, :, None] * u[:, None, :]; i = pr[:, 0]; j = pr[:, 1]; a = np.arange(3)
    R, C, V = [], [], []
    for (I, J, s) in ((i, j, -1.0), (j, i, -1.0), (i, i, 1.0), (j, j, 1.0)):
        R.append((3*I[:, None, None] + a[None, :, None]).repeat(3, 2).ravel())
        C.append((3*J[:, None, None] + a[None, None, :]).repeat(3, 1).ravel())
        V.append((s * B).ravel())
    return coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                      shape=(3*n, 3*n)).tocsr()


def modes(ref, k, cutoff, log=None):
    """Lowest k non-trivial modes. ESCALATES before giving up -- see policy 1.
    Returns (V, attempts, seconds) or (None, attempts, seconds) after all escalations fail."""
    H = hessian(ref, cutoff)
    if H is None: return None, 0, 0.0
    plans = [dict(sigma=1e-6, maxiter=None, ncv=None),                 # 1: default
             dict(sigma=1e-4, maxiter=20000, ncv=min(4*(k+6)+20, H.shape[0]-1)),  # 2: shifted + more iters
             dict(sigma=1e-8, maxiter=50000, ncv=min(8*(k+6)+40, H.shape[0]-1))]  # 3: larger Krylov space
    t0 = time.time()
    for att, p in enumerate(plans, 1):
        try:
            w, V = eigsh(H, k=k+6, which='LM', **{kk: vv for kk, vv in p.items() if vv is not None})
            o = np.argsort(w)
            if log: log(f"      ANM cutoff {cutoff} A converged on attempt {att} ({time.time()-t0:.0f}s)")
            return V[:, o][:, 6:], att, time.time() - t0
        except Exception as e:
            if log: log(f"      ANM attempt {att} failed ({type(e).__name__}); escalating")
    return None, len(plans), time.time() - t0


def fve(ho_centered, V, sst):
    """Held-out FVE of the ANM subspace. V is orthonormal (3N x k)."""
    p = ho_centered @ V
    return float((p ** 2).sum() / (sst + 1e-12))


def select_cutoff_on_train(train_systems, k, log=print):
    """Sweep cutoffs on TRAINING systems only; return the single best. Never touches held-out data."""
    log(f"  ANM cutoff selection on {len(train_systems)} TRAINING systems (held-out never seen):")
    means = {}
    for c in CUTOFF_SWEEP:
        vals = []
        for s in train_systems:
            V, _, _ = modes(s["ref"], k, c)
            if V is not None: vals.append(fve(s["ho"], V, s["sst"]))
        means[c] = float(np.mean(vals)) if vals else float('nan')
        log(f"    cutoff {c:>4} A: mean train FVE {means[c]:.4f}  (n={len(vals)})")
    best = max((c for c in means if np.isfinite(means[c])), key=lambda c: means[c])
    log(f"  -> SELECTED {best} A by training FVE; applied unchanged to held-out systems.")
    log(f"     (ANM-{CONTINUITY_CUTOFF:.0f} retained separately as the labelled CONTINUITY baseline; never pooled.)")
    return best, means
