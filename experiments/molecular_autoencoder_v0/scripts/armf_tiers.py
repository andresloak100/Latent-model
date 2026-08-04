"""Leak-free tier table for the arm-F viability question.

T1  temporal-split PCA: fit modes on frames 1-50, evaluate reconstruction on
    51-100. Leak-free per-system ceiling (fit rank <= 49, so L<=32 is clean).
T3  ANM modes from the REFERENCE contact graph alone (no MD): the non-learned
    "modes from static conditioning" baseline. CA-level (all-atom ANM Hessian is
    3N x 3N and infeasible to eigendecompose at these sizes); T1 is also reported
    CA-level so T3 and T1 compare on the same atoms.
    T3 ~= T1(CA) -> structure predicts modes; the trained gap is optimisation.
    T3 << T1(CA) -> modes are system-specific; modal approach has a ceiling.
T2  cross-system PCA is ill-defined here (systems differ in N / atom identity),
    so a universal structure->modes map (T3) is the meaningful cross-system test.
Reduction vs the zero-displacement null, exactly like the other tiers.
"""
import glob, numpy as np
from pathlib import Path
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"


def split_pca_reduction(d, Ls):
    """d: (T,M) displacement. Fit PCA on first half, eval reconstruction on second."""
    T = d.shape[0]; h = T // 2
    tr, ev = d[:h], d[h:]
    mean = tr.mean(0)
    U, S, Vt = np.linalg.svd(tr - mean, full_matrices=False)   # Vt: (<=h-1, M)
    null = np.sqrt((ev ** 2).sum(-1).mean()) if ev.ndim == 2 else np.sqrt((ev ** 2).mean())
    # ev is (Te, M); reshape null to per-(frame,atom) later -- here M already flat
    out = {}
    for L in Ls:
        Lc = min(L, Vt.shape[0])
        B = Vt[:Lc]                                            # (Lc, M)
        rec = (ev - mean) @ B.T @ B + mean
        rmsd = np.sqrt(((rec - ev) ** 2).sum(-1).mean() / (ev.shape[1] // 3))
        nullr = np.sqrt((ev ** 2).sum(-1).mean() / (ev.shape[1] // 3))
        out[L] = 100 * (nullr - rmsd) / nullr
    return out


def anm_modes(ca, cutoff=13.0):
    """ANM Hessian from CA contact graph; return eigenvectors ascending, drop 6."""
    n = ca.shape[0]; H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dv = ca - ca[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j <= i: continue
            u = dv[j] / dist[j]; blk = np.outer(u, u)
            H[3*i:3*i+3, 3*j:3*j+3] -= blk; H[3*j:3*j+3, 3*i:3*i+3] -= blk
            H[3*i:3*i+3, 3*i:3*i+3] += blk; H[3*j:3*j+3, 3*j:3*j+3] += blk
    w, V = np.linalg.eigh(H)                                   # ascending
    return V[:, 6:]                                            # drop 6 rigid-body


def anm_reduction(dca, modes, Ls):
    """dca: (T, 3nCA) CA displacement (eval frames). Project onto lowest-L ANM modes."""
    out = {}
    npera = dca.shape[1] // 3
    null = np.sqrt((dca ** 2).sum(-1).mean() / npera)
    for L in Ls:
        B = modes[:, :L].T                                     # (L, 3nCA)
        rec = dca @ B.T @ B
        rmsd = np.sqrt(((rec - dca) ** 2).sum(-1).mean() / npera)
        out[L] = 100 * (null - rmsd) / null
    return out


Ls = [8, 16, 32]
print(f"{'system':8s}{'nCA':>5} | T1_allatom 8/16/32 | T1_CA 8/16/32 | T3_ANM(CA) 8/16/32")
for p in sorted(glob.glob(f"{WR}/armf_data/*.npz")):
    z = np.load(p, allow_pickle=True)
    c = z["coords"].astype(np.float64); names = np.asarray(z["atom_name"])
    T, N, _ = c.shape
    d_all = (c - c[0]).reshape(T, -1)                          # (T, 3N)
    t1a = split_pca_reduction(d_all, Ls)
    ca_mask = np.array([str(n) == "CA" for n in names])
    if ca_mask.sum() < 12:
        print(f"{Path(p).stem:8s}{int(ca_mask.sum()):>5} | <too few CA>"); continue
    ca = c[:, ca_mask, :]                                      # (T, nCA, 3)
    dca = (ca - ca[0]).reshape(T, -1)
    t1c = split_pca_reduction(dca, Ls)
    # ANM: modes from reference (frame 0), eval on held-out (second-half) frames
    modes = anm_modes(ca[0])
    t3 = anm_reduction(dca[T // 2:], modes, Ls)
    f = lambda o: "/".join(f"{o[L]:2.0f}" for L in Ls)
    print(f"{Path(p).stem:8s}{int(ca_mask.sum()):>5} |   {f(t1a):12s}   |  {f(t1c):10s}  |   {f(t3)}")
