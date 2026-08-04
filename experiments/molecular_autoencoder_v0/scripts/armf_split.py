"""Is all-atom displacement two mixed signals -- collective (compressible) + local
jitter (incompressible, mostly side-chain)? Leak-free split PCA (fit frames 1-50,
eval 51-100) at MATCHED L for CA / backbone / all-atom / side-chain, with the
ABSOLUTE residual in Angstrom next to every percentage, plus variance shares."""
import glob, numpy as np
from pathlib import Path
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
BB = {"N", "CA", "C", "O"}


def split_pca(d, L):
    """d: (T,n,3). Fit PCA on first half, eval on second. Return (reduction%, residualA, nullA)."""
    T = d.shape[0]; h = T // 2
    X = d.reshape(T, -1); tr, ev = X[:h], X[h:]
    mean = tr.mean(0)
    _, _, Vt = np.linalg.svd(tr - mean, full_matrices=False)
    Lc = min(L, Vt.shape[0]); B = Vt[:Lc]
    rec = (ev - mean) @ B.T @ B + mean
    npera = d.shape[1]
    resid = np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * npera))
    null = np.sqrt((ev ** 2).sum() / (ev.shape[0] * npera))
    return 100 * (null - resid) / null, resid, null


Ls = [8, 16, 32]
rows = {"CA": [], "backbone": [], "all-atom": [], "side-chain": []}
sc_share = []
for p in sorted(glob.glob(f"{WR}/armf_data/*.npz")):
    z = np.load(p, allow_pickle=True)
    c = z["coords"].astype(np.float64); names = np.asarray([str(x) for x in z["atom_name"]])
    d = c - c[0]
    bb = np.array([n in BB for n in names]); sc = ~bb; ca = names == "CA"
    sets = {"CA": ca, "backbone": bb, "all-atom": np.ones(len(names), bool), "side-chain": sc}
    for name, m in sets.items():
        if m.sum() < 12:
            continue
        rows[name].append({L: split_pca(d[:, m, :], L) for L in Ls})
    # variance share: fraction of total all-atom displacement variance in side-chain atoms
    pv = (d ** 2).sum(-1).mean(0)   # per-atom variance
    sc_share.append(pv[sc].sum() / pv.sum())

print("Leak-free split PCA -- reduction% (absolute residual A) at matched L; null in header")
for name in ("CA", "backbone", "all-atom", "side-chain"):
    r = rows[name]
    if not r:
        continue
    nullA = np.mean([r[i][8][2] for i in range(len(r))])
    line = f"  {name:11s} null={nullA:.2f}A |"
    for L in Ls:
        red = np.mean([x[L][0] for x in r]); res = np.mean([x[L][1] for x in r])
        line += f"  L{L}: {red:2.0f}% ({res:.2f}A)"
    print(line)
print(f"\n  side-chain share of all-atom displacement variance: {100*np.mean(sc_share):.0f}% "
      f"(range {100*min(sc_share):.0f}-{100*max(sc_share):.0f}%)")
print(f"  C-C bond ~1.54 A for scale; chemistry needs residual well under that.")
