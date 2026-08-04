"""Per-system nonlinear-vs-linear probe on MISATO displacement (Part 2).

Tests whether a NONLINEAR L-bottleneck beats PCA-L (the linear ceiling) on the
same per-system displacement. No cross-system sharing (per system), no static-
conditioning shortcut (input is displacement only), matched temporal split (frames
1-50 train, 51-100 eval) and matched L.

Design: 50 train frames cannot train a raw-3N MLP without overfitting, so the AE
operates in the TRAIN-PCA span (<=49 resolvable dims). Both PCA-L and the AE are
limited to that span, so the comparison is fair -- the only difference is
nonlinear vs linear compression WITHIN it. Reconstruction RMSD is computed in full
3N space (includes the out-of-span residual, identical for both)."""
import glob, numpy as np, torch, torch.nn as nn
from pathlib import Path
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
BB = {"N", "CA", "C", "O"}


def prep(d):
    """d:(T,n,3) -> train/eval PCA coeffs + basis + mean + eval displacement (3N)."""
    T = d.shape[0]; h = T // 2; M = d.shape[1] * 3
    X = d.reshape(T, -1); tr, ev = X[:h], X[h:]
    mean = tr.mean(0)
    U, S, Vt = np.linalg.svd(tr - mean, full_matrices=False)   # Vt:(r,M), r<=h-1
    ctr = (tr - mean) @ Vt.T; cev = (ev - mean) @ Vt.T          # coeffs (.,r)
    return ctr, cev, Vt, mean, ev, d.shape[1]


def rmsd_from_coeff(rec_c, Vt, mean, ev, npera):
    rec = rec_c @ Vt + mean
    return np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * npera))


def pca_L(cev, Vt, mean, ev, npera, L):
    c = cev.copy(); c[:, L:] = 0.0
    return rmsd_from_coeff(c, Vt, mean, ev, npera)


class AE(nn.Module):
    def __init__(self, r, L, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(r, h), nn.GELU(), nn.Linear(h, L))
        self.dec = nn.Sequential(nn.Linear(L, h), nn.GELU(), nn.Linear(h, r))
    def forward(self, x): return self.dec(self.enc(x))


def ae_L(ctr, cev, Vt, mean, ev, npera, L, seed=0):
    torch.manual_seed(seed)
    r = ctr.shape[1]
    Xtr = torch.tensor(ctr, dtype=torch.float32)
    m = AE(r, min(L, r)); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(600):
        opt.zero_grad(); loss = ((m(Xtr) - Xtr) ** 2).sum(1).mean(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        rec = m(torch.tensor(cev, dtype=torch.float32)).numpy()
    return rmsd_from_coeff(rec, Vt, mean, ev, npera)


Ls = [8, 16, 32]
agg = {("all-atom", k): {"pca": [], "ae": []} for k in Ls}
agg.update({("CA", k): {"pca": [], "ae": []} for k in Ls})
print(f"{'system':8s}{'set':9s}{'null':>6} | " + " ".join(f"L{L}:PCA/AE" for L in Ls))
for p in sorted(glob.glob(f"{WR}/armf_data/*.npz")):
    z = np.load(p, allow_pickle=True)
    c = z["coords"].astype(np.float64); names = np.asarray([str(x) for x in z["atom_name"]])
    for setname, mask in (("all-atom", np.ones(len(names), bool)), ("CA", names == "CA")):
        if mask.sum() < 12: continue
        d = c[:, mask, :] - c[0:1, mask, :]
        ctr, cev, Vt, mean, ev, npera = prep(d)
        null = np.sqrt((ev.reshape(ev.shape[0], -1) ** 2).sum() / (ev.shape[0] * npera))
        cells = []
        for L in Ls:
            rp = pca_L(cev, Vt, mean, ev, npera, L); ra = ae_L(ctr, cev, Vt, mean, ev, npera, L)
            agg[(setname, L)]["pca"].append(100 * (null - rp) / null)
            agg[(setname, L)]["ae"].append(100 * (null - ra) / null)
            cells.append(f"{100*(null-rp)/null:2.0f}/{100*(null-ra)/null:2.0f}")
        print(f"{Path(p).stem:8s}{setname:9s}{null:>6.2f} | " + "  ".join(cells))
print("\n=== MEAN reduction% (PCA vs nonlinear AE), matched L ===")
for setname in ("CA", "all-atom"):
    line = f"  {setname:9s}"
    for L in Ls:
        pca = np.mean(agg[(setname, L)]["pca"]); ae = np.mean(agg[(setname, L)]["ae"])
        line += f"  L{L}: PCA {pca:2.0f}% / AE {ae:2.0f}%"
    print(line)
