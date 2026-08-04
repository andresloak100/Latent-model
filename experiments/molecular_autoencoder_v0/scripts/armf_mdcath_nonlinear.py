"""Nonlinear-vs-PCA rerun on mdCATH (Part 2, now with enough frames).

Same per-system constraint, no cross-system sharing, no static shortcut, matched
temporal split and L as MISATO -- but on mdCATH Window C (all 500 frames, 250/250
split), so the nonlinear AE has 250 train frames instead of 25. AE operates in the
train-PCA span; both PCA-L and the AE are limited to it; RMSD in full 3N space.
  Nonlinear clearly beats PCA -> linear ceiling was misleading, codec viable.
  Nonlinear ~= PCA (now meaningful, not overfit-limited) -> linear ceiling real."""
import glob, numpy as np, torch, torch.nn as nn
from pathlib import Path
import h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
Ls = [8, 16, 32, 64]


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    atoms = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in atoms])


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pcen = P[mask].mean(0); Pc = P[mask] - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q); d = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pcen) @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T + rcen
    return out


class AE(nn.Module):
    def __init__(self, r, L, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(r, h), nn.GELU(), nn.Linear(h, L))
        self.dec = nn.Sequential(nn.Linear(L, h), nn.GELU(), nn.Linear(h, r))
    def forward(self, x): return self.dec(self.enc(x))


def evaluate(d, Ls):
    """d:(T,n,3). PCA vs nonlinear-AE at each L. Returns {L:(pca_red, ae_red, null)}."""
    T = d.shape[0]; h = T // 2; npera = d.shape[1]
    X = d.reshape(T, -1); tr, ev = X[:h], X[h:]; mean = tr.mean(0)
    U, S, Vt = np.linalg.svd(tr - mean, full_matrices=False)
    ctr = (tr - mean) @ Vt.T; cev = (ev - mean) @ Vt.T
    null = np.sqrt((ev ** 2).sum() / (ev.shape[0] * npera))
    def r_of(c):
        rec = c @ Vt + mean
        return np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * npera))
    out = {}
    for L in Ls:
        cp = cev.copy(); cp[:, L:] = 0.0
        pca = r_of(cp)
        torch.manual_seed(0)
        m = AE(ctr.shape[1], min(L, ctr.shape[1])); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        Xtr = torch.tensor(ctr, dtype=torch.float32)
        for _ in range(800):
            opt.zero_grad(); loss = ((m(Xtr) - Xtr) ** 2).sum(1).mean(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            rec = m(torch.tensor(cev, dtype=torch.float32)).numpy()
        ae = r_of(rec)
        out[L] = (100 * (null - pca) / null, 100 * (null - ae) / null, null)
    return out


agg = {("all-atom", L): {"pca": [], "ae": []} for L in Ls}
agg.update({("CA", L): {"pca": [], "ae": []} for L in Ls})
for fp in sorted(glob.glob(f"{DATA}/*.h5")):
    try:
        f = h5py.File(fp, "r")
    except Exception:
        continue
    with f:
        dom = list(f.keys())[0]; g = f[dom]
        if "320" not in g or "0" not in g["320"]:
            continue
        z = np.array(g["z"]); N = len(z); heavy = z != 1
        names = parse_names(g, N)
        if len(names) != N:
            continue
        nmh = names[heavy]; ca = nmh == "CA"
        if ca.sum() < 3:
            continue
        coords = g["320"]["0"]["coords"][:].astype(np.float64)
        al = kabsch(coords[:, heavy, :], ca); dall = al - al[0]
        for sname, m in (("all-atom", np.ones(heavy.sum(), bool)), ("CA", ca)):
            res = evaluate(dall[:, m, :], Ls)
            for L in Ls:
                agg[(sname, L)]["pca"].append(res[L][0]); agg[(sname, L)]["ae"].append(res[L][1])

print("=== mdCATH nonlinear-vs-PCA (per-system, Window C 250/250 split) ===")
for s in ("all-atom", "CA"):
    line = f"  {s:9s}"
    for L in Ls:
        p = np.mean(agg[(s, L)]["pca"]); a = np.mean(agg[(s, L)]["ae"])
        line += f"  L{L}: PCA {p:2.0f}% / AE {a:2.0f}%"
    print(line)
aa = {L: (np.mean(agg[("all-atom", L)]["pca"]), np.mean(agg[("all-atom", L)]["ae"])) for L in Ls}
print(f"\n  read (all-atom L64): PCA {aa[64][0]:.0f}% vs AE {aa[64][1]:.0f}% -> "
      f"{'AE BEATS PCA (linear ceiling misleading, codec viable)' if aa[64][1]>aa[64][0]+3 else 'AE ~= PCA (linear ceiling real, now on 250 frames not overfit)'}")
