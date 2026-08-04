"""PCA-initialised AE (Part 2a, the fair nonlinear test). The AE STARTS at PCA-L
(linear select/place) with zero-initialised nonlinear residual branches, then
trains. Any improvement over PCA-L is REAL nonlinear structure; no improvement
settles the linear-ceiling caveat. mdCATH Window C (250/250 split), per-system, no
cross-system/no static shortcut, matched L, report vs PCA in % and A."""
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


class PCAInitAE(nn.Module):
    """z = select-top-L(x) + nl(x);  recon = place(z) + nl(z). Both nl branches
    zero-initialised, so at init this IS PCA-L; training can only add nonlinearity."""
    def __init__(self, r, L, h=128):
        super().__init__()
        self.enc_lin = nn.Linear(r, L, bias=False); self.dec_lin = nn.Linear(L, r, bias=False)
        with torch.no_grad():
            self.enc_lin.weight.zero_(); self.dec_lin.weight.zero_()
            for j in range(L):
                self.enc_lin.weight[j, j] = 1.0     # select coeff j
                self.dec_lin.weight[j, j] = 1.0     # place coeff j
        self.enc_nl = nn.Sequential(nn.Linear(r, h), nn.GELU(), nn.Linear(h, L))
        self.dec_nl = nn.Sequential(nn.Linear(L, h), nn.GELU(), nn.Linear(h, r))
        for m in (self.enc_nl[-1], self.dec_nl[-1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.enc_lin(x) + self.enc_nl(x)
        return self.dec_lin(z) + self.dec_nl(z)


def evaluate(d, Ls):
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
        Lc = min(L, ctr.shape[1])
        cp = cev.copy(); cp[:, Lc:] = 0.0; pca = r_of(cp)
        torch.manual_seed(0)
        m = PCAInitAE(ctr.shape[1], Lc); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        Xtr = torch.tensor(ctr, dtype=torch.float32)
        for _ in range(800):
            opt.zero_grad(); ((m(Xtr) - Xtr) ** 2).sum(1).mean().backward(); opt.step()
        m.eval()
        with torch.no_grad():
            rec = m(torch.tensor(cev, dtype=torch.float32)).numpy()
        ae = r_of(rec)
        out[L] = (100 * (null - pca) / null, 100 * (null - ae) / null)
    return out


agg = {(s, L): {"pca": [], "ae": []} for s in ("all-atom", "CA", "backbone") for L in Ls}
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
        nmh = names[heavy]; ca = nmh == "CA"; bb = np.isin(nmh, ["N", "CA", "C", "O"])
        if ca.sum() < 3:
            continue
        al = kabsch(g["320"]["0"]["coords"][:].astype(np.float64)[:, heavy, :], ca)
        d = al - al[0]
        for s, msk in (("all-atom", np.ones(heavy.sum(), bool)), ("CA", ca), ("backbone", bb)):
            res = evaluate(d[:, msk, :], Ls)
            for L in Ls:
                agg[(s, L)]["pca"].append(res[L][0]); agg[(s, L)]["ae"].append(res[L][1])

print("=== PCA-initialised AE vs PCA (mdCATH Window C, per-system) ===")
for s in ("all-atom", "CA", "backbone"):
    line = f"  {s:9s}"
    for L in Ls:
        p = np.mean(agg[(s, L)]["pca"]); a = np.mean(agg[(s, L)]["ae"])
        line += f"  L{L}: PCA {p:2.0f}% / AE {a:2.0f}% ({a-p:+.0f})"
    print(line)
d64 = np.mean(agg[("all-atom", 64)]["ae"]) - np.mean(agg[("all-atom", 64)]["pca"])
print(f"\n  read (all-atom L64): AE - PCA = {d64:+.1f} pts -> "
      f"{'REAL nonlinear structure (codec viable beyond linear)' if d64 > 3 else 'no gain over PCA -> LINEAR CEILING settled'}")
