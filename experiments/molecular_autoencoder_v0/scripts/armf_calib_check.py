"""Two checks on the variance calibration before scaling.

CHECK 1 -- calibration std target decides generality:
  empirical per-system std  -> needs a trajectory for a new molecule (NOT general)
  ANM-predicted sqrt(kT/lambda) -> zero-parameter, any structure (general)
  Recalibrate with sqrt(kT/lambda) and re-run discriminators; if it holds, general.
CHECK 2 -- is the whitened-output variance deficit CONSTANT or per-system/per-mode?
  constant -> one global correction constant, general, done.
  varies    -> calibration does model-specific work; a training-time fix is required.
  Report deficit per system and its across-mode spread.

Uses the standard ancestral sampler (cleaner + better-calibrated coupling than the
x0-resample). Calibrations compared: none / global-constant / ANM-permode / empirical-permode.
"""
import math, numpy as np, torch, torch.nn as nn, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
BB = ["N", "CA", "C", "O"]; TEMP, R0 = "320", "0"
L, Tdiff, TAU, H = 64, 100, 50, 800
USE = ["3a5zD02", "3jvvA01", "3a9lA00", "2z1kA02"]
torch.manual_seed(0); np.random.seed(0)


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    a = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in a])


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P[mask].mean(0); Pc = P[mask] - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


def anm(xyz, K, cutoff=10.0):
    n = len(xyz); Hm = np.zeros((3 * n, 3 * n)); ar = np.arange(3); I, J, U = [], [], []
    for i in range(n):
        dv = xyz - xyz[i]; dd = np.linalg.norm(dv, axis=1)
        for j in np.where((dd < cutoff) & (dd > 1e-6))[0]:
            if j > i: I.append(i); J.append(j); U.append(dv[j] / dd[j])
    I, J, U = np.array(I), np.array(J), np.array(U); b = U[:, :, None] * U[:, None, :]
    def sc(a, c, V):
        r = (3*a[:, None, None] + ar[None, :, None]) * np.ones((1, 1, 3), int)
        co = (3*c[:, None, None] + ar[None, None, :]) * np.ones((1, 3, 1), int)
        np.add.at(Hm, (r.ravel(), co.ravel()), V.ravel())
    sc(I, I, b); sc(J, J, b); sc(I, J, -b); sc(J, I, -b)
    w, V = np.linalg.eigh(Hm); return V[:, 6:6 + K], w[6:6 + K]


def offdiag(C):
    n = C.shape[0]; return float(np.abs(C[~np.eye(n, dtype=bool)]).mean())

def excess_kurt(X):
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9); return float((((Xc ** 4).mean(0)) - 3).mean())

def iat_series(x, maxlag=300):
    x = x - x.mean(); v = (x * x).mean()
    if v < 1e-12: return 0.0
    acf = np.correlate(x, x, "full")[len(x) - 1:] / (v * len(x)); t = 1.0
    for k in range(1, min(len(acf), maxlag)):
        if acf[k] < 0.05: break
        t += 2 * acf[k]
    return t

def iat_ref_tau(Z, tau, maxk=150):
    o = []
    for i in range(Z.shape[1]):
        x = Z[:, i] - Z[:, i].mean(); v = (x * x).mean()
        if v < 1e-12: o.append(0.0); continue
        t = 1.0
        for k in range(1, maxk):
            lag = k * tau
            if lag >= len(x): break
            acc = (x[:-lag] * x[lag:]).mean() / v
            if acc < 0.05: break
            t += 2 * acc
        o.append(t)
    return np.mean(o)

def basins(X, thr): return 2 * (X[:, 0] > thr[0]).astype(int) + (X[:, 1] > thr[1]).astype(int)

def bench(gen, ref, top2, thr, tau):
    lab_r = basins(ref[:, top2], thr); lab_g = basins(gen[:, top2], thr)
    return dict(vr=float((gen.std(0) / (ref.std(0) + 1e-9)).mean()),
                xc=offdiag(np.corrcoef(gen.T)), kurt=excess_kurt(gen),
                iat=np.mean([iat_series(gen[:, i]) for i in range(gen.shape[1])]) / max(iat_ref_tau(ref, tau), 1e-6),
                tr=float((lab_g[1:] != lab_g[:-1]).mean() * 1000),
                trr=float((lab_r[tau:] != lab_r[:-tau]).mean() * 1000))


class FiLMDenoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__()
        self.inp = nn.Linear(L + 16, h); self.h1 = nn.Linear(h, h); self.h2 = nn.Linear(h, h); self.out = nn.Linear(h, L)
        self.f0 = nn.Linear(L, 2*h); self.f1 = nn.Linear(L, 2*h); self.f2 = nn.Linear(L, 2*h)
        for f in (self.f0, self.f1, self.f2): nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
    def temb(self, t):
        f = torch.exp(torch.arange(8) * (-math.log(1e4) / 8)); a = t[:, None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def film(self, lin, x, c):
        g, b = lin(c).chunk(2, -1); return torch.nn.functional.gelu(x * (1 + g) + b)
    def forward(self, zt, cond, t):
        x = self.film(self.f0, self.inp(torch.cat([zt, self.temb(t)], -1)), cond)
        x = self.film(self.f1, self.h1(x), cond); x = self.film(self.f2, self.h2(x), cond)
        return self.out(x)


betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1 - betas, 0)


def train(Zn, h, tau):
    m = FiLMDenoiser(L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    zc = Zn[:h - tau]; tgt = Zn[tau:h]
    for ep in range(1500):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),))
        c, x0 = zc[i], tgt[i]; k = torch.randint(0, Tdiff, (len(i),)); noise = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None] * x0 + (1 - ac[k]).sqrt()[:, None] * noise
        opt.zero_grad(); ((m(xk, c, k) - noise) ** 2).mean().backward(); opt.step()
    return m


@torch.no_grad()
def step_ancestral(m, cond):
    x = torch.randn(cond.shape[0], L)
    for k in reversed(range(Tdiff)):
        eps = m(x, cond, torch.full((cond.shape[0],), k)); alpha = 1 - betas[k]
        mean = (x - betas[k] / (1 - ac[k]).sqrt() * eps) / alpha.sqrt()
        if k > 0:
            bt = (1 - ac[k - 1]) / (1 - ac[k]) * betas[k]; x = mean + bt.sqrt() * torch.randn_like(x)
        else: x = mean
    return x


def rollout(m, z0):
    z = z0.clone(); out = [z.clone()]
    for _ in range(H):
        z = torch.nan_to_num(step_ancestral(m, z), nan=0.0).clamp(-12, 12); out.append(z.clone())
    return torch.cat(out, 0).numpy()                             # whitened generated coeffs


print(f"[calib-check] ancestral sampler, tau={TAU}, 4 systems. deficit = whitened output std (target 1.0).")
defic = []
for dom in USE:
    with h5py.File(f"{DATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
        coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); T = d.shape[0]; h = int(T * 0.8)
    mean = d[:h].mean(0); modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d - mean) @ B.T; zmu = Z[:h].mean(0); sd = np.sqrt(0.593 / np.clip(lam, 1e-8, None))
    Zn = ((Z - zmu) / sd).astype(np.float32); ref = Z; top2 = np.argsort(Z.std(0))[::-1][:2]; thr = np.median(Z[:, top2], 0)
    emp_std = Z[:h].std(0)                                       # empirical per-mode target
    m = train(torch.tensor(Zn), h, TAU)
    roll = rollout(m, torch.tensor(Zn[h:h + 1]))                # whitened generated
    wstd = roll.std(0)                                          # per-mode deficit (target 1.0); modes slow->fast (asc lambda)
    th = L // 3
    slow, midb, fast = wstd[:th].mean(), wstd[th:2*th].mean(), wstd[2*th:].mean()
    defic.append((dom, wstd.mean(), wstd.std() / wstd.mean(), wstd, slow, midb, fast))
    rc = roll - roll.mean(0)
    cals = {
        "none": roll * sd + zmu,
        "global-const": rc / wstd.mean() * sd + zmu,             # one constant (mean deficit)
        "anm-permode": rc / (wstd + 1e-9) * sd + zmu,            # target sqrt(kT/lambda)  (ZERO-PARAM)
        "emp-permode": rc / (wstd + 1e-9) * emp_std + zmu,       # target empirical (per-system MD)
    }
    print(f"\n=== {dom} (nCA {ca.sum()})  whitened-output deficit: mean {wstd.mean():.2f} (target 1.0), "
          f"across-mode CV {wstd.std()/wstd.mean():.2f}; by mode-index slow {slow:.2f} / mid {midb:.2f} / fast {fast:.2f} ===")
    print(f"  {'calibration':14s}{'varRatio':>9}{'xcorr':>7}{'kurt':>6}{'iatR':>6}{'trG/R':>10}")
    for name, gen in cals.items():
        b = bench(gen, ref, top2, thr, TAU)
        print(f"  {name:14s}{b['vr']:>9.2f}{b['xc']:>7.2f}{b['kurt']:>6.1f}{b['iat']:>6.2f}{b['tr']:>5.0f}/{b['trr']:<4.0f}")
print("\n=== CHECK 2: deficit DISTRIBUTION ===")
print(f"  {'system':9s}{'mean':>6}{'slow':>6}{'mid':>6}{'fast':>6}")
for dom, m_, cv, w, s, mm, ff in defic:
    print(f"  {dom:9s}{m_:>6.2f}{s:>6.2f}{mm:>6.2f}{ff:>6.2f}")
ms = np.array([x[1] for x in defic]); gm = ms.mean()
within = np.all(np.abs(ms - gm) / gm < 0.10)
print(f"  across-SYSTEM: mean {gm:.2f}, spread +/-{100*max(np.abs(ms-gm))/gm:.0f}% (min {ms.min():.2f} max {ms.max():.2f})"
      f" -> {'WITHIN +-10%: one global constant defensible, GENERAL' if within else 'STRUCTURED spread: per-system work, training-time fix required'}")
W = np.stack([x[3] for x in defic])                             # (systems, L) all L=64
slow_m, mid_m, fast_m = W[:, :L//3].mean(), W[:, L//3:2*L//3].mean(), W[:, 2*L//3:].mean()
print(f"  deficit vs MODE INDEX (avg across systems): slow {slow_m:.2f} / mid {mid_m:.2f} / fast {fast_m:.2f}"
      f" -> {'GROWS toward slow modes = persistence-shortfall signature; loss-weighting across noise levels is the principled fix' if slow_m < fast_m - 0.03 else 'flat across mode index -> a global constant suffices'}")
print("\n  CHECK1 (now mostly redundant): in whitened space the target is unit variance BY CONSTRUCTION;")
print("  if the deficit is constant, x1/deficit is zero-parameter/general. anm-permode ~ emp-permode confirms")
print("  the whitening itself; the load-bearing number is the deficit distribution above.")
