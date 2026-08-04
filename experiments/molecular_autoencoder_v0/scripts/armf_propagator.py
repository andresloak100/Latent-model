"""STEP 3: learned propagator vs OU, strengthened acceptance test, lag-time sweep.

Acceptance metrics (gen vs reference), the discriminators OU cannot pass by design:
  marginals   std ratio, per-mode JS, excess kurtosis (Gaussian=0)
  kinetics    IAT ratio (tau-aware), basin-transition RATE per 1000 steps
  COUPLING    cross-mode linear corr (off-diag) and amplitude coupling corr(|c_i|,|c_j|)
              -- OU in the ANM basis is independent per mode, so these are ~0 for OU;
              any nonzero reference coupling is unreachable by OU, reachable by a learned model.

Sweep tau = 1/10/50/100 ns; DDPM with FiLM per-layer conditioning; delta-vs-absolute
ablation; OU at matched lag; step-1 conditioning guard reported per DDPM; steps-to-1ms.
Marginals/coupling use the FULL reference (stationary, stride-free); kinetics use
tau-separated reference pairs (robust at all tau).
"""
import argparse, math, numpy as np, torch, torch.nn as nn, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
BB = ["N", "CA", "C", "O"]; TEMP, R0 = "320", "0"
L, Tdiff, kT = 64, 100, 0.593
USE = ["3a5zD02", "3jvvA01"]; TAUS = [1, 10, 50, 100]
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
    n = len(xyz); H = np.zeros((3 * n, 3 * n)); ar = np.arange(3); I, J, U = [], [], []
    for i in range(n):
        dv = xyz - xyz[i]; dd = np.linalg.norm(dv, axis=1)
        for j in np.where((dd < cutoff) & (dd > 1e-6))[0]:
            if j <= i: continue
            I.append(i); J.append(j); U.append(dv[j] / dd[j])
    I, J, U = np.array(I), np.array(J), np.array(U); b = U[:, :, None] * U[:, None, :]
    def sc(a, c, V):
        r = (3*a[:, None, None] + ar[None, :, None]) * np.ones((1, 1, 3), int)
        co = (3*c[:, None, None] + ar[None, None, :]) * np.ones((1, 3, 1), int)
        np.add.at(H, (r.ravel(), co.ravel()), V.ravel())
    sc(I, I, b); sc(J, J, b); sc(I, J, -b); sc(J, I, -b)
    w, V = np.linalg.eigh(H); return V[:, 6:6 + K], w[6:6 + K]


# ---------------- strengthened metrics ----------------
def offdiag(C):
    n = C.shape[0]; return float(np.abs(C[~np.eye(n, dtype=bool)]).mean())

def excess_kurt(X):
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9); return float((((Xc ** 4).mean(0)) - 3).mean())

def marg_js(gen, ref, bins=30):
    js = []
    for i in range(gen.shape[1]):
        lo = min(gen[:, i].min(), ref[:, i].min()); hi = max(gen[:, i].max(), ref[:, i].max())
        e = np.linspace(lo, hi, bins + 1)
        pg = np.histogram(gen[:, i], e)[0] / len(gen) + 1e-12; pr = np.histogram(ref[:, i], e)[0] / len(ref) + 1e-12
        m = 0.5 * (pg + pr); js.append(0.5 * (pg * np.log(pg / m)).sum() + 0.5 * (pr * np.log(pr / m)).sum())
    return float(np.mean(js))

def iat_series(x, maxlag=400):
    x = x - x.mean(); v = (x * x).mean()
    if v < 1e-12: return 0.0
    acf = np.correlate(x, x, "full")[len(x) - 1:] / (v * len(x)); t = 1.0
    for k in range(1, min(len(acf), maxlag)):
        if acf[k] < 0.05: break
        t += 2 * acf[k]
    return t

def iat_ref_tau(Z, tau, maxk=200):                               # ref IAT in tau-units from stride-1 pairs
    out = []
    for i in range(Z.shape[1]):
        x = Z[:, i] - Z[:, i].mean(); v = (x * x).mean()
        if v < 1e-12: out.append(0.0); continue
        t = 1.0
        for k in range(1, maxk):
            lag = k * tau
            if lag >= len(x): break
            ac = (x[:-lag] * x[lag:]).mean() / v
            if ac < 0.05: break
            t += 2 * ac
        out.append(t)
    return np.mean(out)

def basins(X, thr):                                             # top-2-mode median quadrants
    return 2 * (X[:, 0] > thr[0]).astype(int) + (X[:, 1] > thr[1]).astype(int)

def trans_rate(labels):
    return float((labels[1:] != labels[:-1]).mean() * 1000)

def bench(gen, ref, top2, thr, tau, ref_full):
    m = {}
    m["std"] = float((gen.std(0) / (ref_full.std(0) + 1e-9)).mean())
    m["js"] = marg_js(gen, ref_full)
    m["kurt_g"] = excess_kurt(gen); m["kurt_r"] = excess_kurt(ref_full)
    m["xcorr_g"] = offdiag(np.corrcoef(gen.T)); m["xcorr_r"] = offdiag(np.corrcoef(ref_full.T))
    m["amp_g"] = offdiag(np.corrcoef(np.abs(gen).T)); m["amp_r"] = offdiag(np.corrcoef(np.abs(ref_full).T))
    m["iat_g"] = np.mean([iat_series(gen[:, i]) for i in range(gen.shape[1])])
    m["iat_r"] = iat_ref_tau(ref_full, tau)
    m["trans_g"] = trans_rate(basins(gen[:, top2], thr))
    lab_r = basins(ref_full[:, top2], thr); m["trans_r"] = float((lab_r[tau:] != lab_r[:-tau]).mean() * 1000)
    return m


# ---------------- FiLM denoiser (per-layer conditioning) ----------------
class FiLMDenoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__()
        self.inp = nn.Linear(L + 16, h); self.h1 = nn.Linear(h, h); self.h2 = nn.Linear(h, h); self.out = nn.Linear(h, L)
        self.f0 = nn.Linear(L, 2 * h); self.f1 = nn.Linear(L, 2 * h); self.f2 = nn.Linear(L, 2 * h)
        for f in (self.f0, self.f1, self.f2):                     # zero-init FiLM -> identity at start (stable)
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
    def temb(self, t):
        f = torch.exp(torch.arange(8) * (-math.log(1e4) / 8)); a = t[:, None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def film(self, lin, x, cond):
        g, b = lin(cond).chunk(2, -1); return torch.nn.functional.gelu(x * (1 + g) + b)
    def forward(self, zt, cond, t):
        x = self.film(self.f0, self.inp(torch.cat([zt, self.temb(t)], -1)), cond)
        x = self.film(self.f1, self.h1(x), cond); x = self.film(self.f2, self.h2(x), cond)
        return self.out(x)


betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1 - betas, 0)


def train_ddpm(Zn, h, tau, param):
    m = FiLMDenoiser(L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    zc = Zn[:h - tau]; tgt = (Zn[tau:h] - Zn[:h - tau]) if param == "delta" else Zn[tau:h]
    for ep in range(1500):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),))
        c, x0 = zc[i], tgt[i]; k = torch.randint(0, Tdiff, (len(i),)); noise = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None] * x0 + (1 - ac[k]).sqrt()[:, None] * noise
        opt.zero_grad(); ((m(xk, c, k) - noise) ** 2).mean().backward(); opt.step()
    return m


@torch.no_grad()
def ddpm_step(m, cond):
    x = torch.randn(cond.shape[0], L)
    for k in reversed(range(Tdiff)):
        eps = m(x, cond, torch.full((cond.shape[0],), k)); x0 = (x - (1 - ac[k]).sqrt() * eps) / ac[k].sqrt()
        x = ac[k - 1].sqrt() * x0 + (1 - ac[k - 1]).sqrt() * torch.randn_like(x) if k > 0 else x0
    return x


def rollout_ddpm(m, z0, H, param):
    z = z0.clone(); out = [z.clone()]
    for _ in range(H):
        s = ddpm_step(m, z); z = (z + s) if param == "delta" else s
        z = torch.nan_to_num(z, nan=0.0).clamp(-12, 12)          # guard blow-up (whitened space)
        out.append(z.clone())
    return torch.cat(out, 0).numpy()


ap = argparse.ArgumentParser(); ap.add_argument("--H", type=int, default=1500); args = ap.parse_args()
print(f"[propagator] strengthened test, tau sweep {TAUS}, FiLM cond, delta/absolute, vs OU. H={args.H}")
for dom in USE:
    with h5py.File(f"{DATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
        coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); T = d.shape[0]; h = int(T * 0.8)
    mean = d[:h].mean(0); modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d - mean) @ B.T; zmu = Z[:h].mean(0); sd = np.sqrt(0.593 / np.clip(lam, 1e-8, None))
    Zn_all = ((Z - zmu) / sd).astype(np.float32); ref_full = Z
    top2 = np.argsort(Z.std(0))[::-1][:2]; thr = np.median(Z[:, top2], 0)
    v = kT / np.clip(lam, 1e-8, None); v *= (Z[:h].var(0).sum() / v.sum())
    a1 = (Z[:h - 1] * Z[1:h]).mean(0) / ((Z[:h - 1] ** 2).mean(0) + 1e-9)
    gamma = float(np.median((-lam / np.log(np.clip(a1, 0.02, 0.98)))[lam > 0]))
    print(f"\n=== {dom} (nCA {ca.sum()}) ===")
    print(f"  {'model':16s}{'tau':>4}{'std':>6}{'js':>6}{'kurtG/R':>10}{'xcorrG/R':>11}{'ampG/R':>11}"
          f"{'iatRat':>7}{'transG/R':>11}{'->1ms':>8}")
    for tau in TAUS:
        steps1ms = int(1e6 / tau)
        # OU at lag tau
        a_ou = np.exp(-lam * tau / gamma); x = Z[h].copy(); roll = [x.copy()]
        for _ in range(args.H):
            x = a_ou * x + np.sqrt(v * (1 - a_ou ** 2)) * np.random.randn(L); roll.append(x.copy())
        mo = bench(np.array(roll), ref_full, top2, thr, tau, ref_full)
        print(f"  {'OU':16s}{tau:>4}{mo['std']:>6.2f}{mo['js']:>6.2f}{mo['kurt_g']:>5.1f}/{mo['kurt_r']:<4.1f}"
              f"{mo['xcorr_g']:>5.2f}/{mo['xcorr_r']:<5.2f}{mo['amp_g']:>5.2f}/{mo['amp_r']:<5.2f}"
              f"{mo['iat_g']/max(mo['iat_r'],1e-6):>7.2f}{mo['trans_g']:>5.0f}/{mo['trans_r']:<5.0f}{steps1ms:>8}")
        for param in ("absolute", "delta"):
            m = train_ddpm(torch.tensor(Zn_all), h, tau, param)
            gen = rollout_ddpm(m, torch.tensor(Zn_all[h:h + 1]), args.H, param) * sd + zmu
            md = bench(gen, ref_full, top2, thr, tau, ref_full)
            # step-1 cond guard: a_ddpm vs a_ref at this lag
            idx = np.random.choice(h - tau, min(150, h - tau), replace=False)
            with torch.no_grad():
                g1 = ddpm_step(m, torch.tensor(Zn_all[idx])).numpy()
            g1 = (Zn_all[idx] + g1) if param == "delta" else g1
            cnd = Zn_all[idx]
            add = ((cnd - cnd.mean(0)) * (g1 - g1.mean(0))).mean(0) / (((cnd - cnd.mean(0)) ** 2).mean(0) + 1e-9)
            aref_tau = (Zn_all[:h - tau] * Zn_all[tau:h]).mean(0) / ((Zn_all[:h - tau] ** 2).mean(0) + 1e-9)
            guard = f"cg{abs(add).mean()/max(abs(aref_tau).mean(),1e-6):.2f}"
            print(f"  {'DDPM-' + param:16s}{tau:>4}{md['std']:>6.2f}{md['js']:>6.2f}{md['kurt_g']:>5.1f}/{md['kurt_r']:<4.1f}"
                  f"{md['xcorr_g']:>5.2f}/{md['xcorr_r']:<5.2f}{md['amp_g']:>5.2f}/{md['amp_r']:<5.2f}"
                  f"{md['iat_g']/max(md['iat_r'],1e-6):>7.2f}{md['trans_g']:>5.0f}/{md['trans_r']:<5.0f}{steps1ms:>8} {guard}")
print("\n  discriminators: OU has xcorr~0, amp~0, kurt~0 BY CONSTRUCTION. If ref xcorr/amp/kurt ~0 too")
print("  -> task is Gaussian/single-basin at this lag, no learned propagator needed (pre-registered).")
print("  A learned win = matching ref xcorr/amp/kurt/trans that OU misses. cg = conditioning guard (a_ddpm/a_ref).")
