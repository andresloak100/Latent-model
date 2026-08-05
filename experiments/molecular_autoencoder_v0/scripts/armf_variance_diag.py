"""Variance-collapse diagnosis (decide SCALE vs STRUCTURAL before any redesign).

STEP 1  post-hoc per-mode rescale of the generated coeffs to reference std, re-run the
        FULL discriminator set. xcorr/amp/kurt are scale-invariant, so if they were
        captured they survive; the test is whether marginals+basin-transitions then pass.
        pure scale -> a per-mode CALIBRATION is a legitimate fix (reported AS a calibration).
STEP 2a posterior variance: current x0-resample vs standard ancestral beta_tilde (narrow)
        vs beta (wide). Report variance ratio for each.
STEP 2b reverse-step count: DDIM(eta=1) respaced at 25/50/100/250, variance ratio vs steps.
Guidance scale = 1.0 (no classifier-free guidance in the model) -- ruled out.
Variance ratio reported alongside every discriminator on every row.
"""
import math, numpy as np, torch, torch.nn as nn, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
BB = ["N", "CA", "C", "O"]; TEMP, R0 = "320", "0"
L, Tdiff, TAU, H = 64, 100, 50, 800
USE = ["3a5zD02", "3jvvA01"]
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
    vr = float((gen.std(0) / (ref.std(0) + 1e-9)).mean())
    js = 0.0
    for i in range(gen.shape[1]):
        lo = min(gen[:, i].min(), ref[:, i].min()); hi = max(gen[:, i].max(), ref[:, i].max())
        e = np.linspace(lo, hi, 31)
        pg = np.histogram(gen[:, i], e)[0] / len(gen) + 1e-12; pr = np.histogram(ref[:, i], e)[0] / len(ref) + 1e-12
        mm = 0.5 * (pg + pr); js += 0.5 * (pg * np.log(pg / mm)).sum() + 0.5 * (pr * np.log(pr / mm)).sum()
    lab_r = basins(ref[:, top2], thr); lab_g = basins(gen[:, top2], thr)
    return dict(vr=vr, js=js / gen.shape[1], xc=offdiag(np.corrcoef(gen.T)),
                amp=offdiag(np.corrcoef(np.abs(gen).T)), kurt=excess_kurt(gen),
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
def step_current(m, cond):
    x = torch.randn(cond.shape[0], L)
    for k in reversed(range(Tdiff)):
        eps = m(x, cond, torch.full((cond.shape[0],), k)); x0 = (x - (1 - ac[k]).sqrt() * eps) / ac[k].sqrt()
        x = ac[k - 1].sqrt() * x0 + (1 - ac[k - 1]).sqrt() * torch.randn_like(x) if k > 0 else x0
    return x

@torch.no_grad()
def step_ancestral(m, cond, wide):
    x = torch.randn(cond.shape[0], L)
    for k in reversed(range(Tdiff)):
        eps = m(x, cond, torch.full((cond.shape[0],), k)); alpha = 1 - betas[k]
        mean = (x - betas[k] / (1 - ac[k]).sqrt() * eps) / alpha.sqrt()
        if k > 0:
            bt = (1 - ac[k - 1]) / (1 - ac[k]) * betas[k]
            sig = betas[k].sqrt() if wide else bt.sqrt()
            x = mean + sig * torch.randn_like(x)
        else: x = mean
    return x

@torch.no_grad()
def step_ddim(m, cond, nstep, eta):
    seq = list(reversed(np.unique(np.linspace(0, Tdiff - 1, nstep).round().astype(int)).tolist())) + [-1]
    x = torch.randn(cond.shape[0], L)
    for i in range(len(seq) - 1):
        t, tp = seq[i], seq[i + 1]; eps = m(x, cond, torch.full((cond.shape[0],), t))
        act = ac[t]; acp = ac[tp] if tp >= 0 else torch.tensor(1.0)
        x0 = (x - (1 - act).sqrt() * eps) / act.sqrt()
        if tp >= 0:
            sigma = eta * ((1 - acp) / (1 - act)).sqrt() * (1 - act / acp).clamp(min=0).sqrt()
            x = acp.sqrt() * x0 + (1 - acp - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * torch.randn_like(x)
        else: x = x0
    return x


def rollout(stepfn, z0):
    z = z0.clone(); out = [z.clone()]
    for _ in range(H):
        z = torch.nan_to_num(stepfn(z), nan=0.0).clamp(-12, 12); out.append(z.clone())
    return torch.cat(out, 0).numpy()


for dom in USE:
    with h5py.File(f"{DATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
        coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); T = d.shape[0]; h = int(T * 0.8)
    mean = d[:h].mean(0); modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d - mean) @ B.T; zmu = Z[:h].mean(0); sd = np.sqrt(0.593 / np.clip(lam, 1e-8, None))
    Zn = ((Z - zmu) / sd).astype(np.float32); ref = Z; top2 = np.argsort(Z.std(0))[::-1][:2]; thr = np.median(Z[:, top2], 0)
    m = train(torch.tensor(Zn), h, TAU); z0 = torch.tensor(Zn[h:h + 1])
    print(f"\n=== {dom} (nCA {ca.sum()}) tau={TAU}, guidance=1.0 (none) ===")
    print(f"  {'config':22s}{'varRatio':>9}{'js':>6}{'xcorr':>7}{'amp':>6}{'kurt':>6}{'iatR':>6}{'trG/R':>10}")
    def row(name, gen):
        b = bench(gen, ref, top2, thr, TAU)
        print(f"  {name:22s}{b['vr']:>9.2f}{b['js']:>6.2f}{b['xc']:>7.2f}{b['amp']:>6.2f}{b['kurt']:>6.1f}"
              f"{b['iat']:>6.2f}{b['tr']:>5.0f}/{b['trr']:<4.0f}")
        return b
    gcur = rollout(lambda c: step_current(m, c), z0) * sd + zmu; bcur = row("current(x0-resample)", gcur)
    # STEP 1: per-mode rescale current -> ref std
    gr = (gcur - gcur.mean(0)) * (ref.std(0) / (gcur.std(0) + 1e-9)) + gcur.mean(0); row("current+RESCALE(step1)", gr)
    # STEP 2a
    row("ancestral-tilde(2a)", rollout(lambda c: step_ancestral(m, c, False), z0) * sd + zmu)
    row("ancestral-beta-wide(2a)", rollout(lambda c: step_ancestral(m, c, True), z0) * sd + zmu)
    # STEP 2b: reverse-step count sweep (DDIM eta=1)
    for ns in (25, 50, 100, 250):
        row(f"ddim-eta1-T{ns}(2b)", rollout(lambda c, n=ns: step_ddim(m, c, n, 1.0), z0) * sd + zmu)
    row("ddim-eta0-T100(min var)", rollout(lambda c: step_ddim(m, c, 100, 0.0), z0) * sd + zmu)
print("\n  reference xcorr/amp/kurt are the targets (printed in step 3). varRatio=1 is the goal.")
print("  step1: if RESCALE makes marginals+basins pass while xcorr/kurt hold -> PURE SCALE -> calibration.")
