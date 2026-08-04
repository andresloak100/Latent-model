"""STEP 1 diagnostic: is the propagator's CONDITIONING PATH DEAD?

Variance 0.34x + autocorrelation 6x too short is one cause, not two: a conditional
model that IGNORES its conditioning turns each step into a near-independent draw from
an over-smoothed marginal -- shrinking variance and destroying temporal correlation
at once. Same failure class as arm F's frame-invariant latent, caught the same way.

TEST: condition on two VERY DIFFERENT x_t and compare the generated next-state
distributions -- the shift between them relative to the within-condition spread, and
the effective AR slope a_ddpm = d(mean out)/d(condition) vs the reference lag-1
autocorrelation a_ref the model SHOULD reproduce. a_ddpm ~ 0 while a_ref is large
=> conditioning is dead.

Parameterisation: the current Denoiser is eps-prediction of the ABSOLUTE next state
(x0 = z_{t+1}) with conditioning by a SINGLE input concat. Absolute-state + dead
conditioning => independent sampler (short autocorr, low variance) = what we observe.
"""
import math, numpy as np, torch, torch.nn as nn, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
BB = ["N", "CA", "C", "O"]; TEMP, R0 = "320", "0"
L, Tdiff, EPOCHS = 64, 100, 2000
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
    w, V = np.linalg.eigh(H)
    return V[:, 6:6 + K], w[6:6 + K]


class Denoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * L + 16, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, L))
    def temb(self, t, dev):
        f = torch.exp(torch.arange(8, device=dev) * (-math.log(1e4) / 8)); a = t[:, None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def forward(self, zt, cond, t):
        return self.net(torch.cat([zt, cond, self.temb(t, zt.device)], -1))


betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1 - betas, 0)


def sample_batch(m, cond):                                        # cond (B,L) -> one draw each (B,L)
    x = torch.randn(cond.shape[0], L)
    for k in reversed(range(Tdiff)):
        eps = m(x, cond, torch.full((cond.shape[0],), k)); x0 = (x - (1 - ac[k]).sqrt() * eps) / ac[k].sqrt()
        x = ac[k - 1].sqrt() * x0 + (1 - ac[k - 1]).sqrt() * torch.randn_like(x) if k > 0 else x0
    return x


print(f"[cond-diag] eps-prediction of ABSOLUTE next state, conditioning = single input concat")
print(f"  {'system':9s}{'a_ddpm':>8}{'a_ref':>8}{'ratio':>7}{'shift/spread(gen)':>18}{'(ideal)':>9}  verdict")
for dom in USE:
    with h5py.File(f"{DATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
        coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); T = d.shape[0]
    h = int(T * 0.8); mean = d[:h].mean(0)
    modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d - mean) @ B.T; zmu = Z[:h].mean(0); sd = np.sqrt(0.593 / np.clip(lam, 1e-8, None))
    Zn = ((Z - zmu) / sd).astype(np.float32)
    Znt = torch.tensor(Zn)
    m = Denoiser(L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    zc, zn1 = Znt[:h - 1], Znt[1:h]
    for ep in range(EPOCHS):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),))
        c, x0 = zc[i], zn1[i]; k = torch.randint(0, Tdiff, (len(i),)); noise = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None] * x0 + (1 - ac[k]).sqrt()[:, None] * noise
        opt.zero_grad(); ((m(xk, c, k) - noise) ** 2).mean().backward(); opt.step()
    m.eval()

    # reference lag-1 AR slope per mode (whitened): a_ref_i = <z_t z_{t+1}>/<z_t^2>
    zt_, zn_ = Zn[:h - 1], Zn[1:h]
    a_ref = (zt_ * zn_).mean(0) / ((zt_ ** 2).mean(0) + 1e-9)

    # regression a_ddpm: many conditions, one draw each -> slope of generated next on condition, per mode
    idx = np.random.choice(h - 1, min(200, h - 1), replace=False)
    with torch.no_grad():
        gen = sample_batch(m, torch.tensor(Zn[idx])).numpy()
    cnd = Zn[idx]
    a_ddpm = ((cnd - cnd.mean(0)) * (gen - gen.mean(0))).mean(0) / (((cnd - cnd.mean(0)) ** 2).mean(0) + 1e-9)

    # two-condition test: most-distant pair, M draws each
    D = ((Zn[:h, None] - Zn[None, :h]) ** 2).sum(-1); iA, iB = np.unravel_index(np.argmax(D), D.shape)
    zA = torch.tensor(Zn[iA])[None].repeat(300, 1); zB = torch.tensor(Zn[iB])[None].repeat(300, 1)
    with torch.no_grad():
        sA = sample_batch(m, zA).numpy(); sB = sample_batch(m, zB).numpy()
    shift = np.linalg.norm(sA.mean(0) - sB.mean(0))
    spread = np.sqrt(0.5 * (sA.var(0).sum() + sB.var(0).sum()))
    ideal = np.linalg.norm(a_ref * (Zn[iA] - Zn[iB]))            # what a perfect AR would shift
    r_gen, r_ideal = shift / (spread + 1e-9), ideal / (spread + 1e-9)
    dead = abs(a_ddpm).mean() < 0.3 * abs(a_ref).mean()
    print(f"  {dom:9s}{abs(a_ddpm).mean():>8.3f}{abs(a_ref).mean():>8.3f}"
          f"{abs(a_ddpm).mean()/max(abs(a_ref).mean(),1e-6):>7.2f}{r_gen:>18.2f}{r_ideal:>9.2f}  "
          f"{'CONDITIONING DEAD' if dead else 'conditioning live'}")
print("\n  read: a_ddpm << a_ref (ratio << 1) and shift/spread(gen) << ideal -> conditioning path is dead;")
print("        fix = inject conditioning at every layer (FiLM / cross-attn), re-run ensemble before redesign.")
