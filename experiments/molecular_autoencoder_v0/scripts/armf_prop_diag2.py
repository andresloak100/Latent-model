"""Two cheap closing diagnostics on the propagator, then STOP propagator work.
 1. lambda-conditioning read: per-mode deficit (gen/ref std) and coupling vs mode index
    (= vs lambda; modes are lambda-ascending).
 2. lambda_min/kurtosis check: is the heavy-tail kurtosis blow-up concentrated in the
    SOFTEST modes (lowest lambda)? Report per-mode kurtosis of the generated rollout,
    softest-3 vs the rest, for a heavy-tail system (2lklA01) and a clean one (3a5zD02).
"""
import math, numpy as np, torch, torch.nn as nn
BBCACHE = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/bb_cache"
L, Tdiff, TAU, H = 64, 100, 50, 800
USE = ["2lklA01", "3a5zD02", "2m1xA00"]
torch.manual_seed(0); np.random.seed(0)


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


def kurt_permode(X): Xc = (X - X.mean(0)) / (X.std(0) + 1e-9); return ((Xc ** 4).mean(0)) - 3


class FiLMDenoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__()
        self.inp = nn.Linear(L+16, h); self.h1 = nn.Linear(h, h); self.h2 = nn.Linear(h, h); self.out = nn.Linear(h, L)
        self.f0 = nn.Linear(L, 2*h); self.f1 = nn.Linear(L, 2*h); self.f2 = nn.Linear(L, 2*h)
        for f in (self.f0, self.f1, self.f2): nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
    def temb(self, t):
        f = torch.exp(torch.arange(8)*(-math.log(1e4)/8)); a = t[:, None].float()*f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def film(self, lin, x, c): g, b = lin(c).chunk(2, -1); return torch.nn.functional.gelu(x*(1+g)+b)
    def forward(self, zt, cond, t):
        x = self.film(self.f0, self.inp(torch.cat([zt, self.temb(t)], -1)), cond)
        x = self.film(self.f1, self.h1(x), cond); x = self.film(self.f2, self.h2(x), cond)
        return self.out(x)


betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1 - betas, 0)


def train(Zn, h):
    m = FiLMDenoiser(L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Z = torch.tensor(Zn); zc = Z[:h-TAU]; tgt = Z[TAU:h]
    for ep in range(1500):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),))
        c, x0 = zc[i], tgt[i]; k = torch.randint(0, Tdiff, (len(i),)); noise = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None]*x0 + (1-ac[k]).sqrt()[:, None]*noise
        opt.zero_grad(); ((m(xk, c, k)-noise)**2).mean().backward(); opt.step()
    return m


@torch.no_grad()
def rollout(m, z0):
    z = torch.tensor(z0); out = [z.clone()]
    for _ in range(H):
        x = torch.randn(1, L)
        for k in reversed(range(Tdiff)):
            eps = m(x, z, torch.full((1,), k)); alpha = 1-betas[k]
            mean = (x - betas[k]/(1-ac[k]).sqrt()*eps)/alpha.sqrt()
            x = mean + ((1-ac[k-1])/(1-ac[k])*betas[k]).sqrt()*torch.randn_like(x) if k > 0 else mean
        z = torch.nan_to_num(x, nan=0.0).clamp(-12, 12); out.append(z.clone())
    return torch.cat(out, 0).numpy()


print("[prop-diag2] lambda-conditioning + lambda_min/kurtosis (modes lambda-ascending: idx0 = softest)")
for dom in USE:
    dd = np.load(f"{BBCACHE}/{dom}.npz"); coords = dd["coords"].astype(np.float64); ca = dd["ca_in_bb"]
    al = kabsch(coords, ca); d = (al-al[0]).reshape(len(al), -1); h = int(len(d)*0.8)
    mean = d[:h].mean(0); modes, lam = anm(al[0], L); lam = lam[:L]
    Z = (d-mean)@modes[:, :L]; zmu = Z[:h].mean(0); sd = np.sqrt(0.593/np.clip(lam, 1e-8, None))
    Zn = ((Z-zmu)/sd).astype(np.float32)
    gen = rollout(train(Zn, h), Zn[h:h+1])*sd + zmu; ref = Z
    th = L//3
    dfc = gen.std(0)/(ref.std(0)+1e-9)                        # per-mode deficit
    kg = kurt_permode(gen); kr = kurt_permode(ref)            # per-mode kurtosis gen/ref
    print(f"\n=== {dom} ===")
    print(f"  lambda-conditioning (deficit gen/ref by mode-index): soft {dfc[:th].mean():.2f} / mid {dfc[th:2*th].mean():.2f} / stiff {dfc[2*th:].mean():.2f}")
    print(f"  kurtosis gen  by mode-index: soft {kg[:th].mean():.1f} / mid {kg[th:2*th].mean():.1f} / stiff {kg[2*th:].mean():.1f}   (ref soft {kr[:th].mean():.1f})")
    print(f"  softest-3 modes kurtosis gen: {np.round(kg[:3],1)}   max-kurt mode index: {int(np.argmax(kg))} (kurt {kg.max():.1f}); is it in softest third? {int(np.argmax(kg)) < th}")
print("\n  read: heavy-tail (high kurt) concentrated in softest modes -> tails are soft-mode over-excursions;")
print("        spread across modes -> generic sampler heavy tails. Either way: propagator work STOPS after this.")
