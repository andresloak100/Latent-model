"""ENSEMBLE VALIDATION: did the propagator learn DYNAMICS, or bounded noise?

"0% out of range" is necessary but not sufficient -- a model that emits the training
mean every step also scores 0%, so does one with far too little variance. Compare the
GENERATED rollout to the REFERENCE MD trajectory, in ANM-mode coefficient space, with
sqrt(kT/lambda) whitening, on held-out (frozen-7) systems:
  1. per-mode marginal std ratio (generated/reference) -- variance-collapse test
  2. per-mode integrated autocorrelation time, both -- the KINETIC test (right
     marginals + wrong decay = right distribution, wrong dynamics)
  3. 2D free energy on the top-2 (highest-variance) modes -- missing basins / filled
     barriers, visible nowhere else. JS divergence + basin coverage; heatmaps saved.

Pre-registered read: std ratios ~1 AND autocorr times within ~2x -> learned dynamics.
Ratios well under 1 -> narrow blob, the in-range result was vacuous. Marginals right
but autocorr wrong -> learned the distribution, not the kinetics -- say so.
"""
import argparse, glob, math, os, numpy as np, torch, torch.nn as nn, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"
OUT = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/ensemble_validate"
BB = ["N", "CA", "C", "O"]
TEMP, R0 = "320", "0"
_ap = argparse.ArgumentParser()
_ap.add_argument("--epochs", type=int, default=2000); _ap.add_argument("--horizon", type=int, default=2000)
_ap.add_argument("--smoke", action="store_true")
_a = _ap.parse_args()
L, Tdiff, EPOCHS, HORIZON = 64, 100, _a.epochs, _a.horizon
USE = ["3a5zD02"] if _a.smoke else ["3a5zD02", "3jvvA01", "3a9lA00"]   # frozen-7 subset spanning n
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    a = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in a])


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pcen = P[mask].mean(0); Pc = P[mask] - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pcen) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
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
        f = torch.exp(torch.arange(8, device=dev) * (-math.log(1e4) / 8))
        a = t[:, None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def forward(self, zt, cond, t):
        return self.net(torch.cat([zt, cond, self.temb(t, zt.device)], -1))


def iat(x, cutoff=0.05, maxlag=400):                              # integrated autocorrelation time
    x = x - x.mean(); v = (x * x).mean()
    if v < 1e-12: return 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (v * len(x))
    t = 1.0
    for k in range(1, min(len(ac), maxlag)):
        if ac[k] < cutoff: break
        t += 2 * ac[k]
    return t


def js_2d(gen, ref, bins=40):                                    # Jensen-Shannon divergence of 2D histograms
    lo = np.minimum(gen.min(0), ref.min(0)); hi = np.maximum(gen.max(0), ref.max(0))
    edges = [np.linspace(lo[i], hi[i], bins + 1) for i in range(2)]
    Pg, _, _ = np.histogram2d(gen[:, 0], gen[:, 1], bins=edges)
    Pr, _, _ = np.histogram2d(ref[:, 0], ref[:, 1], bins=edges)
    Pg = Pg / Pg.sum() + 1e-12; Pr = Pr / Pr.sum() + 1e-12; M = 0.5 * (Pg + Pr)
    kl = lambda a, b: (a * np.log(a / b)).sum()
    js = 0.5 * kl(Pg, M) + 0.5 * kl(Pr, M)
    ref_occ = Pr > (1.5e-12); cover = ((Pg > 1.5e-12) & ref_occ).sum() / ref_occ.sum()   # basins hit
    spurious = ((Pg > 1e-4) & ~ref_occ).sum() / max(1, (Pg > 1e-4).sum())               # filled barriers
    return float(js), float(cover), float(spurious), (-np.log(Pg)).T, (-np.log(Pr)).T, edges


betas = torch.linspace(1e-4, 0.02, Tdiff); ac_ = torch.cumprod(1 - betas, 0)
frozen = [d for d in open(FROZEN).read().split() if d]
use = [d for d in USE if d in frozen]
print(f"[ensemble] ANM codec + sqrt(kT/lambda) whitening, L={L}, horizon={HORIZON}, systems {use}")
print(f"  {'system':9s}{'nCA':>5}{'stdRatio':>9}{'IATgen':>8}{'IATref':>8}{'IATrat':>8}{'JS':>7}{'cover':>7}{'spur':>7}")
agg = []
for dom in use:
    with h5py.File(f"{DATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
        coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); T = d.shape[0]
    h = int(T * 0.8); mean = d[:h].mean(0)
    modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d - mean) @ B.T                                          # (T, L) reference coefficients (raw)
    zmu = Z[:h].mean(0); sd = np.sqrt(0.593 / np.clip(lam, 1e-8, None))   # ANM-predicted whitening
    Zn = torch.tensor((Z - zmu) / sd, dtype=torch.float32)

    m = Denoiser(L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    zc, zn1 = Zn[:h - 1], Zn[1:h]
    for ep in range(EPOCHS):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),))
        c, x0 = zc[i], zn1[i]; k = torch.randint(0, Tdiff, (len(i),)); noise = torch.randn_like(x0)
        xk = ac_[k].sqrt()[:, None] * x0 + (1 - ac_[k]).sqrt()[:, None] * noise
        opt.zero_grad(); ((m(xk, c, k) - noise) ** 2).mean().backward(); opt.step()

    m.eval()
    @torch.no_grad()
    def step(cond):
        x = torch.randn(1, L)
        for k in reversed(range(Tdiff)):
            eps = m(x, cond, torch.tensor([k])); x0 = (x - (1 - ac_[k]).sqrt() * eps) / ac_[k].sqrt()
            x = ac_[k - 1].sqrt() * x0 + (1 - ac_[k - 1]).sqrt() * torch.randn_like(x) if k > 0 else x0
        return x
    zt = Zn[h:h + 1]; roll = [zt]
    with torch.no_grad():
        for _ in range(HORIZON):
            zt = step(zt); roll.append(zt)
    gen = torch.cat(roll, 0).numpy() * sd + zmu                   # (H+1, L) generated, raw coeffs
    ref = Z                                                       # (T, L) reference, raw coeffs

    sr = gen.std(0) / (ref.std(0) + 1e-9)                        # 1. marginal std ratio
    ig = np.array([iat(gen[:, i]) for i in range(L)]); ir = np.array([iat(ref[:, i]) for i in range(L)])  # 2. IAT
    top2 = np.argsort(ref.std(0))[::-1][:2]                       # 3. 2D FE on top-2 modes
    js, cover, spur, fg, fr, edges = js_2d(gen[:, top2], ref[:, top2])
    np.savez(f"{OUT}/{dom}.npz", gen_logP=fg, ref_logP=fr, edges=np.array(edges, dtype=object),
             std_ratio=sr, iat_gen=ig, iat_ref=ir, top2=top2)
    agg.append((sr.mean(), ig.mean(), ir.mean()))
    print(f"  {dom:9s}{ca.sum():>5}{sr.mean():>9.2f}{ig.mean():>8.1f}{ir.mean():>8.1f}"
          f"{ig.mean()/max(ir.mean(),1e-6):>8.2f}{js:>7.3f}{cover:>7.0%}{spur:>7.0%}")

a = np.array(agg); srm, igm, irm = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
print(f"\n  MEAN over systems: stdRatio {srm:.2f}  IAT gen/ref {igm:.1f}/{irm:.1f} (ratio {igm/max(irm,1e-6):.2f})")
iatr = igm / max(irm, 1e-6)
verdict = ("std ratio ~1 AND IAT within ~2x -> LEARNED DYNAMICS" if 0.7 < srm < 1.4 and 0.5 < iatr < 2.0 else
           "std ratio << 1 -> VARIANCE COLLAPSE; the in-range result was vacuous" if srm < 0.7 else
           "std ratio >> 1 -> OVER-DISPERSED; the rollout is noisier than the reference" if srm > 1.4 else
           "marginals ~ok but IAT off -> learned the DISTRIBUTION, not the KINETICS")
print(f"  VERDICT: {verdict}")
print(f"  (-logP heatmaps saved to {OUT}/*.npz)")
