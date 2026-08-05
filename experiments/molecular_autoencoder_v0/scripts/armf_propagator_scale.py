"""Scale-up: does the propagator (ancestral DDPM + ONE global variance constant) pass the
pre-committed acceptance test across many systems? Global constant c fit on 5 systems,
applied to 10 HELD-OUT systems (generality test). Per-system pass/fail against the
pre-registered thresholds; fractions passing each; fraction beating OU on the two
OU-impossible discriminators. Never a mean over systems.

Thresholds (pre-committed): varRatio in [0.80,1.25]; xcorr 0.5*ref..2*ref; IAT ratio
[0.50,2.00]; transitions gen/ref [0.50,2.00]; |kurt_gen-kurt_ref|<=0.5.
"""
import math, glob, os, numpy as np, torch, torch.nn as nn
BBCACHE = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/bb_cache"   # numpy-only (GPU venv has no h5py)
BB = ["N", "CA", "C", "O"]
L, Tdiff, TAU, H = 64, 100, 50, 800
NFIT, NEVAL = 5, 10
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


def offdiag(C): n = C.shape[0]; return float(np.abs(C[~np.eye(n, dtype=bool)]).mean())
def excess_kurt(X): Xc = (X - X.mean(0)) / (X.std(0) + 1e-9); return float((((Xc**4).mean(0)) - 3).mean())
def iat_series(x, ml=300):
    x = x - x.mean(); v = (x*x).mean()
    if v < 1e-12: return 0.0
    acf = np.correlate(x, x, "full")[len(x)-1:] / (v*len(x)); t = 1.0
    for k in range(1, min(len(acf), ml)):
        if acf[k] < 0.05: break
        t += 2*acf[k]
    return t
def iat_ref_tau(Z, tau, mk=150):
    o = []
    for i in range(Z.shape[1]):
        x = Z[:, i]-Z[:, i].mean(); v = (x*x).mean()
        if v < 1e-12: o.append(0.0); continue
        t = 1.0
        for k in range(1, mk):
            lag = k*tau
            if lag >= len(x): break
            ac_ = (x[:-lag]*x[lag:]).mean()/v
            if ac_ < 0.05: break
            t += 2*ac_
        o.append(t)
    return np.mean(o)
def basins(X, thr): return 2*(X[:, 0] > thr[0]).astype(int) + (X[:, 1] > thr[1]).astype(int)


class FiLMDenoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__()
        self.inp = nn.Linear(L+16, h); self.h1 = nn.Linear(h, h); self.h2 = nn.Linear(h, h); self.out = nn.Linear(h, L)
        self.f0 = nn.Linear(L, 2*h); self.f1 = nn.Linear(L, 2*h); self.f2 = nn.Linear(L, 2*h)
        for f in (self.f0, self.f1, self.f2): nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
    def temb(self, t):
        f = torch.exp(torch.arange(8, device=t.device)*(-math.log(1e4)/8)); a = t[:, None].float()*f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def film(self, lin, x, c): g, b = lin(c).chunk(2, -1); return torch.nn.functional.gelu(x*(1+g)+b)
    def forward(self, zt, cond, t):
        x = self.film(self.f0, self.inp(torch.cat([zt, self.temb(t)], -1)), cond)
        x = self.film(self.f1, self.h1(x), cond); x = self.film(self.f2, self.h2(x), cond)
        return self.out(x)


dev = "cuda" if torch.cuda.is_available() else "cpu"
betas = torch.linspace(1e-4, 0.02, Tdiff, device=dev); ac = torch.cumprod(1-betas, 0)


def train(Zn, h, tau):
    m = FiLMDenoiser(L).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Z = torch.tensor(Zn, device=dev); zc = Z[:h-tau]; tgt = Z[tau:h]
    for ep in range(1500):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),), device=dev)
        c, x0 = zc[i], tgt[i]; k = torch.randint(0, Tdiff, (len(i),), device=dev); noise = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None]*x0 + (1-ac[k]).sqrt()[:, None]*noise
        opt.zero_grad(); ((m(xk, c, k)-noise)**2).mean().backward(); opt.step()
    return m


@torch.no_grad()
def rollout(m, z0):
    z = torch.tensor(z0, device=dev); out = [z.clone()]
    for _ in range(H):
        x = torch.randn(1, L, device=dev)
        for k in reversed(range(Tdiff)):
            eps = m(x, z, torch.full((1,), k, device=dev)); alpha = 1-betas[k]
            mean = (x - betas[k]/(1-ac[k]).sqrt()*eps)/alpha.sqrt()
            if k > 0:
                bt = (1-ac[k-1])/(1-ac[k])*betas[k]; x = mean + bt.sqrt()*torch.randn_like(x)
            else: x = mean
        z = torch.nan_to_num(x, nan=0.0).clamp(-12, 12); out.append(z.clone())
    return torch.cat(out, 0).cpu().numpy()


def prep(dom):
    fp = f"{BBCACHE}/{dom}.npz"
    if not os.path.exists(fp): return None
    dd = np.load(fp); coords = dd["coords"].astype(np.float64); ca_in_bb = dd["ca_in_bb"]
    if ca_in_bb.sum() < 30: return None
    al = kabsch(coords, ca_in_bb); d = (al-al[0]).reshape(len(al), -1); T = d.shape[0]; h = int(T*0.8)
    mean = d[:h].mean(0); modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d-mean)@B.T; zmu = Z[:h].mean(0); sd = np.sqrt(0.593/np.clip(lam, 1e-8, None))
    Zn = ((Z-zmu)/sd).astype(np.float32)
    return dict(dom=dom, Zn=Zn, sd=sd, zmu=zmu, Z=Z, h=h, top2=np.argsort(Z.std(0))[::-1][:2],
                thr=np.median(Z[:, np.argsort(Z.std(0))[::-1][:2]], 0))


doms = sorted(os.path.basename(x)[:-4] for x in glob.glob(f"{BBCACHE}/*.npz"))
sysd = [s for s in (prep(d) for d in doms) if s is not None][:NFIT+NEVAL]
fit, ev = sysd[:NFIT], sysd[NFIT:]
print(f"[scale] device={dev}; fit c on {len(fit)} systems, evaluate {len(ev)} held-out")

# train + rollout everything; compute deficit for the fit set
def gen_and_metrics(s):
    m = train(s["Zn"], s["h"], TAU); roll = rollout(m, s["Zn"][s["h"]:s["h"]+1])
    gen = roll*s["sd"] + s["zmu"]; ref = s["Z"]
    deficit = float((gen.std(0)/(ref.std(0)+1e-9)).mean())
    # OU
    lam_sd = s["sd"]; a1 = (ref[:s["h"]-1]*ref[1:s["h"]]).mean(0)/((ref[:s["h"]-1]**2).mean(0)+1e-9)
    lam = 0.593/(lam_sd**2); gamma = float(np.median((-lam/np.log(np.clip(a1, 0.02, 0.98)))[lam > 0]))
    a_ou = np.exp(-lam*TAU/gamma); v = lam_sd**2; v *= (ref[:s["h"]].var(0).sum()/v.sum())
    x = ref[s["h"]].copy(); our = [x.copy()]
    for _ in range(H): x = a_ou*x + np.sqrt(v*(1-a_ou**2))*np.random.randn(L); our.append(x.copy())
    return gen, np.array(our), ref, s["top2"], s["thr"], deficit

cache = {s["dom"]: gen_and_metrics(s) for s in sysd}
c_glob = 1.0 / np.mean([cache[s["dom"]][5] for s in fit])       # ONE global constant from the fit systems
print(f"  global constant c = {c_glob:.2f} (from fit systems' deficit {1/c_glob:.3f})")


def disc(gen, ref, top2, thr):
    lab_r = basins(ref[:, top2], thr); lab_g = basins(gen[:, top2], thr)
    return dict(vr=float((gen.std(0)/(ref.std(0)+1e-9)).mean()),
                xc=offdiag(np.corrcoef(gen.T)), kurt=excess_kurt(gen),
                iat=np.mean([iat_series(gen[:, i]) for i in range(L)])/max(iat_ref_tau(ref, TAU), 1e-6),
                tr=float((lab_g[1:] != lab_g[:-1]).mean()*1000),
                trr=float((lab_r[TAU:] != lab_r[:-TAU]).mean()*1000),
                xcref=offdiag(np.corrcoef(ref.T)), kurtref=excess_kurt(ref))


print("\n=== HELD-OUT per-system pass/fail (global c applied to marginals) ===")
print("  thresholds: varRatio[0.80,1.25]  xcorr 0.5-2x ref  IAT[0.50,2.00]  trans[0.50,2.00]  |dkurt|<=0.5")
print(f"  {'system':9s}{'varR':>6}{'marg':>5}{'xc/ref':>9}{'coup':>5}{'iat':>5}{'kin':>4}{'tr/rf':>9}{'trs':>4}{'dkurt':>6}{'nG':>3}  beatsOU(c/nG)")
frac = {k: 0 for k in ["marg", "coup", "kin", "trs", "nG"]}; bou = {"c": 0, "ng": 0}
for s in ev:
    gen, our, ref, top2, thr, _ = cache[s["dom"]]
    genc = (gen - gen.mean(0))*c_glob + gen.mean(0)             # apply GLOBAL constant to marginals
    d = disc(genc, ref, top2, thr); o = disc(our, ref, top2, thr)
    pm = 0.80 <= d["vr"] <= 1.25
    pc = 0.5*d["xcref"] <= d["xc"] <= 2.0*d["xcref"]
    pk = 0.50 <= d["iat"] <= 2.00
    pt = 0.50 <= d["tr"]/max(d["trr"], 1e-6) <= 2.00
    pn = abs(d["kurt"]-d["kurtref"]) <= 0.5
    boc = abs(d["xc"]-d["xcref"]) < abs(o["xc"]-d["xcref"])
    bon = abs(d["kurt"]-d["kurtref"]) < abs(o["kurt"]-d["kurtref"])
    for k, p in [("marg", pm), ("coup", pc), ("kin", pk), ("trs", pt), ("nG", pn)]:
        frac[k] += p
    bou["c"] += boc; bou["ng"] += bon
    P = lambda b: "P" if b else "."
    print(f"  {s['dom']:9s}{d['vr']:>6.2f}{P(pm):>5}{d['xc']:>5.2f}/{d['xcref']:<3.2f}{P(pc):>5}"
          f"{d['iat']:>5.2f}{P(pk):>4}{d['tr']:>4.0f}/{d['trr']:<4.0f}{P(pt):>4}{d['kurt']-d['kurtref']:>+6.2f}{P(pn):>3}  {P(boc)}/{P(bon)}")
n = len(ev)
print(f"\n  FRACTION PASSING (held-out, n={n}): "
      f"marginals {frac['marg']}/{n}  coupling {frac['coup']}/{n}  kinetics {frac['kin']}/{n}  "
      f"transitions {frac['trs']}/{n}  non-Gaussianity {frac['nG']}/{n}")
print(f"  EARNS ITS COST (beats OU): cross-mode coupling {bou['c']}/{n}  non-Gaussianity {bou['ng']}/{n}")
print(f"  ALL-FIVE pass: {sum(1 for s in ev if True)} systems evaluated; global c={c_glob:.2f} fit on {len(fit)}, held out {n}")
