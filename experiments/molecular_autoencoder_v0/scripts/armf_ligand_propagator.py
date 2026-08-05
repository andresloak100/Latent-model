"""Does the propagator STACK transfer from proteins to LIGANDS? (objective-1 end-to-end)

Full stack on a non-protein molecule: banked graph codec -> per-ligand torsional-mode latents ->
the EXISTING propagator machinery UNCHANGED (FiLMDenoiser DDPM, sqrt(kT/lambda) whitening, ONE
global variance constant, the pre-committed acceptance test + thresholds). Reports (1) whether the
protein-fit constant c=2.87 transfers or the ligand deficit refits it -- itself a generality
result; (2) fractions passing each discriminator; (3) beats-OU on the two OU-impossible ones.

DATA REALITY (measured, armf_ligand_prop_precheck): MISATO ligand latents are decorrelated frame-
to-frame (IAT~1, lag-1 median 0.08). So the DYNAMICS discriminators (kinetics IAT, transitions)
are near-trivial here (ref IAT~1); the MEANINGFUL transfer test is the EQUILIBRIUM-distribution
discriminators (marginals, cross-mode coupling, non-Gaussianity). TAU=50 (long-protein lag) is
inapplicable at T=100; use TAU=1 (data resolution). Everything else unchanged.

Thresholds (pre-committed, verbatim): varRatio[0.80,1.25]; xcorr 0.5*ref..2*ref; IAT[0.50,2.00];
transitions gen/ref [0.50,2.00]; |kurt_gen-kurt_ref|<=0.5."""
import math, pickle, numpy as np, torch, torch.nn as nn
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"; CACHE = f"{WR}/graph_codec_cache.pkl"
L, Tdiff, TAU, H = 8, 100, 1, 500; C_PROTEIN = 2.87
torch.manual_seed(0); np.random.seed(0)
data = pickle.load(open(CACHE, "rb")); data = [d for d in data if d["ntors"] >= L]


class GraphCodec(nn.Module):                                     # verbatim from armf_graph_codec.py
    def __init__(self, fin, Hd=48, layers=3):
        super().__init__(); self.embed = nn.Linear(fin, Hd)
        self.self_w = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(layers)])
        self.nbr_w = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(4 * Hd + 2, Hd), nn.GELU(), nn.Linear(Hd, Hd), nn.GELU(), nn.Linear(Hd, 1))
    def forward(self, xat, edges, n, tidx, tsc):
        h = self.embed(xat); src, dst = edges[0], edges[1]
        deg = torch.zeros(n, dtype=h.dtype).index_add_(0, dst, torch.ones(len(dst), dtype=h.dtype)).clamp(min=1)[:, None]
        for sw, nw in zip(self.self_w, self.nbr_w):
            agg = torch.zeros_like(h).index_add_(0, dst, h[src]) / deg
            h = torch.nn.functional.gelu(sw(h) + nw(agg))
        e = torch.cat([h[tidx[:, 0]], h[tidx[:, 1]], h[tidx[:, 2]], h[tidx[:, 3]], tsc], -1)
        return self.head(e).squeeze(-1)


def modes_eig(logk, G, Lm):                                     # softest-L modes AND eigenvalues (torsional stiffness)
    D = G.shape[0]; k = torch.nn.functional.softplus(logk) + 1e-3
    Lc = torch.linalg.cholesky(G + 1e-6 * torch.eye(D, dtype=G.dtype)); Linv = torch.linalg.inv(Lc)
    A = 0.5 * (Linv @ torch.diag(k) @ Linv.T + (Linv @ torch.diag(k) @ Linv.T).T)
    w, U = torch.linalg.eigh(A); return (Linv.T @ U[:, :Lm]), w[:Lm]


# --- retrain the graph codec (same recipe) then extract per-ligand latents Z + eigenvalues ---
for d in data:
    d["t_xat"] = torch.tensor(d["xat"]).double(); d["t_edges"] = torch.tensor(d["edges"])
    d["t_tidx"] = torch.tensor(d["tidx"]); d["t_tsc"] = torch.tensor(d["tsc"]).double(); d["t_G"] = torch.tensor(d["G"].astype(np.float64))
tr_idx = [i for i in range(len(data)) if i % 4 != 0]
model = GraphCodec(data[0]["xat"].shape[1]).double(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
print(f"[ligand-propagator] {len(data)} ligands (ntors>={L}); retraining codec then running the stack")
for ep in range(500):
    perm = np.random.permutation(tr_idx); opt.zero_grad(); loss = 0.0; nb = 0
    for i in perm[:32]:
        d = data[i]; logk = model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"]); dt = torch.tensor(d["dphi"].astype(np.float64))
        for Ltr in (4, 8):
            M, _ = modes_eig(logk, d["t_G"], Ltr); c = dt @ (d["t_G"] @ M); loss = loss + ((c @ M.T - dt) ** 2).mean()
        nb += 1
    (loss / nb).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

sysd = []
for d in data:
    with torch.no_grad():
        M, lam = modes_eig(model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"]), d["t_G"], L)
    M = M.numpy(); lam = np.clip(lam.numpy(), 1e-6, None)
    dphi = d["dphi"].astype(np.float64); Z = dphi @ (d["G"].astype(np.float64) @ M)   # (T, L) codec latents
    T = len(Z); h = int(T * 0.8); sd = np.sqrt(0.593 / lam); zmu = Z[:h].mean(0)
    if np.any(~np.isfinite(sd)) or sd.min() < 1e-8: continue
    Zn = ((Z - zmu) / sd).astype(np.float32); top2 = np.argsort(Z.std(0))[::-1][:2]
    sysd.append(dict(dom=d["dom"], Zn=Zn, sd=sd, zmu=zmu, Z=Z, h=h, top2=top2, thr=np.median(Z[:, top2], 0)))
NFIT = 5; sysd = sysd[:NFIT + 15]; fit, ev = sysd[:NFIT], sysd[NFIT:]
print(f"  {len(sysd)} ligands with valid latents: fit c on {len(fit)}, evaluate {len(ev)} held-out; TAU={TAU}")

# --- propagator machinery: VERBATIM from armf_propagator_scale.py ---
betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1 - betas, 0)


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
        x = self.film(self.f1, self.h1(x), cond); x = self.film(self.f2, self.h2(x), cond); return self.out(x)


def train(Zn, h, tau):
    m = FiLMDenoiser(L); o = torch.optim.Adam(m.parameters(), lr=1e-3); Z = torch.tensor(Zn); zc = Z[:h-tau]; tgt = Z[tau:h]
    for _ in range(1200):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),)); c, x0 = zc[i], tgt[i]
        k = torch.randint(0, Tdiff, (len(i),)); noise = torch.randn_like(x0); xk = ac[k].sqrt()[:, None]*x0 + (1-ac[k]).sqrt()[:, None]*noise
        o.zero_grad(); ((m(xk, c, k)-noise)**2).mean().backward(); o.step()
    return m


@torch.no_grad()
def rollout(m, z0):
    z = torch.tensor(z0); out = [z.clone()]
    for _ in range(H):
        x = torch.randn(1, L)
        for k in reversed(range(Tdiff)):
            eps = m(x, z, torch.full((1,), k)); alpha = 1-betas[k]; mean = (x - betas[k]/(1-ac[k]).sqrt()*eps)/alpha.sqrt()
            x = mean + ((1-ac[k-1])/(1-ac[k])*betas[k]).sqrt()*torch.randn_like(x) if k > 0 else mean
        z = torch.nan_to_num(x, nan=0.0).clamp(-12, 12); out.append(z.clone())
    return torch.cat(out, 0).numpy()


def offdiag(C): n = C.shape[0]; return float(np.abs(C[~np.eye(n, dtype=bool)]).mean())
def excess_kurt(X): Xc = (X-X.mean(0))/(X.std(0)+1e-9); return float((((Xc**4).mean(0))-3).mean())
def iat_series(x, ml=200):
    x = x-x.mean(); v = (x*x).mean()
    if v < 1e-12: return 0.0
    acf = np.correlate(x, x, "full")[len(x)-1:]/(v*len(x)); t = 1.0
    for k in range(1, min(len(acf), ml)):
        if acf[k] < 0.05: break
        t += 2*acf[k]
    return t
def basins(X, thr): return 2*(X[:, 0] > thr[0]).astype(int) + (X[:, 1] > thr[1]).astype(int)


def gen_and_metrics(s):
    m = train(s["Zn"], s["h"], TAU); gen = rollout(m, s["Zn"][s["h"]:s["h"]+1])*s["sd"] + s["zmu"]; ref = s["Z"]
    dpm = gen.std(0)/(ref.std(0)+1e-9); deficit = float(dpm.mean())
    a1 = (ref[:s["h"]-1]*ref[1:s["h"]]).mean(0)/((ref[:s["h"]-1]**2).mean(0)+1e-9)
    lam = 0.593/(s["sd"]**2); gamma = float(np.median((-lam/np.log(np.clip(a1, 0.02, 0.98)))[lam > 0]))
    a_ou = np.exp(-lam*TAU/gamma); v = s["sd"]**2; v *= (ref[:s["h"]].var(0).sum()/v.sum()); x = ref[s["h"]].copy(); our = [x.copy()]
    for _ in range(H): x = a_ou*x + np.sqrt(v*(1-a_ou**2))*np.random.randn(L); our.append(x.copy())
    return gen, np.array(our), ref, s["top2"], s["thr"], deficit, float(dpm.std()/(dpm.mean()+1e-9))


cache = {s["dom"]: gen_and_metrics(s) for s in sysd}
c_lig = 1.0 / np.mean([cache[s["dom"]][5] for s in fit]); cv = np.mean([cache[s["dom"]][6] for s in fit])
print(f"\n  c=2.87 TRANSFER: ligand-fit c = {c_lig:.2f} (protein c=2.87). per-mode deficit CV {cv:.2f} "
      f"(low => one global constant works; the codec's latent sd is uniformly off, not per-mode)")


def disc(gen, ref, top2, thr):
    lr = basins(ref[:, top2], thr); lg = basins(gen[:, top2], thr)
    return dict(vr=float((gen.std(0)/(ref.std(0)+1e-9)).mean()), xc=offdiag(np.corrcoef(gen.T)), kurt=excess_kurt(gen),
                iat=np.mean([iat_series(gen[:, i]) for i in range(L)])/max(np.mean([iat_series(ref[:, i]) for i in range(L)]), 1e-6),
                tr=float((lg[1:] != lg[:-1]).mean()*1000), trr=float((lr[1:] != lr[:-1]).mean()*1000),
                xcref=offdiag(np.corrcoef(ref.T)), kurtref=excess_kurt(ref))


for cname, cval in [("protein c=2.87 (TRANSFER)", C_PROTEIN), ("ligand-fit c", c_lig)]:
    print(f"\n=== HELD-OUT per-ligand pass/fail -- {cname} ===")
    print(f"  {'lig':9s}{'varR':>6}{'mrg':>4}{'xc/ref':>10}{'cp':>3}{'iat':>6}{'kn':>3}{'tr/rf':>10}{'ts':>3}{'dkurt':>7}{'nG':>3}  OU(c/nG)  [dyn*]")
    frac = {k: 0 for k in ["marg", "coup", "kin", "trs", "nG"]}; bou = {"c": 0, "ng": 0}
    for s in ev:
        gen, our, ref, top2, thr, _, _ = cache[s["dom"]]; genc = (gen-gen.mean(0))*cval + gen.mean(0)
        dd = disc(genc, ref, top2, thr); oo = disc(our, ref, top2, thr)
        pm = 0.80 <= dd["vr"] <= 1.25; pc = 0.5*dd["xcref"] <= dd["xc"] <= 2.0*dd["xcref"]
        pk = 0.50 <= dd["iat"] <= 2.00; pt = 0.50 <= dd["tr"]/max(dd["trr"], 1e-6) <= 2.00; pn = abs(dd["kurt"]-dd["kurtref"]) <= 0.5
        boc = abs(dd["xc"]-dd["xcref"]) < abs(oo["xc"]-dd["xcref"]); bon = abs(dd["kurt"]-dd["kurtref"]) < abs(oo["kurt"]-dd["kurtref"])
        for k, p in [("marg", pm), ("coup", pc), ("kin", pk), ("trs", pt), ("nG", pn)]: frac[k] += p
        bou["c"] += boc; bou["ng"] += bon; P = lambda b: "P" if b else "."
        print(f"  {s['dom']:9s}{dd['vr']:>6.2f}{P(pm):>4}{dd['xc']:>5.2f}/{dd['xcref']:<4.2f}{P(pc):>3}{dd['iat']:>6.2f}{P(pk):>3}"
              f"{dd['tr']:>5.0f}/{dd['trr']:<4.0f}{P(pt):>3}{dd['kurt']-dd['kurtref']:>+7.2f}{P(pn):>3}   {P(boc)}/{P(bon)}")
    n = len(ev)
    print(f"  FRACTION PASSING (n={n}): marginals {frac['marg']}/{n}  coupling {frac['coup']}/{n}  "
          f"[dyn*] kinetics {frac['kin']}/{n}  transitions {frac['trs']}/{n}  non-Gaussianity {frac['nG']}/{n}")
    print(f"  EARNS ITS COST (beats OU): coupling {bou['c']}/{n}  non-Gaussianity {bou['ng']}/{n}")
print("\n  [dyn*] kinetics+transitions are near-trivial here: MISATO ligand latents are decorrelated")
print("         (IAT~1), so these test decorrelated-sample matching, not learned dynamics. The")
print("         meaningful transfer test is marginals + coupling + non-Gaussianity (equilibrium).")
