"""Stage B: corpus-scale complex propagator -- the first corpus-scale training run in the project.

The joint 16-dim propagator is trained ACROSS many complexes (pooled pairs), not one complex's ~79
pairs. Learning curve over 8/50/200/max training complexes. PRIMARY QUESTION (pre-registered): does
the joint model recover MARGINALS at scale while keeping its cross-block coupling advantage?
  joint reaches independent's marginal pass rate AND keeps coupling capture -> joint wins outright,
      the tie was a corpus-use artifact (demonstrated, not asserted).
  joint still loses marginals at 10-50x the data -> NOT a data limit; "independent for marginals,
      joint when coupling matters", documented, stop.
Same whitening, calibration, acceptance test, pre-committed thresholds, per-system then fractions.

Byproducts: (1) is c PREDICTABLE from structure (regress per-complex c vs mean-lambda / protein size
/ ligand size / ratio)? If so the last per-system fitted quantity becomes general. (2) first
corpus-scale run -- overfitting was the recurring diagnosis at 5-20 systems.

Latents are decorrelated (measured) -> evaluation samples via batched ONE-STEP conditioning, not
sequential rollout (equivalent here, far cheaper). Does NOT reopen the codec (closed on the
B-factor oracle ceiling; stopping rule stands)."""
import math, pickle, numpy as np, torch, torch.nn as nn
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"; CACHE = f"{WR}/complex_scale_cache.pkl"
Lp = Ld = 8; L = Lp + Ld; Tdiff, TAU = 100, 1; SCALES = [8, 50, 200]
torch.manual_seed(0); np.random.seed(0)
data = pickle.load(open(CACHE, "rb"))
print(f"[complex-scale] {len(data)} complexes in cache", flush=True)


class GraphCodec(nn.Module):
    def __init__(self, fin, Hd=48, layers=3):
        super().__init__(); self.embed = nn.Linear(fin, Hd)
        self.self_w = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(layers)]); self.nbr_w = nn.ModuleList([nn.Linear(Hd, Hd) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(4*Hd+2, Hd), nn.GELU(), nn.Linear(Hd, Hd), nn.GELU(), nn.Linear(Hd, 1))
    def forward(self, xat, edges, n, tidx, tsc):
        h = self.embed(xat); src, dst = edges[0], edges[1]
        deg = torch.zeros(n, dtype=h.dtype).index_add_(0, dst, torch.ones(len(dst), dtype=h.dtype)).clamp(min=1)[:, None]
        for sw, nw in zip(self.self_w, self.nbr_w):
            h = torch.nn.functional.gelu(sw(h) + nw(torch.zeros_like(h).index_add_(0, dst, h[src]) / deg))
        return self.head(torch.cat([h[tidx[:, 0]], h[tidx[:, 1]], h[tidx[:, 2]], h[tidx[:, 3]], tsc], -1)).squeeze(-1)


def modes_eig(logk, G, Lm):
    D = G.shape[0]; k = torch.nn.functional.softplus(logk) + 1e-3
    Lc = torch.linalg.cholesky(G + 1e-6*torch.eye(D, dtype=G.dtype)); Linv = torch.linalg.inv(Lc)
    A = Linv @ torch.diag(k) @ Linv.T; w, U = torch.linalg.eigh(0.5*(A+A.T)); return (Linv.T @ U[:, :Lm]), w[:Lm]


for d in data:
    d["t_xat"] = torch.tensor(d["xat"]).double(); d["t_edges"] = torch.tensor(d["edges"]); d["t_tidx"] = torch.tensor(d["tidx"])
    d["t_tsc"] = torch.tensor(d["tsc"]).double(); d["t_G"] = torch.tensor(d["G"].astype(np.float64))
he_idx = list(range(0, len(data), 6)); tr_idx = [i for i in range(len(data)) if i not in he_idx]
print(f"  {len(tr_idx)} train / {len(he_idx)} held-out complexes", flush=True)

# --- codec (retrain on training ligands; using the codec at scale, not re-litigating it) ---
model = GraphCodec(data[0]["xat"].shape[1]).double(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400)
print("  retraining ligand codec ...", flush=True)
for ep in range(400):
    perm = np.random.permutation(tr_idx); opt.zero_grad(); loss = 0.0
    for i in perm[:32]:
        d = data[i]; logk = model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"]); dt = torch.tensor(d["dphi"].astype(np.float64))
        for Ltr in (4, 8): Mx, _ = modes_eig(logk, d["t_G"], Ltr); loss = loss + (((dt@(d["t_G"]@Mx))@Mx.T-dt)**2).mean()
    (loss/32).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()

# --- per-complex whitened joint latent ---
for d in data:
    with torch.no_grad():
        Ml, laml = modes_eig(model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"]), d["t_G"], Ld)
    Ml = Ml.numpy(); laml = np.clip(laml.numpy(), 1e-6, None); zlig = d["dphi"].astype(np.float64) @ (d["G"].astype(np.float64) @ Ml)
    Z = np.concatenate([d["zprot"].astype(np.float64), zlig], 1); lam = np.concatenate([d["lamp"].astype(np.float64), laml])
    T = len(Z); h = int(T*0.8)
    sd = Z[:h].std(0) + 1e-6                                     # EMPIRICAL per-complex whitening: required for coherent
    d["h"] = h; d["sd"] = sd; d["zmu"] = Z[:h].mean(0); d["Z"] = Z   # cross-complex pooling (structure-lambda scale is per-complex-inconsistent)
    d["Zn"] = ((Z-d["zmu"])/sd).astype(np.float32); d["meanlam"] = float(lam.mean()); d["ratio"] = d["n"]/d["nb"]

# --- propagator ---
betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1-betas, 0)
class FiLM(nn.Module):
    def __init__(self, Ld_, h=256):
        super().__init__(); self.inp = nn.Linear(Ld_+16, h); self.h1 = nn.Linear(h, h); self.h2 = nn.Linear(h, h); self.out = nn.Linear(h, Ld_)
        self.f0 = nn.Linear(Ld_, 2*h); self.f1 = nn.Linear(Ld_, 2*h); self.f2 = nn.Linear(Ld_, 2*h)
        for ff in (self.f0, self.f1, self.f2): nn.init.zeros_(ff.weight); nn.init.zeros_(ff.bias)
    def temb(self, t): fq = torch.exp(torch.arange(8)*(-math.log(1e4)/8)); a = t[:, None].float()*fq[None]; return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def film(self, lin, x, c): g, b = lin(c).chunk(2, -1); return torch.nn.functional.gelu(x*(1+g)+b)
    def forward(self, zt, cond, t):
        x = self.film(self.f0, self.inp(torch.cat([zt, self.temb(t)], -1)), cond); x = self.film(self.f1, self.h1(x), cond)
        return self.out(self.film(self.f2, self.h2(x), cond))


def pool_pairs(idxs, cols):
    C, X = [], []
    for i in idxs:
        Zn = data[i]["Zn"][:, cols]; h = data[i]["h"]; C.append(Zn[:h-TAU]); X.append(Zn[TAU:h])
    return torch.tensor(np.concatenate(C)), torch.tensor(np.concatenate(X))


def train_pooled(idxs, cols, steps):
    Ld_ = len(cols); m = FiLM(Ld_); o = torch.optim.Adam(m.parameters(), lr=1e-3); C, X = pool_pairs(idxs, cols)
    for _ in range(steps):
        i = torch.randint(0, C.shape[0], (256,)); c, x0 = C[i], X[i]; k = torch.randint(0, Tdiff, (256,)); ns = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None]*x0+(1-ac[k]).sqrt()[:, None]*ns; o.zero_grad(); ((m(xk, c, k)-ns)**2).mean().backward(); o.step()
    return m


@torch.no_grad()
def gen_batch(m, cond):                                          # batched one-step (decorrelated -> equilibrium samples)
    B = cond.shape[0]; x = torch.randn(B, cond.shape[1]); ct = torch.tensor(cond)
    for k in reversed(range(Tdiff)):
        eps = m(x, ct, torch.full((B,), k)); al = 1-betas[k]; mn = (x-betas[k]/(1-ac[k]).sqrt()*eps)/al.sqrt()
        x = mn+((1-ac[k-1])/(1-ac[k])*betas[k]).sqrt()*torch.randn_like(x) if k > 0 else mn
    return torch.nan_to_num(x, nan=0.0).clamp(-12, 12).numpy()


def offdiag(Cm): nn_ = Cm.shape[0]; return float(np.abs(Cm[~np.eye(nn_, dtype=bool)]).mean())
def xblock(Cm): return float(np.abs(Cm[:Lp, Lp:]).mean())
def ekurt(X): Xc = (X-X.mean(0))/(X.std(0)+1e-9); return float((((Xc**4).mean(0))-3).mean())


def evaluate(scale, jm, im_p, im_l):
    GJ, GI, RF = [], [], []; dj, di = [], []
    for i in he_idx:                                            # generate once; each model gets its OWN c
        d = data[i]; cond = d["Zn"][:d["h"]]; ref = d["Z"]
        gj = gen_batch(jm, cond)*d["sd"]+d["zmu"]
        gp = gen_batch(im_p, cond[:, :Lp]); gl = gen_batch(im_l, cond[:, Lp:]); gi = np.concatenate([gp, gl], 1)*d["sd"]+d["zmu"]
        GJ.append(gj); GI.append(gi); RF.append(ref)
        dj.append(float((gj.std(0)/(ref.std(0)+1e-9)).mean())); di.append(float((gi.std(0)/(ref.std(0)+1e-9)).mean()))
    cj = 1.0/np.mean(dj); ci = 1.0/np.mean(di); fr = {"jm": 0, "jc": 0, "im": 0, "ic": 0}; xbj, xbi, xbr = [], [], []
    for gj, gi, ref in zip(GJ, GI, RF):
        gjc = (gj-gj.mean(0))*cj+gj.mean(0); gic = (gi-gi.mean(0))*ci+gi.mean(0)
        xcr = offdiag(np.corrcoef(ref.T))
        fr["jm"] += 0.80 <= float((gjc.std(0)/(ref.std(0)+1e-9)).mean()) <= 1.25
        fr["im"] += 0.80 <= float((gic.std(0)/(ref.std(0)+1e-9)).mean()) <= 1.25
        fr["jc"] += 0.5*xcr <= offdiag(np.corrcoef(gjc.T)) <= 2.0*xcr
        fr["ic"] += 0.5*xcr <= offdiag(np.corrcoef(gic.T)) <= 2.0*xcr
        xbj.append(xblock(np.corrcoef(gj.T))); xbi.append(xblock(np.corrcoef(gi.T))); xbr.append(xblock(np.corrcoef(ref.T)))
    n = len(he_idx)
    print(f"  scale {scale:>4}: JOINT marg {fr['jm']}/{n} coup {fr['jc']}/{n} x-blk {np.mean(xbj):.3f} (c={cj:.2f}) "
          f"| INDEP marg {fr['im']}/{n} coup {fr['ic']}/{n} x-blk {np.mean(xbi):.3f} (c={ci:.2f}) | ref x-blk {np.mean(xbr):.3f}", flush=True)
    return cj, dj


print("\n=== LEARNING CURVE: joint marginals vs corpus scale (primary question) ===", flush=True)
scales = [s for s in SCALES if s <= len(tr_idx)] + ([len(tr_idx)] if len(tr_idx) not in SCALES else [])
im_p = train_pooled(tr_idx, list(range(Lp)), 4000); im_l = train_pooled(tr_idx, list(range(Lp, L)), 4000)  # independent at full scale
last = None
for S in scales:
    jm = train_pooled(tr_idx[:S], list(range(L)), 4000)
    cfit, defs = evaluate(S, jm, im_p, im_l); last = (jm, cfit)

# --- byproduct 1: is c predictable from structure? (per-complex, at full-scale joint model) ---
jm = last[0]; rows = []
for i in he_idx:
    d = data[i]; gj = gen_batch(jm, d["Zn"][:d["h"]])*d["sd"]+d["zmu"]
    ci = 1.0/float((gj.std(0)/(d["Z"].std(0)+1e-9)).mean()); rows.append((ci, d["meanlam"], d["nb"], d["n"], d["ratio"]))
R = np.array(rows); names = ["mean_lambda", "protein_beads", "ligand_atoms", "lig/prot_ratio"]
print("\n=== BYPRODUCT 1: is c predictable from structure? (per-complex c vs features, held-out) ===", flush=True)
print(f"  per-complex c: mean {R[:,0].mean():.2f} std {R[:,0].std():.2f} range [{R[:,0].min():.2f},{R[:,0].max():.2f}]")
for j, nm in enumerate(names):
    r = np.corrcoef(R[:, 0], R[:, 1+j])[0, 1]; print(f"  corr(c, {nm}) = {r:+.2f}")
print("  (strong corr => c becomes a learnable function of structure; the last per-system fit disappears)")
print("\n=== BYPRODUCT 2: first corpus-scale training run (prior learned components used 5-20 systems;")
print("    overfitting was the recurring diagnosis). Codec question stays closed (B-factor oracle ceiling).", flush=True)
