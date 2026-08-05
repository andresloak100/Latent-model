"""First MULTI-MOLECULE test: protein-ligand COMPLEX with both codecs (ANM protein + graph-codec
ligand), one concatenated latent. §7 additivity has never been measured; this measures it.

GATE (pre-registered, report the number FIRST then branch):
  cross-block coupling ~= null  -> protein & ligand latents effectively INDEPENDENT. Do NOT build
      the joint propagator; run TWO INDEPENDENT propagators. This VALIDATES §7 additivity (the
      54 modes x 231 systems latent-budget arithmetic) -- a WIN, stated as such.
  cross-block coupling SIGNIFICANT -> build the JOINT propagator; report gen-vs-ref CROSS-BLOCK
      coupling capture as its OWN line (known weakness: propagator under-captures strong coupling,
      adds ~0.03-0.18); AND report joint vs TWO INDEPENDENT propagators (must earn its complexity).

Same whitening, same global-c calibration (report whether c=2.87 transfers to the mixed latent),
same acceptance test + pre-committed thresholds, per-system pass/fail then fractions. Protein
coarse-grained to residue beads (sequential runs of atoms_residue). Latents L_p=L_l=8."""
import math, pickle, numpy as np, torch, torch.nn as nn, h5py
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"; CACHE = f"{WR}/graph_codec_cache.pkl"
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
Lp = Ld = 8; L = Lp + Ld; Tdiff, TAU, H = 100, 1, 400; C_PROTEIN = 2.87
torch.manual_seed(0); np.random.seed(0)
data = pickle.load(open(CACHE, "rb")); data = [d for d in data if d["ntors"] >= Ld]


class GraphCodec(nn.Module):                                     # verbatim
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


def kabsch(traj):
    ref = traj[0]; rc = ref.mean(0); Q = ref-rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P-pc; U, S, Vt = np.linalg.svd(Pc.T@Q); dd = np.sign(np.linalg.det(Vt.T@U.T))
        out[t] = (P-pc)@(Vt.T@np.diag([1, 1, dd])@U.T).T + rc
    return out


def anm(xyz, K, cutoff=12.0):
    n = len(xyz); Hm = np.zeros((3*n, 3*n)); ar = np.arange(3); I, J, Uu = [], [], []
    for i in range(n):
        dv = xyz-xyz[i]; dd = np.linalg.norm(dv, axis=1)
        for j in np.where((dd < cutoff) & (dd > 1e-6))[0]:
            if j > i: I.append(i); J.append(j); Uu.append(dv[j]/dd[j])
    I, J, Uu = np.array(I), np.array(J), np.array(Uu); b = Uu[:, :, None]*Uu[:, None, :]
    def sc(a, c, V):
        r = (3*a[:, None, None]+ar[None, :, None])*np.ones((1, 1, 3), int); co = (3*c[:, None, None]+ar[None, None, :])*np.ones((1, 3, 1), int)
        np.add.at(Hm, (r.ravel(), co.ravel()), V.ravel())
    sc(I, I, b); sc(J, J, b); sc(I, J, -b); sc(J, I, -b); w, V = np.linalg.eigh(Hm); return V[:, 6:6+K], w[6:6+K]


# --- retrain ligand codec ---
for d in data:
    d["t_xat"] = torch.tensor(d["xat"]).double(); d["t_edges"] = torch.tensor(d["edges"]); d["t_tidx"] = torch.tensor(d["tidx"])
    d["t_tsc"] = torch.tensor(d["tsc"]).double(); d["t_G"] = torch.tensor(d["G"].astype(np.float64))
tr = [i for i in range(len(data)) if i % 4 != 0]; model = GraphCodec(data[0]["xat"].shape[1]).double()
opt = torch.optim.Adam(model.parameters(), lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
print("[complex] retraining ligand codec ...", flush=True)
for ep in range(500):
    perm = np.random.permutation(tr); opt.zero_grad(); loss = 0.0
    for i in perm[:32]:
        d = data[i]; logk = model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"]); dt = torch.tensor(d["dphi"].astype(np.float64))
        for Ltr in (4, 8): Mx, _ = modes_eig(logk, d["t_G"], Ltr); loss = loss + (((dt@(d["t_G"]@Mx))@Mx.T-dt)**2).mean()
    (loss/32).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()

# --- build per-system joint latent ---
f = h5py.File(MD, "r"); sysd = []
for d in data:
    dom = d["dom"]; g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); Pn = int(mbi[-1])
    res = np.array(g["atoms_residue"])[:Pn]; resid = np.concatenate([[0], np.cumsum(res[1:] != res[:-1])]); nb = int(resid.max()+1)
    if nb < 30 or nb > 450: continue
    pc = np.array(g["trajectory_coordinates"])[:, :Pn, :].astype(np.float64); T = len(pc)
    beads = np.stack([np.array([pc[t][resid == r].mean(0) for r in range(nb)]) for t in range(T)])  # (T,nb,3)
    al = kabsch(beads); dcart = (al-al[0]).reshape(T, -1); Mp, lamp = anm(al[0], Lp); lamp = np.clip(lamp, 1e-8, None)
    hh = int(T*0.8); zmup = (dcart @ Mp)[:hh].mean(0); sdp = np.sqrt(0.593/lamp); Zp = dcart @ Mp
    with torch.no_grad():
        Ml, laml = modes_eig(model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"]), d["t_G"], Ld)
    Ml = Ml.numpy(); laml = np.clip(laml.numpy(), 1e-6, None); Zl = d["dphi"].astype(np.float64) @ (d["G"].astype(np.float64) @ Ml)
    sdl = np.sqrt(0.593/laml)
    if min(sdp.min(), sdl.min()) < 1e-8 or T != len(Zl): continue
    Z = np.concatenate([Zp, Zl], 1); sd = np.concatenate([sdp, sdl]); zmu = Z[:hh].mean(0)
    Zn = ((Z-zmu)/sd).astype(np.float32); top2 = np.argsort(Z.std(0))[::-1][:2]
    sysd.append(dict(dom=dom, Z=Z, Zn=Zn, sd=sd, zmu=zmu, h=hh, nb=nb, top2=top2, thr=np.median(Z[:, top2], 0)))
    if len(sysd) >= 12: break
print(f"[complex] {len(sysd)} complexes with joint latent (Lp={Lp}+Ld={Ld}); protein beads {[s['nb'] for s in sysd]}", flush=True)


def xblock(C): return float(np.abs(C[:Lp, Lp:]).mean())         # mean |corr| across the protein-ligand block


# --- ADDITIVITY GATE (report first) ---
print("\n=== §7 ADDITIVITY: cross-block coupling (report FIRST, then branch) ===", flush=True)
cref, cnull = [], []
for s in sysd:
    Z = s["Z"]; Cr = np.corrcoef(Z.T); cr = xblock(Cr)
    Zp2 = Z[:, :Lp].copy(); Zl2 = Z[np.random.permutation(len(Z))][:, Lp:]; cn = xblock(np.corrcoef(np.concatenate([Zp2, Zl2], 1).T))
    cref.append(cr); cnull.append(cn); print(f"  {s['dom']:8s} nb{s['nb']:>4}  cross|corr| ref {cr:.3f}  null(perm) {cn:.3f}", flush=True)
cref, cnull = np.array(cref), np.array(cnull)
SIG = (cref.mean() > cnull.mean() + 2*cnull.std()) and (cref.mean() - cnull.mean() > 0.05)
print(f"\n  MEAN cross-block |corr|: ref {cref.mean():.3f} +/- {cref.std():.3f}   null {cnull.mean():.3f} +/- {cnull.std():.3f}")
print(f"  frac systems ref>null+0.05: {np.mean(cref > cnull+0.05):.0%}   ==> coupling {'SIGNIFICANT' if SIG else '~= NULL (additive)'}", flush=True)

# --- propagator machinery (verbatim) ---
betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1-betas, 0)
class FiLM(nn.Module):
    def __init__(self, L, h=256):
        super().__init__(); self.inp = nn.Linear(L+16, h); self.h1 = nn.Linear(h, h); self.h2 = nn.Linear(h, h); self.out = nn.Linear(h, L)
        self.f0 = nn.Linear(L, 2*h); self.f1 = nn.Linear(L, 2*h); self.f2 = nn.Linear(L, 2*h)
        for ff in (self.f0, self.f1, self.f2): nn.init.zeros_(ff.weight); nn.init.zeros_(ff.bias)
    def temb(self, t): fq = torch.exp(torch.arange(8)*(-math.log(1e4)/8)); a = t[:, None].float()*fq[None]; return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def film(self, lin, x, c): g, b = lin(c).chunk(2, -1); return torch.nn.functional.gelu(x*(1+g)+b)
    def forward(self, zt, cond, t):
        x = self.film(self.f0, self.inp(torch.cat([zt, self.temb(t)], -1)), cond); x = self.film(self.f1, self.h1(x), cond)
        return self.out(self.film(self.f2, self.h2(x), cond))
def train(Zn, h, Ld_):
    m = FiLM(Ld_); o = torch.optim.Adam(m.parameters(), lr=1e-3); Z = torch.tensor(Zn); zc = Z[:h-TAU]; tg = Z[TAU:h]
    for _ in range(1200):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),)); c, x0 = zc[i], tg[i]; k = torch.randint(0, Tdiff, (len(i),)); ns = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None]*x0+(1-ac[k]).sqrt()[:, None]*ns; o.zero_grad(); ((m(xk, c, k)-ns)**2).mean().backward(); o.step()
    return m
@torch.no_grad()
def rollout(m, z0, Ld_):
    z = torch.tensor(z0); out = [z.clone()]
    for _ in range(H):
        x = torch.randn(1, Ld_)
        for k in reversed(range(Tdiff)):
            eps = m(x, z, torch.full((1,), k)); al = 1-betas[k]; mn = (x-betas[k]/(1-ac[k]).sqrt()*eps)/al.sqrt()
            x = mn+((1-ac[k-1])/(1-ac[k])*betas[k]).sqrt()*torch.randn_like(x) if k > 0 else mn
        z = torch.nan_to_num(x, nan=0.0).clamp(-12, 12); out.append(z.clone())
    return torch.cat(out, 0).numpy()
def offdiag(C): n = C.shape[0]; return float(np.abs(C[~np.eye(n, dtype=bool)]).mean())
def ekurt(X): Xc = (X-X.mean(0))/(X.std(0)+1e-9); return float((((Xc**4).mean(0))-3).mean())

def gen_joint(s):
    g = rollout(train(s["Zn"], s["h"], L), s["Zn"][s["h"]:s["h"]+1], L)*s["sd"]+s["zmu"]; return g
def gen_indep(s):
    gp = rollout(train(s["Zn"][:, :Lp], s["h"], Lp), s["Zn"][s["h"]:s["h"]+1, :Lp], Lp)
    gl = rollout(train(s["Zn"][:, Lp:], s["h"], Ld), s["Zn"][s["h"]:s["h"]+1, Lp:], Ld)
    return np.concatenate([gp, gl], 1)*s["sd"]+s["zmu"]

print("\n=== propagator: JOINT vs TWO INDEPENDENT (deficit -> global c) ===", flush=True)
J = {s["dom"]: gen_joint(s) for s in sysd}; IN = {s["dom"]: gen_indep(s) for s in sysd}
NFIT = 4; fit, ev = sysd[:NFIT], sysd[NFIT:]
defj = np.mean([float((J[s["dom"]].std(0)/(s["Z"].std(0)+1e-9)).mean()) for s in fit]); cj = 1.0/defj
print(f"  c TRANSFER to mixed latent: joint-fit c = {cj:.2f} (protein-fit was 2.87, ligand-fit 1.03)", flush=True)

def disc_block(gen, ref):                                        # cross-block coupling capture as its own number
    return xblock(np.corrcoef(gen.T)), xblock(np.corrcoef(ref.T))

for name, GEN, cval in [("JOINT (c=2.87 transfer)", J, C_PROTEIN), ("JOINT (refit c)", J, cj), ("INDEPENDENT (refit c)", IN, cj)]:
    fr = {k: 0 for k in ["marg", "coup", "nG"]}; xb_g, xb_r = [], []
    for s in ev:
        gen = GEN[s["dom"]]; genc = (gen-gen.mean(0))*cval+gen.mean(0); ref = s["Z"]
        vr = float((genc.std(0)/(ref.std(0)+1e-9)).mean()); xc = offdiag(np.corrcoef(genc.T)); xcr = offdiag(np.corrcoef(ref.T))
        kg = ekurt(genc); kr = ekurt(ref); xbg, xbr = disc_block(genc, ref); xb_g.append(xbg); xb_r.append(xbr)
        fr["marg"] += 0.80 <= vr <= 1.25; fr["coup"] += 0.5*xcr <= xc <= 2.0*xcr; fr["nG"] += abs(kg-kr) <= 0.5
    n = len(ev)
    print(f"\n  {name}: marginals {fr['marg']}/{n}  coupling(all) {fr['coup']}/{n}  non-Gaussianity {fr['nG']}/{n}")
    print(f"    CROSS-BLOCK |corr| capture (its own line): gen {np.mean(xb_g):.3f}  vs ref {np.mean(xb_r):.3f}  (deficit {np.mean(xb_r)-np.mean(xb_g):+.3f})", flush=True)
print("\n  read: coupling ~=null -> independent WINS (validates §7 additivity). significant -> joint must")
print("        beat independent on CROSS-BLOCK capture to earn its complexity.", flush=True)
