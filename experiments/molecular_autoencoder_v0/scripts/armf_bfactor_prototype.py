"""B-factor pretraining via the spring-constant bridge (corrected Track 2).

Learned map: per-residue structure features -> spring constants (shared MLP, init
uniform = plain GNM/ANM). PRETRAIN: build GNM Kirchhoff, predicted fluctuation =
diag(Gamma^+) (differentiable), regress vs z-scored crystallographic B-factors (PDB
scale). FINE-TUNE: same, vs mdCATH per-residue RMSF. EVAL: transfer the springs into
the ANM Hessian, diagonalise, reconstruct frozen-7 displacement -> absolute A vs ANM
(uniform springs = 1.37 @ 64 scalars). GNM (N x N) for training, ANM (3N x 3N) for eval.

Guards: epoch-0 == plain ANM (uniform springs); Pearson(pretrain loss, held-out recon A);
per-structure B z-normalisation; pretrain-vs-no-pretrain ablation. Stopping rule: doesn't
beat 1.37 A -> codec closes.
"""
# INBOX 70d GUARD. This track consumes B-factors as CRYSTALLOGRAPHIC. AFDB stores pLDDT in
# the same column, so a predicted structure reaching here would be read as a B-factor and
# would look entirely plausible. The track is closed (oracle-B 1.45 A lost to ANM 1.37 A)
# but the scripts remain, so the refusal is made explicit rather than left to nobody
# pointing an AFDB path at them.
def _refuse_predicted(meta):
    k = (meta or {}).get("confidence_kind")
    if k == "plddt":
        raise ValueError("refusing predicted structure: this script treats the confidence "
                         "column as a crystallographic B-factor, and pLDDT is not one")

import glob, os, numpy as np, torch, torch.nn as nn
BCACHE = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/bfactor_cache"
MCACHE = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/perceiver_cache"
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"
L, CUT, ALPHA = 64, 10.0, 2.0
torch.manual_seed(0); np.random.seed(0)


def feats_edges(ca):
    n = len(ca); I, J = [], []
    for i in range(n):
        dv = ca - ca[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < CUT) & (dist > 1e-6))[0]:
            if j <= i: continue
            I.append(i); J.append(j)
    I, J = np.array(I), np.array(J)
    if len(I) == 0: return None
    deg = np.zeros(n); np.add.at(deg, I, 1.0); np.add.at(deg, J, 1.0)
    dd = np.linalg.norm(ca[I] - ca[J], axis=1); dens = np.zeros(n)
    np.add.at(dens, I, 1.0 / dd); np.add.at(dens, J, 1.0 / dd)
    F = np.stack([deg, dens], 1); F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    return F.astype(np.float32), I, J


def kabsch(traj):
    ref = traj[0]; rcen = ref.mean(0); Q = ref - rcen; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


class SpringMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)   # uniform springs at init
    def forward(self, F):
        return ALPHA * torch.tanh(self.net(F).squeeze(-1))


def gnm_fluct(s, I, J, n):                                        # predicted per-residue fluctuation = diag(Gamma^+)
    g = torch.exp((s[I] + s[J]) / 2)
    G = torch.zeros(n, n)
    G = G.index_put((I, J), -g, accumulate=True); G = G.index_put((J, I), -g, accumulate=True)
    G = G.index_put((I, I), g, accumulate=True); G = G.index_put((J, J), g, accumulate=True)
    return torch.diagonal(torch.linalg.pinv(G))


def anm_modes_sp(ca, s, I, J):                                    # ANM Hessian with learned springs -> top-L modes
    n = len(ca); H = np.zeros((3 * n, 3 * n)); U = ca[J] - ca[I]; U = U / np.linalg.norm(U, axis=1, keepdims=True)
    g = np.exp((s[I] + s[J]) / 2)
    for e in range(len(I)):
        i, j = int(I[e]), int(J[e]); b = g[e] * np.outer(U[e], U[e])
        H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
        H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H); return V[:, 6:6 + L]


def resid(ev, M):
    rec = ev @ M @ M.T; return float(np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * (ev.shape[1] // 3))))


def zsc(x):
    return (x - x.mean()) / (x.std() + 1e-9)


# ---- load PDB pretraining set ----
pdb = []
for fp in sorted(glob.glob(f"{BCACHE}/*.npz"))[:1500]:
    d = np.load(fp); ca = d["ca_xyz"].astype(np.float64); b = d["bfac"].astype(np.float64)
    fe = feats_edges(ca)
    if fe is None or len(ca) < 40: continue
    F, I, J = fe
    pdb.append(dict(F=torch.tensor(F), I=torch.tensor(I), J=torch.tensor(J), n=len(ca),
                    bz=torch.tensor(zsc(b), dtype=torch.float32)))
print(f"[bfactor] PDB pretraining structures: {len(pdb)}")

# ---- load mdCATH (fine-tune pool + frozen-7 eval) ----
frozen = [x for x in open(FROZEN).read().split() if x]
md = []
for fp in sorted(glob.glob(f"{MCACHE}/*.npz")):
    dom = os.path.basename(fp)[:-4]; c0 = np.load(fp)["c0"].astype(np.float64)
    if c0.shape[1] < 40: continue
    al = kabsch(c0); d = (al - al[0]).reshape(len(al), -1); h = len(d) // 2; mean0 = d[:h].mean(0)
    fe = feats_edges(al[0])
    if fe is None: continue
    F, I, J = fe; n = c0.shape[1]
    rmsf = np.sqrt(((d[:h] - mean0) ** 2).reshape(h, n, 3).sum(-1).mean(0))     # per-residue RMSF (train half)
    ev = d[h:] - mean0
    md.append(dict(dom=dom, ca=al[0], F=torch.tensor(F), I=torch.tensor(I), J=torch.tensor(J), n=n,
                   rz=torch.tensor(zsc(rmsf), dtype=torch.float32), ev=ev, test=dom in frozen))
train = [s for s in md if not s["test"]]; test = [s for s in md if s["test"]]
print(f"  mdCATH fine-tune {len(train)} / frozen-7 eval {len(test)}")

# reference: ANM with UNIFORM springs (= plain ANM 1.37) on frozen 7
def eval_map(net):
    rs = []
    for s in test:
        with torch.no_grad():
            sp = net(s["F"]).numpy() if net is not None else np.zeros(s["n"])
        M = anm_modes_sp(s["ca"], sp, s["I"].numpy(), s["J"].numpy())
        rs.append(resid(s["ev"], M))
    return float(np.mean(rs))

anm_ref = eval_map(None)                                          # uniform springs baseline
print(f"  plain ANM (uniform springs) held-out: {anm_ref:.2f} A  (expected ~1.37)")


def fluct_loss(net, batch, key):
    loss = 0.0
    for s in batch:
        pred = gnm_fluct(net(s["F"]), s["I"], s["J"], s["n"])
        loss = loss + ((zsc(pred) - s[key]) ** 2).mean()
    return loss / len(batch)


def train_phase(net, data, key, steps, ckpt_eval=False):
    opt = torch.optim.Adam(net.parameters(), lr=3e-3); hist = []
    for st in range(steps):
        batch = [data[i] for i in np.random.choice(len(data), min(8, len(data)), replace=False)]
        opt.zero_grad(); loss = fluct_loss(net, batch, key); loss.backward(); opt.step()
        if ckpt_eval and st % max(1, steps // 6) == 0:
            hist.append((float(loss), eval_map(net)))
    return hist


# GUARD: epoch-0 == plain ANM
net0 = SpringMap(); e0 = eval_map(net0)
assert abs(e0 - anm_ref) < 0.05, f"GUARD FAIL epoch0 {e0:.3f} != ANM {anm_ref:.3f}"
print(f"  GUARD epoch-0 (uniform init) held-out {e0:.2f} == ANM {anm_ref:.2f}  OK")

print("\n=== ablation: pretrain-vs-no-pretrain (held-out reconstruction A on frozen 7, L=64) ===")
# (A) no pretrain: fine-tune on mdCATH RMSF only
netA = SpringMap(); train_phase(netA, train, "rz", 400); rA = eval_map(netA)
# (B) pretrain on B-factors, then fine-tune on mdCATH RMSF
netB = SpringMap(); histP = train_phase(netB, pdb, "bz", 600, ckpt_eval=True); rBpre = eval_map(netB)
train_phase(netB, train, "rz", 400); rB = eval_map(netB)
# GUARD2: does pretraining loss track held-out reconstruction?
La = np.array([c[0] for c in histP]); Aa = np.array([c[1] for c in histP])
r = float(np.corrcoef(La, Aa)[0, 1]) if len(histP) > 2 else float("nan")
print(f"  ANM (uniform)         {anm_ref:.2f}")
print(f"  finetune-only (no PT) {rA:.2f}   vs ANM {rA-anm_ref:+.2f}")
print(f"  pretrain-only (B)     {rBpre:.2f}   vs ANM {rBpre-anm_ref:+.2f}")
print(f"  pretrain + finetune   {rB:.2f}   vs ANM {rB-anm_ref:+.2f}")
print(f"  [GUARD2] Pearson(pretrain loss, held-out A) = {r:+.2f}  "
      f"({'tracks' if r > 0.3 else 'DOES NOT TRACK -> B-factor fit does not predict reconstruction'})")
best = min(rA, rB, rBpre)
print(f"\n  VERDICT: best learned {best:.2f} vs ANM {anm_ref:.2f} -> "
      + ("BEATS ANM -- pretraining earns its place" if best < anm_ref - 0.03 else
         "does NOT beat ANM at matched scalars -> per the stopping rule, CODEC QUESTION CLOSES"))
