"""Learning curve for a codec that GENERALISES across systems, with ALL references
on IDENTICAL held-out systems (cross-cohort mismatch made structurally impossible).

Two learned variants, both variance-weighted, both initialised at plain ANM:
  PRIMARY  spring-constant reparameterisation. A shared net maps per-residue
           STRUCTURE features -> softness scalars s_i; springs g_ij=exp((s_i+s_j)/2)
           (uniform => plain ANM). Trained by minimising tr(H C)/tr(H) -- the
           elastic energy of the observed high-variance modes relative to total
           stiffness (= the Boltzmann likelihood of the covariance under the
           network, normaliser relaxed) -- differentiable in the springs, NO eigh
           in the training loop; eigh only at eval to read out modes. Smaller
           hypothesis class (O(N) scalars, physically constrained), degenerates to
           ANM at zero correction so it cannot beat itself into non-physical modes.
  CONTROL  direct mode prediction: shared A (KxL) mixes ANM modes, orthonormalise,
           variance-weighted subspace loss. The 64x3N-output class that overfit.

References per held-out system, absolute A, on the SAME held-out frames (rep0 2nd
half): null / ANM / cross-replica (rep1's modes, the honest general ceiling) /
within (rep0's own past, temporal split) / learned_spring / learned_direct.
Headroom closed = (ANM - learned)/(ANM - cross), now well-defined (one cohort).
Level: CA (ANM-natural; CA within/cross = 0.89/1.16 == backbone 0.93/1.17)."""
import glob, os, numpy as np, torch, torch.nn as nn, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
L, K, CUT = 64, 128, 13.0
TEMP, R0, R1 = "320", "0", "1"
ALPHA = 2.0
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"
torch.manual_seed(0)


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


def edges_of(ref):
    n = len(ref); I, J, U = [], [], []
    for i in range(n):
        dv = ref - ref[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < CUT) & (dist > 1e-6))[0]:
            if j <= i: continue
            I.append(i); J.append(j); U.append(dv[j] / dist[j])
    return np.array(I), np.array(J), np.array(U, dtype=np.float64)


def anm_from_edges(n, I, J, U, g):                                  # weighted ANM Hessian -> modes
    H = np.zeros((3 * n, 3 * n)); ar = np.arange(3)
    b = g[:, None, None] * (U[:, :, None] * U[:, None, :])          # (E,3,3)
    def scat(a, c, V):
        rows = (3 * a[:, None, None] + ar[None, :, None]) * np.ones((1, 1, 3), int)
        cols = (3 * c[:, None, None] + ar[None, None, :]) * np.ones((1, 3, 1), int)
        np.add.at(H, (rows.ravel(), cols.ravel()), V.ravel())
    scat(I, I, b); scat(J, J, b); scat(I, J, -b); scat(J, I, -b)
    w, V = np.linalg.eigh(H)
    return V[:, 6:6 + K]                                            # (3n,K) orthonormal


def feats(ref, I, J, n):
    deg = np.zeros(n); dens = np.zeros(n)
    d = np.linalg.norm(ref[I] - ref[J], axis=1)
    np.add.at(deg, I, 1.0); np.add.at(deg, J, 1.0)
    np.add.at(dens, I, 1.0 / d); np.add.at(dens, J, 1.0 / d)
    F = np.stack([deg, dens], 1)
    return (F - F.mean(0)) / (F.std(0) + 1e-6)                      # z-scored within system


def pca(dc):
    _, S, Vt = np.linalg.svd(dc, full_matrices=False)
    return Vt[:L].T, (S[:L] ** 2)                                  # modes (3n,L), variances (L,)


def resid(ev, M):
    rec = ev @ M @ M.T
    return float(np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * (ev.shape[1] // 3))))


def rigid_content(ref, Mpca, var):                                 # variance-weighted PCA content in H's null space
    n = len(ref); cen = ref - ref.mean(0); T = np.zeros((3 * n, 6))
    for a in range(3):
        v = np.zeros((n, 3)); v[:, a] = 1.0; T[:, a] = v.reshape(-1)        # 3 translations
    for a in range(3):
        ax = np.zeros(3); ax[a] = 1.0
        T[:, 3 + a] = np.cross(np.tile(ax, (n, 1)), cen).reshape(-1)        # 3 rotation generators
    Q, _ = np.linalg.qr(T)                                                  # orthonormal rigid basis (3n,6)
    proj = Q.T @ Mpca                                                       # (6,L)
    return float((var * (proj ** 2).sum(0)).sum() / var.sum())


# ---------- build the one cohort (both replicas), precompute everything ----------
S = []
for fp in sorted(glob.glob(f"{DATA}/*.h5")):
    try:
        f = h5py.File(fp, "r")
    except Exception:
        continue
    with f:
        dom = list(f.keys())[0]; g = f[dom]
        if TEMP not in g or R0 not in g[TEMP] or R1 not in g[TEMP]:
            continue
        z = np.array(g["z"]); N = len(z); names = parse_names(g, N)
        if len(names) != N:
            continue
        heavy = z != 1; ca = names[heavy] == "CA"
        if ca.sum() < K // 3 + 10:
            continue
        c0 = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, ca, :]
        c1 = g[TEMP][R1]["coords"][:].astype(np.float64)[:, heavy, :][:, ca, :]
    n = ca.sum()
    a0 = kabsch(c0, np.ones(n, bool)); a1 = kabsch(c1, np.ones(n, bool))
    d0 = (a0 - a0[0]).reshape(len(a0), -1); d1 = (a1 - a1[0]).reshape(len(a1), -1)
    h = len(d0) // 2; mean0 = d0[:h].mean(0)
    Mpca, var = pca(d0[:h] - mean0)                                # target = rep0 past
    Mcross, _ = pca(d1 - d1.mean(0))                              # honest general ceiling = rep1
    ref = a0[0]; I, J, Ue = edges_of(ref)
    Manm = anm_from_edges(n, I, J, Ue, np.ones(len(I)))          # plain ANM (uniform springs)
    ev = d0[h:] - mean0
    S.append(dict(
        dom=dom, n=n, I=I, J=J, U=Ue, ref=ref, F=feats(ref, I, J, n),
        Mpca=Mpca, var=var / var.sum(), Mcross=Mcross, Manm=Manm, ev=ev,
        null=float(np.sqrt((ev ** 2).sum() / (ev.shape[0] * n))),
        within=resid(ev, Mpca), cross=resid(ev, Mcross), anm=resid(ev, Manm[:, :L])))
print(f"[curve] {len(S)} systems (both replicas), n_CA {min(s['n'] for s in S)}..{max(s['n'] for s in S)}")

S.sort(key=lambda s: s["n"])
if os.path.exists(FROZEN):                                          # test set FROZEN before any new domain entered
    frozen = [d for d in open(FROZEN).read().split() if d]
    test = [s for s in S if s["dom"] in frozen]
    missing = set(frozen) - {s["dom"] for s in test}
    assert not missing, f"FROZEN test domains missing from cohort: {missing}"
    train = [s for s in S if s["dom"] not in frozen]
else:                                                              # first run ever: derive and persist
    test = S[2::4]; train = [s for s in S if s not in test]
    with open(FROZEN, "w") as fh: fh.write("\n".join(s["dom"] for s in test) + "\n")
assert not ({s["dom"] for s in test} & {s["dom"] for s in train}), "test/train overlap"
print(f"  train {len(train)} / test {len(test)} (test FROZEN; training pool grows only)")
print(f"  FROZEN TEST ({len(test)}): " + " ".join(s["dom"] for s in test))

# torch tensors for training
for s in S:
    s["tI"] = torch.tensor(s["I"]); s["tJ"] = torch.tensor(s["J"])
    s["tU"] = torch.tensor(s["U"], dtype=torch.float32)
    s["tMp"] = torch.tensor(s["Mpca"], dtype=torch.float32).reshape(s["n"], 3, L)
    s["tvar"] = torch.tensor(s["var"], dtype=torch.float32)
    s["tF"] = torch.tensor(s["F"], dtype=torch.float32)
    s["tManm"] = torch.tensor(s["Manm"], dtype=torch.float32)
    s["tMpca"] = torch.tensor(s["Mpca"], dtype=torch.float32)


def spring_softness(net, s):                                       # per-residue softness scalars
    return ALPHA * torch.tanh(net(s["tF"]).squeeze(-1))            # (n,), 0 at init -> uniform ANM


def spring_loss(net, s):
    sm = spring_softness(net, s)
    g = torch.exp((sm[s["tI"]] + sm[s["tJ"]]) / 2)                # (E,) springs
    dM = s["tMp"][s["tI"]] - s["tMp"][s["tJ"]]                    # (E,3,L)
    proj = (dM * s["tU"][:, :, None]).sum(1)                      # (E,L)
    Ek = (g[:, None] * proj ** 2).sum(0)                         # energy per mode
    return (s["tvar"] * Ek).sum() / g.sum() + 1e-3 * sm.pow(2).mean()   # tr(HC)/tr(H) + tiny reg


def train_spring(subset, tag=""):
    net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)   # init -> plain ANM (zero correction)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    heldA = lambda: float(np.mean([eval_spring(net, s) for s in test]))
    e0 = heldA()                                                   # GUARD1: epoch-0 must equal plain ANM
    assert abs(e0 - anm_m) < 0.03, f"GUARD1 FAIL: epoch0 heldout {e0:.3f} != ANM {anm_m:.3f} -> init/eval bug"
    hist = []; tripped = False
    for ep in range(400):
        opt.zero_grad(); loss = sum(spring_loss(net, s) for s in subset) / len(subset)
        loss.backward(); opt.step()
        if ep % 80 == 79:
            hA = heldA(); hist.append((float(loss), hA))
            if hA > anm_m + 0.05:                                  # monotone away from ANM = wrong-way objective
                print(f"  [GUARD1 tsz{tag}] WRONG-WAY: heldout {hA:.2f} > ANM {anm_m:.2f}+.05 at ep{ep+1}"
                      f" -> objective points wrong; STOP.")
                tripped = True; break
    return net, hist, e0, tripped


def train_direct(subset):
    A = torch.zeros(K, L, requires_grad=True)
    with torch.no_grad(): A[:L] = torch.eye(L)
    opt = torch.optim.Adam([A], lr=1e-2)
    for _ in range(1500):
        opt.zero_grad(); loss = 0.0
        for s in subset:
            Q, _ = torch.linalg.qr(s["tManm"] @ A)
            cap = (Q.T @ s["tMpca"]).pow(2).sum(0)               # (L,) capture per target mode
            loss = loss + (s["tvar"] * (1 - cap)).sum()          # variance-weighted subspace loss
        (loss / len(subset)).backward(); opt.step()
    return A.detach().numpy()


def eval_spring(net, s):
    with torch.no_grad():
        sm = spring_softness(net, s).numpy()
    g = np.exp((sm[s["I"]] + sm[s["J"]]) / 2)
    M = anm_from_edges(s["n"], s["I"], s["J"], s["U"], g)[:, :L]
    return resid(s["ev"], M)


def eval_direct(An, s):
    Q, _ = np.linalg.qr(s["Manm"] @ An)
    return resid(s["ev"], Q)


# ---------- per-mode overlap: does the leading (high-variance) end overlap better? ----------
print("\n=== per-mode cross-replica capture (mean over test systems, ||Mcross^T m_k||^2) ===")
caps = np.zeros(L)
for s in test:
    caps += ((s["Mcross"].T @ s["Mpca"]) ** 2).sum(0)
caps /= len(test)
bins = [(0, 8), (8, 16), (16, 32), (32, 48), (48, 64)]
print("  modes " + "  ".join(f"{a+1}-{b}:{caps[a:b].mean():.2f}" for a, b in bins) +
      f"   (overall mean {caps.mean():.2f})")
rc = np.mean([rigid_content(s["ref"], s["Mpca"], s["var"]) for s in S])
print(f"  [GUARD3] residual rigid-body content of variance-weighted PCA target "
      f"(Kabsch should -> ~0; else tr(HC) is contaminated): {rc:.4f}")

# ---------- learning curve ----------
def rowstats(vals):
    a = np.array(vals); worst = int(np.argmax([s["within"] for s in test]))
    keep = np.array([i for i in range(len(test)) if i != worst])
    return a.mean(), a[keep].mean()


anm_m = np.mean([s["anm"] for s in test]); cross_m = np.mean([s["cross"] for s in test])
within_m = np.mean([s["within"] for s in test]); null_m = np.mean([s["null"] for s in test])
print(f"\n=== LEARNING CURVE (held-out mean absolute A, CA, L={L}; {len(test)} test systems) ===")
print(f"  fixed refs: null {null_m:.2f}  ANM {anm_m:.2f}  cross(ceiling) {cross_m:.2f}  within {within_m:.2f}")
print(f"  {'train_sz':>8}{'spring':>8}{'direct':>8}{'ANM':>7}{'cross':>7}   headroom(sp/dir)  guard")
all_ckpts = []; net20 = An20 = None
for sz in [x for x in (5, 10, 20, 40, 60) if x <= len(train)]:
    idx = np.unique(np.linspace(0, len(train) - 1, sz).round().astype(int))
    sub = [train[i] for i in idx]
    net, hist, e0, tripped = train_spring(sub, tag=str(sz)); An = train_direct(sub)
    net20, An20, last_sz = net, An, sz
    ls = [eval_spring(net, s) for s in test]; ld = [eval_direct(An, s) for s in test]
    ls_m, ls_ko = rowstats(ls); ld_m, ld_ko = rowstats(ld)
    hs = (anm_m - ls_m) / (anm_m - cross_m); hd = (anm_m - ld_m) / (anm_m - cross_m)
    print(f"  {sz:>8}{ls_m:>8.2f}{ld_m:>8.2f}{anm_m:>7.2f}{cross_m:>7.2f}   {hs:>5.0%} / {hd:>4.0%}"
          f"   ep0 {e0:.2f}{' TRIPPED' if tripped else ' ok'} (excl-worst sp {ls_ko:.2f} dir {ld_ko:.2f})")
    all_ckpts += hist

# GUARD2: does the global-spectral surrogate track the top-64 held-out metric?
La = np.array([c[0] for c in all_ckpts]); Aa = np.array([c[1] for c in all_ckpts])
r = float(np.corrcoef(La, Aa)[0, 1]) if len(all_ckpts) > 2 else float("nan")
print(f"\n[GUARD2] surrogate-vs-metric Pearson(train_loss, heldout_A) over {len(all_ckpts)} ckpts = {r:+.2f}  "
      f"({'tracks -> objective valid' if r > 0.3 else 'DOES NOT TRACK -> switch to variance-weighted subspace loss on leading modes'})")

# full per-system table at the largest size (reuse trained maps)
print(f"\n=== per-system at train_sz={last_sz} (absolute A) ===")
print(f"  {'system':9s}{'n':>5}{'null':>7}{'ANM':>7}{'cross':>7}{'within':>8}{'spring':>8}{'direct':>8}")
for s in test:
    print(f"  {s['dom']:9s}{s['n']:>5}{s['null']:>7.2f}{s['anm']:>7.2f}{s['cross']:>7.2f}"
          f"{s['within']:>8.2f}{eval_spring(net20, s):>8.2f}{eval_direct(An20, s):>8.2f}")
print("\n  read: spring < ANM at growing train_sz AND spring < direct -> reparam helps as predicted;"
      "\n        spring ~ ANM -> structure features too weak; direct still <= ANM -> overfitting persists.")
