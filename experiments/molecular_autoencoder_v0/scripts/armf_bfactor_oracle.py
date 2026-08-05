"""Two checks BEFORE training the B-factor map.

CHECK 1: recalibrate plain GNM-vs-B correlation (0.38 was n=1, cutoff 10). Report at
standard 7.3 A and 10/13, on a real sample with a CI, termini excluded.

CHECK 2 (the gate): the ORACLE. Fit per-residue springs to reproduce a fluctuation
diagonal AS CLOSELY AS POSSIBLE (per system, differentiable diag(Gamma^+), no learned
map), diagonalise, reconstruct mdCATH held-out frames in absolute A. That is the UPPER
BOUND on what any per-residue-fluctuation supervision can buy.
  - Oracle-RMSF: fit to the mdCATH trajectory's OWN per-residue RMSF -> the true
    diagonal of the covariance. This upper-bounds crystallographic B (an external,
    noisier diagonal), so if it fails, B-factor supervision cannot work.
  oracle < 1.37 (ANM) -> headroom, train the map. oracle >= 1.37 -> the diagonal is
  insufficient to determine useful modes (the concrete many-to-one concern); CLOSE.
"""
import glob, os, numpy as np, torch, h5py
BCACHE = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/bfactor_cache"
MCACHE = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/perceiver_cache"
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"
L = 64
torch.manual_seed(0); np.random.seed(0)


def edges(ca, cutoff):
    n = len(ca); I, J = [], []
    for i in range(n):
        dv = ca - ca[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j <= i: continue
            I.append(i); J.append(j)
    return np.array(I), np.array(J)


def gnm_diag_np(ca, cutoff):
    n = len(ca); I, J = edges(ca, cutoff)
    if len(I) == 0: return None
    G = np.zeros((n, n))
    for e in range(len(I)):
        i, j = I[e], J[e]; G[i, j] -= 1; G[j, i] -= 1; G[i, i] += 1; G[j, j] += 1
    return np.diagonal(np.linalg.pinv(G))


def kabsch(traj):
    ref = traj[0]; rcen = ref.mean(0); Q = ref - rcen; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


def anm_modes(ca, s, I, J):
    n = len(ca); H = np.zeros((3 * n, 3 * n)); U = ca[J] - ca[I]; U = U / np.linalg.norm(U, axis=1, keepdims=True)
    g = np.exp((s[I] + s[J]) / 2)
    for e in range(len(I)):
        i, j = int(I[e]), int(J[e]); b = g[e] * np.outer(U[e], U[e])
        H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
        H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H); return V[:, 6:6 + L]


def resid(ev, M):
    rec = ev @ M @ M.T; return float(np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * (ev.shape[1] // 3))))


def zt(x): return (x - x.mean()) / (x.std() + 1e-9)


def fit_springs(ca, target_z, cutoff=13.0, steps=400):            # per-system oracle: springs -> diag(Gamma^+) ~ target
    I, J = edges(ca, cutoff); n = len(ca)
    Ii, Jj = torch.tensor(I), torch.tensor(J); tz = torch.tensor(target_z, dtype=torch.float32)
    s = torch.zeros(n, requires_grad=True); opt = torch.optim.Adam([s], lr=5e-2)
    for _ in range(steps):
        g = torch.exp((s[Ii] + s[Jj]) / 2); G = torch.zeros(n, n)
        G = G.index_put((Ii, Jj), -g, accumulate=True); G = G.index_put((Jj, Ii), -g, accumulate=True)
        G = G.index_put((Ii, Ii), g, accumulate=True); G = G.index_put((Jj, Jj), g, accumulate=True)
        d = torch.diagonal(torch.linalg.pinv(G)); dz = (d - d.mean()) / (d.std() + 1e-9)
        opt.zero_grad(); ((dz - tz) ** 2).mean().backward(); opt.step()
    return s.detach().numpy(), I, J


# ---------------- CHECK 1: GNM-vs-B correlation recalibration ----------------
print("[check1] plain GNM vs crystallographic B, Pearson (mean +/- 95% CI over structures)")
files = sorted(glob.glob(f"{BCACHE}/*.npz"))[:250]
for cutoff in (7.3, 10.0, 13.0):
    full, trim = [], []
    for fp in files:
        d = np.load(fp); ca = d["ca_xyz"].astype(np.float64); b = d["bfac"].astype(np.float64)
        if len(ca) < 30: continue
        dg = gnm_diag_np(ca, cutoff)
        if dg is None or dg.std() < 1e-9: continue
        full.append(np.corrcoef(dg, b)[0, 1])
        if len(ca) > 20:
            trim.append(np.corrcoef(dg[5:-5], b[5:-5])[0, 1])
    full, trim = np.array(full), np.array(trim)
    ci = 1.96 * full.std() / np.sqrt(len(full))
    print(f"  cutoff {cutoff:>4} A: full {full.mean():.3f}+-{ci:.3f} (n={len(full)})   termini-excluded {trim.mean():.3f}")

# ---------------- CHECK 2: the oracle ----------------
frozen = [x for x in open(FROZEN).read().split() if x]
print(f"\n[check2] ORACLE on frozen 7 (fit springs to fluctuation diagonal -> modes -> reconstruction A)")
print(f"  {'system':9s}{'nCA':>5}{'null':>7}{'ANM':>7}{'PCA':>7}{'oracleRMSF':>11}")
rows = []
for fp in sorted(glob.glob(f"{MCACHE}/*.npz")):
    dom = os.path.basename(fp)[:-4]
    if dom not in frozen: continue
    c0 = np.load(fp)["c0"].astype(np.float64); al = kabsch(c0); d = (al - al[0]).reshape(len(al), -1)
    n = c0.shape[1]; h = len(d) // 2; mean0 = d[:h].mean(0); ev = d[h:] - mean0
    rmsf = np.sqrt(((d[:h] - mean0) ** 2).reshape(h, n, 3).sum(-1).mean(0))
    I, J = edges(al[0], 13.0)
    null = float(np.sqrt((ev ** 2).sum() / (ev.shape[0] * n)))
    r_anm = resid(ev, anm_modes(al[0], np.zeros(n), I, J))       # uniform springs
    _, _, Vt = np.linalg.svd(d[:h] - mean0, full_matrices=False); r_pca = resid(ev, Vt[:L].T)
    s_fit, I2, J2 = fit_springs(al[0], zt(rmsf))
    r_orc = resid(ev, anm_modes(al[0], s_fit, I2, J2))
    rows.append((dom, n, null, r_anm, r_pca, r_orc))
    print(f"  {dom:9s}{n:>5}{null:>7.2f}{r_anm:>7.2f}{r_pca:>7.2f}{r_orc:>11.2f}")
a = np.array([r[2:] for r in rows])
mn = a.mean(0)
print(f"  {'MEAN':9s}{'':>5}{mn[0]:>7.2f}{mn[1]:>7.2f}{mn[2]:>7.2f}{mn[3]:>11.2f}")
print(f"\n  VERDICT: oracle-RMSF {mn[3]:.2f} vs ANM {mn[1]:.2f} -> "
      + ("HEADROOM: fitting the fluctuation diagonal beats uniform ANM -> train the map (oracle is the ceiling)"
         if mn[3] < mn[1] - 0.03 else
         "oracle >= ANM: the fluctuation DIAGONAL does not determine useful modes (many-to-one) -> "
         "B-factor supervision cannot work; CLOSE the track before training"))
