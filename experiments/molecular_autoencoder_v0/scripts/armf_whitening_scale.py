"""Is the sqrt(kT/lambda) whitening offset just the arbitrary spring-constant units?

ANM sets all springs to 1, so lambda carries an arbitrary overall stiffness and
kT/lambda is correct only up to a GLOBAL multiplicative constant. The model predicts
only the RELATIVE per-mode variance. Test: ratio = ref.std / sqrt(kT/lambda) per mode,
per system. Report CV across MODES and across SYSTEMS separately -- those pick the branch:
  constant across modes AND systems -> arbitrary spring scale; ONE global number, zero
    per-system params -> sqrt(kT/lambda) whitening general -> withdraw 'no general calibration'.
  constant across modes, varies across systems -> one SCALAR per system; test if predictable
    from structure (N, contact density, mean lambda, Rg).
  varies WITH MODE INDEX -> ANM's relative variance prediction is genuinely wrong (real error).
No DDPM -- pure codec-level quantity on trajectories in hand.
"""
import glob, numpy as np, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"
BB = ["N", "CA", "C", "O"]; TEMP, R0, kT, L = "320", "0", 0.593, 64


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
            if j > i: I.append(i); J.append(j); U.append(dv[j] / dd[j])
    I, J, U = np.array(I), np.array(J), np.array(U); b = U[:, :, None] * U[:, None, :]
    def sc(a, c, V):
        r = (3*a[:, None, None] + ar[None, :, None]) * np.ones((1, 1, 3), int)
        co = (3*c[:, None, None] + ar[None, None, :]) * np.ones((1, 3, 1), int)
        np.add.at(H, (r.ravel(), co.ravel()), V.ravel())
    sc(I, I, b); sc(J, J, b); sc(I, J, -b); sc(J, I, -b)
    w, V = np.linalg.eigh(H); return V[:, 6:6 + K], w[6:6 + K], len(I)


frozen = [x for x in open(FROZEN).read().split() if x]
allfiles = sorted(glob.glob(f"{DATA}/*.h5"))
doms = []
for fp in allfiles:
    try:
        with h5py.File(fp, "r") as f:
            dom = list(f.keys())[0]
        doms.append((fp, dom))
    except Exception:
        pass
# frozen 7 + first 5 non-frozen, for an across-system estimate
use = [(fp, d) for fp, d in doms if d in frozen] + [(fp, d) for fp, d in doms if d not in frozen][:5]

print("[whitening-scale] ratio = ref.std / sqrt(kT/lambda) per mode (target: units constant)")
print(f"  {'system':9s}{'nCA':>5}{'meanRatio':>10}{'CVmodes':>8}{'slow':>7}{'mid':>7}{'fast':>7}{'meanLam':>8}")
rows = []
for fp, dom in use:
    try:
        with h5py.File(fp, "r") as f:
            g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
            if len(nm) != N: continue
            heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
            if ca.sum() < 30: continue
            coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    except Exception:
        continue
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); h = len(d) // 2
    modes, lam, _ = anm(al[0], L); lam = lam[:L]
    Z = (d - (d[:h].mean(0))) @ modes[:, :L]                     # (T, L) coefficients
    ref_std = Z[:h].std(0); sd = np.sqrt(kT / np.clip(lam, 1e-8, None))
    ratio = ref_std / sd                                          # target: constant if only units differ
    th = L // 3
    cvm = ratio.std() / ratio.mean()
    rows.append((dom, ca.sum(), ratio.mean(), cvm, ratio[:th].mean(), ratio[th:2*th].mean(), ratio[2*th:].mean(), lam.mean(), ratio))
    print(f"  {dom:9s}{ca.sum():>5}{ratio.mean():>10.2f}{cvm:>8.2f}{ratio[:th].mean():>7.2f}"
          f"{ratio[th:2*th].mean():>7.2f}{ratio[2*th:].mean():>7.2f}{lam.mean():>8.3f}")

means = np.array([r[2] for r in rows]); cvm_all = np.array([r[3] for r in rows])
cv_across_sys = means.std() / means.mean()
# mode-index tilt averaged across systems (all L=64)
Rm = np.stack([r[8] for r in rows]) / means[:, None]             # normalise each system by its own mean, then compare shape
slow, mid, fast = Rm[:, :L//3].mean(), Rm[:, L//3:2*L//3].mean(), Rm[:, 2*L//3:].mean()
print(f"\n  CV ACROSS MODES (mean over systems): {cvm_all.mean():.2f}")
print(f"  CV ACROSS SYSTEMS (of per-system mean ratio): {cv_across_sys:.2f}  (means {means.min():.2f}..{means.max():.2f})")
print(f"  mode-index shape (each system normalised to its own mean): slow {slow:.2f} / mid {mid:.2f} / fast {fast:.2f}")
if cvm_all.mean() < 0.15 and cv_across_sys < 0.15:
    v = "CONSTANT across modes AND systems -> arbitrary spring scale; ONE global constant -> GENERAL, withdraw 'no general calibration'."
elif cvm_all.mean() < 0.15:
    v = "constant across modes, VARIES across systems -> one scalar per system; test if predictable from structure (N, mean lambda)."
else:
    v = "VARIES WITH MODE INDEX -> ANM's relative variance prediction is genuinely wrong (real model error)."
print(f"  VERDICT: {v}")
