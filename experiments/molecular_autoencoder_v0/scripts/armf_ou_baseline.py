"""STEP 2: the propagator's ANM -- an overdamped Ornstein-Uhlenbeck baseline.

The codec lesson: a zero-parameter physics baseline beat every learned alternative.
The propagator has the exact analogue. In the ANM mode basis each mode is an
independent overdamped OU process:
    stationary variance  v_i = kT / lambda_i
    lag-1 AR coefficient a_i = exp(-lambda_i * dt / gamma)   (relaxation time gamma/lambda_i)
ONE free parameter (friction gamma), fit to the reference lag-1 autocorrelations;
per-mode gamma_i spread measures how non-Markovian the real dynamics are. Sampling
is free. Run through the SAME ensemble acceptance test as the DDPM.

Pre-registered: OU PASSES marginals + autocorrelation near-trivially, FAILS basin
coverage (Gaussian, single-basin). If OU fails marginals/IAT, the test or whitening
is wrong. If OU passes basins too, the reference is single-basin at this lag and the
benchmark is too easy -- say so.
"""
import numpy as np, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
BB = ["N", "CA", "C", "O"]; TEMP, R0 = "320", "0"
L, HORIZON, kT = 64, 2000, 0.593
USE = ["3a5zD02", "3jvvA01", "3a9lA00"]
np.random.seed(0)


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


def iat(x, cutoff=0.05, maxlag=400):
    x = x - x.mean(); v = (x * x).mean()
    if v < 1e-12: return 0.0
    acf = np.correlate(x, x, "full")[len(x) - 1:] / (v * len(x)); t = 1.0
    for k in range(1, min(len(acf), maxlag)):
        if acf[k] < cutoff: break
        t += 2 * acf[k]
    return t


def js_2d(gen, ref, bins=40):
    lo = np.minimum(gen.min(0), ref.min(0)); hi = np.maximum(gen.max(0), ref.max(0))
    edges = [np.linspace(lo[i], hi[i], bins + 1) for i in range(2)]
    Pg = np.histogram2d(gen[:, 0], gen[:, 1], bins=edges)[0] / len(gen) + 1e-12
    Pr = np.histogram2d(ref[:, 0], ref[:, 1], bins=edges)[0] / len(ref) + 1e-12
    M = 0.5 * (Pg + Pr); kl = lambda a, b: (a * np.log(a / b)).sum()
    ro = Pr > 1.5e-12; cover = ((Pg > 1.5e-12) & ro).sum() / ro.sum()
    return float(0.5 * kl(Pg, M) + 0.5 * kl(Pr, M)), float(cover)


print(f"[OU] overdamped Ornstein-Uhlenbeck in the ANM basis, one friction gamma; same acceptance test")
print(f"  {'system':9s}{'gamma':>8}{'g_spread':>9}{'stdRatio':>9}{'IATou':>7}{'IATref':>8}{'IATrat':>8}{'JS':>7}{'cover':>7}")
for dom in USE:
    with h5py.File(f"{DATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
        coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); T = d.shape[0]
    h = int(T * 0.8); mean = d[:h].mean(0)
    modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d - mean) @ B.T                                          # reference coefficients (raw)
    zt_, zn_ = Z[:h - 1], Z[1:h]
    a_ref = (zt_ * zn_).mean(0) / ((zt_ ** 2).mean(0) + 1e-9)     # reference lag-1 AR per mode
    a_clip = np.clip(a_ref, 0.02, 0.98)
    gamma_i = -lam / np.log(a_clip)                              # per-mode friction (dt=1 frame)
    gamma = float(np.median(gamma_i[np.isfinite(gamma_i) & (gamma_i > 0)]))
    a_ou = np.exp(-lam / gamma)                                  # OU AR coeff with the single global gamma
    v = kT / np.clip(lam, 1e-8, None)                            # structure variance
    v *= (Z[:h].var(0).sum() / v.sum())                         # one global scale to reference total variance
    # generate OU rollout
    x = Z[h].copy(); roll = [x.copy()]
    for _ in range(HORIZON):
        x = a_ou * x + np.sqrt(v * (1 - a_ou ** 2)) * np.random.randn(L); roll.append(x.copy())
    gen = np.array(roll); ref = Z
    sr = gen.std(0) / (ref.std(0) + 1e-9)
    ig = np.array([iat(gen[:, i]) for i in range(L)]); ir = np.array([iat(ref[:, i]) for i in range(L)])
    top2 = np.argsort(ref.std(0))[::-1][:2]; js, cover = js_2d(gen[:, top2], ref[:, top2])
    gspread = float(np.percentile(gamma_i[np.isfinite(gamma_i) & (gamma_i > 0)], 90) /
                    (np.percentile(gamma_i[np.isfinite(gamma_i) & (gamma_i > 0)], 10) + 1e-9))
    print(f"  {dom:9s}{gamma:>8.1f}{gspread:>9.1f}{sr.mean():>9.2f}{ig.mean():>7.1f}{ir.mean():>8.1f}"
          f"{ig.mean()/max(ir.mean(),1e-6):>8.2f}{js:>7.3f}{cover:>7.0%}")
print("\n  read: OU should PASS marginals (stdRatio~1) + IAT (~1) and FAIL basin coverage (Gaussian,")
print("        single-basin). gamma_spread >> 1 => the real dynamics are non-Markovian (one gamma is a")
print("        compromise). Any learned propagator must BEAT OU on basin coverage to earn its cost.")
