"""GATE: is per-system PCA a structural property, or trajectory-specific noise?

mdCATH ships 5 INDEPENDENT replicas per domain at the same temperature. Fit PCA-64
on replica 0, reconstruct replica 1's trajectory (absolute A, held-out across an
independent run of the SAME molecule) and compare against PCA fit on replica 1
itself (the within-trajectory ~1 A ceiling), ANM, and the null. Also report the
principal-angle subspace overlap between the two replicas' L=64 mode sets.

cross ~= within -> modes are structural; cross-molecule is the only problem.
cross >> within, near ANM -> per-system PCA modes are trajectory-specific; a HARD
upper bound on any structure->modes codec, and the 0.99 A gate was a demo.

Backbone target (matches the step-2 gate) AND CA (matches the codec)."""
import glob, numpy as np, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
L, K = 64, 128
BB = ["N", "CA", "C", "O"]
TEMP, R0, R1 = "320", "0", "1"


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


def anm(xyz, K, cutoff):
    n = len(xyz); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dv = xyz - xyz[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j <= i: continue
            u = dv[j] / dist[j]; b = np.outer(u, u)
            H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
            H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H)
    return V[:, 6:6 + K]                                             # (3n, K) orthonormal


def disp_mc(coords, align_mask):
    al = kabsch(coords, align_mask)
    d = (al - al[0]).reshape(al.shape[0], -1)
    return d - d.mean(0), al[0]                                      # mean-removed disp, ref frame


def pca(dc):
    _, _, Vt = np.linalg.svd(dc, full_matrices=False)
    return Vt[:L].T                                                  # (3n, L) orthonormal


def resid(dc, M):                                                    # absolute A / atom
    rec = dc @ M @ M.T
    return float(np.sqrt(((rec - dc) ** 2).sum() / (dc.shape[0] * (dc.shape[1] // 3))))


# ---- gather domains with BOTH replicas present, spanning size ----
cand = []
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
        heavy = z != 1; nmh = names[heavy]
        if (nmh == "CA").sum() < K // 3 + 10:
            continue
        cand.append((fp, dom, int((nmh == "CA").sum())))
cand.sort(key=lambda c: c[2])
pick = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, min(8, len(cand))).round().astype(int))]
print(f"[cross-replica] {len(cand)} domains have both replicas; testing {len(pick)} spanning n_CA "
      f"{pick[0][2]}..{pick[-1][2]}")

for level in ("CA", "backbone"):
    cutoff = 13.0 if level == "CA" else 10.0
    print(f"\n=== {level} target (ANM cutoff {cutoff} A, L={L}) ===")
    print(f"  {'domain':9s}{'nCA':>5}{'null':>7}{'within':>8}{'cross':>7}{'ANM':>7}{'overlap':>9}")
    rows = []
    for fp, dom, nca in pick:
        try:
            with h5py.File(fp, "r") as f:
                g = f[dom]; z = np.array(g["z"]); N = len(z); names = parse_names(g, N)
                heavy = z != 1; nmh = names[heavy]
                sel = (nmh == "CA") if level == "CA" else np.isin(nmh, BB)
                amask = np.ones(sel.sum(), bool) if level == "CA" else (nmh[sel] == "CA")
                c0 = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, sel, :]
                c1 = g[TEMP][R1]["coords"][:].astype(np.float64)[:, heavy, :][:, sel, :]
        except Exception:
            continue
        dc0, ref0 = disp_mc(c0, amask); dc1, _ = disp_mc(c1, amask)
        M0, M1 = pca(dc0), pca(dc1)
        ca_ref = ref0[amask] if level == "backbone" else ref0
        Manm = anm(ca_ref, K, cutoff) if level == "CA" else anm(ref0.reshape(-1, 3), K, cutoff)
        null = float(np.sqrt((dc1 ** 2).sum() / (dc1.shape[0] * (dc1.shape[1] // 3))))
        within = resid(dc1, M1)                                      # fit-on-rep1 ceiling
        cross = resid(dc1, M0)                                       # rep0 modes on rep1 (held-out)
        r_anm = resid(dc1, Manm[:, :L])
        overlap = float(((M0.T @ M1) ** 2).sum() / L)               # mean cos^2 principal angle
        rows.append((null, within, cross, r_anm, overlap))
        print(f"  {dom:9s}{nca:>5}{null:>7.2f}{within:>8.2f}{cross:>7.2f}{r_anm:>7.2f}{overlap:>8.2f}")
    a = np.array(rows)
    mn = a.mean(0); sd = a.std(0)
    print(f"  {'MEAN':9s}{'':>5}{mn[0]:>7.2f}{mn[1]:>8.2f}{mn[2]:>7.2f}{mn[3]:>7.2f}{mn[4]:>8.2f}")
    print(f"  {'SD':9s}{'':>5}{sd[0]:>7.2f}{sd[1]:>8.2f}{sd[2]:>7.2f}{sd[3]:>7.2f}{sd[4]:>8.2f}")
    within_m, cross_m, anm_m = mn[1], mn[2], mn[3]
    if cross_m < within_m + 0.15:
        v = "cross ~= within -> modes ARE structural; cross-molecule is the only problem. PROCEED to learning curve."
    elif cross_m > anm_m - 0.15:
        v = "cross >> within, NEAR ANM -> per-system PCA modes are TRAJECTORY-SPECIFIC; hard upper bound. STOP."
    else:
        v = f"cross between within and ANM (within {within_m:.2f} < cross {cross_m:.2f} < ANM {anm_m:.2f}); partial reproducibility."
    print(f"  VERDICT [{level}]: {v}")
