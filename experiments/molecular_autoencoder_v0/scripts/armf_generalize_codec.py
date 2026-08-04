"""A codec that GENERALISES across systems (the demo->model gap).

Per system, ANM modes (structure-only) are mixed by a SHARED learned map A (KxL)
into a predicted L-subspace, supervised toward the per-system PCA modes via a
SUBSPACE loss (projection-matrix Frobenius = L - ||M_pred^T M_pca||_F^2, sign- and
rotation-invariant). Trained on training systems, evaluated on HELD-OUT systems
(split by system) as absolute-A reconstruction vs per-system PCA (ceiling), ANM
(free baseline), and the zero-displacement null. CA level (ANM-natural; CA~=bb)."""
import glob, numpy as np, torch, torch.nn as nn
import h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
L, K = 64, 128


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


def anm(ca, K, cutoff=13.0):
    n = len(ca); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dv = ca - ca[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j <= i: continue
            u = dv[j] / dist[j]; b = np.outer(u, u)
            H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
            H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H)
    return V[:, 6:6 + K]                                        # (3n, K) orthonormal


def resid(ev, M):                                              # ev:(T,3n), M:(3n,L) orthonormal
    rec = ev @ M @ M.T
    return float(np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * (ev.shape[1] // 3))))


sys_data = []
for fp in sorted(glob.glob(f"{DATA}/*.h5")):
    try:
        f = h5py.File(fp, "r")
    except Exception:
        continue
    with f:
        dom = list(f.keys())[0]; g = f[dom]
        if "320" not in g or "0" not in g["320"]:
            continue
        z = np.array(g["z"]); N = len(z); names = parse_names(g, N)
        if len(names) != N:
            continue
        heavy = z != 1; nmh = names[heavy]; ca = nmh == "CA"
        if ca.sum() < K // 3 + 10:                              # need enough CA for K ANM modes
            continue
        coords = g["320"]["0"]["coords"][:].astype(np.float64)[:, heavy, :][:, ca, :]
    al = kabsch(coords, np.ones(ca.sum(), bool)); d = (al - al[0]).reshape(al.shape[0], -1)
    h = d.shape[0] // 2
    _, _, Vt = np.linalg.svd(d[:h] - d[:h].mean(0), full_matrices=False)
    Mpca = Vt[:L].T                                            # per-system PCA (fit first half)
    Manm = anm(al[0], K)
    sys_data.append(dict(dom=dom, n=ca.sum(), Mpca=Mpca, Manm=Manm, ev=d[h:]))
print(f"[generalize] {len(sys_data)} systems with >= enough CA")

sys_data.sort(key=lambda s: s["n"])
test = sys_data[2::4]; train = [s for s in sys_data if s not in test]       # size-spanning split
print(f"  train {len(train)} / test {len(test)} (split BY SYSTEM)")

A = torch.zeros(K, L, requires_grad=True)
with torch.no_grad():
    A[:L] = torch.eye(L)                                       # init: ANM top-L (the free baseline)
opt = torch.optim.Adam([A], lr=1e-2)
Tr = [(torch.tensor(s["Manm"], dtype=torch.float32), torch.tensor(s["Mpca"], dtype=torch.float32)) for s in train]
for ep in range(1500):
    opt.zero_grad(); loss = 0.0
    for Manm, Mpca in Tr:
        Q, _ = torch.linalg.qr(Manm @ A)                      # orthonormalise predicted modes
        loss = loss + (L - (Q.T @ Mpca).pow(2).sum())         # subspace loss (proj Frobenius)
    (loss / len(Tr)).backward(); opt.step()
An = A.detach().numpy()

print(f"\n=== HELD-OUT reconstruction (absolute A, L={L}, CA) ===")
print(f"  {'system':9s}{'n':>5}{'null':>7}{'PCA':>7}{'ANM':>7}{'learned':>8}{'gap closed':>11}")
rows = []
for s in test:
    ev = s["ev"]; null = float(np.sqrt((ev ** 2).sum() / (ev.shape[0] * s["n"])))
    r_pca = resid(ev, s["Mpca"]); r_anm = resid(ev, s["Manm"][:, :L])
    Q, _ = np.linalg.qr(s["Manm"] @ An); r_lrn = resid(ev, Q)
    gap = (r_anm - r_lrn) / (r_anm - r_pca) if r_anm > r_pca else float("nan")
    rows.append((r_pca, r_anm, r_lrn, gap))
    print(f"  {s['dom']:9s}{s['n']:>5}{null:>7.2f}{r_pca:>7.2f}{r_anm:>7.2f}{r_lrn:>8.2f}{gap:>10.0%}")
mp, ma, ml = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows])
gc = np.nanmean([r[3] for r in rows])
print(f"\n  MEAN: PCA {mp:.2f}A  ANM {ma:.2f}A  learned {ml:.2f}A  | ANM->PCA gap closed by learned: {gc:.0%}")
verdict = ("learned ~= PCA -> GENERALISATION SOLVED" if ml < mp + 0.05 else
           "learned <= ANM -> learning adds nothing; USE ANM as the general codec" if ml >= ma - 0.02 else
           f"ANM < learned < PCA -> learning adds value ({gc:.0%} of gap)")
print(f"  VERDICT: {verdict}")
