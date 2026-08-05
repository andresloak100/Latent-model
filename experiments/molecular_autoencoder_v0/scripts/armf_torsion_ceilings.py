"""Two ceilings before building the ligand torsion codec (armf_intcoord.md pre-reg).

CEILING 1 -- TORSION-ONLY PCA (per-system): PCA on dihedrals only (sin/cos), reconstruct with
reference bonds/angles. The true ceiling for ANY torsion-based method. Plus the VARIANCE SPLIT:
Cartesian amplitude of bond-only / angle-only / dihedral-only motion.
CEILING 2 -- PER-TORSION-SOFTNESS ORACLE: fit per-torsion softness k PER SYSTEM (torch, through
eigh) so the softest L generalised modes of (diag(k), G=J^T J) reconstruct the trajectory's
torsional displacement; eval Cartesian. Ceiling for what a learned softness map could predict.

Reads: torsion-only PCA ~0.49 -> torsions carry the signal. softness oracle ~= torsion-only PCA
-> softness is the right knob, build the learned map; falls short -> full graph codec.
Same ligands/frames/L/metric (Cartesian A after round-trip) as the oracle."""
import numpy as np, torch, h5py
from scipy.linalg import eigh as geigh
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
LS = [4, 8, 16]; np.random.seed(0); torch.manual_seed(0)


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc); dd = np.sign(np.linalg.det(Vt.T @ U.T))
    return np.sqrt((((Pc @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T) - Qc) ** 2).sum() / len(P))


def perceive_bonds(ref):
    n = len(ref); adj = [[] for _ in range(n)]; D = np.sqrt(((ref[:, None] - ref[None]) ** 2).sum(-1))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < 1.8: adj[i].append(j); adj[j].append(i)
    return adj, D


def zmatrix(adj, D, n):
    import collections
    order = [0]; parent = {0: -1}; seen = {0}; q = collections.deque([0])
    while q:
        u = q.popleft()
        for v in sorted(adj[u], key=lambda x: D[u, x]):
            if v not in seen: seen.add(v); parent[v] = u; order.append(v); q.append(v)
    for v in range(n):
        if v not in seen:
            nearest = min(order, key=lambda a: D[v, a]); parent[v] = nearest; order.append(v); seen.add(v)
    pos = {a: i for i, a in enumerate(order)}; refs = {}
    for a in order:
        ch = []; x = parent[a]
        while x != -1 and len(ch) < 3: ch.append(x); x = parent.get(x, -1)
        for e in order:
            if len(ch) >= 3: break
            if e != a and e not in ch and pos[e] < pos[a]: ch.append(e)
        refs[a] = ch
    children = {a: [] for a in order}
    for a in order:
        if parent[a] != -1: children[parent[a]].append(a)
    return order, refs, parent, children


def subtree(a, children):
    out = []; st = [a]
    while st: u = st.pop(); out.append(u); st.extend(children[u])
    return out


def dih(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2; b1 = b1 / (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def raw_internal(xyz, order, refs):
    bonds, angs, dihs = [], [], []
    for a in order:
        c = refs[a]
        if len(c) >= 1: bonds.append(np.linalg.norm(xyz[a] - xyz[c[0]]))
        if len(c) >= 2:
            u = xyz[a] - xyz[c[0]]; v = xyz[c[1]] - xyz[c[0]]
            angs.append(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))
        if len(c) >= 3: dihs.append(dih(xyz[a], xyz[c[0]], xyz[c[1]], xyz[c[2]]))
    return np.array(bonds), np.array(angs), np.array(dihs)


def nerf(c0, c1, c2, l, th, ch):
    bc = c0 - c1; bc = bc / (np.linalg.norm(bc) + 1e-9)
    nv = np.cross(c1 - c2, bc); nv = nv / (np.linalg.norm(nv) + 1e-9); m = np.cross(nv, bc)
    d2 = np.array([-l * np.cos(th), l * np.sin(th) * np.cos(ch), l * np.sin(th) * np.sin(ch)])
    return c0 + d2[0] * bc + d2[1] * m + d2[2] * nv


def build_cart(order, refs, bonds, angs, dihs):
    xyz = {}; bi = ai = di = 0
    for k, a in enumerate(order):
        c = refs[a]
        if k == 0: xyz[a] = np.zeros(3)
        elif k == 1: xyz[a] = np.array([bonds[bi], 0, 0.]); bi += 1
        elif k == 2:
            l = bonds[bi]; bi += 1; th = angs[ai]; ai += 1
            u = xyz[c[1]] - xyz[c[0]]; u = u / (np.linalg.norm(u) + 1e-9); perp = np.array([-u[1], u[0], 0.])
            xyz[a] = xyz[c[0]] + l * (np.cos(th) * u + np.sin(th) * perp)
        else:
            l = bonds[bi]; bi += 1; th = angs[ai]; ai += 1; ch = dihs[di]; di += 1
            xyz[a] = nerf(xyz[c[0]], xyz[c[1]], xyz[c[2]], l, th, ch)
    return np.array([xyz[a] for a in order])


def anm_hessian(ca, cutoff=8.0):
    n = len(ca); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dv = ca - ca[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j <= i: continue
            u = dv[j] / dist[j]; b = np.outer(u, u)
            H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
            H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    return H


def jacobian(refx, order, refs, children, pos):                 # refx = coords in ORIGINAL index; rows in `order` position
    n = len(order); tors = [a for a in order if len(refs[a]) >= 3]; J = np.zeros((3 * n, len(tors)))
    for t, a in enumerate(tors):
        c = refs[a]; axis = refx[c[1]] - refx[c[0]]; axis = axis / (np.linalg.norm(axis) + 1e-9)
        for k in subtree(a, children):
            J[3*pos[k]:3*pos[k]+3, t] = np.cross(axis, refx[k] - refx[c[0]])
    return J, tors


def pca_recon(tr, ev, L):
    mean = tr.mean(0); _, _, Vt = np.linalg.svd(tr - mean, full_matrices=False)
    M = Vt[:L]; return mean + (ev - mean) @ M.T @ M


def softness_oracle(dphi_tr, G, L, steps=250):                  # fit per-torsion k, softest L modes reconstruct dphi
    Gt = torch.tensor(G, dtype=torch.float64); D = G.shape[0]
    Lc = torch.linalg.cholesky(Gt + 1e-8 * torch.eye(D, dtype=torch.float64)); Linv = torch.linalg.inv(Lc)
    dt = torch.tensor(dphi_tr, dtype=torch.float64); theta = torch.zeros(D, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    for _ in range(steps):
        k = torch.nn.functional.softplus(theta) + 1e-4
        A = Linv @ torch.diag(k) @ Linv.T; A = 0.5 * (A + A.T)
        w, U = torch.linalg.eigh(A); M = Linv.T @ U[:, :L]       # softest L (G-orthonormal)
        c = dt @ (Gt @ M); rec = c @ M.T
        opt.zero_grad(); ((rec - dt) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        k = torch.nn.functional.softplus(theta) + 1e-4; A = Linv @ torch.diag(k) @ Linv.T
        w, U = torch.linalg.eigh(0.5 * (A + A.T)); M = (Linv.T @ U[:, :L]).numpy()
    return M                                                     # (D, L) G-orthonormal torsional modes


f = h5py.File(MD, "r"); allk = list(f.keys()); doms = allk[::max(1, len(allk)//12)][:12]
print("[torsion-ceilings] torsion-only PCA + per-torsion-softness oracle; variance split")
print(f"  {'system':8s}{'nTors':>6}  " + "  ".join(f"L{L}(cANM/intPCA/torPCA/softOracle)" for L in LS)
      + "   var%(bond/ang/dih)")
agg = {L: [] for L in LS}
for dom in doms:
    g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); N = g["atoms_element"].shape[0]
    el = np.array(g["atoms_element"])[int(mbi[-1]):N]
    coords = np.array(g["trajectory_coordinates"])[:, int(mbi[-1]):N, :].astype(np.float64)
    hv = el != 1; coords = coords[:, hv, :]; n = hv.sum()
    if n < 8: continue
    adj, Dm = perceive_bonds(coords[0]); order, refs, parent, children = zmatrix(adj, Dm, n)
    if any(len(refs[a]) < 3 for a in order[3:]): continue
    pos = {a: i for i, a in enumerate(order)}; co = coords[:, order, :]; T = len(co); h = T // 2
    rb, ra, rd = raw_internal(coords[0], order, refs)
    RI = [raw_internal(coords[t], order, refs) for t in range(T)]
    Bo = np.array([x[0] for x in RI]); An = np.array([x[1] for x in RI]); Ph = np.array([x[2] for x in RI])
    dphi = ((Ph - rd + np.pi) % (2 * np.pi)) - np.pi
    null = np.mean([kabsch_rmsd(co[0], co[h + i]) for i in range(T - h)])
    # variance split: Cartesian amplitude of each DOF group varying alone
    def amp(bg, ag, dg): return np.mean([kabsch_rmsd(build_cart(order, refs, bg[t], ag[t], dg[t]), co[0]) for t in range(h, T)])
    a_b = amp(Bo, np.tile(ra, (T, 1)), np.tile(rd, (T, 1))); a_a = amp(np.tile(rb, (T, 1)), An, np.tile(rd, (T, 1)))
    a_d = amp(np.tile(rb, (T, 1)), np.tile(ra, (T, 1)), Ph); s = a_b + a_a + a_d + 1e-9
    # torsion PCA (sin/cos) + Cartesian ANM + softness oracle
    H = anm_hessian(co[0]); wA, VA = np.linalg.eigh(H); Mcart = VA[:, 6:6 + max(LS)]
    J, tors = jacobian(coords[0], order, refs, children, pos); G = J.T @ J + 1e-8 * np.eye(len(tors))
    dc = (co - co[0]).reshape(T, -1); mc = dc[:h].mean(0)
    SC = np.concatenate([np.sin(Ph), np.cos(Ph)], 1); nd = Ph.shape[1]
    cells = []
    for L in LS:
        Le = min(L, len(tors))
        sc_rec = pca_recon(SC[:h], SC[h:], min(L, 2 * nd))
        tor_pca = np.mean([kabsch_rmsd(build_cart(order, refs, rb, ra, np.arctan2(sc_rec[i][:nd], sc_rec[i][nd:])), co[h + i]) for i in range(T - h)])
        cart_anm = np.mean([kabsch_rmsd((co[0] + (mc + (dc[h + i] - mc) @ Mcart[:, :Le] @ Mcart[:, :Le].T).reshape(n, 3)), co[h + i]) for i in range(T - h)])
        # internal PCA (all internal) reference
        IC = np.concatenate([Bo, An, np.sin(Ph), np.cos(Ph)], 1); ic_rec = pca_recon(IC[:h], IC[h:], min(L, IC.shape[1]))
        nb = len(rb); na = len(ra)
        int_pca = np.mean([kabsch_rmsd(build_cart(order, refs, ic_rec[i][:nb], ic_rec[i][nb:nb+na], np.arctan2(ic_rec[i][nb+na:nb+na+nd], ic_rec[i][nb+na+nd:])), co[h + i]) for i in range(T - h)])
        M = softness_oracle(dphi[:h], G, Le); c = dphi[h:] @ (G @ M); dphi_rec = c @ M.T
        soft = np.mean([kabsch_rmsd(build_cart(order, refs, rb, ra, rd + dphi_rec[i]), co[h + i]) for i in range(T - h)])
        cells.append((cart_anm, int_pca, tor_pca, soft)); agg[L].append((cart_anm, int_pca, tor_pca, soft))
    cs = "  ".join(f"{c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}/{c[3]:.2f}" for c in cells)
    print(f"  {dom:8s}{len(tors):>6}  {cs}   {a_b/s:.0%}/{a_a/s:.0%}/{a_d/s:.0%}")

print("\n=== MEAN over ligands (absolute A, held-out) ===")
for L in LS:
    a = np.array(agg[L])
    print(f"  L={L}: Cartesian ANM {a[:,0].mean():.2f}  internal PCA {a[:,1].mean():.2f}  TORSION-ONLY PCA {a[:,2].mean():.2f}  SOFTNESS-ORACLE {a[:,3].mean():.2f}")
print("\n  read: torsion-only PCA ~0.49 -> torsions carry signal (else bonds/angles do). softness-oracle ~= torsion")
print("        PCA -> softness is the right knob, BUILD the learned map; falls short -> full graph codec.")
