"""Torsional ANM (torsional network model) -- the torsional analogue of ANM for ligands.
Zero-parameter, structure-only, general by construction (no trajectory). Same protocol/metric
as the internal-coord oracle so it is directly comparable.

Build the Cartesian ANM Hessian H; the Jacobian J = dx/dphi (rotate each z-matrix torsion's
subtree about its bond axis); torsional Hessian K = J^T H J with metric G = J^T J; generalised
eigh(K,G) -> softest L torsional modes. Reconstruct: project the trajectory's torsional
displacement onto the modes, keep reference bonds/angles, NeRF -> Cartesian, absolute A.
Baselines on the same rows: null, Cartesian ANM, internal-coord PCA (the per-system ceiling).
Also report local-geometry validity (bond/angle deviation, min interatomic distance) WITHOUT
relaxation, for internal-coord-decoded structures.

Pre-registered read (armf_intcoord.md): tors-ANM ~= internal PCA (0.49) -> general ligand codec
solved zero-parameter (obj-1 codec blocker falls); between 0.49 and 0.77 -> partial, report gap
closed; ~= or worse than Cartesian ANM (0.77) -> structure alone insufficient, learned codec
needed (and we know why).
"""
import numpy as np, h5py
from scipy.linalg import eigh as geigh
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
LS = [4, 8, 16]
np.random.seed(0)


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc); dd = np.sign(np.linalg.det(Vt.T @ U.T))
    Pr = Pc @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T
    return np.sqrt(((Pr - Qc) ** 2).sum() / len(P))


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
        chain = []; x = parent[a]
        while x != -1 and len(chain) < 3: chain.append(x); x = parent.get(x, -1)
        for e in order:
            if len(chain) >= 3: break
            if e != a and e not in chain and pos[e] < pos[a]: chain.append(e)
        refs[a] = chain
    children = {a: [] for a in order}
    for a in order:
        if parent[a] != -1: children[parent[a]].append(a)
    return order, refs, parent, children


def subtree(a, children):
    out = []; st = [a]
    while st:
        u = st.pop(); out.append(u); st.extend(children[u])
    return out


def dih(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2; b1 = b1 / (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def raw_internal(xyz, order, refs):                              # bonds, angles, dihedrals (raw)
    bonds, angs, dihs = [], [], []
    for a in order:
        c = refs[a]
        if len(c) >= 1: bonds.append(np.linalg.norm(xyz[a] - xyz[c[0]]))
        if len(c) >= 2:
            u = xyz[a] - xyz[c[0]]; v = xyz[c[1]] - xyz[c[0]]
            angs.append(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))
        if len(c) >= 3: dihs.append(dih(xyz[a], xyz[c[0]], xyz[c[1]], xyz[c[2]]))
    return np.array(bonds), np.array(angs), np.array(dihs)


def nerf(c0, c1, c2, l, theta, chi):
    bc = c0 - c1; bc = bc / (np.linalg.norm(bc) + 1e-9)
    nvec = np.cross(c1 - c2, bc); nvec = nvec / (np.linalg.norm(nvec) + 1e-9); m = np.cross(nvec, bc)
    d2 = np.array([-l * np.cos(theta), l * np.sin(theta) * np.cos(chi), l * np.sin(theta) * np.sin(chi)])
    return c0 + d2[0] * bc + d2[1] * m + d2[2] * nvec


def build_cart(order, refs, bonds, angs, dihs):                 # internal -> Cartesian (in `order` ordering)
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


def jacobian(ref, order, refs, children):                       # dx/dphi for each torsion (atoms with 3 refs)
    n = len(ref); tors = [a for a in order if len(refs[a]) >= 3]; D = len(tors)
    idx = {a: i for i, a in enumerate(order)}                    # atom -> row-in-`order`
    J = np.zeros((3 * n, D))
    for t, a in enumerate(tors):
        c = refs[a]; axis = ref[idx[c[1]]] - ref[idx[c[0]]]; axis = axis / (np.linalg.norm(axis) + 1e-9)
        pivot = ref[idx[c[0]]]
        for k in subtree(a, children):
            J[3*idx[k]:3*idx[k]+3, t] = np.cross(axis, ref[idx[k]] - pivot)
    return J, tors


def pca_recon(dc_tr, dc_ev, L):
    mean = dc_tr.mean(0); _, _, Vt = np.linalg.svd(dc_tr - mean, full_matrices=False)
    M = Vt[:L]; return mean + (dc_ev - mean) @ M.T @ M


f = h5py.File(MD, "r"); allk = list(f.keys()); doms = allk[::max(1, len(allk)//12)][:12]
print("[torsional-ANM] zero-parameter torsional modes vs null / Cartesian ANM / internal-coord PCA")
print(f"  {'system':8s}{'nHvy':>5}{'nTors':>6}  " + "  ".join(f"L{L}(null/cANM/intPCA/torANM)" for L in LS)
      + "   bondDev/angDev/minD(torANM)")
agg = {L: [] for L in LS}
for dom in doms:
    g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); N = g["atoms_element"].shape[0]
    el = np.array(g["atoms_element"])[int(mbi[-1]):N]
    coords = np.array(g["trajectory_coordinates"])[:, int(mbi[-1]):N, :].astype(np.float64)
    hv = el != 1; coords = coords[:, hv, :]; n = hv.sum()
    if n < 8: continue
    adj, Dm = perceive_bonds(coords[0]); order, refs, parent, children = zmatrix(adj, Dm, n)
    if any(len(refs[a]) < 3 for a in order[3:]): continue
    co = coords[:, order, :]; T = len(co); h = T // 2
    ref = co[0]
    rb, ra, rd = raw_internal(coords[0], order, refs)           # internal coords from ORIGINAL-index coords
    rt = kabsch_rmsd(build_cart(order, refs, rb, ra, rd), co[0])   # round-trip sanity (must be ~0)
    # torsional modes (structure only, on the reference)
    H = anm_hessian(ref); J, tors = jacobian(ref, order, refs, children)
    K = J.T @ H @ J; G = J.T @ J + 1e-8 * np.eye(J.shape[1])
    w, V = geigh(K, G)                                           # ascending; softest = smallest
    # trajectory torsional displacement (wrapped)
    Phi = np.array([raw_internal(coords[t], order, refs)[2] for t in range(T)])   # (T, D) from original coords
    dphi = ((Phi - rd + np.pi) % (2 * np.pi)) - np.pi           # wrapped displacement
    # internal-coord PCA baseline (full internal, sin/cos) for the same rows
    def full_ic(x):
        b, a2, d2 = raw_internal(x, order, refs); return np.concatenate([b, a2, np.sin(d2), np.cos(d2)])
    IC = np.array([full_ic(coords[t]) for t in range(T)])
    Manm = None
    dcart = (co - ref).reshape(T, -1); mc = dcart[:h].mean(0); ev = dcart[h:]
    # cartesian ANM modes
    wA, VA = np.linalg.eigh(H); Mcart = VA[:, 6:6 + max(LS)]
    null = np.mean([kabsch_rmsd(ref, co[h + i]) for i in range(len(ev))])
    cells = []; bd = ad = md = 0.0
    for L in LS:
        Le = min(L, len(tors))
        M = V[:, :Le]                                            # softest L torsional modes (G-orthonormal)
        c = dphi[h:] @ (G @ M); dphi_rec = c @ M.T              # project + reconstruct torsional displacement
        tor_rmsd, bdv, adv, mdv = [], [], [], []
        for i in range(len(dphi_rec)):
            xr = build_cart(order, refs, rb, ra, rd + dphi_rec[i])   # ref bonds/angles + recon torsions
            tor_rmsd.append(kabsch_rmsd(xr, co[h + i]))
            b2, a2, _ = raw_internal(xr, order, refs)
            bdv.append(np.abs(b2 - rb).mean()); adv.append(np.degrees(np.abs(a2 - ra)).mean())
            DD = np.sqrt(((xr[:, None] - xr[None]) ** 2).sum(-1)); np.fill_diagonal(DD, 9); mdv.append(DD.min())
        cart_anm = np.mean([kabsch_rmsd((ref + (mc + (ev[i] - mc) @ Mcart[:, :Le] @ Mcart[:, :Le].T).reshape(n, 3)), co[h + i]) for i in range(len(ev))])
        ic_rec = pca_recon(IC[:h], IC[h:], min(L, IC.shape[1]))
        nb = len(rb); na = len(ra); nd = len(rd)
        int_pca = np.mean([kabsch_rmsd(build_cart(order, refs, ic_rec[i][:nb], ic_rec[i][nb:nb+na],
                          np.arctan2(ic_rec[i][nb+na:nb+na+nd], ic_rec[i][nb+na+nd:])), co[h + i]) for i in range(len(ic_rec))])
        tr = np.mean(tor_rmsd); cells.append((null, cart_anm, int_pca, tr)); agg[L].append((cart_anm, int_pca, tr))
        if L == LS[-1]: bd, ad, md = np.mean(bdv), np.mean(adv), np.mean(mdv)
    cs = "  ".join(f"{c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}/{c[3]:.2f}" for c in cells)
    print(f"  {dom:8s}{n:>5}{len(tors):>6} rt{rt:.2f}  {cs}   {bd:.3f}/{ad:.1f}/{md:.2f}")

print("\n=== MEAN over ligands (absolute A, held-out) ===")
for L in LS:
    a = np.array(agg[L])
    if len(a) == 0: continue
    ca_, ip, tr = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
    gap = (ca_ - tr) / (ca_ - ip) if ca_ > ip else float('nan')
    print(f"  L={L}: Cartesian ANM {ca_:.2f}  internal-coord PCA {ip:.2f}  TORSIONAL-ANM {tr:.2f}  (closes {gap:.0%} of the cANM->intPCA gap)")
print("\n  read: torsional-ANM ~= internal PCA -> general ligand codec zero-parameter (obj-1 falls);")
print("        between -> partial (gap % = headroom for a learned codec); ~= Cartesian ANM -> structure insufficient.")
