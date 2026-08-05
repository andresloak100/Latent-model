"""Internal-coordinate codec ORACLE (objective 1). Does a torsional (internal-coordinate)
representation beat Cartesian ANM on MISATO ligands? PCA the internal-coord trajectory,
NeRF back to Cartesian, compare to null / Cartesian ANM / Cartesian PCA. Oracle, not a model.
Pre-registered read in armf_intcoord.md."""
import numpy as np, h5py
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
LS = [4, 8, 16]
np.random.seed(0)


def kabsch_rmsd(P, Q):                                            # RMSD of P onto Q (both (n,3)), per atom
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc); dd = np.sign(np.linalg.det(Vt.T @ U.T))
    Pr = Pc @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T
    return np.sqrt(((Pr - Qc) ** 2).sum() / len(P))


def perceive_bonds(ref):
    n = len(ref); adj = [[] for _ in range(n)]
    D = np.sqrt(((ref[:, None] - ref[None]) ** 2).sum(-1))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < 1.8: adj[i].append(j); adj[j].append(i)
    return adj, D


def zmatrix(adj, D, n):                                          # BFS order + 3 earlier refs per atom
    import collections
    root = 0; order = [root]; parent = {root: -1}; seen = {root}
    q = collections.deque([root])
    while q:
        u = q.popleft()
        for v in sorted(adj[u], key=lambda x: D[u, x]):
            if v not in seen: seen.add(v); parent[v] = u; order.append(v); q.append(v)
    # atoms not reached (disconnected): attach to nearest placed atom
    for v in range(n):
        if v not in seen:
            placed = [a for a in order]; nearest = min(placed, key=lambda a: D[v, a])
            parent[v] = nearest; order.append(v); seen.add(v)
    pos = {a: i for i, a in enumerate(order)}
    refs = {}
    for a in order:
        chain = []; x = parent[a]
        while x != -1 and len(chain) < 3:
            chain.append(x); x = parent.get(x, -1)
        # pad with earliest placed atoms distinct from a and chain
        for e in order:
            if len(chain) >= 3: break
            if e != a and e not in chain and pos[e] < pos[a]: chain.append(e)
        refs[a] = chain
    return order, refs


def dihedral(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2; b1 /= np.linalg.norm(b1) + 1e-9
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def to_internal(xyz, order, refs):                              # -> [bonds, angles, sin(dih), cos(dih)]
    bonds, angs, sind, cosd = [], [], [], []
    for a in order:
        c = refs[a]
        if len(c) >= 1:
            bonds.append(np.linalg.norm(xyz[a] - xyz[c[0]]))
        if len(c) >= 2:
            u = xyz[a] - xyz[c[0]]; v = xyz[c[1]] - xyz[c[0]]
            angs.append(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))
        if len(c) >= 3:
            ch = dihedral(xyz[a], xyz[c[0]], xyz[c[1]], xyz[c[2]]); sind.append(np.sin(ch)); cosd.append(np.cos(ch))
    return np.concatenate([bonds, angs, sind, cosd])


def nerf(c0, c1, c2, l, theta, chi):                            # place atom given refs c0(parent),c1,c2
    bc = c0 - c1; bc /= np.linalg.norm(bc) + 1e-9
    n = np.cross(c1 - c2, bc); n /= np.linalg.norm(n) + 1e-9
    m = np.cross(n, bc)
    d2 = np.array([-l * np.cos(theta), l * np.sin(theta) * np.cos(chi), l * np.sin(theta) * np.sin(chi)])
    return c0 + d2[0] * bc + d2[1] * m + d2[2] * n


def from_internal(vec, order, refs, ref0):                      # NeRF reconstruct Cartesian
    n = len(order); nb = sum(1 for a in order if len(refs[a]) >= 1)
    na = sum(1 for a in order if len(refs[a]) >= 2); nd = sum(1 for a in order if len(refs[a]) >= 3)
    bonds = vec[:nb]; angs = vec[nb:nb+na]; sind = vec[nb+na:nb+na+nd]; cosd = vec[nb+na+nd:]
    xyz = {}; bi = ai = di = 0
    for k, a in enumerate(order):
        c = refs[a]
        if k == 0: xyz[a] = np.zeros(3)
        elif k == 1: xyz[a] = np.array([bonds[bi], 0, 0]); bi += 1
        elif k == 2:
            l = bonds[bi]; bi += 1; th = angs[ai]; ai += 1
            u = xyz[c[1]] - xyz[c[0]]; u = u / (np.linalg.norm(u) + 1e-9)
            perp = np.array([-u[1], u[0], 0.0])
            xyz[a] = xyz[c[0]] + l * (np.cos(th) * u + np.sin(th) * perp)
        else:
            l = bonds[bi]; bi += 1; th = angs[ai]; ai += 1; ch = np.arctan2(sind[di], cosd[di]); di += 1
            xyz[a] = nerf(xyz[c[0]], xyz[c[1]], xyz[c[2]], l, th, ch)
    return np.array([xyz[a] for a in order])                    # in `order` atom-ordering


def pca_recon(dc_train, dc_eval, L):
    mean = dc_train.mean(0); _, _, Vt = np.linalg.svd(dc_train - mean, full_matrices=False)
    M = Vt[:L]; return mean + (dc_eval - mean) @ M.T @ M


def anm_modes(ca, L, cutoff=8.0):
    n = len(ca); H = np.zeros((3*n, 3*n))
    for i in range(n):
        dv = ca - ca[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j <= i: continue
            u = dv[j]/dist[j]; b = np.outer(u, u)
            H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
            H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H); return V[:, 6:6+L]


f = h5py.File(MD, "r"); allk = list(f.keys()); doms = allk[::max(1, len(allk)//12)][:12]
print(f"[intcoord-oracle] {len(doms)} MISATO ligands; internal-coord PCA vs Cartesian ANM/PCA/null")
print(f"  {'system':8s}{'nHvy':>5}{'roundtrip':>10}  " + "  ".join(f"L{L}(null/ANM/cartPCA/INT)" for L in LS))
agg = {L: [] for L in LS}
for dom in doms:
    g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); N = g["atoms_element"].shape[0]
    lig = slice(int(mbi[-1]), N); el = np.array(g["atoms_element"])[lig]
    coords = np.array(g["trajectory_coordinates"])[:, lig, :].astype(np.float64)
    heavy = el != 1; coords = coords[:, heavy, :]; n = heavy.sum()
    if n < 8: continue
    adj, D = perceive_bonds(coords[0])
    order, refs = zmatrix(adj, D, n)
    if any(len(refs[a]) < 3 for a in order[3:]): continue        # need full refs
    T = len(coords); h = T // 2
    IC = np.array([to_internal(coords[t], order, refs) for t in range(T)])   # (T, nIC)
    # round-trip sanity on frame 0
    rt = kabsch_rmsd(from_internal(IC[0], order, refs, coords[0]), coords[0][order])
    # Cartesian references (on `order` atoms for consistency with intcoord recon)
    co = coords[:, order, :]; dcart = (co - co[0]).reshape(T, -1); mc = dcart[:h].mean(0)
    ev_cart = dcart[h:]
    null = np.mean([kabsch_rmsd(co[0], co[h+i]) for i in range(len(ev_cart))])   # null = predict the reference
    Manm = anm_modes(co[0], max(LS))
    cells = []
    for L in LS:
        Le = min(L, IC.shape[1], 3*n-6)
        ic_rec = pca_recon(IC[:h], IC[h:], Le)                   # internal-coord PCA reconstruction
        int_rmsd = np.mean([kabsch_rmsd(from_internal(ic_rec[i], order, refs, co[0]), co[h+i]) for i in range(len(ic_rec))])
        # cartesian PCA + ANM on the same held-out frames
        cp = pca_recon(dcart[:h]-mc, ev_cart-mc, Le)             # actually PCA on centered disp
        cart_pca = np.mean([kabsch_rmsd((co[0] + (mc + cp[i]).reshape(n, 3)), co[h+i]) for i in range(len(cp))])
        rec_anm = (ev_cart - mc) @ Manm[:, :Le] @ Manm[:, :Le].T
        cart_anm = np.mean([kabsch_rmsd((co[0] + (mc + rec_anm[i]).reshape(n, 3)), co[h+i]) for i in range(len(rec_anm))])
        cells.append((null, cart_anm, cart_pca, int_rmsd)); agg[L].append((cart_anm, cart_pca, int_rmsd))
    cs = "  ".join(f"{c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}/{c[3]:.2f}" for c in cells)
    print(f"  {dom:8s}{n:>5}{rt:>10.3f}  {cs}")

print("\n=== MEAN over ligands (absolute A, held-out) ===")
for L in LS:
    a = np.array(agg[L])
    if len(a) == 0: continue
    print(f"  L={L}: Cartesian ANM {a[:,0].mean():.2f}  Cartesian PCA {a[:,1].mean():.2f}  INTERNAL-coord PCA {a[:,2].mean():.2f}")
print("\n  read: internal >> Cartesian ANM (toward Cartesian PCA) -> representation is the fix, build the codec.")
print("        internal ~= Cartesian PCA (no better) -> representation not the lever, different diagnosis needed.")
