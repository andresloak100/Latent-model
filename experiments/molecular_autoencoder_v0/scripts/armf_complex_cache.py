"""Stage A for the corpus-scale complex propagator: process ~700 MISATO complexes into a compact
latent cache. Per complex: ligand graph data (for the GNN codec) + dphi/G, and protein z_prot
(residue-bead ANM, 8 slowest modes) + eigenvalues. Stage B trains the joint propagator across
hundreds of these. Protein ANM uses a partial eigensolver (eigsh) for the lowest modes."""
import numpy as np, h5py, collections, pickle, os, time
from scipy.sparse.linalg import eigsh
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
OUT = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/complex_scale_cache.pkl"
NWANT = 700; Lp = 8; ELEMS = [6, 7, 8, 16, 15, 9, 17]


def perceive_bonds(ref):
    n = len(ref); adj = [[] for _ in range(n)]; D = np.sqrt(((ref[:, None] - ref[None]) ** 2).sum(-1))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < 1.8: adj[i].append(j); adj[j].append(i)
    return adj, D


def in_ring(adj, u, v):
    seen = {u}; st = [u]
    while st:
        x = st.pop()
        for y in adj[x]:
            if (x, y) == (u, v) or (x, y) == (v, u): continue
            if y not in seen: seen.add(y); st.append(y)
    return v in seen


def zmatrix(adj, D, n):
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
    return order, refs, parent, children, pos


def subtree(a, children):
    out = []; st = [a]
    while st: u = st.pop(); out.append(u); st.extend(children[u])
    return out


def dih(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2; b1 = b1 / (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def raw_dih(xyz, order, refs):
    return np.array([dih(xyz[a], xyz[refs[a][0]], xyz[refs[a][1]], xyz[refs[a][2]]) for a in order if len(refs[a]) >= 3])


def jacobian(refx, order, refs, children, pos):
    n = len(order); tors = [a for a in order if len(refs[a]) >= 3]; J = np.zeros((3 * n, len(tors)))
    for t, a in enumerate(tors):
        c = refs[a]; axis = refx[c[1]] - refx[c[0]]; axis = axis / (np.linalg.norm(axis) + 1e-9)
        for k in subtree(a, children):
            J[3*pos[k]:3*pos[k]+3, t] = np.cross(axis, refx[k] - refx[c[0]])
    return J, tors


def elem_feat(e):
    v = [0.0] * (len(ELEMS) + 1); v[ELEMS.index(e) if e in ELEMS else -1] = 1.0; return v


def anm_low(xyz, K, cutoff=12.0):                               # dense build (vectorised) + partial eigensolver
    n = len(xyz); H = np.zeros((3*n, 3*n)); ar = np.arange(3); I, J, U = [], [], []
    for i in range(n):
        dv = xyz - xyz[i]; dd = np.linalg.norm(dv, axis=1)
        for j in np.where((dd < cutoff) & (dd > 1e-6))[0]:
            if j > i: I.append(i); J.append(j); U.append(dv[j]/dd[j])
    I, J, U = np.array(I), np.array(J), np.array(U); b = U[:, :, None]*U[:, None, :]
    def sc(a, c, V):
        r = (3*a[:, None, None]+ar[None, :, None])*np.ones((1, 1, 3), int); co = (3*c[:, None, None]+ar[None, None, :])*np.ones((1, 3, 1), int)
        np.add.at(H, (r.ravel(), co.ravel()), V.ravel())
    sc(I, I, b); sc(J, J, b); sc(I, J, -b); sc(J, I, -b)
    w, V = eigsh(H, k=K+6, sigma=1e-6, which='LM'); idx = np.argsort(w); w = w[idx]; V = V[:, idx]
    return V[:, 6:6+K], w[6:6+K]


def kabsch(traj):
    ref = traj[0]; rc = ref.mean(0); Q = ref-rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P-pc; Uu, S, Vt = np.linalg.svd(Pc.T@Q); dd = np.sign(np.linalg.det(Vt.T@Uu.T))
        out[t] = (P-pc)@(Vt.T@np.diag([1, 1, dd])@Uu.T).T + rc
    return out


f = h5py.File(MD, "r"); allk = list(f.keys()); doms = allk[::max(1, len(allk)//NWANT)][:NWANT]
out = []; t0 = time.time()
for ci, dom in enumerate(doms):
    try:
        g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); N = g["atoms_element"].shape[0]; Pn = int(mbi[-1])
        # ligand
        el = np.array(g["atoms_element"])[Pn:N]; lc = np.array(g["trajectory_coordinates"])[:, Pn:N, :].astype(np.float64)
        hv = el != 1; lc = lc[:, hv, :]; eln = np.array(g["atoms_number"])[Pn:N][hv]; n = int(hv.sum())
        if n < 8 or n > 90: continue
        adj, Dm = perceive_bonds(lc[0]); order, refs, parent, children, pos = zmatrix(adj, Dm, n)
        if any(len(refs[a]) < 3 for a in order[3:]): continue
        tors = [a for a in order if len(refs[a]) >= 3]
        if len(tors) < 8: continue
        T = len(lc); rd = raw_dih(lc[0], order, refs); Ph = np.array([raw_dih(lc[t], order, refs) for t in range(T)])
        dphi = (((Ph - rd + np.pi) % (2*np.pi)) - np.pi).astype(np.float32)
        Jj, _ = jacobian(lc[0], order, refs, children, pos); G = (Jj.T @ Jj + 1e-8*np.eye(len(tors))).astype(np.float32)
        xat = np.array([elem_feat(int(eln[a])) + [len(adj[a])/4.0] for a in range(n)], np.float32)
        edges = np.array([[i, j] for i in range(n) for j in adj[i]], np.int64).T
        tidx = np.array([[a, refs[a][0], refs[a][1], refs[a][2]] for a in tors], np.int64)
        tsc = np.array([[np.linalg.norm(lc[0][refs[a][0]]-lc[0][refs[a][1]]), 1.0 if in_ring(adj, refs[a][0], refs[a][1]) else 0.0] for a in tors], np.float32)
        # protein
        res = np.array(g["atoms_residue"])[:Pn]; resid = np.concatenate([[0], np.cumsum(res[1:] != res[:-1])]); nb = int(resid.max()+1)
        if nb < 30 or nb > 400: continue
        pc = np.array(g["trajectory_coordinates"])[:, :Pn, :].astype(np.float64)
        beads = np.stack([np.array([pc[t][resid == r].mean(0) for r in range(nb)]) for t in range(T)])
        al = kabsch(beads); dcart = (al-al[0]).reshape(T, -1); Mp, lamp = anm_low(al[0], Lp); lamp = np.clip(lamp, 1e-8, None)
        zprot = (dcart @ Mp).astype(np.float32)
        out.append(dict(dom=dom, n=n, ntors=len(tors), nb=nb, xat=xat, edges=edges, tidx=tidx, tsc=tsc,
                        G=G, dphi=dphi, zprot=zprot, lamp=lamp.astype(np.float32)))
        if len(out) % 50 == 0: print(f"  {len(out)} complexes cached ({ci+1} scanned, {time.time()-t0:.0f}s)", flush=True)
    except Exception:
        continue
pickle.dump(out, open(OUT, "wb"))
print(f"[complex-cache] DONE: {len(out)} complexes -> {OUT} ({time.time()-t0:.0f}s)", flush=True)
