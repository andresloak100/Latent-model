"""Learned per-torsion-softness map (the codec the two ceilings point to).

Structure-only per-torsion features (element of the 4 dihedral atoms, central-bond length, degree
of the two central atoms, ring membership) -> shared MLP -> per-torsion stiffness k. Per ligand:
modes = softest L generalised eigenvectors of (diag(k), G=J^T J); reconstruct the torsional
displacement -> NeRF -> Cartesian. TRAINED across training ligands (end-to-end through eigh on
their torsional displacement), EVALUATED on HELD-OUT ligands with NO per-system fitting.

Win condition: held-out learned-map RMSD ~= softness ORACLE (0.51 at L8) and << Cartesian ANM
(0.74) -> a general, structure->softness ligand codec (zero per-system params at test time).
Same ligands/frames/L/metric as the ceilings (Cartesian A after round-trip, second-half frames)."""
import numpy as np, torch, h5py, collections
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
LS = [4, 8, 16]; np.random.seed(0); torch.manual_seed(0)
ELEMS = [6, 7, 8, 16]                                            # C N O S; else -> "other" bucket


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


def in_ring(adj, u, v):                                          # is edge u-v in a cycle? remove it, still connected?
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


def jacobian(refx, order, refs, children, pos):
    n = len(order); tors = [a for a in order if len(refs[a]) >= 3]; J = np.zeros((3 * n, len(tors)))
    for t, a in enumerate(tors):
        c = refs[a]; axis = refx[c[1]] - refx[c[0]]; axis = axis / (np.linalg.norm(axis) + 1e-9)
        for k in subtree(a, children):
            J[3*pos[k]:3*pos[k]+3, t] = np.cross(axis, refx[k] - refx[c[0]])
    return J, tors


def tors_features(refx, adj, order, refs, tors, el):             # per-torsion structure-only features
    feats = []
    for a in tors:
        c = refs[a]; c0, c1 = c[0], c[1]
        blen = np.linalg.norm(refx[c0] - refx[c1])
        deg0 = len(adj[c0]); deg1 = len(adj[c1]); ring = 1.0 if in_ring(adj, c0, c1) else 0.0
        oh = []
        for at in (a, c0, c1, c[2]):
            v = [0.0] * (len(ELEMS) + 1); e = int(el[at])
            v[ELEMS.index(e) if e in ELEMS else -1] = 1.0
            oh += v
        feats.append([blen, deg0 / 4.0, deg1 / 4.0, ring] + oh)
    return np.array(feats)


class SoftMap(torch.nn.Module):
    def __init__(self, nf, h=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(nf, h), torch.nn.GELU(),
                                       torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, 1))
    def forward(self, X): return self.net(X).squeeze(-1)         # log-stiffness per torsion


def modes_from_k(logk, G, L):                                   # softest L generalised modes of (diag(k), G)
    D = G.shape[0]; k = torch.nn.functional.softplus(logk) + 1e-3
    Lc = torch.linalg.cholesky(G + 1e-6 * torch.eye(D, dtype=G.dtype)); Linv = torch.linalg.inv(Lc)
    A = Linv @ torch.diag(k) @ Linv.T; A = 0.5 * (A + A.T)
    w, U = torch.linalg.eigh(A); return Linv.T @ U[:, :min(L, D)]  # (D, L) G-orthonormal


f = h5py.File(MD, "r"); allk = list(f.keys()); doms = allk[::max(1, len(allk)//12)][:12]
data = []
for dom in doms:
    g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); N = g["atoms_element"].shape[0]
    el_all = np.array(g["atoms_element"])[int(mbi[-1]):N]
    coords = np.array(g["trajectory_coordinates"])[:, int(mbi[-1]):N, :].astype(np.float64)
    hv = el_all != 1; coords = coords[:, hv, :]; el = el_all[hv]; n = hv.sum()
    if n < 8: continue
    adj, Dm = perceive_bonds(coords[0]); order, refs, parent, children, pos = zmatrix(adj, Dm, n)
    if any(len(refs[a]) < 3 for a in order[3:]): continue
    co = coords[:, order, :]; T = len(co); h = T // 2
    rb, ra, rd = raw_internal(coords[0], order, refs)
    Ph = np.array([raw_internal(coords[t], order, refs)[2] for t in range(T)])
    dphi = ((Ph - rd + np.pi) % (2 * np.pi)) - np.pi
    J, tors = jacobian(coords[0], order, refs, children, pos); G = J.T @ J + 1e-8 * np.eye(len(tors))
    X = tors_features(coords[0], adj, order, refs, tors, el)
    data.append(dict(dom=dom, order=order, refs=refs, co=co, T=T, h=h, rb=rb, ra=ra, rd=rd,
                     dphi=dphi, G=torch.tensor(G), X=torch.tensor(X, dtype=torch.float64), ntors=len(tors)))

nf = data[0]["X"].shape[1]
# split ligands: 8 train / rest held-out (structure-only generalisation across MOLECULES)
tr_idx = list(range(0, len(data), 3)) + list(range(1, len(data), 3)); tr_idx = sorted(set(tr_idx))
he_idx = [i for i in range(len(data)) if i not in tr_idx]
print(f"[softness-map] {len(data)} ligands: {len(tr_idx)} train / {len(he_idx)} held-out; nfeat={nf}")

# normalise features on training torsions
Xtr = torch.cat([data[i]["X"] for i in tr_idx], 0); mu = Xtr.mean(0); sig = Xtr.std(0) + 1e-6
for d in data: d["Xn"] = (d["X"] - mu) / sig

model = SoftMap(nf).double(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for ep in range(400):                                            # train end-to-end: reconstruct training dphi
    opt.zero_grad(); loss = 0.0
    for i in tr_idx:
        d = data[i]; L = 8                                       # train at L=8 (the operating point)
        M = modes_from_k(model(d["Xn"]), d["G"], L)
        dt = torch.tensor(d["dphi"][:d["h"]]); c = dt @ (d["G"] @ M); rec = c @ M.T
        loss = loss + ((rec - dt) ** 2).mean()
    loss = loss / len(tr_idx); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if ep % 100 == 0: print(f"  ep{ep} train dphi-recon loss {loss.item():.4f}")

print(f"\n  {'system':8s}{'set':>5}{'nT':>4}  " + "  ".join(f"L{L}(learned/oracle*/torPCA*/cANM)" for L in LS))
agg = {L: {"tr": [], "he": []} for L in LS}
for i, d in enumerate(data):
    tag = "he" if i in he_idx else "tr"; co = d["co"]; T, h = d["T"], d["h"]; n = co.shape[1]
    with torch.no_grad():
        logk = model(d["Xn"])
    H = np.zeros((3*n, 3*n))                                     # Cartesian ANM baseline
    ca = co[0]
    for a in range(n):
        dv = ca - ca[a]; dist = np.linalg.norm(dv, axis=1)
        for b in np.where((dist < 8.0) & (dist > 1e-6))[0]:
            if b <= a: continue
            u = dv[b] / dist[b]; bb = np.outer(u, u)
            H[3*a:3*a+3, 3*b:3*b+3] -= bb; H[3*b:3*b+3, 3*a:3*a+3] -= bb
            H[3*a:3*a+3, 3*a:3*a+3] += bb; H[3*b:3*b+3, 3*b:3*b+3] += bb
    _, VA = np.linalg.eigh(H); Mc = VA[:, 6:6+max(LS)]; dc = (co - co[0]).reshape(T, -1); mc = dc[:h].mean(0)
    cells = []
    for L in LS:
        with torch.no_grad():
            M = modes_from_k(logk, d["G"], L).numpy()
        c = d["dphi"][h:] @ (d["G"].numpy() @ M); dphi_rec = c @ M.T
        learned = np.mean([kabsch_rmsd(build_cart(d["order"], d["refs"], d["rb"], d["ra"], d["rd"] + dphi_rec[k]), co[h+k]) for k in range(T-h)])
        Le = min(L, Mc.shape[1])
        cANM = np.mean([kabsch_rmsd((co[0] + (mc + (dc[h+k]-mc) @ Mc[:, :Le] @ Mc[:, :Le].T).reshape(n, 3)), co[h+k]) for k in range(T-h)])
        cells.append((learned, cANM)); agg[L][tag].append(learned)
    cs = "  ".join(f"{c[0]:.2f}/  -  /  -  /{c[1]:.2f}" for c in cells)
    print(f"  {d['dom']:8s}{tag:>5}{d['ntors']:>4}  {cs}")

print("\n=== MEAN learned-map RMSD (absolute A, second-half frames) ===")
print("  (*oracle/torPCA per-system ceilings from armf_torsion_ceilings.py: L4 0.72/0.53  L8 0.51/0.46  L16 0.30/0.39)")
for L in LS:
    tr = np.array(agg[L]["tr"]); he = np.array(agg[L]["he"])
    print(f"  L={L}: learned-map  train {tr.mean():.2f}   HELD-OUT {he.mean():.2f}")
print("\n  read: held-out learned-map ~= softness oracle and << Cartesian ANM -> general structure->softness")
print("        codec (zero per-system params at test). Falls to Cartesian ANM -> features don't carry softness.")
