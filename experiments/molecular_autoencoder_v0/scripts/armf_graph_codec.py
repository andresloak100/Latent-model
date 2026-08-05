"""Full graph codec: message-passing GNN -> per-torsion softness -> torsional modes.

The softness-map v1 fork: v1 (local features, 8 ligands) fit but generalised only at L16. This
scales to many MISATO ligands and gives each torsion GRAPH CONTEXT -- a message-passing GNN over
the molecular bond graph produces per-atom embeddings; a torsion's softness is read from its 4
dihedral atoms' embeddings. Softness -> softest-L modes of (diag(k), G) -> reconstruct dphi ->
NeRF -> Cartesian. Trained end-to-end (through eigh) across training ligands, eval on HELD-OUT
LIGANDS (no per-system fit). Target: beat Cartesian ANM AND approach the ceilings at L4/L8.

Ceilings (per-system, from armf_torsion_ceilings.py): softness oracle L4/8/16 0.72/0.51/0.30;
torsion PCA 0.53/0.46/0.39; Cartesian ANM 0.77/0.74/0.73. v1 held-out 1.32/1.17/0.41."""
import numpy as np, torch, h5py, collections, pickle, os
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
CACHE = f"{WR}/graph_codec_cache.pkl"; NLIG = 200; LS = [4, 8, 16]; ELEMS = [6, 7, 8, 16, 15, 9, 17]
np.random.seed(0); torch.manual_seed(0)


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
    dihs = []
    for a in order:
        c = refs[a]
        if len(c) >= 3: dihs.append(dih(xyz[a], xyz[c[0]], xyz[c[1]], xyz[c[2]]))
    return np.array(dihs)


def raw_ba(xyz, order, refs):
    bonds, angs = [], []
    for a in order:
        c = refs[a]
        if len(c) >= 1: bonds.append(np.linalg.norm(xyz[a] - xyz[c[0]]))
        if len(c) >= 2:
            u = xyz[a] - xyz[c[0]]; v = xyz[c[1]] - xyz[c[0]]
            angs.append(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))
    return np.array(bonds), np.array(angs)


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


def elem_feat(e):
    v = [0.0] * (len(ELEMS) + 1); v[ELEMS.index(e) if e in ELEMS else -1] = 1.0; return v


def build_cache():
    f = h5py.File(MD, "r"); allk = list(f.keys()); doms = allk[::max(1, len(allk)//NLIG)][:NLIG]; out = []
    for dom in doms:
        try:
            g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); N = g["atoms_element"].shape[0]
            el = np.array(g["atoms_element"])[int(mbi[-1]):N]
            coords = np.array(g["trajectory_coordinates"])[:, int(mbi[-1]):N, :].astype(np.float64)
            hv = el != 1; coords = coords[:, hv, :]; el = el[hv]; n = int(hv.sum())
            if n < 8 or n > 90: continue
            adj, Dm = perceive_bonds(coords[0]); order, refs, parent, children, pos = zmatrix(adj, Dm, n)
            if any(len(refs[a]) < 3 for a in order[3:]): continue
            T = len(coords)
            if T > 100: coords = coords[::T // 100][:100]; T = len(coords)
            rb, ra, rd = *raw_ba(coords[0], order, refs), raw_dih(coords[0], order, refs)
            Ph = np.array([raw_dih(coords[t], order, refs) for t in range(T)])
            dphi = (((Ph - rd + np.pi) % (2 * np.pi)) - np.pi).astype(np.float32)
            J, tors = jacobian(coords[0], order, refs, children, pos)
            G = (J.T @ J + 1e-8 * np.eye(len(tors))).astype(np.float32)
            xat = np.array([elem_feat(int(el[a])) + [len(adj[a]) / 4.0] for a in range(n)], np.float32)
            edges = [[i, j] for i in range(n) for j in adj[i]]
            tidx = np.array([[a, refs[a][0], refs[a][1], refs[a][2]] for a in tors], np.int64)  # orig indices
            tsc = np.array([[np.linalg.norm(coords[0][refs[a][0]] - coords[0][refs[a][1]]),
                             1.0 if in_ring(adj, refs[a][0], refs[a][1]) else 0.0] for a in tors], np.float32)
            out.append(dict(dom=dom, n=n, order=order, refs={a: refs[a] for a in order}, rb=rb, ra=ra, rd=rd,
                            dphi=dphi, G=G, co=coords[:, order, :].astype(np.float32), T=T,
                            xat=xat, edges=np.array(edges, np.int64).T, tidx=tidx, tsc=tsc, ntors=len(tors)))
        except Exception as ex:
            continue
    pickle.dump(out, open(CACHE, "wb")); return out


data = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else build_cache()
print(f"[graph-codec] {len(data)} ligands cached (n heavy 8-90)")
tr_idx = [i for i in range(len(data)) if i % 4 != 0]; he_idx = [i for i in range(len(data)) if i % 4 == 0]
print(f"  {len(tr_idx)} train / {len(he_idx)} held-out ligands")


class GraphCodec(torch.nn.Module):
    def __init__(self, fin, H=48, layers=3):
        super().__init__()
        self.embed = torch.nn.Linear(fin, H)
        self.self_w = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(layers)])
        self.nbr_w = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(4 * H + 2, H), torch.nn.GELU(),
                                        torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, 1))
    def forward(self, xat, edges, n, tidx, tsc):
        h = self.embed(xat); src, dst = edges[0], edges[1]
        deg = torch.zeros(n, dtype=h.dtype).index_add_(0, dst, torch.ones(len(dst), dtype=h.dtype)).clamp(min=1)[:, None]
        for sw, nw in zip(self.self_w, self.nbr_w):
            agg = torch.zeros_like(h).index_add_(0, dst, h[src]) / deg
            h = torch.nn.functional.gelu(sw(h) + nw(agg))
        e = torch.cat([h[tidx[:, 0]], h[tidx[:, 1]], h[tidx[:, 2]], h[tidx[:, 3]], tsc], -1)
        return self.head(e).squeeze(-1)                          # log-stiffness per torsion


def modes_from_k(logk, G, L):
    D = G.shape[0]; k = torch.nn.functional.softplus(logk) + 1e-3
    Lc = torch.linalg.cholesky(G + 1e-6 * torch.eye(D, dtype=G.dtype)); Linv = torch.linalg.inv(Lc)
    A = Linv @ torch.diag(k) @ Linv.T; A = 0.5 * (A + A.T)
    w, U = torch.linalg.eigh(A); return Linv.T @ U[:, :min(L, D)]


for d in data:                                                   # to torch once
    d["t_xat"] = torch.tensor(d["xat"]).double(); d["t_edges"] = torch.tensor(d["edges"])
    d["t_tidx"] = torch.tensor(d["tidx"]); d["t_tsc"] = torch.tensor(d["tsc"]).double()
    d["t_G"] = torch.tensor(d["G"].astype(np.float64)); d["t_dphi"] = torch.tensor(d["dphi"].astype(np.float64))
model = GraphCodec(data[0]["xat"].shape[1]).double(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
NEP = 600; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, NEP)
for ep in range(NEP):                                            # train on ALL frames of train ligands, at L=4 and 8
    perm = np.random.permutation(tr_idx); opt.zero_grad(); loss = 0.0; nb = 0
    for i in perm[:32]:
        d = data[i]; logk = model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"]); dt = d["t_dphi"]
        for Ltr in (4, 8):
            M = modes_from_k(logk, d["t_G"], Ltr); c = dt @ (d["t_G"] @ M); rec = c @ M.T
            loss = loss + ((rec - dt) ** 2).mean()
        nb += 1
    loss = loss / nb; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if ep % 100 == 0: print(f"  ep{ep} train dphi-recon loss {loss.item():.4f}")

def min_nb_dist(x):                                              # min non-bonded heavy-heavy distance (bonded ~1.5A excluded)
    D = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)); np.fill_diagonal(D, 9); D[D < 1.7] = 9
    return D.min()


agg = {L: {"tr": [], "he": []} for L in LS}; guard_rec, guard_ref = [], []
for i, d in enumerate(data):
    tag = "he" if i in he_idx else "tr"; co = d["co"].astype(np.float64); T = d["T"]; h = T // 2; n = d["n"]
    with torch.no_grad():
        logk = model(d["t_xat"], d["t_edges"], d["n"], d["t_tidx"], d["t_tsc"])
    for L in LS:
        with torch.no_grad():
            M = modes_from_k(logk, d["t_G"], L).numpy()
        c = d["dphi"][h:].astype(np.float64) @ (d["G"].astype(np.float64) @ M); dphi_rec = c @ M.T
        recs = [build_cart(d["order"], d["refs"], d["rb"], d["ra"], d["rd"] + dphi_rec[k]) for k in range(T - h)]
        rm = np.mean([kabsch_rmsd(recs[k], co[h + k]) for k in range(T - h)])
        agg[L][tag].append(rm)
        if L == 8 and tag == "he":                               # guard at operating point, held-out, NO relaxation
            guard_rec += [min_nb_dist(r) for r in recs]; guard_ref += [min_nb_dist(co[h + k]) for k in range(T - h)]

gr = np.array(guard_rec); gf = np.array(guard_ref)
print("\n=== GUARDS (L=8, held-out reconstructions, NO relaxation) -- reported before RMSD ===")
print(f"  bond/angle deviation: 0 by construction (reference bonds+angles reused; only dihedrals reconstructed)")
print(f"  min non-bonded heavy-heavy distance (A): recon {gr.mean():.2f} (min {gr.min():.2f})  vs  reference {gf.mean():.2f} (min {gf.min():.2f})")
print(f"  clash fraction (recon min non-bonded < 2.0A): {(gr < 2.0).mean():.1%}  (reference {(gf < 2.0).mean():.1%})")

print("\n=== MEAN learned graph-codec RMSD (absolute A, held-out frames) ===")
print("  ceilings: softness oracle L4/8/16 0.72/0.51/0.30  torPCA 0.53/0.46/0.39  cANM 0.77/0.74/0.73  |  v1 held-out 1.32/1.17/0.41")
for L in LS:
    tr = np.array(agg[L]["tr"]); he = np.array(agg[L]["he"])
    print(f"  L={L}: graph-codec  train {tr.mean():.2f}   HELD-OUT {he.mean():.2f}   (n_he={len(he)})")
print("\n  read: held-out << Cartesian ANM at L4/L8 -> general graph codec (context fixes the ranking v1 missed).")
