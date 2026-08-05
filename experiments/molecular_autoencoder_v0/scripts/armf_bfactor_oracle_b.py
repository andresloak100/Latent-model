"""Crystallographic-B oracle: fit per-residue springs to the frozen-7 PARENT-PDB
experimental B-factors, reconstruct the mdCATH trajectory. oracle-RMSF (1.26) is the
upper bound (trajectory's own diagonal); this measures whether the ACTUAL crystal
signal reaches it or falls below ANM 1.37.

Match: mdCATH domain ID = PDB(0:4)+chain(4)+domain. Get mdCATH domain CA residue
numbers from the h5 pdbProteinAtoms; download the parent PDB; match chain CA B-factors
by residue number; fit springs to matched B (z-scored); reconstruct the matched-residue
mdCATH trajectory vs ANM(uniform)/PCA/null on the SAME matched subset.
"""
import os, subprocess, numpy as np, torch, h5py
MDATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"
L = 64; os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
torch.manual_seed(0); np.random.seed(0)


def parse_ca_resid(g, N):                                        # mdCATH domain: CA residue numbers (heavy, in c0 order)
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    lines = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    nm = np.array([ln[12:16].strip() for ln in lines]); rs = np.array([int(ln[22:26]) for ln in lines])
    return nm, rs


def pdb_chain_B(txt, chain):                                     # parent PDB: resnum -> B for a chain's CA
    out = {}
    for ln in txt.splitlines():
        if not ln.startswith("ATOM") or ln[12:16].strip() != "CA": continue
        if ln[16] not in (" ", "A") or ln[21] != chain: continue
        try: out[int(ln[22:26])] = float(ln[60:66])
        except ValueError: pass
    return out


def edges(ca, cutoff=13.0):
    n = len(ca); I, J = [], []
    for i in range(n):
        dv = ca - ca[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j > i: I.append(i); J.append(j)
    return np.array(I), np.array(J)


def kabsch(traj):
    ref = traj[0]; rcen = ref.mean(0); Q = ref - rcen; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


def anm_modes(ca, s, I, J):
    n = len(ca); H = np.zeros((3 * n, 3 * n)); U = ca[J] - ca[I]; U = U / np.linalg.norm(U, axis=1, keepdims=True)
    g = np.exp((s[I] + s[J]) / 2)
    for e in range(len(I)):
        i, j = int(I[e]), int(J[e]); b = g[e] * np.outer(U[e], U[e])
        H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
        H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H); return V[:, 6:6 + L]


def resid(ev, M):
    rec = ev @ M @ M.T; return float(np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * (ev.shape[1] // 3))))


def zt(x): return (x - x.mean()) / (x.std() + 1e-9)


def fit_springs(ca, tz, steps=400):
    I, J = edges(ca); n = len(ca); Ii, Jj = torch.tensor(I), torch.tensor(J)
    tzt = torch.tensor(tz, dtype=torch.float32); s = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([s], lr=5e-2)
    for _ in range(steps):
        g = torch.exp((s[Ii] + s[Jj]) / 2); G = torch.zeros(n, n)
        G = G.index_put((Ii, Jj), -g, accumulate=True); G = G.index_put((Jj, Ii), -g, accumulate=True)
        G = G.index_put((Ii, Ii), g, accumulate=True); G = G.index_put((Jj, Jj), g, accumulate=True)
        d = torch.diagonal(torch.linalg.pinv(G)); dz = (d - d.mean()) / (d.std() + 1e-9)
        opt.zero_grad(); ((dz - tzt) ** 2).mean().backward(); opt.step()
    return s.detach().numpy(), I, J


frozen = [x for x in open(FROZEN).read().split() if x]
print("[oracle-B] fit springs to parent-PDB crystallographic B, reconstruct mdCATH held-out")
print(f"  {'domain':9s}{'PDB':>6}{'match':>7}{'null':>7}{'ANM':>7}{'PCA':>7}{'oracleB':>9}{'GNMvsB':>8}")
rows = []
for dom in frozen:
    pdb, chain = dom[:4], dom[4]
    with h5py.File(f"{MDATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm, rs = parse_ca_resid(g, N)
        heavy = z != 1; ca_m = nm[heavy] == "CA"; resid_md = rs[heavy][ca_m]
        c0 = g["320"]["0"]["coords"][:].astype(np.float64)[:, heavy, :][:, ca_m, :]
    r = subprocess.run(["curl", "-sf", f"https://files.rcsb.org/download/{pdb}.pdb"], capture_output=True, text=True)
    if r.returncode != 0: print(f"  {dom:9s}{pdb:>6}  download failed"); continue
    b_by_res = pdb_chain_B(r.stdout, chain)
    mask = np.array([rr in b_by_res for rr in resid_md])
    if mask.sum() < 30: print(f"  {dom:9s}{pdb:>6}{mask.sum():>7}  too few matched"); continue
    B = np.array([b_by_res[rr] for rr in resid_md[mask]])
    c0 = c0[:, mask, :]; n = mask.sum()
    al = kabsch(c0); d = (al - al[0]).reshape(len(al), -1); h = len(d) // 2; mean0 = d[:h].mean(0); ev = d[h:] - mean0
    rmsf = np.sqrt(((d[:h] - mean0) ** 2).reshape(h, n, 3).sum(-1).mean(0))
    I, J = edges(al[0])
    null = float(np.sqrt((ev ** 2).sum() / (ev.shape[0] * n)))
    r_anm = resid(ev, anm_modes(al[0], np.zeros(n), I, J))
    _, _, Vt = np.linalg.svd(d[:h] - mean0, full_matrices=False); r_pca = resid(ev, Vt[:L].T)
    s_fit, I2, J2 = fit_springs(al[0], zt(B))
    r_orc = resid(ev, anm_modes(al[0], s_fit, I2, J2))
    gnmb = float(np.corrcoef(rmsf, B)[0, 1])                     # how well crystal B tracks MD RMSF
    rows.append((null, r_anm, r_pca, r_orc)); frac = f"{mask.sum()}/{len(mask)}"
    print(f"  {dom:9s}{pdb:>6}{frac:>7}{null:>7.2f}{r_anm:>7.2f}{r_pca:>7.2f}{r_orc:>9.2f}{gnmb:>8.2f}")
a = np.array(rows); mn = a.mean(0)
print(f"  {'MEAN':9s}{'':>6}{'':>7}{mn[0]:>7.2f}{mn[1]:>7.2f}{mn[2]:>7.2f}{mn[3]:>9.2f}")
print(f"\n  VERDICT: oracle-B {mn[3]:.2f} vs ANM {mn[1]:.2f} (oracle-RMSF ceiling 1.26, PCA 0.78) -> "
      + ("crystal B beats ANM -- train the map (bounded by ~1.26)" if mn[3] < mn[1] - 0.03 else
         "crystal B does NOT beat ANM -> the ACTUAL signal cannot work; CLOSE the track"))
