"""SECONDARY A: non-protein generality. "ANM generalises by construction" is a CLAIM;
every test so far is proteins. MISATO ligands are real small-molecule MD. Run the
codec on LIGAND heavy atoms only, vs null / ANM / per-system PCA.

Prediction on record: ANM does substantially WORSE on ligands than proteins -- its
low-frequency modes describe collective domain motion, but a 40-atom ligand's
dynamics is torsional and local, exactly what ANM handles worst. If so, the codec is
a PROTEIN codec (a direct hit on objective 1).

CARE: L is reported RELATIVE to 3N-6 (compression ratio) on every row; L=16 on a
10-heavy-atom ligand is nearly the full space and compression is vacuous.
"""
import numpy as np, h5py
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
LS = [4, 8, 16]


def kabsch(traj):
    ref = traj[0]; rcen = ref.mean(0); Q = ref - rcen; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pcen = P.mean(0); Pc = P - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pcen) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


def anm(xyz, K, cutoff=8.0):
    n = len(xyz); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dv = xyz - xyz[i]; dist = np.linalg.norm(dv, axis=1)
        for j in np.where((dist < cutoff) & (dist > 1e-6))[0]:
            if j <= i: continue
            u = dv[j] / dist[j]; b = np.outer(u, u)
            H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
            H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H)
    return V[:, 6:6 + K]


def pca(dc, L):
    _, _, Vt = np.linalg.svd(dc, full_matrices=False); return Vt[:L].T


def resid(ev, M):
    rec = ev @ M @ M.T
    return float(np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * (ev.shape[1] // 3))))


f = h5py.File(MD, "r"); allk = list(f.keys())
doms = allk[::max(1, len(allk) // 24)][:24]                       # spanning sample of ~24 systems
print(f"[ligand] {len(doms)} of {len(allk)} MISATO systems (spanning sample); codec on LIGAND heavy atoms")
print(f"  {'system':8s}{'nHvy':>5}{'3N-6':>6}  " + "  ".join(f"L{L}(null/ANM/PCA/ratio)" for L in LS))
rows = []
for dom in doms:
    g = f[dom]; mbi = np.array(g["molecules_begin_atom_index"]); N = g["atoms_element"].shape[0]
    lig = slice(int(mbi[-1]), N)                                  # ligand = last molecule
    el = np.array(g["atoms_element"])[lig]
    coords = np.array(g["trajectory_coordinates"])[:, lig, :].astype(np.float64)
    heavy = el != 1; coords = coords[:, heavy, :]; n = heavy.sum()
    if n < 6: continue
    al = kabsch(coords); d = (al - al[0]).reshape(len(al), -1)
    h = len(d) // 2; mean0 = d[:h].mean(0); ev = d[h:] - mean0
    nm = 3 * n - 6                                                # available non-trivial modes
    Manm = anm(al[0], min(max(LS), nm)); Mpca_all = pca(d[:h] - mean0, min(max(LS), nm))
    null = float(np.sqrt((ev ** 2).sum() / (ev.shape[0] * n)))
    cells = []
    for L in LS:
        Le = min(L, nm)
        r_anm = resid(ev, Manm[:, :Le]); r_pca = resid(ev, Mpca_all[:, :Le])
        cells.append((null, r_anm, r_pca, Le / nm))
    rows.append((dom, n, nm, null, cells))
    cellstr = "  ".join(f"{c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}/{c[3]:.0%}" for c in cells)
    print(f"  {dom:8s}{n:>5}{nm:>6}  {cellstr}")

# aggregate: ANM as a FRACTION of the PCA ceiling (the protein number was ~2/3 for CA)
print(f"\n=== ANM captured fraction of PCA ceiling (1 - r_anm/null) / (1 - r_pca/null), per L ===")
for li, L in enumerate(LS):
    fracs, ratios = [], []
    for dom, n, nm, null, cells in rows:
        _, r_anm, r_pca, cr = cells[li]
        anm_cap = 1 - r_anm / null; pca_cap = 1 - r_pca / null
        if pca_cap > 0.05: fracs.append(anm_cap / pca_cap)
        ratios.append(cr)
    print(f"  L={L}: ANM/PCA ceiling = {np.mean(fracs):.0%} (mean over systems)  "
          f"mean compression ratio {np.mean(ratios):.0%}"
          + ("  [VACUOUS: L approaches full space]" if np.mean(ratios) > 0.5 else ""))
print("\n  read: if ANM/PCA fraction is far below the protein ~2/3, ANM is a PROTEIN codec (obj-1 hit).")
