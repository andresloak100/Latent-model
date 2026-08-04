"""USABILITY TEST (the one 8.1 named): learning-worked != codec-usable.

Usability is not RMSD -- it is whether a RELAXED decoded backbone is physically
valid. Decode held-out backbone frames with each codec, run a SHORT LOCAL
RELAXATION (labelled LJ+bond+angle stand-in -- no real FF is callable in this
venv), then score physical validity: backbone bonds, angles, CA clashes,
Ramachandran. Same pipeline on per-system PCA (best), cross-replica (general
ceiling), and ANM (the zero-parameter general codec), so the question is HOW MUCH
WORSE ANM is after relaxation, not an absolute pass/fail.

  ANM-relaxed ~= PCA-relaxed             -> codec question CLOSES at ANM (no eigh build).
  ANM fails, PCA passes, transition >1.01 -> need a different codec CLASS, not a better map.
  ANM fails, PCA passes, transition 1.01-1.37 -> that is the eigh build's trigger.
"""
import glob, numpy as np, torch, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
L, K = 64, 128
BB = ["N", "CA", "C", "O"]
TEMP, R0, R1 = "320", "0", "1"
USE = ["3a5zD02", "3jvvA01", "2k4qA00", "3a9lA00"]                 # frozen-test subset spanning n, incl. the flex outlier
NFR = 10
# ideal backbone geometry (labelled stand-in constants)
B0 = {("N", "CA"): 1.46, ("CA", "C"): 1.52, ("C", "O"): 1.23, ("C", "N"): 1.33}
A0 = {("N", "CA", "C"): 111.0, ("CA", "C", "O"): 120.5, ("CA", "C", "N"): 116.5,
      ("O", "C", "N"): 123.0, ("C", "N", "CA"): 121.5}
CLASH = 3.5                                                         # CA-CA min for |i-j|>=2


def parse(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    a = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in a]), np.array([int(ln[22:26]) for ln in a])


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pcen = P[mask].mean(0); Pc = P[mask] - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pcen) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


def anm(xyz, K, cutoff=10.0):
    n = len(xyz); H = np.zeros((3 * n, 3 * n)); ar = np.arange(3)
    I, J, U = [], [], []
    for i in range(n):
        dv = xyz - xyz[i]; d = np.linalg.norm(dv, axis=1)
        for j in np.where((d < cutoff) & (d > 1e-6))[0]:
            if j <= i: continue
            I.append(i); J.append(j); U.append(dv[j] / d[j])
    I, J, U = np.array(I), np.array(J), np.array(U)
    b = (U[:, :, None] * U[:, None, :])
    def scat(a, c, V):
        rows = (3 * a[:, None, None] + ar[None, :, None]) * np.ones((1, 1, 3), int)
        cols = (3 * c[:, None, None] + ar[None, None, :]) * np.ones((1, 3, 1), int)
        np.add.at(H, (rows.ravel(), cols.ravel()), V.ravel())
    scat(I, I, b); scat(J, J, b); scat(I, J, -b); scat(J, I, -b)
    w, V = np.linalg.eigh(H)
    return V[:, 6:6 + K]


def pca(dc):
    _, _, Vt = np.linalg.svd(dc, full_matrices=False)
    return Vt[:L].T


def topology(names_bb, res_bb):
    order, ra = [], {}
    for idx, (nm, rs) in enumerate(zip(names_bb, res_bb)):
        if rs not in ra: ra[rs] = {}; order.append(rs)
        ra[rs][nm] = idx
    bonds, angles, phipsi = [], [], []
    for k, rs in enumerate(order):
        r = ra[rs]
        if not all(a in r for a in BB): continue
        bonds += [(r["N"], r["CA"], B0[("N", "CA")]), (r["CA"], r["C"], B0[("CA", "C")]),
                  (r["C"], r["O"], B0[("C", "O")])]
        angles.append((r["N"], r["CA"], r["C"], A0[("N", "CA", "C")]))
        angles.append((r["CA"], r["C"], r["O"], A0[("CA", "C", "O")]))
        if k + 1 < len(order) and all(a in ra[order[k + 1]] for a in BB):
            rn = ra[order[k + 1]]
            bonds.append((r["C"], rn["N"], B0[("C", "N")]))
            angles.append((r["CA"], r["C"], rn["N"], A0[("CA", "C", "N")]))
            angles.append((r["O"], r["C"], rn["N"], A0[("O", "C", "N")]))
            angles.append((r["C"], rn["N"], rn["CA"], A0[("C", "N", "CA")]))
            psi = (r["N"], r["CA"], r["C"], rn["N"])
            phi = None
            if k > 0 and all(a in ra[order[k - 1]] for a in BB):
                phi = (ra[order[k - 1]]["C"], r["N"], r["CA"], r["C"])
            phipsi.append((phi, psi))
    return bonds, angles, phipsi


def dihedral(x, idx):                                              # numpy, degrees
    p0, p1, p2, p3 = x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]]
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def rama_ok(phi, psi):
    return ((-160 <= phi <= -30 and -80 <= psi <= -5) or                     # alpha-R
            (-160 <= phi <= -40 and (psi >= 90 or psi <= -160)) or           # beta
            (-160 <= phi <= -40 and -40 <= psi <= 90) or                     # PPII/bridge
            (20 <= phi <= 90 and -30 <= psi <= 90))                          # alpha-L


def validity(x, bonds, angles, phipsi, caidx):
    bd = np.array([abs(np.linalg.norm(x[a] - x[b]) - b0) for a, b, b0 in bonds])
    an = []
    for a, b, c, a0 in angles:
        u = x[a] - x[b]; v = x[c] - x[b]
        ang = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
        an.append(abs(ang - a0))
    an = np.array(an)
    ca = x[caidx]; D = np.sqrt(((ca[:, None] - ca[None]) ** 2).sum(-1))
    sep = np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None]) >= 2
    nb = D[sep]; clashes = int((nb < CLASH).sum() // 2); minca = float(nb.min())
    rok = [rama_ok(dihedral(x, ph), dihedral(x, ps)) for ph, ps in phipsi if ph is not None]
    return (bd.mean(), (bd < 0.1).mean(), an.mean(), (an < 15).mean(),
            clashes, minca, np.mean(rok) if rok else float("nan"))


def relax(x0, bonds, angles, caidx, steps=250):                   # short local LJ+bond+angle stand-in
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    ba = torch.tensor([[a, b] for a, b, _ in bonds]); bl = torch.tensor([b0 for _, _, b0 in bonds])
    aa = torch.tensor([[a, b, c] for a, b, c, _ in angles]); al_ = torch.tensor([np.radians(a0) for _, _, _, a0 in angles])
    cai = torch.tensor(caidx); sep = (torch.abs(torch.arange(len(caidx))[:, None] - torch.arange(len(caidx))[None]) >= 2)
    x0t = torch.tensor(x0, dtype=torch.float32)
    opt = torch.optim.Adam([x], lr=5e-3)
    for _ in range(steps):
        opt.zero_grad()
        d = (x[ba[:, 0]] - x[ba[:, 1]]).norm(dim=1)
        Eb = ((d - bl) ** 2).sum()
        u = x[aa[:, 0]] - x[aa[:, 1]]; v = x[aa[:, 2]] - x[aa[:, 1]]
        cos = (u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1) + 1e-6)
        Ea = ((torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) - al_) ** 2).sum()
        ca = x[cai]; Dc = torch.cdist(ca, ca)
        Ec = (torch.relu(CLASH - Dc)[sep] ** 2).sum()
        Et = ((x - x0t) ** 2).sum()
        (5.0 * Eb + 0.5 * Ea + 2.0 * Ec + 0.05 * Et).backward()
        opt.step()
    return x.detach().numpy()


print(f"[usability] systems {USE}, {NFR} frames each, codecs PCA/cross/ANM, backbone + short local relax")
hdr = f"  {'system':9s}{'codec':7s}{'recon_A':>8}  | after relax: {'bond|dA':>8}{'bond%':>6}{'ang|dd':>7}{'ang%':>6}{'clash':>6}{'minCA':>6}{'rama%':>6}"
agg = {c: [] for c in ("PCA", "cross", "ANM")}
for dom in USE:
    fp = f"{DATA}/mdcath_dataset_{dom}.h5"
    with h5py.File(fp, "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm, rs = parse(g, N)
        heavy = z != 1; nmh, rsh = nm[heavy], rs[heavy]; bb = np.isin(nmh, BB)
        c0 = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
        c1 = g[TEMP][R1]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    names_bb, res_bb = nmh[bb], rsh[bb]; caidx = np.where(names_bb == "CA")[0]
    a0 = kabsch(c0, names_bb == "CA"); a1 = kabsch(c1, names_bb == "CA")
    d0 = (a0 - a0[0]).reshape(len(a0), -1); d1 = (a1 - a1[0]).reshape(len(a1), -1)
    h = len(d0) // 2; mean0 = d0[:h].mean(0)
    codecs = {"PCA": pca(d0[:h] - mean0), "cross": pca(d1 - d1.mean(0)), "ANM": anm(a0[0], K)[:, :L]}
    bonds, angles, phipsi = topology(names_bb, res_bb)
    fr = np.linspace(h, len(d0) - 1, NFR).astype(int)
    print(f"\n{hdr}")
    for cname, M in codecs.items():
        rec_err, vpre, vpost = [], [], []
        for t in fr:
            dec = mean0 + (d0[t] - mean0) @ M @ M.T                # decoded displacement
            rec_err.append(np.sqrt(((dec - d0[t]) ** 2).sum() / len(names_bb)))   # absolute A / atom
            x = a0[0] + dec.reshape(-1, 3)                         # decoded backbone coords
            vpre.append(validity(x, bonds, angles, phipsi, caidx))
            vpost.append(validity(relax(x, bonds, angles, caidx), bonds, angles, phipsi, caidx))
        rec = np.mean(rec_err); post = np.array(vpost).mean(0)
        agg[cname].append((rec, *post))
        print(f"  {dom:9s}{cname:7s}{rec:>8.2f}  | after relax: {post[0]:>8.3f}{post[1]*100:>5.0f}%"
              f"{post[2]:>7.1f}{post[3]*100:>5.0f}%{post[4]:>6.0f}{post[5]:>6.2f}{post[6]*100:>5.0f}%")

print("\n=== MEAN over systems (after relaxation) ===")
print(f"  {'codec':7s}{'recon_A':>8}{'bond|dA':>9}{'bond%':>6}{'ang|dd':>7}{'ang%':>6}{'clash':>6}{'minCA':>6}{'rama%':>6}")
for c in ("PCA", "cross", "ANM"):
    a = np.array(agg[c]).mean(0)
    print(f"  {c:7s}{a[0]:>8.2f}{a[1]:>9.3f}{a[2]*100:>5.0f}%{a[3]:>7.1f}{a[4]*100:>5.0f}%{a[5]:>6.0f}{a[6]:>6.2f}{a[7]*100:>5.0f}%")
print("\n  read: ANM-relaxed ~= PCA-relaxed -> codec CLOSES at ANM. ANM fails & PCA passes ->"
      "\n        locate the transition in 1.37->0.78; >1.01 = new codec class, 1.01-1.37 = eigh trigger.")
