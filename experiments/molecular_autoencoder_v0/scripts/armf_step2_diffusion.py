"""STEP 2 (the priority): the first end-to-end compress -> diffuse -> decode
skeleton. Nothing in this project has run this pipeline before; this is a skeleton
that RUNS, not a good model.

Target: BACKBONE collective (the only arm-F signal that cleared the gate --
backbone L64 ~0.97 A, 75% on mdCATH Window C). Per system:
  compress: Kabsch-align backbone trajectory, PCA-64 of the displacement (the
            working linear codec) -> z_t (64-dim latent per frame)
  diffuse:  a small CONDITIONAL DDPM p(z_{t+1} | z_t) trained on the latent's
            temporal transitions
  decode:   z -> displacement -> backbone coords
Validation WITHOUT side chains: CA-CA distance distribution, backbone bond/angle
geometry, backbone-backbone clashes, and DRIFT over a short rollout. Not RMSD.
"""
import argparse, glob, math, numpy as np, torch, torch.nn as nn
import h5py

BB = ["N", "CA", "C", "O"]


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    atoms = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in atoms])


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pcen = P[mask].mean(0); Pc = P[mask] - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q); d = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pcen) @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T + rcen
    return out


def anm(xyz, K, cutoff=10.0):                                       # structure-only ANM modes (general codec)
    n = len(xyz); H = np.zeros((3 * n, 3 * n)); ar = np.arange(3)
    I, J, U = [], [], []
    for i in range(n):
        dv = xyz - xyz[i]; dd = np.linalg.norm(dv, axis=1)
        for j in np.where((dd < cutoff) & (dd > 1e-6))[0]:
            if j <= i: continue
            I.append(i); J.append(j); U.append(dv[j] / dd[j])
    I, J, U = np.array(I), np.array(J), np.array(U); b = U[:, :, None] * U[:, None, :]
    def scat(a, c, V):
        rows = (3 * a[:, None, None] + ar[None, :, None]) * np.ones((1, 1, 3), int)
        cols = (3 * c[:, None, None] + ar[None, None, :]) * np.ones((1, 3, 1), int)
        np.add.at(H, (rows.ravel(), cols.ravel()), V.ravel())
    scat(I, I, b); scat(J, J, b); scat(I, J, -b); scat(J, I, -b)
    w, V = np.linalg.eigh(H)
    return V[:, 6:6 + K], w[6:6 + K]                                # modes + eigenvalues (for ANM-predicted whitening)


class Denoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * L + 16, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, L))
    def temb(self, t, dev):
        f = torch.exp(torch.arange(8, device=dev) * (-math.log(1e4) / 8))
        a = t[:, None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)          # (B,16)
    def forward(self, zt, cond, t):
        return self.net(torch.cat([zt, cond, self.temb(t, zt.device)], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data")
    ap.add_argument("--n-systems", type=int, default=2)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--Tdiff", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--codec", default="pca", choices=["pca", "anm"])   # anm = general (structure-only) codec
    ap.add_argument("--whiten", default="diag", choices=["diag", "zca", "anm"])   # diag=per-mode std; zca=full cov; anm=sqrt(kT/lambda)
    args = ap.parse_args()
    torch.manual_seed(0)
    files = sorted(glob.glob(f"{args.data_dir}/*.h5"))[:args.n_systems]

    betas = torch.linspace(1e-4, 0.02, args.Tdiff)
    ac = torch.cumprod(1 - betas, 0)                                # alpha_bar

    for fp in files:
        with h5py.File(fp, "r") as f:
            dom = list(f.keys())[0]; g = f[dom]
            if "320" not in g or "0" not in g["320"]:
                continue
            z_at = np.array(g["z"]); N = len(z_at)
            names = parse_names(g, N)
            if len(names) != N:
                continue
            bb = np.isin(names, BB); ca = names == "CA"
            coords = g["320"]["0"]["coords"][:].astype(np.float64)   # (T, N, 3)
        al = kabsch(coords[:, bb, :], ca[bb])                        # align backbone on CA
        d = (al - al[0]).reshape(al.shape[0], -1)                    # (T, 3*nbb) displacement
        T = d.shape[0]
        # compress: PCA (fit on first 80%, per-system) OR ANM (structure-only, GENERAL codec)
        h = int(T * 0.8); mean = d[:h].mean(0); lam = None
        if args.codec == "anm":
            modes, lam = anm(al[0], args.L, 10.0); B = modes[:, :args.L].T; lam = lam[:args.L]
        else:
            _, _, Vt = np.linalg.svd(d[:h] - mean, full_matrices=False); B = Vt[:args.L]
        Z = (d - mean) @ B.T                                         # (T, L) latent
        zmu = Z[:h].mean(0); Zc = Z[:h] - zmu
        # whitening: diag = empirical per-mode std (trajectory); zca = empirical FULL covariance
        # (trajectory, removes cross-mode correlation); anm = structure-derived sqrt(kT/lambda) (ZERO
        # trajectory, gamma=1, kT=0.593) -- the theoretically-correct equilibrium prediction.
        if args.whiten == "zca":
            C = np.cov(Zc.T) + 1e-6 * np.eye(Zc.shape[1]); w2, V2 = np.linalg.eigh(C)
            W = V2 @ np.diag(1 / np.sqrt(w2)) @ V2.T; Winv = V2 @ np.diag(np.sqrt(w2)) @ V2.T
            whiten = lambda X: (X - zmu) @ W; unwhiten = lambda Y: Y @ Winv + zmu
        elif args.whiten == "anm" and lam is not None:
            sd = np.sqrt(0.593 / np.clip(lam, 1e-8, None))          # ANM-predicted std (structure-only)
            whiten = lambda X: (X - zmu) / sd; unwhiten = lambda Y: Y * sd + zmu
        else:
            zsd = Zc.std(0) + 1e-6
            whiten = lambda X: (X - zmu) / zsd; unwhiten = lambda Y: Y * zsd + zmu
        Zn = torch.tensor(whiten(Z), dtype=torch.float32)

        # diffuse: conditional DDPM p(z_{t+1} | z_t)
        m = Denoiser(args.L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        zc, zn1 = Zn[:h - 1], Zn[1:h]                                # (cond, target) train pairs
        for ep in range(args.epochs):
            i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),))
            c, x0 = zc[i], zn1[i]
            k = torch.randint(0, args.Tdiff, (len(i),))
            noise = torch.randn_like(x0)
            xk = ac[k].sqrt()[:, None] * x0 + (1 - ac[k]).sqrt()[:, None] * noise
            opt.zero_grad(); ((m(xk, c, k) - noise) ** 2).mean().backward(); opt.step()

        # rollout: start at z_0 (held-out region start), step forward `horizon`
        m.eval()
        @torch.no_grad()
        def step(cond):
            x = torch.randn(1, args.L)
            for k in reversed(range(args.Tdiff)):
                eps = m(x, cond, torch.tensor([k]))
                x0 = (x - (1 - ac[k]).sqrt() * eps) / ac[k].sqrt()
                if k > 0:
                    x = ac[k - 1].sqrt() * x0 + (1 - ac[k - 1]).sqrt() * torch.randn_like(x)
                else:
                    x = x0
            return x
        z = Zn[h:h + 1]                                              # start of held-out
        roll = [z]
        with torch.no_grad():
            for _ in range(args.horizon):
                z = step(z); roll.append(z)
        Zroll = unwhiten(torch.cat(roll, 0).numpy())                 # (H+1, L) un-whitened
        drec = Zroll @ B + mean                                      # (H+1, 3*nbb)
        bbc = (al[0] + drec.reshape(-1, bb.sum(), 3))                # backbone coords, (H+1, nbb, 3)

        # ---- geometry validation (backbone only) ----
        caidx = np.where(ca[bb])[0]
        cac = bbc[:, caidx, :]                                       # CA over rollout
        cad = np.linalg.norm(np.diff(cac, axis=1), axis=-1)         # consecutive CA-CA
        step_d = np.linalg.norm(np.diff(bbc, axis=1), axis=-1)      # adjacent backbone-atom dist
        def minnb(x):
            D = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)); np.fill_diagonal(D, 9); return D.min()
        drift = np.linalg.norm(cac - cac[0], axis=-1).mean(-1)      # CA drift vs rollout start
        H = args.horizon
        print(f"\n=== {dom}  nbb={bb.sum()} nCA={ca.sum()}  codec={args.codec.upper()}  whiten={args.whiten}  horizon={H} ===")
        finite = np.isfinite(bbc).all()
        print(f"  finite structures: {finite}  latent L={args.L} Tdiff={args.Tdiff}")

        # ---- PRIMARY: latent coefficients vs the TRAINING distribution ----
        # (Cartesian saturation is decoder-forced: output = ref + sum c_i v_i is
        #  confined to the mode affine span; the real test is whether the diffusion
        #  model keeps coefficients inside the training range.)
        Ztr = Z[:h]                                                  # (h, L) training coeffs
        tr_mu, tr_sd = Ztr.mean(0), Ztr.std(0) + 1e-6
        tr_min, tr_max = Ztr.min(0), Ztr.max(0)
        Zro = Zroll                                                  # (H+1, L) rollout coeffs
        out = (Zro < tr_min) | (Zro > tr_max)
        oor = out.any(1).mean()                                      # frac of steps ANY coeff out of range
        modes_out = int(out.any(0).sum())                           # modes that ever leave range
        exc = np.abs(Zro - tr_mu) / tr_sd                          # excursion in training-sigma units
        half = args.L // 2
        lo, hi = exc[:, :half].mean(), exc[:, half:].mean()
        print(f"  [PRIMARY latents] per-step worst-mode excursion {exc.max(1).mean():.2f}sd (max {exc.max():.2f}sd)")
        print(f"    steps with ANY coeff outside training [min,max]: {oor:.0%}   modes ever-out: {modes_out}/{args.L}")
        print(f"    drift concentration: low-idx modes(0-{half-1}) {lo:.2f}sd  vs high-idx({half}-{args.L-1}) {hi:.2f}sd"
              f"  -> {'LOW (slow, large-amplitude)' if lo > hi else 'HIGH (fast)'}")

        # drift-vs-step curve (DECODER-FORCED: PCA span confines it -- not a stability result)
        pts = [p for p in (1, 10, 50, 100, 200, 400, 800, 1000) if p <= H]
        print("  [decoder-forced] CA drift vs start (A): " + "  ".join(f"t{p}={drift[p]:.2f}" for p in pts) +
              "  -> " + ("SATURATES (span-confined, expected)" if drift[-1] < 1.6 * drift[min(50, H)] else "GROWS"))
        # geometry START vs END of the rollout
        def geo(i):
            c = np.linalg.norm(np.diff(cac[i:i+1], axis=1), axis=-1)
            b = np.linalg.norm(np.diff(bbc[i:i+1], axis=1), axis=-1)
            return c.mean(), c.std(), b.mean(), b.std(), minnb(cac[i])
        s, e = geo(1), geo(H)
        print(f"  START(t1)  CA-CA {s[0]:.2f}+-{s[1]:.2f}  bb-bond {s[2]:.2f}+-{s[3]:.2f}  minCA {s[4]:.2f}")
        print(f"  END(t{H})   CA-CA {e[0]:.2f}+-{e[1]:.2f}  bb-bond {e[2]:.2f}+-{e[3]:.2f}  minCA {e[4]:.2f}  (native CA-CA~3.80 bond~1.3-1.5 minCA>3.7)")


if __name__ == "__main__":
    main()
