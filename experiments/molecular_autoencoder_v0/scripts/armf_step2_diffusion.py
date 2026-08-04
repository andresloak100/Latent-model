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
        # compress: PCA-L (fit on first 80% for an honest codec)
        h = int(T * 0.8); mean = d[:h].mean(0)
        _, _, Vt = np.linalg.svd(d[:h] - mean, full_matrices=False); B = Vt[:args.L]
        Z = (d - mean) @ B.T                                         # (T, L) latent
        zmu, zsd = Z[:h].mean(0), Z[:h].std(0) + 1e-6
        Zn = torch.tensor((Z - zmu) / zsd, dtype=torch.float32)      # standardised

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
        Zroll = torch.cat(roll, 0).numpy() * zsd + zmu               # (H+1, L) unstandardised
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
        print(f"\n=== {dom}  nbb={bb.sum()} nCA={ca.sum()}  compress->diffuse->decode RAN, horizon={args.horizon} ===")
        print(f"  latent L={args.L}, DDPM Tdiff={args.Tdiff}")
        print(f"  CA-CA consecutive (A): mean={cad.mean():.2f} sd={cad.std():.2f}  (native ~3.80)")
        print(f"  adjacent backbone-atom dist (A): mean={step_d.mean():.2f} sd={step_d.std():.2f}  (native ~1.3-1.5)")
        print(f"  min CA-CA over rollout (clash proxy): {min(minnb(cac[i]) for i in range(len(cac))):.2f} A (native >~3.7)")
        print(f"  CA drift vs rollout start: t=1 {drift[1]:.2f}A  t={args.horizon} {drift[-1]:.2f}A  "
              f"(blow-up if monotonic->large; native fluctuation ~1-3A)")
        finite = np.isfinite(bbc).all()
        print(f"  finite structures: {finite}  -> SKELETON {'RUNS end-to-end' if finite else 'PRODUCED NaNs'}")


if __name__ == "__main__":
    main()
