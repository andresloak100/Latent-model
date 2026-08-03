"""Build (T,N,3) CA-aligned heavy-atom trajectories for the dimensionality probe.

`scripts/trajectory_dimensionality.py` assumes frames are ALREADY aligned -- it
takes deviation from frame 0 directly. Raw MISATO coordinates carry global
tumbling, which would inflate the deviation spectrum with rigid-body motion. So
every frame is Kabsch-superposed onto frame 0 using the protein CA atoms
(standard protein-MD reference: remove global translation+rotation, then measure
internal deviation over all heavy atoms). One npz per system, key 'coords'.

Usage:
    python scripts/build_traj_probe.py \
        --md   /.../misato/MD.hdf5 \
        --topo /.../misato/topo_extract/parameter_restart_files_MD \
        --misato-build /.../latent-model-workspace/misato_build \
        --out  /.../latent-model-workspace/misato_traj_probe --n 20

Then:
    python scripts/trajectory_dimensionality.py --traj-dir <--out> --limit 20
"""
import argparse, os, sys
import numpy as np


def kabsch_to_ref(traj, mask):
    """Superpose every frame onto frame 0 using atoms in `mask`; apply to all."""
    ref = traj[0]
    rcen = ref[mask].mean(0)
    Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(traj.shape[0]):
        P = traj[t]
        pcen = P[mask].mean(0)
        Pc = P[mask] - pcen
        H = Pc.T @ Q
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        out[t] = (P - pcen) @ R.T + rcen
    return out


def _selftest():
    rng = np.random.RandomState(0)
    Q0 = rng.randn(50, 3)
    th = 0.7
    Rt = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    P0 = Q0 @ Rt.T + np.array([3.0, -2.0, 1.0])
    al = kabsch_to_ref(np.stack([Q0, P0]), np.ones(50, bool))
    err = np.sqrt(((al[1] - Q0) ** 2).sum(-1).mean())
    assert err < 1e-8, f"Kabsch self-test failed: {err}"
    print(f"[selftest] Kabsch recovers a known transform, residual RMSD={err:.2e} A")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="MISATO MD.hdf5")
    ap.add_argument("--topo", required=True, help="topo dir (per-system production.top.gz)")
    ap.add_argument("--misato-build", required=True, help="dir with misato_convert.py (_fast_topo)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20, help="how many systems to build")
    ap.add_argument("--spread", type=int, default=60, help="candidates spread across the corpus")
    args = ap.parse_args()

    sys.path.insert(0, args.misato_build)
    import h5py
    import misato_convert as M

    _selftest()
    os.makedirs(args.out, exist_ok=True)
    md = h5py.File(args.md, "r")
    mm = {s.lower(): s for s in md.keys()}
    systems = [mm[t] for t in sorted(os.listdir(args.topo)) if t.lower() in mm]
    cand = sorted(set(int(k * len(systems) / args.spread) for k in range(args.spread)))
    print(f"[info] {len(systems)} systems; probing {len(cand)} spread candidates for {args.n} builds")

    built = 0
    for ci in cand:
        if built >= args.n:
            break
        su = systems[ci]
        try:
            g = md[su]
            nsol = g["trajectory_coordinates"].shape[1]
            names, elems, resnames, res_of, bonds_all, natom = M._fast_topo(
                f"{args.topo}/{su.lower()}/production.top.gz")
            if natom < nsol or "MOL" not in set(resnames[:nsol]):
                print(f"  skip {su}: mismatch/no-ligand")
                continue
            heavy = np.array([elems[i] != "H" for i in range(nsol)])
            is_ca = np.array([names[i] == "CA" and resnames[i] != "MOL" for i in range(nsol)])
            traj = g["trajectory_coordinates"][:].astype(np.float64)[:, heavy, :]
            ca = is_ca[heavy]
            mask = ca if ca.sum() >= 3 else np.ones(traj.shape[1], bool)
            pre = np.sqrt(((traj - traj[0]) ** 2).sum(-1).mean(-1)).mean()
            traj = kabsch_to_ref(traj, mask)
            post = np.sqrt(((traj - traj[0]) ** 2).sum(-1).mean(-1)).mean()
            np.savez(f"{args.out}/{su}.npz", coords=traj.astype(np.float32))
            built += 1
            ref = "CA" if ca.sum() >= 3 else "allheavy"
            print(f"  built {su}: T={traj.shape[0]} N={traj.shape[1]} nCA={int(ca.sum())} "
                  f"align={ref}  RMSD0 {pre:.2f}->{post:.2f} A")
        except Exception as e:
            print(f"  ERR {su}: {type(e).__name__}: {str(e)[:80]}")
    print(f"[done] built {built} trajectories -> {args.out}")


if __name__ == "__main__":
    main()
