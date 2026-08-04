"""Arm F trajectory dataset: (ref=frame0, target=frame_t) pairs with full static
features, CA-aligned so displacement = internal deviation. One npz per system:
element_idx, bonds, atom_name, coords(T,N,3) aligned to frame 0. WL/canonical are
computed at load time from element_idx+bonds (like molae.dataset)."""
import os, sys, numpy as np
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
sys.path.insert(0, WR + "/misato_build")
import h5py, misato_convert as M

MDP  = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
TOPO = "/network/scratch/j/jacob-junqi.tian/datasets/misato/topo_extract/parameter_restart_files_MD"
OUT  = WR + "/armf_data"; os.makedirs(OUT, exist_ok=True)
N_SYS = 8

def kabsch_to_ref(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(traj.shape[0]):
        P = traj[t]; pcen = P[mask].mean(0); Pc = P[mask] - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        out[t] = (P - pcen) @ R.T + rcen
    return out

md = h5py.File(MDP, "r"); mm = {s.lower(): s for s in md.keys()}
systems = [mm[t] for t in sorted(os.listdir(TOPO)) if t.lower() in mm]
cand = sorted(set(int(k * len(systems) / 40) for k in range(40)))
built = 0
for ci in cand:
    if built >= N_SYS: break
    su = systems[ci]
    try:
        out, err = M.convert_multiframe(su, md, TOPO, list(range(100)))
        if err: print(f"  skip {su}: {err}"); continue
        f0 = out[0]
        coords = np.stack([out[fr]["coords"] for fr in range(100)], 0).astype(np.float64)  # (T,N,3)
        names = np.asarray(f0["atom_name"])
        is_ca = np.array([str(n) == "CA" for n in names])
        mask = is_ca if is_ca.sum() >= 3 else np.ones(coords.shape[1], bool)
        pre = np.sqrt(((coords - coords[0]) ** 2).sum(-1).mean(-1)).mean()
        coords = kabsch_to_ref(coords, mask)
        post = np.sqrt(((coords - coords[0]) ** 2).sum(-1).mean(-1)).mean()
        np.savez(f"{OUT}/{su}.npz",
                 element_idx=np.asarray(f0["element_idx"], np.int64),
                 bonds=np.asarray(f0["bonds"], np.int64),
                 atom_name=names,
                 coords=coords.astype(np.float32))                 # (T,N,3), aligned
        built += 1
        print(f"  built {su}: N={coords.shape[1]} nCA={int(is_ca.sum())} "
              f"dev0 {pre:.2f}->{post:.2f} A  (mean internal deviation ~{post:.2f})")
    except Exception as e:
        print(f"  ERR {su}: {type(e).__name__}: {str(e)[:80]}")
print(f"[done] built {built} arm-F trajectories -> {OUT}")
