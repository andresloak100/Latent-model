"""Backbone cache for the GPU propagator scale-up (the GPU venv lacks h5py). Extract
per-system backbone coords + CA-in-backbone mask -> npz. Run locally with h5py."""
import glob, os, numpy as np, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
OUT = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/bb_cache"
BB = ["N", "CA", "C", "O"]; TEMP, R0 = "320", "0"
os.makedirs(OUT, exist_ok=True)


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    a = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in a])


n = 0
for fp in sorted(glob.glob(f"{DATA}/*.h5"))[:20]:
    try:
        with h5py.File(fp, "r") as f:
            dom = list(f.keys())[0]; g = f[dom]
            if TEMP not in g or R0 not in g[TEMP]: continue
            z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
            if len(nm) != N: continue
            heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
            if ca.sum() < 30: continue
            coords = g[TEMP][R0]["coords"][:].astype(np.float32)[:, heavy, :][:, bb, :]
            ca_in_bb = (nm[heavy][bb] == "CA")
    except Exception:
        continue
    np.savez(f"{OUT}/{dom}.npz", coords=coords, ca_in_bb=ca_in_bb)
    n += 1; print(f"{dom}  bb{coords.shape}  nCA {ca_in_bb.sum()}")
print(f"=== cached {n} backbone systems -> {OUT} ===")
