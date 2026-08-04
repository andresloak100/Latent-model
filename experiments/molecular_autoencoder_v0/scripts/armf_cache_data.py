"""Pre-extract mdCATH CA coords to an .npz cache so GPU jobs need only numpy+torch
(the shared venv tarball lacks h5py, and it must not be overwritten). Run locally
with the h5py-capable venv. Saves per-domain {dom}.npz with c0 (rep0 CA coords) and
c1 (rep1 CA coords, when present -- needed for the frozen-7 cross reference)."""
import glob, os, numpy as np, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
OUT = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/perceiver_cache"
TEMP, R0, R1 = "320", "0", "1"
os.makedirs(OUT, exist_ok=True)


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    a = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in a])


n_ok = 0
for fp in sorted(glob.glob(f"{DATA}/*.h5")):
    try:
        f = h5py.File(fp, "r")
    except Exception:
        continue
    with f:
        dom = list(f.keys())[0]; g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        if len(nm) != N or TEMP not in g or R0 not in g[TEMP]:
            continue
        heavy = z != 1; ca = nm[heavy] == "CA"
        if ca.sum() < 20:
            continue
        d = {"c0": g[TEMP][R0]["coords"][:].astype(np.float32)[:, heavy, :][:, ca, :]}
        if R1 in g[TEMP]:
            d["c1"] = g[TEMP][R1]["coords"][:].astype(np.float32)[:, heavy, :][:, ca, :]
    np.savez(f"{OUT}/{dom}.npz", **d)
    n_ok += 1
    print(f"{dom}  c0{d['c0'].shape}" + (f"  c1{d['c1'].shape}" if "c1" in d else "  (no rep1)"))
print(f"=== cached {n_ok} domains -> {OUT} ===")
