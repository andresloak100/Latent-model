"""Propagator pre-check on LIGAND latents: is there frame-to-frame dynamics to propagate, or are
MISATO snapshots decorrelated (=> an equilibrium SAMPLER, not a propagator)?

Uses the graph-codec cache (per-ligand dphi, G). Per ligand: top-8 torsional-PCA latents c_t;
measure per-mode lag-1 autocorrelation and integrated autocorrelation time (IAT). Also peek at any
MISATO time metadata. Read: lag-1 ~0 across modes/ligands -> snapshots decorrelated, the
"propagator" is equilibrium sampling (design changes). lag-1 high -> real dynamics -> build the
conditional propagator + OU baseline with the standard discriminators."""
import numpy as np, pickle, os, h5py
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"; CACHE = f"{WR}/graph_codec_cache.pkl"
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
data = pickle.load(open(CACHE, "rb")); L = 8


def iat(x):                                                      # integrated autocorrelation time (sum of positive lags)
    x = x - x.mean(); v = (x * x).mean() + 1e-12; tau = 1.0
    for k in range(1, min(50, len(x) // 2)):
        c = (x[:-k] * x[k:]).mean() / v
        if c <= 0.05: break
        tau += 2 * c
    return tau


# MISATO time metadata peek
f = h5py.File(MD, "r"); k0 = list(f.keys())[0]; g = f[k0]
print("[ligand-prop-precheck] MISATO trajectory metadata peek:")
print(f"  keys in a system: {[k for k in g.keys()][:12]}")
print(f"  trajectory_coordinates shape: {g['trajectory_coordinates'].shape}  (frames x atoms x 3)")
for tk in ("frames", "time", "dt", "trajectory_time"):
    if tk in g: print(f"  {tk}: {np.array(g[tk])[:5]}")

lag1_all, iat_all, T_all = [], [], []
for d in data:
    dphi = d["dphi"].astype(np.float64); T = len(dphi); T_all.append(T)
    dphi = dphi - dphi.mean(0)
    _, _, Vt = np.linalg.svd(dphi, full_matrices=False); C = dphi @ Vt[:L].T   # top-L torsion PCA latents (T x L)
    for m in range(min(L, C.shape[1])):
        c = C[:, m]
        if c.std() < 1e-6: continue
        lag1_all.append(np.corrcoef(c[:-1], c[1:])[0, 1]); iat_all.append(iat(c))

lag1 = np.array(lag1_all); iatv = np.array(iat_all)
print(f"\n  ligands {len(data)}, frames/ligand median {int(np.median(T_all))} (min {min(T_all)}, max {max(T_all)})")
print(f"  per-mode lag-1 autocorrelation: mean {lag1.mean():.2f}  median {np.median(lag1):.2f}  "
      f"frac>0.3 {np.mean(lag1>0.3):.0%}  frac>0.5 {np.mean(lag1>0.5):.0%}")
print(f"  per-mode IAT (frames): median {np.median(iatv):.1f}  frac>2 {np.mean(iatv>2):.0%}  (IAT~1 => decorrelated)")
print("\n  READ: lag-1 high / IAT>>1 -> real dynamics, build the propagator. lag-1 ~0 / IAT~1 ->")
print("        snapshots decorrelated -> propagator = equilibrium sampler; propagate structure differently.")
