"""Freeze the held-out test set BEFORE any new domain can enter the cohort.

Reproduces armf_learning_curve.py's cohort construction and split EXACTLY (same
filter, same stable sort by n, same S[2::4]) and writes the test domain IDs to a
persistent file. From then on the learning curve reads this file: the test set is
fixed, only the TRAINING POOL grows, and every curve point (5/10/20/40/60) is
scored on the identical systems -- so "still improving" cannot be a change of test
set. Run once, now, on the current 28-domain cohort."""
import glob, numpy as np, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
K = 128
TEMP, R0, R1 = "320", "0", "1"
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    a = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in a])


S = []
for fp in sorted(glob.glob(f"{DATA}/*.h5")):                         # same glob order as the main script
    try:
        f = h5py.File(fp, "r")
    except Exception:
        continue
    with f:
        dom = list(f.keys())[0]; g = f[dom]
        if TEMP not in g or R0 not in g[TEMP] or R1 not in g[TEMP]:
            continue
        z = np.array(g["z"]); N = len(z); names = parse_names(g, N)
        if len(names) != N:
            continue
        heavy = z != 1; ca = names[heavy] == "CA"
        if ca.sum() < K // 3 + 10:
            continue
    S.append((dom, int(ca.sum())))

S.sort(key=lambda x: x[1])                                           # stable sort by n (ties keep glob order)
test = S[2::4]                                                       # identical to the main script's split
with open(FROZEN, "w") as fh:
    fh.write("\n".join(d for d, _ in test) + "\n")
print(f"cohort {len(S)} (both replicas); FROZEN TEST SET ({len(test)}):")
print("  " + "  ".join(f"{d}({n})" for d, n in test))
print(f"written -> {FROZEN}")
