"""ATLAS cache builder: download -> SUBSAMPLE -> store compact -> DELETE the archive.

ATLAS HOSTS BOTH AXES. The MISATO(N-axis)/mdCATH(capacity-axis) split is RETIRED -- it was a
workaround for a bind ATLAS dissolves. MISATO had a 37.5x N range but 80 frames and a broken ceiling;
mdCATH had a valid ceiling but only 21 training domains. ATLAS has a 56.7x range, 10,001
frames/replica AND 1,938 proteins. Two of the last three nulls came from a corpus that could satisfy
only one constraint at a time.

WHY SUBSAMPLE RATHER THAN STREAM. MISATO's PCA baseline overfits because 80 frames sit BELOW the
median rank90 (~168); the fix is frames comfortably ABOVE rank90. 2,501 frames/replica -> ~2,000
train, which also keeps the DM=512 ceiling rank-valid (see STRIDE).

WHY STORE RATHER THAN STREAM. Streaming was right for the b-exponent (spectra out, data discarded).
It is WRONG here: TRAINING reads frames many times over, so the subsampled set must be reusable
rather than re-downloaded per run. The archive is deleted immediately; only the subsample is kept.

MEASURED (not assumed):
  - 10,001 frames/replica at 10 ps, confirmed against the production .mdp (nsteps 50e6 x dt 2 fs)
  - atoms = 15.77*L - 8, R^2 0.9894 over 24 REAL topologies; atoms/residue median 15.89
    (range 13.11-16.66). NOT ~9.5 and not the 14.7 taken from a single protein.
  - => ATLAS N range ~591 - 33,541 atoms (56.7x). The TOP EXCEEDS MISATO's 26,861, so ATLAS extends
    BOTH ENDS of the N axis and gives a slightly LONGER lever for the 1e6 extrapolation, not the same
    one. CAVEAT: the fit is anchored on L=38-212, so L=2,128 is a 10x extrapolation -- confirm by
    caching the largest entries directly before treating the top as established.
  - 2,501 frames x 3 replicas -> ~157 GB for 700 proteins stored UNCOMPRESSED .npy so it can be
    MEMMAPPED. 108 GB compressed would be smaller on disk but npz CANNOT be memmapped -- it must be
    fully inflated, and neither figure fits in RAM. Training reads 8 random frames per step, so
    memmap reads only those slices.
  - ATLAS bandwidth 4.9 MB/s (NOT HuggingFace's 25): 700 proteins ~ 0.34 TB ~ 19 h serial, ~5 h on 4
  - mdtraj 1.10.3 loaded from a wheel unzipped onto PYTHONPATH (the venv has no SSL, so pip is dead;
    the shared venv is never modified)

SCOPE NOTE (uniform subsampling): taking every 4th frame across the full 100 ns preserves the SLOW
COLLECTIVE MODES that dominate variance at low k. What is lost is fast local motion, which is not
what a k<=24 ceiling measures. State this wherever the ATLAS ceiling is used.

SCOPE NOTE (corpus): ATLAS is SINGLE CHAINS; MISATO is COMPLEXES. Different chemistry and mobility
regimes -- do not pool without checking, and expect ATLAS to sit on the floppier side (mdCATH single
domains measured 1.65x the RMSF of MISATO complexes). ATLAS extends BOTH ends of the N axis
(measured ~591-33,541 vs MISATO's 717-26,861)."""
import os, sys, csv, json, time, subprocess, shutil, numpy as np, warnings
warnings.filterwarnings("ignore")
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
LIB = f"{WR}/pylibs"; sys.path.insert(0, LIB)
OUT = f"{WR}/atlas_cache"; INFO = f"{WR}/atlas_info.tsv"
URL = "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{p}/{p}_protein.zip"
STRIDE = 4    # 10,001 -> 2,501 frames/replica -> ~2,000 train after an 80/20 split.
# WHY NOT 1,000: a valid PCA-512 ceiling needs >=1,707 usable frames under the 30% rank rule.
# 2,000 train puts DM=512 at 512/1999 = 25.6% -- edge-but-valid. At 1,000 frames (800 train) the
# DM=512 ceiling would be VOID BY CONSTRUCTION and the capacity null would repeat with better data.
NSEL = int(os.environ.get("ATLAS_N", "40"))
np.random.seed(0)


def select(n):
    """Stratified across chain length (the only size proxy available before download).
    Length -> atoms measured on 24 real topologies: atoms = 15.77*L - 8, R^2 0.9894."""
    rows = list(csv.DictReader(open(INFO), delimiter="\t"))
    rows = [r for r in rows if r["length"] not in ("", "NA")]
    rows.sort(key=lambda r: float(r["length"]))
    L = np.array([float(r["length"]) for r in rows])
    # LOG-UNIFORM, not rank-uniform: rank-uniform follows the length distribution and clusters near
    # the median (measured quartiles 108/176/300), starving both ends -- and the guard is a SLOPE,
    # so it lives on the extremes. Log-uniform deliberately over-weights them.
    tgt = np.logspace(np.log10(L.min()), np.log10(L.max()), n)
    idx = sorted({int(np.argmin(np.abs(L - t))) for t in tgt})
    return [rows[i] for i in idx]


def main():
    import mdtraj as md
    w = int(sys.argv[1]); nw = int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    tmp = os.environ.get("SLURM_TMPDIR", f"{WR}/atlas_tmp"); os.makedirs(tmp, exist_ok=True)
    items = [x for i, x in enumerate(select(NSEL)) if i % nw == w]
    print(f"[w{w}] {len(items)} proteins assigned", flush=True)
    t0 = time.time(); nd = 0
    for it in items:
        pc = it["PDB"]; dst = f"{OUT}/{pc}.npy"
        if os.path.exists(dst): continue
        wd = f"{tmp}/w{w}"; shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
        z = f"{wd}/a.zip"
        try:
            r = subprocess.run(["curl", "-sL", "--max-time", "3600", URL.format(p=pc), "-o", z],
                               capture_output=True)
            if r.returncode != 0 or not os.path.exists(z): raise RuntimeError("download failed")
            subprocess.run(["unzip", "-qo", z, "-d", wd], check=True)
            os.remove(z)                                        # archive gone before any compute
            top = [f for f in os.listdir(wd) if f.endswith(".pdb")][0]
            xts = sorted(f for f in os.listdir(wd) if f.endswith("_fit.xtc"))
            reps = []
            for x in xts:
                tr = md.load(f"{wd}/{x}", top=f"{wd}/{top}", stride=STRIDE)
                reps.append((tr.xyz.astype(np.float32) * 10.0))  # nm -> Angstrom
            el = [a.element.atomic_number if a.element else 0
                  for a in md.load(f"{wd}/{top}").topology.atoms]
            arr = np.stack(reps)                                 # (R, F, N, 3) float32
            np.save(dst, arr)                                    # UNCOMPRESSED -> memmap-able
            json.dump({"pdb": pc, "length": int(float(it["length"])), "atoms": int(arr.shape[2]),
                       "reps": int(arr.shape[0]), "frames": int(arr.shape[1]),
                       "elem": [int(x) for x in el],
                       "rmsf_ref": float(it.get("avg_RMSF", "nan") or "nan"),
                       "bytes": int(arr.nbytes)}, open(dst.replace(".npy", ".json"), "w"))
            nd += 1
            print(f"[w{w}] {pc} L={it['length']} {reps[0].shape} -> {os.path.getsize(dst)/1e6:.0f} MB "
                  f"({time.time()-t0:.0f}s, {nd} done)", flush=True)
        except Exception as e:
            # FAMILY A: a failure is a FILTER. Log it with its length so the exclusion is auditable.
            fl = f"{OUT}/fail_w{w}.json"
            fj = json.load(open(fl)) if os.path.exists(fl) else {}
            fj[pc] = {"length": it["length"], "err": f"{type(e).__name__}: {e}"}
            json.dump(fj, open(fl, "w"))
            print(f"[w{w}] FAIL {pc} L={it['length']}: {type(e).__name__}", flush=True)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    print(f"[w{w}] FINISHED {nd} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
