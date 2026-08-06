"""ATLAS cache builder: download -> SUBSAMPLE -> store compact -> DELETE the archive.

WHY SUBSAMPLE RATHER THAN STREAM. MISATO's PCA baseline overfits because 80 frames sit BELOW the
median rank90 (~168); the fix is frames comfortably ABOVE rank90, not all 10,001. ~1,000 frames per
replica gives ~800 train frames, ~5x the median rank90, which is well-conditioned. Frames beyond that
buy almost nothing for a PCA ceiling at k<=24.

WHY STORE RATHER THAN STREAM. Streaming was right for the b-exponent (spectra out, data discarded).
It is WRONG here: TRAINING reads frames many times over, so the subsampled set must be reusable
rather than re-downloaded per run. The archive is deleted immediately; only the subsample is kept.

MEASURED (not assumed):
  - 10,001 frames/replica at 10 ps, confirmed against the production .mdp (nsteps 50e6 x dt 2 fs)
  - 14.7 atoms/residue all-atom (2,701 atoms for a 184-residue chain) -- NOT ~9.5
  - 1,001 frames x 3 replicas = 97 MB/protein stored float32 -> ~65 GB for 700, ~179 GB for 1,938
  - ATLAS bandwidth 4.9 MB/s (NOT HuggingFace's 25): 700 proteins ~ 0.34 TB ~ 19 h serial, ~5 h on 4
  - mdtraj 1.10.3 loaded from a wheel unzipped onto PYTHONPATH (the venv has no SSL, so pip is dead;
    the shared venv is never modified)

SCOPE NOTE (uniform subsampling): taking every 10th frame across the full 100 ns preserves the SLOW
COLLECTIVE MODES that dominate variance at low k. What is lost is fast local motion, which is not
what a k<=24 ceiling measures. State this wherever the ATLAS ceiling is used.

SCOPE NOTE (corpus): ATLAS is SINGLE CHAINS; MISATO is COMPLEXES. Different chemistry and mobility
regimes -- do not pool without checking, and expect ATLAS to sit on the floppier side (mdCATH single
domains measured 1.65x the RMSF of MISATO complexes). ATLAS's top is ~20,216 atoms vs MISATO's
26,861, so it extends the BOTTOM of the N axis, not the top: it buys a valid ceiling and large n,
NOT a longer lever for the 1e6-atom extrapolation."""
import os, sys, csv, json, time, subprocess, shutil, numpy as np, warnings
warnings.filterwarnings("ignore")
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
LIB = f"{WR}/pylibs"; sys.path.insert(0, LIB)
OUT = f"{WR}/atlas_cache"; INFO = f"{WR}/atlas_info.tsv"
URL = "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{p}/{p}_protein.zip"
STRIDE = 10                       # 10,001 -> 1,001 frames per replica (~800 train after an 80/20 split)
NSEL = int(os.environ.get("ATLAS_N", "40"))
np.random.seed(0)


def select(n):
    """Stratified across chain length (the only size proxy available before download).
    Length correlates with atoms at 14.7 atoms/residue, measured."""
    rows = list(csv.DictReader(open(INFO), delimiter="\t"))
    rows = [r for r in rows if r["length"] not in ("", "NA")]
    rows.sort(key=lambda r: float(r["length"]))
    idx = np.linspace(0, len(rows) - 1, n).astype(int)          # even spread over the WHOLE range
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
        pc = it["PDB"]; dst = f"{OUT}/{pc}.npz"
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
            np.savez_compressed(dst, coords=np.stack(reps), elem=np.array(el, np.int16),
                                length=int(float(it["length"])), pdb=pc,
                                rmsf_ref=float(it.get("avg_RMSF", "nan") or "nan"))
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
