"""Resolve b (the atom-count exponent of intrinsic dimensionality) at an UNCENSORED frame budget.

b governs whether latent width must scale with system size, and the width chain turns on it:
b=0.10 -> ~489 dims for a 1e6-atom bound complex, b=0.14 -> ~622, b=0.20 -> ~914, b=0.30 -> ~1720,
where the single-global-latent claim needs qualification rather than a wider DM. Currently b is known
only as a CENSORED LOWER BOUND (+0.101 +/- 0.021, MISATO, 79 frames) because MISATO has only 100
frames, and mdCATH's 28 local domains cannot measure it (CI spans zero at every budget).

The blocker is n=28, not mdCATH: the repo holds 5,398 domains. sqrt(700/28) ~ 5x CI shrink; the ~12x
N range costs ~1.5x against MISATO's 37x, netting ~3.3x, so a +/-0.25 half-width becomes ~+/-0.075 --
enough to separate b=0.14 from b=0.30 at ~2 sigma. And measured at 2,400 frames it is the UNCENSORED
b, not another bound.

STREAM, DO NOT STORE. 3.61 TB total, mean 670 MB/domain -- far too much for a borrowed account's
scratch. Per domain: download -> compute the rank90/RMSF ladder -> write only the spectra -> DELETE
the h5 immediately. Peak disk is (workers x ~2 GB); retained output is a few MB.

STRATIFIED across projected atom count (bytes->atoms calibration measured on the 28 local files:
atoms = 3.0387e-06*bytes + 36.3, R^2 = 0.9944, median error 2.7%). An unstratified draw concentrates
at the median and starves the ends, which is exactly where a slope needs its leverage.

Measured bandwidth 25.0 MB/s single-stream, so bandwidth is not binding; compute (~25 s/domain) is
comparable, hence concurrent workers to overlap the two."""
import json, os, sys, time, subprocess, numpy as np, warnings
warnings.filterwarnings("ignore")
SC = os.environ.get("BSC", "/network/scratch/j/jacob-junqi.tian/latent-model-workspace")
INV = f"{SC}/mdcath_inv.json"; OUTDIR = f"{SC}/bexp"
# TEMPERATURE: 320K ONLY. mdCATH ships five; 28/28 domains are denatured at 450K and 20/28 at 413K,
# and denatured systems have LOW rank90 because unfolding collapses variance into one dominant
# collective motion. Pooling temperatures would corrupt BOTH coefficients -- b through the atom-count
# term and c through the mobility term it conditions on. SCOPE LIMIT: b is measured for FOLDED
# NATIVE-STATE dynamics only.
#
# PER-REPLICA, NOT CONCATENATED. Concatenating the 5 replicas puts between-replica structural offsets
# into the covariance as extra variance directions, inflating rank90 -- an artifact that scales with
# however many replicas (and however long) each domain happens to have, which VARIES (pilot saw
# F = 2190/2200/2500). So: rank90 per replica, averaged within a domain, with the between-replica
# spread kept as a per-domain error bar. The concatenated value is ALSO computed, purely to quantify
# the inflation factor -- which retroactively bounds the earlier concatenated mdCATH numbers,
# including the rank90 asymptote 168 that anchors the width chain.
# REPLICA-COUNT SWEEP. The two artifacts move in OPPOSITE directions with join count -- concatenation
# inflation grows with joins, censoring shrinks with total frames -- so there is an optimum and it can
# be MEASURED rather than argued. For each join count we record: total frames, rank90 as % of usable
# rank (censoring axis), budget-matched inflation vs 1 replica (concatenation axis), and b.
# If b is stable across joins 2-5 the choice does not matter and that stability IS the answer; if b
# moves systematically with join count, that is the concatenation slope-bias detected directly --
# cleaner than inferring it from an inflation-vs-N correlation.
JOINS = [1, 2, 3, 5]
MATCHNF = 400                              # fixed budget for the inflation axis, so joins are comparable
BUDGETS = [79, 200, 400]                   # per-replica ladder (artifact check)
CAT_BUDGETS = [79, 200, 400, 800, 1600, 2000]   # frame-budget ladder on the full 5-join series
# stratification targets: oversample the sparse tails, take the whole top bin (only 50 exist)
STRATA = [((0, 1000), 140), ((1000, 1500), 140), ((1500, 2500), 140),
          ((2500, 4000), 140), ((4000, 6000), 90), ((6000, 99999), 50)]
URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/"
np.random.seed(0)


def kabsch_to(traj, ref):
    rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def select():
    inv = json.load(open(INV)); out = []
    for (lo, hi), k in STRATA:
        pool = sorted([x for x in inv if lo <= x["N"] < hi], key=lambda z: z["path"])
        if len(pool) > k:
            idx = np.linspace(0, len(pool) - 1, k).astype(int)   # even spread within the bin, deterministic
            pool = [pool[i] for i in idx]
        out += pool
    return out


def _one(disp, nf):
    d = disp[:nf]; d = d - d.mean(0)
    mag = np.linalg.norm(d, axis=-1)
    rmsf = float(np.median(np.sqrt((mag ** 2).mean(0))))
    S = np.linalg.svd(d.reshape(nf, -1), compute_uv=False)
    cum = np.cumsum(S ** 2) / ((S ** 2).sum() + 1e-12)
    return rmsf, int(np.searchsorted(cum, 0.90) + 1)


def ladder(per_rep, cat):
    """per_rep: list of (F_i, 3N) displacement arrays, one per replica. cat: concatenated."""
    rec = {}
    # ---- REPLICA-COUNT SWEEP ----
    for J in JOINS:
        if len(per_rep) < J: continue
        j = np.concatenate(per_rep[:J], 0)
        m_, r_ = _one(j, len(j))
        rec[f"j{J}_r"] = int(r_); rec[f"j{J}_nf"] = int(len(j)); rec[f"j{J}_rmsf"] = float(m_)
        rec[f"j{J}_pct"] = float(r_ / max(len(j) - 1, 1))          # censoring axis
        take = MATCHNF // J                                        # concatenation axis, BUDGET-MATCHED
        if all(len(d) >= take for d in per_rep[:J]) and take >= 8:
            mm = np.concatenate([d[:take] for d in per_rep[:J]], 0)
            _, rm = _one(mm, len(mm)); rec[f"j{J}_mix"] = int(rm); rec[f"j{J}_mixnf"] = int(len(mm))
    for nf in BUDGETS:                                    # PRIMARY: per-replica, averaged
        rs, ms = [], []
        for d in per_rep:
            if nf <= len(d):
                m_, r_ = _one(d, nf); ms.append(m_); rs.append(r_)
        if not rs: continue
        rec[f"r{nf}"] = float(np.mean(rs)); rec[f"rmsf{nf}"] = float(np.mean(ms))
        rec[f"rsd{nf}"] = float(np.std(rs)); rec[f"nrep{nf}"] = len(rs)   # between-replica spread
    # BUDGET-MATCHED inflation test: the same nf frames, but SPREAD ACROSS replicas instead of taken
    # from one. cat[:nf] would just be replica 0 for nf <= replica length, which tests nothing.
    for nf in BUDGETS:
        per_take = max(2, nf // max(len(per_rep), 1))
        chunks = [d[:per_take] for d in per_rep if len(d) >= per_take]
        if len(chunks) < 2: continue
        mixed = np.concatenate(chunks, 0)
        if len(mixed) < 8: continue
        m_, r_ = _one(mixed, len(mixed))
        rec[f"mix_r{nf}"] = int(r_); rec[f"mix_nf{nf}"] = int(len(mixed))
    for nf in CAT_BUDGETS:                                # full concatenated series (what earlier runs used)
        if nf > len(cat): continue
        m_, r_ = _one(cat, nf); rec[f"cat_r{nf}"] = int(r_); rec[f"cat_rmsf{nf}"] = float(m_)
    return rec


def main():
    import h5py
    w = int(sys.argv[1]); nw = int(sys.argv[2])
    tmp = os.environ.get("SLURM_TMPDIR", f"{SC}/bexp_tmp"); os.makedirs(tmp, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    outf = f"{OUTDIR}/w{w}.json"
    done = json.load(open(outf)) if os.path.exists(outf) else {}
    items = [x for i, x in enumerate(select()) if i % nw == w]
    print(f"[w{w}] {len(items)} domains assigned, {len(done)} already done", flush=True)
    t0 = time.time(); nd = 0
    for it in items:
        dom = it["path"].split("mdcath_dataset_")[-1][:-3]
        if dom in done: continue
        loc = f"{tmp}/w{w}.h5"
        try:
            r = subprocess.run(["curl", "-sL", "--max-time", "900", URL + it["path"], "-o", loc],
                               capture_output=True)
            if r.returncode != 0 or not os.path.exists(loc): raise RuntimeError("download failed")
            ff = h5py.File(loc, "r"); dm = list(ff.keys())[0]; g = ff[dm]
            T0 = sorted([k for k in g.keys() if k.isdigit()], key=int)[0]   # 320K ONLY (lowest); see header
            reps = sorted(g[T0].keys(), key=lambda x: int(x))
            ref = np.array(g[T0][reps[0]]["coords"][0]).astype(np.float64)
            per = [kabsch_to(np.array(g[T0][r_]["coords"]).astype(np.float64), ref) - ref for r_ in reps]
            ff.close()
            cat = np.concatenate(per, 0)
            rec = {"N": int(cat.shape[1]), "projN": it["N"], "F": int(len(cat)),
                   "nrep": len(per), "replen": [int(len(x)) for x in per], "T": int(T0)}
            rec.update(ladder(per, cat))
            done[dom] = rec; json.dump(done, open(outf, "w")); nd += 1
            if nd % 10 == 0:
                print(f"[w{w}] {nd} done ({time.time()-t0:.0f}s, {(time.time()-t0)/nd:.1f}s/domain)", flush=True)
        except Exception as e:
            print(f"[w{w}] {dom}: {type(e).__name__}: {e}", flush=True)
        finally:
            if os.path.exists(loc): os.remove(loc)          # DELETE IMMEDIATELY -- never accumulate
    print(f"[w{w}] FINISHED {nd} new, {len(done)} total ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
