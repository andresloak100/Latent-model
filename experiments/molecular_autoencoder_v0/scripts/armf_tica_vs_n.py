"""DOES SLOW-MODE DIMENSIONALITY GROW WITH ATOM COUNT? (INBOX 007/008)

rank90 grows out of sample at b >= +0.9285 +/- 0.2220 -- near-linear in N. But rank90 is
VARIANCE-weighted: it counts modes carrying 90% of displacement variance, and near-linear growth in
that is close to what independent local thermal motion would produce, since every extra atom brings
its own fast low-amplitude degrees of freedom. Those modes are real. They are NOT the dynamics a
latent generator has to represent.

The slowness-weighted analogue was measured for objective 3 and was FLAT: TICA dimensionality held at
32-42 across five temperatures at 36-38% of its basis (uncensored), with a bound 5x tighter than
rank90's. But that was flat across EFFECTIVE TIME. It has never been measured across N.

    THE QUESTION THAT DECIDES THE ARCHITECTURE:
    does TICA dimensionality grow with atom count, or is slow-mode dimensionality flat in N
    the way it is flat in time?

DEFINITION -- deliberately IDENTICAL to armf_slowness.py:54 so the two results are comparable:
PCA-project to a CONSTANT basis m, generalised eigenproblem eigh(Ctau, C0), |autocorrelation| per
independent component sorted slowest-first, and the dimension is the count reaching 90% of sum(lam^2).
A basis that varies across systems would make the dimension track the basis rather than the slowness.

FOUR GUARDS, every one of which bit the rank90 version:

  FAMILY E -- THE LAG TIME IS A HYPERPARAMETER. tau is SWEPT ON TRAINING SYSTEMS, one value picked,
    and applied unchanged to held-out. A per-system tau would make the dimension track the lag. The
    whole sweep is reported, not just the winner.

  FAMILY B -- TICA NEEDS MORE SAMPLES THAN PCA (it estimates a time-lagged covariance, so the
    effective sample count is what matters, and n_eff is 1-7% of frames on this corpus). Reported:
    the dimension as a % of basis, n_eff per dimension, AND a re-measurement at a LARGER basis. If
    the exponent moves with the basis, the constant-basis number is censored and a flat slope would
    be a ceiling rather than a result.

  FAMILY A -- ANY SYSTEM FAILING TO REACH 90% OF THE SLOW SPECTRUM IS AN EXCLUSION, and for rank90
    those concentrated at high N (the 5 dropped systems had median N=14,018 against a corpus median
    of 3,249). Kept-vs-dropped median N is reported.

  IN-SAMPLE AND OUT-OF-SAMPLE BOTH. rank90's exponent DOUBLED between them (+0.4657 -> +0.9285), so
    nothing is assumed for TICA. Out-of-sample means: PCA basis and TIC vectors both fitted on
    replicas 0+1, then the held-out replica projected onto those fixed TICs and each component's
    autocorrelation MEASURED ON HELD-OUT, keeping the train ordering. If the train ordering
    generalises badly, more components are needed -- which is exactly the effect being looked for.

CROSS-REPLICA CORRECTNESS: the train set is replicas 0+1 CONCATENATED, so a lagged covariance taken
across the join would pair the last frames of replica 0 with the first of replica 1 -- unrelated
conformations from different runs, entering as spurious decorrelation. Lagged products are
accumulated WITHIN each replica only.

PRE-REGISTERED READS (recorded before results exist)
  exponent flat (CI includes 0, excludes ~0.5) -> slow dynamics are low-dimensional regardless of
      system size; the one-token architecture is viable FOR THE DYNAMICS THAT MATTER, and the
      N^0.93 variance result describes fast local noise a generator need not represent explicitly.
  exponent tracks rank90 (~0.9) -> the information limit is real even for slow dynamics and fixed
      width faces a genuine ceiling. Report it plainly.
  in between -> report the exponent with CI and NO verdict.

IF FLAT, the consequence to state (but NOT to act on yet): a variance-weighted objective -- plain MSE
on displacement -- spends the token's capacity in proportion to variance, i.e. mostly on the fast
local modes growing as N^0.93. If slow-mode dimensionality is flat, MSE is the wrong training
objective for this architecture. That is the "objective mismatch" row of the 004c fan-out, and it
would then have evidence behind it instead of being one hypothesis among six. Measure first."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats, linalg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
OUT = f"{WR}/atlas_tica_vs_n.json"
CHUNK = 20000
TICA_BASIS = 100          # canonical, matches armf_slowness.py so the numbers are comparable
BIG_BASIS = 400           # (W//10 = 500 caps it) # Family B: does the exponent move when the cap is raised?
TAUS = [5, 10, 20, 40]    # Family E: swept on TRAINING systems, winner applied unchanged
FRAC = 0.90


def _dim(lam, frac=FRAC):
    kv = np.clip(np.abs(lam), 0, 1) ** 2
    c = np.cumsum(kv) / (kv.sum() + 1e-12)
    hit = np.nonzero(c >= frac)[0]
    return (int(hit[0] + 1) if len(hit) else None)


def tica_train(Ztr, reps, tau, m):
    """TICA on the train PCA coords. `reps` are (start, end) spans; lagged pairs stay WITHIN a span."""
    Z = Ztr[:, :m]; W = len(Z)
    C0 = Z.T @ Z / (W - 1) + 1e-8 * np.eye(m)
    A = np.zeros((m, m)); npair = 0
    for s, e in reps:                                  # never pair across the replica join
        if e - s <= tau: continue
        A += Z[s:e - tau].T @ Z[s + tau:e]; npair += (e - s - tau)
    if npair < m: return None, None, None
    Ct = (A + A.T) / (2 * npair)
    try:
        lam, V = linalg.eigh(Ct, C0)
    except Exception:
        return None, None, None
    o = np.argsort(np.abs(lam))[::-1]
    return np.clip(np.abs(lam[o]), 0, 1), V[:, o], C0


def tica_holdout(Zho, V, tau, m):
    """Project held-out onto the TRAIN TICs and measure each one's autocorrelation ON HELD-OUT,
    keeping the TRAIN ordering -- a train ordering that generalises badly then costs more components."""
    S = Zho[:, :m] @ V
    S = S - S.mean(0); sd = S.std(0) + 1e-12; S = S / sd
    if len(S) <= tau: return None
    return np.clip(np.abs((S[:-tau] * S[tau:]).mean(0)), 0, 1)


def coords(path):
    """Train PCA coords (frames 0..2F) and held-out coords IN THE SAME BASIS, frame-space throughout."""
    a = np.load(path, mmap_mode="r"); F, N = a.shape[1], a.shape[2]
    D3 = 3 * N
    mu = np.zeros(D3)
    for r in (0, 1): mu += np.asarray(a[r]).reshape(F, -1).astype(np.float64).sum(0)
    mu /= (2 * F)
    Ftr = 2 * F
    G = np.zeros((Ftr, Ftr)); M = np.zeros((F, Ftr))
    for c0 in range(0, D3, CHUNK):
        c1 = min(c0 + CHUNK, D3)
        tr = np.concatenate([np.asarray(a[r]).reshape(F, -1)[:, c0:c1].astype(np.float64)
                             for r in (0, 1)], 0) - mu[c0:c1]
        ho = np.asarray(a[2]).reshape(F, -1)[:, c0:c1].astype(np.float64) - mu[c0:c1]
        G += tr @ tr.T; M += ho @ tr.T
        del tr, ho
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]; w = np.clip(w[o], 0, None); U = U[:, o]
    nz = int((w > w[0] * 1e-12).sum()); s = np.sqrt(w[:nz])
    return (U[:, :nz] * s), (M @ U[:, :nz]) / s, int(N), int(F), nz


def measure(path, tau, m_req):
    Ytr, Yho, N, F, nz = coords(path)
    m = max(2, min(m_req, (2 * F) // 10, nz))
    reps = [(0, F), (F, 2 * F)]
    lam_tr, V, _ = tica_train(Ytr, reps, tau, m)
    if lam_tr is None: return None
    lam_ho = tica_holdout(Yho, V, tau, m)
    # ---- TRUNCATION-MATCHED CONTROL (the control this whole comparison needs) ----
    # TICA is computed inside an m-dimensional PCA basis; rank90 is not truncated at all. So
    # "slow-mode dimensionality is flatter in N than variance dimensionality" could be nothing but a
    # TRUNCATION effect -- any count confined to m components is bounded by m and its N-slope is
    # compressed toward zero. The only way to tell is to measure VARIANCE dimensionality inside the
    # SAME m-dimensional basis and compare like with like:
    #   pca_dim_in_m  ~ TICA's exponent  => the flatness is TRUNCATION, and says nothing about slowness
    #   pca_dim_in_m  >> TICA's exponent => the flatness is SLOWNESS, which is the 007 claim
    wtr = (Ytr[:, :m] ** 2).sum(0)
    c = np.cumsum(wtr) / (wtr.sum() + 1e-12)
    pdi = int(np.searchsorted(c, FRAC) + 1)
    who = (Yho[:, :m] ** 2).sum(0)
    ws = np.sort(who)[::-1]
    c2 = np.cumsum(ws) / (ws.sum() + 1e-12)
    pdo = int(np.searchsorted(c2, FRAC) + 1)
    # TWO out-of-sample readings, because they answer different questions and the dry run showed the
    # first alone pins against the basis (87% of m) and would report a CEILING as a flat slope:
    #   dim_out       -- train ORDER kept. Ordering-sensitive: if the train TICs generalise in the
    #                    wrong order, more components are needed. Directly analogous to rank90_out.
    #   dim_out_sorted-- ranked by ACTUAL held-out slowness. Ordering-free, so it measures the
    #                    DIMENSIONALITY of held-out slow content in the train basis and nothing else.
    # dim_out >> dim_out_sorted means the basis is fine but its ORDER does not transfer, which is a
    # different defect from slow dynamics being high-dimensional -- and only the second bears on the
    # architecture question.
    return dict(N=N, F=F, nz=nz, m=m, tau=tau,
                pca_dim_in_m=pdi, pca_dim_out_m=pdo,
                dim_in=_dim(lam_tr),
                dim_out=(_dim(lam_ho) if lam_ho is not None else None),
                dim_out_sorted=(_dim(np.sort(lam_ho)[::-1]) if lam_ho is not None else None),
                lam5=[float(x) for x in lam_tr[:5]])


def reg(x, y, lab, ref=None):
    if len(x) < 3: print(f"  {lab}: n={len(x)}, too few"); return None, None
    lr = stats.linregress(x, y); hw = stats.t.ppf(0.975, len(x) - 2) * lr.stderr
    tag = "FLAT (CI includes 0)" if abs(lr.slope) < hw else ("GROWS" if lr.slope > 0 else "SHRINKS")
    extra = ""
    if ref is not None:
        extra = ("   EXCLUDES rank90's %+.3f" % ref) if (lr.slope + hw < ref or lr.slope - hw > ref) \
                else ("   consistent with rank90's %+.3f" % ref)
    print(f"  log10({lab:26s}) ~ {lr.slope:+.4f} +/- {hw:.4f} * log10(N)  R^2 {lr.rvalue**2:.3f}  "
          f"n={len(x)}  -> {tag}{extra}", flush=True)
    return lr.slope, hw


if __name__ == "__main__":
    man = json.load(open(f"{WR}/atlas_manifest.json"))
    store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: m for m in store.meta}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]
    print(f"[tica-vs-N] held-out {len(ho_ids)} cached, train pool {len(tr_ids)} cached", flush=True)

    # ---------- FAMILY E: sweep tau on TRAINING systems only ----------
    swp_f = f"{WR}/atlas_tica_tausweep.json"
    swp = json.load(open(swp_f)) if os.path.exists(swp_f) else {}
    probe = [tr_ids[i] for i in np.linspace(0, len(tr_ids) - 1, min(12, len(tr_ids))).astype(int)]
    print(f"\n=== FAMILY E: lag-time sweep on {len(probe)} TRAINING systems (never held-out) ===",
          flush=True)
    for tau in TAUS:
        k = str(tau)
        if k not in swp:
            ds = []
            for p in probe:
                try:
                    r = measure(have[p]["path"], tau, TICA_BASIS)
                    if r and r["dim_in"]: ds.append(r["dim_in"])
                except Exception as e:
                    print(f"    tau={tau} {p}: {type(e).__name__}", flush=True)
            swp[k] = ds; json.dump(swp, open(swp_f, "w"))
        d = swp[k]
        print(f"  tau={tau:>3} ({tau*40:>5} ps): median dim {np.median(d) if d else float('nan'):.1f}  "
              f"n={len(d)}", flush=True)
    # pick the LOWEST tau whose median dim has stabilised (within 10% of the next) -- a longer lag
    # discards fast modes but also throws away lagged pairs, so the smallest stable tau is the choice.
    meds = {int(k): float(np.median(v)) for k, v in swp.items() if v}
    ks = sorted(meds)
    TAU = ks[-1]
    for i in range(len(ks) - 1):
        if abs(meds[ks[i]] - meds[ks[i + 1]]) <= 0.10 * max(meds[ks[i + 1]], 1e-9):
            TAU = ks[i]; break
    print(f"  -> TAU = {TAU} ({TAU*40} ps), chosen on TRAINING systems, applied UNCHANGED to held-out",
          flush=True)

    # ---------- measure on held-out systems ----------
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["pdb"], r["m_req"]) for r in rows}
    for m_req in (TICA_BASIS, BIG_BASIS):
        print(f"\n=== held-out systems at basis {m_req} ===", flush=True)
        for p in ho_ids:
            if (p, m_req) in done: continue
            try:
                r = measure(have[p]["path"], TAU, m_req)
                if r is None:
                    print(f"  FAIL {p} N={have[p]['atoms']}: eigenproblem", flush=True); continue
                r["pdb"] = p; r["m_req"] = m_req
                rows.append(r); json.dump(rows, open(OUT, "w"))
                if len(rows) % 20 == 0: print(f"    {len(rows)}  last N={r['N']}", flush=True)
            except Exception as e:
                print(f"  FAIL {p} N={have[p]['atoms']}: {type(e).__name__}: {e}", flush=True)

    print(f"\n{'='*78}\n=== RESULT: DOES SLOW-MODE DIMENSIONALITY GROW WITH N? ===\n{'='*78}", flush=True)
    RANK90_OUT = 0.9285
    for m_req in (TICA_BASIS, BIG_BASIS):
        R = [r for r in rows if r["m_req"] == m_req]
        if not R: continue
        N = np.array([r["N"] for r in R], float); x = np.log10(N)
        di = np.array([r["dim_in"] if r["dim_in"] else np.nan for r in R], float)
        do = np.array([r["dim_out"] if r["dim_out"] else np.nan for r in R], float)
        ds = np.array([r.get("dim_out_sorted") or np.nan for r in R], float)
        mm = np.array([r["m"] for r in R], float)
        print(f"\n--- basis {m_req} (realised median m={np.median(mm):.0f}), tau={TAU}, n={len(R)}, "
              f"N {int(N.min())}-{int(N.max())} ---", flush=True)
        # FAMILY A
        for nm, v in (("in-sample", di), ("out-of-sample", do), ("out-sorted", ds)):
            bad = ~np.isfinite(v)
            if bad.sum():
                print(f"  FAMILY A [{nm}]: {bad.sum()}/{len(v)} never reach {FRAC:.0%} of the slow "
                      f"spectrum. dropped median N {np.median(N[bad]):.0f} vs kept "
                      f"{np.median(N[~bad]):.0f}"
                      + ("   <-- DROPS CONCENTRATE AT HIGH N: the slope below is biased LOW"
                         if np.median(N[bad]) > np.median(N[~bad]) else ""), flush=True)
            else:
                print(f"  FAMILY A [{nm}]: 0 exclusions.", flush=True)
        # FAMILY B
        for nm, v in (("in-sample", di), ("out-of-sample", do), ("out-sorted", ds)):
            g = np.isfinite(v)
            if not g.sum(): continue
            pct = 100 * np.median(v[g] / mm[g])
            print(f"  FAMILY B [{nm}]: dim is {pct:.0f}% of basis (median {np.median(v[g]):.0f} of "
                  f"{np.median(mm[g]):.0f})"
                  + ("   <-- OVER 60% OF BASIS: CENSORED, a flat slope here would be a CEILING"
                     if pct > 60 else "   (clear of the basis cap)"), flush=True)
        # the exponents
        print(f"  vs rank90 out-of-sample b = +{RANK90_OUT:.4f} +/- 0.2220 (itself a LOWER bound):",
              flush=True)
        pdi = np.array([r.get("pca_dim_in_m") or np.nan for r in R], float)
        pdo = np.array([r.get("pca_dim_out_m") or np.nan for r in R], float)
        for nm, v in (("TICA dim in-sample", di), ("TICA dim OUT (train order)", do),
                      ("TICA dim OUT (sorted)", ds),
                      ("PCA dim in SAME basis", pdi), ("PCA dim OUT in SAME basis", pdo)):
            g = np.isfinite(v)
            if g.sum() >= 3: reg(x[g], np.log10(v[g]), nm, ref=RANK90_OUT)
        gi = np.isfinite(di) & np.isfinite(pdi)
        if gi.sum() >= 3:
            st_, _ = stats.linregress(x[gi], np.log10(di[gi]))[:2]
            sp_, _ = stats.linregress(x[gi], np.log10(pdi[gi]))[:2]
            print(f"  TRUNCATION CONTROL: at basis {m_req}, VARIANCE dimensionality inside the SAME "
                  f"basis grows at {sp_:+.4f} while SLOWNESS grows at {st_:+.4f}.", flush=True)
            if abs(sp_) < 2 * abs(st_) + 0.02:
                print(f"    => BOTH are flattened by the truncation. The TICA flatness is a "
                      f"TRUNCATION artifact and says NOTHING about slowness. Family D.", flush=True)
            else:
                print(f"    => variance dimensionality is {abs(sp_)/max(abs(st_),1e-9):.1f}x steeper "
                      f"than slowness IN THE SAME BASIS, so the flatness is SLOWNESS, not "
                      f"truncation. That is the 007 claim, controlled.", flush=True)

    # ---------- verdict on the canonical basis, using the pre-registered rule ----------
    R = [r for r in rows if r["m_req"] == TICA_BASIS]
    Rb = [r for r in rows if r["m_req"] == BIG_BASIS]
    if R:
        N = np.array([r["N"] for r in R], float)
        # VERDICT uses the ORDERING-FREE out-of-sample dimension: the train-order variant pins
        # against the basis, and a quantity at its measurement ceiling cannot support a flat call.
        do = np.array([r.get("dim_out_sorted") or np.nan for r in R], float)
        mm = np.array([r["m"] for r in R], float)
        g = np.isfinite(do)
        if g.sum() >= 3 and 100 * np.median(do[g] / mm[g]) > 60:
            print("\n=== VERDICT WITHHELD: the ordering-free out-of-sample dimension is still above")
            print("    60% of its basis, so it is CENSORED and a flat slope would be a ceiling.")
            print("    Read the larger-basis rows instead. ===", flush=True)
            g = np.zeros_like(g)
        if g.sum() >= 3:
            s, hw = reg(np.log10(N[g]), np.log10(do[g]), "VERDICT (ordering-free, out-of-sample)")
            print(f"\n=== VERDICT (pre-registered) ===", flush=True)
            if abs(s) < hw and s + hw < 0.5:
                print("  TICA EXPONENT IS FLAT and EXCLUDES ~0.5.")
                print("  => SLOW dynamics are low-dimensional REGARDLESS OF SYSTEM SIZE. The one-token")
                print("     architecture is viable FOR THE DYNAMICS THAT MATTER, and the N^0.93 variance")
                print("     result describes fast local noise a generator need not represent explicitly.")
                print("  => CONSEQUENCE TO STATE, NOT YET TO ACT ON: a variance-weighted objective (plain")
                print("     MSE on displacement) spends the token's capacity in proportion to VARIANCE,")
                print("     i.e. mostly on the fast local modes that grow as N^0.93. If slow-mode")
                print("     dimensionality is flat, MSE IS THE WRONG TRAINING OBJECTIVE here. That is the")
                print("     'objective mismatch' row of the 004c fan-out, now with evidence behind it")
                print("     rather than being one hypothesis among six. DO NOT change the loss yet.")
            elif s - hw > 0.5:
                print("  TICA EXPONENT TRACKS rank90.")
                print("  => The information limit is REAL even for slow dynamics, and a fixed-width token")
                print("     faces a genuine ceiling. Report plainly; this is a finding, not a setback.")
            else:
                print(f"  IN BETWEEN: {s:+.4f} +/- {hw:.4f}. Report the exponent with CI. NO VERDICT.")
    if Rb and R:
        print("\n  BASIS CHECK (Family B): if the two bases disagree, the constant-basis exponent is")
        print("  censored and the flat/steep call belongs to the LARGER basis.", flush=True)
