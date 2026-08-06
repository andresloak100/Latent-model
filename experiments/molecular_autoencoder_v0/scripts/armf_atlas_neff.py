"""EFFECTIVE SAMPLE SIZE, not frame count -- and whether IAT-vs-N explains the guard slope.

n_eff = F / tau_int, tau_int = 1 + 2*sum_{k>=1} rho(k) truncated at rho < 0.05, computed on PCA-mode
coefficient time series. The Gram's eigenvectors ARE those series: for X (F,3N), G = X X^T (F,F),
eigh -> U,w, and mode i's coefficients are U[:,i]*sqrt(w_i). No 3N-sized array is ever formed.

FOUR ARTIFACT CHECKS RUN BEFORE THIS WAS TRUSTED (5 systems, N = 598 -> 33,377):

  A. RIGID-BODY MOTION -- RULED OUT. If frames were not superposed, PC1 would be global rotation,
     whose correlation time grows with radius and would manufacture BOTH a huge tau and a spurious
     IAT-vs-N slope. Measured: Kabsch removes 1.5%/0.8%/0.3%/0.1%/0.0% of variance (DEcreasing with
     N), COM drift <= 0.55 A, net rotation <= 2.4 deg, and tau_raw vs tau_aligned agree to <1%
     (716.6 vs 719.0, 1060.1 vs 1060.0). ATLAS ships superposed trajectories. No alignment needed.

  B. LAG-CAP CENSORING -- FOUND AND FIXED. The first implementation capped maxlag at 500; 4 of 5
     systems needed lags of 625-841 to reach rho < 0.05. That truncation censors LARGE tau
     preferentially -- i.e. it censors exactly the variable under test, and biases the IAT-vs-N slope
     DOWNWARD if tau correlates with N. This is Family B (a count pinned to its measurement ceiling).
     maxlag is now F//2 and every system carries a `conv` flag; non-converged tau are LOWER BOUNDS
     and are reported as such rather than averaged in silently.

  C. MODE-INDEX DEPENDENCE -- CHANGES THE CONCLUSION. tau is not a per-system scalar; it collapses
     with mode index: ~700-1060 (mode 0), ~17 (mode 49), ~3 (mode 199), ~1-2 (mode 499). So the
     LEADING modes are starved while the bulk of rank90 is effectively independent frame-to-frame.
     Comparing n_eff(PC1) against rank90 would pit the worst-case mode against a count dominated by
     well-sampled ones. Everything below is therefore MODE-RESOLVED: n_eff_k uses the mean tau over
     modes 0..k-1, at exactly the k the guard uses (24 and 256).

  D. SELECTION EFFECT -- REAL, QUANTIFIED, NOT CORRECTED OUT. PCA chooses the highest-variance
     direction, which on a finite trajectory preferentially finds slow drift, so tau(PC1) is an
     extreme order statistic and is upward-biased by construction. Measured against a random-
     projection null: 3.1x / 6.1x / 5.0x / 3.5x / 9.5x. tau(PC1) therefore OVERSTATES how starved a
     typical direction is; the random-null tau is reported alongside as the unselected reference.

WHAT THIS DECIDES. If tau grows with N, bigger proteins have fewer effective samples at fixed
trajectory length, no amount of denser subsampling or extra replicas fixes it (n_eff is set by
tau_int, not by stride, and replicas measured at a 1.18x between/within ratio add little), and the
ceiling is IRREDUCIBLY N-biased on ATLAS. That is survivable: the PRIMARY comparison -- codec vs ANM,
both zero-shot, same held-out frames -- needs no ceiling and is untouched. The oracle-fraction would
become a permanently caveated SECONDARY rather than something to fix."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
OUT = f"{WR}/atlas_neff.json"
KS = [24, 256]                       # the k the guard actually uses
IDX = [0, 1, 4, 9, 49, 199, 499]     # mode-index ladder for the tau profile
RHO_CUT = 0.05
NRAND = 12                           # random-projection null draws


def acf_batch(Y):
    """Full autocorrelation for every column of Y (F,K) at once, via FFT. Unbiased 1/(F-k) norm."""
    F = Y.shape[0]
    Y = Y - Y.mean(0)
    n2 = 1 << (2 * F - 1).bit_length()
    F_ = np.fft.rfft(Y, n=n2, axis=0)
    ac = np.fft.irfft(F_ * np.conj(F_), n=n2, axis=0)[:F]
    ac /= np.arange(F, 0, -1)[:, None]
    return ac / (ac[0] + 1e-300)


def taus_from_acf(R, maxlag):
    """tau_int per column + convergence flag. Truncate at the FIRST lag with rho < RHO_CUT."""
    out, conv, lags = [], [], []
    for j in range(R.shape[1]):
        r = R[:maxlag, j]
        below = np.nonzero(r < RHO_CUT)[0]
        k = int(below[0]) if len(below) else maxlag
        out.append(max(1.0 + 2.0 * float(r[1:k].sum()), 1.0))
        conv.append(bool(len(below))); lags.append(k)
    return np.array(out), np.array(conv), np.array(lags)


def per_replica(path, r=0):
    a = np.load(path, mmap_mode="r"); F, N = a.shape[1], a.shape[2]
    G = np.zeros((F, F))
    for c0 in range(0, 3 * N, 20000):
        c1 = min(c0 + 20000, 3 * N)
        X = np.asarray(a[r]).reshape(F, -1)[:, c0:c1].astype(np.float64)
        X -= X.mean(0); G += X @ X.T; del X
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]; w = np.clip(w[o], 0, None); U = U[:, o]
    nz = int((w > w[0] * 1e-12).sum()); maxlag = F // 2
    cum = np.cumsum(w) / (w.sum() + 1e-12); r90 = int(np.searchsorted(cum, 0.90) + 1)
    # K must COVER rank90: beyond ~256 the modes are the FAST ones (tau ~1-3), so averaging tau over
    # only the first 256 would UNDERSTATE n_eff for the full rank90 subspace and unfairly deflate the
    # n_eff/rank90 comparison. Denominator and numerator must span the same modes.
    K = min(max(max(KS), r90), nz)
    C = U[:, :K] * np.sqrt(w[:K])                       # PCA coefficient series, (F,K)
    tau, conv, lags = taus_from_acf(acf_batch(C), maxlag)
    d = dict(N=int(N), F=int(F), nz=nz, rank90=r90,
             tau_lead=float(tau[0]), conv_lead=bool(conv[0]), lag_lead=int(lags[0]),
             ncens=int((~conv).sum()), K=int(K),
             r90_pct=float(r90 / max(nz, 1)))            # rank90 as % of usable rank (censoring check)
    for k in KS:                                         # n_eff supporting a k-dim subspace
        kk = min(k, K)
        d[f"tau{k}"] = float(tau[:kk].mean()); d[f"neff{k}"] = float(F / tau[:kk].mean())
        d[f"kk{k}"] = kk
        d[f"per_dim{k}"] = float(F / tau[:kk].mean() / kk)   # <1 => fitting more dims than samples
    kr = min(r90, K)                                     # tau over EXACTLY the rank90 modes
    d["tau_r90"] = float(tau[:kr].mean()); d["neff_r90"] = float(F / tau[:kr].mean())
    d["neff_per_r90"] = float(F / tau[:kr].mean() / r90)
    for i in IDX:
        d[f"tau_m{i}"] = float(tau[i]) if i < K else None
    # D: unselected reference direction
    rng = np.random.default_rng(0); tr = []
    for _ in range(NRAND):
        v = rng.normal(size=K); v /= np.linalg.norm(v)
        tr.append(taus_from_acf(acf_batch((C @ v)[:, None]), maxlag)[0][0])
    d["tau_rand"] = float(np.median(tr))
    return d


def reg(x, y, lab, note=""):
    lr = stats.linregress(x, y); hw = stats.t.ppf(0.975, len(x) - 2) * lr.stderr
    sig = "SIGNIFICANT" if (lr.slope - hw > 0 or lr.slope + hw < 0) else "not significant"
    print(f"  log10({lab:24s}) ~ {lr.slope:+.4f} +/- {hw:.4f} * log10(N)   R^2 {lr.rvalue**2:.3f}   "
          f"p={lr.pvalue:.3f}   {sig}{note}", flush=True)
    return lr.slope, hw


if __name__ == "__main__":
    store = AtlasStore(f"{WR}/atlas_cache")
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["pdb"] for r in rows}
    for m in store.meta:
        if m["pdb"] in done: continue
        try:
            d = per_replica(m["path"]); d["pdb"] = m["pdb"]
            rows.append(d); json.dump(rows, open(OUT, "w"))
            if len(rows) % 25 == 0: print(f"    {len(rows)}  last N={d['N']}", flush=True)
        except Exception as e:
            print(f"  FAIL {m['pdb']} N={m['atoms']}: {type(e).__name__}: {e}", flush=True)

    N = np.array([r["N"] for r in rows], float); F = np.array([r["F"] for r in rows], float)
    r90 = np.array([r["rank90"] for r in rows], float)
    tl = np.array([r["tau_lead"] for r in rows]); tr_ = np.array([r["tau_rand"] for r in rows])
    x = np.log10(N)
    ncens = sum(r["ncens"] for r in rows); ntot = sum(r["K"] for r in rows)
    nlead = sum(1 for r in rows if not r["conv_lead"])
    print(f"\n[atlas-neff] {len(rows)} systems, N {int(N.min())}-{int(N.max())} "
          f"({N.max()/N.min():.1f}x), log-N span {x.max()-x.min():.2f} decades", flush=True)
    print(f"  B (lag-cap censoring): {ncens}/{ntot} mode-series did not reach rho<0.05 "
          f"({100*ncens/max(ntot,1):.1f}%); of the LEADING modes {nlead}/{len(rows)} are LOWER BOUNDS",
          flush=True)
    print(f"  D (selection effect): tau(PC1)/tau(random) median {np.median(tl/tr_):.1f}x -- PC1 is an "
          f"extreme order statistic and OVERSTATES how starved a typical direction is", flush=True)

    print(f"\n=== C. tau BY MODE INDEX -- the leading modes are starved, the bulk is not ===", flush=True)
    print(f"  {'mode':>6}{'tau (median)':>15}{'n_eff (median)':>17}{'% of frames':>14}")
    for i in IDX:
        v = np.array([r[f"tau_m{i}"] for r in rows if r.get(f"tau_m{i}") is not None])
        if not len(v): continue
        print(f"  {i:>6}{np.median(v):>15.1f}{np.median(F[:len(v)]/v):>17.0f}"
              f"{100*np.median(1.0/v):>13.1f}%", flush=True)

    print(f"\n=== 1. THE FRAMES-PER-rank90 TABLE, RE-REPORTED IN n_eff ===", flush=True)
    print(f"  raw column = F/rank90 (what was reported before); n_eff column uses tau averaged over")
    print(f"  EXACTLY the rank90 modes, so numerator and denominator span the same subspace.")
    print(f"  {'N bin':>17}{'n':>5}{'rank90':>8}{'raw 1rep':>10}{'raw 2rep':>10}"
          f"{'n_eff(r90)':>12}{'n_eff/rank90':>14}{'overstatement':>15}")
    n24 = np.array([r["neff24"] for r in rows]); n256 = np.array([r["neff256"] for r in rows])
    nr9 = np.array([r["neff_r90"] for r in rows])
    e = np.percentile(N, [0, 25, 50, 75, 100])
    for i in range(4):
        m = (N >= e[i]) & ((N <= e[i + 1]) if i == 3 else (N < e[i + 1]))
        if m.sum() == 0: continue
        raw = np.median(F[m] / r90[m]); eff = np.median(nr9[m] / r90[m])
        print(f"  {int(e[i]):>8}-{int(e[i+1]):<8}{int(m.sum()):>5}{np.median(r90[m]):>8.0f}"
              f"{raw:>9.1f}x{2*raw:>9.1f}x{np.median(nr9[m]):>12.0f}{eff:>13.2f}x"
              f"{raw/max(eff,1e-9):>14.0f}x", flush=True)
    print("  The raw 1->2 replica doubling is an UPPER BOUND: at the measured 1.18x between/within", flush=True)
    print("  RMSD ratio the second replica is largely redundant, so the effective gain is well below 2x.", flush=True)

    print(f"\n=== 1b. CAN THE CEILING SUBSPACE EVEN BE FITTED? n_eff per fitted dimension ===", flush=True)
    print(f"  A value < 1 means the ceiling fits MORE dimensions than it has independent samples.")
    for k in KS:
        pd = np.array([r[f"per_dim{k}"] for r in rows])
        print(f"  k={k:<4} n_eff {np.median([r[f'neff{k}'] for r in rows]):>7.0f}  per dimension "
              f"{np.median(pd):>6.2f}   below 1.0 in {(pd<1).sum()}/{len(pd)} systems", flush=True)
    pr = np.array([r["r90_pct"] for r in rows])
    print(f"  rank90 as % of usable rank: median {100*np.median(pr):.1f}%, max {100*pr.max():.1f}%"
          + ("   <-- OVER 30%: rank90 itself is near its measurement ceiling (Family B)"
             if pr.max() > 0.30 else "   (clear of the 30% censoring edge)"), flush=True)

    print(f"\n=== 2. IS IAT-vs-N THE MECHANISM BEHIND THE GUARD SLOPE? ===", flush=True)
    s_tl, h_tl = reg(x, np.log10(tl), "tau leading mode")
    reg(x, np.log10(tr_), "tau random projection", "   <- unselected reference")
    for k in KS: reg(x, np.log10([r[f"tau{k}"] for r in rows]), f"tau mean over {k} modes")
    for k, v in ((24, n24), (256, n256)): reg(x, np.log10(v), f"n_eff at k={k}")
    s_n, h_n = reg(x, np.log10(n256), "n_eff at FIXED k=256", "   <- attribution basis")
    s_r, h_r = reg(x, np.log10(r90), "rank90", "   <- for attribution")
    s_c, h_c = s_n - s_r, h_n + h_r
    s_m, h_m = reg(x, np.log10(nr9), "n_eff over rank90 modes", "   <- COMPOSITION-CONFOUNDED, see 2b")
    reg(x, np.log10(nr9 / r90), "n_eff(r90) / rank90", "   <- same confound, reported for completeness")

    print(f"\n=== 2b. ATTRIBUTION -- and why it must use FIXED k ===", flush=True)
    print(f"  n_eff averaged over rank90 modes has slope {s_m:+.4f} -- POSITIVE. That is NOT improved")
    print(f"  sampling: at large N rank90 is larger, so the average sweeps in more of the FAST")
    print(f"  high-index modes (tau ~1-4 beyond mode 200), pulling the mean tau down. It is a")
    print(f"  COMPOSITION effect of a moving window, not a statement about how well any given")
    print(f"  direction is sampled. Attribution therefore uses FIXED k, where the window is held")
    print(f"  still and the comparison is like-for-like:")
    tot = abs(s_n) + abs(s_r)
    print(f"    n_eff at k=256   {s_n:+.4f}   (IAT growing with N)           {100*abs(s_n)/max(tot,1e-9):>5.1f}%")
    print(f"    -rank90          {-s_r:+.4f}   (more modes needed at large N) {100*abs(s_r)/max(tot,1e-9):>5.1f}%")
    print(f"    conditioning     {s_c:+.4f}   at fixed k", flush=True)

    print(f"\n=== 3. VERDICT ===", flush=True)
    tau_grows = s_n + h_n < 0          # n_eff FALLING at fixed k <=> tau GROWING at fixed k
    print(f"  tau(PC1) vs N: [{s_tl-h_tl:+.4f}, {s_tl+h_tl:+.4f}] -- PC1 alone is a noisy extreme order")
    print(f"  statistic (R^2 ~0.08); the MODE-AVERAGED tau at fixed k is the well-measured version.")
    if s_c + h_c < 0:
        print(f"  Conditioning (n_eff/rank90) DEGRADES with N: [{s_c-h_c:+.4f}, {s_c+h_c:+.4f}].", flush=True)
        dom = "IAT growth" if abs(s_n) > abs(s_r) else "rank90 growth"
        print(f"  DOMINANT TERM: {dom}. IAT-vs-N is {'CONFIRMED' if s_n+h_n<0 else 'NOT confirmed'} as a")
        print(f"  mechanism, but it accounts for {100*abs(s_n)/max(abs(s_n)+abs(s_r),1e-9):.0f}% of the degradation, not all of it.")
        print("  BOTH terms are immune to the two fixes available on this corpus:")
        print("    - more frames from the same trajectories: n_eff is set by tau_int, NOT by stride, so")
        print("      denser subsampling adds no independent information and does not reduce rank90;")
        print("    - more replicas: measured between/within RMSD ratio 1.18, so a second replica is")
        print("      largely redundant -- the effective gain is far below the nominal 2x.")
        print("  Only LONGER trajectories would raise n_eff, and ATLAS does not have them.")
        print("  => THE CEILING IS IRREDUCIBLY N-BIASED ON THIS CORPUS. The oracle-fraction is a")
        print("     PERMANENTLY CAVEATED SECONDARY, not something to fix. The ceiling-free PRIMARY")
        print("     (codec vs ANM, both zero-shot, same held-out frames) is the MAIN LINE, not a")
        print("     fallback -- it has no denominator and is untouched by any of this.")
    else:
        print(f"  Conditioning does NOT degrade significantly with N [{s_c-h_c:+.4f}, {s_c+h_c:+.4f}];")
        print("  the ceiling may be usable, subject to the guard's own measured slope.")
