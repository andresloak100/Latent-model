"""INBOX 31d: WHICH OF THE THREE QUESTIONS IS THE PROJECT TESTING? This one discriminates (1).

31d asks that the next experiment NAME what it discriminates, because three rounds of arms that do
not distinguish the options is the expensive failure mode from here. The three:

  (1) the encoder cannot reach the ~54-mode target;
  (2) the objective is wrong -- MSE on displacement may not select for a good basis (17c put 65% of
      the gap in basis quality);
  (3) the comparison is unwinnable as posed -- ANM gets a SYSTEM-SPECIFIC basis and one shared model
      may not match that at any capacity.

THIS EXPERIMENT DISCRIMINATES (1) AND ONLY (1). Hold the architecture, the objective and the
comparison fixed, and vary ONLY the training-set size: n_train in {50, 130, 300}. Nothing else moves.

  FVE rises materially across the ladder -> the tied arm is DATA-limited. (1) is not the binding
      constraint, more data is the lever, and (2)/(3) are premature.
  FVE flat across a 6x corpus range  -> (1) STANDS as measured: this encoder family does not reach
      the target by being given more of the same data, and the live options become (2) and (3),
      which need different experiments.

It cannot separate (2) from (3) and does not try. 19c pre-authorised exactly this -- "run the full
ladder ONCE, on the winner" -- and the modal comparison has now settled which arm that is.

THE N-ONLY SCALE CORRECTION IS NOT APPLIED HERE. It is zero-shot and reportable, but it is a
post-hoc correction to the OUTPUT; folding it in would confound "does more data help" with "does the
calibration fix help". The raw curve is the one that answers (1).

(Derived from armf_modal_seeds.py so the training procedure is bit-identical -- forking the loop is
the procedure drift this project spent a night measuring.)

ORIGINAL HEADER FOLLOWS.
INBOX 24c: DOES THE LR SWEEP HAVE THE RESOLUTION TO RULE OUT FAMILY E? Replicate and find out.

THE OBJECTION, which is correct. I concluded "the modal arm is not losing on an unswept
hyperparameter" from a five-point LR sweep at n=1 per rate. But two of the five collapse to a
constant code (PR 1.0), so the usable sweep is THREE points, and across those three the adjacent-rate
scatter is up to 0.0433 (3e-5 +0.0996, 1e-4 +0.0563, 3e-4 +0.0974) -- NON-MONOTONIC, which is the
signature of run-to-run noise dominating the LR effect. The gap being adjudicated is 0.0350
(control +0.1346 vs best untied +0.0996).

    The sweep's own scatter is 1.24x the difference it is being used to settle, at n=1 per rate.

That is FAMILY C: an underpowered null believed. Ruling out Family E with an instrument this noisy
just substitutes one family for another, and the fix is not more argument, it is replication.

WHAT THIS MEASURES. The three usable rates x 3 seeds, for BOTH the control and the untied modal arm --
the control included deliberately, because a between-seed spread is only interpretable against the
spread of the thing it is being compared to. Reports mean +/- spread per (variant, rate), the
between-seed spread at fixed rate, and then the comparison that decides it:

    if between-seed spread at fixed rate ~ the between-RATE scatter (0.0433)
        -> the LR sweep never had the resolution. The correct statement is "the gap is not
           resolvable against training noise at n=1", which is a fine thing to say and is NOT what is
           currently written.
    if between-seed spread << between-rate scatter
        -> the non-monotonicity is a real LR effect and the original conclusion stands.

Then the gap itself is tested properly: control-vs-untied at matched rate, as a difference of means
with a CI from the seed replicates, instead of a difference of two single draws.

NOT re-opening the modal-arm question in either direction on the current numbers -- that is the point.
INBOX 23d is unaffected: it rests on three quantities moving together across the whole sweep, which is
the threshold-free form that survives this objection."""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata
import armf_atlas_dm as D
import armf_stamp as STAMP
from armf_modal_decoder import ModalCodec

WR = D.WR
RES = os.environ.get("LADDER_RES",
      "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/tied_ladder.json")
REPORT_ONLY = bool(os.environ.get("LADDER_REPORT_ONLY", ""))
DM, NTR = 256, 50
USABLE_LRS = [3e-5]
LADDER = [50, 130, 300]     # 31d: the ONLY thing that varies
SEEDS = [0, 1, 2]        # INBOX 32b: 3 per rung -- see the verdict block for why
VARIANTS = ["tied"]
BETWEEN_RATE_SCATTER = 0.0433        # measured, from the n=1 sweep
GAP_UNDER_TEST = 0.0350              # control best - untied best, from the n=1 sweep
dev = D.dev


def make(kind):
    base = D.Codec
    if kind == "control":
        return lambda Fs, L, dm, dlat=None, sub_z0=False: base(Fs, L, dm, dlat=dlat, sub_z0=sub_z0)
    return lambda Fs, L, dm, dlat=None, sub_z0=False: ModalCodec(
        Fs, L, dm, dlat=dlat, sub_z0=sub_z0, tie_encoder=False)


if __name__ == "__main__":
    print(f"[modal-seeds] INBOX 24c: is the LR sweep resolvable? {VARIANTS} x {USABLE_LRS} x "
          f"{len(SEEDS)} seeds. Between-rate scatter {BETWEEN_RATE_SCATTER:.4f}, gap under test "
          f"{GAP_UNDER_TEST:.4f}.", flush=True)
    # INBOX 36a made the verdict path reachable only by a 20 h GPU job, so its new branch could
    # not be exercised before it shipped. That is the 25a/ctx failure exactly: the tests covered
    # the kernel and nothing called the CALLER. REPORT_ONLY replays stored arms through the SAME
    # report code -- no fork, no copy -- and also re-prints the verdict from the store later
    # without a GPU. LADDER_RES points it at a fixture.
    if not REPORT_ONLY:
        man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
        have = {m["pdb"]: i for i, m in enumerate(store.meta)}
        ho_ids = [p for p in man["heldout"] if p in have]
        tr_ids = [p for p in man["train_ordered"] if p in have]
        HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
        _ord = sorted(range(len(HO)), key=lambda i: HO[i]["N"])
        HOt = [HO[_ord[i]] for i in np.linspace(0, len(HO) - 1, min(D.NHO_TRACK, len(HO))).astype(int)]
        TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:NTR]) if x is not None]
        print(f"  {len(TR)} train / {len(HOt)} tracked / {len(HO)} held-out", flush=True)

        # LADDER_MAXSTEPS raises the step cap. INERT unless set, and the default stamp stays BYTE
        # IDENTICAL so the six arms already on disk are not invalidated. When it IS set, the cap enters
        # the stamp explicitly -- D.train's AST is unchanged by a monkeypatch of maxsteps_for, so without
        # this key re-run arms would silently pool with 90k arms, which is Family F inside the resume.
        _cfg = dict(dm=DM, lrs=str(USABLE_LRS),            # NOT seeds/ladder: they select WHICH draws and
                                  # WHICH rungs, not how any arm is computed (27b's NSYS lesson)
                                  warmup=D.WARMUP, maxsteps=D.MAXSTEPS)
        _cap = os.environ.get("LADDER_MAXSTEPS", "")
        if _cap:
            _capi = int(_cap); _base = D.maxsteps_for
            D.maxsteps_for = lambda lr, _b=_base, _c=_capi: min(_c, max(_b(lr), 1))
            _cfg["cap"] = _capi
            print(f"  LADDER_MAXSTEPS={_capi}: cap raised from {_base(USABLE_LRS[0])} to "
                  f"{D.maxsteps_for(USABLE_LRS[0])}. Stamp changed, so 90k arms are NOT reused.",
                  flush=True)
        ST = STAMP.stamp(_cfg, ModalCodec, D.train)
        rows = json.load(open(RES)) if os.path.exists(RES) else []
        STAMP.report(rows, ST, "arms")
        rows = [r for r in rows if STAMP.same_stamp(r, ST)]
        done = {(r["variant"], r["lr"], r["seed"], r.get("n_train")) for r in rows}
        orig = D.Codec

        for kind in VARIANTS:
          for NTR_ in LADDER:
            TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:NTR_]) if x is not None]
            for lr in USABLE_LRS:
                for sd in SEEDS:
                    if (kind, lr, sd, NTR_) in done: continue
                    D.Codec = make(kind)
                    try:
                        torch.manual_seed(sd); np.random.seed(sd + 1)
                        t0 = time.time()
                        mdl, hist, used, stopped, improving = D.train(
                            TR, HOt, DM, lr, f"{kind} n{NTR_} lr{lr:g} s{sd}", 1)
                        mdl.eval()
                        per = [D.fve_model(mdl, x) for x in HO]
                        rec = dict(variant=kind, lr=lr, seed=sd, dm=DM, n_train=NTR_, stamp=ST,
                                   fve=float(np.mean(per)), med=float(np.median(per)),
                                   steps=used, stopped=stopped,
                                   improving=improving, best_track=max(h[1] for h in hist),
                                   secs=time.time() - t0)
                        rows.append(rec); json.dump(rows, open(RES, "w")); done.add((kind, lr, sd))
                        print(f"    {kind} n{NTR_} lr{lr:g} s{sd}: FVE {rec['fve']:+.4f}  steps {used} "
                              f"({stopped}){'  *** VOID ***' if improving else ''}  "
                              f"[{time.time()-t0:.0f}s]", flush=True)
                    except Exception as e:
                        print(f"    {kind} lr{lr:g} s{sd}: FAIL {type(e).__name__}: {e}", flush=True)
                    finally:
                        D.Codec = orig
    else:
        rows = json.load(open(RES)) if os.path.exists(RES) else []
        print(f"  [REPORT-ONLY] replaying {len(rows)} stored arms from {RES} "
              f"-- no training, no stamp filter", flush=True)

    if not rows: raise SystemExit
    # ---------------- REPORT ----------------
    # ---------------- INBOX 31d/32b LADDER VERDICT ----------------
    # 32b: 31d's yardstick was the RANGE of three draws WITHIN one rung. The quantity this has to
    # resolve is a DIFFERENCE BETWEEN rungs, which is a different statistic:
    #     SD(difference of two k-seed rungs) = SD_arm * sqrt(2/k),  95% half-width = 1.96 * that.
    # At one seed per rung and SD_arm = 0.033 that is 0.092 -- an 88% relative lift against a control
    # mean of +0.1042 -- so a flat result could not have retired option (1). That is Family C, and it
    # would retire the hypothesis the ladder exists to test.
    #
    # AND THE SD IS TAKEN FROM THIS LADDER'S OWN ARMS, not borrowed. 32a's point is that TIED's seed
    # spread was never measured; the control's varies 0.0071-0.0330 across rates, so borrowing it
    # would be a comparator measured on a different arm. Three seeds per rung measures it here.
    print(f"\n=== 31d/32b: DOES THE TIED ARM'S CEILING MOVE WITH DATA? (discriminates option 1 only) ===",
          flush=True)
    print(f"    {'n_train':>9}{'seeds':>7}{'mean FVE':>11}{'SD':>9}{'median FVE':>13}", flush=True)
    pts, sds, sd_ks = [], [], []
    rung_sd, rung_f = {}, {}
    for nt in LADDER:
        v = [r for r in rows if r.get("n_train") == nt and not r["improving"]]
        if not v: continue
        f = np.array([r["fve"] for r in v], float)
        m = np.array([r.get("med", np.nan) for r in v], float)
        sd = float(np.std(f, ddof=1)) if len(f) > 1 else float("nan")
        if np.isfinite(sd): sds.append(sd); sd_ks.append(len(f))
        rung_sd[nt] = sd; rung_f[nt] = f          # INBOX 37a: kept for the Welch/HC3 pair
        pts.append((nt, float(f.mean()), float(np.nanmean(m)), len(f)))
        print(f"    {nt:>9}{len(f):>7}{f.mean():>+11.4f}{sd:>9.4f}{np.nanmean(m):>+13.4f}", flush=True)
    # ---- INBOX 36a: IS THIS THE LADDER THAT WAS POSED? ----
    # The verdict fires at len(pts)>=2 and the primary slope at len(allr)>=4, so TWO full rungs (6
    # arms) trips both. That would print a bound measured over 50->130 = 2.6x for a question posed
    # over 50->300 = 6.0x, and the truncation is at the END THAT SETS THE LEVER ARM -- the same shape
    # as 18d, where a median-based coverage check passed a sample missing the top 18% of log-range.
    # A reader who sees "=>" does not re-derive the span, so the span has to travel ON the verdict.
    # Rungs missing => PARTIAL: the lever arm itself is short, and the monitor must NOT stop on it.
    # Seeds short within a present rung is a different defect -- it costs precision, not lever arm,
    # and 34a/34b's unequal-k pooling already prices it -- so it is noted but is NOT non-terminal.
    # Making a VOID arm non-terminal would leave the watch running forever after the chain ended.
    rungs_have = [p[0] for p in pts]
    missing = [nt for nt in LADDER if nt not in rungs_have]
    span_have = (max(rungs_have) / min(rungs_have)) if rungs_have else float("nan")
    span_reg = max(LADDER) / min(LADDER)
    PARTIAL = f" [PARTIAL LADDER -- {len(rungs_have)} of {len(LADDER)} rungs, span {span_have:.1f}x, " \
              f"not the pre-registered {span_reg:.1f}x; missing n{','.join(str(m) for m in missing)}]" \
              if missing else ""
    if PARTIAL:
        print(f"\n  !! {PARTIAL.strip(' []')}", flush=True)
        print(f"     This is an EARLY READ, not the fork. Any bound below is measured over "
              f"{span_have:.1f}x and cannot be quoted as the answer to a {span_reg:.1f}x question.",
              flush=True)
    # ---- STEP BUDGET BY RUNG. Family D and Family A, both live, neither previously printed. ----
    # maxsteps_for(3e-5) = min(90000, 30000*sqrt(1e-3/3e-5)) = min(90000, 173205) = 90000: the CAP
    # binds, giving 48% less than this project's own sqrt scaling prescribes, and the ladder runs at
    # 3e-5 exclusively -- the rate armf_atlas_dm.py:100 already records as "hit MAXSTEPS still
    # improving and are flagged VOID ... a Family D hole opened by the Family E repair".
    #
    # The budget is CONSTANT across rungs, so compute is matched. What is not matched is how far that
    # budget gets each rung from convergence, and that distance grows with n_train -- the regressor of
    # the primary test. Two consequences, and neither was visible in the output:
    #   FAMILY D  truncation understates FVE at high n, biasing the slope TOWARD ZERO. A bounded null
    #             would then be partly a measurement of the step budget rather than of the data --
    #             which is this file's own doctrine: "undertrained at a fixed budget ... manufactures
    #             a flat curve" (armf_atlas_dm.py:33).
    #   FAMILY A  VOID is "still improving at maxsteps", so the EXCLUSION RATE rises with n_train and
    #             the surviving high-n arms are the fast-converging subset, not a random one.
    CEIL = D.maxsteps_for(USABLE_LRS[0])
    print(f"\n  --- STEP BUDGET BY RUNG (ceiling {CEIL}, constant across rungs) ---", flush=True)
    cfrac = {}
    for nt in LADDER:
        v = [r for r in rows if r.get("n_train") == nt]
        if not v: continue
        at = sum(1 for r in v if r["steps"] >= CEIL); vd = sum(1 for r in v if r["improving"])
        cfrac[nt] = at / len(v)
        print(f"    n{nt:<5} {len(v)} arms, mean {np.mean([r['steps'] for r in v]):.0f} steps, "
              f"{at}/{len(v)} at the ceiling, {vd} VOID (excluded)", flush=True)
    CEILING = ""
    if len(cfrac) > 1:
        lo_n, hi_n = min(cfrac), max(cfrac)
        if cfrac[hi_n] > cfrac[lo_n]:
            CEILING = (f" [CEILING BINDS HARDER AT HIGH n: {cfrac[lo_n]:.2f} at n{lo_n} -> "
                       f"{cfrac[hi_n]:.2f} at n{hi_n}]")
            print(f"    !! {CEILING.strip(' []')}", flush=True)
            print(f"       Family D: high-n arms are truncated further from convergence, which "
                  f"understates their FVE and", flush=True)
            print(f"       biases the slope TOWARD ZERO -- so a bounded null here partly measures the "
                  f"step budget, not the data.", flush=True)
            print(f"       Family A: VOID means 'still improving at the ceiling', so the exclusion "
                  f"rate rises with n_train and", flush=True)
            print(f"       the surviving high-n arms are the fast-converging subset. Raising the cap "
                  f"MID-LADDER would not fix", flush=True)
            print(f"       either -- it would make the rungs incomparable on budget, so the budget "
                  f"stays fixed and this prints.", flush=True)
    short = [(nt, k_) for (nt, _, _, k_) in pts if k_ < len(SEEDS)]
    if short:
        print(f"     (also short of seeds, which costs PRECISION not lever arm and is already priced "
              f"by the df below: {', '.join(f'n{nt}={k_}/{len(SEEDS)}' for nt, k_ in short)})",
              flush=True)
    if len(pts) >= 2:
        n0, f0, m0, k0 = pts[0]; n1, f1, m1, k1 = pts[-1]
        d_mean = f1 - f0
        # INBOX 34a/34b: pooling and the standard error must both handle UNEQUAL seed counts.
        #   34a  RMS pooling ignores k_i entirely and is only correct when the rungs are equal. The
        #        weighted form is s_p^2 = sum((k_i-1) s_i^2) / sum(k_i-1) -- the same denominator as
        #        the df already computed, which is what made the inconsistency visible.
        #   34b  SD_arm*sqrt(2/k) carries the same assumption; the general form is
        #        s_p*sqrt(1/k_i + 1/k_j).
        # Both are no-ops when the rungs come back full at (3,3,3), which is the expected case.
        var_num = sum((k_ - 1) * sd_ ** 2 for sd_, k_ in zip(sds, sd_ks) if k_ > 1)
        dfree = int(sum(max(k_ - 1, 0) for (_, _, _, k_) in pts))
        sd_arm = float(np.sqrt(var_num / dfree)) if dfree > 0 else float("nan")
        if np.isfinite(sd_arm) and dfree >= 1 and k0 >= 1 and k1 >= 1:
            tq = float(stats.t.ppf(0.975, dfree))
            se = sd_arm * np.sqrt(1.0 / k0 + 1.0 / k1)      # 34b: general, not sqrt(2/k)
            hw = tq * se
        else:
            tq = se = hw = float("nan")
        # INBOX 36b: a constant from ONE arm at ONE rung, living in a script that reports across
        # three. Fine as a display denominator; it would be Family F the moment it entered a verdict
        # CONDITION, so its scope prints wherever it does.
        # ---- INBOX 37a: EQUAL VARIANCE ACROSS RUNGS is what pooling and OLS both assume ----
        # It was untested and silent. Printing the per-rung SDs and their max/min ratio makes it
        # visible -- the same move as printing df.
        #
        # 37a asked for a switch to Welch/HC3 above ~4x. I am recording a STRICTER rule instead, and
        # the reason is a number: at k=3 the ratio barely carries information. Simulating 400k draws
        # of three rungs under TRUE equal variance, P(max/min SD ratio > 4) = 0.24 and the MEDIAN
        # ratio is 2.52. A 4x switch therefore fires a quarter of the time when the assumption holds
        # perfectly, so which estimator reports the verdict would be decided by noise -- which is 28e
        # arriving through the mechanism built to prevent it.
        #
        # PRE-REGISTERED, FIXED BEFORE n300 EXISTS: the ROBUST pair -- Welch for the pairwise, HC3
        # for the slope -- is authoritative UNCONDITIONALLY. Pooled/OLS prints beside it as a
        # sensitivity check, never as an alternative verdict. This removes the estimator choice from
        # the data entirely rather than making it a coin flip. It is 34a's standard again: Welch and
        # HC3 are valid under BOTH regimes and collapse to the pooled answer when the spreads match,
        # so the correction is a no-op in the common case and correct when the unusual one arrives.
        #
        # The price is MEASURED on this exact design (3 rungs x 3 seeds, x = log10(50,130,300)),
        # 200k simulations per row, not asserted:
        #     true variances      OLS coverage   HC3 coverage   HC3/OLS width
        #     equal                  0.952          0.960           1.16
        #     unequal (extreme)      0.868          0.947           1.47
        #     n130 tight only        0.970          0.937           0.96
        # HC3 costs 16% width when the assumption holds. When it does not, OLS's nominal 95%
        # interval is really 87% -- an under-covering interval on the PRIMARY test is exactly what
        # manufactures a false "THE CEILING MOVES WITH DATA". Honest caveat: in the one-tight-rung
        # pattern the early data hints at, HC3 itself under-covers slightly (0.937). Neither is
        # exact at n=9; HC3 is closer to nominal in both non-equal cases, which is why it reports.
        fs = {k_: v_ for k_, v_ in rung_sd.items() if np.isfinite(v_) and v_ > 0}
        sd_ratio = (max(fs.values()) / min(fs.values())) if len(fs) > 1 else float("nan")
        print(f"\n  --- 37a: EQUAL VARIANCE ACROSS RUNGS (the assumption behind pooling and OLS) ---",
              flush=True)
        print(f"    per-rung SD: " + "   ".join(
              f"n{nt}={rung_sd.get(nt, float('nan')):.4f} (k={k_})" for (nt, _, _, k_) in pts),
              flush=True)
        print(f"    max/min ratio {sd_ratio:.2f}. The ROBUST pair below is authoritative regardless "
              f"of this number:", flush=True)
        print(f"    at k=3, under TRUE equal variance, P(ratio>4)=0.24 and the median ratio is 2.52, "
              f"so a", flush=True)
        print(f"    threshold switch would pick the estimator by noise. Pooled/OLS is a SENSITIVITY "
              f"CHECK, not a verdict.", flush=True)

        CTRL_MEAN = 0.1042; CTRL_PROV = "control lr3e-4, n50, 3 seeds, 24c"

        # ---- INBOX 34c: THE PRIMARY TEST IS FIXED HERE, BEFORE THE NUMBERS ----
        # PRIMARY  = slope of FVE on log10(n_train) over ALL arms (df = n_arms - 2).
        # SECONDARY= pairwise n50 vs n300, reported as descriptive.
        # Chosen because the question is a TREND across the ladder, the slope uses all nine points
        # rather than six, and it has more df. Picking whichever of the two reads better once both
        # exist is the post-hoc statistic choice 28e caught, so it is recorded now.
        allr = [r for r in rows if not r["improving"] and r.get("n_train") in LADDER]
        print(f"\n  --- PRIMARY (fixed before results, INBOX 34c): slope of FVE on log10(n_train) ---",
              flush=True)
        if len(allr) >= 4:
            x = np.log10([r["n_train"] for r in allr]); y = np.array([r["fve"] for r in allr])
            sl = stats.linregress(x, y); dfs = len(x) - 2
            hs_ols = float(stats.t.ppf(0.975, dfs)) * sl.stderr
            span = np.log10(max(x_ for x_ in 10 ** x) / min(x_ for x_ in 10 ** x))
            # HC3 sandwich for a simple regression. e_i/(1-h_i) inflates each residual by its own
            # leverage, which is the small-sample variant; at n=9 it is deliberately conservative.
            xb = x.mean(); Sxx = float(((x - xb) ** 2).sum())
            e = y - (sl.intercept + sl.slope * x)
            h = 1.0 / len(x) + (x - xb) ** 2 / Sxx
            se_hc3 = float(np.sqrt((((x - xb) ** 2) * (e / (1.0 - h)) ** 2).sum() / Sxx ** 2))
            hs = float(stats.t.ppf(0.975, dfs)) * se_hc3          # AUTHORITATIVE (37a)
            print(f"    slope {sl.slope:+.4f} +/- {hs:.4f}  (HC3, t df={dfs}, n={len(x)} arms)  "
                  f"R^2 {sl.rvalue**2:.3f}   <- AUTHORITATIVE", flush=True)
            print(f"    slope {sl.slope:+.4f} +/- {hs_ols:.4f}  (OLS, assumes equal variance)"
                  f"          [sensitivity only]", flush=True)
            print(f"    implied change over the measured {10**span:.1f}x range: "
                  f"{sl.slope*span:+.4f} +/- {hs*span:.4f}", flush=True)
            moves = abs(sl.slope) > hs
        else:
            moves = False; sl = None
            print(f"    only {len(allr)} arms -- primary test not yet computable", flush=True)

        print(f"\n  --- SECONDARY (descriptive): pairwise n{n0} vs n{n1} ---", flush=True)
        print(f"    difference {d_mean:+.4f};  pooled SD_arm {sd_arm:.4f} (df={dfree}, "
              f"k={k0}/{k1});  SE {se:.4f};  t {tq:.3f};  95% half-width {hw:.4f}"
              f"   [sensitivity only]", flush=True)
        s0, s1 = rung_sd.get(n0, float("nan")), rung_sd.get(n1, float("nan"))
        if np.isfinite(s0) and np.isfinite(s1) and k0 > 1 and k1 > 1 and (s0 > 0 or s1 > 0):
            se_w = float(np.sqrt(s0 ** 2 / k0 + s1 ** 2 / k1))
            df_w = ((s0 ** 2 / k0 + s1 ** 2 / k1) ** 2 /
                    ((s0 ** 2 / k0) ** 2 / (k0 - 1) + (s1 ** 2 / k1) ** 2 / (k1 - 1)))
            hw_w = float(stats.t.ppf(0.975, max(df_w, 1e-9))) * se_w
            print(f"    WELCH (no equal-variance assumption): SE {se_w:.4f};  df {df_w:.2f};  "
                  f"95% half-width {hw_w:.4f}   <- AUTHORITATIVE", flush=True)
        else:
            print(f"    WELCH: not computable (a rung has k<2 or zero spread)", flush=True)

        if moves and sl is not None:
            print(f"\n  => THE CEILING MOVES WITH DATA.{PARTIAL}{CEILING} Slope {sl.slope:+.4f} "
                  f"+/- {hs:.4f} excludes")
            print(f"     zero, so the tied arm is DATA-limited over this range: 31d-(1) is NOT the")
            print(f"     binding constraint and (2)/(3) are premature.")
            print(f"     INBOX 32c CONSEQUENCE, pre-registered: 14a, 17c, 25a and the ENTIRE peer")
            print(f"     comparison were measured at n50, so a rise means each was measured on an")
            print(f"     UNDER-TRAINED model and is a FLOOR, not an estimate -- including FVE_perp ~ 0")
            print(f"     and the 0%-of-systems peer loss. It does not soften the n50 peer result; it")
            print(f"     means the headline carries 'at n_train=50' until the peer comparison is")
            print(f"     re-run at the best rung.", flush=True)
        elif sl is not None:
            bound = abs(hs * span)
            print(f"\n  => Across a {10**span:.1f}x range in n_train, NO EFFECT LARGER THAN "
                  f"{bound:.4f}{PARTIAL}{CEILING}")
            print(f"     (t-interval on the slope, df={dfs}) -- which is {100*bound/CTRL_MEAN:.0f}% of")
            print(f"     the control's mean ({CTRL_MEAN:+.4f}; {CTRL_PROV}).")
            print(f"     OPTION (1) IS NOT RETIRED; effects below that size are not excluded here.")
            print(f"     Reporting this as 'not data-limited' would be Family C -- retiring the")
            print(f"     hypothesis the ladder exists to test.", flush=True)

    print(f"\n  SCOPE: architecture, objective and comparator held FIXED; only n_train varies, so this")
    print(f"  discriminates (1) alone and cannot separate (2) from (3). The N-only scale correction is")
    print(f"  deliberately NOT applied -- folding it in would confound 'does more data help' with")
    print(f"  'does the calibration fix help'.", flush=True)

    print(f"\n=== 24c: MEAN +/- SPREAD PER (VARIANT, RATE) ===", flush=True)
    print(f"    {'variant':>9}{'lr':>9}{'n':>4}{'mean FVE':>11}{'spread':>10}{'sd':>9}", flush=True)
    cell = {}
    for kind in VARIANTS:
        for lr in USABLE_LRS:
            v = np.array([r["fve"] for r in rows
                          if r["variant"] == kind and r["lr"] == lr and not r["improving"]], float)
            if not len(v): continue
            cell[(kind, lr)] = v
            print(f"    {kind:>9}{lr:>9.0e}{len(v):>4}{v.mean():>11.4f}"
                  f"{(v.max()-v.min()):>10.4f}{(v.std(ddof=1) if len(v) > 1 else float('nan')):>9.4f}",
                  flush=True)

    print(f"\n=== IS THE SWEEP RESOLVABLE? ===", flush=True)
    seed_spreads = [float(v.max() - v.min()) for v in cell.values() if len(v) > 1]
    if seed_spreads:
        ms = float(np.median(seed_spreads))
        print(f"  between-SEED spread at fixed rate: median {ms:.4f}  "
              f"(range {min(seed_spreads):.4f}-{max(seed_spreads):.4f}, {len(seed_spreads)} cells)")
        print(f"  between-RATE scatter from the n=1 sweep: {BETWEEN_RATE_SCATTER:.4f}")
        print(f"  gap being adjudicated (control - untied):  {GAP_UNDER_TEST:.4f}")
        ratio = ms / max(BETWEEN_RATE_SCATTER, 1e-9)
        if ratio > 0.5:
            print(f"  => SEED SPREAD IS {ratio:.2f}x THE BETWEEN-RATE SCATTER. The LR sweep NEVER HAD")
            print(f"     THE RESOLUTION, and the non-monotonicity was noise, not an LR effect. The")
            print(f"     correct statement is: THE GAP IS NOT RESOLVABLE AGAINST TRAINING NOISE AT")
            print(f"     n=1. 'The modal arm is not losing on an unswept hyperparameter' is WITHDRAWN")
            print(f"     as unsupported -- not refuted, unsupported.", flush=True)
        else:
            print(f"  => seed spread is {ratio:.2f}x the between-rate scatter, so the rate effect is")
            print(f"     real and the original conclusion stands.", flush=True)

    print(f"\n=== THE GAP, TESTED PROPERLY (difference of means, not of two single draws) ===",
          flush=True)
    for lr in USABLE_LRS:
        a, b = cell.get(("control", lr)), cell.get(("untied", lr))
        if a is None or b is None or len(a) < 2 or len(b) < 2: continue
        # The p-value is WELCH (equal_var=False) but the half-width used t.ppf(.975, na+nb-2) --
        # the POOLED df. Welch SE with a Student quantile is two different distributions on the same
        # line, and it made the interval TOO NARROW: at lr1e-4 it printed +0.0381 +/- 0.0364, a CI
        # excluding zero, beside p=0.080. A 95% CI that excludes zero cannot sit next to p>0.05.
        # Welch-Satterthwaite df there is 2.44, not 4; the correct half-width is 0.0478 and the CI
        # [-0.0097, +0.0858] contains zero. "control ahead" was an artifact of the mismatch, and it
        # was the ONLY non-null cell in the table. This is 33a one level down: right statistic, wrong
        # distribution for it.
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = a.mean() - b.mean()
        va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
        se = np.sqrt(va / na + vb / nb)
        dfw = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        hw = stats.t.ppf(0.975, max(dfw, 1e-9)) * se
        print(f"    lr{lr:g}: control {a.mean():+.4f} - untied {b.mean():+.4f} = {d:+.4f} "
              f"+/- {hw:.4f}  (Welch df={dfw:.2f}, p={p:.3f})", flush=True)
        # And "INDISTINGUISHABLE" is an underpowered null unless it is BOUNDED. The number that makes
        # it mean something is GAP_UNDER_TEST -- the very gap this test exists to adjudicate.
        STAMP.null_report(d, hw, GAP_UNDER_TEST, f"      control - untied @ lr{lr:g}")
    print(f"\n  Whatever this says, INBOX 23d is UNAFFECTED: it rests on three quantities moving")
    print(f"  together across the whole sweep (PR up, identity down, FVE down), which is the")
    print(f"  threshold-free form that survives an objection about resolution at any single rate.",
          flush=True)
