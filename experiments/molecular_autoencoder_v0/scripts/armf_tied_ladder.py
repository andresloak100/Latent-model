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
    #   FAMILY D  truncation -- high-n arms are cut off before convergence, so their FVE is
    #             understated. Pushes the slope DOWN. Affects 3/3 arms at n130.
    #   FAMILY A  VOID exclusion -- VOID *means* still improving, so the SLOW movers are dropped and
    #             the plateaued subset survives, leaving the high-n mean above a random draw's.
    #             Pushes the slope UP. Affects 1/3 arms at n130.
    # INBOX 38a: these OPPOSE, and an earlier version of this comment claimed the net was "toward
    # zero", which is stronger than the argument supports. Truncation touches three times as many
    # arms and probably dominates, so the net is MOST LIKELY downward -- but that is not established,
    # and a one-directional claim is what a reader leans on. Both directions therefore print.
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
            # Report EVERY rung's fraction, not just the endpoints. The observed pattern is
            # 0.00 / 1.00 / 0.50 at n50 / n130 / n300 -- the MIDDLE rung binds hardest, and an
            # endpoint-only tag reading "binds harder at high n" would assert a monotonicity the data
            # does not have. My own extrapolation from n130 (that n300 would be worse) was wrong:
            # n300 s1 plateaued at 80000, below the cap.
            CEILING = (" [CEILING BINDS UNEVENLY, fraction at cap: "
                       + " ".join(f"n{k_}={cfrac[k_]:.2f}" for k_ in sorted(cfrac)) + "]")
            print(f"    !! {CEILING.strip(' []')}  (not monotonic in n -- see per-rung table above)",
                  flush=True)
            print(f"       Family D (slope DOWN, 3/3 arms at n130): high-n arms are cut off before "
                  f"convergence, understating FVE.", flush=True)
            print(f"       Family A (slope UP,   1/3 arms at n130): VOID *means* still improving, so "
                  f"the SLOW movers are dropped", flush=True)
            print(f"         and the plateaued subset survives, leaving the high-n mean above a "
                  f"random draw's.", flush=True)
            print(f"       THESE OPPOSE. Truncation touches more arms, so the net is most likely "
                  f"DOWNWARD (INBOX 38a).", flush=True)
            # 38a said neither direction was quantified. Family A now IS: the VOID arms have measured
            # FVEs, so how far the exclusion moved each rung mean is arithmetic, not conjecture.
            # Family D stays unquantified -- nobody knows where a truncated arm would have converged --
            # but each VOID arm's FVE is a LOWER BOUND on it, which bounds the suppression from below.
            for nt in LADDER:
                v = [r for r in rows if r.get("n_train") == nt]
                ok = [r["fve"] for r in v if not r["improving"]]
                vd = [r["fve"] for r in v if r["improving"]]
                if ok and vd:
                    shift = float(np.mean(ok)) - float(np.mean(ok + vd))
                    print(f"         FAMILY A, MEASURED at n{nt}: excluding {len(vd)} VOID arm(s) moved "
                          f"the rung mean by {shift:+.4f}", flush=True)
                    print(f"           (usable {np.mean(ok):+.4f} vs all-arms {np.mean(ok+vd):+.4f}; "
                          f"VOID FVEs are themselves lower bounds)", flush=True)
            print(f"       Raising the cap MID-LADDER would not fix either: it would make the rungs "
                  f"incomparable on compute,", flush=True)
            print(f"         so the budget stays fixed and this prints instead.", flush=True)
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
        print(f"    max/min ratio {sd_ratio:.2f}. INBOX 39a: read this as evidence about WHICH RUNGS "
              f"HIT THE CAP, not", flush=True)
        print(f"    about the data. A rung whose arms all stop at the SAME step loses the "
              f"stopping-point component of", flush=True)
        print(f"    variance, so its SD is deflated MECHANICALLY -- which narrows its interval and "
              f"makes a rise EASIER to", flush=True)
        print(f"    declare. That is a third ceiling effect, and it opposes the other two. The ROBUST "
              f"pair is authoritative", flush=True)
        print(f"    regardless of this number:", flush=True)
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
            # INBOX 39a/39b APPLIED TO THE PRIMARY, which needs it more than the secondary does.
            # x is CONSTANT WITHIN A RUNG, so the slope depends only on the rung MEANS: inflating
            # within-rung deviations leaves the point estimate EXACTLY unchanged and moves only the
            # SE. Same number, different bar -- which is what a sensitivity should be.
            _nts = np.array([r["n_train"] for r in allr])
            _free = [rung_sd[k_] for k_ in rung_sd
                     if cfrac.get(k_, 0) == 0 and np.isfinite(rung_sd.get(k_, float("nan")))]
            if _free:
                _sref = max(_free); _e2 = e.copy(); _touched = []
                for k_ in sorted(set(_nts.tolist())):
                    m_ = _nts == k_; sdk = rung_sd.get(k_, float("nan"))
                    if cfrac.get(k_, 0) > 0 and np.isfinite(sdk) and 0 < sdk < _sref and m_.sum() > 1:
                        dev = y[m_] - y[m_].mean()
                        _e2[m_] = dev * (_sref / sdk) + (e[m_] - dev)   # rung-level residual kept
                        _touched.append(f"n{k_} {sdk:.4f}->{_sref:.4f}")
                if _touched:
                    se_p = float(np.sqrt((((x - xb) ** 2) * (_e2 / (1.0 - h)) ** 2).sum() / Sxx ** 2))
                    hs_p = float(stats.t.ppf(0.975, dfs)) * se_p
                    print(f"    39b PESSIMISTIC (ceiling-deflated SDs restored: {', '.join(_touched)}): "
                          f"slope {sl.slope:+.4f} +/- {hs_p:.4f}", flush=True)
                    print(f"       -> {'STILL excludes zero' if abs(sl.slope) > hs_p else 'NO LONGER excludes zero'}"
                          f". Point estimate is unchanged by construction; only the bar moved.",
                          flush=True)
                    if moves and abs(sl.slope) <= hs_p:
                        print(f"       THE PRIMARY VERDICT DOES NOT SURVIVE THE PESSIMISTIC SD. The "
                              f"rise below rests on rungs whose", flush=True)
                        print(f"       spread was compressed by the cap, so it is NOT safe to read as "
                              f"a rise (INBOX 39a).", flush=True)
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
        # INBOX 39b: assume the deflation is ENTIRELY mechanical -- substitute the widest
        # non-ceiling-bound rung's SD into every ceiling-bound rung and re-run. Pessimistic by
        # construction; a no-op when no rung is ceiling-bound.
        free = [rung_sd[k_] for k_ in rung_sd
                if cfrac.get(k_, 0) == 0 and np.isfinite(rung_sd.get(k_, float("nan")))]
        if free and np.isfinite(s0) and np.isfinite(s1) and k0 > 1 and k1 > 1:
            sref = max(free)
            p0 = sref if cfrac.get(n0, 0) > 0 and s0 < sref else s0
            p1 = sref if cfrac.get(n1, 0) > 0 and s1 < sref else s1
            if (p0, p1) != (s0, s1):
                vnp = (k0 - 1) * p0 ** 2 + (k1 - 1) * p1 ** 2; dfp = (k0 - 1) + (k1 - 1)
                spp = float(np.sqrt(vnp / dfp))
                hwp = float(stats.t.ppf(0.975, dfp)) * spp * np.sqrt(1.0 / k0 + 1.0 / k1)
                ap_, bp_ = p0 ** 2 / k0, p1 ** 2 / k1
                sew = float(np.sqrt(ap_ + bp_))
                dfw2 = (ap_ + bp_) ** 2 / (ap_ ** 2 / (k0 - 1) + bp_ ** 2 / (k1 - 1))
                hwpw = float(stats.t.ppf(0.975, max(dfw2, 1e-9))) * sew
                print(f"    39b PESSIMISTIC (ceiling-bound rung SDs := {sref:.4f}, the widest "
                      f"cap-free rung):", flush=True)
                print(f"       pooled half-width {hwp:.4f} -> excludes 0: "
                      f"{'YES' if abs(d_mean) > hwp else 'NO'}   [sensitivity only]", flush=True)
                print(f"       WELCH  half-width {hwpw:.4f} (df {dfw2:.2f}) -> excludes 0: "
                      f"{'YES' if abs(d_mean) > hwpw else 'NO'}   <- the authoritative form",
                      flush=True)
                if abs(d_mean) > hwp and abs(d_mean) <= hwpw:
                    print(f"       NOTE: the pairwise survives the pessimistic SD under POOLED but "
                          f"NOT under Welch. Per 34c the", flush=True)
                    print(f"       PRIMARY test is the slope, not this pairwise, so this does not "
                          f"decide the fork -- but it is the", flush=True)
                    print(f"       reason the pooled row is labelled sensitivity only.", flush=True)

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
            if CEILING:
                print(f"\n     INBOX 38b -- THIS BRANCH IS FULLY INTERPRETABLE. The instrument is biased "
                      f"AGAINST finding a rise", flush=True)
                _worst = max(cfrac, key=lambda k_: cfrac[k_])
                print(f"     (truncation dominates -- n{_worst} runs {cfrac[_worst]:.0%} at the cap), "
                      f"so a rise measured THROUGH the", flush=True)
                print(f"     confound is CONSERVATIVE: the true", flush=True)
                print(f"     effect is at least this large. 38c does not apply and no re-run is "
                      f"triggered.", flush=True)

        elif sl is not None:
            bound = abs(hs * span)
            print(f"\n  => Across a {10**span:.1f}x range in n_train, NO EFFECT LARGER THAN "
                  f"{bound:.4f}{PARTIAL}{CEILING}")
            print(f"     (t-interval on the slope, df={dfs}) -- which is {100*bound/CTRL_MEAN:.0f}% of")
            print(f"     the control's mean ({CTRL_MEAN:+.4f}; {CTRL_PROV}).")
            print(f"     OPTION (1) IS NOT RETIRED; effects below that size are not excluded here.")
            print(f"     Reporting this as 'not data-limited' would be Family C -- retiring the")
            print(f"     hypothesis the ladder exists to test.", flush=True)
            if CEILING:
                print(f"\n     INBOX 38b -- THIS BRANCH IS NOT INTERPRETABLE AS A NULL. Flat is "
                      f"indistinguishable from the step", flush=True)
                print(f"     budget: a bound here is partly a statement about 90000 steps, not about "
                      f"data, so it CANNOT retire", flush=True)
                print(f"     option (1) -- which is exactly what 32b and 33c established the ladder "
                      f"must be able to do.", flush=True)
                print(f"     INBOX 38c, COMMITTED BEFORE THE NUMBER EXISTED: a flat result TRIGGERS A "
                      f"RE-RUN at the prescribed", flush=True)
                print(f"     173205 steps (LADDER_MAXSTEPS=173205, ~2x compute over 9 arms) before "
                      f"option (1) may be retired.", flush=True)
                print(f"     Chosen over 'report as confounded and leave it open' because THAT is an "
                      f"underpowered null wearing a", flush=True)
                print(f"     result's clothes -- the thing 35a and null_verdict exist to refuse -- and "
                      f"because 32c makes this fork", flush=True)
                print(f"     decide whether the n50 peer loss is a finding or an artefact of "
                      f"under-training.", flush=True)

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
