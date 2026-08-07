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
RES = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/tied_ladder.json"
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
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]
    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    _ord = sorted(range(len(HO)), key=lambda i: HO[i]["N"])
    HOt = [HO[_ord[i]] for i in np.linspace(0, len(HO) - 1, min(D.NHO_TRACK, len(HO))).astype(int)]
    TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:NTR]) if x is not None]
    print(f"  {len(TR)} train / {len(HOt)} tracked / {len(HO)} held-out", flush=True)

    ST = STAMP.stamp(dict(dm=DM, lrs=str(USABLE_LRS),   # NOT seeds/ladder: they select WHICH draws and
                          # WHICH rungs, not how any arm is computed (27b's NSYS lesson)
                          warmup=D.WARMUP, maxsteps=D.MAXSTEPS), ModalCodec, D.train)
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
    for nt in LADDER:
        v = [r for r in rows if r.get("n_train") == nt and not r["improving"]]
        if not v: continue
        f = np.array([r["fve"] for r in v], float)
        m = np.array([r.get("med", np.nan) for r in v], float)
        sd = float(np.std(f, ddof=1)) if len(f) > 1 else float("nan")
        if np.isfinite(sd): sds.append(sd); sd_ks.append(len(f))
        pts.append((nt, float(f.mean()), float(np.nanmean(m)), len(f)))
        print(f"    {nt:>9}{len(f):>7}{f.mean():>+11.4f}{sd:>9.4f}{np.nanmean(m):>+13.4f}", flush=True)
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
        CTRL_MEAN = 0.1042      # control lr3e-4 mean over 3 seeds (24c)

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
            hs = float(stats.t.ppf(0.975, dfs)) * sl.stderr
            span = np.log10(max(x_ for x_ in 10 ** x) / min(x_ for x_ in 10 ** x))
            print(f"    slope {sl.slope:+.4f} +/- {hs:.4f}  (t, df={dfs}, n={len(x)} arms)  "
                  f"R^2 {sl.rvalue**2:.3f}", flush=True)
            print(f"    implied change over the measured {10**span:.1f}x range: "
                  f"{sl.slope*span:+.4f} +/- {hs*span:.4f}", flush=True)
            moves = abs(sl.slope) > hs
        else:
            moves = False; sl = None
            print(f"    only {len(allr)} arms -- primary test not yet computable", flush=True)

        print(f"\n  --- SECONDARY (descriptive): pairwise n{n0} vs n{n1} ---", flush=True)
        print(f"    difference {d_mean:+.4f};  pooled SD_arm {sd_arm:.4f} (df={dfree}, "
              f"k={k0}/{k1});  SE {se:.4f};  t {tq:.3f};  95% half-width {hw:.4f}", flush=True)

        if moves and sl is not None:
            print(f"\n  => THE CEILING MOVES WITH DATA. Slope {sl.slope:+.4f} +/- {hs:.4f} excludes")
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
            print(f"\n  => Across a {10**span:.0f}x range in n_train, NO EFFECT LARGER THAN "
                  f"{bound:.4f}")
            print(f"     (t-interval on the slope, df={dfs}) -- which is {100*bound/CTRL_MEAN:.0f}% of")
            print(f"     the control's mean ({CTRL_MEAN:+.4f}). OPTION (1) IS NOT RETIRED; effects")
            print(f"     below that size are not excluded by this design.")
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
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        hw = stats.t.ppf(0.975, max(len(a) + len(b) - 2, 1)) * se
        print(f"    lr{lr:g}: control {a.mean():+.4f} - untied {b.mean():+.4f} = {d:+.4f} "
              f"+/- {hw:.4f}  (p={p:.3f})  -> "
              f"{'control ahead' if d - hw > 0 else ('untied ahead' if d + hw < 0 else 'INDISTINGUISHABLE')}",
              flush=True)
    print(f"\n  Whatever this says, INBOX 23d is UNAFFECTED: it rests on three quantities moving")
    print(f"  together across the whole sweep (PR up, identity down, FVE down), which is the")
    print(f"  threshold-free form that survives an objection about resolution at any single rate.",
          flush=True)
