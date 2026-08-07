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
SEEDS = [0]
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

    ST = STAMP.stamp(dict(dm=DM, ntr=NTR, lrs=str(USABLE_LRS), seeds=str(SEEDS),
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
    # ---------------- INBOX 31d LADDER VERDICT ----------------
    print(f"\n=== 31d: DOES THE TIED ARM'S CEILING MOVE WITH DATA? (discriminates option 1 only) ===",
          flush=True)
    print(f"    {'n_train':>9}{'n arms':>8}{'mean FVE':>11}{'median FVE':>13}{'steps':>9}", flush=True)
    pts = []
    for nt in LADDER:
        v = [r for r in rows if r.get("n_train") == nt and not r["improving"]]
        if not v: continue
        f = float(np.mean([r["fve"] for r in v])); m = float(np.mean([r.get("med", np.nan) for r in v]))
        pts.append((nt, f, m))
        print(f"    {nt:>9}{len(v):>8}{f:>+11.4f}{m:>+13.4f}{int(np.mean([r['steps'] for r in v])):>9}",
              flush=True)
    if len(pts) >= 2:
        n0, f0, m0 = pts[0]; n1, f1, m1 = pts[-1]
        d_mean, d_med = f1 - f0, m1 - m0
        # the yardstick is the SEED spread measured under 24c at this architecture's operating point,
        # not a threshold invented here: a ladder move smaller than run-to-run noise is not a move.
        SEED_SPREAD = 0.0656
        print(f"\n  n_train {n0} -> {n1} ({n1/n0:.1f}x corpus):  mean {f0:+.4f} -> {f1:+.4f} "
              f"({d_mean:+.4f})   median {m0:+.4f} -> {m1:+.4f} ({d_med:+.4f})", flush=True)
        print(f"  yardstick: the largest between-SEED spread measured at fixed settings (24c) is "
              f"{SEED_SPREAD:.4f}.", flush=True)
        if max(abs(d_mean), abs(d_med)) > SEED_SPREAD:
            print(f"  => THE CEILING MOVES WITH DATA. The tied arm is DATA-limited over this range, so")
            print(f"     31d-(1) is NOT the binding constraint and more data is a real lever. (2) and")
            print(f"     (3) are premature until the curve flattens.", flush=True)
        else:
            print(f"  => FLAT ACROSS A {n1/n0:.0f}x CORPUS RANGE, within the seed spread. 31d-(1) STANDS")
            print(f"     AS MEASURED: this encoder family does not reach the target by being given more")
            print(f"     of the same data. The live options become (2) the objective and (3) the")
            print(f"     per-system-basis advantage -- which need DIFFERENT experiments, and this one")
            print(f"     cannot separate them.", flush=True)
    print(f"\n  SCOPE: holds architecture, objective and comparator FIXED and varies ONLY n_train, so")
    print(f"  it discriminates (1) alone. The N-only scale correction is deliberately NOT applied --")
    print(f"  folding it in would confound 'does more data help' with 'does the calibration fix help'.",
          flush=True)

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
