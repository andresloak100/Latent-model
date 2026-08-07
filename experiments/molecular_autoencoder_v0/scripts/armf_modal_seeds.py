"""INBOX 24c: DOES THE LR SWEEP HAVE THE RESOLUTION TO RULE OUT FAMILY E? Replicate and find out.

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
RES = f"{WR}/modal_seeds.json"
DM, NTR = 256, 50
USABLE_LRS = [3e-5, 1e-4, 3e-4]      # the three that do not collapse to a constant code
SEEDS = [0, 1, 2]
VARIANTS = ["control", "untied"]
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
    done = {(r["variant"], r["lr"], r["seed"]) for r in rows}
    orig = D.Codec

    for kind in VARIANTS:
        for lr in USABLE_LRS:
            for sd in SEEDS:
                if (kind, lr, sd) in done: continue
                D.Codec = make(kind)
                try:
                    torch.manual_seed(sd); np.random.seed(sd + 1)
                    t0 = time.time()
                    mdl, hist, used, stopped, improving = D.train(
                        TR, HOt, DM, lr, f"{kind} lr{lr:g} s{sd}", 1)
                    mdl.eval()
                    per = [D.fve_model(mdl, x) for x in HO]
                    rec = dict(variant=kind, lr=lr, seed=sd, dm=DM, n_train=NTR, stamp=ST,
                               fve=float(np.mean(per)), steps=used, stopped=stopped,
                               improving=improving, best_track=max(h[1] for h in hist),
                               secs=time.time() - t0)
                    rows.append(rec); json.dump(rows, open(RES, "w")); done.add((kind, lr, sd))
                    print(f"    {kind} lr{lr:g} s{sd}: FVE {rec['fve']:+.4f}  steps {used} "
                          f"({stopped}){'  *** VOID ***' if improving else ''}  "
                          f"[{time.time()-t0:.0f}s]", flush=True)
                except Exception as e:
                    print(f"    {kind} lr{lr:g} s{sd}: FAIL {type(e).__name__}: {e}", flush=True)
                finally:
                    D.Codec = orig

    if not rows: raise SystemExit
    # ---------------- REPORT ----------------
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
