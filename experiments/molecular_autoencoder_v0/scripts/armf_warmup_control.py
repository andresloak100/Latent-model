"""DOES THE CURRENT TRAINING PROCEDURE REPRODUCE THE ARMS ALREADY IN atlas_dm.json?

WHY THIS EXISTS. Re-training the sweep's best arm (n_train=50, DM=256, lr=3e-4, seed=1) for INBOX
14b did NOT reproduce it: best tracked FVE +0.0515 against the recorded +0.0972 on the SAME 24
tracked systems, and it plateau-stopped at 22,500 steps rather than 15,000. Same architecture (dlat
== dm, so no projection is inserted), same seeds, same training pool, same tracked set.

WHAT CHANGED BETWEEN THEM. The 13 arms persisted in `atlas_dm.json` were trained BEFORE two edits
made to rescue the wide low-LR arms from a Family D hole:
    WARMUP = 1000        linear LR ramp over the first 1,000 steps  (was: none)
    maxsteps_for(lr)     lr-scaled budget, 54,779 steps at 3e-4     (was: flat 30,000)
Completed arms SKIP on re-run, by design, so those 13 will never be retrained. That means
`atlas_dm.json` now MIXES TWO TRAINING PROCEDURES, and the sweep's own tables compare them as one
experiment: at n_train=50 the legacy rows cover DM=16 (5 LRs), DM=64 (3 LRs) and DM=256 (5 LRs),
while DM=512 and the two missing DM=64 LRs will be trained under the new procedure. A DM curve built
from that is FAMILY E -- a comparator separated from its peers by an unswept hyperparameter -- and
the LR winner chosen at n_train=50 propagates to every higher rung.

THIS FILE DOES NOT ASSUME THE CAUSE. One failed re-run is one draw, and the difference could be GPU
nondeterminism rather than the procedure. So both procedures run HERE, in ONE process on ONE card,
at TWO seeds each:

    LEGACY   WARMUP = 1     (off), flat 30,000-step budget   -- what the persisted arms used
    CURRENT  WARMUP = 1000, lr-scaled budget                 -- what every new arm uses

READ:
  LEGACY reproduces ~0.10 and CURRENT lands near ~0.05  -> the procedure change is the cause, the
      mixing is real, and the persisted arms must be invalidated rather than reported alongside
      new ones.
  BOTH land together, wherever that is                  -> the single re-run was seed/hardware
      noise, the arms are comparable, and nothing needs invalidating.
  BOTH scatter across seeds by more than the gap        -> the arm is not seed-stable at all, and
      NO single-seed arm in the sweep should be quoted as a width result.

The comparison is against `best_track` on the tracked systems -- NOT against the recorded `fve`,
which is a mean over ALL held-out systems and runs ~1.6x higher (0.1553 vs 0.0913 for seed 1).
Comparing across those two denominators is what made the first divergence report overstate itself."""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata
import armf_atlas_dm as D

WR = D.WR
RES = f"{WR}/warmup_control.json"
DM, LR, NTR = 256, 3e-4, 50
SEEDS = [1, 3]
# THE HYBRID, ADDED BEFORE THE VERDICT so the follow-up is not designed after seeing which way it
# went. If LEGACY beats CURRENT, "revert the warmup" is NOT automatically the right move: warmup was
# introduced alongside the lr-scaled budget to close a Family D hole (3e-5 arms hitting the step cap
# STILL IMPROVING, hence VOID), and reverting both would reopen it. The hybrid separates the two
# edits -- warmup OFF, lr-scaled budget KEPT -- which is the only combination that could fix the void
# without paying for it at the operating point. Adopting it untested would repeat exactly the error
# under investigation, so it is measured at the same LR as the others (does it match LEGACY?) and at
# 3e-5 (does it converge instead of VOIDing?).
HYBRID_LRS = [3e-4, 3e-5]


if __name__ == "__main__":
    print(f"[warmup-control] does the CURRENT procedure reproduce the PERSISTED arms? "
          f"n_train={NTR} DM={DM} lr={LR:g}, seeds {SEEDS}, both procedures, one card.", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]

    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    _ord = sorted(range(len(HO)), key=lambda i: HO[i]["N"])
    HOt = [HO[_ord[i]] for i in np.linspace(0, len(HO) - 1, min(D.NHO_TRACK, len(HO))).astype(int)]
    TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:NTR]) if x is not None]
    print(f"  {len(TR)} train systems, {len(HOt)} tracked, {len(HO)} held-out total", flush=True)

    rec = [r for r in json.load(open(D.RES))
           if r.get("L") == 1 and r["dm"] == DM and abs(r["lr"] - LR) < 1e-12
           and r.get("arch") == "network"]
    print(f"\n  PERSISTED (legacy-procedure) ARMS FOR THIS CELL:", flush=True)
    for r in sorted(rec, key=lambda r: r["seed"]):
        print(f"    seed{r['seed']}: best_track {r['best_track']:+.4f}  tail_track {r['tail_track']:+.4f}"
              f"  steps {r['steps']} ({r['stopped']})   [all-{r['nho']} fve {r['fve']:+.4f}]", flush=True)
    legacy_bt = np.array([r["best_track"] for r in rec], float)
    print(f"    -> legacy best_track: mean {legacy_bt.mean():+.4f}, spread "
          f"{legacy_bt.max()-legacy_bt.min():.4f} across {len(legacy_bt)} seeds", flush=True)

    out = json.load(open(RES)) if os.path.exists(RES) else {}
    orig_ms = D.maxsteps_for
    # (name, warmup, maxsteps_fn, [lrs]). HYBRID separates the two edits that shipped together.
    PLANS = [("LEGACY ", 1, lambda lr: 30000, [LR]),
             ("CURRENT", 1000, orig_ms, [LR]),
             ("HYBRID ", 1, orig_ms, HYBRID_LRS)]
    for proc, warm, msf, lrs in PLANS:
        for lr_ in lrs:
            for sd in SEEDS:
                # Key format is BACKWARD COMPATIBLE: arms at the default LR keep the original
                # `NAME_sSEED` key, so results already written by an in-flight job are recognised
                # and not silently retrained.
                key = (f"{proc.strip()}_s{sd}" if lr_ == LR
                       else f"{proc.strip()}_lr{lr_:g}_s{sd}")
                if key in out: continue
                D.WARMUP = warm; D.maxsteps_for = msf
                torch.manual_seed(sd); np.random.seed(sd + 1)
                t0 = time.time()
                mdl, hist, used, stopped, improving = D.train(
                    TR, HOt, DM, lr_, f"{proc} lr{lr_:g} s{sd}", 1)
                mdl.eval()
                bt = max(h[1] for h in hist); tt = float(np.mean([h[1] for h in hist[-5:]]))
                full = float(np.mean([D.fve_model(mdl, x) for x in HO]))
                out[key] = dict(proc=proc.strip(), warmup=warm, seed=sd, lr=lr_, best_track=bt,
                                tail_track=tt, full_fve=full, steps=used, stopped=stopped,
                                improving=improving, secs=time.time() - t0)
                json.dump(out, open(RES, "w"))
                print(f"  {proc} lr{lr_:g} seed{sd}: best_track {bt:+.4f}  tail {tt:+.4f}  "
                      f"all-{len(HO)} FVE {full:+.4f}  steps {used} ({stopped})"
                      f"{'  *** STILL IMPROVING -> VOID ***' if improving else ''}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    D.maxsteps_for = orig_ms

    print(f"\n=== VERDICT ===", flush=True)
    def bt(nm, lr_=LR, field="best_track"):
        return np.array([v[field] for v in out.values()
                         if v["proc"] == nm and v.get("lr", LR) == lr_], float)
    L, C, H = bt("LEGACY"), bt("CURRENT"), bt("HYBRID")
    # REPORT BOTH METRICS, because they answer different questions and can disagree.
    #   best_track = the PEAK of the tracking curve on the 24 stratified systems during training
    #   full_fve   = the FINAL weights evaluated on ALL held-out systems
    # The sweep SELECTS ITS WINNERS ON full_fve, so that is the metric the procedure decision must
    # turn on; best_track is a property of the training curve's shape and two procedures can differ
    # sharply there while landing in the same place. Deciding on best_track alone would optimise a
    # diagnostic instead of the reported quantity.
    Lf, Cf, Hf = bt("LEGACY", field="full_fve"), bt("CURRENT", field="full_fve"), bt("HYBRID", field="full_fve")
    print(f"\n  METRIC 1 -- best_track (peak of the tracking curve, 24 stratified systems):")
    for nm, v in (("LEGACY", L), ("CURRENT", C), ("HYBRID", H)):
        if len(v): print(f"    {nm:>8}: mean {v.mean():+.4f}  n={len(v)}  spread {v.max()-v.min():.4f}")
    print(f"  METRIC 2 -- full_fve (FINAL weights, ALL held-out systems) <- WHAT THE SWEEP SELECTS ON:")
    for nm, v in (("LEGACY", Lf), ("CURRENT", Cf), ("HYBRID", Hf)):
        if len(v): print(f"    {nm:>8}: mean {v.mean():+.4f}  n={len(v)}  spread {v.max()-v.min():.4f}")
    if len(Lf) and len(Cf):
        gf = Lf.mean() - Cf.mean(); wf = max(Lf.max()-Lf.min(), Cf.max()-Cf.min())
        print(f"    LEGACY - CURRENT on full_fve = {gf:+.4f}, within-procedure seed spread {wf:.4f}"
              f"  -> {'SEPARATED' if abs(gf) > wf else 'NOT SEPARATED by these seeds'}", flush=True)
    if not len(L) or not len(C):
        print("  incomplete."); raise SystemExit
    print(f"  LEGACY   best_track {L.mean():+.4f}  (n={len(L)}, spread {L.max()-L.min():.4f})")
    print(f"  CURRENT  best_track {C.mean():+.4f}  (n={len(C)}, spread {C.max()-C.min():.4f})")
    print(f"  recorded legacy arms  {legacy_bt.mean():+.4f}  (n={len(legacy_bt)}, spread "
          f"{legacy_bt.max()-legacy_bt.min():.4f})", flush=True)
    if len(H):
        print(f"  HYBRID   best_track {H.mean():+.4f}  (n={len(H)}, spread {H.max()-H.min():.4f})"
              f"   [warmup OFF, lr-scaled budget KEPT]", flush=True)
        v = [x for x in out.values() if x["proc"] == "HYBRID" and x.get("lr") != LR]
        for x in sorted(v, key=lambda z: (z["lr"], z["seed"])):
            print(f"    HYBRID lr{x['lr']:g} s{x['seed']}: best_track {x['best_track']:+.4f}  "
                  f"steps {x['steps']} ({x['stopped']})"
                  f"{'  *** VOID: still improving ***' if x['improving'] else '  converged'}",
                  flush=True)
        print(f"  The 3e-5 HYBRID rows answer the question reverting warmup would otherwise reopen:")
        print(f"  does the lr-scaled budget ALONE close the Family D hole warmup was added for?",
              flush=True)
    gap = L.mean() - C.mean()
    within = max(L.max() - L.min(), C.max() - C.min())
    if gap > within and gap > 0.02:
        print(f"  => THE PROCEDURE CHANGE IS THE CAUSE. Gap {gap:+.4f} exceeds the within-procedure")
        print(f"     seed spread ({within:.4f}). atlas_dm.json MIXES two procedures and its DM/LR")
        print(f"     tables are FAMILY E. The persisted arms must be INVALIDATED and retrained under")
        print(f"     the current procedure, not reported beside new ones.", flush=True)
    elif abs(gap) <= within:
        print(f"  => NOT THE PROCEDURE. Gap {gap:+.4f} is within the seed spread ({within:.4f}); the")
        print(f"     single failed re-run was one draw. But note what that means: this arm's")
        print(f"     seed-to-seed spread is {within:.4f} on a value of ~{C.mean():.3f}, so NO")
        print(f"     single-seed arm should be quoted as a width result.", flush=True)
    else:
        print(f"  => CURRENT is BETTER by {-gap:+.4f}, exceeding the seed spread. The persisted arms")
        print(f"     UNDERSTATE what the architecture does; the mixing is still real and still")
        print(f"     Family E, just in the other direction.", flush=True)
