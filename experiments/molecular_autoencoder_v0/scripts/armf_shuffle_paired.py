#!/usr/bin/env python3
"""INBOX 115a: the SHUFFLE arm read through the PAIRED verdict, on all held-out systems.

WHAT 112b GOT WRONG, AND WHY IT MATTERS. 112b saw `SHUFFLE 10/11, failing only iat` on the two gate
systems and concluded "trans does not discriminate either, and iat carries the entire temporal
signal." Two numbers already on the record contradict that:

    trans under a time shuffle (112a, measured)       231.37 -> 713.73   +208%
    trans under the PAIRED test on lv_r8              18/24, p = 0.0227, median +80.4

trans MORE THAN TRIPLES when time order is destroyed, and it FIRES under the paired rule. It failed
to flag the shuffle because the VERDICT was per-system interval overlap, which 113d measured as
catching 0/24 at a 3.7x timescale error. The metric was never the problem; the rule was, and it is
the same rule 113d already replaced. The correct sentence is "the shuffle passes trans UNDER INTERVAL
OVERLAP" -- a statement about power, not about the statistic.

THE PREDICTION, WRITTEN BEFORE THIS RUNS (115a):
    trans FIRES against the shuffle under the paired rule
    the shuffle scores 9/11 rather than 10/11
    if trans still does not fire against a +208% shift, the PAIRING is broken and that finding is
    worth more than the run

WHY THIS IS A SEPARATE CPU SCRIPT AND NOT A WAIT. The shuffle arm needs no model: it is the reference
windows with their rows permuted. Every input is the same ANM preparation the GPU run does, so the
whole arm is numpy and runs while the GPU trains. 10350959 computes its own SHUFFLE on the same 24
systems and will confirm or contradict this independently -- two paths to one number, which is worth
the duplication given that 112b's error survived three items.

NOTHING IS RE-IMPLEMENTED. prepare, segments, stats_ext, consistent, band and the geometry all come
from armf_latent_video_run, which imports its estimator from armf_propagator -- so this is the same
instrument on the same data, with only the ACROSS-SYSTEM aggregation changed.
"""
import sys, os, json, time
import numpy as np
from scipy import stats as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from armf_latent_video_run import (prepare, segments, stats_ext, XMETRICS, K, T, RGRP, DLAT,
                                   KEVAL, NEVAL)
from armf_propagator import band, consistent, MIN_H
from armf_paired_verdict import paired
from armf_atlas_data import AtlasStore
import armf_atlas_dm as D
import armf_io

WR = D.WR
RES = os.environ.get("SP_RES", f"{WR}/shuffle_paired.json")
TIME_AWARE = {"iat", "trans"}

if __name__ == "__main__":
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    print(f"[115a] SHUFFLE through the paired verdict. T={T} K={K} MIN_H={MIN_H}", flush=True)
    HO = prepare(store, have, ho_ids, NEVAL)
    print(f"  prepared {len(HO)} held-out systems", flush=True)
    rng = np.random.default_rng(0)
    rows, t0 = {}, time.time()
    for i, sysd in enumerate(HO, 1):
        try:
            if T < MIN_H:
                print(f"  [{i}/{len(HO)}] {sysd['pdb']}: UNEVALUABLE T={T} < {MIN_H}", flush=True)
                continue
            parts = [segments(c, T, KEVAL, rng, disjoint=True) for c in sysd["C_ref"]]
            parts = [x for x in parts if x is not None]
            if not parts: continue
            refw = np.concatenate(parts)
            pool = np.concatenate(list(sysd["C_ref"]))
            top2 = np.argsort(pool.std(0))[::-1][:2]
            thr = np.median(pool[:, top2], 0)
            rs = [stats_ext(w.reshape(T, K), top2, thr, pool) for w in refw]
            # THE SHUFFLE: identical frames, rows permuted. Same marginals, same instantaneous mode
            # covariance, time destroyed -- the ceiling a purely STATIC model reaches.
            shuf = np.stack([w.reshape(T, K)[rng.permutation(T)].reshape(T, RGRP, DLAT)
                             for w in refw])
            ss = [stats_ext(w.reshape(T, K), top2, thr, pool) for w in shuf]
            inside = {m: bool(consistent(ss, rs, m)) for m in XMETRICS}
            rows[sysd["pdb"]] = dict(
                n_win=len(refw), agree=int(sum(inside.values())), inside=inside,
                med={m: float(band(ss, m)[0]) for m in XMETRICS},
                ref_med={m: float(band(rs, m)[0]) for m in XMETRICS})
            print(f"  [{i}/{len(HO)}] {sysd['pdb']:10s} windows={len(refw):>2} "
                  f"overlap {sum(inside.values())}/{len(inside)}  "
                  f"fails={[k for k,v in inside.items() if not v]}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(HO)}] {sysd['pdb']}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(HO), complete=(i == len(HO)), n_failed=0,
                          T=T, K=K)

    if not rows: raise SystemExit("  nothing scored")
    print(f"\n=== 115a: SHUFFLE, per-system OVERLAP vs the PAIRED rule ({len(rows)} systems) ===")
    print(f"  Ties are dropped at 108.1's own |rel| < 1e-6 -- see armf_paired_verdict.paired(). The")
    print(f"  UNFIXED sign test fired on nine of these metrics at differences of 1e-16, because it")
    print(f"  counted an exact zero as a negative. This arm is what exposed that.")
    print(f"  {'metric':>13}{'n_pos/n':>10}{'ties':>6}{'sign p':>10}{'wilcoxon p':>12}{'skew':>8}"
          f"{'median diff':>14}{'overlap caught':>16}  time?")
    fired_paired = 0
    for m in XMETRICS:
        d = np.array([r["med"][m] - r["ref_med"][m] for r in rows.values()])
        ref = [r["ref_med"][m] for r in rows.values()]
        caught = sum(1 for r in rows.values() if not r["inside"][m])
        npos, nk, p_sign, p_wil, sk, skewed, nt = paired(d, ref)
        # 115c's rule, fixed in advance: |skew| > 1 and the sign test governs.
        p = p_sign if skewed else p_wil
        if p < 0.05: fired_paired += 1
        note = "  <-- PAIRED FIRES, overlap did not" if (p < 0.05 and caught == 0) else ""
        if skewed: note += "  [SKEWED: sign test governs]"
        if nt == len(d): note += "  [ALL TIED: invariant by construction]"
        print(f"  {m:>13}{f'{npos}/{nk}':>10}{nt:>6}{p_sign:>10.4f}{p_wil:>12.4f}{sk:>8.2f}"
              f"{np.median(d):>14.4f}{f'{caught}/{len(d)}':>16}  "
              f"{'TIME' if m in TIME_AWARE else 'static'}{note}")
    print(f"\n  SHUFFLE passes {len(XMETRICS)-fired_paired}/{len(XMETRICS)} under the PAIRED rule")
    print(f"  115a predicted 9/11 with trans firing. A pass count of 10/11 with trans NOT firing")
    print(f"  against a +208% shift means the PAIRING is broken, not that trans is blind.")
