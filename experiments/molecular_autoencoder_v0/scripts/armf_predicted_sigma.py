#!/usr/bin/env python3
"""INBOX 109c, owed since 107d: the ANGSTROM error under PREDICTED sigma.

WHY THIS IS THE DECIDING NUMBER. The ANM basis is zero-shot from the reference structure -- that
claim holds. The WHITENING is not: `sd` is 64 numbers currently obtained from that protein's own MD.
106c measured the correlation between equipartition's prediction and the truth (pooled r = +0.730,
53% of the variance in log sigma) and 107d showed the miss is within-system SHAPE, not a per-system
scale, so a B-factor scalar does not rescue it.

But a correlation is not a decision. The decision-relevant number is what substituting the predicted
sigma COSTS IN ANGSTROMS on held-out systems: if the per-system reconstruction is essentially
unchanged, the pipeline is zero-shot FOR THIS PURPOSE whatever r says; if it degrades materially,
"needs a short simulation" is established rather than inferred.

INBOX 115d, AND IT CORRECTS A LABEL OF MINE. 115d asks for the rank-64 reconstruction of TRUE frames
as a floor beneath the 3.009 A baseline. That floor was already the baseline and I had mislabelled
it: `rec_meas = ((C / sd_meas) * sd_meas) @ V.T + mu` is algebraically `C @ V.T + mu`, the rank-64
projection of the true frames, with NO generative model anywhere in it. Calling it "RMSD with
measured sigma" implied a model was involved. The row is renamed `rmsd_floor` and the identity is
asserted in code, so the two names cannot drift apart again.

That makes 115d's second reading the one that holds: the floor is the WHOLE of the 3.009 A, so the
sigma penalty is an increment on top of a truncation error that is already 3 A. Milestone 4's demo
is capped there regardless of the generator or of where sigma comes from -- which raises the
question 115d's two readings do not cover: IS the floor buyable? So the floor is swept over
K = 8..256 on the same frames and the same basis. If it falls steeply, 3 A is a K choice; if it
plateaus, 3 A is what a linear ANM subspace can represent and no amount of K fixes it.

INBOX 116 item 2 -- THE SWEEP NEEDED ITS OWN NULL. K=0 is `mu` alone: the mean structure, no modes.
Without it 3.009 A has no denominator, which is the same gap 115d flagged for 3.009 itself, one level
down. K=0 is now the first row of the sweep and every later row is quoted as a fraction of the
variance it removes.

INBOX 116 item 3 -- BOTH SWEEPS, BECAUSE "3 A IS A CHOICE OF K" MAY NOT BE A CHOICE AVAILABLE
ZERO-SHOT. ANM orders by eigenvalue, so modes 65-256 are the stiffer, faster, more local ones, and
three things pull against buying the floor with rank: the tilt hypothesis says the model already
over-weights fast modes; high modes are the ones most determined by CUTOFF, so a floor bought at
K=256 may be bought with modes that are not collective motion at all; and their sigma is the hardest
to predict from equipartition.

THE PREDICTION, RECORDED BEFORE THE NUMBERS (116.3):
    the MEASURED-sigma floor keeps falling with K
    the PREDICTED-sigma floor FLATTENS, because equipartition's error grows with mode index
    if so, the ZERO-SHOT floor is NOT buyable with rank even though the oracle floor is, and the GAP
      BETWEEN THE TWO CURVES is the true cost of not simulating the target -- far more useful than a
      single +0.528 A at one K
    if the curves track, raising K is a clean win and the penalty is a fixed offset

The overall-scale match that 107d motivated is applied WITHIN each rank, so a rank-k row is a
self-consistent zero-shot prediction at that rank rather than one rescaled by a 64-mode constant.

108.1's PRE-REGISTERED BUG CHECK APPLIES HERE AND IS RUN. A sigma error is a per-mode SCALE, so it
MUST move std, js and trans and MUST leave xcorr, amp, kurt and iat unchanged to |rel| < 1e-6 --
not to zero, because kurt sits at 1.1e-08 from float reassociation. If xcorr or amp moves, there is
a leak and the substitution is touching something it should not.
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from armf_atlas_data import AtlasStore, sysdata
from armf_anm import modes as anm_modes, hessian
from armf_propagator import stats_of, METRICS
import armf_atlas_dm as D
import armf_io

WR = D.WR
K = int(os.environ.get("PS_K", "64"))
CUTOFF = 5.0
NEVAL = int(os.environ.get("PS_NEVAL", "24"))
NFRAME = int(os.environ.get("PS_NFRAME", "400"))
RES = os.environ.get("PS_RES", f"{WR}/predicted_sigma.json")
kT = 0.593
# 115d: the floor sweep. KMAX bounds the one basis every rank is sliced from.
KSWEEP = [int(x) for x in os.environ.get("PS_KSWEEP", "0,8,16,32,64,128,256").split(",")]
KMAX = max(KSWEEP + [K])


def ca_index(pdb, N):
    p = f"{WR}/atlas_topo/{pdb}.pdb"
    if not os.path.exists(p): return None
    at = [l for l in open(p, errors="ignore").read().splitlines()
          if l.startswith(("ATOM", "HETATM"))]
    if len(at) != N: return None
    return np.array([i for i, l in enumerate(at) if l[12:16].strip() == "CA"])


if __name__ == "__main__":
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho = [p for p in man["heldout"] if p in have][:NEVAL]
    print(f"[109c] Angstrom cost of PREDICTED sigma, {len(ho)} held-out systems, K={K}", flush=True)
    rows, t0 = {}, time.time()
    for i, p in enumerate(ho, 1):
        try:
            d = sysdata(store, have[p])
            # 115d: one basis computed at KMAX and sliced, so every rank in the sweep is the SAME
            # modes truncated -- a rank-k row is not a separately-fitted basis.
            kmax = min(KMAX, 3 * d["N"] - 6)
            Vmax, _, _ = anm_modes(d["ref"], kmax, CUTOFF)
            if Vmax is None: continue
            V = Vmax[:, :K]
            H = hessian(d["ref"], CUTOFF)
            # 116.3: eigenvalues for EVERY mode in the sweep, computed once. The old loop rebuilt
            # H @ v per mode for 64 modes; this is one sparse matmul for all of them.
            lam_all = np.clip(np.einsum("ij,ij->j", Vmax, H @ Vmax), 1e-12, None)
            lam = lam_all[:K]
            a = np.load(d["path"], mmap_mode="r")
            idx = np.arange(0, d["F"], max(1, d["F"] // NFRAME))[:NFRAME]
            X = np.asarray(a[2][idx]).astype(np.float64).reshape(len(idx), -1) - d["mu"]
            C = X @ V
            sd_meas = C.std(0) + 1e-8
            sd_pred = np.sqrt(kT / lam)
            # match the OVERALL scale so this measures SHAPE error, not a units mismatch --
            # 107d already showed a per-system offset is one number and is separable
            sd_pred = sd_pred * (sd_meas.mean() / sd_pred.mean())
            # round-trip through each whitening: encode -> whiten -> unwhiten -> decode
            rec_meas = ((C / sd_meas) * sd_meas) @ V.T + d["mu"]
            rec_pred = ((C / sd_meas) * sd_pred) @ V.T + d["mu"]
            truth = X + d["mu"]

            def rmsd(P):
                return float(np.sqrt(((P - truth) ** 2).reshape(len(X), -1, 3).sum(-1).mean()))

            e_meas = rmsd(rec_meas)
            e_pred = rmsd(rec_pred)
            # 115d: the measured-sigma round trip IS the rank-K reconstruction of true frames --
            # ((C/sd)*sd) @ V.T is C @ V.T. Asserted so the floor and the baseline cannot drift into
            # two names for one number again.
            assert abs(rmsd(C @ V.T + d["mu"]) - e_meas) < 1e-6 * max(e_meas, 1e-12)
            # IS THE FLOOR BUYABLE, AND IS IT BUYABLE ZERO-SHOT? 116 items 2 and 3. Two sweeps
            # over the same frames and the same basis: the ORACLE floor (measured sigma, which is
            # just the rank-k projection) and the ZERO-SHOT floor (equipartition sigma at rank k).
            # K=0 is `mu` alone -- the null the sweep was missing -- and falls out of the same
            # expression, since Vmax[:, :0] makes the projection term exactly zero.
            sweep, sweep_pred = {}, {}
            for k in KSWEEP:
                if k > kmax: continue
                Vk = Vmax[:, :k]
                sweep[k] = rmsd((X @ Vk) @ Vk.T + d["mu"])
                if k == 0:
                    sweep_pred[k] = sweep[k]        # no modes, no sigma: the two curves must meet
                    continue
                Ck = X @ Vk
                sdm = Ck.std(0) + 1e-8
                sdp = np.sqrt(kT / lam_all[:k])
                # 107d's overall-scale match applied WITHIN this rank, so the row is a
                # self-consistent zero-shot prediction at rank k rather than one rescaled by a
                # 64-mode constant.
                sdp = sdp * (sdm.mean() / sdp.mean())
                sweep_pred[k] = rmsd(((Ck / sdm) * sdp) @ Vk.T + d["mu"])
            ca = ca_index(p, d["N"])
            def span(P):
                if ca is None or len(ca) < 2: return float("nan")
                Q = P.reshape(len(P), -1, 3)[:, ca]
                return float(np.linalg.norm(Q[:, 1:] - Q[:, :-1], axis=-1).mean())
            # 108.1 bug check: which statistics may move under a pure per-mode rescale?
            # `thr` must be in the units of the series it is applied to. Computing it from the
            # UNWHITENED C and scoring the WHITENED series made every frame fall in one basin, so
            # trans came out identical on both arms and the bug check flagged a LEAK that was mine.
            Cw = C / sd_meas
            top2 = np.argsort(Cw.std(0))[::-1][:2]; thr = np.median(Cw[:, top2], 0)
            A = stats_of(Cw[:256], top2, thr, Cw)
            B = stats_of((Cw * (sd_pred / sd_meas))[:256], top2, thr, Cw)
            moved = {m: abs(B[m] - A[m]) / max(abs(A[m]), 1e-30) for m in METRICS}
            rows[p] = dict(N=d["N"], rmsd_floor=e_meas, rmsd_meas=e_meas, rmsd_pred=e_pred,
                           kmax=kmax, sweep={str(k): v for k, v in sweep.items()},
                           sweep_pred={str(k): v for k, v in sweep_pred.items()},
                           ca_truth=span(truth), ca_meas=span(rec_meas), ca_pred=span(rec_pred),
                           moved=moved)
            print(f"  [{i}/{len(ho)}] {p:10s} N={d['N']:>6}  rank-{K} FLOOR {e_meas:6.3f} A  "
                  f"predicted-sigma {e_pred:6.3f} A  ({e_pred-e_meas:+.3f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(ho)}] {p}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(ho), complete=(i == len(ho)), n_failed=0, K=K)

    if not rows: raise SystemExit("  nothing scored")
    em = np.array([r["rmsd_meas"] for r in rows.values()])
    ep = np.array([r["rmsd_pred"] for r in rows.values()])
    print(f"\n=== 109c: WHAT PREDICTED SIGMA COSTS ({len(rows)} systems) ===")
    print(f"  rank-{K} reconstruction of TRUE frames (THE FLOOR): median {np.median(em):.3f} A")
    print(f"    -- 115d: this IS the measured-sigma row. ((C/sd)*sd) @ V.T is C @ V.T, so no")
    print(f"       generative model enters it. The old label 'RMSD with measured sigma' implied one.")
    print(f"  same frames, PREDICTED sigma                     : median {np.median(ep):.3f} A")
    print(f"  cost: {np.median(ep-em):+.3f} A median, {100*np.median((ep-em)/em):+.1f}% relative; "
          f"worse on {100*(ep>em).mean():.0f}% of systems")
    print(f"\n  115d/116.2/116.3: IS THE FLOOR BUYABLE, AND IS IT BUYABLE ZERO-SHOT?")
    print(f"  Same frames, one basis, rank truncated. K=0 is `mu` alone -- the null the sweep was")
    print(f"  missing, and the denominator 3.009 A did not have.")
    print(f"    {'K':>5}{'ORACLE sigma':>14}{'vs prev':>9}{'% of K=0':>10}"
          f"{'ZERO-SHOT sigma':>17}{'vs prev':>9}{'GAP':>8}{'n':>6}")
    prev = prevp = k0 = None
    for k in KSWEEP:
        v = [r["sweep"][str(k)] for r in rows.values() if str(k) in r.get("sweep", {})]
        vp = [r.get("sweep_pred", {})[str(k)] for r in rows.values()
              if str(k) in r.get("sweep_pred", {})]
        if not v: continue
        med, medp = float(np.median(v)), (float(np.median(vp)) if vp else float("nan"))
        if k0 is None: k0 = med
        d1 = f"{100*(prev-med)/prev:+7.1f}%" if prev else " " * 9
        d2 = f"{100*(prevp-medp)/prevp:+7.1f}%" if (prevp and medp == medp) else " " * 9
        print(f"    {k:>5}{med:>13.3f}A{d1}{100*med/k0:>9.1f}%{medp:>16.3f}A{d2}"
              f"{medp-med:>+7.3f}{len(v):>6}")
        prev, prevp = med, (medp if medp == medp else prevp)
    print(f"\n  116.3 PREDICTED, before these numbers: the oracle curve keeps falling and the")
    print(f"  zero-shot curve FLATTENS, because equipartition's error grows with mode index. If so")
    print(f"  the zero-shot floor is NOT buyable with rank and the GAP column is the real cost of")
    print(f"  not simulating the target. If the curves track, raising K is a clean win.")
    print(f"\n  108.1 PRE-REGISTERED BUG CHECK -- a per-mode scale MUST move std/js/trans and MUST")
    print(f"  leave xcorr/amp/kurt/iat below |rel| < 1e-6:")
    for m in METRICS:
        v = np.median([r["moved"][m] for r in rows.values()])
        must_move = m in ("std", "js", "trans")
        ok = (v > 1e-6) if must_move else (v < 1e-6)
        print(f"    {m:>6}: median |rel| {v:.3e}  {'moved' if v>1e-6 else 'unchanged':>9}  "
              f"{'OK' if ok else '*** LEAK ***'}")
    print(f"\n  -> {'predicted sigma is usable: the Angstrom cost is small' if np.median((ep-em)/em) < 0.1 else 'predicted sigma is NOT usable at this cost; a short simulation of the target is required'}")
