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
            V, _, _ = anm_modes(d["ref"], K, CUTOFF)
            if V is None: continue
            H = hessian(d["ref"], CUTOFF)
            lam = np.clip([float(V[:, k] @ (H @ V[:, k])) for k in range(V.shape[1])], 1e-12, None)
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
            e_meas = float(np.sqrt(((rec_meas - truth) ** 2).reshape(len(X), -1, 3).sum(-1).mean()))
            e_pred = float(np.sqrt(((rec_pred - truth) ** 2).reshape(len(X), -1, 3).sum(-1).mean()))
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
            rows[p] = dict(N=d["N"], rmsd_meas=e_meas, rmsd_pred=e_pred,
                           ca_truth=span(truth), ca_meas=span(rec_meas), ca_pred=span(rec_pred),
                           moved=moved)
            print(f"  [{i}/{len(ho)}] {p:10s} N={d['N']:>6}  RMSD measured-sigma {e_meas:6.3f} A  "
                  f"predicted-sigma {e_pred:6.3f} A  ({e_pred-e_meas:+.3f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(ho)}] {p}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(ho), complete=(i == len(ho)), n_failed=0, K=K)

    if not rows: raise SystemExit("  nothing scored")
    em = np.array([r["rmsd_meas"] for r in rows.values()])
    ep = np.array([r["rmsd_pred"] for r in rows.values()])
    print(f"\n=== 109c: WHAT PREDICTED SIGMA COSTS ({len(rows)} systems) ===")
    print(f"  RMSD with MEASURED sigma : median {np.median(em):.3f} A")
    print(f"  RMSD with PREDICTED sigma: median {np.median(ep):.3f} A")
    print(f"  cost: {np.median(ep-em):+.3f} A median, {100*np.median((ep-em)/em):+.1f}% relative; "
          f"worse on {100*(ep>em).mean():.0f}% of systems")
    print(f"\n  108.1 PRE-REGISTERED BUG CHECK -- a per-mode scale MUST move std/js/trans and MUST")
    print(f"  leave xcorr/amp/kurt/iat below |rel| < 1e-6:")
    for m in METRICS:
        v = np.median([r["moved"][m] for r in rows.values()])
        must_move = m in ("std", "js", "trans")
        ok = (v > 1e-6) if must_move else (v < 1e-6)
        print(f"    {m:>6}: median |rel| {v:.3e}  {'moved' if v>1e-6 else 'unchanged':>9}  "
              f"{'OK' if ok else '*** LEAK ***'}")
    print(f"\n  -> {'predicted sigma is usable: the Angstrom cost is small' if np.median((ep-em)/em) < 0.1 else 'predicted sigma is NOT usable at this cost; a short simulation of the target is required'}")
