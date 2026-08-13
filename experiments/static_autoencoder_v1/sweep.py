#!/usr/bin/env python3
"""The grid driver. Writes incrementally, reports completeness, never silently drops a cell.

    python sweep.py --synthetic            # harness self-test, minutes on a CPU
    python sweep.py --out results.json     # the real grid (needs RealCorpus implemented)

A cell that fails is RECORDED AS FAILED. A partial grid presented as a full one is the most
common way a scaling study misleads, so `completeness` is written into the results file.
"""

from __future__ import annotations

import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sae.data import SyntheticStructures
from sae.model import AEConfig, config_for_params
from sae.scaling import fit_scaling, reading
from sae.train import TrainConfig, run

PARAMS = [500_000, 2_000_000, 8_000_000, 32_000_000]
LATENTS = [32, 64, 128, 256, 512, 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--intrinsic-dim", type=int, default=16)
    ap.add_argument("--params", type=int, nargs="*", default=None)
    ap.add_argument("--latents", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-atoms", type=int, default=64)
    ap.add_argument("--structures", type=int, default=4096)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    params = a.params or PARAMS
    latents = a.latents or LATENTS

    if a.synthetic:
        ds = SyntheticStructures(n_structures=a.structures, n_atoms=a.n_atoms,
                                 intrinsic_dim=a.intrinsic_dim, n_elements=4, seed=0)
        print(f"[synthetic] {len(ds)} structures, {a.n_atoms} atoms, "
              f"INTRINSIC DIM {a.intrinsic_dim} -- the curve MUST saturate at or before "
              f"latent size {a.intrinsic_dim}", flush=True)
    else:
        from sae.data import RealCorpus
        ds = RealCorpus()

    base = AEConfig(max_atoms=max(a.n_atoms, 1024), n_elements=16)
    rows, t0 = [], time.time()
    total = len(params) * len(latents)

    for P in params:
        for L in latents:
            cfg = config_for_params(P, replace(base, latent_width=L, n_latent_tokens=1))
            r = run(cfg, TrainConfig(steps=a.steps, seed=0), ds)
            row = r.as_dict(); row["target_params"] = P
            rows.append(row)
            tag = "FAILED " + r.failed if r.failed else \
                (f"ho_mse {r.heldout_mse:.5f}  gap {r.gap:+.5f}  "
                 f"eff_rank {r.effective_rank:5.1f}/{L}  probe_r2 {r.probe_r2:+.3f}"
                 + (f"  fve {r.frac_var_explained:+.3f}")
                 + ("" if r.converged else f"  [{r.status.upper()} tail={r.tail_improvement:+.1%}]"))
            print(f"  [{len(rows):>2}/{total}] target_P {P:>10,} actual {r.n_params:>10,} "
                  f"L {L:>4}  {tag}  ({r.seconds:.0f}s)", flush=True)
            Path(a.out).write_text(json.dumps({"rows": rows, "complete": False}, indent=2))

    # UNCONVERGED cells are excluded from the fit, not silently averaged in. An under-trained
    # cell is flat, and a row of flat cells reads as saturation -- the exact wrong conclusion.
    ok = [r for r in rows if not r["failed"] and r["converged"]]
    unconv = [r for r in rows if not r["failed"] and not r["converged"]]
    out = {
        "rows": rows,
        "complete": True,
        "completeness": {"cells_requested": total, "cells_run": len(rows),
                         "cells_failed": sum(1 for r in rows if r["failed"]),
                         "cells_stuck": sum(1 for r in unconv if r["status"] == "stuck"),
                         "cells_still_improving": sum(1 for r in unconv
                                                      if r["status"] == "still_improving"),
                         "cells_used_in_fit": len(ok)},
        "sweep": {"params": params, "latents": latents, "steps": a.steps,
                  "synthetic": a.synthetic,
                  "intrinsic_dim": a.intrinsic_dim if a.synthetic else None},
        "seconds": time.time() - t0,
    }

    if ok:
        # GROUP BY TARGET, NOT ACTUAL. config_for_params re-solves the width for every latent
        # size, so actual parameter counts differ by a few thousand between cells in the same
        # row -- grouping on the actual count matched exactly one row and silently reduced the
        # latent-axis fit to n=1. Caught by the synthetic self-test, which is what it is for.
        big_L = max(r["latent_size"] for r in ok)
        by_P = [r for r in ok if r["latent_size"] == big_L]
        big_P = max(r["target_params"] for r in ok)
        by_L = [r for r in ok if r["target_params"] == big_P]
        fp = fit_scaling([r["n_params"] for r in by_P], [r["heldout_mse"] for r in by_P])
        fl = fit_scaling([r["latent_size"] for r in by_L], [r["heldout_mse"] for r in by_L])
        out["fit_params_axis"] = fp.as_dict()
        out["fit_latent_axis"] = fl.as_dict()
        out["reading"] = reading(fp, fl)
        print("\n" + out["reading"])

    Path(a.out).write_text(json.dumps(out, indent=2))
    if unconv:
        n_stuck = sum(1 for r in unconv if r["status"] == "stuck")
        n_imp = sum(1 for r in unconv if r["status"] == "still_improving")
        print(f"\nWARNING: {len(unconv)} cell(s) excluded from the fit -- "
              f"{n_stuck} STUCK (never left the do-nothing baseline), "
              f"{n_imp} STILL_IMPROVING (under-trained). Both look FLAT, and a flat row reads "
              f"as saturation, which is the wrong pre-registered verdict. Fix them before "
              f"reading anything.")
    print(f"\nwrote {a.out}  ({len(ok)}/{total} cells used in fit, "
          f"{len(unconv)} unconverged, {sum(1 for r in rows if r['failed'])} failed)")


if __name__ == "__main__":
    main()
