#!/usr/bin/env python3
"""Plot held-out RMSD vs training steps, one line per config — the scaling curves.

Reads the `steps` / `val_rmsd` pairs that train.py now writes into
train_log.json, so every run carries its own curve and no extra jobs are needed.

Beyond drawing the lines it answers the question the grid was built to settle:
which cells are STILL DESCENDING at the end of their budget, and which flattened
early. A cell that plateaus at a better value can still be the worse bet if a
slower one has not run out of road — that is the whole point of looking at the
curve instead of the endpoint.

"Still descending" is measured as the slope over the final third of each curve,
in A per doubling of steps, using the same plateau criterion applied to the
earlier compute control: |improvement| < ~0.05 A over the last doubling counts
as flat. A threshold on the final VALUE is deliberately not used -- that is the
mistake that would have labelled a model "floored at 0.51" while it was in fact
descending fastest.

Usage:
    python scripts/plot_curves.py --results-dir outputs/cluster --out outputs/curves.png
    python scripts/plot_curves.py --results-dir outputs/cluster --pattern 'pgrid_*'
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path

import numpy as np

# matplotlib is imported lazily inside main(): the analysis helpers below
# (load_curves, tail_slope) are pure numpy and must be importable in
# environments without a plotting stack -- the cluster venv has neither
# matplotlib nor pytest, and a test failing on a missing plotting library is
# noise, not signal.


def load_curves(results_dir: Path, pattern: str):
    """Return {name: (steps[], val_rmsd[], final_train)} for runs that have a curve."""
    out = {}
    for log in sorted(results_dir.glob("*/train_log.json")):
        name = log.parent.name
        if not fnmatch.fnmatch(name, pattern):
            continue
        try:
            rows = json.load(open(log)).get("log", [])
        except (json.JSONDecodeError, OSError):
            continue
        pts = [(r["steps"], r["val_rmsd"]) for r in rows
               if "val_rmsd" in r and "steps" in r and r["val_rmsd"] == r["val_rmsd"]]
        if len(pts) < 3:
            continue                       # not enough to call a trend
        pts.sort()
        train = [r["rmsd"] for r in rows if "rmsd" in r]
        out[name] = (np.array([p[0] for p in pts], dtype=float),
                     np.array([p[1] for p in pts], dtype=float),
                     float(np.median(train[-3:])) if train else float("nan"))
    return out


def tail_slope(steps, vals, frac=1 / 3):
    """A per doubling of steps over the final `frac` of the curve (negative = improving)."""
    k = max(3, int(len(steps) * frac))
    s, v = steps[-k:], vals[-k:]
    if s[0] <= 0 or s[-1] / s[0] < 1.05:
        return float("nan")
    return float(np.polyfit(np.log2(s), v, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs/cluster")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default="outputs/curves.png")
    ap.add_argument("--plateau-bar", type=float, default=0.05,
                    help="|A per doubling| below this counts as flat")
    ap.add_argument("--reference", type=float, default=1.02,
                    help="held-out all-atom RMSD of the incumbent design")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent.parent
    curves = load_curves(root / args.results_dir, args.pattern)
    if not curves:
        raise SystemExit(
            f"no curves under {args.results_dir} matching {args.pattern!r}. "
            "Runs need `val_rmsd` in train_log.json (train.py logs it every "
            "eval_every epochs); older runs predate that and have none.")

    rows = []
    for name, (s, v, train) in curves.items():
        rows.append((name, s, v, train, tail_slope(s, v)))
    rows.sort(key=lambda r: r[2][-1])          # best final val first

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    cmap = plt.get_cmap("viridis")
    for i, (name, s, v, train, slope) in enumerate(rows):
        c = cmap(i / max(len(rows) - 1, 1))
        flat = (not math.isnan(slope)) and abs(slope) < args.plateau_bar
        ax.plot(s, v, "-o", ms=3, lw=1.8, color=c,
                ls="--" if flat else "-",
                label=f"{name}  {v[-1]:.2f}Å  ({slope:+.3f}/dbl{', flat' if flat else ''})")
    ax.axhline(args.reference, color="crimson", lw=1.2, ls=":",
               label=f"incumbent design ({args.reference:.2f}Å)")
    ax.set_xscale("log")
    ax.set_xlabel("optimiser steps")
    ax.set_ylabel("held-out all-atom RMSD (Å)")
    ax.set_title("Scaling curves — solid = still descending, dashed = plateaued")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False, ncol=1)

    # Right panel: is a cell's advantage real, or is it just not finished yet?
    names = [r[0] for r in rows]
    finals = [r[2][-1] for r in rows]
    slopes = [r[4] for r in rows]
    y = np.arange(len(names))
    ax2.barh(y, finals, color=["#4c78a8" if abs(sl) >= args.plateau_bar else "#bab0ac"
                              for sl in slopes])
    for i, (f, sl) in enumerate(zip(finals, slopes)):
        ax2.text(f + 0.02, i, f"{sl:+.3f}/dbl", va="center", fontsize=7, color="dimgrey")
    ax2.axvline(args.reference, color="crimson", lw=1.2, ls=":")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("final held-out all-atom RMSD (Å)")
    ax2.set_title("Endpoint (blue = still improving, grey = flat)")
    ax2.grid(alpha=0.3, axis="x")

    fig.text(0.5, -0.04,
             "Slope is Å per DOUBLING of steps over the final third of each curve. "
             "A cell with a worse endpoint but a steeper slope has not run out of road; "
             "one that is flat has.\nEndpoint alone would have called an earlier model "
             '"floored" precisely while it was descending fastest — hence the slope column.',
             ha="center", fontsize=8, color="dimgrey")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")

    print(f"{'run':<26}{'final val':>10}{'final train':>12}{'Å/doubling':>12}  verdict")
    for name, s, v, train, slope in rows:
        flat = (not math.isnan(slope)) and abs(slope) < args.plateau_bar
        print(f"{name:<26}{v[-1]:>10.3f}{train:>12.3f}{slope:>12.3f}  "
              f"{'plateaued' if flat else 'still descending'}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
