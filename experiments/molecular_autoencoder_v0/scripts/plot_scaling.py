#!/usr/bin/env python3
"""Scaling figure: held-out RMSD vs data / compute / parameters.

Three slices through the same scaling surface, one line per model size --
the "contour" view. Numbers come from the committed summary tables
(capacity_ladder.md, data_scaling_2d.md) rather than being recomputed, so this
script is a plotter, not an experiment.

IMPORTANT, and annotated on the figure: the parameter slice was scored on
processed's val (293 structures) while the data and compute slices were scored
on processed_small's val (758). Absolute values are NOT comparable across
panels; slopes within a panel are.

Usage:
    python scripts/plot_scaling.py --out outputs/scaling.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- data (from outputs/cluster/{capacity_ladder,data_scaling_2d}.md) --------

# Panel A: held-out all-atom vs n_train, per model size. val = 758.
DATA_SLICE = {
    "1.1M":  [(450, 1.185), (878, 0.993), (2000, 0.858), (2272, 0.916)],
    "15.2M": [(450, 0.934), (878, 0.930), (2000, 0.837), (2272, 0.844)],
}
# Panel B: held-out all-atom vs optimiser steps at fixed n=2272. val = 758.
COMPUTE_SLICE = {
    "1.1M":  [(60_000, 0.916), (240_000, 0.863)],
    "15.2M": [(60_000, 0.844), (240_000, 0.752)],
}
# Panel C: held-out all-atom vs params at fixed n=878, lr-matched. val = 293.
PARAM_SLICE = [(1.104e6, 1.085), (3.814e6, 0.938), (6.973e6, 0.910), (15.227e6, 0.873)]

COLORS = {"1.1M": "#1f77b4", "15.2M": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/scaling.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Protein structure autoencoder — scaling slices (held-out all-atom RMSD)",
                 fontsize=13, y=1.00)

    ax = axes[0]
    for label, pts in DATA_SLICE.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=COLORS[label], label=f"{label} params", lw=2, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("unique training structures")
    ax.set_ylabel("held-out all-atom RMSD (Å)")
    ax.set_title("A. data  (matched 60k steps, val=758)")
    ax.axvspan(2272, 3853, alpha=0.12, color="grey")
    ax.annotate("PDB band ceiling\n(~3,853 total)", xy=(3000, 1.10), fontsize=8,
                ha="center", color="dimgrey")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, pts in COMPUTE_SLICE.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=COLORS[label], label=f"{label} params", lw=2, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("optimiser steps")
    ax.set_title("B. compute  (fixed n=2272, val=758)")
    ax.annotate("still descending —\nno capacity floor", xy=(1.3e5, 0.80),
                fontsize=8, color="dimgrey")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    xs, ys = zip(*PARAM_SLICE)
    ax.plot(xs, ys, "s-", color="#2ca02c", lw=2, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("model parameters")
    ax.set_title("C. capacity  (fixed n=878, val=293)")
    ax.annotate("−0.056 Å / doubling", xy=(3e6, 1.02), fontsize=9, color="dimgrey")
    ax.grid(alpha=0.3)

    for a in axes:
        a.set_ylim(0.70, 1.25)

    fig.text(0.5, -0.06,
             "Panels A/B scored on processed_small val (758 structures); panel C on processed val (293). "
             "Absolute values are NOT comparable across panels — slopes within a panel are.\n"
             "Panel A rungs are matched on optimiser steps, so high-n cells received fewer epochs and are "
             "under-converged: the data slope shown is a LOWER BOUND.",
             ha="center", fontsize=8, color="dimgrey")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
