#!/usr/bin/env python3
"""Is the error INTERNAL geometry or ASSEMBLY placement?

Both complex_d8 arms plateau near 5 A with train ~= val, and 4x the parameters
buys nothing. That rules out capacity, optimisation and undertraining, and
leaves a representational limit — but "representational" is not a diagnosis.
This narrows it.

The hypothesis
--------------
Global RMSD superimposes the WHOLE structure once. If the model reconstructs
each chain (or domain) accurately but places them wrongly relative to each
other, a single global superposition cannot absorb that, and the reported
RMSD is dominated by the placement error while the local geometry is fine.

That fits everything observed: bb ~= aa everywhere (a global displacement
moves backbone and sidechains equally, whereas a local-geometry failure would
show aa > bb); error rising with size (more and larger parts to place); and
capacity-independence (placing two rigid bodies is not a parameter-hungry
problem).

Why the chain-count null result did not test it
-----------------------------------------------
corr(aa, n_chains) = 0.075 is a LINEAR correlation across 1-8 chains. If the
failure is inter-part placement it is a STEP function: two chains already give
you one placement to get wrong, and eight do not give proportionally more
error per atom. A linear correlation is nearly blind to a step. The right
comparison is 1 chain versus >=2, plus the decomposition below.

What this measures
------------------
For each structure:
  global_rmsd  -- one Kabsch superposition over everything (what we report)
  part_rmsd    -- each part superimposed INDEPENDENTLY, then averaged
                  (internal geometry only; placement error removed)
  placement    -- global_rmsd - part_rmsd

"Parts" are chains when there is more than one. For single-chain structures
the chain is split into contiguous halves, which stands in for domains — a
two-domain protein has the same placement problem with one chain, and that is
precisely the case a chain-count test cannot see.

Reading it:
  part_rmsd ~= global_rmsd  -> the error is genuinely local geometry. The
      representation is failing at reconstruction, and assembly is a red
      herring.
  part_rmsd << global_rmsd  -> the parts are right and the assembly is wrong.
      The fix is in how global placement is represented, not in the decoder's
      per-residue readout, and NOT in model size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae.dataset import ProteinStructureDataset, collate_fn, sample_from_arrays  # noqa: E402
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy  # noqa: E402
from molae import utils  # noqa: E402


def parts_of(chain_idx, n_min=30):
    """Index groups to superimpose independently.

    Chains when there are several. A single chain is split into contiguous
    halves as a domain stand-in: a two-domain protein has the same placement
    problem with one chain, and a chain-count test cannot see it.
    """
    uniq = sorted(set(int(c) for c in chain_idx))
    if len(uniq) > 1:
        parts = [np.where(chain_idx == c)[0] for c in uniq]
    else:
        n = len(chain_idx)
        parts = [np.arange(0, n // 2), np.arange(n // 2, n)]
    return [p for p in parts if len(p) >= n_min]


@torch.no_grad()
def decompose(model, sample, device):
    batch = collate_fn([sample])
    gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    pred = model.decode(model.encode(gb), gb)
    n = int(gb["mask"][0].sum().item())
    p = pred[0, :n].cpu().numpy().astype(np.float64)
    t = gb["coords"][0, :n].cpu().numpy().astype(np.float64)
    chain = sample["chain_idx"].numpy()[:n]

    glob = kabsch_rmsd_numpy(p, t)
    parts = parts_of(chain)
    if len(parts) < 2:
        return {"global_rmsd": glob, "part_rmsd": float("nan"),
                "placement": float("nan"), "n_parts": len(parts), "n_atoms": n}
    per = [kabsch_rmsd_numpy(p[idx], t[idx]) for idx in parts]
    weights = np.array([len(idx) for idx in parts], dtype=float)
    part = float((np.array(per) * weights).sum() / weights.sum())
    return {"global_rmsd": float(glob), "part_rmsd": part,
            "placement": float(glob - part), "n_parts": len(parts),
            "n_atoms": n, "n_chains": len(set(chain.tolist()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/error_decomposition.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    processed = Path(args.processed_dir or (ROOT / cfg.data.processed_dir))
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    keys = splits[args.split]
    if args.limit:
        keys = keys[:args.limit]

    model = make_autoencoder(cfg.model).to(device)
    utils.load_checkpoint(args.checkpoint, model, None)
    model.eval()

    rows = []
    for k in keys:
        f = processed / f"{k}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        r = decompose(model, sample_from_arrays(d), device)
        r["pdb_id"] = k
        rows.append(r)

    ok = [r for r in rows if not np.isnan(r["part_rmsd"])]
    if not ok:
        raise SystemExit("no structures with >=2 parts")

    g = float(np.mean([r["global_rmsd"] for r in ok]))
    p = float(np.mean([r["part_rmsd"] for r in ok]))
    single = [r for r in rows if r.get("n_chains", 1) == 1]
    multi = [r for r in rows if r.get("n_chains", 1) > 1]

    summary = {
        "n_structures": len(rows),
        "mean_global_rmsd": g,
        "mean_part_rmsd": p,
        "mean_placement_error": g - p,
        "placement_share": (g - p) / g if g > 0 else float("nan"),
        "single_chain_mean_rmsd": float(np.mean([r["global_rmsd"] for r in single]))
        if single else float("nan"),
        "multi_chain_mean_rmsd": float(np.mean([r["global_rmsd"] for r in multi]))
        if multi else float("nan"),
        "n_single_chain": len(single), "n_multi_chain": len(multi),
        "structures": rows,
    }
    utils.save_json(summary, ROOT / args.out)

    print(f"=== error decomposition ({len(rows)} structures, split={args.split}) ===")
    print(f"global RMSD (one superposition) : {g:.3f} A")
    print(f"per-part RMSD (each aligned)    : {p:.3f} A")
    print(f"placement error                 : {g - p:.3f} A "
          f"({100 * (g - p) / g:.0f}% of the total)")
    print(f"\nsingle-chain ({len(single):3d}) : "
          f"{summary['single_chain_mean_rmsd']:.3f} A")
    print(f"multi-chain  ({len(multi):3d}) : "
          f"{summary['multi_chain_mean_rmsd']:.3f} A")
    print("\n  part ~= global -> error is local geometry; assembly is a red herring")
    print("  part << global -> parts are right, assembly is wrong; the fix is in")
    print("     how global placement is represented, not in model size")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
