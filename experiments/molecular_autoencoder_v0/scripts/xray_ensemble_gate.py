#!/usr/bin/env python3
"""Conformational resolution measured IN-DOMAIN, on crystal structures.

Why this exists
---------------
The NMR gate found a ~1.3 A reconstruction floor that is flat from latent 4 to
latent 16, against 0.75 A on X-ray novel folds. The natural reading is "domain
gap: NMR solution ensembles differ from the crystal-structure training set."
That reading has a fix attached (train on NMR conformers), and the fix is a
significant detour, so the diagnosis needs to be right first.

The problem with concluding "domain gap" from the NMR cohort alone is that it
confounds two different things:

  (a) the experimental method's effect on deposited coordinates, and
  (b) the fact that proteins SOLVED by NMR are a biased sample of protein
      space -- smaller, more flexible, frequently containing disordered
      regions, often things that do not crystallise at all.

Both get called "domain gap" but they are different problems, and neither is
the domain we actually care about: MD trajectories are conformers of whatever
protein you simulate, with coordinates from a force field. So the number we
need -- "can the codec resolve 1-2 A fluctuations of a protein around a
basin?" -- is not answered by either the NMR floor or the novel-fold number.

This script answers it without leaving the training domain. The X-ray corpus
already contains the same protein solved many times (different crystal forms,
ligands, space groups); those repeats ARE a conformational ensemble, drawn from
exactly the distribution the codec was trained on. Grouping by sequence and
measuring resolution within each group isolates "can it resolve conformers"
from "is NMR different."

Reading the result
------------------
floor ~0.75 A in-domain   -> the NMR 1.3 A really is a domain gap, the codec
                             resolves conformers fine in its own domain, and
                             the NMR-training detour is justified if we care
                             about NMR-like data
floor still ~1.3 A        -> NOT a domain gap. It is a genuine limit on
                             conformational resolution, training on NMR would
                             not fix it, and the fix belongs in the codec

Structures of the same protein rarely have identical atom sets (different loops
are resolved in different crystals), so members are compared on the INTERSECTION
of their (res_seq, atom_name) keys. Reconstruction still runs on each full
structure; only the comparison is restricted.

Usage:
    python scripts/xray_ensemble_gate.py --config configs/direct_d8.yaml \
        --checkpoint <direct_d8 final.pt> --out outputs/xray_ensemble_gate.json
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from molae.config import ExperimentConfig  # noqa: E402
from molae.dataset import sample_from_arrays, collate_fn  # noqa: E402
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy, kabsch_transform_numpy  # noqa: E402
from molae import utils  # noqa: E402
from latent_suitability import ensemble_resolution, smoothness  # noqa: E402


def load_groups(processed_dir, min_members, min_common_atoms, allowed_keys=None):
    """Group processed structures by exact sequence; keep groups >= min_members.

    ``allowed_keys`` restricts membership to one split. This is NOT optional in
    practice: same-sequence entries are clustered into the SAME split, so an
    unfiltered run is dominated by training structures and reports a training
    reconstruction number. Comparing that against a held-out NMR figure
    conflates the domain gap with the train/test gap.
    """
    by_seq = collections.defaultdict(list)
    for f in sorted(Path(processed_dir).glob("*.npz")):
        if allowed_keys is not None and f.stem not in allowed_keys:
            continue
        d = np.load(f, allow_pickle=True)
        by_seq[str(d["sequence"])].append((f.stem, {k: d[k] for k in d.files}))

    groups = []
    for seq, members in by_seq.items():
        if len(members) < min_members:
            continue
        # Common atoms = intersection of (res_seq, atom_name_idx) keys.
        keysets = []
        for _, d in members:
            keysets.append({(int(r), int(a)) for r, a in
                            zip(d["res_seq"], d["atom_name_idx"])})
        common = set.intersection(*keysets)
        if len(common) < min_common_atoms:
            continue
        order = sorted(common)
        index_maps = []
        for _, d in members:
            pos = {(int(r), int(a)): i for i, (r, a) in
                   enumerate(zip(d["res_seq"], d["atom_name_idx"]))}
            index_maps.append(np.array([pos[k] for k in order], dtype=np.int64))
        groups.append({"sequence": seq, "members": members,
                       "index_maps": index_maps, "n_common": len(order)})
    return groups


@torch.no_grad()
def run_group(model, group, device):
    """Encode/decode each member in full; compare on the common atom subset."""
    zs, preds, trues = [], [], []
    for (_, d), idx in zip(group["members"], group["index_maps"]):
        batch = collate_fn([sample_from_arrays(d)])
        gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        z = model.encode(gb)
        pred = model.decode(z, gb)
        n = int(gb["mask"][0].sum().item())
        zs.append(z)
        preds.append(pred[0, :n].cpu().numpy().astype(np.float64)[idx])
        trues.append(gb["coords"][0, :n].cpu().numpy().astype(np.float64)[idx])

    # Superimpose truths (and each prediction onto its own truth's frame) so
    # crystal-frame differences do not masquerade as conformational change.
    ref = trues[0]
    for i in range(1, len(trues)):
        trues[i] = kabsch_transform_numpy(trues[i], ref)
    preds = [kabsch_transform_numpy(p, t) for p, t in zip(preds, trues)]
    return zs, preds, trues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed-dir", default=None,
                    help="defaults to the config's data.processed_dir")
    ap.add_argument("--out", default="outputs/xray_ensemble_gate.json")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="which split to draw group members from. Default val: "
                         "same-sequence entries cluster into ONE split, so an "
                         "unfiltered run reports a TRAINING reconstruction "
                         "number and is not comparable to held-out figures.")
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--min-common-atoms", type=int, default=100)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    processed = args.processed_dir or (ROOT / cfg.data.processed_dir)

    model = make_autoencoder(cfg.model).to(device)
    utils.load_checkpoint(args.checkpoint, model, None)
    model.eval()

    allowed = None
    if args.split != "all":
        splits = utils.load_json(ROOT / cfg.data.splits_file)
        allowed = set(splits[args.split])
    groups = load_groups(processed, args.min_members, args.min_common_atoms, allowed)
    print(f"[xray_gate] split={args.split}  {len(groups)} same-sequence groups with "
          f">={args.min_members} members and >={args.min_common_atoms} common atoms")
    if args.split == "all":
        print("  WARNING: split=all mixes training structures into the measurement; "
              "the result is NOT comparable to held-out numbers.")
    if not groups:
        raise SystemExit(
            "No same-sequence groups found. The corpus may have been "
            "deduplicated; try --min-common-atoms lower, or build a dataset "
            "that deliberately keeps repeat entries of the same protein.")

    rows, recon_all = [], []
    for g in groups:
        zs, preds, trues = run_group(model, g, device)
        rmsds = [kabsch_rmsd_numpy(p, t) for p, t in zip(preds, trues)]
        ratio, spread_p, spread_t = ensemble_resolution(preds, trues)
        rho, _ = smoothness(zs, trues)
        ids = [m[0] for m in g["members"]]
        rows.append({
            "ids": ids, "n_members": len(ids), "n_common_atoms": g["n_common"],
            "mean_recon_rmsd": float(np.mean(rmsds)),
            "spread_true_conformers": spread_t,
            "spread_reconstructions": spread_p,
            "resolution_ratio": ratio, "smoothness_spearman": rho,
        })
        recon_all.extend(rmsds)
        print(f"  {ids[0]}+{len(ids)-1}  {g['n_common']:4d} atoms  "
              f"recon {np.mean(rmsds):.3f} A  spread {spread_t:.3f} A  "
              f"resolution {ratio:.3f}")

    def m(key):
        v = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(v)) if v else float("nan")

    summary = {
        "split": args.split,
        "n_groups": len(rows), "n_structures": len(recon_all),
        "mean_recon_rmsd": float(np.mean(recon_all)),
        "mean_spread_true_conformers": m("spread_true_conformers"),
        "mean_resolution_ratio": m("resolution_ratio"),
        "mean_smoothness_spearman": m("smoothness_spearman"),
        "groups": rows,
    }
    utils.save_json(summary, ROOT / args.out)

    print(f"\n=== in-domain (crystal) conformational resolution [split={args.split}] ===")
    print(f"groups {summary['n_groups']}  structures {summary['n_structures']}")
    print(f"reconstruction         : {summary['mean_recon_rmsd']:.3f} A")
    print(f"true spread            : {summary['mean_spread_true_conformers']:.3f} A")
    print(f"resolution ratio       : {summary['mean_resolution_ratio']:.3f}")
    print("\nCompare against the NMR cohort's 1.3 A floor and the 0.75 A")
    print("novel-fold number:")
    print("  ~0.75 A here -> the NMR floor IS a domain gap; the codec resolves")
    print("     conformers fine in-domain.")
    print("  ~1.3 A here  -> NOT a domain gap; it is a real limit on")
    print("     conformational resolution and training on NMR will not fix it.")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
