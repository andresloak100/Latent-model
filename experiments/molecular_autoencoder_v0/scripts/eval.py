#!/usr/bin/env python3
"""Evaluate the trained autoencoder and compare against classical baselines.

Reports per-structure geometry metrics for the learned model, a backbone-cohort
comparison against PCA / mean-shape / identity at matched latent budgets, timing
and memory, and writes example reconstructions as PDB.

Usage:
    python scripts/eval.py --config configs/stage_a_overfit.yaml
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae.dataset import ProteinStructureDataset, collate_fn  # noqa: E402
from molae.model import MolecularAutoencoder  # noqa: E402,F401
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.metrics import build_topology_info, compute_all_metrics  # noqa: E402
from molae.baselines import build_backbone_cohort, evaluate_baselines, structure_backbone  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy  # noqa: E402
from molae.pdb_io import write_pdb  # noqa: E402
from molae import utils  # noqa: E402


def eval_keys(cfg):
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    keys = list(splits["train"]) + list(splits["val"]) if cfg.train.overfit else list(splits["val"])
    if not keys:
        keys = list(splits["train"])
    return keys


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = ROOT / cfg.train.out_dir
    device = torch.device(args.device)
    utils.set_seed(cfg.train.seed)

    keys = eval_keys(cfg)
    processed = ROOT / cfg.data.processed_dir
    paths = [processed / f"{k}.npz" for k in keys if (processed / f"{k}.npz").exists()]
    dataset = ProteinStructureDataset(paths)
    print(f"[eval] {len(dataset)} structures")

    model = make_autoencoder(cfg.model).to(device)
    ckpt_path = Path(args.ckpt) if args.ckpt else (out_dir / "final.pt")
    if not ckpt_path.exists():
        ckpt_path = out_dir / "latest.pt"
    utils.load_checkpoint(ckpt_path, model, None, map_location=device)
    model.eval()
    print(f"[eval] loaded {ckpt_path.name}  ({model.num_parameters():,} params)")

    latent_floats = model.latent_floats
    per_structure = []
    recon_cache = {}
    struct_dicts = []
    infer_times = []

    examples_dir = out_dir / "reconstructions"
    n_saved = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            s = dataset[i]
            batch = collate_fn([s])
            gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            t0 = time.time()
            preds, z = model(gb)
            infer_times.append(time.time() - t0)
            pred = preds[0, : s["n_atoms"]].cpu().numpy().astype(np.float64)
            target = gb["coords"][0, : s["n_atoms"]].cpu().numpy().astype(np.float64)

            topo = build_topology_info(
                s["atom_name_idx"].numpy(), s["res_pos"].numpy(),
                s["bonds"].numpy(), s["element_symbol"],
            )
            n_res_i = int(s["res_pos"].numpy().max()) + 1
            # Per-atom latents scale with atoms, per-residue ones with residues,
            # fixed bottlenecks with neither. Getting this wrong silently
            # inflates the reported compression ratio (it once did, by ~60x).
            if hasattr(model, "latent_floats_for_atoms"):
                lf_i = model.latent_floats_for_atoms(int(s["n_atoms"]))
            elif hasattr(model, "latent_floats_for"):
                lf_i = model.latent_floats_for(n_res_i)
            else:
                lf_i = latent_floats
            m = compute_all_metrics(pred, target, topo, latent_floats=lf_i)
            m["pdb_id"] = s["pdb_id"]
            m["chain_id"] = s["chain_id"]
            m["n_residues"] = int(s["res_pos"].numpy().max()) + 1
            per_structure.append(m)
            recon_cache[s["pdb_id"]] = (pred, target)
            struct_dicts.append({
                "coords": target,
                "atom_name_idx": s["atom_name_idx"].numpy(),
                "res_pos": s["res_pos"].numpy(),
                "pdb_id": s["pdb_id"],
            })

            if n_saved < args.n_examples:
                write_pdb(examples_dir / f"{s['pdb_id']}_pred.pdb", pred,
                          s["residue_idx"].numpy(), s["atom_name_idx"].numpy(),
                          s["res_seq"].numpy() if "res_seq" in s else np.arange(s["n_atoms"]),
                          s["element_symbol"], chain_id="A")
                write_pdb(examples_dir / f"{s['pdb_id']}_true.pdb", target,
                          s["residue_idx"].numpy(), s["atom_name_idx"].numpy(),
                          s["res_seq"].numpy() if "res_seq" in s else np.arange(s["n_atoms"]),
                          s["element_symbol"], chain_id="A")
                n_saved += 1

    # Aggregate learned-model metrics.
    def agg(key):
        vals = [m[key] for m in per_structure if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    metric_keys = ["all_atom_rmsd", "backbone_rmsd", "pairwise_distance_error",
                   "bond_length_error", "chirality_violation_rate",
                   "clashes_per_1000_atoms", "contact_f1", "compression_ratio"]
    learned_summary = {k: agg(k) for k in metric_keys}

    # Trivial all-atom baselines (apples-to-apples for the full variable-size
    # setting, unlike the fixed backbone cohort used for PCA below).
    #   identity  -> predict truth (RMSD 0, no compression)
    #   centroid  -> predict every atom at the centroid; its aligned RMSD equals
    #                the radius of gyration (0 latent floats, maximal compression)
    rgs = []
    for _pid, (_pred, target) in recon_cache.items():
        c = target - target.mean(axis=0, keepdims=True)
        rgs.append(float(np.sqrt((c ** 2).sum(axis=1).mean())))
    trivial_all_atom = {
        "identity_all_atom_rmsd": 0.0,
        "centroid_all_atom_rmsd": float(np.mean(rgs)),
        "learned_all_atom_rmsd": learned_summary["all_atom_rmsd"],
    }

    # ---- Baseline comparison on a fixed-length backbone cohort ----
    n_res_cohort = min(d["res_pos"].max() + 1 for d in struct_dicts)
    n_res_cohort = int(min(n_res_cohort, cfg.data.min_residues))

    # Fit PCA / mean-shape on the TRAIN cohort and score on the EVAL cohort so
    # the comparison is not data-leaked. Both cohorts are aligned to the same
    # reference frame (required for a fitted PCA basis to transfer).
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    train_dicts = []
    for k in splits["train"]:
        p = processed / f"{k}.npz"
        if p.exists():
            dd = np.load(p, allow_pickle=True)
            c = dd["coords"].astype(np.float64)
            train_dicts.append({"coords": c - c.mean(0, keepdims=True),
                                "atom_name_idx": dd["atom_name_idx"],
                                "res_pos": dd["res_pos"], "pdb_id": str(dd["pdb_id"])})
    X_train, train_ids, ref = build_backbone_cohort(train_dicts, n_res_cohort)
    X_test, cohort_ids, _ = build_backbone_cohort(struct_dicts, n_res_cohort, ref=ref)
    k_list = [2, 4, 8, 16, 32, 64, latent_floats]
    baseline_results = (evaluate_baselines(X_train, X_test, n_res_cohort, k_list)
                        if X_train.shape[0] and X_test.shape[0] else {})
    ran_k = {int(v["latent_floats"]) for name, v in baseline_results.items() if name.startswith("pca_k")}
    skipped_k = [k for k in k_list if k not in ran_k and k > 0]
    X = X_test  # for downstream references

    # Learned model on the SAME cohort (backbone RMSD over first n_res residues).
    ae_cohort_rmsds = []
    for d in struct_dicts:
        pid = d["pdb_id"]
        pred, target = recon_cache[pid]
        bb_pred = structure_backbone(pred, d["atom_name_idx"], d["res_pos"], n_res_cohort)
        bb_true = structure_backbone(target, d["atom_name_idx"], d["res_pos"], n_res_cohort)
        if bb_pred is not None and bb_true is not None:
            ae_cohort_rmsds.append(kabsch_rmsd_numpy(bb_pred, bb_true))
    ae_cohort = {
        "mean_backbone_rmsd": float(np.mean(ae_cohort_rmsds)) if ae_cohort_rmsds else float("nan"),
        "latent_floats": latent_floats,
        "compression_ratio": float(12 * n_res_cohort) / float(latent_floats),
    }

    peak_gpu = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None
    summary = {
        "n_structures": len(dataset),
        "checkpoint": ckpt_path.name,
        "latent_floats": latent_floats,
        "learned_model": learned_summary,
        "trivial_all_atom_baselines": trivial_all_atom,
        "per_structure": per_structure,
        "cohort": {
            "n_res": n_res_cohort,
            "n_atoms": 4 * n_res_cohort,
            "eval_ids": cohort_ids,
            "pca_fit_ids": train_ids,
            "pca_fit_on_train_scored_on_eval": True,
            "pca_k_skipped_capped_by_n_train": skipped_k,
            "learned_autoencoder": ae_cohort,
            "baselines": baseline_results,
        },
        "timing": {
            "mean_inference_s_per_structure": float(np.mean(infer_times)),
            "n_structures_timed": len(infer_times),
        },
        "memory": {
            "peak_host_rss_mb": peak_rss_mb(),
            "peak_gpu_mb": peak_gpu,
            "gpu_note": "N/A (CPU-only run)" if peak_gpu is None else "CUDA",
        },
    }
    utils.save_json(summary, out_dir / "metrics.json")
    _write_tables(summary, out_dir / "metrics_table.md")
    print(f"[eval] wrote {out_dir/'metrics.json'} and metrics_table.md")
    print(f"[eval] learned all-atom RMSD: {learned_summary['all_atom_rmsd']:.3f} A  "
          f"backbone RMSD: {learned_summary['backbone_rmsd']:.3f} A")
    print(f"[eval] example reconstructions -> {examples_dir}")


def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if np.isnan(x):
            return "nan"
        if x == float("inf"):
            return "inf"
        return f"{x:.4g}"
    return str(x)


def _write_tables(summary, path):
    lm = summary["learned_model"]
    lines = ["# Metrics", "", "## Learned autoencoder (all-atom, per-structure mean)", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in lm.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    tb = summary.get("trivial_all_atom_baselines", {})
    lines += ["", "## All-atom trivial baselines (full variable-size setting)", "",
              "| method | all_atom_rmsd (A) |", "|---|---|"]
    lines.append(f"| identity (predict truth) | {_fmt(tb.get('identity_all_atom_rmsd'))} |")
    lines.append(f"| centroid (predict Rg) | {_fmt(tb.get('centroid_all_atom_rmsd'))} |")
    lines.append(f"| **learned autoencoder** | {_fmt(tb.get('learned_all_atom_rmsd'))} |")

    lines += ["", "## Per-structure (learned model)", ""]
    cols = ["pdb_id", "n_residues", "n_atoms", "all_atom_rmsd", "backbone_rmsd",
            "bond_length_error", "clashes_per_1000_atoms", "chirality_violation_rate",
            "contact_f1", "compression_ratio"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for m in summary["per_structure"]:
        lines.append("| " + " | ".join(_fmt(m.get(c)) for c in cols) + " |")

    coh = summary["cohort"]
    lines += ["", f"## Baseline comparison (fixed backbone cohort, "
              f"n_res={coh['n_res']}, {coh['n_atoms']} atoms)", "",
              "Backbone RMSD (A) vs latent-float budget; lower RMSD + higher "
              "compression is better.", "",
              "| method | latent_floats | compression_ratio | mean_backbone_rmsd |",
              "|---|---|---|---|"]
    ae = coh["learned_autoencoder"]
    lines.append(f"| **learned_autoencoder** | {_fmt(ae['latent_floats'])} | "
                 f"{_fmt(ae['compression_ratio'])} | {_fmt(ae['mean_backbone_rmsd'])} |")
    for name, r in coh["baselines"].items():
        lines.append(f"| {name} | {_fmt(r['latent_floats'])} | "
                     f"{_fmt(r['compression_ratio'])} | {_fmt(r['mean_backbone_rmsd'])} |")

    lines += ["", "## Timing & memory", "",
              f"- mean inference: {_fmt(summary['timing']['mean_inference_s_per_structure'])} s/structure",
              f"- peak host RSS: {_fmt(summary['memory']['peak_host_rss_mb'])} MB",
              f"- GPU memory: {summary['memory']['gpu_note']}"]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
