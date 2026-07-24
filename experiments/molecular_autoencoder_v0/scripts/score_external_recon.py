#!/usr/bin/env python3
"""Score an EXTERNAL model's reconstructions on our leak-free held-out bar.

Model-agnostic: give it a directory of ground-truth PDBs and a directory of
reconstructed PDBs (matched by filename, e.g. 1ABC.pdb), and it reports the
same metrics + PCA/mean-shape baselines as our own eval.py — so any external
autoencoder (ProteinAE, etc.) is judged by exactly the same standard as our
models. Atoms are matched by (residue position, atom name), so it works for
backbone-only or all-atom reconstructions and only scores the common atoms.

Usage:
    # after running the external model to produce pred PDBs:
    python scripts/score_external_recon.py --config configs/stage_b_heldout.yaml \
        --true-dir outputs/external/true --pred-dir outputs/external/proteinae \
        --tag proteinae
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae.parsing import parse_structure, perceive_bonds  # noqa: E402
from molae.metrics import build_topology_info, compute_all_metrics  # noqa: E402
from molae.baselines import build_backbone_cohort, evaluate_baselines, structure_backbone  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy  # noqa: E402
from molae import utils  # noqa: E402


def parsed_to_arrays(ps):
    return {
        "coords": ps.coords.astype(np.float64),
        "atom_name_idx": ps.atom_name_idx,
        "res_pos": ps.res_pos,
        "element_symbol": list(ps.element_symbol),
    }


def match_common(true_ps, pred_ps):
    """Return (coords_true, coords_pred, atom_name_idx, res_pos, elements) on
    the atoms present in BOTH, keyed by (res_pos, atom_name_idx)."""
    def index(ps):
        return {(int(ps.res_pos[i]), int(ps.atom_name_idx[i])): i
                for i in range(ps.n_atoms)}
    ti, pi = index(true_ps), index(pred_ps)
    common = sorted(set(ti) & set(pi))
    it = [ti[k] for k in common]
    ip = [pi[k] for k in common]
    ct = true_ps.coords[it].astype(np.float64)
    cp = pred_ps.coords[ip].astype(np.float64)
    ani = true_ps.atom_name_idx[it]
    rp = true_ps.res_pos[it]
    # renumber res_pos to be contiguous 0..L-1 over the common set
    _, rp = np.unique(rp, return_inverse=True)
    el = [true_ps.element_symbol[i] for i in it]
    return ct, cp, ani.astype(np.int64), rp.astype(np.int64), el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--true-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--tag", default="external")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    true_dir, pred_dir = ROOT / args.true_dir, ROOT / args.pred_dir
    if not true_dir.is_absolute():
        true_dir = ROOT / args.true_dir

    per_structure, struct_dicts, rgs = [], [], []
    for tp in sorted(Path(true_dir).glob("*.pdb")):
        pid = tp.stem
        pp = Path(pred_dir) / f"{pid}.pdb"
        if not pp.exists():
            print(f"  [skip] no prediction for {pid}")
            continue
        true_ps = parse_structure(str(tp), pdb_id=pid)
        pred_ps = parse_structure(str(pp), pdb_id=pid)
        if true_ps is None or pred_ps is None:
            print(f"  [skip] unparseable {pid}")
            continue
        ct, cp, ani, rp, el = match_common(true_ps, pred_ps)
        if ct.shape[0] < 4:
            print(f"  [skip] <4 common atoms for {pid}")
            continue
        bonds = perceive_bonds(ct, el, rp)
        topo = build_topology_info(ani, rp, bonds, el)
        m = compute_all_metrics(cp, ct, topo)
        m["pdb_id"] = pid
        m["n_common_atoms"] = int(ct.shape[0])
        per_structure.append(m)
        struct_dicts.append({"coords": ct - ct.mean(0, keepdims=True),
                             "atom_name_idx": ani, "res_pos": rp, "pdb_id": pid})
        c = ct - ct.mean(0, keepdims=True)
        rgs.append(float(np.sqrt((c ** 2).sum(1).mean())))

    if not per_structure:
        raise SystemExit("No matched true/pred structures found.")

    def agg(k):
        v = [m[k] for m in per_structure if k in m and not np.isnan(m[k])]
        return float(np.mean(v)) if v else float("nan")

    keys = ["all_atom_rmsd", "backbone_rmsd", "pairwise_distance_error",
            "bond_length_error", "chirality_violation_rate",
            "clashes_per_1000_atoms", "contact_f1"]
    summary = {k: agg(k) for k in keys}

    # Leak-free cohort baselines: PCA fit on the TRAIN split, scored on these
    # (held-out) structures, aligned to a common frame — identical to eval.py.
    n_res = int(min(min(d["res_pos"].max() + 1 for d in struct_dicts),
                    cfg.data.min_residues))
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    processed = ROOT / cfg.data.processed_dir
    train_dicts = []
    for k in splits["train"]:
        p = processed / f"{k}.npz"
        if p.exists():
            dd = np.load(p, allow_pickle=True)
            c = dd["coords"].astype(np.float64)
            train_dicts.append({"coords": c - c.mean(0, keepdims=True),
                                "atom_name_idx": dd["atom_name_idx"],
                                "res_pos": dd["res_pos"], "pdb_id": str(dd["pdb_id"])})
    X_train, _, ref = build_backbone_cohort(train_dicts, n_res)
    X_test, _, _ = build_backbone_cohort(struct_dicts, n_res, ref=ref)
    baselines = (evaluate_baselines(X_train, X_test, n_res, [2, 4, 8, 16])
                 if X_train.shape[0] and X_test.shape[0] else {})

    out = {
        "tag": args.tag,
        "n_structures": len(per_structure),
        "external_model": summary,
        "trivial_all_atom_baselines": {
            "identity_all_atom_rmsd": 0.0,
            "centroid_all_atom_rmsd": float(np.mean(rgs)),
            "external_all_atom_rmsd": summary["all_atom_rmsd"],
        },
        "cohort": {"n_res": n_res, "baselines": baselines},
        "per_structure": per_structure,
    }
    out_dir = ROOT / "outputs" / "external"
    utils.save_json(out, out_dir / f"score_{args.tag}.json")

    lines = [f"# External model held-out score: {args.tag}", "",
             f"{len(per_structure)} matched held-out structures. "
             "Same leak-free metrics/baselines as eval.py.", "",
             "| metric | value |", "|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {summary[k]:.4g} |")
    lines += ["", "| backbone-cohort baseline | backbone RMSD (A) |", "|---|---|"]
    lines.append(f"| centroid (all-atom Rg) | {np.mean(rgs):.3g} |")
    for name, v in baselines.items():
        lines.append(f"| {name} | {v['mean_backbone_rmsd']:.3g} |")
    (out_dir / f"score_{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[score] wrote {out_dir/('score_'+args.tag+'.md')}")


if __name__ == "__main__":
    main()
