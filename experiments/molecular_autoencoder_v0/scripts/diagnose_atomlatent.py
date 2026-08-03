#!/usr/bin/env python3
"""Where does the atom-latent architecture fail? Instrumentation, not RMSD.

RMSD says whether it works. It does not say why. At matched budget the shared
latent already has MORE capacity than the per-residue reference (4,096 scalars
against ~1,600 for a 200-residue structure) and reconstructs about twice as
badly, so the failure is not compression -- it is how the latent gets used.
This measures that directly.

Five questions, each with a failure signature:

  ANCHOR COVERAGE     do the L anchors spread over the structure, or collapse?
      Collapsed anchors make locality meaningless: every atom routes to the
      same handful of latents and the k-nearest selection carries no
      information. Signature: anchor spread << structure spread, and mean
      atom-to-nearest-anchor distance comparable to the radius of gyration.

  LATENT UTILISATION  do all L tokens carry information, or are most dead?
      A latent whose effective rank is far below L is not using the budget it
      was given, which would explain "more capacity, worse result" exactly.
      Signature: effective rank << L, or a long tail of near-zero-variance
      tokens.

  ROUTING LOAD        how many atoms does each latent serve?
      If a few latents absorb most atoms, the rest are wasted and the model is
      effectively running at a much smaller L. Signature: high Gini, low
      fraction of latents used.

  LOCALITY PAYOFF     does pass 2 (local) actually improve on pass 1 (global)?
      If not, the locality machinery is inert and arm C reduces to arm B.
      Signature: coarse and fine RMSD within noise of each other.

  ERROR vs DISTANCE   is per-atom error worse for atoms far from any anchor?
      This is the direct test of whether anchor placement is the bottleneck.
      Signature: strong positive correlation.

Usage:
    python scripts/diagnose_atomlatent.py --config configs/atomlatent_base.yaml \\
        --checkpoint outputs/.../final_ema.pt --limit 50
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
from molae.dataset import collate_fn, sample_from_arrays, add_graph_features  # noqa: E402
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.alignment import kabsch_align_torch  # noqa: E402
from molae.model_atomlatent import topk_latents  # noqa: E402
from molae import utils  # noqa: E402


def gini(x):
    """0 = every latent serves equally many atoms, 1 = one latent serves all."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def effective_rank(z):
    """exp(entropy of the normalised singular spectrum). L if all tokens are
    independent and equally used; far below L if the latent has collapsed."""
    s = torch.linalg.svdvals(z.double())
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def diagnose_one(model, batch, cfg):
    out = {}
    z = model.encode(batch)                                   # (1, L, d)
    L = z.shape[1]
    anchors = z[..., :3] * cfg.model.coord_scale              # (1, L, 3)
    coords = batch["coords"]
    m = batch["mask"][0] > 0.5
    real = coords[0][m]

    # --- anchor coverage -------------------------------------------------
    rg = float(torch.linalg.norm(real - real.mean(0), dim=-1).mean())
    a = anchors[0]
    out["anchor_spread"] = float(torch.linalg.norm(a - a.mean(0), dim=-1).mean())
    out["structure_rg"] = rg
    out["anchor_spread_ratio"] = out["anchor_spread"] / max(rg, 1e-6)
    d_to_anchor = torch.cdist(real, a).min(dim=1).values
    out["mean_dist_to_nearest_anchor"] = float(d_to_anchor.mean())
    out["dist_to_anchor_over_rg"] = float(d_to_anchor.mean()) / max(rg, 1e-6)

    # --- latent utilisation ---------------------------------------------
    content = z[0, :, 3:]
    out["latent_L"] = L
    out["effective_rank"] = effective_rank(content)
    out["effective_rank_frac"] = out["effective_rank"] / L
    var = content.var(dim=1)
    out["dead_token_frac"] = float((var < 0.01 * var.mean()).float().mean())

    # --- routing load ----------------------------------------------------
    fine, coarse = model.decoder(z, batch)
    if getattr(model.decoder, "local", False):
        idx = topk_latents(coarse.detach(), anchors, cfg.model.dec_local_k)[0][m]
        counts = torch.bincount(idx.reshape(-1), minlength=L).float()
        out["routing_gini"] = gini(counts.cpu().numpy())
        out["latents_used_frac"] = float((counts > 0).float().mean())
    else:
        out["routing_gini"] = float("nan")
        out["latents_used_frac"] = float("nan")

    # --- locality payoff -------------------------------------------------
    def rmsd(p):
        al = kabsch_align_torch(p, coords, batch["mask"])[0][m]
        return float(torch.sqrt(((al - real) ** 2).sum(-1).mean()))

    out["rmsd_coarse"] = rmsd(coarse)
    out["rmsd_fine"] = rmsd(fine)
    out["locality_gain"] = out["rmsd_coarse"] - out["rmsd_fine"]

    # --- error vs distance to anchor -------------------------------------
    al = kabsch_align_torch(fine, coords, batch["mask"])[0][m]
    err = torch.linalg.norm(al - real, dim=-1)
    if err.numel() > 2:
        e = err - err.mean()
        dd = d_to_anchor - d_to_anchor.mean()
        denom = (e.norm() * dd.norm()).clamp_min(1e-9)
        out["corr_err_vs_anchor_dist"] = float((e * dd).sum() / denom)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="outputs/atomlatent_diagnosis.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    processed = Path(args.processed_dir or (ROOT / cfg.data.processed_dir))
    keys = utils.load_json(ROOT / cfg.data.splits_file)[args.split][:args.limit]
    graph = getattr(cfg.model, "atom_addressing", "group") == "graph"

    model = make_autoencoder(cfg.model)
    utils.load_checkpoint(args.checkpoint, model, None)
    model.eval()

    rows = []
    for k in keys:
        f = processed / f"{k}.npz"
        if not f.exists():
            continue
        s = sample_from_arrays(np.load(f, allow_pickle=True))
        if graph:
            s = add_graph_features(s, key=str(f))
        r = diagnose_one(model, collate_fn([s]), cfg)
        r["pdb_id"] = k
        rows.append(r)

    if not rows:
        raise SystemExit("no structures found")

    keys_num = [k for k in rows[0] if isinstance(rows[0][k], (int, float))
                and not isinstance(rows[0][k], bool)]
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in keys_num}

    print(f"\n=== atom-latent diagnosis over {len(rows)} structures ===\n")
    print("ANCHOR COVERAGE   (collapse -> locality is meaningless)")
    print(f"  anchor spread / structure Rg      {summary['anchor_spread_ratio']:.3f}"
          "     (<<1 = collapsed)")
    print(f"  mean atom->nearest anchor / Rg    {summary['dist_to_anchor_over_rg']:.3f}"
          "     (~1 = anchors are not near the atoms)")
    print("\nLATENT UTILISATION  (dead tokens -> budget unused)")
    print(f"  effective rank                    {summary['effective_rank']:.1f}"
          f" of {int(summary['latent_L'])}  ({summary['effective_rank_frac']:.2f})")
    print(f"  near-zero-variance token fraction {summary['dead_token_frac']:.3f}")
    print("\nROUTING LOAD        (concentration -> effective L is smaller than L)")
    print(f"  gini over latents                 {summary['routing_gini']:.3f}"
          "     (0 = even, 1 = one latent serves all)")
    print(f"  latents receiving any atom        {summary['latents_used_frac']:.3f}")
    print("\nLOCALITY PAYOFF     (no gain -> arm C reduces to arm B)")
    print(f"  RMSD pass 1 (global)              {summary['rmsd_coarse']:.3f} A")
    print(f"  RMSD pass 2 (local)               {summary['rmsd_fine']:.3f} A")
    print(f"  gain from locality                {summary['locality_gain']:+.3f} A")
    print("\nERROR vs ANCHOR DISTANCE  (the direct anchor-placement test)")
    print(f"  corr(per-atom error, dist to nearest anchor)  "
          f"{summary.get('corr_err_vs_anchor_dist', float('nan')):+.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    utils.save_json({"summary": summary, "rows": rows}, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
