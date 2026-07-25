#!/usr/bin/env python3
"""Latent-space geometry diagnostics for the TRAINED autoencoder.

Read-only analysis of a finished AE checkpoint: no training, no sampling, no
noise schedule, no trajectories. Relevant to Step 2 because these are the
properties latent diffusion would depend on, but nothing here builds one.

Milestone 1 asked "can we reconstruct?" and the answer is yes (0.75 A held-out).
But reconstruction accuracy is necessary and NOT sufficient for latent
diffusion. A denoising transformer runs *inside* the latent space, so it needs
two further properties that nothing so far has measured:

  1. it works on the Step-2/3 distribution -- conformers of the SAME molecule,
     not novel folds;
  2. the latent manifold is smooth -- nearby conformers map to nearby latents,
     and points BETWEEN two valid latents decode to physically valid structures.

Property 2 is load-bearing. Diffusion spends all its time at points that are
not exactly any training latent. If the space is pitted -- if the midpoint of
two good latents decodes to a tangle -- then no amount of reconstruction
accuracy will save the diffusion stage, and the fix (KL/VAE regularisation,
contrastive smoothing) belongs in the autoencoder, i.e. BEFORE Step 2 is built
on top of it. That is exactly the "fix it now or pay 10x later" lesson from the
decoder bug.

Data: NMR ensembles, free of MD compute. An NMR entry deposits 10-20 models of
the same molecule with identical atom composition -- a conformational ensemble
already sitting in the PDB. Our parser previously discarded all but model 0;
``parse_structure(..., model_index=k)`` now exposes them.

Three measurements
------------------
reconstruction   per-conformer RMSD. Should be AT LEAST as good as novel folds;
                 if it is worse, the codec is not suited to its own use case.
smoothness       Spearman correlation between pairwise latent distance and
                 pairwise structural RMSD across the ensemble. High = the
                 latent metric tracks the structural metric, which is what
                 makes a diffusion trajectory in latent space mean something.
                 SCORED AGAINST A RANDOM-INIT CONTROL, because an untrained
                 encoder is a random projection and random projections already
                 preserve distances (Johnson-Lindenstrauss) -- a smoke test on
                 untrained weights scored rho = 0.71. "Positive rho" is
                 therefore not evidence of anything; only rho clearly above the
                 random-init control is.
interpolation    decode the midpoint of two conformers' latents and score its
                 PHYSICS (bond error, chirality, clashes) against the two
                 endpoints. This is the direct test: if midpoints are as
                 physical as endpoints, the manifold is filled in. If they
                 degrade sharply, it is pitted and Step 2 will fail.

Usage:
    python scripts/latent_suitability.py --checkpoint outputs/direct_d8/final.pt \
        --config configs/direct_d8.yaml --pdb-list data/nmr_ids.txt
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
from molae.parsing import parse_structure, to_npz_dict, AllResiduesFiltered  # noqa: E402
from molae.dataset import sample_from_arrays, collate_fn  # noqa: E402
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy, kabsch_transform_numpy  # noqa: E402
from molae.metrics import build_topology_info, compute_all_metrics  # noqa: E402
from molae import utils  # noqa: E402


def load_ensemble(path, pdb_id, max_models, max_atoms, superimpose=True):
    """Parse every deposited model of one entry into a list of samples.

    Members are superimposed onto model 0 by default. This matters: our
    encoders are not rotation-invariant, so leaving the deposited frames alone
    would let a global rotation masquerade as a conformational difference and
    corrupt the smoothness measurement. Deposited NMR models are usually
    already superimposed, but that is a convention, not a guarantee.
    """
    first = parse_structure(path, pdb_id=pdb_id, model_index=0)
    if first is None:
        return []
    n_models = int(first.record.get("n_models", 1))
    if n_models < 2 or first.record["n_atoms"] > max_atoms:
        return []
    # Reject CA-only / backbone-only depositions. They perceive zero bonds and
    # zero stereocentres, so every physics metric is undefined for them, and
    # they are not the all-atom structures this codec is defined on. (1ANP is
    # one: 28 residues, 28 atoms.)
    if first.bonds.shape[0] == 0:
        return []
    out = []
    for k in range(min(n_models, max_models)):
        try:
            ps = parse_structure(path, pdb_id=pdb_id, model_index=k)
        except (IndexError, AllResiduesFiltered, ValueError):
            continue
        if ps is None:
            continue
        # Composition must match model 0 or the ensemble is not comparable.
        if ps.record["n_atoms"] != first.record["n_atoms"]:
            continue
        if superimpose and k > 0:
            ps.coords = kabsch_transform_numpy(
                ps.coords.astype(np.float64), first.coords.astype(np.float64)
            ).astype(np.float32)
        out.append(ps)
    return out


@torch.no_grad()
def encode_all(model, structures, device):
    """Return (latents list, predicted coords list, true coords list)."""
    zs, preds, trues = [], [], []
    for ps in structures:
        batch = collate_fn([sample_from_arrays(to_npz_dict(ps))])
        gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        z = model.encode(gb)
        pred = model.decode(z, gb)
        n = int(gb["mask"][0].sum().item())
        zs.append(z)
        preds.append(pred[0, :n].cpu().numpy().astype(np.float64))
        trues.append(gb["coords"][0, :n].cpu().numpy().astype(np.float64))
    return zs, preds, trues


def _ranks(x):
    """Average ranks, so tied values share a rank (plain argsort does not)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):                       # average ranks within each tie run
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return r


def spearman(a, b):
    """Rank correlation without a scipy dependency (tie-aware)."""
    ra, rb = _ranks(np.asarray(a, dtype=np.float64)), _ranks(np.asarray(b, dtype=np.float64))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def ensemble_resolution(preds, trues):
    """Does the decoder RESOLVE conformers, or collapse them onto one structure?

    THE decisive number for latent MD, and the one thing smoothness cannot tell
    you. Spearman rho is a RANK correlation and therefore scale-free: a codec
    that mapped every conformer of a molecule to nearly the same output, but
    ordered them correctly, would score rho ~ 1.0 and be completely useless for
    Step 2/3. Ranks survive collapse; magnitudes do not.

    So measure magnitude directly:

        ratio = mean pairwise RMSD among RECONSTRUCTIONS
              / mean pairwise RMSD among the TRUE conformers

      ~1.0  conformational differences survive the round trip
      ~0.0  every conformer decodes to the same structure (mean collapse) --
            fatal for latent MD regardless of how good single-structure
            reconstruction looks

    Also returns the true ensemble spread so reconstruction error can be read
    against it: if per-conformer RMSD is comparable to the spread, the codec's
    own error swamps the conformational signal it has to represent, which is
    just as fatal as collapse and equally invisible in a reconstruction number.
    """
    n = len(preds)
    pred_d, true_d = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if preds[i].shape != preds[j].shape:
                continue
            pred_d.append(kabsch_rmsd_numpy(preds[i], preds[j]))
            true_d.append(kabsch_rmsd_numpy(trues[i], trues[j]))
    if not true_d:
        return float("nan"), float("nan"), float("nan")
    mp, mt = float(np.mean(pred_d)), float(np.mean(true_d))
    return (mp / mt if mt > 1e-9 else float("nan")), mp, mt


def smoothness(zs, trues):
    """Spearman(pairwise latent distance, pairwise structural RMSD).

    Necessary but NOT sufficient, and blind to collapse -- see
    ``ensemble_resolution``, which must be read alongside it.

    Note the asymmetry this has to respect: structural distance is Kabsch RMSD
    (rotation-invariant) but our encoders are NOT rotation-invariant, so two
    conformers differing only by a global rotation would score zero structural
    distance and large latent distance. Ensemble members are therefore
    superimposed onto a common frame before encoding (see ``load_ensemble``),
    otherwise this metric measures deposition frame as much as conformation.
    """
    n = len(zs)
    lat_d, str_d = [], []
    for i in range(n):
        for j in range(i + 1, n):
            zi, zj = zs[i], zs[j]
            if zi.shape != zj.shape:            # ragged; skip
                continue
            lat_d.append(float(torch.linalg.norm(zi - zj).item()))
            str_d.append(float(kabsch_rmsd_numpy(trues[i], trues[j])))
    if len(lat_d) < 3:
        return float("nan"), 0
    return spearman(np.array(lat_d), np.array(str_d)), len(lat_d)


@torch.no_grad()
def interpolation_physics(model, structures, zs, trues, device, max_pairs=10):
    """Decode midpoint latents; compare their physics to the endpoints'.

    A smooth (diffusible) latent gives midpoints whose bond error / chirality /
    clash rate are close to the endpoints'. A pitted latent gives midpoints that
    are geometrically plausible on average but chemically broken -- exactly the
    mean-collapse signature, and invisible to RMSD.

    A midpoint latent has no ground-truth structure, so the two target-relative
    metrics need a reference that does not depend on the conformer. Both have
    one: **bond lengths and chirality are conformer-invariant**. Covalent bonds
    are rigid on the scale that distinguishes NMR models of one molecule (they
    vary by ~0.01 A, against the ~0.1 A errors we are measuring), and a
    stereocentre never inverts between conformers of the same deposited entry.
    So an endpoint's *true* coordinates are a valid reference for the midpoint's
    bonds and handedness, while clash rate needs no reference at all.
    """
    rows = []
    n = len(structures)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:max_pairs]
    for i, j in pairs:
        ps = structures[i]
        batch = collate_fn([sample_from_arrays(to_npz_dict(ps))])
        gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if zs[i].shape != zs[j].shape:
            continue
        topo = build_topology_info(
            ps.atom_name_idx, ps.res_pos, ps.bonds, ps.element_symbol)
        n_at = int(gb["mask"][0].sum().item())
        ref = trues[i]                      # conformer-invariant reference

        def physics(z):
            pred = model.decode(z, gb)[0, :n_at].cpu().numpy().astype(np.float64)
            m = compute_all_metrics(pred, ref, topo, latent_floats=1)
            return (m["bond_length_error"], m["chirality_violation_rate"],
                    m["clashes_per_1000_atoms"])

        e_i, e_j = physics(zs[i]), physics(zs[j])
        mid = physics(0.5 * (zs[i] + zs[j]))
        endpoint = tuple((a + b) / 2 for a, b in zip(e_i, e_j))
        rows.append({
            "pair": [structures[i].pdb_id, i, j],
            "endpoint_bond_err": endpoint[0], "midpoint_bond_err": mid[0],
            "endpoint_chirality": endpoint[1], "midpoint_chirality": mid[1],
            "endpoint_clash_per_1k": endpoint[2], "midpoint_clash_per_1k": mid[2],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pdb-list", required=True, help="file of PDB ids (NMR entries)")
    ap.add_argument("--raw-dir", default="data/raw_nmr")
    ap.add_argument("--out", default="outputs/latent_suitability.json")
    ap.add_argument("--max-models", type=int, default=12)
    ap.add_argument("--max-entries", type=int, default=25)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    model = make_autoencoder(cfg.model).to(device)
    utils.load_checkpoint(args.checkpoint, model, None)
    model.eval()

    # Random-init control. An untrained encoder is a random projection, and
    # random projections already preserve pairwise distances well, so the
    # trained model's smoothness is only meaningful relative to this.
    utils.set_seed(cfg.train.seed)
    control = make_autoencoder(cfg.model).to(device).eval()

    raw_dir = ROOT / args.raw_dir
    ids = [ln.strip() for ln in open(args.pdb_list) if ln.strip()][: args.max_entries]

    entries, recon, smooth_rows, ctrl_rows, res_rows, interp = [], [], [], [], [], []
    for pid in ids:
        path = raw_dir / f"{pid}.cif"
        if not path.exists():
            continue
        try:
            structs = load_ensemble(str(path), pid, args.max_models,
                                    cfg.data.max_atoms)
        except Exception as exc:                                # noqa: BLE001
            print(f"  [skip] {pid}: {exc}")
            continue
        if len(structs) < 3:
            continue

        zs, preds, trues = encode_all(model, structs, device)
        rmsds = [kabsch_rmsd_numpy(p, t) for p, t in zip(preds, trues)]
        rho, n_pairs = smoothness(zs, trues)
        res_ratio, spread_pred, spread_true = ensemble_resolution(preds, trues)
        rows = interpolation_physics(model, structs, zs, trues, device)

        zs_c, _, _ = encode_all(control, structs, device)
        rho_c, _ = smoothness(zs_c, trues)

        entries.append({
            "pdb_id": pid, "n_models": len(structs),
            "n_residues": structs[0].record["n_residues"],
            "n_atoms": structs[0].record["n_atoms"],
            "mean_recon_rmsd": float(np.mean(rmsds)),
            "smoothness_spearman": rho,
            "smoothness_spearman_random_init": rho_c, "n_pairs": n_pairs,
            "resolution_ratio": res_ratio,
            "spread_reconstructions": spread_pred,
            "spread_true_conformers": spread_true,
        })
        recon.extend(rmsds)
        if not np.isnan(rho):
            smooth_rows.append(rho)
        if not np.isnan(rho_c):
            ctrl_rows.append(rho_c)
        if not np.isnan(res_ratio):
            res_rows.append((res_ratio, spread_pred, spread_true))
        interp.extend(rows)
        print(f"  {pid}: {len(structs)} models ({structs[0].record['n_residues']} res), "
              f"recon {np.mean(rmsds):.3f} A vs spread {spread_true:.3f} A, "
              f"resolution {res_ratio:.2f}, rho={rho:.3f} (ctrl {rho_c:.3f})")

    def mean(key, rows):
        """NaN-safe: a structure with no bonds or no stereocentres yields NaN
        for that metric and must not poison the aggregate."""
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_entries": len(entries),
        "n_conformers": len(recon),
        "mean_recon_rmsd": float(np.mean(recon)) if recon else float("nan"),
        "mean_smoothness_spearman": float(np.mean(smooth_rows)) if smooth_rows else float("nan"),
        "mean_smoothness_spearman_random_init": (
            float(np.mean(ctrl_rows)) if ctrl_rows else float("nan")),
        "mean_resolution_ratio": (
            float(np.mean([r[0] for r in res_rows])) if res_rows else float("nan")),
        "mean_spread_reconstructions": (
            float(np.mean([r[1] for r in res_rows])) if res_rows else float("nan")),
        "mean_spread_true_conformers": (
            float(np.mean([r[2] for r in res_rows])) if res_rows else float("nan")),
        "interpolation": {
            "n_pairs": len(interp),
            "endpoint_bond_err": mean("endpoint_bond_err", interp),
            "midpoint_bond_err": mean("midpoint_bond_err", interp),
            "endpoint_chirality": mean("endpoint_chirality", interp),
            "midpoint_chirality": mean("midpoint_chirality", interp),
            "endpoint_clash_per_1k": mean("endpoint_clash_per_1k", interp),
            "midpoint_clash_per_1k": mean("midpoint_clash_per_1k", interp),
        },
        "entries": entries,
        "interpolation_rows": interp,
    }
    out = ROOT / args.out
    utils.save_json(summary, out)

    ip = summary["interpolation"]
    print("\n=== latent suitability for diffusion ===")
    print(f"entries {summary['n_entries']}  conformers {summary['n_conformers']}")
    print(f"reconstruction on ensembles : {summary['mean_recon_rmsd']:.3f} A")
    print(f"true conformational spread  : {summary['mean_spread_true_conformers']:.3f} A"
          f"   <- recon error must be WELL BELOW this")
    print(f"reconstruction spread       : {summary['mean_spread_reconstructions']:.3f} A")
    print(f"RESOLUTION RATIO            : {summary['mean_resolution_ratio']:.3f}"
          f"   (1.0 = conformers survive, 0.0 = collapsed)")
    rho_t = summary["mean_smoothness_spearman"]
    rho_c = summary["mean_smoothness_spearman_random_init"]
    print(f"latent/structure smoothness : rho = {rho_t:.3f}  "
          f"(random-init control {rho_c:.3f}, delta {rho_t - rho_c:+.3f})")
    print(f"interp bond error  endpoint {ip['endpoint_bond_err']:.3f} -> midpoint {ip['midpoint_bond_err']:.3f} A")
    print(f"interp chirality   endpoint {ip['endpoint_chirality']:.3f} -> midpoint {ip['midpoint_chirality']:.3f}")
    print(f"interp clashes/1k  endpoint {ip['endpoint_clash_per_1k']:.1f} -> midpoint {ip['midpoint_clash_per_1k']:.1f}")
    print("\nGATE, in order of decisiveness:")
    print(" 1. RESOLUTION RATIO near 1.0, and recon error well below the true")
    print("    spread. If the codec cannot tell conformers apart, nothing else")
    print("    matters -- rho is rank-based and stays high under collapse.")
    print(" 2. Midpoints close to endpoints on ALL THREE physics rows.")
    print(" 3. rho clearly above the random-init control (a random projection")
    print("    already preserves distances, so bare rho means nothing).")
    print("Failing 1 or 2 means fixing the autoencoder BEFORE latent diffusion.")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
