#!/usr/bin/env python3
"""Train the molecular autoencoder.

Resumable: if ``<out_dir>/latest.pt`` exists it continues from there. Saves
config, environment (versions + git commit), checkpoints, and a training log.

Usage:
    python scripts/train.py --config configs/stage_a_overfit.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae.dataset import ProteinStructureDataset, collate_fn  # noqa: E402
from molae.model import MolecularAutoencoder  # noqa: E402,F401
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.losses import LossComputer  # noqa: E402
from molae.alignment import aligned_rmsd_torch  # noqa: E402
from molae.curriculum import Curriculum, mixture_indices  # noqa: E402
from molae import utils  # noqa: E402


def build_datasets(cfg):
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    processed = ROOT / cfg.data.processed_dir
    train_keys = list(splits["train"])
    if cfg.train.overfit:
        train_keys = train_keys + list(splits["val"])
    train_paths = [processed / f"{k}.npz" for k in train_keys]
    train_paths = [p for p in train_paths if p.exists()]
    val_paths = [processed / f"{k}.npz" for k in splits.get("val", [])]
    val_paths = [p for p in val_paths if p.exists()]
    return (ProteinStructureDataset(train_paths), train_keys,
            ProteinStructureDataset(val_paths) if val_paths else None)


def build_tiered_datasets(cfg):
    """Multi-source training set with a per-tier index, for the curriculum.

    Each tier is a processed directory with its own splits file, so tiers can
    be built independently and mixed at train time. Returns
    ``(dataset, tier_index, val_dataset)`` where ``tier_index`` maps a tier
    name to the dataset positions belonging to it.

    Validation comes from the HIGHEST tier only. That is not a default, it is
    the point: a val set containing predicted structures reports how well the
    model learned the prediction distribution, and nothing in a loss curve
    distinguishes that from success.
    """
    tiers = dict(cfg.data.tiers)
    high = "experimental" if "experimental" in tiers else sorted(tiers)[-1]

    paths, tier_index, val_paths = [], {}, []
    for name, rel in tiers.items():
        d = ROOT / rel
        splits_file = d.parent / f"splits_{d.name.replace('processed_', '')}.json"
        if not splits_file.exists():
            splits_file = ROOT / cfg.data.splits_file
        splits = utils.load_json(splits_file)
        keys = list(splits["train"])
        start = len(paths)
        found = [d / f"{k}.npz" for k in keys]
        found = [p for p in found if p.exists()]
        paths.extend(found)
        tier_index[name] = np.arange(start, len(paths))
        if name == high:
            val_paths = [p for p in (d / f"{k}.npz" for k in splits.get("val", []))
                         if p.exists()]
        print(f"[train] tier {name!r}: {len(found)} structures from {rel}")

    if not val_paths:
        raise SystemExit(
            f"tier {high!r} produced no validation structures; held-out "
            "evaluation must come from the highest-quality tier")
    return (ProteinStructureDataset(paths), tier_index,
            ProteinStructureDataset(val_paths))


@torch.no_grad()
def quick_rmsd(model, loader, device, max_batches=4):
    model.eval()
    vals = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        preds, _ = model(gb)
        r = aligned_rmsd_torch(preds, gb["coords"], gb["mask"])
        vals.extend(r.tolist())
    model.train()
    return sum(vals) / max(len(vals), 1)


@torch.no_grad()
def heldout_rmsd(model, loader, device):
    """Mean aligned RMSD over the WHOLE val split -- deterministic.

    Distinct from quick_rmsd, which samples 4 random TRAIN batches (64 of 878+
    structures) and carries ~8% relative noise even when converged. A scaling
    curve read off that is unusable; this one is exact and cheap (forward only).
    """
    model.eval()
    aa, bb, n = 0.0, 0.0, 0
    for batch in loader:
        gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        preds, _ = model(gb)
        r = aligned_rmsd_torch(preds, gb["coords"], gb["mask"])
        aa += float(r.sum())
        n += r.numel()
    model.train()
    return aa / max(n, 1)


def corrupt_coords(coords, mask, frac, mode="zero", generator=None):
    """Corrupt a random subset of atoms' INPUT coordinates (target unchanged).

    A masked-reconstruction objective: the encoder never sees the true position
    of the corrupted atoms, so it cannot copy them through -- it has to infer
    them from the rest of the structure. This multiplies the effective training
    signal from a fixed set of structures, which matters because the matched
    experimental band contains only ~3,853 entries in the entire PDB.

    Only coordinates are corrupted, never identity: the decoder is conditioned
    on atom identity by design, so masking that would change the task rather
    than make it harder.
    """
    if frac <= 0.0:
        return coords
    keep = torch.rand(coords.shape[:2], device=coords.device) >= frac
    keep = keep | (mask < 0.5)          # never "corrupt" padding
    out = coords * keep.unsqueeze(-1)
    if mode == "noise":
        noise = torch.randn_like(coords) * coords.std().clamp_min(1e-3)
        out = out + noise * (~keep).unsqueeze(-1)
    return out


def log_eval_schedule(epoch, epochs, log_every, eval_every, has_val):
    """Decide independently whether this epoch logs train stats and/or evals val.

    These two schedules must NOT be nested. An earlier version evaluated the
    held-out set inside the ``log_every`` branch, which silently required an
    epoch to satisfy both, so any pair where neither divides the other (208 vs
    417, 42 vs 85) produced a held-out "curve" consisting of its two endpoints.
    Returns ``(do_log, do_eval)``.
    """
    last = epoch == epochs - 1
    do_log = epoch % log_every == 0 or last
    do_eval = bool(has_val) and eval_every > 0 and (epoch % eval_every == 0 or last)
    return do_log, do_eval


def resolve_device(name):
    if name in ("auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def random_rotate(coords):
    """Apply an independent random proper rotation (det=+1) per batch element.

    Kabsch alignment already makes the loss rotation-invariant, so this is pure
    input augmentation: it teaches non-equivariant encoders (baseline, rope) to
    be rotation-robust. A no-op in effect for the invariant encoder.
    """
    B = coords.shape[0]
    q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=coords.device))
    det = torch.linalg.det(q)
    q = torch.cat([q[:, :, :2], q[:, :, 2:] * det.view(B, 1, 1)], dim=2)  # force det=+1
    return torch.einsum("bni,bij->bnj", coords, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--amp", action="store_true", help="mixed precision (GPU; experimental)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = ROOT / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    utils.set_seed(cfg.train.seed)
    cfg.save(out_dir / "config.yaml")
    utils.save_json(utils.capture_environment(str(ROOT)), out_dir / "environment.json")

    tier_index, curriculum = None, None
    if cfg.data.tiers:
        dataset, tier_index, val_dataset = build_tiered_datasets(cfg)
        curriculum = Curriculum.from_spec(cfg.data.curriculum)
        keys = []
        print(f"[train] curriculum {cfg.data.curriculum or '(defaults)'}; "
              f"weights at start {curriculum.weights_at(0.0)}, "
              f"end {curriculum.weights_at(1.0)}")
    else:
        dataset, keys, val_dataset = build_datasets(cfg)
    if len(dataset) == 0:
        raise SystemExit("No processed structures found. Run prepare_dataset.py first.")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=(args.pin_memory and device.type == "cuda"))
    val_loader = (DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)
                  if val_dataset is not None and len(val_dataset) else None)
    # A FIXED, deterministic slice of train, evaluated exactly like val.
    #
    # quick_rmsd samples 4 random batches -- 32 structures at batch 8 -- and
    # its own docstring says a curve read off it is unusable. Comparing that
    # against a full deterministic val pass cannot resolve a train/val gap,
    # which is the whole question when distinguishing underfitting from
    # capacity. Same size as val so the two numbers are like for like.
    n_te = min(len(dataset), len(val_dataset) if val_dataset is not None else 128)
    train_eval_loader = DataLoader(
        Subset(dataset, list(range(n_te))), batch_size=cfg.train.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    print(f"[train] {len(dataset)} structures on {device} (amp={use_amp}); "
          f"{'first ids: ' + str(keys[:8]) if len(keys) > 8 else keys}")

    model = make_autoencoder(cfg.model).to(device)
    print(f"[train] model params: {model.num_parameters():,}  latent floats: {model.latent_floats}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = LossComputer(cfg.loss, clash_dist=cfg.train.clash_dist, max_atoms=cfg.train.loss_max_atoms)

    start_epoch = 0
    log = []
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = utils.load_checkpoint(latest, model, opt)
        start_epoch = ckpt["epoch"] + 1
        log = ckpt["extra"].get("log", [])
        # Restore RNG *after* set_seed above, so a resumed run continues the
        # same random stream instead of restarting it from the base seed.
        if ckpt.get("rng_torch") is not None:
            torch.set_rng_state(ckpt["rng_torch"].to("cpu", torch.uint8))
        if ckpt.get("rng_numpy") is not None:
            import numpy as _np
            _np.random.set_state(ckpt["rng_numpy"])
        print(f"[train] resumed from epoch {start_epoch}")

    steps_per_epoch = max(1, -(-len(dataset) // cfg.train.batch_size))
    t0 = time.time()
    rng = np.random.default_rng(cfg.train.seed)
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        # Curriculum: re-draw the epoch's sample at the scheduled tier
        # proportions. Progress is fraction of TRAINING done, so the schedule
        # is expressed once and holds whatever the epoch budget is.
        epoch_loader = loader
        if curriculum is not None:
            progress = epoch / max(cfg.train.epochs - 1, 1)
            idx = mixture_indices(curriculum, tier_index, progress,
                                  len(dataset), rng)
            epoch_loader = DataLoader(
                Subset(dataset, idx.tolist()), batch_size=cfg.train.batch_size,
                shuffle=True, collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=(args.pin_memory and device.type == "cuda"))
        ep_comps, n_batches = {}, 0
        for batch in epoch_loader:
            gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if cfg.train.augment_rotation:
                gb["coords"] = random_rotate(gb["coords"])
            # Masked reconstruction: the model sees corrupted coordinates, the
            # loss is scored against the CLEAN ones. Two separate batch dicts so
            # the target cannot be contaminated -- models read batch["coords"],
            # so corrupting in place would corrupt the target too.
            model_batch = gb
            if getattr(cfg.train, "corrupt_frac", 0.0) > 0.0:
                model_batch = dict(gb)
                model_batch["coords"] = corrupt_coords(
                    gb["coords"], gb["mask"], cfg.train.corrupt_frac,
                    getattr(cfg.train, "corrupt_mode", "zero"))
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if hasattr(model, "training_loss"):      # flow-matching objective
                    loss, comp = model.training_loss(gb)
                else:
                    preds, _ = model(gb)
                    loss, comp = loss_fn(preds, gb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in comp.items():
                ep_comps[k] = ep_comps.get(k, 0.0) + v
            n_batches += 1
        for k in ep_comps:
            ep_comps[k] /= max(n_batches, 1)

        do_log, do_eval = log_eval_schedule(
            epoch, cfg.train.epochs, cfg.train.log_every, cfg.train.eval_every,
            val_loader is not None)
        if do_log or do_eval:
            rmsd = quick_rmsd(model, loader, device) if do_log else float("nan")
            row = {"epoch": epoch, "rmsd": rmsd, "elapsed_s": time.time() - t0,
                   "steps": (epoch + 1) * steps_per_epoch, **ep_comps}
            if curriculum is not None:
                # Record the mixture actually used, so a curve can be read
                # against the schedule rather than against an assumption.
                row["tier_weights"] = curriculum.weights_at(
                    epoch / max(cfg.train.epochs - 1, 1))
            # Held-out every eval_every: gives a SCALING CURVE vs steps from a
            # single run, instead of needing one run per step budget.
            if do_eval:
                row["val_rmsd"] = heldout_rmsd(model, val_loader, device)
                # Deterministic and same-sized, so train_rmsd vs val_rmsd is a
                # real comparison rather than a sample-noise difference.
                row["train_rmsd"] = heldout_rmsd(model, train_eval_loader, device)
                row["val_over_train"] = row["val_rmsd"] / max(row["train_rmsd"], 1e-6)
            log.append(row)
            if do_log:
                extra = " ".join(f"{k}={ep_comps[k]:.4f}" for k in
                                 ("coord", "bond", "clash", "flow_mse") if k in ep_comps)
                print(f"  epoch {epoch:5d}  total={ep_comps.get('total', float('nan')):.4f}  "
                      f"{extra}  rmsd={rmsd:.3f}A"
                      + (f"  VAL={row['val_rmsd']:.3f}A" if "val_rmsd" in row else "")
                      + (f"  TRAIN={row['train_rmsd']:.3f}A"
                         f"  v/t={row['val_over_train']:.2f}"
                         if "train_rmsd" in row else ""))
            elif "val_rmsd" in row:
                print(f"  epoch {epoch:5d}  VAL={row['val_rmsd']:.3f}A"
                      + (f"  TRAIN={row['train_rmsd']:.3f}A"
                         f"  v/t={row['val_over_train']:.2f}"
                         if "train_rmsd" in row else ""))

        if epoch % cfg.train.ckpt_every == 0 or epoch == cfg.train.epochs - 1:
            utils.save_checkpoint(latest, model, opt, epoch, extra={"log": log})

    utils.save_checkpoint(out_dir / "final.pt", model, opt, cfg.train.epochs - 1,
                          extra={"log": log})
    utils.save_json({"log": log, "train_seconds": time.time() - t0,
                     "n_structures": len(dataset), "keys": keys},
                    out_dir / "train_log.json")
    print(f"[train] done in {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
