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

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae.dataset import ProteinStructureDataset, collate_fn  # noqa: E402
from molae.model import MolecularAutoencoder  # noqa: E402,F401
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.losses import LossComputer  # noqa: E402
from molae.alignment import aligned_rmsd_torch  # noqa: E402
from molae import utils  # noqa: E402


def build_datasets(cfg):
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    processed = ROOT / cfg.data.processed_dir
    train_keys = list(splits["train"])
    if cfg.train.overfit:
        train_keys = train_keys + list(splits["val"])
    train_paths = [processed / f"{k}.npz" for k in train_keys]
    train_paths = [p for p in train_paths if p.exists()]
    return ProteinStructureDataset(train_paths), train_keys


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

    dataset, keys = build_datasets(cfg)
    if len(dataset) == 0:
        raise SystemExit("No processed structures found. Run prepare_dataset.py first.")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=(args.pin_memory and device.type == "cuda"))
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

    t0 = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        ep_comps, n_batches = {}, 0
        for batch in loader:
            gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if cfg.train.augment_rotation:
                gb["coords"] = random_rotate(gb["coords"])
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

        if epoch % cfg.train.log_every == 0 or epoch == cfg.train.epochs - 1:
            rmsd = quick_rmsd(model, loader, device)
            row = {"epoch": epoch, "rmsd": rmsd, "elapsed_s": time.time() - t0, **ep_comps}
            log.append(row)
            extra = " ".join(f"{k}={ep_comps[k]:.4f}" for k in
                             ("coord", "bond", "clash", "flow_mse") if k in ep_comps)
            print(f"  epoch {epoch:5d}  total={ep_comps.get('total', float('nan')):.4f}  "
                  f"{extra}  rmsd={rmsd:.3f}A")

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
