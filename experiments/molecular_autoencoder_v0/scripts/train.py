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
from molae.model import MolecularAutoencoder  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = ROOT / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    utils.set_seed(cfg.train.seed)
    cfg.save(out_dir / "config.yaml")
    utils.save_json(utils.capture_environment(str(ROOT)), out_dir / "environment.json")

    dataset, keys = build_datasets(cfg)
    if len(dataset) == 0:
        raise SystemExit("No processed structures found. Run prepare_dataset.py first.")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    print(f"[train] {len(dataset)} structures: {keys}")

    model = MolecularAutoencoder(cfg.model).to(device)
    print(f"[train] model params: {model.num_parameters():,}  latent floats: {model.latent_floats}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    loss_fn = LossComputer(cfg.loss, clash_dist=cfg.train.clash_dist)

    start_epoch = 0
    log = []
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = utils.load_checkpoint(latest, model, opt)
        start_epoch = ckpt["epoch"] + 1
        log = ckpt["extra"].get("log", [])
        print(f"[train] resumed from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        ep_comps, n_batches = {}, 0
        for batch in loader:
            gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            preds, _ = model(gb)
            loss, comp = loss_fn(preds, gb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            for k, v in comp.items():
                ep_comps[k] = ep_comps.get(k, 0.0) + v
            n_batches += 1
        for k in ep_comps:
            ep_comps[k] /= max(n_batches, 1)

        if epoch % cfg.train.log_every == 0 or epoch == cfg.train.epochs - 1:
            rmsd = quick_rmsd(model, loader, device)
            row = {"epoch": epoch, "rmsd": rmsd, "elapsed_s": time.time() - t0, **ep_comps}
            log.append(row)
            print(f"  epoch {epoch:5d}  total={ep_comps['total']:.4f}  "
                  f"coord={ep_comps['coord']:.4f}  bond={ep_comps['bond']:.4f}  "
                  f"clash={ep_comps['clash']:.4f}  rmsd={rmsd:.3f}A")

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
