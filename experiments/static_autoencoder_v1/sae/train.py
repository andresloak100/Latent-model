"""One training run: config in, metrics out. No chemistry in the loss -- coordinate MSE only."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

from .data import StructureBatch, center_and_scale, random_rotations
from .metrics import (masked_mse, rmsd_unaligned, effective_rank, active_dims,
                      latent_spectrum, interpolation_error, linear_probe)
from .model import AEConfig, StaticAutoencoder, n_params


@dataclass
class TrainConfig:
    steps: int = 2000
    batch: int = 32
    lr: float = 3e-4
    warmup: int = 100
    augment_rotation: bool = True
    coord_scale: float = 10.0
    seed: int = 0
    eval_batches: int = 8
    log_every: int = 500
    device: str = "cpu"
    # A cell whose loss still drops by more than this over its final fifth is STILL_IMPROVING.
    converged_tail: float = 0.05
    # A cell that explains less than this fraction of the coordinate variance never left the
    # do-nothing baseline and is STUCK, however flat its curve looks.
    min_frac_var_explained: float = 0.10


@dataclass
class RunResult:
    ae: dict
    train: dict
    n_params: int
    latent_size: int
    n_train: int
    n_heldout: int
    steps: int
    seconds: float
    # AN UNDER-TRAINED CELL IS INDISTINGUISHABLE FROM A SATURATED ONE, and that is the most
    # dangerous failure this harness can have. On the synthetic self-test at 400 steps every
    # latent size from 4 to 32 returned ho_mse 0.1154 -- a perfectly flat curve, which reads as
    # pre-registered reading 2, "saturates in latent size, capacity is not the bottleneck". The
    # same cells at 6000 steps separate cleanly. A cell still improving at the end is therefore
    # reported as UNCONVERGED and must not be scored.
    converged: bool
    status: str                      # "converged" | "still_improving" | "stuck"
    tail_improvement: float          # fractional loss drop over the last fifth of training
    null_mse: float                  # MSE of predicting the centroid -- the do-nothing baseline
    frac_var_explained: float        # 1 - heldout_mse / null_mse
    loss_curve: list
    train_mse: float
    heldout_mse: float
    heldout_rmsd_A: float
    gap: float
    effective_rank: float
    active_dims: int
    spectrum: list = field(default_factory=list)
    interp_mse: float = float("nan")
    probe_r2: float = float("nan")
    failed: str = ""

    def as_dict(self):
        return asdict(self)


def _draw(ds, idx, rng, tcfg, device):
    sel = rng.choice(idx, size=min(tcfg.batch, len(idx)), replace=False)
    b = ds.batch(sel).to(device)
    xyz = center_and_scale(b.xyz, b.mask, tcfg.coord_scale)
    if tcfg.augment_rotation:
        R = random_rotations(len(xyz), device=xyz.device)
        xyz = torch.bmm(xyz, R)
    return StructureBatch(xyz, b.elem, b.mask)


def run(ae_cfg: AEConfig, tcfg: TrainConfig, ds, split=None) -> RunResult:
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    tr_idx, ho_idx = split if split is not None else ds.split()
    dev = torch.device(tcfg.device)

    model = StaticAutoencoder(ae_cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), tcfg.lr)
    t0 = time.time()

    curve = []
    try:
        model.train()
        for st in range(1, tcfg.steps + 1):
            for g in opt.param_groups:
                g["lr"] = tcfg.lr * min(1.0, st / max(tcfg.warmup, 1))
            b = _draw(ds, tr_idx, rng, tcfg, dev)
            rec, _, _ = model(b.xyz, b.elem, b.mask)
            m = b.mask[..., None].to(rec.dtype)
            loss = ((rec - b.xyz) ** 2 * m).sum() / (m.sum() * 3).clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
            if st % max(tcfg.steps // 20, 1) == 0:
                curve.append((st, float(loss.detach())))
    except Exception as exc:
        return RunResult(asdict(ae_cfg), asdict(tcfg), n_params(model), ae_cfg.latent_size,
                         len(tr_idx), len(ho_idx), tcfg.steps, time.time() - t0,
                         False, "failed", float("nan"), float("nan"), float("nan"), curve,
                         *(float("nan"),) * 4, float("nan"), 0,
                         failed=f"{type(exc).__name__}: {exc}")

    # ---- evaluation ---------------------------------------------------------------------
    model.eval()
    def _eval(idx):
        mses, Zs, ys = [], [], []
        r = np.random.default_rng(tcfg.seed + 7)
        with torch.no_grad():
            for _ in range(tcfg.eval_batches):
                b = _draw(ds, idx, r, tcfg, dev)
                rec, z, _ = model(b.xyz, b.elem, b.mask)
                mses.append((masked_mse(rec, b.xyz, b.mask), rmsd_unaligned(rec, b.xyz, b.mask)))
                Zs.append(z.reshape(len(z), -1).cpu().numpy())
                # GENERIC probe target: the spatial extent of the point set. Not a chemical label.
                ys.append(b.xyz.pow(2).sum(-1).amax(-1).sqrt().cpu().numpy())
        return (float(np.mean([m[0] for m in mses])), float(np.mean([m[1] for m in mses])),
                np.concatenate(Zs), np.concatenate(ys))

    tr_mse, _, _, _ = _eval(tr_idx)
    ho_mse, ho_rmsd, Z, y = _eval(ho_idx)
    b = _draw(ds, ho_idx, np.random.default_rng(tcfg.seed + 11), tcfg, dev)

    # TWO conditions, because a flat curve alone means nothing. The first version of this guard
    # tested only "has the loss stopped falling", and on the self-test a 300-step run that never
    # left the mean-prediction regime reported tail=+0.1% and PASSED as converged, while a
    # 3000-step run that had learned most of the signal reported +20.7% and failed. A model that
    # never learns is flat too. So convergence requires BOTH a small tail AND having actually
    # moved off the do-nothing baseline.
    #
    # The baseline is the MSE of predicting the centroid, which after centring is zero -- i.e.
    # the variance of the data. It costs nothing and it turns the loss into a fraction.
    with torch.no_grad():
        nb = _draw(ds, ho_idx, np.random.default_rng(tcfg.seed + 13), tcfg, dev)
        nm = nb.mask[..., None].to(nb.xyz.dtype)
        null_mse = float((nb.xyz.pow(2) * nm).sum() / (nm.sum() * 3).clamp_min(1))
    fve = float(1.0 - ho_mse / null_mse) if null_mse > 0 else float("nan")

    tail = float("nan")
    if len(curve) >= 10:
        k = max(len(curve) // 5, 1)
        prev = np.mean([v for _, v in curve[-2 * k:-k]])
        last = np.mean([v for _, v in curve[-k:]])
        tail = float((prev - last) / prev) if prev > 0 else float("nan")

    if not np.isfinite(tail):
        status = "stuck"
    elif fve < tcfg.min_frac_var_explained:
        status = "stuck"
    elif tail >= tcfg.converged_tail:
        status = "still_improving"
    else:
        status = "converged"
    conv = status == "converged"

    return RunResult(
        ae=asdict(ae_cfg), train=asdict(tcfg), n_params=n_params(model),
        latent_size=ae_cfg.latent_size, n_train=len(tr_idx), n_heldout=len(ho_idx),
        steps=tcfg.steps, seconds=time.time() - t0,
        converged=conv, status=status, tail_improvement=tail,
        null_mse=null_mse, frac_var_explained=fve,
        loss_curve=[[int(s_), float(v)] for s_, v in curve],
        train_mse=tr_mse, heldout_mse=ho_mse,
        heldout_rmsd_A=ho_rmsd * tcfg.coord_scale,      # back into Angstrom
        gap=ho_mse - tr_mse,
        effective_rank=effective_rank(Z), active_dims=active_dims(Z),
        spectrum=[float(v) for v in latent_spectrum(Z)[:64]],
        interp_mse=interpolation_error(model, b, dev),
        probe_r2=linear_probe(Z, y),
    )
