"""Reproducibility helpers: seeding, environment capture, checkpoint I/O."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def git_commit(repo_dir: str = ".") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def capture_environment(repo_dir: str = ".") -> dict:
    pkgs = {}
    for name in ("torch", "numpy", "scipy", "sklearn", "gemmi"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "packages": pkgs,
        "git_commit": git_commit(repo_dir),
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def grow_embedding_rows(state, model):
    """Pad saved embedding tables that are SHORTER than the model's.

    The element and residue vocabularies are append-only, but adding ligand
    symbols still changes ``N_ELEMENTS`` / ``N_RESIDUES`` and therefore the
    embedding shape, which would make every Milestone-1 checkpoint fail to
    load. Because new symbols are appended, rows 0..k-1 keep their meaning, so
    the saved table is copied verbatim and only the NEW rows are initialised
    (small random, matching nn.Embedding's N(0,1) scaled down).

    Only ever grows, and only for 2-D parameters whose trailing dim matches --
    a shrunk or otherwise mismatched tensor is left alone so load_state_dict
    still raises rather than silently loading the wrong weights.
    """
    tgt = model.state_dict()
    grown = []
    for k, saved in list(state.items()):
        cur = tgt.get(k)
        if cur is None or saved.ndim != 2 or cur.ndim != 2:
            continue
        if saved.shape[1] != cur.shape[1] or saved.shape[0] >= cur.shape[0]:
            continue
        new = torch.empty_like(cur)
        new.normal_(0.0, 0.02)
        new[: saved.shape[0]] = saved
        state[k] = new
        grown.append((k, tuple(saved.shape), tuple(cur.shape)))
    return grown


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    grown = grow_embedding_rows(ckpt["model"], model)
    if grown:
        for k, old, new in grown:
            print(f"[ckpt] grew {k} {old} -> {new} (vocabulary extended; "
                  f"existing rows preserved, new rows randomly initialised)")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
