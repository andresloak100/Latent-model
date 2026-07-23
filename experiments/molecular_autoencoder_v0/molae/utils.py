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


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
