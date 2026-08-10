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
    # INBOX 73a/74b: written through a temp file and os.replace, which is atomic on
    # one filesystem. A kill landing inside a direct write leaves a truncated file
    # at the destination; this leaves either the old complete file or the new one.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    """INBOX 73a. Atomic, with one generation of fallback kept alongside.

    WHY. On a preemptible partition the kill can land *inside* the write. A bare
    ``torch.save`` onto ``latest.pt`` therefore leaves a truncated file that still
    satisfies ``.exists()``, and ``train.py``'s resume branch keys on exactly that.
    The requeued job then either dies on load every time -- a requeue loop that
    reads as a scheduler problem rather than a corrupt file -- or resumes from
    something that is not what the log names.

    ``os.replace`` is atomic on one filesystem, so the destination is always either
    the previous complete checkpoint or the new complete one and never a partial.
    The previous generation is kept as ``.prev`` because the write *before* this one
    could already have been bad, and because the rename leaves a brief window in
    which the destination does not exist -- ``resume_checkpoint_path`` below covers
    both cases.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "extra": extra or {},
        # INBOX 73b: a requeued run is a second training under the first one's name,
        # and --open-mode=append hides that in the log. Stamped here so the artefact
        # carries how many times its producer died.
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    if path.exists():
        os.replace(path, path.with_name(path.name + ".prev"))
    os.replace(tmp, path)


def resume_checkpoint_path(path):
    """INBOX 73a. Return the newest checkpoint that actually LOADS, or None.

    ``latest.pt`` existing is not evidence that it is readable, and the window
    between the two renames in ``save_checkpoint`` can leave it absent while
    ``latest.prev`` is intact. Both fall back to the previous generation, and the
    fallback is announced -- a silent step back to an older checkpoint is a step
    count that disagrees with the log, which is the same defect class as resolving a
    checkpoint the caller did not name (INBOX 43b).
    """
    path = Path(path)
    for cand in (path, path.with_name(path.name + ".prev")):
        if not cand.exists():
            continue
        try:
            torch.load(cand, map_location="cpu", weights_only=False)
        except Exception as exc:                      # truncated, or half-written
            print(f"[ckpt] {cand.name} exists but will not load ({type(exc).__name__}: {exc}); "
                  f"treating it as absent")
            continue
        if cand != path:
            print(f"[ckpt] FALLING BACK to {cand.name} -- {path.name} was missing or unreadable. "
                  f"The resumed step count comes from the OLDER checkpoint.")
        return cand
    return None


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


def remap_legacy_keys(state, model):
    """PositionEncoding used to BE an nn.Embedding; it now WRAPS one as `self.table`. So a checkpoint
    holding `x.res_pos_emb.weight` cannot load into `x.res_pos_emb.table.weight`, and every direct
    checkpoint predating that refactor fails -- including the one behind Section 5's 0.79 A, and the
    one latent_suitability.py needs for the 0.868 NMR number. scaling.py's docstring promises
    "every existing checkpoint loads and every prior result reproduces"; this is what keeps it.

    The remap is exact WITHIN max_positions. Beyond it the new forward CLAMPS where the old raised,
    so callers evaluating oversized structures must check that themselves -- it is not silently safe.
    Fixed here rather than per-script because three callers already hit it and the next one would too.
    """
    want = set(model.state_dict().keys())
    out, moved = {}, []
    for k, v in state.items():
        if k not in want and k.endswith(".weight"):
            cand = k[: -len(".weight")] + ".table.weight"
            if cand in want:
                out[cand] = v; moved.append((k, cand)); continue
        out[k] = v
    for a, b in moved:
        print(f"[ckpt] remapped legacy key {a} -> {b}")
    return out, moved


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    ckpt["model"], _moved = remap_legacy_keys(ckpt["model"], model)
    grown = grow_embedding_rows(ckpt["model"], model)
    if grown:
        for k, old, new in grown:
            print(f"[ckpt] grew {k} {old} -> {new} (vocabulary extended; "
                  f"existing rows preserved, new rows randomly initialised)")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
