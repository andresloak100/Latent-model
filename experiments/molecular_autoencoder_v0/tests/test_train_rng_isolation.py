"""Evaluating must not change training.

A DataLoader with num_workers > 0 draws its worker base seed from the global
torch RNG each time an iterator is created, and shuffle=True draws its
permutation from the same place. So an added eval pass -- or merely a change
to how often one fires -- shifts the stream that decides batch order and
rotation augmentation for every later epoch.

Two runs of identical code and data then diverge because their logging
differed. That makes a before/after comparison across an eval-schedule change
uninterpretable, which is precisely what a regression check is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class _Toy(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        return torch.tensor([float(i)])


def _train_order_after_eval_passes(n_eval_passes, eval_generator, seed=0):
    """Batch order of a shuffled train loader, after N eval iterations."""
    torch.manual_seed(seed)
    train = DataLoader(_Toy(), batch_size=8, shuffle=True)
    for _ in range(n_eval_passes):
        ev = DataLoader(
            Subset(_Toy(), list(range(16))), batch_size=8, shuffle=False,
            generator=(torch.Generator().manual_seed(seed + 1)
                       if eval_generator else None))
        for _ in ev:
            pass
    return [b.flatten().tolist() for b in train]


def test_eval_passes_shift_the_training_stream_without_a_generator():
    """The bug this guards: more eval passes -> different training run."""
    a = _train_order_after_eval_passes(0, eval_generator=False)
    b = _train_order_after_eval_passes(3, eval_generator=False)
    assert a != b


def test_a_dedicated_generator_isolates_training_from_eval():
    a = _train_order_after_eval_passes(0, eval_generator=True)
    b = _train_order_after_eval_passes(3, eval_generator=True)
    c = _train_order_after_eval_passes(7, eval_generator=True)
    assert a == b == c, "training order must not depend on the eval schedule"


def test_train_script_gives_eval_loaders_their_own_generator():
    """Pin it in the script itself, not just in principle."""
    src = (ROOT / "scripts" / "train.py").read_text()
    assert "eval_gen = torch.Generator()" in src
    assert src.count("generator=") >= 2, "val and train-eval loaders both need one"


def test_eval_generator_is_derived_from_the_run_seed():
    """Two seeds must still give two different runs; this is isolation, not
    a fixed constant that would make the eval order seed-independent."""
    src = (ROOT / "scripts" / "train.py").read_text()
    assert "cfg.train.seed + 1" in src
    assert "cfg.train.seed + 2" in src
