"""YAML-backed configuration (no hard-coded paths in scripts)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .model import ModelConfig
from .losses import LossWeights


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    splits_file: str = "data/splits.json"
    min_residues: int = 20
    max_residues: int = 120
    max_atoms: int = 1200
    # Quality-tiered pretraining. ``tiers`` maps a tier name to a processed
    # directory, e.g. {"predicted": "data/processed_csm",
    #                  "experimental": "data/processed_big"}.
    # Left empty, training is single-source and behaves exactly as before.
    #
    # ``curriculum`` is the schedule as four sweepable scalars -- hq_start,
    # hq_end, ramp_start, ramp_end -- so a sweep over it is a config grid
    # rather than a code change. See molae/curriculum.py.
    #
    # VALIDATION IS ALWAYS DRAWN FROM THE HIGH TIER ONLY. A held-out set
    # containing predicted structures measures how well the model learned the
    # prediction distribution, and that failure looks exactly like success.
    tiers: dict = field(default_factory=dict)
    curriculum: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 2000
    batch_size: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 200
    ckpt_every: int = 200
    clash_dist: float = 1.5
    loss_max_atoms: int = 1200  # subsample cap for distance/clash loss (per-atom O(N^2)); raise for large proteins
    out_dir: str = "outputs/stage_a"
    overfit: bool = False  # if true, train on train+val (Stage A sanity)
    augment_rotation: bool = False
    # Masked-reconstruction objective: fraction of atoms whose INPUT coordinates
    # are corrupted each step (target unchanged). Multiplies the signal from a
    # fixed structure set -- the matched PDB band holds only ~3,853 entries.
    corrupt_frac: float = 0.0
    # Weight EMA. 0.0 = off (every result before this existed was measured with
    # it off). Averaging the weights is the only fix for the final-checkpoint
    # lottery that changes the run rather than the report -- held-out RMSD over
    # the last ten evaluations of one complex run spreads sd 0.19-0.45A.
    ema_decay: float = 0.0
    # Masked geometric denoising (molae/masking.py). All default off.
    mask_atom_frac: float = 0.0
    mask_region_frac: float = 0.0
    mask_region_span: int = 8
    mask_noise_std: float = 0.0
    # Symmetry-corrected RMSD alongside the strict one. Costs a Hungarian
    # match per symmetry class per structure, so it is opt-in.
    report_symmetry_rmsd: bool = False
    corrupt_mode: str = "zero"      # "zero" | "noise"  # random SO(3) input rotations (teaches non-equivariant encoders invariance)


@dataclass
class ExperimentConfig:
    name: str = "stage_a_overfit"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_yaml(path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return ExperimentConfig(
            name=raw.get("name", "experiment"),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossWeights(**raw.get("loss", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "loss": asdict(self.loss),
            "train": asdict(self.train),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
