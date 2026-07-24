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
    out_dir: str = "outputs/stage_a"
    overfit: bool = False  # if true, train on train+val (Stage A sanity)
    augment_rotation: bool = False  # random SO(3) input rotations (teaches non-equivariant encoders invariance)


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
