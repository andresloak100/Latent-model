"""SE(3)-invariant encoder prototype (candidate fix for the alignment crutch).

The baseline `Encoder` (model.py) feeds *raw coordinates* into attention, so it
is not rotation/translation invariant — invariance is bolted on via input
centring and Kabsch alignment at eval. The held-out experiments suggest that
non-equivariance is a prime suspect for the poor data efficiency (the model
must spend capacity learning that a rotated protein is the same protein).

This module replaces only the ENCODER with one whose latent is invariant *by
construction*: every input feature it sees is an SE(3) invariant of the
geometry —
  * sorted distances to the k nearest (real) neighbours, and
  * distance to the (masked) centroid,
combined with the same atom-identity embeddings. Because attention over
invariant features yields invariant outputs, `InvariantEncoder(coords)` ==
`InvariantEncoder(R·coords + t)` for any rotation R and translation t (verified
exactly in tests/test_equivariance.py).

The decoder is unchanged: it predicts coordinates in a canonical frame from the
invariant latent, and reconstruction is scored up to a rigid transform (Kabsch),
which is precisely the quantity we care about. This is a *minimal* invariant
featurisation; a richer equivariant variant (GVP vector channels / IPA frames)
that also keeps directional information is the natural follow-up.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import (
    ModelConfig, MolecularAutoencoder, AtomFeaturizer, CrossAttention,
    SelfAttention, Bottleneck, Decoder,
)


def make_autoencoder(cfg: ModelConfig):
    """Factory: pick the encoder variant from ``cfg.encoder_type``.

    "baseline" -> raw-coordinate Perceiver AE (model.MolecularAutoencoder)
    "invariant" -> SE(3)-invariant encoder (this module)
    Default keeps existing behaviour, so all current configs are unchanged.
    """
    if getattr(cfg, "objective", "reconstruct") == "flowmatch":
        from .flow import FlowMatchingAutoencoder
        return FlowMatchingAutoencoder(cfg)
    etype = getattr(cfg, "encoder_type", "baseline")
    if etype == "invariant":
        return InvariantAutoencoder(cfg, k_neighbors=getattr(cfg, "k_neighbors", 8))
    if etype == "rope":
        from .model_rope import RoPEAutoencoder
        return RoPEAutoencoder(cfg)
    if etype == "direct":
        from .model_direct import DirectAutoencoder
        return DirectAutoencoder(cfg)
    if etype == "perresidue":
        from .model_perresidue import PerResidueAutoencoder
        return PerResidueAutoencoder(cfg)
    if etype == "atomlatent":
        from .model_atomlatent import AtomLatentAutoencoder
        return AtomLatentAutoencoder(cfg)
    if etype == "perceiver_direct":
        from .model_perceiver_direct import PerceiverDirectAutoencoder
        return PerceiverDirectAutoencoder(
            cfg, group_self_layers=getattr(cfg, "group_self_layers", 0))
    if etype in ("peratom", "peratom_elem"):
        from .model_peratom import PerAtomAutoencoder
        return PerAtomAutoencoder(cfg, element_only=(etype == "peratom_elem"))
    if etype == "baseline":
        return MolecularAutoencoder(cfg)
    raise ValueError(f"unknown encoder_type: {etype!r}")

_BIG = 1.0e4          # sentinel distance for padded / missing neighbours
_MAX_DIST = 50.0      # clamp (angstrom) so features stay bounded


def invariant_geometry_features(coords: torch.Tensor, mask: torch.Tensor, k: int):
    """Return (B, N, k+1) SE(3)-invariant per-atom geometric features.

    Columns: sorted distances to the k nearest real neighbours, then distance
    to the masked centroid. All are invariant to global rotation/translation.
    """
    B, N, _ = coords.shape
    m = mask.unsqueeze(-1)                                   # (B, N, 1)
    n = m.sum(dim=1, keepdim=True).clamp_min(1.0)            # (B, 1, 1)
    centroid = (coords * m).sum(dim=1, keepdim=True) / n     # (B, 1, 3)
    dist_centroid = torch.linalg.norm(coords - centroid, dim=-1)  # (B, N)

    d = torch.cdist(coords, coords)                          # (B, N, N)
    # Exclude self and padded atoms from the neighbour search.
    eye = torch.eye(N, device=coords.device, dtype=torch.bool).unsqueeze(0)
    d = d.masked_fill(eye, _BIG)
    pad_cols = (mask < 0.5).unsqueeze(1)                     # (B, 1, N) padded atoms
    d = d.masked_fill(pad_cols, _BIG)

    kk = min(k, max(N - 1, 1))
    knn, _ = torch.topk(d, kk, dim=-1, largest=False)        # (B, N, kk) ascending
    if kk < k:                                               # pad feature width
        pad = knn.new_full((B, N, k - kk), _MAX_DIST)
        knn = torch.cat([knn, pad], dim=-1)
    knn = knn.clamp(max=_MAX_DIST)
    feats = torch.cat([knn, dist_centroid.unsqueeze(-1).clamp(max=_MAX_DIST)], dim=-1)
    return feats                                             # (B, N, k+1)


class InvariantEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, k_neighbors: int = 8):
        super().__init__()
        self.cfg = cfg
        self.k = k_neighbors
        # Identity-only atom features (no coordinates enter directly).
        self.feat = AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=False)
        self.geom = nn.Sequential(
            nn.Linear(k_neighbors + 1, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.latents = nn.Parameter(torch.randn(cfg.n_latent_tokens, cfg.d_model) * 0.02)
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
        self.self_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.enc_self_layers)]
        )

    def forward(self, batch, coords):
        # coords come in angstrom; scale the invariant features consistently.
        feats = invariant_geometry_features(coords, batch["mask"], self.k) / self.cfg.coord_scale
        tokens = self.ln(self.feat(batch) + self.geom(feats))
        pad = ~batch["mask"].bool()
        lat = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        lat = self.cross(lat, tokens, key_padding_mask=pad)
        for blk in self.self_blocks:
            lat = blk(lat)
        return lat


class InvariantAutoencoder(nn.Module):
    """Same interface as MolecularAutoencoder, with the invariant encoder."""

    def __init__(self, cfg: ModelConfig, k_neighbors: int = 8):
        super().__init__()
        self.cfg = cfg
        self.encoder = InvariantEncoder(cfg, k_neighbors)
        self.bottleneck = Bottleneck(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, batch):
        lat = self.encoder(batch, batch["coords"])          # raw coords -> invariants inside
        return self.bottleneck.encode(lat)

    def decode(self, z, batch):
        return self.decoder(self.bottleneck.decode(z), batch)

    def forward(self, batch):
        z = self.encode(batch)
        return self.decode(z, batch), z

    @property
    def latent_floats(self) -> int:
        return self.cfg.n_latent_tokens * self.cfg.latent_dim

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
