"""Per-residue latent autoencoder ("latent point cloud", ProteinAE-style).

The failed baseline compressed a whole protein into a *fixed* global bottleneck
(16x8 = 128 floats regardless of size) — ~10-20x over-compression. ProteinAE
instead uses a **per-residue** latent (one token per residue, dim 8), so the
latent *scales with the protein* and preserves locality. This module builds
that: atoms are pooled into one latent token per residue; the number of latent
tokens equals the number of residues.

Design: reuses the shared Bottleneck and Decoder unchanged. The encoder pools
atom features to residues with a scatter-mean (O(N), linear in atoms), runs
self-attention over the residue tokens, and produces a latent of shape
(B, R, latent_dim) where R = #residues. The decoder (with a residue padding
mask) lets each atom attend over all residue latents.

Symmetry: like the baseline it sees raw coordinates, so it relies on rotation
augmentation for invariance (the ProteinAE bet), not built-in equivariance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import ModelConfig, AtomFeaturizer, SelfAttention, Bottleneck, Decoder
from .scaling import make_group_attention


def n_residues(res_pos: torch.Tensor) -> int:
    return int(res_pos.max().item()) + 1 if res_pos.numel() else 0


def residue_mask(res_pos: torch.Tensor, mask: torch.Tensor, R: int) -> torch.Tensor:
    """(B, R) 1.0 where residue r has at least one real atom."""
    B = res_pos.shape[0]
    counts = res_pos.new_zeros(B, R, dtype=torch.float32)
    counts.scatter_add_(1, res_pos, mask.to(counts.dtype))
    return (counts > 0).float()


def scatter_mean_residues(tokens, res_pos, mask, R):
    """Mean-pool atom features (B, N, d) into per-residue tokens (B, R, d)."""
    B, N, d = tokens.shape
    idx = res_pos.unsqueeze(-1).expand(-1, -1, d)
    res_sum = tokens.new_zeros(B, R, d)
    res_sum.scatter_add_(1, idx, tokens * mask.unsqueeze(-1))
    counts = tokens.new_zeros(B, R)
    counts.scatter_add_(1, res_pos, mask.to(tokens.dtype))
    res_feats = res_sum / counts.clamp_min(1.0).unsqueeze(-1)
    return res_feats, (counts > 0).float()


class PerResidueEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=True,
                                   max_chains=cfg.max_chains,
                                   use_slot_emb=cfg.use_slot_emb)
        # Group tokens, so this is the other O(R^2) term alongside the
        # decoder's. Dense by default; cfg.attn_window makes it linear.
        self.self_blocks = make_group_attention(cfg, cfg.enc_self_layers)

    def forward(self, batch, coords):
        tokens = self.feat(batch, coords)                       # (B, N, d)
        R = n_residues(batch["res_pos"])
        res_feats, res_mask = scatter_mean_residues(
            tokens, batch["res_pos"], batch["mask"], R)         # (B, R, d)
        pad = ~res_mask.bool()
        for blk in self.self_blocks:
            res_feats = blk(res_feats, key_padding_mask=pad)
        return res_feats, res_mask


class PerResidueAutoencoder(nn.Module):
    per_residue = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PerResidueEncoder(cfg)
        self.bottleneck = Bottleneck(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, batch):
        coords_in = batch["coords"] / self.cfg.coord_scale
        res_feats, _ = self.encoder(batch, coords_in)
        return self.bottleneck.encode(res_feats)               # (B, R, latent_dim)

    def decode(self, z, batch):
        R = z.shape[1]
        rmask = residue_mask(batch["res_pos"], batch["mask"], R)
        lat = self.bottleneck.decode(z)
        return self.decoder(lat, batch, latent_key_padding_mask=~rmask.bool())

    def forward(self, batch):
        z = self.encode(batch)
        return self.decode(z, batch), z

    @property
    def latent_floats(self) -> int:
        # Per-residue budget only. The TOTAL latent is latent_dim x n_residues
        # and therefore depends on the structure -- use latent_floats_for() for
        # anything that reports compression, or the ratio is inflated by ~n_res.
        return self.cfg.latent_dim

    def latent_floats_for(self, n_residues: int) -> int:
        """Actual latent size for a structure with ``n_residues`` residues."""
        return self.cfg.latent_dim * max(int(n_residues), 1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
