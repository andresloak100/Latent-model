"""Flow-matching (diffusion) autoencoder — ProteinAE-style objective.

Instead of the decoder predicting coordinates directly (trained with a
coordinate+Kabsch loss), it becomes a **conditional velocity field**:

  * encode the clean structure -> latent z (any of our encoders);
  * define a linear interpolation x_t = (1-t) x0 + t x1 between zero-CoM noise
    x0 and the true (centred) coords x1;
  * train the decoder v_theta(x_t, t, z, identity) to predict the velocity
    x1 - x0 (rectified-flow / v-target), masked over real atoms;
  * reconstruct by integrating dx/dt = v_theta from noise at t=0 to t=1.

This is the "diffusion autoencoder": the encoder fixes the latent, the decoder
is a small conditional generator. Symmetry is handled by zero-CoM + rotation
augmentation (non-equivariant), matching ProteinAE. Composes with the
per-residue latent (the recommended pairing).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import (
    ModelConfig, AtomFeaturizer, CrossAttention, SelfAttention, Encoder, Bottleneck,
)


def zero_com(x, mask):
    """Subtract the masked centre of mass (per batch element)."""
    w = mask.unsqueeze(-1)
    n = w.sum(dim=1, keepdim=True).clamp_min(1.0)
    com = (x * w).sum(dim=1, keepdim=True) / n
    return (x - com) * w


def timestep_embedding(t, dim):
    """Sinusoidal embedding of scalar t in [0, 1]. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    ang = t.unsqueeze(-1) * freqs * 1000.0
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], dim - emb.shape[-1])], dim=-1)
    return emb


class FlowDecoder(nn.Module):
    """Conditional velocity field: (x_t, t, latents, identity) -> velocity."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=False)
        self.coord_proj = nn.Linear(3, cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
        self.self_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.dec_self_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 3),
        )

    def forward(self, x_t, t, latents, batch, latent_key_padding_mask=None):
        for blk in self.self_blocks:
            latents = blk(latents, key_padding_mask=latent_key_padding_mask)
        q = self.feat(batch)                                   # identity (B, N, d)
        q = q + self.coord_proj(x_t / self.cfg.coord_scale)    # noised coords
        temb = self.time_mlp(timestep_embedding(t, self.cfg.d_model))
        q = q + temb.unsqueeze(1)                              # broadcast time over atoms
        h = self.cross(q, latents, key_padding_mask=latent_key_padding_mask)
        return self.head(h) * self.cfg.coord_scale             # velocity (B, N, 3)


class FlowMatchingAutoencoder(nn.Module):
    objective = "flowmatch"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.per_residue = (cfg.encoder_type == "perresidue")
        if self.per_residue:
            from .model_perresidue import PerResidueEncoder
            self.encoder = PerResidueEncoder(cfg)
        else:
            self.encoder = Encoder(cfg)                        # raw-coord Perceiver
        self.bottleneck = Bottleneck(cfg)
        self.decoder = FlowDecoder(cfg)

    def encode(self, batch):
        """Return (latents (B, L, d), latent_key_padding_mask or None)."""
        coords_in = batch["coords"] / self.cfg.coord_scale
        if self.per_residue:
            res_feats, res_mask = self.encoder(batch, coords_in)
            z = self.bottleneck.encode(res_feats)
            return z, ~res_mask.bool()
        lat = self.encoder(batch, coords_in)
        return self.bottleneck.encode(lat), None

    def _decode_v(self, x_t, t, z, batch, latent_mask):
        return self.decoder(x_t, t, self.bottleneck.decode(z), batch,
                            latent_key_padding_mask=latent_mask)

    def training_loss(self, batch):
        mask = batch["mask"]
        x1 = zero_com(batch["coords"], mask)
        x0 = zero_com(torch.randn_like(x1), mask)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1.0 - t).view(-1, 1, 1) * x0 + t.view(-1, 1, 1) * x1
        v_target = (x1 - x0) * mask.unsqueeze(-1)
        z, latent_mask = self.encode(batch)
        v_pred = self._decode_v(xt, t, z, batch, latent_mask) * mask.unsqueeze(-1)
        sq = ((v_pred - v_target) ** 2).sum(-1) * mask
        loss = sq.sum() / mask.sum().clamp_min(1.0)
        return loss, {"total": float(loss.detach()), "flow_mse": float(loss.detach())}

    @torch.no_grad()
    def reconstruct(self, batch, n_steps=None):
        n_steps = n_steps or self.cfg.flow_steps
        mask = batch["mask"]
        z, latent_mask = self.encode(batch)
        x = zero_com(torch.randn(*mask.shape, 3, device=mask.device), mask)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((mask.shape[0],), k * dt, device=mask.device)
            v = self._decode_v(x, t, z, batch, latent_mask)
            x = zero_com(x + v * dt, mask)
        return x

    def forward(self, batch):
        z, _ = self.encode(batch)
        return self.reconstruct(batch), z

    @property
    def latent_floats(self) -> int:
        return self.cfg.latent_dim if self.per_residue else self.cfg.n_latent_tokens * self.cfg.latent_dim

    def latent_floats_for(self, n_residues: int) -> int:
        """Actual latent size for one structure (per-residue latents scale with it)."""
        if self.per_residue:
            return self.cfg.latent_dim * max(int(n_residues), 1)
        return self.cfg.n_latent_tokens * self.cfg.latent_dim

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
