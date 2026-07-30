"""LLM-style encoder: atoms as a 1D token sequence with layered position codes.

the brief's proposal — treat the structure like a "sentence of atoms" and use a
plain (Perceiver) transformer, letting the model learn geometry from a rich
coordinate *position encoding* rather than baking in equivariance:

  0. atom identity   — element / residue / atom-name embeddings (token type)
  1. list position   — a 128-dim sinusoidal code of the atom's index in the 1D
                       list (like an LLM token position)
  2. coordinate code — a Fourier / RoPE-style encoding of each atom's 3D
                       position (Cartesian, or spherical r/θ/φ), default 768 dims

These are concatenated and projected to d_model, then a Perceiver cross-attends
to the latent bottleneck (shared with the other encoders, so the A/B is fair).

Symmetry note: coordinates are centred (→ translation invariant), but this
encoder is **not** rotation invariant — it is the "learn invariance from data +
augmentation" bet (Bet B), the opposite of model_equivariant.py (Bet A). Train
with random-rotation augmentation to teach invariance; the Kabsch loss already
makes the *target* rotation-invariant. Spherical coordinates are only invariant
in a canonical frame, which is left as a follow-up (documented).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import (
    ModelConfig, AtomFeaturizer, CrossAttention, SelfAttention, Bottleneck, Decoder,
)


def sinusoidal_list_pe(n: int, dim: int, device) -> torch.Tensor:
    """(n, dim) standard sinusoidal position encoding over atom index."""
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def fourier_coord_pe(coords, mask, n_freqs, spherical=False,
                     lambda_min=0.7, lambda_max=64.0):
    """(B, N, 3*2*n_freqs) Fourier/RoPE-style encoding of centred coordinates.

    Frequencies are log-spaced over physical wavelengths ``lambda_min`` ..
    ``lambda_max`` angstrom (not the aliasing 2^k schedule), so the code is
    smooth, numerically stable, and translation-invariant (coords are centred).
    """
    w = mask.unsqueeze(-1)
    n = w.sum(dim=1, keepdim=True).clamp_min(1.0)
    c = coords - (coords * w).sum(dim=1, keepdim=True) / n      # centre (transl. inv.)
    if spherical:
        r = torch.linalg.norm(c, dim=-1, keepdim=True)          # (B, N, 1) invariant
        theta = torch.atan2(c[..., 1:2], c[..., 0:1])           # azimuth
        phi = torch.acos((c[..., 2:3] / r.clamp_min(1e-6)).clamp(-1.0, 1.0))
        comp = torch.cat([r, theta, phi], dim=-1)               # (B, N, 3)
    else:
        comp = c                                                # (B, N, 3) angstrom
    wl = torch.logspace(math.log10(lambda_min), math.log10(lambda_max),
                        n_freqs, device=coords.device)
    freqs = 2.0 * math.pi / wl                                  # (F,) rad/angstrom
    ang = comp.unsqueeze(-1) * freqs                            # (B, N, 3, F)
    pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)    # (B, N, 3, 2F)
    return pe.flatten(start_dim=2)                              # (B, N, 3*2F)


class RoPEEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, list_pe_dim: int = 128,
                 n_freqs: int = 128, spherical: bool = False):
        super().__init__()
        self.cfg = cfg
        self.list_pe_dim = list_pe_dim
        self.n_freqs = n_freqs
        self.spherical = spherical
        self.id_feat = AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=False)
        coord_dim = 3 * 2 * n_freqs
        self.in_proj = nn.Linear(cfg.d_model + list_pe_dim + coord_dim, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.latents = nn.Parameter(torch.randn(cfg.n_latent_tokens, cfg.d_model) * 0.02)
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
        self.self_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.enc_self_layers)]
        )

    def forward(self, batch, coords):
        B, N = coords.shape[0], coords.shape[1]
        ids = self.id_feat(batch)                                        # (B, N, d)
        lpe = sinusoidal_list_pe(N, self.list_pe_dim, coords.device)     # (N, 128)
        lpe = lpe.unsqueeze(0).expand(B, -1, -1)
        cpe = fourier_coord_pe(coords, batch["mask"], self.n_freqs,
                               self.spherical)                          # (B, N, 3*2F)
        tokens = self.ln(self.in_proj(torch.cat([ids, lpe, cpe], dim=-1)))
        pad = ~batch["mask"].bool()
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        lat = self.cross(lat, tokens, key_padding_mask=pad)
        for blk in self.self_blocks:
            lat = blk(lat)
        return lat


class RoPEAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = RoPEEncoder(cfg, n_freqs=getattr(cfg, "rope_freqs", 128),
                                   spherical=getattr(cfg, "rope_spherical", False))
        self.bottleneck = Bottleneck(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, batch):
        return self.bottleneck.encode(self.encoder(batch, batch["coords"]))

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
