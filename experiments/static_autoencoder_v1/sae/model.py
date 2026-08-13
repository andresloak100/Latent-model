"""A GENERIC autoencoder over sets of atoms. No molecular inductive bias.

WHAT IS DELIBERATELY ABSENT, because the study measures how a general method scales and any
of these would confound exactly that measurement:

  * no SE(3)/E(3) equivariance -- rotation is handled by DATA AUGMENTATION, which is the
    generic way to acquire an invariance rather than to build one in;
  * no internal coordinates, torsions, frames or distance matrices;
  * no bond, angle, clash or energy terms anywhere in the loss;
  * no residue, chain, backbone/side-chain or secondary-structure concepts. Tokens are ATOMS.

WHAT IS PRESENT is the standard transformer kit: learned embeddings, multi-head attention,
learned positional embedding on the atom index, pre-norm residual blocks.

ARCHITECTURE (Perceiver-style, chosen because it handles variable-length inputs without
padding waste and gives the two sweep axes as clean independent knobs):

    atoms -> embed -> self-attention stack -> cross-attention into LATENT QUERIES
          -> bottleneck of shape (n_latent_tokens, latent_width)
          -> cross-attention from ATOM QUERIES -> self-attention stack -> xyz

    total latent size = n_latent_tokens * latent_width

A CHOICE THAT MATERIALLY CHANGES THE TASK, stated here because BRIEF.md requires open choices
to be recorded where they are made: by default the decoder is given the atom COMPOSITION
(element id and index) as its queries and must predict only coordinates. The latent therefore
carries geometry, not identity. Setting `predict_composition=True` makes the decoder recover
element identity from the latent as well, which is a strictly harder task and a different
scaling curve. Both are legitimate; they are not the same experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AEConfig:
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    ffn_mult: int = 4
    n_latent_tokens: int = 1
    latent_width: int = 64
    max_atoms: int = 1024
    n_elements: int = 16
    predict_composition: bool = False
    dropout: float = 0.0

    @property
    def latent_size(self) -> int:
        return self.n_latent_tokens * self.latent_width


class Block(nn.Module):
    """Pre-norm transformer block. Self-attention when `kv` is None, cross-attention otherwise."""

    def __init__(self, d, n_heads, ffn_mult, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.n_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d), nn.Dropout(dropout)
        )

    def forward(self, x, kv=None, key_padding_mask=None):
        q = self.n1(x)
        k = q if kv is None else self.n_kv(kv)
        a, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.n2(x))


class StaticAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.coord_in = nn.Linear(3, d)
        self.elem_emb = nn.Embedding(cfg.n_elements, d)
        self.pos_emb = nn.Embedding(cfg.max_atoms, d)

        self.enc = nn.ModuleList([Block(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
                                  for _ in range(cfg.n_enc_layers)])
        self.latent_q = nn.Parameter(torch.randn(cfg.n_latent_tokens, d) * 0.02)
        self.enc_cross = Block(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout)

        # THE BOTTLENECK. Everything the decoder sees about geometry passes through these numbers.
        self.to_latent = nn.Linear(d, cfg.latent_width)
        self.from_latent = nn.Linear(cfg.latent_width, d)

        self.dec_cross = Block(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
        self.dec = nn.ModuleList([Block(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
                                  for _ in range(cfg.n_dec_layers)])
        self.n_out = nn.LayerNorm(d)
        self.coord_out = nn.Linear(d, 3)
        self.elem_out = nn.Linear(d, cfg.n_elements) if cfg.predict_composition else None

    # -- pieces, exposed separately so metrics can encode without decoding ------------------

    def encode(self, xyz, elem, mask):
        """(B,N,3), (B,N) long, (B,N) bool where True = real atom -> (B, n_latent_tokens, latent_width)."""
        B, N, _ = xyz.shape
        idx = torch.arange(N, device=xyz.device).clamp_max(self.cfg.max_atoms - 1)
        h = self.coord_in(xyz) + self.elem_emb(elem) + self.pos_emb(idx)[None]
        pad = ~mask
        for blk in self.enc:
            h = blk(h, key_padding_mask=pad)
        q = self.latent_q[None].expand(B, -1, -1)
        q = self.enc_cross(q, kv=h, key_padding_mask=pad)
        return self.to_latent(q)

    def decode(self, z, elem, mask):
        B, N = elem.shape
        idx = torch.arange(N, device=z.device).clamp_max(self.cfg.max_atoms - 1)
        q = self.pos_emb(idx)[None].expand(B, -1, -1)
        if not self.cfg.predict_composition:
            q = q + self.elem_emb(elem)
        h = self.dec_cross(q, kv=self.from_latent(z))
        for blk in self.dec:
            h = blk(h, key_padding_mask=~mask)
        h = self.n_out(h)
        logits = self.elem_out(h) if self.elem_out is not None else None
        return self.coord_out(h), logits

    def forward(self, xyz, elem, mask):
        z = self.encode(xyz, elem, mask)
        rec, logits = self.decode(z, elem, mask)
        return rec, z, logits


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def config_for_params(target: int, cfg: AEConfig, tol: float = 0.05,
                      lo: int = 16, hi: int = 2048) -> AEConfig:
    """Binary-search `d_model` so the model lands within `tol` of `target` parameters.

    The sweep varies parameter count as an INDEPENDENT axis from latent size, so the width has
    to be solved for rather than guessed -- otherwise the two axes are entangled and the grid
    measures their sum. `n_heads` is kept dividing `d_model`, and depth is held fixed so a row
    of the grid differs in one thing.
    """
    from dataclasses import replace

    def build(d):
        heads = max(1, min(cfg.n_heads, d // 16))
        while d % heads:
            heads -= 1
        return replace(cfg, d_model=d, n_heads=heads)

    best = build(lo)
    while lo <= hi:
        mid = ((lo + hi) // 2 // 8) * 8 or 8
        c = build(mid)
        n = n_params(StaticAutoencoder(c))
        if abs(n - target) < abs(n_params(StaticAutoencoder(best)) - target):
            best = c
        if abs(n - target) / target <= tol:
            return c
        if n < target:
            lo = mid + 8
        else:
            hi = mid - 8
    return best
