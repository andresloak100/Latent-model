#!/usr/bin/env python3
"""A generic transformer autoencoder over atoms. Nothing here knows what a molecule is.

The encoder sees, per atom, an element token, an index, and three numbers. It has no notion of
bonds, residues, chains, distances, or rotation. The bottleneck is a mean-pool followed by a single
linear map to L numbers -- one vector for the whole structure -- and the decoder rebuilds coordinates
from that vector plus each atom's element and index.

The decoder is GIVEN the element and index of every atom. Only the geometry passes through the
bottleneck; the atom inventory is not being compressed, which is what makes L a clean capacity axis.

Deliberately absent: equivariant layers, frames, invariant features, distance matrices, internal
coordinates, and every chemistry-specific loss term. See SPEC.md.
"""
import math
import torch
import torch.nn as nn

NVOCAB = 20            # element ids 0..19; the shard uses 0..15


def _blocks(d, depth, heads, ff):
    layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                       dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class StructAE(nn.Module):
    def __init__(self, d_model=256, depth=6, latent=128, na=256, heads=None, ff_mult=4):
        super().__init__()
        heads = heads or max(1, d_model // 64)
        ff = ff_mult * d_model
        self.na, self.latent = na, latent
        # --- encoder ---
        self.e_elem = nn.Embedding(NVOCAB, d_model)
        self.e_pos = nn.Embedding(na, d_model)
        self.e_xyz = nn.Linear(3, d_model)
        self.enc = _blocks(d_model, depth, heads, ff)
        self.enc_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent)
        # --- decoder ---
        self.from_latent = nn.Linear(latent, d_model)
        self.d_elem = nn.Embedding(NVOCAB, d_model)
        self.d_pos = nn.Embedding(na, d_model)
        self.dec = _blocks(d_model, depth, heads, ff)
        self.dec_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def encode(self, elem, xyz):
        B, N = elem.shape
        idx = torch.arange(N, device=elem.device).unsqueeze(0).expand(B, N)
        h = self.e_elem(elem) + self.e_pos(idx) + self.e_xyz(xyz)
        h = self.enc_norm(self.enc(h))
        return self.to_latent(h.mean(1))                     # (B, L) -- one vector per structure

    def decode(self, z, elem):
        B, N = elem.shape
        idx = torch.arange(N, device=elem.device).unsqueeze(0).expand(B, N)
        h = self.d_elem(elem) + self.d_pos(idx) + self.from_latent(z).unsqueeze(1)
        return self.out(self.dec_norm(self.dec(h)))          # (B, N, 3)

    def forward(self, elem, xyz):
        return self.decode(self.encode(elem, xyz), elem)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
