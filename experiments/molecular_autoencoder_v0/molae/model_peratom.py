"""Per-atom latents with identity routing — the chemistry-general variant.

Why this exists
---------------
`model_direct.py` fixed the codec (9.5 A -> 0.75 A held-out) by removing the
attention lookup from the decoder: atom *i* reads a dedicated output slot
instead of having to *find* its coordinates by attending over a global latent.

But the way it earns that dedicated slot is protein-specific. It allocates a
bank of ``N_ATOM_NAMES`` slots per residue and indexes it with
``(res_pos, atom_name_idx)`` — which needs (a) a residue decomposition and
(b) a fixed atom-name vocabulary. Neither exists for ligands, solvent, ions,
lipids, nucleic acids, or arbitrary MD systems.

The important property was never "per-residue". It was **static index routing**:
a fixed, non-learned map from latent slot to output atom. This module takes that
to its limit — one latent per atom, and the routing map is the identity:

    atom i  ->  z_i  ->  self-attention over atom tokens  ->  xyz_i

No residues, no atom-name vocabulary, no gather. Permutation-equivariant by
construction (permute the input atoms and the latents and outputs permute with
them), so no canonical ordering is required either.

Two arms
--------
``peratom``       same identity featurisation as every other arm (element,
                  residue type, atom name, residue index) so the A/B against
                  ``direct`` isolates *routing granularity* alone.
``peratom_elem``  element + coordinates only. No residue type, no atom name, no
                  residue index. This arm carries **zero protein assumptions**
                  and is the one that would transfer to arbitrary molecules; it
                  is also the honest test of whether the identity side-channel
                  was doing the work.

The compression ceiling (read this before running it)
-----------------------------------------------------
A per-atom latent of dimension ``d`` costs ``d`` floats per atom against 3
coordinate floats per atom, so compression is exactly ``3/d`` *regardless of
system size*:

    d = 1  ->  3.0x        d = 2  ->  1.5x        d >= 3  ->  no compression

At ``d >= 3`` the latent can hold the coordinates verbatim and the experiment is
vacuous. So per-atom arms are only meaningful at ``latent_dim`` 1 or 2, and
**3x is a hard ceiling** for this family. Per-residue reaches 2.9x at 8
floats/residue and 11.7x at 2 floats/residue precisely because pooling amortises
the latent over ~7.8 atoms. Generality and compression trade off directly here;
see REPORT.md section 4.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import constants as C
from .model import ModelConfig, AtomFeaturizer, SelfAttention, Bottleneck


class ElementOnlyFeaturizer(nn.Module):
    """Chemistry-general atom embedding: element (+ optional coordinates).

    Deliberately omits residue type, atom name and residue index — the three
    inputs that assume a protein. Anything with elements and coordinates can be
    featurised by this, which is the point.
    """

    def __init__(self, d_model: int, use_coords: bool):
        super().__init__()
        self.elem_emb = nn.Embedding(C.N_ELEMENTS, d_model, padding_idx=C.PAD_ELEMENT_IDX)
        self.use_coords = use_coords
        if use_coords:
            self.coord_proj = nn.Linear(3, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, batch, coords=None):
        x = self.elem_emb(batch["element_idx"])
        if self.use_coords and coords is not None:
            x = x + self.coord_proj(coords)
        return self.ln(x)


def _featurizer(cfg: ModelConfig, element_only: bool, use_coords: bool):
    if element_only:
        return ElementOnlyFeaturizer(cfg.d_model, use_coords=use_coords)
    return AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=use_coords)


class PerAtomEncoder(nn.Module):
    """Atom tokens -> self-attention -> one feature vector per atom."""

    def __init__(self, cfg: ModelConfig, element_only: bool = False):
        super().__init__()
        self.cfg = cfg
        self.feat = _featurizer(cfg, element_only, use_coords=True)
        self.blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.enc_self_layers)]
        )

    def forward(self, batch, coords):
        h = self.feat(batch, coords)                    # (B, N, d)
        pad = ~batch["mask"].bool()
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad)
        return h


class DirectAtomDecoder(nn.Module):
    """Per-atom latent -> xyz for that same atom. Routing is the identity."""

    def __init__(self, cfg: ModelConfig, element_only: bool = False):
        super().__init__()
        self.cfg = cfg
        self.up = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.feat = _featurizer(cfg, element_only, use_coords=False)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.dec_self_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 3),
        )

    def forward(self, z, batch):
        """z: (B, N, latent_dim) -> coords (B, N, 3), row i is atom i."""
        h = self.ln(self.up(z) + self.feat(batch))
        pad = ~batch["mask"].bool()
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad)
        return self.head(h) * self.cfg.coord_scale


class PerAtomAutoencoder(nn.Module):
    """One latent per atom; no residue decomposition anywhere in the model."""

    per_residue = False
    per_atom = True

    def __init__(self, cfg: ModelConfig, element_only: bool = False):
        super().__init__()
        self.cfg = cfg
        self.element_only = element_only
        self.encoder = PerAtomEncoder(cfg, element_only=element_only)
        self.bottleneck = Bottleneck(cfg)
        self.decoder = DirectAtomDecoder(cfg, element_only=element_only)

    def encode(self, batch):
        h = self.encoder(batch, batch["coords"] / self.cfg.coord_scale)
        return self.bottleneck.encode(h)                # (B, N, latent_dim)

    def decode(self, z, batch):
        return self.decoder(z, batch)

    def forward(self, batch):
        z = self.encode(batch)
        return self.decode(z, batch), z

    @property
    def latent_floats(self) -> int:
        # Per-ATOM budget. Total latent is latent_dim x n_atoms; reporting code
        # must use latent_floats_for_atoms() or compression is inflated by ~N.
        return self.cfg.latent_dim

    def latent_floats_for_atoms(self, n_atoms: int) -> int:
        return self.cfg.latent_dim * max(int(n_atoms), 1)

    def latent_floats_for(self, n_residues: int) -> int:
        """Compatibility shim for per-residue reporting call sites.

        The per-atom latent does not scale with residues, so a residue count
        cannot determine it. Callers that know the atom count must use
        ``latent_floats_for_atoms``; raising here prevents a silent repeat of
        the compression mis-report that inflated the per-residue arms by ~60x.
        """
        raise TypeError(
            "PerAtomAutoencoder latent scales with atoms, not residues; "
            "use latent_floats_for_atoms(n_atoms)."
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
