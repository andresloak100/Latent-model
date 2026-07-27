"""A Perceiver-style molecular autoencoder (simplest credible baseline).

Encoder:  atom tokens (identity + coordinates) --cross-attention--> a small,
          fixed set of L learnable latent tokens --self-attention--> a
          bottleneck of shape (L, latent_dim). The latent size is independent
          of the number of atoms, so compression grows with protein size.

Decoder:  per-atom *identity* queries (element, residue, atom name, residue
          index) --cross-attention--> the latent tokens --> an MLP head that
          predicts (x, y, z). The decoder never sees input coordinates, so
          all geometry must pass through the bottleneck. It IS conditioned on
          atom identity/sequence (analogous to a video decoder knowing the
          frame shape); only 3D geometry is compressed.

Equivariance: this architecture is NOT rotation/translation equivariant.
Invariance is handled by (a) centring inputs and (b) Kabsch alignment in the
loss/metrics. This is a deliberate baseline choice, documented in the README.

Modularity: `Encoder`, `Bottleneck`, and `Decoder` are separate modules so a
future equivariant or diffusion variant can swap any one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import constants as C


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_latent_tokens: int = 16
    latent_dim: int = 8
    enc_self_layers: int = 2
    dec_self_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    max_res_pos: int = 1024
    coord_scale: float = 10.0
    # >1 adds a chain embedding for protein-protein complexes. Left at 1 the
    # module is not created at all, so single-chain checkpoints load unchanged.
    max_chains: int = 1
    # "baseline" | "invariant" | "rope" | "perresidue" | "direct"
    #   | "peratom" | "peratom_elem"   (per-atom latents; see model_peratom.py --
    #     compression is 3/latent_dim, so only latent_dim 1-2 is meaningful)
    encoder_type: str = "baseline"
    k_neighbors: int = 8             # neighbours for the invariant encoder
    rope_freqs: int = 128            # Fourier frequencies per axis for the rope encoder (3*2*F coord dims)
    rope_spherical: bool = False     # rope: encode spherical (r/θ/φ) instead of Cartesian
    objective: str = "reconstruct"   # "reconstruct" (direct coord + Kabsch loss) | "flowmatch" (diffusion AE)
    flow_steps: int = 50             # ODE integration steps for flow-matching reconstruction
    # perceiver_direct only: >0 re-enables self-attention over group tokens,
    # which costs O(R^2). Left at 0 the model is O(N.L) end to end.
    group_self_layers: int = 0
    # perceiver_direct only: ALL-ATOM self-attention layers before the latents
    # read the atoms. O(N^2) each -- the expensive kind -- so keep this at 0-2.
    # Jacob's spec allows "very few"; 0 is the cheapest and 1-2 is where local
    # geometry (bonds, chirality) is most likely to be resolved.
    atom_self_layers: int = 0


class AtomFeaturizer(nn.Module):
    """Embed atom identity (+ optional coordinates) into d_model tokens."""

    def __init__(self, d_model: int, max_res_pos: int, use_coords: bool,
                 max_chains: int = 1):
        super().__init__()
        self.chain_emb = nn.Embedding(max_chains, d_model) if max_chains > 1 else None
        self.elem_emb = nn.Embedding(C.N_ELEMENTS, d_model, padding_idx=C.PAD_ELEMENT_IDX)
        self.res_emb = nn.Embedding(C.N_RESIDUES, d_model, padding_idx=C.PAD_RESIDUE_IDX)
        self.atom_emb = nn.Embedding(C.N_ATOM_NAMES, d_model, padding_idx=C.PAD_ATOM_IDX)
        self.pos_emb = nn.Embedding(max_res_pos, d_model)
        self.max_res_pos = max_res_pos
        self.use_coords = use_coords
        if use_coords:
            self.coord_proj = nn.Linear(3, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, batch, coords=None):
        rp = batch["res_pos"].clamp(max=self.max_res_pos - 1)
        x = (
            self.elem_emb(batch["element_idx"])
            + self.res_emb(batch["residue_idx"])
            + self.atom_emb(batch["atom_name_idx"])
            + self.pos_emb(rp)
        )
        if self.chain_emb is not None:
            # Without this, two chains are only distinguishable by their global
            # res_pos, so the model cannot tell a chain break from a peptide bond.
            x = x + self.chain_emb(batch["chain_idx"].clamp(max=self.chain_emb.num_embeddings - 1))
        if self.use_coords and coords is not None:
            x = x + self.coord_proj(coords)
        return self.ln(x)


class CrossAttention(nn.Module):
    """Pre-norm cross-attention block: query attends to context."""

    def __init__(self, d_model, n_heads, ff_mult, dropout):
        super().__init__()
        self.nq = nn.LayerNorm(d_model)
        self.nc = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.nf = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, q, ctx, key_padding_mask=None):
        qn, cn = self.nq(q), self.nc(ctx)
        a, _ = self.attn(qn, cn, cn, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        q = q + self.ff(self.nf(q))
        return q


class SelfAttention(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model, n_heads, ff_mult, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        xn = self.n1(x)
        a, _ = self.attn(xn, xn, xn, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.n2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=True,
                                   max_chains=cfg.max_chains)
        self.latents = nn.Parameter(torch.randn(cfg.n_latent_tokens, cfg.d_model) * 0.02)
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
        self.self_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.enc_self_layers)]
        )

    def forward(self, batch, coords):
        tokens = self.feat(batch, coords)            # (B, N, d)
        pad = ~batch["mask"].bool()                   # True where padded
        B = tokens.shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        lat = self.cross(lat, tokens, key_padding_mask=pad)
        for blk in self.self_blocks:
            lat = blk(lat)
        return lat                                    # (B, L, d)


class Bottleneck(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.down = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.up = nn.Linear(cfg.latent_dim, cfg.d_model)

    def encode(self, lat):
        return self.down(lat)

    def decode(self, z):
        return self.up(z)


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=False,
                                   max_chains=cfg.max_chains)
        self.self_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.dec_self_layers)]
        )
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 3),
        )

    def forward(self, latents, batch, latent_key_padding_mask=None):
        # latent_key_padding_mask (B, L): True where a latent token is padding
        # (used by the per-residue latent, whose token count varies per sample).
        for blk in self.self_blocks:
            latents = blk(latents, key_padding_mask=latent_key_padding_mask)
        q = self.feat(batch)                          # (B, N, d) identity only
        h = self.cross(q, latents, key_padding_mask=latent_key_padding_mask)
        coords = self.head(h) * self.cfg.coord_scale  # (B, N, 3) angstrom
        return coords


class MolecularAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.bottleneck = Bottleneck(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, batch):
        coords_in = batch["coords"] / self.cfg.coord_scale
        lat = self.encoder(batch, coords_in)
        return self.bottleneck.encode(lat)            # (B, L, latent_dim)

    def decode(self, z, batch):
        lat = self.bottleneck.decode(z)
        return self.decoder(lat, batch)

    def forward(self, batch):
        z = self.encode(batch)
        coords = self.decode(z, batch)
        return coords, z

    @property
    def latent_floats(self) -> int:
        return self.cfg.n_latent_tokens * self.cfg.latent_dim

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
