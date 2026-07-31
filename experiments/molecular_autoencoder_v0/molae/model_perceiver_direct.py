"""Perceiver encoder + group-mediated direct readout — decoupled, dynamic latent.

The design this resolves
------------------------
Two requirements were in tension.

*Ours, from measurement:* the whole codec only started working when the decoder
stopped making each ATOM find its own coordinates by attention. Per-atom
identity queries cross-attending over a latent scored 9.5-11 A -- worse than a
0-parameter mean-shape baseline -- at every latent size tried, including
per-residue latents that were LARGER than the fixed one. Swapping in a direct
readout (output slot i simply IS atom i) took the same setup to 0.75 A. The
failure was routing, not latent size.

*From the product requirements:* the latent must be O(N.L) to build, decoupled from
atom count, and choosable at inference -- L=100 for a 100k-atom system, L=10,000
for 1M atoms, with more latents and more diffusion steps as a paid quality dial.
A per-residue latent cannot do that: it scales with N by construction and caps
at ~2.9x compression.

Both hold at once if the routing problem is solved at GROUP granularity instead
of atom granularity:

    atoms --cross-attn--> L latents          O(N.L)   <- decoupled, dynamic
    self-attn over latents                   O(L^2)
    L latents --cross-attn--> R group tokens O(L.R)
    each group emits its own atoms' slots    O(N)     <- the fix that worked

R queries retrieving a rigid-unit description, then a deterministic expansion,
is a far better-conditioned problem than N queries (up to 10^6) each hunting
through L latents for its own three numbers. The expansion is learned once per
group TYPE, not per atom INSTANCE.

Note what is absent: there is no all-atom self-attention anywhere, and -- unlike
`model_direct.py` -- no self-attention over groups either. Group tokens take
their context from the latents. That removes the O(R^2) term, which matters at
scale: 1M atoms is ~125k residues, so R^2 ~ 1.6e10.

Dynamic L
---------
Latent queries are built from a SINUSOIDAL index encoding rather than a learned
parameter block of fixed size, so L is a free argument at inference and is not
bounded by anything seen in training. Group queries are built the same way, so R
is unbounded too (a learned `nn.Embedding(max_res_pos)` would cap at 1024 --
useless for 125k residues).

For the quality dial to actually work, train with L VARYING (pass `n_latents`
per batch). A model trained at a single L will not generalise to another one
just because the encoding admits it.

Compression is 3N / (L * latent_dim) -- unbounded in N, which is the point. Be
aware the aggressive end is lossy for information-theoretic rather than
architectural reasons: L=100, d=8 over 100k atoms is 0.008 floats/atom against 3
raw. Sweep it and read the curve rather than assuming a floor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import constants as C
from .model import ModelConfig, AtomFeaturizer, CrossAttention, SelfAttention, Bottleneck
from .model_perresidue import residue_mask, n_residues


def sinusoidal_index(idx: torch.Tensor, d_model: int) -> torch.Tensor:
    """Standard sinusoidal encoding of an integer index -> (..., d_model).

    Used instead of `nn.Embedding` so neither the latent count L nor the group
    count R is bounded by a table size chosen at construction.
    """
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=idx.device, dtype=torch.float32) / max(half, 1)
    )
    ang = idx.float().unsqueeze(-1) * freqs
    enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if enc.shape[-1] < d_model:                       # odd d_model
        enc = torch.cat([enc, enc[..., :1]], dim=-1)
    return enc


class PerceiverEncoder(nn.Module):
    """Atoms -> L latent tokens. O(N.L); no all-atom self-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=True,
                                   max_chains=cfg.max_chains,
                                   use_slot_emb=cfg.use_slot_emb)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)   # shapes the index encoding
        # OPTIONAL all-atom self-attention. O(N^2) per layer -- the expensive
        # kind -- so it is off by default and meant to stay at 0-2. Included
        # because removing it entirely may cost local geometry, and that should
        # be measured rather than assumed.
        self.atom_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(getattr(cfg, "atom_self_layers", 0))]
        )
        self.cross = CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
        self.self_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.enc_self_layers)]
        )

    def latent_queries(self, B, L, device):
        idx = torch.arange(L, device=device)
        return self.q_proj(sinusoidal_index(idx, self.cfg.d_model)).unsqueeze(0).expand(B, -1, -1)

    def forward(self, batch, coords, n_latents=None):
        tokens = self.feat(batch, coords)                       # (B, N, d)   O(N)
        pad_atoms = ~batch["mask"].bool()
        for blk in self.atom_blocks:                            #             O(N^2)
            tokens = blk(tokens, key_padding_mask=pad_atoms)
        B = tokens.shape[0]
        L = n_latents or self.cfg.n_latent_tokens
        lat = self.latent_queries(B, L, tokens.device)
        pad = ~batch["mask"].bool()
        lat = self.cross(lat, tokens, key_padding_mask=pad)     # (B, L, d)   O(N.L)
        for blk in self.self_blocks:
            lat = blk(lat)                                      #             O(L^2)
        return lat


class GroupDirectDecoder(nn.Module):
    """L latents -> R group tokens -> each group emits its own atoms.

    The group queries cross-attend the latents (O(L.R)); they deliberately do
    NOT self-attend, so no O(R^2) term appears. `group_self_layers > 0` re-enables
    it for ablation only.
    """

    def __init__(self, cfg: ModelConfig, group_self_layers: int = 0):
        super().__init__()
        self.cfg = cfg
        self.n_slots = C.N_ATOM_NAMES
        self.up = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.gtype_emb = nn.Embedding(C.N_RESIDUES, cfg.d_model, padding_idx=C.PAD_RESIDUE_IDX)
        self.cross_blocks = nn.ModuleList(
            [CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(max(1, getattr(cfg, "dec_cross_layers", 1)))]
        )
        self.self_blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(group_self_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, self.n_slots * 3),
        )

    def forward(self, z, batch):
        """z: (B, L, latent_dim) -> coords (B, N, 3); row i is atom i."""
        group_idx = batch["res_pos"]                            # group = residue
        B, N = group_idx.shape
        R = n_residues(group_idx)
        device = z.device

        lat = self.up(z)                                        # (B, L, d)

        # Group queries: unbounded index encoding + which group this is.
        gi = torch.arange(R, device=device)
        q = self.q_proj(sinusoidal_index(gi, self.cfg.d_model)).unsqueeze(0).expand(B, -1, -1)
        gtype = torch.zeros(B, R, dtype=torch.long, device=device)
        gtype.scatter_(1, group_idx, batch["residue_idx"])
        h = q + self.gtype_emb(gtype)

        gpad = ~residue_mask(group_idx, batch["mask"], R).bool()
        for i, blk in enumerate(self.cross_blocks):             # (B, R, d)   O(L.R) each
            h = blk(h, lat)
            if i < len(self.self_blocks):                       # interleave refinement
                h = self.self_blocks[i](h, key_padding_mask=gpad)
        for blk in self.self_blocks[len(self.cross_blocks):]:
            h = blk(h, key_padding_mask=gpad)

        slots = self.head(h).view(B, R, self.n_slots, 3) * self.cfg.coord_scale
        flat = slots.view(B, R * self.n_slots, 3)
        gather_idx = (group_idx * self.n_slots + batch["atom_name_idx"]).clamp(
            max=R * self.n_slots - 1)
        return torch.gather(flat, 1, gather_idx.unsqueeze(-1).expand(-1, -1, 3))   # O(N)


class PerceiverDirectAutoencoder(nn.Module):
    """Decoupled dynamic latent (Perceiver) + the direct readout that works."""

    per_residue = False
    dynamic_latent = True

    def __init__(self, cfg: ModelConfig, group_self_layers: int = 0):
        super().__init__()
        self.cfg = cfg
        self.encoder = PerceiverEncoder(cfg)
        self.bottleneck = Bottleneck(cfg)
        self.decoder = GroupDirectDecoder(cfg, group_self_layers=group_self_layers)

    def encode(self, batch, n_latents=None):
        lat = self.encoder(batch, batch["coords"] / self.cfg.coord_scale, n_latents)
        return self.bottleneck.encode(lat)                      # (B, L, latent_dim)

    def decode(self, z, batch):
        return self.decoder(z, batch)

    def forward(self, batch, n_latents=None):
        z = self.encode(batch, n_latents)
        return self.decode(z, batch), z

    @property
    def latent_floats(self) -> int:
        """Total latent size -- INDEPENDENT of atom or residue count."""
        return self.cfg.n_latent_tokens * self.cfg.latent_dim

    def latent_floats_for(self, n_residues: int) -> int:
        return self.latent_floats          # constant, by design

    def latent_floats_for_atoms(self, n_atoms: int) -> int:
        return self.latent_floats          # constant, by design

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
