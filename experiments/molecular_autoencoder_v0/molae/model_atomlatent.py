"""Atom tokens -> L shared latent tokens -> atom outputs.

Every previous arm in this repo tied latent count to input structure: one
latent per residue (works, 0.79 A) or one per atom (fails, 4.68-6.32 A with
coin-flip chirality). This one unties them. L is a free config argument, and
the same L serves 200 atoms or 200,000.

The design problem, and the resolution
--------------------------------------
Locality-guided cross-attention needs to know WHERE an atom is, in order to
send it to nearby latents. But the atom's position is the thing being
predicted -- the decoder never sees input coordinates, by requirement. Taking
them from the encoder would be a content skip, and at diffusion time there is
no encoder to take them from.

So the decoder runs in two passes:

    pass 1   global cross-attention over all L latents  ->  provisional xyz
    pass 2   locality-guided cross-attention, each atom attending only to the
             k latents whose ANCHORS are nearest its provisional xyz

Pass 1's positions come entirely from the latent, so pass 2's locality is
derived, not leaked. Nothing bypasses the bottleneck.

Anchors
-------
Each latent token carries an explicit 3D anchor among its channels. Anchors
are computed at encode time as the attention-weighted centroid of the atom
coordinates that latent attends to -- literally "where this latent is
looking" -- so they are grounded in real geometry from step one rather than
having to be discovered.

They are stored INSIDE the latent and counted against its budget: a token of
``latent_dim`` d spends 3 channels on the anchor and d-3 on content. That is
the honest accounting, and it is what makes the anchors available at decode
time without a side channel. The diffusion stage will have to generate
anchors along with content, which is the intended consequence.

Cost
----
Encoder: O(N.w) local atom attention + O(N.L) cross-attention. Decoder:
O(N.L) for pass 1 and O(N.k) for pass 2. No O(N^2) anywhere, which objective
4 makes binding rather than merely desirable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import constants as C
from .model import ModelConfig, AtomFeaturizer, CrossAttention, SelfAttention
from .scaling import WindowedSelfAttention, sinusoidal_index

WL_VOCAB = 512          # hashed; real molecules use far fewer classes
MAX_CHARGE = 8          # formal charge range [-8, +8]


class GraphAtomFeaturizer(nn.Module):
    """Atom identity from the GRAPH alone -- no residues, no atom names.

    Everything here comes out of an .sdf: element, formal charge, bond orders,
    degree, and the Weisfeiler-Lehman class those induce. Two extra channels
    carry ADDRESSABILITY rather than chemistry -- the canonical rank, and the
    ordinal within a symmetry class -- because WL classes are deliberately
    non-unique and identical queries would place two atoms on top of each
    other. Both are sinusoidal, so neither caps the atom count.
    """

    def __init__(self, d_model: int, use_coords: bool):
        super().__init__()
        self.d_model = d_model
        self.elem = nn.Embedding(C.N_ELEMENTS, d_model, padding_idx=C.PAD_ELEMENT_IDX)
        self.charge = nn.Embedding(2 * MAX_CHARGE + 1, d_model)
        self.wl = nn.Embedding(WL_VOCAB, d_model)
        self.degree = nn.Embedding(16, d_model)
        self.bond_type = nn.Embedding(8, d_model)
        self.rank_proj = nn.Linear(d_model, d_model)
        self.ord_proj = nn.Linear(d_model, d_model)
        self.use_coords = use_coords
        if use_coords:
            self.coord_proj = nn.Linear(3, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, batch, coords=None):
        z = batch["element_idx"]
        g = lambda k: batch[k] if k in batch else torch.zeros_like(z)   # noqa: E731
        x = (self.elem(z)
             + self.charge((g("formal_charge") + MAX_CHARGE).clamp(0, 2 * MAX_CHARGE))
             + self.wl(g("wl_class").clamp(0, WL_VOCAB - 1) % WL_VOCAB)
             + self.degree(g("degree").clamp(0, 15))
             + self.bond_type(g("max_bond_type").clamp(0, 7))
             + self.rank_proj(sinusoidal_index(g("canonical_rank"), self.d_model))
             + self.ord_proj(sinusoidal_index(g("class_ordinal"), self.d_model)))
        if self.use_coords and coords is not None:
            x = x + self.coord_proj(coords)
        return self.ln(x)


def make_featurizer(cfg, use_coords: bool):
    if getattr(cfg, "atom_addressing", "group") == "graph":
        return GraphAtomFeaturizer(cfg.d_model, use_coords=use_coords)
    return AtomFeaturizer(cfg.d_model, cfg.max_res_pos, use_coords=use_coords,
                          max_chains=cfg.max_chains, use_slot_emb=cfg.use_slot_emb)


class AnchorPool(nn.Module):
    """Cross-attention that also returns where each latent is looking.

    A plain attention module would give the pooled FEATURES and throw the
    weights away. Here the same weights pool the atom COORDINATES too, which
    is what makes an anchor a real position instead of a number the model has
    to learn from scratch.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.nq = nn.LayerNorm(d_model)
        self.nk = nn.LayerNorm(d_model)

    def forward(self, lat, atoms, coords, mask):
        B, L, D = lat.shape
        N = atoms.shape[1]
        q = self.q(self.nq(lat)).view(B, L, self.h, self.dk).transpose(1, 2)
        kk = self.k(self.nk(atoms)).view(B, N, self.h, self.dk).transpose(1, 2)
        vv = self.v(atoms).view(B, N, self.h, self.dk).transpose(1, 2)

        att = (q @ kk.transpose(-1, -2)) / (self.dk ** 0.5)          # (B,h,L,N)
        att = att.masked_fill(~mask.bool().view(B, 1, 1, N), float("-inf"))
        att = att.softmax(-1)

        pooled = (att @ vv).transpose(1, 2).reshape(B, L, D)
        # Head-averaged weights pool coordinates -> one anchor per latent.
        w = att.mean(1)                                              # (B,L,N)
        anchors = w @ coords                                         # (B,L,3)
        # Atom-level traceability: which latents each atom WRITES to. Kept only
        # when explicitly recording, so training allocates nothing extra.
        if getattr(self, "record", False):
            self.last_write = w.detach()
        return lat + self.o(pooled), anchors


def topk_latents(coords, anchors, k):
    """Indices of the k nearest latent anchors for each atom. (B, N, k)"""
    d = torch.cdist(coords, anchors)                                 # (B,N,L)
    return d.topk(min(k, anchors.shape[1]), dim=-1, largest=False).indices


class LocalCrossAttention(nn.Module):
    """Each atom attends only to its k nearest latents. O(N.k), not O(N.L)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.nq = nn.LayerNorm(d_model)
        self.nc = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.nf = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, atoms, lat, idx, rel=None):
        B, N, D = atoms.shape
        k = idx.shape[-1]
        # Gather from the SMALL latent (B, L, D) with a flattened index, NOT by
        # expanding it to (B, N, L, D) first. At B=8, N=3000, L=128, D=128 the
        # expanded form is 393M elements -- 1.57 GB if the gather materialises
        # it, per layer per step. This form touches only B.L.D.
        ctx = lat.gather(1, idx.reshape(B, N * k, 1).expand(B, N * k, D)
                         ).view(B, N, k, D)
        if rel is not None:
            ctx = ctx + rel
        q = self.q(self.nq(atoms)).view(B, N, self.h, self.dk)
        kk = self.k(self.nc(ctx)).view(B, N, k, self.h, self.dk)
        vv = self.v(ctx).view(B, N, k, self.h, self.dk)
        att = torch.einsum("bnhd,bnkhd->bnhk", q, kk) / (self.dk ** 0.5)
        att = att.softmax(-1)
        if getattr(self, "record", False):
            self.last_local = (att.mean(2).detach(), idx.detach())   # (B,N,k),(B,N,k)
        out = torch.einsum("bnhk,bnkhd->bnhd", att, vv).reshape(B, N, D)
        atoms = atoms + self.o(out)
        return atoms + self.ff(self.nf(atoms))


class AtomLatentEncoder(nn.Module):
    """N atom tokens -> L latent tokens. One token per atom throughout."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = make_featurizer(cfg, use_coords=True)
        # Atoms are NOT pooled into residues. Local attention keeps this O(N.w)
        # -- dense atom self-attention would be O(N^2), the expensive kind.
        self.atom_blocks = nn.ModuleList([
            WindowedSelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout,
                                  window=max(int(cfg.attn_window), 32),
                                  n_global=int(cfg.attn_global))
            for _ in range(max(int(cfg.enc_self_layers), 1))])
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.pool = AnchorPool(cfg.d_model, cfg.n_heads)
        self.lat_blocks = nn.ModuleList([
            SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(1)])                     # L is small; O(L^2) is fine
        self.to_content = nn.Linear(cfg.d_model, max(cfg.latent_dim - 3, 1))

    def forward(self, batch, coords, n_latents=None):
        L = int(n_latents or self.cfg.n_latent_tokens)
        x = self.feat(batch, coords)
        mask = batch["mask"]
        pad = ~mask.bool()

        # The local atom attention windows over TENSOR order, which permuting
        # the input changes -- so run it in CANONICAL order instead and undo
        # the sort afterwards. Without this, permutation invariance is only
        # empirical: measured at 2e-4 on an untrained 60-atom chain, with
        # nothing keeping it there once the attention has learned to use its
        # window. Sorting makes the property structural.
        order = inv = None
        if "canonical_rank" in batch:
            key = batch["canonical_rank"] + (~mask.bool()).long() * (2 ** 30)
            order = torch.argsort(key, dim=1)              # padding sorts last
            inv = torch.argsort(order, dim=1)
            g = order.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.gather(x, 1, g)
            pad = torch.gather(pad, 1, order)

        for blk in self.atom_blocks:
            x = blk(x, key_padding_mask=pad)

        if inv is not None:
            x = torch.gather(x, 1, inv.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            pad = ~mask.bool()

        # Latent queries from a table-free index encoding, so L is free.
        idx = torch.arange(L, device=x.device)
        lat = self.q_proj(sinusoidal_index(idx, self.cfg.d_model))
        lat = lat.unsqueeze(0).expand(x.shape[0], -1, -1)

        lat, anchors = self.pool(lat, x, coords, mask)
        for blk in self.lat_blocks:
            lat = blk(lat)
        content = self.to_content(lat)
        # Anchors ride INSIDE the latent and are counted in its budget.
        return torch.cat([anchors / self.cfg.coord_scale, content], dim=-1)


class AtomLatentDecoder(nn.Module):
    """L latents -> N atom coordinates, atom-level throughout.

    Sees atom IDENTITY (element, atom name, residue index, chain) -- routing
    metadata, explicitly allowed -- and never input coordinates.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat = make_featurizer(cfg, use_coords=False)
        self.up = nn.Linear(max(cfg.latent_dim - 3, 1), cfg.d_model)
        self.anchor_proj = nn.Linear(3, cfg.d_model)
        # Pass 1: global. Every atom sees every latent; gives provisional xyz.
        self.global_blocks = nn.ModuleList([
            CrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(max(int(cfg.dec_cross_layers), 1))])
        self.head1 = nn.Sequential(nn.LayerNorm(cfg.d_model),
                                   nn.Linear(cfg.d_model, 3))
        # Pass 2: local, keyed on the provisional positions.
        self.local = bool(cfg.dec_local_cross)
        if self.local:
            self.rel_proj = nn.Linear(3, cfg.d_model)
            self.local_blocks = nn.ModuleList([
                LocalCrossAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
                for _ in range(max(int(cfg.dec_local_layers), 1))])
            self.head2 = nn.Sequential(nn.LayerNorm(cfg.d_model),
                                       nn.Linear(cfg.d_model, 3))

    def forward(self, z, batch):
        anchors = z[..., :3] * self.cfg.coord_scale
        lat = self.up(z[..., 3:]) + self.anchor_proj(anchors / self.cfg.coord_scale)

        q = self.feat(batch)                                  # identity only
        for bi, blk in enumerate(self.global_blocks):
            if getattr(self, "record", False):
                q, w = blk(q, lat, return_weights=True)
                self.last_read = w.detach()                   # (B,N,L), last layer
            else:
                q = blk(q, lat)
        coarse = self.head1(q) * self.cfg.coord_scale
        if not self.local:
            return coarse, coarse

        # Locality from the PROVISIONAL positions -- derived from the latent,
        # not read from the input. This is what keeps pass 2 leak-free.
        idx = topk_latents(coarse.detach(), anchors, self.cfg.dec_local_k)
        k = idx.shape[-1]
        B, N, _ = coarse.shape
        near = torch.gather(anchors.unsqueeze(1).expand(B, N, anchors.shape[1], 3),
                            2, idx.unsqueeze(-1).expand(B, N, k, 3))
        rel = self.rel_proj((near - coarse.unsqueeze(2)) / self.cfg.coord_scale)
        for blk in self.local_blocks:
            q = blk(q, lat, idx, rel=rel)
        return self.head2(q) * self.cfg.coord_scale, coarse


class AtomLatentAutoencoder(nn.Module):
    """Arbitrary L shared latents, atom-level in and out."""

    per_residue = False

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AtomLatentEncoder(cfg)
        self.decoder = AtomLatentDecoder(cfg)

    def encode(self, batch, n_latents=None):
        return self.encoder(batch, batch["coords"] / self.cfg.coord_scale,
                            n_latents=n_latents)

    def decode(self, z, batch):
        return self.decoder(z, batch)[0]

    def training_aux(self, batch):
        """Auxiliary supervision for the PROVISIONAL coordinates.

        Pass 2 routes each atom to the k latents whose anchors are nearest its
        pass-1 position, and that selection is taken through .detach() -- so
        the routing receives no gradient at all. Pass 1 otherwise trains only
        through the relative-offset term, which means `coarse` is never
        required to be a COORDINATE. Locality then gets computed in a space
        that was never trained to be physical space, and the measured gradient
        reaching head1 is 5% of what reaches head2.

        This makes pass 1 predict actual positions, so the neighbourhood it
        selects is a real neighbourhood. Weight it with loss.coarse.
        """
        c = getattr(self, "last_coarse", None)
        if c is None:
            return None
        from .alignment import kabsch_align_torch
        m = batch["mask"]
        aligned = kabsch_align_torch(c, batch["coords"], m)
        d2 = ((aligned - batch["coords"]) ** 2).sum(-1) * m
        return d2.sum() / m.sum().clamp_min(1.0)

    def forward(self, batch, n_latents=None):
        z = self.encode(batch, n_latents=n_latents)
        fine, coarse = self.decoder(z, batch)
        self.last_coarse = coarse            # supervised via cfg.loss coarse term
        return fine, z

    @property
    def latent_floats(self) -> int:
        return self.cfg.n_latent_tokens * self.cfg.latent_dim

    def latent_floats_for(self, n_residues: int) -> int:
        """Independent of structure size -- the entire point of this arm."""
        return self.cfg.n_latent_tokens * self.cfg.latent_dim

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
