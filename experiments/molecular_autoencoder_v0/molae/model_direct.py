"""Direct per-residue readout decoder — the "make it work" architecture.

Diagnosis this is built from:

  * A flat MLP autoencoder on fixed-size aligned backbones BEATS PCA
    (0.31-0.66 A at 2/4/8 floats). Neural compression is fine.
  * The same-sized latent in our Perceiver pipeline loses to a 0-float
    mean-shape baseline on whole proteins.

The structural difference: in the MLP, output slot *i* simply IS atom *i* --
the latent-to-position mapping is free. In our decoder, each per-atom identity
query has to *find* its own coordinates by attending over a latent that encodes
the whole protein. That is a hard credit-assignment problem, and it is the most
likely cause of the observed mean-collapse (near-random chirality, ~3.6 A bond
errors, tiny per-fold RMSD spread).

This decoder removes the lookup entirely while keeping variable size:

    per-residue latent z_r  ->  self-attention over residue tokens
                            ->  a head that emits ALL atom slots for residue r
                            ->  atom i reads slot (res_pos[i], atom_name_idx[i])

Every atom therefore has a direct, dedicated path from its own residue's
latent, exactly like the MLP's fixed output slot, but the number of residues
can vary. Slots are indexed by atom NAME (a fixed vocabulary), so the mapping
is unambiguous and needs no learned matching.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import constants as C
from .model import ModelConfig, SelfAttention, Bottleneck
from .model_perresidue import PerResidueEncoder, residue_mask, n_residues
from .scaling import PositionEncoding, make_group_attention


def rot_from_6d(x):
    """Continuous 6D rotation parameterisation (Zhou et al.) -> (..., 3, 3).

    Gram-Schmidt on two predicted 3-vectors. Chosen over quaternions or Euler
    angles because those are discontinuous as functions of the rotation, which
    puts a seam in the thing a network has to regress; 6D has none.
    """
    a, b = x[..., :3], x[..., 3:]
    e1 = torch.nn.functional.normalize(a, dim=-1, eps=1e-6)
    b = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = torch.nn.functional.normalize(b, dim=-1, eps=1e-6)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)


def residue_chain_index(res_pos, chain_idx, R: int, max_chains: int):
    """Per-residue chain id, and the residue's position WITHIN that chain.

    Both are derived from tensors the batch already carries; nothing new has
    to be parsed or stored. Chains are laid out contiguously in ``res_pos``,
    so a chain's first global residue index is the minimum over its residues,
    and the within-chain position is the offset from it.
    """
    B = res_pos.shape[0]
    device = res_pos.device
    rchain = torch.zeros(B, R, dtype=torch.long, device=device)
    rchain.scatter_(1, res_pos, chain_idx)
    rchain = rchain.clamp(max=max_chains - 1)

    idxR = torch.arange(R, device=device).unsqueeze(0).expand(B, R)
    # amin over residue indices per chain. Padding residues never lower a
    # minimum (they sit at HIGHER indices than the chain's real first residue),
    # so they cannot corrupt the offset.
    first = torch.full((B, max_chains), R, dtype=torch.long, device=device)
    first = first.scatter_reduce(1, rchain, idxR, reduce="amin", include_self=True)
    within = idxR - first.gather(1, rchain)
    return rchain, within.clamp_min(0)


class DirectResidueDecoder(nn.Module):
    """Emit a coordinate slot per (residue, atom-name); atoms gather their own.

    Chain awareness (``cfg.dec_chain_aware``) closes an asymmetry, not a
    feature gap. The ENCODER's featuriser already carries a chain embedding,
    with the reason stated on it: without one, "two chains are only
    distinguishable by their global res_pos, so the model cannot tell a chain
    break from a peptide bond". Every word of that applies to this decoder,
    which had neither a chain embedding nor any within-chain index -- it saw a
    single sequential ``res_pos`` running across the whole assembly.

    That is worse than absent information, it is wrong information. The
    ``res_pos`` embedding's value is the prior that residue r sits ~3.8A from
    residue r+1; that holds along a chain and is FALSE at every chain
    boundary, and nothing marked where the boundaries were. On a single chain
    the prior is true everywhere -- which is precisely the asymmetry that needs
    explaining: 0.79A single-chain against ~5.9A on complexes.

    Off by default, so every prior ``direct`` result stays reproducible.

    Frame readout (``cfg.dec_frames``) attacks a different and larger term.
    The plain head maps one residue token to 14x3 ABSOLUTE coordinates through
    a single linear layer at a single output scale. That layer has to express
    both a ~50A placement and a 1.5A bond length in the same units, and it
    emits the placement independently 14 times per residue -- so placement
    error is injected once per atom and corrupts the internal geometry it
    shares an output with.

    Instead predict, per residue: a local atom cloud at ``local_scale`` (a few
    angstrom), a rotation, and a translation at ``coord_scale``. Placement
    then flows through 3 numbers, not 42, and moves the residue RIGIDLY, so it
    cannot deform it.

    The measurement that ranks this above chain awareness: at residue
    granularity, error decomposes 5.884A global into 1.829A residue-internal
    and the rest placement -- 69% linearly, ~90% of the squared error. And the
    SINGLE-CHAIN members of complex val sit at 4.126A, implying ~3.7A of
    placement error where there is no chain boundary to blame, against a
    multi-minus-single differential of only 2.11A. Chain awareness can address
    the 2.11; only frames reach the 3.7.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_slots = C.N_SLOTS
        self.up = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.res_pos_emb = PositionEncoding(
            cfg.d_model, cfg.max_res_pos,
            unbounded=getattr(cfg, "unbounded_positions", False))
        self.res_type_emb = nn.Embedding(C.N_RESIDUES, cfg.d_model, padding_idx=C.PAD_RESIDUE_IDX)
        self.chain_aware = bool(getattr(cfg, "dec_chain_aware", False))
        # dec_chain_aware bundles TWO things, and they are not the same claim.
        # chain_emb marks WHICH chain (2k params, no-op on a single chain).
        # res_in_chain_emb is a SECOND position table (131k params) that on a
        # single chain is an exact duplicate of the global one -- pure extra
        # positional capacity with zero chain content. Bundled, a single-chain
        # improvement cannot be attributed. These split them.
        self.use_chain_emb = self.chain_aware and bool(
            getattr(cfg, "dec_chain_emb", True))
        self.use_res_in_chain = self.chain_aware and bool(
            getattr(cfg, "dec_res_in_chain_emb", True))
        if self.use_chain_emb:
            self.chain_emb = nn.Embedding(cfg.max_chains, cfg.d_model)
            # Kept ALONGSIDE the global index, not instead of it: global says
            # how far into the assembly a residue sits, within-chain restores
            # the adjacency prior the global index breaks at each boundary.
        if self.use_res_in_chain:
            self.res_in_chain_emb = PositionEncoding(
                cfg.d_model, cfg.max_res_pos,
                unbounded=getattr(cfg, "unbounded_positions", False))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.blocks = make_group_attention(cfg, cfg.dec_self_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, self.n_slots * 3),
        )
        self.frames = bool(getattr(cfg, "dec_frames", False))
        if self.frames:
            # Translation is emitted at coord_scale (tens of angstrom), local
            # atom offsets at local_scale (a few angstrom). That separation is
            # the entire point -- see the class docstring.
            self.trans_head = nn.Linear(cfg.d_model, 3)
            self.rot_head = nn.Linear(cfg.d_model, 6)
            nn.init.zeros_(self.rot_head.weight)
            with torch.no_grad():   # 6D -> identity rotation at initialisation
                self.rot_head.bias.copy_(
                    torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, z, batch):
        """z: (B, R, latent_dim) -> coords (B, N, 3) for the batch's atom list."""
        B, R, _ = z.shape
        res_pos = batch["res_pos"]                                  # (B, N)
        device = z.device

        # Residue tokens: latent + which residue this is + what residue it is.
        idx = torch.arange(R, device=device)
        h = self.up(z) + self.res_pos_emb(idx).unsqueeze(0)
        # First residue-type seen at each position (identity is given, as before).
        rtype = torch.zeros(B, R, dtype=torch.long, device=device)
        rtype.scatter_(1, res_pos, batch["residue_idx"])
        h = h + self.res_type_emb(rtype)
        if self.chain_aware:
            rchain, within = residue_chain_index(
                res_pos, batch["chain_idx"], R, self.cfg.max_chains)
            if self.use_chain_emb:
                h = h + self.chain_emb(rchain)
            if self.use_res_in_chain:
                h = h + self.res_in_chain_emb(within)
        h = self.ln(h)

        rmask = residue_mask(res_pos, batch["mask"], R)
        pad = ~rmask.bool()
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad)

        if self.frames:
            # Local atom cloud in the residue's own frame, then place it.
            local = self.head(h).view(B, R, self.n_slots, 3) * self.cfg.local_scale
            rot = rot_from_6d(self.rot_head(h))                  # (B, R, 3, 3)
            t = self.trans_head(h) * self.cfg.coord_scale        # (B, R, 3)
            slots = torch.einsum("brsi,brij->brsj", local, rot) + t.unsqueeze(2)
        else:
            slots = self.head(h).view(B, R, self.n_slots, 3) * self.cfg.coord_scale

        # Each atom reads its own dedicated slot -- no attention lookup.
        #
        # The slot is addressed by `slot_idx`, which for a protein residue IS
        # atom_name_idx -- so this is bit-identical to the original
        # `res_pos * n_slots + atom_name_idx` on protein-only data (asserted in
        # tests/test_ligands.py). A ligand has no vocabulary atom name, so its
        # atoms carry an ORDINAL slot within their group instead, which is what
        # lets non-polymer atoms share this decoder at all.
        flat = slots.view(B, R * self.n_slots, 3)
        slot = batch["slot_idx"] if "slot_idx" in batch else batch["atom_name_idx"]
        gather_idx = (res_pos * self.n_slots + slot).clamp(max=R * self.n_slots - 1)
        return torch.gather(flat, 1, gather_idx.unsqueeze(-1).expand(-1, -1, 3))


class DirectAutoencoder(nn.Module):
    """Per-residue encoder + direct-readout decoder."""

    per_residue = True

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PerResidueEncoder(cfg)
        self.bottleneck = Bottleneck(cfg)
        self.decoder = DirectResidueDecoder(cfg)

    def encode(self, batch):
        res_feats, _ = self.encoder(batch, batch["coords"] / self.cfg.coord_scale)
        return self.bottleneck.encode(res_feats)

    def decode(self, z, batch):
        return self.decoder(z, batch)

    def forward(self, batch):
        z = self.encode(batch)
        return self.decode(z, batch), z

    @property
    def latent_floats(self) -> int:
        return self.cfg.latent_dim

    def latent_floats_for(self, n_residues: int) -> int:
        return self.cfg.latent_dim * max(int(n_residues), 1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
