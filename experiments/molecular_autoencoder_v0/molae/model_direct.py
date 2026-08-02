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
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_slots = C.N_SLOTS
        self.up = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.res_pos_emb = nn.Embedding(cfg.max_res_pos, cfg.d_model)
        self.res_type_emb = nn.Embedding(C.N_RESIDUES, cfg.d_model, padding_idx=C.PAD_RESIDUE_IDX)
        self.chain_aware = bool(getattr(cfg, "dec_chain_aware", False))
        if self.chain_aware:
            self.chain_emb = nn.Embedding(cfg.max_chains, cfg.d_model)
            # Kept ALONGSIDE the global index, not instead of it: global says
            # how far into the assembly a residue sits, within-chain restores
            # the adjacency prior the global index breaks at each boundary.
            self.res_in_chain_emb = nn.Embedding(cfg.max_res_pos, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList(
            [SelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
             for _ in range(cfg.dec_self_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, self.n_slots * 3),
        )

    def forward(self, z, batch):
        """z: (B, R, latent_dim) -> coords (B, N, 3) for the batch's atom list."""
        B, R, _ = z.shape
        res_pos = batch["res_pos"]                                  # (B, N)
        device = z.device

        # Residue tokens: latent + which residue this is + what residue it is.
        idx = torch.arange(R, device=device).clamp(max=self.cfg.max_res_pos - 1)
        h = self.up(z) + self.res_pos_emb(idx).unsqueeze(0)
        # First residue-type seen at each position (identity is given, as before).
        rtype = torch.zeros(B, R, dtype=torch.long, device=device)
        rtype.scatter_(1, res_pos, batch["residue_idx"])
        h = h + self.res_type_emb(rtype)
        if self.chain_aware:
            rchain, within = residue_chain_index(
                res_pos, batch["chain_idx"], R, self.cfg.max_chains)
            h = (h + self.chain_emb(rchain)
                 + self.res_in_chain_emb(within.clamp(max=self.cfg.max_res_pos - 1)))
        h = self.ln(h)

        rmask = residue_mask(res_pos, batch["mask"], R)
        pad = ~rmask.bool()
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad)

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
