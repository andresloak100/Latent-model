"""Joint-segment latent diffusion — the "video" stage.

This is the part of the plan nothing published does. Positioning, as of the
literature check:

  * Structure Language Models (ICLR'25) generate in a LATENT space, but a
    discrete one, autoregressively, and for STATIC conformations.
  * ATMOS (2026) models atom-level MD trajectories, but generates in
    COORDINATE space and decodes frames AUTOREGRESSIVELY, one at a time.

Neither does continuous all-atom latent diffusion over a whole trajectory
segment at once. That joint-over-a-segment property is exactly what modern
video diffusion does, and it is what buys long-horizon coherence and sampling
speed: a T-frame segment costs one denoising trajectory, not T of them, and
frame 0 is conditioned on frame T rather than only the reverse.

Shape and cost
--------------
The codec's latent is PER RESIDUE, so a segment is (T, R, latent_dim) -- the
direct analogue of a video's (frames, tokens, channels). At T=16, R=100,
latent_dim=8 that is 12,800 floats, which is small.

Attention is FACTORISED into spatial (over R, within a frame) and temporal
(over T, within a residue), alternating. Full joint attention over T*R tokens
would be O((TR)^2); factorised is O(T.R^2 + R.T^2). For T=16, R=100 that is
2.6M versus 2.56M -- a wash here -- but at R=400 it is 10x cheaper, and the
saving grows with protein size, which is the direction this has to scale.
Factorised is NOT autoregressive: every frame still sees every other frame
through the temporal blocks, in both directions.

The latent for a frame is produced by a FROZEN codec. This stage never
touches coordinates; it learns the distribution of latent trajectories, and
decoding is the codec's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .flow import timestep_embedding
from .model_perceiver_direct import sinusoidal_index


@dataclass
class LatentVideoConfig:
    latent_dim: int = 8          # must match the codec's bottleneck
    d_model: int = 256
    n_heads: int = 8
    depth: int = 6               # each is one spatial + one temporal block
    ff_mult: int = 4
    dropout: float = 0.0
    # Sinusoidal, not learned: segment length and protein size are then free
    # parameters at inference, exactly as the codec's latent count is.
    max_frames_hint: int = 64


class AdaLN(nn.Module):
    """DiT-style modulation: the diffusion time sets scale/shift/gate.

    Zero-initialised output, so every block starts as the identity and the
    model begins training as a well-behaved residual stream.
    """

    def __init__(self, d_model: int, n_groups: int = 3):
        super().__init__()
        self.n_groups = n_groups
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(d_model, n_groups * d_model))
        nn.init.zeros_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, cond):
        return self.net(cond).chunk(self.n_groups, dim=-1)


class AxisAttention(nn.Module):
    """Self-attention along ONE axis of a (B, T, R, d) tensor.

    ``axis="spatial"`` attends over residues within each frame; ``"temporal"``
    attends over frames within each residue. The tensor is folded so the
    attended axis is the sequence dimension and everything else joins the
    batch, which is what makes the factorisation cheap.
    """

    def __init__(self, cfg: LatentVideoConfig, axis: str):
        super().__init__()
        assert axis in ("spatial", "temporal")
        self.axis = axis
        self.norm = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ada = AdaLN(cfg.d_model, 3)

    def forward(self, x, cond, group_mask, frame_mask):
        B, T, R, D = x.shape
        shift, scale, gate = self.ada(cond)              # (B, d) each
        h = self.norm(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]

        if self.axis == "spatial":
            h = h.reshape(B * T, R, D)
            pad = (~group_mask.bool()).repeat_interleave(T, dim=0)     # (B*T, R)
        else:
            h = h.permute(0, 2, 1, 3).reshape(B * R, T, D)
            pad = (~frame_mask.bool()).repeat_interleave(R, dim=0)     # (B*R, T)

        # A fully-masked row makes softmax produce NaN. Any such row is pure
        # padding whose output is discarded, so keep one key alive to stay
        # finite rather than poisoning the batch.
        pad = pad.clone()
        pad[pad.all(dim=1), 0] = False

        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        if self.axis == "spatial":
            a = a.reshape(B, T, R, D)
        else:
            a = a.reshape(B, R, T, D).permute(0, 2, 1, 3)
        return x + gate[:, None, None, :] * a


class FeedForward(nn.Module):
    def __init__(self, cfg: LatentVideoConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * cfg.ff_mult), nn.GELU(),
            nn.Linear(cfg.d_model * cfg.ff_mult, cfg.d_model),
        )
        self.ada = AdaLN(cfg.d_model, 3)

    def forward(self, x, cond):
        shift, scale, gate = self.ada(cond)
        h = self.norm(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        return x + gate[:, None, None, :] * self.net(h)


class SegmentDiT(nn.Module):
    """Denoiser over a whole latent segment (B, T, R, latent_dim)."""

    def __init__(self, cfg: LatentVideoConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.t_embed = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList()
        for _ in range(cfg.depth):
            self.blocks.append(nn.ModuleList([
                AxisAttention(cfg, "spatial"),
                AxisAttention(cfg, "temporal"),
                FeedForward(cfg),
            ]))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, t, group_mask=None, frame_mask=None):
        """z: (B, T, R, latent_dim), t: (B,) in [0, 1] -> velocity, same shape."""
        B, T, R, _ = z.shape
        dev = z.device
        if group_mask is None:
            group_mask = torch.ones(B, R, device=dev)
        if frame_mask is None:
            frame_mask = torch.ones(B, T, device=dev)

        x = self.inp(z)
        # Sinusoidal so T and R are unbounded and free at inference.
        x = x + sinusoidal_index(torch.arange(R, device=dev), self.cfg.d_model)[None, None]
        x = x + sinusoidal_index(torch.arange(T, device=dev), self.cfg.d_model)[None, :, None]
        cond = self.t_embed(timestep_embedding(t, self.cfg.d_model))

        for spatial, temporal, ff in self.blocks:
            x = spatial(x, cond, group_mask, frame_mask)
            x = temporal(x, cond, group_mask, frame_mask)
            x = ff(x, cond)
        return self.out(self.out_norm(x))


class LatentVideoDiffusion(nn.Module):
    """Rectified-flow diffusion over latent segments.

    Rectified flow rather than DDPM for the same reason flow.py uses it: the
    target is a straight-line velocity, which integrates in few steps and has
    no noise schedule to tune.
    """

    def __init__(self, cfg: LatentVideoConfig):
        super().__init__()
        self.cfg = cfg
        self.net = SegmentDiT(cfg)

    def training_loss(self, z1, group_mask=None, frame_mask=None, generator=None):
        """z1: (B, T, R, latent_dim) clean latent segment from the frozen codec."""
        B, T, R, D = z1.shape
        dev = z1.device
        z0 = torch.randn(z1.shape, device=dev, generator=generator)
        t = torch.rand(B, device=dev, generator=generator)
        zt = (1 - t)[:, None, None, None] * z0 + t[:, None, None, None] * z1
        target = z1 - z0
        pred = self.net(zt, t, group_mask, frame_mask)

        w = torch.ones(B, T, R, 1, device=dev)
        if group_mask is not None:
            w = w * group_mask[:, None, :, None]
        if frame_mask is not None:
            w = w * frame_mask[:, :, None, None]
        err = ((pred - target) ** 2 * w).sum() / w.sum().clamp_min(1.0) / D
        return err, {"flow_mse": float(err.detach())}

    @torch.no_grad()
    def sample(self, shape, device, steps=50, group_mask=None, frame_mask=None,
               generator=None):
        """Integrate dz/dt = v from noise at t=0 to a sample at t=1.

        The WHOLE segment is denoised together: there is no frame ordering in
        this loop and no conditioning on previously generated frames.
        """
        z = torch.randn(shape, device=device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i * dt, device=device)
            z = z + self.net(z, t, group_mask, frame_mask) * dt
        return z

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
