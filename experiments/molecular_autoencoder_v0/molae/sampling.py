"""Sample a latent trajectory, decode it, and score it as physics.

The last piece before the joint-segment diffusion stage can be trained on
anything: turning a sampled latent segment back into coordinates and judging
whether the result is a plausible piece of molecular dynamics.

What has to be scored, and why each is separate
-----------------------------------------------
A latent sample is only as good as what it decodes to, and there are three
independent ways for it to be wrong:

  1. PHYSICS. Bond lengths, chirality, clashes -- per frame. A frame can be
     structurally reasonable and physically impossible.
  2. TEMPORAL CONTINUITY. Consecutive frames must be close. A segment of
     individually-plausible frames that jump around is not a trajectory, and
     no per-frame metric can see it.
  3. DIVERSITY. Frames must actually differ. This is the failure that looks
     like success: a model emitting one frame T times scores PERFECTLY on
     physics and continuity. The same trap as a codec that averages
     conformers to a mean structure, which is why `ensemble_resolution` had
     to be added to the codec gate -- and it is worth catching here before a
     training run rather than after.

Decoding needs a TEMPLATE. The codec's decoder is conditioned on atom
identity, so a sample is geometry for a KNOWN molecule -- exactly as a video
decoder needs the frame shape. That is by design, not a limitation: it is
what makes the latent carry geometry alone.
"""

from __future__ import annotations

import numpy as np
import torch

from .metrics import (build_topology_info, bond_length_error,
                      chirality_violation_rate, clash_metrics)
from .alignment import kabsch_rmsd_numpy


@torch.no_grad()
def decode_segment(codec, z, batch):
    """(T, R, latent_dim) latent -> list of (N, 3) coordinate frames.

    The codec is frozen and shared across frames: this stage produces latent
    trajectories, and turning a latent into coordinates is the codec's job.
    """
    was_training = codec.training
    codec.eval()
    frames = []
    n = int(batch["mask"][0].sum().item())
    for t in range(z.shape[0]):
        coords = codec.decode(z[t:t + 1], batch)
        frames.append(coords[0, :n].cpu().numpy().astype(np.float64))
    if was_training:
        codec.train()
    return frames


def temporal_continuity(frames):
    """Consecutive-frame RMSD: mean, max, and the jump ratio.

    A sequence of individually-plausible frames that teleport is not a
    trajectory. ``jump_ratio`` is the largest step over the mean step -- a
    smooth trajectory sits near 1-2, and a sampler that occasionally jumps
    between basins spikes it, which the mean alone hides.
    """
    if len(frames) < 2:
        return {"mean_step": float("nan"), "max_step": float("nan"),
                "jump_ratio": float("nan")}
    steps = [kabsch_rmsd_numpy(frames[i + 1], frames[i])
             for i in range(len(frames) - 1)]
    m = float(np.mean(steps))
    return {"mean_step": m, "max_step": float(np.max(steps)),
            "jump_ratio": float(np.max(steps) / m) if m > 1e-9 else float("nan")}


def diversity(frames):
    """Spread across the segment, and its ratio to the consecutive step.

    THE failure that looks like success. A model emitting one frame T times
    is physically perfect and temporally perfect; only this catches it.

    ``spread`` near 0 means collapse. ``spread_over_step`` near 1 means
    consecutive frames are as far apart as the segment's extremes -- i.e. the
    frames are independent samples rather than a path through conformational
    space, which is the OTHER way to fail while scoring well per frame.
    """
    if len(frames) < 2:
        return {"spread": float("nan"), "spread_over_step": float("nan")}
    d = [kabsch_rmsd_numpy(frames[i], frames[j])
         for i in range(len(frames)) for j in range(i + 1, len(frames))]
    spread = float(np.mean(d))
    cont = temporal_continuity(frames)
    ratio = spread / cont["mean_step"] if cont["mean_step"] > 1e-9 else float("nan")
    return {"spread": spread, "spread_over_step": ratio}


def physics(frames, sample, clash_dist=1.5):
    """Per-frame bond, chirality and clash statistics, averaged.

    Scored against the TEMPLATE's own geometry for bond lengths (the sample
    has no ground truth), so this measures whether the generated geometry is
    self-consistent chemistry, not whether it matches a target.
    """
    bonds = sample["bonds"].numpy()
    topo = build_topology_info(
        sample["atom_name_idx"].numpy(), sample["res_pos"].numpy(),
        bonds, sample["element_symbol"])
    ref = np.asarray(sample["coords"], dtype=np.float64)
    bl, ch, cl = [], [], []
    for f in frames:
        bl.append(bond_length_error(f, ref, bonds))
        ch.append(chirality_violation_rate(f, ref, topo))
        cl.append(clash_metrics(f, bonds, topo.radii, clash_dist=clash_dist)
                  ["clashes_per_1000_atoms"])
    return {"bond_length_error": float(np.mean(bl)),
            "chirality_violation_rate": float(np.mean(ch)),
            "clashes_per_1000_atoms": float(np.mean(cl))}


@torch.no_grad()
def sample_and_score(diffusion, codec, sample, batch, n_frames=8, steps=50,
                     generator=None, device=None, clash_dist=1.5):
    """Sample a latent segment, decode it, and score all three axes.

    Returns the frames alongside the metrics so a caller can write structures
    out for inspection -- a number that says "physically fine" is not the
    same as a trajectory anyone has looked at.
    """
    device = device or next(diffusion.parameters()).device
    R = int(batch["res_pos"].max().item()) + 1
    z = diffusion.sample((1, n_frames, R, codec.cfg.latent_dim), device,
                         steps=steps, generator=generator)[0]
    frames = decode_segment(codec, z, batch)
    out = {"n_frames": len(frames), "n_atoms": len(frames[0])}
    out.update(physics(frames, sample, clash_dist))
    out.update(temporal_continuity(frames))
    out.update(diversity(frames))
    return frames, out
