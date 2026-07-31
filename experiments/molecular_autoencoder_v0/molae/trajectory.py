"""Frame sequences -> latent segments for the joint-segment diffusion stage.

A "segment" is T consecutive frames of ONE system: identical atom composition,
differing only in coordinates. That invariant is what makes a segment stackable
into (T, R, latent_dim) and is checked rather than assumed, because the two
sources violate it in different ways if you are careless:

  * NMR ensembles -- many deposited models of one molecule. Real conformers,
    available today, but an ENSEMBLE, not a time series: the models have no
    temporal order, so they can validate that the latent space is smooth
    enough to diffuse in, and nothing about dynamics.
  * MD trajectories (mdCATH, MISATO) -- genuinely ordered in time. This is
    what the temporal axis is for.

The codec is FROZEN here. This stage learns the distribution of latent
trajectories; turning latents back into coordinates is the codec's job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dataset import sample_from_arrays, collate_fn
from .parsing import parse_structure, to_npz_dict


class InconsistentFrames(ValueError):
    """Raised when frames of a segment do not share an atom composition."""


def frames_from_nmr(path, pdb_id=None, max_frames=None, **parse_kw):
    """Every deposited model of one entry, as a list of parsed-array dicts.

    Deposited models of an NMR entry share atom composition by construction,
    which is exactly the property a segment needs -- and the reason these are
    a usable stand-in before trajectory data arrives.
    """
    import gemmi
    n_models = len(gemmi.read_structure(str(path)))
    if max_frames is not None:
        n_models = min(n_models, max_frames)
    out = []
    for i in range(n_models):
        ps = parse_structure(str(path), pdb_id=pdb_id, model_index=i, **parse_kw)
        if ps is None:
            continue
        out.append(to_npz_dict(ps))
    return out


def check_consistent(frames):
    """Verify frames differ ONLY in coordinates. Returns the atom count.

    A silent composition change across frames would misalign the latent
    sequence -- residue r in frame 0 would not be residue r in frame 1 -- and
    nothing downstream could detect it.
    """
    if not frames:
        raise InconsistentFrames("no frames")
    ref = frames[0]
    n = len(ref["coords"])
    for k, f in enumerate(frames[1:], start=1):
        if len(f["coords"]) != n:
            raise InconsistentFrames(
                f"frame {k} has {len(f['coords'])} atoms, frame 0 has {n}")
        for key in ("element_idx", "residue_idx", "atom_name_idx", "res_pos"):
            if not np.array_equal(np.asarray(f[key]), np.asarray(ref[key])):
                raise InconsistentFrames(f"frame {k} differs from frame 0 in {key}")
    return n


def segments_from_frames(frames, seg_len, stride=None, drop_last=True):
    """Slice a frame list into fixed-length segments.

    ``stride`` defaults to ``seg_len`` (non-overlapping). A shorter stride
    gives overlapping segments, which multiplies the training signal from a
    fixed trajectory -- worth having, since the matched experimental band is
    small and MD trajectories are expensive.
    """
    stride = stride or seg_len
    out = []
    for start in range(0, max(len(frames) - seg_len + 1, 0), stride):
        out.append(frames[start:start + seg_len])
    if not drop_last and len(frames) % stride and len(frames) >= seg_len:
        tail = frames[-seg_len:]
        if not out or tail is not out[-1]:
            out.append(tail)
    return out


@torch.no_grad()
def encode_segment(model, frames, device, batch_frames=8):
    """Encode a segment through the FROZEN codec -> (T, R, latent_dim).

    Frames are encoded independently: the codec has no notion of time, and
    giving it one would move temporal modelling into the wrong stage.
    """
    check_consistent(frames)
    was_training = model.training
    model.eval()
    zs = []
    for i in range(0, len(frames), batch_frames):
        chunk = frames[i:i + batch_frames]
        batch = collate_fn([sample_from_arrays(f) for f in chunk])
        gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        zs.append(model.encode(gb).cpu())
    if was_training:
        model.train()
    return torch.cat(zs, dim=0)


def collate_segments(segments):
    """Pad a list of (T, R, D) tensors to a batch plus frame/group masks."""
    B = len(segments)
    T = max(s.shape[0] for s in segments)
    R = max(s.shape[1] for s in segments)
    D = segments[0].shape[2]
    z = torch.zeros(B, T, R, D)
    frame_mask = torch.zeros(B, T)
    group_mask = torch.zeros(B, R)
    for i, s in enumerate(segments):
        t, r, _ = s.shape
        z[i, :t, :r] = s
        frame_mask[i, :t] = 1.0
        group_mask[i, :r] = 1.0
    return {"z": z, "frame_mask": frame_mask, "group_mask": group_mask}


def latent_drift(z):
    """Mean per-frame latent change along a segment: (T-1,) tensor.

    The cheap smoothness diagnostic for the stage. If consecutive frames land
    far apart in latent space relative to the spread of the whole segment, the
    trajectory is not a smooth path and diffusing over it will be hard -- that
    is a property of the CODEC, measurable before any diffusion model is
    trained, and worth knowing early because the fix would be to the codec.
    """
    if z.shape[0] < 2:
        return torch.zeros(0)
    return (z[1:] - z[:-1]).flatten(1).norm(dim=1)


def drift_ratio(z):
    """Consecutive-frame drift over segment spread. Small = smooth path.

    ~1 means consecutive frames are as far apart as the segment's extremes,
    i.e. the latent is not tracking the conformational change continuously.
    """
    d = latent_drift(z)
    if d.numel() == 0:
        return float("nan")
    spread = (z - z.mean(dim=0, keepdim=True)).flatten(1).norm(dim=1).mean()
    return float(d.mean() / spread.clamp_min(1e-8))
