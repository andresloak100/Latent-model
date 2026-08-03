"""Masked geometric denoising — the objective that makes a wide latent honest.

Why this is not optional at high latent_dim
-------------------------------------------
Capacity and task are independent. A latent with three or more floats per
effective atom CAN hold the coordinates verbatim, and plain reconstruction
gives it no reason not to: copying scores perfectly. Shrinking the latent is
the wrong lever, because it also removes the capacity we want for hard
structures. The right lever is to make copying insufficient.

So the encoder never sees the masked atoms' positions at all, while the loss
is scored against the CLEAN structure including them. Those atoms can only be
reconstructed by inference from the rest, which is exactly the capability the
latent is supposed to hold.

The repo has a prior null on this -- "masked reconstruction is neutral on the
direct codec (1.016 vs 1.020)" -- and it does not apply here. That was
measured at per-residue d=8, i.e. 3x compression, where the model could not
copy in the first place, so forbidding copying changed nothing. The objection
bites precisely where the bottleneck is wide.

Three corruptions, because they test different things
-----------------------------------------------------
``atom``    scattered individual atoms. Reconstructible from immediate
            covalent neighbours -- the easy case, and mostly a test of local
            chemistry.
``region``  contiguous spans of groups. No local neighbours survive, so this
            can only be solved from global context. This is the one that
            distinguishes a model that has learned structure from one that
            has learned interpolation.
``noise``   Gaussian displacement of the VISIBLE atoms. Stops the model
            treating visible coordinates as exact, which is what makes the
            latent robust enough to be worth diffusing in later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskingConfig:
    """Off by default: every existing result must remain reproducible."""
    atom_frac: float = 0.0        # fraction of atoms masked individually
    region_frac: float = 0.0      # fraction of atoms masked as contiguous spans
    region_span: int = 8          # groups per contiguous span
    noise_std: float = 0.0        # angstrom, applied to VISIBLE atoms only

    @property
    def enabled(self) -> bool:
        return (self.atom_frac > 0 or self.region_frac > 0 or self.noise_std > 0)


def _region_mask(res_pos, mask, frac, span, generator=None):
    """Mask whole contiguous runs of groups, not scattered atoms."""
    B, N = res_pos.shape
    out = torch.zeros(B, N, dtype=torch.bool, device=res_pos.device)
    if frac <= 0:
        return out
    for b in range(B):
        valid = mask[b] > 0.5
        if not bool(valid.any()):
            continue
        gmax = int(res_pos[b][valid].max()) + 1
        target = int(frac * int(valid.sum()))
        if target <= 0:
            continue
        got, guard = 0, 0
        while got < target and guard < 8 * max(gmax, 1):
            guard += 1
            start = int(torch.randint(0, max(gmax, 1), (1,), generator=generator,
                                      device=res_pos.device))
            sel = ((res_pos[b] >= start) & (res_pos[b] < start + span) & valid
                   & ~out[b])
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            if got + n_sel > target:
                # Trim the final span instead of swallowing it whole. Without
                # this the achieved fraction overshoots by up to one span --
                # measured at 2x for region_frac=0.1 -- so the number in the
                # config would not be the number in the experiment.
                keep = target - got
                idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)[:keep]
                sel = torch.zeros_like(sel)
                sel[idx] = True
                n_sel = keep
            got += n_sel
            out[b] |= sel
    return out


def apply_masking(coords, mask, res_pos, cfg: MaskingConfig, generator=None):
    """Corrupt the ENCODER's view. Returns ``(corrupted, is_masked)``.

    ``is_masked`` marks atoms whose position was withheld entirely, so
    reconstruction can be reported separately for them (the only way to show
    the model inferred rather than copied).

    Coordinates only -- never identity. The decoder is conditioned on atom
    identity by design, so masking that would change the task rather than
    make it harder.
    """
    B, N, _ = coords.shape
    valid = mask > 0.5
    is_masked = torch.zeros(B, N, dtype=torch.bool, device=coords.device)

    if cfg.region_frac > 0:
        is_masked |= _region_mask(res_pos, mask, cfg.region_frac,
                                  cfg.region_span, generator)
    if cfg.atom_frac > 0:
        r = torch.rand(B, N, device=coords.device, generator=generator)
        is_masked |= (r < cfg.atom_frac) & valid
    is_masked &= valid                       # padding is not "masked"

    out = coords.clone()
    if cfg.noise_std > 0:                    # visible atoms only
        noise = torch.randn(coords.shape, device=coords.device,
                            generator=generator) * cfg.noise_std
        out = out + noise * (valid & ~is_masked).unsqueeze(-1)
    out = out * (~is_masked).unsqueeze(-1)   # withhold masked positions
    return out, is_masked


@torch.no_grad()
def split_rmsd(pred, target, mask, is_masked, align_fn):
    """Aligned RMSD reported for masked / visible / all atoms.

    ONE superposition, computed over all real atoms, then the error is
    partitioned. Superimposing the masked atoms on their own would let a rigid
    motion absorb exactly the error being measured, and would flatter the
    number that carries the whole claim.
    """
    out = {}
    B = pred.shape[0]
    # align_fn returns per-sample RMSD; here we need per-atom error under the
    # shared alignment, so the caller passes an aligner that yields coords.
    aligned = align_fn(pred, target, mask)                 # (B, N, 3)
    err = ((aligned - target) ** 2).sum(-1)                # (B, N)
    for name, sel in (("masked", is_masked),
                      ("visible", (mask > 0.5) & ~is_masked),
                      ("all", mask > 0.5)):
        n = sel.sum()
        out[name] = (float(torch.sqrt((err * sel).sum() / n)) if n > 0
                     else float("nan"))
        out[f"n_{name}"] = int(n)
    return out
