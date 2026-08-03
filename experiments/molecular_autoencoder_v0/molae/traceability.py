"""Atom-level traceability through a shared bottleneck.

The latent count may be independent of the atom count. The atom
CORRESPONDENCE may not be. A codec that reproduces a plausible overall shape
while losing which output is which atom has not compressed the structure, it
has compressed a summary of it -- and RMSD alone cannot tell the two apart,
because a shape that is right on average scores well.

So this measures the address, not the geometry:

  write route     which latents each atom writes to in the encoder
  read route      which latents each output atom reads from in the decoder
  agreement       whether the two routes name the same latents for one atom
  collapse        how many atoms share an identical routing pattern
  utilisation     latents that are never written/read, or that absorb everything
  error vs route  whether diffuse routing predicts bad reconstruction
  symmetry        whether equivalent atoms are label-swapped (fine) or
                  physically stacked on one point (broken)

The last one is the sharpest test of the requirement. Symmetric atoms are
interchangeable, so a swap is not an error -- but if the model puts them at
the SAME place it has represented "there is a methyl here" instead of "these
three hydrogens are at these three positions", which is exactly the failure
mode a shared bottleneck invites.
"""

from __future__ import annotations

import numpy as np
import torch


def _entropy(p, axis=-1, eps=1e-12):
    p = p / p.sum(axis=axis, keepdims=True).clip(eps)
    return -(p * np.log(p.clip(eps))).sum(axis=axis)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def random_jaccard(L, k=8):
    """Expected top-k Jaccard for two INDEPENDENT random routes.

    E[|A n B|] = k^2 / L, so this falls as L grows -- which makes the raw
    Jaccard meaningless across cells with different L. Reported alongside the
    measured value so the comparison is always against the right baseline.
    """
    inter = (k * k) / max(L, 1)
    return float(inter / max(2 * k - inter, 1e-9))


def route_metrics(write, read, k=8):
    """``write`` (L, N) encoder attention, ``read`` (N, L) decoder attention.

    Returns the address-level diagnostics. Both are normalised per ATOM, so a
    latent that simply attends to everything does not look like a route.
    """
    out = {}
    L, N = write.shape
    w_atom = write.T                                    # (N, L)
    w_atom = w_atom / w_atom.sum(1, keepdims=True).clip(1e-12)
    r_atom = read / read.sum(1, keepdims=True).clip(1e-12)

    kk = min(k, L)
    w_top = np.argsort(-w_atom, axis=1)[:, :kk]
    r_top = np.argsort(-r_atom, axis=1)[:, :kk]

    # (3) do the encoder and decoder name the same latents for one atom?
    jac = [len(set(a) & set(b)) / len(set(a) | set(b))
           for a, b in zip(w_top, r_top)]
    out["route_agreement_jaccard"] = float(np.mean(jac))
    out["route_agreement_random"] = random_jaccard(L, kk)
    out["route_agreement_over_random"] = (
        out["route_agreement_jaccard"] / max(out["route_agreement_random"], 1e-9))
    out["route_agreement_top1"] = float(np.mean(w_top[:, 0] == r_top[:, 0]))
    out["route_top1_random"] = 1.0 / max(L, 1)

    # (4) how many atoms are indistinguishable by their route?
    pats = [tuple(sorted(t.tolist())) for t in r_top]
    uniq, counts = np.unique(np.array([hash(p) for p in pats]), return_counts=True)
    out["distinct_read_patterns"] = int(len(uniq))
    out["distinct_pattern_frac"] = float(len(uniq) / max(N, 1))
    out["max_atoms_per_pattern"] = int(counts.max())
    out["atoms_in_shared_pattern_frac"] = float(counts[counts > 1].sum() / max(N, 1))

    # (5) unused or overloaded latents, by mass rather than by top-k
    wm, rm = w_atom.sum(0), r_atom.sum(0)
    out["write_gini"] = gini(wm)
    out["read_gini"] = gini(rm)
    out["write_unused_frac"] = float((wm < 0.01 * wm.mean()).mean())
    out["read_unused_frac"] = float((rm < 0.01 * rm.mean()).mean())

    # (6) routing concentration per atom
    out["read_entropy_mean"] = float(_entropy(r_atom).mean())
    out["read_entropy_max"] = float(np.log(L))
    out["read_entropy_frac"] = out["read_entropy_mean"] / max(np.log(L), 1e-9)
    out["write_entropy_frac"] = float(_entropy(w_atom).mean() / max(np.log(L), 1e-9))
    return out, r_atom


def error_vs_routing(err, r_atom):
    """(6) does a diffuse route predict a badly placed atom?"""
    ent = _entropy(r_atom)
    if len(err) < 3 or ent.std() < 1e-9 or err.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(err, ent)[0, 1])


def symmetry_collapse(pred, target, classes):
    """(7) equivalent atoms: label-swapped, or physically stacked?

    For each symmetry class with >= 2 members, compare the mean pairwise
    spread of the PREDICTED positions against the TARGET's. A swap preserves
    the spread; collapsing them onto one point destroys it.

    Returns the mean spread ratio -- 1.0 is faithful, 0.0 is fully collapsed --
    and the fraction of classes below half the target spread.
    """
    ratios = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        if len(idx) < 2:
            continue
        dt = np.linalg.norm(target[idx][:, None] - target[idx][None], axis=-1)
        dp = np.linalg.norm(pred[idx][:, None] - pred[idx][None], axis=-1)
        m = ~np.eye(len(idx), dtype=bool)
        if dt[m].mean() < 1e-6:
            continue
        ratios.append(dp[m].mean() / dt[m].mean())
    if not ratios:
        return {"symmetry_spread_ratio": float("nan"),
                "symmetry_collapsed_frac": float("nan"),
                "n_symmetry_classes": 0}
    r = np.asarray(ratios)
    return {"symmetry_spread_ratio": float(r.mean()),
            "symmetry_collapsed_frac": float((r < 0.5).mean()),
            "n_symmetry_classes": int(len(r))}


class RouteRecorder:
    """Turn on attention recording for one forward pass, then turn it off."""

    def __init__(self, model):
        self.model = model
        self.mods = []

    def __enter__(self):
        m = self.model
        for mod in (getattr(m.encoder, "pool", None), m.decoder,
                    *getattr(m.decoder, "local_blocks", [])):
            if mod is not None:
                mod.record = True
                self.mods.append(mod)
        return self

    def __exit__(self, *exc):
        for mod in self.mods:
            mod.record = False

    def write(self):
        w = getattr(self.model.encoder.pool, "last_write", None)
        return None if w is None else w[0].cpu().numpy()      # (L, N)

    def read(self):
        r = getattr(self.model.decoder, "last_read", None)
        return None if r is None else r[0].cpu().numpy()      # (N, L)
