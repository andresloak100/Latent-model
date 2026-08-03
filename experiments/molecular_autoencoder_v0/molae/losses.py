"""Differentiable reconstruction losses.

Each term operates on a single structure's real atoms (N, 3); the
``LossComputer`` sums them over a batch (proteins have different sizes, so a
per-sample loop is clearer and correct than padded batched ops). All terms
are rotation/translation invariant: the coordinate term Kabsch-aligns first,
the rest are internal-geometry functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .alignment import kabsch_align_torch


def coord_loss_single(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared per-atom deviation after rigid alignment. (scalar)"""
    mask = torch.ones(pred.shape[0], device=pred.device, dtype=pred.dtype).unsqueeze(0)
    aligned = kabsch_align_torch(pred.unsqueeze(0), target.unsqueeze(0), mask)[0]
    return ((aligned - target) ** 2).sum(dim=-1).mean()


def distance_loss_single(pred, target, max_atoms=1200):
    n = pred.shape[0]
    if n < 2:  # no atom pairs -> empty reduction would be NaN
        return pred.new_zeros(())
    if n > max_atoms:
        idx = torch.linspace(0, n - 1, max_atoms, device=pred.device).long()
        pred, target = pred[idx], target[idx]
    dp = torch.cdist(pred, pred)
    dt = torch.cdist(target, target)
    iu = torch.triu_indices(pred.shape[0], pred.shape[0], offset=1, device=pred.device)
    return F.smooth_l1_loss(dp[iu[0], iu[1]], dt[iu[0], iu[1]], beta=1.0)


def bond_loss_single(pred, target, bonds):
    if bonds.shape[0] == 0:
        return pred.new_zeros(())
    lp = torch.linalg.norm(pred[bonds[:, 0]] - pred[bonds[:, 1]], dim=1)
    lt = torch.linalg.norm(target[bonds[:, 0]] - target[bonds[:, 1]], dim=1)
    return F.smooth_l1_loss(lp, lt, beta=0.1)


def angle_triples(bonds, n_atoms, max_per_centre=4, max_angles=4000):
    """Index triples (i, j, k) for every bond angle, built without Python loops.

    These depend ONLY on the bond graph, which never changes for a structure.
    The first version rebuilt them inside the training step with a triple
    nested Python loop and per-element int() conversions -- py-spy caught the
    training thread sitting in it, GIL held and the GPU at 2-4% utilisation.
    So this is both vectorised AND cacheable: compute once per structure, reuse
    for every step of every epoch.

    Neighbours are capped at ``max_per_centre`` so a high-degree atom cannot
    dominate; the cap makes the pair set a fixed small number of columns, which
    is what allows the vectorised form.
    """
    if bonds.numel() == 0 or bonds.shape[0] < 2:
        return bonds.new_zeros((0, 3))
    # Drop self-loops and undirected duplicates first. A self-loop makes an
    # atom its own neighbour, and a repeated bond lists the same neighbour
    # twice -- both produce degenerate triples (i == j, or i == k) that
    # contribute nothing to the loss while consuming slots under max_angles.
    bonds = bonds[bonds[:, 0] != bonds[:, 1]]
    if bonds.shape[0] < 2:
        return bonds.new_zeros((0, 3))
    lo = torch.minimum(bonds[:, 0], bonds[:, 1])
    hi = torch.maximum(bonds[:, 0], bonds[:, 1])
    key = torch.unique(lo * n_atoms + hi)
    bonds = torch.stack([key // n_atoms, key % n_atoms], dim=1)
    e = torch.cat([bonds, bonds.flip(1)], dim=0)
    centre, other = e[:, 0], e[:, 1]
    order = torch.argsort(centre, stable=True)
    centre, other = centre[order], other[order]

    # Rank of each edge within its centre, without a loop: subtract the index
    # of the centre's first edge from the running position.
    counts = torch.bincount(centre, minlength=n_atoms)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(centre.numel(), device=bonds.device) - starts[centre]

    keep = rank < max_per_centre
    centre, other, rank = centre[keep], other[keep], rank[keep]

    # Padded neighbour table (n_atoms, max_per_centre).
    nbr = torch.full((n_atoms, max_per_centre), -1, dtype=torch.long,
                     device=bonds.device)
    nbr[centre, rank] = other

    a, c = torch.triu_indices(max_per_centre, max_per_centre, offset=1,
                              device=bonds.device)
    ii = nbr[:, a]                                     # (n_atoms, n_pairs)
    kk = nbr[:, c]
    jj = torch.arange(n_atoms, device=bonds.device).unsqueeze(1).expand_as(ii)
    valid = (ii >= 0) & (kk >= 0)
    triples = torch.stack([ii[valid], jj[valid], kk[valid]], dim=1)
    return triples[:max_angles]


def angle_loss_single(pred, target, bonds, max_angles=4000, triples=None):
    """Bond ANGLES, from pairs of bonds sharing a central atom.

    Bond lengths alone leave a structure free to fold at every joint: a chain
    with perfect lengths and wrong angles is geometrically wrong everywhere
    while scoring perfectly on the bond term. Angles are the cheapest signal
    that constrains local shape rather than local scale, which matters most
    for the masked atoms, whose lengths can be satisfied by a hinge in the
    wrong direction.

    Compared as cosines to avoid the derivative blow-up of arccos near 0 and
    pi -- collinear triples are common (carbonyls, aromatics) and would
    otherwise dominate the gradient.

    Pass ``triples`` precomputed to skip rebuilding them every step.
    """
    if triples is None:
        triples = angle_triples(bonds, pred.shape[0], max_angles=max_angles)
    if triples.numel() == 0:
        return pred.new_zeros(())
    ii, jj, kk = triples[:, 0], triples[:, 1], triples[:, 2]

    def cosines(x):
        return F.cosine_similarity(x[ii] - x[jj], x[kk] - x[jj], dim=1, eps=1e-6)

    return F.smooth_l1_loss(cosines(pred), cosines(target), beta=0.1)


def knn_pairs(target, k=16):
    """Each atom's k nearest TARGET neighbours, as flat index pairs.

    Step-invariant, exactly like the angle triples: the neighbour SET depends
    only on the target structure, which never changes. The O(N^2) cdist that
    finds it was being paid every step -- 55 ms per 3000-atom structure, 7.7
    hours over a 900-epoch run -- to recompute an identical answer.

    Caching it also removes the need to subsample. The old version drew a
    random ``max_atoms`` subset per call to bound the cdist; once the cdist is
    paid once per structure there is nothing to bound, so every atom keeps its
    real neighbours instead of a fresh random slice each step.
    """
    n = target.shape[0]
    if n < 3:
        return (target.new_zeros(0, dtype=torch.long),
                target.new_zeros(0, dtype=torch.long))
    kk = min(k + 1, n)
    idx = torch.cdist(target, target).topk(kk, largest=False).indices[:, 1:]
    rows = torch.arange(n, device=target.device).unsqueeze(1).expand_as(idx)
    return rows.reshape(-1), idx.reshape(-1)


def local_distance_loss_single(pred, target, k=16, max_atoms=1200, knn=None):
    """Pairwise distances to each atom's k nearest TARGET neighbours.

    The plain distance term subsamples pairs uniformly, so at N atoms it spends
    almost all of its budget on far-apart pairs whose distance is easy and
    uninformative. Local neighbourhoods are where clashes, packing and covalent
    geometry live. Neighbours are chosen on the TARGET so the set does not
    drift as the prediction moves.

    Pass ``knn`` precomputed to skip the O(N^2) search every step.
    """
    if knn is None:
        n = pred.shape[0]
        if n > max_atoms:
            sel = torch.randperm(n, device=pred.device)[:max_atoms]
            pred, target = pred[sel], target[sel]
        knn = knn_pairs(target, k=k)
    rows, cols = knn
    if rows.numel() == 0:
        return pred.new_zeros(())
    dp = torch.linalg.norm(pred[rows] - pred[cols], dim=-1)
    dt = torch.linalg.norm(target[rows] - target[cols], dim=-1)
    return F.smooth_l1_loss(dp, dt, beta=0.1)


def clash_loss_single(pred, bonds, clash_dist=1.5, max_atoms=1200):
    """Hinge penalty on non-bonded atom pairs closer than ``clash_dist``.

    Takes the SPARSE (M, 2) bond list rather than a dense (N, N) mask. The dense
    form cost 34 MB per structure at 3000 atoms and 382 MB at 10,000 -- built in
    a dataloader worker and shipped to the GPU for every sample -- which made
    protein-protein complexes untrainable long before the model itself was the
    limit. Bond membership is instead tested on encoded pair keys, which is
    O(M) memory. Numerically identical to the dense version.
    """
    n = pred.shape[0]
    if n > max_atoms:  # subsample for tractability on large proteins
        idx = torch.linspace(0, n - 1, max_atoms, device=pred.device).long()
        # Re-index bonds into the subsampled numbering, dropping any bond whose
        # partner was not sampled (it is no longer a pair we can score).
        remap = torch.full((n,), -1, dtype=torch.long, device=pred.device)
        remap[idx] = torch.arange(idx.numel(), device=pred.device)
        if bonds.numel():
            rb = remap[bonds]
            bonds = rb[(rb >= 0).all(dim=1)]
        pred = pred[idx]

    m = pred.shape[0]
    d = torch.cdist(pred, pred)
    iu = torch.triu_indices(m, m, offset=1, device=pred.device)
    dv = d[iu[0], iu[1]]

    if bonds.numel():
        lo = torch.minimum(bonds[:, 0], bonds[:, 1])
        hi = torch.maximum(bonds[:, 0], bonds[:, 1])
        nonbonded = ~torch.isin(iu[0] * m + iu[1], lo * m + hi)
    else:
        nonbonded = torch.ones(dv.shape[0], dtype=torch.bool, device=pred.device)

    penalty = F.relu(clash_dist - dv) * nonbonded
    denom = nonbonded.sum().clamp_min(1)
    return penalty.sum() / denom


def chirality_loss_single(pred, target, centers, margin=1.0):
    """Push predicted signed volumes to match the target sign (hinge)."""
    if centers.shape[0] == 0:
        return pred.new_zeros(())

    def signed(coords):
        ca = coords[centers[:, 0]]
        v1 = coords[centers[:, 1]] - ca
        v2 = coords[centers[:, 2]] - ca
        v3 = coords[centers[:, 3]] - ca
        return (torch.cross(v1, v2, dim=-1) * v3).sum(dim=-1)

    sp = signed(pred)
    st = signed(target)
    return F.relu(margin - torch.sign(st) * sp).mean()


@dataclass
class LossWeights:
    coord: float = 1.0
    distance: float = 1.0
    bond: float = 1.0
    clash: float = 0.5
    chirality: float = 0.2
    # Bond ANGLES: lengths alone leave every joint free to hinge.
    angle: float = 0.0
    # Distances to each atom's k nearest neighbours, rather than uniformly
    # sampled pairs that are mostly far apart and mostly easy.
    local_distance: float = 0.0
    # Supervises the atomlatent decoder's PROVISIONAL (pass-1) coordinates.
    # Without it the locality routing selects neighbours using a quantity that
    # was never trained to be a position.
    coarse: float = 0.0
    # Penalty on uneven latent usage (see AtomLatentAutoencoder.load_balance_loss).
    # 1.0 = perfectly balanced, L = one latent carries everything.
    load_balance: float = 0.0


def _per_sample_mean(values, sample_idx, B):
    """Mean of `values` grouped by `sample_idx`, returned as (B,).

    Groups with no elements yield 0 (matching the empty-input guards in the
    per-sample loss functions).
    """
    sums = values.new_zeros(B).index_add_(0, sample_idx, values)
    counts = values.new_zeros(B).index_add_(0, sample_idx, torch.ones_like(values))
    return sums / counts.clamp_min(1.0)


def batched_losses(preds, batch, clash_dist=1.5, chirality_margin=1.0):
    """All five loss terms for a padded batch, without a Python per-sample loop.

    Mathematically identical to summing the ``*_single`` functions over the
    batch (verified in tests/test_losses_vectorized.py), but computed with
    batched tensor ops so the GPU is not left idle between samples.

    Bonded pairs and chirality centres are gathered via flat indices offset by
    ``b * Nmax``, so per-structure topology of differing sizes is handled in one
    pass.
    """
    device = preds.device
    B, Nmax, _ = preds.shape
    mask = batch["mask"].to(device)                                  # (B, Nmax)
    target = batch["coords"].to(device)
    w = mask.unsqueeze(-1)

    # --- coordinate loss: batched Kabsch, then masked per-sample mean --------
    aligned = kabsch_align_torch(preds, target, mask)
    sq = ((aligned - target) ** 2).sum(-1) * mask                    # (B, Nmax)
    coord = (sq.sum(1) / mask.sum(1).clamp_min(1.0)).mean()

    # --- pair masks ---------------------------------------------------------
    pair_valid = (mask.unsqueeze(2) * mask.unsqueeze(1))             # (B, N, N)
    triu = torch.triu(torch.ones(Nmax, Nmax, device=device, dtype=preds.dtype), diagonal=1)
    pair_valid = pair_valid * triu

    dp = torch.cdist(preds * w, preds * w)
    dt = torch.cdist(target * w, target * w)

    # --- distance loss (smooth L1 over valid pairs) -------------------------
    dist_elem = F.smooth_l1_loss(dp, dt, beta=1.0, reduction="none") * pair_valid
    npairs = pair_valid.sum((1, 2)).clamp_min(1.0)
    distance = (dist_elem.sum((1, 2)) / npairs).mean()

    # --- bonds: gathered sparsely; no dense (B, N, N) bond mask is built -----
    bond_rows, bond_a, bond_b = [], [], []
    for i, bonds in enumerate(batch["bonds"]):
        if bonds.numel() == 0:
            continue
        bonds = bonds.to(device)
        bond_rows.append(torch.full((bonds.shape[0],), i, device=device, dtype=torch.long))
        bond_a.append(bonds[:, 0])
        bond_b.append(bonds[:, 1])
    if bond_rows:
        br = torch.cat(bond_rows); ba = torch.cat(bond_a); bb = torch.cat(bond_b)
        lp = torch.linalg.norm(preds[br, ba] - preds[br, bb], dim=-1)
        lt = torch.linalg.norm(target[br, ba] - target[br, bb], dim=-1)
        bond_elem = F.smooth_l1_loss(lp, lt, beta=0.1, reduction="none")
        bond = _per_sample_mean(bond_elem, br, B).mean()
    else:
        br = None
        bond = preds.new_zeros(())

    # --- clash: hinge over all valid pairs, then subtract the bonded ones ----
    # (cheaper than materialising dense bonded / non-bonded masks).
    clash_sum = (F.relu(clash_dist - dp) * pair_valid).sum((1, 2))
    n_nonbonded = npairs.clone()
    if br is not None:
        bonded_hinge = F.relu(clash_dist - dp[br, ba, bb])
        clash_sum = clash_sum - torch.zeros_like(clash_sum).index_add_(0, br, bonded_hinge)
        n_nonbonded = n_nonbonded - torch.zeros_like(n_nonbonded).index_add_(
            0, br, torch.ones_like(bonded_hinge))
    clash = (clash_sum / n_nonbonded.clamp_min(1.0)).mean()

    # --- chirality (signed volume sign agreement) ---------------------------
    c_rows, c_idx = [], []
    for i, centers in enumerate(batch["chirality_centers"]):
        if centers.numel() == 0:
            continue
        centers = centers.to(device)
        c_rows.append(torch.full((centers.shape[0],), i, device=device, dtype=torch.long))
        c_idx.append(centers)
    if c_rows:
        cr = torch.cat(c_rows); ci = torch.cat(c_idx, dim=0)

        def signed(x):
            ca = x[cr, ci[:, 0]]
            v1 = x[cr, ci[:, 1]] - ca
            v2 = x[cr, ci[:, 2]] - ca
            v3 = x[cr, ci[:, 3]] - ca
            return (torch.cross(v1, v2, dim=-1) * v3).sum(-1)

        chir_elem = F.relu(chirality_margin - torch.sign(signed(target)) * signed(preds))
        chirality = _per_sample_mean(chir_elem, cr, B).mean()
    else:
        chirality = preds.new_zeros(())

    return {"coord": coord, "distance": distance, "bond": bond,
            "clash": clash, "chirality": chirality}


class LossComputer:
    """Reconstruction loss.

    Two equivalent implementations (agreement proven to <1e-4 on values *and*
    gradients in tests/test_losses_vectorized.py):

      * ``vectorized=False`` (default) -- per-sample Python loop, cost
        proportional to sum_b n_b^2.
      * ``vectorized=True`` -- batched tensor ops, cost proportional to
        B * Nmax^2.

    **Measured on CPU, batch of 16 real proteins (327-832 atoms, padded to
    832): loop 85 ms/step vs vectorized 178 ms/step -- the loop wins 2x.**
    Ragged protein sizes mean the padded batch computes ~2.7x more pairs than
    the loop does, which outweighs the removal of Python overhead. On tiny
    uniform batches the vectorized path is ~6.5x faster, so the crossover is
    entirely about padding waste.

    The loop therefore stays the default. The vectorized path is kept because
    the trade-off may invert on GPU (where per-kernel launch overhead, not
    arithmetic, tends to dominate for small tensors) -- that needs measuring on
    real hardware before switching. The bigger win for either path is
    length-bucketed batching, which would cut the padding waste at the source.
    """

    def __init__(self, weights: LossWeights, clash_dist: float = 1.5,
                 vectorized: bool = False, max_atoms: int = 1200):
        self.w = weights
        self.clash_dist = clash_dist
        self.vectorized = vectorized
        self.max_atoms = max_atoms  # subsample cap for distance/clash (loop path)

    def __call__(self, preds, batch):
        if self.vectorized:
            totals = batched_losses(preds, batch, clash_dist=self.clash_dist)
            total = (self.w.coord * totals["coord"]
                     + self.w.distance * totals["distance"]
                     + self.w.bond * totals["bond"]
                     + self.w.clash * totals["clash"]
                     + self.w.chirality * totals["chirality"])
            comp = {k: float(v.detach()) for k, v in totals.items()}
            comp["total"] = float(total.detach())
            return total, comp
        return self._loop(preds, batch)

    def _loop(self, preds, batch):
        """preds: (B, Nmax, 3) padded. batch: dict from collate_fn."""
        device = preds.device
        totals = {k: torch.zeros((), device=device)
                  for k in ("coord", "distance", "bond", "clash", "chirality",
                            "angle", "local_distance")}
        B = preds.shape[0]
        for i in range(B):
            n = int(batch["n_atoms"][i])
            p = preds[i, :n]
            t = batch["coords"][i, :n].to(device)
            bonds = batch["bonds"][i].to(device)
            centers = batch["chirality_centers"][i].to(device)
            totals["coord"] = totals["coord"] + coord_loss_single(p, t)
            totals["distance"] = totals["distance"] + distance_loss_single(p, t, max_atoms=self.max_atoms)
            totals["bond"] = totals["bond"] + bond_loss_single(p, t, bonds)
            totals["clash"] = totals["clash"] + clash_loss_single(p, bonds, self.clash_dist, max_atoms=self.max_atoms)
            totals["chirality"] = totals["chirality"] + chirality_loss_single(p, t, centers)
            if self.w.angle:
                tri = batch.get("angle_triples")
                totals["angle"] = totals["angle"] + angle_loss_single(
                    p, t, bonds, triples=(tri[i].to(device) if tri else None))
            if self.w.local_distance:
                kn = batch.get("knn_pairs")
                totals["local_distance"] = (
                    totals["local_distance"] + local_distance_loss_single(
                        p, t, max_atoms=self.max_atoms,
                        knn=((kn[i][0].to(device), kn[i][1].to(device))
                             if kn else None)))
        for k in totals:
            totals[k] = totals[k] / max(B, 1)
        total = (
            self.w.coord * totals["coord"]
            + self.w.distance * totals["distance"]
            + self.w.bond * totals["bond"]
            + self.w.clash * totals["clash"]
            + self.w.chirality * totals["chirality"]
            + self.w.angle * totals["angle"]
            + self.w.local_distance * totals["local_distance"]
        )
        comp = {k: float(v.detach()) for k, v in totals.items()}
        comp["total"] = float(total.detach())
        return total, comp
