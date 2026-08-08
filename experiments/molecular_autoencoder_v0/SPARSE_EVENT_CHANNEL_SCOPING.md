# Sparse event channel — scoping note (INBOX 53b)

**Status: scoping only. No implementation is proposed and none should start on this note alone.**

## The position this starts from

Every negative result in the ATLAS line measures `x_i(t) = f(S_i, g_t, e_t)` **with `e_t` absent**.
Static per-atom conditioning and a global latent exist; the sparse event channel has never been
built. So "the codec loses to zero-shot ANM on 100% of 123 systems" is a measurement of two thirds
of the design.

The other branch is already closed. The ladder measured **+0.0483 FVE per decade** of `n_train`;
against a mean gap of 0.6114 that is ~12.7 decades, and exhausting the entire remaining corpus
(300 → 697 systems) buys **+0.0177**. More data will not produce a peer win. The channel is the only
untested thing that could.

## The minimal experiment: an ORACLE upper bound, not a trained channel

The cheapest informative test is not to build the channel. It is to ask whether a channel *could*
help even with perfect information, because that can be answered with **inference only**:

1. Take an existing trained arm and its per-frame reconstruction on held-out systems.
2. Compute the residual `E = true − decoded` per frame.
3. Pick the top-K atoms by residual magnitude and hand the decoder **their exact displacements** —
   this is the maximum information any sparse channel of budget K could carry.
4. Recompute FVE. Sweep **K = 1, 4, 16, 64, 85** atoms per frame.

K=85 is the budget-matched point: 3K ≈ 256 = DM, so the channel costs what the global latent costs.

**Why an oracle and not a trained channel.** A learned channel cannot beat one that is told exactly
which atoms matter and exactly how they move. So the oracle is a **strict upper bound**, and that
makes the negative result decisive while costing no training.

**The asymmetry, stated in advance** (38b's move applied to this question):

- **Oracle fails to close the gap → the branch is dead.** No learned channel can do better.
- **Oracle closes the gap → the branch stays alive, and nothing more.** It does not show a learnable
  channel exists; it shows the architecture is not the obstacle.

## Pre-registered falsifier

> At **K = 85** atoms per frame (budget-matched to DM = 256), if the oracle sparse channel recovers
> **less than 25%** of the codec-to-ANM gap on the held-out set, the sparse event channel is refused
> as a path to a peer win, and this item closes.

The gap is 0.6114 (mean) / 0.3956 (Q1 median), so 25% is ~0.15 / ~0.10 FVE. Reported per quartile,
never pooled, and with the same guards as everything else: median, mean and failure fraction
together, not a mean alone.

If the oracle at K=85 lands between 25% and full closure, that is **GRADED** — report the K at which
it crosses 25% and 50%, and the item stays open with the cost of the next step stated.

## Cost, and what it reuses

Inference only, no training. It reuses:

- `armf_tied_peer.py` — already computes codec and ANM FVE **in one pass on the same frames**, which
  is the join this must not get wrong;
- the ATLAS cache and the held-out manifest;
- an existing checkpoint (now loadable, after the `remap_legacy_keys` fix).

Estimated **2–4 GPU-hours** for a K-sweep over the held-out quartiles. That is comparable to a single
ladder arm.

## The measurement question 53b raises, and my answer

004 recorded that ANM *"has no generator and cannot be the product architecture."* If the channel's
value is generative, per-frame reconstruction FVE cannot express it — **Family D at the level of the
research question**.

That is correct, and it does not block this experiment, because **the oracle test is not asking a
generative question.** It asks whether the architecture can *represent* the residual at a given
sparse budget. Representation is exactly what FVE measures, so FVE is the right metric here.

But it bounds what a positive result may claim: it would establish representational sufficiency and
say nothing about whether the events can be *generated* at sampling time. That is a separate
experiment with a separate metric, and this note does not propose it. Writing a positive oracle
result as "the sparse channel works" would be the Family D error 53b names.

## What would make this item stop here

If the oracle cannot be run cheaply — for example if the residual cannot be attributed to atoms
without retraining, or if no existing harness computes codec and ANM on the same frames — then the
falsifier is not cheap, and per 53b's own condition the item should stop rather than become an
open-ended optimisation loop. That is the failure INBOX 004 removed from the ANM-decoder branch.

**It does not stop here.** `armf_tied_peer.py` already provides the same-frames join, the residual is
a subtraction on arrays that pass through it, and the K-sweep is a loop over one masking operation.
The falsifier is cheap, so the item proceeds to the oracle run — **when the GPU frees**, behind the
tied ladder and 41c.
