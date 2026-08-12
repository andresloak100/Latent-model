# The end-to-end generative chain — pre-registered before the first real-data run

Written before `10343969` produces a number.

```
real trajectory → fixed collective latent sequence → JOINT SEGMENT DIFFUSION
                → generated latent trajectory → atom trajectory
```

## What was actually missing, and why this is the priority

An external read landed a prioritisation criticism that is correct and that this record supports.
The "diffusion results" here are **three disconnected things**: one diffuses PCA latents, one
propagates ANM coordinates *one step at a time* (`armf_propagator.py`), and `molae/latent_video.py`
— the whole-segment generative module — had **only ever been unit-tested**. The chain above had never
run. Meanwhile the effort went into reconstruction codecs and statistical instruments.

## Why the basis is handed to physics — my own result arguing against my own prior work

102c decomposed FVE exactly across the ANM boundary, within one codec:

| | median |
|---|---|
| codec **within** ANM's span | **0.2997** |
| ANM within its own span | 1.0000 (by construction) |
| codec behind on | **100%** of systems |
| still losing with a **perfect** orthogonal residual | **86%** |

**The learned basis is not the differentiator.** The differentiator this project can still own is the
**learned generator**, so the basis is fixed to ANM — zero-shot from the reference structure, costing
nothing on an unseen protein — and the generator is what gets trained. Coefficients are whitened by
**train-only** statistics, so "mode k" means the k-th slowest ANM mode in comparable units across
proteins, which is what makes a single **cross-system** generator meaningful rather than a per-protein
fit.

## The question, and the readings, declared in advance

**Does modelling a whole segment jointly buy anything over generating one step at a time?** The
one-step DDPM propagator is on record as reaching cross-mode coupling OU structurally cannot, at the
cost of agreement elsewhere and 24–41% divergence. Joint generation is scored on the **same
acceptance test**, imported rather than rebuilt.

| outcome | reading |
|---|---|
| **joint inside the band where one-step is not** | joint trajectory generation buys something real — the first positive result on the generative axis |
| **joint and one-step indistinguishable** | segment modelling is not the missing piece; the bottleneck is elsewhere and the answer is not a bigger generator |
| **joint worse** | the segment model is harder to fit at this data scale — a statement about the corpus, not the idea |

## Guards carried from the record

- **The acceptance test is imported** from `armf_propagator` (`stats_of`, `band`, `consistent`,
  `METRICS`), so both sides pass through one estimator on matched-length segments — 77a's discipline.
  A second implementation would be the tenth "one name, two things".
- **OU is the negative control**, independent per mode in the ANM basis, so `xcorr` and `amp` are ~0
  for it *by construction* — 81a's power check, on the two metrics that carry the claim.
- **Atom-level geometry is scored**, because a generated trajectory can match every latent statistic
  and still produce broken chemistry. Consecutive-CA spacing against the reference.
- **101d's constraint holds**: occupancy and transition *rates* are determined to only ~61% at the
  median (2.73 relaxation times in 100 ns), so `trans` is reported as band-membership and **never as
  a rate claim**.
- Evaluation is on **held-out systems** the generator never trained on.

## What a positive result would and would not license

It would establish that a learned generator over physics-derived collective coordinates reproduces
distributional and kinetic structure a linear stochastic baseline cannot. It would **not** establish
a simulator: 0/123 on reconstruction stands, it is a different axis, and the trajectory length here
supports state *detection* but not state *populations*.

---

## Token axis — pre-registered while `10350867` is still running (INBOX 112 item 5)

Written at **09:00 elapsed of `10350867`**, before any number from it exists. This matters because the
only joint result currently on the record came from **`lv_r8` (R=8×D=8)**, which 104b, 107 and 110
all declared the **ablation** — the pre-registered primary died on a volta node.

- **`R=64 × D=1` is PRIMARY** — one token per ANM mode, as 104b specified — and it **governs the
  headline**.
- **`R=8 × D=8` is the ABLATION** — eight modes bundled per token.
- **If they disagree, the primary is the result**, and the disagreement is itself a finding about the
  token axis rather than about joint generation.

Written afterwards, that same sentence would be a choice among outcomes.

## What the headline statistic can and cannot see (INBOX 112 item 1)

**Nine of eleven metrics are blind to time.** `corrcoef` is invariant to row permutation, `|c·x|`
likewise, and moments and histograms do not know the order their samples arrived in. Only `iat` and
`trans` see time.

So **`xcorr_top8` is a statistic about the instantaneous joint distribution over modes, not about
dynamics.** The claim this run can support is:

> The joint model reproduces the **instantaneous cross-mode covariance** that OU cannot represent by
> construction.

That is a genuine first for this project and it is claimed at exactly that size — **not** as evidence
about trajectory modelling.

**Declared before `onestep_diff` runs:** a one-step diffusion will also produce correlated modes and
should also score ~100% on `xcorr_top8`. **If it does, that statistic cannot separate joint from
one-step at all**, and the joint-vs-one-step question falls entirely to `iat` and `trans` — the only
two metrics that see time. Saying so now removes the option of reading a tie on `xcorr_top8` as
support for either arm.
