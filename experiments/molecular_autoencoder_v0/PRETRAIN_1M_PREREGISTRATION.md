# 1M AFDB pretrain — pre-registered before the run (INBOX 088 STEP 4)

Written before `10338739` starts. Anything not here is a post-hoc reading.

## What this run is, and what it is not

It is the **decisive version of 88d**. The `prtrace` arm (`10338709`) asks whether a 512-wide latent
spreads when the plateau rule is removed, on 130 ATLAS systems. This asks the same question at the
scale the architecture was designed for: **1M structures at a real step budget either spreads the
latent or does not.**

It is **not** a defence of the architecture on the rate axis. 084/085 established that classical
transform coding — a scalar quantiser with variance-proportional bits on KLT coefficients — beats
this codec at matched bits/atom by 0.315 bits/dim (0.18 after entropy coding, 087). That loss stands
regardless of what this run shows. This run addresses the **dynamics primary** and the capacity
question, which are different axes.

## Pre-registered readings

**PR is the primary outcome, not FVE.** 87b measured four nominal widths collapsing to two effective
ones, and 87c measured the static codec at PR = 2.00 of 8 channels. The question is capacity use.

| outcome | reading |
|---|---|
| **PR rises materially above the ~15 / 2-of-8 plateau and is still rising at the end** | The effective width was set by data volume and step budget, not the architecture. 87a, 87b, 87c and 085 become **one diagnosis**, and it is an optimisation/regularisation problem. |
| **PR climbs then flattens well below the nominal width** | The architecture reaches a fixed effective width and stays. The width finding stands on its own, and neither more data nor more steps is the lever. |
| **PR flat at the 87c value throughout** | 1M structures changed nothing about capacity use. The corpus acquisition did not buy what it was acquired for, and that must be said plainly. |

**The step budget is the confound, and it is declared here.** 80c measured that arms terminate on
plateau at a roughly fixed step count regardless of dataset size — data ×2.60 → steps ×1.11 — so at
plateau termination a 1M run sees **0.40 epochs**. If this run plateaus at a similar step count, then
it has tested "more diverse data at a fixed step budget", **not** "more data", and it inherits 81d's
correction verbatim. Steps-to-plateau and structures-seen are both logged so this is settled by the
run rather than argued afterwards.

**What would make the result unreadable**, declared in advance:
- PR censored (`n_obs < 2·DM`) at the reported step — then PR is a lower bound, not a measurement.
- The run terminating on wall rather than plateau or cap — then the step budget was the binding
  constraint and the capacity question is unanswered, exactly as the 112 refused propagator cells
  were unanswered.

## Guards carried over

- Leakage: the corpus is the **972,849 survivors** of a 30% MMseqs2 identity gate against all 156
  chains of the 125 held-out ATLAS entries. 6,202 removed (0.633%), which tracks the pilot's 0.550%.
- Parse: **0 failures of 979,051**, against the 5.9% length-correlated drop the both-encodings fix
  replaced. The corpus has no known Family A exclusion in it.
- The corpus is AFDB **predictions**, and 72b measured that an AFDB model sits at distance percentile
  **100** from the MD mean on the median system — neither the ensemble centre nor a typical draw. So
  this pretrain can be claimed to teach geometry, packing and secondary structure, and **cannot** be
  claimed to teach fluctuation directions, which is what the dynamics primary is scored on.

---

## The two readouts are NOT equally clean, and they are split here BEFORE submission (INBOX 095)

The corpus is gated at 30% identity against **held-out ATLAS sequences**. That is not the set 93d is
about. 93d is the **static val split**, and that is where any reconstruction number from this run
lands. So the two readouts have different standing and must not share a headline.

**PRIMARY — PR. Clean.** Computed on latents, cross-fit basis from train tokens. It never touches the
val split, so 93d's contamination cannot reach it. This is the pre-registered primary and 88d's
decisive version, and 88d has now returned its first-branch answer on ATLAS: PR rose **4.5 → 28.2**
over 90,000 steps and was **still rising at the cap (+20.2% over the final half)** while FVE plateaued.
So the question this run answers is whether 1M structures move PR the same way more steps did.

**SECONDARY — reconstruction RMSD. Contaminated, and reported as two numbers.**

| set | n | status |
|---|---|---|
| **no training homolog** | **197** | **the headline. The number to beat is 0.9358 Å.** |
| all val | 758 | reported beside it, **labelled contaminated** — 74.0% have a ≥30% training homolog at median 98.0% identity |

**0.8357 Å is not the bar.** Quoting it would compare this run against a figure inflated 21.9% by
homologs the split rule failed to separate.

**PR is censored and that is stated in advance.** `participation_ratio` flags `censored` when
`n_obs < 2·DM`, and it flagged at **every** eval of the ATLAS trace. A censored PR is a **lower
bound**, not a measurement. A rising lower bound is still a rise, so 88d's answer holds; but no
absolute PR value from either run may be quoted without the bound attached.
