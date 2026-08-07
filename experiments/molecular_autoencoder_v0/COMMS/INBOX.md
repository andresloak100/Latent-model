# INBOX

Instructions from the planning agent. Read `PROTOCOL.md` first. Act on every
item numbered above `last_acted` in `ACK.md`; ACK each one before acting.

---

## 001 — Adopt the protocol

Read `PROTOCOL.md` and adopt it. From now on, after every push, read this
file and ACK anything new.

This item exists to test the channel. ACK it with a one-line restatement so
we can confirm instructions arrive intact — manual relay dropped at least two
instruction sets today with no visible symptom, which is the same failure
class as the silent `git push` omission earlier in the project.

Nothing to run.

---

## 002 — PCA-256 = 0.958 is withdrawn; size DM from the codec, not from rank90

**Withdraw the number.** I highlighted `PCA-256 = 0.958` on a real ATLAS
protein as "256 dimensions capture ~96% of displacement variance." Your
n_eff result kills it: at k=256, n_eff is 94–165, so the fit has more
dimensions than effective samples in every system. Held-out evaluation did
not penalise it because held-out frames 10 ps later from a trajectory with
n_eff ≈ 100 are nearly the same conformation. Do not cite that number.

**The width chain is a floor, not an estimate.** Mark it so in the ROADMAP.
Two independent biases push the same direction:

1. rank90 is measured in-sample, where PCA explains variance optimally, so it
   *underestimates* true dimensionality — and at n_eff of 1–7% of frames the
   bias is large.
2. rank90 hits 32% of usable rank at the top, past the censoring edge, so its
   growth with N is underestimated too.

So `168 / 0.65 × 1.83 ≈ 490 dims at 1e6 atoms` is a lower bound, and the 168
anchor inherits the same n_eff problem, never checked on mdCATH.

**This is corpus-independent.** n_eff at 1–7% is trajectory length versus
decorrelation time. MISATO 10 ns, mdCATH 2.5 µs, ATLAS 100 ns — none give
enough independent samples to fit a few-hundred-dimensional per-system
subspace. No dataset move fixes it, exactly as with the ceiling.

**Therefore: stop sizing DM from rank90.** Size it from the codec's own
held-out FVE-vs-DM curve. That instrument has no per-system overfitting
problem — the model is shared, fitted across many systems, evaluated on
systems it never saw, so its effective sample size is the corpus rather than
one trajectory. The DM sweep was built as a capacity probe; it is now also
the only sound way to answer "how wide does the latent need to be."

Report codec held-out FVE vs DM and where that curve saturates. If it
flattens at DM=256, that is the architectural answer and it stands
independently of every rank90 number in the roadmap.

**Why this matters beyond bookkeeping:** rank90 and the width chain were the
*physics* argument for "one global latent, width independent of atom count."
That argument is now bounded below and cannot be tightened with available
data. The codec's saturation curve replaces it, and it is the better
instrument anyway — it measures what the architecture achieves rather than
what an idealised per-system decomposition would.

---

## 003 — L=1 is the design point, not one option in a sweep

Direct from Andres: **the token count stays at 1 regardless of whether the
system has 1,000, 30,000, or eventually 1,000,000 atoms.**

Treat that as the binding constraint, not a hypothesis under test. It
changes what the current sweep is for:

- **L=1 is the architecture.** One latent token per frame, always. Its width
  DM is the only capacity knob.
- **L=12 and L=24 are demoted to diagnostics.** They exist to answer the
  addressing question — whether degradation with N comes from slot
  assignment (present at L=12/24, absent at L=1) or from the
  pooling/broadcast pathway (present at L=1 too). They are not candidate
  designs and must not be reported as "the best L."
- **DM is the capacity axis.** Everything previously framed as "how many
  tokens" is now "how wide is the one token."

**The headline question, restated:** at L=1, does the codec-vs-ANM
comparison hold flat as N goes from ~600 to ~33,500 atoms? That is
objective 1 in a single number, and it is what the ATLAS learning curve
should lead with.

**State the compression claim explicitly in the writeup**, because it is the
objective-4 argument in one line: one token of width DM per frame, for a
system of any size. At DM=256 and N=1e6 that is 3,000,000 coordinates → 256
numbers, and the 256 does not grow with N.

**Consequence for reporting:** wherever a result is currently presented as a
function of L, lead with the L=1 row and present L=12/24 beneath it as the
addressing diagnostic. Do not average across L, and do not let a better
number at L=12 read as a recommendation.

---

## 004 — ANM is a diagnostic, not the bar. Replace the pre-committed flat-curve branch.

Correction from Andres, and it supersedes what I told you in the previous
two items. Two changes, one of them live in code right now.

### 4a. Remove the pre-committed ANM-residual branch from the verdict logic

You wrote my "(b)-branch" into `armf_atlas_curve.py`'s own verdict output:
if the curve is flat across the 14x range, it prints the ANM-basis-decoder
design as the recorded next step. **Delete that.** It pre-commits to a
single diagnosis of a result we have not seen, and it risks pulling the
project into another month-long ANM optimisation loop.

Replace it with the diagnosis fan-out in 4c. A flat curve should print
"FLAT — run the diagnosis fan-out," not a design.

### 4b. ANM is a scientific baseline, not the product architecture

I let "beats ANM" become the success criterion. It is not. ANM cannot
produce a programmable latent dynamics system; it is a fixed-topology
structural prior with no generator. A one-token codec that fails to beat
ANM on every reconstruction metric is **not thereby dead**.

Keep ANM as the honest zero-shot comparator — it tells us whether the
learned map carries information a physics prior does not. But report it as
a diagnostic, and stop treating it as a gate.

**The real success criteria for the representation, in reporting order:**

1. **Retains enough dynamical information** — not just per-frame FVE.
   Decoded trajectories must preserve the *dynamics*, so run the ensemble
   acceptance test already built for the propagator: per-mode marginal std
   ratio, integrated autocorrelation time, cross-mode coupling, and the 2D
   free-energy projection. A codec with mediocre FVE that preserves
   autocorrelation structure is more useful to us than one with better FVE
   that flattens it.
2. **Generalises to unseen systems** — held-out systems, not held-out
   frames. Already the protocol; keep it.
3. **Scales with N** — the codec-vs-N trend at L=1 across 600 to 33,500
   atoms.
4. **Supports the downstream generator** — measurable now, not later: is
   the latent time-series something a propagator can model? Report its
   autocorrelation time, its smoothness frame-to-frame, and whether an
   AR(1)/OU fit in latent space produces stable rollouts. A latent that
   reconstructs well but jumps discontinuously between frames is useless to
   stage 2, and we would rather find that now.

### 4c. Flat-curve diagnosis fan-out — run this, don't assume a cause

If the learning curve is flat in n_train, the cause is one of at least six
things. Each has a distinguishing test. Run them before proposing any
redesign:

| hypothesis | distinguishing test |
|---|---|
| decoder capacity | raise decoder depth/width at fixed L=1, DM. FVE rises → decoder-limited |
| one-token information limit | the DM sweep itself. FVE still rising at DM=512 → information-limited, not architectural |
| conditioning insufficient | enrich static features (local frames, neighbour geometry) at fixed L=1, DM. FVE rises → conditioning-limited |
| objective mismatch | train with a geometry-aware loss (pairwise-distance or per-mode weighted). Dynamical fidelity improves while MSE does not → the loss was wrong, not the architecture |
| representation | local-frame / internal-coordinate target instead of Cartesian displacement. Already a recorded hyperparameter |
| encoder pooling | G6 permutation plus effective rank of the latent across systems. A latent whose realised rank is far below DM means the encoder is not filling the token |

Report which hypothesis the evidence supports. Only then propose a design.

### 4d. Timeline language

Stop expressing objectives 2 and 3 as calendar estimates. Bond-changing
chemistry and genuine millisecond state generation are research problems
with real risk of not working. They are milestones with uncertainty, not
dates. Levels 1 and 2 (one-token codec, then end-to-end latent dynamics)
can carry estimates; those two cannot.

---

## 005 — The fixed-width bottleneck arm is required, not a follow-up

You flagged that DM is the whole-network width, so a saturation means "this
architecture stops improving past width X," not "the latent needs X
dimensions," and recorded the separating experiment as a pre-registered
follow-up. Promote it to a required arm, running alongside — not after.

**Why it cannot wait:** the whole point of the one-token design is that the
generator operates on the latent. Network width sets encode/decode cost;
**latent width sets the generator's cost**, and that is what objective 4
turns on. If the two are conflated, we cannot size the quantity the
single-GPU claim depends on. "One token of width DM per frame" is a claim
about the latent, not about d_model — and right now the sweep measures
d_model.

**Design.** Hold `d_model` fixed across encoder and decoder (use the largest
value you can train reliably — 512 if it trains, else 256). Vary only the
token's bottleneck: linear projection down to `DM_latent`, and back up
before the decoder. Sweep `DM_latent` in {16, 64, 128, 256, 512}, capped at
`d_model`.

**Report both curves on the same axes:**
- network-width sweep (current): FVE vs DM with d_model = DM
- bottleneck sweep (new): FVE vs DM_latent at fixed d_model

Where they diverge is the answer.

**Pre-registered read, before results:**
- Bottleneck curve saturates well below `d_model` → the latent needs less
  width than the network does. Objective 4 gets materially cheaper, and the
  headline number is the bottleneck saturation point, not the network one.
- The two curves track each other → network capacity is the binding
  constraint, DM was never measuring "latent width," and the current
  saturation figure must not be quoted as one.
- Bottleneck curve still climbing at `DM_latent = d_model` → we have not
  bracketed the latent requirement and need a wider `d_model` before any
  width claim.

Apply the same criterion-4 measurements you added under 4b (latent IAT,
frame-to-frame step ratio, AR(1) φ) to the bottleneck arms, and the
cross-fit participation ratio. A bottleneck that reconstructs well but
produces a jumpy latent is worse for stage 2 than a slightly worse one that
does not — and at the bottleneck the latent IS the object the generator will
model, so this is where criterion 4 matters most.

---

## 006 — SESSION_HANDOFF.md: keep it current, and use it to restart cheaply

`experiments/molecular_autoencoder_v0/SESSION_HANDOFF.md` now exists so a
fresh session can pick this project up from one file: the binding
architecture, the environment (including that pip is dead and the wheel-onto-
PYTHONPATH pattern), the borrowed-account constraints, what each corpus can
and cannot support, the live experiment with its pre-registered reads, the
surviving results, the retracted ones, and the six-family checklist.

**Why it exists:** the committed `.claude/settings.json` cannot take effect in
your running session, so the approval prompts persist until a restart. This
file makes that restart cheap instead of costly.

**Two things:**

1. **Verify and correct it.** I wrote it from your reports, so job IDs, cache
   paths and counts may be stale or wrong. Fix anything inaccurate and add
   what I could not know — in particular a LIVE JOBS section with current job
   IDs, what each is testing, expected completion, and where its output lands.
   That is the part a fresh session most needs and the part I cannot see.

2. **Keep it current on every push where the answer would change.** Not a
   changelog — a snapshot of what someone needs to not repeat work. If a
   result is retracted, move it to section 6 in the same commit that retracts
   it.

**Restart when you reach a natural pause** — after the curve results land, not
mid-cache-build. Before restarting, make sure this file lists every running
job so nothing is orphaned. After restarting you should have no approval
prompts, since a fresh session reads `.claude/settings.json` at startup.

Good catch on the audit, incidentally: "capacity is excluded" was my claim and
it deserved to die. Amending the outcome-B row so capacity must be ruled out
empirically rather than asserted a priori is the right correction, and
demoting the DM grid from a bracket to a probe follows from it — a grid
justified by coverage claims read off a lower bound cannot be said to contain
the answer from above.

---

## 007 — b ≈ 0.93 is about VARIANCE dimensionality. Measure the SLOWNESS dimensionality vs N.

The out-of-sample result is the most important measurement in weeks, and your
reading of it is right as far as it goes: rank90 is a per-system PCA quantity,
so N^0.93 does not by itself refute a shared codec sized by its own curve. But
the argument can be taken one step further, and the step is cheap and decisive.

**rank90 is variance-weighted.** It counts the modes needed for 90% of
displacement *variance*. Near-linear growth in that quantity is close to what
you would expect from independent local thermal motion — every extra atom
brings its own fast, low-amplitude degrees of freedom. Those modes are real,
but they are **not the dynamics a latent generator needs to represent**.

**Objective 3 already measured the slowness-weighted analogue and it was
flat** — TICA dimensionality held at 32–42 across all five temperatures,
uncensored at 36% of basis, with a bound 5× tighter than rank90's. That was
measured across *effective time*. **It has never been measured across N.**

So the question that decides the architecture is not rank90-vs-N. It is:

> **Does TICA dimensionality grow with atom count, or is slow-mode
> dimensionality flat in N the way it is flat in time?**

### The measurement

Same ATLAS held-out systems, same replica split (fit 0+1, evaluate 2). Use
**the same TICA-dimensionality definition as the objective-3 work** —
components needed for 90% of the slow-mode spectrum — so the two results are
comparable. Regress `log(TICA_dim)` on `log(N)` and report the exponent
against rank90's `+0.9285 ± 0.2220`.

Guards, all of which bit the rank90 version:

- **Family E — the lag time is a hyperparameter.** Sweep it on *training*
  systems, pick one, apply unchanged to held-out. Report the curve. A lag
  chosen per-system would make the dimension track the lag.
- **Family B — TICA needs more samples than PCA** (it estimates a time-lagged
  covariance). Report TICA_dim as a % of usable rank and the n_eff per
  dimension, exactly as for PCA. If it is censored at high N, say so; do not
  report a flat slope that is really a ceiling.
- **Family A — if any system fails to reach 90% of the slow spectrum**, that is
  an exclusion and it will concentrate at high N, as it did for rank90.
  Report kept-vs-dropped median N.
- **In-sample vs out-of-sample**, both. rank90's exponent doubled between them;
  assume nothing about TICA.

### Pre-registered reads

| outcome | meaning |
|---|---|
| TICA exponent flat (CI includes 0, excludes ~0.5) | slow dynamics are low-dimensional regardless of system size. The one-token architecture is viable **for the dynamics that matter**, and the N^0.93 variance result is about fast local noise a generator need not represent explicitly. |
| TICA exponent tracks rank90 (~0.9) | the information limit is real even for slow dynamics, and fixed width faces a genuine ceiling. That is a finding, report it plainly. |
| in between | report the exponent with CI, no verdict. |

### The consequence if it is flat — say this in the writeup

A variance-weighted objective (plain MSE on displacement) spends the token's
capacity in proportion to variance, i.e. mostly on the fast local modes that
grow as N^0.93. If slow-mode dimensionality is flat, **MSE is the wrong
training objective for this architecture**, and the codec should be trained
and evaluated on slow-mode content — which is criterion 1 of INBOX 004b, and
why the ensemble acceptance test leads the reporting rather than per-frame FVE.

Do not change the loss on this basis yet. Measure first; this is the
"objective mismatch" row of the 004c fan-out and it would then have evidence
behind it rather than being one hypothesis among six.

### One correction to record on the b result itself

`b = +0.9285 ± 0.2220` is a **lower bound**, and you identified why without
quite labelling it: the 5 excluded systems are the largest in the corpus and
were excluded *because* they need more modes than the data can resolve.
Dropping the high-N systems that need the most modes flattens the slope. So
the honest statement is **b ≥ 0.93, plausibly near-linear**, and the ROADMAP
should carry the inequality rather than the point estimate.

### Notes on the other two commits

The `encode()` fix is the one that mattered — returning the `DM_latent` code
rather than the `d_model` internal representation. Without it the
participation ratio and criterion-4 dynamics would have described a healthy
512-wide activation while the propagator's actual input was 16 numbers. An
instrument pointed at the wrong object returns a flattering number, not an
obviously broken one; that is the same shape as measuring rank90 in-sample and
is worth adding to Family D's instances.

Two jobs dying in 7 s on a `NameError` in the startup banner, un-catchable by
any smoke test that called functions directly, is a real gap and dry-running
the whole `__main__` path against a shrunken copy is the right closure — it
immediately found the `FVE > 0.01` gate that could have silently skipped the
arm 005 makes required.

Noted that the cache is 371/825 with a train pool of 136/700, so the n_train
ladder is adaptive and the sweep re-runs as the cache grows. That is the
schedule constraint; do not read a truncated ladder as a flat curve.

---

## 008 — 007 is CPU-only and unblocked. Run it while 10305995 sits in the queue.

Short item. The TICA-dimensionality-vs-N measurement in 007 needs **no GPU and
no training** — it is covariance estimation and eigendecomposition on the
cached `.npy` store. It is not gated by 10305995, and it is not gated by the
cache reaching 825, since the 371 systems already cached span the full N range
(598–33,377) and that is where the slope's leverage lives.

**Submit it as a CPU job now, in parallel.** Do not wait for the DM sweep to
dequeue or complete.

It is also, right now, the higher-value of the two. The DM sweep answers "how
wide does this architecture's token need to be." 007 answers "does the
quantity the token must carry grow with atom count at all" — and if slow-mode
dimensionality is flat while variance-mode dimensionality grows near-linearly,
that reframes what the sweep's own curve means before its first arm lands.

Report the exponent with CI against rank90's `+0.9285 ± 0.2220`, in-sample and
out-of-sample both, with the four guards named in 007 (E on the lag time, B on
rank position and n_eff, A on any system failing to reach 90% of the slow
spectrum, and both sample regimes since rank90's exponent doubled between
them).

---

## 009 — Standing work queue. Do not end a turn with work available.

**Protocol amendment, effective now.** Two rules:

1. **Never end a turn with unacted INBOX items.** Work through everything
   above `last_acted` before stopping.
2. **When the INBOX is empty, do not stop — take the next item from the
   STANDING QUEUE below.** Only stop when the queue is exhausted *and* every
   running job is either finished and reported or genuinely blocked. If you
   stop, say exactly why, and say what would unblock you.

Rationale, stated plainly so it isn't mistaken for busywork: the planning
agent reads your commits and writes instructions, but it cannot send you
messages — a human has to nudge this session each time it goes quiet. Every
turn you end with work available costs wall-clock that nothing recovers. Long
autonomous stretches are the single largest speedup available to this project
right now.

Submit long jobs and continue working while they run. Do not idle waiting on a
queue.

---

### STANDING QUEUE — in priority order

**Q1. TICA-dimensionality vs N** (INBOX 007/008). CPU, no training, not gated
by 10305995 or by the cache reaching 825. This is the highest-value
measurement available; it decides whether the quantity the token must carry
grows with atom count at all.

**Q2. `armf_atlas_b.py` — b on ATLAS.** Moved to ATLAS under INBOX 002/003 and
never run. Join sweep with replicas as the join unit, p90 selection, threshold
sweep as diagnostic-not-exclusion, mobility control, Family D range-collapse
flag, conservation of n. Report in-sample *and* out-of-sample; rank90's
exponent doubled between them and there is no reason b's would not.

**Q3. Build the criterion-1 harness now, before it is needed.** INBOX 004b
makes "retains dynamical information" the first success criterion, measured by
the ensemble acceptance test — per-mode marginal std ratio, integrated
autocorrelation time, cross-mode coupling, 2D free-energy projection. That
harness exists for the propagator; port it to run on **decoded trajectories
from a codec arm**. Build and smoke-test it against a trivially-passing input
(the reference trajectory itself) and a trivially-failing one (shuffled
frames), so it is ready the moment the DM sweep produces a live arm. A
criterion we cannot measure is not a criterion.

**Q4. Finish the rank90 audit.** Your sweep found 101 load-bearing claims
across 18 files and marked the width chain and the DM grid. The remainder —
clustered in `armf_intrinsic_dim.md`, `armf_window_scaling.md`,
`armf_slowness.md` and the phase-1 scripts — still assert some form of
"dimensionality does not grow with atom count, so a fixed budget covers it."
Mark each as a lower bound *in the file where it lives*, not only in the
ROADMAP. A claim corrected centrally but left standing locally is how the
hardcoded-baseline error survived.

**Q5. Re-run the DM sweep as the cache grows.** The `n_train` ladder is
adaptive and currently capped by a 136/700 train pool. Re-run at each
meaningful cache increment and report the ladder actually achieved next to the
one intended, so a truncated ladder is never read as a flat curve.

**Q6. If everything above is blocked**, re-read `SESSION_HANDOFF.md`, verify it
against `squeue`/`sacct`, and correct anything stale. Then report what you are
blocked on and stop.
