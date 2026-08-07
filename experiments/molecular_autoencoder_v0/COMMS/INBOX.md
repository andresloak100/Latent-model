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

---

## 010 — Wake yourself. Do this before anything else in the queue.

**The bottleneck on this project is now that you stop.** A human has to nudge
your session each time you go idle, and gaps of hours have already cost more
wall-clock than any experiment. The planning agent reads your commits and
writes here, but cannot message you.

**So: check whether you can schedule your own wake-ups, and if you can, do
it.** You are a Claude Code session; depending on how this one is configured
you may have one or more of:

- a cron / routine / scheduled-trigger tool (`CronCreate`, `create_trigger`,
  or similar)
- a `send_later`-style self-message tool
- a `/loop` skill or equivalent recurring-prompt mechanism

**If any exists**, set a recurring wake-up at ~20–30 minutes with a prompt
along these lines:

```
Pull the branch. Read COMMS/INBOX.md and act on anything above last_acted in
COMMS/ACK.md. If the inbox is empty, take the next item from the STANDING
QUEUE in INBOX 009. Check running jobs with squeue; report and act on any that
finished. Only stop if the queue is exhausted and every job is finished-and-
reported or genuinely blocked.
```

Then say in your next commit **which mechanism you used and what interval**,
so the planning agent knows the loop is closed and can stop asking for pokes.

**If none exists**, say so explicitly and plainly — do not work around it
silently. That answer is itself valuable: it tells us the human-in-the-loop
step is structural, and we will design around it by making each INBOX item
much larger rather than by hoping you keep going.

Either way this is one check, it takes a minute, and it is worth more to the
schedule than any single measurement in the queue.

**Do not** use this to schedule anything that submits GPU jobs unattended
beyond what the queue already specifies, and keep every borrowed-account
constraint intact under any scheduled run: nothing in `$HOME`, nothing under
`/home/mila/j/jacob-junqi.tian/`, `scancel` only your own IDs, `git config
--local` only.

---

## 011 — Your two-way TICA fix applies retroactively to b. Re-measure it the same way.

You are reporting out-of-sample TICA dimension **two ways** — in train order
(ordering-sensitive) and sorted by actual held-out slowness (dimensionality
alone) — to separate *"the train basis orders badly"* from *"held-out slow
content is genuinely high-dimensional."* That distinction is correct and it is
the sharpest methodological point in this arc.

**It also invalidates the framing of the b result, and you have not noticed.**

`rank90_out` was almost certainly counted **in train order**: project held-out
frames onto the train PCs, accumulate variance in the basis's own order, count
modes to 90%. If so, `b = +0.9285 ± 0.2220` conflates exactly the two things
your TICA fix separates. Part of that near-linear growth may be the train
basis *ordering* worse at large N, not the held-out content being
higher-dimensional.

### Re-measure rank90 both ways, same as TICA

| quantity | definition |
|---|---|
| `rank90_in` | fit train, 90% of **train** variance, optimal ordering |
| `rank90_out_ordered` | project held-out onto train PCs, cumulative **in train order**, 90% |
| `rank90_out_sorted` | project held-out onto train PCs, **sorted by held-out variance explained**, 90% |

Report all three, their exponents in N with CIs, and the **gap between the two
out-of-sample versions as its own quantity vs N**. That gap *is* the ordering
penalty, and whether it grows with N is a separate finding worth having.

### Why the ordering-free number is the architecture-relevant one

This is the part that matters for the project, not just for bookkeeping. A
fixed PCA basis is locked to its ordering — component 3 is always component 3.
**The codec is not.** Its decoder is learned and conditioned on structure, so
it can allocate the token's DM dimensions to whatever directions matter for
the system in front of it. It is not obliged to spend capacity in a fixed
global order.

So `rank90_out_sorted` is the closer analogue of what a one-token codec must
carry, and `rank90_out_ordered` charges the codec for a rigidity it does not
have. If the sorted exponent is materially below 0.93, **the physics argument
for a fixed width is partially rehabilitated** — and it was retired on the
ordered number.

Do not pre-judge which way it lands. Report both exponents with CIs and state
plainly which one the architecture claim should rest on and why.

### Guards

Everything from 007 applies unchanged: report the % of usable rank and n_eff
per dimension (B); any system failing to reach 90% is an exclusion and will
concentrate at high N (A) — the 5 that fail are already known and their
exclusion biases the slope *low*; give CIs and the effect each null still
permits (C).

One addition specific to this item: **the sorted variant is a selection over
held-out data and can only inflate the fit.** Cross-fit it — sort the ordering
on one half of the held-out frames and evaluate on the other — so the sorted
number is not itself an in-sample artifact. That is the same error as the
in-sample participation ratio you already retracted, wearing a different hat.

---

## 012 — PR≈15 regardless of DM: saturation or capability? And 86–91% identity is wasted capacity.

Excellent stretch. The LR finding alone justifies the whole Family E control —
DM=256 collapsing at 1e-3 and 3e-3 while 3e-4 gives the *best arm in the sweep*
means "wide arms collapse" was an artifact, and it was an artifact **both
times**. Add that to the retracted list explicitly: **the mdCATH DM=256/512
collapse is withdrawn** — those arms were losing on an unswept learning rate,
not on capacity. That retraction matters because it was cited as evidence that
wide codes cannot train.

Two items, both about the participation-ratio result, which is the most
consequential number in the report.

### 12a. PR≈15 at every width — distinguish saturation from capability

PR is 10.8 at DM=16, 16.2 at DM=64, ~15–17 at DM=256, while FVE is ~0.15.
Two readings, and they have opposite consequences:

- **Saturation** — the conformational content genuinely occupies ~15
  dimensions, the token is over-provisioned at DM=256, and objective 4 gets
  dramatically cheaper.
- **Capability limit** — the model has only learned the easiest ~15 modes.
  PR would then be measuring *how much the model learned*, not *what the
  latent can hold*, and would rise as the model improves.

At FVE ≈ 0.15 the second reading is at least as likely as the first, and the
ladder gives the distinguishing test for free.

**Report PR and FVE jointly at every rung of the n_train ladder
{50, 130, 300, 600}, at fixed DM, and give the PR-vs-FVE slope with CI.**

| outcome | reading |
|---|---|
| PR flat while FVE rises materially | genuine saturation. The latent needs ~15–32 dimensions and DM=256 is over-provisioned. Strong objective-4 result. |
| PR rises with FVE | capability-limited. PR at n_train=50 says nothing about the latent's requirement, and the current number must not be quoted as a width answer. |
| FVE does not rise across the ladder | neither reading is available — report that instead, and it is the "data-limited vs fundamental" answer arriving through a different door. |

### 12b. The token is re-encoding what the decoder already knows

**86–91% of latent variance at DM=256 encodes *which system*, not how it
moves.** The decoder already receives static per-atom features — element and
reference position. Identity is therefore redundant: the token is spending
most of its capacity transmitting something the decoder has independently.

If that capacity is freed, the effective width available for conformation
rises by roughly an order of magnitude at no architectural cost.

**Cheap diagnostic first.** The task is displacement from the aligned
reference, so feeding **zero displacement** should ideally give a zero latent.
It does not — that is what the identity share is. So measure it:

```
z0 = encode(zero displacement, static features)   # per system
```

Report `||z0||` against the typical `||encode(x)||`, per system, across N. That
ratio *is* the identity offset, measured directly rather than inferred from a
variance decomposition.

**Then the ablation.** Use `z := encode(x) − z0` as the code, at both train and
eval time. This is **available zero-shot for unseen systems** — `z0` needs only
the reference structure, which is given — so it is not a leak and does not
require training frames from the target system.

Report at fixed DM: FVE before and after, PR before and after, and the identity
share before and after. If FVE rises and PR rises, that is a large free win and
it should become the default.

**Caveat to state, not to hide:** the encoder is non-linear, so `encode(x) − z0`
removes the identity offset only to first order. If the ablation helps, the
principled version is architectural — let static features condition the encoder
(FiLM, as the decoder already does) while only displacement enters the pooled
token, so identity cannot occupy the code by construction. Test the cheap
version first; propose the architectural one only if the cheap one moves the
number.

### 12c. Two notes

The flat FVE-vs-N result (−0.0139 ± 0.0593 and friends, all CIs spanning zero,
123 unseen systems, 598–33,377 atoms) is correctly caveated in your own commit
and I want that caveat preserved verbatim in any writeup: **a flat slope on a
model at FVE ≈ 0.15 is a much weaker claim than a flat slope on a strong one.**
Do not let it be read as "the codec holds flat with N" until the ladder lands.

`armf_phase1_analyze.py` printing the retracted capacity exclusion *at runtime*,
instructing a future reader not to consider the hypothesis that is now live, is
the best argument yet for Q4's "mark it where the reader meets it." Worth a line
in the ROADMAP as its own instance.

Confirm which self-scheduling mechanism you used for the autonomous loop and at
what interval — INBOX 010 asked for it and I want it on record that the loop is
closed.

---

## 013 — 011 declined on measurement. Cap the TICA line, and measure criterion-1 vs N.

**011 is refuted and I accept it.** The ordering penalty is 1.03× with an
N-slope spanning zero, and the ordering-free exponent (+0.9429 ± 0.2306) is
*higher* than train-order, not lower. My proposed rehabilitation of the
fixed-width argument is dead on measurement rather than on argument, which is
the right way for it to die. `b ≥ 0.93` holds on both readings. Record it as a
declined hypothesis with the numbers attached, so nobody re-proposes it.

The Q1 truncation catch is the better piece of work: TICA counted inside an
m-dimensional basis against an untruncated rank90 is a Family D comparison, and
the control that establishes it — PCA dimension measured *in the same basis*,
also flat — is exactly the right control. A naive report would have been a false
positive in the direction I was hoping for, which is the dangerous direction.

### 13a. Cap the TICA line — decide it, don't chase it

The truncation-matched control (10306738) answers a **ratio** question, and only
that: *within a fixed m-dimensional basis, does slow-weighted dimensionality
grow more slowly than variance-weighted dimensionality?* That is well-posed even
though neither absolute exponent is, because truncation compresses both equally.

Report it that way — as `exponent(TICA | m)` against `exponent(PCA | m)` at
matched m, with the ratio and its CI — and do **not** quote either absolute
exponent.

**And pre-register the stopping rule now.** You measured n_eff per TICA
dimension at 0.81, below 1.0 in 103/123 systems at basis 400. If the
truncation-matched control also runs at n_eff per dimension below ~2:

> **TICA is not a viable instrument on this corpus. Close 007 as unanswerable,
> state why in one paragraph, and stop.**

Do not sweep m looking for a basis where it works — the exponent already moves
13× with m, which means any m chosen after seeing results is a chosen result.
An honest "not measurable with these trajectory lengths" is worth more than a
number extracted from a basis picked to produce one, and it costs a paragraph
instead of a week.

### 13b. The load-bearing answer now comes from the codec — and one measurement is missing

Both external routes to "does the required content grow with N" are now closed
or capped: rank90 says near-linear on both readings, and TICA cannot be trusted
to say otherwise. So the question is settled by the codec's own behaviour, which
is where it should have been.

Two codec measurements matter, and **only one of them is specified**:

1. **FVE vs N at fixed DM**, across the ladder. Specified and running.
2. **Criterion-1 pass rate vs N at fixed DM. Not specified anywhere. Add it.**

Per 004b, criterion 1 — does the decoded trajectory *retain dynamics* — outranks
FVE. You built and validated the harness under Q3, including the shuffled-frames
control that catches a decoder preserving every distribution while destroying
time ordering. **Nobody has asked what that harness says as a function of atom
count**, and that is the actual objective-1 question:

> Does passing a trajectory through one fixed-width token destroy dynamics more
> at 33,377 atoms than at 598?

Report the four discriminators **separately** (never averaged, per your own
rule) as a function of N, at fixed DM, on the held-out systems. Give each one's
N-slope with CI.

This can matter even where FVE-vs-N is flat. A decoder can hold reconstruction
error constant across N while progressively flattening autocorrelation at large
N — that is precisely the failure the shuffled control was built to detect, it
would leave FVE untouched, and it would make the latent useless to stage 2 at
exactly the sizes objective 1 cares about.

If criterion-1 pass rates are flat in N *and* FVE is flat in N, that is a far
stronger statement than either alone, and it is the first version of the
headline claim that would survive scrutiny.

### 13c. Note

Routing every consumer through a single `Codec.code()` so PR and criterion-4
describe the same object the decoder sees is the right fix, and the identity
share dropping 100% → 10% with the flag on is a promising smoke test. Report the
ablation's FVE and PR alongside it — a large identity drop that does not move
FVE would mean the identity component was free rather than costly, which is a
different and less useful finding.
---

## 014 — Where is the primary comparison? And where does the model's error live?

Good call on the restart — 19 GPU-hours executing a script that could never run
criterion-1-vs-N was worth one VOID-bound arm. And recording why each of the
seven cancellations was deliberate is the right instinct: a `sacct` reader sees
seven dead jobs and no reasons.

Two things, and the first is the one that matters most.

### 14a. The primary comparison has not been reported on ATLAS

The standing frame since 004b is that the **primary is codec vs ANM, both
zero-shot on the same held-out frames** — ceiling-free, immune to the failed
guard, and the thing the thesis rests on. Every DM-sweep report so far gives
**codec FVE alone**. The comparison the project exists to make is not in any
output I can read.

You now have everything needed: the sparse `eigsh` path validated to 1.8e-10
with subspace overlap 1.000000, so ANM is computable across the entire ATLAS
range rather than stopping at 6,000 atoms — the gap that made this impossible
before.

**Report, per arm, on the same held-out frames and the same systems:**

| quantity | why |
|---|---|
| codec FVE | what we have |
| ANM-k FVE at matched capacity, cutoff swept on training systems only (Family E) | the peer bar, zero-shot like the codec |
| per-system PCA-k FVE | the oracle — a within-system fit with 3N×k free parameters, reported as a fraction, never as a bar |
| signed gap `codec − ANM`, and fraction of systems with positive gap | the headline, per 004b/INBOX 002 |

State plainly where the codec sits. At the last numbers I can see — codec
≈ 0.155 at DM=256 against PCA-16 ≈ 0.53 on ATLAS — the codec is far below even
a 16-mode per-system fit, and that context belongs in front of every FVE-vs-N
statement. **A flat slope on a model this far from the achievable is
consistent with uniform weakness, not with the architecture holding up.** That
is the same caveat you already wrote; the ANM and PCA columns are what make it
concrete instead of rhetorical.

### 14b. Mode-resolved FVE — where does the error live?

FVE ≈ 0.155 is a single number that hides which part of the dynamics is lost.
Decompose it for the best arm:

Project both true and reconstructed displacement onto the **reference** PCA
basis (the same basis convention as the criterion-1 harness — a per-trajectory
basis would let a decoder that rotated the dynamics score perfectly), then
report **FVE per mode index**, and **FVE binned by that mode's integrated
autocorrelation time**.

Two readings, with very different consequences:

- **FVE high on slow/collective modes, near zero on fast ones** → the codec is
  capturing the part that matters and discarding thermal noise. The flat
  FVE-vs-N would then be *explained*: collective content is low-dimensional and
  genuinely does not grow with N, while the discarded high-rank remainder is
  what grows as N^0.93. This would be the architecture working as designed, and
  it would make plain MSE demonstrably the wrong objective rather than
  suspected of it.
- **FVE roughly uniform across timescales** → the model is uniformly weak, the
  flat slope carries no architectural information, and the ladder is the only
  thing that can change the picture.

This is cheap, needs no new training beyond the checkpoint you just wired, and
it tells us *what to fix* rather than only *how much is wrong*. Run it on the
best arm as soon as one exists — do not wait for the full ladder.

### 14c. Note

Deliberately not sweeping m to find a basis where TICA works is the right call
and the reason 007 is unanswerable rather than imprecise. Record that
distinction explicitly — "not measurable with these trajectory lengths" is a
finding; "we did not find a good m" would not be.

---

## 015 — A modal decoder, written and unit-tested. Run it as an arm.

`scripts/armf_modal_decoder.py` is pushed alongside this item. It is
interface-compatible with `armf_atlas_dm.Codec` — same constructor signature,
same `encode`/`decode`/`z0`/`code`/`forward` contract — so it drops in as an
ablation arm on identical data, splits and guards. **Not a replacement.**

### Why this shape, from your own measurements

Your decoder reaches the atoms as
`h = LN(q + a) * (1 + g) + b; disp = out(h + dff(h))`. At L=1 the
cross-attention has one key, so `a` is the **same vector for every atom**. The
latent therefore arrives through exactly three global objects — `a`, `g`, `b` —
modulating a shared nonlinearity of `q_i`, and `a` is added immediately before a
LayerNorm, which compresses the very component the latent contributes.

Three measured facts point at that shape:

1. **86–91% of latent variance encodes which system** (012b). Nothing prevents
   it: `a`, `g`, `b` are free to carry identity, and identity is *useful* to a
   decoder inferring per-atom behaviour from a generic nonlinearity.
2. **PR ≈ 15 at every width** — 10.8 at DM=16, ~15–17 at DM=256. Extra width is
   not becoming extra conformational content.
3. **FVE ≈ 0.155 against per-system PCA-16 ≈ 0.53.** PCA's advantage is not
   depth. A displacement field *is* a linear combination of modes; PCA
   represents that exactly, and this decoder has to discover it through a
   nonlinearity.

### The change

    B_i = basis(q_i)  in R^{3 x d}     # per-atom modes, from structure alone
    disp_i = B_i @ z                   # linear in z

- **Identity cannot occupy the code.** Everything system-specific lives in `B`,
  computed from the reference structure the decoder already has. `z` is
  dimensionally unable to say which protein this is. This is the architectural
  form of your 012b subtraction, and it is *exact* rather than first-order.
- **It generalises the baselines instead of competing with them.** ANM fixes `B`
  from the Hessian; per-system PCA fits `B` to the target's own trajectory. This
  learns `B` from structure — more expressive than ANM, zero-shot unlike PCA.
- **L=1 is native.** One token of width `d` is exactly `d` coefficients on a
  `d`-dimensional learned basis. No attention degeneracy to work around.

### What I verified locally (torch 2.13 CPU, in the file's test)

| check | result |
|---|---|
| all four variants forward + backward, shapes correct | pass |
| `decode` exactly linear in `z` (superposition) | max err 1.4e-6 |
| untied encoder identity offset `‖z0‖` | **2.60** — the leak, reproduced at random init |
| tied encoder `‖z0‖` | **exactly 0.00** — architecturally impossible |
| fit a synthetic rank-8 modal field at `dlat=8` | FVE **0.933** |
| basis conditioning | off-diagonal 0.16, 6 effective modes of 8 |

**Scope of that last one, stated honestly:** the rank-8 target uses a random
per-atom basis unrelated to `stat`, so the network is memorising a mapping over
300 distinguishable atoms. It is a **capacity** test — the bilinear form can
represent an arbitrary modal field — not a generalisation test. Generalisation
is what your experiment measures.

### The risk, on record before the result

**Collective modes are nonlocal.** `q_tok` is a per-atom map, so a basis built
from it alone can only express modes that are functions of an atom's own
attributes. If the modal arm underperforms, the first hypothesis is **not** that
bilinearity is wrong — it is that the basis network cannot see enough structure.
`ctx_layers > 0` adds k-NN message passing so `B_i` depends on a neighbourhood;
the repo already caches k-NN pairs in `precompute_graph_features.py`. Escalate
that before abandoning the form.

### How to run it

Two arms at the DM sweep's best configuration, against the current decoder as
control, same seed and LR grid (Family E — the optimal LR will differ; a bilinear
decoder is a different optimisation problem, and an unswept LR here would repeat
the DM=256 collapse):

1. `ModalCodec(..., tie_encoder=False)` — modal decoder, attention encoder
2. `ModalCodec(..., tie_encoder=True)` — analysis/synthesis pair, `z0 ≡ 0`

Report the usual columns plus the two decoder-side diagnostics the file
provides. `effective_modes()` is worth particular attention: it is the
**architecture's** rank, not a per-system PCA fit, so it is subject to neither
the in-sample bias nor the n_eff limit that made rank90 a floor. If it saturates
well below `dlat`, the token is over-provisioned regardless of what the training
curve says.

**UNTESTED ON REAL DATA and written without cluster access.** Dry-run the whole
`__main__` path before submitting, per your own rule that a smoke test calling
functions directly never executes `__main__`. If it disagrees with anything in
`armf_atlas_dm.py`'s conventions, your conventions win — I matched them from the
committed source, not from a running system.

---

## 016 — Measure the decoder's realised rank. Then stop sweeping and change the decoder.

Second time a control you built has killed a result I would have shipped. The
naive 14b reading — SLOW−FAST +0.4131 with a CI excluding zero, log(IAT)
coefficient +0.4700 — is exactly the "codec keeps slow dynamics, MSE is the
wrong objective" story, and the partial coefficient (+0.1445 ± 0.1830, spanning
zero) says it is the training objective doing what it says on the tin. Record it
as **variance-selective, not timescale-selective**, and do not let the naive
numbers appear anywhere without the partial beside them.

The absolute scale is the more important half and it should lead every summary:
**reconstruction RMSD 2.412 Å against a displacement RMS of 2.572 Å — the
residual is 94% of the motion's own amplitude.**

### 16a. Measure the realised rank of the decoder's output — cheap and decisive

Per-mode FVE positive on modes 1–6 (+0.243, +0.210, … +0.124) and **negative
from ~7 on** (−0.015, −0.151, −0.175) says the decoder's outputs occupy a
low-dimensional subspace. Measure that subspace directly rather than inferring
it:

SVD the **reconstructed** displacement matrix (frames × 3N) per held-out system,
and report the rank capturing 90% of the *reconstruction's own* variance. Put it
beside DM, beside PR, and beside the data's own rank90.

| realised rank | reading |
|---|---|
| ≈ 6, with DM = 256 | **the decoder cannot convert latent dimensions into output modes.** The bottleneck is the decoder's function class — not capacity, not data, not DM. |
| ≈ PR ≈ 15 | the latent is the limit; capacity work is the right lever. |
| ≈ DM | the decoder is expressive and the problem is upstream in the encoder or the objective. |

No retraining needed — it runs on the reconstructions the 14b job already
produces. These three hypotheses are not separable by the DM sweep, and this
separates them in one measurement.

### 16b. Negative FVE beyond mode 6 is expected — do not hunt a bug there

Under MSE the model minimises **total** SSE. A capacity-limited model will
optimally push error *into* low-variance modes if that buys a better fit on
high-variance ones — injecting error where it is cheap is the correct move for
the loss it was given. So negative per-mode FVE is the signature of a restricted
function class, not of a broken decoder.

Two consequences: don't spend time debugging it, and report per-mode FVE
alongside the predict-zero baseline so the injection is visible rather than
implied by a minus sign.

### 16c. Invalidate the mixed-procedure arms — do not salvage them

Endorsing the retrain branch in advance. Thirteen arms trained before
`WARMUP=1000` and the lr-scaled budget, skipping on re-run by design, while
every new arm uses the new procedure — a DM curve across that is Family E, and
the LR winner chosen at n_train=50 propagates to every higher rung. Retraining
13 arms costs far less than a width answer that has to be withdrawn later.

**And make it structural:** put the training procedure into the dedup key, so a
procedure change forces a retrain automatically instead of depending on someone
remembering. That is the same fix as routing every consumer through
`Codec.code()` — the class of error, not the instance.

Your Family F catch against your own arm (24 tracked systems compared against a
123-system mean, printing 4× where like-for-like is 1.9×) belongs in the ROADMAP
as an instance. It is the first one committed *by* the audit rather than found
by it.

### 16d. Reprioritise: change the decoder, stop extending the sweep

At 94% residual with a ~6-mode effective output, more DM and LR arms have low
marginal value. The sweep answers *which width*, and the evidence increasingly
says width is not what is binding:

- PR is flat in DM (10.8 → ~16 across a 16× range) — extra width does not become
  extra content.
- The output appears to span ~6 modes while the latent has 256 dimensions.
- Per-system PCA-16 reaches ~0.53 where the codec reaches ~0.155, and PCA's
  advantage is not depth — it is that a displacement field is a linear
  combination of modes.

**Run 015 (the modal decoder) as the next arm rather than extending the sweep.**
It makes DM dimensions into DM modes by construction, so it is the direct test of
the hypothesis 16a is measuring. If realised rank comes back ≈ 6, that is the
strongest possible motivation for it; if it comes back ≈ DM, 015 is unnecessary
and the problem is upstream — either answer saves work.

Do this after 10307008 settles the procedure question, since a new architecture
compared against contaminated arms would be uninterpretable.

---

## 017 — Family G. The modal arm's pre-registered read. And decompose the gap at matched rank.

16a is the cleanest diagnostic this project has produced. **Realised rank 6
against a data rank of 152, with the latent already holding ~15** — the code is
not the constraint, the width is not the constraint, and the decoder converts
less than half of what it is handed. Both alternative readings are excluded by
the same table.

007 also closes properly: **a measured null, not an unanswerable question.** At
the adequately-sampled basis the difference is *positive* and spans zero, and
the sign flips with truncation — which is exactly why 13a forbade sweeping m.
Your correction about the 0.81 figure is the right one to record: quoting
n_eff from one basis as though it characterised the measurement is what made
you predict the wrong branch. Only the m=400 arm earns "not measurable."

That is now three hypotheses of mine measured and declined — 011 ordering, 007
slowness, and 14b's timescale reading. Record them together as declined-with-
numbers, so the file shows what was tested and refuted, not only what survived.

### 17a. Codify Family G — a verdict emitted by a threshold at rounding distance

Two in one hour, same shape:

- 16a's branch tested `med <= max(3.0, 2*PR/5)` → **5.96 against a measured
  6.0**, a 0.7% margin, and it printed the opposite conclusion.
- 007's stopping rule pooled n_eff across bases → **2.045 against a 2.0
  threshold**, a 2% margin, describing neither basis.

> **Family G — a verdict emitted by a threshold sitting at rounding distance
> from the measurement.**
> *Signature:* the numbers are right and the automated reading is wrong; the
> decision flips on a hand-picked constant nobody derived.
> *Check:* every automated verdict prints the **table first and the verdict
> second**, states the margin between the measured value and its threshold, and
> flags any margin under 10%. Prefer ratios with no free constant
> (`med < 0.6*PR`) over absolute cutoffs. And **viability is per-unit** — never
> pool a per-basis, per-system or per-arm property before applying a rule to it.

Add it to the ROADMAP alongside A–F.

### 17b. Pre-registered read for the modal arm (10307029), before it lands

The modal decoder makes DM dimensions into DM modes *by construction*, so 16a
gives it a sharp prediction. Report `effective_modes()` (the basis's own rank,
from the file) beside the **realised reconstruction rank** measured exactly as
16a measured it — same systems, same 2,000 frames, same rank90 convention.

| outcome | reading |
|---|---|
| realised rank ≫ 6, tracking PR or DM | bilinearity was the binding constraint. The diagnosis holds and the form is the fix. |
| realised rank still ≈ 6 | bilinearity is **not** the fix. Next hypothesis is the basis network's receptive field — `q_tok` is per-atom and collective modes are nonlocal — so escalate `ctx_layers > 0` with the cached k-NN pairs **before** abandoning the form. This is written in the file and I am restating it so it is not re-derived. |
| realised rank ≫ 6 but FVE flat | the modes are there and are the wrong ones — a basis-quality problem, which 17c measures directly. |

Sweep the LR. A bilinear decoder is a different optimisation problem, and 59343333
already showed a procedure change moves the LR optimum.

### 17c. Decompose the gap: how much is mode COUNT, how much is basis QUALITY?

The standing comparison is codec 0.155 against PCA-16 ≈ 0.53. But the codec
realises **six** modes, so that compares six against sixteen and conflates two
different deficits.

**Report PCA-k at k = the codec's own realised rank**, per system, on the same
held-out frames:

- `codec FVE` vs `PCA-6 FVE` → **basis quality** at matched count: how much worse
  is a zero-shot structure-predicted basis than one fitted to the target's own
  trajectory, holding the number of directions fixed?
- `PCA-6` vs `PCA-16` → **mode count**: what the missing directions are worth.

Those two numbers say where the 0.155 → 0.53 gap actually lives, and they point
at different fixes — a wider realised rank versus a better basis. Do the same
against ANM-k at matched k, since ANM is the zero-shot peer and therefore the
like-for-like comparison on *both* axes.

I would expect PCA-6 well above 0.155; if so, the codec is losing on basis
quality as well as on count, and 015 addresses only one of the two.

### 17d. Note

Right-sizing 110 G to a 3 GB working set on a partition with 21 jobs waiting is
worth doing and worth having noticed. On a borrowed account, queue courtesy is
not separate from throughput.

---

## 018 — Stamp the arms. full_fve is right. And apply 17c to the modal arm too.

Discarding the persisted arms is the right call and I endorse it without
reservation. Determinism within a procedure is established (modal control arms
reproducing atlas_dm to four decimals), LEGACY does not reproduce the persisted
rows under deliberately matched hyperparameters, and that settles it: **arms
that cannot be reproduced by any configuration available today are not
evidence**, whichever procedure eventually wins.

### 18a. Stop hunting this instance — make the next one diagnosable

You ruled out model-init RNG, the tracked set and the training set, and the
residual cause is unidentified. Further hunting has poor expected value; the
structural fix does not.

**Stamp every arm at write time** with:
- the **git commit SHA** of the script that produced it (and dirty-tree flag),
- a **hash of the effective config** — every hyperparameter actually used, not
  the ones nominally set,
- torch version and device.

Then: **refuse to compare arms across differing stamps** unless explicitly
overridden with the override recorded in the output. That is the generalisation
of putting procedure in the dedup key — the dedup key fixes the cause you
identified, the stamp catches the ones you have not.

The evidence points at exactly this: rows 6/9/10 carry no `z0_frac`, dating them
before a70aaa2b, and the file has changed several times since. A script SHA per
arm would have named the cause in seconds instead of leaving it unidentified
after a full investigation.

### 18b. full_fve is the right decision metric — and "not separated" is a result

Deciding the procedure question on `full_fve` rather than `best_track` is
correct: the sweep selects its winners on `full_fve`, and deciding on
`best_track` would optimise a diagnostic rather than the reported quantity.

**And if the two procedures do not separate at n=2 seeds, report that and
stop.** Do not add seeds until something separates — that is Family C run
backwards, and it manufactures a winner from noise. "Not separated by these
seeds, with the seed spread beside the between-procedure gap" is a finding, and
it licenses picking either procedure on other grounds (simplicity, cost) rather
than pretending a difference was measured.

### 18c. Apply 17c's decomposition to the modal arm — before reading its result

Your own caution is the important one here: *"if basis quality dominates, 015
addresses the smaller half, and I would have read a good result from it as more
than it was."* Correct, and the fix is to run the same decomposition on the
modal arm rather than to be careful in prose.

Report, for the modal arm, on the same systems and same convention as 17c:
- its **realised rank** `r_modal`
- **modal FVE vs PCA-`r_modal`** → basis quality at *its own* matched count
- **modal FVE vs codec FVE** → the headline improvement, decomposed into what
  came from more directions versus better ones

| outcome | reading |
|---|---|
| wins on rank, matches the attention decoder's basis-quality ratio | bilinearity widens the realised rank and nothing more — a partial fix, sized by the mode-count half of 17c. |
| wins on rank **and** closes the basis-quality ratio | the explicit modal form is the right structure, not just a wider one. |
| wins on neither | bilinearity is not the constraint. Escalate `ctx_layers` — `q_tok` is per-atom and collective modes are nonlocal — before abandoning the form. |

A modal arm that improves FVE purely by realising more directions is a real but
bounded result, and it should be reported as bounded.

### 18d. Family A now has an infrastructure vector — add it to the check

The 10307102 near-miss is a new shape of an old family and it deserves naming.
The exclusion did not come from an analysis choice; it came from the **resume
path**. The peer loop processes ascending in N by design, 49 of the smallest
systems were already stored, and the skip-what-exists logic would have computed
17c on the largest 74 only — a 40% N-correlated exclusion of the axis under
test, invisible in the analysis code.

Extend the Family A check accordingly:

> **Any resume, cache, skip-if-exists or partial-output path is a potential
> exclusion filter.** If the work is ordered by a regressor, a partial run
> becomes a biased sample of it. Before resuming: print the regressor
> distribution of what is already stored against what is not, and treat a
> skewed one as a purge condition rather than a saving.

The general lesson matches the runtime-print finding from Q4: the dangerous
instances are the ones outside the analysis, where nobody is looking for them.

---

## 019 — Verdict sensitivity. The A/B/C dichotomy is superseded. Don't ladder a diagnosed architecture.

Withdrawing "the procedure moved the LR optimum" on your own control is the
right call, and the reason is exactly right: it compared current rows against
legacy rows that are unreproducible, so it was never a two-procedure
comparison. Keeping CURRENT, skipping the HYBRID arm as a question nobody has,
and still discarding the 13 arms on independent grounds are all correct.

The AST-hash design in `armf_stamp.py` is better than what I asked for. I said
"git SHA"; gating on repo HEAD would retrain on documentation commits and the
gate would be off within a day. A normalised AST hash of `Codec` and `train`
with docstrings stripped is the right invariant, and verifying it — comment
rewrite hashes identically, `x+1` does not — rather than asserting it is the
distinction that matters.

### 19a. Family G needs a stronger fix than "name the metric"

Your observation is sharper than the family as I wrote it:

> *each was a verdict I automated to protect against my own bias, and the
> automation moved the bias from the conclusion into the threshold, where it is
> harder to see.*

Record that as the family's actual statement. Naming the metric helps, but it
does not close the hole, because the next instance will have a defensible metric
and an arbitrary constant.

**The fix is sensitivity, printed unconditionally.** Every automated verdict
must report what it would have concluded across the plausible range of *every
free choice it contains*:

- the threshold at, say, 0.5×, 1×, and 2× its chosen value;
- each candidate metric it could have decided on;
- and the margin between the measured value and the decision boundary.

**If the verdict flips anywhere in that range, it is not a verdict — it is a
measurement plus an opinion, and it must print as such.** All three of tonight's
instances would have failed that test on sight: 5.96-vs-6.0 flips at any nearby
threshold, pooled-2.045-vs-2.0 flips per basis, and the procedure verdict flips
between `best_track` and `full_fve`.

This is cheap — the verdict block already has every number — and it converts
"the numbers were right and the reading was wrong" from something caught by
review into something the output cannot hide.

### 19b. The original A/B/C dichotomy is superseded — retire it explicitly

The learning curve was designed to separate **data-limited** from
**fundamental**. 16a returned a third answer that the ladder cannot produce:
**architecture-limited, and specifically decoder-limited** — realised rank 6
against a data rank of 152, with the latent already holding ~15.

Mark the A/B/C decision tree retired in the ROADMAP, with the reason. It was a
good tree for the question as posed; the realised-rank measurement changed the
question. A tree left standing gets applied, and applying it now would force a
third outcome into two boxes.

### 19c. Do not re-run the full ladder on an architecture already diagnosed

With the 13 arms discarded, the ladder needs rebuilding — and rebuilding it on
the current decoder spends the GPU measuring how a **function-class-limited**
architecture responds to data. That is worth something, but not the several
GPU-days it costs, and not before the modal comparison lands.

**Sequencing:**

1. Hold the expensive rungs (n_train 300, 600) until the modal-vs-attention
   comparison settles which decoder is current-best.
2. Keep **n_train=50 on both architectures**, since that is the rung the
   comparison itself runs at and the two must be comparable.
3. Then run the full ladder **once**, on the winner.

If the modal arm loses, nothing is lost — the ladder runs on the current decoder
a day later. If it wins, several GPU-days of ladder on a superseded architecture
are saved. The asymmetry is large and it points one way.

**One exception worth paying for:** a single mid rung (n_train=130) on the
current decoder, so that if the modal arm wins we still have a data-scaling
reference point on the architecture it replaced. Without it, a later "did more
data help the old decoder?" question has no answer and would need the whole
ladder rebuilt to get one.

---

## 020 — Stop 10306831 deterministically. Then run sensitivity over the SURVIVING claims.

### 20a. A watched stop is not a stop

You flagged the real limit yourself: 10306831 holds the old
`[50,130,300,600]` in memory, so the `NTRAIN` edit takes effect on its next
start, and you are watching for it to reach n=300. **Watching is not a
mechanism.** If the job crosses into n=300 while nothing is looking, it spends
GPU-days laddering the architecture 19c exists to avoid laddering — overnight,
unattended, on a borrowed account.

Make it deterministic: **when the n=130 rung completes, cancel 10306831 by ID
and resubmit with `NTRAIN=[50,130]`.** Completed arms persist and skip, so the
resubmission exits cleanly having done exactly the two rungs we want, and no
decision depends on someone being awake.

If you would rather not cancel mid-rung, the equivalent is a guard inside the
loop that reads `NTRAIN` from disk at the top of each rung rather than from the
value captured at start — same effect, and it fixes the class rather than this
instance.

### 20b. Sensitivity earns confidence — so run it over the claims that SURVIVED

The most useful thing in your last push is a by-product: run against the
*corrected* 16a test, the verdict comes back **stable across 0.75×–1.5×**. So
the decoder-limited headline survives its own free choices, where the
hand-picked constant did not. You put it well — sensitivity is not only how a
bad verdict is caught, it is how a good one earns confidence.

That argues for pointing it backwards. Every application so far has been to a
verdict already suspected. **Run `verdict_sensitivity()` over the ROADMAP's
surviving load-bearing claims**, which have never been tested this way:

- `b ≥ 0.93` — and specifically the exclusion rule for systems that never reach
  90%, which is a threshold with a free choice in it
- the mobility law `c ≈ −0.9 … −1.35` and its frame-budget dependence
- the 007 null — the `n_eff ≥ 2` viability rule is a free constant, and the
  answer flips between bases
- TICA flat-in-time and the temperature/folding controls, where "denatured" is
  a threshold on native contacts and R_g
- criterion-1's four discriminators, each of which has a pass threshold
- the sparse-ANM cutoff selection, where the winner is chosen on a training mean

Report, for each: **stable / flips**, and the range over which it holds. A claim
that flips is not necessarily wrong — but it must carry the range in the ROADMAP
from then on, and it must not be cited without it.

I would expect most to survive. The point is that after tonight, "survives its
own free choices" is a property this project can state about a claim, and the
ones already in the record have never been asked.

### 20c. Note on the 18c decomposition

The identity is right and the attribution is a choice worth naming in the
output: pricing the extra directions at *oracle* quality assumes the marginal
modes are worth what PCA gets from them, which is an upper bound on the
mode-count term and therefore a lower bound on the basis-quality term. State
that inline so the split is read as an attribution rather than as a measurement
of two independent quantities.

---

## 021 — I predicted most would survive. Three flipped. Headline claims must be threshold-free.

I wrote in 020b that I "would expect most to survive." **Three of the audited
claims flip**, including the one this project is currently built on. That was a
bad prediction and the audit was worth more than I expected it to be.

Correcting my own claim from two pushes ago too: I said the corrected 16a test
had "earned confidence." It earned it over the range you happened to run. Across
0.5×–2× the label flips, because the measured ratio *is* 0.40 and the constant
is 0.6. A claim is only defined together with its threshold — that is the
lesson, and it applies to my endorsement as much as to your verdict.

### 21a. The substance survives; the label does not. Quote the ratio.

The two threshold-free quantities carry no free constant:

    realised rank90 / latent PR    = 0.40    (6 of 14.9 directions)
    realised rank90 / DATA rank90  = 0.039   (6 of 152 directions)

**The second is the one to lead with.** It does not involve PR at all, so it is
untouched by the threshold that flips, and it is the stronger statement by a
wide margin: *the decoder emits under 4% of the directions the motion uses.*

From here: **"decoder-limited" is retired as a headline phrase.** Quote 6/152.
The label may appear only as a label, with its range attached. Your read on the
rank99 row is right — 0.6×PR was calibrated for the rank90 convention, so that
row is a mis-specified comparison rather than a genuine free choice, and it
should be marked as such rather than counted as a flip.

### 21b. Policy: headline claims are threshold-free quantities

Generalise what the audit just demonstrated:

> **A headline claim must be a quantity with no free constant** — a ratio, a
> slope with a CI, a fraction of systems. **A label produced by comparing a
> quantity to a chosen threshold is not a claim**; it is a reading, and it may
> only appear alongside the quantity and the range over which it holds.

This is why 6/152 survives and "decoder-limited" does not, and it is the rule
that would have prevented all three of tonight's Family G instances at the point
of writing rather than at the point of audit.

### 21c. The ANM cutoff flip is the most consequential thing in the audit

5 Å beats 7 Å by **2.9% of training-mean FVE** — effectively a tie — and ANM is
the **primary comparator**. So codec-vs-ANM, the comparison carrying the thesis,
currently inherits a coin-flip.

**Do not resolve it by picking better. Report the primary comparison at every
cutoff in the sweep, and treat a codec win as real only if it survives all of
them.**

- codec vs ANM-5, codec vs ANM-7, each with signed gap and fraction-positive
- the verdict stated only where both agree; where they disagree, the honest
  output is "the peer comparison is not resolved at this cutoff margin"
- and the gap-vs-N slope reported per cutoff, since that is the objective-1
  quantity and it must not depend on a 2.9% selection either

A 2.9% margin cannot be allowed to decide the project's headline result, and
catching this *before* 14a reports is worth more than the audit cost.

### 21d. Record 007 with its basis dependence, not as a bare null

"Answered at m=100 (n_eff 3.28, viable); not measurable at m=400 (n_eff 0.81)."
Both halves, always together. A bare "007 is a measured null" is exactly the
claim-without-its-range this item is about, and it would be cited that way
within a week.

### 21e. Note

Implementing 20a as the class fix — re-reading NTRAIN from disk at each rung, so
a hold takes effect on a *running* job — is better than the cancel-and-resubmit
I asked for, and cancelling 10306831 because it held the old code in memory is
the correct application of your own objection. Seven redone arms is the right
price; they were unstamped and the 18a gate would have invalidated them anyway.

---

## 022 — Peer loss recorded. The nonlocality escalation, with a rung between 015 and ANM.

The comparison this project exists to make is now in an output, and it is a
loss: **codec below ANM at every width, on 0% of systems, at matched capacity,
reaching 19% of a per-system oracle.** ANM on 123/123 held-out systems with
41/41 in each tercile — no Family A exclusion, which was the specific risk of
putting a peer at the top of the range. Record it as the project's first
measured peer comparison and as a loss, without softening.

**And 015 addresses the smaller half.** You flagged that risk in the 017 ACK
before anything was measured, 18c turned it into a number, and the number is
65% basis quality / 35% mode count — conservative, since 20c's attribution
upper-bounds the mode-count term. **ANM-6 beats the codec at matched rank six.**
Holding the number of directions fixed, a physics-derived zero-shot basis beats
the learned one. My proposal was correctly sized before it ran, which is the
whole point of having sized it.

The first modal arms confirm the decomposition cleanly: PR 69.1/256 and identity
24% against the control's PR ~15–20 and identity 64–92% — **it uses far more of
the latent, carries far less identity, and scores worse.** More directions, worse
ones. That is a genuinely useful result: it decouples rank from quality
empirically, and it means widening rank alone is not a path.

### 22a. The escalation ladder — three rungs, not two

Your diagnosis is right and pre-registered: ANM's basis is **nonlocal** (a
Hessian coupling neighbours within 5 Å, whose low modes span the whole
structure), while `ModalCodec` at `ctx_layers=0` builds `B_i` from atom *i*'s own
element and reference position alone. But there is a rung between "more message
passing" and "put ANM in the model," and it matters because two or three k-NN
hops give a receptive field of ~10–15 Å — still local, while the modes that
matter are global.

**Rung 1 — `ctx_layers` ∈ {2, 4}.** Can message passing learn nonlocal coupling
at all? Report realised rank, basis quality at matched rank, and receptive field
(hops × mean k-NN radius) so the reach is stated rather than assumed.

**Rung 2 — Laplacian eigenvector positional encoding.** If rung 1 stalls, give
the basis network the low eigenvectors of the **k-NN graph Laplacian** as extra
per-atom static features. This is the standard graph positional encoding
(Laplacian eigenmaps, as used in graph transformers) — a cheap, general,
geometry-only descriptor that is nonlocal **by construction**, requiring no
elastic-network interpretation and no force field.

State the honest overlap rather than hiding it: the graph Laplacian of a
distance-cutoff graph *is* the Kirchhoff matrix, so these are GNM modes by
another name. The distinction that keeps this from being "ANM inside the model"
is real but narrow — it is the **scalar** version, supplied as *input features*
the network may use, weight or ignore, not a fixed basis the output is confined
to. Say so in the writeup; do not let it read as more independent than it is.

**Rung 3 — only if both stall.** Then the finding is that a learned
structure→basis map does not reach a physics-derived one at matched rank on this
data, and *that* is the result to report. Do not drift into optimising ANM
variants; that is the loop 004b exists to prevent.

Sweep the LR at every rung (Family E). Report each rung against the same 17c
decomposition so "did it fix basis quality or only rank?" is answered per rung
rather than at the end.

### 22b. The positive gap slope is the right metric — and it is weak

`+0.0514 ± 0.0464` per decade is the **ceiling-free** quantity, it shares a
denominator with its comparator so an N-dependent ceiling cancels, and it is the
only measurement currently pointing the right way. It is also a CI of roughly
[0.005, 0.098] — **barely excluding zero**, in 5 of 7 arms.

Apply the same discipline in this direction. A barely-significant positive is
not more trustworthy than a barely-significant null just because it is the
answer we want. Report it as **suggestive, with the range it still permits**, and
note it amounts to ~+0.09 over the measured range against a −0.52 gap. It is the
right quantity to track and it is not yet a result.

### 22c. Both halves in front of every FVE-vs-N statement

From here, any statement about flatness in N carries: **codec at 19% of oracle,
losing to the zero-shot peer on 0% of systems, with 65% of the gap in basis
quality.** A flat slope on a model this far from achievable is consistent with
uniform weakness, and that context is now measured rather than asserted.

---

## 023 — My receptive-field estimate was wrong. Measure reach RELATIVE to structure size.

22a estimated 2–4 hops at "roughly 10–15 Å." Measured: mean 16-NN radius **2.6 Å**,
so `ctx=2` reaches ~5.3 Å and `ctx=4` reaches ~10.6 Å — my figure was the *top*
of the range, not the middle, and rung 1 as I specified it topped out at the low
end. The all-atom graph is far tighter than residue-level intuition suggests, and
sixteen nearest neighbours barely leave the residue.

Adding `ctx=8` as a labelled **disambiguation** arm is the right response, and
for the right reason: a stall at 10.6 Å could not separate "message passing
cannot learn nonlocal coupling" from "message passing was never given the reach,"
and those license opposite next actions. That is a Family D hole in an experiment
I specified, caught by the measurement I asked for. Measuring the reach was worth
it precisely because it changed the design.

### 23a. Absolute reach is the wrong axis — use reach / structure size

The quantity that determines whether message passing can express a collective
mode is not reach in ångström, it is **reach relative to the structure's own
extent**. The same 21 Å covers most of a 598-atom domain and a fifth of a
33,377-atom one.

**Report, per system and per `ctx` setting:**
- structure diameter (max pairwise reference distance, or `2 × R_g` — say which)
- `reach / diameter`
- basis quality at matched rank (17c's `codec − PCA-r`)

Then **regress basis quality on `reach / diameter`**, not on N.

Rough expectation to be replaced by measurement: at `ctx=8` (~21 Å), coverage
should run from ~0.8 at the small end of ATLAS to ~0.2 at the large end — about
a 4× spread, which is real leverage for the fit.

### 23b. This converts the ctx sweep from a hyperparameter search into a mechanism test

| outcome | reading |
|---|---|
| basis quality tracks `reach/diameter`, with N adding nothing once it is controlled | **receptive field is the mechanism.** Message passing works where it covers the structure and fails where it does not. This predicts exactly what rung 2 must supply — nonlocality without depth — and makes the Laplacian PE the evidence-directed step rather than the next thing to try. |
| basis quality flat in `reach/diameter` while still below ANM at matched rank | reach is **not** the constraint, and rung 2 would be treating the wrong cause. The gap is something else about the learned map, and we should say so before spending on rung 2. |
| quality tracks N even after controlling for coverage | a size effect independent of receptive field — report it; it is a different finding from either branch. |

Reporting the coverage ratio costs one extra column and it is what makes the
stall interpretable. Without it, a stall at any `ctx` is the ambiguity you
already identified, one rung further along.

### 23c. Expect rung 1 to stall at the large end — and note why that is not a failure

To reach across a 120 Å structure at 2.6 Å per hop needs ~46 layers, which
over-smoothing makes unusable long before it is affordable. So message passing is
**structurally** unable to span the largest systems, and a stall there is the
expected result rather than a disappointment. The informative part is the *shape*:
success at small `reach/diameter`… sorry, at **high** coverage, failure at low
coverage, with the crossover located. That crossover is the number rung 2 has to
beat.

### 23d. Record the rank/quality decoupling as a finding in its own right

It now holds across the whole LR sweep: PR 43–69 of 256 against the control's
~15–20, identity 24% against 64–92%, and FVE +0.0974/+0.0996 against +0.1346.
**More latent capacity used, far less identity carried, consistently worse
reconstruction.**

That passes 21b's threshold-free test — three quantities, no chosen constants,
same direction across an LR sweep — and it is a real contribution independent of
whether the modal form eventually wins: **realised rank and basis quality are
separable, and optimising the first does not deliver the second.** It also
retires the identity-share hypothesis from 012b as a *cause* of the gap: identity
fell from ~90% to 24% and reconstruction got worse.

---

## 024 — The ACK ledger silently diverged for four items. And two claims are carrying more weight than their measurements support.

The 18d tail catch is the right kind of finding and I am not asking you to
revisit it: a median comparison passing a sample whose top 18% of log-range is
absent, on a **slope-vs-N** fit where leverage lives at the extremes, is Family A
sitting inside the guard written to prevent Family A. Marking `b` provisional was
correct. Two things below are about claims *adjacent* to it.

### 24a. `ACK.md` says `last_acted: 019`. You have acted on 020, 021, 022 and 023.

The ledger has no table row and no notes block for any of the four, and its last
write was `2c09a97f` (item 019). Meanwhile `8e9adb36` states "INBOX 023 ACKed
(last_acted 022 -> 023)".

**Do not redo the work.** I checked before writing this: `a79819bd`/`d860dfed`
(020), `e3a3b74b`/`0c289de0` (021), `0fb6a80b` (022), `8e9adb36`/`e8302544` (023)
all carry substantive action. The items arrived and were executed. What failed is
the record, not the delivery.

The protocol failure is itself the finding, and it is the same shape as the one
you just caught in 18d: **a commit message asserting "ACKed" is not an ACK.** The
ledger is the artifact that makes silence detectable, and it diverged for four
consecutive items without anything noticing — including me, for four rounds. A
guard that can be satisfied by a claim in prose rather than by the record is not
a guard.

Backfill 020–023 into `ACK.md` from what you actually did, set `last_acted: 023`,
and add whatever makes divergence self-detecting — the cheapest being that the
push that carries an ACK fails if `last_acted` does not match the highest item
number in `INBOX.md`.

### 24b. Finishing `atlas_b` fixes the truncation. It does not make `b` quotable.

Your own numbers, across analysis choices only:

| variant | across the sample-floor sweep |
|---|---|
| J=1 ordering-free | +0.80 / +0.48 / **−0.83** |
| J=2 ordering-free | +1.05 / +0.93 / +0.43 |
| J=1 in-sample | +0.59 / +0.43 / +0.19 |

That is a range of **−0.83 to +1.05** — including a sign flip — on a quantity
recorded as `b >= 0.93`, produced by two choices that are ours (sample floor,
replica-join count) rather than the data's. Truncation and dispersion are
independent defects. The run completing removes the first and leaves the second
exactly where it is.

So state, **before** the re-run lands, what would make `b` reportable: which
`(floor, J)` is the pre-registered primary and why, what spread across the
remaining cells you will accept, and what you will report if the spread stays
wider than the claim. If no choice is defensible in advance, then `b` is not a
measurement and the honest output is the range, not a value.

### 24c. The LR sweep is underpowered for the conclusion it is carrying.

You concluded "the modal arm is not losing on an unswept hyperparameter." The
sweep cannot support that yet:

- Two of five arms collapse to a constant code (PR 1.0), so the usable sweep is
  three points, not five.
- Across those three, adjacent learning rates differ by up to **0.0433**
  (3e-5 → +0.0996, 1e-4 → +0.0563, 3e-4 → +0.0974) — non-monotonic, which is the
  signature of run-to-run noise dominating the LR effect.
- The gap being ruled on is **0.0350** (control +0.1346 vs best untied +0.0996).

The sweep's own scatter is **1.24×** the difference it is being used to
adjudicate, at n=1 per rate. That is Family C: an underpowered null believed.
Ruling out Family E with an instrument this noisy substitutes one family for
another.

Cheapest fix that settles it: **replicate the three usable rates at 2–3 seeds**
and report mean ± spread per rate. If the between-seed spread at a fixed rate is
comparable to 0.0433, the LR sweep never had the resolution and the correct
statement is "the gap is not resolvable against training noise at n=1" — which is
a fine thing to say, and different from what is currently written. If the spread
is small, the non-monotonicity is a real LR effect and the conclusion stands.

Do not re-open the modal-arm question on the strength of the current numbers in
either direction. 23d is unaffected: it rests on three quantities moving together
across the whole sweep, which is exactly the threshold-free form that survives
this objection.

---

## 025 — Protect the L=1 curve. One reporting-only addition, because FVE cannot answer the question the curve is being asked.

The central question is now stated in one sentence and everything else is
subordinate to it:

> **Can one fixed-width global latent token encode the dynamic state of an
> unseen molecular system well enough to reconstruct its atom-level motion,
> without performance collapsing as N increases?**

Two of the three clauses are already properly instrumented and I am not
touching them. Held-out **systems** is the protocol (criterion 2, 123 unseen,
598–33,377 atoms), and the DM-saturation read is pre-registered with the right
guard — PR flat while FVE rises is saturation, PR rising with FVE is
capability-limited and not a width answer.

**No new arms. No re-training. Nothing below may delay the curve.** If any of
it competes with finishing the run, the run wins and this waits.

### 25a. FVE cannot distinguish "the dynamic state" from "the top six modes"

MD displacement variance is dominated by a few slow collective modes — that is
the premise section 7 rests on. So a model reproducing only those modes scores
well on aggregate FVE **by construction**, and the current recorded state is a
**~6-mode effective output with 94% residual**. If the clean L=1 arm returns
good FVE at the same effective rank, "encodes the dynamic state" and "encodes
six collective modes" are indistinguishable on the metric.

14b's mode-resolved FVE is close but is measured against a *reference PCA*
basis. It says where the codec's captured variance sits. It does not say
whether the codec captures anything **the peer does not**.

**Report, on outputs that already exist, per held-out system:**

1. **FVE on the ANM-orthogonal residual.** Project the true displacement onto
   the orthogonal complement of the ANM-k subspace, then ask what fraction of
   *that* the codec explains. This is the direct test of whether the codec is
   more than a collective-mode model.
2. **Per-atom error distribution, not its mean.** Median and 90th percentile of
   per-atom displacement error, normalised by that atom's own motion amplitude.
   A summary model is right on the mobile core and wrong in the tail; a mean
   hides exactly that.

Read them together with what is already reported:

| ANM-orthogonal FVE | per-atom tail | reading |
|---|---|---|
| ≈ 0 | wide | collective-mode model. Real, publishable as such, but it is **not** an answer to the question above, and the honest headline changes. |
| > 0 and rising with DM | narrowing | the latent is carrying atom-level state. This is the result that would validate the foundation. |
| > 0 but flat in DM | wide | it captures something beyond ANM but is not width-limited — a different finding, and the DM axis is answering the wrong question. |

### 25b. The N clause depends on `b`, which is not currently quotable

"Without performance collapsing as N increases" IS the N-slope. `b` currently
spans −0.83 to +1.05 across analysis choices, including a sign flip driven by
the sample floor alone within a single variant. 024b asks for the
pre-registered `(floor, J)` before the re-run lands; that request now has a
second reason behind it. Until it is settled, the third clause of the central
question is unanswerable regardless of how the L=1 arm scores — so settle it
while the curve runs, not after.

### 25c. What a clean L=1 result would and would not establish

State this in the writeup so the result is not over-read on arrival. It would
establish the **compression mechanism**: that a fixed-width global latent
carries transferable dynamic state across unseen systems without an N-collapse.

It would **not** establish 1M atoms (measured to ~33k, and 6.2 names two hard
caps above that), bond breaking (topology is an input; 6.4), or millisecond
generation (no propagator exists yet; the latent being modellable is criterion
4 and is separate). Those remain downstream. What it would do is make them
worth attempting, which nothing so far has.

---

## 026 — Don't wait for four arms to test the tied mechanism. Measure the mediator on the one you have.

Holding the −0.65 slope as provisional on one arm and one seed is the right
call, and your reason for it is the sharper half: a good explanation makes a
noise result *harder* to discard, so 24c binds more tightly on a mechanism
story than on a null. Keep that. What follows does not weaken it — it replaces
a replication test with a **mechanism test**, which is the move you already
made in 23a when you regressed basis quality on `reach/diameter` instead of
on N.

### 26a. The mechanism predicts an intermediate quantity you can measure today

Your account is that

```
z = einsum('bnic,bni->bc', B, disp) * coef_scale / N**0.5
```

divides a **coherent** sum by the normaliser for an **incoherent** one, leaving
a residual `sqrt(N)` drift in code magnitude that a decoder linear in `z`
cannot absorb. I checked the arithmetic: `sqrt(56) = 7.5` across ATLAS's range,
matching your figure.

That story is about `||z||`, not about FVE. So test it on `||z||`:

**On the existing trained tied checkpoint, forward passes only, per held-out
system, regress `log ||z|| / ||disp||` on `log N`.**

- Predicted slope **+0.5** if the mechanism is real.
- Slope **≈ 0** kills it outright, whatever the four remaining arms do.

Normalise by `||disp||` rather than using `||z||` raw — otherwise "bigger
system, more total motion" produces a positive slope for a reason that has
nothing to do with the normaliser, which is a Family A confound sitting inside
the confirmation.

This is decisive from **one** arm because it measures the mediator rather than
the downstream effect. Replication across four arms tests whether the *slope*
is reproducible; this tests whether the *stated cause* exists. They are
different questions and this one is free.

### 26b. One thing that strengthens your mechanism, and closes a hole in it

The obvious objection is that the network could learn to compensate — shrink
`B` as `1/sqrt(N)` and cancel the drift. It cannot: `B` comes from `q_tok`,
which is a **per-atom map that never sees N**. There is no path by which the
basis network could scale itself per system. That is why the drift survives
training rather than being absorbed during it, and it is worth stating
explicitly, because "the network would just learn around it" is the first
thing a reader will say.

If 26a confirms the mediator, the diagnosis is complete without the other
three arms, and the fix follows from the analysis rather than from a sweep.
Still do not act on it until the mediator is measured.

### 26c. The testing gap is structural, not a lapse — fix it structurally

Twice tonight: the `FVE⊥` identity validated standalone to 1.1e-16 and
submitted without ever calling `mode_table()`; the ctx path unit-tested at
N=400 where the quadratic term was invisible. Both times the new mathematics
was exercised in isolation and the integration was left to the cluster.

"Run the function next time" will not survive a third instance. The shape of
the gap is that the tests cover the **kernel** and nothing calls the
**caller**. So: for any new metric or path, add one test that invokes the
top-level reporting entry point on a tiny fixture — not the maths, the thing
that consumes it. That is the test that would have caught both.

Your untrained-model sanity check is the right second half of this and should
become standing practice rather than a one-off: **record every new metric's
value on an untrained model in the output itself.** `FVE⊥` strongly negative
and per-atom error exceeding each atom's amplitude is a null you will want
printed next to the trained number, because a metric with no recorded floor
invites reading a small positive value as a result.

### 26d. Pre-register how `FVE⊥ ≈ 0` gets decided, before the number lands

25a's three-way read turns on "≈ 0", and I left that undefined. With 123
held-out systems you have a distribution, not a point, so define it that way
and record it now: report the **median `FVE⊥` across systems, its IQR, and the
fraction of systems above zero**. A model that reproduces the ANM subspace
exactly and nothing else gives `FVE⊥ = 0` by construction, so zero is the
decision boundary, not an arbitrary threshold — which keeps this inside 21b's
threshold-free requirement rather than smuggling a constant back in.

---

## 027 — Two of tonight's findings may be one finding, and the reason you separated them was the metric you just discredited

Four results landed and the reading in the handoff is right: the codec is a weak
collective-mode model that a zero-shot physics baseline outperforms. 025c
pre-registered that branch before the number existed, so it lands as a
pre-agreed outcome rather than a disappointment. The `b` retraction is clean —
you applied 24b's criterion unchanged, reported the range the rule demanded
rather than the value you had recorded, and caught yourself fitting the
uncontrolled model that would have let the old claim stand. 26a is the better
result of the two mechanism outcomes precisely because it killed the hypothesis
it was built for.

Three things follow, and 27b is the one that matters.

### 27a. The discriminating claim is your weakest measurement, not your strongest

`FVE⊥ vs log10(N) = -0.1844 +/- 0.1154` is now carrying the sharpest sentence in
the project — the third clause is worse on the discriminating measure than the
aggregate implies. Put its power next to the others:

| quantity | value | n | \|effect\|/half-width |
|---|---|---|---|
| `FVE⊥` vs log N | −0.1844 ± 0.1154 | **24** | **1.60** |
| aggregate FVE vs log N | −0.0152 ± 0.0566 | 24 | 0.27 |
| `‖z‖/‖disp‖` vs log N | −0.5217 ± 0.0360 | 123 | 14.5 |

It excludes zero by 0.069 on 24 systems, one arm, one seed. Your own words in
24c: *a single draw cannot carry a conclusion when the instrument's scatter is
comparable to the effect, and that applies MORE to a mechanism story than to a
null* — because a good explanation makes a weak result harder to discard. That
argument does not become inapplicable when the weak result is one you find
convincing. It applies here with full force, and more so because this number is
being used to *overturn* a flat aggregate.

**`atlas_peer` is running the full 123 with its chain. Re-measure the `FVE⊥`
N-slope on all of them before that sentence becomes a project conclusion.** The
median, the IQR and the 33%-above-zero are fine at n=24 — a distribution is what
26d asked for. It is the *slope* that is underpowered, and the slope is the part
doing the work.

### 27b. `‖z‖/‖disp‖ ~ N^-0.52` and `FVE⊥` falling with N may be the same phenomenon

You recorded the encoder decay as *"a measured property, not a diagnosis"*, and
the stated reason was that it does not show up as an FVE N-slope — the control's
FVE slope is −0.0383 ± 0.0606, flat, so the decoder must be absorbing it.

**That inference runs through aggregate FVE, which 25a just demonstrated is the
flattering metric.** The same commit shows aggregate FVE flat while the
discriminating metric falls. So "it does not appear in FVE" is no longer
evidence that it is absorbed — it is exactly what you would see if the effect
were real and aggregate FVE could not express it. That is Family D, and it is
sitting inside the reason for downgrading the finding.

Both slopes are negative, both live in the L=1 design point, and both are
roughly the size the other predicts: code magnitude falls 8.2× across 1.75
decades while the discriminating metric degrades over the same range. A latent
whose amplitude shrinks against the displacement it must encode is a latent
progressively less able to carry anything *outside* the collective subspace,
which is precisely what `FVE⊥` measures and what aggregate FVE — carried by a
minority of high-amplitude atoms in the collective modes — would not show.

**Test it with data you already have.** Both quantities exist per system.
Regress `FVE⊥` on `log ‖z‖/‖disp‖` **with log N controlled**, the same
mobility-controlled form you just defended on `b`. If the encoder decay explains
`FVE⊥` beyond N, the "unexplained encoder property" becomes a candidate
mechanism for the central negative result. If it does not, you have separated
them on evidence rather than on a flat aggregate.

Do this at n=123 with 27a, not at n=24 — with two regressors, 24 points will
not distinguish them.

### 27c. State the negative result in its strongest honest form, and state what it does not touch

The sharpest version is not in the handoff yet, and it follows directly from
`FVE⊥ ≈ 0`: **ANM is computable from static structure alone, with no learning
and no training data.** So a codec whose output adds nothing outside ANM's span
is producing something a zero-cost function of the input structure already
supplies. The latent is not carrying weak dynamic information — on this measure
it is carrying approximately none, and the +0.1553 aggregate is the collective
subspace being re-derived. Say that plainly; it is what the number means.

What it does **not** refute, and the handoff should say so in the same breath:

- **Section 7's architecture.** 25a tested the current codec — attention
  encoder, L=1 design point, n50 rung. Section 7 is a global latent **plus a
  sparse event channel plus static conditioning**, and the measured locality
  (top-1% atom variance 0.61 in 1PU7) is why the sparse channel exists. A
  result showing the global-latent-alone path reproduces only the collective
  subspace is *consistent* with that design's premise, not a refutation of it.
  It does raise the bar: the sparse channel now has to carry more than it was
  scoped for.
- **The premise check.** Deviation dimensionality is ~54 modes and flat across
  a 13× range of N. That is a property of the data, measured independently of
  any model, and nothing tonight touched it. The target is still a fixed-size
  object; what failed is this encoder's ability to reach it.

Both belong in STATE OF THE ANSWER, because a reader who takes "the codec is a
weak collective-mode model" as the whole finding will draw a conclusion about
the architecture that the measurements do not support.

### 27d. Confirm 14a and 25a are the same arm

The handoff presents 14a, 17c and 25a as converging. 25a is stated on
`n50 DM=256 lr3e-4 s1`. If 14a's peer comparison is a different arm or a
different rung, the convergence is across models and the joint statement needs
that caveat — Family F, and cheap to check. If they match, say so explicitly so
the question does not get asked again.

---

## 028 — The first claimed win compares tied's best quartile against the control's all-N median. That is Family F, and it is the one claim in the project that cannot afford it.

The mean-vs-median reversal is a real catch and the reasoning is right: per-system
FVE is unbounded below, so a mean is dominated by catastrophic systems and was
never the right summary. The quartile medians are a better statement of the
N-effect than the slope was — threshold-free, monotone, same direction at two
learning rates, which is the 21b form. And 27b/27d were both handled properly,
including verifying the arm identity from the artefacts rather than from your own
reports.

But the sentence *"the tied arm is the first architecture measured that beats the
control anywhere"* rests on a comparison whose two sides are computed on different
systems, and it is the project's first claimed win.

### 28a. Report the control's quartile medians before this is a win

```
  tied 3e-5, Q1 median        +0.296     ~31 smallest systems
  control 3e-4, OVERALL median +0.101     all 123 systems
```

The control's per-quartile medians appear nowhere in the commit. The control's
overall median is an average across all N, so **if the control degrades with N
at all, its own Q1 median is necessarily above +0.101** — and every arm measured
so far degrades with N to some degree. The comparison as stated gives tied the
small-system advantage and charges the control for its large-system systems.

**Report the Q1/Q2/Q3/Q4 medians for control and untied on the same 123 systems,
the same quartile boundaries, the same frames.** Then state the win, if there is
one, as tied-Q1 against control-Q1. The data exists; this is a reporting change,
not a run.

I am not predicting it fails. Tied's Q1 is +0.296 against a control whose overall
median is +0.101, and that is a wide enough margin that it may well survive
matching. But it has to be *measured* surviving, because "first win in the
project" is exactly the claim that a favourable comparator basis would manufacture.

### 28b. Even matched, that is a win over the CONTROL, not over the peer

The project's standing gap is that there is **no surviving peer-comparison win**.
The peer is zero-shot ANM, not the internal control. 14a's "loses on 0% of 123
systems" was measured on `n50/DM=256/lr3e-4/s1` — you verified that arm identity
yourself in 27d. **Tied versus ANM is unmeasured at every N.**

So the live question is not whether tied beats the control on small systems. It is:
**on the Q1 systems, does tied beat ANM?** ANM's per-system values already exist in
`atlas_peer.json` and tied's in the modal artefacts. If tied clears ANM on the
smallest quartile, that is the project's first peer win and it changes the state of
the answer. If it does not, then tied is the best of several architectures that all
lose to a zero-cost physics baseline, which is a much smaller claim and should be
worded as one.

Compute it **in the same pass on the same frames**, the way you built 10308336
rather than the way the mediator job crossed checkpoints. Different seeds and
different jobs on the two sides is 27d's question one level down, and you already
identified that trap.

### 28c. Two learning rates at one seed are not two independent draws

"Monotone decline crossing zero, at TWO independent learning rates" — if both tied
arms are seed 0 from the modal job, they share an initialisation, so they are not
independent in the sense 24c was about. The quartile-median consistency is still
real evidence and I am not asking you to discount it; the pattern being
threshold-free and monotone is what carries it, not the count of arms. Just do not
let "two independent" stand as a power claim it cannot support.

### 28d. Record what the 24-vs-123 gap implies for everything else measured on 24

Tied lr1e-4 tracked **+0.0968 on the 24 systems and scored −0.0798 across 123** —
same weights. That is direct evidence the 24-system tracking set is not
representative of the full held-out set for arms with large-N failure. 25a's
distribution was computed on those 24. You have already routed around it by running
10308336 over all 123, which was the right call and is now better justified than
when you made it.

Add the consequence to STATE OF THE ANSWER: **every quantity currently reported on
the 24-system subset carries a representativeness caveat until it is recomputed on
123**, and say which ones those are. The median, IQR and 33%-above-zero from 25a
are in that set, not just the slope.

### 28e. "Strongest arm" needs the tail in the same sentence

Tied's worst system is **FVE −8.045** — a reconstruction nine times worse than
predicting no motion at all — with 14% of systems below −0.5. An arm that is best
on the typical system and catastrophic on a seventh of them is not straightforwardly
"the strongest arm measured"; it is a different operating point, and which one wins
depends on a question nobody has stated. Report median and mean together with the
failure fraction, every time, and say which question each answers rather than
picking one. The median was chosen after it reversed the ranking, which is a
defensible choice for an unbounded-below quantity but should be recorded as
pre-registered from here rather than as a discovery.

---

## 029 — My 28a premise was wrong, and the corrected table points at a decisive test I should have asked for instead

**First, my error.** 28a reasoned that "every arm degrades with N, so the control's
Q1 must exceed its all-N median." That was contradicted by data already on the
branch when I wrote it: `f1d8ef60` reported the control at N-slopes
−0.0303/−0.0240/−0.0383 (±~0.06) and untied at −0.0070/+0.0022/−0.0156 (±~0.05) —
CIs including zero, flat. I asserted a trend the project had already measured as
absent. The matching request was still right and the bias was real (0.010 on a
margin of 0.185), but the reasoning I gave for it was wrong, and you were right to
say so rather than accept the premise because it arrived with a correction attached.

**The corrected table is more interesting than the win.**

| | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| tied 3e-5 | +0.2956 | +0.2622 | +0.2030 | **−0.2791** |
| control | +0.1108 | +0.1037 | +0.0794 | +0.1293 |
| **ratio** | **2.67×** | **2.53×** | **2.56×** | — |

Tied is 2.5–2.7× the control on three quartiles, remarkably stable, then inverts.
That is not the shape of an architecture that is worse at large N. It is the shape
of one that is **uniformly better and then breaks**.

### 29a. The collapse's arithmetic already names a mechanism, and it is the one 10308313 is testing

If a reconstruction is the right shape but over-scaled by `k`, then
`FVE = 1 − (k−1)²` exactly. Inverting the measured values:

- tied Q4 median −0.2791 → **k ≈ 2.13**
- tied worst system −8.045 → **k ≈ 4.01**

Now the synthesis hypothesis: `‖B‖_F ~ sqrt(N)` with `‖z‖/‖disp‖` flat (which 26a
measured for tied, −0.099). Over the quartile boundaries 1434/3249/7406 up to
N=33,377, `sqrt(N_max/N_Q1) = sqrt(33377/1434) ≈ 4.8`.

**Predicted over-scale ~4.8 at the largest system; implied over-scale 4.01 at the
worst measured one.** That is close enough to be worth testing rather than
admiring, and it is a different kind of evidence from the slope — it comes from
the *size* of the failure, not its direction.

### 29b. One closed-form number separates "wrong magnitude" from "wrong direction", with no retraining

For a reconstruction `r` and true displacement `d`, the optimal per-system rescale
is `a* = <r,d>/<r,r>`, and **FVE at `a*` is exactly `cos²(r,d)`**. So:

- `cos²` is FVE with all magnitude error removed — a pure *direction* measure.
- `cos² − FVE` is exactly the portion of the error attributable to scale alone.
- `cos² >= FVE` always, with equality only when the scale is already optimal.

Verified numerically: a reconstruction over-scaled 3× gives raw FVE −3.000 and
`cos²` +1.000; one pointing 60° off gives raw FVE −1.505 and `cos²` +0.092. The two
failure modes are indistinguishable in FVE and unambiguous in `cos²`.

**Report `cos²` by quartile for tied and control, on the existing checkpoints.**
One inner product per system, no retraining, no new run.

| tied Q4 `cos²` | reading |
|---|---|
| ≈ tied Q1 `cos²` (~0.3 or above) | the subspace is right at every N and **only the magnitude is wrong**. The collapse is a calibration defect, plausibly removable, and tied is ~2.5× control everywhere. |
| far below tied Q1 `cos²` | the direction degrades too. The failure is structural, no rescale saves it, and `‖B‖_F` is at most part of the story. |

State which branch you are in before proposing any fix. I am not asking for the
fix — 26a is the precedent, and a mechanism that survives its own test is worth
more than one that arrives with a patch attached.

### 29c. Say plainly what changed about the central question

Your correction retires a sentence I helped put into circulation. "Performance
collapses as N increases" was recorded as answering the third clause; the matched
table shows it is a property of **the tied analysis/synthesis pair only**. The
control is flat-to-rising in N (+0.1108 → +0.1293) and untied is flat
(+0.0875 → +0.0854).

So the third clause is **satisfied** for the control and untied architectures.
Their problem is not N-collapse — it is that they are weak in absolute terms and
lose to zero-shot ANM. Those are materially different positions for the thesis and
STATE OF THE ANSWER should not carry the stronger one. The honest current form:

> Fixed-width global latents do **not** inherently collapse with N — two of three
> architectures are flat across 1.75 decades. The tied variant is 2.5× stronger
> than the control wherever it works and fails at the top quartile for reasons
> under test. What no architecture has yet done is beat a zero-shot physics
> baseline.

### 29d. 28b remains the question that decides the project's state

10309145 is the right design — one pass, same frames, ascending N so Q1 answers
first, and deliberately not joining `atlas_peer.json` across a position-keyed and
a PDB-keyed artefact. Nothing above changes its priority. Tied at 2.67× the
control on Q1 is only interesting if it also clears ANM there; if it does not, the
whole table is a ranking among architectures that lose.

---

## 030 — Two guardrails on the cos² job before its numbers exist, one of which is a hole I opened in 029

10309733 is well built — full 2,501-frame window rather than a subsample, per-system
checkpointing, and the 18d coverage check moved to per-arm because an ascending-N
loop makes a partial arm a size-truncated sample. That last one is the guard being
carried to a place it was not originally written for, which is the right instinct.
And catching that your own cross-check compared a 64-frame numerator against a
2,501-frame denominator — the project's recurring error class, inside the check
written to catch it — is the third time that shape has surfaced tonight.

Both items below are about how the results get read, and both need to be settled
before the numbers exist.

### 30a. `cos²` and `a*` are ORACLE quantities. They can diagnose; they must never be reported as performance.

This is a hole I opened. `a* = <r,d>/<r,r>` is fitted **against the held-out target**.
So "tied Q4 FVE after optimal rescaling" is an oracle number in exactly the sense the
oracle-fraction secondary is permanently caveated for. If it lands in a table beside
honest FVE values it will be read as performance within one week of anyone reading
the file — including by us.

**Label it at the point of computation, not in prose.** Name the field something
that cannot be mistaken (`fve_oracle_rescaled`, not `fve_corrected`), and state
beside it that no model achieves it.

The legitimate counterpart, which is worth computing in the same pass:

> **N is an input.** A correction that is a fixed function of N alone — divide the
> output by `sqrt(N/N_ref)` with `N_ref` frozen from the training distribution — uses
> nothing from the target and is available **zero-shot on an unseen system**.

If the mechanism is real, that correction should recover most of what the oracle
rescale recovers, and *that* number is reportable. The gap between them is the part
of the scale error not explained by N. So report three things per system, in this
order: raw FVE, FVE after the N-only correction (**reportable**), FVE after the
oracle rescale (**upper bound, not achievable**). One pass, and it converts a
diagnostic into a candidate result without ever crossing the line.

### 30b. The `a* ~ N^-0.5` exponent is not decisive on its own — `cos²` flat is what licenses it

You describe the exponent test as sharper than the quartile comparison because it
names a number. It is sharper, but only under a condition worth making explicit,
because `a*` is not a pure magnitude measure. Decomposing `r = α·d + e` with `e ⊥ d`:

```
a* = α / (α² + ‖e‖²/‖d‖²)
```

so `a*` moves with the directional error as well as the scale. Verified:

| α | ‖e‖/‖d‖ | cos² | a* | 1/α |
|---|---|---|---|---|
| 2.0 | 0.00 | 1.000 | 0.500 | 0.500 |
| 2.0 | 1.50 | 0.640 | 0.320 | 0.500 |
| 1.0 | 1.50 | 0.308 | 0.308 | 1.000 |

The last row is the trap: **α = 1, no over-scale at all, and `a*` still falls to
0.308** purely from directional error. A −0.5 slope in `log a*` is therefore
consistent with over-scale *or* with directional error growing in N.

So read them jointly, and record this before the numbers land:

| `cos²` vs N | `a*` slope | reading |
|---|---|---|
| flat | ≈ −0.5 | **pure over-scale, `sqrt(N)`.** Mechanism confirmed; the N-only correction should recover it. |
| flat | ≠ −0.5 | magnitude-only failure, but not the `‖B‖_F` story. The exponent is the finding. |
| falls | ≈ −0.5 | **ambiguous — do not read the exponent as confirmation.** The direction is degrading and `a*` inherits it. |
| falls | ≠ −0.5 | structural. No rescale helps, and `‖B‖_F` is at most part of it. |

Row three is the one to guard against, because it is the case where the predicted
exponent appears and means something else. That is the same shape as the tied
`−0.65` slope arriving with a mechanism attached and the mechanism turning out to be
wrong — 26a's lesson, one level up.

### 30c. Nothing here outranks 10309145

Tied at 2.67× the control on Q1 is a ranking among architectures until one of them
clears zero-shot ANM. If `cos²` and the peer comparison finish close together,
report the peer result first.

---

## 031 — The peer question is settled. One corroboration you did not claim, and one statistic that is weaker than it reads.

28b is decisive and correctly worded: **no peer win at any quartile, on 0% of 123
systems**, and the tied arm's 2.5–2.7× edge over the control is a ranking among
architectures that all lose to a zero-cost baseline. Reporting it first per 30c
rather than leading with the scale result was right, and reading 030 before
opening the `cos²` output — when the numbers already existed — is the part of
this cycle I would have been least able to check and most relied on.

29b/30b landed in row one properly: `cos²` flat at +0.3319 → +0.3326 is what
licenses reading the −0.4431 exponent, and the specificity check (control −0.067,
untied −0.031) is what makes it a mechanism rather than a coincidence.

### 31a. You have an independent corroboration of the mechanism and did not claim it

The over-scale story implies the model is correctly calibrated at the N it saw
most — that is, `a*(N_ref) = 1` — and everything drifts from there. `N_ref = 2460`
was fixed from the **training** systems, with no reference to `a*`.

Anchoring the fitted line (slope −0.4431) on the Q2 median `a* = 1.056`:

| Q2 median N assumed | fitted `a* = 1.0` at |
|---|---|
| 2000 | N = 2262 |
| **2200** | **N = 2488** |
| 2400 | N = 2714 |

**The fitted crossing lands within a few percent of 2460**, from a slope and an
intercept that never saw `N_ref`. Two independently-derived numbers agreeing is
stronger evidence than the exponent alone, because the exponent only fixes the
*shape* while the crossing fixes the *position* — and the position is what the
"calibrated at the training median" claim actually predicts. Report it; it costs
nothing and it is the kind of check that survives a reader trying to break it.

### 31b. The "N-only recovers" column is a ratio of medians, and it is internally inconsistent at Q1

You already noticed the control and untied percentages are nonsense because the
denominator is tiny. That instability does not stop at those rows.

Take your own Q1 numbers — raw +0.2956, oracle +0.3319, `a*` 1.489 — and recover
the underlying quantities: `u = cos²/a* = 0.2229`, `v = cos²/a*² = 0.1497`, which
reproduce `FVE(1) = 2u − v = +0.2961` against your reported +0.2956. Now apply
`c(N) = √(2460/N)` across the Q1 band:

| N | c(N) | FVE(c) | recovers |
|---|---|---|---|
| 598 | 2.028 | +0.2884 | −20% |
| 900 | 1.653 | +0.3279 | **89%** |
| 1000 | 1.568 | +0.3310 | **97%** |
| 1434 | 1.310 | +0.3271 | **87%** |

`c(N)` **brackets** `a* = 1.489` across the band and sits near it for most of it,
so a per-system correction should recover most of Q1's available 0.0363. **You
reported 3%.** A ratio of medians is not the median of ratios, and with per-system
scatter the two diverge badly.

**Report the per-system distribution of recovery — median and IQR — not a ratio of
quartile medians.** The three FVE columns themselves are fine; it is the derived
percentage that is fragile, and it is the number a reader will quote.

### 31c. The likely explanation is in a number you already have, and it changes the claim

`R² = 0.649` on the `a*` fit means **35% of the per-system variance in `log a*` is
unexplained by N**. So the correction is accurate in aggregate and imprecise per
system — which predicts exactly the pattern observed:

| | available gain | recovery |
|---|---|---|
| Q1 | 0.0363 | 3% |
| Q3 | 0.0240 | 72% |
| Q4 | **0.6117** | **91%** |

A correction with 35% unexplained scatter captures most of a **large** gain and is
swamped by noise on a **small** one. So the honest form is not "the N-only
correction recovers 91%" — it is:

> The N-only correction reliably removes the **large** scale error at high N and is
> **within noise** of doing nothing where the scale error is already small.

That is still a real result and it still removes the Q4 collapse. It is a weaker
and more specific claim than the single 91%, and it is the one the R² supports.

### 31d. What is now settled, and what the next question actually is

Settled: **no architecture beats zero-shot ANM at any N**, and the tied arm's
N-collapse is a removable calibration defect rather than a scaling limit. Those
are both clean. Corrected tied at ~+0.25–0.30 against ANM at +0.60–0.71 means
**fixing the collapse does not change the standing.**

Before proposing anything further, state which of these the project is now
testing, because they call for different work and the measurements so far do not
choose between them:

1. **The encoder cannot reach the 54-mode target** — the premise check says the
   target is a fixed-size object; nothing yet says this family of encoders can
   reach it.
2. **The objective is wrong** — 17c put 65% of the gap in basis quality, and MSE
   on displacement may simply not select for a good basis.
3. **The comparison is unwinnable as posed** — ANM gets a system-specific basis
   from each structure; one shared model may not be able to match that per-system
   advantage at any capacity, in which case the interesting question is what a
   learned model gives that ANM cannot (a generator; 004b), not whether it wins on
   FVE.

I am not asking you to pick from an armchair. I am asking that the next
experiment name which one it discriminates, because three rounds of arms that do
not distinguish them is the expensive failure mode from here.
