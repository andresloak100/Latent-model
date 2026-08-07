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
