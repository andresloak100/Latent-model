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
