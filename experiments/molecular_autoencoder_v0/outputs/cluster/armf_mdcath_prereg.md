# Pre-registration: mdCATH matched-window test + nonlinear probe (before the data)

Committed before the mdCATH subset is pulled, so the interpretation cannot be
fitted to the numbers. Arm F on MISATO showed all-atom displacement is not
linearly compressible into 8-32 modes at chemical accuracy over 80 ps-8 ns; this
tests whether that was the wrong dynamical regime.

## mdCATH matched-window test (PRIMARY decision point)

Within ONE mdCATH trajectory -- **same domain, same temperature (320 K), same
replica** (replica/temperature variation would confound the window comparison) --
subsample two windows with matched frame count / rank ceiling / PCA conditions;
only the time window changes:
- **A**: 50 frames over ~50 ns (~1 ns stride)
- **B**: 50 frames over ~464 ns (~9 ns stride)

Leak-free tier table (fit first half, eval second half) for **CA / backbone /
all-atom / side-chain** at **L = 8/16/32**. Subset: 20-30 domains, broad size
range, ~20-25 GB on $SCRATCH.

**Window A at MULTIPLE offsets** (0-50, 200-250, 400-450 ns) and report the
spread: the first 50 ns may contain relaxation away from the starting structure,
inflating A's deviation and its apparent compressibility; multiple offsets guard
against that and against cherry-picking. If offsets disagree materially, that is
itself a reported result.

**DECISION IS ON ABSOLUTE ANGSTROM, NOT PERCENTAGE.** B spans 464 ns so its null
is much larger than A's; B can post a HIGHER % reduction while leaving a WORSE
absolute residual (60% off a 4A null = 1.6A > A's 1.15A). The "chemically invalid"
argument is absolute, so the read must be too:

- **B beats A iff B's ABSOLUTE residual (A) is lower at matched L.** Report both %
  and A; the decision is on A.
- **Report both nulls explicitly.** If B's null is several times A's, flag it
  prominently -- that alone reframes what "compressible" means here.

**Locked interpretation (on absolute A):**
- **B's absolute residual substantially below A's** -> slow collective dynamics
  are more compressible; MISATO tested the wrong regime and the codec question
  REOPENS on the relevant (slow) signal. Then rerun the tier + nonlinear tests on
  the slow window.
- **B's absolute residual ~= or above A's** -> the negative generalises across
  timescale, a much stronger verdict (all-atom displacement is not chemically
  compressible into 8-32 modes at any accessible timescale).

## Nonlinear probe (per-system, MISATO) -- ALREADY RUN, asymmetric interpretation

Per system, no cross-system sharing, no static-conditioning shortcut, matched
temporal split and L, nonlinear AE vs PCA.
- **Nonlinear clearly beats PCA** -> the linear ceiling was misleading; the codec
  remains viable even on thermal fluctuations.
- **Nonlinear approximately matches PCA (or worse)** -> INCONCLUSIVE until the
  mdCATH window test resolves the regime question. Does NOT count as a second
  independent negative.

RESULT: nonlinear UNDERPERFORMED PCA (all-atom L32: 34% vs 42%), attributable to
50-frame overfitting, not a nonlinear ceiling -> **inconclusive**, not a negative.
AE 34% vs PCA 42% on 50 frames means 50 frames is too few to FIT a nonlinear model
at all, so the nonlinearity question is gated on FRAME COUNT, not just regime.
Rerun the probe on mdCATH AFTER the window control lands (same per-system
constraint, same temporal split, same matched bottleneck, now with enough frames
that a negative would mean something). **mdCATH therefore resolves BOTH open
caveats -- window regime AND linear-vs-nonlinear -- which is the strongest
argument for the ingest.**

## Domain filter (at pull time)

Trajectory length is variable (mean 464 ns, std 76), so a domain whose trajectory
is e.g. 300 ns cannot supply window B. **Read numFrames per file (from the index)
and keep only domains supporting BOTH windows**, so B is not ragged across systems
(same common long span for all kept domains). **Report how many of the 25-30
survive the filter**; if it is a large cut, say so rather than silently shrinking
the cohort.

**Filter-bias check (adaptive-length simulation could correlate length with
dynamics).** From the index: retained (numFrames>=451, n=3293) vs excluded
(n=2105) are **near-indistinguishable** -- heavy 1067 vs 1113, residues 135 vs
141, gyration range 0.21 vs 0.22, coil 44% vs 45%, alpha 33.6% vs 33.4%. Excluded
(short) are MARGINALLY more flexible (~5% gyration range) -- the direction that
would bias toward B~=A -- but the magnitude is negligible. Noted, proceed; not a
stability-selected cohort.

## Window C (added -- breaks span/stride coupling; leak-free L=64)

A (50 fr / 50 ns / 1 ns stride) and B (50 fr / 500 ns / ~10 ns stride) differ in
BOTH span and stride; a 10 ns stride aliases fast motion. Window **C = all 500
frames, 1 ns stride, 250/250 split, L=8/16/32/64** breaks the coupling: A->B moves
span+stride at fixed rank; B->C moves stride+rank at fixed span; C is best-case.
C also gives the **leak-free L=64 test** MISATO couldn't (250-frame fit -> up to
249 modes), checking the prediction that L=64 lands at **43-44% all-atom** in the
thermal regime -- if it comes in much higher, the saturation argument was wrong.
**C is reported SEPARATELY, not as the headline**, since it is not rank-matched to
A/B. (A/B are 50 fr -> 25 fit -> rank<=24, so their L=32 is rank-capped at 24,
matched across A and B; L=8/16 clean.)

## High-temperature (450 K) trajectories

Analysed SEPARATELY for the transition/unfolding question; NOT mixed into the
equilibrium tier table (elevated-temperature kinetics are a different regime).

## Ordering

The mdCATH matched-window result is the primary decision point. Report the
matched-window result first. If long ~= short, the codec line gets a stronger
negative; if long is substantially more compressible, rerun the relevant tier and
nonlinear tests on the slower signal.

_Pre-registered 2026-08-04, before the mdCATH pull._
