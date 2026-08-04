# Pre-registration: mdCATH matched-window test + nonlinear probe (before the data)

Committed before the mdCATH subset is pulled, so the interpretation cannot be
fitted to the numbers. Arm F on MISATO showed all-atom displacement is not
linearly compressible into 8-32 modes at chemical accuracy over 80 ps-8 ns; this
tests whether that was the wrong dynamical regime.

## mdCATH matched-window test (PRIMARY decision point)

Within ONE mdCATH trajectory (same domain, temperature, replica; 320 K
equilibrium), subsample two windows with matched frame count / rank ceiling / PCA
conditions -- only the time window changes:
- **A**: 50 frames over ~50 ns (~1 ns stride)
- **B**: 50 frames over ~464 ns (~9 ns stride)

Leak-free tier table (fit first half, eval second half) for **CA / backbone /
all-atom / side-chain** at **L = 8/16/32**, reported as **% variance captured AND
absolute residual A**. Subset: 20-30 domains, broad size range, ~20-25 GB on
$SCRATCH.

**Locked interpretation:**
- **B substantially better than A** -> slow collective dynamics are more
  compressible; MISATO tested the wrong regime and the codec question REOPENS on
  the relevant (slow) signal. Then rerun the tier + nonlinear tests on the slow
  window.
- **B approximately equal to A** -> the negative generalises across timescale and
  becomes a much stronger verdict (all-atom displacement is not chemically
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
The nonlinear question is deferred to mdCATH's larger frame count.

## High-temperature (450 K) trajectories

Analysed SEPARATELY for the transition/unfolding question; NOT mixed into the
equilibrium tier table (elevated-temperature kinetics are a different regime).

## Ordering

The mdCATH matched-window result is the primary decision point. Report the
matched-window result first. If long ~= short, the codec line gets a stronger
negative; if long is substantially more compressible, rerun the relevant tier and
nonlinear tests on the slower signal.

_Pre-registered 2026-08-04, before the mdCATH pull._
