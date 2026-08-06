# Phase 1 A/B/C read (n=276, L in {1,12,24} at DM=64, MISATO) -- OUTCOME C, neither branch excluded

## Arm quality ordering -- do NOT average these
1. **L=12 is the HEADLINE ARM**: n=69 held-out, well-powered, uncontaminated.
2. **L=24 is COMPROMISED**: Family A hit -- 23 void rows with medN **8,293 void vs 2,325 kept**, so
   G8 voided preferentially at HIGH N and its slope sits on a truncated N-axis.
3. **L=1 is VALID BUT HOPELESS for discrimination**: guards all pass (G1 0.841, G4 +0.072->-0.128,
   G6 identical) but the slope CI half-width is **+/-1.305**. This is exactly the pre-registered
   weak-arm case; it is reported as "valid but underpowered for A/B discrimination", NOT resolved by
   borrowing the L=12/24 answer (which would be Family C on the headline question).

## The script's printed verdict was WRONG -- the SPAN rule is itself a Family B defect
`armf_phase1_misato.py` printed **INDEX-ADDRESSING at all three L**. That is an artifact.
`SPAN = max/min` explodes when the smallest bucket approaches zero -- L=1's 46.92x comes from a
single bucket at 0.01. The rule is pinned to whatever the minimum bucket happens to be rather than to
the effect. **Replace the span rule with the slope test before it reports again.** Worse, the
absolute-gap arm of the same rule "confirmed" it via max-min > 0.03 while the actual gap SLOPES are
all NEGATIVE (gap shrinking with N, the opposite of the failure mode).

## Per-system slopes (the correct statistic)
| L | n | ratio slope | p | gap slope (CI) | N effect? |
|---|---|---|---|---|---|
| 1 | 69 | +0.479 +/- 1.305 | 0.467 | -0.059 [-0.120,+0.001] p=0.053 | NO |
| **12** | **69** | **+0.049 +/- 0.111** | **0.385** | -0.023 [-0.066,+0.021] p=0.301 | **NO** |
| 24 | 46 | +0.055 +/- 0.087 | 0.206 | -0.027 [-0.075,+0.020] p=0.250 | NO |
No ratio slope excludes zero. **No gap slope excludes zero either** -- L=1 misses by +0.0008 (p=0.053)
but L=1 is the uninformative arm, so this is not positive evidence.

## THE THRESHOLD MAPPING CHANGED -- stated, not applied silently in either direction
The earlier "2.0x <-> slope +0.175" was derived with **baseline ratio 0.27 and range 1.54 decades**
(the n=25 run). This run's L=12 baseline is **0.144, roughly half**. Since a 2x span requires
`b*rng = min_fitted`, a LOWER baseline makes the threshold STRICTER:

| L | median ratio | min(fit) | thr(2x) | thr(1.3x) | slope CI upper | verdict |
|---|---|---|---|---|---|---|
| 12 | 0.144 | 0.090 | **+0.0569** | +0.0171 | +0.1603 | 2x NOT excluded |
| 24 | 0.212 | 0.146 | +0.0925 | +0.0277 | +0.1416 | 2x NOT excluded |

Against the STALE +0.175 the L=12 CI upper (0.160) would look excluded; against this run's own
baseline the threshold is +0.057 and the CI upper is ~3x above it. **Reporting exclusion on the old
number would be a stale-threshold UPGRADE -- the mirror of a silent downgrade.**
Assumption-free form -- relative change across the measured range (2x span ~ +100%):
- **L=12: +53%, CI [-68%, +175%]** -> spans it.
- L=24: +41%, CI [-23%, +105%] -> marginally tightest, but this is the contaminated arm.
- L=1: [-434%, +937%] -> uninformative.

**The pre-registered asymmetry did NOT materialise.** It predicted half-width ~0.075 and exclusion of
2x. Actual L=12 half-width is **0.111**, and the baseline halved. Both moved against discrimination,
so this run excludes NEITHER branch. That prediction is scored as wrong.

## Density-matched pairs -- three CORRELATED marginal results, not independent evidence
Pair A (avail 410 vs 416) holds in the same direction at every L: L=12 0.045 vs 0.248 (p=0.068);
L=24 0.159 vs 0.226 (p=0.097); L=1 0.295 vs 0.349 (p=0.644). **These three tests share systems, so
they must NOT be combined as though independent.** Pair B goes the other way (L=12 0.095 vs 0.127,
p=0.738) and REVERSES at L=1 (0.401 vs 0.012). Suggestive, unresolved.

## Objective 4: steps-to-plateau vs N, inconsistent across L
L=1 +0.074 +/- 0.313 (flat); **L=12 +0.179 +/- 0.145, R^2 0.745 (GROWS)**; L=24 +0.000 +/- 0.077
(flat). Inconsistent, so no claim that training cost scales with N.

## VERDICT: OUTCOME C -- no N effect detectable at this resolution, NEITHER branch excluded
Per the pre-committed rule, the response is **NOT** to buy 565 systems: three inconclusive reads on
the same statistic indicts the STATISTIC. Two defects must be fixed first --
**(1) the span verdict rule** (replace with the slope test) and **(2) the L=24 G8 exclusion**
(void-at-high-N truncates the axis) -- then rerun with **realised effective rank and subspace overlap
as the PRIMARY instruments** (both now wired in, plus checkpoint saving).
