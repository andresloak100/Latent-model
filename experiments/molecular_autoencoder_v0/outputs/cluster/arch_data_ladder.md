# Architecture x data ladder: does DATA close the Perceiver's memorization gap?

The depth grid showed the Perceiver's failure is memorization (fits train ~0.7,
val ~7.5; ~10x gap) not decoder capacity. This tests the remaining lever -- DATA --
by running both architectures over a data ladder and comparing the **val-vs-n
slope**, not the endpoint.

## Design
- Both architectures at their best-known config, both un-handicapped: 242k steps,
  lr 3e-4, processed_small, **identical 758-structure val across all rungs**.
- Data ladder: n_train = 450 / 878 / 2272.
- **Size bracket:** direct at 1.1M AND 3.8M brackets the Perceiver's 3.2M
  (1.1M < 3.2M < 3.8M), so the size contribution to the slope is measurable and
  can be subtracted. **The 3.2M-vs-1.1M gap is deliberate (best-config, NOT a
  like-for-like size comparison) -- read the slopes, not the raw offsets.**

## Results (held-out, 242k steps)
| arch | n450 aa | n878 aa | n2272 aa | **val slope** | n2272 bb | n2272 chir | n2272 F1 |
|---|---|---|---|---|---|---|---|
| direct 1.1M | 1.171 | 1.089 | 0.878 | **-0.127** | 0.610 | 0.0005 | 0.953 |
| direct 3.8M | 1.067 | 0.882 | 0.792 | **-0.114** | 0.510 | 0.0002 | 0.963 |
| perc 3.2M | 8.320 | 7.618 | 6.770 | **-0.661** | 6.282 | 0.0353 | 0.499 |

slope = A per doubling of n, least-squares on log2(n).

### val/train gap per rung (does the gap close with data?)
| arch | n450 | n878 | n2272 |
|---|---|---|---|
| direct 1.1M | 4.27 | 2.87 | 1.50 |
| direct 3.8M | 9.10 | 4.40 | 2.79 |
| perc 3.2M | 18.20 | 6.70 | 4.16 |

## Findings

**1. The size effect on the slope is ~zero.** direct 3.8M slope -0.114 vs direct
1.1M -0.127 -> difference **+0.013 A/doubling** (the bigger model is if anything
marginally FLATTER, not steeper). The "bigger model sheds more overfit as data
grows -> steeper slope" hypothesis does NOT hold here. So any steeper Perceiver
slope cannot be attributed to its larger size.

**2. The Perceiver slope IS steeper -- and it's architectural.** perc -0.661 vs
direct ~-0.12: **~5x steeper**, and since the size contribution is only +0.013,
the architecture slope beyond size is **-0.534 A/doubling**. By the pre-registered
criterion (steeper perc slope, exceeding the size effect) **the scaling bet is
LIVE in direction**: the Perceiver improves with data much faster than the direct
codec does, and that is a real architecture effect, not a size artifact.

**3. But the crossover is impractical.** The Perceiver starts ~7x worse (6.77 vs
0.878 at n2272). Extrapolating its own slope, it reaches the direct codec's
CURRENT n2272 quality (0.878) only at **n = 2^20 ~ 1.1 MILLION structures (~480x
n2272)** -- and the direct codec keeps improving with data too, so the real target
moves further. Live in direction, but not reachable at any realistic data scale
for this dataset.

**4. The memorization gap IS partly data-starvation.** The Perceiver's val/train
gap closes steeply with data (18.2 -> 6.70 -> 4.16), the fastest relative closing
of the three -- consistent with the steep slope. So the gap is not purely
architectural; more data genuinely helps. But at n2272 it is still 4.16x (vs the
direct codec's 1.50x), nowhere near closed.

**5. The direct codec keeps improving with data.** direct 1.1M 1.17 -> 0.88,
direct 3.8M 1.07 -> 0.79 (bb 0.51, chir 0.0002, F1 0.963 at n2272) -- the working
codec is data-limited too and has not plateaued; the target the Perceiver is
chasing is still descending.

## Bottom line
Data is a real and architecturally-genuine lever for the Perceiver -- its val
slope (-0.661) is ~5x the direct codec's and the size effect is ~0, so the
steepness is not a size artifact and the scaling bet is live in DIRECTION. **But
it is not live in PRACTICE:** from a 7x deficit, closing to today's direct-codec
quality needs ~1e6 structures, and the direct codec is itself still improving with
data. The direct per-residue codec remains the design of record (0.79 A aa /
0.51 A bb / chir 0.0002 at n2272, 3.8M); the fixed-size Perceiver's size-independent
compression is not worth ~7x reconstruction error that only data at millions-of-
structures scale could close. (Masking was neutral for both architectures; see
perceiver_depth_grid.md.)
