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

**Read the RELATIVE column, not the absolute slope.** RMSD is bounded below by 0,
and these arms sit ~7x apart in level, so A/doubling is not comparable between
them: -0.661 A off 8.32 is a SMALLER fractional move than -0.127 off 1.171.

| arch | absolute A/doubling | **relative %/doubling** |
|---|---|---|
| direct 1.1M | -0.127 | **-11.6%** |
| direct 3.8M | -0.114 | **-12.0%** |
| perc 3.2M | -0.661 | **-8.5%** |

In the comparable frame the Perceiver descends **more slowly** than either direct
arm, not 5x faster.

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

**2. The Perceiver slope is NOT steeper once levels are accounted for -- the
scaling bet is DEAD in direction.** The apparent "5x steeper" is entirely an
artifact of reading A/doubling off a 7x higher starting value. Relative rates:
perc **-8.5%/doubling** vs direct **-11.6%** (1.1M) and **-12.0%** (3.8M). The
Perceiver improves with data more slowly than the codec it is chasing.

The decisive check is the ratio, since "overtake" means the ratio reaches 1:

| n | perc / direct-1.1M | perc / direct-3.8M |
|---|---|---|
| 450 | 7.105 | 7.798 |
| 878 | 6.995 | 8.637 |
| 2272 | **7.711** | **8.548** |

The gap **widens** with data -- +3.6%/doubling against direct-1.1M and
+4.0%/doubling against direct-3.8M. More data does not move the Perceiver toward
the codec; it moves it further away. The pre-registered criterion is not met.

**3. There is no crossover to quantify.** The earlier "~1.1 million structures"
figure came from extrapolating the ABSOLUTE slope (-0.661 A/doubling) as a
straight line down to 0.878 A. That is invalid for a positive-definite quantity:
sustaining -0.661 A/doubling from 6.77 A would mean a **-44%/doubling** relative
rate near 1.5 A, 5x the Perceiver's measured -8.5%. Under its actual measured
rate the two curves **diverge**, so they never cross. The honest statement is not
"live but impractical" -- it is that the ladder shows no convergence in direction
at all.

**4. The val/train closure is dominated by TRAIN degradation, not by
generalization.** The rungs are matched on STEPS (242k), so per-structure exposure
falls 4.9x from n450 (8345 epochs) to n2272 (1704). Train fit therefore degrades
mechanically with n, which shrinks the ratio whether or not val improves.
Decomposing (implied train = val / gap):

| arch | val change | train change | train 450 -> 2272 |
|---|---|---|---|
| direct 1.1M | **-25.0%** | +113.4% | 0.274 -> 0.585 |
| direct 3.8M | **-25.8%** | +142.1% | 0.117 -> 0.284 |
| perc 3.2M | **-18.6%** | +256.0% | 0.457 -> 1.627 |

The Perceiver's gap "closes fastest" because its train fit collapses fastest --
its val improved the LEAST of the three. So this does not support "the gap is
partly data-starvation"; it mostly says the Perceiver degrades hardest when given
fewer passes per structure. Separating the two would need matched EPOCHS, which
the 2-D study already named as the fix (data_scaling_2d.md, caveat 3). Not worth
launching on this evidence.

**5. The direct codec keeps improving with data.** direct 1.1M 1.17 -> 0.88,
direct 3.8M 1.07 -> 0.79 (bb 0.51, chir 0.0002, F1 0.963 at n2272) -- the working
codec is data-limited too and has not plateaued; the target the Perceiver is
chasing is still descending.

## Bottom line
**Data is not the Perceiver's missing lever.** Measured in the only frame where
arms 7x apart in level are comparable, it improves at -8.5%/doubling against the
direct codec's -11.6% / -12.0%, and the perc/direct ratio WIDENS with data
(7.11 -> 7.71 vs 1.1M; 7.80 -> 8.55 vs 3.8M). There is no crossover at any data
scale on the measured trend, so the scaling bet fails its own pre-registered
criterion -- dead in direction, not merely impractical. The size bracket still
does its job (slope difference +0.013 A/doubling, ~0), so this is an architecture
result, not a size artifact.

The direct per-residue codec remains the design of record (0.79 A aa / 0.51 A bb /
chir 0.0002 / F1 0.963 at n2272, 3.8M) and is itself still data-limited and
descending. The fixed-size Perceiver's size-independent compression is not worth a
~7x reconstruction deficit that data is not closing. (Masking was neutral for both
architectures; see perceiver_depth_grid.md.)

**Caveat on what this does and does not rule out.** The ladder spans 450 -> 2272
(2.3 doublings) under a matched-STEP budget, so every rung is somewhat
under-converged at high n and the train-side numbers are confounded with epoch
count. It establishes that the Perceiver is not closing on this range under this
protocol. It does not establish that no training protocol could -- a matched-EPOCH
sweep, or a fundamentally larger corpus, remains untested.
