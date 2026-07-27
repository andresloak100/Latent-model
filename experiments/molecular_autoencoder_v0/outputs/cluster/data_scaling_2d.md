# 2-D scaling study: data x model capacity (direct decoder, working codec)

## CRITICAL COMPARABILITY CAVEATS (read first)
1. These cells score against **processed_small val = 758**; the capacity ladder and
   all earlier direct-decoder numbers scored against **processed val = 293** -- a
   DIFFERENT eval set. Do NOT cross-compare held-out absolutes across the two.
2. **Do NOT compare TRAIN RMSD across the two studies either** (see "Unexplained"
   below): 15.2M @ n=878 gives train ~0.300 in the capacity ladder but ~0.5-0.6
   here (same model/n/lr, different dataset + more steps). Unresolved.
3. Grid is matched at **60k optimizer-steps**, so epochs/rung = 2069/1091/480/423.
   Train RMSD is confounded with epoch count across rungs -- do not read the
   train-vs-n trend as capacity.

## Grid (held-out all-atom RMSD, A; matched 60k steps, lr 3e-4)

| n_train | epochs | 1.1M train | 1.1M val | v/t | 15.2M train | 15.2M val | v/t |
|---|---|---|---|---|---|---|---|
| 450  | 2069 | 0.576 | 1.185 | 2.06 | 0.252 | 0.934 | 3.71 |
| 878  | 1091 | 0.786 | 0.993 | 1.26 | 0.598* | 0.930 | 1.56 |
| 2000 | 480  | 0.800 | 0.858 | 1.07 | 0.638 | 0.837 | 1.31 |
| 2272 | 423  | 0.884 | 0.916 | 1.04 | 0.684 | 0.844 | 1.23 |

*train RMSD is augment-noisy (15.2M@878 bounces 0.5-0.87, min/final 0.498);
mean-of-last-3 overstates it. Top rung 2272 = matched-band ceiling (3853 ids).

NOTE (undertraining applies to the VAL numbers too): matched 60k steps gave the
high-n rungs only 423-480 epochs vs 2069 at n=450. The resume control (below)
shows 15.2M train falling 0.68 -> 0.45 with 2-4x more steps at n=2272, so the
**n=2000/2272 val numbers here are pessimistic** -- they are under-converged, not
converged data points. Q1's direction survives this (all rungs are undertrained
in the same direction, and the low-n rungs got MORE epochs, if anything biasing
against the observed descent); the slope magnitude does not (see Q2).

## Q1: does held-out descend with data? YES (SOLID)
1.1M 1.185->0.993->0.858; 15.2M 0.934->0.930->0.837 (2272 within noise). And
val/train ratio closes toward ~1.0 as data grows (1.1M 2.06->1.04; 15.2M
3.71->1.23): more data closes the generalization gap. **REPORT.md's "more data
won't help" is void** (it was the broken 9.5 A attention decoder). This is the
headline and it stands.

## Q2: data slope (fit log2(n) over measured range 450->2000 = 2.15 doublings; 2272 excluded, 1.14x near-duplicate of 2000)
1.1M -0.150 A/doubling; 15.2M -0.046 A/doubling (15.2M is better at low data and
flatter -> LESS data-hungry, not more, on val).

**These slopes are LOWER BOUNDS on the converged data slope, not the true slope.**
The high-n rungs are under-converged (Q1 note above), which flattens the measured
val-vs-n line -- a converged high-n point would sit lower, making the descent
steeper. Consequently the naive 0.5 A projections (1.1M ~9.8k structs ~4x beyond
2272; 15.2M ~370k ~160x beyond) are **OVERESTIMATES of the data actually needed**:
with the true (steeper) slope, 0.5 A is reached at fewer structures than these
numbers imply. They remain projections, not measurements.

To measure the TRUE data slope you would need a **converged data sweep** -- matched
*epochs* across rungs, or matched steps at a much higher budget so every rung
converges -- not the matched-60k-steps grid used here.

## Q3: is there a capacity floor? NO -- both models compute-limited (RESOLVED by resume control below)
The prior version claimed a capacity floor ("neither reaches 0.5 A at any data")
from train RMSD rising with n. That is NOT supported: (a) the rise is confounded
-- high-n rungs got 1/5 the epochs (matched steps); (b) train RMSD is
augment-noisy; (c) the capacity ladder shows 15.2M reaching train 0.300 at n=878,
so 0.68 is not its floor. **Retracted: the capacity-floor reading and the
"0.5 A needs joint capacity+data scaling" bottom line that rested on it.**

Settled by a compute control (resume the n=2272 checkpoints -- no fresh training):
continue both models to 120k and 240k steps.

### Resume control -- FINAL (n=2272 checkpoints resumed 60k -> 120k -> 240k steps)
Windowed train RMSD (median of ~4 quick-RMSD points/window; median-robust to the
augment/4-batch noise that swings single points 0.3-0.9):

| model | @60k | @120k | @240k | 120k->240k | verdict (plateau test, bar 0.05) |
|---|---|---|---|---|---|
| 1.1M  | 0.83 | 0.69 | 0.51 | **-0.185** | COMPUTE-LIMITED, still descending |
| 15.2M | 0.68 | 0.46 | 0.39 | **-0.073** | COMPUTE-LIMITED, still descending |

**Neither model is at a capacity floor at 240k.** Both train curves are still
descending; both clear the plateau bar by a wide margin.
* **15.2M: the 0.68 "floor" was undertraining -- confirmed.** Train 0.68 -> 0.39
  and still going (decelerating but -0.073 over the last doubling, > 0.05).
* **1.1M: also NOT floored.** Train 0.69 -> 0.51 (-0.185), the steepest segment of
  the whole run; it would cross 0.5 and keep going. It is compute-limited, not
  capacity-limited. (This is the threshold-vs-plateau distinction: a 0.5 A cutoff
  would have mislabeled it "floored at ~0.51" while it was in fact descending fastest.)

Held-out at 240k (same processed_small val=758; direct-comparable to the grid rows):

| model | val aa @60k (grid) | val aa @240k | val bb @240k | contact F1 @240k |
|---|---|---|---|---|
| 1.1M  | 0.916 | 0.863 | 0.605 | 0.950 |
| 15.2M | 0.844 | **0.752** | **0.476** | 0.968 |

Both held-out numbers IMPROVED with 4x compute (confirming the grid val rows were
pessimistic/under-converged), but far less than train did -- so the train/val gap
WIDENED at fixed n=2272 (15.2M v/t 1.23 -> ~2.5). **At fixed data, extra compute
mostly buys train fit; held-out is data-limited once train converges.** 15.2M @
240k = 0.752 A all-atom / 0.476 A backbone is the best held-out on this val set.

## Bottom line (FINAL)
1. **Data is a real lever** (Q1, solid): held-out descends with n and the train/val
   gap closes. REPORT.md's "more data won't help" is void.
2. **No capacity floor was found for either model.** The resume control settles it:
   both 1.1M and 15.2M train curves are still descending at 240k steps (train 0.51
   and 0.39, plateau test -0.185 / -0.073). The 15.2M's apparent 0.68 "floor" and
   the earlier joint-scaling story were undertraining artifacts. RETRACTED and now
   positively refuted.
3. **The grid's data slopes are lower bounds; the projections are overestimates.**
   High-n rungs were under-converged, so a converged sweep would show a steeper
   data slope and 0.5 A reached at fewer than the naive 9.8k (1.1M) / 370k (15.2M)
   structures. Measuring the TRUE slope needs a converged data sweep (matched
   epochs, or matched steps at a much higher budget).
4. **At fixed data, compute and data trade off as expected:** extra compute drives
   train down but held-out only modestly (gap widens) -> held-out is data-limited
   once train converges. Best held-out on this val set: 15.2M @ 240k = 0.752 A
   all-atom / 0.476 A backbone.
