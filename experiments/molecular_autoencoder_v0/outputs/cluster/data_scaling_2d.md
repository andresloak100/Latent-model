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

## Q3: is there a capacity floor? UNRESOLVED -- earlier "floor" claim RETRACTED
The prior version claimed a capacity floor ("neither reaches 0.5 A at any data")
from train RMSD rising with n. That is NOT supported: (a) the rise is confounded
-- high-n rungs got 1/5 the epochs (matched steps); (b) train RMSD is
augment-noisy; (c) the capacity ladder shows 15.2M reaching train 0.300 at n=878,
so 0.68 is not its floor. **Retracted: the capacity-floor reading and the
"0.5 A needs joint capacity+data scaling" bottom line that rested on it.**

Settled by a compute control (resume the n=2272 checkpoints -- no fresh training):
continue both models to 120k and 240k steps.

### Resume control -- interim (through ~120k steps)
* **15.2M: undertraining CONFIRMED.** Train RMSD fell 0.68 (@60k) -> ~0.45
  (@~113-120k), clearly below 0.5 with a third of the budget still to run. The
  0.68 was never a floor. (Augment-noisy: single quick-RMSD points swing 0.45-0.87
  on 4-batch samples; the ~0.45 is the windowed level.)
* **1.1M: still descending, NOT plateaued.** 0.806 (@65k) -> ~0.68 windowed
  (@120k) = -0.144 A per compute-doubling, still going down. That slope crosses
  0.5 near ~285k steps, past the 240k endpoint, so it will likely finish ~0.53-0.55
  while still improving. **Do NOT call it capacity-limited at 240k on a threshold.**
  Decide by PLATEAU: if 120k -> 240k buys < ~0.05 A it is a floor; otherwise it is
  still compute-limited and is reported that way.
[240k endpoint pending; this section finalizes when it lands.]

## Bottom line (current)
Data is a real lever: held-out descends and the train/val gap closes (Q1, solid).
The "capacity floor" reading is retracted -- the 15.2M's apparent floor was
undertraining (0.68 -> 0.45 with more steps). Because the grid's high-n rungs are
under-converged, the measured data slopes are lower bounds and the structure-count
projections are overestimates: less data than 9.8k/370k is likely needed, but the
true amount requires a converged data sweep to measure. Whether the small 1.1M
model has a genuine floor is pending the 240k plateau test.
