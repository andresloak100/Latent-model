# 2-D scaling study: data x model capacity (direct decoder, working codec)

Motivation: REPORT.md's "more data won't help" was measured on the 9.5 A
attention decoder (broken) -> void for the direct decoder. This grid tests data
scaling with the WORKING codec, at two model sizes, to find the compute-optimal
frontier.

## CRITICAL COMPARABILITY CAVEAT (read before comparing any numbers)
These cells score against **processed_small's val = 758 structures**. The
capacity ladder and all earlier direct-decoder numbers (1.020 / 0.938 / 0.873 /
1.085 ...) scored against **processed's val = 293** -- a DIFFERENT evaluation
set. Do NOT cross-compare absolutes across the two (e.g. this n=878 cell is NOT
comparable to the capacity ladder's 1.085, despite the matching train count).
Read TRENDS WITHIN this grid only. (Same class of error as the earlier
backbone-vs-all-atom mix-up; stated here so nobody repeats it.)

## Grid (held-out all-atom RMSD, A; matched 60k optimizer-steps, lr 3e-4)

| n_train | 1.1M train | 1.1M val | v/t | 15.2M train | 15.2M val | v/t |
|---|---|---|---|---|---|---|
| 450  | 0.576 | 1.185 | 2.06 | 0.252 | 0.934 | 3.71 |
| 878  | 0.786 | 0.993 | 1.26 | 0.598 | 0.930 | 1.56 |
| 2000 | 0.800 | 0.858 | 1.07 | 0.638 | 0.837 | 1.31 |
| 2272 | 0.884 | 0.916 | 1.04 | 0.684 | 0.844 | 1.23 |

Top rung is 2272 (matched single-chain X-ray <=2.5A 20-110res band ceiling =
3853 ids -> 3030 structs). The 4000 rung does not exist for this band.

## Q1: does held-out descend with data? YES (both models)
1.1M: 1.185 -> 0.993 -> 0.858 (2272 = 0.916, within noise of 2000). 15.2M: 0.934
-> 0.930 -> 0.837. **REPORT.md's "more data won't help" is void** -- it was the
broken decoder. And val/train ratio SHRINKS toward 1.0 as data grows (1.1M
2.06->1.04; 15.2M 3.71->1.23): more data CLOSES the generalization gap. (This
does not contradict the capacity ladder, where the ratio grew as MODEL grew at
FIXED data=878 -- different axis. Both hold: data closes the gap; capacity at
fixed data widens it.)

## Q2: data slope + extrapolation (fit on log2(n), range 450->2000 = 2.15 doublings; 2272 excluded as a 1.14x near-duplicate of 2000)
- 1.1M:  slope -0.150 A/doubling
- 15.2M: slope -0.046 A/doubling

Naive extrapolation to 0.5 A: 1.1M ~9,800 structs (~4x beyond 2272 / 3x past the
3853 ceiling); 15.2M ~370,000 structs (~160x beyond / ~95x past ceiling). **These
are PROJECTIONS, not measurements, and both are unreliable -- see Q3: capacity
floors block 0.5 A regardless of data, so neither slope holds to 0.5.**

## Q3: is 15.2M hungrier for data than 1.1M? NO -- and both are capacity-floored
- 15.2M is BETTER at low data (0.934 vs 1.185 @ n=450) and has a FLATTER slope --
  it extracts more from little data; it is LESS data-hungry, not more.
- But note train RMSD: 1.1M rises 0.576->0.884 as data grows (it can no longer
  memorize; converging to its ~0.9 CAPACITY FLOOR); 15.2M rises 0.252->0.684
  (lower floor). **Both models' TRAIN error at 2272 (0.88 / 0.68) already exceeds
  0.5 A -- so neither can reach 0.5 A val at ANY data volume.** Reaching 0.5 A
  requires a model whose train floor is < 0.5 (bigger than 15.2M) AND enough data
  to close the gap. The compute-optimal path is JOINT scaling of capacity + data,
  not either axis alone -- which is exactly why the single-axis extrapolations in
  Q2 are not targets.

## Bottom line
Data is a real lever (gap closes, held-out descends) -- REPORT.md's claim is
void. But data alone plateaus at each model's capacity floor (1.1M ~0.9, 15.2M
~0.84 val at this data). 0.5 A needs BOTH more capacity and more data. For the
eventual pretraining run, size the model and the dataset together.
