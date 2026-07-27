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

## Q1: does held-out descend with data? YES (SOLID)
1.1M 1.185->0.993->0.858; 15.2M 0.934->0.930->0.837 (2272 within noise). And
val/train ratio closes toward ~1.0 as data grows (1.1M 2.06->1.04; 15.2M
3.71->1.23): more data closes the generalization gap. **REPORT.md's "more data
won't help" is void** (it was the broken 9.5 A attention decoder). This is the
headline and it stands.

## Q2: data slope (fit log2(n) over measured range 450->2000 = 2.15 doublings; 2272 excluded, 1.14x near-duplicate of 2000)
1.1M -0.150 A/doubling; 15.2M -0.046 A/doubling (15.2M is better at low data and
flatter -> LESS data-hungry, not more, on val). Naive 0.5 A projections (1.1M
~9.8k structs ~4x beyond 2272; 15.2M ~370k ~160x beyond) are PROJECTIONS not
measurements, and are only meaningful if the slope is real rather than
undertraining -- see Q3.

## Q3: is there a capacity floor? UNRESOLVED -- earlier "floor" claim RETRACTED
The prior version claimed a capacity floor ("neither reaches 0.5 A at any data")
from train RMSD rising with n. That is NOT supported: (a) the rise is confounded
-- high-n rungs got 1/5 the epochs (matched steps); (b) train RMSD is
augment-noisy; (c) the capacity ladder shows 15.2M reaching train 0.300 at n=878,
so 0.68 is not its floor. **Retracted: the capacity-floor reading and the
"0.5 A needs joint capacity+data scaling" bottom line that rested on it.**

Settled by a compute control (resume the n=2272 checkpoints -- no fresh training):
continue both models to 120k and 240k steps.
  * train RMSD descends below 0.5 -> the "floor" was undertraining; data
    extrapolation is back on the table.
  * train RMSD plateaus -> the capacity-floor reading is earned.
[Control in progress; this section updates when it lands.]

## Bottom line (current)
Data is a real lever: held-out descends and the train/val gap closes (Q1, solid).
Whether that extrapolates toward 0.5 A, or hits a capacity wall, is UNRESOLVED
pending the resume control -- no joint-scaling conclusion is drawn yet.
