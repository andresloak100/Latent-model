# Perceiver depth grid + masked objective (RESULT: depth is a bounded lever; failure is memorization)

Tests whether decoder depth rescues the fixed-size Perceiver latent that scored
~9.3 A in the prior sweep. Grid: **x = dec_cross_layers {1,2,4,8}** x
**s = group_self_layers {0,2,4}**, <=800-atom set (data/processed, 878 train /
293 val), 242k steps. Reference: grid_direct_d8_reco = 0.75 A bb / 1.02 A aa /
chir 0.001 at 2.9x (direct per-residue codec, lr 1e-3).

## lr 1e-3 is unusable for the Perceiver -- the depth verdict comes from lr 3e-4
The grid was launched at lr 1e-3 to match the reference, but **lr 1e-3
destabilizes the Perceiver** (it did NOT destabilize the direct codec -- mask_direct_d8
below trained fine at 1e-3). Failure mode scales with attention depth:
- **Terminal nan-weight divergence:** x4_s2 (~ep400), x8_s4 (~ep400), x2_s2 (~ep500);
  even s0 cells eventually died -- x1_s0 (~ep1700), x2_s0 (~ep2100).
- **Chronic nan / frozen:** x4_s4 and mask_pgrid_x4_s2 froze at their ep-400 value
  (GradScaler skipped every update); perceiver_deep_L64_lr1e3 nan'd every epoch
  350-899 but weights survived and it matched its 3e-4 twin (8.21 vs 8.26).
- Deeper (group-self >= 2) dies fastest; the R x R group-self stack is the culprit.
All diverged lr-1e-3 cells were scancelled (dead compute). **Verdict below uses the
lr-3e-4 controls**, which trained cleanly to 242k. (This confirms the earlier
lr-artifact prediction: on the arms that survived, 1e-3 and 3e-4 reach the same
result; 1e-3 just risks death.)

## Scaling curves -- lr 3e-4 controls, full 242k (val all-atom vs steps)
| cell | 33k | 66k | 132k | 198k | 242k | verdict |
|---|---|---|---|---|---|---|
| x8/s0 | 9.29 | 9.28 | 9.27 | 9.30 | 9.34 | plateaued by 33k |
| x4/s4 | 7.79 | 7.72 | 7.75 | 7.78 | 7.79 | plateaued by 66k |
| x8/s4 | 7.66 | 7.57 | 7.53 | 7.50 | 7.50 | descends to ~180k, then flat |

**All cells plateau by 242k** -- x8/s4 descends longest but flattens at 7.50. More
STEPS will not help; depth is exhausted at ~7.5 A.

## Final numbers (lr 3e-4, 242k; deep retries 900 ep; train NEXT to val)
| arm | cell | train | val aa | val bb | chir | bond | F1 |
|---|---|---|---|---|---|---|---|
| pgrid_x8_s4_lr3e4 | x8/s4 | 0.714 | **7.508** | 6.977 | 0.047 | 0.565 | 0.426 |
| pgrid_x4_s4_lr3e4 | x4/s4 | 0.812 | 7.804 | 7.266 | 0.090 | 0.629 | 0.409 |
| pgrid_x8_s0_lr3e4 | x8/s0 | 0.869 | 9.351 | 8.853 | 0.085 | 0.692 | 0.350 |
| perceiver_deep_L64 | x4/s2 | 1.322 | 8.261 | 7.720 | 0.173 | 0.665 | 0.375 |
| perceiver_deep_L256 | x4/s2 | 1.263 | 8.168 | 7.666 | 0.170 | 0.664 | 0.374 |
| **grid_direct_d8_reco** | direct | ~0.5 | **1.02** | **0.75** | **0.001** | - | ~0.95 |

## Findings
1. **Group-self (s) is the lever; decoder cross-depth (x) is nearly inert alone.**
   Group-self ladder (lr3e-4): s0 9.35 -> s2 8.26 -> s4 ~7.5-7.8 (~1.85 A). Cross-depth
   at s0 was flat ~9.3-9.7 across x1->x8 (prior sweep); it helps only WITH group-self
   (x4/s4 7.80 -> x8/s4 7.51, ~0.3 A). Best cell x8/s4 = **7.508 A aa / 6.977 bb**.
2. **Independent confirmation:** perceiver_deep_L64 (x4/s2, separate config/code path)
   lands at 8.26 -- exactly the s2 rung of that ladder -- so the group-self finding
   replicates. L64 = L256 (8.26 vs 8.17): latent width is irrelevant, as before.
3. **The failure is MEMORIZATION, not decoder capacity or optimization.** The best
   cell fits TRAIN to 0.714 A -- nearly the direct codec's ~0.5 -- but generalizes to
   7.508 A: a **10.5x train->val gap** (vs the direct codec's ~2x). Better configs fit
   train better and the gap WIDENS. Depth improved the fit and bought little on val.
4. **Still ~7x worse than the direct codec** (7.51 vs 1.02 aa; 6.98 vs 0.75 bb; chir
   0.047 vs 0.001). Depth moves within the failure regime; it does not escape it.

## Masked objective on the direct codec (mask_direct_d8) -- NEUTRAL (standalone result)
Direct codec + 15% coord corruption vs plain reconstruction, only the objective differs:
| | backbone | all-atom | chir | bond | F1 | comp |
|---|---|---|---|---|---|---|
| grid_direct_d8_reco | 0.75 | 1.02 | 0.001 | - | ~0.95 | 2.9x |
| **mask_direct_d8** | **0.777** | **1.016** | 0.0007 | 0.168 | 0.938 | 2.92x |

Identical within noise; converges to the same ~1.0 A plateau by ~33k steps. **Masking
neither helps nor hurts the codec that works** -- expected given its small ~1.9x gap.
(The masked Perceiver, where the 6-10x gap could let a regularizer bite, is being
retried at lr 3e-4 -- mask_pgrid at 1e-3 nan-froze at 8.08.)

## Bottom line
Decoder depth is a real but **bounded** lever inside the Perceiver: group-self buys
~1.85 A, cross-depth ~0.3 A, plateauing at ~7.5 A -- still ~7x off the direct codec.
The residual failure is **memorization** (train 0.71 / val 7.51), not capacity or
optimization. So the open question is no longer depth but **data** -- tested by the
architecture x data ladder (running).
