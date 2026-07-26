# Model-capacity sweep — corrected, lr-matched: the codec is DATA-limited

Same <=800 dataset, latent_dim=8, 900 epochs. lr=1e-3 (tuned for 1.1M) diverges/
underfits at larger sizes (train RMSD rises with size: 0.61->0.83->1.06->NaN),
so the valid ladder is lr-matched at 3e-4. The 1.1M point is now also run at 3e-4
(previously its only point was the confounded 1e-3 run).

## Corrected 4-point ladder — all lr=3e-4

| params | d_model/layers | train RMSD (ep899) | held-out all-atom | val/train ratio |
|---|---|---|---|---|
| 1.1M  | 128 / 2+2 | 0.662 | 1.085 | 1.64 |
| 3.8M  | 256 / 2+2 | 0.517 | 0.938 | 1.81 |
| 7.0M  | 256 / 4+4 | 0.316 | 0.910 | 2.88 |
| 15.2M | 384 / 4+4 | 0.273 | 0.873 | 3.20 |

(contrast: 1.1M @ 1e-3, its tuned lr: val 1.020, train 0.521, ratio 1.96 — slightly
better val than 1.1M @ 3e-4, confirming 1e-3 was tuned for the small model.)

## Verdict: DATA-limited, not a capacity dial

- Bigger models fit TRAIN monotonically better (0.66 -> 0.27) — capacity is usable.
- But held-out plateaus (1.085 -> 0.873, gains decelerating) while the val/train
  gap GROWS (1.64 -> 3.20). Generalization gap widening with capacity = the model
  is data-limited, not capacity-limited.
- 15.2M hits 0.273 A on SEEN data but 0.873 on held-out: it can already represent
  the precision we need; it just can't generalize from 878 training structures.
- lr-matched slope ~ -0.03 A per parameter-doubling -> ~1e11 params for 0.5 A.
  Scaling parameters is NOT the path. **878 training structures is the binding
  constraint.**

Size-generality (does 0.75 A hold past 800 atoms) is UNANSWERED: the 1.1M model
cannot fit the <=3000-atom / 3726-struct TRAIN set at any lr tried (train ~5.0 at
both 1e-3 and 3e-4, vs ~0.6 on <=800) — it is under-capacity for larger structures.
A clean size test needs a bigger model on the big data (capacity + size together).
