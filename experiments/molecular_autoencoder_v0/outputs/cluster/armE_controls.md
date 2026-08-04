# Arm E controls (E1/E2) on the shared-latent verdict

Two controls on the C/D verdict (`atomlatent_verdict.md`), not successor
architectures. All on the complex corpus; reference arm A aa = 5.462.

| cell | aa RMSD | x arm A |
|---|---|---|
| E1 fixed-L128, canonical, d8 | 13.09 | 2.40 |
| E1 fixed-L128, canonical, d32 | 9.08 | 1.66 |
| E1 fixed-L128, file, d8 | 7.22 | 1.32 |
| E1 fixed-L128, file, d32 | 6.60 | 1.21 |
| E2 r=8 (budget-violating), d8 | 7.83 | 1.43 |
| E2 r=8 (budget-violating), d32 | 12.05 | 2.21 |

## E1 prediction — CONFIRMED (routing was never the problem)

Locked prediction: E1 (deterministic routing, matched L=128 budget) ~= C/D
(learned routing) -> routing was never the problem, the shared-latent bottleneck
is; E1 >> C/D -> learned routing was the defect. **E1_canonical_d32 = 1.66x vs
C/D best (C_local_group_L128_d32) 1.81x = an 8% difference, inside the +/-15%
band.** Deterministic routing did NOT rescue it. Recorded as confirmed --
including that the prediction would have been wrong had E1 jumped to near arm A.
Consistent with the B-row (no routing also fails).

## Scalars, not token count, is the binding variable (the useful byproduct)

At **fixed L=128 (constant token count)**, widening tokens d8 -> d32 moved
canonical E1 from **13.09 -> 9.08, a 30% gain**. Token count held constant, only
scalars-per-token increased -> direct evidence that the binding budget variable is
**scalars, not tokens** -- exactly how §7 frames the per-molecule budget (require
in scalars, d-agnostic). This wasn't what the E cells were designed to test and is
the most useful thing they produced.

## Ordering caveat (point 3), and an E2 anomaly

E1 **file order beats canonical** (d32: 6.60 vs 9.08; d8: 7.22 vs 13.09).
Deterministic segment pooling on file order rides protein **sequence coherence**
(file order ~= sequence), which does NOT transfer to general molecules -- so E1's
best cell (file, 1.21x) is a sequence-coherence artifact, not the mechanism
working generally. E2 is anomalous (d32 2.21x WORSE than d8 1.43x); noted, not
pursued (E2 is the budget-violating control, not a candidate).

## Bottom line

No shared-latent variant -- learned routing (C/D), deterministic routing (E1),
matched or budget-violating (E2) -- reaches arm A; every one is 1.2-3.4x. The
verdict stands, and the scalars-not-tokens result is the byproduct worth keeping.
