# Ensemble validation of the rollout: did the propagator learn dynamics?

"0% out of range" is necessary but not sufficient -- a model emitting the training
mean every step also scores 0%, so does one with far too little variance. Test the
sufficient conditions: generated rollout vs REFERENCE MD in ANM-mode coefficient
space, sqrt(kT/lambda) whitening, held-out (frozen-7) systems, 2000-step rollout.

## Result

| system | nCA | std ratio (gen/ref) | IAT gen | IAT ref | IAT ratio | JS | basin cover | spurious |
|---|---|---|---|---|---|---|---|---|
| 3a5zD02 | 64  | 0.48 | 1.6 | 8.3  | 0.19 | 0.29 | 47% | 28% |
| 3jvvA01 | 100 | 0.31 | 1.6 | 4.5  | 0.35 | 0.34 | 35% | 19% |
| 3a9lA00 | 207 | 0.24 | 3.0 | 25.8 | 0.11 | 0.50 | 18% | 25% |
| MEAN    |     | **0.34** | 2.0 | 12.9 | **0.16** | | | |

## Verdict: VARIANCE COLLAPSE -- the in-range result was vacuous

Pre-registered first failure branch (std ratio well under 1 -> narrow blob, in-range
vacuous) is what happened.

- **Marginals collapsed:** generated per-mode std ~1/3 of reference (0.34). The
  rollout samples a tight blob, not the conformational distribution.
- **Kinetics wrong:** generated integrated autocorrelation time ~6x too short (ratio
  0.16). Real MD has slow collective modes (ref IAT up to 25.8 frames); generated
  decorrelates in ~2-3 frames -- fast, structureless noise.
- **Free-energy landscape wrong:** rollout covers only 18-47% of reference basins
  (misses the majority), 19-28% spurious density (filled barriers).

**This retroactively guts the earlier "strongest positive."** The 1000-step "0% out
of range, ensemble sampling" result was VACUOUS: a collapsed narrow blob trivially
stays inside the training range. The propagator did NOT learn dynamics; it learned
bounded noise around a too-narrow region.

## Consequence

Physics solved the codec; learning belongs in the propagator, and the current
skeleton propagator does not do the job. A small conditional MLP DDPM p(z_{t+1}|z_t)
trained on one trajectory is mean-reverting, low-variance, fast-decorrelating. The
obj-3 problem is now sharp: build a propagator whose generated ensemble matches the
reference in (1) per-mode variance, (2) autocorrelation time, (3) free-energy basins.
Ensemble validation (this script) is the acceptance test for any future propagator.
