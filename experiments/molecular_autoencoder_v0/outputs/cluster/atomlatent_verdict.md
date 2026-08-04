# Atom-latent C/D cross — verdict (against the pre-registration in atomlatent_verdict_prereg.md)

16 cells, all post-fix (4f66deb unit bug + fps_anchors; group and graph halves
both rerun so the addressing comparison is attributable). Diagnostic run BEFORE
RMSD per protocol. Reference: arm A (direct per-residue) aa = 5.462 A, same
in-job eval pipeline. Pre-registered gates (committed 84c486e) applied in order.

| cell | L | anchor | distinct | Jac/null | locgain | aa RMSD | aa/A |
|---|---|---|---|---|---|---|---|
| C_local_graph_L32_d8   | 32  | 1.21 | 0.29 | 1.72 | -0.42 | 17.81 | 3.26 |
| C_local_graph_L32_d32  | 32  | 1.19 | 0.29 | 2.71 | -0.40 | 18.30 | 3.35 |
| C_local_graph_L128_d8  | 128 | 1.08 | 0.81 | 1.96 | -0.01 | 17.69 | 3.24 |
| C_local_graph_L128_d32 | 128 | 1.07 | 0.88 | 2.96 | -0.35 | 17.39 | 3.18 |
| C_local_group_L32_d8   | 32  | 1.19 | 0.03 | 3.56 | +2.77 | 13.88 | 2.54 |
| C_local_group_L32_d32  | 32  | 1.19 | 0.16 | 1.28 | +3.08 | 12.88 | 2.36 |
| C_local_group_L128_d8  | 128 | 1.08 | 0.25 | 2.05 | +2.64 | 13.75 | 2.52 |
| C_local_group_L128_d32 | 128 | 1.07 | 0.54 | 1.84 | +5.89 |  9.91 | 1.81 |
| D_local_masked_graph_L32_d8    | 32  | 1.19 | 0.34 | 1.28 | -0.27 | 17.61 | 3.22 |
| D_local_masked_graph_L32_d32   | 32  | 1.19 | 0.64 | 0.90 | -0.36 | 17.40 | 3.19 |
| D_local_masked_graph_L128_d8   | 128 | 1.08 | 0.80 | 1.55 | -0.14 | 18.47 | 3.38 |
| D_local_masked_graph_L128_d32  | 128 | 1.07 | 0.83 | 2.41 | -0.71 | 17.33 | 3.17 |
| D_local_masked_group_L32_d8    | 32  | 1.19 | 0.07 | 1.22 | +3.29 | 13.67 | 2.50 |
| D_local_masked_group_L32_d32   | 32  | 1.19 | 0.11 | 1.79 | +2.76 | 12.92 | 2.37 |
| D_local_masked_group_L128_d8   | 128 | 1.07 | 0.31 | 1.60 | +2.91 | 13.51 | 2.47 |
| D_local_masked_group_L128_d32  | 128 | 1.07 | 0.23 | 1.88 | +3.25 | 13.37 | 2.45 |

## Gates, in the pre-registered order

1. **GATE (anchor_spread_ratio < 0.3): CLEARS.** All 16 cells 1.07-1.21 (>1.0);
   the fps_anchors fix worked (the void pre-fix run was 0.016). Anchors are
   well-spread; **RMSDs are interpretable.**
2. **Route collapse (distinct < 0.5): 10/16.** But it splits by addressing:
   graph-L128 preserves distinct routing (0.80-0.88), graph-L32 partial
   (0.29-0.64), **group collapses hard (0.03-0.54).**
3. **Jaccard > null (k^2/L at each cell's own L): 15/16 above.** Only
   D_graph_L32_d32 (0.90) is marginally below. Addressing is preserved in 15/16.
4. **Graph vs group RMSD: graph 3.25x, group 2.38x arm A.** Graph is **36% worse
   than group** in-distribution -- past the pre-registered ~15% line, so the
   **genuine-negative side for the graph path in-distribution.** (Caveat kept from
   the pre-reg: graph's payoff is non-protein / past-training-length
   generalization, which this cross does not measure -- so this is not a verdict
   on graph's intended use.)

   **Prediction miss, recorded:** parity was predicted; graph came in 36% worse,
   past the 15% line. This makes graph-derived identity's **in-distribution cost
   measurable**, which turns the OOD / past-training-length test from optional to
   **mandatory** -- it is the only place that cost can be earned back. Until it
   runs, the graph path is **unjustified, not merely unproven**. Queued behind
   E1/E2.

## The decisive result: the abandon-arm-C condition is met

The pre-registration named one outcome that abandons arm C: **anchors healthy +
routing healthy + Jaccard above null + RMSD still ~2x arm A** -> addressing is
fine and the deficit is **capacity allocation through the bottleneck, which
anchors and locality cannot fix.**

The graph-L128 cells satisfy it and then some: anchors 1.07-1.08, distinct
0.80-0.88, Jaccard 2-3x null -- **addressing is demonstrably intact** -- yet aa
RMSD is **~3.2x arm A.** Two corroborations:

- **The inversion IS the finding** (stronger than the address-destroyed path we
  pre-registered). The best-reconstructing cells (group, 2.38x) are the ones that
  COLLAPSE routing (distinct 0.03-0.54); the cells that PRESERVE per-atom routing
  (graph-L128, distinct 0.8+) reconstruct WORST (3.2x). So **at fixed capacity,
  addressability and fidelity compete directly** -- spending latent capacity on
  keeping atoms individually addressed costs reconstruction. That convicts the
  bottleneck design's *premise* (that per-atom addressability through shared
  latents is what to preserve), not merely its execution.
- **Locality is not an independent good.** locality_gain is <= 0 on every graph
  cell (local cross-attention inert or harmful where routing already works) and
  large positive (+2.6 to +5.9) only on group cells, where it *compensates* for
  routing collapse. Arm C's local cross-attention earns nothing on top of intact
  addressing.

And every one of the 16 cells is **1.81-3.38x arm A.** No shared-latent
configuration -- any arm, addressing, L, or d -- approaches the direct
per-residue codec. The single best (C_group_L128_d32, 1.81x) is still ~2x arm A
and is carried by locality (locgain 5.89), not by addressing (distinct 0.54,
marginal).

## Corroboration from the B row (no routing)

Two pre-fix monitors fired late with an L128_d32 snapshot and a "~9.3 A bar"
whose definition does not match the current baselines (arm A aa 5.462; cohort
mean_shape bb ~5.0; pca_k8 bb ~1.9) -- so that bar framing is discarded. But the
numbers are the current post-fix runs and add the **B row** (no routing), which
the 16-cell verdict did not cover:

| arm, L128_d32 | graph aa | group aa | graph aa/A | group aa/A |
|---|---|---|---|---|
| **B_global (NO routing)** | 17.53 | 14.02 | 3.21 | 2.57 |
| C_local | 17.39 | 9.91 | 3.18 | 1.81 |
| D_local_masked | 17.33 | 13.37 | 3.17 | 2.45 |

B has **no routing at all**, yet fails identically (graph 3.2x, group 2.6x arm A).
Routing (C/D) vs no routing (B) barely moves the result -- confirming the deficit
is the shared-latent **bottleneck itself**, not the routing layered on it. And
graph > group holds in all three arms, so the graph penalty is general, not a
routing artefact. Every shared-latent variant loses to the direct per-residue
codec. (The "universal path live" reading from the stale monitor's own framework
requires both < bar; both are *above* it -- the architecture-fails branch.)

## Verdict

**The shared-latent atom bottleneck (arms C/D) is refuted.** With the anchor-
collapse confound removed by the fps fix, this is a clean read: even with healthy
anchors, preserved distinct routing (graph half), and above-null Jaccard,
reconstruction is ~3x arm A. The deficit is the **bottleneck's capacity
allocation**, not addressing -- exactly the pre-registered abandon condition.

Note this is NOT the "address destroyed -> arm E" path anticipated earlier: on
the graph half the address is *intact* and it still fails. That is a stronger
result -- the problem is the learned shared-latent bottleneck itself, not the
addressing scheme layered on it. **Arm E** (strided-conv pooling, atom i -> latent
floor(i/r), no learned bottleneck routing) is the indicated intervention: it
removes the exact component this cross convicts.
