# Atom-latent C/D cross — pre-registered interpretation (before the numbers)

Committed **before** the diagnostic is read, so the interpretation cannot be
fitted to the results afterward. The 16-cell cross is B/C/D x {graph,group} x
{L32,L128} x {d8,d32}, all post-fix (4f66deb + fps_anchors, per-molecule sound),
group and graph halves both rerun so the addressing comparison is attributable.

Protocol: run `diagnose_atomlatent.py` FIRST; anchor spread before anything; then
distinct-patterns + gini primary, Jaccard-over-random secondary; L printed on
every row. Val RMSD stays unread until the gate below clears.

## Decision thresholds (locked)

1. **GATE — `anchor_spread_ratio < 0.3`** -> anchors collapsed; arm C has
   mechanically reduced to arm B. **No RMSD from these 16 cells is
   interpretable.** Everything below assumes this gate clears.

2. **`anchors > 0.6` but `locality_gain ~ 0` within noise** -> local
   cross-attention is inert and does not earn its cost. Kills the **locality
   machinery**, not the shared latent.

3. **`distinct_pattern_frac < 0.5`** -> route collapse; atoms are not
   individually addressed and the model is representing summaries. **Decisive
   against the current bottleneck design EVEN IF the RMSDs look good.**

4. **Jaccard** must beat `k^2 / L` computed at **each cell's OWN L**. Below that
   null = address not preserved. (Never compare Jaccard across cells of different
   L.)

## Graph vs group — stated in advance

Expectation: **graph ~= group in-distribution, possibly slightly worse.** Group
addressing has a real uniqueness advantage that graph-derived identity cannot
have for automorphic atoms, so **parity IS a success for the graph path** — its
payoff is non-protein and past-training-length generalization, which this cross
does not measure. **Graph worse by more than ~15% is the genuine negative.**

## What would abandon arm C

Anchors healthy + routing healthy + Jaccard above null + **RMSD still ~2x arm A**
-> addressing is fine and the deficit is **capacity allocation through the
bottleneck**, which anchors and locality cannot fix. That would mean the arm
needs a different intervention, not tuning.

_Pre-registered 2026-08-03._
