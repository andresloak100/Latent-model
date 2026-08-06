# Effective sample size, not frame count — and why the ceiling is N-biased on ATLAS

**Status:** guard result FINAL (n=115 held-out, stress-tested). n_eff result INTERIM (n=20 spanning
the full N axis); full-corpus run is job 10305556.

The frames-per-rank90 table previously reported (13.3× → 26.7× going 1 → 2 replicas) counted **frames**.
Frames are not samples. The right denominator is `n_eff = F / tau_int`, with
`tau_int = 1 + 2*sum_{k>=1} rho(k)` truncated at `rho < 0.05` — the same convention as the ensemble
acceptance test. Measured on PCA-mode coefficient time series, which the Gram gives for free: for
`X (F,3N)`, `G = X X^T` (F,F), `eigh -> U,w`, and mode *i*'s coefficient series is `U[:,i]*sqrt(w_i)`.
No 3N-sized array is ever formed.

## The guard fails under the new protocol

Re-measured with PCA fitted on replicas 0+1 and evaluated on replica 2 (job 10304109, n=115):

| k | median FVE | slope vs log10(N) | p | verdict |
|---|---|---|---|---|
| 24 | 0.5742 | **−0.1430 ± 0.0818** | 0.001 | degrades with N |
| 256 | 0.8614 | **−0.1880 ± 0.0439** | 0.000 | degrades with N |

Realised span 1.75 decades, rank-void 0/115, median s_k/s_1 = 1.31e-01 (k=24) and 2.87e-02 (k=256).

The stale number on record was −0.1096 ± 0.1311, **not** significant. The new one is, and is larger.
The old number is retired and must not be quoted — it evaluated on held-out frames from the *same*
trajectory, which is a strictly easier test.

Stress-tested four ways, because 2 held-out systems are still uncached:

| k | OLS | bootstrap 95% | drop-most-influential | adversarial +2 at Nmax, best-in-corpus FVE |
|---|---|---|---|---|
| 24 | −0.1430 | [−0.2271, −0.0624] | −0.1606 | −0.0960 |
| 256 | −0.1880 | [−0.2381, −0.1387] | −0.1754 | −0.1541 |

Negative under all four, at both k. (An earlier +10-point adversarial was run before the cache caught
up; with only 2 systems actually missing, +2 is the honest worst case.)

## Four artifact checks, run before any of this was trusted

**A. Rigid-body motion — RULED OUT.** If frames were not superposed, PC1 would be global rotation,
whose correlation time grows with radius — manufacturing both a huge tau *and* a spurious IAT-vs-N
slope. Measured across N = 598 → 33,377: Kabsch removes 1.5% / 0.8% / 0.3% / 0.1% / **0.0%** of
variance (*decreasing* with N), COM drift ≤ 0.55 Å, net rotation ≤ 2.4°, and tau agrees to <1%
between raw and aligned (716.6 vs 719.0; 1060.1 vs 1060.0). ATLAS ships superposed trajectories.

**B. Lag-cap censoring — FOUND AND FIXED.** The first implementation capped `maxlag` at 500; 4 of 5
systems needed lags of 625–841 to reach rho < 0.05. That truncation censors *large* tau
preferentially — i.e. it censors exactly the variable under test, and biases the IAT-vs-N slope
**downward**. This is Family B (a count pinned to its measurement ceiling), and it was live in a
submitted job, which was cancelled and its partial output deleted. `maxlag` is now F//2 and every
system carries a convergence flag; non-converged tau are reported as lower bounds.

**C. Mode-index dependence — CHANGES THE FRAMING.** tau is not a per-system scalar. It collapses with
mode index:

| mode | 0 | 1 | 4 | 9 | 49 | 199 | 499 |
|---|---|---|---|---|---|---|---|
| tau (frames) | 719–1060 | 136–628 | 117–322 | 49–177 | ~17 | ~3 | ~1–2 |

The *leading* modes are starved; the bulk of rank90 is effectively independent frame-to-frame.
Comparing n_eff(PC1) against rank90 would pit the worst-case mode against a count dominated by
well-sampled ones. Everything below is therefore mode-resolved, with tau averaged over **exactly** the
modes being counted.

**D. Selection effect — REAL, QUANTIFIED, NOT CORRECTED OUT.** PCA picks the highest-variance
direction, which on a finite trajectory preferentially finds slow drift, so tau(PC1) is an extreme
order statistic and upward-biased by construction. Against a random-projection null: 3.1× / 6.1× /
5.0× / 3.5× / 9.5×. tau(PC1) therefore *overstates* how starved a typical direction is — which is why
the mode-averaged quantity, not PC1, carries the result.

## IAT does grow with N — but it is the minority mechanism

Interim, n=20 spanning N = 598 → 33,377 (slopes are per decade of N, 95% CI):

| quantity | slope | R² | |
|---|---|---|---|
| tau, PC1 alone | +0.1405 ± 0.1696 | 0.14 | n.s. — noisy extreme order statistic |
| tau, random projection | +0.1082 ± 0.2287 | 0.05 | n.s. — unselected reference |
| **tau, mean over 24 modes** | **+0.1713 ± 0.0554** | 0.70 | **significant** |
| **tau, mean over 256 modes** | **+0.1745 ± 0.0416** | 0.81 | **significant** |
| n_eff at k=256 | −0.1745 ± 0.0416 | 0.81 | significant |
| n_eff / rank90 | −0.8239 ± 0.4569 | 0.44 | significant — the conditioning quantity |
| rank90 | +0.6494 ± 0.4712 | 0.32 | significant — for attribution |

**The hypothesis is confirmed but is not the whole story.** IAT-vs-N is real: τ roughly **doubles**
across the realised 1.75-decade span (10^(0.1745×1.75) = 2.02×), so n_eff falls from ~165 to ~94 at
k=256. But the conditioning slope decomposes exactly:

```
log(n_eff/rank90) = log(n_eff) − log(rank90)
    −0.1745  (IAT growing with N)        =  21% of the degradation
    −0.6494  (more modes needed at large N) =  79% of the degradation
    ───────
    −0.8239  matches the measured −0.8239
```

So slower collective modes in bigger proteins are a **confirmed minority mechanism at ~21%**. The
dominant term is simply that large proteins need more modes to reach 90% variance, against an n_eff
that is small for every system regardless of size.

Note the single most alarming number is not a slope at all: **n_eff at k=24 is 14–24 raw-equivalent
samples**, and at k=256 it is 94–165 — against 2,501 frames. Efficiency is ~1–7%. At k=256 the ceiling
is fitting a 256-dimensional subspace from fewer than 256 effective samples in **every** system.

## Neither more frames nor more replicas fixes this

Both terms are immune to the two levers available on this corpus:

- **Denser subsampling of the same 100 ns adds nothing.** n_eff is set by tau_int, not by stride.
  Halving the stride doubles frames and doubles tau in frame units, leaving n_eff unchanged. It also
  does not reduce rank90.
- **More replicas add far less than their nominal factor.** Measured between/within RMSD ratio is
  **1.18** (range 1.085–1.442) across 10 proteins spanning N = 598–33,377 — replicas do not
  decorrelate over 100 ns. The raw 1 → 2 replica doubling in the old table is an upper bound; the
  effective gain is well below 2×.

Only **longer trajectories** would raise n_eff, and ATLAS does not have them.

## Consequence, stated plainly

**The PCA ceiling is irreducibly N-biased on this corpus.** This is not a bug to fix; it is a property
of 100 ns trajectories applied to proteins whose slowest modes have correlation times of ~12–42 ns.

That is survivable, and the reason it is survivable was designed in beforehand:

- **PRIMARY — codec vs ANM, both zero-shot, scored on the same held-out replica-2 frames — needs no
  ceiling.** It has no denominator, no per-system oracle, and nothing measured here touches it. This
  is the **main line**, not a fallback.
- **SECONDARY — the oracle-fraction (codec FVE as a fraction of the PCA ceiling) — is now permanently
  caveated.** Every future report of it must carry the slope (−0.1880 ± 0.0439 at k=256) and the bias
  direction: a ceiling that degrades with N makes the high-N oracle-fraction an **upper bound**, i.e.
  it flatters the codec at large N. It is reported with that attached or not at all.

The decision this forecloses: do **not** spend another corpus move chasing a sound ceiling. No
available dataset supplies the trajectory length that would fix it.

## Open

- rank90 reaches 807 of 2,501 usable rank (32%) at the top of the N axis, past the 30% edge where
  rank90 itself starts pressing against its own measurement ceiling (Family B). The full run reports
  this per system; if it bites, the +0.6494 rank90 slope is itself an underestimate.
- 2 held-out systems (3vth_A, 2po4_A) still uncached; guard is provisional on 123/125 pending the
  cache job. Extremes are preserved (realised range == selected range) and FAILED = 0.
- `conservation_report` fired a false "RUN IS VOID" banner on the guard: its denominator was the whole
  126-system store rather than the held-out list the guard deliberately processes. Fixed by making the
  intended population an explicit argument, and by distinguishing *never attempted* (coverage
  shortfall, cache still building) from *failed* (size-correlated dropout). A warning that cries wolf
  trains the reader to ignore the one warning that must never be ignored.
