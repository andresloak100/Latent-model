# Intrinsic dimensionality of MD displacement vs system size -- the N^0.21 reading does NOT survive

## What prompted this
The Phase-1 ceiling table showed rank90 (PCA components for 90% of displacement variance) rising
29 -> 57 while N rose 959 -> 23,895 on MISATO: apparently ~N^0.21, "2x dimensionality for 25x
atoms", which would extrapolate to rank90 ~125-250 at 1e6 atoms -- a latent budget in the low
hundreds covering a million atoms, and the most direct evidence yet for objective 1. Two suspicions
had to be tested before it could be believed: (a) rank90=57 against a 79-frame cap is 72% of
available rank, so the exponent is CENSORED; (b) nothing controlled for how MOBILE each system is.

## Measurement
mdCATH, 28 local domains, ALL-ATOM, 320K, all 5 replicas concatenated -> 2,500 frames/domain
(vs MISATO's 100). Kabsch-aligned displacement from a common reference. rank90 swept over
nf = 79/200/400/800/1600/2400 to test convergence; exponent fitted with mobility control.
PBC screen (N-robust, see below): 0/28 excluded.

## Result 1 -- censoring is real and severe
Same domain, more frames -> much higher rank90 (4cd8A00: 53 @79 -> 191 @400; 3a9lA00: 52 @79 ->
387 @2400, still climbing). **8/28 domains are STILL not converged at 2,400 frames.** Max rank90
= 387 is only 16% of available rank, so this is genuine high-dimensionality, not a rank artifact.
=> MISATO's rank90 = 29-57 are substantial UNDERESTIMATES; anything fitted on 79 frames is a lower
bound on dimensionality.

## Result 2 -- the size effect is a MOBILITY CONFOUND (the headline)
Multiple regression, log10(rank90) ~ b*log10(N) + c*log10(medRMSF), n=28:

| term | estimate (95% CI) | significant |
|---|---|---|
| b (atom count) | **+0.135 +/- 0.302** | **NO -- CI spans 0** |
| c (mobility) | **-1.350 +/- 0.245** | yes, large |

R^2 = **0.858** with mobility vs **0.124** for N alone. Converged-only subset (n=20) agrees:
b = +0.157 +/- 0.349, c = -1.413 +/- 0.301, R^2 0.861.

Mediation, directly measured:
- corr(log RMSF, log rank90) = **-0.924** (p = 2.5e-12) -- mobility drives dimensionality
- corr(log N, log RMSF) = -0.310 -- larger domains tend to be more rigid
- corr(log N, log rank90) = +0.353 (p = 0.066) -- the naive "size effect", marginal

Mechanism: a floppy domain's variance is dominated by ONE large collective motion -> LOW rank90
(3jamY00, RMSF 12.9 A -> rank90 15). A stable domain's motion is diffuse thermal fluctuation spread
over many modes -> HIGH rank90 (3a9lA00, RMSF 1.35 A -> rank90 387). Because bigger proteins are
somewhat more rigid, size correlates with dimensionality WITHOUT causing it. **rank90 ~ RMSF^-1.35
with no detectable N dependence over an 11x size range** (N 711-7,524; RMSF 1.35-12.9 A; rank90
15-387). Counter-example in the raw data: 3h7lB02 (N=7,524, RMSF 1.82) converges at rank90 293,
LOWER than 3a9lA00 (N=3,173, RMSF 1.35) at 387 -- the larger system is the lower-dimensional one.

## What this does and does not license for objective 1
**Favourable in MECHANISM, unsupported as a NUMBER.**
- Favourable: dimensionality does not grow detectably with atom count at fixed dynamical character.
  A 1e6-atom assembly as rigid as these proteins would not need a latent budget scaled to its atom
  count. This is a better argument for objective 1 than N^0.21 was, because it is causal rather
  than correlational.
- Unsupported: the N range here is 11x and 1e6 atoms is ~130x beyond it. Extrapolating a NULL
  effect with CI +/-0.302 across 6 decades gives **65x uncertainty in each direction** from b's CI
  alone -- before counting the 8/28 unconverged domains. **No numerical 1M-atom latent budget is
  licensed by this data.** The earlier "rank90 ~125 at 1e6" figure should not be quoted.
- Design consequence: plan latent budget against **dynamical complexity (flexibility/mobility)**,
  not atom count. Two systems of equal size can differ 26x in required dimensionality.

## PBC / artifact screen (methodological, applies to both corpora)
A max-per-atom-displacement threshold is WRONG: max-over-atoms grows with N by extreme-value
statistics alone, so any fixed threshold preferentially drops large systems -- it dropped 7/8 and
8/8 of the two largest MISATO buckets, exactly the N-correlated exclusion bias that would bias a
size trend. Correct test: a PBC wrap is a DETACHED step population at ~box scale, so screen on the
fraction of atoms whose step exceeds 3x that system's OWN p99.9. Measured: step distributions are
smooth (p50 -> p99 -> p99.9 -> p99.99 -> max, no discontinuity), 0.00% detached atoms everywhere,
and max steps (MISATO 9.8-12.6 A, mdCATH 17.9-26.0 A) sit far below box/extent scale (37-65 A).
**Zero exclusions in either corpus.** Both earlier flags were false positives: 1IL3 is a fat-tailed
floppy system, 1CBR has its whole step distribution shifted up (genuinely mobile, medRMSF 5.05 A).
mdCATH steps exceed MISATO's only because frames are ~1 ns vs ~80 ps apart.
