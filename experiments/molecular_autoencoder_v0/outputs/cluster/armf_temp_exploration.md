# Does exploration add dimensions? A(T) across 5 temperatures, with a folding control (objective 3)

The 320K saturation result (21/28) could have been saturation WITHIN THE SAMPLED BASIN -- 2.5 us
cannot contain ms rare events. mdCATH's 5 temperatures accelerate barrier crossing, so A(T) as a
curve tests whether exploring more landscape raises the asymptote. 28 domains x 5 temperatures x 5
replicas, ALL-ATOM, common reference (320K rep0 frame0).

## Effective time (Arrhenius, t_eff = 2.5 us x k(T)/k(320K)); assumed Ea stated
| T | Ea=5 | Ea=10 | Ea=15 |
|---|---|---|---|
| 348K | 5 us | 9 us | 17 us |
| 379K | 9 us | 29 us | 98 us |
| 413K | 15 us | **86 us** | 507 us |
| 450K | 24 us | 235 us | 2.28 ms |

Only Ea=15 at 450K reaches ms -- and 450K is unusable (below). **Max usable effective time is
~86 us (Ea=10) at 413K**, ~35x the 320K window.

## Folding control -- MANDATORY, and it changed the answer
| T | Rg/Rg320 (med) | Q/Q320 (med) | DENATURED |
|---|---|---|---|
| 320K | 1.000 | 1.000 | 0/28 |
| 348K | 0.998 | 0.963 | 2/28 |
| 379K | 1.002 | 0.909 | 7/28 |
| 413K | 1.022 | 0.794 | **20/28** |
| 450K | 1.096 | **0.521** | **28/28** |

**At 450K every domain is denatured** (native contacts down to 52%). Pooling it would have
attributed unfolding to "exploration". Denatured points have LOW A (median 37-59) -- unfolding
COLLAPSES measured dimensionality (one dominant collective motion), consistent with rank90 ~
RMSF^-1.35.

## A(T) among FOLDED domains: FLAT (paired, survivorship-controlled)
Cross-temperature medians compare shrinking, increasingly thermostable subsamples (28->26->21->8),
so the load-bearing analysis is PAIRED within-domain -- each domain against ITS OWN 320K value:

| T | n folded | A(T)/A(320) | rank90@2400 ratio | stayed saturating |
|---|---|---|---|---|
| 348K | 26 | 0.985 | 0.985 | 14/19 |
| 379K | 21 | 0.830 | 0.905 | 8/16 |
| 413K | 8 | 0.949 | 1.082 | 4/7 |

Paired regression on log10(effective time), Ea=10, n=55 domain-temperature pairs:
- log10(A ratio): slope **-0.036 +/- 0.127**, R^2 0.006 -> **FLAT**
- log10(rank90@2400 ratio): slope **-0.012 +/- 0.121**, R^2 0.001 -> **FLAT**

**EQUIVALENCE BOUND (what makes this useful):** extrapolating the CI to 1 ms effective time (400x)
bounds the change in dimensionality to **[0.38x, 1.73x]** for A and **[0.45x, 1.92x]** for observed
rank90. Explosive growth is EXCLUDED over this range. The pre-registered read "A(T) flat, folded
throughout -> saturation robust to exploration, ms concern drops substantially" is the one that
fires.

## Two honest qualifications
1. **The saturating CLASSIFICATION does degrade with heating** (14/19, 8/16, 4/7 of saturating
   domains stay saturating). The curve SHAPE becomes less clearly saturating -- but the MAGNITUDE
   (A and observed rank90) does not rise, and in paired terms slightly falls. So this is a change in
   curve shape, not evidence of new dimensions. Reported because it was the pre-registered
   headline condition; it fires on shape but not on scale.
2. **rank90 is VARIANCE-WEIGHTED, and that is the real limit of this whole approach.** It counts
   directions carrying >=10% of variance. A rare, brief excursion into a NEW basin contributes
   little variance and would barely move rank90. So "flat rank90 under heating" establishes *no new
   HIGH-VARIANCE dimensions*, not *no new states*. The ms concern is REDUCED, not eliminated, and a
   metric sensitive to rarely-visited states (not variance-weighted PCA) would be needed to close it.

## Group split (kept flagged as suggestive-unresolved, per the underpowered-null rule)
climbing@320K (n=7) A ratio stays ~1 (34 -> 41 -> 39 -> 36 across T); saturating@320K (n=21) shows
no rise either. The 2.6x mobility gap between the groups remains suggestive and unresolved at this n.
