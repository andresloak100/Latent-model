# Does intrinsic dimensionality saturate with observation window? (objective 3, measured)

mdCATH, 28 domains, ALL-ATOM, 320K, 5 replicas concatenated (2,500 frames). rank90(T) on a dense
T grid (50...2400), PCA only, no training. Three models fitted in LINEAR space, selected by AIC:
saturating A*T/(K+T) | power a*T^c | logarithmic a + b*ln(T).
(Efficiency: all windows come from ONE frame Gram matrix via double-centred submatrices --
verified identical to direct SVD, rank90=75 both ways.)

## Classification
| model | domains |
|---|---|
| **saturating** | **21/28** |
| power | 6/28 |
| logarithmic | 1/28 |

- Saturating asymptote **A: median 168, range 13-500**; observed rank90 at T=2400 is **86% of A**.
- Power-law exponent **c: median 0.18, range 0.03-0.55** -- i.e. even the non-saturating domains
  grow very slowly.

## Read: favourable for objective 3, but the measurement cannot see the regime that matters
**What the data supports firmly:** growth DECELERATES strongly with observation window. Dimensionality
is not exploding with simulated time in the sampled regime.

**What it does NOT support:** a confident hard finite asymptote. Over a 48x window a saturating
curve and a c~0.18 power law look alike -- AIC preferring "saturating" for 21/28 does not sharply
exclude slow unbounded growth, since both describe decelerating growth. Quantified: at median
c=0.18, extending 2.5 us -> 1 ms (400x) multiplies rank90 by 400^0.18 ~ **2.9x**; at the worst
observed c=0.55 it is ~27x.

**The load-bearing caveat: 2,500 frames x ~1 ns = ~2.5 us.** This bounds the ns-us regime ONLY.
Millisecond dynamics are dominated by RARE BARRIER CROSSINGS that do not occur in 2.5 us, so the
saturation observed here may be **saturation within the sampled basin**. New basins at ms scale
would add dimensions DISCONTINUOUSLY, which no smooth extrapolation of this curve can predict.
**Objective 3 is therefore not settled by this result** -- it is bounded from below.

## Which domains keep climbing -- suggestive, NOT resolved
climbing n=7 vs saturating n=21 (Mann-Whitney U):

| property | climbing (med) | saturating (med) | p |
|---|---|---|---|
| mobility medRMSF | **6.36** | **2.46** | 0.189 |
| size N | 1168 | 1844 | 0.172 |
| helix fraction | 0.29 | 0.32 | 0.717 |
| sheet fraction | 0.23 | 0.28 | 0.915 |

**These p-values are UNDERPOWERED (n=7 vs 21) and must not be read as "no difference"** -- the same
error as the p=0.58 drop-rate null. The mobility direction is a 2.6x difference in medians and is
the physically coherent one: floppy domains have LOW absolute rank90 (variance concentrated in one
collective motion) but keep creeping up as they explore new conformations, while stable domains
reach a HIGH but finite thermal-fluctuation basis and saturate. Note this is exactly the population
relevant to the ms concern -- the domains that explore are the ones that do not saturate.
Direction suggestive, effect unresolved at this n.

## Reinforces the ordered-latent direction
Saturating asymptotes span **13-500 (38x)** across domains. A single fixed L is simultaneously
over- and under-provisioned across the corpus; see the ordered/nested latent entry in ROADMAP.

## Decisive follow-up (NOT run -- proposed)
mdCATH ships 5 temperatures to 450K, which accelerate barrier crossing and unfolding. Comparing
rank90(T) asymptotes at 320K vs 450K directly tests whether exploring MORE of the landscape raises
the asymptote: similar A -> saturation is robust to exploration; much larger A -> exploration adds
dimensions and the ms concern is real. Same data on disk, CPU-only, no download.
