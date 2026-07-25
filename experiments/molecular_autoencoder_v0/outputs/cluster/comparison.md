Matched cohort: bb(cohort), mean-shape and PCA are all the 20-residue/80-atom cohort.
bb(full) is the full-length backbone (NOT comparable to the baselines).

| config | bb RMSD (cohort) | bb RMSD (full) | contact F1 | mean-shape | PCA k8 | verdict |
|---|---|---|---|---|---|---|
| grid_direct_d16_reco | 0.681 | 0.740 | 0.937 | 5.067 | 1.987 | BEATS PCA |
| grid_direct_d8_reco | 0.693 | 0.754 | 0.943 | 5.067 | 1.987 | BEATS PCA |
| grid_direct_d4_reco | 0.707 | 0.750 | 0.938 | 5.067 | 1.987 | BEATS PCA |
| grid_direct_d2_reco | 1.405 | 1.531 | 0.826 | 5.067 | 1.987 | BEATS PCA |
| grid_direct_d1_reco | 4.069 | 6.236 | 0.440 | 5.067 | 1.987 | beats mean-shape |
| recipe_proteinae | 4.826 | 5.013 | 0.438 | 5.067 | 1.987 | beats mean-shape |
| grid_direct_d8_flow | 4.957 | 5.798 | 0.386 | 5.067 | 1.987 | beats mean-shape |
| arch_ab_perresidue | 6.181 | 9.728 | 0.296 | 5.067 | 1.987 | loses to mean-shape |
| grid_perresidue_d8_reco | 6.187 | 9.498 | 0.292 | 5.067 | 1.987 | loses to mean-shape |
| grid_perresidue_d4_reco | 6.420 | 9.942 | 0.283 | 5.067 | 1.987 | loses to mean-shape |
| arch_ab_invariant | 6.931 | 10.170 | 0.267 | 5.067 | 1.987 | loses to mean-shape |
| arch_ab_baseline | 7.137 | 10.473 | 0.262 | 5.067 | 1.987 | loses to mean-shape |
| arch_ab_rope | 8.132 | 11.157 | 0.236 | 5.067 | 1.987 | loses to mean-shape |
| grid_direct_d4_flow | 11.614 | 11.865 | 0.179 | 5.067 | 1.987 | loses to mean-shape |
