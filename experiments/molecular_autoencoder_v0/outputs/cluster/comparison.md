Matched cohort: bb(cohort), mean-shape and PCA are all the 20-residue/80-atom cohort.
bb(full) is the full-length backbone (NOT comparable to the baselines).

| config | bb RMSD (cohort) | bb RMSD (full) | contact F1 | mean-shape | PCA k8 | verdict |
|---|---|---|---|---|---|---|
| recipe_proteinae | 4.826 | 5.013 | 0.438 | 5.067 | 1.987 | beats mean-shape |
| arch_ab_perresidue | 6.181 | 9.728 | 0.296 | 5.067 | 1.987 | loses to mean-shape |
| arch_ab_invariant | 6.931 | 10.170 | 0.267 | 5.067 | 1.987 | loses to mean-shape |
| arch_ab_baseline | 7.137 | 10.473 | 0.262 | 5.067 | 1.987 | loses to mean-shape |
| arch_ab_rope | 8.132 | 11.157 | 0.236 | 5.067 | 1.987 | loses to mean-shape |
