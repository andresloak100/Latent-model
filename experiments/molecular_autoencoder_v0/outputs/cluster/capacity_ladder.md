# Model-capacity sweep — is the ~0.9 A all-atom floor architectural or a dial?

Latent-size ladder (d2..d16) and the conformer-margin fine-tune both held model
capacity fixed at d_model=128 / 2+2 layers / 1.1M params. This sweep varies
MODEL SIZE (the untested axis). Same <=800 dataset, latent_dim=8, 900 epochs.

## Valid sweep (bigger arms at lr=3e-4; 1.1M at its tuned 1e-3)

| params | d_model/layers | lr | final train RMSD | held-out all-atom | held-out bb | chirality | bond err | contact F1 |
|---|---|---|---|---|---|---|---|---|
| 1.1M  | 128 / 2+2 | 1e-3 | 0.606 | 1.020 | 0.754 | 0.001 | 0.161 | 0.943 |
| 3.8M  | 256 / 2+2 | 3e-4 | 0.442 | 0.938 | 0.639 | 0.000 | 0.147 | 0.948 |
| 7.0M  | 256 / 4+4 | 3e-4 | 0.334 | 0.910 | 0.578 | 0.000 | 0.120 | 0.953 |
| 15.2M | 384 / 4+4 | 3e-4 | 0.300 | 0.873 | 0.548 | 0.000 | 0.102 | 0.956 |

Verdict: the floor is a CAPACITY DIAL, not architectural. Train RMSD and held-out
all-atom both fall monotonically with model size; every geometry metric improves.
The earlier "architectural limit" read was on the wrong axis (latent + objective,
at fixed 1.1M model capacity).

Caveats: gains decelerate (-0.082 / -0.028 / -0.037 per step in all-atom) and the
train->val gap widens (0.41 -> 0.57): a real dial with diminishing returns and
rising overfit, not clearly reaching the 0.1-0.5 A crystal-form scale as-is.

## Invalid contrast: lr=1e-3 (tuned for 1.1M) does NOT test capacity

| params | lr | final train RMSD | held-out all-atom |
|---|---|---|---|
| 3.8M  | 1e-3 | ~0.83 | 2.034 |
| 7.0M  | 1e-3 | ~1.06 | 1.558 |
| 15.2M | 1e-3 | NaN (diverged) | — |

Train RMSD RISES with model size at 1e-3 — a bigger model fitting train worse is an
optimization failure, not capacity. Reporting final train RMSD (not just held-out)
is what separated the lr artifact from the real capacity signal.
