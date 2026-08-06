# Capacity axis (L=1, DM sweep on mdCATH) -- DM=256 and DM=512 arms are VOID

## FVE vs DM at L=1, with the PCA-DM ceiling alongside (n=7 held-out)
| bucket | medN | DM=16 | DM=64 | DM=256 | DM=512 | PCA-DM ceiling 16 -> 512 |
|---|---|---|---|---|---|---|
| 700-1.2k | 1,083 | 0.032 | **-0.050** | -0.000 | -0.000 | 0.483 -> 0.970 |
| 1.2-2k | 1,739 | 0.051 | 0.031 | -0.000 | -0.000 | 0.530 -> 0.940 |
| 2-3.5k | 2,356 | 0.059 | **0.093** | 0.000 | 0.000 | 0.649 -> 0.958 |
| 3.5-8k | 4,823 | 0.029 | 0.033 | 0.000 | -0.000 | 0.493 -> 0.852 |

## GUARDS: DM=16 and DM=64 PASS; DM=256 and DM=512 FAIL
- DM=16: G1 cos 0.9238 PASS, G4 +0.091 -> -0.215 PASS, G6 PASS
- DM=64: G1 cos 0.9403 PASS, G4 +0.070 -> -0.113 PASS, G6 PASS
- **DM=256: G1 cos 1.0000 FAIL, G4 base -0.000 FAIL**
- **DM=512: G1 cos 1.0000 FAIL, G4 base -0.000 FAIL**
G1 = 1.0000 means the latent is IDENTICAL across frames: both wide arms collapsed to a constant
output. **They are VOID -- not measurements of 256/512-dimensional capacity.**

## Consequences
- **The DM=512 rank-usage caveat is MOOT.** The 10/28 void-domain issue would only matter if there
  were a number to qualify; the arm never trained.
- **The curve shape cannot be read.** From the clean points only (DM=16, 64) it is essentially flat:
  0.032->-0.050, 0.051->0.031, 0.059->0.093, 0.029->0.033 -- one bucket improves, one degrades, two
  flat. Against ceilings of 0.48-0.82 the model realises **<=0.09 FVE at best**. This does NOT decide
  whether 512 is a sufficient width or a floor; **the arm never became competent enough to ask.**
- **Signed gap** is negative everywhere (-0.175 to -0.742); **fraction-positive is 0/7 at every DM.**
  The model never beats PCA on mdCATH -- expected, since PCA here has 2,000 well-conditioned train
  frames rather than MISATO's overfitting 80.

## Diagnosis
21 training domains against a 512-wide FiLM-conditioned decoder: wider code, more parameters, same
tiny corpus -> collapse to the constant solution. The competence gate worked as designed (it kept the
dead arms from early-stopping below 0.05, so they ran to plateau and were then caught by G1/G4), but
no gate can make 21 domains train a 512-dimensional code.
