# Perceiver-direct sweep — hypothesis test (RESULT: NEGATIVE)

**Question.** A fixed-size latent (L x 8 floats, independent of structure size)
read by *per-atom cross-attention* previously scored 9.5-11 A — worse than a
zero-parameter mean-shape baseline. The fix under test: route the readout through
**group tokens** (Perceiver encoder atoms->L latents, O(N.L); group-mediated
direct readout decoder). Does group routing rescue the fixed-size latent?

**Answer: no.** All 8 arms land at 9.2-10.5 A held-out all-atom — the same
failure regime, barely better than the 11.9 A mean-shape baseline and ~9x worse
than the per-residue direct codec (1.02 A). Hypothesis REFUTED.

## Setup
- encoder_type `perceiver_direct`, <=800-atom set (data/processed, splits.json),
  matched to grid_direct_d8_reco: 900 ep, batch 16, d_model 128, latent_dim 8,
  enc/dec self-attn 2, augment_rotation on, AMP.
- Dial: L in {16,64,256}; a = # all-atom self-attention layers in {0,1} (the
  O(N^2) knob; verified pre-flight to emit exactly `a` NxN attention calls).
- lr: 6 arms at 3e-4 (compression dial); 2 L64 arms at 1e-3 (lr-matched to the
  reference, which was trained at 1e-3). All 8 COMPLETED, ~2h each, clean exit.
- Params: a0 = 1.006M, a1 = 1.205M (the one atom self-attn layer adds ~198k).

## Full results — TRAIN next to HELD-OUT (all-atom RMSD, A)

| arm | lr | L | a | comp | **train aa** | **val aa** | val bb | chir | bond | cF1 | clash/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| perceiver_L16_a0  | 3e-4 | 16  | 0 | 14.2 | 2.49 | 9.53 | 8.98 | 0.142 | 0.598 | 0.333 | 882 |
| perceiver_L16_a1  | 3e-4 | 16  | 1 | 14.2 | 2.38 | 9.68 | 9.14 | 0.157 | 0.602 | 0.324 | 978 |
| perceiver_L64_a0  | 3e-4 | 64  | 0 | 3.56 | 2.43 | 9.30 | 8.72 | 0.157 | 0.596 | 0.337 | 900 |
| perceiver_L64_a1  | 3e-4 | 64  | 1 | 3.56 | 2.34 | 9.58 | 9.03 | 0.151 | 0.579 | 0.338 | 937 |
| perceiver_L256_a0 | 3e-4 | 256 | 0 | 0.89 | 2.37 | 9.35 | 8.80 | 0.160 | 0.587 | 0.327 | 964 |
| perceiver_L256_a1 | 3e-4 | 256 | 1 | 0.89 | 2.35 | 9.19 | 8.67 | 0.142 | 0.571 | 0.334 | 1021 |
| perceiver_L64_a0_lr1e3 | **1e-3** | 64 | 0 | 3.56 | 2.82 | 9.82 | 9.33 | 0.049 | 0.610 | 0.315 | 1190 |
| perceiver_L64_a1_lr1e3 | **1e-3** | 64 | 1 | 3.56 | 9.65 | 10.47 | 9.99 | 0.002 | 0.417 | 0.215 | 1969 |
| **grid_direct_d8_reco (ref, lr1e3)** | 1e-3 | - | - | **2.9** | ~0.5 | **1.02** | **0.75** | **0.001** | - | ~0.95 | low |
| centroid / mean-shape baseline | - | - | - | - | - | 11.89 | - | - | - | - | - |

train aa = final quick-RMSD on train batches (all-atom, augmented); val = held-out eval.

## (a) Architecture verdict — lr-matched L64 vs grid_direct_d8_reco (both lr 1e-3, both ~3x compression)
- L64_a0_lr1e3: train 2.82 / **val 9.82** / bb 9.33 / cF1 0.315.
- L64_a1_lr1e3: train 9.65 / val 10.47 — **failed to optimize** (train ~= val ~=
  mean-shape). Its low chir 0.002 / bond 0.417 are NOT quality: they are the
  signature of a collapsed near-constant prediction. One atom self-attn layer
  destabilized training at 1e-3 even at 1.2M params.
- Reference: val **1.02** / bb 0.75 / chir 0.001, at 2.9x.

**Decisive loss: 9.82 vs 1.02 A at matched lr and matched compression.** The
architecture does not work.

## (b) Compression dial (lr 3e-4): latent width is NOT the bottleneck
| L | comp | a0 val aa | a1 val aa |
|---|---|---|---|
| 16  | 14.2x | 9.53 | 9.68 |
| 64  | 3.56x | 9.30 | 9.58 |
| 256 | 0.89x | 9.35 | 9.19 |

Val is **flat across a 16x swing in latent width** — accuracy does not improve as
capacity is added. L256 @ 0.89x is an **over-complete** latent (2048 floats > the
~1800 coords of a 600-atom structure) and still gives 9.35 A. If the fixed-size
bottleneck were the constraint, the over-complete latent would fix it. It does
not -> the **group-readout path**, not the latent width, is what fails. The
compression dial the fixed-size latent was supposed to buy is unusable because
nothing can be reconstructed from it.

## Why train-next-to-held-out matters here
The 3e-4 arms fit TRAIN only to ~2.4 A all-atom (vs the direct codec's ~0.5). The
model cannot fit even the training set to the codec's quality -> this is a
genuine architecture/optimization failure, **not an lr artifact and not a data
limit**. (Same diagnostic that earlier unmasked an "undertraining" scare as
fixable; here it confirms a real fit ceiling.) 3e-4 vs 1e-3 costs only ~0.065 A
at this 1.1M scale (capacity ladder) — negligible against the ~8 A gap.

## a0 vs a1
One all-atom self-attention layer (the O(N^2) term we were trying to avoid) buys
**no** accuracy at 3e-4 (L64 9.30 -> 9.58, no improvement; L256 9.35 -> 9.19,
within noise) and **destabilizes** at 1e-3 (a1 diverged). Local geometry is not
the missing ingredient — the failure is upstream of it.

## Bottom line
Perceiver + group-mediated direct readout stays at ~9-10 A regardless of latent
width (16->256), atom self-attention (0->1), or lr (3e-4/1e-3). It reproduces the
prior 9.5-11 A fixed-size-latent result — group-token routing did not change the
outcome. The **per-residue direct codec** (structure-sized latent, 2.9x, 0.75 A
backbone) remains the only working design; the size-independent compression dial
it lacks is not recoverable via this architecture.
