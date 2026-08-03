# ProteinAE (arXiv 2510.10634) — what it does, and what it says about our arms

Read because it is the closest published system to ours and was cited as
evidence that "a similar architecture works". It does work, and it is *not*
the architecture our shared-latent arms are testing. Three findings below are
load-bearing; the rest is context.

## What the architecture actually is

| | ProteinAE default | our direct codec |
|---|---|---|
| latent | one token **per residue** | one token per residue |
| latent dim | 8 | 8 |
| token count vs length | `N/r`, **never** independent of N | scales with N |

`r` is a *length downsampling ratio* and `r=1` — no length compression — is
the shipped default. `d` is the channel bottleneck (256 → 8). So the config
name `r1_d8` describes our arm A, not arms B/C/D.

Their own ablation degrades with `r`: **0.28 Å (r=1) → 0.35 (r=2) → 0.50
(r=4)** Cα on CASP15 TS-domains. Compression by token reduction costs
accuracy, and even at r=4 the count is proportional to length.

## Finding 1 — they tested our architecture, and named why it fails

**ProteinAE-Register** uses "learnable register tokens instead of bottleneck
compression", fixed in count and independent of protein length. That is arms
B/C/D.

Result: comparable to the bottleneck version *within the training length
distribution*, then

> "performance degrades dramatically for structures exceeding this training
> length limit"

with reconstruction "breaking down abruptly around residue 231" past the
256-residue training cap. Their explanation:

> "The success of registers as latent representations in vision tasks may be
> partly attributable to the fixed size of image tokens after patchification,
> a characteristic not present in native protein sequences of arbitrary
> length."

**Two consequences.** A fixed-L latent is a length-generalisation hazard, which
matters enormously for the 1M-atom objective — a model trained at 3k atoms
with fixed L has no reason to hold at 10⁶. But also: registers were
*competitive in-distribution*. Our grid trains and evaluates on the same
distribution, so arms B/C/D sitting at 9–10 Å against arm A's 4.95 Å is worse
than the reference says this design should be. That points at a defect in ours
rather than a verdict on the design — and the traceability instrumentation
located one: routing gini 0.001 → 0.90, distinct read patterns 0.894 → 0.57.

## Finding 2 — their decoder is a diffusion decoder with a coordinate scaffold

Ours emits `xyz` in one shot from atom identity plus the latent. Theirs takes
`(x_t, z, t)` — a **noisy structure being denoised**, the latent, and the
timestep — and predicts a velocity, iterated by an ODE sampler.

So their decoder always has a coordinate state to refine; ours must
hallucinate positions from identity alone. That is a large difference and the
most likely remaining explanation for the quality gap after data and scale.

Confirmed clean on the leak question: no information from the *clean* input
reaches the decoder outside the latent. The pair/attention bias is computed
from the **noisy** structure, which is the diffusion state, not a skip.

Note our own flow-matching attempt scored 11.87 Å — but that was on the
pre-`model_direct` architecture that failed at everything, so it is not
evidence against this.

## Finding 3 — the numbers are not comparable to ours without care

- **Backbone only** (Cα, N, C, O). Our headline is all-atom, which carries
  every sidechain and is strictly harder. Use `ca_rmsd` for comparison.
- **588,318 AFDB structures**, lengths 32–256, single chains. We used 2,272.
- **~20M (Base) / ~100M (Large) params**, token dim 256, 5 layers, 4×A100,
  10 epochs. Our codec is 1.1M–3.8M.

Measured reproduction on the matched architecture: **0.365 ± 0.135 Cα**
(3.8M params, 2,272 structures) against their **0.28 ± 0.20** — inside either
error bar, with 259× less data. Different benchmarks (their CASP15 TS-domains
vs our PDB single-chain val), so this is a regime match, not a head-to-head.

## What follows

1. Arms B/C/D are testing a design the reference tried and did not ship. Their
   in-distribution result should be *competitive*; ours is not, so read the
   traceability before the RMSD.
2. `N/r` strided pooling (arm E, `atom_bottleneck: seqpool`) is their answer to
   the length problem and preserves the atom address by construction.
3. The diffusion decoder is the largest untested lever we have.
4. Fixed-L latents carry a specific, published length-generalisation risk. Any
   fixed-L result must be validated *outside* the training length range before
   it is believed for the 1M-atom path.
