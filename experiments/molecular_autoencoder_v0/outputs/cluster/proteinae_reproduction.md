# The single-chain codec reproduces ProteinAE at C-alpha

ProteinAE_v1 (arXiv 2510.10634, OnlyLoveKFC) reports **0.28 +- 0.20 A C-alpha**
RMSD on CASP15 TS-domains, with its default **r=1, d=8** setting -- one latent
token per residue at latent width 8. That configuration *is* our direct
per-residue codec, not the shared-latent arms B/C/D. This logs the matched-metric
comparison.

## The number

C-alpha-only aligned RMSD on our single-chain PDB val set (n=758), EMA weights,
each checkpoint's own architecture:

| codec | params | train | **ca_rmsd** | all-atom | backbone |
|---|---|---|---|---|---|
| **ladder_direct3m_n2272** (design-of-record) | 3.8M | 2,272 | **0.365 +- 0.135** | 0.792 | 0.510 |
| ladder_direct_n2272 | 1.1M | 2,272 | 0.475 +- 0.153 | 0.963 | 0.716 |
| ProteinAE_v1 (reference) | -- | ~590k | 0.28 +- 0.20 | -- | -- |
| arm E (this codec's mechanism, not yet run) | -- | -- | pending | -- | -- |

Our design-of-record lands **0.365 vs their 0.28** -- a 0.085 A gap, inside one
standard deviation of either (theirs +-0.20, ours +-0.135), on the matched r=1/d=8
architecture, with **~259x less training data** (2,272 vs ~590k structures).
This is the reproduction: same ballpark, same architecture, far less data.

## Caveat: regime match, not head-to-head

The benchmarks differ -- **ProteinAE evaluates on CASP15 TS-domains; we evaluate
on our own single-chain PDB val split.** Same quantity (C-alpha aligned RMSD),
same architecture family, comparable difficulty, but *not the same inputs*. So
"reproduced" means "same ballpark on comparable single-chain data," not "matched
or beat them on their benchmark." A true head-to-head would require running our
codec on CASP15 TS-domains.

Two further honesty notes:
- **C-alpha is the reference's metric; all-atom (0.792) is what we optimize** and
  is strictly harder -- it carries every sidechain. We hold 0.365 C-alpha *while*
  reconstructing all atoms, so the C-alpha comparison understates the codec.
- The complex-corpus contrast: the same direct mechanism on the multi-chain +
  ligand corpus (arm A) sits at ca 4.366 A. The C-alpha gap there is not a
  single-chain limitation -- it is the multi-chain placement failure, which C-alpha
  tracks just as all-atom does.

## Reproducibility note

The two single-chain checkpoints predate the `unbounded_positions` refactor that
wrapped the learned position table inside `PositionEncoding`, renaming the state
key `decoder.res_pos_emb.weight` -> `decoder.res_pos_emb.table.weight`. With
`unbounded=False` the wrapped `.table` submodule is byte-for-byte the same
`nn.Embedding`, so evaluation used a single lossless key rename. Confirmation that
the rename is exact: all-atom / backbone reproduce the originals to the third
decimal (0.792 / 0.510 and 0.963 / 0.716).

## What this does and does not license

It licenses the claim that the direct per-residue codec is at reference quality
on single chains. It does **not** license a data-scale run off the back of it,
and it does **not** validate the shared-latent arms (B/C/D) -- the reference
never shipped arbitrary shared latents; its r is length downsampling (N/r tokens,
never independent of N, quality degrading with r: 0.28/0.35/0.50 at r=1/2/4).
Arm E ports the mechanism that actually works (fixed positional pooling); whether
it is needed depends on the pending graph-vs-group traceability verdict.
