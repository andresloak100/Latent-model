# Pre-registered readings — committed before any cell finishes

BRIEF.md requires the interpretation to be fixed before results exist. This file is that commitment.
**No results existed in this repository when it was committed.**

## The four readings, from BRIEF.md, restated as the rule I will apply

| outcome | reading |
|---|---|
| `alpha` steep, `L_inf` low, neither axis saturated | generic scaling works here; the next investment is compute |
| saturates in **latent size** below the top of the range | latent capacity is not the bottleneck; widening cannot help |
| saturates in **parameter count** | the architecture is the limit; a different generic architecture is next |
| **flat on both axes** | generic scaling does not work at this data scale. A real result, reported as one. |

## Choices this brief leaves open, recorded at the point of making them

BRIEF.md §"If something here is underspecified, choose the most standard option and record the
choice." Each of these is a choice, not a finding.

1. **Token = one residue**, per the granularity axis (§3). Per-residue tokens carry a fixed-size
   coordinate vector; the 2- and 4-residue settings concatenate adjacent tokens. This is the axis the
   brief names, so residue boundaries enter as *tokenisation*, not as an architectural prior.
2. **Context window fixed across the sweep** so parameter count and latent size are the only things
   varying. Structures shorter than the window are excluded rather than padded, so no mask logic can
   change what the metric means.
3. **Coordinates centred, not aligned.** Translation is a gauge; rotation is not removed, so a model
   that wants a canonical frame must learn one. This makes the task harder than the usual literature
   setup and is stated because it affects every absolute number.
4. **MSE per coordinate is the loss and the primary metric**, per §"Metrics — generic only". RMSD is
   reported only as a readability aid and never as the headline.
5. **Homology split by sequence identity** (§Data). The threshold and the resulting train/held-out
   counts will be reported. A random split is not acceptable here and is not used.

## Two baselines, reported beside every cell

- `CENTROID` — predict the centroid for every token. The zero-information floor.
- `MEAN_SHAPE` — predict the training mean position per token index.

**A measurement already exists for these and is recorded now because it is a property of the task,
not of any model:** on a 3,000-structure random-crop shard drawn from the AlphaFold DB corpus,
`CENTROID` gives 12.407 A and `MEAN_SHAPE` gives 12.402 A median RMSD. **They agree to within
0.005 A**, because with structures in arbitrary orientations a per-index mean averages over rotations
and collapses to the centroid. So `MEAN_SHAPE` carries no information under choice 3 above, and the
bar is a single figure. Any cell not clearly below it has learned nothing structure-specific.

## Scope note

`model.py` and `data_prep.py` in this directory were written against an earlier, self-authored spec
before BRIEF.md was located, and they **do not yet meet it**: the split is random rather than
homology-based, tokens are atoms rather than residues, and the sweep points are not the brief's.
They are kept as a starting point and every one of those gaps is open work, listed here rather than
in a commit message so it cannot be mistaken for done.
