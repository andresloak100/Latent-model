# Milestone 1 — static structure autoencoder, and whether it scales

**This directory is clean-room. It imports nothing from any other experiment and carries no
conclusions from one.** What follows is the task and the measurement, not a summary of what anyone
expects to happen.

## The hypothesis under test

> Can a generic scalable learner discover a useful compressed representation of atomic structure
> without being told molecular physics?

This is a **falsification experiment**, not a product build. The result that matters is the *shape*
of the scaling curve, not the best achievable RMSD. If reconstruction improves predictably with
parameters and with latent capacity, milestone 2 (latent diffusion over the learned representation)
is justified. If a persistent wall appears that more parameters and more latent capacity do not move,
that is evidence the underlying idea needs to change rather than be patched.

## The task

    input   an atomic structure: per atom, an element symbol and an (x, y, z) coordinate
    model   a generic transformer encoder -> a single latent vector of size L -> a decoder
    output  the same coordinates, reconstructed

## What is deliberately NOT allowed

The point is to test a *general* architecture, so none of the following may be used:

- SE(3)- or E(3)-equivariant layers, frames, or invariant featurisation
- hand-designed internal coordinates (bond lengths, angles, torsions, distance matrices as input)
- chemistry-specific loss terms: energy functions, bond/angle/clash penalties, Ramachandran terms
- protein-specific structure: residue identity, backbone/side-chain distinction, chain breaks,
  secondary structure, contact maps, MSAs, templates
- superposition, Kabsch alignment, or any canonical-frame construction as preprocessing
- any auxiliary loss beyond coordinate reconstruction

Anything on that list would be "manually encoding what we know about molecules", which is the thing
being tested against.

## What IS used, and why each is data rather than a prior

| choice | justification |
|---|---|
| element symbol as a token id | the raw observable identifying an atom, like a token id in text |
| learned positional embedding over atom index | standard sequence-model positioning, not chemistry |
| fixed context window of `NA` atoms, random contiguous crop | the LM context-window convention |
| subtract the crop centroid | translation is a trivial gauge; nothing is rotated or aligned |
| coordinates in Angstroms, unscaled | no learned or hand-set normalisation of geometry |

**Rotation is NOT removed.** Structures are used in whatever orientation the file provides. A model
that wants an aligned frame has to learn one. This is the honest form of the question and it is
recorded here because it makes the task strictly harder than the usual literature setup.

**The decoder is given each atom's element and index**, and predicts only coordinates. The geometry
is what passes through the bottleneck; the atom inventory is not being compressed.

## Measurement

- **RMSD** in Angstroms on **held-out structures**: `sqrt(mean over atoms of squared distance)`,
  computed without superposition, in the same frame the input was given.
- Reported per configuration, on a fixed held-out set that no configuration trains on.
- Two baselines are reported beside every model, computed on the identical held-out crops:
  - `CENTROID` — predict the crop centroid for every atom. The zero-information floor.
  - `MEAN_SHAPE` — predict the training-set mean position for each atom *index*. What is achievable
    with no per-structure information at all.
  A model that does not clearly beat `MEAN_SHAPE` has learned nothing structure-specific, whatever
  its RMSD looks like in isolation.

## The scaling axes

1. **parameters** — width and depth together, at fixed latent size
2. **latent size L** — at fixed parameters

Both sweeps pass through a shared centre configuration so the two curves are commensurable. Every
configuration sees the **same number of training tokens** and the same data order, so a difference
between configurations is a difference in capacity and not in budget.

## Success and failure, stated before the runs

- **Clean scaling**: RMSD falls monotonically along both axes, and the parameter sweep does not
  saturate before the largest size tested. -> proceed to milestone 2.
- **Wall**: RMSD flattens along *both* axes while remaining far above the `MEAN_SHAPE` baseline's
  distance to the data. -> the generic architecture does not extract the structure, and the idea
  needs changing rather than patching.
- **Latent-bound**: parameters do nothing, latent size does everything. -> the bottleneck is the
  binding constraint and the capacity question is about representation size, not model size.
- **Parameter-bound**: the mirror image.

These four are written down now so the outcome is read rather than chosen.

## Layout

    data_prep.py   AFDB mmCIF -> a coordinate shard. Parsing only; no featurisation.
    model.py       the generic transformer autoencoder
    train.py       one configuration: train, evaluate, append a row to results.json
    ladder.py      the two sweeps, run sequentially in one job
    RESULTS.md     measured numbers only
