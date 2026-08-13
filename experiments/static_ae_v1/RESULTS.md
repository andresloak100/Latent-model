# Results — measured numbers only

Nothing here is interpreted beyond what SPEC.md pre-registered.

## Setup as run

    corpus       AlphaFold DB mmCIF, 979,051 files
    crop         256 contiguous atoms, one crop per structure, start drawn once at seed 0
    features     element symbol token + atom index; coordinates in Angstroms, centroid subtracted
    NOT applied  rotation alignment, superposition, internal coordinates, bonds, residues,
                 equivariance, any chemistry loss term
    metric       held-out median RMSD in Angstroms, no superposition

Parser check on 200 files: 200/200 parsed at 65 files/s; atoms per structure min 300, median 2,459,
max 9,722; elements present are C, N, O, S only. Centroids are **not** at the origin in the source
files, so centring is a real preprocessing step rather than a no-op.

## Baselines

Measured on a 3,000-structure smoke shard, 256 held out:

| baseline | median RMSD |
|---|---|
| CENTROID (predict the crop centroid for every atom) | 12.407 Å |
| MEAN_SHAPE (predict the training mean position per atom index) | 12.402 Å |

**These are the same number to within 0.005 Å, and that is a consequence of refusing alignment.**
With structures in arbitrary orientations, averaging position-by-index across structures averages
over rotations and collapses to the centroid. So MEAN_SHAPE carries no information here and the bar
is a single ~12.4 Å figure. Recorded before any model result.

## Scaling ladder

Pending: `10361485` (shard, 200,000 structures) -> `10361486` (ladder, 9 configurations).
