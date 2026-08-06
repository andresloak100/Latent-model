# RETRACTION: the graph-codec "win" over ANM does not survive a correct comparison

## What was claimed
"The learned graph codec beats Cartesian ANM on ligands" -- held-out RMSD **0.64 A vs 0.74 A at
L=8**, cited as the project's evidence that a learned codec beats a zero-shot structural baseline.

## What was actually done -- the baseline came from a DIFFERENT MOLECULE SET
`armf_graph_codec.py:239` prints `cANM 0.77/0.74/0.73` as a **HARDCODED STRING**. Line 228 shows only
the codec's own RMSD ever entered the aggregation. **The ANM baseline was never computed on the graph
codec's 48 held-out ligands.** Those numbers came from `armf_torsion_ceilings.py`, which ran on
**12 ligands** -- a different, much smaller sample. This is worse than an unswept hyperparameter: the
two sides of the comparison were measured on different molecules.

## Correct comparison, same 190-ligand cache, same i%4 split, same protocol
Held-out RMSD (A), lower is better, n=48:
| L | graph codec | ANM swept (cutoff) | bond-graph ENM (w13) | ANM-8 continuity | verdict |
|---|---|---|---|---|---|
| 4 | 0.800 | **0.594** (6 A) | 1.294 (0.0) | 0.599 | **CODEC LOSES** |
| 8 | 0.640 | **0.571** (5 A) | 1.501 (0.1) | 0.579 | **CODEC LOSES** |
| 16 | **0.440** | 0.549 (8 A) | 1.486 (0.1) | 0.549 | codec wins |

**At L=8 the claimed comparison inverts: on the SAME molecules ANM-8 gives 0.579, which BEATS the
codec's 0.640.** The original 0.74 was an artifact of the 12-ligand sample. Sweeping the cutoff makes
the peer stronger still (ANM-5 = 0.571).

**The codec wins only at L=16** -- and Family B qualifies even that: 9/48 held-out molecules have
L=16 past 30% of their 3N-6 available modes (worst **76.2%**), so for those the comparison is
rank-bound rather than a clean capacity result.

## Cutoff sweep (training molecules only; held-out never seen)
| cutoff | train L=4 | train L=8 | train L=16 |
|---|---|---|---|
| 3 A | 1.1982 | 1.2342 | 1.2200 |
| 4 A | 0.6696 | 0.6513 | 0.6323 |
| 5 A | 0.6283 | **0.6017** | 0.5892 |
| 6 A | **0.6236** | 0.6039 | 0.5804 |
| 8 A | 0.6333 | 0.6063 | **0.5787** |
Selection on TRAINING data, applied unchanged to held-out. 3 A under-connects badly; 5-8 A is a broad
optimum. ANM-8 was NOT badly chosen for these small molecules -- unlike ANM-10 on proteins, where
ANM-5 beat it by +0.135 FVE. **The ligand error was the molecule set, not the cutoff.**

## Second peer: bond-graph ENM is WORSE, not better
1-2 bonds plus 1-3 neighbours, 1-3 weight swept on training data: held-out 1.29-1.50 A versus
0.55-0.59 for distance-cutoff ANM. **A distance cutoff is the better network at this scale**, which
was not the expected outcome -- the hypothesis that a bond graph suits small molecules better is
refuted. (MISATO stores no bond orders; bonds are perceived by distance <1.8 A, so the weighting is
TOPOLOGICAL rather than by bond order. Stated, not hidden.)

## Status
**The project has NO surviving peer-comparison win.** The ligand result is RETRACTED at L=4 and L=8
and stands only at L=16 with a rank-bound caveat on 9/48 molecules. The ATLAS run is now the first
place a peer comparison could be established rather than re-litigated.
