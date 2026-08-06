# RE-RUN: the graph-codec "win" over ANM does NOT survive. WITHDRAWN.

## Headline
Recomputed on the SAME 48 held-out ligands, with baselines swept on TRAINING molecules only:
| L | codec | ANM swept (cutoff) | bond-graph ENM (w13) | ANM-8 continuity | verdict |
|---|---|---|---|---|---|
| 4 | 0.800 | **0.594** (6 A) | 1.294 (0.0) | 0.599 | **CODEC LOSES to ANM-6** |
| 8 | 0.640 | **0.571** (5 A) | 1.501 (0.1) | 0.579 | **CODEC LOSES to ANM-5** |
| 16 | **0.440** | 0.549 (8 A) | 1.486 (0.1) | 0.549 | CODEC WINS |
(held-out RMSD in A, lower is better; 190 ligands, 142 train / 48 held-out, same i%4 split)

## THE ROOT ERROR WAS NOT THE UNSWEPT CUTOFF -- IT WAS A CROSS-DATASET COMPARISON
The recorded claim was "graph codec 0.64 vs Cartesian ANM 0.74 at L=8". **ANM was never computed on
the 190-ligand set.** `armf_graph_codec.py:239` prints `cANM 0.77/0.74/0.73` as a HARDCODED STRING
LITERAL; the token appears nowhere else in that file. Those numbers came from
`armf_intcoord_oracle.py`, which sampled **12 ligands** (`allk[::len(allk)//12][:12]`).
**A model evaluated on 48 held-out molecules was compared against a baseline measured on a different
12-molecule sample.** Recomputed on the correct 48, ANM-8 gives **0.599 / 0.579 / 0.549**, not
0.77 / 0.74 / 0.73 -- the baseline was ~0.16 A stronger than reported at L=8.

## The Family E hypothesis was WRONG here (and that is worth recording)
The re-run was commissioned on the theory that ANM-8 was handicapped by an unswept cutoff, as ANM-10
demonstrably was on proteins (ANM-5 beat it by +0.135 FVE). **On ligands the cutoff barely matters:**
train-selected optima give 0.594/0.571/0.549 against ANM-8's 0.599/0.579/0.549 -- a 0.005-0.008 A
difference, and at L=16 ANM-8 IS the optimum. Cutoff 3 A is the only bad choice (1.20-1.23, the
network fragments). So Family E was real for proteins and negligible for ligands; the ligand win fell
for a different and more basic reason.

## Second peer: the bond graph is the WRONG network at this scale
Bond-graph ENM (1-2 springs + 1-3 at a swept weight) gives 1.294-1.501 A -- **more than twice the
error** of distance-cutoff ANM. Best 1-3 weights were 0.0-0.1, i.e. the sweep pushed toward 1-2 only,
and it still lost badly. A distance cutoff is the right network for 8-90-atom molecules; adding this
peer does not rescue the codec, and it does not threaten ANM either.
(NOTE: MISATO stores no bond orders -- bonds are perceived by distance <1.8 A -- so weighting is
TOPOLOGICAL, 1-2 vs 1-3, not by bond order. Stated, not hidden.)

## FAMILY B, applied to this arm for the first time
| L | rank usage (median) | worst | molecules past 30% |
|---|---|---|---|
| 4 | 5.3% | 19.0% | 0/48 |
| 8 | 10.7% | 38.1% | 1/48 |
| 16 | 21.3% | **76.2%** | **9/48** |
**The single L where the codec still wins (L=16) is also where the metric is most bound-like** --
9 of 48 molecules have L past 30% of their 3N-6 available modes, one at 76%. That win is therefore
the weakest of the three, not the strongest.

## STATUS: WITHDRAWN
The codec beats a properly-computed, properly-swept zero-shot peer at **L=16 only**, on the arm where
Family B is worst. It **loses at L=4 and L=8**. This can no longer be cited as evidence that a
learned codec beats classical zero-shot decomposition on ligands.
