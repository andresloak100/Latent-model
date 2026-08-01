# complex_d8: first protein-COMPLEX + LIGAND codec (RESULT: capacity-limited, representation sound)

First codec run on multi-chain assemblies + non-protein groups, after fixing the
data representation (disulfides via struct_conn, covalent anchoring incl. glycan
trees, duplicate-ligand collapse, valence <=4). 1.1M params, lr 3e-4, batch 8,
max_atoms 3000, 900 epochs. Dataset: 742 X-ray complexes (556 train / 186 val),
537 multi-chain, 322 ligand-bearing (566 genuine-ligand groups), 762 disulfides.
Reference (single-chain protein <=800 atoms): 0.79 aa / 0.51 bb / 0.0002 chir / 0.963 F1.

## (a) Overall held-out
aa 5.584 | bb 5.529 | chir 0.0071 | bond 0.441 | contact F1 0.601  (centroid baseline 17.16)
final TRAIN quick-rmsd 4.80 -> val/train gap 1.16x => UNDER-CAPACITY, not overfitting
(the model cannot fit even the training complexes; bb ~= aa, a uniform failure not a sidechain one)

## (c) RMSD vs structure size -- the dominant effect
| atoms | n | aa RMSD |
|---|---|---|
| <=600 | 12 | 2.26 |
| 600-1000 | 32 | 3.31 |
| 1000-1500 | 35 | 4.43 |
| 1500-2000 | 77 | 5.60 |
| 2000-3000 | 30 | 10.65 |
corr(n_atoms, aa_rmsd) = 0.53. Smallest complexes (~559 atoms) reach 2.3 A;
largest blow up to 10.6 A. The 0.79 A reference was measured at <=800 atoms;
this run spans up to 2546. Accuracy is lost with SCALE.

## (b) Per-category RMSD (atom-weighted, pooled over val)
| category | RMSD (A) | n_atoms |
|---|---|---|
| protein | 7.98 | 273,164 |
| ligand (covalent) | 8.28 | 462 |
| ligand (non-covalent) | 9.11 | 2,639 |
| modified residue | 9.99 | 1,519 |
The ligand ORDINAL-SLOT path (the untested one) is NOT a special failure:
covalently-anchored ligands (8.28) reconstruct about as well as protein (7.98),
so the anchoring work paid off. Free ligands (9.11) and modified residues (9.99)
are ~1-2 A harder, as expected. (Atom-weighted pooling upweights large/worse
structures, so these exceed the 5.58 per-structure mean; the RELATIVE ordering is
the valid read.)

## Verdict (SUPERSEDED -- see the correction below)
Capacity-limited, not representation- or ligand-broken. Data is correct (762
disulfides, anchors intact incl. glycan trees, max bond degree <=4, duplicates
collapsed); the ligand path works; but 1.1M params cannot hold accuracy on
1500-3000-atom complexes -- matching the earlier "1.1M under-capacity for <=3000"
finding. Next lever is model CAPACITY, not more data plumbing. The scaled
experimental dataset (queued behind MISATO) and a larger codec are the path.

## Correction: capacity was NOT the ceiling

Two controls have since run, and both came back negative:

| arm | change | held-out aa |
|---|---|---|
| complex_d8 | 1.1M params, latent_dim 8 | 5.584 |
| complex_d8_3m | 3.8M params (3.5x) | 5.804 |
| complex_d16 | latent_dim 16 (2x budget) | worse than d8 |

Three-and-a-half times the parameters made it slightly WORSE, and doubling the
latent budget did too. Neither is a null result that more of the same would
fix: the train RMSD does not drop either, so the model is not failing to
*store* the answer, it is failing to fit even what it has seen. Combined with
the earlier eliminations -- undertraining (4x the steps bought nothing) and
size (rmsd/rg flat at 0.29-0.38, so "degrades with size" was largely a
property of the metric) -- the "next lever is model CAPACITY" line above is
falsified. It read the 1.16x val/train gap as under-capacity when the same
evidence is equally consistent with an architecture that cannot express the
target at all.

Two candidates remain, and the per-category table above now reads differently
in their light:

  * **Data volume.** 556 training complexes, never controlled. The one arm not
    yet run, and now unblocked (see docs/CORPORA.md).
  * **The per-residue readout.** Every atom reads a slot addressed by
    `(res_pos, slot_idx)` from its own residue's latent. That is what made
    variable-size decoding work at all, but it gives an atom no path to
    anything outside its own residue -- which is precisely what placing one
    chain relative to another requires. The 22%-of-total assembly-placement
    error and the uniform bb ~= aa failure both sit where this predicts.

The parts of the verdict that survive: the data is correct, and the ligand
ordinal-slot path is not a special failure (covalent ligands 8.28 vs protein
7.98).
