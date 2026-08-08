# Atom-to-atom reconstruction — direct per-residue codec

Generated 2026-08-08 11:16 · every number below carries the provenance of the run that produced it.

> **These are different arms and are reported at different scales.** They are in separate sections
> because they are not comparable; a number from one section does not describe the other.

## Arm: `complex`

| provenance | |
|---|---|
| encoder | `direct` (DirectAutoencoder) |
| latent | **per-residue (scales with residue count)** |
| splits | `data/splits_complex.json` |
| data | `data/processed_complex` |
| held-out pool | 186 structures |
| checkpoint | `final.pt` sha256 `ca197bea6e912acb` (2026-07-31T22:42:56) |
| code | git `67acdef5-dirty`, torch 2.13.0+cu130 |
| checkpoint shim | **1 key(s) remapped** to load under current code (`decoder.res_pos_emb.weight -> decoder.res_pos_emb.table.weight`); exact for structures within max_positions=1024, which is asserted per structure |

| structure | atoms | res | all-atom Å | backbone Å | chirality | clashes/1k | contact F1 | latent floats | ×compression |
|---|---|---|---|---|---|---|---|---|---|
| `1BYZ` | 408 | 52 | **2.23** | 2.17 | 0.0000 | 1,906.9 | 0.959 | 416 | 2.94× |
| `3DS4` | 1281 | 164 | **6.06** | 6.03 | 0.0000 | 1,831.4 | 0.482 | 1312 | 2.93× |
| `8HJY` | 1821 | 233 | **7.47** | 7.46 | 0.0000 | 2,298.7 | 0.286 | 1864 | 2.93× |
| `5S3D` | 2529 | 334 | **15.54** | 15.43 | 0.0000 | 8,378.8 | 0.073 | 2672 | 2.84× |

### Headline — the full held-out set (n=186)

| | all-atom Å |
|---|---|
| mean | **5.88** |
| median | **2.49** |
| sd | 4.78 |
| quartiles | 2.26 / 2.49 / 10.14 |
| range | 1.91 – 21.57 |
| above 2 Å | 185 of 186 (99.5%) |

That set is **52–334 residues**. The headline describes structures of that size, not proteins in general.

The commonly quoted **5.88 Å is the mean**; the median is **2.49 Å**. The mean sits **2.4× the median**, so the distribution is **right-skewed** with a long tail and quoting the mean alone **overstates** the typical error. Quote both.


**Size regimes (49c).** One median over the whole set describes neither end of it:

| residues | n | median all-atom Å | median clashes/1k |
|---|---|---|---|
| 50–100 | 20 | 2.22 | 1,740 |
| 100–150 | 35 | 2.29 | 1,744 |
| 150–200 | 34 | 5.60 | 1,954 |
| 200–300 | 89 | 6.08 | 2,132 |
| 300–334 | 8 | 15.46 | 6,538 |

Rolling median crosses **2 Å** at ≥52 residues and **5 Å** at ≥122 residues; Spearman(residues, RMSD) = **0.499**.

The 4 structures below are chosen to **span the size range**, not drawn at random, so their median (6.76 Å) is an illustration and not an estimate of the set's.

**Exclusion audit (47b):** 0 structure(s) skipped for exceeding `max_positions`. The exclusion is empty, so it cannot bias these numbers.

Latent is **per-residue (scales with residue count)** — 8.0 floats per residue, so it grows with the structure. This is **not** the fixed-size codec.

## Arm: `single-chain`

| provenance | |
|---|---|
| encoder | `direct` (DirectAutoencoder) |
| latent | **per-residue (scales with residue count)** |
| splits | `data/splits_small_n2272.json` |
| data | `data/processed_small` |
| held-out pool | 758 structures |
| checkpoint | `final.pt` sha256 `662cf35e8a1b5a17` (2026-07-30T19:10:01) |
| code | git `67acdef5-dirty`, torch 2.13.0+cu130 |
| checkpoint shim | **1 key(s) remapped** to load under current code (`decoder.res_pos_emb.weight -> decoder.res_pos_emb.table.weight`); exact for structures within max_positions=1024, which is asserted per structure |
| vocabulary | **3 embedding(s) extended** since training; new rows are randomly initialised, so any structure indexing them is refused rather than reported |

| structure | atoms | res | all-atom Å | backbone Å | chirality | clashes/1k | contact F1 | latent floats | ×compression |
|---|---|---|---|---|---|---|---|---|---|
| `1O06` | 157 | 20 | **1.26** | 0.91 | 0.0000 | 38.2 | 0.889 | 160 | 2.94× |
| `2PPX` | 493 | 61 | **0.97** | 0.65 | 0.0000 | 77.1 | 0.971 | 488 | 3.03× |
| `2QKU` | 635 | 83 | **0.91** | 0.61 | 0.0000 | 12.6 | 0.957 | 664 | 2.87× |
| `2DYJ` | 734 | 91 | **0.99** | 0.66 | 0.0000 | 17.7 | 0.944 | 728 | 3.02× |
| `8ZXJ` | 799 | 96 | **0.64** | 0.33 | 0.0000 | 5.0 | 0.977 | 768 | 3.12× |

### Headline — the full held-out set (n=758)

| | all-atom Å |
|---|---|
| mean | **0.79** |
| median | **0.84** |
| sd | 0.21 |
| quartiles | 0.66 / 0.84 / 0.93 |
| range | 0.27 – 2.38 |
| above 2 Å | 1 of 758 (0.1%) |

That set is **20–109 residues**. The headline describes structures of that size, not proteins in general.

The commonly quoted **0.79 Å is the mean**; the median is **0.84 Å**. The mean sits **below** the median, so the distribution is **left-skewed** and quoting the mean alone **understates** the typical error. Quote both.


**Size regimes (49c).** One median over the whole set describes neither end of it:

| residues | n | median all-atom Å | median clashes/1k |
|---|---|---|---|
| 0–50 | 57 | 0.89 | 20 |
| 50–100 | 613 | 0.84 | 12 |
| 100–109 | 88 | 0.77 | 9 |

Rolling median crosses **2 Å** at ≥None residues and **5 Å** at ≥None residues; Spearman(residues, RMSD) = **0.011**.

The 5 structures below are chosen to **span the size range**, not drawn at random, so their median (0.97 Å) is an illustration and not an estimate of the set's.

**Exclusion audit (47b):** 0 structure(s) skipped for exceeding `max_positions`. The exclusion is empty, so it cannot bias these numbers.

Latent is **per-residue (scales with residue count)** — 8.0 floats per residue, so it grows with the structure. This is **not** the fixed-size codec.

---

## Conformers (NMR ensembles)

Source `nmr_suitability_full.json` — **11 entries, 126 conformers**.

| | |
|---|---|
| true conformational spread | **3.544 Å** |
| spread after the round trip | **3.371 Å** |
| ratio | **0.932** — 6.8% of the spread is lost |
| mean reconstruction error | 1.164 Å |

Conformers **do not collapse**: 6.8% of the spread is lost, not all of it. And the reconstruction error (1.16 Å) is **well below** the true spread (3.54 Å), so the codec resolves conformers rather than blurring them together.

The ratio is **sample-dependent** — it is 0.932 here and 0.868 in §5 on a different entry count — so it is quoted with its n, never alone.

---

## Scope

- The latent **scales with residue count**. This is not the fixed-size compression the project is about.
- No claim is made about **dynamics**. On the ATLAS line the fixed-size codec loses to zero-shot ANM on **100% of 123 systems**.
- Numbers from different arms appear in **separate sections at their own scales** and are not comparable.
- Both headline means hide their distributions: the single-chain mean sits *below* its median (left skew), the complex mean is *2.4x* its median (long right tail). Medians and ranges are given above; quote both.
