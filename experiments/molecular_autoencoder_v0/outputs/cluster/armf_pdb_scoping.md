# PDB pretraining corpus -- scoping report (report only, no download)

Current RCSB holdings (~Aug 2026): 257,571 experimental structures (X-ray 206,039,
EM 35,940, NMR 14,807). Main archive is **CC0 1.0 (public domain)** -- coordinates and
RCSB API outputs, commercial use allowed. Derived sets differ (PDBbind is restricted).

## Source 1 -- crystallographic B-factors  [HIGHEST-YIELD, LOWEST-CAVEAT]
- **~206,039 X-ray entries**, per-atom `_atom_site.B_iso_or_equiv` on essentially every
  atom -- order **1e8 labeled atoms**. Largest per-atom label in structural biology.
- Access: mmCIF bulk rsync/HTTPS from files.rcsb.org / PDBe / PDBj; RCSB Data+Search API
  for metadata (resolution/refinement, needed for normalisation). No pairing needed.
- Caveat: B conflates true fluctuation with crystal packing, refinement, resolution,
  static disorder; absolute scale not comparable -> **per-structure z-normalisation
  mandatory** (else the model learns resolution). Anisotropic ADPs are rare (~2-3% of
  X-ray, sub-1.5A only); per-atom directional fluctuation is only broadly available as
  scalar isotropic B. ANM/GNM-vs-B validation lit: Bahar 1997 (GNM), Atilgan 2001 (ANM),
  Eyal 2006 (corr ~0.5-0.6).

## Source 2 -- NMR multi-model ensembles
- **~14,807 entries**, ~20 models each (~300k conformers -> ~15k RMSF-labeled structures).
- Signal: per-atom RMSF across superposed models. Caveat: spread reflects restraint
  sparsity as much as real dynamics (not Boltzmann-weighted); skewed to small solubles.

## Source 3 -- apo/holo pairs (matched displacement)
- Clean curated: CryptoSite **93**, CryptoBench **1,107** pairs. Large automated: AHoJ-DB
  ~4.68M quality-filtered pairs (noisy). PDBbind 14,108 holo (RESTRICTED license, not CC0).
- Pair by UniProt; signal = per-residue apo->holo displacement. Caveat: contaminated by
  crystal-form/packing/construct differences; needs packing-aware filtering. Clean yield
  ~1e2-1e3; noisy yield millions.

## Bottom line
**B-factors are the pretraining signal:** ~200k structures, ~1e8 atoms, pure CC0,
trivially extractable, single well-understood confound (scale, fixed by per-structure
normalisation), mature ANM validation to benchmark against. Recipe: pretrain per-atom on
normalised B-factors (~200k), fine-tune collective-motion heads on NMR RMSF (~15k) +
curated apo/holo (~1k). Scoping is CLEAN -- the "build if clean" condition is met.
