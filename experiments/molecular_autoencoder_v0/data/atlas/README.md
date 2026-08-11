# ATLAS split and index — versioned because the split is a DECISION, not data (INBOX 092c)

`armf_atlas_dm.py:61` reads `MAN = $WR/atlas_manifest.json`, and **every ATLAS number in this project
is conditioned on it** — the 0/123 peer result, the tied ladder, the DM sweep, 091's rate–distortion
peer test. Until now neither file existed anywhere in the repo, only on scratch. This is a borrowed
account: if scratch goes away, *"which 123 systems"* has no answer, and no result on this line is
reproducible.

**The raw data is deliberately NOT versioned** — 282 GB of ATLAS cache and 289 GB of AFDB CIFs, both
re-downloadable from their source. What is versioned is the kilobytes that cannot be regenerated
because they encode a choice.

| file | bytes | what it is |
|---|---|---|
| `atlas_manifest.json` | 12,750 | the train / held-out split: `heldout`, `train_ordered`, `n_train_curve` |
| `atlas_info.tsv` | 1,801,991 | the ATLAS index the selection was drawn from |

## How to regenerate the cache from these

```bash
export WR=/network/scratch/<u>/<user>/latent-model-workspace
cp data/atlas/atlas_manifest.json data/atlas/atlas_info.tsv $WR/
ATLAS_N=825 $WR/venv/bin/python scripts/armf_atlas_cache.py     # download -> subsample -> store -> delete archive
```

`armf_atlas_cache.py` reads `atlas_manifest.json` when it exists and re-fetches exactly
`heldout + train_ordered`, so the split survives a cache rebuild rather than being re-drawn.

## The two acquisition parameters, and why they are what they are

| parameter | value | why |
|---|---|---|
| `STRIDE` | **4** | 10,001 frames/replica → **2,501**. Frames must sit comfortably above the median rank90 (~168), because MISATO's PCA baseline overfits at 80 frames. Also keeps the DM=512 ceiling rank-valid. |
| `NSEL` (`ATLAS_N`) | **825** | systems selected from `atlas_info.tsv`; 841 landed in cache, 123 of them held out. |

**Frame spacing is 10 ps before striding, 40 ps after** — measured against the production `.mdp`
(nsteps 50e6 × dt 2 fs), not assumed. This matters beyond acquisition: 81c recorded that ATLAS is
**100 ns continuous per replica**, and the propagator's τ is applied in *frames* while labelled *ns*,
which coincide only on mdCATH. Anything reading τ off this cache must convert.

## Still owed

The same treatment for the AFDB corpus when `prep1m` lands: the 972,849 gated accession list, the
30% MMseqs2 identity threshold, and the query set (all 156 chains of the 125 held-out ATLAS entries).
That list is likewise a decision — it is what the pretrain is allowed to see.
