# Complex codec: the add-more levers are exhausted; one architectural lever (mechanism pending)

Results only. The mechanism behind the one positive result is **confounded and
under test** (two runs in flight); the "information vs parameterization" reading
is deliberately withheld until they land.

## 1. Data does not help, over 8x (candidate #2, dead)

Composition-matched, leakage-controlled complex corpus (72.4% multi-chain /
43.4% ligand held flat across rungs; anchor = complex_d8's 556 via `--pin-train`;
near-duplicate leakage held constant at the baseline's own 163 via
exclude-then-match). Last-10 mean +- sd of held-out val RMSD (EMA off):

| n_train | val RMSD (mean +- sd) |
|---|---|
| 556  | 6.003 +- 0.304 |
| 1112 | 5.778 +- 0.344 |
| 2224 | 5.829 +- 0.271 |
| 4448 | 5.891 +- 0.787 |

Flat, if anything up at the top. The apparent descent by *final* checkpoint
(6.13 -> 5.38) was the final-checkpoint lottery; by mean it is flat inside the
noise. **8x more data does not help.** Bounds the data effect over the range we
can span (up to ~4,848 structures); does not rule out 10-100x, which is
untestable here as it was for the single-chain reference.

## 2. Model weaker AND corpus harder (the 2x2, closed)

| | reference val | complex val |
|---|---|---|
| reference codec | 0.878 | (not feedable) |
| complex_d8 | 2.352 (n=733) | 5.884 |

- Same structures, two models: complex_d8 2.352 vs reference 0.878 on identical
  reference-val structures (n=733, 25 leakage keys excluded) -> **the model is
  genuinely weaker** (2.7x, well-powered).
- Same model, two corpora: complex_d8 5.884 (complex val) vs 2.352 (reference
  val) -> **the corpus is also harder.**

Both real. Roughly ~1.5A model weakness + ~3.5A corpus difficulty. Not
"model not corpus" -- both.

## 3. The failure is placement, not internal geometry

Residue-granularity decomposition (per-residue independent Kabsch removes what
no rigid motion can fix = internal geometry; the rest is placement). The
Kabsch-DOF bias inflates the placement share, so the **degradation ratio**
(bias-robust, reference as internal control) is the honest statement:

|  | internal | placement |
|---|---|---|
| reference codec | 0.520 | 0.707 |
| complex_d8 | 1.829 | 5.593 |
| **degradation** | **3.5x** | **7.9x** |

Placement degrades 2.2x faster than internal. Corroborated, DOF-independent, by
the **bb/aa ratio on identical structures**: reference 0.463/0.666 = 0.70
(normal: backbone easier than sidechains), complex_d8 2.224/2.230 = 1.00 (the
rigid-displacement signature -- the residue ordering is gone).

## 4. STEP 1: frames null, chain-aware positive -- but the positive is confounded

Three arms, matched except one readout flag, all EMA 0.9998, 900 epochs (~63k
steps), same 556-structure corpus. Headline = last-10 mean +- sd of
`val_rmsd_ema`; residue decomposition on the EMA checkpoint alongside.

| arm | val_rmsd_ema | vs A | global | internal | placement | single | multi |
|---|---|---|---|---|---|---|---|
| A baseline | 5.769 +- 0.028 | -- | 5.744 | 1.870 | 3.874 | 4.189 | 6.055 |
| B frames | 5.734 +- 0.031 | -0.035 | 5.710 | 1.959 | 3.751 | 4.171 | 6.018 |
| C chain-aware | 4.904 +- 0.019 | **-0.865** | 4.943 | 1.872 | **3.070** | 3.779 | 5.175 |

**B (frames): a clean null.** Global -0.03, internal *up* (1.870 -> 1.959),
placement barely moved. Both frame heads trained to non-trivial weights
(trans_head L2 1.22, rot_head L2 2.49), so the readout *engaged* and still did
nothing -- the rigid-frame parameterization is not the bottleneck.

**C (chain-aware): -0.865A, real, works through placement** (3.874 -> 3.070,
internal untouched), multi-preferring (multi -0.88 vs single -0.41).

## The open question (mechanism WITHHELD)

`dec_chain_aware` bundled two changes:
- `chain_emb`: which chain a residue is in (+2,048 params)
- `res_in_chain_emb`: position within that chain (+131,072 params)

On a single chain, within-chain position == global position, so
`res_in_chain_emb` is an exact duplicate of a table the decoder already had --
pure positional capacity carrying zero chain information. **98.5% of C's added
parameters are the redundant position table**, and that -- not a chain marker --
is the obvious candidate for C's 0.41A gain on single chains, which have no
boundary to mark.

So C's -0.865A is real, but **it cannot yet be attributed to chain-marking vs
extra positional capacity.** Two runs decide it (in flight):
- `markeronly` (+2k, chain marker only) reproduces the multi gain, ~nothing on
  single -> chain-marking is the mechanism.
- `posonly` (+131k, extra positions only) reproduces both -> it is under-
  parameterized position encoding, and the chain story is dead.

Until then the "information vs parameterization" reading is unsupported and is
**not** recorded here.

## Bottom line (results)

Every add-more lever on the complex codec is negative -- parameters 3.5x,
latent 2x, steps 4x, structures 8x -- and none moved train RMSD. The failure is
placement, not internal geometry, and it is a weaker model on a harder corpus.
The first non-add-more experiment found frames null and a chain-aware readout
worth -0.865A -- the first real architectural movement in the arc -- but whether
that is a chain marker or positional capacity is undecided pending two runs.
