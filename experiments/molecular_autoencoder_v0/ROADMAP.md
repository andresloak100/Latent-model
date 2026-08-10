# Roadmap: capability gaps and technical debt

The system is a latent molecular-dynamics model: compress structures into a
compact latent, run diffusion there rather than in coordinate space, decode
back. Four stages — autoencoder → latent diffusion → mid-training on
high-quality trajectories → RL against quantum-accurate feedback.

**The four objectives it is being built to hit** (§6 audits the stack against
each, and the ordering of work follows from them):

1. Scale to **1M+ atoms**, general molecules and not only proteins.
2. Model **enzyme-like bond breaking and forming**.
3. Scale to **multi-millisecond** timescales.
4. Be efficient enough to **inference on a single GPU** — training compute is
   worth spending to buy inference efficiency.

This document records the distance between what the codebase does now and
those four targets. Nothing here is a reason not to proceed, and several items
are cheap; but a roadmap that named none of them would read as though the
objectives fall out of scaling the current design, and two of them do not.

---

## 1. Capability gaps

### 1.1 Design vs. modelling — the largest unstated leap
Everything built is a **forward** model: structure in, structure out. Machines
require the inverse — function to structure. Latent diffusion supplies a
generative prior, which is the right substrate for conditional design, but the
conditioning has never been specified. This is the biggest gap in the four-
stage plan and it is not addressed by any stage as currently written.

### 1.2 Interfaces, not global RMSD
Function lives in a binding interface of a few dozen atoms. A global all-atom
RMSD over a 2,500-atom complex can look excellent while the interface is
wrong. We have no interface-restricted metric.

*Partly mitigated:* `molae/pocket.py` crops to the ligand's neighbourhood,
which makes the interface the whole measurement instead of ~2% of it. Cropping
is not the same as an interface metric on the full complex.

### 1.3 Fixed covalent topology — bond forming/breaking is out of reach
Three specific blockers, in order of how hard they are to remove:

1. **The objective fights it.** Bond topology is fixed from the input frame
   (perceived, or read from `struct_conn`/prmtop) and then *restrained*. A
   trajectory in which a bond breaks is penalised by the bond-length loss.
2. **The atom set is fixed.** The decoder is conditioned on atom identity and
   emits into fixed slots. Atoms move; none appear or disappear. Reactions
   that change composition are outside the representation. Hydrogens are
   stripped, and that is where most reaction chemistry lives.
3. **No energetics.** Nothing in the model knows the barrier to breaking a
   bond. Classical restraints cannot supply it; that is quantum. Stage 4, and
   the least specified part of the plan.

Coordinates *do* implicitly encode topology — two atoms at 1.5 Å are bonded —
so the representation is not intrinsically barred from expressing bond
changes. Closing this needs: keep hydrogens, replace the fixed-topology
restraint with a predicted bond graph, and train on reactive trajectories
(AIMD/DFT) rather than classical MD. MISATO ships a QM component we deferred.

### 1.4 Assembly scale
Machines are 10⁴–10⁶ atoms. The latent is per-residue so it scales linearly,
but the clash loss is O(N²) (capped via `loss_max_atoms`) and reconstruction
quality currently degrades with size — see 3.1, where the cause is still open.

### 1.5 Non-canonical residue identity — MEASURED as the worst class
Non-canonical residues (SeMet, hydroxyproline, D-amino acids) are kept in
their chain with correct bonds, but collapse to a single `UNK` residue index
with ordinal slots. The model gets almost no chemical prior about *what* the
residue is.

Flagged as a known limitation when built; now measured. In `complex_d8`'s
four-way split the modified-residue class is worst at **9.99 Å** (protein
7.98, ligand-covalent 8.28, ligand-free 9.11), and the two worst structures
overall are small and modified-residue-heavy — 735 atoms with 30 such residues
at **19.4 Å**, 936 atoms with 4 at 14.6 Å.

Non-canonical amino acids are a primary tool for engineering designed
proteins, so this sits on the critical path rather than at the edge. Closing
it means real identity embeddings per non-canonical type, not one `UNK`
bucket.

---

## 2. Untested design choices

### 2.1 Equivariance, on the architecture that works
The codec is **not** equivariant and uses no graph/message-passing structure:
plain transformer attention, with invariance handled by input centring,
rotation augmentation, and Kabsch alignment in the loss. Bonds enter only
through the loss, never the forward pass.

An SE(3)-**invariant** encoder was tested (6.931 Å vs 7.137 Å baseline) — but
that A/B predates the direct decoder, and *every* arm in it loses to a
mean-shape baseline. It says nothing about whether equivariance helps the
codec that works. This is untested, cheap to test, and one of the more likely
remaining wins.

### 2.2 Temporal encoding in the codec
Frames are currently encoded **independently** and stacked into `(T, R, d)`;
all temporal modelling lives in the diffusion stage. Video VAEs compress time
as well as space, and MISATO's ~80 ps stride means consecutive frames are
highly redundant, so per-frame latents waste capacity.

Variants worth sweeping: windowed encoder (temporal context, no compression);
temporal compression (k frames → 1 latent frame); residual latents. Tension:
a per-frame codec trains on *all* static PDB, a temporal one only where
trajectories exist. Resolution is inflation — train spatial on PDB, add
zero-init temporal blocks, fine-tune on MD.

`drift_ratio` (in `molae/trajectory.py`) decides whether this is needed, and
needs no diffusion model to run.

### 2.3 Curriculum schedule
`molae/curriculum.py` exposes the schedule as four scalars, so it sweeps as a
config grid. Untested: whether a discrete mid-training phase beats a gradual
ramp, and where the shift should sit.

---

## 3. The complex-codec ceiling — what has been eliminated

`complex_d8` plateaus at ~5.5–5.9 Å all-atom against the 0.79 Å single-chain
reference. Six explanations have been tested and five are closed. Only
training-set size and the representation itself are still live.

**First, the noise floor — it qualifies every number below.** Held-out RMSD
over the last ten evaluations of a *single* complex run spreads far more than
the effects being measured:

| run | final | mean | sd | range |
|---|---|---|---|---|
| `complex_d8` (242k) | 5.885 | 5.935 | 0.192 | 5.72–6.36 |
| `complex_d16` (63k) | 6.172 | 5.761 | 0.449 | 4.88–6.37 |
| `complex_d8_3m` (242k) | 5.804 | 5.869 | 0.199 | 5.47–6.22 |
| ladder n556 | 6.126 | 6.003 | 0.304 | 5.60–6.50 |
| ladder n1112 | 5.902 | 5.778 | 0.344 | 5.19–6.31 |
| ladder n2224 | 5.368 | 5.829 | 0.271 | 5.37–6.13 |

The evaluation is a deterministic pass over the whole val split, so this is
weight jitter, not measurement noise. It means **the final checkpoint is a
draw, not a measurement**, and two conclusions drawn from single draws do not
survive contact with it:

- The data ladder read 6.13 → 5.90 → 5.37 by final checkpoint — a 0.75 Å
  descent. By last-ten mean it is 6.00 → 5.78 → 5.83: **flat, inside the
  band**. n556 drew high and n2224 drew low.
- `complex_d16` read 0.59 Å worse than `complex_d8`. By mean it is 5.761 vs
  5.935 — **tied**, with the largest spread in the set. "Latent 16 is worse"
  is dead; "latent 16 does not help" is what the data supports.

`complex_d8_3m` is unaffected (5.869 vs 5.935, deep inside the band) — capacity
was already a null and stays one.

Two separate noise sources, and they need different treatment. *Temporal
jitter* — the final-checkpoint lottery — is what the table measures, and it is
fixable: `train.ema_decay` averages the weights, and `val_summary` in
`train_log.json` now reports last-N mean ± sd so a headline number cannot be a
single draw. *Seed variance* is only measurable by running seeds; the one
estimate we have is 0.806 / 0.908 on the single-chain task, spread 0.102.
Seeds run before the jitter is fixed measure the two convolved together.

**Undertraining — RULED OUT.** 63k steps gave 5.584 Å; 242k gave 5.884. Four
times the steps bought nothing.

**Model capacity — RULED OUT.** 1.1M gave 5.584 Å held-out; 3.8M (3.5×) gave
5.804 Å. Both plateau at train ~5.3, v/t ~1.1. More parameters bought
nothing and cost a little. This rules out *model* capacity only; `latent_dim`
was 8 in both arms and `d_model` does not touch it — that axis is (1) below.

**Assembly placement — SECONDARY, not primary.** Superimposing each chain or
domain independently recovers only 1.25 Å of the 5.79 Å total (22%). There
IS a real multi-chain penalty — single-chain 4.24 Å vs multi-chain 6.18 Å, a
step the linear `corr(rmsd, n_chains) = 0.075` was blind to — but it is not
the main term.

**Size — RULED OUT, and this one was a misreading.** Absolute per-part RMSD
rises 2.67 → 8.49 Å with part size, which read as "large chains are harder".
Normalised by radius it is FLAT: `rmsd/rg ≈ 0.29–0.38` across every real
chain/domain bin. The codec has **constant fractional precision**; absolute
RMSD grows only because the structures are physically bigger. "Quality
degrades with size" was substantially a property of the metric.

That reframes the problem. Against the reference at `rmsd/rg ≈ 0.06`, the
complex codec sits at ~0.3 — a **uniform ~5× relative gap**, not a
size-dependent failure. Three candidates remain:

1. **Latent budget — RULED OUT.** Constant fractional precision looked like
   the signature of a fixed per-residue budget: 8 floats buys a fixed
   fraction of the dynamic range. Counter-evidence was already on record —
   on the ≤800-atom band the latent saturates by dim 4 (d1 4.069, d2 1.405,
   d4 0.707, d8 0.693, d16 0.681) — and `complex_d16` confirmed it at
   complex scale: doubling the budget made it *worse*, not flat.
2. **Training-set size — MEASURED FLAT over 4×, pending the last rung.** A
   matched-composition, leakage-controlled ladder at 556 → 1112 → 2224 gives
   6.00 → 5.78 → 5.83 by last-ten mean (sd ~0.3, SE ~0.1). No slope. Note what
   this does and does not say: it bounds the effect of data over the range the
   corpus can span (4,848 train structures total), not over the 10–100× that
   the single-chain result was never tested against either. n4448 is running
   and could still bend it.
3. **The representation itself** (multi-chain + ligands + modified residues),
   which is what we wanted to add.

**Where in (3) the error actually is — residue-granularity decomposition.**
Chain granularity put placement at 22%, which read as "the parts are right,
the assembly is mostly fine, so the failure is local geometry". That inference
had a hole: a chain superimposed as a *rigid body* still carries every
per-residue misplacement inside it. Superimposing each residue independently
closes it:

| quantity | value |
|---|---|
| global RMSD | 5.884 |
| residue-internal `part_rmsd` | 1.829 |
| placement (linear) | 4.055 — **69%** |
| placement (share of *squared* error) | **~90%** |
| single-chain val (n=31) | 4.126 |
| multi-chain val (n=155) | 6.236 |

RMSDs add in quadrature, not linearly, so the linear 69% understates it:
internal geometry is only `(1.829/5.884)² ≈ 10%` of the squared error.

The decisive line is the single-chain row. Those structures have **no chain
boundary**, and they still imply `√(4.126² − 1.829²) ≈ 3.70 Å` of placement
error — *larger* than the 2.11 Å multi-minus-single differential. So most
placement error is intra-chain and exists where the chain-boundary story
predicts none. That reorders the two candidate fixes:

- **Frame readout** (`model.dec_frames`) — reaches the 3.70 Å. The plain head
  maps one residue token through a single linear layer at a single output
  scale to 14×3 *absolute* coordinates: it must express a ~50 Å placement and
  a 1.5 Å bond length in the same units, and it re-emits the placement
  independently per atom, where it deforms the geometry it shares an output
  with. Frames send placement through 3 numbers and a rotation, moving the
  residue rigidly. Cost: **+1,161 params (+0.1%)**.
- **Chain awareness** (`model.dec_chain_aware`) — reaches at most the 2.11 Å
  differential. Still worth running: it is orthogonal, and it closes a real
  asymmetry (the encoder's featuriser has a chain embedding; the decoder had
  none, so its `res_pos` adjacency prior was false at every unmarked
  boundary). Cost: +133k params (+12%).

One methodological caveat on `part_rmsd`: Kabsch on a 4-atom group removes 6
DOF from 12 numbers, so per-residue superposition is permissive and the
internal term is biased *low* — which biases the placement share *high*. The
control is to run the same decomposition on the 0.79 Å single-chain reference
codec; the bias applies equally to both, so the comparison survives it even
though the absolute number does not.

With (1) closed and (2) measured flat, (3) is the only arm left standing.
Every lever that adds *more of something* — parameters (3.5×), latent budget
(2×), steps (4×), training structures (4×) — has now come back negative, and
none of them made the *train* RMSD better either. A model that cannot fit what
it has already seen is not short of anything you can add more of. The next
real experiment is a representational change, not a bigger version of this one.

One caveat on the v/t ratio, which fell 1.11 → 0.88 → 0.78 across the ladder
and looked like independent corroboration of a slope. At matched step counts
it is not clean: more data at the same steps means a worse train fit, which
lowers the ratio mechanically. It cannot be read as generalisation improving.

The specific suspicion about the readout: every atom reads slot
`(res_pos, slot_idx)` emitted from its own residue's latent. That is what made
variable-size decoding work at all, but it gives an atom no path to anything
outside its own residue — which is exactly what placing one chain relative to
another requires. The 22% assembly-placement term and the uniform `bb ≈ aa`
failure both sit where that predicts.

**Outstanding control:** rerun `ladder_direct_n2272` on current HEAD. Every
complex number was produced by code that changed substantially since 0.878 Å
was measured, and declared bonds in particular added 762 disulfides that
enter the *bond loss*. The loss is not the same function it was. If the
rerun reproduces 0.878 the chain of reasoning above is sound; if not, the
ceiling is measuring our own regression.

---

## 3b. Small molecules are relatively hardest

Parts with Rg < 8 Å (ligands, fragments) score `rmsd/rg = 0.85` against
0.29–0.38 for real chains. Small molecules are hard as a fraction of their
own radius. MISATO's ligands have a median of 28 heavy atoms, so this is
directly on the path — deliverable (b) should keep reporting the ligand class
separately rather than folding it into an all-atom mean.

---

## 4. Technical debt

### 4.1 `chain_idx` is overloaded
It means both "which polymer chain" (where the residue-adjacency bonding rule
is right) and "which separate molecule" (where it is not). Several
representation defects were cases where those two meanings disagreed. The
current behaviour is correct on every case tested, so this is not urgent — but
it is why the bugs arrived one at a time rather than together.

### 4.2 Geometric bond perception vs. declared topology
Reading `struct_conn` replaced three geometric heuristics and recovered a bond
class we had never perceived at all (disulfides: 0 across the entire project
before, 762 in the complex set after). It does not cover everything:

- **28%** of PDB entries carry no `struct_conn` at all, so intra-residue and
  intra-ligand topology there is still distance-perceived.
- Fe₄S₄ clusters sit entirely unbonded — the metal radii that correctly stop
  Zn–His *coordination* being read as covalent also sever the genuine Fe–S
  bonds *inside* a cluster, where the metal is part of the molecule. 33 atoms
  across 4 of 47 cached structures; recorded rather than patched, because a
  seventh geometric heuristic is the habit worth breaking.

The Chemical Component Dictionary answers both directly. **Closed for
MISATO** — AMBER prmtop supplies per-system bonds for all 16,972 systems — so
this is now a PDB-only question.

### 4.3 Milestone 1's bond metric never saw a disulfide
`REPORT.md`'s `bond_length_error` was computed on a corpus with zero
disulfides. It will not change the headline — reconstruction RMSD, chirality
and contact F1 do not depend on the bond-loss term — but it needs a footnote
once the difference is measured. Cheapest route: re-evaluate the committed
`ladder_direct3m_n2272` checkpoint against a `struct_conn`-corrected dataset.
Evaluation only, no retraining.

---

## 5b. WHAT THE MEASUREMENT PROGRAMME HAS EXCLUDED (state, not interpretation)

| route | status | rests on |
|---|---|---|
| global latent alone | **0/123**, win rate **< 2.4%** (rule of three) | n300 vs ANM, **one producer, no join** |
| + a *perfect* sparse channel | **5.7%** at matched budget; needs **12–27% of atoms** for parity | oracle; its K=0 column reproduces the n300 column exactly, **same producer** |
| static path | **DOES NOT EXTRAPOLATE IN SIZE beyond its training range** (66a — *not* "SMALL-PROTEIN"; see below) | a different arm entirely; **confounded by its own training set** |
| more data | **67.5% of systems worse**, sign test **p=0.000132**; the aggregate rise is a tail repair | **RESTORED** — same-producer recompute `10318733` confirms it (below) |

**Three of the four rows are load-bearing and none of them is a join.** "Every route to a peer win on
per-frame reconstruction FVE is measured and closed" rests on the **n300 absolute** and the **oracle**,
so the headline survives without the fourth row. The data-scaling row is **withdrawn from this table
until the recompute lands** — an exclusion list containing a provisional row invites exactly the
premature closure it exists to prevent (63a). It is not restated here in weakened form, because the
purpose of this section is to be quoted without reading the body.

**The one axis never tested** is the one 004 named at the start and 53b named again: **ANM has no
generator.** Reconstruction FVE cannot express a generative advantage — Family D at the level of the
research question — so a benchmark on that axis is not a consolation prize, it is the only untested
direction left. Recorded so the next question is chosen against what is known rather than re-derived.

## 5a. STANDING SCOPE LIMIT: every static-path corpus is capped at 3,000 atoms

`manifest.json` records `max_atoms 3000` on the complex corpus, and `processed_big` — which has no
manifest — measures out at **170–2,977 atoms, zero above 3,000**. So the cap is not one experiment's
footnote; it bounds **every** static-path result in this project.

Against the ATLAS held-out set (n=123, 598–33,377 atoms, quartiles 1,434 / 3,249 / 7,406):

> **58 of 123 ATLAS systems (47.2%) fall below the 3,000-atom cap. 65 (52.8%) are above it and have
> never been seen by any static-path model here.**

Every present and future claim of the form *"the static path scales"* is bounded by this, including
the sparse-channel work if it ever runs on these corpora. Capping at 3,000 atoms was a reasonable
construction choice; leaving it implicit in one experiment's verdict is the defect.

## 5. What is actually established

Worth stating alongside the gaps, since the list above is all deficits:

- The direct per-residue codec reconstructs held-out protein structures at
  **0.79 Å all-atom / 0.51 Å backbone**, chirality 0.0002, contact F1 0.963.
  **That 0.79 is a MEAN over 758 held-out structures; the MEDIAN is 0.84** (sd 0.21,
  quartiles 0.66 / 0.84 / 0.93, range 0.27–2.38, 1 above 2 Å). The distribution is
  left-skewed so the mean sits below the median — quote both. The set is
  **20–109 residues**, so this describes structures of that size, not proteins in
  general. Latent is **8 floats/residue**, i.e. only **~3× compression**.
  **The checkpoint is `ladder_direct3m_n2272`** — *not* any of the three
  `*perresidue*` result dirs, which sit at ~10 Å. The name does not identify it.
  **Two defects block reproducing this** (see below): the checkpoint does not load
  under current code, and loading it pads three embeddings with random rows.
- It is **data-limited, not plateaued** — val still descending at n=2272,
  val/train 1.50 and closing. We have used ~1% of the experimental archive.
- The fixed-size Perceiver alternative is **dead in direction**: its gap to
  the direct codec *widens* with data (ratio 7.11 → 7.71 vs direct-1.1M).
- The ordinal-slot addressing that makes ligands representable **works**:
  ligand-covalent 8.28 ≈ protein 7.98 in the same structures.
- Masked reconstruction is neutral on the direct codec (1.016 vs 1.020).
- The codec does **not** average conformers away: on NMR ensembles the
  conformational spread survives the round trip at ratio 0.868. That was the
  failure that would have killed the temporal stage outright.
- But its latent is only weakly organised *by* conformation: rho 0.933 against
  a random-projection control at 0.875, a gain of just +0.058. A codec that
  has never seen two conformers of one molecule has no reason to organise
  them, and the number reflects that — which is the measured argument for
  temporal context in the encoder (2.2).
- Predicted structures differ from experimental ones in GLOBAL SHAPE, not in
  bonds: within-type bond-length spread matches (1.05×), but 56% of raw
  computed structure models are extended (rg_ratio > 1.5) against 2% of
  experimental entries. Confidence filtering of the predicted tier is
  therefore load-bearing, not optional, and mixing ratios must be set on the
  filtered count.

---

## 6. The four objectives, audited

These are the stated targets for the system, not aspirations for the codec.
(Objective order does NOT match section order: **obj 1 -> 6.2, obj 2 -> 6.4,
obj 3 -> 6.3, obj 4 -> 6.1**. Objective 2 is bond breaking/forming.)
Each is audited against what exists, with the blocking item named. Ordered by
how far the current stack is from them, nearest first.

### 6.1 Single-GPU inference — ALIGNED, nothing to reverse

"Worth spending more training compute to inference more efficiently" is the
one objective the current design already satisfies, partly by luck and partly
because the alternatives were measured and rejected:

- **Per-group latents, not a fixed-size bottleneck.** The latent scales with
  the system instead of over-compressing it. At 1M atoms (~125k groups at 8
  atoms each) and `latent_dim` 8 the latent is 1M floats = **4 MB**. The
  fixed-size Perceiver alternative is dead in direction — its gap to the
  direct codec *widens* with data (7.11 -> 7.71).
- **Direct readout, no attention lookup.** Decoding is O(N) in atoms.
- **Flow matching is already implemented** (`objective: flowmatch`), which is
  the right family for trading training compute against few-step sampling.

The constraint does rule things out, and it should be treated as binding: no
O(N^2) component survives at 125k groups, and no many-step sampler survives
if it is O(N^2) per step.

### 6.2 1M+ atoms, general molecules — TRACTABLE, two hard caps

Blocking items, both in code rather than in principle:

1. **`max_res_pos = 1024`** — an *absolute* positional embedding over group
   index. Above ~8k atoms indices clamp and collide. Needs relative or
   continuous positional encoding. This is a 125x shortfall against target.
2. **O(R^2) attention** in the encoder trunk and decoder trunk. At 125k groups
   that is 1.56e10 pairs = **31 GB per head per layer** in fp16; ~250 GB for
   the current 4 heads x 2 layers. A k=32 neighbourhood is 0.01 GB. Half of
   this is already built: the `invariant` encoder has a `k_neighbors` path.

What carries over unchanged, and is the reason this is an extension rather
than a rewrite: **group-mediated addressing**. `(group, slot)` generalises
`(residue, atom name)` -- proteins use name-derived slots, everything else
uses ordinal slots -- and it is measured working, with ligand-covalent
geometry at 8.28 against protein 7.98 in the same structures.

The residual is harder than vocabulary, and it is the part of "general
molecules" that a bigger element table does not buy. 23 residues and 39 atom
names is protein-shaped, but the deeper issue is that `(group, slot)`
addressing presumes a *group decomposition exists*. Polymers have one
(residue, then atom name); small molecules get an ordinal fallback that works
because they are small. A covalent solid, a lattice, or a nanotube has no
residues, no sequence, and no natural grouping, and periodicity is not the
same thing. So the addressing scheme reaches polymers-with-sidechains plus
small ordinal groups, and stops.

General chemistry wants element plus local connectivity as the primary
identity, with named slots kept as a protein-specific specialisation on top.
Note this is the same design decision implicated in §3's placement ceiling —
the per-group readout is simultaneously what made variable-size decoding work,
what may be capping complex reconstruction, and what bounds materials scope.
Changing it is therefore one decision, not three.

### 6.3 Multi-millisecond timescales — ARITHMETIC WORKS, STABILITY IS THE RISK

Direct integration is 5.0e11 steps at a 2 fs timestep. That is the whole
reason for a latent dynamics model. A latent stepping at 1 ns reaches 1 ms in
1e6 steps; at 10 ns, 1e5. Both are feasible wall-clock.

The risk is not throughput, it is **error accumulation over 1e6
autoregressive steps**, and the relevant measurement is already on record and
is not encouraging: the latent is only weakly organised *by* conformation --
rho 0.933 against a random-projection control at 0.875, a gain of just
**+0.058**. It does preserve conformational spread through the round trip
(0.868), which is what keeps the temporal stage alive at all, but preserving
spread is not the same as structuring it.

Implications for stage 3: train on **long-lag pairs**, not consecutive frames,
so the model learns the coarse-timestep transition directly rather than
composing 1e6 fine ones; and put temporal context in the encoder (2.2), which
the +0.058 is the measured argument for.

### 6.4 Enzyme-like bond breaking and forming — NOT A SCALING PROBLEM

The furthest from the current stack, and the one that will not fall out of
scale. Bonds are an **input**: the decoder is conditioned on a fixed atom list
with fixed covalent topology and emits coordinates. Bond breaking is not slow
here, it is *inexpressible*.

Two separable sub-problems, and conflating them wastes effort:

- **Variable topology.** There is a concrete path that is not a rewrite. The
  repo already has geometric bond perception (`perceive_bonds`) alongside
  declared topology, and currently trains against *declared* bonds -- the
  model is handed the answer. Training against perceived bonds makes topology
  emergent, i.e. an output. See 4.2.
- **Correct energetics.** Bond breaking is a quantum event; classical
  potentials cannot rank it, which is why QM/MM exists. This is exactly what
  stage 4's quantum-accurate compute is for, and it cannot be substituted by
  more data at the codec level.

Plan this as its own arc with its own gates. Do not assume it emerges.

### 6.5 What this changes about the current queue

Nothing immediately, and for a specific reason: **placement error is on the
critical path to 7.2**. Section 3 measures ~90% of the complex codec's squared
error as placement rather than internal geometry. That failure mode gets worse
with system size, not better -- a decoder that cannot place two chains cannot
place ten thousand fragments. The frames readout is a prerequisite for scale
independent of whether it closes the complex gap today.

After it resolves, the ordering follows the audit above rather than incremental
codec polish: relative positional encoding and neighbourhood attention (6.2)
are the cheapest large unlocks, long-lag temporal training (6.3) is gated on
trajectory corpora that already exist (MISATO built, mdCATH identified), and
topology-as-output (6.4) needs its own design pass before any GPU is spent.

### 6.6 Leverage ordering, ranked against the objectives

Per unit effort, once the frames result lands. Each entry names which
objective it serves, so the ranking is auditable rather than a matter of
taste.

1. **Relative positional encoding** (obj 1). Removes a hard 125x cap for a
   contained change. Nothing else in the list matters if group indices
   collide above 8k atoms.
2. **Neighbourhood attention in both trunks** (obj 1, 4). 31 GB per head per
   layer -> 0.01 GB. Half-built already via the invariant encoder's
   `k_neighbors`. Serves the efficiency constraint at the same time, which no
   other item does.
3. **Long-lag trajectory training** (obj 3). Gated on corpora that already
   exist (MISATO built, mdCATH identified), and directly targets the +0.058
   conformational organisation that is the measured risk to 1e6-step
   stability. **Second, independent justification** (added, ranking unchanged):
   trajectory length is the single bind behind three separate limits --
   dev_modes_99 at 93% of ceiling, the tight-tau latent-width rows (§7), and
   basin-hopping for obj 3 -- so longer trajectories (mdCATH) now also gate the
   §7 latent-width decision, not only timescale. **mdCATH scoped** (task b, from
   primary sources): HF `compsciencelab/mdCATH`, CC-BY-4.0, **3.3 TB** HDF5,
   one file per domain (5,398 domains x 5 replicas x 5 temps). **~464 frames/traj
   at 1 frame/ns = 4.6x MISATO's frames and 58x its time** -- comfortably lifts
   the T=100 rank ceiling (pool replicas/temps for thousands of frames/system).
   Topology is **shipped** (atomic numbers `z` + PSF bond list -> no perception),
   coords+forces all-atom, torchmd-net has a loader. Ingest path: read `z`
   (element_idx) + parse PSF (bonds) once/domain, stack `coords` per (temp,
   replica). Caveats: 3.3 TB -> ingest a domain SUBSET, not the whole set; temps
   are 320-450 K (450 K drives unfolding transitions but is non-physiological, so
   temperature is a variable to condition on, not ignore).
4. **Perceived- rather than declared-bond training** (obj 2). Cheap probe of
   whether emergent topology is viable at all. Answers a yes/no that decides
   whether obj 2 is an extension or a separate model.
5. **Element+connectivity identity** (obj 1). Larger change, and it is
   entangled with the readout decision in 6.2 — worth doing once, after the
   frames result tells us what the readout should be.
6. **An energetic output** (obj 2). A learned force head converts the codec
   from a shape model into something that can rank states. Prerequisite for
   any bond-breaking work to be scoreable, but it is downstream of 4 deciding
   whether topology can vary at all.

Deliberately NOT on this list: inverse conditioning, interface metrics, and
non-canonical residue identity. All three were ranked highly under the
previous framing. None of them serve the four objectives directly, and design
capability in particular is a different product from a dynamics model.

---

## 7. Target architecture for the simulation stage

One shared **global latent per timestep** carrying the dynamic state, with
atom identity, bonding and reference geometry supplied as **static per-atom
conditioning**, plus a **sparse local event channel** for bond breaking and
other microscopic changes the global latent should not have to carry.

```
static, once per system:   identity + bonds + reference geometry -> S_i  (N x d_s)
per timestep t:            global dynamic latent g_t  (L_g x d_g, FIXED)
                           sparse events e_t          (few (atom_idx, vector))
decode:                    x_i(t) = f(S_i, g_t, e_t)
```

### Why this is the well-founded version, and the earlier attempts were not

Everything that failed this session was compressing **static structure**
through a small bottleneck: our fixed-size Perceiver (gap widens with data),
ProteinAE-Register (breaks past the training length). Static structure of a
large system is genuinely high-dimensional and there is no evidence anywhere
that one global latent holds it.

The per-timestep **deviation** is a different object, and PCA-of-MD, tICA and
Markov-state modelling all report it dominated by a few slow collective modes.
That is the quantity the global latent carries here, and the static conditioning
supplies everything the deviation is measured *against*.

Note this also changes what counts as a leak. The no-content-skip rule was
right for the autoencoder, where the structure WAS the thing being compressed.
Here the structure is given and the *dynamics* is compressed, so reference
geometry as conditioning is the task definition, not a shortcut.

### Cost, which is the point

Per-timestep latent, 1M atoms (~231 protein-ligand systems at MISATO's mean
4,320 heavy atoms), 10^6 steps for 1 ms, 1000 candidates per evolutionary round.
The premise check (below) shows the dynamic modes **add across independent
molecules**, so the latent scales with **molecule count, not atom count** --
2/4/8 tokens per molecule at 231 molecules is 462 / 924 / 1,848 tokens:

| per-timestep latent | floats/step | 1 ms | x1000 candidates |
|---|---|---|---|
| per-atom d=8 | 8,000,000 | 8.0e12 | 8.0e15 |
| per-residue d=8 | 1,000,000 | 1.0e12 | 1.0e15 |
| seqpool r=64 | 125,000 | 1.25e11 | 1.25e14 |
| **per-molecule 4 tok/mol x 231, d=16** | **14,784** | **1.48e10** | **1.48e13** |

~68x cheaper per step than per-residue and **constant in atoms-per-molecule**,
which is the property the search needs. The earlier "one fixed global 1024x16"
row is struck: a single fixed latent cannot carry the additive dynamic state
(effective dimension >> 1024, below). It holds the box's **global** state only;
the per-molecule tokens carry the additive part. (Token count: 125,000
per-residue vs 924 per-molecule at 4 tok/mol = 135x fewer.)

**Width is a fidelity-target decision, pinned in scalars** (stated in scalars so
the choice survives a change in d). The sizing quantity is the per-molecule-
fidelity requirement `sum_k modes_abs(block_k)` -- the minimum modes to bring EACH
molecule's residual per-atom variance under a target tau, additive by construction
and pool-free (the only sound criterion; see the control below for why). It
depends on the fidelity target:

| target RMSD | tau (A^2) | scalars required (~231 x mean) |
|---|---|---|
| 0.05 A | 0.0025 | ~22,700 (truncation-limited) |
| 0.10 A | 0.01 | ~21,800 (truncation-limited) |
| 0.20 A | 0.04 | ~19,000 |
| 0.30 A | 0.09 | ~15,400 |
| 0.40 A | 0.16 | ~11,700 |
| 0.50 A | 0.25 | ~8,224 |
| ~0.53 A | 0.28 | ~7,392  (= 2 tok/mol at d=16) |
| 0.60 A | 0.36 | ~5,359 |
| 1.00 A | 1.00 | ~520 |

The sweep does not *select* a token count -- it shows the **budget is
tolerance-dominated**. The 2-vs-4 crossover sits at ~0.53 A against a nominal
0.5 A target: a ~6% move in tau flips the decision, so the measurement does not
distinguish the options; the choice of tau does. **And 0.5 A itself is
unjustified and circular** -- arm A already reconstructs C-alpha at 0.365 A and
the measured mean RMSF is 1.05 A, so "0.5 A" is approximately *as good as we
already are*, a status quo dressed as a requirement. The tolerance has to be
derived from what breaks downstream (next subsection), not assumed.

Two structural facts from the extended sweep, and they are a **live risk to the
4 tok/mol recommendation, not a footnote**: below ~0.3 A the requirement
**exceeds 4 tok/mol** (14,784) and climbs steeply, and the tightest rows
(<= 0.1 A) are **rank-ceiling-limited at T=100** (~98 of 99 modes), so they are
**lower bounds, not estimates**. The r^-12 argument (next subsection) says the
force-derived tau should land under 0.5 A -- in which case **4 tok/mol may be
insufficient by an amount we currently cannot measure**. Read "4 tok/mol
recommended" as provisional, not settled.

This is the **third place trajectory length binds the design**: dev_modes_99 at
93% of ceiling, these tight-tau width rows, and basin-hopping for objective 3.
One fix serves all three -- longer trajectories (mdCATH); see 6.6.

**Recommendation stands: 4 tok/mol** for the measured 8 ns / ~0.5 A regime --
headroom is cheap and the 8 ns caveat argues for slack -- but it is a **tolerance
choice, not a measurement outcome**. (dev_modes_90's 9,518 is a scale-inflated
relative proxy; absolute modes_abs is the sizing number.)

**Caveat carried onto the pinned numbers:** these are the **8 ns within-basin**
figures. Basin-hopping over a millisecond adds modes this control never touches --
the budget is pinned for the regime measured, **not for the 1 ms objective**.

### Deriving the tolerance from the validator, and the fork it opens

The tolerance must be set by where reconstruction error becomes invisible to
whatever scores structures downstream -- energies and forces, which are violently
sensitive to coordinate error. A repulsive term going as r^-12 means a 0.5 A
displacement at a 4 A contact moves that term by factors of several, so energy
error can sit orders of magnitude above the ~1 kcal/mol chemical-accuracy scale
while RMSD still looks fine. The right tau is where |dE| stays under ~1 kcal/mol
per molecule and force error under a stated fraction of typical force magnitude.

MEASUREMENT (pending, queued behind the C/D cross; CPU, no training): inject
isotropic Gaussian coordinate noise (sigma 0.05 .. 1.0 A) into real corpus
structures and read dE, dF vs sigma. Tooling caveat: **no force field is callable
in-env** (no OpenMM/ASE/OpenFF/RDKit; only parmed, which reads Amber parameters
but does not evaluate). The measurement will use a **labelled LJ + harmonic-bond
stand-in built from the real Amber LJ/bond parameters** parmed extracts per
prmtop -- correct for the curve SHAPE and the close-contact story, absolute
kcal/mol indicative only. Expected: the curve is dominated by a few close
contacts, so tau is set by the tightest contacts, not the average atom -- which
would make a per-atom uniform tau the wrong shape.

**The fork (latent budget and validator budget are coupled through tau).** If the
derived tau is tight, two routes trade against each other:
- **(a) bigger latent** -- buy tight reconstruction directly (>4 tok/mol; the
  extended sweep shows the scalar cost climbing steeply below 0.3 A).
- **(b) loose reconstruction + a short local relaxation before scoring** --
  tolerate larger tau and pay validator/relaxation compute to repair the close
  contacts that dominate the energy.
Route (b) spends downstream per-evaluation compute; route (a) spends latent
width. **Neither budget can be set without the other, through tau.** NOTE: the
ROADMAP does not yet contain a validator / duty-cycle budget (referenced as "§8"
in planning, but no such section exists; objective-2 scale caps are in 6.2,
bond-breaking energetics/QM-MM in 6.4). The validator-compute side of this fork
is therefore currently **unquantified** and needs its own section before route
(b) can be costed.

### The premise, and how it gets checked

The design rests on the deviation being low-dimensional at our system sizes.
`scripts/trajectory_dimensionality.py` measures it on real trajectories before
anything is built on it: PCA modes for 90/95/99% of the displacement variance,
the same for frame-to-frame deltas, and the fraction of variance sitting in the
top 1% most mobile atoms.

Validated on synthetic cases with known answers:

| case | modes for 90% | variance in top 1% of atoms |
|---|---|---|
| 5 collective modes | 5 | 0.02 |
| one localised event | 3 | **1.00** |
| independent per-atom motion | 172 (of a 199 ceiling) | 0.01 |

The middle row is why locality is reported alongside the mode count: a purely
local event is ALSO low-dimensional, so mode count alone would wrongly endorse
a global latent. High locality is the measured argument for the sparse channel.

### Measured result (20 systems, 8 ns): premise supported for single systems, two limits

Ran on 20 CA-superposed MISATO protein-ligand trajectories (T=100 frames, 80 ps
stride = **8.0 ns**; N = 998-16,522 heavy atoms, mean 4,320). Deviation is taken
from frame 0 after Kabsch alignment on protein CA (removes global tumbling;
self-test residual < 1e-8 A). Artifacts: `$WORKROOT/misato_traj_probe/`.

| quantity | mean | of ceiling 99 |
|---|---|---|
| dev modes 90% | 53.9 | 54% |
| dev modes 95% | 71.1 | 72% |
| dev modes 99% | 91.8 | 93% (truncation-limited) |
| delta modes 90% | 57.2 | -- |
| top-1% atom variance | 0.131 (max 0.61 in 1PU7) | -- |

**Core call -- deviation is genuinely low-dimensional, and its dimensionality
does not scale with system size.** 54% of the T-1 ceiling is real concentration,
not a short-window artifact: the matched isotropic null (N=4,320, T=200) sits at
175 of 199 = 88%. The load-bearing result is that the 90% mode count is **flat at
~44-67 across a 13x range of size** (N=1.5k -> 16.5k) -- latent width is set by
the dynamics, not by N. Locality reaches 0.61 in 1PU7 (a system with only 26
collective modes but a dominant local event), so the sparse channel is
load-bearing, not optional. **Premise supported for single systems at ~8 ns.**

Caveat carried forward: dev_modes_99 = 91.8 at 93% of ceiling is
truncation-limited and **sizes nothing** -- pinning latent width needs longer
trajectories (raise T to lift the 99-mode ceiling).

Two limits this result does NOT clear:

- **Timescale.** 8.0 ns is 125,000x short of the 1 ms objective. 54 modes
  describes fluctuation *within one basin*; it says nothing about
  basin-to-basin transitions, which are exactly what a millisecond model is for.
  This does not validate the millisecond target.
- **Multi-molecule additivity.** Measured on ONE protein-ligand system per
  trajectory. The size-independence above was across DIFFERENT single systems --
  not evidence that many molecules in one box share modes. Independent molecules
  add modes, so the effective dynamic dimension at 1M atoms (~231 systems) is the
  pooled count of 231 real deviation spectra, not the per-system 54.

  A control settles the scale (`scripts/additivity_control.py`; exact
  pooled-spectrum calc = independent-blocks / infinite-T limit, ceiling-free, K
  up to 231):

  | metric / block regime | K=2 | K=16 | K=32 | K=231 |
  |---|---|---|---|---|
  | dev_modes_90, identical blocks | 1.000 | 0.996 | 0.996 | 0.995 |
  | PR, identical blocks | 1.000 | 1.000 | 1.000 | 1.000 |
  | dev_modes_90, heterogeneous (real mix) | 0.927 | 0.791 | 0.780 | **0.766** |
  | PR, heterogeneous (real mix) | 0.726 | 0.365 | 0.324 | **0.285** |
  | modes_abs 0.5A, heterogeneous | 0.941 | 0.775 | 0.756 | **0.738** |
  | modes_abs 1.0A, heterogeneous | 0.863 | 0.414 | 0.375 | **0.344** |

  Four results:
  1. **The discount does not compound.** `dev_modes_90` drops fast then flattens
     at ~0.77 (0.791 at K=16 -> 0.766 at K=231): a measured **~9,518 pooled modes**
     vs the naive 54 x 231 = 12,474. Independent-additive is therefore ~9,500, not
     12,474 -- the worst case is only ~1.3x pessimistic, not uselessly so.
  2. **The fixed global latent stays dead.** ~9,518 modes is **9.3x over** a 1024
     global latent; physical coupling can only *reduce* modes below the ~9,518
     independent baseline, never raise it. So the budget scales with molecule
     count regardless. Budget decision unchanged.
  3. **PR is exactly additive only for IDENTICAL blocks** (ratio 1.000 across all
     K -- pooled spectrum is each eigenvalue with multiplicity K, so both Sum L and
     Sum L^2 scale by K and PR = (Sum L)^2/Sum L^2 = K x PR_block). This pins the
     sub-additivity to the 90% *threshold* (greedy selection goes deep into flat
     blocks, shallow into concentrated ones), not to physics. But for the real
     HETEROGENEOUS case PR collapses to **0.285** -- worse than dev_modes_90 --
     because scale spread makes (Sum V)^2/Sum Q track the largest block. So PR is
     NOT the cleaner additivity metric for a mixed box; its 1.000 baseline is a
     homogeneity artifact, and `dev_modes_90` (baseline ~0.77) is better behaved
     there.
  4. **Size per-molecule -- it is the only SOUND criterion, not a workaround.**
     The absolute `modes_abs(tau)` also sub-adds when pooled (0.738 at 0.5 A,
     0.344 at 1.0 A, K=231), and it does so **even for IDENTICAL blocks**: exact
     duplication is 1.000 at K=1,2 but drops to 0.980 by K=231 (114 of 5,775 modes
     shed). The reason is that a pooled absolute criterion is
     `sum(residual)/sum(N) <= tau` -- an **average over the box**. That average
     lets a floppy molecule stay badly reconstructed while near-rigid ones pull
     the mean under tau (for identical blocks, accumulated per-mode
     over-satisfaction is the same slack). So pooled modes_abs does not merely
     fail to add; it **certifies boxes containing individually-failed molecules.**
     The sound criterion enforces the tolerance PER MOLECULE:
     `sum_k modes_abs(block_k)`, additive by construction and pool-free. At 0.5 A
     that is ~231 x 35.6 = **~8,224 scalars** (the pinned width requirement; see
     Cost for the fidelity sweep). **This is the same principle as the atom-level
     traceability tests** -- average-over-box fidelity is exactly the failure mode
     they were built to catch (an aggregate that looks right while individual
     atoms are lost). Sizing criterion and architecture requirement are one
     constraint at two scales, not a metric technicality. (dev_modes_90's 9,518 is
     a scale-inflated relative proxy that only happens to agree on 4 tok/mol.)

  The pooling ratio itself cleanly quantifies how wrong pooling gets as
  heterogeneity grows: 0.980 for identical blocks (pure overshoot slack) down to
  0.344 for the real heterogeneous mix at 1.0 A -- both numbers measured.

  Caveat: the 231 blocks were sampled from only 20 distinct real shapes; a truly
  231-distinct mix could push the ratios somewhat lower, but the non-compounding
  (flat from K=16) behaviour is robust. Still pending -- the *physics*: genuine
  multi-solute trajectories vs the independent-concatenation baseline of the SAME
  molecules is the only thing that isolates real shared modes from this
  metric/heterogeneity baseline. MISATO has no multi-solute boxes at scale.

  **Physical reading of the collapse (record, don't build yet).** The steep
  sub-additivity is not only a metric artifact: per-molecule variance scale is
  spread very wide across the real mix (V/N 0.81-6.16 A^2 -- floppy complexes vs
  near-rigid), so a minority of molecules carries most of the pooled variance
  (order ~30% from the ratio -- an estimate, not a law). Consequence for the
  encoder's per-molecule head: allocation should scale with each molecule's OWN
  `modes_abs`, with flat per-molecule allocation as the fallback when per-molecule
  spectra are not available at box-assembly time. Nothing to rebuild now -- it
  changes the per-molecule head later.

  (Metric roles, kept separate: latent WIDTH is sized by the absolute per-molecule
  `modes_abs` sum at a target fidelity (above); `dev_modes_90/95/99` are secondary
  relative cross-checks; PR measures spectral concentration only and must never
  size width -- on a concentrated spectrum it lands well below the modes_abs count.)

### Invariants the implementation must hold

1. The global latent `g_t` has fixed size and carries the box's global state.
   The additive per-molecule dynamics is carried by dynamic tokens that are
   fixed *per molecule* and scale with molecule count, not atoms per molecule --
   the additivity arithmetic (limit 2 above) forbids a single fixed latent from
   holding the whole dynamic state.
2. Every output atom is individually addressable -- the address comes from the
   static per-atom code, so it cannot be lost in the bottleneck.
3. Sparse events reach only their own atoms.
4. Removing `g_t` must degrade the prediction: if it does not, the static
   conditioning alone is doing the work and the latent is decorative.

## 8. The validator, the reaction channel, and what they cost

Layer one of the target stack -- the trajectory proposal -- is §7, not new work.
The full stack:

```
static atom conditioning + global box latent + per-molecule dynamic tokens
  + sparse local reaction-event tokens
    -> atom-level trajectory proposal            (§7)
      -> energy / force / chemistry validator     (8.1-8.3)
        -> reaction channel                       (8.4-8.5)
          -> inverse design + experimental loop   (downstream, ~zero effort)
```

### 8.1 Validator duty cycle -- the hard budget

Per-frame validation reinstates exactly the cost latent rollout exists to avoid,
and kills the throughput objective. Order-of-magnitude arithmetic (labelled as
such):

- 1 ms at 80 ps stride = **1.25e7 decoded frames**.
- MACE-class E+F over 1e6 atoms ~**10 s/GPU**.
- Per-frame validation ~= 1.25e8 GPU-s ~= **4 GPU-years per millisecond trajectory**.
- 1-in-1e3 duty cycle ~= **35 h**; 1-in-1e4 ~= **3.5 h**.

So the validator cannot be per-frame. It is a **sparse global audit** at a fixed
low duty cycle **plus dense local checks** on the 1e2-1e3 atoms named by the event
tokens. DFT escalation needs its own rate budget: a few hundred QM atoms is
minutes to hours, so ~1e3 events per trajectory is days -- a hard cap on the event
rate, not just on event handling.

**The decode floor is ~1.2 A backbone and no codec work removes it (measured).**
The cross-replica gate (armf_generalize_prereg.md) showed per-system PCA modes
reconstruct an INDEPENDENT run of the same molecule at 1.17 A backbone vs 0.93 A
for the run's own past -- the residual **0.24 A is genuinely trajectory-specific**
(independent runs sample different sub-states of a rough landscape) and is
therefore **unlearnable from structure**. Even a perfect structure->modes codec
tops out at ~1.17 A. Consequence: **every downstream stage must tolerate ~1.2 A
backbone error**, so the (b) fork below -- **loose reconstruction + a short local
relaxation before scoring** (§6.6, and the (a)-vs-(b) fork at line ~649) -- is
**MANDATORY, not optional**: a scorer that needs sub-A geometry cannot be fed
decoded frames directly, because the codec cannot deliver sub-A even in principle.
The relaxation duty cost sits next to this per-frame budget and 8.3 prices it.

**The relax-before-scoring fork is now empirically validated (armf_usability.md).**
A short local relaxation of an ANM-decoded backbone (recon 1.54 A) yields a
physically valid structure ~= a PCA-decoded one (recon 0.71 A): bonds 97% vs 100%,
angles 94% vs 98%, Rama 88% vs 90%, minCA 3.85 vs 3.88. So the ~1.2 A codec floor
is not a blocker for downstream scoring -- relaxation repairs it. The codec stage
is settled: **ANM modes are the general codec** (contact graph from any structure,
no fit, no corpus), and they are usable after relaxation.

**Cost note (MEASURED, armf_generality_scaling.md): the eigensolve is the scaling
bottleneck.** Full `eigh` is O(N^3). Sparse SHIFT-INVERT Lanczos (scipy eigsh) does
return the correct leading 64 modes (matches dense to ~1e-9) BUT scales O(N^2.03) --
the LU factorisation of a 3D-mesh Hessian has O(N^2) fill-in -- so 1e6 atoms projects
to ~1.9e6 s (~22 days) per eigensolve and the search economics do not close (~0
candidates/GPU-hour). The fix is a MATRIX-FREE PRECONDITIONED iterative eigensolver
(LOBPCG + multigrid/Jacobi) that avoids the LU factorisation; not yet tested. So
"Lanczos at a fraction of the cost" holds for correctness but NOT for scaling with
shift-invert -- matrix-free is required and is on the build list.

### 8.2 Barrier accuracy, not energy MAE -- where objectives 2 and 3 couple

Barrier error enters rates **exponentially** (Arrhenius), so energy MAE can look
fine while the chemistry -- the rates -- is wrong. The validator's correctness
metric is **barrier accuracy**, not energy MAE. Primary home is **objective 2**
(bond breaking/forming = 6.4; objective order != section order, see §6). The claim
had no home in the doc; 6.4 is nearest and did not state it.

Cross-referenced to **6.3** because barrier accuracy is **where objectives 2 and 3
couple**: a millisecond trajectory is mostly rare events, and their statistics are
set by barriers, so barrier error corrupts long rollouts as **wrong event counts**,
not as noise.

The arithmetic makes the target actionable. Rate ~ exp(-dG_barrier / kT), with
kT = 0.593 kcal/mol at 300 K, so a barrier error dG is a rate-error factor
exp(dG/kT):

| barrier error | rate error |
|---|---|
| 1 kcal/mol | 5.4x |
| 2 kcal/mol | 29x |
| 3 kcal/mol | 157x |

So **chemical accuracy (1 kcal/mol) buys only ~5x in rate.** Over a 1 ms window
that is the difference between an event occurring a handful of times vs dozens, or
occurring vs not. "Target 1 kcal/mol" reads as sufficiency and is not -- it is a
floor, not the goal.

### 8.3 Route (b) is costable -- via the contact-violation rate

§7's fork is (a) bigger latent vs (b) loose reconstruction + a short local
relaxation before scoring. Full-box relaxation per frame is the same order as the
validator it was meant to avoid and dies on the same 1.25e7-frame arithmetic --
so route (b) is viable **only localized to atoms in tight contact**, and its cost
then scales with the **contact-violation rate**. The queued noise sweep (§7)
produces that for free with one added column: at each sigma, the **fraction of
atoms whose local energy error exceeds threshold**. That fraction x the per-atom
relaxation cost is route (b)'s price, making the (a)-vs-(b) fork **quantitative
instead of named**. The column is to be added when the sweep runs.

### 8.4 Reaction channel -- untrainable on the current corpus, not undertrained

MISATO and mdCATH are fixed-topology classical force fields: **zero reaction
events by construction**. No amount of training on them produces bond-breaking --
it is untrainable here, not undertrained. This needs a separate **reactive
corpus**, and it is plausibly the **earliest-start / critical-path** item.

Candidate sources, **verified against primary sources** (findings recorded, not
acquired):

- **Transition1x** -- 9.6M DFT configs from 10,073 reactions; **full NEB/CINEB
  pathways incl. TS**, real bond break/form (<=6 changes/rxn). wB97x/6-31G(d),
  gas phase, CHON, <=7 heavy atoms. Best substrate for a potential that must be
  accurate *through* the TS region.
- **RGD1** -- 176,992 reactions with validated TS (endpoints + TS, not dense MEP).
  GFN2-xTB + B3LYP-D3/TZVP, gas phase, CHON, <=10 heavy atoms. Better for
  barrier / TS-geometry ML than along-path training.
- **ANI-1x / ANI-2x, SPICE** -- **CORRECTION to an earlier draft: these contain
  ZERO reactions.** ANI are off-equilibrium conformers (fixed topology); SPICE is
  equilibrium + perturbed conformers. SPICE alone has biomolecular *context*
  (dipeptides, solvated amino acids, aa-ligand pairs) but no bond changes -- same
  category as MISATO for reactions, at QM level. Pre-training substrate for a
  force head, not a reactive corpus.
- **Newly surfaced, still gas-phase organic:** Reaction-QM (~2.3M rxns GFN2-xTB +
  ~200k at B3LYP-D3 with full IRC trajectories ~23M configs; extends to
  Si/P/S/Cl -- the strongest breadth option), HORM (1.84M Hessians for TS opt),
  QMrxn20 (E2/SN2 TS). **M-CSA** is the enzyme-mechanism database but is 961
  entries of *2D electron-flow schemes* linked to PDB -- a mechanism/priors
  source, NOT a training corpus of 3D reactive trajectories.

**The enzyme / condensed-phase hole is confirmed real.** No large public dataset
of enzyme active-site or condensed-phase reactive events (QM/MM reaction
trajectories in protein context) exists at scale; enzyme-reactive work is bespoke
per-system QM/MM (e.g. AbyU EMLE). Consequence: reactive protein-context data
must be **generated** (QM/MM or ML/MM electrostatic-embedding + metadynamics +
active learning). The gas-phase reactive sets above are viable **pre-training**
for the QM reactive core; the active-site distribution has no public equivalent.
Earliest-start / critical-path item, and it is a generation problem, not a
download.

### 8.5 Identity contract change (record, do not implement)

Bond changes break graph-derived addressing: WL classes and canonical order
shift, so atoms **silently re-index mid-trajectory** -- exactly the traceability
failure the atom-latent work has been guarding against. When the reaction channel
is built, the identity contract must change: **identity anchors to (element,
persistent t=0 index)**, and graph features (WL class, canonical rank) demote from
**identity key** to **time-varying conditioning**. Recorded as a contract change
at the top of `molae/graph_identity.py`. Do not implement until the reaction
channel exists.

**Amendment (simpler than freeze-at-t=0).** With a CONTENT-ADDRESSED decoder (§9)
-- each atom's decoder query carries its own (element, reference position,
persistent index) -- graph-derived canonical order is **no longer load-bearing for
identity at all**. The original §9-style fear (a broken bond re-indexes atoms via
canonical order) dissolves: don't use graph-derived order for addressing in the
first place. Content addressing on (element, coordinates, persistent index) is
topology-robust for free. WL/canonical order stays useful only for symmetry-aware
METRICS, and those do not have to survive bond changes. So the fix is not "freeze
canonical order at t=0"; it is "never address by graph-derived order."

## 9. Latent architecture: position, content addressing, and the ordering caveat

The atom-native Perceiver (generic slots, learned allocation) is the intended
encoder. Four points fix why it is well-posed and where its boundary is.

### 9.1 Two kinds of "position", opposite requirements

INDEX position (array order) is physically meaningless; permutation-invariance with
respect to it is **desirable**. SPATIAL position is essential and we already have
it -- **coordinates enter as CONTENT features on each atom token, not as positional
encodings.** Omitting positional encoding therefore loses only the arbitrary
ordering, not the geometry. This is precisely why the Perceiver (P) arm is
well-posed: it drops index order and keeps spatial content. (It is also why the
key/value split matters: identity+position in the KEY sets frame-stable attention
weights; displacement in the VALUE carries the dynamics -- a uniform pool over an
aligned zero-mean displacement field averages to ~0, the measured frame-invariant
bug.)

### 9.2 Content addressing keeps the latent usable, for free

The decoder is queried PER ATOM, each query carrying that atom's own identity
(element, reference position, persistent index). Correspondence comes from the
QUERY, so permutation invariance costs nothing in traceability. Consequence: the
WL / canonical-order apparatus is **no longer load-bearing for identity** (see the
§8.5 amendment) -- content addressing on (element, coordinates, persistent index)
is topology-robust for free.

### 9.3 SFC ordering is not time-stable -- an architectural argument for P-init

Morton/Hilbert rank is computed FROM COORDINATES, so it is not stable along a
trajectory: as atoms move, codes change and atoms reshuffle between slots, making
the latent discontinuous in time. Pinning SFC to the REFERENCE (as the sweep spec
does) avoids the discontinuity but inherits reference staleness -- it degrades
under large conformational change and is undefined across a topology change.

**Measured (frac of atoms whose slot assignment changes, reference vs last frame):**

| system | n | CA drift A | L=8 | L=16 | L=32 | L=64 |
|---|---|---|---|---|---|---|
| 3a5zD02 | 64 | 1.79 | 36% | 61% | 81% | 89% |
| 3jvvA01 | 100 | 1.60 | 22% | 38% | 59% | 74% |
| 2k4qA00 | 156 | 11.1 | 81% | 90% | 94% | 97% |
| 3h7lB02 | 482 | 2.57 | 39% | 60% | 71% | 83% |

At L=64, 74-97% of atoms would land in a different slot -- the reference-pinned
segment assignment is badly stale, worst where flexibility is highest. Learned
attention over coordinate CONTENT re-weights continuously and has no such issue.

So the architectural ranking is clearer than the current optimisation results
suggest: **segments train better NOW, the Perceiver has the right long-run
properties, and P-init gets both.** This is an architectural justification for the
P-init arm, not merely an optimisation convenience -- **P-init must not be dropped
even if the segment arm wins the current sweep.**

### 9.4 Known boundary: absolute coordinates as content are not equivariant

Attention over ABSOLUTE coordinates is not rotation- or translation-invariant. We
get away with it only because every frame is Kabsch-aligned to the reference,
putting everything in a canonical frame -- a **PREPROCESSING property, not an
architectural one.** It BREAKS for multi-molecule systems, where one global
alignment is meaningless. Relative geometry or an equivariant encoder becomes
necessary at that point. Recorded as a known boundary, not work now.

## MILESTONE (2026-08-05): first objective-1 movement -- general LIGAND codec

Objective 1 (general molecules) has moved for the first time; every prior codec result was
proteins. On MISATO ligands, a learned **graph codec** (message-passing GNN over the bond graph
-> per-torsion softness -> softest-L modes of (diag(k), G) -> NeRF) is a **general ligand codec**:

- **Beats Cartesian ANM** at L8 (held-out 0.64 vs 0.74 A) and L16 (0.44 vs 0.73 A).
- **Generalises across held-out ligands**: train ~= held-out (L8 0.60/0.64; L16 0.43/0.44) over
  48 held-out molecules -- zero test-time params.
- Measured against a **per-trajectory-fit ceiling** (softness oracle 0.51 / torsion-PCA 0.46 at
  L8) that **no general method can reach by construction** (the ceiling sees each test molecule's
  own trajectory). The L4/L8 residual gap is **documented-and-accepted**, not a failure; it is the
  system-specific part of softness that structure alone cannot predict. At L16 the gap nearly
  closes (0.44 vs 0.39).
- **Opposite of the protein result** (where ANM beat 4 learned attempts): ligand softness is not
  predicted by ANM's contact graph but IS learnable by a GNN.
- **Scaling headroom on record**: trained on **190 of ~17k MISATO ligands** with a small GNN.
  Whenever the L8 ceiling gap matters, more ligands + a bigger model is the untried lever.

Full writeup + guards (bond/angle exact by construction; min non-bonded 1.96 vs 2.05 A ref, no
catastrophic clash): outputs/cluster/armf_intcoord.md.

**HONEST SCOPE of both objective-1 milestones (codec + propagator transfer):** ligand-SIZED
molecules (8-90 heavy atoms), FIXED topology (no bond changes), NANOSECOND MISATO trajectories that
are decorrelated frame-to-frame (equilibrium ensembles, not dense dynamics). The propagator-
transfer result is an EQUILIBRIUM-distribution transfer (marginals/coupling/non-Gaussianity);
ligand dynamics are untestable on MISATO. These are the first objective-1 movements, not general-
reactive-molecule generality.

## NEGATIVE (2026-08-05): MISATO frame counts cannot support the arbitrary-L measurement at L=64

Phase-1 arbitrary-L scaling test (N atom tokens -> L=64 latent slots -> N atom outputs) needs a
per-system PCA-L ceiling with a temporal split. **MISATO has exactly 100 frames/complex** (uniform,
measured over 400 complexes). Temporal split 80/20 -> PCA-64 is fit on ~80 frames, i.e. 64/79 of
the frame rank. G7 pre-check (armf_phase1_g7precheck.py), held-out FVE, temporal split:
- PCA-64 **train** FVE 0.94-0.98 (rank-SATURATED); **held-out** FVE only 0.33-0.61; train-vs-held-out
  gap **+0.31 to +0.69**. Signal rank90 = 40-56 (comparable to 64, not << it).
=> the L=64 ceiling is a rank artifact on MISATO; the deficit-ratio read is void -- the same error
that made the earlier L=64 extrapolation wrong.

**RULE (general, any corpus): L must stay below ~30% of usable rank**, where usable rank = frames
surviving the temporal split. Below 20% is solidly clean; 20-30% is the deliberate edge; above ~50%
the PCA ceiling is frame-limited and the deficit-ratio read is void. MISATO is not dead as a
corpus -- only for large L.

MISATO instantiated (100 frames/complex, 80/20 split -> 79 usable):
- L=12 -> 15% of rank, clean
- L=24 -> 30% of rank, deliberate upper slope-test point
- L=64 -> 81% of rank, VOID (the G7 negative above)

Phase-1 MISATO run therefore uses **L in {12, 24}**. Two L values guard the low-L confound: matching
deficit-ratio-vs-N slopes mean the slope is a property of the addressing scheme; diverging slopes
mean bottleneck saturation and the single-L version would have fooled us.

**Corpus switch for Phase 1: mdCATH.** 500 frames/replica (5 temperatures x 5 replicas/domain) ->
400 train frames, L=64 at 16% of frame rank, G7 clean. Heavy-atom spread 11.3x (342-3869, med 904)
> the 4x floor, so the 1.3x/2x deficit-ratio thresholds are meaningful. Caveats on record: only 28
domains; high-N tail is THIN (3 domains in 2-4k heavy, 0 above 3869); single proteins, not
complexes; N tops at ~3.9k heavy (not 8k). The scaling axis is 11x but pinned low-mid-N.

### Corpus sizing for the arbitrary-L scaling axis (measured 2026-08-05)

**N-AXIS CONVENTION: ALL-ATOM including hydrogens** -- model input, PCA ceiling, ANM baseline and
the regression N-axis are all counted this way (what the model consumes; what the 1M-atom objective
counts). NOTE: earlier ligand/protein codec work used HEAVY-atom counts; the two are not mixed
silently. ANM baselines are skipped above 10k atoms (eigensolver cost) and reported NaN, not faked.

**MISATO (all-atom):** min 673, median 4,977, **max 40,530**. Availability at stride-2 (8,486
complexes scanned): (700,1k) 18 | (1k,2k) 410 | (2k,4k) 1,970 | (4k,8k) 3,935 | (8k,16k) 1,717 |
(16k,32k) 416; 434 complexes >=16k, 18 >=32k. Realized bucket span ~20x -- ample power for the
1.3x/2x deficit-ratio thresholds. **MISATO is the only corpus here with real N-range.**

**mdCATH:** 5,398 domains, 500 frames/replica x 5 temperatures x 5 replicas. File size is a VALID
atom-count proxy -- calibrated on all 28 local domains, bytes = 327,232*atoms - 7.84e6, **R^2 =
0.9944**, residual mean 3.2% / max 17.4% (=> frames/temps uniform corpus-wide). **Projected max
~7,520 all-atom (~3,685 heavy); ZERO domains above 8,000 all-atom at any download volume.**
=> mdCATH cannot supply a high-N arm. Under the all-atom convention its 7.5k ceiling sits inside
MISATO's (4k,8k) bucket, so its role is a **frame-rich replication at low-to-mid N** (useful to
check the deficit-ratio trend is not a 79-frame artifact), NOT the high-N arm.
**mdCATH stores `element` at domain level** (per-atom array beside `coords`) -- static atom identity
is read directly, no derivation and no ordering-desync risk.

### SCOPE LIMIT on the arbitrary-L result (written BEFORE the result exists, 2026-08-05)

Measured PCA ceilings on MISATO held-out frames (temporal split, all-atom), median over 3
complexes/bucket: PCA-12 0.50/0.37/0.28/0.21/0.21/0.24 and PCA-24 0.59/0.45/0.34/0.26/0.24/0.27
across buckets (700,1k)...(16k,32k); signal rank90 rises 29 -> 57 with N.

**The displacement field over 100 frames of ns-scale MISATO MD is genuinely low-dimensional
(~24-57 modes carry it).** Whatever the arbitrary-L test shows about L-independence therefore holds
**FOR THAT REGIME ONLY**. It says nothing about multi-millisecond trajectories, where the accessible
conformational space is far richer and the intrinsic dimensionality is NOT ~24. **Objective 3
(multi-millisecond timescales) is untouched by this result in either direction** -- a "content-
addressed, arbitrary-L survives" verdict does not license extrapolating L=12-24 to long-timescale
sampling, and an "index-addressing" verdict would not condemn it there either.

**Two measurement properties that constrain how the result may be read:**
1. **The ceiling FALLS ~2.2x across the N range** (0.59 -> 0.27 at L=24). The task genuinely gets
   harder with N (more atoms -> more independent local modes at fixed 79 frames). Raw model FVE
   must therefore fall with N for reasons unrelated to addressing; the deficit RATIO normalises
   this, raw FVE does not. Corollary: the absolute-gap criterion is **lenient at high N by
   construction** (gap cannot exceed a ~0.27 ceiling), so ratio carries the discriminating power at
   high N and absolute gap at low N -- they are not symmetric across the axis.
2. **Per-system ceiling heterogeneity is large and mobility-driven**: ~8% of complexes have a
   dominant large-amplitude motion (e.g. 1CBR, median RMSF 5.05 A vs typical 1.2 A) giving PCA-24
   ceilings of 0.92 vs bucket-typical 0.25 -- 3-4x. Aggregating as ratio-of-means would let the
   outlier draw per bucket manufacture a cross-bucket trend. Reporting therefore uses **paired
   per-system deficit ratios aggregated by MEDIAN** (each system against its own ceiling, so system
   difficulty cancels) with IQRs and a high-mobility count per bucket. Not excluded, just robust.

### CORRECTION (2026-08-05): intrinsic dimensionality scales with MOBILITY, not atom count

The rank90 29->57 vs N 959->23,895 trend on MISATO (apparently ~N^0.21) **does not survive
controls**. Measured on mdCATH with 2,500 frames/domain (28 domains, all-atom, 320K, 5 replicas):
log10(rank90) ~ b*log10(N) + c*log10(medRMSF) gives **b = +0.135 +/- 0.302 (NOT significant)** and
**c = -1.350 +/- 0.245**, R^2 0.858 (vs 0.124 for N alone). The apparent size effect is a mediated
confound: corr(logRMSF, logRank90) = **-0.924**, corr(logN, logRMSF) = -0.31 -- bigger proteins are
more rigid, and rigidity (not size) raises dimensionality. Floppy domain -> one dominant collective
motion -> rank90 15; stable domain -> diffuse thermal motion -> rank90 387, at LARGER N in one case.

Also: rank90 is severely frame-censored (2-3x higher at 400 vs 79 frames; **8/28 domains still
unconverged at 2,400 frames**), so all 79-frame values are lower bounds.

**Consequence for objective 1:** the mechanism is MORE favourable than N^0.21 (dimensionality does
not grow with atom count at fixed dynamical character), but **no numerical 1M-atom latent budget is
licensed** -- 11x measured N range, ~130x extrapolation, and b's CI alone gives 65x uncertainty each
way. Do not quote "rank90 ~125 at 1e6". **Plan latent budget against dynamical complexity, not atom
count**: two systems of equal size differed 26x in required dimensionality here.
Full writeup: outputs/cluster/armf_intrinsic_dim.md

**Methodological (both corpora):** artifact screening on MAX per-atom displacement is invalid --
max-over-atoms grows with N by extreme-value statistics and preferentially drops large systems (it
dropped 7/8 and 8/8 of the top two MISATO buckets). Screen instead on a DETACHED step population
(fraction of atoms above 3x the system's own p99.9). Result: **0 exclusions in either corpus**; no
PBC unwrapping is present in MISATO or mdCATH.

### ARCHITECTURAL DIRECTION (record, do not implement): ORDERED / NESTED LATENT CODE

Earned by the mobility result above: intrinsic dimensionality spans **26x at fixed N** (rank90
15-387 across mdCATH domains of comparable size), and it is predicted by mobility, not atom count.
A single fixed L is therefore **over-provisioned for floppy systems and under-provisioned for rigid
ones simultaneously**. Sizing L for the worst case spends inference compute on most of the corpus
that the corpus does not need -- which cuts directly against **objective 4 (single-GPU inference)**.

Direction: an **ORDERED / NESTED latent code** -- variance-ordered slots (a variance-ordering loss)
or Matryoshka-style nesting -- so that ONE trained model serves variable L at inference **by
truncation**, with L chosen per system from its dynamical character rather than fixed globally.
This also composes with the codec/propagator stack already built, since both consume a latent
vector whose leading components would carry the most variance by construction.

Not implemented, not scheduled. Recorded so the Phase-2 grid (skip connections, denoising objective,
local-frame coordinates -- all remedies for latent UNDER-utilisation) is not confused with this,
which is a remedy for latent MIS-allocation across systems. Measure before prescribing either.

### METHODOLOGICAL RULES carried forward (both earned by errors made here)

1. **`max`-over-atoms is an extreme-value statistic**, so any FIXED threshold on it is an
   N-correlated filter BY CONSTRUCTION -- larger systems draw more extreme values and get dropped
   preferentially. Screen on within-system relative criteria (e.g. a detached population above 3x
   the system's own p99.9), never on a global max threshold.
2. **An underpowered null is not evidence of no effect.** A drop-rate-vs-N regression returned
   p=0.58 ("uniform in N") while the raw counts showed 7/8 and 8/8 of the two largest buckets being
   dropped. The raw counts were right. Same failure shape as the n=1 GNM-vs-B r=0.38.

### G8 CONVERGENCE GUARD (added 2026-08-05) -- mandatory, same standing as G1/G4/G6/G7

A fixed step budget can produce a FALSE "index-addressing" verdict. If large-N systems converge more
slowly than small-N ones -- the DEFAULT expectation, since more atom tokens means more
cross-attention work per system -- the deficit ratio grows with N for pure OPTIMISATION reasons and
would be misread as the architecture failing to reach 1M atoms.

**G8:** train to a PLATEAU, not a step count. Patience-based early stop on held-out FVE; every
bucket must show <1% relative improvement over the final 25% of steps; a bucket that has not
plateaued is **VOID FOR THAT BUCKET** and excluded from the regression rather than included.
Steps-to-plateau is reported PER BUCKET and regressed on N.

**Steps-to-plateau vs N is a finding in its own right, belonging to objective 4:** training cost
scaling with system size is exactly what "spend training compute to buy inference efficiency" must
be priced against. Not a footnote.

Related sampler fix found while implementing this: FRAMES_PER_STEP was memory-budgeted
(`300000//N`), giving small systems 80 frames/step and large systems 13 -- the same
undertraining-at-high-N artifact G8 exists to catch, baked into the sampler. Now CONSTANT across N.

Also fixed in the same pass: the model was trained to predict RAW displacement while being scored
against a TRAIN-MEAN-CENTERED denominator, so it had to additionally learn each complex's drift --
work the PCA ceiling gets for free (its reconstruction is mu + projection). Predicting ~0 against a
centered denominator yields FVE ~ -1, which is exactly what the first run produced. Input, target
and metric are now all centered, making the model's task identical to PCA's.

### DESIGN TARGET (2026-08-05): L=1, DM as the capacity axis -- "one global latent per frame,
### width independent of atom count"

**An objective-3 measurement has directly specified an objective-1 design parameter.** This is the
first time that has happened in the project, and it is worth naming: the rank90 saturating asymptote
(median **168**, range **13-500**) does NOT track atom count -- the mobility-controlled N-exponent CI
spans zero -- it tracks MOBILITY. That measurement sizes this architecture. DM=256 covers the median
system; DM=512 covers the observed tail. Hence the sweep DM in {16, 64, 256, 512} at L=1.

> ***AMENDED (INBOX 002): rank90 NO LONGER SIZES THIS ARCHITECTURE.*** 168 is an IN-SAMPLE rank90 and
> therefore a FLOOR on the median system's true dimensionality; "DM=256 covers the median, DM=512
> covers the tail" are both **coverage claims read off a lower bound**, so neither is established.
> The "N-exponent CI spans zero" clause is separately SUPERSEDED (line 1439: b = +0.101 +/- 0.021).
> The DM grid {16, 64, 256, 512} is KEPT -- but it is now a **probe, not a bracket**: it is no longer
> claimed to contain the answer from above. `armf_atlas_dm.py` carries the pre-registered extension
> rule for exactly this: if DM=512 still uses >=50% of its width, the sweep is CENSORED AT ITS OWN
> TOP, the saturating DM is another floor, and the grid must be extended to 1024 before any width
> answer is reported.

**Statement of the objective-4 argument in one line:** L=1 x DM=256 is 256 floats per frame for a
system of ANY size; for a 1M-atom system that is 3e6 coordinates -> 256 numbers (a ~11,700:1
compression). Always report DM alongside L -- "one latent token" is otherwise read as "one number",
and L=1 x DM=64 is a 64-DIMENSIONAL code.

**Ceilings (three, because PCA-1 is the wrong one):** PCA-DM (unconstrained, per system) --
**but RANK-VOID above DM=23 on MISATO's 79 train frames**, the same G7 failure that killed L=64, so
only DM=16 has a valid PCA ceiling there; **ANM-DM** (zero-parameter, structure-predicted, and
crucially FRAME-INDEPENDENT so valid at every DM) is the PRIMARY and honest comparator, since the
decoder receives static geometry and can learn a basis from it -- L=1 x DM=64 is functionally
"learned ANM with 64 coefficients"; and the graph codec at matched capacity. NOTE on the third: the
graph codec is a LIGAND torsional codec (8-90 heavy atoms) and is not defined on 700-32,000-atom
protein complexes, so it is reported as context on its own domain rather than as a matched ceiling
here -- stating that rather than manufacturing a comparison.

**G6 (permutation) carries the most weight at L=1:** with a single slot there is no routing to
learn, so a permutation failure would mean the ENCODER index-addresses -- a different and more
serious problem than slot assignment.

**THREE LIMITS, on the record BEFORE results:**
1. **256 floats cannot reconstruct 3M coordinates.** It reconstructs the part PREDICTABLE FROM
   STATIC STRUCTURE. The target is achievable exactly to the extent dynamics are low-dimensional
   given structure -- which is what rank90 measured, and why the number is 168 and not 3N.
2. **The 26x dimensionality spread across systems at fixed N** means a fixed DM is over-provisioned
   for floppy systems and under-provisioned for rigid ones simultaneously -- the argument for the
   ordered/nested latent code already recorded (one trained model, truncatable at inference).
3. **rank90 was measured over ns-to-us windows.** The ms regime remains open, and specifically open
   for rare states that variance-weighted PCA cannot see. Sizing DM from rank90 sizes it for the
   SAMPLED regime, not for multi-millisecond dynamics.

### PRE-REGISTERED (before results): what the n=276 scale-up CAN and CANNOT decide

Held-out n goes 25 -> ~69, so the slope CI half-width goes 0.125 -> **0.075**. Mapping the
pre-registered SPAN thresholds onto the regression slope (baseline ratio 0.27, N range 1.54 decades):

| threshold | implied slope | resolvable at n=69 held-out? |
|---|---|---|
| 2.0x = "INDEX-ADDRESSING" | +0.175 | **YES** -- excluded even if the point estimate stays at +0.099 (CI upper 0.174) |
| 1.3x = "ARBITRARY-L SURVIVES" | +0.053 | **NO** -- the half-width alone (0.075) exceeds it, so it is unconfirmable at ANY point estimate |

**The run is therefore ASYMMETRIC by construction: it can falsify the failure mode but cannot
certify the success mode.** That is a real result and must be written as such, NOT as another
no-verdict. Confirming "arbitrary-L survives" needs half-width < 0.053, i.e. **~141 held-out
(~565 sampled systems)** -- recorded as the cost of that answer.

### DENSITY-MATCHED PAIRS -- the sharpest diagnostic, and it runs against the density hypothesis

Two pairs hold corpus availability roughly fixed while varying N by ~8-16x. From the n=98 run:

| pair | availability | medN | ratio |
|---|---|---|---|
| A | 410 vs 416 | 1,444 vs 22,443 | 0.27 vs **0.52** |
| B | 1,970 vs 1,717 | 2,934 vs 9,639 | 0.18 vs **0.30** |

Both show the higher-N member worse at matched density, which argues the density hypothesis does
NOT explain the effect and an N effect survives density control. This does not depend on the
regression, so it is the cleanest positive evidence available. **Revision to the earlier "U-shape"
reading:** the left arm (N=959, ratio 0.50) sits in the ONLY genuinely density-starved bucket
(18 available, 12 sampled), so the U is plausibly a density artifact on the left glued to a real
N effect on the right (0.18 -> 0.52 across density-matched buckets). At n=276 these buckets carry
~11-12 held-out each; both pairs are to be reported with CIs.

### G8 WATCH AT DM=512
MAXSTEPS was raised 30k -> 50k because 3/6 buckets voided on the cap at L=24. DM=512 at L=1 is a
larger latent than anything trained so far: **if it voids on the cap, raise and rerun that arm
rather than reading a capped bucket.** Report steps-to-plateau vs DM alongside steps-to-plateau vs
N. If convergence cost scales with DM but not with N, that is a SECOND objective-4 number and it
prices the width choice directly.

### PRE-REGISTERED THREE-WAY READ on the L=1 arm (supersedes the earlier two-way framing)

**CORRECTION to what was written earlier in this file and in armf_phase1_dm.py:** an N effect at
L=1 was described as a "CAPACITY limit". That is wrong and contradicts a measurement already in
hand -- rank90's mobility-controlled N-exponent CI spans zero, and TICA dimensionality is flat and
uncensored. **The physics says intrinsic dimensionality does not grow with atom count, so a
fixed-width code suffices.** Capacity is therefore already excluded as an explanation.

> ***RETRACTED (INBOX 002). CAPACITY IS NOT EXCLUDED.*** This paragraph is the single most
> load-bearing inference the width chain ever carried, and both of its premises have failed.
> (1) "The N-exponent CI spans zero" was **superseded** at line 1439: b = +0.101 +/- 0.021 is
> clearly positive, and is itself a censored lower bound. (2) The whole argument rests on rank90 as
> an ESTIMATE of intrinsic dimensionality, and rank90 is a **FLOOR** -- measured in-sample (its own
> modes explain a median of 74.7% of held-out variance, not 90%, with the shortfall GROWING with N)
> and censored at large N (32.3% of usable rank at the top).
> **Consequence: "not capacity" may no longer be asserted a priori for outcome B below.** Capacity
> returns as a live hypothesis, and it is now tested EMPIRICALLY rather than excluded by argument --
> it is the "one-token information limit" row of the 004c fan-out, whose distinguishing test is the
> DM sweep itself (`armf_atlas_dm.py`): if FVE is still rising at DM=512, the limit is information,
> not addressing. The outcome table below is amended accordingly.

| outcome | observation | localisation | fix |
|---|---|---|---|
| **A** | N effect at L=12/24 but NOT at L=1 | slot assignment / routing | addressing mechanism |
| **B** | N effect at L=1 as well | **POOLING / BROADCAST PATHWAY** -- encoder aggregating N tokens into a fixed code, or decoder broadcasting one code back to N atoms. ~~NOT capacity.~~ **AMENDED (INBOX 002): capacity is NOT excluded** -- the "rank90/TICA flat in N" premise is retracted above, so B must be SEPARATED from a one-token information limit by the DM sweep (FVE still rising at DM=512 => information-limited, not pooling) before any aggregation redesign is proposed. | first distinguish capacity via the DM sweep; only then aggregation architecture: hierarchical pooling, deeper cross-attention, relative-position conditioning |
| **C** | no N effect anywhere | arbitrary-L holds, subject to the stated power limit | -- |

The outcome must be labelled A/B/C explicitly in the report. **If B, do not write "capacity limit"**
-- that phrasing contradicts an existing measurement and would send the next round at the wrong fix.

### PRE-COMMITTED DECISION TREE (so it is not relitigated after results)
- **Outcome C:** do NOT immediately spend 565 sampled systems to certify <1.3x. Excluding
  index-addressing is enough to proceed; the certification run is scheduled only if something
  downstream depends on the tighter bound.
- **Outcome A:** next run is the ADDRESSING MECHANISM, not more systems.
- **Outcome B:** next run is the AGGREGATION ARCHITECTURE, not more systems.

In all three cases the follow-up is **mechanistic, not statistical**. The price of statistical
certainty is now known (~565 sampled systems) and it is to be paid deliberately, not by default.

### RUN-KILLING BUGS FOUND IN THE n=276 SCALE-UP (2026-08-05) -- all four L=1 arms VOID

**1. Early stopping killed every L=1 arm below competence.** A plateau detector cannot distinguish
"flat because converged" from "flat because not yet taken off" -- identical signal. All four L=1
arms (DM=16/64/256/512) sat at FVE ~0.000 and were early-stopped at 8k-17k steps. G8 then reported
4/6 buckets as `plateau YES, status ok, ratio 1.00` -- **a dead model certified as converged,
producing exactly the index-addressing signature.** Only G4 (base > 0.05) caught it, at arm level.
**G8 tests CONVERGENCE, not COMPETENCE.**
FIX APPLIED: early stopping does not activate until some bucket exceeds held-out FVE 0.05
(`COMPETENCE`); below that, train to MAXSTEPS regardless of curve flatness. TAKEOFF STEP (first
crossing of 0.05) is now recorded per bucket and per arm, to be regressed on N and on DM.
Steps-to-TAKEOFF is a different quantity from steps-to-PLATEAU and has never been measured.

**2. L=1 was untestable as built -- a decoder degeneracy, not a finding.** With L=1, decoder
cross-attention has ONE key, so softmax is identically 1.0 for every query: `dec(q,lat,lat)` returns
the SAME vector for every atom and the latent enters only as a uniform additive shift before
LayerNorm -- the weakest possible conditioning. L=12 trains because it has 12 keys to attend over.
**Any N effect from the old L=1 arms would have reflected this bug, NOT the pooling/broadcast
pathway** -- outcome B must not be read from them.
FIX APPLIED: FiLM conditioning in the decoder (per-channel scale+shift from the pooled code), which
works even when attention is constant.

**3. The powered L=12/L=24 arms are UNDERTRAINED.** All six buckets at L=12 are G8-VOID with
rel-improve +0.05 to +0.16 at 50,000 steps -- still climbing. MAXSTEPS raised 50k -> 150k.
Steps-to-plateau was flat in N (31k-40k), so the undertraining is NOT N-correlated.

**Signal that survives (undertrained, so provisional):** at n=276 the U-shape is GONE and the
deficit ratio rises monotonically with N at both L: L=12 gives 0.24/0.19/0.23/0.27/0.30/**0.40**
and L=24 gives 0.25/0.28/0.27/0.31/0.31/**0.38** across medN 906->19,702. **Both density-matched
pairs confirm at power**: A (avail 410 vs 416) 0.19 vs 0.40; B (avail 1970 vs 1717) 0.23 vs 0.30.

### CG-ANM IS NOT USABLE AS AN N-COMPARISON BASELINE -- retracted before use

The earlier "CG-ANM validates within ~0.01" claim was based on TWO systems at N=2,627 and 2,621,
i.e. effectively one size point, and it was wrong. Across 160 systems the median gap at k=64 is
**-0.059** (CG-ANM systematically OVERSHOOTS all-atom ANM), an order of magnitude larger and in the
direction that matters. **And the gap is N-dependent**: -0.122 at N~965, -0.058 at ~1,840, -0.046 at
~3,280, -0.043 at ~4,607; slope **+0.110 +/- 0.039** vs log10(N), significant. Extrapolated to
N=22,755 the gap flips sign (+0.043), which is meaningless.

Diagnosis: QR-orthonormalising rigid-residue modes in all-atom space does not approximate all-atom
ANM -- it produces a DIFFERENT, better-aligned subspace (rigid-residue collective motion explains
more held-out variance at matched k), whose bias varies systematically with N. A baseline whose bias
moves with N cannot support an N-trend comparison, which is the only thing it was introduced for.
**Consequence: PCA-k is the primary ceiling where rank-valid (k <= 23, available at ALL N, no
N-dependent bias). Above k=23 no valid per-system ceiling exists on MISATO's 79 train frames --
state that as a limitation rather than substituting a biased baseline.**

### RELAUNCH DESIGN (2026-08-05): axes split by corpus, architecture held constant

**1. FiLM AT EVERY L.** The decoder uses cross-attention over the L tokens AND FiLM from the pooled
latent, identically at L=1, 12 and 24. If L=1 used FiLM while L=12/24 used attention alone, the
A-vs-B read would confound LATENT COUNT with CONDITIONING MECHANISM. Architecture constant, only L
varies.

**2. AXES SPLIT BY CORPUS -- each used for what it can support.**
| axis | corpus | why |
|---|---|---|
| **N-axis** (L in {1,12,24} at DM=64) -- the A/B/C run | **MISATO**, n=276, 79 train frames | PCA-k rank-valid at k=1/12/24; wide N range (906-19,702) |
| **CAPACITY axis** (L=1, DM in {16,64,256,512}) | **mdCATH**, 2,500 frames -> 2,000 train | PCA-k valid to k~600, covering DM=512. On MISATO PCA-k is void above k=23, so the deficit ratio for DM=64/256/512 is simply UNDEFINED there |
**The two axes are measured on DIFFERENT CORPORA. Absolute numbers must NOT be cross-compared** --
mdCATH PCA-16 runs 0.48-0.65 vs MISATO PCA-12 ~0.10-0.33. **The cause is MOBILITY, not frame count.**
(An earlier draft of this line blamed frame count; that has the sign backwards -- more frames sample
MORE dynamics, so PCA-k at fixed k captures LESS. Frames push mdCATH DOWN relative to MISATO; it is
higher anyway, so frames work AGAINST the observed direction and cannot be the mechanism.)
MEASURED, identical all-atom Kabsch convention, 79 frames each: **MISATO medRMSF 0.904 A (bound
complexes, n=276) vs mdCATH 1.493 A (unpartnered single domains, n=28) = 1.65x floppier,
p=2e-10**. Time-matched control (mdCATH's first 10 frames ~ 10 ns = MISATO's ENTIRE window) gives
1.502 A, ratio 1.66x -- unchanged, so this is a corpus effect, not an observation-window effect.
Floppy concentrates variance into fewer modes (c<0 below), which raises PCA-k at fixed k.

**3. PLATEAU CRITERION TIGHTENED.** The old test ("<1% relative improvement over the final 25%")
fires on a genuine slow monotone climb -- which is exactly how the last run reported steps-to-plateau
of 31-40k while every curve was still rising at 50k. Both cannot be true. New criterion: the
**absolute held-out FVE slope over the final 20% must be statistically indistinguishable from zero**,
and the **raw tail of every curve is printed alongside the verdict**. This matters because
"steps-to-plateau flat in N" is load-bearing for arguing the N-trend is not optimisation-driven, so
it needs a criterion that slow monotone growth cannot satisfy.

**4. RETAINED:** competence gate (early stopping cannot fire below held-out FVE 0.05) and takeoff-step
recording, to be regressed on N and on DM.

**Ceiling policy:** PCA-k is the SOLE ceiling, used only where rank-valid. Above the rank limit no
per-system ceiling is reported and the limitation is stated -- rather than substituting CG-ANM, whose
validation showed a -0.059 median bias that is itself N-dependent (slope +0.110 +/- 0.039).

### CROSS-CORPUS VALIDATION of the mobility law -- and TWO CORRECTIONS it forces

Fitting log10(rank90) ~ b*log10(N) + c*log10(medRMSF) at a MATCHED 79-frame budget on both corpora:

| corpus | n | b(N) | c(RMSF) | R^2 |
|---|---|---|---|---|
| MISATO (bound complexes) | 276 | **+0.101 +/- 0.021** | **-0.887 +/- 0.075** | 0.743 |
| mdCATH (single domains) | 28 | -0.057 +/- 0.129 | **-0.950 +/- 0.123** | 0.917 |

**The mobility law REPRODUCES on an independent corpus**: c = -0.887 +/- 0.075 (MISATO) vs
-0.950 +/- 0.123 (mdCATH) overlap tightly. Free cross-corpus validation of the finding that sizes DM.

**CORRECTION 1 -- c depends on FRAME BUDGET.** The headline c = -1.350 +/- 0.245 was measured on
mdCATH at 2,400 frames. At 79 frames the same corpus gives -0.950 +/- 0.123; the CIs do NOT overlap.
Censoring at 79 frames compresses rank90's dynamic range and attenuates the exponent toward zero, so
**-1.35 (uncensored) is the better estimate of the true exponent, and -0.9 is the attenuated
frame-matched value.** Cross-corpus comparisons must be made at matched frame budget; only the
matched numbers above license the "reproduces" claim.

**CORRECTION 2 -- dimensionality DOES grow with atom count, weakly.** The earlier conclusion
"b's CI spans zero, so intrinsic dimensionality does not grow with N" was measured on mdCATH at
n=28 and was UNDERPOWERED. On MISATO at n=276, **b = +0.101 +/- 0.021 is clearly positive**:
rank90 ~ N^0.10. Over the measured range (906 -> 19,702) that is 1.36x; extrapolated from medN 4,415
to 1e6 atoms it is ~1.7x. Small, but real and no longer dismissible.
**This does NOT undermine the deficit-ratio logic**: rank90 growth is already absorbed by the PCA-k
CEILING (which falls with N), and the ratio measures the model's shortfall RELATIVE to its own
ceiling. But it does weaken the blanket premise "the physics excludes capacity" used in the
three-way read -- capacity grows weakly with N, and outcome B's wording must say that the deficit
ratio (not raw FVE) is what isolates the pooling/broadcast pathway from it.

### SCOPE LIMIT ON THE CAPACITY AXIS (mdCATH is the floppy end)

mdCATH is 1.65x floppier than MISATO. With c ~ -0.9, that implies mdCATH systems need
1.65^-0.9 ~ **0.65x the dimensionality** of MISATO complexes at matched N -- i.e. **DM sized on
mdCATH is sized for the FLOPPY end and would UNDER-PROVISION bound complexes by roughly 1.5x.**
Per the 26x spread across systems this is exactly the direction that needs LESS width, so the
capacity axis gives a LOWER BOUND on the DM required for deployment on protein-ligand complexes.
Recorded before the capacity results exist so the number is not over-read later.

### IS b ATTENUATED TOO? -- paired frame-budget test on mdCATH, and the WIDTH CHAIN

Censoring was shown to attenuate c (-0.950 at 79 frames -> -1.350 at 2,400). There is no reason it
would spare b, and b governs whether latent width must scale with system size. Paired test on the
SAME 28 mdCATH domains at six budgets (each system its own control):

| frames | b(N) | c(RMSF) | max rank90 (% of cap) |
|---|---|---|---|
| 79 | -0.057 +/- 0.129 | -0.950 +/- 0.123 | 68% |
| 200 | -0.052 +/- 0.197 | -1.282 +/- 0.185 | 55% |
| 400 | -0.004 +/- 0.230 | -1.350 +/- 0.206 | 48% |
| 800 | +0.042 +/- 0.232 | -1.384 +/- 0.192 | 30% |
| 1600 | +0.063 +/- 0.278 | -1.370 +/- 0.223 | 20% |
| 2400 | +0.246 +/- 0.404 | -1.313 +/- 0.455 | 16% |

**THE MULTIPLICATIVE-FACTOR METHOD FAILS FOR b, and its output must not be used.** b CROSSES ZERO
across the ladder, so the ratio 2400/79 = -4.33x divides through a sign change. Propagating it gives
b = -0.438 (dimensionality SHRINKING with atom count) and a 16-dimension width chain -- nonsense.
Both discarded.

What the data does support:
1. **c's attenuation factor is well determined: 1.38x**, independently reproducing the known
   -0.950 -> -1.350 steepening.
2. **b is UNMEASURABLE on mdCATH at any budget** -- its CI spans zero at all six. n=28 is too few for
   the N axis; this is the same underpowering that produced the original "b spans zero" error.
3. **The DIRECTION supports the attenuation hypothesis**: b drifts monotonically upward with budget
   (-0.057 -> -0.052 -> -0.004 -> +0.042 -> +0.063 -> +0.246), consistent with attenuation toward
   zero at low budgets. So **MISATO's b = +0.101 +/- 0.021 (79 frames) is a LOWER BOUND on the true
   exponent.** It cannot be de-attenuated directly: MISATO has only 100 frames, so a
   higher-budget MISATO fit is impossible.
4. Provisional correction using c's factor (assumes censoring attenuates both coefficients of the
   same regression similarly -- defensible, not proven): **b ~ 0.14**.

### WIDTH CHAIN -- ***RETIRED AS A SIZING INSTRUMENT. EVERY ROW BELOW IS A FLOOR.*** (INBOX 002)
> **DO NOT SIZE DM FROM THIS TABLE.** Two independent biases, both pushing the SAME direction, make
> every number here a lower bound of unknown tightness, and neither can be fixed with available data.
> Superseded by the codec's own held-out FVE-vs-DM curve (`armf_atlas_dm.py`). Retained for audit
> only. See "WIDTH CHAIN IS A FLOOR" below for the measurements.

(anchor: mdCATH asymptote 168 at medN 1,804; floppy->bound divide by 0.65)

| b | N^b to 1e6 | dims for a 1e6-atom BOUND complex |
|---|---|---|
| 0.101 (raw, censored -> LOWER bound) | 1.89 | **>= 489** |
| 0.139 (x1.38 c-derived, provisional) | 2.41 | **>= 622** |
| 0.20 | 3.54 | >= 914 |
| 0.30 | 6.65 | >= 1720 |

**Design sensitivity -- ITSELF VOID.** It formerly read: "at b ~ 0.10-0.14, DM ~ 500-620 covers a
1M-atom bound complex and DM=512 sits right at the edge." **That conclusion is withdrawn.** It
compared a DM budget against a quantity now known to be a floor, so "DM=512 sits right at the edge"
is unsupported in the only direction that matters -- the true requirement can only be HIGHER, never
lower. What survives is the qualitative sensitivity: the single-global-latent claim needs
qualification rather than a wider DM once the requirement passes ~900, and nothing here bounds it
below that.

**CAVEATS ON THE SAME LINE:** 2.5-decade extrapolation (1,804 -> 1e6 atoms); ns-us regime only;
variance-weighted so rare states are excluded (the coverage question is open, not answered); floppy
-> bound factor 0.65 derived from a 1.65x RMSF ratio via c ~ -0.9; **b itself is a censored lower
bound**; and now **the 168 anchor is a lower bound too** -- it was measured in-sample on mdCATH,
which never had its n_eff checked.

### b-EXPONENT MEASUREMENT: design, staged (2026-08-05)

Blocker was n=28, not mdCATH -- the repo holds **5,398 domains, 3.61 TB**. Streaming 700 stratified
domains: download -> compute ladder -> write spectra -> DELETE h5. Peak disk ~2 GB/worker; retained
output a few MB. **Measured bandwidth 25.0 MB/s** and **42 s/domain end-to-end** in pilot, so ~2-3 h
on 4 workers; bandwidth is NOT the binding constraint and n=700 stands (no cut to 300 needed).

**bytes->atoms calibration MEASURED** on the 28 local files: atoms = 3.039e-06*bytes + 36.3,
**R^2 = 0.9944**, 321 KB/atom, median error 2.7%. Stratified selection: 700 domains, N 710-7,678
(10.8x), bins 140/140/140/140/90/50 -- the entire sparse >6000 bin taken, since that is where a
slope gets its leverage.

**320K ONLY.** 28/28 domains are denatured at 450K and 20/28 at 413K, and denatured systems have LOW
rank90 (unfolding collapses variance into one dominant mode). Pooling temperatures would corrupt b
through the atom-count term AND c through the mobility term it conditions on.
**SCOPE LIMIT: b is measured for FOLDED NATIVE-STATE dynamics.**

**PER-REPLICA, NOT CONCATENATED.** rank90 is computed per replica and averaged within a domain, with
the between-replica spread kept as a per-domain error bar -- concatenation would let replica count
and length (which VARY: pilot saw replen 430-500, F 2190-2500) drive the number.
Two checks now permanent in the output:
- **Budget-matched replica-offset inflation is SMALL: x0.96-1.21, median ~1.05.** Spreading the same
  budget across replicas raises rank90 ~5%. So the earlier concatenated mdCATH numbers (asymptote
  168, window and temperature sweeps) are higher chiefly because they used MORE FRAMES -- a window
  effect, not an offset artifact. The anchor does not need a large correction.
  (The naive test -- cat[:nf] vs per-replica -- tests nothing, since cat[:nf] IS replica 0 for
  nf <= replica length. The valid test holds the budget fixed and spreads frames across replicas.)
- **Between-replica spread**: within-domain CV 10-17% vs a ~6x between-domain range, so between-domain
  variation dominates and the CIs are not concealing replica noise.

Top per-replica budget is 400 frames (replicas are ~430-500). Guard retained: if b(79) sits near
zero the analysis reports an ADDITIVE shift, never a multiplicative factor -- the failure that
produced -4.33x and b=-0.438 cannot recur.

### PRIMARY LADDER FLIPPED: concatenated-at-2,000, not per-replica-at-400

**Rank usage decides it, and the two errors are not comparable in size:**
| ladder | usable rank | rank90 position | status |
|---|---|---|---|
| per-replica 400 | ~399 | tail near ~50% of rank | **past the 30% void line** |
| concatenated 2,000 | ~1,999 | ~10% median, ~19-24% tail | clean / edge |

**Censoring is the error that DEMONSTRABLY flips b's sign** (-0.057 at 79 frames -> +0.246 at 2,400).
**Concatenation is a measured ~3-5% median inflation.** Trading a quantified small bias for an
artifact known to invert the answer is the wrong trade. Concatenated-at-2,000 is PRIMARY; the
per-replica ladder is retained and reported as the ARTIFACT CHECK.

**The deciding test -- inflation factor vs N -- run on the 28 local domains:**
| budget | median inflation | slope vs log10(N) | r | p |
|---|---|---|---|---|
| 200 frames (n=17) | x1.029 | -0.016 +/- 0.281 | -0.031 | 0.905 |
| 400 frames (n=17) | x1.000 | +0.178 +/- 0.276 | +0.334 | 0.190 |

Point estimates say **INTERCEPT-ONLY** (concatenation shifts the level, not the slope, so b is
unaffected). **But this is UNDERPOWERED at n=17 and must not be banked**: the 400-frame CI upper
bound (+0.454) would move the inflation factor by ~0.47 across the N range, biasing b by roughly
**+0.077 -- comparable to b itself**. So the test is run again inside the n=700 job, where it is
reported with its CI and the implied bias on b at both the point estimate and the CI upper bound.
The primary/secondary choice above rests on the censoring argument, which is quantified; the
intercept-vs-slope question is settled at n=700, not asserted now.

Width-chain anchor: apply the measured median inflation (~1.03) to the 168 asymptote if the n=700
test confirms intercept-only.

### REPLICA-COUNT SWEEP: measure the artifact optimum instead of arguing it

The two artifacts move in OPPOSITE directions with join count -- concatenation inflation grows with
joins, censoring shrinks with total frames -- so there is an optimum and it is measurable. Sweep
joins = 1, 2, 3, 5, reporting per join: total/usable frames, rank90 as % of usable rank (censoring
axis), budget-matched inflation vs 1 replica (concatenation axis), and b with CI.

**Pilot (3 domains) confirms the shape**, e.g. 2ac1A02 (N=3,044):
| joins | frames | rank usage | matched inflation |
|---|---|---|---|
| 1 | 440 | **36.9%** | 1.00 (baseline) |
| 2 | 880 | 25.8% | x1.05 |
| 3 | 1320 | 18.7% | x1.01 |
| 5 | 2200 | **11.2%** | x1.07 |
Rank usage falls monotonically as expected; inflation stays ~1.00-1.07 rather than climbing steeply,
so the censoring axis is the steeper of the two.

**Selection is now made FROM THE CURVE, not by argument:** the analysis picks the lowest-inflation
join among those clearing the 30% rank line, reports b at every join, and explicitly tests whether b
is stable across joins 2-5 -- printing "STABLE: the join choice does not matter, and that stability
IS the answer" or "MOVES WITH JOIN COUNT: concatenation slope-bias detected DIRECTLY" according to
whether the b-spread exceeds the typical CI half-width. This detects the slope-bias directly and is
cleaner than inferring it from the inflation-vs-N correlation. If no join clears the line, it says so
rather than picking one.

**Censoring is SYSTEM-DEPENDENT, so the median would hide it:** the pilot's three domains sit at
4.1% / 22.6% / 36.9% of rank at J=1 -- floppy systems barely use any rank, rigid ones approach
saturation. Selection therefore uses the **p90** of rank usage, not the median, and the full
distribution (median / p90 / max / fraction over the void line) is printed per join. Otherwise a join
that looks clean on average would be picked while censored precisely for the rigid tail, which is
where the slope gets its leverage.

### FOUR SAFEGUARDS ON THE b MEASUREMENT (recorded BEFORE results exist)

**1. NO EXCLUSION OF CENSORED DOMAINS -- the primary fit keeps every domain.** Dropping domains that
fail a rank-usage threshold would be a non-random exclusion CORRELATED WITH N (rigid -> high rank90
-> censored -> dropped, and rigid correlates with large). **This is the same shape as the maxDisp bug
that dropped 7/8 and 8/8 of the top MISATO buckets -- the THIRD occurrence of this pattern**, after
that one and the underpowered drop-rate null. p90 rank usage selects a JOIN COUNT; it is never a
filter on which domains are fitted.
AUDIT RESULT: the primary join-sweep fit keeps every domain that has the given join. One leak found
and now made visible rather than silent -- the secondary LADDER fit uses a paired subset requiring
2,000 frames, which drops SHORT-REPLICA domains, and short replicas correlate with unfolding
(floppy). That exclusion is now quantified in the output (n dropped, medRMSF kept vs dropped) and the
ladder is labelled DIAGNOSTIC ONLY.

**2. RANK-USAGE THRESHOLD SWEEP AT FIXED JOIN.** Fit b on subsets with rank usage below
10/15/20/30/40%, reporting b, CI, n retained, N range retained and median RMSF at each. Stable ->
censoring is not biasing b. Drifting upward as the threshold tightens -> that IS the censoring bias
measured directly, and the trend extrapolates to a corrected b. **DIAGNOSTIC SUBSETTING, NOT AN
EXCLUSION RULE.** Axes are crossed ONE AT A TIME -- joins at fixed threshold, threshold at fixed join
-- since varying both confounds them. Tightening the threshold preferentially retains floppy/low-rank
systems, shrinking the N and mobility ranges, so both are printed and the loss of leverage is visible.

**3. ADAPTIVE PER-DOMAIN JOIN -- held in reserve as the remedy if (2) shows drift.** Choose the join
per domain so every domain clears the same rank-usage target, equalising censoring instead of letting
it track mobility. Cost: join count then correlates with mobility, so inflation correlates with c --
**if used, report c BOTH ways.** This is the right side of the trade, since inflation is x1.00-1.07
while censoring is the artifact that flips b's sign.

**4. DIRECTIONAL PREDICTION, ON RECORD BEFORE THE NUMBERS EXIST: de-censored b should come back
HIGHER than +0.101.** Mechanism: bigger proteins are more rigid (corr(logN, logRMSF) = -0.31), rigid
means higher rank90 (c ~ -0.9 to -1.35), higher rank90 means more censoring -- so censoring truncates
the dependent variable preferentially at HIGH N and FLATTENS the slope. **If b comes back LOWER, that
mechanism is wrong, and that fact is worth as much as the number.**

## PRE-FLIGHT CHECKLIST -- run all four against EVERY new arm before submission

Each family below has fired 3-5 times in this project. They have different signatures and different
fixes, and **the error-discovery rate -- not compute -- is the pace limiter here.** Running these
costs minutes; discovering them one job at a time has cost several runs.

### FAMILY A -- non-random exclusion correlated with a regressor
**Instances:** maxDisp>25A threshold (dropped 7/8 and 8/8 of the top MISATO buckets); the
2,000-frame paired subset (drops short replicas = floppy); download/parse failures under a fixed
timeout (large files fail preferentially -> drops high N).
**Signature:** a fixed cutoff on a quantity whose scale varies systematically with N or mobility.
**CHECK:** for every filter, report **n dropped and the mean of each regressor in kept vs dropped**.
Never apply a fixed ABSOLUTE threshold to a quantity whose scale varies with a regressor --
normalise per system first (e.g. "3x this system's own p99.9", not "25 A").

### FAMILY B -- a count or dimension pinned to its own measurement limit
**Instances:** L=64 at 81% of available rank; the 2 A cutoff at 2399/2400 states; TICA basis varying
with temperature; state count as a fraction of frames; rank90 censoring at high N; steps-to-plateau
pinned to MAXSTEPS.
**Signature:** the reported number tracks the MEASUREMENT'S CEILING rather than the system.
**CHECK:** print every count/dimension as a **PERCENTAGE OF ITS CEILING** next to the raw value.
**Above ~30% it is a BOUND, not a value**, and must be reported as such.

### FAMILY C -- an underpowered null believed
**Instances:** GNM-vs-B at n=1 (0.38 vs a true 0.55); exclusion-uniformity p=0.58 while raw counts
showed 7/8 and 8/8; b's CI spanning zero at n=28; inflation-vs-N at n=17.
**Signature:** a non-significant result read as "no effect."
**CHECK:** never report a null without **the CI and the effect size the CI still permits**. State
explicitly what the upper bound would do to the DOWNSTREAM number (e.g. "permits a b-shift of 0.08,
which moves the 1e6 width chain by 1.6x").

### FAMILY D -- a measurement that cannot express the effect it tests for
**Instances:** the L=1 softmax degeneracy (one key -> uniform output, so the arm measured the
decoder, not the pathway); G8 certifying a dead model as plateaued; the centering mismatch scoring
the model against a denominator PCA got for free.
**Signature:** a configuration in which the quantity under test is STRUCTURALLY unable to vary.
**CHECK:** before any arm, ask **what it would report if the mechanism were perfect, and if it were
absent. If those two answers coincide, the arm is void before it runs.**

### AUDIT OF THE CURRENT ARMS (four new instances found and fixed)
- **A:** download/parse failures were an invisible filter -- large files fail a fixed timeout first.
  Timeout raised to 2400 s and every failure now logged with its projected N; the analysis prints
  failed-vs-kept median N and flags an N-correlated failure set.
- **A:** G8-VOID buckets are dropped from the deficit-ratio regression -- if plateau failure tracks
  N that is a Family A exclusion. Regressor means for kept-vs-void buckets must be reported.
- **B:** steps-to-plateau and TAKEOFF now print as a **% of MAXSTEPS**; near the cap they are bounds.
- **C:** the join-stability verdict is a NULL, so it now states the b-shift it still permits and the
  resulting width-chain factor.
- **D:** the rank-usage threshold sweep can collapse the retained N range, making b unidentifiable
  while returning noise that mimics drift; it now flags when the retained log10(N) range < 0.5.

### FAMILY A, GENERALISED: CONSERVATION OF n -- every silent failure is a filter

The curl timeout was not a filter anyone wrote; it was one the ENVIRONMENT imposed. The same holds
for OOMs, NaNs, h5py read errors, eigensolver non-convergence, missing ceilings, and any system
skipped for any reason. **None of them announce themselves.**

**RULE: every pipeline stage prints n_in -> n_out. Any stage where they differ must characterise the
shortfall against EVERY regressor (N, mobility) before its output is used downstream. No stage
silently passes fewer rows than it received.** Implemented as `conserve(stage, rows_in, rows_out)`,
which prints the kept-vs-dropped median of each regressor and marks *SKEWED* past 25% relative
difference.

**AUDIT HIT -- and it was live in a RUNNING job.** The ANM ceiling returns NaN above ANM_MAXN, a
known N-correlated missing-data pattern. `armf_phase1_dm.py` filtered `if np.isfinite(ratio_anm)`
before the deficit regression, so on mdCATH (top bucket 3,500-8,000 atoms, ANM_MAXN=4,000) **the top
bucket's ratio was computed only from its sub-4,000 half** -- the regression's own N axis truncated
inside its widest bucket. Family A operating through MISSINGNESS rather than through a written
filter, exactly as predicted.

**FIX:** the capacity axis now uses **PCA-DM** as its ceiling. That is why the axis was moved to
mdCATH in the first place -- PCA-k is rank-valid there to k~600, at every DM and **every N, with no
missing-data pattern**. Using ANM there reintroduced a gap PCA does not have. ANM is retained as a
secondary reference where it exists. The running job saves both `pca` and `anm` per row, so the
PCA-based ratio is recomputable offline -- no rerun needed.

### PRE-REGISTERED: THE WEAK-ARM CASE FOR THE A/B/C READ

L=1's G4 base is **+0.072**, barely above the 0.05 competence gate. A model that weak may yield a
deficit-ratio CI too wide to separate flat from sloped -- in which case **A-vs-B discrimination is
underpowered even though the arm is VALID on guards.**

**If that happens, report exactly: "L=1 arm valid but UNDERPOWERED for A/B discrimination", with the
CI and the slope range it still permits.** Do NOT resolve it by defaulting to whichever of A or B the
L=12/24 arms suggest. That would be **Family C on the headline question itself**, and the L=1 arm
exists precisely because the L=12/24 arms CANNOT separate routing from pooling -- borrowing their
answer would assume what the arm was built to test.

**The remedy in that case is a STRONGER L=1 arm** -- more steps, or DM raised so one token carries
more capacity -- **not a softer verdict.**

### FAMILY B RUN ON THE NEW PCA-DM CEILING -- DM=512 is edge-to-void, per domain

Measured on the 28 mdCATH domains actually cached (full replica set, T 1,760-2,500, h 1,408-2,000):

| DM | rank usage (median) | worst | domains VOID (>30%) |
|---|---|---|---|
| 16 | 1% | 1% | 0/28 |
| 64 | 3% | 5% | 0/28 |
| 256 | 13% | 18% | 0/28 |
| **512** | **26%** | **36%** | **10/28** |

The full replica set does put DM=512 at **26% median -- edge but valid** as expected. But **h varies
per domain (1,408-2,000)** because replica lengths vary, so **10 of 28 domains exceed 30% of their
OWN usable rank**. The script's `RANKVOID` was a FIXED `0.30*1999 = 600`, which would have passed
DM=512 for every domain and hidden all ten. **Rank validity is now computed PER DOMAIN**
(`DM <= 0.30*(h-1)`), with per-domain rank usage stored and the median/worst/void-count printed for
every DM arm. DM=256 at 13% is fine either way, as expected.

**Reported BOTH ways, never dropped.** For any DM where some domains are void, the output prints the
ratio over ALL domains and over the rank-valid subset side by side. Dropping the void set would be
**Family A** -- void here means short replicas, which means unfolding, which means floppy, so the
exclusion would correlate with mobility.

### PER-REPLICA DOES NOT APPLY TO THE CAPACITY AXIS -- recorded in BOTH script headers

Concatenation is an artifact **for DIMENSIONALITY measurement**: between-replica structural offsets
add apparent variance directions and inflate rank90. That is why `armf_b_exponent.py` needs the
replica-join sweep.

It is **NOT** an artifact **for PER-FRAME AUTOENCODING**. The AE encodes single frames with no time
dependence, so frames drawn from five replicas are simply BROADER SAMPLING of the same equilibrium
ensemble -- **strictly better training data, not discontinuities**. `armf_phase1_dm.py` therefore
keeps concatenated frames, and both headers now carry the cross-reference so neither gets "fixed"
into the other's convention later. **Different use, different concern.**

### PRE-COMMITTED: THE INCONCLUSIVE-AGAIN BRANCH (do NOT reflexively scale n)

If the deficit ratio again fails to separate flat from sloped, the reflex is "more systems".
**Resist it. Three inconclusive reads on the same statistic is evidence the STATISTIC is weak, not
only that n is small** -- and certainty is priced at ~565 sampled systems, a large bill for an
instrument that has not yet discriminated.

**The deficit ratio is a scalar summary of a subspace comparison, and it discards the structure that
would answer the question.** Two more direct statistics, both now computed INLINE:

1. **EFFECTIVE RANK OF THE MODEL'S RECONSTRUCTIONS vs N.** SVD the model's reconstruction matrix per
   system, take the 90%-variance rank of what the model actually REALISES, compare to that system's
   own rank90.
   - model rank TRACKS system rank90 across N -> the model realises the available dimensionality,
     arbitrary-L holds.
   - model rank SATURATES while system rank90 grows -> **the bottleneck is measured DIRECTLY, in
     dimensions**, not inferred from a ratio of scalars.
2. **SUBSPACE OVERLAP** -- principal angles / Grassmann distance between the model's realised
   subspace and the system's top-k PCA subspace, as a function of N. This says whether the model
   finds the **RIGHT DIRECTIONS**, which the FVE ratio conflates with how much variance it captures.

If both also come back flat, then no N-dependence exists at the available resolution and paying for
565 systems is justified. If they discriminate where the ratio did not, the run is saved and the
instrument is better.

**CORRECTION TO THE PREMISE:** this is NOT post-hoc on the runs finishing now -- `fve()` computes
predictions and discards them, and no model checkpoint was saved either. So the current jobs cannot
support it. Cost of the branch is therefore **one re-run at the SAME n** (cheap), not a scale-up.
Both statistics are now computed INLINE per system, and **`torch.save` of each arm's state_dict is
added**, so they never again require retraining to recover.

**FAMILY B APPLIES TO THE NEW INSTRUMENT TOO:** both ranks are capped by the frame count -- MISATO
held-out is 20 frames, all-frames cap is 99, and system rank90 already runs 29-57 there, i.e. 30-58%
of cap. Model and system rank usage are therefore printed as a % of cap next to the values, and
flagged as BOUNDS above 30%. The instrument is consequently **much cleaner on mdCATH (2,500 frames)
than on MISATO (100)** -- which is worth knowing before reading its MISATO output.

## PERMANENT VERDICT RULE: log10(ratio) ~ log10(N). SPAN AND MAX-MIN ARE RETIRED.

The slope IS the power-law exponent, and a "K-fold change across the range" maps to a threshold that
depends ONLY on the N range -- never on the baseline, so it cannot move between runs:

    threshold slope = log10(K) / log10(N_max / N_min)

**Use the ACTUAL fitted per-system N range, not bucket medians.** For this corpus the per-system
range is **717 - 26,861 = 37.5x, log10 = 1.574** (bucket medians 913-22,755 give 1.396 and slightly
looser thresholds -- the fitted range is the correct denominator):
| K | threshold slope (this corpus) |
|---|---|
| 1.3x -- arbitrary-L holds | **+0.0724** |
| 2.0x -- index-addressing | **+0.1913** |
Comparable across L, across reruns, and across corpora after rescaling by that corpus's own log N
range. **The span (max/min) and max-min gap rules are retired** -- span was pinned to whichever
bucket happened to be smallest and fired INDEX-ADDRESSING on an arm whose gap slope was NEGATIVE.

Implementation, both required: **bootstrap CIs over systems** (log(ratio) has high leverage as a
ratio approaches zero -- measured max leverage 0.074-0.090 here, so no single point dominates), and
**ratios must be strictly positive; a ratio <= 0 is a FAMILY D SIGNAL to be reported, never clipped.**

### FAMILY D FIRED: THE PCA "CEILING" IS NOT A CEILING, AND THE RATIO IS INVALID
**22/69 systems at L=1, 14/69 at L=12, 6/46 at L=24 have ratio <= 0 -- the model BEATS its own PCA
ceiling** (184L reaches -10.07). This is not a bug. **PCA-L is fitted on 80 TRAIN frames and applied
to HELD-OUT frames, so it is NOT an upper bound on held-out performance**: a model trained across 207
systems generalises better than a per-system PCA that overfits its 80 frames. Consequence: the
deficit ratio's denominator can vanish AND change sign, which is the actual disease behind the
U-shapes, the L=1 scatter, and the near-zero bucket that broke the span rule.

**The log-log form therefore drops 9-32% of systems by construction.** That exclusion is NOT
N-correlated (p=0.738 / 0.254 / 0.937) but it IS **ceiling-correlated**: dropped systems have median
PCA ceiling 0.036 vs 0.135 (L=1), 0.200 vs 0.286 (L=12), 0.275 vs 0.377 (L=24). It removes exactly
the systems where the ceiling is weakest -- a property of the CEILING, not of the model -- so the
surviving fit is biased toward systems where PCA happens to generalise well.

### ROBUST ALTERNATIVE: the ABSOLUTE GAP (ceiling - model), in FVE points
Defined for either sign, no denominator, **no drops**. Thresholds set directly in FVE rather than via
a ratio: a 0.05-FVE degradation across the range -> slope +0.0318.
| L | n | gap slope | bootstrap CI | excludes 0? |
|---|---|---|---|---|
| 1 | 69 | -0.0594 | [-0.1267,+0.0065] | no |
| 12 | 69 | -0.0227 | [-0.0715,+0.0251] | no |
| 24 | 46 | -0.0274 | [-0.0681,+0.0164] | no |

### RE-PRICED (the 565-systems figure came from the stale mapping and is VOID)
n for CI half-width below half the 0.0724-0.1913 gap (target 0.0594), from THIS run's residuals:
| L | current half-width | n held-out needed | systems to sample (4:1) |
|---|---|---|---|
| 1 | 0.176 (n=47) | 411 | ~1,644 |
| **12** | **0.204 (n=55)** | **651** | **~2,604** |
| 24 | 0.131 (n=40) | 195 | ~780 |
Gap-based pricing agrees: L=12 needs n=637 held-out. **So certainty on the ratio costs ~2,600 sampled
systems, not 565** -- and it buys certainty about a statistic that is invalid on 9-32% of systems.

**CONCLUSION: do not buy it.** The binding defect is the CEILING, not n. Fixes, in order:
(1) a frame-rich corpus so per-system PCA stops overfitting (mdCATH has 2,000 train frames vs
MISATO's 80); (2) the direct instruments -- realised effective rank and subspace overlap -- which
need no ceiling at all and are already wired in.

## FINDING (reframed): A SHARED LEARNED CODEC BEATS PER-SYSTEM PCA ON HELD-OUT FRAMES

**PCA-L is a per-system BASELINE fitted on 80 TRAIN frames -- it was never a ceiling.** A single
codec trained across 207 systems beating it on HELD-OUT frames is **the thesis of this project**: a
learned shared representation extracting more generalisable structure than per-system classical
decomposition. Reported as a finding; the ratio framing that made it look like a broken denominator
is RETIRED.
| L | systems where the codec beats per-system PCA | best case |
|---|---|---|
| 1 | 22/69 (32%) | 1A0Q **+0.136 FVE** |
| 12 | 14/69 (20%) | 184L +0.076 FVE |
| 24 | 6/46 (13%) | 1AU2 +0.041 FVE |

### PRIMARY STATISTIC: SIGNED GAP = model FVE - PCA-L FVE
Continuous, no denominator, no sign pathology, **no drops -- every system retained**.
| L | n | slope vs log10(N) | half-width | change permitted across range | vs median abs(gap) |
|---|---|---|---|---|---|
| 1 | 69 | +0.0594 | 0.0666 | 0.105 | **3.1x** |
| 12 | 69 | +0.0227 | 0.0483 | 0.076 | **1.8x** |
| 24 | 46 | +0.0274 | 0.0423 | 0.067 | 0.9x |
**The CI permits a gap change 1-3x the size of the gap itself. No discrimination.**

### SUPPORTING (DEMOTED, Family C on a statistic I proposed): FRACTION WITH POSITIVE GAP
Binary-per-system discards magnitude, and at ~12 systems/bucket the **binomial SE floor is 9-13
percentage points** while the entire observed spread is 29-36 pp -- **2xSE alone spans most of the
range**. Slopes are +0.079+/-0.332, -0.058+/-0.445, -0.051+/-0.719: a null with no discriminating
power, NOT evidence of flatness. Retained as an interpretable descriptive number ("the codec beats
per-system PCA on ~20-30% of systems"), **not as an N-scaling test.**

### THE DECISIVE RESULT: MISATO CANNOT ANSWER THE N QUESTION BY ANY BASELINE-REFERENCED STATISTIC
Not merely underpowered -- **structurally confounded. The PCA baseline itself degrades with N,
significantly:**
| L | PCA baseline slope (CI) | model FVE slope (CI) |
|---|---|---|
| 1 | **-0.0877 [-0.1647,-0.0055]** | -0.0283 [-0.0749,+0.0196] flat |
| 12 | **-0.1660 [-0.2588,-0.0654]** | -0.1433 [-0.2145,-0.0685] |
| 24 | **-0.1790 [-0.2742,-0.0782]** | -0.1516 [-0.2197,-0.0733] |
All three PCA slopes EXCLUDE ZERO. PCA-L is fitted on 80 frames in a 3N-dimensional space, so as N
grows frames/dimensions falls and it overfits progressively worse -- **a baseline-side artifact**.
It degrades FASTER than the model (-0.166 vs -0.143 at L=12), which is exactly why the signed-gap
slope comes out positive. **Any baseline-referenced N-scaling statistic on MISATO inherits this.**
Only the CEILING-FREE instruments (realised effective rank, subspace overlap) can work there.

## ATLAS SCOPED VIA API -- NO DOWNLOAD (specs differ from recollection; verify, don't assume)
| spec | recalled | **measured** |
|---|---|---|
| proteins | ~1,390 | **1,938** |
| length | 3 x 100 ns | **100 ns confirmed** (nsteps 50e6 x dt 2 fs) |
| frames | ~1e4 | **10,001/replica** at 10 ps (nstxout_compressed=5000); ~30,000 per protein |
| size | 1k-20k atoms | **38-2,128 residues** (median 175) = ~361-20,216 all-atom |

**Download volume MEASURED by HEAD on 3 entries** (383/708/750 MB at 184/322/350 residues):
`bytes = 2.26 MB/residue x L - 30 MB`, **R^2 0.9971** -> **TOTAL 0.94 TB** for protein-only bundles
(vs the ~1.2 TB estimate -- close). The `_total.zip` variant including solvent is **5.2 GB for a
single protein**, ~7x larger, and is NOT wanted. Streaming a stratified subset: 300 -> 0.15 TB
(0.4 h on 4 workers), 700 -> 0.34 TB (1.0 h), all 1,938 -> 0.94 TB (2.6 h).
**Stream-compute-delete as with mdCATH; do NOT stage ~1 TB on a borrowed account's scratch.**

### ATLAS EXTENDS THE BOTTOM OF THE N AXIS, NOT THE TOP
| corpus | N min | N median | N max | span | train frames |
|---|---|---|---|---|---|
| ATLAS | 361 | 1,662 | **20,216** | 56x | **~30,000** |
| MISATO | 717 | 4,977 | **26,861** | 37.5x | 80 |
| mdCATH | 711 | 1,804 | 7,524 | 10.6x | 2,000 |
**ATLAS's top (20,216) is BELOW MISATO's (26,861); its 56x span comes from reaching DOWN to ~361.**
It buys a valid ceiling and large n -- it does NOT give a longer lever at high N, and **the 1e6-atom
extrapolation still rests on the same upper end already in hand.**
**Scope difference:** ATLAS is SINGLE CHAINS, MISATO is COMPLEXES -- different chemistry and mobility
regimes. Do not pool without checking, and expect ATLAS to sit on the FLOPPIER side as mdCATH did
(mdCATH single domains measured 1.65x the RMSF of MISATO complexes).

## ATLAS: SUBSAMPLE-AND-STORE (not stream). Feasibility VERIFIED on real data.

**Why subsample rather than take all 10,001 frames.** MISATO's PCA baseline overfits because 80
frames sit BELOW the median rank90 (~168). The fix is frames comfortably ABOVE rank90, not more
frames without limit: **~1,000 frames/replica -> ~800 train frames ~ 5x median rank90**, which is
well-conditioned. Frames beyond that buy almost nothing for a PCA ceiling at k<=24.

**Why store rather than stream.** Streaming was correct for the b-exponent (spectra out, data
discarded). It is WRONG here: **training reads frames many times over**, so the subsample must be
reusable rather than re-downloaded per run. The archive is deleted immediately; only the subsample
is kept.

### FEASIBILITY BLOCKERS FOUND AND RESOLVED (neither was anticipated)
1. **No trajectory reader anywhere**: no mdtraj/MDAnalysis/pytraj/chemfiles in the venv, no GROMACS
   module, and **pip is dead -- the venv's Python has no SSL module**. ATLAS ships `.xtc`, so without
   a reader its specs are irrelevant. RESOLVED by fetching the cp310 manylinux **wheel with curl and
   unzipping it onto PYTHONPATH** (`$WR/pylibs`), plus pure-Python deps pyparsing and packaging the
   same way. **mdtraj 1.10.3 works; the shared venv is never modified.**
2. **ATLAS bandwidth is 4.9 MB/s, not HuggingFace's 25** (383 MB in 78 s, measured). 700 proteins
   ~ 0.34 TB ~ **19 h serial, ~5 h on 4 workers**. The download is the real cost, paid ONCE.

### MEASURED ON REAL DATA (2y44_A)
- **10,001 frames/replica CONFIRMED**, matching the production .mdp exactly
- **14.7 atoms/residue all-atom** (2,701 atoms for 184 residues) -- NOT the ~9.5 assumed, so the
  raw storage estimate was ~3.5x low
- stride 10 -> 1,001 frames in 1.5 s; compressed npz observed **16-32 MB/protein at L=38-81**
  (~0.35 MB/residue) -> **~43 GB for 700 proteins, ~118 GB for all 1,938**. Storable and reusable.

### SCOPE NOTES to carry wherever the ATLAS ceiling is used
- **Uniform subsampling across the full 100 ns preserves the SLOW COLLECTIVE MODES** that dominate
  variance at low k. What is lost is fast local motion, which is not what a k<=24 ceiling measures.
- **ATLAS is SINGLE CHAINS, MISATO is COMPLEXES** -- different chemistry and mobility regimes. Do not
  pool without checking; expect ATLAS on the floppier side (mdCATH single domains measured 1.65x the
  RMSF of MISATO complexes).
- **ATLAS extends the BOTTOM of the N axis**: top ~20,216 atoms vs MISATO's 26,861. It buys a valid
  ceiling and large n, **NOT a longer lever** -- the 1e6-atom extrapolation still rests on the
  existing upper end.

### GUARD-FIRST PROTOCOL FOR THE ATLAS N-AXIS RERUN
**Run the PCA-baseline-vs-N slope FIRST, before reading anything else.** On MISATO that slope was
-0.088/-0.166/-0.179 with all CIs EXCLUDING ZERO -- the baseline degraded with N because 80 frames in
3N dimensions overfit worse as N grows, which contaminated every baseline-referenced statistic.
- **flat on ATLAS -> the ceiling is sound and every downstream statistic is trustworthy.**
- **still sloping -> subsample deeper (2,000 frames) and re-check.**
It is a GUARD, not a result. Then: direct instruments (realised effective rank, subspace overlap)
alongside as an independent ceiling-free read, with signed gap and fraction-positive supporting.
**What it answers:** whether the model's own -0.143 degradation with N is real or shares the
baseline's cause -- the objective-1 question, unanswerable on MISATO by construction.

## ATLAS HOSTS BOTH AXES -- the MISATO/mdCATH corpus split is RETIRED

It was a workaround for a bind ATLAS dissolves, and **two of the last three nulls were caused by a
corpus that could satisfy only one constraint at a time**:
| | N range | frames | training systems | ceiling |
|---|---|---|---|---|
| MISATO (was N-axis) | 37.5x | 80 | 207 | **BROKEN** (baseline degrades with N, CIs exclude 0) |
| mdCATH (was capacity) | 10.6x | 2,000 | **21** | valid |
| **ATLAS** | **56.7x** | **10,001** | **1,938** | valid |

### 1. SUBSAMPLE 2,500 FRAMES, NOT 1,000 (stride 4)
A valid PCA-512 ceiling needs **>=1,707 usable frames** under the 30% rank rule. 2,501 subsampled ->
~2,000 train -> **DM=512 sits at 512/1999 = 25.6%, edge-but-valid**. At 1,000 frames (800 train) the
DM=512 ceiling would be **VOID BY CONSTRUCTION** and the capacity null would repeat with better data.
Storage ~43 -> **~108 GB for 700 proteins**; still storable, still reusable.

### 2. PRE-REGISTERED COLLAPSE TEST -- data-limited or architectural?
DM=256 and DM=512 collapsed to a constant (G1 cos 1.0000, G4 -0.000) at n=21 training domains.
**Run DM=256 at n=21 (REPRODUCE the collapse) and at n=700.**
- **trains at 700, collapses at 21 -> purely DATA-LIMITED**, and the capacity axis is answerable by
  scaling the corpus.
- **collapses at BOTH -> ARCHITECTURAL.** A 256/512-wide FiLM code collapsing to a constant is a
  known failure shape and the fixes differ: latent normalisation, a variance floor on the code, or
  removing the pooled-FiLM bottleneck.
**Do NOT skip the n=21 reproduction** -- without it, "we fixed it" cannot be distinguished from "we
changed two things at once".

### 3. THE PCA-vs-N GUARD IS THE ENTRY CONDITION FOR EVERY ATLAS RUN
Run it BEFORE reading anything else. Flat -> ceiling sound. Still sloping -> subsample deeper and
re-check. Report the realised log-N span beside the slope and CI; **under ~1.2 decades the guard has
not been tested regardless of what it says**, and Family C applies -- a flat slope with wide CIs is
an UNTESTED ceiling, not a clean one, so state the slope the CI still permits and what that does to
the downstream N-reading.
Selection is now **LOG-UNIFORM** across length, deliberately over-weighting both extremes:
rank-uniform followed the length distribution and clustered at quartiles 108/176/300, starving
exactly the ends where a slope gets its leverage.

### 4. THE FAIR TEST OF THE THESIS (stated so it cannot drift)
**Does a shared codec trained on ~700 systems beat per-system PCA fitted on 2,000 WELL-CONDITIONED
frames?**
- On MISATO the BASELINE was broken (80 frames, overfits, degrades with N).
- On mdCATH the MODEL was starved (21 training domains).
- **ATLAS is the first setting where both sides are sound.**
**Report fraction-positive and signed gap there as the HEADLINE.** That number is the project's
central claim and **it has never actually been measured.**

## THE COMPARISON SET, CORRECTED: per-system PCA is an ORACLE, not a peer

**The earlier "fair test" framing was wrong** and would have produced a headline that reads as
refuting the thesis while refuting nothing.

| | fitted on | evaluated on | per-system parameters |
|---|---|---|---|
| **per-system PCA-k** | 2,000 frames **OF THE TARGET SYSTEM** | held-out frames of that SAME system | **3N x k** -- at N=2,077, k=256 that is **~1.6 MILLION** |
| **the codec** | OTHER systems | held-out **SYSTEMS**, target trajectory never seen | **ZERO** |

Within-system fit vs zero-shot transfer. **Beating it would be remarkable; losing it is the NULL
EXPECTATION.** (The parameter asymmetry was measured earlier and the conclusion not drawn.)

### CORRECT COMPARISON SET
- **ORACLE (upper bound):** per-system PCA-k. Report the codec as a **FRACTION of it** --
  *"zero-shot reaches X% of a within-system fit"* -- which is the honest and genuinely interesting
  number.
- **PEER (the bar that matters):** methods that ALSO never see the target trajectory -- **ANM /
  structure-predicted basis**, and the existing **graph codec**. This is where the thesis lives, and
  it is the comparison the project already won once (learned codec beat ANM on ligands).
- **FLOOR:** zero displacement, i.e. predicting the train mean -- FVE = 0 by construction.

### ANM PEER-BASELINE COVERAGE ON ATLAS (measured)
Dense eigensolver limit ~6,000 atoms (~276 s/system; ~87 s at 4,000, ~650 s at 8,000).
| N bucket | n | ANM computable |
|---|---|---|
| 591-1,159 | 190 | 100% |
| 1,159-2,272 | 575 | 100% |
| 2,272-4,454 | 647 | 100% |
| 4,454-8,731 | 427 | **54.6% PARTIAL** |
| 8,731-17,115 | 94 | **0% -- NO ZERO-SHOT PEER** |
| 17,115-33,551 | 5 | **0% -- NO ZERO-SHOT PEER** |
**ANM covers the bottom ~85% of the corpus** (1,645/1,938 at the 6,000 limit). Above ~6,000 atoms
there is **NO zero-shot peer baseline until a sparse solver exists** -- state this plainly rather
than filling the gap. **CG-ANM is NOT a substitute**: its bias was measured at -0.059 median and is
**N-DEPENDENT** (slope +0.110 +/- 0.039), which is exactly the contamination it would introduce into
an N-scaling comparison. Retracted and staying retracted.

### HEADLINE, RESTATED
1. **PRIMARY: codec vs ANM at matched k, both zero-shot**, on the buckets where ANM is computable.
2. **SECONDARY: codec FVE as a FRACTION of the per-system PCA oracle**, across the full N range.
3. **THE N-QUESTION (objective 1): does EITHER of those degrade with N?**

Note the consequence for the top of the N axis: the primary comparison is unavailable above ~6,000
atoms, so the N-trend for the PEER comparison rests on 591-8,731 (~15x span, still ample), while the
ORACLE-fraction trend spans the full 591-33,551. Report both spans explicitly.

## SPARSE ANM: validated, and the CUTOFF turns out to be a first-order choice

### Validation (your protocol, passed)
Sparse eigsh on the SAME Hessian vs dense, on the N<=6,000 overlap:
| protein | N | dense | sparse | max rel eig err | subspace overlap |
|---|---|---|---|---|---|
| 1fd3_A | 610 | 0.6 s | 0.7 s | 5.0e-09 | 1.000000 |
| 1fm4_A | 2,449 | 35.2 s | **15.4 s** | **4.7e-10** | **1.000000** |
| 1fs1_C | 853 | 1.5 s | 1.1 s | 1.8e-10 | 1.000000 |
**Same eigenproblem, not a new baseline** -- so extending to the full N range carries NO caveat,
unlike CG-ANM (which was an approximation with an N-DEPENDENT bias, slope +0.110 +/- 0.039).
At scale: N=9,195-10,254 takes 45-205 s (nnz 24-28M, RSS 3.5-3.7 GB) where **dense at the ATLAS top
would be ~25 HOURS**. Sparse is the only route above ~6,000 atoms.

### THE CUTOFF IS A FIRST-ORDER CHOICE -- ANM-10 IS A WEAK BASELINE (measured)
10 A on all-atom gives ~142 pairs/atom and over-connects; the all-atom literature sits at 5-7 A
(~22 pairs/atom at 5 A, also 6.5x cheaper). Measured at k=64 on cached ATLAS proteins:
| protein | N | ANM-5 | ANM-7 | ANM-10 |
|---|---|---|---|---|
| 1fd3_A | 610 | **0.4273** | 0.3584 | 0.2991 |
| 1fm4_A | 2,449 | **0.4294** | 0.4027 | 0.3539 |
| 1fs1_C | 853 | **0.6816** | 0.6273 | 0.5739 |
| 1j8e_A | 598 | **0.6101** | 0.5344 | 0.3808 |
| **mean** | | **0.5371** | 0.4807 | **0.4019** |
**ANM-5 beats ANM-10 by +0.135 FVE on EVERY protein.** A codec beating ANM-10 would be a hollow win
on the comparison that carries the thesis.

**SELECTION PROTOCOL (keeps it a baseline, not an oracle):** sweep {5, 7, 10} A on TRAINING systems
only, pick the single best cutoff by mean TRAINING FVE, apply that one cutoff unchanged to held-out
systems. All three training curves reported so the choice is visible. **ANM-10 is retained as the
labelled CONTINUITY baseline** for comparison against earlier results, and the two are NEVER pooled.

### RETROACTIVE CAVEAT ON THE PROJECT'S ONE PRIOR PEER WIN
The ligand result -- "the learned graph codec beats Cartesian ANM" (0.64 vs 0.74 at L=8) -- used an
**8 A cutoff on ligand HEAVY ATOMS** (8-90 atoms, molecular diameter ~10 A), which connects nearly
the whole molecule. **That win may rest on the same weak-baseline problem** and must be re-checked
against a swept cutoff before being cited as evidence for the thesis. It is currently the only
peer-comparison victory on record.

### RETRY-WITH-ESCALATION (Family A, in the solver)
Convergence time varies 45 s vs 205 s at matched N -- that is shift-invert convergence, which is
TUNABLE. A timeout that preferentially kills slow-converging systems is an **N-CORRELATED EXCLUSION**
if convergence difficulty tracks size, which would void the comparison it feeds. Policy: attempt 1 at
default, attempt 2 with adjusted sigma/maxiter, attempt 3 with a larger Krylov subspace, and ONLY
then record a Family A exclusion. Retrying costs minutes against a bias that would void the result.

### MEMORY AT THE TOP -- test submitted before wiring in (job 10301505)
Largest ATLAS entry is **6sup_A, 33,377 real atoms** (the 15.77*L-8 fit predicted 33,541: 0.5% error,
so the N-range extrapolation was sound). Shift-invert factorises (H - sigma*I) and fill-in on a
100k-DOF 3D connectivity problem grows superlinearly, so peak RSS could land far above the ~12 GB a
linear scaling suggests. **Testing on the real structure at all three cutoffs before committing the
pipeline** -- if it exceeds node memory we need to know now, not when a cache job dies two-thirds
through. Fetching only the reference .pdb (2.6 MB) rather than the 4.8 GB trajectory bundle.

## FAMILY E -- a comparator handicapped by an unswept hyperparameter
**Instances:** ANM-10 on all-atom proteins (ANM-5 beats it by **+0.135 FVE**, on every protein tested);
ANM-8 on 8-90-atom ligands (tested: the cutoff turned out to matter only 0.005-0.008 A there, so this
instance was a FALSE POSITIVE -- recorded because the hypothesis was reasonable and the test was cheap).
**Signature:** a win reported against a baseline whose settings were never optimised. The comparison
cannot fail, so it proves nothing.
**CHECK:** before reporting any comparison, sweep EVERY baseline hyperparameter on TRAINING data,
report the curve, use the best. **A baseline you didn't try to make strong is not a baseline.**

### FAMILY A — INFRASTRUCTURE VECTOR (INBOX 18d): resume/cache/skip-if-exists paths are exclusion filters
The 10307102 near-miss is a new shape of an old family, and the exclusion came from **no analysis
choice at all**. The peer loop processes systems **ascending in N** by design; 49 of the smallest were
already stored by an earlier build; and skip-if-exists would have computed 17c's matched-rank
decomposition on the **largest 74 only** — a 40% N-correlated exclusion of the axis under test,
invisible in the analysis code because it lives in the resume path.

> **CHECK:** any resume, cache, skip-if-exists or partial-output path is a potential exclusion filter.
> If the work is ordered by a regressor, a partial run is a *biased sample* of it. Before resuming,
> print the regressor distribution of what is stored against what is not, and treat a skew as a purge
> condition rather than a saving. Implemented as `armf_stamp.coverage_by()`, called by the peer loop
> before it resumes.

**Same lesson as the Q4 runtime-print finding:** the dangerous instances are the ones *outside* the
analysis, where nobody is looking.

## FAMILY C, COMPILED: "fails to reject zero" written into a verdict branch as "is zero"
Family C is an underpowered null believed. The instances that matter here were not in prose — they
were **compiled into verdict branches**, so they would have printed the same wrong reading on every
future run, with the authority of an automated conclusion.

**Three sites, all firing on `abs(estimate) <= halfwidth` alone:**

| site | printed | |
|---|---|---|
| `armf_atlas_modes` 27b | *"the code decay adds nothing once N is held — they are SEPARATE"* | fired where the non-significant coefficient had the **larger point estimate** (1.16× N's) |
| `armf_tied_mediator` | *"flat (mechanism absent)"* | "CI includes zero" ≠ "mechanism absent" |
| `armf_modal_ctx` 23b | *"reach is NOT the constraint"* | would have sent the project to rung 2 on a null it could not support |

**The correct form was already in the project**, in `armf_tica_vs_n`'s pre-registered read: *"exponent
flat (CI includes 0, **excludes ~0.5**)"*. **Rejecting the alternative is what licenses a null; failing
to reject zero licenses nothing.**

**CHECK — `armf_stamp.null_verdict(estimate, halfwidth, relevant)`.** Every caller must NAME the
smallest effect that would change the conclusion, and gets **three** answers rather than two:

- `EXCLUDES_ZERO` — CI excludes 0, and reaches past `relevant` → a real effect that matters
- **`SIGNIFICANT_BUT_BELOW_RELEVANCE`** — CI excludes 0 **and** sits entirely inside ±`relevant` →
  statistically real, **practically below the bar the caller itself set**. This is the mirror of
  believing an underpowered null: acting on an effect already declared irrelevant. It must be tested
  **before** the plain significance branch or that branch swallows it.
- `EQUIVALENT` — CI excludes ±`relevant` → a genuine null, **bounded by that number**
- `NOT_RESOLVABLE` — CI contains both 0 and `relevant` → underpowered, **nothing is licensed**

Worked on the 27b partials: `log N` −0.2668 ± 0.1843 → REAL EFFECT; `log z_ratio` −0.3103 ± 0.3375 →
**NOT RESOLVABLE**, not "separate". And on 26a's mediator, −0.0992 ± 0.0639 against a predicted +0.5 →
the CI excludes both 0 and +0.5, so that refutation was **sound** — it rejected the alternative rather
than failing to reject zero.

## FAMILY G — a VERDICT emitted by a THRESHOLD sitting at rounding distance from the measurement
**The family's actual statement (INBOX 19a), which is sharper than "pick better thresholds":**
> *every instance was a verdict AUTOMATED TO GUARD AGAINST BIAS, and the automation moved the bias
> from the conclusion into the THRESHOLD, where it is harder to see.*

**Signature:** the measured numbers are right and the *automated reading* of them is wrong. The
decision flips on a hand-picked constant nobody derived, at a margin far below the uncertainty in the
quantity being tested.

**THE FIX IS SENSITIVITY, PRINTED UNCONDITIONALLY** — naming the metric does not close the hole,
because the next instance will have a defensible metric *and* an arbitrary constant. Every automated
verdict must report what it would have concluded across the plausible range of **every free choice it
contains**: the threshold at 0.5×/1×/2×, every candidate metric it could have decided on, and the
margin to the decision boundary. **If the verdict flips anywhere in that range it is not a verdict —
it is a measurement plus an opinion, and it must print as such.** Implemented as
`armf_stamp.verdict_sensitivity()`; it costs nothing, since the verdict block already holds every
number. Validated against all three instances — each is flagged — and against the *corrected* 16a
ratio test, which comes back **STABLE across 0.75×–1.5×**. That stability is itself a result: the
decoder-limited headline survives its own free choices, where the original threshold did not.

**Two instances within one hour, identical in shape:**
- **16a's realised-rank branch** tested `med <= max(3.0, 2·PR/5)` → **5.96 against a measured 6.0**, a
  **0.7%** margin. It printed *"the latent is the limit"* when the measured ratio (6/14.9 = 0.40) says
  the decoder is. The opposite conclusion, on a rounding-scale margin.
- **007's stopping rule** applied the n_eff threshold to the median **pooled across bases** →
  **2.045 against 2.0**, a **2%** margin, from a pool describing neither basis (3.28 viable, 0.81
  censored). Pooling let a censored basis drag a viable one toward the cliff.

**CHECK — all four, on every automated verdict:**
1. Print the **table first and the verdict second**, so a reader checks the reading rather than
   trusting it.
2. State the **margin** between the measured value and its threshold, and flag any margin under 10%.
3. Prefer **ratios with no free constant** (`med < 0.6·PR`) over absolute cutoffs.
4. **Viability is per-unit.** Never pool a per-basis, per-system or per-arm property before applying a
   rule to it — assess per unit, then report the units separately.

**Why this family is worse than it looks.** A Family G defect produces a *confident, well-formatted,
plausible* conclusion backed by correct numbers. Nothing looks wrong, the arithmetic checks out, and
the only tell is a margin nobody printed. Both instances above were caught by reading the numbers
next to the verdict — which is exactly why check 1 is first.

## FAMILY F -- a comparator computed on DIFFERENT DATA from the model
**Instance:** the graph-codec ligand win. `armf_graph_codec.py:239` printed `cANM 0.77/0.74/0.73` as a
HARDCODED STRING; ANM was never computed on that script's 190-ligand set. The numbers came from
`armf_intcoord_oracle.py`, which sampled **12 ligands**. A model evaluated on 48 held-out molecules
was compared against a baseline measured on a different 12-molecule sample. Recomputed correctly,
ANM-8 is **0.599/0.579/0.549**, not 0.77/0.74/0.73 -- the baseline was ~0.16 A stronger at L=8 than
reported, and the win reverses at L=4 and L=8.
**Signature:** a baseline number that is a literal in the reporting code rather than an array computed
from the same rows as the model.
**CHECK:** every comparator must be COMPUTED IN THE SAME SCRIPT, over the SAME held-out rows, in the
same run. No baseline number may be a hardcoded constant. If a number is quoted from another run,
label it with its own n and never place it in the same table as a differently-sampled result.

### SECOND INSTANCE — COMMITTED *BY* THE AUDIT, IN THE AUDIT'S OWN CODE (INBOX 016)
The 14b reproduction guard compared a re-trained arm's mean over the **24 N-stratified TRACKED
systems** against the recorded `fve`, a mean over **all 123 held-out systems**. Those denominators
differ systematically — for the three DM=256 arms, 0.1453/0.1553/0.1497 over 123 against
0.0969/0.0913/0.0958 over the 24, a consistent ~1.6× — because the stratified sample deliberately
over-weights the large end and is therefore the *harder* set, not a noisier estimate of the same
quantity. The guard printed a **4× divergence where the like-for-like gap is 1.9×**.

**Why this instance matters more than the first.** The first was found by the audit in old code. This
one was *written* by the audit, in a guard whose entire purpose was to detect a discrepancy — and it
fired correctly while overstating the magnitude, which is the failure mode most likely to be believed.
A check that compares the right two things for the wrong reason is not a check.
**Signature:** two aggregates with the same *name* (`FVE`) computed over different row sets, compared
without either being restated. **CHECK:** any guard that compares a new number to a stored one must
name the denominator of both, and prefer a stored field computed on the *identical* set —
`best_track` here, not `fve`. Fixed in `armf_atlas_modes.py`, which now compares `best_track` to
`best_track` and prints the all-system figure separately, labelled as a different denominator.

## GRAPH-CODEC WIN: WITHDRAWN (was PROVISIONAL pending this re-run)
Held-out, same 48 ligands, baselines swept on training molecules only:
codec **0.800/0.640/0.440** vs swept ANM **0.594/0.571/0.549** at L=4/8/16.
**The codec LOSES at L=4 and L=8 and wins only at L=16** -- which is also where Family B is worst
(9/48 molecules past 30% of 3N-6, one at 76%). Full writeup: outputs/cluster/armf_ligand_peer_rerun.md.
**It must not be cited as evidence that a learned codec beats classical zero-shot decomposition.**

### WHAT IS *NOT* AT RISK (checked, and one conclusion STRENGTHENS)
- **The codec-arc closure stands and gets STRONGER.** It rested on ANM at 1.37 A beating oracle-B at
  1.45 A on proteins, i.e. the B-factor track was already dead. A STRONGER ANM (swept cutoff) makes
  the B-factor track **deader, not less dead**. That conclusion strengthens.
- **The propagator results are unaffected** -- they never referenced an ANM baseline for a win claim.
- **The intrinsic-dimensionality / mobility-law results are unaffected** -- no comparator involved.
- **AT RISK and now withdrawn:** only the ligand graph-codec peer win.

## FAMILY E -- a comparator handicapped by an unswept hyperparameter

**Instances:** ANM-10 on all-atom proteins (ANM-5 beats it by **+0.135 FVE**, on every protein
tested); ANM-8 on 8-90-atom ligands.
**Signature:** a win reported against a baseline whose settings were never optimised. **The
comparison cannot fail, so it proves nothing.**
**CHECK:** before reporting any comparison, sweep every baseline hyperparameter on TRAINING data,
report the curve, and use the best. **A baseline you did not try to make strong is not a baseline.**

### FAMILY E's first application RETRACTED THE PROJECT'S ONLY PEER WIN -- and found a worse bug
Re-running the ligand comparison revealed the cited ANM numbers (`cANM 0.77/0.74/0.73`) were a
**HARDCODED PRINT STRING** in `armf_graph_codec.py:239`, carried over from a **12-ligand** run, while
the codec was evaluated on **48 held-out ligands**. The baseline was never computed on the codec's
own held-out set -- the two sides were measured on DIFFERENT MOLECULES.
Correct, same-split comparison: **codec LOSES at L=4 (0.800 vs 0.594) and L=8 (0.640 vs 0.571)**, and
wins only at L=16 (0.440 vs 0.549) where Family B flags 9/48 molecules past 30% of their 3N-6 rank.
Note the cutoff itself was NOT the ligand problem: 5-8 A is a broad optimum there and ANM-8 was a
reasonable choice; on proteins ANM-10 genuinely is weak. Two different errors, one family.
Second peer tested and REFUTED: a bond-graph ENM (1-2 + 1-3, weight swept) scores 1.29-1.50 A vs
0.55-0.59 A for distance-cutoff ANM -- a distance cutoff is the better network at this scale.
Full writeup: outputs/cluster/armf_ligand_peer_rerun.md

### GRAPH-CODEC WIN: RETRACTED (was PROVISIONAL, now resolved against it)
The ligand codec result must NOT be cited as supporting evidence for the thesis. **The project
currently has NO surviving peer-comparison win**, which is precisely why the ATLAS run matters: it is
the first setting where such a comparison could be *established* rather than re-litigated.

### WHAT IS **NOT** AT RISK -- and one conclusion that STRENGTHENS
The **codec-arc closure** rested on ANM (1.37 A) beating oracle-B (1.45 A) on the B-factor track.
A STRONGER ANM makes that track **deader, not less dead** -- the stopping rule stands and is
reinforced. Also unaffected: the intrinsic-dimensionality/mobility law (c ~ -0.9 to -1.35,
reproduced cross-corpus), the TICA/window objective-3 results, the propagator transfer results, and
the §7 additivity measurement -- none of these involve a comparator with an unswept hyperparameter.

## THREE CORRECTIONS BEFORE THE ATLAS CURVE

### 1. THE GUARD NUMBER WAS STALE -- RE-MEASURED, AND IT NOW FAILS (job 10304109, n=115)
The -0.1096 +/- 0.1311 was measured with held-out frames from the SAME trajectory, and was **not
significant**. Under the new split (fit replicas 0+1, evaluate replica 2) the slope is larger and
**significant at both k**. The old number is RETIRED and may not be quoted.
| k | median FVE | slope vs log10(N) | p | bootstrap 95% | drop-most-influential | adversarial +2 |
|---|---|---|---|---|---|---|
| 24 | 0.5742 | **-0.1430 +/- 0.0818** | 0.001 | [-0.2271, -0.0624] | -0.1606 | -0.0960 |
| 256 | 0.8614 | **-0.1880 +/- 0.0439** | 0.000 | [-0.2381, -0.1387] | -0.1754 | -0.1541 |
Negative under all four tests at both k. Span 1.75 decades, rank-void 0/115, s_k/s_1 medians 1.31e-01
and 2.87e-02. **THE CEILING DEGRADES WITH N.** See the n_eff section below for the mechanism.

### 2. ATLAS REPLICAS ARE ONLY WEAKLY INDEPENDENT -- the claim is WALKED BACK
Replicas share a start structure and differ only by velocity seed. Measured across 10 proteins
spanning N=598-33,377:
| statistic | value |
|---|---|
| between-replica / within-replica RMSD | **median 1.177**, range 1.085-1.442 |
| ratio = 1.0 would mean | replicas INDISTINGUISHABLE from within-replica variation |
Replicas sit only **~18% further apart than frames within a replica**: 100 ns does NOT fully
decorrelate them. Replica 2 is a **somewhat harder test than a temporal split, not a categorically
different one** -- the phrase "independent trajectory" is withdrawn from the scripts and the writeups.
N-dependence of the ratio: slope +0.0695 +/- 0.1604, p=0.347 at n=10. **That is UNDERPOWERED
(FAMILY C), not "no effect"** -- the CI permits +-0.28 across the range against an observed spread of
0.36, so it must be re-measured at full n. If decorrelation DOES vary with N, the held-out test
changes difficulty along the very axis under test, which would be an N-correlated artifact in the
EVALUATION itself.

### 3. b MOVED TO ATLAS; THE mdCATH STREAM IS CANCELLED
| | ATLAS | mdCATH |
|---|---|---|
| N range | **56x** | 11x |
| systems | **825 cached** (1,938 available) | 28 |
| frames | **7,503** | 2,000 |
| rank position of k~500 | **10% (clean)** | 31% (EDGE) |
b decides whether "one global latent, width independent of atom count" holds at 1e6 atoms, and the
ATLAS cache is already built for the learning curve, so this costs nothing. `armf_b_exponent.py` is
marked SUPERSEDED, `bexp.sbatch` deleted, and the 0.34 TB mdCATH download is dropped entirely (it was
staged, never submitted -- nothing to abort).
`armf_atlas_b.py` carries the machinery over unchanged with **replicas as the join unit**: join sweep
(1/2/3) with the optimum MEASURED rather than argued, p90-based selection (censoring is
system-dependent, and a median-clean join can still be censored for the RIGID tail where the slope's
leverage lives), threshold sweep as DIAGNOSTIC subsetting that never excludes from the primary fit,
mandatory mobility control (the naive N-exponent is a mediated confound: bigger proteins are more
rigid at corr -0.31, and rigidity raises rank90 at c ~ -0.9 to -1.35), Family D flagging when a
subset's N range collapses, and conservation of n with every failure logged against its N.

## EFFECTIVE SAMPLE SIZE: THE CEILING IS IRREDUCIBLY N-BIASED ON ATLAS

Frames are not samples. `n_eff = F / tau_int`, `tau_int = 1 + 2*sum rho(k)` truncated at rho<0.05 --
the acceptance-test convention -- measured on PCA-mode coefficient series (the Gram's eigenvectors ARE
those series, so this costs nothing extra). Full writeup: `outputs/cluster/armf_neff.md`.

### Four artifact checks before the number was trusted
| check | result |
|---|---|
| **A. rigid-body motion** (would manufacture BOTH large tau and a fake IAT-vs-N slope) | **RULED OUT.** Kabsch removes 1.5/0.8/0.3/0.1/**0.0**% of variance, *decreasing* with N; COM drift <=0.55 A; net rotation <=2.4 deg; tau raw vs aligned agree <1%. ATLAS ships superposed. |
| **B. lag-cap censoring** | **FOUND AND FIXED.** maxlag=500 truncated 4/5 systems (needed 625-841). That censors LARGE tau preferentially -- i.e. the variable under test -- biasing the slope DOWNWARD. **Family B, and it was live in a submitted job**, which was cancelled and its partial output deleted. maxlag is now F//2 with a per-system convergence flag. |
| **C. mode-index dependence** | **REFRAMES IT.** tau is not a per-system scalar: ~700-1060 (mode 0), ~17 (mode 49), ~3 (mode 199), ~1-2 (mode 499). Leading modes starved, bulk of rank90 effectively independent. All comparisons are now mode-resolved with tau averaged over EXACTLY the modes being counted. |
| **D. selection effect** | **REAL, QUANTIFIED, NOT CORRECTED OUT.** PCA picks the slowest direction by construction, so tau(PC1) is an extreme order statistic: 3.1-9.5x a random-projection null. PC1 OVERSTATES the problem; the mode-averaged quantity carries the result. |

### IAT-vs-N is CONFIRMED but is the MINORITY mechanism (21%)
Interim n=20 spanning N=598-33,377, slopes per decade of N:
| quantity | slope | R^2 | |
|---|---|---|---|
| tau, PC1 alone | +0.1405 +/- 0.1696 | 0.14 | n.s. -- noisy order statistic |
| tau, random projection | +0.1082 +/- 0.2287 | 0.05 | n.s. -- unselected reference |
| **tau, mean over 24 modes** | **+0.1713 +/- 0.0554** | 0.70 | **significant** |
| **tau, mean over 256 modes** | **+0.1745 +/- 0.0416** | 0.81 | **significant** |
| n_eff / rank90 | -0.8239 +/- 0.4569 | 0.44 | the conditioning quantity |
| rank90 | +0.6494 +/- 0.4712 | 0.32 | for attribution |

tau roughly **doubles** across the 1.75-decade span (2.02x), so n_eff falls ~165 -> ~94 at k=256. But
the conditioning slope decomposes exactly: `-0.1745 (IAT) + -0.6494 (rank90) = -0.8239` measured.
**Slower modes in bigger proteins explain ~21%; the dominant 79% is simply that large proteins need
more modes.** The hypothesis is confirmed, and is not the whole story.

The starkest number is not a slope: **n_eff is 14-24 at k=24 and 94-165 at k=256, against 2,501
frames** (~1-7% efficiency). At k=256 the ceiling fits a 256-dim subspace from **fewer than 256
effective samples in every system**.

### Neither more frames nor more replicas fixes it
- **Denser subsampling adds nothing**: n_eff is set by tau_int, not stride -- halving the stride
  doubles frames AND doubles tau in frame units. It also does not reduce rank90.
- **More replicas add far less than 2x**: measured between/within RMSD ratio **1.18**. The raw
  1->2 replica doubling in the old frames-per-rank90 table is an UPPER BOUND.
- Only **longer trajectories** would, and ATLAS does not have them.

### THE STANDING DECISION THIS FORECLOSES
**Do NOT spend another corpus move chasing a sound ceiling. No available dataset supplies the
trajectory length that would fix it.** Consequently:
- **PRIMARY = codec vs ANM**, both zero-shot, same held-out replica-2 frames. **No ceiling, no
  denominator, no per-system oracle.** Nothing measured here touches it. This is the **MAIN LINE**,
  not a fallback.
- **SECONDARY = the oracle-fraction**, now **PERMANENTLY CAVEATED**. Every future report of it must
  carry the slope (-0.1880 +/- 0.0439 at k=256) and the bias direction: **a ceiling that degrades
  with N makes the high-N oracle-fraction an UPPER BOUND -- it FLATTERS the codec at large N.**
  Reported with that attached, or not at all.

### Open
- rank90 hits 807 of 2,501 usable rank (**32%**) at the top of the N axis, past the 30% edge where
  rank90 presses against its own measurement ceiling (Family B). If it bites, +0.6494 is an
  UNDERestimate.
- A false "RUN IS VOID" banner fired on the guard: `conservation_report` used the whole store as the
  denominator rather than the held-out list the guard deliberately processes. Fixed by making the
  intended population explicit and by distinguishing **never attempted** (coverage shortfall, cache
  still building) from **failed** (size-correlated dropout). **A warning that cries wolf trains the
  reader to ignore the one warning that must never be ignored.**

## WIDTH CHAIN IS A FLOOR, NOT AN ESTIMATE -- AND rank90 IS RETIRED AS A SIZING INSTRUMENT

INBOX 002. Two independent biases push the SAME direction, so they compound rather than cancel.

### BIAS 1 -- rank90 is measured IN-SAMPLE. MEASURED, not argued.
PCA maximises explained variance on the very frames it was fitted to, so rank90 is a LOWER BOUND on
the modes a NEW frame needs. With n_eff at 1-7% of raw frames the gap is not a rounding error.
Measured on ATLAS under the actual protocol (fit replicas 0+1, evaluate replica 2),
`armf_rank90_insample.py`:

| N | rank90_in | rank90_out | ratio | held-out FVE at rank90_in |
|---|---|---|---|---|
| 598 | 31 | 70 | 2.26x | 80.2% |
| 1,173 | 103 | 257 | 2.50x | 79.9% |
| 1,709 | 27 | 13 | 0.48x | 93.9% |
| 2,433 | 217 | 960 | 4.42x | 69.0% |
| 3,475 | 259 | 889 | 3.43x | 79.0% |
| 4,919 | 399 | 2,136 | 5.35x | 69.0% |
| 7,495 | 314 | 2,417 | 7.70x | 70.3% |
| 33,377 | 914 | **never reaches 90%** | **CENSORED** | 56.3% (max 72.5%) |

**The in-sample rank90 modes explain a median of 74.7% of held-out variance, not 90%.** At the top of
the N axis rank90 is not merely underestimated -- it is **UNDEFINED out of sample**.
**And the ratio GROWS with N** (2.26x -> 7.70x -> censored), so bias 1 attacks the **b exponent**
too, independently of bias 2.

### BIAS 2 -- rank90 is CENSORED at large N (Family B)
rank90 reaches **32.3%** of usable rank at the top of the N axis, past the 30% edge where a count
starts pressing against its own measurement ceiling. That censors rank90 **at large N specifically**,
which is exactly where the b slope takes its leverage, so the measured growth understates the true
growth.

### CONSEQUENCE
`168 / 0.65 x 1.83 ~= 490 dims at 1e6 atoms` is a **FLOOR**. The 168 anchor inherits the same n_eff
problem and mdCATH never had it checked. **"DM=512 sits right at the edge" is withdrawn** -- a budget
compared against a floor is unsupported in the only direction that matters.

### THIS IS CORPUS-INDEPENDENT -- there is NO dataset move that fixes it
n_eff at 1-7% is trajectory length versus decorrelation time, not a property of ATLAS.
MISATO **10 ns** / mdCATH **2.5 us** / ATLAS **100 ns** -- none supplies enough independent samples to
fit a few-hundred-dimensional per-system subspace. **Exactly as with the ceiling, the answer is to
change instrument, not corpus.**

### WITHDRAWN: `PCA-256 = 0.958` ("256 dimensions capture ~96% of displacement variance")
Do not cite it. At k=256, n_eff is 94-165, so the fit carries **more dimensions than effective
samples in 237 of 239 systems** (n_eff per fitted dimension **0.53**; at k=24 it is 0.87, below 1 in
188/239). Held-out evaluation could not penalise it because the held-out replica sits only **1.18x**
further away than the training frames sit from each other -- 100 ns does not decorrelate the
replicas. Under the current protocol the PCA-256 median is **0.8614**, and that number is subject to
the identical objection: it is not usable as a capacity claim either.
(The `0.958` is not in the committed record -- a repo-wide search finds it only as an unrelated
mdCATH PCA-DM-512 ceiling in `armf_capacity_axis.md`. Recorded here so it cannot be resurrected.)

### THEREFORE: SIZE THE LATENT FROM THE CODEC, NOT FROM rank90
The codec's held-out FVE-vs-DM curve has **no per-system overfitting problem**: the model is SHARED,
fitted across many systems, and evaluated on **systems it never saw**, so its effective sample size
is the **CORPUS**, not one trajectory. This was built as a capacity probe; it is now also the only
sound way to answer "how wide does the latent need to be."

**What this costs, stated honestly:** rank90 and the width chain were the *physics* argument for "one
global latent, width independent of atom count." That argument is now bounded below and cannot be
tightened with available data. The codec curve replaces it and is the better instrument anyway -- it
measures what the architecture **achieves** rather than what an idealised per-system decomposition
would need. But it is an *empirical* answer on a *measured* N range, not a physics argument, so the
1e6-atom extrapolation is no longer supported by anything and must be dropped or re-derived.

### PRE-REGISTERED READ (before the DM run exists)
- Held-out FVE rises with DM then flattens; the flattening DM is the architectural width answer.
- **A flat curve is only width saturation if the wide arms actually USED their width.** The prior
  mdCATH DM sweep collapsed at DM=256/512 to constant output (G1 cos = 1.0000) on 21 training
  domains and was VOID. So each arm reports the **participation ratio** of the learned latent
  alongside FVE: `PR = (sum lambda_i)^2 / sum lambda_i^2` on the latent covariance across held-out
  frames and systems. **PR ~ DM => genuine saturation. PR << DM => capacity that failed to train,
  which is a DIFFERENT finding and must not be reported as the first.**

## ANM IS A DIAGNOSTIC, NOT THE BAR -- AND THE FLAT-CURVE PRE-COMMITMENT IS DELETED (INBOX 004)

### 004a -- the pre-committed (b)-branch is REMOVED from live code
`armf_atlas_curve.py` printed, as its own verdict output, the ANM-basis-decoder design ("predict the
ANM basis from structure, use it as the DECODER'S BASIS, latent supplies coefficients plus a learned
residual") as the recorded next step whenever the curve came back flat. **Deleted from both the
verdict block and the module docstring** before the curve produced any results. It named ONE cause
for a result nobody had seen, and the cure it named risked pulling the project into an open-ended ANM
optimisation loop. A flat curve now prints **"FLAT -- run the diagnosis fan-out"** and enumerates the
six mechanisms, never a design.

### 004b -- "beats ANM" is RETIRED as the success criterion
ANM is a fixed-topology structural prior **with no generator**. It cannot produce a programmable
latent dynamics system, which is the entire point of the project. **A one-token codec that loses to
ANM on per-frame reconstruction is NOT thereby dead.** ANM stays as the honest zero-shot comparator
answering exactly one question -- does the learned map carry information a physics prior does not --
and is reported as a **diagnostic**, never as a gate.

**THE ACTUAL SUCCESS CRITERIA, in reporting order:**
1. **RETAINS DYNAMICAL INFORMATION** -- not just per-frame FVE. Decoded trajectories must preserve the
   DYNAMICS: per-mode marginal std ratio, integrated autocorrelation time, cross-mode coupling, 2D
   free-energy projection -- the ensemble acceptance test **already built** for the propagator.
   **A codec with mediocre FVE that preserves autocorrelation structure is worth more than one with
   better FVE that flattens it.**
2. **GENERALISES TO UNSEEN SYSTEMS** -- held-out systems, not held-out frames. Already the protocol.
3. **SCALES WITH N** -- the codec-vs-N trend at L=1 from 600 to 33,500 atoms.
4. **SUPPORTS THE DOWNSTREAM GENERATOR** -- measurable NOW, not later: the latent time-series'
   autocorrelation time, its frame-to-frame smoothness, and whether an AR(1)/OU fit in latent space
   yields stable rollouts. **A latent that reconstructs well but jumps discontinuously between frames
   is useless to stage 2**, and that is far better found now than after the propagator exists.

### 004c -- FLAT-CURVE DIAGNOSIS FAN-OUT (run it; do not infer a cause from flatness)
| hypothesis | distinguishing test |
|---|---|
| decoder capacity | raise decoder depth/width at fixed L=1, DM. FVE rises => decoder-limited |
| one-token information limit | the DM sweep itself. FVE still rising at DM=512 => information-limited, not architectural |
| conditioning insufficient | enrich static features (local frames, neighbour geometry) at fixed L=1, DM. FVE rises => conditioning-limited |
| objective mismatch | geometry-aware loss (pairwise-distance or per-mode weighted). Dynamical fidelity improves while MSE does not => the LOSS was wrong, not the architecture |
| representation | local-frame / internal-coordinate target instead of Cartesian displacement (already a recorded hyperparameter) |
| encoder pooling | G6 permutation + effective rank of the latent across systems. Realised rank << DM => the encoder is not filling the token |

Report which hypothesis the evidence supports. **Only then** propose a design.

### 004d -- NO CALENDAR ESTIMATES ON OBJECTIVES 2 AND 3
Bond-changing chemistry and genuine millisecond state generation are **research problems with real
risk of not working**. They are milestones with uncertainty, **not dates**. Levels 1 and 2 (one-token
codec, then end-to-end latent dynamics) may carry estimates; those two may not.

### AUDIT SWEEP: 175 rank90-derived claims found; 101 load-bearing (91 after dedup)
An adversarial audit swept ROADMAP.md, every `outputs/cluster/*.md` writeup, and every
`scripts/armf_*.py` for claims that size the latent, the model width, or an architectural decision
from rank90 or from b. Result: **175 claims, 101 load-bearing**, spread across 18 files. The
inference chain is far wider than the width-chain table.

**The single most consequential one was not a number, it was an exclusion** (ROADMAP ~1310):
> "The physics says intrinsic dimensionality does not grow with atom count, so a fixed-width code
> suffices. **Capacity is therefore already excluded as an explanation.**"

Both premises have failed -- the "N-exponent CI spans zero" clause was already superseded by
b = +0.101 +/- 0.021, and the argument treats rank90 as an ESTIMATE when it is a FLOOR.
**Capacity is no longer excluded.** It returns as a live hypothesis and is now settled empirically
(the "one-token information limit" row of the 004c fan-out, whose test is the DM sweep: FVE still
rising at DM=512 => information-limited). The outcome-B row of the localisation table is amended so
"NOT capacity" can no longer be asserted a priori.

**Also marked, in live code rather than prose** -- `armf_atlas_b.py` and `armf_b_analyze.py` both
hold `ASYMPTOTE, FLOPPY_TO_BOUND = 168.0, 0.65` and PRINT "dims for a 1e6-atom bound complex" when
they run. Those outputs now print `>=` and carry the floor reasoning inline, so a future reader of
the log cannot mistake a lower bound for an estimate the way the DM=512 sufficiency claim was.

**And the DM grid's own justification** (ROADMAP ~1233, "DM=256 covers the median system; DM=512
covers the observed tail. Hence the sweep DM in {16,64,256,512}"): both coverage claims are read off
a lower bound. The grid is KEPT but is now a **probe, not a bracket** -- it is no longer claimed to
contain the answer from above, which is precisely why `armf_atlas_dm.py` carries the pre-registered
extension rule to DM=1024.

Remaining load-bearing claims are concentrated in the objective-3 writeups (`armf_intrinsic_dim.md`,
`armf_window_scaling.md`, `armf_slowness.md`) and the phase-1 scripts, all of which assert some form
of "dimensionality does not grow with atom count, so a fixed budget covers it." Every one of those
inherits the same floor and is to be read as a lower bound until re-derived from the codec curve.

### FULL IN-SAMPLE RESULT (n=123 held-out ATLAS systems) -- b DOUBLES OUT OF SAMPLE
`armf_rank90_insample.py`, job 10305712. Fit PCA on replicas 0+1, evaluate on replica 2.

| quantity | value |
|---|---|
| held-out variance covered by the **in-sample rank90 modes** | median **77.1%**, not 90% |
| systems where that falls below 90% | **123/123** (min 49.9%) |
| `rank90_out > rank90_in` | **118/118** of those that reach 90% at all |
| `rank90_out / rank90_in` | median **3.49x**, IQR 2.26-6.08x |
| systems that NEVER reach 90% at any k | **5/123** -- N 13,543-33,377, median 14,018 (corpus median 3,249) |
| **in-sample b** (`log10 rank90_in ~ log10 N`) | **+0.4657 +/- 0.2158** |
| **out-of-sample b** | **+0.9285 +/- 0.2220** |

**The b exponent roughly DOUBLES out of sample** -- and the out-of-sample figure is *itself* biased
low, because the 5 systems excluded from it are precisely the high-N ones that failed to reach 90%
(FAMILY A: a non-random exclusion correlated with the regressor). The censored systems being the
LARGEST is the signature the second bias predicts.

rank90_in sits at median 4.0% / max 19.3% of usable rank under the 2-replica train set (milder than
the 32.3% seen with one replica), but that percentage **RISES with N at +0.0348 +/- 0.0152** -- the
measurement ceiling binds hardest exactly where the b slope takes its leverage.

**Consequence for the "one global latent, width independent of atom count" claim:** measured out of
sample on this corpus, rank90 grows at roughly `N^0.93` -- close to linear. That does not by itself
refute the architecture, because rank90 is a per-system PCA quantity and the codec is a shared model
sized by its own saturation curve; but it does remove the last version of the *physics* argument that
a fixed width suffices. The claim now stands or falls on `armf_atlas_dm.py`.

### ⬛⬛ `b >= 0.93` IS RETRACTED. The completed measurement gives a RANGE, and it excludes 0.93.
`atlas_b` COMPLETED — **841 systems, 1.82 decades, 12:24:51**, no truncation. The 24b criterion was
fixed *before* these numbers existed and is applied here unchanged.

**Primary cell** (ordering-free `rsort` × J by the script's own inflation rule × no floor),
**mobility-controlled** — `log rank90 ~ b·log N + c·log RMSF`, which is what `b` has meant since the
original work:

| J | n | b | 95% CI | R² | vs the recorded 0.93 |
|---|---|---|---|---|---|
| 1 | 752 | **+0.6788** | [+0.600, +0.757] | 0.711 | **EXCLUDES** |
| **2 (primary)** | 831 | **+0.8280** | [+0.766, +0.890] | 0.783 | **EXCLUDES** |
| 3 | — | ordering-free variant not computed for this join | | | |

**Acceptance, as pre-registered:** spread **0.1493** ≤ 0.25 ✓, but the CIs **do not overlap** ✗.
Criterion 1 required *both*. **→ REPORT A RANGE: `b ∈ [+0.68, +0.83]` depending on join count.**

**`b ≥ 0.93` is not supported.** Both cells exclude 0.93 *from below*. The inequality form came from a
censoring argument on a smaller, truncated sample; on the complete corpus the ordering-free exponent
is **+0.828 ± 0.062**. The single-value and lower-bound forms are both retired — per 21b the **range
is the reportable quantity**.

**Criterion 3 fires too:** across the floor sweep the ordering-free J=2 value moves +1.01 → +0.85 →
+0.32, and J=1 flips sign (+0.75 → −0.78). **`b` is not robust to sample restriction**, and that
caveat travels with it.

**For contrast, at the primary J=2:** in-sample **+0.2977 ± 0.0371**, train-ordered **+0.8379 ±
0.0598**, ordering-free **+0.8280 ± 0.0620**. The in-sample/out-of-sample gap is **2.8×** — the
in-sample exponent answers a different question, as INBOX 011 established.

**What this does NOT change:** the width chain was already retired as a sizing instrument under INBOX
002, so nothing downstream depended on `b`'s value. What changes is the recorded number.

### ⬛ PRE-REGISTERED (INBOX 24b): WHAT WOULD MAKE `b` REPORTABLE — written BEFORE the re-run lands
Finishing `atlas_b` removes the **truncation** (the top 18% of log-range was missing). It does
nothing about the **dispersion**, and those are independent defects. On the partial data `b` ranges
**−0.83 to +1.05 including a sign flip**, produced by two choices that are *ours* — the sample floor
and the replica-join count — not the data's.

**PRIMARY CELL, fixed now, before any number is seen:**
- **variant** = `j{J}_rsort`, out-of-sample **sorted** — ordering-free, which INBOX 011 established
  is the architecture-relevant one.
- **J** = chosen by `armf_atlas_b.py`'s OWN pre-existing rule: *lowest budget-matched inflation among
  joins whose p90 rank usage is under the 30% censoring line*. That rule predates this question and
  was not written by me after seeing `b`.
- **floor** = **none**, all systems. The floor sweep is my sensitivity probe; it was never part of
  the claim, and introducing a floor would itself be a free choice.

**ACCEPTANCE, also fixed now:**
1. Report `b` as a **value** only if, across the uncensored `J` cells at no floor on the
   ordering-free variant, the point estimates span **≤ 0.25** *and* their CIs mutually overlap.
2. If the spread exceeds 0.25, **report `b` as a RANGE** over those cells and retire the
   single-value form: `b ≥ 0.93` becomes `b ∈ [lo, hi], depending on join count`.
3. The floor sweep can only ever **add a caveat**, never upgrade `b`. If it flips the sign or moves
   `b` by more than 0.5, record that `b` is not robust to sample restriction.
4. If no `J` clears the 30% line, report as **censored** — the script already does this.

**And if the spread stays wider than the claim, the honest output is the range, not a value** — with
it stated that `b` is produced by analysis choices as much as by data, so any width chain built on it
inherits the range rather than the number. Per **21b** the range *is* the reportable quantity; `b ≥
0.93` is a label.

### CORRECTION: b is an INEQUALITY, `b >= 0.93` (INBOX 007)
The out-of-sample exponent `+0.9285 +/- 0.2220` must be recorded as **`b >= 0.93`**, not as a point
estimate. The 5 excluded systems are the LARGEST in the corpus and were excluded **precisely because
they need more modes than the data can resolve** -- so the exclusion is not incidental to the
measurement, it is the mechanism that flattens the slope. Dropping the high-N systems that need the
most modes biases b downward by construction (FAMILY A). The honest statement is **b >= 0.93,
plausibly near-linear.**

### Q1 SUBMITTED: does SLOWNESS dimensionality grow with N? (job 10306540, `armf_tica_vs_n.py`)
rank90 is **variance**-weighted. Near-linear growth in it is close to what independent local thermal
motion would produce -- every extra atom brings its own fast, low-amplitude degrees of freedom. Those
modes are real but are **not the dynamics a latent generator must represent.** The slowness-weighted
analogue was measured for objective 3 and was FLAT (TICA dim 32-42 across five temperatures at 36-38%
of basis, uncensored, a bound 5x tighter than rank90's) -- but that was flat across **effective
time**, and it has never been measured across **N**.

Definition taken verbatim from `armf_slowness.py:54` so the two results are comparable. Guards: lag
swept on TRAINING systems and applied unchanged (E); dimension reported as % of basis with a
larger-basis re-measurement (B); kept-vs-dropped median N for any system failing to reach 90% of the
slow spectrum (A); in-sample AND out-of-sample, since rank90's exponent doubled between them.
Cross-replica correctness: lagged pairs accumulate WITHIN each replica, never across the 0/1 join.

**A CENSORING PROBLEM FOUND IN THE DRY RUN, BEFORE SUBMISSION.** The out-of-sample TICA dimension
pins at **86-87% of basis at every basis tried (100, 150, 300)** -- over the 60% line, so a flat
slope there would be a CEILING. Two consequences, both wired into the script:
1. The in-sample dimension is **itself basis-tracking** (36% of 100, 38% of 300), which is the exact
   pathology `armf_slowness.py` documented and why it fixed the basis at 100. **The fixed-basis TICA
   number is comparable across systems but is NOT an absolute dimensionality and must not be reported
   as one.**
2. The out-of-sample reading is split in two: **train-order** (ordering-sensitive, the direct
   rank90_out analogue) and **ordering-free** (ranked by actual held-out slowness). If the first far
   exceeds the second, the train basis is sound but its ORDER does not transfer -- a different defect
   from slow dynamics being high-dimensional, and **only the second bears on the architecture
   question**. The verdict uses the ordering-free quantity and is WITHHELD entirely if it is still
   above 60% of basis.

**Pre-registered consequence if flat:** a variance-weighted objective (plain MSE on displacement)
spends the token's capacity in proportion to variance -- i.e. mostly on the fast local modes growing
as N^0.93. If slow-mode dimensionality is flat, **MSE is the wrong training objective for this
architecture.** That is the "objective mismatch" row of the 004c fan-out, and it would then carry
evidence rather than being one hypothesis among six. **Do not change the loss on this basis yet.**

## Q3 DONE: THE CRITERION-1 HARNESS EXISTS AND IS PROVEN TO DISCRIMINATE (`armf_criterion1.py`)
INBOX 004b makes "retains dynamical information" the FIRST success criterion, ahead of per-frame FVE.
**A criterion that cannot be measured is not a criterion**, so the harness was built and validated
BEFORE the DM sweep produced a live arm.

What changes from the propagator's version: the propagator ROLLS OUT a trajectory, the codec DECODES
the reference frames. So the question is not "did it invent plausible dynamics" but **"did passing
the trajectory through the bottleneck destroy the dynamics that were already there"** -- the failure
that makes a latent useless to stage 2 while looking fine on FVE.

Four discriminators, all in ONE basis fitted on the REFERENCE (a per-trajectory basis would let a
decoder that rotated the dynamics into other directions still score perfectly -- Family D):
1. marginal std ratio per mode · 2. **IAT per mode -- the kinetic test** · 3. cross-mode coupling ·
4. 2D free-energy JS divergence on the top-2 modes.

**VALIDATED AGAINST THREE KNOWN INPUTS (n=800 frames, real ATLAS system):**

| control | 1 std | 2 IAT (kinetic) | 3 coupling | 4 free energy | total |
|---|---|---|---|---|---|
| identity (decoded == reference) | PASS 100% | PASS 100% | PASS 100% | PASS 0.0000 | **4/4** |
| **frames SHUFFLED** | PASS 100% | **FAIL 0%** (30.0 -> 1.0 frames) | PASS 100% | PASS 0.0000 | 3/4 |
| variance collapse (0.3x) | FAIL 0% | PASS 100% | PASS 100% | FAIL 0.9272 | 2/4 |

**The shuffled control is the one that matters.** Shuffling preserves every marginal, every
cross-mode coupling and the entire free-energy surface EXACTLY, destroying only time ordering -- so it
passes 1, 3 and 4 and must fail 2. **A harness that passed shuffled frames would be measuring
distribution rather than dynamics and would certify a latent the propagator cannot use.** It fails it,
at 0% of modes retained. The variance-collapse control confirms discriminator 1 catches an
MSE-minimising decoder that regresses toward the mean -- which is the specific failure INBOX 004b
warns FVE would reward.

Reporting rule enforced in the code: **FRACTIONS PER DISCRIMINATOR, never a mean across them.** A mean
would let a catastrophic kinetic failure hide behind three passing distributional checks -- precisely
the case the harness exists to catch.

## FIRST REAL DM-SWEEP ARMS (job 10305995, n_train=50, L=1) -- GUARDS FIRST

**GUARDS, before any number is quoted.** 11 arms completed at n_train=50. Of those:
- **2 VOID (Family D, still improving at MAXSTEPS):** DM=16 lr=3e-4, DM=256 lr=3e-3.
- **4 COLLAPSED to a constant code (PR = 1.0-3.3, FVE ~= 0):** DM=64 lr=3e-3, DM=256 lr=1e-3,
  DM=256 lr=3e-3, DM=512 lr=3e-4.
- **5 VALID:** DM=16 lr=1e-3/3e-3, DM=64 lr=3e-4/1e-3, DM=256 lr=3e-4 (x3 seeds).

### THE LR SWEEP WAS NOT OPTIONAL -- it changes the conclusion at every width
| DM | best LR | FVE at best | FVE at 1e-3 | FVE at 3e-3 |
|---|---|---|---|---|
| 16 | 3e-3 | +0.1185 | +0.1143 | +0.1185 |
| 64 | 3e-4 | +0.1068 | +0.0986 | **+0.0000 (collapsed)** |
| 256 | 3e-4 | **+0.1553** | **-0.0001 (collapsed)** | **-0.0004 (collapsed)** |
| 512 | -- | **collapsed at 3e-4, the grid FLOOR** | -- | -- |

**The optimal LR falls monotonically with width, and the mdCATH collapse is explained.** DM=256 does
not collapse because 256 dimensions cannot be trained; it collapses **at the wrong learning rate**.
With lr=3e-4 it is the BEST arm. Had the LR not been swept, this run would have reproduced the
mdCATH conclusion -- "the wide arms collapse" -- and it would have been an artifact both times.
**FAMILY E, caught by the control built to catch it.**

**AND THE SAME ERROR WAS STILL LIVE AT THE TOP.** DM=512 collapsed at **3e-4, the floor of my grid**,
so the widest arm was losing on an unswept hyperparameter rather than on capacity -- the grid simply
did not extend far enough. Floor lowered to **3e-5** and the sweep resubmitted (job 10306611). Cost
controlled by running the full grid only at the cheapest n_train and transferring the winner plus one
lower neighbour up the ladder.

### SEED REPRODUCIBILITY (control 5) -- the valid wide arm is stable
DM=256, lr=3e-4, three seeds: **+0.1453 / +0.1553 / +0.1497** (spread 0.0100), PR 15.6 / 14.9 / 16.7.
**Collapse rate 0/3 at the right LR**, against 2/2 at the wrong one. A one-seed collapse would have
been an anecdote; this is a rate.

### PRELIMINARY, AND THE MOST INTERESTING NUMBER SO FAR: PR DOES NOT GROW WITH DM
| DM | PR (conformational, cross-fit, within-system centred) | PR/DM |
|---|---|---|
| 16 | 10.8 | 68% |
| 64 | 16.2 | 25% |
| 256 | 15.6 / 14.9 / 16.7 | 6-7% |

**The code uses ~15 effective conformational dimensions regardless of how wide the token is.** PR is
not censored here (n_obs ~1,845 >> 2xDM=512 at DM=256), and it is measured on within-system-centred
latents, so it counts conformational directions and not system identity. If this survives the wider
LR grid and the n_train ladder, **the latent saturates far below DM** -- which is what the 005
bottleneck arm exists to confirm independently.
**Identity share is 86-91% at DM=256**: most latent variance encodes WHICH system, not how it moves.

### THE HEADLINE QUESTION, PRELIMINARY: FVE-vs-N IS FLAT
Across N = 598-33,377 (1.75 decades), evaluated on 123 UNSEEN systems, the valid arms give slopes
**-0.0139 +/- 0.0593, -0.0152 +/- 0.0566, -0.0143 +/- 0.0547** (DM=256, three seeds) and
**-0.0055 +/- 0.0433** (DM=16). Consistently slightly negative, **every CI spanning zero**.
**SCOPE: n_train=50 only, and absolute FVE is ~0.15 -- low. A flat slope on a weak model is a much
weaker claim than a flat slope on a strong one, and the ladder must land before this is read as
"the codec holds flat with N".**

### Q4 DONE: the audit is now marked IN THE FILES, not only here
A claim corrected centrally but left standing locally is how the Family-F hardcoded baseline
survived, so the remaining load-bearing rank90 claims are marked where a reader will actually meet
them. Five writeups carry a **LOWER BOUND banner directly under the title**
(`armf_intrinsic_dim.md`, `armf_window_scaling.md`, `armf_slowness.md`, `armf_temp_exploration.md`,
`armf_phase1_abc.md`) giving the out-of-sample numbers, and the individual load-bearing sentences are
struck through in place. Live code marked at the point of use: `armf_atlas_b.py`,
`armf_b_analyze.py`, `armf_intrinsic_dim.py`, `armf_phase1_dm.py`, `armf_phase1_analyze.py`,
`armf_atlas_data.py`.

**Two that mattered more than the rest:**
- `armf_phase1_analyze.py` PRINTED, at runtime, that rank90 and TICA "already EXCLUDE capacity -- the
  physics says a fixed-width code suffices. DO NOT call this a capacity limit." That is the retracted
  inference, in an executable telling a future reader not to consider the hypothesis that is now
  live. Replaced with an instruction to rule capacity out via the DM sweep FIRST.
- `armf_slowness.md` claimed the TICA result "licenses sizing DM from rank90/TICA". **The SIZING half
  is retired; the TIME half stands.** Flat-in-time was never flat-in-N, and TICA-dim vs atom count
  was never measured until job 10306540 -- the distinction the writeup silently elided.

## INBOX 011 ANSWERED: THE ORDERING HYPOTHESIS IS REFUTED (job 10306553, n=123)
011 proposed that part of rank90's near-linear out-of-sample growth might be the train basis
**ordering** worse at large N rather than held-out content being higher-dimensional, and that if so
the physics argument for a fixed width would be partially rehabilitated. **Measured, it is not.**

| quantity | median | exponent in N |
|---|---|---|
| rank90_in | 138 | +0.4657 +/- 0.2158 |
| rank90_out, TRAIN ORDER | 426 | **+0.9285 +/- 0.2220** |
| rank90_out, ORDERING-FREE (cross-fit) | 382 | **+0.9429 +/- 0.2306** |
| ORDERING PENALTY (ordered/sorted) | **1.03x** | **-0.0145 +/- 0.0195** (CI spans 0) |

The ordering penalty is **3%**, it does **not** grow with N, and the ordering-free exponent is
**slightly HIGHER** than the ordered one, not lower. Sorting on the evaluation data inflates by only
2% (naive/cross-fit 1.02x), so the cross-fit was worth doing and changes nothing.
**The growth is real dimensionality. The fixed-width argument stays retired, and `b >= 0.93` stands
on the ordering-free number as well as the ordered one.**

## Q1 FIRST RESULT AND ITS OWN CONTROL: THE FLATNESS WAS TRUNCATION, NOT SLOWNESS
Job 10306540 (n=123, tau=20 chosen on TRAINING systems) returned TICA exponents far below rank90's:
`+0.0089 +/- 0.0041` (in-sample, basis 400) up to `+0.1162 +/- 0.0130` (in-sample, basis 100), every
variant at every basis EXCLUDING rank90's `+0.9285`. Read naively that is the 007 result: slow
dynamics nearly flat in N while variance dimensionality grows near-linearly.

**It does not survive the control, and the control had to be built because TICA is computed inside an
m-dimensional PCA basis while rank90 is not truncated at all.** Any count confined to m components is
bounded by m, so its N-slope is compressed toward zero *whatever* it weights by. Measuring VARIANCE
dimensionality inside the SAME basis (n=14 spanning the full N range, m=100):

| quantity | exponent in N |
|---|---|
| TICA dim, in-sample | +0.0903 +/- 0.0200 |
| **PCA dim, SAME basis, in-sample** | **-0.0580 +/- 0.2891** |
| TICA dim, out-of-sample sorted | +0.0061 +/- 0.0191 |
| **PCA dim, SAME basis, out-of-sample** | **+0.0291 +/- 0.1525** |

**Variance dimensionality inside the truncated basis is ALSO flat.** So "TICA +0.06 vs rank90 +0.94"
compared a truncated count against an untruncated one -- apples to oranges, and a **FAMILY D** error
if reported: the measurement is structurally unable to express growth beyond m.

Three further reasons the raw TICA number cannot carry the claim, all measured:
- **The dimension TRACKS THE BASIS**: median 41 of 100 and 164 of 400 -- exactly 41% both times. It
  is not an absolute dimensionality, which is the pathology `armf_slowness.py` documented and why it
  fixed the basis at 100.
- **The exponent MOVES 13x with the basis** (+0.1162 at m=100 vs +0.0089 at m=400, in-sample), which
  is direct evidence the number is a property of the truncation.
- **n_eff per TICA dimension is 0.81, below 1.0 in 103/123 systems** at basis 400 -- TICA fits more
  dimensions than it has independent samples, exactly the Family B failure 007 warned of. (007 asked
  for this statistic explicitly and my first implementation did not report it.)

The verdict block WITHHELD a verdict on censoring grounds, which was the right instinct for a
slightly wrong reason: **no system is pinned at the cap** (max 364 of 400), so the >60%-of-basis flag
was a conservative proxy. The real defect is the truncation compressing the slope for every system.

**Re-running with the truncation-matched control at n=123 (job 10306738).** The controlled quantity
is TICA-vs-PCA *within the same basis*; only if variance dimensionality is materially steeper than
slowness dimensionality IN THAT BASIS does 007's claim hold. **No 007 verdict may be quoted until
that lands.**

### RETIRED (INBOX 19b): THE A/B/C DECISION TREE — data-limited vs fundamental
The learning curve was built to separate **(a) data-limited** from **(b) fundamental**. **16a returned
a third answer the ladder cannot produce: ARCHITECTURE-LIMITED, and specifically DECODER-limited** —
realised rank **6** against a data rank of **152**, with the latent already carrying **~15**.

It was a good tree for the question as posed. The realised-rank measurement **changed the question**,
and a tree left standing gets applied: applying it now would force a third outcome into two boxes and
would read "more data does not help" as evidence for *fundamental* when the actual constraint is a
decoder that converts fewer than half the directions it is handed. **Do not use the A/B/C framing in
any new report.** The live question is which decoder, not how much data — see INBOX 015/17b.

### HYPOTHESES MEASURED AND DECLINED — WITH NUMBERS (INBOX 017)
Kept together deliberately. A file that records only what survived reads as a run of successes and
tells a later reader nothing about what was already tried and refuted. Each of these was proposed,
measured properly, and **did not hold** — and in each case the measurement, not an argument, is what
declined it.

| # | hypothesis | what was measured | outcome |
|---|---|---|---|
| **011** | `rank90_out` was inflated by TRAIN-ORDER counting, so the ordering-free exponent should be lower | ordering penalty **1.03×**; N-slope of the gap **−0.0145 ± 0.0195** (spans zero); ordering-free exponent **+0.9429 ± 0.2306**, *higher* than train-order | **DECLINED.** The ordering artifact is real but negligible, and removing it moves the exponent the *wrong* way for the hypothesis. |
| **007** | slow-weighted dimensionality grows more slowly in N than variance-weighted | matched-*m* `exponent(TICA) − exponent(PCA)` at the viable basis (m=100, n_eff/dim 3.28): **+0.0873 ± 0.1163** out-of-sample — positive, spans zero | **DECLINED — a measured null,** not an unanswerable question. Sign flips at m=400, confirming truncation dependence. |
| **14b** | the codec keeps slow/collective motion and discards thermal noise, so MSE is the wrong objective | naive `FVE ~ log(IAT)` **+0.4700 ± 0.1917** (significant), but **partial** coefficient holding variance share fixed **+0.1445 ± 0.1830** — spans zero | **DECLINED.** Variance-selective, not timescale-selective: MSE selects for variance by construction. |

**What survived instead:** 16a — realised rank **6** against a data rank of **152** with the latent
already carrying **~15**. That one is not a null, and it is the only one of the four that points at a
specific fix.

### 007: **ANSWERED at m=100 (n_eff/dim 3.28, viable); NOT MEASURABLE at m=400 (n_eff/dim 0.81)**
> **INBOX 21d — ALWAYS BOTH HALVES, TOGETHER.** A bare "007 is a measured null" is precisely the
> claim-without-its-range that 21b forbids, and it would be cited that way within a week. The null
> holds *at the basis where the instrument is adequately sampled*; at the larger basis the instrument
> cannot answer at all, and the difference changes sign between them.
Ratio-only, per INBOX 13a. **No absolute TICA exponent is quoted, here or in the output.**

| basis | n_eff per TICA dim | in-sample DIFFERENCE | out-of-sample DIFFERENCE |
|---|---|---|---|
| **m = 100** | **3.28 — VIABLE** | +0.1310 ± 0.1252 | **+0.0873 ± 0.1163** |
| m = 400 | 0.81 — **CENSORED** | −0.0766 ± 0.1713 | −0.0384 ± 0.1575 |

`DIFFERENCE = exponent(TICA|m) − exponent(PCA|m)`, conservative CI (sum of half-widths). 007's claim
requires this to be **significantly negative**. At the adequately-sampled basis it is **positive and
spans zero** on the architecture-relevant out-of-sample variant. **007's claim is NOT SUPPORTED.**

**And the sign FLIPS between m=100 (+0.087) and m=400 (−0.038)** — a quantity whose sign depends on
where the basis is truncated is a property of the truncation. That is precisely why only the
matched-*m* difference at a viable basis is admissible, and why sweeping *m* for a basis that
"works" would have manufactured whichever answer was wanted.

> **I PREDICTED THIS WRONG, AND THE REASON MATTERS.** In the 013 ACK I wrote that the stopping rule
> *"will very likely fire — I already measured n_eff per TICA dimension at 0.81."* That 0.81 is the
> **m=400** figure; at the canonical m=100 basis it is **3.28**, comfortably viable. I quoted a number
> from one basis as though it characterised the measurement. The instrument was adequately sampled
> where it counted, so 007 gets a **measured null**, not a closure for unanswerability — the exact
> distinction 14c asked to be preserved, landing on the opposite side from my prediction.

> **AGGREGATION DEFECT, FOUND AND FIXED.** The first run applied the stopping rule to the median
> n_eff **pooled across both bases** — median([3.28, 3.28, 0.81, 0.81]) = **2.045**, missing the 2.0
> threshold by 2%. Viability is a **per-basis** property; pooling produced a number describing neither
> basis and let a censored basis drag a viable one toward the cliff. Now applied per basis, with
> censored bases excluded from answering and printed as excluded. **Same defect as 16a's hand-picked
> threshold: a verdict decided on a knife-edge by an aggregation nobody examined.** In both cases the
> measured numbers never moved — only the automated reading of them. Re-run as job 10307083; the
> pooled-verdict log is archived as `ticaN_10306738_v1_pooledverdict.log`.

**INBOX 14c — RECORD WHICH KIND OF NEGATIVE THIS IS.** If the stopping rule fires, the finding is
**"TICA dimensionality is NOT MEASURABLE at these trajectory lengths"** — n_eff per TICA dimension
0.81 against a threshold of 2.0 pre-registered before the re-run. That is a statement about the
corpus, and it is corpus-independent in the same way INBOX 002 is: **no dataset supplies the
trajectory length that would fix it**, because the requirement grows with the dimensionality being
fitted. It is emphatically **not** "we did not find a good *m*", which would be an admission of
insufficient search and would license exactly the *m*-sweep that turns the answer into a chosen
result. The exponent moves **13×** across the *m* values tried; picking the one that reports a
result is picking the result. The two statements license opposite next actions — one closes the
question, the other reopens it as a search — and the writeups say which one this is.

### ⬛ THE CENTRAL QUESTION, AND WHAT A CLEAN L=1 RESULT WOULD *NOT* ESTABLISH (INBOX 025)
> **Can one fixed-width global latent token encode the dynamic state of an unseen molecular system
> well enough to reconstruct its atom-level motion, without performance collapsing as N increases?**

Three clauses. **Unseen systems** is properly instrumented (criterion 2: 123 held-out systems,
598–33,377 atoms). **Width** is pre-registered with the right guard (PR flat while FVE rises =
saturation; PR rising with FVE = capability-limited, not a width answer). The other two need care:

**"Encode the dynamic state" cannot be read off FVE.** MD displacement variance is dominated by a few
slow collective modes, so a model reproducing *only those* scores well on aggregate FVE **by
construction** — and the measured state is a **~6-mode effective output with 94% residual**. The
discriminating measurement is **FVE on the ANM-orthogonal residual**: of the motion the zero-shot peer
does *not* span, how much does the codec explain? Plus the **per-atom error tail** (p90/median,
normalised per atom), because a summary model is right on the mobile core and wrong in the tail and a
mean hides that. Both are reporting-only, on outputs that already exist.

**"Without collapsing as N increases" IS the N-slope, and `b` is not currently quotable** — it spans
−0.83 to +1.05 across analysis choices, with a sign flip driven by the sample floor alone *within a
single variant*. Until the 24b pre-registration is settled the third clause is unanswerable
**regardless of how the L=1 arm scores**, which is why it is being settled while the curve runs.

**WHAT A CLEAN L=1 RESULT WOULD ESTABLISH:** the **compression mechanism** — that a fixed-width global
latent carries transferable dynamic state across unseen systems without an N-collapse.

**WHAT IT WOULD NOT ESTABLISH, recorded now so the result is not over-read on arrival:**
- **not 1M atoms** — measured to ~33k, and §6.2 names two hard caps above that;
- **not bond breaking** — topology is an input (§6.4);
- **not millisecond generation** — no propagator exists; whether the latent is *modellable* is
  criterion 4 and is a separate measurement.

Those remain downstream. What a clean result would do is make them **worth attempting**, which
nothing so far has.

> **STANDING CONTEXT (INBOX 22c) — carry all three in front of ANY statement about flatness in N:**
> **the codec sits at 19% of a per-system oracle, loses to the zero-shot peer on 0% of systems, and
> 65% of that gap is basis quality.** A flat slope on a model this far from achievable is consistent
> with uniform weakness — and that context is now *measured*, not asserted.

### ⬛ 14a RESULT (job 10307401, n=123 held-out, **n_train = 50**): THE CODEC LOSES TO THE ZERO-SHOT PEER, EVERYWHERE
The comparison the project exists to make, finally in an output. **ANM computable on 123/123 systems
across all three N terciles — no Family A exclusion.** Cutoff 5 Å, swept on TRAINING systems only.

| n_train | DM=k | codec | ANM-k | codec−ANM | frac codec>ANM | PCA-k oracle | % of oracle |
|---|---|---|---|---|---|---|---|
| 50 | 16 | 0.1185 | 0.4032 | **−0.2847** | **0%** | 0.5274 | 22% |
| 50 | 64 | 0.1068 | 0.5460 | **−0.4393** | **0%** | 0.6929 | 15% |
| 50 | 256 | **0.1553** | 0.6720 | **−0.5168** | **0%** | 0.8242 | **19%** |

**The codec loses to ANM at every width, on 0% of systems, at matched capacity.** Best arm reaches
**19%** of a per-system PCA oracle. Per 004b this is *not fatal* — ANM has no generator and cannot be
the product — but it belongs in front of every FVE-vs-N statement, and **a flat slope on a model this
far from the achievable is consistent with uniform weakness.**

**ONE THING RUNS THE OTHER WAY — AND IT IS SUGGESTIVE, NOT A RESULT (INBOX 22b).** The codec−ANM
*gap* shares a denominator with its comparator, so an N-dependent ceiling cancels; it is the
ceiling-free quantity and the only measurement currently pointing the right way. At DM=256 it is
**+0.0514 ± 0.0464 per decade** — a CI of roughly **[0.005, 0.098]**, *barely* excluding zero, in
5 of 7 arms. **The same discipline applies in this direction: a barely-significant positive is not
more trustworthy than a barely-significant null because it is the answer we want.** It amounts to
≈ **+0.09 over the measured range against a −0.52 gap**. Right quantity to track; not yet a result.

**DM=512 has no matched peer column** (ladder capped at k=256 on ANM cost) and this is printed, not
silent. It does not hide the winner: DM=512's best is +0.1427, below DM=256's +0.1553.

### ⬛⬛ 25a RESULT: THE CODEC IS A COLLECTIVE-MODE MODEL — the pre-registered verdict fired
Best arm (n50, DM=256, lr 3e-4, seed 1, FVE **+0.1553**), 24 N-stratified held-out systems.
Decided as a **distribution** per 26d, with **zero as the constructed boundary** (a model reproducing
the ANM subspace exactly and nothing else gives `FVE⊥ = 0` identically):

| | ANM-6 | ANM-16 |
|---|---|---|
| peer spans | 24% of the motion | 37% |
| **`FVE⊥` median** | **−0.0261** | **−0.0439** |
| IQR | [−0.0985, +0.0410] | [−0.1328, +0.0006] |
| **above zero** | **33% of systems** | **25%** |
| `FVE⊥` vs log10(N) | **−0.1844 ± 0.1154** | **−0.2232 ± 0.1635** |

**On the motion the zero-shot peer does not span, the codec explains nothing — and for two-thirds of
systems it slightly *adds* error.** That is the pre-registered "collective-mode model" branch: real,
publishable as such, **but it is not an answer to "can one token encode the dynamic state," and the
honest headline changes.**

**And the discriminating metric DEGRADES WITH N while aggregate FVE stays flat.** `FVE⊥` vs log10(N)
is **−0.1844 ± 0.1154** — the CI **excludes zero** — against the same arm's aggregate FVE slope of
−0.0152 ± 0.0566, which does not. **This is exactly the failure 25a was built to detect: the metric
the central question is usually judged on is flat, and the metric that can actually distinguish the
two hypotheses is falling.** The third clause of the central question is *worse* on the discriminating
measure than the aggregate one suggests.

**PER-ATOM (25a-2): median 1.001, IQR [0.955, 1.037], p90 1.207.** A predict-zero baseline gives
**exactly 1.000** by construction — so **the median atom is reconstructed no better than predicting no
motion at all**, and the +0.1553 aggregate FVE is carried by a minority of high-amplitude atoms.

**One correction to how the pre-registered table reads here.** 025's first row is "`FVE⊥` ≈ 0 **with a
wide tail** → collective-mode model", the wide tail expressing "right on the mobile core, wrong in
the tail". The measured tail is **narrow** (p90/median 1.20×) — but narrow *around 1.0*, with 50% of
systems below it. That is not selective success with a bad tail; it is **uniform failure at the
per-atom level**. The conclusion is the same and arrives by a simpler route than the table
anticipated, and saying so is more accurate than forcing the observation into the row.

### ⬛ 26a **(at n_train = 50)**: MY √N MECHANISM IS REFUTED — and the controls found something I did not predict
`log10(‖z‖/‖disp‖)` vs `log10(N)`, trained checkpoints, forward passes only, **123 held-out systems
over 1.75 decades**:

| arm | slope | 95% CI | R² |
|---|---|---|---|
| **tied** | **−0.0992** | [−0.163, −0.035] | 0.072 |
| untied | **−0.5217** | [−0.558, −0.485] | **0.868** |
| control | **−0.5217** | [−0.558, −0.486] | **0.871** |

**The mechanism predicted +0.5 for tied. Measured −0.099. It is dead.** My reasoning was that a
*coherent* analysis sum grows like N and the `1/√N` normaliser leaves a residual `√N`. The tied arm
being nearly **flat** says the opposite: the normaliser roughly cancels the growth, so the analysis
sum behaves ~`√N`, i.e. per-atom displacement contributions are quasi-**independent** in the fixed
reference frame, not coherent. The premise was wrong, not just the exponent.

**And the controls — which existed only to show the effect was specific — carry the real finding.**
Both attention encoders sit at **−0.52 with R² ≈ 0.87**, a far tighter relationship than anything on
the tied arm: `‖z‖/‖disp‖ ∝ 1/√N`. That is what a softmax-weighted **mean** over N tokens gives when
per-atom contributions are quasi-independent — the latent is an average, so it shrinks relative to a
`‖disp‖` that grows like `√N`. **Across ATLAS's range the code magnitude falls 7.5× relative to the
displacement it must encode, in the L=1 design-point architecture.**

**Stated with the discipline 22b demanded, in the direction that cuts against making this a
headline:** it does **not** currently manifest as an FVE N-slope — the control's is **−0.0383 ±
0.0606, flat** — so the decoder evidently absorbs it (it is nonlinear, with FiLM and layer norms).
This is a **measured property of the encoder**, not a diagnosis of anything, and it is recorded
because it is a strong, unexplained, systematic N-dependence sitting inside the design point.

**Consequence for the tied −0.65 FVE slope: it now has NO mechanism behind it.** One arm, one seed,
unexplained. 26a was built to be decisive from one arm and it was — it just decided against the
hypothesis it was built to test.

### ⬛ 32a: the seed finding does NOT touch the peer result, and its direction is NOT established
**Direction, unresolved.** "The control's favourable seed understates tied's advantage" resolves the
bias from **one side's** variance. **Tied's Q1 +0.2956 is also a single draw**, seed 0, spread
unmeasured. So the 2.67× ratio is **one draw over one draw**, with the denominator's SD known
(0.017–0.033 depending on rate) and the numerator's unknown. Tied's spread is now being measured at
each rung by the 3-seed ladder (10311078).

**What it does not touch, and this is the headline.** Tied Q1 **+0.2956** against ANM **+0.6912** is a
gap of **0.396 — 12× the measured single-arm SD.** No plausible seed draw closes it.
**"No architecture beats zero-shot ANM at any N" survives the seed finding intact.**

### ⬛ SETTLED (24c, job 10307865, COMPLETE 18/18 arms): the control's "+0.1346" is the TOP of a 3-seed range

| lr | seeds | mean | spread |
|---|---|---|---|
| 3e-5 | +0.0978 / +0.1179 / +0.1318 | +0.1159 | 0.0341 |
| 1e-4 | +0.1112 / +0.1216 / +0.1250 | +0.1193 | 0.0137 |
| **3e-4** | **+0.0690 / +0.1089 / +0.1346** | **+0.1042** | **0.0656** |

**The +0.1346 quoted throughout this file as "the control's best arm" is the top of that range**, not
a typical draw; its mean is +0.1042. Every control comparison here — including tied's Q1 ratio of
2.67× — is against a favourable seed. *(The bias understates tied's advantage rather than inflating
it. That is convenient, and it is a separate fact from the number not being what it was presented
as.)*

**24c's objection is being borne out:** at lr3e-4 the between-**seed** spread (0.0656) exceeds both
the between-**rate** scatter (0.0433) and the gap the sweep adjudicated (0.0350). Median seed spread
0.0239 — a ratio of **0.55**, above the 0.5 line pre-registered for "the LR sweep never had the
resolution".

**FINAL, all 18 arms.** Median between-seed spread **0.0353** against between-rate scatter 0.0433 —
a ratio of **0.82**. *The LR sweep never had the resolution, and the non-monotonicity was noise.*
"The modal arm is not losing on an unswept hyperparameter" is **WITHDRAWN as unsupported** — not
refuted, unsupported. This is Family E dismissed on evidence that could not carry it.

**And the gap table had a defect that changed a verdict word.** As first printed:

| lr | difference | as printed | Welch df | corrected half-width | corrected |
|---|---|---|---|---|---|
| 3e-5 | +0.0114 | ± 0.0298 → INDISTINGUISHABLE | 2.70 | ± 0.0364 | NOT RESOLVABLE |
| **1e-4** | **+0.0381** | **± 0.0364 → "control ahead"** | **2.44** | **± 0.0478** | **NOT RESOLVABLE** |
| 3e-4 | +0.0230 | ± 0.0608 → INDISTINGUISHABLE | 3.15 | ± 0.0678 | NOT RESOLVABLE |

The p-value was **Welch** (`equal_var=False`) but the half-width multiplied the Welch SE by a
**Student** quantile, `t.ppf(.975, nₐ+n_b−2)` = df 4. Two distributions on one line. It printed
`+0.0381 ± 0.0364` — a CI *excluding zero* — beside **p=0.080**, which is impossible; the true Welch
df is **2.44**, the half-width 0.0478, and the CI [−0.0097, +0.0858] contains zero. **"control ahead"
was the only non-null cell in the table and it was an artifact of the mismatch.** This is
[33a](#) one level down: the right statistic with the wrong distribution for it.

Both remaining cells were also unbounded nulls. Routed through `null_verdict` against
`GAP_UNDER_TEST = 0.0350`, **all three rates are NOT RESOLVABLE** — every CI contains both zero and
the gap being adjudicated. This *strengthens* 24c's own conclusion: the gap is not resolvable
against training noise, and now no cell claims otherwise.

### ⬛ 044: every checkpoint-derived result now carries its `n_train`, and atlas_dm's rung math

**44b — the provenance the filenames now state, the headings now state too.** All 15 legacy
checkpoints were renamed `_n50`; the three results computed *from* them are 28b, 29b/30 and 26a, and
32c had attached "at n_train=50" only to the peer comparison. All now carry it in their heading. This
is the specific failure mode behind 41a: a number joined across contexts because its scope was
reconstructable rather than attached.

**44a — the rung math, measured rather than hoped.** 47 arms in a 20:00:08 wall = **25.5 min/arm**.
The log reached the bottleneck sweep (`DMOD=512`) and completed `dlat` 512/16/64, so n50 owes
**~2 bottleneck arms + ~4 addressing arms ≈ 2.5 h**. n130's grid is *narrower* by construction —
`grid = {winner, winner/3}` instead of all five rates — so ≈ 23 arms ≈ 9.8 h at the n50 rate.

**But the n50 rate is the wrong rate for n130, and the ladder says so directly:** at n130 the tied
ladder ran **3/3 arms to the step cap** while n50 plateaued at 57–70k. Arms that hit `maxsteps` cost
3–5× one that plateaus early. So n130 could plausibly take 20 h+, not 9.8 h — 044's concern is real,
though the mechanism is arm duration rather than rung order.

**The mitigation is already in place:** the requeue was submitted as a **chain** (10314124 →
10314125, `afterany`), giving 40 h, and the resume is per-arm from `atlas_dm.json`. So n130 completes
across the two walls without a second writer. A separate n130 job is **not** launched — two processes
appending to one `atlas_dm.json` is a lost-update race, which is worse than the problem it solves.

### ⚠ 41c FIRST ATTEMPT REPORTED n50 NUMBERS UNDER AN n300 HEADING — 043's defect one layer up

Job `10317061` finished in **3:34** — far too fast for 123 ANM eigensolves. It had:

    loading tied arm trained at n_train=300: tied_dm256_lr3e-05_s0_n300.pt   <- checkpoint CORRECT
    [stamp] all 123 stored systems match the current stamp                   <- computed NOTHING

and reprinted the n50 quartile table (Q1 +0.2956 vs ANM +0.6912, 0%) as though it were the re-run.

**Cause.** The peer stamp keyed on `dm/lr/seed/cutoff/arm` and **not on `n_train`**, so an n300 run is
indistinguishable from an n50 one and every stored system "matches". 043 fixed the **checkpoint path**
so an n300 run could not load n50 *weights*; it did not fix the **results key**, so the same run
happily reused n50 *results*. Moving a guard one level down is not the same as installing it.

Second contributing error, mine: the sbatch set `LADDER_RES`, which is the **ladder's** variable and
was never read by the peer script, so the intended separate output file was never in effect.

**Fixed:** `n_train` is now in the peer stamp **and** in the results filename, so two rungs can share
neither. Verified: with the key added, **0 of 123** stored n50 systems match an n300 stamp — it must
recompute. Cost, stated: the completed n50 results are now stamp-orphaned for *resume* purposes. They
are intact as data and already reported, and correctness beats resume convenience — the same trade
taken for `modal_seeds` under 42a.

**The n50 peer result is undamaged** — `tied_peer.json`'s mtime never changed. Relaunched as
`10317064`, which will compute all 123.

### ⬛ 41c RESULT — the peer loss SURVIVES the best rung. And the ladder slope, pinned at 8 arms.

**41c (`10317064`, 123 systems, stamp carries `n_train: 300` so it recomputed rather than reusing
n50).** Pre-registered under 41c as *"expected to gain ≈ +0.018–0.036 and to still lose on ~100% of
systems"*:

| quartile | tied @ n50 | **tied @ n300** | ANM | gap | tied > ANM |
|---|---|---|---|---|---|
| Q1 | +0.2956 | **+0.2538** | +0.6912 | −0.4484 | **0%** |
| Q2 | +0.2622 | **+0.2633** | +0.7148 | −0.4104 | **0%** |
| Q3 | +0.2030 | **+0.1698** | +0.6042 | −0.3795 | **0%** |
| Q4 | −0.2791 | **+0.0292** | +0.6790 | −0.6248 | **0%** |

**The prediction's direction was right and its shape was wrong.** Still 0% at every quartile — the
floor is a floor, and the n50 peer loss is not an artefact of under-training. But the expected uniform
+0.018–0.036 did not happen: **Q1 and Q3 got worse** (−0.042, −0.033) while **Q4 improved by +0.308**.
Worst-system FVE went −4.040 → **−1.673** and the fraction below −0.5 went 9% → **2%**.

So more data did not lift the arm; it **traded quartiles**, repairing the catastrophic large-system
tail at the small systems' expense. Nothing in the pre-registration anticipated that, and it is the
part worth carrying forward.

**This also makes every decades-to-close extrapolation unnecessary**, not merely risky: 41c is the
same-harness measurement those were approximating.

### ⬛ 10318733 RESULT — the cross-producer defect was real and immaterial. Headline STANDS.

Read under the rule committed at `07b4682d` **before this number existed**. `c50` recomputed from
`ladder_ckpt/`, so both sides of the paired analysis now share a producer.

| quartile | old c50 (`modal_arm`) | new c50 (ladder) | shift |
|---|---|---|---|
| Q1 | +0.2956 | +0.2926 | **−0.0022** |
| Q2 | +0.2622 | +0.2585 | −0.0014 |
| Q3 | +0.2030 | +0.2002 | −0.0039 |
| Q4 | −0.2791 | −0.2531 | +0.0046 |

**The producer difference is 0.001–0.005 FVE** — an order of magnitude below every effect it was
feared to contaminate. So the join was a genuine methodological defect and its *magnitude* was
negligible, which is only knowable because it was measured rather than argued.

**Pre-registered verdict: STANDS.** 83/123 = **67.5% worse**, sign test **p = 0.000132** — clears
both bars (≥60%, p<0.01). **Q4 = +0.4496** against the provisional +0.4474, so the entire positive
result on the main line survives, and 61a's leverage finding is unaffected at that magnitude.

The §5b `more data` row is **restored** on this basis.

### ⏳ PRE-REGISTERED: how `10318733`'s same-producer c50 gets read, written before it exists

The recompute revises the project's most-quoted negative and **both directions are narratively
attractive**, so the rule goes in first — the way 58b's did at `65cf5b32` before the 9th arm was read.
`c300` is unchanged, so every delta moves by exactly **−(new_c50 − old_c50)** per system, and the
producer difference **is** the measurement of how much the defect mattered.

> **Report old vs new c50 per quartile, and the per-system distribution of the difference** — not only
> the new deltas. A silent replacement destroys the only measurement of the defect's size.
>
> - **HEADLINE STANDS** if the sign test keeps direction *and* significance: **≥60% of systems worse,
>   p < 0.01**.
> - **HEADLINE RETRACTED** if the fraction worse falls **below 50%**, or the sign test loses
>   significance at **p > 0.05**.
> - **BETWEEN**: **NOT RESOLVABLE** — the §5b row stays out and "more data" returns to an **open
>   question**, not an excluded route.
> - **Q4 SEPARATELY**, since it is the entire positive result on the main line: does **+0.4474**
>   survive, and does 61a's leverage finding (Q2 one system, Q4 robust) still hold on the new deltas?

**What is provisional, in full (63b).** Everything differencing n50 against n300 on the peer harness:
the 67%/sign test, every per-quartile delta, 59a's decomposition, 60b's regressions, 61a's leverage
analysis, and 41c's n50 column. **What stands, each checked rather than assumed:** 41c's n300
absolute, the oracle, the ladder slope (`ladder_ckpt` throughout), 48b, and the 3,000-atom cap.

### ⬛ 63d: which n50 is canonical, and for what

Two legitimate n50 peer readings are on disk. **Neither is wrong and they are not interchangeable:**

- **`ladder_ckpt/` n50 → canonical for the PAIRED ANALYSIS.** It shares a producer and a procedure
  with n130 and n300, which is what makes a difference between rungs interpretable at all.
- **`tied_peer_n50_MODALARM_PRODUCER.json` → canonical for the HISTORICAL RECORD.** 28b, 29b/30 and
  26a were computed against that model, and `7a9cf5be` established those stored results stand even
  though the checkpoint behind them was overwritten.

**The switch is not a correction to the old numbers.** They were right about the old model. Anyone
treating the new c50 as superseding 28b/29b/26a would be making the cross-producer error in reverse.

### ⛔ 062: the sparse branch is REFUSED, and the paired result rests on a cross-producer join

**62a — REFUSED, and this honours the pre-registration rather than overriding it.** 53b's own text
says *"closing 25% or 50% still loses to ANM, so neither is a peer win"* and then implements a
different rule as a threshold. Prose and implementation disagree, **the prose predates the data**, so
following it repairs a mis-implementation instead of choosing a statistic post hoc. This is 49a one
level up — a sentence contradicting its own table, where the table is a falsifier.

| | |
|---|---|
| codec alone, win rate | 0/123, 95% upper bound **2.4%** |
| **+ a PERFECT oracle channel at K=74** | **5.7%** |
| K₁₀₀ to stop losing | 256 on Q1 = **27.2% of the structure**; 1024 on Q3; **>1024 REFUSED on Q4** |

The oracle is a strict upper bound, so **5.7% bounds the win rate of any learned sparse channel at the
matched budget** — no trainable channel does better. And the budget at which it stops losing is a
fifth to a quarter of every structure every frame, at which point it is not an event channel and the
constant-per-step cost argument is gone.

**What survives, stated narrowly:** at *non-sparse* budgets the representation is sufficient. That
says the decoder is not the obstacle. It is not a route to a peer win under the cost model.

**62b — K as a fraction of N, and the top row is inside its own ceiling.**

| K | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| 74 | 5.2–12.4% | 2.4–5.1% | 1.0–2.3% | 0.2–1.0% |
| 256 | 17.9–42.8% | 8.2–17.8% | 3.5–7.9% | 0.8–3.4% |
| **1024** | **71.7–171.2%** | 32.8–71.1% | 14.0–31.5% | 3.1–13.6% |

**Q1 spans 598–1429 atoms, so K=1024 is 72%–171% of the structure** — for the smaller half of Q1 it is
*every atom*. Its 100% win rate at K=1024 partly means "the oracle was handed the whole molecule
exactly", which is a **Family B** ceiling, not a result. Only the **K=74 row is genuinely sparse
everywhere** (0.2%–12.4%) and only it carries the sparse claim.

**62c — the paired result is a cross-producer join, and it was invisible.** Checked rather than
assumed:

| | git | written | cfg_raw | checkpoint producer |
|---|---|---|---|---|
| `tied_peer.json` (c50) | 6ba90650 | **Aug 7 13:05** | no `n_train` | **`modal_arm`** (pre-043 n-less path) |
| `tied_peer_n300.json` (c300) | 00496b2d | Aug 8 17:10 | `n_train: 300` | **ladder** |

So 67%-worse, Q1 −0.0755 and Q4 +0.4474 all join a **`modal_arm`-trained n50** model to a
**ladder-trained n300** model. Both sat at paths that resolved, so nothing failed. Option (1) taken:
**c50 is being recomputed from `ladder_ckpt/`** (`10318733`) so both sides share a producer, with the
Aug-7 file preserved as `tied_peer_n50_MODALARM_PRODUCER.json`. **Until it lands, the paired numbers
are provisional.**

**62d — sha256 at write time.** A path check cannot detect present-and-wrong, now the third instance
(`complex_d8`, `ladder_direct_n2272`, the overwrite). The peer stamp now carries the checkpoint's
**sha256**, recorded when the numbers are produced, so a reader can verify the weights still are what
made them.

### ⚠ DATA LOSS I CAUSED: the ladder overwrote `modal_arm`'s n50 checkpoint

The 42b change that made the ladder save checkpoints wrote them into **`modal_arm_ckpt/` under
`modal_arm`'s own naming scheme** — and 043 had just renamed `modal_arm`'s file to exactly
`tied_dm256_lr3e-05_s0_n50.pt`. So the ladder's n50 seed-0 arm **overwrote the checkpoint 28b, 29b/30
and 26a were computed from.**

The evidence is in the directory: every other tied n50 file is **Aug 7 at 1,862,320/401 bytes**
(`modal_arm`'s), while the three `lr3e-05` files are **Aug 8 at 1,862,452** — a different size,
written by the ladder run.

**043's guard could not catch this.** It raises when a checkpoint is *absent*; here a real file sat at
the expected path holding a **different model**, which is 050's failure exactly. A guard against
missing files does nothing about wrong ones.

**What is lost and what is not.** The stored *results* are unaffected — `tied_peer.json` and the rest
were computed on Aug 7 from the original. What is gone is the ability to *reproduce* them from that
path: anything re-run against it today silently gets the ladder's model. The file is not recoverable.

**Fixed:** the ladder now writes `ladder_ckpt/`, its nine checkpoints have been moved there, and
`ckpt_path()` searches both namespaces **and prints which one resolved** — a checkpoint's producer is
part of its identity. 043's raise-on-absent is unchanged (verified: `n999` still raises).

*Two producers must not share a namespace, however well each is named inside it.* That is the
generalisation; the specific fix is worth less than the rule.

### ⬛ ORACLE RESULT — the falsifier does not fire, and the channel still is not a path to a peer win

`10318365`, 123/123 systems, tied arm at n300. **Family B ceiling passed: K=all median FVE
+1.000000**, and K=0 reproduces `fve_model` to 0.000e+00 — so every intermediate K is readable.

| quartile | n | K=0 | **K=74** | ANM | K=74 closes | K₁₀₀ | K₁₀₀ as %N |
|---|---|---|---|---|---|---|---|
| Q1 (598–1429) | 31 | +0.2538 | +0.6038 | +0.6912 | **80.0%** | 256 | **27.2%** |
| Q2 (1440–3126) | 30 | +0.2633 | +0.5364 | +0.7148 | **60.5%** | 256 | 11.8% |
| Q3 (3249–7297) | 31 | +0.1698 | +0.4060 | +0.6042 | **54.4%** | 1024 | 21.3% |
| Q4 (7516–33377) | 31 | +0.0292 | +0.1437 | +0.6790 | **17.6% → REFUSED** | >1024 | >9% |

**Pooled, the form the falsifier was pre-registered in: 60.5% closure at K=74 — NOT REFUSED.** By
53b's asymmetry that means *the branch stays alive and nothing more.*

**But the falsifier measured the wrong sufficient statistic, and the run shows it.** Gap closure in
median FVE is not "would it win":

| | Q1 | Q2 | Q3 | Q4 | ALL |
|---|---|---|---|---|---|
| beats ANM at **K=74** | 19.4% | 3.3% | 0.0% | 0.0% | **5.7%** |
| beats ANM at K=256 | 80.6% | 53.3% | 35.5% | 0.0% | 42.3% |
| beats ANM at K=1024 | 100% | 100% | 90.3% | 25.8% | 78.9% |

**Closing 60.5% of the gap still loses to ANM on 94.3% of systems.** 54e half-anticipated this by
adding K₁₀₀; the lesson is sharper than that — I should have pre-registered the **win rate**, not the
gap fraction, and 53b's own text says as much ("closing 25% or 50% still loses to ANM, so neither is
a peer win") while the threshold it then fixed was a gap fraction. Family D **about the falsifier**.

**And K₁₀₀ says it is not sparse.** To stop losing, the channel needs **256 atoms on Q1 = 27.2% of the
structure**, 11.8% on Q2, 21.3% on Q3, and **more than 1024 on Q4**. A channel transmitting a fifth to
a quarter of all atoms every frame is not a sparse event channel, and the constant-per-step cost
argument dies with it — which is exactly the outcome 54e said would be more decisive than the 25%
rule, arriving through the number it added.

**Verdict.** The pre-registered falsifier does not fire, so the item does not close by its own rule.
But the honest summary is narrower than "alive": at a budget-matched 74 atoms the *oracle* — a strict
upper bound, told exactly which atoms matter and how they move — wins on **5.7%** of systems, and Q4
refuses outright. What survives is a weaker claim: **a sparse channel could close most of the gap on
small systems, at a budget where it still loses to ANM on 4 of 5 of them.**

Scope, unchanged from 53b: this is representational sufficiency. It says nothing about whether such
events could be **generated** at sampling time, and FVE cannot express that.

### ⬛ 061: Q2's slope was one system; Q4's survives. Plus a submission guard and two pre-registrations.

**61a — leverage settles it, and the suspicion was right.** Per-point leverage on the delta-on-mean
regression, refit with the top-3 leverage points dropped:

| band | max leverage | full slope | p | drop top-3 | p | |
|---|---|---|---|---|---|---|
| **Q2** | **0.705 (10.6× mean)** | −1.5594 | 7×10⁻⁹ | **−0.6264** | **0.056** | **COLLAPSES** |
| **Q4** | 0.485 (7.5×) | −0.7751 | 1×10⁻¹⁰ | **−1.2391** | 2×10⁻⁶ | **SURVIVES** |
| Q1 | 0.156 (2.4×) | −0.1359 | 0.048 | −0.1285 | 0.135 | collapses |
| Q3 | 0.285 (4.4×) | +0.0544 | 0.64 | −0.2480 | 0.015 | **sign flips** |

**Q2's slope was a single system** — `4tsh_A`, delta **+2.935**, leverage 0.705. So Q2 gets **no
slope**, matching the NOT RESOLVABLE its deltas already carry. Q1's was marginal and does not survive;
Q3's *flips sign* and becomes significant, so it is not determined either. **Only Q4's is robust**, and
it strengthens when the high-leverage points are removed — so Q4's baseline-dependence is a property
of the band, not of three systems.

**61d — submission guard, structural.** `scripts/armf_submit.sh` refuses to `sbatch` a job whose
`--job-name` is already in the queue, printing the existing ids, with `ARMF_FORCE=1` for a deliberate
second chain. Driven both ways: it refuses `oracle_ch` (correctly listing 10318365/66) and passes a
name that is absent.

*Correction to the record:* 061 credits me with catching and cancelling the duplicate. I did not —
I submitted **10318365/66**; **10318412/13** appeared 13 minutes later and were cancelled by another
actor at 0:00 elapsed. I noticed them only afterwards, in a monitor event. That strengthens 61d's
point rather than weakening it: the race was resolved by luck, not by anyone's alertness.

### ⏳ PRE-REGISTERED (61b): the n50 seed replicates, and what they can and cannot settle

RTM-from-noise and genuine repair-of-the-worst predict the **same** negative slope, so no regression
separates them. An independent per-system read at one rung can, and
`tied_dm256_lr3e-05_s{0,1,2}_n50.pt` are on disk.

> **Logic, stated before the run:** the across-seed spread at n50 estimates the measurement-noise
> variance; that variance implies **how much** regression to the mean to expect; Q4's observed
> +0.4474 is then inside or outside that expectation.
>
> **Three seeds gives a noisy variance estimate, so this produces a BOUND, not a point.** Said now
> rather than after, per 061's own condition.

### ⏳ PRE-REGISTERED (61c): capacity competition costs one arm, and it is not free

`atlas_dm` is ruled out as the test (wrong arm, wrong rung pair), and the tied ladder ran at DM=256
only — so this is **new compute, not a groupby**. Cheapest sharp version: **one arm — tied, n300,
DM=512**, everything else matched.

> **Reading, fixed now.** Q1's degradation at DM=256 is **−0.0755** [−0.0950, −0.0559].
> **SHRINKS** = Q1 delta at DM=512 above **−0.0378** (half the magnitude) *and* its CI excluding
> −0.0755. **DIES** = Q1 delta within the DM=256 CI. Between = NOT RESOLVABLE by one arm.
>
> **Carries the 002 guard:** report the learned latent's **participation ratio** alongside. The prior
> DM sweep saw DM=256/512 collapse to constant output (G1 cos = 1.0000) on 21 domains and be declared
> VOID — without PR a null reads as "capacity does not help" when it is "the wide arm failed to train".

Queued behind the oracle and the seed replicates.

### ⬛ 060: the two statistics both hold, the RTM control survives its own artefact, and the question had a missing branch

**60a — both tests, each on its own statistic.** They are not in conflict:

| statistic | value | its own test |
|---|---|---|
| mean paired Δ | **+0.1141** [+0.0315, +0.1967] | paired t, **p = 0.0072** → more data helps |
| systems worse | **83/123 = 67%** | sign test z = **+3.88**, exact binomial **p = 0.000132** → more data hurts |

> **PROVISIONAL (62c/63a):** cross-producer join; awaiting `10318733`.

The sign test is **54× stronger in p**, and it is what the headline rests on. The mean is positive
because a minority moves far; the sign test is negative because the majority moves the other way.
Stated twice it is more convincing than either alone. Sign test rather than Wilcoxon, as 060 asked —
it assumes nothing about the shape of the Δ distribution, which matters given Q2.

**60b — the control was run, and it needed a correction of its own.** Regressing Δ = c300 − c50 on
**c50** puts the same noisy quantity on both sides, forcing a negative slope even with no real effect
(Oldham's fallacy). The tell is Q2 at **−0.9605**, near the −1.0 that pure coupling predicts. Regressing
on the **mean** of the two removes the coupling:

| band | slope on baseline | slope on mean | p (on mean) |
|---|---|---|---|
| **Q4** | −0.5924 | **−0.7751** | <10⁻⁵ |
| Q1 | −0.1775 | −0.1359 | 0.048 |
| Q2 | −0.9605 | −1.5594 | <10⁻⁵ |
| Q3 | −0.1375 | +0.0544 | 0.64 |

**Q4's baseline-dependence survives the artefact correction.** What it cannot separate is
*measurement-noise* regression to the mean from *real headroom* — the systems that were worst had the
most room. That needs replicate measurements at one rung, and the peer runs are single-seed, so it
stays bounded rather than settled. Q4's repair is concentrated in the systems that were worst, which
is what "repairing the tail" means; whether its magnitude is inflated is open.

**60c — the third branch, added to the question.** The two-branch design cannot express what was
measured: *data-limited* predicts systems improve (67% got worse); *fundamental* predicts nothing
moves (Q4 moved +0.4474, p=0.0003). **Capacity competition** — a fixed DM=256 latent asked to cover
6× more systems, reallocating budget rather than learning more — is a third hypothesis, fits both
observations, and is actionable where neither other is. Recorded as a branch of the question.

**It is NOT tested by `atlas_dm`, and the test I ran says so.** The cross exists (DM 16/64/256 have
both rungs, `per` carries all 123 per-system FVEs) but:

- it is the **attention/control arm over 50→130**, while the paired result is the **tied arm over
  50→300** — different arm *and* different rung pair;
- the Q1 deltas are **non-monotonic** (+0.0136, +0.0650, +0.0171) and the slope on log₂(DM) is
  **+0.0009, p = 0.961** on three points. My own script printed "CONSISTENT with capacity
  competition" from `slope > 0` without testing resolution — a Family C read, and it is wrong. The
  honest verdict is **NOT RESOLVABLE**.

**60d — Q2 gets no direction.** CI [−0.1070, +0.3287], width 0.436, **eleven times Q1's 0.039**.
Through `null_verdict` against a relevance bound of 0.0755 it returns **NOT RESOLVABLE** — the CI
contains both zero and an effect that would matter. It is rendered as such, not as a positive.

**60e — oracle launched:** `10318365 → 10318366` (afterany chain). Harness validated first: K=0
reproduces `fve_model` to **0.000e+00**, K=all reaches exactly 1.0 with residual SSE 0 (Family B
ceiling), monotone in K.

### ⬛ 060: both ALL-row tests are significant and point opposite ways — and Q4's gain has a live confound

**60a — the sign test is the stronger statistic, and the headline rests on it.** Reported with each
number carrying its own test, since the previous row paired a descriptive fraction with a p-value
computed on the other statistic:

| statistic | value | test | p |
|---|---|---|---|
| mean paired Δ | **+0.1141** [+0.0315, +0.1967] | paired t | 0.00716 |
| **systems worse** | **83 / 123 = 67%** | **exact binomial (sign), z=3.88** | **0.000132** |

The sign test is **54× stronger in p**. These are **not in conflict**: the mean is positive because a
minority moves far, the sign test is negative because the majority moves the other way. That *is* the
finding, stated twice, and it is more convincing with both than with either. The sign test carries the
headline because it assumes nothing about the shape of the delta distribution — which matters given
Q2's spread.

**60b — the regression-to-the-mean control fires, and Q4 is the entire positive result.** Regressing
each system's paired Δ on its n50 value:

| | slope | R² | p |
|---|---|---|---|
| **Q4** | **−0.5924 ± 0.0829** | **0.881** | <0.0001 |
| Q1 | −0.1775 ± 0.1149 | 0.256 | 0.0037 |

Not flat — steeply negative, explaining **88% of the variance in Q4's gains**. Systems that were worst
at n50 gained most.

**But the slope alone does not settle it, and saying it debunks Q4 would overclaim.** Regression to the
mean and a genuine repair-of-the-worst-cases predict *the same* negative slope. Separating them needs an
independent n50 read per system, which one arm cannot provide. **That read is cheap and available:**
`tied_dm256_lr3e-05_s{0,1,2}_n50.pt` all exist, so running the peer harness on a second seed gives an
independent per-system n50 and splits the two explanations. Queued behind the oracle.

**60d — Q2 is NOT RESOLVABLE and is recorded as such.** CI [−0.1070, +0.3287], width 0.436 — **11×
Q1's 0.039**. Its +0.1108 must not be rendered as a positive in any table scanned by sign; per
`null_verdict`'s own four states the honest cell is *not resolvable*.

### ⚠ 60c: the live question has no branch for what was measured — add the third

31d asks **data-limited** vs **fundamental**. The result fits neither:

- **data-limited** predicts systems improve with data. **67% got worse.**
- **fundamental** predicts nothing moves. **Q4 moved +0.4474, p=0.0003.**

Both rungs draw from the same 697-system pool, so the training distribution's *shape* is unchanged
between them — there is simply more of it. A **fixed-width latent (DM=256)** asked to cover six times
as many systems, improving the tail while degrading the majority, is the signature of **capacity
competition**: the model reallocating a fixed budget rather than learning more. That is a **third
hypothesis**, neither of the two on offer, and unlike both it is *actionable* — it points at
conditioning or per-size capacity, not at more data and not at giving up.

This is **Family D at the level of the research question**: a two-branch design cannot express
capacity-limited, so whichever branch returned, the answer would have been mis-assigned.

**Testable on data already on disk.** `atlas_dm` sweeps DM at `NTRAIN=[50,130]` — exactly the cross
needed. Per-quartile paired Δ at each DM: if Q1's degradation **shrinks as DM grows**, it is capacity
competition and the fixed-width assumption is the binding constraint; if **flat in DM**, the hypothesis
dies cheaply. A groupby over completed arms, no new compute.

### ⬛ 059: six times the data made **67% of systems worse**. The slope is a tail repair.

The ladder's rise and 41c's quartiles are one result, and the paired per-system test settles it.
Both rungs evaluated **the same 123 systems on the same frames** (N matched exactly), so this is a
paired comparison, not a difference of summaries:

| quartile | n | mean paired Δ | 95% CI | **fraction worse** | p (paired t) |
|---|---|---|---|---|---|
| Q1 | 31 | **−0.0755** | [−0.0950, −0.0559] | **97%** | <0.0001 |
| Q2 | 30 | +0.1108 | [−0.1070, +0.3287] | **80%** | 0.31 |
| Q3 | 31 | −0.0264 | [−0.0566, +0.0038] | 77% | 0.085 |
| Q4 | 31 | **+0.4474** | [+0.2274, +0.6675] | 16% | 0.0003 |
| **ALL** | **123** | **+0.1141** | [+0.0315, +0.1967] | **67%** | 0.0072 |

**Q1 did not merely fail to improve — it degraded, and the effect is unambiguous:** 97% of its systems
got worse and the CI is nowhere near zero. 059's cautious "no improvement detected" is too weak for
Q1; it is right for Q3 (CI still touches zero at −0.0264, p=0.085).

**Q2 is the mean/median trap one level down.** Its mean is *positive* (+0.1108) while **80% of its
systems got worse** — a few large gains outvoting a declining majority. A quartile mean hides this the
same way the overall mean hides Q4.

**The headline, restated.** Six times the training data made **67% of held-out systems worse**. The
aggregate mean rose only because Q4's catastrophic failures were repaired. So the ladder's
+0.1227 ± 0.0871 and JT p=0.00179 are real but they order **arm means**, and the means rise because
the tail rises. The supported statement is **"more data monotonically repairs the large-system tail"**,
not "the codec improves with data". That is Family D about the ladder's own statistic — mean FVE
cannot separate "everything improved a little" from "the tail improved a lot while the rest declined".

**And Q4 was repaired from harmful to useless, not to good:**

| | Q4 |
|---|---|
| n50 | **−0.2791** — worse than predicting nothing |
| n300 | **+0.0292** — approximately no signal |
| ANM | **+0.6790** — still +0.6498 ahead |

Worst-system FVE −4.040 → −1.673 and sub-−0.5 9% → 2% say the same thing: the model **stopped being
actively wrong** on large systems. It did not start being right about them.

**The negative, stated with its bound (59c).** Not "we have not seen a win": by the rule of three,
**0 wins in 123 puts the 95% upper bound on the win rate at 3/123 = 2.4%** — at every cutoff, at the
best rung the ladder can reach.

### ⬛ 31d/34c LADDER COMPLETE — 9/9 arms on the GENUINE TIED arm. The ceiling moves with data.

Read under the rule committed at `65cf5b32` **before the slope was looked at**. It did not return
across zero; it tightened.

| rung | k | mean FVE | SD | at step ceiling | VOID |
|---|---|---|---|---|---|
| n50 | 3 | +0.1041 | 0.0372 | 0/3 | 0 |
| n130 | 3 | +0.1616 | 0.0148 | 0/3 | 0 |
| n300 | 3 | **+0.1993** | 0.0218 | 0/3 | 0 |

| arms | slope (HC3) | zero |
|---|---|---|
| 7 | +0.1048 ± 0.1375 | includes |
| 8 | +0.1142 ± 0.1006 | excludes |
| **9 — authoritative** | **+0.1227 ± 0.0871** (df=7, R² 0.763) | **EXCLUDES** |

**Assumption-free: Jonckheere–Terpstra 26 of 27, exact one-sided p = 3/1680 = 0.00179.**

**This is the cleanest run in the project.** All three rungs at k=3, **zero VOID arms**, **nothing at
the step ceiling** (0/3, 0/3, 0/3) — so none of the Family D truncation, Family A exclusion or 39a
SD-deflation effects that qualified the untied ladder apply here. No `PARTIAL` and no `THIN RUNG` tag
fires: the ladder is the one that was posed.

**Per 42c, tied did not reproduce untied and was not expected to.** Tied sits higher at every rung
(+0.1041/+0.1616/+0.1993 against +0.1045/+0.1275/+0.1403) and converges without ever hitting the cap.

**What it establishes, and what it does not.** The tied arm is **not saturated in `n_train` over
50–300** — direction at p=0.0018, magnitude resolved at +0.1227 ± 0.0871. It does **not** establish
that data is a path to the peer: **41c measured that directly at n300 and the arm still loses on 0%
of systems at every quartile.** Those two facts are now both same-harness measurements, so the
conclusion needs no extrapolation and none should be quoted.

### ⏳ PRE-REGISTERED: how the 9-arm ladder slope is read, written before it was looked at

The learning curve has resolved in magnitude for the first time:

| arms | slope (HC3) | zero |
|---|---|---|
| 7 | +0.1048 ± 0.1375 | includes |
| **8** | **+0.1142 ± 0.1006** | **EXCLUDES** (df=6, n300 k=2 at +0.1907) |

The 9th arm is on disk and **has not been read**. Recorded first, so the question cannot be reopened:

> **The 9-arm HC3 fit is authoritative by 37a and supersedes the 8-arm one whichever way it falls.**
> If the CI returns across zero, the reading is **"the 8-arm exclusion was not resolved"** — *not*
> "the 9th arm is an outlier". An outlier claim would require an independent reason to discount that
> arm, and "it moved the answer back" is not one. The 8-arm value stays visible as the retract trail
> rather than being overwritten.

### ⬛ 057: the 48b sample is size-representative, and 41a's extrapolation is RETIRED not recomputed

**57a — the verdict is quotable.** `--limit 1400` capped the run; the rule is `linspace` over the
**size-sorted index**, a systematic sample of order statistics, so it spans the range rather than
truncating it. Against the full 4,715 clean pool:

| band | full % | eval % | full n | eval n |
|---|---|---|---|---|
| 0–110 | 4.0% | 3.9% | 188 | 55 |
| 110–150 | 22.8% | 23.1% | 1,076 | 324 |
| 150–200 | 24.4% | 24.0% | 1,152 | 336 |
| 200–250 | 13.0% | 12.9% | 611 | 181 |
| 250–300 | 16.6% | 16.9% | 783 | 237 |
| 300–395 | 19.2% | 19.1% | 905 | 267 |

**KS D = 0.0050, p = 1.000** — but this is a **construction check, not an independent test
(58c)**. A `linspace` over a sorted index reproduces the sort key's CDF *by construction*, so D≈0 is
guaranteed by the sampler rather than discovered about the sample. It certifies **size and nothing
else**, and must not later be cited as evidence the sample is unbiased in general.
The 0–110 band is 29.3% covered against 29.7% overall, so its n=55 is
proportional and not the residue of a size-ordered truncation. Cost is precision only.

**57b — STRUCK ENTIRELY (INBOX 58a), not superseded.** Every decades-to-close figure in this
project — 41a's ~10¹⁰, and the 4.9M / 52,217 / 74.9× / 92× numbers computed under 057 — joined
**ladder** mean FVE to a **peer**-derived gap. The same arm reads **+0.1041 on the ladder and +0.0607
on the peer at n50, a 1.71× ratio**. None was measured on one harness, so none is a result. Struck
rather than recomputed, because recomputing a cross-harness join more carefully is still one. 41c
now supplies the same-harness number and no extrapolation is needed.

*(historical, retained as the retract trail)* **57b — the extrapolation is retired, not recomputed.** The ladder has since reached **8 arms**
(n300 now k=2, +0.1907; slope **+0.1142 ± 0.1006**, HC3 df=6, which now **excludes zero**). Rerunning
41a's arithmetic on it gives mean-gap closure at 4.2 decades (~7,000× the pool) and, at the optimistic
CI end, **74.9×** the pool — so 057's surviving Q1 branch (3.5× the pool) **closes**: on the tighter
8-arm CI the same branch needs **92×**.

But every one of those numbers joins the **ladder's mean FVE** to a **peer quartile median from a
different harness**, which is precisely what made 41a wrong and what 42b refused. The ladder's n50 is
+0.1041 while the peer's n50 codec mean is +0.0607 — the same arm, two harnesses. So no version of
this extrapolation is quotable, including the ones above, and they are recorded only to show the
direction is unchanged. **41c (`10317064`, running) computes tied-at-n300 against ANM on the same
frames; that is the number, and it needs no extrapolation.**

**57c — PARTIAL correctly did not fire, and that exposed a gap.** All three rungs are present, so
`missing` is empty and the tag keys on missing *rungs*. But 036's principle is that the flag **rides on
the verdict line**, and a rung that is present-but-thin at the end setting the lever arm was warned
about only *above* the line. Added as its own tag rather than folded into PARTIAL, which would conflate
two different defects:

    => THE CEILING MOVES WITH DATA. [THIN RUNG: n300=2/3 seeds] Slope +0.1142 +/- 0.1006 excludes

**57d — contact-F1 null implemented; the clash null is degenerate and is stated as such.** A collapsed
prediction puts every atom on every other, so its clash rate is maximal by construction and carries no
information — that is said rather than reported as a number. For contact F1 the same prediction *is*
readable: recall = 1, precision = true contact density, F1 = 2d/(1+d), which **falls with N** (0.113 at
N=200 → 0.012 at N=2,900) and supplies exactly the calibration the absolute 0.90/0.75 thresholds lack.
**It is a FLOOR, never a normaliser (58d).** Contact density falls roughly as N^−0.86, so the
null falls *faster* than the model and the ratio to null **rises** with size — 26.8× at ≤110 residues
against 46.1× at 300–395. Normalising the F1 column by it would show the degradation flattening or
inverting and conclude the fold survives at 2,600 atoms, against 3.59 Å RMSD and 180 clashes/1k saying
it does not. Report only that the model clears the trivial baseline at every band, and say on the line
why it is not divided by. The current 48b run predates it; per 057 this is precision, not correctness, and the SMALL-PROTEIN
verdict rests on RMSD, which is controlled and crosses on its own.

### ⬛ SEQUENCE-LEAKAGE GATE built and self-tested, before the 1M draw exists

AlphaFold DB covers essentially all of UniProt, so a 1M draw **will** contain predicted structures of
the held-out ATLAS proteins unless they are removed — and that would invalidate every zero-shot claim
on the record: 0/123 at every cutoff, the 2.4% win-rate bound, the whole peer comparison.

**MMseqs2 15-6f452** installed as a static binary (`pip` is dead in this venv). Query set is **all
chains of all 125 held-out entries = 156 sequences**, deliberately over-inclusive: over-inclusion can
only remove more candidates, while under-inclusion leaves a held-out protein unscreened, which is the
failure the gate exists to prevent.

**Two defects caught while building the query set, both silent:**

- fragile chain-matching against RCSB's header variants lost **24 of 125 entries** (101 retrieved);
- the id file had **no trailing newline**, so `while read` dropped the last line — **`6sup_A` had zero
  sequences in the query set** until the 124-vs-125 discrepancy was chased rather than rounded off.

Either would have left held-out proteins unscreened while the gate reported success.

**Self-tested against a known positive:** feeding held-out sequences back in as candidates removes
**2 of 2 at 100% identity**. A gate that has never rejected anything is not known to work.

**Stated in the output, not just here:** the **training** systems are **not** excluded — pretraining
on the training distribution is the point — and that is a decision rather than an oversight, printed
every run because a reader seeing a leakage gate will reasonably assume it removed both.

### ⬛ 064 RECONCILED, 065 ACCOUNTED, and 70b/70e PRE-REGISTERED before the corpus exists

**64a/64b — the table was three different statistics under one heading.** My script printed
`median(old)`, `median(new)`, and `median(new−old)`; a reader subtracting the columns gets
`median(new)−median(old)`, and **median(Δ) ≠ Δ(median)**. Every column now carries its statistic:

| Q | med(old) | med(new) | med(new)−med(old) | med(new−old) | mean(new−old) |
|---|---|---|---|---|---|
| Q1 | +0.2956 | +0.2926 | −0.0030 | −0.0022 | −0.0008 |
| Q2 | +0.2622 | +0.2585 | −0.0038 | −0.0014 | +0.0096 |
| Q3 | +0.2030 | +0.2002 | −0.0028 | −0.0039 | −0.0042 |
| Q4 | −0.2791 | −0.2531 | **+0.0260** | +0.0046 | −0.0021 |

**64b closes exactly:** Q4's paired delta went **+0.4474 → +0.4496**, a change of **+0.0021**, which
equals **−mean(new−old) = +0.0021** to the digit. The delta is a *mean*; the c50 columns are *medians*.
Both were right and neither could be checked against the other — one-name-two-things in **statistics**
rather than filenames.

**64c — my stated range was for one statistic only, and I should have said which:**

| statistic | per-quartile range |
|---|---|
| med(new−old) | −0.0039 … +0.0046 ← what I quoted as "0.001–0.005" |
| mean(new−old) | −0.0042 … +0.0096 |
| **med(new)−med(old)** | −0.0038 … **+0.0260** ← what a reader subtracting columns sees |

The largest producer difference is **+0.0260** at Q4, **5.2×** the top of my quoted range. It remains
**5.8% of Q4's +0.4496**, so the verdict is unaffected — but Q4 carries the entire positive result, so
it is exactly where the range had to be exact.

**065 — compute accounted, with its own limitation stated.** `sacct` records `gres/gpu=N` but **not
the GPU model**, so an A100-equivalent figure cannot come from sacct alone. Joining each job's elapsed
time to the model in its `nvidia-smi` epilogue:

| model | jobs | wall h | factor | A100-eq h |
|---|---|---|---|---|
| Quadro RTX 8000 | 33 | 117.0 | 0.30 | 35.1 |
| NVIDIA RTX A6000 | 1 | 1.1 | 0.42 | 0.5 |
| NVIDIA L40S | 2 | 0.0 | 0.55 | 0.0 |
| **unattributed** | | **368.4** | — | — |

**Only 24% of GPU wall-time is model-attributable from logs.** Attributed: **35.6 A100-eq h** of 118.1
wall. If the remaining 368.4 h ran on the dominant model, the project total is **~146 A100-equivalent
GPU-hours** against **486.4 wall**. Recorded with the 76%-unattributed caveat on its face, because a
single blended factor would be a guess wearing an accounting's clothes.

*Note this revises the "251.5 GPU-h spent" figure used in 68b/69c — that was wall-hours on mixed
hardware, not A100-equivalent. The cap-lift comparison holds either way: ~18 GPU-h against 486 wall.*

### ⏳ PRE-REGISTERED (70b/70e): the 1M re-run's reading, written before the corpus exists

**70b — source is a second axis and must not ride along.** AFDB is **predicted**; every corpus so far
is **experimental**. As framed, the re-run would compare *0.79 Å experimental-small* against
*X Å predicted-size-diverse*, with **size** the axis under test and **source** moving with it — 48a's
warning, which 41a/42b/045/050 have each already paid for.

> **Hold out an AFDB in-range slice as the reference**, so size is the only axis that moves. Report
> **AFDB→AFDB** and **AFDB→experimental** as **two results, never pooled**. The second is the transfer
> question, worth having, and a different question.

**70e — thresholds, fixed now.** 66a's rule was anchored to *median 0.8357 Å over 758 held-out at
20–109 residues*, from a model trained on that range. With size-diverse training that reference no
longer describes the in-distribution case, so the **rule** survives and the **numbers** must move:

> **Reference:** the held-out in-range AFDB slice (≤110 residues, drawn from the 130,000 available).
> **Structure retained from 66a:** all-atom ≤2× reference, contact F1 ≥0.90, clashes ≤2× reference.
> **Floor retained from 55b:** n ≥ 50 in the deciding band, or **UNDERPOWERED**. It will not fire at
> 1M — and a rule that cannot fire is free.
>
> **THE BRANCH THAT HAS NEVER BEEN WRITTEN DOWN — what says the architecture DOES scale:** the ≥300-
> residue band meets **all three** thresholds against the in-range AFDB reference, **and** the
> degradation slope across bands is flat within its CI. Until now no run could produce this branch, so
> it was never specified; writing it while nobody knows the answer is what made 58b and 63c
> trustworthy.

### ⬛ 66c MEASURED (pending since 066): rate in bits, a likelihood surrogate, and the regularisation decision

`scripts/armf_likelihood.py`, 120 held-out structures, CPU, no retraining.

**Rate–distortion, rate in BITS at uniform scalar quantisation** — the distortion axis is the
project's own `all_atom_rmsd` (Kabsch-**aligned**), because raw and aligned differ ~3× (1.589 vs 0.487
on one structure) and reporting raw against a recorded aligned number would be one-name-two-things in
the distortion axis:

| bits/scalar | bits/atom | median aligned RMSD |
|---|---|---|
| 2 | 2.07 | 6.168 |
| 3 | 3.11 | 3.080 |
| 4 | 4.15 | 1.645 |
| **6** | **6.22** | **0.837** |
| **8** | **8.29** | **0.769** |
| 12 | 12.44 | 0.757 |
| 32 (float, no quantisation) | 33.16 | 0.757 |

**The curve saturates at 8–12 bits.** 8 bits/scalar costs **1.6%** distortion (0.769 vs 0.757) for a
**4× rate reduction**, and **6 bits/scalar already reaches the headline** (0.837 vs the recorded 0.8357
median). So **`compression_ratio`'s float count overstates the rate by 4–5×**: the real operating
point is **~6–8 bits/atom, not 33**.

*Harness check first, per Family B:* `decode(z, batch)` reproduces `forward(batch)` to **0.000e+00**,
and 32-bit passthrough gives 0.757 Å against the recorded 0.79/0.8357 — so the curve reaches its own
ceiling before anything below it is read.

**Gaussian-decoder likelihood surrogate.** σ fitted on 106,923 coordinates from a **disjoint half**:

| | nats/dim | **bits/dim** |
|---|---|---|
| fit half | 1.4188 | 2.0469 |
| **held half** | **1.4050** | **2.0269** ← the honest one |

σ = 0.9999 Å. Refitting σ on the held half would give 2.0267, so the optimism from fitting and scoring
on one set is **+0.0003 bits/dim** — negligible, which is itself worth knowing rather than assuming.

**Stated as a surrogate:** the decoder is deterministic, so this prices reconstruction error *as* a
likelihood. It says nothing about whether the latent is distributed in a way a diffusion model could
sample. The NLL is on **raw** residuals deliberately — alignment is a per-structure rigid fit, i.e.
free parameters the model does not have, and a likelihood must not be paid that discount.

**THE REGULARISATION DECISION, RECORDED (66c.3).** The latent is currently regularised by **neither KL
nor VQ**, and until now that was an omission rather than a decision. Recorded as a decision, with its
evidence and its limit:

> **Neither, for now.** The rate–distortion curve shows the latent survives uniform 8-bit scalar
> quantisation at a 1.6% distortion cost, so it is not pathologically scaled or outlier-dominated —
> which is the failure an unregularised latent usually shows first, and it is absent.
>
> **But that is not sufficient for diffusion.** Quantisability constrains the *marginal* per-scalar
> range; diffusion needs the *aggregate posterior* to be close to something samplable, which nothing
> here measures. **Before the diffusion stage this must be answered, not inherited** — and the 8-bit
> result argues VQ is the cheaper candidate of the two, since the latent already tolerates
> discretisation.

### ⬛ 68a MEASURED: the exponent is **0.44**, and 1M costs **~12.7 GPU-h**

`10324137`, fitted on `processed_big` across 4,976 structures:

| band | n | med R | ms/step | ms/structure |
|---|---|---|---|---|
| 20–60 | 102 | 54 | 17.4 | 1.09 |
| 60–100 | 232 | 85 | 18.1 | 1.13 |
| 100–150 | 1,191 | 129 | 18.1 | 1.13 |
| 150–220 | 1,349 | 167 | 22.2 | 1.39 |
| 220–300 | 1,197 | 257 | 30.8 | 1.92 |
| 300–400 | 905 | 330 | 39.0 | 2.43 |

**Fitted exponent a = 0.44.** Not 2 (068's O(R²)) and not 0 (067's implicit flat). **Attention does
not dominate in this range** — cost is sub-linear in residues, so the model is still fixed-cost
dominated at AFDB's bulk sizes.

**The absolute figure needed anchoring, and my first script version got it wrong.** The benchmark's
17–39 ms/step is **kernel-only** (forward+backward+step, direct indexing, MSE loss); `train_log`'s
150.0 ms/step is the **full loop** — dataloader, masking/curriculum, periodic `quick_rmsd` evals,
checkpointing — a **7.4× non-kernel overhead**. The **exponent transfers; the absolute does not.**
Scaling the real 150 ms/step by the measured size factor:

| | ms/step at AFDB sizes | 4 epochs over 1M |
|---|---|---|
| 067 (assumed a=0) | 150 | 10.4 GPU-h |
| 068 (assumed a=2) | ~1,754 | ~122 GPU-h floor |
| ~~measured, factor 1.22×~~ | ~~183~~ | ~~12.7 GPU-h~~ — **wrong anchor pairing, see 69b** |
| **measured, corrected (69b)** | **263** | **18.2 GPU-h** |
| **+ worst-case tail bound (69c)** | **469** | **32.6 GPU-h** |

**69b — the anchor and the multiplier were on different corpora, and that is my error.** The 150.0
ms/step anchor comes from `ladder_direct3m_n2272`, trained on `splits_small_n2272` at **median 81
residues**. I computed the size factor against `processed_big`'s median of **183** — the corpus the
*fit* used. The 81 → 183 step is real cost and was uncounted:

| reference | factor | ms/step | 4 epochs over 1M |
|---|---|---|---|
| 183 res (fit corpus — wrong pairing) | 1.22× | 184 | 12.7 GPU-h |
| **81 res (the anchor's own corpus)** | **1.75×** | **263** | **18.2 GPU-h** |

A **1.43× correction**, and it is the same two-references-one-number shape as `complex_d8`, `rmsd` and
064's shift column — hard to see precisely because both numbers are individually correct.

**69c — the tail is now BOUNDED, not merely unpriced.** Holding the measured a=0.44 below 330 residues
and assuming the *pessimal* a=2.0 above it (attention fully dominant), anchored continuous at 330, and
weighting by the **actual 85,220-length sample** rather than band midpoints:

    below 330 res: 60% of AFDB       above: 40%
    E[cost] worst-case / E[cost] at a=0.44 throughout = 1.79x

069 estimated 1.58× from midpoints; from the full sample it is **1.79×** — its own caveat about
midpoints was warranted, and in the conservative direction.

**So the whole run is bounded under ~33 GPU-h even if the tail behaves as badly as it possibly can**,
against **251.5 GPU-h already spent**. The cap-lift decision does not depend on measuring the tail and
should not wait for it.

So 067 was close and 068's order-of-magnitude correction does not survive measurement — the O(R²)
worry is real in principle and simply is not what this model does at these sizes.

**CAVEAT, and it is 48b's own lesson turned on this fit:** the fit spans **54–330 residues** and
**40% of AFDB lies above it** (q95 795, max 1,843). `a` may rise there if attention begins to
dominate. **This prices the bulk, not the tail** — and extrapolating a fitted law past its measured
range is precisely what 66a just cost.

**Consequence for 68b:** the cap-lifted branch costs **~18.2 GPU-h measured, bounded under ~33 GPU-h** including a worst-case tail — not ~122. The pair now reads:
cap kept → confound survives, cheap; **cap lifted → 66a answerable, ~12.7 GPU-h plus an unpriced
tail**. That is affordable against 251.5 GPU-h spent to date.

### ⏳ 068: ms/step is being MEASURED, not scaled — and the cap/compute pair recorded together

**68a accepted, and neither estimate is adopted.** 067 implicitly assumed cost is flat in R (a=0);
068 assumes O(R²) (a=2) and derives ~122 GPU-h as a floor. Both are extrapolations from an assumed
exponent, so `scripts/armf_stepcost.py` (`10324137`) **fits the exponent** instead: `processed_big`
spans 22–385 residues, median 183, which already brackets AFDB's median of 277, so the scaling law is
measurable on data already on disk with no AFDB preprocessing.

It evaluates the fitted law at **E[Rᵃ] over AFDB's actual 85,220-length sample**, not at its median —
068's own skew point, applied to the fit rather than to a guess.

**68b — the cap and the compute are one decision, recorded as a pair:**

| | corpus | 66a confound | compute |
|---|---|---|---|
| cap kept (≤3,000 atoms ≈ 411 res) | truncated at AFDB's q75 | **survives** | cheap |
| **cap lifted** | 87% out-of-training-range | **answerable** | the O(Rᵃ) cost, being measured |

Recorded together so the cap is not lifted and the compute discovered afterwards.

**68c — storage repriced, and 068's estimate confirmed then sharpened.** `processed_big` is 135 KB at
median 183 residues = **0.738 KB/residue**. At AFDB's median 277 that is **204 KB** (068 said ~206).
But 1M structures cost the **mean**, not the median, and AFDB's mean is **328** residues:

| | |
|---|---|
| 1M processed (at the mean) | **~242 GB** |
| 1M raw mmCIF | ~321 GB |
| raw kept + processed | ~563 GB |
| **raw deleted after conversion** (the `armf_atlas_cache.py` pattern) | **~242 GB** |

**Headroom checked before the run, per 68c:** `$SCRATCH` has **185 TB available** of 804 TB (78%
used), against an ATLAS cache already holding 263 GB. 242 GB is not a constraint here — but the check
is recorded because "a 1M download that dies at 80% on a full filesystem leaves a partial corpus that
looks complete" is the right failure to have excluded in advance rather than discovered.

**68d — two corpus facts, recorded before the split is drawn:**

- **The in-range slice is itself a 57× scale-up.** 13.0% of 1M = **130,000 structures at ≤110
  residues**, against the current 2,272. So this corpus answers the controlled size study **and**
  scales the regime that already works — **two results, to be reported separately, not averaged**.
- **The tail is thin where the test is hardest.** q95 = 795, max = 1,843, so ~5% sits above 795 —
  against ATLAS scale of ~4,200 residues. **Sampling choice is therefore a decision, not a default:**
  natural sampling gives AFDB's distribution and representativeness; size-stratified buys tail
  coverage at the cost of it. **To be recorded explicitly when the split is drawn**, since inheriting
  one silently is what A7 did for three months.

### ⬛ 067 ACQUISITION PILOT — measured, and it corrects 067's own cost model

**AFDB is reachable and the version is v6, not v4.** Every `AF-*-model_v4.cif` and `_v3` request
404s with an S3 `NoSuchKey`; the EBI FTP accession index carries the version in its last column and it
is **6**. A fetcher written against v4 would have failed on every structure.

**Measured throughput** (160 accessions from the live index, `curl` + `xargs -P`):

| workers | structures/s | MB/s | 1M structures |
|---|---|---|---|
| 1 | 1.96 | 0.6 | 142 h |
| 8 | 16.8 | 5.5 | 16.5 h |
| **32** | **109.1** | **35.9** | **~2.5 h** |

**321 KB/structure raw mmCIF** → ~321 GB download for 1M. So **acquisition is ~2.5 CPU/IO hours, not
the dominant line item** 067 expected — it is comparable to the 10.4 GPU-h of training, not larger.

**067's training basis reproduces exactly**: 36,300 s / 241,968 steps = **150.0 ms/step**, 142.0
steps/epoch, batch **16.0**. Storage reproduces too: 53 and 135 KB/structure against 067's 52.1/133.8.

**AFDB's size distribution is the finding that matters for 66a** (n=85,220 sampled from the index):

| | residues |
|---|---|
| min / q25 / **median** / q75 / q95 / max | 16 / 162 / **277** / 430 / 795 / 1843 |

- **Only 13.0% of AFDB is ≤110 residues** — the range `ladder_direct3m_n2272` was trained on. **87% is
  outside it.**
- AFDB's median is **3.4×** the training median (277 vs 81).
- **27.4% is above the 3,000-atom cap**, so scaling the count *under* the cap would leave 66a's
  confound untouched — which is exactly why the cap must lift in the same pass.

So a 1M AFDB corpus does not merely add data: it is the **controlled re-run 66a needs**, because its
training distribution spans the evaluation range for the first time.

### ⬛ A7 REVISITED AND RE-RECORDED (067): AFDB is now in scope

`PLAN.md` A7 reads *"Small scale is acceptable for a first prototype | ~21 proteins; explicitly
labelled illustrative; **no bulk/auto download; ESM Atlas not used**"*, and `README.md:60` repeats it.
**That exclusion was scoped to a ~21-protein illustrative milestone** and has been inherited unexamined
ever since.

**Re-recorded decision:** bulk download is now **in scope**, and the source is **AlphaFold DB**, on
measured grounds — reachable, 109 struct/s at P=32, 321 KB/structure, and a size distribution that
covers the range the static verdict is confounded on. ESM Atlas remains **not used**, now for a stated
reason rather than an inherited one: AFDB alone supplies 87% out-of-training-range coverage at 1M,
so a second predicted-structure source adds volume without adding the property that is missing.

### ⚠ 66a: 48b's verdict is CONFOUNDED BY ITS OWN TRAINING SET — re-worded, not withdrawn

`ladder_direct3m_n2272` trained on `splits_small_n2272`. Measured, not assumed:

| half | n | residues | median | above 109 |
|---|---|---|---|---|
| train | 2,272 | **21–110** | 81 | **1** |
| val | 758 | 20–109 | 82 | 0 |

**The model saw essentially nothing above 110 residues, and 48b evaluated it to 385.** So the design
cannot separate:

- *the architecture cannot represent large proteins* — the reading "SMALL-PROTEIN PROPERTY" implies;
- *this model was never shown one* — pure out-of-distribution extrapolation.

**Training distribution is a third branch, and the verdict rule I fixed in advance had only two slots.**
That is 60c's shape exactly — a two-way question answered by a third thing — committed in a rule I
wrote myself, and pre-registering it did not help because the missing branch was missing from the
pre-registration too.

**What the run does support**, and it is still worth having: the codec **does not extrapolate in size
beyond its training range**, degrading monotonically and losing physical validity (contact F1
0.959 → 0.607, clashes 13.8 → 179.6/1k). The 54c centroid control still rules out "the task merely got
harder" — the null rises 53% while the error rises 317%. What it does **not** support is any claim
about the architecture's capacity.

**The controlled re-run that would separate them:** train on a size-stratified corpus spanning the
evaluation range, then re-evaluate. That is 66b/067's scale-up, and 66a is the reason the 3,000-atom
cap must be lifted **in the same pass** — scaling the count under the cap leaves this confound exactly
where it is.

### ⬛ 48b RESULT (re-scoped by 66a — reads as extrapolation failure, not architectural capacity)

Run on all of `splits_big` minus the 261 fitted (4,715 clean), 1,400 evaluated, residues 25–395.
**0 excluded** for `max_positions` — the Family A guard fires empty, so the curve is not censored.

| residues | label | n | median all-atom Å | median contact F1 | clashes/1k | centroid null Å |
|---|---|---|---|---|---|---|
| 0–110 | **in-dist** | 55 | **0.86** | 0.959 | 13.8 | 12.7 |
| 110–150 | OOD | 324 | 2.09 | 0.850 | 63.4 | 14.2 |
| 150–200 | OOD | 336 | 2.75 | 0.749 | 109.7 | 15.1 |
| 200–250 | OOD | 181 | 3.17 | 0.677 | 156.3 | 16.6 |
| 250–300 | OOD | 237 | 3.38 | 0.639 | 183.6 | 17.5 |
| **300–395** | OOD | **267** | **3.59** | **0.607** | **179.6** | 19.4 |

**Verdict, by the rule fixed before the run:** top band n=**267** (n≥50 **met**), all-atom **3.59 Å**
(> the 2.51 SMALL-PROTEIN threshold), contact F1 **0.607** (< 0.75), clashes **179.6** (> 60).
**All three cross on the same side: SMALL-PROTEIN PROPERTY.**

**The 54c control is what makes it a finding rather than an artefact.** The predict-the-centroid null
rises only **12.7 → 19.4 Å (+53%)** across the range while the model's error rises **0.86 → 3.59 Å
(+317%)**. The task gets ~1.5× harder; the model gets ~4.2× worse. The degradation is **not** explained
by the metric getting harder with size — which is exactly what that control was added to separate.

The in-distribution band reproduces the reference (0.86 vs 0.8357 median), so the harness agrees with
the recorded evaluation where they overlap.

**Reach (54b/55d):** this is measured to the **~2,980-atom processing cap**, about the ATLAS *median*
system. It says nothing about the upper half of ATLAS, where 65 of 123 systems live.

**One defect in the run, stated rather than buried:** the per-structure `pclass` tag came back
`unknown` for all 1,400 — `cls_map` keys on split keys while the tag looks up `pdb_id`, and the two
differ. The **exclusion still worked** (the log confirms 261 dropped, 4,715 clean), so the curve above
is clean; but 54a's "report it both ways" sensitivity is **not available** from this run and would
need the tag fixed.

### ⏳ 48b PRE-REGISTERED (INBOX 53a) — recorded BEFORE the run, committed before it launched

**The claim under test.** Is **0.79 Å a property of the architecture, or of small proteins?** The
direct per-residue codec has only ever been evaluated on 20–109 residues. Nothing else is being asked.

**Reference — in-distribution held-out** (`splits_small_n2272`, n=758, 20–109 residues), from the
recorded evaluation:

| | value |
|---|---|
| all-atom median | **0.8357 Å** (mean 0.7917) |
| contact F1 median | **0.9639** |
| clashes/1000 atoms median | **11.9** (q75 19.1) |

**Evaluation pool:** `data/processed_big`. **CONTAMINATED, and the contamination is a deterministic
function of the regressor (INBOX 54a).** Measured, not inferred:

| | count |
|---|---|
| `big ∩ small-TRAIN` — **fitted on** | **261** |
| `big ∩ small-VAL` — genuinely held out | 80 |
| `big` only — never seen | 4,635 |

**5.2% of the pool is training data, and all 261 sit at 21–107 residues — zero above 109.** So the
small end is inflated by memorisation and the large end is clean, which *exaggerates* the measured
degradation with size and moves the crossing bands too early. The headline verdict survives (top band
clean vs a clean reference), but the GRADED reading — the likely outcome — is exactly what this
corrupts.

Every structure is therefore tagged **fitted / heldout / unseen** and the curve is reported **both
ways from one pass**. "In-distribution held-out" was three populations under one name; only the middle
is held out. That is the fourth one-name-two-things failure in four items, after `complex_d8`,
`ladder_direct_n2272` and `rmsd`.

**Reach (54b).** `processed_big` tops out at **383 residues ≈ 2,800 atoms**, against ATLAS quartiles
1,434 / 3,249 / 7,406 and a maximum of **33,377**. So 48b reaches about the ATLAS **median** system
and says nothing about the upper half. A clean ARCHITECTURE verdict here licenses "the static path
scales" only to ~2,800 atoms — stated **in the verdict line**, not the discussion.

**Size-calibration control (54c).** The thresholds are absolute and derived at 20–109 residues, which
assumes size-invariance that is untested and unlikely for contact F1. The predict-the-centroid null
(`centroid_rmsd`, 12.165 Å at reference size) grows with radius of gyration, hence with N, and is now
computed per structure. Reported per band beside the learned number, so a crossing separates "the
model got worse" from "the task got harder" — otherwise a SMALL-PROTEIN verdict is not separable from
a metric that simply gets harder, which is Family D applied to the verdict rule.

**Minimum n, pre-registered before band populations were known (INBOX 55b).** The threshold rule
never fixed how many structures the deciding band must hold, so a median over a thin top band could
cross 1.67 Å or 0.90 F1 on sampling noise and be reported as an architectural conclusion — Family C
waiting to happen. Fixed now:

> **n ≥ 50 in the ≥300-residue band.** Below that the verdict prints **UNDERPOWERED** with the n,
> and does **not** choose between ARCHITECTURE and SMALL-PROTEIN. Per-band n prints beside every
> threshold crossing, so a crossing in a thin band is visible as one.

**Pool corrected (55a).** The first relaunch used `splits_big.val` — but that boundary holds data out
from a model trained on `splits_big`, and the checkpoint under test trained on `splits_small_n2272`,
so `splits_big.train` is equally unseen. Restricting to val used **1,191 of 4,715** clean structures
and discarded 3,524 for a reason that does not apply. Now: **all of `splits_big`, minus the 261
fitted-on** — ~4× the sample at no additional risk, and it matters most in the ≥300 band, which is
both the smallest and the one the verdict reads.

**Reach is a CAP, not a data limit (55d).** `processed_big` has no manifest, so measured directly:
residues **22–385**, atoms **170–2,977**, with **zero above 3,000 atoms**. The atom maximum sitting
just under 3,000 is a `max_atoms 3000` processing cap — the same cap `manifest.json` records. So going
higher is a **re-processing job, not a data problem**, and a clean ARCHITECTURE verdict must read
*"to the ~2,980-atom cap"*, not *"the static path scales"*.

**Verdict rule, fixed now.** Comparing the top band (≥300 residues) against the reference:

- **ARCHITECTURE PROPERTY** — all-atom median ≤ **2×** reference (≤1.67 Å) **and** contact F1 ≥ **0.90**
  **and** clashes ≤ **2×** reference (≤24/1k).
- **SMALL-PROTEIN PROPERTY** — *any* of: all-atom median > **3×** reference (>2.51 Å), contact F1 <
  **0.75**, or clashes > **5×** reference (>60/1k).
- **GRADED** — anything between. Report the band at which each of the three crosses, separately.

All three metrics are reported per band with n, because 49b established that all-atom RMSD alone can
look respectable while the structure is unusable (1BYZ: 2.23 Å with ~2 clashes per atom). If RMSD
degrades gracefully while contact F1 collapses, **that is the finding**, and only the second metric
expresses it.

**Family A, stated in advance.** `max_positions = 1024` refuses any structure above it — and *the
exclusion criterion is the regressor*. The excluded count and residue range print per band **even when
zero** (47b's pattern). If anything is excluded, the curve is reported as **right-censored** at that
point, not as a measurement of its top band.

### ⬛ 052 AUDIT: the collision class is 61 runs wide, but the 3m ladder is clean

`outputs/cluster` holds **64 runs; 3 have `final.pt`, 61 do not.** Where there are no weights there is
no sha256, so a collision is undetectable from the record alone. Comparing `train_log.json`
(epochs, steps, wall, final train RMSD) between each committed run and the `$WR/results` run of the
same name:

**62 of 64 identical; 2 differ** — `complex_d8` and `ladder_direct_n2272`.

| ladder rung | weights | committed vs `$WR` |
|---|---|---|
| `ladder_direct3m_n2272` | Y | match |
| `ladder_direct3m_n450` | N | match |
| `ladder_direct3m_n878` | N | match |
| `ladder_direct_n2272` | Y | **DIFFER** (wall 37434/36239, rmsd 0.5838/0.5035; same 1703 ep, same 241968 steps) |
| `ladder_direct_n450` | N | match |
| `ladder_direct_n878` | N | match |

**The 3m ladder — the one behind §5 — is clean on all three rungs**, so its learning-curve slope is
not computed across two models. `ladder_direct_n2272` is a genuine but milder collision: identical
epochs and steps, differing only in wall time and final train RMSD, i.e. a re-run of one config.

**RETRACTED: "the longer training is worse."** `train_log`'s `rmsd` is `quick_rmsd` — an aligned RMSD
over the **first 4 batches of the *training* loader**, averaged over structures. `metrics.json`'s
`all_atom_rmsd` is per-structure over the full held-out set. Different population *and* a 4-batch
subsample, so 4.799 vs 6.722 was never a comparison, and train-RMSD-above-held-out-median is not an
anomaly — it is two different quantities. The three metrics also disagree in direction (train and
mean favour the short run, held-out **median** favours the long one), and 047 established the median
as the statistic describing the typical case.

**Guard added:** the report now warns when a `metrics.json` declares a checkpoint that does not exist
beside it. `outputs/cluster/complex_d8/metrics.json` declares `"final.pt"` against an absent file and
reported anyway — a provenance block naming a file it never opened.

### ⚠ NAME COLLISION: `complex_d8` is two different trainings, and it produced a false defect report

Two evaluations of "the same" checkpoint on the same 186 structures disagreed (mean 5.5844 vs 5.8843,
one structure by 16%). The hypothesis was the `res_pos_emb` load shim. **It is not.**

- The positional table is **(1024, 128) in checkpoint and model** — no growth, and the largest complex
  is 334 residues, so the clamp never fires.
- The demo is **deterministic run-to-run** (identical to 6 dp), so the difference is systematic.

The cause is that **`complex_d8` names two different trainings**:

| | epochs | steps | wall | final train RMSD | weights present |
|---|---|---|---|---|---|
| `outputs/cluster/complex_d8` | **900** | 63,000 | 9,473 s | 4.799 | **no `final.pt`** |
| `$WR/results/complex_d8` | **3457** | 241,990 | 25,134 s | 6.722 | yes |

Every path-shaped label was identical — arm, config basename, splits file, processed dir. Only the
checkpoint sha256 distinguished them, and a record with no weights has no sha to compare. **Neither
run is wrong.** The 049c band table came from `$WR/results`; the independent one came from
`outputs/cluster`; each is internally consistent with its own file on medians, range *and* Spearman.

**Fix:** provenance now carries the **training identity** (epochs, steps, wall, final train RMSD)
read from `train_log.json`, so two runs of the same name are visibly different in the report rather
than distinguishable only by hashing weights that may not exist.

**Consequences.** 48b is **not** blocked — the load path is sound. For any externally published page,
quote the **committed** run, because its `_true.pdb`/`_pred.pdb` pairs are the coordinates being drawn.

Note also, for the record: the longer training is **worse** (final train RMSD 6.72 at 3457 epochs vs
4.80 at 900). Different configs, so not a controlled comparison — but it is not the direction anyone
would assume from the names.

### ⚠ The complex size curve is RIGHT-CENSORED by a corpus filter (INBOX 55c)

Checked before attributing, because this is the cross-corpus join that keeps going wrong:
`data/manifest.json` maps **742 of 742** onto `splits_complex` and only **955 of 4,976** onto
`splits_big` — it is the **complex** manifest, so this applies to `complex_d8` alone.

    filters: min_residues 20, max_residues 400, max_atoms 3000, multi_chain, keep_ligands
    kept 5,880 | rejected 2,120
      residues_out_of_band  1,932   (401–4,802 residues, median 594)
      too_many_atoms          188

So the complex arm's 52–334 residue span is **where the filter cuts, not where the data runs out**.
1,932 real structures between 401 and 4,802 residues were refused at corpus construction, and
`max_atoms 3000` is tighter still — it sits **below the ATLAS median of 3,249 atoms**.

The 49c/50b trend *inside* the window stands. The window is not a property of the molecules, and the
size curve is right-censored at its top by an exclusion whose criterion is the regressor — the same
Family A statement 48b's pre-registration makes about `max_positions`.

### ⬛ 049: the complex arm reconstructs to a number, not to a usable structure

`clashes_per_1000_atoms` was computed and stored for every structure and printed in no table. It is
the column that decides whether a reconstruction is physically real, and the two arms do not overlap
on it:

| arm | all-atom Å | clashes / 1000 atoms |
|---|---|---|
| single-chain | 0.64 – 1.26 | **5.0 – 77.1** |
| complex | 2.23 – 15.54 | **1,831 – 8,379** |

`1BYZ` is the complex arm's **best** case at 2.23 Å — a number that reads as a good reconstruction —
while carrying roughly **1.9 steric clashes per atom**. Across the full 186 the median clash rate never
falls below **1,740 per 1,000 atoms** in any size band. And `chirality 0.0000` sits on every row
including one at 15.5 Å, so a reader saw a stereochemistry check passing and reasonably inferred sound
geometry. **Chirality survives the bottleneck; physical validity does not.**

**Two regimes, not one median** (from the recorded evaluation, n=186):

| residues | n | median all-atom Å | median clashes/1k |
|---|---|---|---|
| 50–100 | 20 | 2.22 | 1,740 |
| 100–150 | 35 | 2.29 | 1,744 |
| 150–200 | 34 | **5.60** | 1,954 |
| 200–300 | 89 | 6.08 | 2,132 |
| 300–334 | 8 | **15.46** | 6,538 |

Rolling median crosses 2 Å at ≥52 residues and **5 Å at ≥122**; Spearman(residues, RMSD) **0.499**.
Contact F1 collapses alongside: 0.959 → 0.482 → 0.286 → 0.073.

**This bounds 48b without answering it.** That curve is `complex_d8` on `processed_complex`; 48b asks
about `ladder_direct3m_n2272` on single chains — different checkpoint, different training
distribution. What it establishes is that *a* per-residue codec of this design degrades steeply with
size rather than holding. 48b still needs its own evaluation above 109 residues.

### ⬛ 048: the project's success and its failure sit on almost disjoint domains — stated positively

The unaided reading of this repo is that "0.79 Å reconstruction" and "loses to zero-shot ANM on 100%
of 123 systems" contradict each other. **They do not, and neither transfers.** They differ on three
axes at once:

| | §5 direct codec | ATLAS modal line |
|---|---|---|
| task | static structure | per-frame displacement |
| size | **157–799 atoms** (20–109 residues) | **598–33,377 atoms** |
| latent | per-residue, **scales** | fixed, **L=1** |

The **entire §5 range sits inside ATLAS's first quartile** (boundaries 1434 / 3249 / 7406), most of it
below the smallest ATLAS system, and the largest §5 structure is **1/42** the largest ATLAS one. Both
statements are true and they share almost no domain.

**48b, the open question this makes visible:** the direct per-residue codec has **never been evaluated
above ~109 residues**. Whether 0.79 Å is a property of the architecture or of small proteins is
unknown, cheap to answer (inference on existing checkpoints, no training), and sits underneath every
plan assuming the static path scales. Queued behind the ladder, 41c and the demo.

### ⬛ Both headline numbers are MEANS, and they mislead in opposite directions

Measured from the recorded per-structure evaluations, not re-derived:

| arm | n | mean | median | range | skew |
|---|---|---|---|---|---|
| §5 single-chain | 758 | **0.7917** | **0.8357** | 0.27–2.38 | left — mean *below* median |
| `complex_d8` | 186 | **5.8843** | **2.4907** | 1.91–21.57 | right — mean **2.4× the median** |

So "complex plateaus at 5.5–5.9 Å" describes a mean dominated by a long tail; the **typical** complex
reconstructs at **2.49 Å**. And §5's 0.79 understates its typical case. Quote both statistics for
both arms — this is 28e's rule (median, mean and failure fraction together) applied to the two numbers
the project leads with.

Demo illustration of the size dependence in the complex arm: 2.23 Å at 52 residues, 6.06 at 164,
7.47 at 233, **15.54 at 334**.

### ⬛ NMR: conformers do not collapse, and the codec does resolve them — on the fuller sample

| | 3 entries × 6 models | **11 entries × 126 conformers** |
|---|---|---|
| true spread | 1.1073 Å | **3.5437 Å** |
| reconstructed spread | 0.9206 Å | 3.3705 Å |
| ratio | 0.831 (16.9% lost) | **0.9324 (6.8% lost)** |
| mean recon error | 1.1969 Å | **1.1639 Å** |
| criterion 1 (recon ≪ spread) | appears to FAIL | **passes** (1.16 ≪ 3.54) |

**The 3-entry read was wrong and is retracted.** Those three happened to be low-spread ensembles
(1.11 Å), so the reconstruction error looked larger than the spread. On 11 entries the spread is
3.54 Å and the error 1.16 Å, comfortably below it — the script's criterion 1 passes. A three-point
sample read as a finding is the error this file spends most of its length cataloguing.

Note the ratio is sample-dependent too: **0.9324 here vs §5's 0.868**, on different entry and model
counts. Report the count beside the ratio.

### ⚠ TWO CHECKPOINT-COMPATIBILITY DEFECTS, found by trying to load the 0.79 Å model

Building the atom-level demo (045/046) required loading the checkpoint behind §5's headline. It is
**`ladder_direct3m_n2272`** — all-atom **0.7917**, backbone **0.5103**, chirality **0.0002261**,
contact F1 **0.9629**, matching §5's four numbers exactly. Note it is *not* any of the three
`*perresidue*` result dirs, which sit at ~10 Å; the naming does not identify it.

**FIXED AT THE CLASS.** The remap now lives in `molae/utils.load_checkpoint` (`remap_legacy_keys`),
so every caller gets it. It was found via the demo but it is not a demo bug: `latent_suitability.py`
failed identically, which means **§5's NMR number (0.868) was also unreproducible** by the script that
produced it. Two of §5's bullets were blocked by one broken promise.

**1. `PositionEncoding`'s backward-compatibility promise is broken.** `molae/scaling.py:73` states
*"Default stays learned so every existing checkpoint loads and every prior result reproduces."* It
does not: the refactor wrapped a bare `nn.Embedding` as `self.table`, so a checkpoint holding
`decoder.res_pos_emb.weight` cannot load into `decoder.res_pos_emb.table.weight`. **Every direct
checkpoint predating that refactor fails to load**, including this one. The remap is exact *within*
`max_positions`; beyond it the new path `clamp`s where the old raised, so the demo asserts
`n_res <= max_positions` per structure rather than assuming it.

**2. `grow_embedding_rows` silently substitutes untrained weights.** Loading grew three embeddings
(elements 16→21, residues 22→23, res-type 22→23) with **randomly initialised** new rows. Any
structure indexing into them is being reconstructed with untrained embeddings, so its number is not
that checkpoint's result. The demo **refuses** such a structure rather than reporting it.

Neither is a demo bug — both are live for any script loading a pre-refactor checkpoint, which
includes anything reproducing §5.

**Verified end-to-end on CPU, 4 held-out structures spanning 157–799 atoms:** all-atom
1.26 / 0.89 / 0.94 / 0.64 Å, median **0.92 Å**, chirality 0.0000 throughout, contact F1 0.889–0.977,
latent **8 floats/residue** giving only **~3× compression**. Not launched; held for a free slot per
045's stated priority.

### ⬛ 22a RUNG 1 (modal_ctx, 10307539, COMPLETE): the pre-registered "flat" branch is REFUSED

Message passing, `ctx_layers` 2/4/8 at k=16, DM=256, n_train=50, LR grid swept:

| ctx | best LR | FVE | PR | identity | eff-modes | basis-off | reach |
|---|---|---|---|---|---|---|---|
| 2 | 3e-4 | 0.0914 | 35.4 | 69% | 21 | 0.186 | ~5.3 Å |
| 4 | 3e-4 | 0.0913 | 31.0 | 64% | 17 | 0.203 | ~10.6 Å |
| 8 | 3e-4 | 0.0814 | 39.5 | 20% | 15 | 0.235 | ~21.2 Å |

22a pre-registered: *"FVE flat in ctx ⇒ the receptive field is the limit, and rung 2 (Laplacian
eigenvector features) is the next rung."* **That branch is refused, because the design cannot support
it.** Every cell is **one seed**. Against 24c's measured single-arm SD for the nearest comparable arm
(untied lr3e-4, **0.0186**, 3 seeds), a difference of two 1-seed arms carries a 95% half-width of
**0.0515**:

| comparison | difference | CI | verdict |
|---|---|---|---|
| ctx2 vs ctx4 | +0.0001 | [−0.0514, +0.0516] | **NOT RESOLVABLE** |
| ctx2 vs ctx8 | +0.0100 | [−0.0415, +0.0615] | **NOT RESOLVABLE** |

Both CIs contain **zero and the 0.0350 gap**, so "flat" is not licensed — that is Family C, and it
would have justified building rung 2 on an underpowered null. Note the SD is *borrowed* from a
different arm, which is 32a's objection; the ctx sweep measured **none of its own**, which is itself
the finding.

**Rung 2 stays unjustified rather than justified.** The cheap repair is seeds, not architecture: 3
seeds per ctx would cost 6 more arms and would make either branch readable. Also note this was
measured at **n_train=50**, which the ladder has now shown to be a floor for the untied arm.

### ⬛ 52d AUDIT COMPLETE: **2** name collisions in 64 committed runs, and one has weights

`complex_d8` was not alone. `scripts/armf_run_collision_audit.py` compares the **final training row**
of every `outputs/cluster/<run>/train_log.json` against `$WR/results/<run>` of the same name — no
weights needed, which is the point: **61 of 64 committed runs have no `final.pt`**, so a sha256 can
never detect a collision in them.

| run | committed (epoch / steps / s / rmsd) | `$WR` | `final.pt` committed |
|---|---|---|---|
| `complex_d8` | 899 / 63,000 / 9,473 / **4.799** | 3456 / 241,990 / 25,134 / **6.722** | no |
| **`ladder_direct_n2272`** | 1703 / 241,968 / 37,434 / **0.584** | 1703 / 241,968 / 36,239 / **0.504** | **yes** |

The second is new and differs in kind. **Identical epoch and steps, different wall time and different
final train RMSD (0.584 vs 0.504)** — the same schedule run twice, not a longer run. A sha256 *would*
have caught this one, since the weights are committed; nobody had compared them. It already had an
outstanding control against it (ROADMAP L269), which now has a second reason.

**What the audit clears.** `ladder_direct3m_n2272` — the headline single-chain arm, 0.79 Å — does
**not** collide, and neither do the other ladder rungs. The learning-curve slope is computed on one
training per rung and is safe to quote.

**Why reproducibility tracked weight availability.** The single-chain arm agreed across both
evaluations *and* is one of only three runs with committed weights; `complex_d8` diverged and has
none. That correlation was the visible signal, but the cause is name reuse, and the audit is what
converts "one detected collision" into "two, out of a bounded 64".

### ⬛ 048: THE SUCCESS AND THE FAILURE ARE ON ALMOST DISJOINT DOMAINS — stated positively

A reader who meets *"the codec reconstructs at 0.79 Å"* and *"the codec loses to a zero-cost baseline
on 100% of systems"* will reach for a contradiction. **There isn't one, and the two results share
almost no domain.** Measured, not asserted:

```
section 5   157 ────── 799                (n=758 held-out, median 635 atoms)
ATLAS                598 ──────────────────────────────── 33,377   (n=123 held-out)
                          Q1 1434    Q2 3249    Q3 7406
```

| overlap, measured | |
|---|---|
| ATLAS systems inside §5's atom range | **10 / 123 (8.1%)** |
| §5 structures below the *smallest* ATLAS system (598 atoms) | **311 / 758 (41.0%)** |
| §5 largest (799) vs ATLAS largest (33,377) | **1/42 the size** |
| does all of §5 sit below ATLAS Q1 (1434)? | **yes** |

The two differ on **three axes at once**:

| | §5 | ATLAS |
|---|---|---|
| task | static structure | per-frame displacement |
| size | 157–799 atoms | 598–33,377 atoms |
| latent | per-residue, **scales** with residue count | fixed, L=1 |

**Both are true and neither transfers.** The 0.79 Å is not weakened by the peer loss, and the peer
loss is not softened by the 0.79 Å. Any claim that spans them is a join across three axes at once —
the failure mode that produced 41a, 42b and 045's ligand row on one axis each.

### ⬜ 048b QUEUED (behind the ladder, 41c and the demo): does the direct codec hold above 109 residues?

**The one architecture in this project that demonstrably works has never been evaluated above ~109
residues** — smaller than the smallest system the rest of the project studies. That is not a
criticism of the held-out set; it is a scope statement that was invisible until §5's range was put
next to ATLAS's.

Unlike most open questions here it is **cheap**: inference on existing checkpoints against existing
structures, no training. It would not touch the dynamics question — different task — but it would
separate *"0.79 Å is a property of the architecture"* from *"0.79 Å is a property of small
proteins"*, and that distinction sits underneath every plan assuming the static path scales.

### ⛔ RETRACTED AS A *TIED* RESULT: the ladder trained the **UNTIED** architecture

`armf_tied_ladder.py` was derived from `armf_modal_seeds.py` so the training loop would be
bit-identical. `make()` came across **unchanged** — and `modal_seeds` only ever passes `"control"` or
`"untied"`, so its else-branch **hardcodes `tie_encoder=False`**. Setting `VARIANTS = ["tied"]` here
therefore built the untied architecture and stamped every record `variant="tied"`.

**Proof, not suspicion.** The ladder's three n50 arms reproduce 24c's *untied* lr3e-5 cell to ten
decimal places with identical step counts:

| seed | 24c untied | ladder "tied" | steps |
|---|---|---|---|
| 0 | +0.0996116863 | +0.0996116863 | 67,500 both |
| 1 | +0.1128945779 | +0.1128945779 | 70,000 both |
| 2 | +0.1009773579 | +0.1009773579 | 57,500 both |

`tie_encoder=True` is the tied arm — `armf_modal_decoder.py:207` and `armf_tied_peer.py`, which builds
`tie_encoder=True`. So the convention is unambiguous and the label was wrong.

**What survives, re-scoped to the untied arm:** the rise is real and the rank test still holds —
untied is **not saturated** in `n_train` over 50–300, JT 27/27, p = 0.00060, slope +0.0483 ± 0.0296.
Nothing about the *measurement* was wrong; only the architecture it was attributed to.

**What does not survive:** every sentence naming the tied arm. **The tied arm's data-scaling is
unmeasured.** And 41a joined the untied slope to the *tied* arm's peer gap — a cross-architecture
join, which is Family F at the level of the conclusion.

**Fixes.** `make()` now maps every variant explicitly and **raises** on an unknown one rather than
defaulting to an architecture. `variants` is now in the stamp — it changes *how* an arm is computed,
unlike seeds and rungs — so the 9 untied arms match the corrected stamp **0/9** and cannot resume as
tied. That hole is the same one the `LADDER_MAXSTEPS` cap had, arriving through a different door. The
untied arms are preserved at `untied_ladder_measured.json`.

**Re-running: 10314088 → 10314089 → 10314090**, the genuine tied ladder.

### ⬛ 31d/34c LADDER (**UNTIED ARM** — see the retraction above): THE CEILING MOVES WITH DATA

Complete, 9/9 arms, job 10311656.

| rung | usable | mean FVE | SD | at cap | VOID |
|---|---|---|---|---|---|
| n50 | 3 | **+0.1045** | 0.0073 | 0/3 | 0 |
| n130 | 2 | **+0.1275** | 0.0008 | 3/3 | 1 |
| n300 | 1 | **+0.1403** | — | 2/3 | 2 |

**PRIMARY (fixed before results, 34c): slope of FVE on log10(n_train) = +0.0483 ± 0.0296**
(HC3, t, df=4, n=6 arms, R² 0.906). **Excludes zero.** Implied change over the measured 6.0× range:
**+0.0376 ± 0.0230**. It survives the 39b pessimistic-SD substitution (±0.0305) — the point estimate
is unchanged by construction there, only the bar moves.

So **31d-(1) is NOT the binding constraint, and (2)/(3) are premature.** Per 32c, pre-registered:
14a, 17c, 25a and the entire peer comparison were measured at n50, so each was measured on an
**under-trained** model and is a **floor, not an estimate** — including `FVE⊥ ≈ 0` and the
0%-of-systems peer loss. It does not soften the n50 peer result; the headline now carries
**"at n_train = 50"** until the peer comparison is re-run at the best rung.

Per **38b**, recorded before the numbers existed, this is the interpretable branch: the instrument is
biased *against* finding a rise, so a rise measured through it is **conservative** — the true effect
is at least this large. **38c does not fire**; no re-run is triggered.

**Family A is now measured, not conjectured.** 38a said neither bias direction was quantified. The
VOID arms have FVEs, so the exclusion effect is arithmetic:

| rung | usable mean | all-arms mean | shift from excluding VOID |
|---|---|---|---|
| n130 | +0.1275 | +0.1261 | **+0.0015** |
| n300 | +0.1403 | +0.1411 | **−0.0007** |

Both are tiny against a +0.0376 effect and they **change sign**, so Family A is negligible here
rather than merely opposed. Family D stays unquantified — nobody knows where a truncated arm would
have converged — but each VOID FVE is a *lower bound* on it, and **n300 s2 reached +0.1527 while
still climbing**, the highest arm in the ladder. The suppression is real and it runs against the
conclusion, which is why the conclusion survives it.

**041: what the slope can actually buy — and why the fork licenses less than 31d said.**

The measured rate is **+0.0483 FVE per decade** of `n_train`. Against the zero-shot peer, on the
statistic the ladder tracks (mean FVE over held-out systems, tied vs ANM-256, measured together per
system in `tied_peer.json`):

| gap statistic | gap | decades to close | systems needed |
|---|---|---|---|
| Q1 median | +0.3956 | 8.2 | 4.7×10¹⁰ |
| median, all 123 | +0.4629 | 9.6 | 1.2×10¹² |
| **mean — like-for-like with the ladder** | **+0.6114** | **12.7** | **1.4×10¹⁵** |

Exhausting the entire remaining ATLAS pool (300 → 697, 0.366 decades) buys **+0.0177 FVE**. The
shortfall is 20–30× on every statistic, so the conclusion does not depend on which one is chosen.

**41b — the pre-registered dichotomy was too coarse, and the verdict text has been narrowed.** 31d
read *"rising → data-limited, (2)/(3) premature"*, which treats a rise of **any size** as retiring the
fundamental-limit hypothesis. It conflates *data helps at the margin* with *data is the binding
constraint*. What is supported:

> The tied arm is **not saturated** in `n_train` over 50–300. The measured rate **cannot close the
> peer gap with any data this project can obtain**, so options (2) and (3) **remain live** and are not
> deferred by this result.

The **direction** is established at p=0.0006 by the assumption-free rank test; the **rate** rests on a
top rung of one usable arm, and this extrapolation inherits the weaker of the two. Learning curves
flatten rather than staying log-linear, so the decade counts are optimistic.

**41c — the n300 peer re-run is pre-registered, before it runs:** the codec is expected to gain
≈ +0.018–0.036 over its n50 value and to **still lose to zero-shot ANM on ~100% of systems**. A loss
confirms the floor is a floor. A **win** falsifies the 41a extrapolation and would be the most
important result in the project. Recorded in advance so a still-losing result is not read as new
information when it is the prediction — the 38b move applied to the peer comparison.

**The result that survives everything (040).** Every concern raised — 37a's equal-variance
assumption, 39a's mechanical SD deflation, 40a's leverage imbalance — is about variances, estimators
or exclusions. A rank test on the ordering depends on none of them:

| rung | arms, sorted |
|---|---|
| n50 | +0.0996 +0.1010 +0.1129 |
| n130 | +0.1231 +0.1270 +0.1281 |
| n300 | +0.1302 +0.1403 +0.1527 |

**The ordering is complete** — every arm at a higher rung exceeds every arm at a lower one, all nine.
Exact Jonckheere–Terpstra **27 of 27**, one-sided **p = 1/1680 = 0.00060**, computed on *all* arms so
the exclusion rule drops out too. This is the most robust form of the result and the weakest in effect
size: it establishes the **direction**, not the magnitude.

**Sensitivities, all printed:**

| variant | slope | HC3 half-width | excludes 0 |
|---|---|---|---|
| pre-registered (usable arms, n=6) | +0.0483 | ±0.0296 | yes |
| 39b pessimistic SD | +0.0483 | ±0.0305 | yes |
| **40b all 9 arms, VOID at truncated FVE** | **+0.0471** | **±0.0263** | **yes** |
| 40a counterfactual: n300 forced to the n130 level | +0.0357 | ±0.0602 | **no** |

The last row is a counterfactual, not a reading of the data — and the data contradicts its premise,
since the *lowest* n300 arm (+0.1302) exceeds the *highest* n130 arm (+0.1281). But it is the honest
statement of what the result rests on: n300 being genuinely above n130.

**Leverage (40a):** n50 h=0.305 each, n130 h=0.208 each, **n300 h=0.668** — one arm carrying more than
twice any other, at the end that sets the lever arm.

**The honest weakness: the top rung rests on ONE usable arm.** n300 is k=1, so it contributes no
spread of its own, and the 6.0× lever arm is anchored by a single draw. The HC3 interval prices this
through the residuals, and the pessimistic sensitivity passes, but a k=3 top rung would be a
materially stronger result than what is reported here.

### ⚠ LIVE CONFOUND found at n130, before n300 lands: the step ceiling binds harder as n_train rises

n130 s2 came back **VOID** (`maxsteps`, still improving). One lost arm is not the finding; the
pattern behind it is:

| rung | arms | mean steps | at the 90k ceiling | VOID |
|---|---|---|---|---|
| n50 | 3 | 65,000 | **0/3** | 0 |
| n130 | 3 | 90,000 | **3/3** | 1 |

`maxsteps_for(3e-5) = min(90000, 30000·√(1e-3/3e-5)) = min(90000, **173,205**) = 90000`. The **cap
binds**, giving arms 48% less than this project's own √-scaling rule prescribes — and the ladder runs
at 3e-5 *exclusively*, the rate [`armf_atlas_dm.py:100`](scripts/armf_atlas_dm.py#L100) already
records as *"hit MAXSTEPS still improving and are flagged VOID … a Family D hole opened by the
Family E repair."*

The budget is **constant across rungs**, so compute is matched. What is not matched is how far that
budget gets each rung from convergence, and that distance grows with `n_train` — the regressor of the
primary test:

- **Family D.** Truncation understates FVE at high n, biasing the slope **toward zero**. A bounded
  null would then partly measure the step budget rather than the data — which is this project's own
  doctrine verbatim: *"undertrained at a fixed budget … manufactures a flat curve."*
- **Family A.** VOID *means* "still improving at the ceiling", so the **exclusion rate rises with the
  regressor** and the surviving high-n arms are the fast-converging subset, not a random one.

**n300 s0 confirms it, and the VOID calls are NOT noise.** The plateau/VOID discriminator is
`max(last 4 evals) > 1.01 x max(all earlier)`, and it separates cleanly:

| arm | ratio | call | at ceiling? |
|---|---|---|---|
| n50 s1 | 0.9326 | kept | no (70k) |
| n50 s2 | 0.9863 | kept | no (57.5k) |
| n130 s0 | 0.9927 | kept | yes |
| n130 s1 | 0.9883 | kept | yes |
| n130 s2 | **1.0550** | VOID | yes |
| n300 s0 | **1.1407** | VOID | yes |

Every kept arm is <= 0.993, every VOID arm >= 1.055, with the 1.01 threshold in the gap. A ratio
below 1.0 means the recent window peaked *below* the earlier max, so n130 s0/s1 had genuinely stopped
climbing. **n300 s0 was still climbing 14% at the ceiling** -- the budget is really too small there,
and its `+0.1302` is a true lower bound that already exceeds every other arm in the ladder.

That last point matters for the direction of the bias: the confound suppresses the rise it would have
to manufacture to be dangerous. FVE by rung is n50 +0.1045, n130 +0.1276, n300 >= +0.1302.

**039: a THIRD ceiling effect, pointing the other way.** A rung whose arms all stop at the *same*
step loses the stopping-point component of variance. n130's arms all ran to exactly 90,000, so its
SD of **0.0008** is at least partly **mechanical** — and the consequence runs opposite to the other
two: a deflated SD **narrows the interval and makes a rise easier to declare**.

| effect | arms | direction on the verdict |
|---|---|---|
| truncation (Family D) | 3/3 at n130 | suppresses the rise |
| VOID exclusion (Family A) | 1/3 at n130 | inflates the high-n mean |
| **SD deflation at ceiling-bound rungs** | 3/3 at n130 | **makes a rise easier to declare** |

This reframes 37a: the SD ratio is evidence about *which rungs hit the cap*, not about the data.
HC3 remains right — it is robust either way.

**The pessimistic sensitivity, and where it bites.** Substituting the widest cap-free rung's SD
(n50's 0.0073) into every ceiling-bound rung:

| test | as measured | pessimistic | excludes 0? |
|---|---|---|---|
| pairwise n50→n130, pooled *[sensitivity]* | ±0.01737 | ±0.02121 | yes |
| pairwise n50→n130, **Welch** *[authoritative]* | ±0.01771 | **±0.02562** | **NO** |
| **PRIMARY: slope, HC3** | ±0.0296 | **±0.0305** | **yes** |

The pairwise survives under pooled but **not** under Welch — 37a's own demotion of the pooled form
biting. Per 34c the primary is the *slope*, so this does not decide the fork, and the primary does
survive. Both sensitivity rows now print. Because `x` is constant within a rung, the slope depends
only on the rung means, so inflating within-rung deviations leaves the point estimate **exactly**
unchanged and moves only the bar.

**038: the two biases OPPOSE, and the verdict is asymmetrically readable.**

| | mechanism | arms at n130 | slope bias |
|---|---|---|---|
| Family D | truncation — high-n arms cut off before convergence, FVE understated | **3/3** | **down** |
| Family A | VOID exclusion — VOID *means* still improving, so the *slow* movers are dropped and the plateaued subset survives | **1/3** | **up** |

An earlier version of this section said the net was "toward zero". That is stronger than the argument
supports: truncation touches 3× as many arms and probably dominates, but the net is **not
established**, and neither direction is quantified by this design.

The consequence is that **the ladder can confirm data-limitation but cannot refute it**:

- **RISE** → measured *through* an instrument biased against rises, so it is **conservative** — the
  true effect is at least as large. Fully interpretable; option (1) stands.
- **FLAT** → indistinguishable from the step budget. A bound here is partly a statement about 90,000
  steps, not about data, so it **cannot retire option (1)** — which is exactly what 32b and 33c
  established the ladder must be able to do.

**Decided before the number exists (38c): a flat result triggers a re-run at 173,205 steps**
(`LADDER_MAXSTEPS=173205`, ~2× compute over 9 arms). The alternative — report it as confounded and
leave the fork open — is an underpowered null wearing a result's clothes, which is precisely what 35a
and `null_verdict` exist to refuse. And 32c makes this fork decide whether the n50 peer loss is a
finding or an artefact of under-training, so it is worth the compute.

**The budget is deliberately NOT being raised mid-ladder.** Doing so would make the rungs
incomparable on compute — trading a visible confound for a hidden one. Instead the verdict now prints
the per-rung step table and carries a `CEILING BINDS HARDER AT HIGH n` tag on the verdict line, beside
`PARTIAL`.

**Risk this creates for the fork:** if n300 voids at a higher rate still, that rung may end with too
few usable arms to enter `pts` at all. The monitor already handles it — its rung check counts
*non-VOID* arms, so it will report `CHAIN ENDED with 2/3 rungs` rather than terminating on a
two-rung verdict.

### ⬛ 037 PRE-REGISTERED, before n300 exists: the ladder reports Welch + HC3, unconditionally

Pooling across rungs and the OLS slope both assume **equal variance across rungs**, which the ladder
never tested and never printed. Now printed: per-rung SDs and their max/min ratio, beside the pooled
value. On the live partial store that ratio is **9.11** (n50 SD 0.0073 at k=3, n130 0.0008 at k=2).

37a proposed switching to Welch/HC3 above ~4x. **A stricter rule was recorded instead, on a measured
number:** at k=3 seeds per rung, simulating 400k draws of three rungs under *true* equal variance,
`P(max/min SD ratio > 4) = 0.24` and the **median ratio is 2.52**. A 4x switch therefore fires a
quarter of the time when the assumption holds perfectly, so which estimator reports the verdict would
be settled by noise -- 28e arriving through the mechanism built to stop it.

**The rule, fixed now:** the robust pair -- **Welch** for the pairwise, **HC3** for the slope -- is
authoritative *unconditionally*. Pooled/OLS prints beside it as a sensitivity check, never as an
alternative verdict. This removes the estimator choice from the data rather than making it a coin flip.

The price is **measured on the exact design** (3 rungs x 3 seeds, x = log10(50,130,300)), 200k
simulations per row:

| true variances | OLS coverage | HC3 coverage | HC3/OLS width |
|---|---|---|---|
| equal | 0.952 | 0.960 | **1.16x** |
| unequal (extreme) | **0.868** | 0.947 | 1.47x |
| n130 tight only | 0.970 | 0.937 | 0.96x |

HC3 costs **16%** width when the assumption holds. When it fails, OLS's nominal 95% interval is
really **87%** -- and an under-covering interval on the *primary* test is exactly what manufactures a
false "THE CEILING MOVES WITH DATA". Honest caveat: in the one-tight-rung pattern the early data
hints at, HC3 itself under-covers slightly (0.937). Neither is exact at n=9; HC3 is closer to nominal
in both non-equal cases, which is why it is the one that reports.

The HC3 sandwich was cross-checked against an independent matrix implementation (agreement to 1e-12);
statsmodels is unavailable in this venv.

### ⬛ 031: the crossing corroboration is REAL but WEAK, and the 91% was a ratio-of-medians artefact

**31a — corroboration, stated with its uncertainty rather than its point estimate.** The fitted
`log10(a*) = −0.4431·log10(N) + 1.4825` crosses `a* = 1.0` at **N = 2217**, against
**N_ref = 2460** fixed independently from the 50 training systems. Point estimates agree to **9.9%**
— and the slope CI puts the crossing anywhere in **901–7187**, a factor of eight.

So this is corroboration, and it is **weak** corroboration. 031 describes it as landing "within a few
percent"; the point estimate is within ten, and the interval is far too wide to carry weight on its
own. Recorded because the *direction* is right and it costs nothing — not because it survives
scrutiny as an independent confirmation. **The same discipline 22b imposed on the positive gap slope
applies here, and this one points the way I want.**

**31b — the "N-only recovers" column was a ratio of medians, and 031 is right that it misreports.**
Per-system recovery, median [IQR]:

| band | available gain | **per-system median** | IQR | ratio-of-medians (what I reported) |
|---|---|---|---|---|
| Q1 | 0.0376 | **80%** | [−313%, 90%] | **3%** |
| Q2 | 0.0132 | 38% | [−55%, 60%] | −15% |
| Q3 | 0.0117 | 73% | [−612%, 94%] | 72% |
| **Q4** | **0.6009** | **97%** | **[94%, 99%]** | 91% |

The Q1 figure was **3% by ratio-of-medians and 80% per system** — the artefact 031 predicted. But the
IQRs are the real story: at Q1–Q3 they span from −600% to +94%, and **only Q4 is tight**.

**31c — so the honest claim is narrower than "recovers 91%", and the R² already said so.** `R² = 0.649`
on the `a*` fit means **35% of per-system `log a*` variance is unexplained by N**. A correction with
that much scatter captures most of a *large* gain and is swamped on a *small* one — exactly the
pattern. The supportable form:

> The N-only correction **reliably removes the large scale error at high N** (Q4: median 97%, IQR
> [94%, 99%]) and is **within noise of doing nothing** where the scale error is already small
> (Q1–Q3: medians 38–80%, IQRs spanning −600% to +94%).

That still removes the Q4 collapse, which is the result. It is not a general-purpose calibration fix.

### ⬛ 21c/27a/27b LANDED: the peer loss survives the cutoff sweep, the gap slope does NOT, and 27b is refuted

**21c — the peer result does not depend on the 2.9% cutoff selection.** All three cutoffs, best arm,
123 systems:

| cutoff | ANM-k | codec−ANM | codec > ANM | **gap vs log10(N)** |
|---|---|---|---|---|
| 5 Å | 0.6720 | −0.5168 | **0%** | **+0.0514 ± 0.0464** |
| 7 Å | 0.6271 | −0.4718 | **0%** | +0.0156 ± 0.0477 |
| 10 Å | 0.5669 | −0.4106 | **0%** | **−0.0207 ± 0.0509** |

**ALL CUTOFFS AGREE: the codec loses, on 0% of systems, at every cutoff.** 14a's headline stands on
its own rather than on a coin flip — which is exactly what 21c was written to establish.

> ### ⛔ RETRACTED: "the codec closes on ANM as systems get larger"
> The positive gap-vs-N slope was recorded (with 22b's caveat that it was weak) as *the one
> measurement pointing the right way*. **It flips sign across the cutoff sweep** — +0.0514 at 5 Å,
> +0.0156 at 7 Å, **−0.0207 at 10 Å**. It was an artefact of the cutoff selection, and 21c caught it
> for precisely the reason 21c existed. The project has **no** measurement showing the codec
> improving relative to the peer with N.

**27a — the `FVE⊥` slope SURVIVES at n=123.** It was −0.1844 ± 0.1154 (ratio 1.60) on 24 systems,
one arm, one seed, and was being used to overturn a flat aggregate. Recomputed on **all 123**:

| | slope | ratio | n |
|---|---|---|---|
| ANM-6 | **−0.1073 ± 0.0628** | 1.71 | 123 |
| ANM-16 | −0.1002 ± 0.0630 | 1.59 | 123 |

Same sign, smaller magnitude, still excluding zero, now on five times the systems. **The
discriminating metric does degrade with N while aggregate FVE is flat**, and the sharpest sentence in
the project keeps its support. The distribution also holds: `FVE⊥` median **−0.0266**, IQR
[−0.0779, +0.0148], **above zero on 32% of 123** (n=24 gave −0.026 and 33%).

**27b — NOT REFUTED. NOT SEPARABLE BY THIS DESIGN (INBOX 35a).** The hypothesis was that the encoder decay (`‖z‖/‖disp‖ ~ N^−0.52`) and
the `FVE⊥` degradation are one finding, and that my reason for separating them appealed to the metric
25a discredited. Tested directly, same arm, same frames, n=123:

| | partial `log(z_ratio)` | partial `log N` |
|---|---|---|
| ANM-6 | **−0.3103 ± 0.3375** (spans zero) | **−0.2668 ± 0.1843** (excludes zero) |
| ANM-16 | −0.2169 ± 0.3410 (spans zero) | −0.2116 ± 0.1862 (excludes zero) |

**I read this as a refutation and it is not one — it is an underpowered null, which is Family C in a
verdict branch I wrote myself.** `|effect|/half-width` for `log(z_ratio)` is **0.92 — below one**, its
point estimate is **1.16× larger in magnitude than N's**, and its CI **[−0.6478, +0.0272]** is
consistent with no effect *and* with a large one. "Adds nothing once N is held" is not licensed by
that interval; my branch treated *does not exclude zero* as *is zero*.

**And the imprecision is structural, so no further systems fix it.** `z_ratio ~ N^−0.52` at R² 0.87
leaves `corr(log z_ratio, log N) = −0.9412`, **VIF 8.76** (SEs inflated ~3×), and only **11%** of
`z_ratio` variance orthogonal to N — that 11% is all the partial has to work with. Reaching a ratio
of 1.0 at this effect size needs **~145 systems; 123 exist.** *(Measured here slightly worse than
035's stated −0.933 / 7.7 / 13%.)*

**The honest form: at n=123 the residual encoder decay is not distinguishable from zero, and this
design cannot separate the two.** The encoder decay therefore remains **open** — neither confirmed as
a contributor to `FVE⊥` nor excluded. The verdict branch in `armf_atlas_modes.py` now prints that,
with the collinearity diagnostics beside it, so the reading cannot recur.

### ⬛⬛⬛ 28b/29b/30 **(all at n_train = 50)**: NO PEER WIN AT ANY N — and the tied N-collapse is a SCALE defect, removable zero-shot
**Reported first, per 30c.** Tied vs zero-shot ANM-256, matched capacity, one pass, same frames,
123 held-out systems:

| band | n | tied median | **ANM median** | gap | tied > ANM |
|---|---|---|---|---|---|
| Q1 | 31 | +0.2956 | **+0.6912** | −0.3702 | **0%** |
| Q2 | 30 | +0.2622 | **+0.7148** | −0.3657 | **0%** |
| Q3 | 31 | +0.2030 | **+0.6042** | −0.3518 | **0%** |
| Q4 | 31 | −0.2791 | **+0.6790** | −0.9424 | **0%** |

**No peer win, at any quartile, on any system.** The tied arm's 2.5–2.7× advantage over the control
is a ranking *among architectures that all lose to a zero-cost physics baseline*. That is the wording
the result licenses.

**29b/30b — the tied collapse is MAGNITUDE, not direction, and the joint reading is 30b's first row:**

| | Q1 | Q2 | Q3 | Q4 | `a*` slope vs N |
|---|---|---|---|---|---|
| tied `cos²` | +0.3319 | +0.3228 | +0.2270 | **+0.3326** | **−0.4431 ± 0.0587** (CI contains −0.5) |
| tied raw FVE | +0.2956 | +0.2622 | +0.2030 | **−0.2791** | |
| control `a*` slope | | | | | −0.0670 ± 0.0632 |
| untied `a*` slope | | | | | −0.0309 ± 0.0678 |

`cos²` is **flat** (Q1 +0.3319 → Q4 +0.3326) while raw FVE collapses — **+0.6117 of the Q4 loss is
scale alone.** With `a*` at −0.4431 (CI containing −0.5) and **specific to tied**, this is 30b's row
one: **pure over-scale, √N, mechanism confirmed.** The `‖B‖_F ~ √N` synthesis hypothesis survives its
own test — after the analysis-side version of the same story was refuted by 26a.

**30a — the ZERO-SHOT correction, and the oracle bound it is measured against.** `c(N) = √(N_ref/N)`
with `N_ref = 2460` frozen from the 50 training systems; it uses **nothing from the target**:

| tied | raw FVE | **N-only (REPORTABLE)** | oracle (UPPER BOUND, unachievable) | N-only recovers |
|---|---|---|---|---|
| Q1 | +0.2956 | **+0.2965** | +0.3319 | 3% |
| Q3 | +0.2030 | **+0.2202** | +0.2270 | 72% |
| **Q4** | **−0.2791** | **+0.2784** | +0.3326 | **91%** |

**The Q4 collapse is removed by a fixed function of N** — −0.2791 → +0.2784 — and tied becomes
**flat in N at ~+0.22 to +0.30**. The same correction makes control and untied *worse*, which is the
specificity check: a correction derived from a tied-specific mechanism helps only tied.

**BUT IT DOES NOT CLOSE THE PEER GAP.** Corrected tied is ~+0.25–0.30 against ANM-256 at **+0.60–0.71**.
Fixing the N-collapse leaves the architecture still losing to a zero-cost baseline at every N.

*(`fve_oracle_rescaled` is named that way at the point of computation, per 30a: `a*` is fitted against
the held-out target, no model achieves it, and it appears only to bound the reportable column.)*

### ⬛ CORRECTED (INBOX 28a): the claimed win compared tied's Q1 against the control's ALL-N median
My sentence *"the tied arm is the first architecture measured that beats the control anywhere"* set
tied's **Q1** median (+0.2956, 31 smallest) against the control's **overall** median (+0.1009, all
123). Two sides on different systems — Family F, on the project's first claimed win.

**Matched: same 123 systems, same quartile boundaries (1434 / 3249 / 7406), same frames.**

| arm | Q1 | Q2 | Q3 | Q4 | overall |
|---|---|---|---|---|---|
| **tied 3e-5** | **+0.2956** | +0.2622 | +0.2030 | **−0.2791** | +0.1990 |
| tied 1e-4 | +0.2670 | +0.2476 | +0.1876 | −0.3838 | +0.1876 |
| **control 3e-4** | **+0.1108** | +0.1037 | +0.0794 | **+0.1293** | +0.1009 |
| untied 3e-4 | +0.0875 | +0.0621 | +0.0567 | +0.0854 | +0.0662 |

**The win survives: Q1 vs Q1, tied +0.2956 against control +0.1108.** The bias was worth **0.010** on
a margin of 0.185.

**And 28a's premise does not hold for the control, which is worth more than the correction.** The
control does **not** degrade with N — its Q4 (+0.1293) is *above* its Q1 (+0.1108), and untied is flat
too (+0.0875 → +0.0854). **Only the tied arm degrades with N.** So the N-collapse is a property of the
tied analysis/synthesis pair, not of the L=1 design point in general — which the earlier framing
("performance DOES collapse as N increases") did not distinguish.

**INBOX 28c — correcting a power claim of mine.** I wrote "monotone decline crossing zero, at TWO
independent learning rates". Both tied arms are **seed 0** from the same job, so they share an
initialisation and are **not independent draws**. What carries the evidence is that the pattern is
threshold-free and monotone across four quartiles, not the number of arms. The word "independent" is
withdrawn.

**INBOX 28e — the tail travels with the claim, every time.** Tied's worst system is **FVE −8.045**
(a reconstruction nine times worse than predicting no motion) with **14%** of systems below −0.5.
Best-on-the-typical-system *and* catastrophic on a seventh is **a different operating point, not a
uniformly better model.** Median, mean and failure fraction are reported together from here, and the
choice of median is **pre-registered from now on** rather than presented as a discovery — it was
selected after it reversed the ranking, which is defensible for an unbounded-below quantity and must
be recorded as such.

**STILL A WIN OVER THE INTERNAL CONTROL, NOT THE PEER (INBOX 28b).** The peer is zero-shot ANM, and
tied-vs-ANM is unmeasured at every N. Job **10309145** computes both sides in one pass on the same
frames, ascending N so Q1 lands first. Until it reports, there is **no surviving peer-comparison win
in this project**.

### ⬛⬛ THE TIED ARM: BEST ON SMALL SYSTEMS, COLLAPSES ON LARGE ONES
**And the mean was hiding it.** Arms have been ranked all night by mean per-system FVE — an
**unbounded-below** quantity, so a handful of catastrophic systems dominates it:

| arm | mean | **median** | min | frac < −0.5 |
|---|---|---|---|---|
| control 3e-4 | **+0.1346** | +0.1009 | −0.376 | 0% |
| untied 3e-5 | +0.0996 | +0.0715 | −0.410 | 0% |
| **tied 3e-5** | +0.0607 | **+0.1990** | **−4.040** | **9%** |
| **tied 1e-4** | **−0.0798** | **+0.1876** | **−8.045** | **14%** |

**On the median the ranking REVERSES: tied (+0.199) > control (+0.101) > untied (+0.072).** The tied
variant has the best typical-system performance of anything measured and the worst tail.

**And the tail is entirely at large N** — FVE median by N quartile:

| arm | Q1 smallest | Q2 | Q3 | Q4 largest |
|---|---|---|---|---|
| tied 3e-5 | **+0.296** | +0.262 | +0.203 | **−0.279** (29% below −0.5) |
| tied 1e-4 | **+0.267** | +0.248 | +0.188 | **−0.384** (45% below −0.5) |

Monotone decline crossing zero, reproduced at **two independent learning rates**. **The −0.65/−0.75
N-slopes were not noise** — 24c's objection about single draws applied to the *slope*, and the
quartile medians are a stronger, threshold-free statement of the same thing (21b form: no free
constant, same direction across two rates).

**This is the central question's third clause answered for this architecture: performance DOES
collapse as N increases.** And it is the first architecture measured that beats the control anywhere.

**The mechanism is still unknown.** 26a refuted the analysis-side √N story — the tied encoder's
`‖z‖/‖disp‖` is nearly flat (−0.099). But the reconstruction is `B @ z`, and `B` is built per atom so
`‖B‖_F` grows like `√N`: with `‖z‖/‖disp‖` flat, `‖Bz‖/‖disp‖` would grow like `√N` and the **output**
would be over-scaled at large N. That is the *other half* of the analysis/synthesis pair, and it is
**being measured, not asserted** — job 10308313 adds `‖decoded‖/‖disp‖` vs N to the same harness.
Having had one mechanism refuted by exactly this test, the second gets the same treatment before it
is claimed.


Procedure-matched, same job, same seed, same data:

| variant | lr 3e-5 / 1e-4 / 3e-4 | N-slope |
|---|---|---|
| control | +0.1318 / +0.1250 / +0.1346 | −0.0303 / −0.0240 / −0.0383 (± ~0.06) |
| untied | +0.0996 / +0.0563 / +0.0974 | −0.0070 / +0.0022 / −0.0156 (± ~0.05) |
| **tied** | **+0.0607** (3e-5 only so far) | **−0.6539 ± 0.2480** |

**A mechanism is visible in the code, and it is specific enough to be checkable.** The tied encoder is
`z = einsum('bnic,bni->bc', B, disp) · coef_scale / N^0.5`. The `1/√N` normalises a sum with *random*
signs — but collective modes are **coherent**, so the analysis sum grows like *N*, leaving a residual
`√N` drift in the code magnitude. Across ATLAS's 56× range that is 7.5×, and the decoder is **linear
in z**, so it cannot absorb it. This predicts a negative N-slope in the **tied variant only** — and the
untied variant, which encodes by attention, shows none of it.

**NOT ACTED ON, and 24c is the reason.** One arm, one seed. A single draw cannot carry a conclusion
when the instrument's own scatter is comparable to the effect — and that applies to a *mechanism
story* at least as much as to a null, because a good explanation makes a noise result harder to
discard. If the remaining four tied arms all show a large negative slope it is structural; if only
this one does, it is noise wearing a plausible explanation.

**Confirmed in a trained model, not merely at init:** tied identity share 13% with `‖z0‖/‖z‖`
**exactly 0%**. The architectural guarantee that identity cannot occupy the code survives training.

### ⬛ REALISED RANK AND BASIS QUALITY ARE SEPARABLE (INBOX 23d) — a finding in its own right
Holding across the **whole LR sweep** of the modal `untied` arms, against a procedure-matched control
trained in the same job:

| | control (attention) | modal untied |
|---|---|---|
| participation ratio | ~15–20 / 256 | **43–69 / 256** |
| identity share | 64–92% | **24%** |
| held-out FVE | **+0.1346** | +0.0974 / +0.0996 |

**More latent capacity used, far less identity carried, consistently worse reconstruction.** Three
quantities, no chosen constants, same direction across an LR sweep — it passes 21b's threshold-free
test, and it holds independently of whether the modal form eventually wins.

**Two consequences.** (1) **Optimising realised rank does not deliver basis quality**; they are
separable, so a design that widens rank is not thereby better. (2) It **retires the 012b identity
share as a *cause* of the gap**: identity fell from ~90% to 24% and reconstruction got *worse*. The
wasted-capacity reading was reasonable and is now measured to be not the binding problem.

### ⬛ 17c: THE GAP IS 65% BASIS QUALITY, 35% MODE COUNT — SO 015 ADDRESSES THE SMALLER HALF
At matched rank **r = 6** (the codec's own realised rank, read from `atlas_modes.json`), same systems:

| quantity | FVE |
|---|---|
| codec (realises 6) | 0.1553 |
| **ANM-6** (zero-shot, matched rank) | **0.2888** |
| PCA-6 (oracle) | 0.3961 |
| PCA-16 (oracle) | 0.5274 |

- **basis quality**, `codec − PCA-6` = **−0.2409**; like-for-like `codec − ANM-6` = **−0.1336**
- **mode count**, `PCA-16 − PCA-6` = **+0.1312**
- Of the 0.3721 gap to PCA-16: **65% basis quality, 35% mode count.**

**ANM-6 beats the codec even at matched rank.** Holding the number of directions *fixed*, a
physics-derived zero-shot basis is better than the learned one. So the deficit is not mainly that the
codec realises too few directions — it is that **the directions it realises are worse ones**.

**Consequence for INBOX 015:** the modal decoder widens the realised rank, which attacks the **35%**
half. That is precisely the risk flagged in the 017 ACK before any of this was measured, and 18c's
decomposition is what turned the caution into a number. *(Per 20c the split is an ATTRIBUTION: the
extra directions are priced at oracle quality, an upper bound on the mode-count term and hence a lower
bound on basis quality — so 65% is if anything conservative.)*

**And the first modal arms are consistent with it.** `untied` reaches PR **69.1/256** and identity
**24%** (control: PR ~15–20, identity 64–92%) with **56 effective basis modes** — it demonstrably uses
far more of the latent and carries far less identity — yet scores FVE **+0.0996 / +0.0563** against the
control's **+0.1346**. More directions, worse ones. Note that **ANM's basis is inherently nonlocal**
(a Hessian coupling neighbours within 5 Å) while `ModalCodec` at `ctx_layers=0` builds `B_i` from atom
*i* alone — which is exactly the escalation 015/17b pre-registered, now with evidence behind it rather
than as a fallback.

### INBOX 14a: THE PRIMARY COMPARISON WAS MISSING FROM EVERY ATLAS REPORT
The standing frame since 004b is that the primary result is **codec vs ANM, both zero-shot on the
same held-out frames** — ceiling-free, immune to the failed guard, and the claim the thesis rests on.
Every ATLAS report so far has given **codec FVE alone**.

**The diagnosis is more specific than an oversight.** `armf_atlas_curve.py` has had the
codec/ANM/oracle columns all along, but it is pinned to `L = 24; DM = 256; KS = [24]` — the
configuration INBOX 003 retired when L=1 became the architecture — and it TRAINS its own learning
curve, so it needs the GPU the DM sweep holds. It has never produced an `atlas_curve.json`. The
comparison was not forgotten; it was **blocked behind a stale script**, and the reportable column got
reported instead of the load-bearing one. *A measurement that exists only in unrun code is not a
measurement.*

**The fix needs no GPU.** `atlas_dm.json` already stores per-system held-out FVE for every finished
arm, so `armf_atlas_peer.py` computes the peer and oracle columns on CPU and joins them to arms
already paid for (job 10306938).

Two properties of the comparison recorded before results, both cutting against us:
- **Matched capacity is not matched information.** At k=DM both emit the same numbers per frame, but
  ANM-k derives a **system-specific** basis from that system's own structure while the codec's DM
  numbers come from **one model shared across every system**. The peer is the more favoured of the
  two at matched k. It is still the right bar — it is what a practitioner does without training
  anything — but that asymmetry belongs beside the number.
- **The PCA column is an ORACLE and never a bar** (3N×k free parameters fitted to the target's own
  trajectory — 1.6M at k=16, N=33,377), reported only as a fraction. Two versions are computed: the
  honest `pca_out` (basis from replicas 0+1, scored on replica 2) and the in-sample `pca_in` that
  earlier numbers used. On the smallest held-out system the in-sample version is **1.68× flattered at
  k=4** — the same in-sample inflation retracted for rank90, arriving through a different door.

**The context this puts in front of every FVE-vs-N statement:** at codec ≈ 0.155 against a per-system
PCA-16 of ≈ 0.53, **a flat slope on a model this far from the achievable is consistent with uniform
weakness, not with the architecture holding up.** INBOX 14b (job 10306939) is the direct test of which
one it is.

### ⚠ THE SECTION BELOW IS WITHDRAWN — the control does not support it (job 10307026)
**Withdrawn claim:** *"the procedure change moved the LR optimum."* It compared **legacy rows against
current rows**, and the legacy rows are now known to be **unreproducible by any configuration
available today** (see below), so the comparison was never between two procedures — it was between a
current procedure and an unrecoverable past state.

**What the control actually measured**, at matched LR (3e-4), matched code, matched data, 2 seeds each:

| metric | LEGACY | CURRENT | gap | seed spread | verdict |
|---|---|---|---|---|---|
| `best_track` (peak of tracking curve, 24 systems) | +0.0973 | +0.0656 | 0.0317 | 0.0282 | separated, **13% margin** |
| **`full_fve`** (final weights, all 123 — **what the sweep selects on**) | **+0.1217** | **+0.1160** | **0.0057** | **0.0269** | **NOT SEPARATED** — gap is 21% of seed noise |

**Conclusion: the procedure changes the SHAPE of the training curve, and is NOT shown to change final
held-out quality.** There is no evidence to prefer LEGACY. **Keep the current procedure** — it
additionally closes the Family D hole (3e-5 arms VOID at a flat cap) that warmup and the lr-scaled
budget were introduced for. The `HYBRID` arm was designed for the case where LEGACY won; it is **not
run**, saving ~1.5 GPU-hours.

> **FAMILY G, THIRD INSTANCE — and this one is in the control written to settle the question.**
> The script's own printed verdict says *"THE PROCEDURE CHANGE IS THE CAUSE"* on a `gap > spread` rule
> **I chose**, evaluated on a metric **I chose** (`best_track`), at a 13% margin. On the metric the
> sweep actually selects winners with, there is no separation at all. Deciding on `best_track` would
> have optimised a diagnostic instead of the reported quantity — and would have justified retraining
> the entire grid to chase a difference that does not exist in the number anyone reports.
> *Check added:* an automated verdict must name **which metric it is deciding on** and why that metric
> is the decision-relevant one — not merely which threshold it crossed.

### WHAT DOES STAND: THE PERSISTED ARMS ARE NOT REPRODUCIBLE, SO THEY MUST BE DISCARDED
Independent of the procedure question. Control LEGACY seed1 gives all-123 **+0.1352**, stopping at
20,000 steps; the persisted seed1 recorded **+0.1553** at 15,000 — under hyperparameters matched
deliberately (warmup off, flat 30,000 cap). Three plausible causes were checked and **excluded**:
model-init RNG (`down`/`up` are `None` at `dlat == dm`, so initialisation is bit-identical),
tracked-set stratification (N-stratified `HOt` predates those arms), and training-set membership (the
cache fills `heldout + train_ordered` in manifest order, so `tr_ids[:50]` was the same 50 systems at
271 cached as at 697). Rows 6/9/10 carry no `z0_frac`, dating them before commit `a70aaa2b`.
**The residual cause is unidentified and the conclusion does not depend on it:** arms that cannot be
regenerated cannot be compared against arms that can, so the 13 legacy rows are retired.

**Determinism itself is confirmed** — CURRENT seed1's `best_track` (+0.0515) reproduces the 14b re-run
exactly, and the modal job's control arms reproduce `atlas_dm` rows to four decimals (+0.1318,
+0.1250). Within a fixed procedure *and* codebase, training is deterministic.

<details><summary>WITHDRAWN — original section retained for the record</summary>

### THE PROCEDURE CHANGE MOVED THE OPTIMAL LEARNING RATE — WHICH IS WHY MIXING IS FATAL, NOT UNTIDY
`atlas_dm.json` at the time of the INBOX 16c fix held 16 rows. Provenance is unambiguous: rows 0–12
are the legacy 13 (flat 30,000-step cap), rows 13–15 come from job 10306831 under the current
procedure — row 13 ran **47,500 steps**, which is only reachable through `maxsteps_for`.

| arm | procedure | best LR found | all-123 FVE |
|---|---|---|---|
| n50 DM=256 | **legacy** | 3e-4 | **+0.1453 / +0.1553 / +0.1497** (3 seeds) |
| n50 DM=256 | **current** | 3e-5 | **+0.1318** |
| n50 DM=256 @ 3e-4 | current | — | tracked **+0.0515** vs legacy's tracked +0.0972 |

**The procedure change did not simply shift performance — it moved where the LR optimum sits.** Under
the legacy procedure DM=256 peaked at lr=3e-4 and *collapsed* at 1e-3/3e-3 (−0.0001, −0.0004). Under
the current procedure the same width does well at **3e-5**, a rate the legacy grid could not even
reach, while 3e-4 falls to roughly half its legacy tracked value.

**So a table mixing the two does not compare two noisy estimates of one quantity — it compares
different points on different LR curves.** The sweep picks "best LR per DM" by maximising over rows;
with legacy rows supplying 3e-4/1e-3/3e-3 and current rows supplying 3e-5/1e-4, the winner is
selected across a procedure boundary. That is Family E in its most damaging form, because the
selected LR then propagates to every higher rung of the ladder.

**Consequence for the fix:** "standardise the procedure" is not merely bookkeeping — **the entire LR
grid must be re-swept under whichever procedure is kept**, since a winner inherited across the
boundary is meaningless. Job 10307026 decides which procedure that should be at matched LR; note that
if LEGACY wins, the correct action is *not* the retrain direction assumed so far — it would be to
**revert or condition the warmup**, which was introduced only to rescue the 3e-5 grid edge and may
have cost the main operating point to do it.

</details>

### INBOX 16a RESULT: THE DECODER'S OUTPUT SPANS ~6 DIRECTIONS. THE FUNCTION CLASS IS WHAT BINDS.
Measured on the best available arm, 24 N-stratified held-out systems, 2,000 consecutive frames:

| quantity | median | range |
|---|---|---|
| **realised rank90 of the RECONSTRUCTION** | **6** | 2–10 |
| realised rank99 of the reconstruction | 23 | 7–39 |
| **rank90 of the DATA on the same frames** | **152** | 22–690 |
| latent participation ratio | 14.9 | — |
| DM | 256 | — |

**The reconstruction spans 4% of the directions the motion actually uses.** And the output rank (6)
is **2.5× below the latent's own effective rank** (14.9) — the code carries roughly fifteen usable
dimensions of conformational variance and the decoder converts about six of them into output modes.

> ### ⬛ HEADLINE FORM (INBOX 21a/21b): **the decoder emits under 4% of the directions the motion uses — 6 of 152.**
> That is the claim to quote. It involves **no free constant**, does not involve PR at all, and is
> therefore untouched by the threshold that flips. **"Decoder-limited" is RETIRED as a headline
> phrase**; it may appear only as a label, with its range attached (it holds above 0.40·PR).
>
> **POLICY, generalised from the audit:** a headline claim must be a **quantity with no free
> constant** — a ratio, a slope with a CI, a fraction of systems. **A label produced by comparing a
> quantity to a chosen threshold is not a claim**; it is a reading, and it appears only beside the
> quantity and the range over which it holds. This rule would have prevented all three Family G
> instances *at the point of writing* rather than at the point of audit.
>
> **INBOX 20b — THE LABEL DEPENDS ON A THRESHOLD; THE RATIOS DO NOT. QUOTE THE RATIOS.**
> Running `verdict_sensitivity()` over this claim (the first time it was pointed at a *surviving*
> result rather than a suspected one) shows the **"decoder-limited" LABEL flips**: it holds for any
> threshold above **0.40·PR** and fails below it, because the measured ratio *is* 0.40. So the label
> is only defined together with its constant and must never be cited without it.
> **The underlying ratios carry no free constant and are the form to quote:**
>
> | ratio | value | |
> |---|---|---|
> | realised rank90 / **latent PR** | **0.40** | 6 of 14.9 directions |
> | realised rank90 / **DATA rank90** | **0.039** | 6 of 152 directions |
>
> The data-rank ratio is the stronger of the two **and does not involve PR at all**, so the substance
> of 16a does not rest on the threshold that flips. *(One qualification in the other direction: the
> audit also lists `rank99` as flipping, but `0.6·PR` was calibrated for the rank90 convention, so
> that row is a MIS-SPECIFIED COMPARISON rather than a genuine free choice, and is marked as such
> rather than counted as a flip.)*
> **This also corrects my own earlier statement** that 16a "survives its own free choices" — that was
> true across 0.75×–1.5× on one metric, and is not true across 0.5×–2×. The narrower test was the one
> I happened to run.

**This selects 016's first reading: the binding constraint is the DECODER'S FUNCTION CLASS — not
capacity, not data, not DM.** It is not the latent (the latent already holds more than the decoder
emits) and not the width (DM=256 against six realised directions). Consequences, per 016d: further DM
and LR arms have low marginal value, and `INBOX 015`'s modal decoder — which makes DM dimensions into
DM modes *by construction* — is the direct test of exactly this diagnosis.

**Corroborated by the injection pattern (16b).** Per-mode error against the predict-zero baseline is
0.757, 0.790, 0.886, 0.863, 0.876, 0.933 on modes 1–6 and crosses **above 1.0 from mode 7** (1.015,
1.151 at 15, 1.175 at 30) — **the decoder injects error into 70% of the 30 resolved modes.** Under
MSE that is the correct move for a restricted function class: push error where it is cheap to buy fit
on the modes that dominate the loss. It is the signature, not a bug.

> **CORRECTION, recorded because it changed a printed verdict.** The first run of this measurement
> printed *"realised rank ~ PR, the latent is the limit"* — the opposite reading. The branch tested
> `med <= max(3.0, 2·PR/5)`, an invented constant that evaluated to **5.96 against a measured median
> of 6.0**, so an arbitrary threshold flipped the conclusion on a margin of 0.04, at rounding scale,
> when the measured ratio is 6/14.9 = 0.40. The test is now that ratio directly (`med < 0.6·PR`) with
> no free constant. **The numbers were always right; the automated reading of them was not** — which
> is why the verdict text is never quoted here without the table above it.

### INBOX 14b RESULT: VARIANCE-SELECTIVE, NOT TIMESCALE-SELECTIVE — AND THE ABSOLUTE SCALE LEADS
**Lead with this, always:** reconstruction RMSD **2.412 Å** (range 1.677–7.740) against a
displacement RMS of **2.572 Å** (range 1.673–8.159). **The residual is 94% of the motion's own
amplitude.** Every FVE ratio below is a ratio on top of that fact.

The naive timescale reading is strong and would have been shipped:
`SLOW − FAST = +0.4131 ± 0.1764` (passes the pre-registered >0.10 with a CI excluding zero) and
`FVE ~ log(IAT)` coefficient `+0.4700 ± 0.1917`. That is the "codec keeps slow collective motion,
discards thermal noise, so MSE is the wrong objective" story.

**It does not survive the control.** With variance share held fixed, the partial `log(IAT)`
coefficient is **+0.1445 ± 0.1830 — the CI spans zero.** Slow modes *are* the high-variance modes and
MSE selects for variance by construction, so the apparent timescale preference is the training
objective doing exactly what it says. **VERDICT: variance-selective, not timescale-selective.**
*The naive numbers must never appear without the partial beside them.*

Per-mode FVE: +0.243 (mode 1, 45.1% of variance), +0.210, +0.114, +0.137, +0.124, +0.067, then
**negative from ~mode 7** (−0.015, −0.151 at 15, −0.175 at 30). Per **INBOX 16b** that is EXPECTED,
not a bug: under MSE a capacity-limited model optimally pushes error into low-variance modes to buy
fit on the ones that dominate the loss. It is the signature of a restricted function class.
Timescale bins are flat in N (SLOW−FAST vs log10(N): −0.1167 ± 0.3879).

**Caveat on the arm:** measured on a re-trained model that did NOT reproduce the recorded arm
(+0.0515 vs +0.0972 best-tracked). At 94% residual the "uniformly weak" reading is not delicate, but
the mode-resolved numbers should be re-measured once the procedure question settles.

### RETRACTED (INBOX 012): the mdCATH DM=256/512 COLLAPSE
`armf_capacity_axis.md` recorded DM=256 and DM=512 collapsing to constant output on mdCATH (G1 cos
1.0000, G4 base -0.000) and this was cited as evidence that **wide codes cannot train**. **Withdrawn.**
On ATLAS at DM=256, lr=1e-3 and lr=3e-3 both collapse to a constant code while **lr=3e-4 gives the
best arm in the sweep (+0.1553)**, and the optimal LR falls monotonically with width. Those arms were
losing on an **unswept learning rate (FAMILY E)**, not on capacity. The diagnosis on record --
"21 training domains against a 512-wide decoder" -- was plausible and wrong; corpus size may still
matter, but it was never shown to, because the hyperparameter was never swept.

### FAMILY D, NEW INSTANCE: an instrument pointed at the wrong object returns a FLATTERING number
Two cases in one week, and the shape is identical -- the measurement does not break, it quietly
describes something other than what it names, so nothing looks wrong:
- **rank90 measured in-sample.** Its own modes cover 77.1% of held-out variance, not 90%.
- **`encode()` returning the d_model representation instead of the DM_latent code.** Had the 005
  bottleneck been bolted on without routing every consumer through one `code()` method, the
  participation ratio and criterion-4 would have described a healthy 512-wide activation while the
  propagator's actual input was 16 numbers.
- **And the executable instance (INBOX 012c):** `armf_phase1_analyze.py` PRINTED the retracted
  capacity exclusion **at runtime** -- "rank90 and TICA already EXCLUDE capacity... DO NOT call this a
  capacity limit" -- so a future reader would have been instructed, by a running program, not to
  consider the hypothesis that is now live. **A claim corrected centrally but left standing locally
  is not corrected.** That is the argument for Q4 marking claims where the reader meets them.

### THE FAMILY-E REPAIR OPENED A FAMILY-D HOLE, AND THE FLAG CAUGHT IT
Widening the LR floor to 3e-5 to rescue DM=512 was correct in direction and incomplete in execution:
the very first 3e-5 arm came back **`STILL IMPROVING -> VOID`** at MAXSTEPS=30,000. A grid extension
whose new arms cannot converge buys nothing **at exactly the width it was meant to rescue** -- so
DM=512 would still have had no valid arm, and the top of the sweep would still be unmeasured. The
VOID flag (control 3) is what made this visible rather than silently producing a flat curve.

**Fix is WARMUP, not brute force.** Quadrupling the step budget costs 2+ hours per wide arm.
1,000-step linear warmup addresses the actual collapse mechanism -- large early updates
destabilising a wide FiLM decoder before the latent has any structure -- so a wide arm can train at
an LR that would otherwise collapse it and the grid does not have to reach as low. A square-root,
capped step-budget scale is applied on top (3e-5 and 1e-4 get 90,000 steps; 3e-4 gets 54,772; 1e-3
and above keep 30,000), so low-LR arms are not judged on a budget built for high-LR ones.

Not applied to the running job (10306611), which is 40 minutes into the full ladder and producing
valid arms at the good LRs; it takes effect on the next re-run, and its VOID arms remain correctly
flagged rather than quietly counted.

## DECLINED HYPOTHESIS (INBOX 011, closed on measurement): rank90's growth is NOT an ordering artifact
Recorded with numbers attached so it is not re-proposed. The hypothesis was that part of rank90's
near-linear out-of-sample growth is the train PCA basis **ordering** worse at large N rather than
held-out content being higher-dimensional -- which, if true, would partially rehabilitate the physics
argument for a fixed latent width, since a learned structure-conditioned decoder is not locked to a
fixed component order.

| quantity | measured (n=123 held-out ATLAS systems) |
|---|---|
| ORDERING PENALTY (ordered / ordering-free) | **1.03x** |
| its N-slope | **-0.0145 +/- 0.0195** (CI spans zero -- the basis does NOT order worse at large N) |
| exponent, TRAIN ORDER | +0.9285 +/- 0.2220 |
| exponent, ORDERING-FREE (cross-fit) | **+0.9429 +/- 0.2306** -- *higher*, not lower |
| selection inflation (naive / cross-fit) | 1.02x |

**Declined.** The growth is real dimensionality. `b >= 0.93` holds on both readings, and the
fixed-width physics argument stays retired.

## 007 IS CAPPED: TICA IS PROBABLY NOT A VIABLE INSTRUMENT ON THIS CORPUS (INBOX 13a)
The truncation-matched control answers a **RATIO** question and only that: *within a fixed
m-dimensional basis, does slow-weighted dimensionality grow more slowly than variance-weighted?*
That is well posed even though neither absolute exponent is, because truncation compresses both
equally and cancels in the comparison. **No absolute TICA exponent may be quoted.**

**PRE-REGISTERED STOPPING RULE, recorded before the re-run lands:** if n_eff per TICA dimension is
below ~2, **007 closes as UNANSWERABLE**. n_eff per TICA dimension measured **0.81** on the first
run (below 1.0 in 103/123 systems), so this is likely to fire. **m will NOT be swept further looking
for a basis where it works** -- the exponent already moves 13x with m, so any m chosen after seeing
results is a chosen result. "Not measurable at these trajectory lengths" is worth more than a number
extracted from a basis picked to produce one, and costs a paragraph instead of a week.

## THE MEASUREMENT THAT WAS MISSING: CRITERION-1 PASS RATE vs N (INBOX 13b)
Both external routes to "does the required content grow with N" are now closed or capped, so the
question is settled by the codec's own behaviour -- where it should have been. Two codec measurements
matter and **only one was specified**: FVE-vs-N was running; **criterion-1-vs-N was specified
nowhere.**

> **Does passing a trajectory through ONE fixed-width token destroy dynamics MORE at 33,377 atoms
> than at 598?**

That is the objective-1 question, and criterion 1 outranks FVE (004b). **It can fail exactly where
FVE-vs-N is flat**: a decoder can hold reconstruction error constant across N while progressively
FLATTENING autocorrelation at large N -- the failure the shuffled-frames control was built to detect,
invisible to FVE, and fatal to stage 2 at precisely the sizes objective 1 cares about. I built and
validated that harness under Q3 and then reported FVE-vs-N as the headline without ever asking what
it says as a function of N.

Now wired at fixed DM on the held-out systems: **four discriminators reported SEPARATELY** (never
averaged) with per-discriminator N-slopes and CIs, on **consecutive** frames because the kinetic
discriminator is meaningless on a strided sample.
**If criterion-1 pass rates are flat in N AND FVE is flat in N, that is far stronger than either
alone -- and the first version of the headline claim that would survive scrutiny.**
