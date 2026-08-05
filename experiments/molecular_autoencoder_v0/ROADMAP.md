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

## 5. What is actually established

Worth stating alongside the gaps, since the list above is all deficits:

- The direct per-residue codec reconstructs held-out protein structures at
  **0.79 Å all-atom / 0.51 Å backbone**, chirality 0.0002, contact F1 0.963.
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
