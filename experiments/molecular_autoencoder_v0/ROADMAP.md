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

| outcome | observation | localisation | fix |
|---|---|---|---|
| **A** | N effect at L=12/24 but NOT at L=1 | slot assignment / routing | addressing mechanism |
| **B** | N effect at L=1 as well (rank90/TICA flat in N) | **POOLING / BROADCAST PATHWAY** -- encoder aggregating N tokens into a fixed code, or decoder broadcasting one code back to N atoms. **NOT capacity.** | aggregation architecture: hierarchical pooling, deeper cross-attention, relative-position conditioning |
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

### WIDTH CHAIN, auditable (anchor: mdCATH asymptote 168 at medN 1,804; floppy->bound divide by 0.65)
| b | N^b to 1e6 | dims for a 1e6-atom BOUND complex |
|---|---|---|
| 0.101 (raw, censored -> LOWER bound) | 1.89 | **489** |
| 0.139 (x1.38 c-derived, provisional) | 2.41 | **622** |
| 0.20 | 3.54 | 914 |
| 0.30 | 6.65 | 1720 |

**Design sensitivity:** at b ~ 0.10-0.14, DM ~ 500-620 covers a 1M-atom bound complex and DM=512 sits
right at the edge. At b >= 0.20 it takes ~900+, and at b ~ 0.30 the single-global-latent claim needs
qualification rather than just a wider DM. The conclusion is sensitive to b over exactly the range
censoring could plausibly move it, and b is currently known only as a lower bound.

**CAVEATS ON THE SAME LINE:** 2.5-decade extrapolation (1,804 -> 1e6 atoms); ns-us regime only;
variance-weighted so rare states are excluded (the coverage question is open, not answered); floppy
-> bound factor 0.65 derived from a 1.65x RMSF ratio via c ~ -0.9; and **b itself is a censored
lower bound.**

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
