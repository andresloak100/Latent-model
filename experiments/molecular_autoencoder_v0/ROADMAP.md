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
   stability.
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
row is struck: 54 modes x 231 systems = 12,474 >> 1024, so a single fixed latent
cannot carry the additive dynamic state. It holds the box's **global** state
only; the per-molecule tokens carry the additive part. (Token count: 125,000
per-residue vs 924 per-molecule at 4 tok/mol = 135x fewer.)

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
  have additive modes: 54 x 231 systems (1M atoms) = 12,474, i.e. 12.2x over a
  1024-token global latent -- which is why the budget scales with molecule count.
  The additive 54 x 231 is the **conservative worst case**: physical coupling
  between molecules in one box can only *reduce* the mode count (shared collective
  modes), never raise it above independent-additive. Direct test pending, and the
  naive version is confounded: `dev_modes_90` is a variance-fraction count that is
  sub-additive across independent blocks even with ZERO sharing (provably-
  independent synthetic systems give box/sum ~0.85-0.88, lower for concentrated
  spectra), so "approach K x 54" is the wrong additive prediction -- the
  concatenation value already IS the independent prediction. And concatenating
  independent MISATO simulations cannot exhibit physical sharing at all (the
  molecules are not in one box). A real test needs genuine multi-solute
  trajectories, compared against the independent-concatenation baseline of the
  same molecules, using an additivity-preserving effective-dimension metric
  (participation ratio (Sum L)^2 / Sum L^2), not a 90%-variance count.

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
