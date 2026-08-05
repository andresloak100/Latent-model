# Reactive-corpus PILOT (objective 2) -- pipeline test, NOT data generation

**LEVEL OF THEORY: GFN2-xTB (semi-empirical). This is a PIPELINE/format/cost test. NONE of its
trajectories are corpus** -- xtb barriers are not accurate, and 8.2 established barrier accuracy
is the objective-2 metric.

## Deliverables

**(a) Validated pipeline (end-to-end).** xtb 6.6.1 static binary (fetched to $SCRATCH, no pip);
reaction = H2O2 O-O bond dissociation via a constrained bond scan (1.45->3.0 A, 20 steps, each
geometry-optimised). Runs clean: single-point 0.11 s, full 20-step dissociation scan 0.28 s,
peak 23 MB, 20 structures along the reaction coordinate in xtbscan.log.

**(b) Event/format spec (discovered by needing it).** A reaction-event record must carry:
  - per-frame coordinates + total energy + gradients (forces) -- forces are needed for any
    force-matching consumer and are free from the QM call;
  - the reaction coordinate: the breaking bond as an ATOM-INDEX PAIR (persistent t=0 indices);
  - the topology-change TIMESTAMP: the frame where the bond crosses a break threshold
    (e.g. > 1.5x its equilibrium length) -- one integer per event;
  - the IDENTITY CONTRACT across the break: atoms keep persistent t=0 indices (element unchanged);
    the BOND LIST changes at the timestamp (edge removed/added); this is exactly the "graph
    features demote from identity key to time-varying conditioning" contract recorded in ROADMAP
    8.5 -- the pilot makes it concrete: identity = (element, persistent index), topology = a
    time-stamped edge-list diff;
  - metadata: level of theory, charge, spin, method version.

**(c) xtb cost/trajectory:** 0.28 s for a 4-atom, 20-step scan. Realistic reactive-site QM region
(~50-100 atoms): xtb single-point ~seconds; a reaction path (~50-200 opt frames) ~minutes.

**(d) xtb->DFT scaling factor:** NO DFT code is installed (no psi4/orca/cp2k/gaussian module; venv
lacks pip). Could not TIME a DFT single-point -- flagged as needing a module request or a
$SCRATCH install before the campaign decision. ESTIMATE from complexity + literature: GFN2-xTB is
~1e2-1e3x faster than hybrid DFT (e.g. B3LYP/def2-TZVP) for drug-to-QM-region sizes, and the gap
GROWS with system size (DFT O(N^3-N^4) with a large prefactor vs xTB O(N^2-N^3)). Use 1e2-1e3x,
ASSUMPTION STATED.

**(e) Implied DFT campaign cost (extrapolation, assumptions explicit):** corpus for objective 2
needs reaction DIVERSITY -- assume ~1e3-1e4 reaction paths, ~50-100-atom QM regions, DFT geometry
opts (~10-50 SCF/frame, ~50-200 frames/path). Per-path DFT ~hours-days; corpus ~1e4-1e6 CPU-hours
= ~1-100 CPU-YEARS depending on scope. That is the "months of DFT" quantified. RESOURCE NOTE:
the pilot fit in the margins (xtb, CPU, sub-second); the full campaign is a serious sustained-DFT
allocation that needs Andres's own resources or an explicit arrangement -- NOT fundable silently
on a borrowed account. Raise at the campaign go/no-go.

## DECISION BLOCK for Andres -- reactive campaign go/no-go (figures, not a recommendation)

**Measured (GFN2-xTB, this pilot):** single-point 0.11 s / 4-atom H2O2; 20-step bond-dissociation
scan 0.28 s / 4 atoms, 23 MB.

**Extrapolation to one realistic reactive trajectory (assumptions explicit):**
- system: ~50 heavy-atom reactive site / QM region (ASSUMPTION);
- path: ~100-300 optimised frames per reactive event, ~10-30 SCF/frame (ASSUMPTION);
- xtb scaling O(N^2-N^3): xtb single-point ~0.5-1 s at 50 atoms -> **xtb ~1-10 min / trajectory**.

**xtb -> DFT scaling factor: ESTIMATED ~1e2-1e3x** (hybrid DFT e.g. B3LYP/def2-TZVP vs GFN2-xTB;
grows with system size). **NOT measured** -- no DFT code installed (no psi4/orca/cp2k module,
venv has no pip). Timing a real DFT single-point needs a module request or a $SCRATCH install; that
is the one missing figure and it is a prerequisite, flagged.

**Implied DFT cost / trajectory:** (1-10 min) x (1e2-1e3) = **~2 h to ~1 week per trajectory**.

**Implied corpus cost (size per ROADMAP 8.4 -- enzyme / condensed-phase active-site events,
a GENERATION problem, ~1e3-1e4 reactive trajectories):**
- at ~2 h/traj (optimistic DFT, small system): 1e3-1e4 traj -> **~2e3-2e4 CPU-hours (~0.2-2 CPU-yr)**;
- at ~1 week/traj (large system / dense path): -> **~1e5-1e6 CPU-hours (~10-100 CPU-yr)**.

**Assumptions behind the extrapolation:** QM region size (50 atoms), frames/event (100-300),
SCF/frame (10-30), xtb->DFT factor (1e2-1e3, unmeasured), corpus size (1e3-1e4 trajectories,
per 8.4's "generation problem" framing since no public enzyme-reactive corpus exists). QM/MM (not
pure DFT) would change the per-frame cost. **No further reactive work until Andres decides.**
