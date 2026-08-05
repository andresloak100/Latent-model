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
