# Static structure autoencoder — scaling harness

Read `BRIEF.md` first. It is the task; this file is how to run it.

**Nothing here knows what a molecule is.** No equivariance, no internal coordinates, no bond or
angle or clash terms, no residue concepts. Tokens are atoms. The loss is coordinate MSE. That is
deliberate and it is the point of the study — see `BRIEF.md`.

## Layout

    BRIEF.md        the task, the sweep, the metrics, the pre-registered readings
    sae/model.py    generic transformer autoencoder; latent size and parameter count as
                    independent knobs, with `config_for_params` solving width for a target
    sae/data.py     SyntheticStructures (KNOWN intrinsic dimension) + the RealCorpus interface
    sae/metrics.py  generic autoencoder metrics only
    sae/scaling.py  fit L = A*N^(-alpha) + L_inf, with intervals and a degeneracy test
    sae/train.py    one run; returns every metric plus a convergence status
    sweep.py        the grid driver
    tests/          16 tests, each pinning a defect the self-test actually found

## Run the harness self-test first

    python sweep.py --synthetic --intrinsic-dim 8 --n-atoms 24 --structures 2048 \
        --params 300000 --latents 4 8 16 32 --steps 6000

The synthetic set has **exactly** `intrinsic-dim` degrees of freedom, so the curve must flatten
at or before that latent size. A sweep that cannot recover a known answer is not evidence about
an unknown one.

Measured on this harness at 300k parameters, latent 16, k=8:

    steps    held-out MSE    effective rank / 16
      400         0.11546                   1.37
     2000         0.03265                   7.34
     6000         0.00650                   9.71

## Then the real grid

Implement `sae.data.RealCorpus` where the corpus lives, honouring the four requirements in its
docstring — **the split must be by sequence homology, not at random** — then:

    python sweep.py --out results.json

## Two guards that matter more than they look

**A cell is scored only if it converged.** An under-trained cell is flat, and a flat row reads as
"saturates in latent size" — which is a pre-registered verdict, and the wrong one. At 400 steps
every latent size from 4 to 32 returned 0.1154; the same cells at 6000 steps separate cleanly.
Worse, a model that never learns is *also* flat, so convergence needs two conditions and the
status is one of `converged` / `still_improving` / `stuck`. Only the first enters the fit, and
the other two are counted in the results file.

**`L_inf` is not always identifiable.** When `alpha` is near zero the power-law term is constant
and `A` and `L_inf` trade off freely — the fit will put the whole level into `A` and report
`L_inf = 0`. On a self-test with truth `L_inf = 0.40` it did exactly that. `fit_scaling` now
tests this directly, sets `degenerate`, and reports `L_extrap_10x` instead, which is identifiable
either way. **If `degenerate` is set, do not quote `L_inf`.**

## What the headline is

`alpha` and `L_inf`, with intervals, from `sweep.py`'s `reading`. `alpha` says how fast scale
helps. `L_inf` says whether scale arrives anywhere useful. The four readings are pre-registered
in `BRIEF.md` and include "flat on both axes", which is a real result and should be reported as
one rather than treated as a failed run.
