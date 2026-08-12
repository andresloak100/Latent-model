# Static structure autoencoder — scaling study

**Self-contained brief.** Everything needed is below. Do not import assumptions, metrics,
baselines or conclusions from any other directory in this repository. If something here is
underspecified, choose the most standard option and record the choice.

---

## The task

Train an autoencoder on static protein structures and measure **how reconstruction quality scales
with model size and latent size.**

That is the whole deliverable. Not a new architecture, not a benchmark win — two scaling curves and
the numbers that describe them.

## The architecture constraint, and it is a hard rule

**Use a generic architecture. No domain-specific structure of any kind.**

- a plain transformer over tokens, learned positional encoding, standard normalisation and
  residual connections;
- **no** equivariance or invariance built into the architecture;
- **no** hand-designed geometric features, frames, internal coordinates, or physics terms;
- **no** chemistry-specific terms in the loss;
- **no** chemistry-specific terms in the evaluation.

The study measures how a *general* method scales. Anything domain-specific confounds exactly the
quantity being measured, so it is out of scope even where it would improve the number. If you
believe a domain prior is necessary to make the model train at all, report that as a finding rather
than adding it.

Reference for why this constraint exists: R. Sutton, *The Bitter Lesson* (2019),
http://www.incompleteideas.net/IncIdeas/BitterLesson.html — general methods that scale with
computation tend to outperform methods built on human domain knowledge, and the domain knowledge
usually turns out to be the thing that caps the ceiling.

## Data

- Prepared structure corpus: `data/processed*/` (see the repository's data README for the exact
  path and record layout). Roughly 10^6 structures.
- **Held out by sequence homology, not at random.** A random split leaks: near-duplicate structures
  land on both sides and reconstruction error reads far better than it is. Use a sequence-identity
  threshold and state it.
- Report the number of structures in train and held-out, and the identity threshold used.

## The sweep

**Primary axes:**

1. **parameter count** — `{0.5M, 2M, 8M, 32M}`, varied by width and depth together
2. **latent size** — total latent dimension per structure, `{32, 64, 128, 256, 512, 1024}`

Full grid where affordable; if not, a staircase along the diagonal plus the two edges, and say which
cells ran.

**Secondary axis, run at one parameter count:**

3. **tokenisation granularity** — how the structure is cut into tokens (per residue, per 2 residues,
   per 4 residues). Latent capacity can be bought either by widening the per-token vector or by
   having more tokens, and these are not the same trade. Sweeping the total latent alone cannot
   separate them.

4. **latent regularisation** — none vs a small KL or variance penalty, at one parameter count and
   one latent size. Standard autoencoder practice and it changes the latent's shape substantially.

## Metrics — generic only

**Reconstruction**
- mean squared error per coordinate, train and held-out
- the held-out curve against latent size (the rate–distortion curve)
- the held-out curve against parameter count
- train / held-out gap at every point

**Latent quality**
- **effective rank of the latent covariance against the nominal latent size** — does the model use
  the latent it is given? A nominal 512 collapsing to an effective 30 is the single most useful
  thing this study can find, and it is invisible unless plotted.
- fraction of latent dimensions carrying variance above a stated threshold
- the latent variance spectrum, plotted
- interpolation error: encode two held-out structures, decode the midpoint of their latents, report
  the reconstruction error of the midpoint against the midpoint of the inputs
- linear probe: predict a held-out scalar from the frozen latent, using a linear model only

**Scaling**
- **isoflop frontier**: at a fixed compute budget, is error lower with more parameters or a larger
  latent? Requires the 2-D grid, not two 1-D sweeps.
- **fit `L = A·N^(-alpha) + L_inf`** along each axis. **Report `alpha` and `L_inf` with confidence
  intervals.** These two numbers are the headline result of the study.

`L_inf` — the irreducible loss the curve is approaching — is what says whether more scale would
reach a useful error at all. It is the number the whole study exists to produce.

## Readings, to be recorded BEFORE the sweep runs

Commit these before any cell finishes, so the interpretation is not chosen after the fact.

| outcome | reading |
|---|---|
| `alpha` steep, `L_inf` low, neither axis saturated | generic scaling works on this problem; the next investment is compute |
| saturates in **latent size** well below the top of the range | latent capacity is not the bottleneck and widening it cannot help |
| saturates in **parameter count** | the architecture is the limit; a different generic architecture is the next experiment |
| **flat on both axes** | generic scaling does not work on this problem at this data scale. That is a real result and it should be reported as one. |

## Discipline requirements

These are method, not domain knowledge, and they are not optional.

1. **No metric may be introduced after results are seen.** The list above is the list.
2. **Report `n` at every point** — structures, parameters, tokens, and steps.
3. **Report sweep completeness**: which cells ran, which failed, and why. A partial grid presented
   as a full one is the most common way a scaling study misleads.
4. **State every choice left open by this brief**, at the point where you make it.
5. If a cell fails, report the failure rather than substituting a nearby cell.

## What success looks like

Two plots and two numbers: held-out error against parameter count, held-out error against latent
size, `alpha`, and `L_inf` — with the effective-rank plot beside them and the pre-registered reading
that matches.

Nothing else is in scope.
