# Latent-model

Research toward a **latent molecular-dynamics model**: compress molecular structure into a
compact latent space, run diffusion there, and decode back — the approach modern
text-to-video models use, applied to proteins and molecular dynamics.

This is a **research record, not a released model.** Several headline results have been
retracted by later measurement, at least one was retracted twice, and the reversals are
kept in place rather than tidied away. If you are evaluating whether something here is
true, the reversal history is the most useful thing in the repository.

---

## Start here

**Read in this order.** Each step tells you whether to keep going.

| # | Read | Why |
|---|---|---|
| 1 | This file | What exists, what it has shown, how to run it. If the status table answers your question, stop. |
| 2 | [`COMMS/INDEX.md`](experiments/molecular_autoencoder_v0/COMMS/INDEX.md) | 96 numbered decisions with line links. **Reversals are flagged.** Fastest route to why anything was done. |
| 3 | [`ROADMAP.md`](experiments/molecular_autoencoder_v0/ROADMAP.md) §5b | The exclusion table — what has been ruled out and what each exclusion rests on. Read before proposing an experiment; several obvious ones are already closed. |
| 4 | `git log` | Full experiment reports live in **commit messages**, not files. This is a primary source here, not a changelog. |

### Run it — about two minutes, no GPU

```bash
cd experiments/molecular_autoencoder_v0
pip install -r requirements.txt
python -m pytest tests/          # 431 passed, 2 skipped, ~68 s
```

The suite needs **no cluster, no GPU and no downloaded data**. If it passes, your
environment is correct. Everything past this point needs either the preprocessed corpora
(regenerable via `scripts/prepare_dataset.py`) or the compute cluster.

---

## What has actually been measured

Including the results that went the wrong way. **There is currently no surviving
peer-comparison win.**

| line | result |
|---|---|
| Static reconstruction, non-homologous held-out (n=197) | **0.9358 Å** median all-atom — this is the generalisation figure |
| Static reconstruction, full held-out (n=758) | 0.8357 Å — but **74% of that set has a ≥30% training homolog at median 98% identity**, so it is not a generalisation figure |
| Rate–distortion vs classical transform coding | **a loss**, ≈0.66 bits/dim, pending a symmetric re-run |
| Dynamics vs ANM (a zero-cost physics baseline), reconstruction FVE | **0 wins in 123** held-out systems; 95% upper bound on the win rate 2.4% |
| Dynamics vs ANM, at matched bit rate | never measured until now — in progress |
| Learned propagator vs Ornstein–Uhlenbeck | reaches cross-mode coupling OU cannot reach *by construction*, on 61% of the corpus, at 24–41% divergence and worse agreement elsewhere |
| Latent capacity | participation ratio **2.00 of 8** channels; 68.4% of variance in a single direction |

The compression result against transform coding has been retracted, un-retracted and
retracted again — each time by a correction to what "rate" or "distortion" was measuring,
not by new data. That sequence is items 084 → 085 → 088 → 090a → 094 in the index.

---

## Layout

```
experiments/molecular_autoencoder_v0/
  molae/        the library. Imported. Model, losses, alignment, metrics, dataset, config.
  scripts/      entry points — see the naming table below
  configs/      one YAML per run
  COMMS/        the two-agent protocol; START AT INDEX.md
  outputs/      committed results (JSON + markdown); weights are ignored
  data/         splits, manifests and indices ONLY — corpora are regenerable, untracked
  slurm/        submission wrappers
  tests/        431 tests, run locally
  ROADMAP.md    the long-form record; §5b is the exclusion table
  REPORT.md     milestone write-up          PLAN.md   milestone scope
```

### Naming, so the file list stops looking arbitrary

| pattern | what it is |
|---|---|
| `molae/*.py` | the library, imported by everything else |
| `scripts/train.py`, `prepare_dataset.py`, `eval.py` | the core pipeline: build a corpus, train, score |
| `scripts/armf_*.py` | **cluster experiments.** Each is run directly by `sbatch` and imported by nothing, so grepping its name finds no callers — that means *top of its own call chain*, *not* dead code. Around 100 of them. The prefix's expansion is not recorded anywhere in the history. |
| `configs/*.yaml` | one per run. `name` and `train.out_dir` are a run's **identity** — two configs sharing them resume each other's checkpoints, which has happened six times |
| `data/**` | splits and manifests are **decisions** and are tracked; the `.npz` corpora are derived and are not |

### Tracing a number

A figure quoted here or in `ROADMAP.md` runs through three places, and you usually need
all three:

- **the value** — a file under `outputs/`
- **the reasoning** — the INBOX item that produced or corrected it, via `COMMS/INDEX.md`
- **the run** — the commit message, carrying the full report and job IDs

Do not trust a bare grep: `0.8357` appears in several files and only one is the figure
quoted above. The index is the reliable route.

---

## How the work is done

Two agents share one branch. A **planning agent** reads results and appends numbered
items to `COMMS/INBOX.md`; a **cluster agent** runs the jobs, reports in commit messages,
and records `last_acted` in `COMMS/ACK.md`. An item above `last_acted` has not been read —
which is deliberately distinguishable from one that was read and rejected.

Every claim is audited against six recurring failure modes, recorded in `ROADMAP.md`:

| | |
|---|---|
| **A** | an exclusion correlated with a regressor |
| **B** | a count pinned to its own measurement ceiling |
| **C** | an underpowered null believed |
| **D** | a measurement that cannot express the effect |
| **E** | an unswept comparator |
| **F** | a comparator computed on different data |

Most corrections here are one of those six, and several are the same defect appearing
somewhere new: **a rule that named one specific thing, while new things arrived with
slightly different names.** That single pattern produced a config collision across six
runs, a parser that silently dropped every protein under 81 residues, and 20,772
regenerable files committed because an ignore rule said `processed/` and the corpora were
called `processed_small`, `processed_big`, `processed_complex` and
`processed_complex_scaled`.

## Scope

Milestone 1 is the autoencoder and its evaluation. **No diffusion, trajectories, RL or
generation is claimed** — the propagator work in progress is exploratory and is reported
as such. Reactions are out of scope by construction: topology and atom count are fixed,
hydrogens are removed, and there is no force, energy, charge or electronic-state model.
