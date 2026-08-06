# Session handoff — read this first

If you are a fresh session picking up this project, read this file, then
`COMMS/PROTOCOL.md`, then `COMMS/INBOX.md` (act on anything above `last_acted`
in `COMMS/ACK.md`). Detailed reasoning for every result below lives in the
**commit messages** — `git log` is the lab notebook, not a changelog.

Kept current by whoever pushes. If it is stale, fix it.

---

## 1. What this is

A latent molecular-dynamics model on the text-to-video recipe: compress →
diffuse in latent → decode. Four stages: autoencoder pretraining, latent
diffusion, mid-training on trajectories, RL with quantum-accurate compute.

**The binding architecture, stated by Andres:**

> N atoms → **1 global latent token of fixed width DM** → dynamics in that
> latent → N atom outputs. The token count stays at 1 whether the system has
> 1,000, 30,000, or eventually 1,000,000 atoms.

L=1 is the design point, not a swept variable. DM is the only capacity knob.
L=12/24 exist solely as addressing diagnostics.

**Four objectives:**
1. Scale to 1M+ atoms, general molecules — not just proteins
2. Model enzyme-like bond breaking/forming
3. Scale to multi-millisecond timescales
4. Run inference on a single GPU (more training compute is worth it)

---

## 2. Environment

| | |
|---|---|
| repo | `/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model` |
| workspace `$WR` | `/network/scratch/j/jacob-junqi.tian/latent-model-workspace` |
| python | `$WR/venv/bin/python` |
| required export | `LD_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/arch/distro/libffi/3.2.1/lib:$LD_LIBRARY_PATH` |
| branch | `claude/latent-diffusion-md-model-8rlhv1` |
| GPU constraint | `SBATCH_CONSTRAINT="turing|ampere|lovelace"` |

**pip is dead in that venv** — the Python has no SSL module. The working
pattern for new packages is: `curl` the cp310 manylinux wheel, unzip onto
`PYTHONPATH`. That is how `mdtraj 1.10.3` got installed (needed for ATLAS
`.xtc`). Do not attempt `pip install`; it is also denied in
`.claude/settings.json`.

**Borrowed account — non-negotiable:**
- Never write to `$HOME`; strict quota. Everything under `$SCRATCH`.
- Never modify or delete anything under `/home/mila/j/jacob-junqi.tian/`.
- `scancel` only your own job IDs. **Never `scancel -u $USER`.**
- `git config --local` only, never `--global`.
- Do not create PRs. Keep collaborator names out of commits, configs, docs.

---

## 3. Data corpora — what each can and cannot support

| corpus | N range | frames | systems | use |
|---|---|---|---|---|
| MISATO | 717–26,861 (37.5×) | 100 | ~8,500 | superseded — 80 train frames, PCA ceiling overfits |
| mdCATH | 591–7,520 (11×) | 2,400 | 5,398 (28 local) | superseded — too few local systems |
| **ATLAS** | **591–33,541 (56.7×)** | **10,001/replica × 3** | **1,938** | **current, both axes** |

ATLAS dissolved a bind that cost several corpus moves: MISATO had N range but
not frames, mdCATH had frames but not range or system count. ATLAS has all
three. `atoms = 15.77 × residues − 8`, R² = 0.9894 (validated against real
topologies, 0.5% error at the top).

Cache: stride 4 → 2,501 frames/replica, all 3 replicas, uncompressed `.npy` +
JSON sidecar (npz cannot be memmapped). Replicas 0+1 train, replica 2
held-out. **Replicas are only ~1.18× between/within RMSD** — 100 ns does not
decorrelate them, so replica 2 is a somewhat harder test than a temporal
split, not a categorically different one.

---

## 4. The current experiment

**ATLAS learning curve**, `n_train ∈ {50, 100, 200, 400, 700}`, fixed held-out
set, nested prefixes of one shuffle so only *how many* systems varies, never
*which*. This separates the two live hypotheses:

- **(a) data-limited** — every negative result so far came at 21–207 training
  systems. Curve climbing → scale up.
- **(b) fundamental** — curve flat across a 14× range in `n_train` → run the
  six-hypothesis diagnosis fan-out in INBOX 004c. **Do not pre-commit to a
  redesign.**

Alongside: the **DM sweep** (how wide the one token must be) and the
**fixed-width bottleneck arm** (INBOX 005) that separates latent width from
network width — `d_model` fixed, only the token's bottleneck varied. Without
it, DM saturation means "this architecture stops improving," not "the latent
needs X dimensions," and objective 4 depends on the latter.

**Primary comparison is ceiling-free:** codec vs ANM, two zero-shot methods on
the same held-out frames. ANM is a *diagnostic, not the bar* — it cannot
produce a programmable latent dynamics system, and failing to beat it on every
reconstruction metric does not kill the design.

**Real success criteria, in reporting order:** retains dynamical information
(run the ensemble acceptance test, not just per-frame FVE) · generalises to
unseen **systems** · scales with N · **supports the downstream generator**
(latent IAT, frame-to-frame smoothness, whether an OU fit gives stable
rollouts).

---

## 5. Established — surviving results

- **Objective 3, half closed.** TICA dimensionality is flat against effective
  time, bounded to [0.87×, 1.16×] at 1 ms, with a temperature-accelerated
  control (Arrhenius; folding control mandatory — 28/28 denatured at 450 K).
  Latent *width* need not grow with simulated time. **Coverage is unresolved**
  — the state-count result was censored and withdrawn; it needs a
  recurrence-sensitive statistic. Objective 3 is shelved half-open.
- **The mobility law.** Intrinsic dimensionality tracks mobility, not atom
  count: `c ≈ −0.9` (79 frames) to `−1.35` (2,400 frames), R² 0.86, validated
  cross-corpus. Two equal-sized systems differed 26× in required dimensionality.
- **Training cost to convergence is flat in N** across a 23× atom-count range,
  at two capacities. Objective-4 finding.
- **Codec arc closed.** B-factor pretraining is dead (oracle-B 1.45 Å > ANM
  1.37 Å); a stronger ANM only makes it deader.
- **Propagator.** Calibration constant collapses to 1.07 ± 0.05 at corpus
  scale; joint ≈ independent, so §7 additivity holds operationally.
- **Sparse ANM.** `eigsh` shift-invert on the sparse Hessian reproduces dense
  to 1.8e-10 with subspace overlap 1.000000 — same eigenproblem, not an
  approximation, so no N-dependent bias. Extends the peer baseline to the full
  ATLAS range. **ANM-5 beats ANM-10 by +0.135 FVE** on all-atom proteins;
  sweep cutoff on training systems only.

## 6. Retracted — do not resurrect

- **The graph-codec ligand peer win.** The ANM baseline was a hardcoded string
  from a *different 12-molecule sample*; the codec was evaluated on 48. Correctly
  computed, the codec **loses** at L=4 and L=8 and wins only at L=16, where 9/48
  molecules exceed 30% of their `3N−6` rank. **The project has no surviving
  peer-comparison win.**
- **`PCA-256 = 0.958`.** At k=256, n_eff is 94–165 — more dimensions than
  effective samples in 237 of 239 systems.
- **`rank90 ~ N^0.21`.** Underpowered at n=28; b is `+0.101 ± 0.021` and is a
  **lower bound**.
- **rank90 as a sizing instrument.** In-sample rank90 modes explain a median
  74.7% of held-out variance, not 90%; `rank90_out/rank90_in` runs 2.26× →
  7.70× and **grows with N**; at N=33,377 rank90 is *undefined* out of sample.
- **"Capacity is excluded."** Both premises are gone. Capacity is a **live
  hypothesis**, settled by the DM sweep, not by argument.
- **CG-ANM.** −0.059 median bias, N-dependent (slope +0.110 ± 0.039).
- **The width chain** (`168 / 0.65 × 1.83 ≈ 490` dims at 1e6 atoms) is a
  **floor**, not an estimate.

---

## 7. The failure-family checklist — run before every submission

Six recurring failure modes, each caught 2–5 times. Full definitions with
instances in `ROADMAP.md`.

| | signature |
|---|---|
| **A** | non-random exclusion correlated with a regressor (incl. *silent* ones — timeouts, OOMs, NaNs). Conservation of n: every stage prints `n_in → n_out` and characterises any shortfall |
| **B** | a count or dimension pinned to its own measurement ceiling. Print every count as a % of its ceiling; >30% means it is a bound |
| **C** | an underpowered null believed. Never report a null without the CI and the effect it still permits |
| **D** | a measurement that cannot express the effect it tests. Ask what it would report if the mechanism were perfect and if absent; if those coincide, the arm is void before it runs |
| **E** | a comparator handicapped by an unswept hyperparameter. Sweep every baseline hyperparameter on training data |
| **F** | a comparator computed on different data. Every comparator in the same script, same rows, same run; no hardcoded baseline constants |

---

## 8. Operating notes

- **Reports go in commit messages.** They are the record. Do not duplicate into
  COMMS.
- **Verify pushes.** Quote the actual `git push` output range, then `git fetch`
  and `git log --oneline -1 origin/<branch>`. A return code is not proof — a
  silent dropped `push` once hid ~15 commits.
- **Results are persisted before guards run.** An ancillary failure must never
  destroy completed measurements; that ordering bug once lost 18 minutes of
  finished results.
- **Pre-register reads before results exist.** Amending a threshold after
  seeing which way it cuts is indistinguishable from moving goalposts, even
  when the amendment is correct.
- Effort mode is Ultracode / xhigh / workflows. Keep it — the adversarial
  design reviews it enables have caught in-sample controls, censored
  thresholds, and a live pre-commitment in a docstring.
