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

Cache: **COMPLETE — 841 systems, 263 GB** (train pool 697/700, held-out 123/125).
Stride 4 → 2,501 frames/replica, all 3 replicas, uncompressed `.npy` +
JSON sidecar (npz cannot be memmapped). Replicas 0+1 train, replica 2
held-out. **Replicas are only ~1.18× between/within RMSD** — 100 ns does not
decorrelate them, so replica 2 is a somewhat harder test than a temporal
split, not a categorically different one.

---

## 4. LIVE JOBS — check these first (verified against squeue/sacct, 2026-08-06)

Workspace `$WR` = `/network/scratch/j/jacob-junqi.tian/latent-model-workspace`. Logs in `$WR/logs/`.
**Five live, one finished.** They are SLURM jobs, not session children — a client restart orphans nothing.

> **⚠ OPEN HAZARD — `atlas_dm.json` MIXES TWO TRAINING PROCEDURES.** The 13 persisted arms were
> trained before `WARMUP = 1000` and the lr-scaled step budget were added; completed arms skip on
> re-run, so they will never be retrained, while every new arm uses the new procedure. Re-training
> the best arm did **not** reproduce it (best tracked FVE **+0.0515 vs +0.0972** on the same 24
> systems). If the procedure is the cause, every DM/LR table built from that file is **Family E**.
> **Job 10307008 is the decisive test** (both procedures × 2 seeds, one card). **Do not quote a DM
> or LR winner from `atlas_dm.json` until it lands.**

| job | what it tests | output | state |
|---|---|---|---|
| **10307026** `warmup_ctl` | **Does the CURRENT procedure reproduce the PERSISTED arms?** LEGACY (warmup off, flat 30k budget) vs CURRENT (warmup 1000, lr-scaled), 2 seeds each, one process on one card, at n50/DM256/lr3e-4. Separates a procedure effect from seed noise before anything is invalidated. | `$WR/warmup_control.json` · `warmupctl_%j.log` | QUEUED, `long` + `--requeue`, 5 h. (**10307008 died in 0 s** — a `git rebase` briefly removed the script from the working tree as the job started. Never rebase while a job is launching from the repo.) |
| **10307029** `modal_arm` | **INBOX 015.** Modal decoder `disp_i = B_i(structure) @ z` as an arm: control vs untied vs tied, DM=256, n50, **LR swept per decoder** (Family E), **control re-trained in the same job** so the comparison does not depend on the `atlas_dm.json` procedure question. Self-test on real ATLAS tensors gates training and passed: superposition err 9.5e-07, **tied `‖z0‖` exactly 0.0** vs untied 5.66. | `$WR/modal_arm.json` · `$WR/modal_arm_ckpt/` · `modal_%j.log` | QUEUED, `long` + `--requeue`, 20 h, resumable |
| **10306938** `atlas_peer` | **INBOX 14a — THE PRIMARY COMPARISON, which has never appeared in an ATLAS output.** codec vs **ANM-k** (the zero-shot peer, cutoff swept on TRAINING systems) vs **per-system PCA-k** (the oracle, reported as a fraction only), at matched capacity k=DM ∈ {16,64,256}. CPU-only: it JOINS to codec arms already in `atlas_dm.json` rather than training anything. Also reports the **codec−ANM gap vs N**, which is ceiling-free because the shared denominator cancels. | `$WR/atlas_peer.json` · `peer_%j.log` | QUEUED, `long-cpu` + `--requeue`, 16 h |
| **10306939** `atlas_modes` | **INBOX 14b — where the error LIVES.** DONE. Verdict **VARIANCE-SELECTIVE, NOT timescale-selective**: the one-variable log(IAT) coefficient is +0.4700 ± 0.1917, but the PARTIAL coefficient holding variance share fixed is **+0.1445 ± 0.1830 — CI spans zero**. Residual is **94% of the motion's own amplitude** (2.412 Å vs 2.572 Å). Per-mode FVE goes **negative beyond mode ~7**. | `$WR/atlas_modes.json` · `modes_10306939.log` | **FINISHED** — but measured on a re-trained arm that did NOT reproduce; re-run once the procedure question (10307008) is settled |
| **10306831** `atlas_dm` | DM sweep + 005 bottleneck + **013b criterion-1-vs-N** + **12b identity ablation**, at L=1. LR {3e-5…3e-3} with **1,000-step warmup** and an lr-scaled step budget; ladder {50,130,300,600}. Checkpoints each arm so criterion-1 scoring never re-trains. | `$WR/atlas_dm.json` · `$WR/atlas_dm_criterion1.json` · `$WR/atlas_dm_ckpt/` · `atlasdm_10306831.log` | RUNNING ~40 min, on the `n50 dm64 lr3e-5` arm |
| **10306738** `tica_vs_n` | **Q1 re-run with the TRUNCATION-MATCHED CONTROL** and the 13a stopping rule. Reports only the matched-*m* difference `exponent(TICA\|m) − exponent(PCA\|m)`; **no absolute exponent**. Closes 007 as unanswerable if n_eff per TICA dimension < 2.0. | `$WR/atlas_tica_vs_n.json` (v1 archived `..._v1_nocontrol.json`) · `ticaN_10306738.log` | RUNNING ~40 min, τ=20 selected on train, ~60/123 held-out |
| **10306590** `atlas_b` | **Q2.** b on ATLAS, replicas as the join unit; fits all three rank90 variants per INBOX 011. | `$WR/atlas_b.json` · `atlasb_10306590.log` | RUNNING ~1:47, 100/841 systems (~12 h pace, 16 h limit) |

**THE CACHE IS COMPLETE** — `10301859` finished in 6:27:23. **841 systems, 263 GB, train pool 697/700,
held-out 123/125.** The full `n_train` ladder {50,130,300,600} is runnable; it no longer gates anything.

**Cancelled deliberately, NOT failures** (so six cancelled `atlas_dm` jobs are not read as a failing
experiment): `10305469`/`10305543` censored `maxlag` · `10305827`/`10305840` pre-005 architecture ·
`10305911`/`10305936` died on a startup `NameError`, fixed and resubmitted · **`10305995` cancelled
after its first 11 arms because the LR grid floor (3e-4) was too high for DM=512** — the completed
arms are saved in `atlas_dm.json` and skip on re-run.

Completed earlier: `10305556` atlas_neff · `10305712` r90_insample (v1) · `10304109` atlas_guard.

**AUTONOMOUS LOOP — RECREATE IT AFTER ANY RESTART.** A `CronCreate` job (`7,34 * * * *`, ~27 min)
drives the 009 work cycle: pull → work the INBOX above `last_acted` → check `squeue` and job logs →
fall through to the STANDING QUEUE in INBOX 009 → push and verify the remote moved → re-read the
inbox. **It is SESSION-ONLY: in memory, never on disk, and it dies when the session exits — including
the restart 006 asks for.** Nothing warns you. A fresh session must recreate it or the project goes
quiet until a human nudges it.

**Restart guidance (INBOX 006):** a natural pause is after the DM sweep's `n_train=130` arms land.
**13 arms from earlier `atlas_dm` runs persist in `atlas_dm.json` and are skipped on re-run** — a
restart of that job costs only the in-flight arm, never the finished ones.
All four jobs survive a restart.

---

## 5. The current experiment

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

## 6. Established — surviving results

- **Objective 3, half closed.** TICA dimensionality is flat against effective
  time, bounded to [0.87×, 1.16×] at 1 ms, with a temperature-accelerated
  control (Arrhenius; folding control mandatory — 28/28 denatured at 450 K).
  Latent *width* need not grow with simulated time. **Coverage is unresolved**
  — the state-count result was censored and withdrawn; it needs a
  recurrence-sensitive statistic. Objective 3 is shelved half-open.
- **The mobility law.** Intrinsic dimensionality tracks mobility, not atom
  count: `c ≈ −0.9` (79 frames) to `−1.35` (2,400 frames), R² 0.86, validated
  cross-corpus. Two equal-sized systems differed 26× in required dimensionality.
- **Effective sample size, n=239 ATLAS systems.** `n_eff = F/tau_int` is only
  **1–7%** of raw frames. Per fitted dimension it is **0.53 at k=256, below 1.0
  in 237/239 systems** — the PCA ceiling fits more dimensions than it has
  independent samples almost everywhere. τ at fixed k grows with N
  (**+0.1794 ± 0.0138**, R² 0.735), so the ceiling degrades with N
  (**−0.1880 ± 0.0439** at k=256, p=0.000, survives bootstrap /
  drop-most-influential / adversarial). **The ceiling is irreducibly N-biased on
  this corpus** — denser subsampling cannot help (n_eff is set by τ, not stride)
  and replicas sit at a measured **1.18×** between/within ratio. The
  *ceiling-free* primary (codec vs ANM) is the main line; the oracle-fraction is
  a permanently caveated secondary that must always carry the slope and the bias
  direction (it **flatters** the codec at large N).
- **rank90 is a FLOOR, measured (n=123 held-out ATLAS systems).** In-sample
  rank90 modes explain a median **77.1%** of held-out variance, not 90% —
  **below 90% in 123/123 systems**, min 49.9%. `rank90_out > rank90_in` in
  **118/118** systems that reach 90% at all, ratio median **3.49×** (IQR
  2.26–6.08×). The other **5 never reach 90% at any k**, and they are the
  largest (N 13,543–33,377, median 14,018 vs corpus median 3,249) — there
  rank90 is **undefined out of sample**, not merely low.
  **The b exponent doubles out of sample:** in-sample `+0.4657 ± 0.2158`,
  out-of-sample **`+0.9285 ± 0.2220`** — and the out-of-sample figure is *itself*
  biased low, because the high-N systems that fail to reach 90% are excluded
  (Family A). **This is corpus-independent** (MISATO 10 ns / mdCATH 2.5 µs /
  ATLAS 100 ns all fail the same way), so **no dataset move fixes it — the fix is
  a change of instrument**, which is why latent width is now sized from the
  codec's own held-out saturation curve.
- **Training cost to convergence is flat in N** across a 23× atom-count range,
  at two capacities. Objective-4 finding.
- **Codec arc closed.** B-factor pretraining is dead (oracle-B 1.45 Å > ANM
  1.37 Å); a stronger ANM only makes it deader.
- **Propagator.** Calibration constant collapses to 1.07 ± 0.05 at corpus
  scale; joint ≈ independent, so §7 additivity holds operationally.
- **Criterion-1 harness exists and is PROVEN to discriminate** (`scripts/armf_criterion1.py`).
  Four discriminators on decoded trajectories in one reference-fitted basis: marginal std ratio,
  **IAT per mode (the kinetic test)**, cross-mode coupling, 2D free-energy JS. Validated on a real
  ATLAS system: identity **4/4**; **frames shuffled 3/4 — passing std/coupling/free-energy at 100%
  and failing the kinetic test at 0%** (IAT 30.0 → 1.0 frames); variance collapse 2/4. Shuffling
  preserves every marginal and the whole free-energy surface, so a harness that passed it would be
  measuring distribution rather than dynamics. Ready the moment a codec arm lands.
- **The DM-sweep LR sweep is load-bearing, not hygiene.** Optimal LR falls monotonically with width
  (3e-3 at DM=16 → 3e-4 at DM=64/256). At DM=256, lr=1e-3 and 3e-3 collapse to a constant code while
  lr=3e-4 is the **best arm in the sweep** (+0.1553). The mdCATH "wide arms collapse" conclusion was
  an LR artifact. Collapse rate 0/3 seeds at the right LR vs 2/2 at the wrong one.
- **Sparse ANM.** `eigsh` shift-invert on the sparse Hessian reproduces dense
  to 1.8e-10 with subspace overlap 1.000000 — same eigenproblem, not an
  approximation, so no N-dependent bias. Extends the peer baseline to the full
  ATLAS range. **ANM-5 beats ANM-10 by +0.135 FVE** on all-atom proteins;
  sweep cutoff on training systems only.

## 7. Retracted — do not resurrect

- **The graph-codec ligand peer win.** The ANM baseline was a hardcoded string
  from a *different 12-molecule sample*; the codec was evaluated on 48. Correctly
  computed, the codec **loses** at L=4 and L=8 and wins only at L=16, where 9/48
  molecules exceed 30% of their `3N−6` rank. **The project has no surviving
  peer-comparison win.**
- **`PCA-256 = 0.958`.** At k=256, n_eff is 94–165 — more dimensions than
  effective samples in 237 of 239 systems.
- **`rank90 ~ N^0.21`.** Underpowered at n=28; b is `+0.101 ± 0.021` and is a
  **lower bound**.
- **rank90 as a sizing instrument.** FULL RESULT, n=123: in-sample rank90 modes
  explain a median **77.1%** of held-out variance, below 90% in **123/123**
  systems (min 49.9%); `rank90_out > rank90_in` in **118/118** that reach 90% at
  all, ratio median **3.49×**; the remaining **5 never reach 90% at any k** and
  are the largest in the corpus, so there rank90 is *undefined* out of sample.
  **b doubles out of sample: `+0.4657 ± 0.2158` in-sample vs `+0.9285 ± 0.2220`
  out** — and the latter is still biased low, since the failing high-N systems
  are excluded (Family A).
- **"Capacity is excluded."** Both premises are gone. Capacity is a **live
  hypothesis**, settled by the DM sweep, not by argument.
- **CG-ANM.** −0.059 median bias, N-dependent (slope +0.110 ± 0.039).
- **The width chain** (`168 / 0.65 × 1.83 ≈ 490` dims at 1e6 atoms) is a
  **floor**, not an estimate.

---

## 8. The failure-family checklist — run before every submission

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

## 9. Operating notes

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
