# ACK log

Append-only. One block per INBOX item. See `PROTOCOL.md`.

```
last_acted: 010
```

| item | restatement | status | commit |
|------|-------------|--------|--------|
| 001 | The COMMS channel replaces manual relay: after every push I read `INBOX.md`, ACK every item numbered above `last_acted` with a restatement in my own words, act, and push `ACK.md` with the next commit; disagreement goes in as `QUESTIONED` with a reason and silence is not a valid response. Reports stay in commit messages. | ACCEPTED | (this commit) |
| 002 | Stop treating `PCA-256 = 0.958` as evidence that 256 dims capture the displacement variance — at k=256 n_eff is 94–165, so the fit has more dimensions than effective samples. Mark the whole width chain (`168 / 0.65 × 1.83 ≈ 490`) as a FLOOR, because rank90 is measured in-sample (understates dimensionality) and is censored at large N (understates its own growth). This is corpus-independent — no dataset supplies the trajectory length that would fix it. Therefore size DM from the codec's own held-out FVE-vs-DM curve, whose effective sample size is the corpus rather than one trajectory, and report where that curve saturates. | ACCEPTED | (this commit) |
| 003 | L=1 is the architecture, not a swept variable: one latent token per frame whatever the atom count, so DM is the only capacity knob and everything once framed as "how many tokens" becomes "how wide is the one token." L=12/24 are demoted to addressing diagnostics — they localise any N-degradation to slot assignment (present at L=12/24, absent at L=1) versus the pooling/broadcast pathway (present at L=1 too) — and must never be reported as "the best L" or averaged across. Lead with the L=1 row, and lead the ATLAS curve with the headline: does codec-vs-ANM hold flat from ~600 to ~33,500 atoms. State the compression claim explicitly. | ACCEPTED | (this commit) |

| 004 | Delete the pre-committed ANM-basis-decoder branch from `armf_atlas_curve.py`'s verdict logic before the curve runs — a flat curve must print "run the diagnosis fan-out," never a design, because naming one cause for an unseen result risks an open-ended ANM optimisation loop. ANM is a diagnostic, not the bar: it has no generator and cannot be the product architecture, so losing to it on per-frame FVE does not kill the codec. Report against four real criteria in order — dynamical fidelity via the existing ensemble acceptance test, generalisation to unseen systems, N-scaling at L=1, and whether the latent is something a propagator can actually model. If the curve is flat, run the six-hypothesis fan-out and report which one the evidence supports before proposing any design. Stop putting calendar estimates on objectives 2 and 3. | ACCEPTED | (this commit) |

| 005 | The bottleneck arm is required and runs alongside, not after: because network width sets encode/decode cost while LATENT width sets the generator's cost, and objective 4 turns on the latter, a sweep of d_model cannot answer "how wide does the one token need to be." Hold d_model fixed at the widest value that trains reliably, vary only a linear down/up projection on the token, sweep DM_latent in {16,64,128,256,512} capped at d_model, and plot both curves on the same axes — where they diverge is the answer. Apply the cross-fit participation ratio and the criterion-4 dynamics measurements to the bottleneck arms, because there the latent IS the object the generator will model. | ACCEPTED | (this commit) |

| 006 | `SESSION_HANDOFF.md` exists so a fresh session can restart cheaply once `.claude/settings.json` takes effect; verify and correct it against reality rather than my reports, add a LIVE JOBS section with job IDs, what each tests, expected completion and where output lands, and keep it current on every push where the answer would change — moving retracted results into section 6 in the same commit that retracts them. Restart at a natural pause, with every running job listed so nothing is orphaned. | ACCEPTED | (this commit) |

| 007 | rank90's b ≈ 0.93 is about VARIANCE dimensionality, and near-linear growth there is close to what independent local thermal motion would give — those modes are real but are not what a latent generator must represent. The slowness-weighted analogue was flat across effective time but has never been measured across N, so measure TICA dimensionality vs atom count using the objective-3 definition, with Family E on the lag time (swept on training systems, applied unchanged), Family B on rank position and n_eff, Family A on any system failing to reach 90% of the slow spectrum, and both sample regimes. If flat, state that MSE is the wrong training objective for this architecture — but do not change the loss on that basis yet. Also: record b as `≥ 0.93`, an inequality, since the 5 excluded systems are the largest and were excluded *because* they need the most modes. | ACCEPTED | (this commit) |
| 008 | 007 is CPU-only, needs no training, and is gated by neither 10305995 nor the cache reaching 825, since the 371 cached systems already span the full N range where the slope's leverage lives — so submit it in parallel now rather than waiting. It is the higher-value of the two, because it reframes what the DM sweep's curve means before its first arm lands. | ACCEPTED | (this commit) |
| 009 | Never end a turn with unacted INBOX items, and when the inbox is empty fall through to the STANDING QUEUE (Q1 TICA-dim vs N · Q2 b on ATLAS · Q3 criterion-1 harness built and smoke-tested before it is needed · Q4 finish the rank90 audit in the files themselves · Q5 re-run the DM sweep as the cache grows · Q6 verify the handoff) rather than stopping. Only stop when the queue is exhausted and every job is finished-and-reported or genuinely blocked, and then say exactly what would unblock me. Submit long jobs and keep working while they run. | ACCEPTED | (this commit) |
| 010 | Check whether I can schedule my own wake-ups and, if so, set a 20–30 minute recurring prompt that pulls, works the inbox, falls through to the standing queue, and checks squeue; report the mechanism and interval, or say plainly that I cannot. | ACCEPTED | (this commit) |

## Notes on 010 — I CAN, with one caveat that matters

**Mechanism: `CronCreate`. Job `7f6daacb`, cron `7,34 * * * *` — fires at :07 and :34, roughly every
27 minutes.** Off-minute deliberately, since :00 and :30 are where every scheduled job on the planet
lands. The prompt pulls, works the INBOX above `last_acted`, checks `squeue` and each running job's
log, falls through to the 009 standing queue when the inbox is empty, pushes with verification that
the remote moved, and re-reads the inbox before stopping.

**THE CAVEAT, because it changes your plan: the schedule is SESSION-ONLY.** It lives in memory, is
never written to disk, and **dies the moment this session exits** — including the client restart 006
asks for so `.claude/settings.json` takes effect. It also only fires while the REPL is idle, and it
auto-expires after 7 days. So the restart and the autonomous loop are in direct tension: restarting
buys no approval prompts but silently kills the loop. **After any restart, the cron must be recreated
— it will not come back on its own, and nothing will warn you.** I have added this to
`SESSION_HANDOFF.md` so a fresh session recreates it rather than assuming it is still running.

## Notes on 007

**Submitted as job 10306540** (`long-cpu`, 8 CPU, 96 GB, 12 h) before finishing this ACK, per 008.
Definition taken verbatim from `armf_slowness.py:54` so the numbers are comparable to the
objective-3 result. Cross-replica correctness handled: the train set is replicas 0+1 concatenated, so
lagged pairs are accumulated **within each replica only** — a lagged covariance across the join would
pair the end of replica 0 with the start of replica 1, unrelated conformations entering as spurious
decorrelation.

**The dry run found a censoring problem before submission, and it changes what can be reported.**
Out-of-sample TICA dimension pins at **86–87% of its basis at every basis I tried (100, 150, 300)**,
which is over the 60% line — so a flat slope there would be a ceiling, exactly the Family B failure
007 names. Two things follow. First, the in-sample dimension is *itself* basis-tracking: 36% of 100,
38% of 300 — which is the pathology `armf_slowness.py` already documented ("tica/basis was a
near-constant 0.42–0.45 across every condition") and why it fixed the basis at 100. The fixed-basis
number is comparable across systems but is **not an absolute dimensionality**, and I will not report
it as one. Second, I split the out-of-sample reading in two: **train-order** (ordering-sensitive, the
direct rank90_out analogue) and **ordering-free** (ranked by actual held-out slowness). If the first
greatly exceeds the second, the train basis is fine but its *order* does not transfer — a different
defect from slow dynamics being high-dimensional, and only the second bears on the architecture
question. The verdict block uses the ordering-free quantity and **withholds the verdict entirely**
if it is still above 60% of basis.

**On the b correction — accepted, and it is the stronger form of what I wrote.** I reported that the
5 excluded systems are the largest and that this biases the slope low, but I stated the result as a
point estimate anyway. The inequality is the honest form: **b ≥ 0.93**, because the exclusion is not
incidental to the measurement — those systems were dropped *precisely because* they need more modes
than the data resolves, which is the mechanism that flattens the slope. ROADMAP now carries the
inequality.

## Notes on 006

Verified against `squeue`/`sacct` rather than against my own reports, and several things were wrong
or missing. Added the LIVE JOBS section with both running jobs, what each tests, where output lands,
and time remaining. Recorded which cancellations were **deliberate** and why (censored `maxlag`;
pre-005 architecture) so a fresh session does not read six cancelled `atlas_dm` jobs as a failing
experiment. Noted that both jobs are SLURM jobs rather than session children, so restarting the
client orphans nothing.

**Correcting my own count while doing this:** the handoff's corpus table said ATLAS is the current
corpus for "both axes" without recording how much of it actually exists yet. The cache is **371 of
825 systems**, and the train pool is only **136 of 700** — which is why the `n_train` ladder is
adaptive and the DM sweep must be **re-run as the cache grows**. A fresh session that missed that
would conclude the higher `n_train` arms had been tested and come back empty.

**I have not restarted yet, deliberately.** The pause 006 describes is "after the curve results
land," and the DM sweep had not produced a single arm — it had *died in 7 seconds*, twice, on a
`NameError` in the startup banner (`L` removed when `L_PRIMARY` was introduced under 003). My smoke
tests called the module's functions directly and never executed `__main__`, so they could not catch
it. Fixed, and the gap closed properly: the whole `__main__` path now gets dry-run on a shrunken
copy of the real script, which exercises every reporting block. That dry run immediately found a
second defect — the bottleneck arm's health gate required `FVE > 0.01`, conflating "d_model trained"
with "d_model performed", which at L=1 could have silently skipped the arm 005 makes required. The
gate is now PR-based (the mdCATH collapse signature is a *constant* latent, PR ≈ 1), and if nothing
passes it the sweep runs against the least-collapsed arm with a loud PROVISIONAL banner rather than
skipping.

## Notes on 005

Implemented and smoke-tested; the bottleneck runs in the same job as the network sweep, not after it.

**The detail that made this worth doing carefully.** `encode()` now returns the DM_latent-dimensional
code rather than the d_model-wide internal representation, so the participation ratio and the
criterion-4 measurements attach to the code automatically. Had I bolted the bottleneck on without
that, PR would have kept measuring the d_model activations — reporting a healthy 512-wide
representation while the actual code the propagator sees was 16 numbers. That is the same class of
error as measuring rank90 in-sample: an instrument pointed at the wrong object returns a flattering
number rather than an obviously broken one. Verified: code shape tracks `dlat` exactly.

**One judgement call I made rather than asking.** "Use the largest d_model you can train reliably —
512 if it trains, else 256" is a runtime decision, so the script *measures* it instead of assuming:
it takes DM=512 from the network sweep and requires that arm to be non-`improving`, PR/DM > 10%, and
FVE > 0.01 before adopting 512 as the fixed d_model, falling back to 256 on the same test. If neither
qualifies the bottleneck sweep is SKIPPED with an explicit message, because a bottleneck measured
against a d_model that never trained would be uninterpretable — and I would rather report "no sound
fixed d_model to hold" than a curve that looks like a result.

**Pre-registered reads wired in verbatim**, printing the corresponding verdict: bottleneck saturates
below half of d_model => the latent needs less width than the network and DM_latent is the headline
number; the two curves track => network capacity is binding, DM never measured latent width, and the
network figure must not be quoted as one; still climbing at DM_latent = d_model => the latent
requirement is not bracketed and d_model must widen before any width claim.

## Notes on 004

**4a done, and it was live in two places, not one.** The pre-commitment was in the verdict block
*and* restated in the module docstring. Docstrings in this project carry load-bearing scientific
claims — the graph-codec retraction and the Family-F hardcoded baseline both started as a claim
someone only ever read in a header — so a deletion that left the docstring standing would have left
the pre-commitment fully intact for the next reader. Both are gone; the verdict now enumerates the
six 4c hypotheses and prints no design. Verified no surviving reference (the remaining "ANM basis"
hits are `armf_ou_baseline.py` / `armf_propagator.py`, which are the OU propagator baseline — a
different object, correctly named).

**On 4b, one thing I want on the record because it cuts against my own recent work.** Criterion 1
(dynamical fidelity over per-frame FVE) means the DM sweep I am about to run optimises and reports
the *wrong* primary quantity: it selects arms by held-out FVE, which is exactly the metric 4b demotes.
The saturation read is still valid for the width question, but "best DM by FVE" is not automatically
"best DM for the propagator" — a wider code could reconstruct better while producing a jumpier latent
trajectory. I am adding the latent-smoothness and latent-IAT measurements (criterion 4) to the DM
sweep's per-arm output so the two rankings can be compared directly rather than assumed to agree. If
they disagree, that disagreement is itself the finding.

**4d noted.** No calendar estimates will appear against objectives 2 or 3.

## Notes on 003

Accepted and already wired into `armf_atlas_dm.py` (the L=24 job was cancelled before it burned GPU
on the wrong architecture). One addition and one scope note.

**Addition — the diagnostic needs a capacity-matched arm to mean what it says.** Total latent
capacity is `L x DM`, so L=12 at the L=1 winner's DM carries **12x** the capacity of L=1. A
difference in N-slope there could be capacity rather than addressing, and the diagnostic would be
naming an effect it cannot separate from the one it holds fixed (Family D). Each diagnostic L
therefore runs **twice**: at the same DM, and capacity-matched at `DM/L`. Addressing is implicated
only if the slope gap survives **both**. If it appears only at same-DM, the cause is capacity and
the addressing reading is void.

**Scope note on the compression claim.** "One token of width DM per frame, any N" is exact and I have
put it in the run's output. But the 1e6-atom figure (3,000,000 coordinates → 256 numbers) is an
**architectural** claim — token count is N-independent *by construction* — not a measured
reconstruction quality at that size. The measured range is N=598–33,377 (56x). I have written it that
way so the compression ratio cannot later be read as a fidelity result at 1e6 atoms, which is the
same class of error as the width chain's 2.5-decade extrapolation.

**A prediction, recorded before the run.** At L=1 the entire conformational state of a 33,377-atom
protein flows through DM numbers — 100,131 coordinates through, at most, 512. I expect held-out FVE
at the top of the N axis to be low in absolute terms. That is not an argument against the design
point; it is what the N-slope is there to measure, and the primary comparison (codec vs ANM, both
zero-shot) is a ratio that does not care about the absolute level.


## Notes on 002

Accepted in full; two corrections to the stated reasoning, neither of which changes the conclusion.

**1. The mechanism for why held-out evaluation failed to penalise PCA-256 is not "frames 10 ps
later."** Under the current protocol held-out is **replica 2** — a separate MD run, not a temporal
continuation. The number is still not protected, but for a different and *measured* reason: the
between-replica / within-replica RMSD ratio is **1.18** (range 1.085–1.442, n=10, N=598–33,377), so
replica 2 sits only ~18% further from the training frames than training frames sit from each other.
100 ns does not decorrelate the replicas. So the correct statement is "a held-out replica at 1.18×
separation cannot penalise a fit with 0.53 effective samples per dimension," which is the same
conclusion resting on a measurement rather than an assumption.

**2. `0.958` is not in the committed record under that phrasing.** A repo-wide search finds it only
as an unrelated `PCA-DM 512` ceiling in `armf_capacity_axis.md` (mdCATH) and in old per-system
metrics tables. It appears to have been a chat-reported feasibility number. I have recorded the
withdrawal explicitly in the ROADMAP anyway, with the governing figures attached, so it cannot be
resurrected from the transcript: under the **new** protocol the PCA-256 median is **0.8614**, and
even that is not usable as a capacity claim — at k=256 the ceiling fits **more dimensions than it has
effective samples in 237 of 239 systems** (n_eff per dimension 0.53).

**Supporting measurements now in hand** (both were assertions when 002 was written, both are now
measured):

- *In-sample bias, ATLAS, fit replicas 0+1 → evaluate replica 2*: the in-sample rank90 modes explain
  a median of **74.7%** of held-out variance, not 90%. rank90_out/rank90_in runs **2.26× → 7.70×**
  and the ratio **grows with N**; at N=33,377 held-out FVE never reaches 90% at any k (max 72.5%), so
  rank90 is not merely underestimated there — it is **undefined out of sample**. Because the
  correction factor grows with N, bias (1) also attacks the b exponent, independently of bias (2).
- *Censoring*: rank90 reaches **32.3%** of usable rank at the top of the N axis, confirming (2).

**One scope limit I am adding to 002's final instruction.** "If it flattens at DM=256, that is the
architectural answer" is only sound if a flat curve cannot instead mean the wide arms were starved.
The prior mdCATH DM sweep produced exactly that failure — DM=256/512 collapsed to constant output
(G1 cos = 1.0000) on 21 training domains and were declared VOID. So the run reports, per arm, the
**participation ratio of the learned latent code** alongside FVE. A flat curve with the wide arms
using their full width is width saturation; a flat curve with DM=512 using ~200 effective dimensions
is capacity that failed to train, which is a different finding and must not be reported as the first.
