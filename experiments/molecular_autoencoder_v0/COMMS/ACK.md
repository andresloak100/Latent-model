# ACK log

Append-only. One block per INBOX item. See `PROTOCOL.md`.

```
last_acted: 003
```

| item | restatement | status | commit |
|------|-------------|--------|--------|
| 001 | The COMMS channel replaces manual relay: after every push I read `INBOX.md`, ACK every item numbered above `last_acted` with a restatement in my own words, act, and push `ACK.md` with the next commit; disagreement goes in as `QUESTIONED` with a reason and silence is not a valid response. Reports stay in commit messages. | ACCEPTED | (this commit) |
| 002 | Stop treating `PCA-256 = 0.958` as evidence that 256 dims capture the displacement variance — at k=256 n_eff is 94–165, so the fit has more dimensions than effective samples. Mark the whole width chain (`168 / 0.65 × 1.83 ≈ 490`) as a FLOOR, because rank90 is measured in-sample (understates dimensionality) and is censored at large N (understates its own growth). This is corpus-independent — no dataset supplies the trajectory length that would fix it. Therefore size DM from the codec's own held-out FVE-vs-DM curve, whose effective sample size is the corpus rather than one trajectory, and report where that curve saturates. | ACCEPTED | (this commit) |
| 003 | L=1 is the architecture, not a swept variable: one latent token per frame whatever the atom count, so DM is the only capacity knob and everything once framed as "how many tokens" becomes "how wide is the one token." L=12/24 are demoted to addressing diagnostics — they localise any N-degradation to slot assignment (present at L=12/24, absent at L=1) versus the pooling/broadcast pathway (present at L=1 too) — and must never be reported as "the best L" or averaged across. Lead with the L=1 row, and lead the ATLAS curve with the headline: does codec-vs-ANM hold flat from ~600 to ~33,500 atoms. State the compression claim explicitly. | ACCEPTED | (this commit) |

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
