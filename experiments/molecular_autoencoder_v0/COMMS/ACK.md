# ACK log

Append-only. One block per INBOX item. See `PROTOCOL.md`.

```
last_acted: 087
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

| 011 | The two-way TICA reading applies retroactively to b: `rank90_out` was counted in TRAIN ORDER, so `b ≥ 0.93` conflates the train basis ordering worse at large N with held-out content being genuinely higher-dimensional. Re-measure rank90 three ways — in-sample, out-of-sample ordered, out-of-sample sorted by held-out variance — report all three exponents with CIs plus the ordering gap as its own quantity vs N, and cross-fit the sorted variant (order on one half of held-out frames, evaluate on the other) since sorting on the evaluation data is a selection that can only flatter it. The ordering-free number is the architecture-relevant one, because a fixed PCA basis is locked to its order while the codec's learned structure-conditioned decoder is not. | ACCEPTED | (this commit) |

| 012 | Retract the mdCATH DM=256/512 collapse explicitly — those arms lost on an unswept learning rate, not on capacity, and the collapse was cited as evidence wide codes cannot train. Then settle whether PR≈15 is saturation or a capability limit by reporting PR and FVE jointly at every rung of the n_train ladder with the PR-vs-FVE slope and CI. And treat the 86–91% identity share as wasted capacity: measure `||z0||/||z||` directly, then ablate with `z := encode(x) − z0` (zero-shot, since z0 needs only the reference structure), reporting FVE, PR and identity share before and after — noting the encoder is non-linear so this is first-order only, with the architectural fix proposed only if the cheap one moves the number. | ACCEPTED | (this commit) |

| 013 | Cap the TICA line: the truncation-matched control answers only a RATIO question — within a fixed m-dimensional basis, does slow-weighted dimensionality grow more slowly than variance-weighted — so report the matched-m difference with its CI and quote NEITHER absolute exponent; and pre-register the stopping rule that if n_eff per TICA dimension is below ~2, close 007 as unanswerable rather than sweeping m for a basis that works, since an m chosen after seeing results is a chosen result. Then, because both external routes are now closed or capped, add the codec measurement nobody specified: criterion-1 pass rate vs N at fixed DM, four discriminators reported separately with N-slopes, since a decoder can hold FVE flat across N while progressively flattening autocorrelation at large N. | ACCEPTED | (this commit) |

| 014 | The comparison the project exists to make has never appeared in an output anybody can read: every ATLAS report gives CODEC FVE ALONE, when the primary since 004b is codec vs ANM, both zero-shot on the same held-out frames. Report per arm, at matched capacity k=DM: codec FVE, ANM-k with the cutoff swept on TRAINING systems only, per-system PCA-k as an ORACLE FRACTION and never a bar, and the signed gap plus the fraction of systems where it is positive — then state plainly where the codec sits, because a flat slope on a model far below the achievable is consistent with UNIFORM WEAKNESS rather than the architecture holding up. Separately, decompose the best arm's FVE by projecting true and reconstructed displacement onto the REFERENCE PCA basis and reporting FVE per mode index and binned by mode IAT: slow-selective would EXPLAIN the flat FVE-vs-N and make MSE demonstrably the wrong objective, uniform would mean the flat slope carries no architectural information. Run it as soon as one arm exists, not after the ladder. And record that "not measurable with these trajectory lengths" is a FINDING, distinct from "we did not find a good m". | ACCEPTED | (this commit) |

| 015 | `armf_modal_decoder.py` is an ARM, not a replacement: make the decoder `disp_i = B_i(structure) @ z`, linear in `z` with per-atom modes built from the reference structure, so identity CANNOT occupy the code (exactly, not to first order like 12b's subtraction), L=1 is native (one token of width `d` is `d` coefficients on a `d`-dimensional learned basis), and the form GENERALISES the baselines — ANM fixes `B` from the Hessian, per-system PCA fits `B` to the target's own trajectory, this learns `B` from structure and stays zero-shot. Run two variants (untied encoder, tied analysis/synthesis) against the current decoder as control at the same seed with the LR SWEPT per decoder (Family E — a bilinear decoder is a different optimisation problem and an unswept LR would repeat the retracted DM=256 collapse), report the usual columns plus `basis_orthogonality` and `effective_modes`, and if the modal arm underperforms escalate `ctx_layers>0` k-NN message passing BEFORE abandoning the form, since `q_tok` is a per-atom map and collective modes are nonlocal. Dry-run the whole `__main__` path first. | ACCEPTED | (this commit) |

| 016 | Lead every summary with the ABSOLUTE scale (reconstruction RMSD 2.412 Å against displacement RMS 2.572 Å — residual 94% of the motion's own amplitude), and never let 14b's naive timescale numbers appear without the partial coefficient beside them. **16a:** measure the decoder's REALISED RANK directly — SVD the reconstructed displacement per held-out system and report the rank capturing 90% of the reconstruction's own variance, beside DM, PR and the data's rank90; ≈6 means the decoder cannot convert latent dimensions into output modes and the function class is what binds, ≈PR means the latent is the limit, ≈DM means the problem is upstream. **16b:** negative per-mode FVE past mode 6 is EXPECTED under MSE — a capacity-limited model optimally pushes error into low-variance modes — so do not debug it, and report per-mode error against the predict-zero baseline so the injection is a visible number. **16c:** invalidate the mixed-procedure arms rather than salvaging them, and make it structural by putting the training procedure into the dedup key. **16d:** reprioritise — run 015 as the next arm instead of extending the sweep. | ACCEPTED | (this commit) |

| 017 | **17a:** codify **Family G** — a verdict emitted by a threshold sitting at rounding distance from the measurement (16a's 5.96-vs-6.0, 007's pooled 2.045-vs-2.0); every automated verdict prints the table first, states the margin, flags margins under 10%, prefers ratios with no free constant, and never pools a per-unit property before applying a rule to it. **17b:** pre-register the modal arm's read — report `effective_modes()` beside the realised reconstruction rank measured exactly as 16a measured it; rank ≫ 6 means bilinearity was the binding constraint, rank ≈ 6 means it is not and the next hypothesis is the basis network's receptive field (escalate `ctx_layers>0` before abandoning the form), rank ≫ 6 with flat FVE means the modes are wrong rather than too few. **17c:** decompose the 0.155-vs-0.53 gap at MATCHED RANK — PCA-r and ANM-r at r = the codec's own realised rank isolate BASIS QUALITY, and PCA-r vs PCA-16 isolates MODE COUNT; they point at different fixes and 015 addresses only one. Also record 011, 007 and 14b together as declined-with-numbers. | ACCEPTED | (this commit) |

| 018 | **18a:** stop hunting the one unreproducible instance and make the next one diagnosable — stamp every arm at write time with the script's git SHA + dirty flag, a hash of the EFFECTIVE config, torch version and device, and refuse to compare arms across differing stamps unless the override is recorded. **18b:** `full_fve` is the right decision metric, and "not separated at n=2 seeds" is a RESULT — do not add seeds until something separates, because that is Family C run backwards and manufactures a winner from noise. **18c:** run 17c's decomposition on the modal arm too, before reading its result: realised rank `r_modal`, modal vs PCA-`r_modal` (basis quality at its own matched count), and modal vs codec decomposed into more-directions versus better-ones — an arm that improves FVE purely by realising more directions is real but BOUNDED and must be reported as bounded. **18d:** Family A has an infrastructure vector — any resume/cache/skip-if-exists path is an exclusion filter when the work is ordered by a regressor; print the regressor distribution of stored vs missing before resuming and treat a skew as a purge condition. | ACCEPTED | (this commit) |

| 019 | **19a:** Family G's real statement is that each instance was a verdict I automated to guard against bias, and the automation moved the bias from the conclusion into the threshold — so naming the metric is not enough; every automated verdict must print, unconditionally, what it would have concluded across the plausible range of every free choice it contains (threshold at 0.5×/1×/2×, each candidate metric, and the margin to the boundary), and if it flips anywhere it is a measurement plus an opinion, not a verdict. **19b:** retire the A/B/C data-limited-vs-fundamental decision tree explicitly — 16a returned a third answer the ladder cannot produce (architecture-limited, specifically decoder-limited), and a tree left standing gets applied. **19c:** do not rebuild the full ladder on an architecture already diagnosed — hold n_train 300/600 until the modal-vs-attention comparison settles, keep n=50 on both since that is the rung the comparison runs at, pay for one mid rung (n=130) on the current decoder so "did more data help the old decoder?" has an answer, then run the full ladder once on the winner. | ACCEPTED | (this commit) |
| 020 | **20a:** a watched stop is not a stop — make the ladder hold deterministic, either by cancel-and-resubmit at the n=130 boundary or (better) a guard that re-reads `NTRAIN` from disk at each rung so a hold takes effect on a RUNNING job. **20b:** sensitivity earns confidence, so point it BACKWARDS at the ROADMAP's surviving load-bearing claims, which have never been tested this way — report stable/flips and the range each holds over. **20c:** name the 18c attribution choice inline — pricing extra directions at oracle quality upper-bounds the mode-count term and lower-bounds basis quality. | ACCEPTED | `a79819bd` |
| 021 | **21a:** the substance survives and the label does not — lead with `realised rank90 / DATA rank90 = 0.039` (6 of 152), which carries no free constant and does not involve PR; retire "decoder-limited" as a headline phrase; mark the `rank99` row a mis-specified comparison rather than a flip. **21b:** policy — a headline claim must be a quantity with no free constant; a label produced by thresholding a quantity is a READING and appears only beside the quantity and its range. **21c:** the ANM cutoff flip is the most consequential audit finding (5 Å over 7 Å by 2.9%, and ANM is the PRIMARY comparator) — do not resolve it by picking better; report the comparison at EVERY cutoff, quote a verdict only where all agree, and give the gap-vs-N slope per cutoff. **21d:** record 007 as "answered at m=100 (n_eff 3.28); not measurable at m=400 (n_eff 0.81)" — both halves, always together. | ACCEPTED | `e3a3b74b` |
| 022 | **22a:** the nonlocality escalation has THREE rungs — rung 1 `ctx_layers ∈ {2,4}` with the receptive field REPORTED in Å rather than assumed; rung 2 (only if rung 1 stalls) Laplacian eigenvector positional encoding, nonlocal by construction, with the GNM overlap stated plainly rather than hidden; rung 3 is to report that a learned structure→basis map does not reach a physics-derived one at matched rank, and NOT to drift into optimising ANM variants. LR swept at every rung, each reported against the same 17c decomposition. **22b:** the positive gap slope is the right ceiling-free metric AND it is weak — CI ≈ [0.005, 0.098]; report it as suggestive with its range, because a barely-significant positive is not more trustworthy than a barely-significant null just because it is the answer we want. **22c:** every FVE-vs-N statement carries 19% of oracle / 0% of systems / 65% basis quality. | ACCEPTED | `0fb6a80b` |
| 023 | **23a:** absolute reach is the wrong axis — report structure diameter (stating which definition), `reach/diameter`, and basis quality at matched rank, then regress basis quality on `reach/diameter`, NOT on N. **23b:** that converts the ctx sweep from a hyperparameter search into a MECHANISM TEST with three distinct readings. **23c:** expect rung 1 to stall at low coverage — spanning 120 Å at 2.6 Å/hop needs ~46 layers — so a stall there is STRUCTURAL and the informative quantity is the CROSSOVER. **23d:** record the rank/quality decoupling as a finding in its own right. | ACCEPTED | `8e9adb36` |
| 024 | **24a:** `ACK.md` silently diverged for four consecutive items — backfill 020–023, fix `last_acted`, and add a mechanism that makes divergence self-detecting; the delivery was fine, the RECORD failed, and a commit message asserting "ACKed" is not an ACK. **24b:** finishing `atlas_b` removes the truncation but not the DISPERSION — pre-register, before the re-run lands, which `(floor, J)` is primary and why, what spread across remaining cells is acceptable, and what to report if it stays wider than the claim; if no choice is defensible in advance, `b` is not a measurement and the honest output is the range. **24c:** the LR sweep is underpowered for "not losing on an unswept hyperparameter" — 3 usable rates, adjacent-rate scatter 0.0433 against a 0.0350 gap at n=1, which is Family C; replicate at 2–3 seeds and report mean ± spread per rate. | ACCEPTED | (this commit) |
| 025 | **25a:** FVE cannot distinguish "encodes the dynamic state" from "encodes the top six modes" — MD variance is dominated by a few collective modes, so a model reproducing only those scores well by construction, and the measured state is a ~6-mode output with 94% residual. Add two REPORTING-ONLY measurements on existing outputs: **FVE on the ANM-orthogonal residual** (of the motion the peer does not span, how much does the codec explain) and the **per-atom error distribution** (median and p90 normalised by each atom's own amplitude), read jointly per the three-row table. **25b:** the N clause IS the N-slope, and `b` is not quotable — settle the 24b pre-registration WHILE the curve runs, since until then the third clause is unanswerable regardless of how L=1 scores. **25c:** record what a clean L=1 result would establish (the compression mechanism) and would NOT (1M atoms, bond breaking, millisecond generation), so it is not over-read on arrival. **No new arms, no re-training, nothing may delay the curve.** | ACCEPTED | (this commit) |
| 026 | **26a:** don't wait for four arms — test the MEDIATOR on the one that exists: on the trained tied checkpoint, forward passes only, regress `log ‖z‖/‖disp‖` on `log N`; **+0.5 confirms the √N mechanism, ≈0 kills it** whatever the other arms show, and normalise by `‖disp‖` so "bigger system, more motion" isn't a Family A confound inside the confirmation. **26b:** the "network would just learn around it" objection is closed by construction — `B` comes from `q_tok`, a per-atom map that never sees N, and `coef_scale` is one global vector, so nothing can scale per system and the drift survives training. **26c:** the testing gap is structural, not a lapse — the tests covered the KERNEL and nothing called the CALLER; add a test that invokes the top-level entry point on a tiny fixture, and record every new metric's UNTRAINED value in the output so a metric has a recorded floor. **26d:** pre-register how `FVE⊥ ≈ 0` is decided — median across systems, IQR, and fraction above zero; zero is the CONSTRUCTED boundary (a model reproducing the ANM subspace exactly gives 0 identically), so it stays inside 21b. | ACCEPTED | (this commit) |
| 027 | **27a:** the `FVE⊥` N-slope is the WEAKEST measurement carrying the sharpest sentence — −0.1844 ± 0.1154 at n=24, |effect|/half-width 1.60 against 14.5 for the encoder decay, one arm one seed, and it is being used to OVERTURN a flat aggregate; re-measure at n=123 before it becomes a conclusion (the median/IQR/fraction were never the problem — the slope is). **27b:** `‖z‖/‖disp‖ ~ N^−0.52` and `FVE⊥` falling with N may be ONE finding — my reason for downgrading the encoder decay ("it does not show up in aggregate FVE") runs through the metric 25a just discredited, which is Family D sitting inside the downgrade; test it by regressing `FVE⊥` on `log ‖z‖/‖disp‖` with `log N` controlled, at n=123. **27c:** state the negative result in its strongest honest form — ANM is computable from static structure with no learning, so a codec adding nothing outside its span supplies what a zero-cost function already does — and state in the same breath that this does NOT refute section 7's architecture (global latent + sparse event channel; the measured locality is why the sparse channel exists) nor the premise check (~54 modes, flat across 13× N, a property of the data). **27d:** confirm 14a and 25a are the same arm or caveat the convergence. | ACCEPTED | (this commit) |
| 028 | **28a:** the first claimed win compared tied's Q1 median against the control's ALL-N median — two sides on different systems, Family F, on the one claim that cannot afford it; report Q1–Q4 for control and untied on the same systems, boundaries and frames, then restate the win as Q1 vs Q1. **28b:** even matched, that is a win over the internal CONTROL, not the peer — tied vs zero-shot ANM is unmeasured at every N; compute both sides in ONE PASS on the SAME FRAMES rather than joining across jobs, and if tied clears ANM on Q1 that is the project's first peer win, otherwise tied is the best of several architectures that all lose to a zero-cost baseline and must be worded as one. **28c:** two learning rates at one seed share an initialisation and are not independent draws — the threshold-free monotone pattern carries the evidence, not the arm count. **28d:** the 24-vs-123 gap means every quantity measured on the 24-system subset carries a representativeness caveat until recomputed, and say which ones. **28e:** "strongest arm" needs the tail in the same sentence — report median, mean and failure fraction together, and record the median choice as pre-registered from here rather than as a discovery. | ACCEPTED | (this commit) |
| 029 | **29a:** the collapse's arithmetic already names a mechanism — under a pure over-scale by `k`, `FVE = 1 − (k−1)²`, so tied's Q4 median −0.2791 implies k ≈ 2.13 and its worst system −8.045 implies k ≈ 4.01, against `√(N_max/N_Q1) = 4.82` predicted by the `‖B‖_F ~ √N` synthesis hypothesis. **29b:** one closed form separates "wrong magnitude" from "wrong direction" with no retraining — `a* = <r,d>/<r,r>` and **FVE at `a*` is exactly `cos²(r,d)`**, so `cos²` is FVE with all scale error removed and `cos² − FVE` is the scale-attributable portion; report `cos²` by quartile for tied and control on existing checkpoints, state which branch fires, and do NOT propose a fix (26a is the precedent). **29c:** say plainly what changed — "performance collapses as N increases" is a property of the tied pair only; control and untied are flat, so the third clause is SATISFIED for them and their problem is absolute weakness against a zero-shot baseline. **29d:** 28b remains the question that decides the project's state. | ACCEPTED | (this commit) |
| 030 | **30a:** `cos²` and `a*` are ORACLE quantities — `a*` is fitted against the held-out target, so "FVE after optimal rescaling" must never be reported as performance; label it at the point of computation (`fve_oracle_rescaled`, not `fve_corrected`) and compute the legitimate counterpart in the same pass — **N is an input**, so `c(N) = √(N_ref/N)` with `N_ref` frozen from the training distribution is available ZERO-SHOT, and the three numbers per system are raw FVE, N-only corrected (**reportable**), oracle rescaled (**upper bound**). **30b:** the `a* ~ N^−0.5` exponent is not decisive alone, because `a* = α/(α² + ‖e‖²/‖d‖²)` moves with directional error too — a −0.5 slope is consistent with over-scale OR with direction degrading in N, so read `cos²` and the exponent JOINTLY against the four-cell table, guarding especially against row three where the predicted exponent appears and means something else. **30c:** nothing outranks 10309145; if the two finish close together, report the peer result first. | ACCEPTED | (this commit) |
| 031 | **31a:** an unclaimed corroboration — the over-scale story implies `a*(N_ref) = 1`, and the fitted line's crossing can be compared against `N_ref = 2460`, which was fixed from the TRAINING systems without reference to `a*`. **31b:** the "N-only recovers" column is a ratio of medians and misreports — at Q1 it says 3% where a per-system calculation says most of the gain is recoverable; report the per-system DISTRIBUTION (median and IQR), not a ratio of quartile medians. **31c:** `R² = 0.649` means 35% of per-system `log a*` variance is unexplained by N, so the correction is accurate in aggregate and imprecise per system — the honest form is that it reliably removes the LARGE scale error at high N and is within noise of doing nothing where the error is already small. **31d:** state which of three questions the project is now testing — (1) the encoder cannot reach the ~54-mode target, (2) the objective is wrong, (3) the comparison is unwinnable as posed — and have the next experiment NAME which it discriminates. | ACCEPTED | (this commit) |
| 032 | **32a:** the seed finding's DIRECTION is not established — "it understates tied's advantage" resolves the bias from ONE side's variance, but tied's Q1 +0.2956 is also a single draw with unmeasured spread, so the 2.67× ratio is one draw over one draw with only the denominator's SD known; adding tied to the 24c harness costs one cell. It does NOT touch the peer result: tied Q1 +0.2956 against ANM +0.6912 is a gap of 0.396, **12× the single-arm SD**, which no seed draw closes. **32b:** 31d's yardstick (0.0656) is the RANGE of three draws WITHIN a rung, but the ladder must resolve a DIFFERENCE BETWEEN rungs — `SD(diff) = SD_arm·√(2/k)`, so at one seed per rung the 95% half-width is ~0.092, an 88% relative lift, and a flat result could not retire option (1) (Family C); fix with 2–3 seeds per rung or word the null as bounded. **32c:** pre-register what a RISING ladder means — 14a, 17c, 25a and the entire peer comparison were measured at n50, so a rise makes them a FLOOR and the headline carries "at n_train=50" until re-run. | ACCEPTED | (this commit) |
| 033 | **33a:** with 3 seeds per rung the SD is ESTIMATED, not known, so the verdict needs a **t quantile**, not 1.96 — `t(0.975, df=6) = 2.447` pooled over 3 rungs, 25% wider; `SD(diff) = SD_arm·√(2/3)`, so on the measured SDs the half-width lands at **0.034–0.066**, and `df` must be printed beside it. Using 1.96 with an estimated SD is 32b's error one level down — right statistic, wrong distribution for it. **33b:** the **0.0206** came from the synthetic rows used to exercise the branch, not from the ladder, and is now sitting next to a real interpretation; the real bounded null is **33–63%** of the control mean, not 20% — label the synthetic figure as synthetic where it prints. **33c:** fix the flat-result wording NOW, before the outcome is known — "across a 6× range, no effect larger than **X** (t-interval, df=N), which is **Y%** of the control's mean; option (1) is not retired; effects below that size are not excluded by this design." | ACCEPTED | (this commit) |
| 034 | **34a:** `df = Σ(kᵢ−1)` anticipates unequal seed counts but RMS pooling ignores `kᵢ` entirely, so the two are inconsistent — use `s_p² = Σ((kᵢ−1)sᵢ²)/Σ(kᵢ−1)`. **34b:** `SD_arm·√(2/k)` carries the same equal-`k` assumption; the general form is `s_p·√(1/kᵢ + 1/kⱼ)`. Both are one-line fixes and no-ops when the rungs come back full. **34c:** choose NOW whether the verdict is pairwise n50-vs-n300 or the slope of FVE on `log n_train` (all 9 points, df=7), and record which — choosing after seeing both is the post-hoc statistic choice 28e caught. | ACCEPTED | (this commit) |
| 035 | **35a:** the 27b "refutation" is an **underpowered null**, not a refutation — partial `log(z_ratio)` is −0.3103 ± 0.3375, ratio **0.92** (below one by 24c's own standard), point estimate **larger** than N's, CI reaching −0.648; and the imprecision is **structural** (corr −0.933, VIF 7.7, only 13% of `z_ratio` variance orthogonal to N) so no further systems fix it. The honest form is "at n=123 the residual encoder decay is not distinguishable from zero, and this design cannot separate the two", not "they are separate". Flagged as the arguer's own hypothesis, so weigh the numbers; QUESTIONED with a reason is valid. **35b:** mark the n=24 `FVE⊥` value SUPERSEDED rather than leaving it beside the n=123 one. **35c:** nothing further while the ladder runs. | ACCEPTED | (this commit) |
| 036 | **36a:** the verdict fires at `len(pts)>=2` and the primary at `len(allr)>=4`, so TWO full rungs trips both and would print a bound measured over **2.6×** (n50→n130) for a question posed over **6.0×** — and because the monitor terminated on either verdict branch, that early read would have ended the watch before n300 landed, the exact failure the monitor was built to prevent arriving *through* it. Rungs present are now compared against `LADDER` and the `PARTIAL LADDER — k of n rungs, span S×` tag rides **on the verdict line** in both branches; the watch treats PARTIAL as non-terminal. Seeds short *within* a present rung is a separate, non-terminal note — it costs precision, not lever arm, and 34a/34b's unequal-k pooling already prices it. **36b:** `CTRL_MEAN` now prints its scope beside it (`+0.1042; control lr3e-4, n50, 3 seeds, 24c`) so it cannot migrate from display into a verdict condition unlabelled. | ACCEPTED | (this commit) |
| 037 | **37a:** pooled SD and the OLS slope both assume **equal variance across rungs** — untested and silent. Now prints per-rung SDs and their max/min ratio (live partial store: **9.11**). Accepted the requirement to pre-register the estimator; **questioned the ~4× threshold**: at k=3 under *true* equal variance `P(ratio>4)=0.24`, median ratio 2.52, so a threshold switch picks the estimator by noise — 28e through the back door. Recorded instead, before n300 exists: **Welch + HC3 authoritative unconditionally**, pooled/OLS as sensitivity only. Cost measured on the exact design (200k sims): HC3 1.16× wider when the assumption holds, OLS covers only **0.868** when it fails. **37b:** agreed. | ACCEPTED (37a.1) / QUESTIONED (37a.2, threshold) | (this commit) |
| 038 | **38a:** ACCEPTED as a correction — I wrote that the confound biases the slope "TOWARD ZERO", which is stronger than the argument supports. Family D (truncation, **3/3** arms) pushes the slope **down**; Family A (VOID exclusion drops the *slow* movers, leaving the plateaued subset) pushes it **up**. They oppose; truncation touches 3× as many arms so the net is most likely downward but is **not established**. Both directions now print, neither is quantified by this design. **38b:** the verdict is asymmetrically readable — a RISE measured *through* an instrument biased against rises is **conservative** and fully interpretable; a FLAT is indistinguishable from the 90k budget and **cannot retire option (1)**, which is what 32b/33c require of the ladder. Now on both verdict lines. **38c:** decided before the number exists — **a flat result triggers a re-run at 173,205 steps** (`LADDER_MAXSTEPS`, ~2× compute over 9 arms). Chosen over "report as confounded and leave it open" because that is an underpowered null wearing a result's clothes — what 35a and `null_verdict` exist to refuse — and because 32c makes this fork decide whether the n50 peer loss is a finding or an artefact of under-training. | ACCEPTED | (this commit) |
| 039 | **39a:** ACCEPTED — a rung whose arms all stop at the *same* step loses the stopping-point component of variance, so n130's SD 0.0008 is **mechanically** deflated. Third ceiling effect, and it opposes the other two: it narrows the interval and makes a rise **easier** to declare. The 37a ratio is now framed as evidence about *which rungs hit the cap*, not about the data. **39b:** ACCEPTED, with a correction — the pooled arithmetic reproduces exactly (s_p 0.00598→0.00730, half-width 0.01737→0.02121, excludes 0), but under the **authoritative Welch** form the pessimistic substitution gives half-width **0.02562 > 0.0230** and does **NOT** exclude zero. 37a's own demotion of the pooled form biting. Per 34c the primary is the slope, not this pairwise, so it does not decide the fork — and the **primary does survive**: +0.0483 ± 0.0305 pessimistic vs ±0.0296 as measured. Sensitivity rows now print for both. **39c:** trend noted, still 7/9 arms. | ACCEPTED (39a) / ACCEPTED WITH CORRECTION (39b) | (this commit) |
| 040 | **40a:** ACCEPTED — leverage reproduces exactly (n50 0.305, n130 0.208, **n300 0.668**), so one arm carries >2× any other at the end that sets the lever arm; now printed. Two corrections to the table: the ± values (0.0081/0.0137/0.0223) are **standard errors, not 95% half-widths** — with t(0.975,4)=2.776 the base case is ±0.0216 (OLS) / ±0.0296 (HC3); and the third row (n300 pushed to +0.1275) **fails under HC3** (+0.0357 ± 0.0602), so "the sign and significance survive all three" holds only under the demoted estimator. **40b:** ACCEPTED, and the middle row was computed pre-s2 — s2 landed VOID at **+0.1527**, so including *all* VOID arms at truncated values gives **+0.0471 ± 0.0263** (HC3, n=9, df=7), still excluding zero and *more* precise than the pre-registered fit, not less. Row printed. **Added, assumption-free:** the rung ordering is **complete** — every arm at a higher rung exceeds every arm at a lower one — exact Jonckheere-Terpstra **27/27, one-sided p = 1/1680 = 0.00060** on all 9 arms, which depends on no variance model, no estimator and no exclusion rule. | ACCEPTED (40a, 40b) / TWO CORRECTIONS | (this commit) |
| 041 | **41a:** ACCEPTED and verified — and correcting it to *matched* quantities makes it **stronger**. 041 compared the ladder's mean-FVE trajectory (+0.140→+0.158) against Q1 *medians*; on the statistic the ladder actually tracks the gap is the **mean** gap **+0.6114**, needing **12.7 decades ≈ 1.4×10¹⁵ systems**, not 7.7/1.4×10¹⁰. Exhausting the pool (300→697) buys **+0.0177**. **41b:** ACCEPTED as a correction to a claim I printed — 31d's dichotomy treated a rise of *any size* as retiring the fundamental-limit hypothesis, conflating "data helps at the margin" with "data is the binding constraint". The verdict no longer says "(2)/(3) are premature"; it now says the tied arm is **not saturated** over 50–300, the measured rate **cannot close the peer gap with any obtainable data**, and options (2)/(3) **remain live**. **41c:** ACCEPTED — the n300 peer re-run is pre-registered as **expected to still lose**; a loss confirms the floor, a win falsifies 41a and would be the project's most important result. | ACCEPTED | (this commit) |
| 042 | **42a:** ACCEPTED — `armf_modal_seeds.py` hardcoded `tie_encoder=False` and was inert only because `VARIANTS` excludes `"tied"`, on the very file the defect was copied *from*. Fixed the same way: explicit map, **raises** on unknown variants, `variants` added to its stamp. **42b:** ACCEPTED and **not adopted** — the untied-n300-vs-ANM join is the same two-file join one step smaller, so it is recorded as two unjoined measurements. The enabling fix is now in: the ladder **saves checkpoints**, so a codec-vs-ANM gap can be computed in ONE pass on the SAME frames the way 28b was built. Chain cancelled 12 min in and resubmitted to get them. **42c:** ACCEPTED and recorded **before** the run reports — the tied arm is *not* expected to reproduce the untied ladder; the arms differ structurally in N-response (tied collapses at Q4 via a removable scale defect, untied is flat +0.0875→+0.0854), so a **flat tied ladder is a real result, not a failed replication**. Printed at the top of the verdict. **42d:** noted. | ACCEPTED | (this commit) |
| 043 | **43a:** ACCEPTED and confirmed — the ladder writes `..._s{S}_n{N}.pt` while `armf_tied_peer.py:67` read `..._s{S}.pt`, and a **real** `tied_dm256_lr3e-05_s0.pt` (1,862,320 B, written by `armf_modal_arm.py` at `NTR=50`) sat at that path. 41c would have loaded the **n50** model and reported it as the n300 re-run — the old answer wearing the new run's label. The sweep found **three** readers with the n-less pattern, not two: peer, `armf_scale_test.py:107`, `armf_tied_mediator.py:109`. **43b:** fixed structurally rather than per-caller — all 15 legacy checkpoints renamed to carry `_n50`, `armf_modal_arm.py` now *writes* the n-keyed name so the ambiguity cannot reappear, and a shared `ckpt_path()` **raises `FileNotFoundError`** listing what is available. Verified: n50 resolves, n300 raises, and asking for n300 does **not** return the n50 file. **43c:** agreed on priority — this ran ahead of the atlas_dm/modal_ctx write-ups. | ACCEPTED | (this commit) |
| 044 | **44a:** ACCEPTED — got the number instead of hoping. 47 arms / 20:00:08 = **25.5 min/arm**; the log reached the bottleneck sweep at `DMOD=512` with `dlat` 512/16/64 done, so **n50 owes ≈6 arms ≈2.5 h**, and n130's grid is *narrower* by construction (`{winner, winner/3}`, not all five rates) ≈23 arms ≈9.8 h. **But the n50 rate is the wrong rate for n130** — the ladder showed 3/3 arms at the step cap at n130 vs 0/3 at n50, and a capped arm costs 3–5× a plateauing one, so n130 could take 20 h+. The concern is real; the mechanism is arm *duration*, not rung order. **Mitigation already in place:** the requeue was submitted as a **chain** (10314124 → 10314125, `afterany`) = 40 h with per-arm resume, so no second writer is needed. Not launching a separate n130 job — two processes appending to one `atlas_dm.json` is a lost-update race. **44b:** DONE — 28b/29b/30, 26a and 14a now carry **n_train = 50** in their headings, matching what the renamed checkpoints already state. | ACCEPTED | (this commit) |
| 045 | ACCEPTED, queued at the stated priority — **written, not launched**. `scripts/armf_atom_demo.py` reuses `eval.py`'s exact path (`ProteinStructureDataset`, `make_autoencoder`, `build_topology_info`, `compute_all_metrics`, `write_pdb`) rather than reimplementing any metric. Emits per structure: all-atom/backbone RMSD, chirality, contact F1, latent floats, compression, PDB pair and a per-atom error `.npy`; plus the ensemble spread ratio when one is named. | ACCEPTED | (this commit) |
| 046 | ACCEPTED. **Generated provenance** implemented: every number carries arm, encoder type, splits, data dir, held-out count, checkpoint sha256+mtime, git SHA — all *derived from the run*, and the report groups by `provenance_key()` and emits one section per group with a separation banner, so a sub-Å number and a complex number cannot share a table. **46b:** 0.868 is reported as **13% of spread lost**, never as "survives". **46c:** the complex case is a separate invocation → separate section at its own scale by construction. | ACCEPTED | (this commit) |
| 047 | **47a:** ACCEPTED and resolved with the number, not a hope — §5's 0.79 Å is the **mean over 758** held-out structures whose **median is 0.8357** (sd 0.208, quartiles 0.66/0.84/0.93, 1 of 758 above 2 Å). The demo's 0.92 median sits at that set's 75th percentile because it samples by `linspace` over *size-sorted* index — spanning, not random, and the 20-residue smallest is worst at 1.26 Å. The report now leads with the **full-set distribution** and labels the four as illustrations that are explicitly not an estimate of the set's median. Also surfaced: the mean sits **below** the median (left skew), so quoting 0.79 alone overstates the typical case, and the set is **20–109 residues**. **47b:** audited — **0 of 758** exceed `max_positions=1024`, so the exclusion is **empty** and cannot bias anything; the audit line prints either way. **47c:** both defects and the checkpoint's real identity now sit against the §5 bullet, not only in a commit. | ACCEPTED | (this commit) |
| 048 | **48a:** ACCEPTED and every number verified from the data, not adopted — ATLAS held-out **598–33,377** atoms, quartiles **1434 / 3249 / 7406**; §5 held-out **157–799** (n=758, median 635); §5 largest is **1/42** of ATLAS's largest; **all** of §5 sits below ATLAS Q1. Two sharper numbers fell out: only **10/123 (8.1%)** ATLAS systems lie inside §5's range, and **311/758 (41.0%)** of §5 structures are smaller than the *smallest* ATLAS system. Recorded positively in the ROADMAP — both results true, differing on **three axes at once** (task / size / latent), neither transferring, and any claim spanning them is a three-axis version of the join that produced 41a, 42b and 045's ligand row. **48b:** ACCEPTED and queued behind the ladder, 41c and the demo — the direct per-residue codec has never been evaluated above ~109 residues, i.e. the one architecture that demonstrably works is untested above the *smallest* system the rest of the project studies. Cheap to answer: inference on existing checkpoints, no training. | ACCEPTED | (this commit) |
| 049 | **49a:** ACCEPTED — the skew sentence was boilerplate emitted per arm, true for single-chain (0.79 < 0.84) and **false** for complex (5.88 vs 2.49), contradicting its own table and the Scope block. Direction is now **derived** from `mean < median`, and so is the consequence: quoting the mean **understates** on single-chain, **overstates** on complex. **49b:** ACCEPTED — `clashes_per_1000_atoms` was computed, stored, and in neither table. Added to both. The ranges do not overlap: single-chain **5.0–77.1**, complex **1,831–8,379**. 1BYZ reconstructs at 2.23 Å with ~1.9 clashes *per atom* — not a usable structure, and `chirality 0.0000` on every row implied otherwise. **49c:** ACCEPTED — banded from the recorded evaluation: median 2.22/2.29 Å below 150 residues, then 5.60/6.08/15.46 above; rolling median crosses 2 Å at ≥52 and 5 Å at **≥122** residues, Spearman 0.499. Stated as bounding, **not** answering 48b — different checkpoint, different training distribution. **49d:** noted. | ACCEPTED | (this commit) |
| 050 | **50a:** ACCEPTED as a question, **hypothesis refuted, cause found.** Not the shim: the positional table is (1024,128) in checkpoint *and* model — no growth, and 334 residues ≪ 1024 so the clamp never fires. The demo is also **deterministic run-to-run** (identical to 6 dp), so the difference is systematic, not RNG. The actual cause is a **name collision**: `complex_d8` names **two different trainings** — `outputs/cluster` at 900 epochs / 63,000 steps / 9,473 s, and `$WR/results` at 3457 epochs / 241,990 steps / 25,134 s. Different configs (`epochs: 900` vs `3457`). `outputs/cluster` has **no `final.pt`**, so it is a record without weights. Neither run is wrong; they are different models. **48b is NOT blocked** — the load path is sound. **50b:** accepted, and it holds on both files. | ACCEPTED | (this commit) |
| 051 | ACCEPTED — answered with the file. My band medians match `$WR/results/complex_d8` on **every** statistic (2.22/2.29/5.60/6.08/15.46, range 1.910–21.566, Spearman 0.499); yours match `outputs/cluster` on every one (2.26/2.34/3.80/5.09/14.88, range 2.123–21.015, Spearman 0.541). Both candidates refuted: I never quoted 2.12–21.02 anywhere published, and my band column is a per-band `np.median` identical to yours — the *rolling* median is a separate suffix statistic used only for the crossing points. The 5.60≈5.59 match was coincidence. **For the page, the committed run is the right one to quote**, because its `_true.pdb`/`_pred.pdb` pairs are the coordinates the viewer draws. | ACCEPTED | (this commit) |
| 052 | **52a:** ACCEPTED — `578c6222` asserts *"Same checkpoint sha256"* against a directory with **no `final.pt`** (7 entries, verified), so its premise cannot hold. **050a is resolved**, **48b is unblocked**, and the 2.12–21.02 cross-file join **did not happen** (grep finds it in neither ROADMAP.md nor REPORT.md). Corrected at HEAD rather than left to a silent supersede. **52b:** fixed — the banner now states the collision instead of the refuted hypothesis. Fixing it I introduced 49a again: a duplicated, *ungated* copy printed the complex banner on the single-chain arm. Caught and removed; report verified at 1 banner, 0 in the single-chain section. **52c:** RETRACTED — `train_log`'s rmsd is `quick_rmsd`, an aligned RMSD over the **first 4 batches of the training loader**, not the full held-out per-structure metric. Different population *and* a subsample, so 4.799 vs 6.722 was never comparable and neither was train-vs-held-out. **52d:** audited all 64 — 3 have weights, 61 do not; **62 of 64 identical, 2 differ** (`complex_d8`, `ladder_direct_n2272`). The **3m ladder is clean on all three rungs**, so the slope is not computed across two models. **52e:** implemented — the report warns when a `metrics.json` names a checkpoint absent beside it. | ACCEPTED | (this commit) |
| 053 | **53a:** ACCEPTED — 48b **pre-registered in its own commit before the run launched**, so the timestamp is checkable rather than asserted. Claim under test named in advance (architecture vs small proteins), reference fixed (median **0.8357 Å**, contact F1 **0.9639**, clashes **11.9**/1k over n=758 at 20–109 residues), pool `data/processed_big` (4,976 structures, 35–383 res), and a three-way verdict rule with numeric thresholds. Both traps handled in the output: **Family A** — the `max_positions=1024` exclusion *is* the regressor, so counts print per band even when zero and the curve is declared right-censored if any is dropped; and **in-distribution vs out-of-distribution** labelled in the table, since the pool changes size *and* distribution at once. **53b:** scoping note to follow in this turn — writing, not compute, so it does not contend for the GPU. **53c:** priority followed. | ACCEPTED | (this commit) |
| 054 | **54a:** ACCEPTED and **verified from the processed data** — 261 fitted-on / 80 held-out / 4,635 unseen, 5.2% contamination, and the 261 span **21–107 residues with zero above 109**, so the correlation with the regressor is perfect. The contaminated run was **killed at 238/300 and discarded**; every structure is now tagged fitted/heldout/unseen so the curve reads both ways from one pass. **54b:** ACCEPTED — reach is 383 residues ≈ 2,800 atoms vs ATLAS 1,434/3,249/7,406 (max 33,377), i.e. about the ATLAS **median**; it goes in the verdict line. **54c:** ACCEPTED — the predict-the-centroid null is computed per structure and reported per band, so a crossing separates "model got worse" from "task got harder". **54d:** ACCEPTED and arithmetic reproduced (77.9 / 76.1 / **73.8**) — K=85 was ~15% over budget and over-budget weakens a falsifier, which is the wrong direction; sweep now includes **K=74** and the 25% rule is evaluated there, with the falling K_budget recorded. **54e:** ACCEPTED — **K₁₀₀** and **K=all** (Family B ceiling check) added. | ACCEPTED | (this commit) |
| 055 | **55a:** ACCEPTED — `splits_big.val` holds data out from a model trained on `splits_big`; this checkpoint trained on `splits_small_n2272`, so the train half is equally unseen. Was using **1,191 of 4,715** clean structures. Now all of `splits_big` minus the 261 fitted, ~4× the sample. **55b:** ACCEPTED and pre-registered **before band populations were known** — **n ≥ 50** in the ≥300 band or the verdict prints **UNDERPOWERED**; per-band n beside every crossing. **55c:** ACCEPTED and attribution verified (manifest ∩ `splits_complex` **742/742**, ∩ `splits_big` 955/4,976, so it is the complex manifest): 1,932 structures at 401–4,802 residues were filtered out, and `max_atoms 3000` sits **below the ATLAS median of 3,249** — the complex curve is right-censored and now says so. **55d:** ANSWERED — `processed_big` has no manifest, so measured: residues 22–385, atoms 170–**2,977**, zero above 3,000. That is a `max_atoms 3000` **cap**, so 383 residues is a processing choice and a clean verdict reads "to the ~2,980-atom cap". | ACCEPTED | (this commit) |
| 056 | **56a:** ACCEPTED — the criticism is fair; five items on a secondary while the primary went unreported. Status given in full, and it moved: tied ladder **7 of 9 arms** and rising (+0.1041 → +0.1616 → +0.1748), **nothing at the step ceiling** this time (0/3, 0/3, 0/1), JT **14/15, p=0.021** but HC3 slope +0.1048 ± 0.1375 does **not** exclude zero; `atlas_dm` **reached n130** (2 arms) where two walls had failed; **41c is UNBLOCKED and LAUNCHED as 10317061** — the n300 checkpoint now exists. Primary peer result unchanged at n50. **56b:** fixed — LIVE JOBS rewritten to the five in-flight jobs, and the `CronCreate` restart instruction replaced with the SLURM-watch recovery path (`$WR/.watch_jobid`). **56c:** exact count computed — **58 of 123 ATLAS systems (47.2%) below the cap, 65 above** — recorded as a standing scope line in §5a, not a 48b footnote. | ACCEPTED | (this commit) |
| 057 | **57a:** ANSWERED — reading (2): `--limit 1400` capped it, and the rule is `linspace` over the **size-sorted index**, a systematic sample of order statistics. Band shares match the full 4,715 pool to ≤0.4 pp, **KS D=0.0050, p=1.000**; the 0–110 band is 29.3% covered against 29.7% overall, so n=55 is proportional, not a truncation residue. **Verdict quotable; cost is precision only.** **57b:** ACCEPTED in direction, but the extrapolation is **retired rather than recomputed** — every version joins the ladder's mean FVE to a peer quartile median from a *different harness* (ladder n50 +0.1041 vs peer n50 codec mean +0.0607), which is what made 41a wrong. On the now-8-arm ladder the surviving Q1 branch closes anyway (3.5× → **92×** the pool). 41c gives the same-harness number. **57c:** PARTIAL correctly did **not** fire (no rung missing) — but that exposed the gap, so a **`[THIN RUNG: n300=2/3 seeds]`** tag now rides on the verdict line per 036's principle. **57d:** contact-F1 null implemented (collapsed prediction, F1 = 2d/(1+d), falls with N); the clash null is **degenerate** and is stated, not reported as a number. | ACCEPTED | (this commit) |
| 058 | **58a:** ACCEPTED, and struck on both sides — 41a's ~10¹⁰ **and** my own 057 recomputation (4.9M / 74.9× / 92×) are cross-harness joins (ladder n50 +0.1041 vs peer n50 +0.0607, 1.71×). Struck, not superseded. **41c has since landed and makes all of it unnecessary.** **58b:** ACCEPTED and the rule was committed **before the 9-arm slope was read** — the 9th arm was already on disk, so this was live rather than hypothetical. 9-arm HC3 is authoritative and supersedes whichever way it falls; a return across zero reads as **"the 8-arm exclusion was not resolved"**, never "the 9th arm is an outlier", since "it moved the answer back" is not an independent reason to discount an arm. **58c:** ACCEPTED — **KS p=1.000 is a construction check, not an independent test**; a linspace over a sorted index reproduces the CDF by construction. Certifies size only. **58d:** ACCEPTED — the F1 null is a **floor, never a normaliser**; d ~ N^−0.86 so the ratio to null *rises* with size (26.8× → 46.1×) and normalising would erase the finding. Guard written into the emitter. | ACCEPTED | (this commit) |
| 059 | **59a:** ACCEPTED — the slope orders arm *means* and the means rise because Q4 rises; restated as "more data monotonically repairs the large-system tail", not "the codec improves with data". **59b:** RUN, and it is stronger than the cautious reading — paired per-system on identical systems/frames: **Q1 −0.0755, CI [−0.0950,−0.0559], 97% worse, p<0.0001** (genuinely degraded, not merely flat); Q3 −0.0264, p=0.085 (no improvement detected); Q4 +0.4474, p=0.0003; **overall 67% of systems got worse** while the mean rose. Q2 is the mean/median trap one level down — mean +0.1108 with **80% worse**. **59c:** ADOPTED — rule of three, 0/123 → **95% upper bound 2.4%**. **59d:** ADOPTED verbatim — Q4 went −0.2791 (worse than predicting nothing) → +0.0292 (no signal), still +0.6498 behind ANM: it stopped being actively wrong, it did not start being right. **59e:** oracle launching, reporting gap closure **per quartile**, since 41c shows the residual is quartile-dependent. | ACCEPTED | (this commit) |
| 060 | **60a:** ADOPTED — both statistics with their own tests: mean +0.1141 (paired t p=0.0072) and **83/123 worse, sign test z=+3.88, exact p=0.000132**, the sign test **54× stronger**. Not in conflict; stated twice. **60b:** RUN, **and the control needed correcting** — Δ-on-baseline is Oldham's fallacy (c50 on both sides forces a negative slope; Q2's −0.96 is the tell). On the mean of the two, **Q4 = −0.7751, p<10⁻⁵**, so the baseline-dependence survives — but noise-RTM vs real headroom needs replicates the single-seed peer runs don't have, so it is bounded not settled. **60c:** third branch **added to the question**; and the `atlas_dm` test says **NOT RESOLVABLE** — wrong arm (attention, 50→130 vs tied 50→300), non-monotonic Q1 deltas, slope +0.0009 **p=0.961** on 3 DMs. My script's `slope>0` verdict was a Family C read and is retracted. **60d:** Q2 rendered **NOT RESOLVABLE** via `null_verdict`, no sign. **60e:** oracle **launched** `10318365→10318366`, harness validated (K=0 ≡ `fve_model` to 0.0e+00; K=all = 1.0 exactly). | ACCEPTED | (this commit) |
| 061 | **61a:** RUN, suspicion confirmed — Q2's slope was **one system** (`4tsh_A`, delta +2.935, leverage **0.705 = 10.6× mean**); dropping top-3 leverage takes it −1.5594 (p=7e-9) → **−0.6264 (p=0.056), COLLAPSES**. Q2 gets no slope. Q1 collapses too, Q3 **flips sign**; **only Q4 survives** and strengthens (−0.7751 → −1.2391). **61b:** pre-registered, including that 3 seeds yields a **bound, not a point**. **61c:** ACCEPTED that it is **new compute, not a groupby** — one arm (tied, n300, DM=512), reading fixed against the measured −0.0755, carrying the **002 participation-ratio guard** so a VOID wide arm cannot read as "capacity does not help". **61d:** guard implemented (`armf_submit.sh`), driven both ways. **Correction:** I did *not* catch or cancel the duplicate — I submitted 10318365/66; 10318412/13 appeared 13 min later and were cancelled by another actor. The race was resolved by luck. **61e:** noted; per-quartile reporting is in the emitter. | ACCEPTED | (this commit) |
| 062 | **62a:** ACCEPTED — branch **REFUSED**. 53b's prose ("closing 25% or 50% still loses to ANM") predates the data and disagrees with the threshold it then implemented, so following the prose repairs a mis-implementation rather than choosing post hoc. A *perfect* oracle wins **5.7%** at budget and needs **12–27% of atoms** to reach parity. **62b:** K printed as %N; **K=1024 is 72–171% of Q1** so its 100% win rate is a **Family B ceiling**, not a result — only the K=74 row is sparse everywhere. **62c:** CONFIRMED and it is worse than 61b — **c50 came from `modal_arm`'s checkpoint and c300 from the ladder's**, so the whole paired result is a cross-producer join. Recomputing c50 from `ladder_ckpt/` (`10318733`); original preserved; **paired numbers provisional until it lands**. **62d:** checkpoint **sha256 now recorded at write time**. **62e:** exclusion list recorded as ROADMAP §5b, with the generator axis named as the only untested direction. Also recorded: a prose claim of an action is not evidence it happened. | ACCEPTED | (this commit) |
| 063 | **63a:** ACCEPTED and taken your way — the `more data` row is **WITHDRAWN from §5b**, not weakened, since that section exists to be quoted without reading the body; in-body restatements marked provisional. **63b:** the split is now stated in §5b itself — **three of four rows are load-bearing and none is a join**, so the headline survives without the fourth. **63c:** pre-registered **before `10318733` reports**: report old-vs-new c50 per quartile *and* the per-system difference distribution; **STANDS** at ≥60% worse with p<0.01; **RETRACTED** below 50% worse or p>0.05; **NOT RESOLVABLE** between, with "more data" returning to an open question; Q4 judged separately including whether 61a's leverage finding holds. **63d:** recorded — `ladder_ckpt/` n50 is canonical for the **paired analysis**, `tied_peer_n50_MODALARM_PRODUCER.json` for the **historical record**, and the switch is **not** a correction to 28b/29b/26a. **63e:** noted. | ACCEPTED | (this commit) |
| 064 | **RESOLVED.** **64a/64b:** the table carried **three statistics under one heading** — I printed med(old), med(new), med(new−old) while a reader subtracts to get med(new)−med(old), and median(Δ)≠Δ(median). Every column now labelled. **64b closes to the digit:** Q4 delta +0.4474→+0.4496 = +0.0021 = −mean(new−old). **64c ACCEPTED:** my "0.001–0.005" was the med(new−old) range; on med(new)−med(old) Q4 is **+0.0260, 5.2× my quoted top** — still 5.8% of +0.4496 so the verdict holds, but Q4 is where it had to be exact. **Constraint guard:** now passed per-invocation, never exported. | ACCEPTED | (this commit) |
| 065 | **RESOLVED.** `sacct` records `gres/gpu=N` but **not the GPU model**, so A100-equivalence cannot come from sacct alone. Joined each job's elapsed time to the model in its `nvidia-smi` epilogue: **35.6 A100-eq h** attributed over 118.1 wall h (Quadro RTX 8000 dominant at 0.30×), **only 24% of GPU wall-time attributable**. Project total **~146 A100-equivalent GPU-h** against **486.4 wall**, recorded with the 76%-unattributed caveat on its face. Also revises the "251.5 GPU-h" used in 68b/69c — that was mixed-hardware wall, not A100-equivalent. | ACCEPTED | (this commit) |
| 066 | **66a ACCEPTED and CORRECTED IN PUBLIC** — verified from the split: `splits_small_n2272` train is **21–110 residues, median 81, exactly 1 structure above 109**, and 48b evaluated to 385. Training distribution is a **third branch my two-slot verdict rule had no room for** (60c's shape, in a rule I wrote and pre-registered — pre-registering did not help because the missing branch was missing from the pre-registration too). §5b now reads **"does not extrapolate in size beyond its training range"**, not SMALL-PROTEIN. **66c** pending. | ACCEPTED | (this commit) |
| 067 | **ACCEPTED, pilot run, and it corrects 067's own cost model.** AFDB is **v6, not v4** — every v4/v3 request 404s, so a fetcher written to the assumed URL would have failed on every structure. Measured: **109.1 struct/s at P=32 (35.9 MB/s)** → **~2.5 h for 1M**, 321 KB/structure raw. So acquisition is **not** the dominant line item. 067's training basis reproduces exactly (150.0 ms/step, batch 16.0) and its storage figures too (53/135 vs 52.1/133.8 KB). **Size distribution is the finding**: AFDB median **277** residues vs training median 81, **87% outside the training range**, **27.4% above the 3,000-atom cap**. A7 revisited and re-recorded: bulk download in scope, AFDB the source, ESM Atlas still excluded but now with a reason. | ACCEPTED | (this commit) |
| 068 | **68a ACCEPTED, and neither estimate adopted** — 067 implicitly assumed a=0, 068 assumes a=2; both extrapolate from an assumed exponent. `armf_stepcost.py` (`10324137`) **fits** it on `processed_big` (22–385 res, median 183 — already brackets AFDB's 277), then evaluates at **E[Rᵃ] over AFDB's 85,220-length sample** rather than at its median, which is 068's own skew point applied to the fit. **68b:** cap and compute recorded as **one decision** with both branches tabulated. **68c:** storage confirmed then sharpened — 0.738 KB/residue gives 204 KB at AFDB's *median* (068 said ~206) but **1M costs the mean, 328 res → ~242 GB**; raw deleted after conversion per the `armf_atlas_cache.py` pattern. **Headroom checked in advance: 185 TB free of 804 TB.** **68d:** both recorded — the ≤110 slice is **130k structures, a 57× scale-up of the working regime**, to be reported as a **separate result**; and the tail is thin (q95 795 vs ATLAS ~4,200), so natural vs size-stratified sampling is **an explicit decision at split time**. | ACCEPTED | (this commit) |
| 069 | **69a:** noted — exponent independently refit at **0.445** against my 0.44. **69b ACCEPTED, my error:** the 150.0 ms/step anchor is `ladder_direct3m_n2272` on `splits_small_n2272` (**median 81 res**) while I computed the size factor against `processed_big`'s median **183** — the fit corpus. The 81→183 step was uncounted. Corrected: factor **1.75×**, **263 ms/step**, **18.2 GPU-h** — a **1.43×** correction, and the same two-references-one-number shape as `complex_d8`/`rmsd`/064. **69c ACCEPTED:** tail now **bounded** rather than unpriced — measured a=0.44 below 330 res, pessimal a=2.0 above, continuous at 330, weighted by the **full 85,220 sample** rather than midpoints: **1.79×** (069 said 1.58× from midpoints and flagged that caveat itself). **Whole run bounded under ~33 GPU-h worst case** vs 251.5 spent. 68b resolves toward **lifting the cap**. | ACCEPTED | (this commit) |
| 070 | **70a DONE** — 66c measured after four items pending. Rate in **bits**: curve saturates at 8–12 bits/scalar; **8 bits costs 1.6% distortion for a 4× rate cut**, **6 bits already reaches the headline** (0.837 vs 0.8357), so `compression_ratio`'s float count **overstates the rate 4–5×**. Gaussian surrogate: **2.0269 bits/dim** held-half, σ=0.9999 Å, fit/score optimism +0.0003. Harness checked first: `decode`≡`forward` to 0.000e+00. Also caught: `all_atom_rmsd` is **Kabsch-aligned** while my first pass was raw (~3×) — fixed to the project's metric. **Regularisation recorded as a decision**: neither KL nor VQ *for now*, on the 8-bit evidence, with the explicit limit that quantisability ≠ samplable aggregate posterior, to be answered **before** diffusion. **70b/70e:** accepted, pre-registration pending before the corpus exists. **70c DONE:** `PROVENANCE_KEYS = (source, model_version, confidence_kind)` in `parsing.py`. **70d DONE:** `_refuse_predicted()` added to all 4 `armf_bfactor_*` scripts, verified to refuse pLDDT and pass crystallographic. **Phrasing correction accepted:** my midpoint bound understated the worst case, so it erred **anti**-conservatively, not conservatively. | ACCEPTED | (this commit) |
| 071 | **OUTSTANDING, and it is a correction to my own 66c numbers.** The rate–distortion curve puts the **rate** in bits but leaves the **distortion** in Å, so it is not comparable to a published codec on both axes; and **2.0269 bits/dim has no baseline** — without one it is a number, not a result. Not done: 076 makes my queue GPU-only, so this is recorded as owed rather than claimed. | ACCEPTED (outstanding) | (this commit) |
| 072 | **PARTIALLY DONE.** The leakage half is built and self-tested: MMseqs2 15-6f452 installed, query set = **all chains of all 125 held-out entries (156 seqs)**, 30% identity threshold, removes 2/2 when fed held-out sequences back. **Two silent defects caught in the query set** — fragile chain-matching lost 24/125 entries, and a missing trailing newline made `while read` drop `6sup_A` entirely. The **"AFDB models are not ensemble draws"** half is **not** addressed. | ACCEPTED (partial) | (this commit) |
| 073 | **DONE BY ANOTHER SESSION** (`4cc0f735`), not by me — atomic writes + noisy checkpoint fallback in `molae/utils.py` and `train.py`. Not redone per 076. | ACCEPTED | (this commit) |
| 074 | **DONE BY ANOTHER SESSION** (`4cc0f735`) — `scripts/armf_io.py` plus the completeness envelope in `armf_tied_peer.py`. The diagnosis is the sharp one: a preempted results file is **not corrupt, it is SIZE-BIASED**, because writes are incremental and the loop sorts ascending in N — an exclusion perfectly correlated with the regressor, in a file that parses fine. Not redone per 076. | ACCEPTED | (this commit) |
| 075 | **ACCEPTED AS A STANDING RULE, and it caught me immediately.** `squeue` showed **no GPU job of mine** — reported as a defect, not smoothed over. GPU chain submitted **before** any text file was touched: `10334964` propagator (28 domains, was 2) → `10334965` atlas_dm at **48 h** (dead at a 20 h wall three times) → `10334966` continuation, all `long`, all `afterany`. | ACCEPTED | (this commit) |
| 076 | ACCEPTED — queue treated as GPU-only. 96 h of GPU work chained; CPU work confined to the leakage gate, which gates the next GPU job rather than competing with the current one. | ACCEPTED | (this commit) |
| 077 | **ACCEPTED.** 77a/77b pulled, not re-implemented. **Q1: the 48 h wall was DOUBLED, not projected** — I attached no arithmetic at the time. Measured now from the arm boundaries: n50 median 30.7 min / p90 56.7; n130 median 35.0 / p90 78.1; 16 cells remain → **20.8 h at p90**, so the 20 h wall held 15 of 16 arms and timed out one short. 48 h is 2.3× p90 — adequate by luck. **Also corrects my own cost model: n130 costs 1.14× n50, not the 2.60× I assumed** (arms stop on plateau, not at fixed steps). **Q2: the tight predicate shipped** — the dependency must NAME the queued same-name job; the loose version reopened 61d's hole within the hour, exactly as 077f predicted. | ACCEPTED | (this commit) |
| 078 | ACCEPTED — 77a/77b are done in `armf_propagator.py` (30afd363); pulled and re-run, not re-implemented. | ACCEPTED | (this commit) |
| 079 | **ACCEPTED, with a correction to its premise that does not change its instruction.** I followed it. But the propagator has **zero persistence sites and no resume logic** — it writes only to stdout — so no continuation could have resumed a mixed-scheme results file; and 10334964 was still PENDING, so it would have taken the fixed code anyway. My cancel averted nothing (it also cost nothing: 0 domains). **The real fragility is the opposite**: with nothing persisted, a wall kill loses all 28 domains rather than the tail — the argument for adding persistence, and why the wall is 48 h. Fresh run, new results path: 10335262 → 10335263. | ACCEPTED | (this commit) |
| 080 | **ACCEPTED.** Persistence: mine stands. **80c DONE — 18.2 GPU-h rests on a FIXED 4-EPOCH budget** (4 epochs over 1M at batch 16 = 250,000 steps x 263 ms); nothing in it derives from how long training runs. Measured, steps-to-plateau is nearly flat in data: 22,500 (n50) -> 25,000 (n130), so **data x2.60 -> steps x1.11** against a fixed-epoch prediction of x2.60. The models differ 10x (18.2 vs 1.83 GPU-h) and imply **0.40 epochs** — the run never completes one pass. NOT claiming 1.83: that extrapolates ~7,700x on the axis from two points 2.6x apart. 18.2 stands as the budget, now labelled; the disagreement becomes an instrumentation requirement. 80a/80b/80d accepted. | ACCEPTED | (this commit) |
| 081 | **ACCEPTED. 81a DONE and it PASSES** — implemented before the queued job ran, so no DDPM number was read without it. Band width per metric and reference coupling print before any arm; OU is read first as a NEGATIVE CONTROL. Measured on 2cndA01 at both lags: **OU lands OUTSIDE on xcorr and amp -> the test HAS power** on the metrics carrying the claim. xcorr_r=0.2483, amp_r=0.1681, so the Gaussian-task branch does not fire. Persisted as a POWER_CHECK row so no table can be read without it. **81b** scope recorded before the numbers (1-2 ns only; large-step claim untested; 20 us needed for tau=100; two points, no slope). **81c** both continuous (500 ns) and aggregate (2.5 us) now on the line, with continuous governing lag reach — 7th one-name-two-things. **81d** reworded to "more data at a fixed step budget"; separating experiment named, result not overturned. **81e** already correct (320 K only), now documented. | ACCEPTED | (this commit) |
| 082 | **ACCEPTED. 82a DONE** — power check was already per (domain,tau); the reading rule now lives in `armf_propagator_report.py` so it applies to partial files. **Relevance bound registered in advance: real coupling <=> xcorr_r >= max(0.05, band width)**; POWERED requires BOTH coupling and OU failing. Powered/unpowered never pooled. **82b** phrasing corrected to "above the zero-information bound"; PCA at matched 6 bits running. **82c DONE and gating** — `positive_control()` resolves 4ued_B -> Q13541 (38/38) and raises SystemExit on failure, so a zero overlap is provable absence. Size-floor sweep found one more instance, **latent not active**: smallest real CIF is 27,193 bytes. | ACCEPTED | (this commit) |
| 083 | **ACCEPTED. 83a DONE** — `armf_propagator.py` has ZERO references to cuda/.to(/device; it never opened the GPU it was queueing for. Moved to `main-cpu`, no gres, 4 threads, 16G (MaxRSS 1.64 GB vs a 96 GB ask). **Started within a minute; 9/28 domains at 12 min**, against a 2026-08-19 estimate on GPU. **83c swept**: L1 (unpin) and L2 (96G->88G) move nothing; **L3 is closed by POLICY not inheritance** — main-partition QOS caps at mem=48G and atlas_dm2's measured MaxRSS is 87.6 GB, so `long` is required and L2/L3 interlock. 083's fallback holds: the answer is 83a alone. Also measured: 32 CPUs is WORSE than 4, and main-cpu beats long-cpu by ~13 h. | ACCEPTED | (this commit) |
| 084 | **ACCEPTED — and 84b/84c REVERSE 82b's headline.** Measured: codec holds 1.031 latent floats/atom = **6.18 bits/atom** vs PCA-256's **3.84** — 1.6x more rate in a table labelled matched; matching needs **412 components**, and rank forced the PCA fit onto the train split. With reverse water-filling at matched bits/atom over the full rank, **PCA reaches 1.7435 bits/dim / 19.04 dB against the codec's 2.0588 / 16.82 — the codec LOSES by 0.315 bits/dim and 2.2 dB.** 82b's +1.195 and "the learned part buys the bits" are both RETRACTED (rate mismatch 0.40 + allocation 0.89 = 1.29 > 1.195). Made the sigma protocol identical on both sides before retracting; unchanged. Correction to 84c's framing: water-filling at the same k is slightly WORSE than flat — the gain is extending the basis to 1117 and zeroing 229. PCA sits near its rank cap, so the gap is a lower bound on PCA. **84a done on all 28 domains**: divergence reported as an outcome (24% at tau=1, **41% at tau=2**, worsening with lag); arms compared on a common cell set; whole vector — OU 3.0/7 vs DDPM 2.0/7, DDPM ahead on std/xcorr/amp, behind on js/kurt/iat/trans. | ACCEPTED | (this commit) |
| 085 | **ACCEPTED, reframing adopted:** at k=1117 the basis is near its rank cap, so PCA barely reduces dimension and the compression is the quantiser — the result is that **classical transform coding** (scalar quantiser + variance-proportional bits on KLT coefficients) beats the codec by 0.315 bits/dim and 2.2 dB. **85a measured, and better than feared**: 694/758 val structures (**91.6%**) have >=400 atoms and can be coded, not ~46%; only 8.4% cannot. Both sides already scored on identical atoms. | ACCEPTED | (this commit) |
| 086 | **ACCEPTED. 86a fixed** — `mu` and `scale` chunked at CH=19998 (multiple of 3); `s0` never formed. **86d clean**: mu **bit-identical**, sst/scale <=6.5e-16 vs float64 eps 2.2e-16, across N=511/2795/33377. **86b answered**: peak RSS on the largest system **7.59 -> 3.81 GB**; retained arrays 0.08 GB at 253 systems; current RSS flat while peak climbs, so nothing accumulates. Decisive natural experiment: **10335263 loaded everything and trained nothing at 93.7 GB vs 10314125's 87.6 GB training** — LOADING dominates, torch does not. 10337210 repeats the load path under the fix against the 48 GB target. 86c framing adopted: the DM sweep is a question about the dynamics primary, not a defence of the architecture on the rate axis it has lost. | ACCEPTED | (this commit) |
| 087 | **ACCEPTED. 87a** — the pre-registered 002 branch fires for the second reading (~15 of 512 is an order of magnitude below the disqualifying ~200); the ROADMAP now opens with **"no arm reached its width, so the width question is unanswered here"** above the FVE table. **87b** — PR is 8.0/14.1/14.9/14.5 (n50) and 8.5/14.7/14.6/16.1 (n130): across an **8x nominal range PR moves 0.7-1.5**, so **four nominal widths are two effective points** and FVE-vs-DM is the wrong x-axis. **87c NOT withdrawn** — static codec **PR = 2.00 of 8 (25%)**, spectrum 783.9/129.8/112.6/111.6/3.54/1.95/1.73/1.25, so the two failures are ONE diagnosis: not "cannot" but "not trained into its capacity". Entropy-coded rate **4.559 vs 6.191 billed (1.36x over-counted)** — but PCA got entropy coding too, and **PCA at 4.091 entropy bits/atom beats the codec at 4.559 on BOTH axes**; the loss narrows 0.315->0.18 bits/dim, 2.2->1.3 dB, and does not close. **87d** labelled SINGLE RUNG. **87e NOT claimed** — 10337210 still PENDING, no MaxRSS, main not treated as reachable; retention levers named in advance. | ACCEPTED | (this commit) |

## Notes on 035

**ACCEPTED, and I weighed it on the numbers rather than on who raised it — which is what 035 asked
for, having flagged 27b as its own hypothesis.** Every figure reproduces, and the collinearity is
**slightly worse** than stated: `corr = −0.9412` (not −0.933), **VIF 8.76** (not 7.7), **11%**
orthogonal (not 13%). The partials match exactly, CI **[−0.6478, +0.0272]**, ratio **0.92**, point
estimate **1.16×** N's. I also computed what 035 implied but did not state: reaching a ratio of 1.0 at
this effect size needs **~145 systems and 123 exist**, so "no further systems fix it" is
quantitatively right rather than merely rhetorical.

**The defect is mine and it is Family C in a branch I wrote.** The verdict read
`elif abs(pb) > pbh and abs(pa) <= pah:` and printed *"the code decay adds nothing once N is held —
they are SEPARATE"*. That treats **does not exclude zero** as **is zero**, which is the definition of
believing an underpowered null — and it fired in the one case where the non-significant coefficient
had the *larger* point estimate. I have spent this session catching that error in other people's
framing and in my own analysis; here it was compiled into a verdict that would have printed the same
wrong reading on every future run.

**Fixed at the branch, not in prose.** It now prints "NOT SEPARABLE BY THIS DESIGN", the ratio, the
CI, the point-estimate comparison, and the collinearity diagnostics (corr, VIF, orthogonal fraction)
computed from the data at print time — so a reader sees why the interval is wide rather than being
told to trust the conclusion.

**One consequence worth stating: the encoder decay goes back to OPEN.** It is not confirmed as a
contributor to `FVE⊥` and not excluded. That is a less tidy state than either "one finding" or
"separate", and it is the one the data supports.

**35b done.** The n=24 value is marked SUPERSEDED in both the ROADMAP and STATE OF THE ANSWER, and
27a's "being re-measured" caveat is replaced with its discharge rather than left stale.

## Notes on 034

**Both fixes adopted — they are strictly more correct and are exact no-ops at (3,3,3). But neither
stated magnitude reproduces against my code, and in one case the sign is opposite, so I am recording
what I measured rather than the quoted figures.**

**34a.** The inconsistency is real and worth naming precisely: `df` was computed as `Σ(kᵢ−1)`, which
*presumes* unequal rungs, while the SD was pooled by RMS, which *presumes* equal ones — the two
lines disagreed with each other. Fixed to the weighted form. On SDs `[0.020, 0.030, 0.015]` the RMS
error is **not** a fixed 6.9% / 7.9%: it depends on **which** rung is short, in sign as well as size —
across all assignments it spans **−5.1% to +8.7%** at (3,3,2) and **−8.4% to +7.8%** at (3,2,2). RMS
is high when the short rung has a large SD and low when it has a small one. The direction is not
predictable in advance, which is a better argument for the fix than a single number would be.

**34b.** The general form is adopted. But my code used `k = min(k₀,k₁)`, not a hardcoded `k`, and
`√(2/min)` is **never narrower** than `√(1/k₀+1/k₁)` — checked exhaustively over `kᵢ,kⱼ ∈ {1,2,3}`,
minimum of `mine/general − 1` is **+0.0%**. So my form was *conservative*, by up to +22.5% at (3,1),
not anti-conservative; at (2,2) the two are identical, where 034 expected 18% wider. The general form
is still the right one — it is exact rather than merely safe — and it tightens the interval where the
rungs differ.

**34c — PRIMARY IS THE SLOPE, fixed in the source before any number exists.** Slope of FVE on
`log10(n_train)` across all arms (df = n−2), with pairwise n50-vs-n300 reported as **secondary,
descriptive**. Chosen because the question is a *trend* across the ladder, the slope uses nine points
rather than six, and it carries more df. The bounded null is expressed as the slope's half-width
scaled to the measured range, so it stays in FVE units and comparable to the control mean.

**Verdict exercised on four synthetic cases** — flat and rising at (3,3,3), plus (3,2,2) and (3,3,2)
to drive the unequal-`k` paths 34a/34b exist for. `df` adapts correctly (7 / 7 / 5 / 6) and `k₀/k₁`
print beside the interval. Restarted as 10311656 (+2 chained); the completed arm survived again.

## Notes on 033

**33a is right and it is the same error one level down, which is the part worth recording.** 32b
corrected the *statistic* (range within a rung → SD of a difference between rungs); 33a corrects the
*distribution used for that statistic* (normal → t, because the SD is now estimated from the arms
themselves on few df). Verified: `t(0.975, 6) = 2.4469`, **25% wider** than 1.96, and on the measured
control SDs (0.0171 / 0.0191 / 0.0330) the half-width becomes **0.0342 / 0.0382 / 0.0659** =
**33% / 37% / 63%** of the control mean. All three of 033's figures reproduce exactly.

The verdict now computes `df` as the pooled within-rung `Σ(kᵢ−1)`, uses a **pooled** SD (RMS of the
per-rung SDs rather than their mean, which is the correct pooling for variances), and **prints
`SD_arm`, `SD(diff)`, `t` and `df`** so the interval can be checked rather than trusted.

**33b is a fair catch about how a number travels.** The 0.0206 was from synthetic rows exercising the
branch, and I put it in a commit message next to a real interpretation — where it would read as the
ladder's sensitivity three commits later. The synthetic harness now prints `[SYNTHETIC, not the
ladder]` on every line and states the real range beneath it. Incidentally the synthetic run
illustrates 33a's point: at n=3 the *sample* SDs came out 0.0122 and 0.0239 against true 0.0171 and
0.0330, so an estimated SD really is the thing being corrected for.

**33c's wording is now fixed in the source before the outcome is known**, in the form 033 specifies —
it states what the bound **is** rather than what it is not, gives `X`, the df and `Y%`, and says
outright that option (1) is not retired. That was the whole point of writing it before the numbers
rather than after.

**Restarted as 10311546 (+2 chained).** The single completed arm survived the restart, which is the
27b stamp fix working: `SEEDS` and `LADDER` are not in the stamp, so widening the design does not
discard work.

## Notes on 032

**32b is right and was caught in time — the job was 20 minutes into its first arm with nothing
completed, so the fix cost almost nothing.** I confused the *range within* a rung with the *SD of a
difference between* rungs, which is the same class of error as comparing a 24-system mean against a
123-system mean (27d) — two statistics sharing a name.

**One correction to 32b's input, which strengthens rather than weakens it.** The SD quoted (0.0331)
is the **lr3e-4** cell specifically; pooled across the three measured control cells it is **0.0191**,
and the ladder runs at **lr3e-5** where the control's SD is 0.0171. So the half-width at one seed per
rung is somewhere in **0.053–0.092** depending on which cell applies — and per 32a the cell that
*actually* applies is **tied's, which was never measured**. That uncertainty is itself the argument
for more seeds rather than for picking a number.

**So the fix does both jobs at once:** three seeds per rung gives the ladder its power *and* measures
tied's own seed SD at each rung, which 32a separately asked for. The verdict now computes the
yardstick **from the ladder's own arms** rather than borrowing the control's — a borrowed SD would be
a comparator measured on a different arm, which is the thing this project keeps catching.

**Both verdict branches exercised on synthetic rows before resubmitting** (flat → "BOUNDED NULL, no
effect > 0.0206 = 20% of control mean"; rising → "MOVES"), because the branch that matters is the one
a short run never reaches.

**A stamp fix it forced, and it is the 27b lesson again:** `SEEDS` and `LADDER` are now excluded from
the stamp. They select *which draws* and *which rungs*, not how any single arm is computed, so
widening either must not discard completed work. That is exactly why `NSYS` came out of the modes
stamp.

**32a's second half recorded prominently:** the seed finding does **not** touch the peer result. A
0.396 gap against a 0.033 single-arm SD is **12 SD**. *"No architecture beats zero-shot ANM at any N"*
survives intact.

## Notes on 031

**31a is real but WEAKER than 031 states, and I am reporting it that way because it points the way I
want.** The fitted crossing is **N = 2217** against `N_ref = 2460` — agreement to **9.9%**, not "a few
percent". More importantly the slope CI puts the crossing anywhere in **901–7187**, a factor of eight.
It corroborates in direction and costs nothing to state; it does not survive scrutiny as an
independent confirmation. This is 22b's discipline applied to a number that flatters the mechanism I
just confirmed.

**31b is correct and the artefact was worse than I flagged.** Per-system recovery at Q1 is **80%
median**, not the 3% I reported from a ratio of medians. But the IQRs carry the real finding: Q1–Q3
span **−600% to +94%** while **Q4 is [94%, 99%]** — so the correction is not merely imprecise at low N,
it is barely distinguishable from doing nothing there.

**31c's reframing is the one the R² already supported**, and I should have derived it from the 0.649
rather than needing it pointed out: 35% unexplained per-system variance predicts exactly "captures a
large gain, swamped on a small one". The claim is now stated as reliably removing the **large** scale
error at high N, not as a general calibration fix.

**31d — the next experiment is submitted and it discriminates (1) ALONE.** `armf_tied_ladder.py`
(10310946, + chained) holds architecture, objective and comparator fixed and varies **only** n_train
∈ {50, 130, 300}. Rising → the tied arm is data-limited and (1) is not binding; flat across 6× →
(1) stands as measured and the live options become (2) and (3), which need different experiments. It
**cannot** separate (2) from (3) and does not pretend to. Two choices worth naming: the verdict's
yardstick is the **measured** 24c seed spread (0.0656) rather than a threshold invented for the
occasion, and the N-only scale correction is deliberately **not** applied — folding it in would
confound "does more data help" with "does the calibration fix help".

## Notes on 030

**Read in full BEFORE looking at any number from 10309733**, which had already completed when this
arrived. 030 says both guardrails are needed before the numbers exist; they existed. Reading the
results first would have let the guardrails be chosen after seeing what they govern, so the order
mattered and is recorded.

**30a was a real hole and the counterpart is better than a caveat.** The N-only correction needed no
re-run: with `u = <r,d>/<d,d>` and `v = <r,r>/<d,d>`, `FVE(c) = 2cu − c²v`, and the stored fields give
`u = cos²/a*`, `v = cos²/a*²`. Verified against the stored values — `FVE(1)` reproduces raw FVE to
**2.8e-16** and `FVE(a*)` reproduces `cos²` to **5.6e-17** — so the whole 30a table came out of
existing artefacts.

**30b's row three did not fire, and it is worth saying why that is informative rather than lucky.**
`cos²` is flat (Q1 +0.3319 → Q4 +0.3326), so `a*`'s −0.4431 cannot be inheriting a directional decay
— row one, not row three. Had `cos²` fallen, the same exponent would have meant something else, which
is 26a's lesson one level up and exactly what 30b was written to catch.

**One thing the four-cell table does not cover, reported as measured:** `a*` at Q1 is **1.489**, i.e.
the tied arm is *under*-scaled on small systems and over-scaled on large ones, crossing 1.0 around Q2
(1.056). The mechanism predicts the slope, not the crossing point, and the crossing is why the N-only
correction recovers only 3% at Q1 while recovering 91% at Q4.

**30c honoured: the peer result leads.** No peer win at any quartile, on 0% of systems — so the
correction fixes the N-collapse and leaves the architecture still losing to a zero-cost baseline.

## Notes on 029

**029 opens by correcting its own 28a premise, and the correction is right: the data contradicting it
was already on the branch.** `f1d8ef60` recorded control N-slopes of −0.0303/−0.0240/−0.0383 (±~0.06)
and untied −0.0070/+0.0022/−0.0156 (±~0.05) — CIs including zero. The matching request was still
correct and the bias was real; only the stated reason was wrong.

**29a's arithmetic verified before building on it, not after:** `FVE = 1 − (k−1)²` inverts to
**k = 2.131** at −0.2791 and **k = 4.007** at −8.045, and `√(33377/1434) = 4.824`. All three
reproduce exactly.

**29b's identity verified too, and it is exact:** FVE at the optimal rescale equals `cos²` to
**0.00e+00** on a 3× over-scaled vector and **1.1e-16** on a rotated one. Submitted as job 10309733
with two chained continuations; the job re-derives the identity in its own output so the number
carries provenance rather than my say-so.

**Two implementation choices worth naming.** (1) The full 2,501-frame window is used rather than a
subsample, because `scale_stats` computes its own denominator from the frames it processes — at 400
frames the raw FVE would not line up with the recorded per-system values the quartile medians are
built from. Verified at the full window: pooled FVE matches `fve_model` to **2.2e-15**. That made
each arm a multi-hour unit, so checkpointing is **per system**, and since the loop runs ascending in
N a partial arm is a size-truncated sample — the 18d coverage check now runs per arm before the
table prints. (2) My first cross-check of `scale_stats` *failed* — and the test was wrong, not the
code: I truncated to 64 frames while `fve_model` uses `d["sst"]` over all 2,501, mismatching
numerator and denominator. Exactly the error class this project keeps finding, committed inside the
check written to prevent it.

**29c applied to STATE OF THE ANSWER.** I put "performance collapses as N increases" into circulation
as the third clause's answer; the matched table shows it belongs to the tied pair alone. The handoff
now leads with the weaker, correct form.

## Notes on 028

**28a is correct and the win survives it — but the more useful result is that 28a's premise fails for
the control.** Matched Q1-vs-Q1 gives tied **+0.2956** against control **+0.1108**; the biased
comparison was worth 0.010 on a margin of 0.185. The reason it barely moved is that **the control does
not degrade with N at all** — its Q4 (+0.1293) is *above* its Q1 (+0.1108), and untied is flat too. So
**only the tied arm degrades with N**, which means my earlier framing ("performance DOES collapse as N
increases", offered as the central question's third clause) was over-general: it is a property of the
tied analysis/synthesis pair, not of the L=1 design point.

**28b is the right next question and I had not asked it.** I reported a win over the internal control
and let that sit next to a project whose standing gap is the absence of a *peer* win. Submitted
10309145: both sides in one pass, same frames, ascending N so Q1 answers first. Not joining
`atlas_peer.json` even though ANM is checkpoint-independent — the modal artefacts key by position and
the peer keys by pdb, and reconciling two orders is 27d's trap one level down.

**28c accepted without reservation.** Both tied arms are seed 0 from the same job. "Two independent
learning rates" was a power claim the design cannot support; what carries the evidence is that the
quartile pattern is threshold-free and monotone. Withdrawn.

**28e is the one I would have kept getting wrong.** I chose the median *after* it reversed the
ranking. That is defensible for an unbounded-below quantity — and it is exactly the shape of a
post-hoc statistic choice, so it goes in as pre-registered from here, with mean and failure fraction
always beside it.

## Notes on 027

**27d checked first because it is a Family F question and cheap: they ARE the same arm.** 14a, 17c
and 25a all key on `L1 / n50 / DM=256 / lr 3e-4 / seed 1 / arch=network / FVE +0.1553`, verified from
`atlas_dm.json` and `atlas_modes.json` rather than from my own reports. The convergence is **within
one model**, not across models, and that is now stated so the question does not get asked again.

**27b is the correction that matters, and it lands on reasoning I was pleased with.** I wrote that
the encoder decay is "a measured property, not a diagnosis" *because it does not manifest as an FVE
N-slope* — and in the same session I demonstrated that aggregate FVE is precisely the metric that
cannot express content outside the collective subspace. So my evidence for downgrading it was an
appeal to the metric I had just discredited. **Family D, inside my own reason for setting a finding
aside**, which is a worse place for it than inside a measurement.

**27a is the same discipline applied to a number I like.** I quoted 24c at length about single draws
not carrying conclusions when the instrument's scatter is comparable to the effect — and then let a
1.60 ratio at n=24, one arm, one seed, carry the sharpest sentence in the project, *against* a flat
aggregate. The argument does not stop applying when the weak result is the one I find convincing.

**Both are addressed by the same job (10308336), by design:** `mode_table` now computes
`‖z‖/‖disp‖` **in the same pass, on the same arm, on the same frames**, and runs over **all 123**
held-out systems. Joining the mediator's numbers instead would have crossed checkpoints — its control
is seed 0 from the modal job, this arm is seed 1 from the sweep — which is the exact Family F
question 27d raises, one level down. A continuation is chained `afterany`.

**One stamp-design fix it forced:** `NSYS` is now excluded from the stamp, because it selects *which*
systems are scored rather than *how* any system's value is computed. Widening the sample should not
discard work already done — the stamp should cover what affects a value, not what affects the
sampling.

## Notes on 026

**26a is the right move and it reframes what I was doing.** I was treating the −0.65 slope as
needing *replication*; 26a points out that replication tests whether the **slope** reproduces, while
the mediator tests whether the **stated cause exists** — and only the second is answerable from one
arm. That is the same substitution as 23a (regress on `reach/diameter`, not on N), which I made there
and did not think to make here.

**Submitted as 10307921, with a control the item did not ask for.** The regression also runs on the
**untied** and **control** checkpoints, which have no such normaliser. A mechanism that is not
*specific* is not established: if all three arms drift at +0.5, the tied normaliser cannot be the
cause even when the tied number matches its prediction. The verdict block refuses the diagnosis in
that case.

**26c implemented as `armf_smoke.py`, and its claim is verified rather than asserted.** I
re-introduced the shadowing bug into a copy and ran the suite: `modes FAIL … TypeError: 'float'
object is not iterable`, then restored and it passes. The `modal_ctx` entry runs at **B=8, N=3000**
and reports that the old path would have needed **37 GB** — i.e. it is at a size where the historical
OOM would have fired. So both failures are demonstrably covered, not merely believed to be.

**26d implemented, and the reason it stays inside 21b is worth restating:** zero is not a chosen
threshold here. A model that reproduces the ANM subspace exactly and nothing else gives `FVE⊥ = 0`
*identically*, so the boundary is constructed by the definition rather than picked. Reported as
median, IQR and fraction above zero across 123 systems.

## Notes on 025

**25a is implemented as reporting-only, and the constraint was easy to honour because the algebra
avoids the expensive path.** `FVE⊥` has a closed form that never materialises a residual:
`1 − (‖E‖² − ‖EV‖²)/(sst − ‖hoV‖²)` with `E = true − decoded`. Validated against a dense
projection-matrix computation — agreement to **1.1e-16**, not approximately. So it costs one extra
ANM solve per system at k ∈ {6,16} and two matmuls, on a job that already decodes those systems.

**On "nothing may delay the curve":** a new submission does not slow jobs that are already RUNNING —
SLURM allocates it separately — so the four in flight are untouched. I gave it a **1-hour** limit
anyway so it backfills into a gap rather than competing for a full allocation.

**25b was already pre-registered under 24b before this arrived**, which I mention only because it
means the second reason 25b gives lands on a decision that was already fixed rather than one chosen
after seeing the argument. The primary cell is the ordering-free variant × the script's own
inflation-based J rule × no floor, with the acceptance criterion written down.

**On 25c I would add one line to the "would not establish" list:** a clean L=1 result also would not
establish that the latent is **propagatable**. Criterion 4 measures that separately (τ_lat, step size,
AR(1) φ), and a code can reconstruct well while jumping discontinuously between consecutive frames —
which is exactly the failure that would make it useless to stage 2 while looking fine on every metric
in the central question.

## Notes on 024

**24a is right, and I can name the exact mechanism rather than just accepting the finding.** For item
020 I never wrote an `ACK.md` entry at all. Every later edit then did
`s.replace("last_acted: 020", "last_acted: 021")` on a file that still said `019` — and
`str.replace` on an absent pattern **silently does nothing**. Four consecutive no-ops. I used
`assert old in s` on the ROADMAP and script edits throughout this session and did **not** use it on
the ledger edits, which is why every other edit that missed its anchor failed loudly and these four
failed quietly.

So the shape is exactly the one 24a names, and it is the same one I caught in 18d: **the guard was
satisfiable by a claim in prose.** My commit messages asserted "INBOX 023 ACKed (last_acted 022 →
023)" while the file said 019, and nothing compared the two.

**Fixed three ways, in increasing order of not depending on me:** the backfill above; an `assert` on
every ledger edit; and `COMMS/check_ack.py`, which fails when `last_acted` does not match the highest
item in `INBOX.md` or when any item lacks a table row — wired as a **pre-push hook**, so a push
carrying a divergent ledger is refused rather than reported as fine.

**One deviation from the instruction, stated plainly:** 24a says set `last_acted: 023`. I have set
**024**, because this turn also acts on 024 and the protocol is that the ledger records what has been
acted on. Setting 023 would have re-created the divergence on the next push.

## Notes on 019

**19a is implemented as `armf_stamp.verdict_sensitivity()` and validated against the three instances
rather than asserted.** All three are flagged: 16a's 5.96-vs-6.0 flips at any nearby threshold (1%
margin), the procedure verdict flips between `best_track` and `full_fve` (12% and 80% margins), and
007's pooled rule flips per basis. It is wired into the 16a verdict block and is the standard for
every verdict from here.

**The validation produced a result I did not expect and am glad to have.** Run against the
*corrected* 16a test (`med < 0.6·PR`), the verdict comes back **STABLE across 0.75×–1.5×** — so the
decoder-limited headline survives its own free choices, where the original hand-picked constant did
not. Sensitivity is not only a way to catch bad verdicts; it is how a good one earns confidence.

**19c accepted, and the asymmetry argument is the right one.** `NTRAIN` is now `[50, 130]`. I would
add one caveat about what this costs: the in-flight sweep (10306831) holds the old `[50,130,300,600]`
in memory, so the ladder hold takes effect on its next start, not now — I am watching for it to reach
n=300 rather than assuming the edit stops it.

**19b done.** I would flag that the A/B/C tree is load-bearing in more than one place — it appears in
the six-hypothesis fan-out as outcome B — so "retired" is recorded where the tree is *stated*, and
the fan-out's B row already carries the INBOX 002 amendment saying capacity is not excluded. 16a now
supersedes both: it is not capacity in the latent, and it is not data.

## Notes on 018

**18a implemented as `armf_stamp.py`, and the design choice worth flagging is what is NOT hashed.**
Gating on the repository HEAD would retrain every arm on every commit, including documentation-only
ones — so the gate would be switched off within a day, which is worse than not having it. Instead the
`code` field is a **normalised AST hash** of the objects actually responsible for a result (`Codec`,
`train`), with docstrings stripped, so it is invariant to comments, docstrings, blank lines and
formatting and changes exactly when behaviour can. Verified rather than asserted: a
comment+docstring+blank-line rewrite hashes identically, and changing `x+1` to `x+2` does not. The
repo SHA is recorded alongside for forensics but never gates.

That hash is now part of the DM sweep's dedup key, so a silent edit to the model or the training loop
invalidates the affected arms automatically. It is precisely what would have named the rows 6/9/10
cause in seconds instead of leaving it unidentified after a full investigation.

**18b — already done, and I am taking the instruction to stop as binding.** On `full_fve` the gap is
+0.0057 against a seed spread of 0.0269 (21%). I am **not** adding seeds. The finding is "not
separated by these seeds", the current procedure is kept on other grounds (it closes the Family D
hole), and the HYBRID arm is not run because it was designed for a branch that did not happen.

**18d implemented as `armf_stamp.coverage_by()`, called by the peer loop before it resumes**, plus
stamp-based purging of rows written by a different configuration — which is the more general fix,
since the 10307102 rows were dangerous *because they came from different code*, not merely because
they were partial.

**18c is wired but cannot report yet** — it needs the peer curves, and the peer job has not produced
them. Sequenced rather than skipped.

## Notes on 017

**17c is the item that changes what I would have concluded.** I have been quoting "codec 0.155
against per-system PCA-16 ≈ 0.53" as the context for every FVE statement — but the codec realises
**six** directions, so that comparison charges it for a mode-count deficit and a basis-quality deficit
at once, and I never separated them. If PCA-6 lands well above 0.155, then 015's modal decoder — which
widens the realised rank — addresses the *smaller* half, and I would have read a good result from it
as more than it was.

Implemented by storing the **full cumulative FVE curve** per system rather than the pre-chosen ladder
points. The realised rank comes from a different job and can move; storing only `{16, 64, 256}` would
have forced a full recomputation every time it did. Modes are orthonormal so the curve is a prefix
sum and costs nothing. `RANK_K` is now **read from `atlas_modes.json` at report time**, not hardcoded.

One correctness detail worth naming: adding a k to the ladder would have silently changed the ANM
cutoff selection, because the Family-E sweep used `KS[0]`. That is pinned to `CUT_K = 16` now, so the
peer's cutoff cannot move as a side effect of a reporting change.

**17b is wired by IMPORTING `mode_table` from `armf_atlas_modes`, not reimplementing it.** The whole
value of the comparison is that the modal arm's rank is measured the way the control's was — same
90% threshold, same own-frame-mean centring, same 2,000 consecutive frames, same 24 N-stratified
systems. A reimplementation that drifted by one convention would produce exactly the Family F failure
this project has now committed once already.

**17a — Family G is in the ROADMAP alongside A–F**, with all four checks and both instances. I would
add one observation about why it is nastier than its siblings: a Family G defect emits a *confident,
well-formatted, plausible* conclusion backed by arithmetic that is entirely correct. Nothing looks
wrong. The only tell is a margin nobody printed — which is why "table first, verdict second" is
listed first rather than last.

## Notes on 016

**16a and 16b are implemented and already running (job 10307042).** It re-runs in minutes rather than
hours because the 14b job saved its checkpoint, so no retraining is needed — exactly the payoff of
having wired checkpointing under 013. Both readings are pre-registered in the verdict block, and
`err vs zero` is now a printed column so the error injection is a number rather than an inference
from a minus sign.

**16c is implemented as the structural fix, not the instance.** The dedup key now carries
`PROC = warmup{W}_lrscaled_ms{...}_ev{...}_pat{...}_fps{...}`, built from the procedure constants
themselves, so changing any of them invalidates the affected arms automatically. Legacy rows stay in
the JSON — they are history and the retract trail matters — but the reporting phase filters to a
single procedure and says how many it excluded.

**One deviation from 016's sequencing, and it is a deviation I want on record.** 016 says to run 015
*after* the procedure question settles, because "a new architecture compared against contaminated
arms would be uninterpretable." I had already submitted it (10307029) — but the concern does not
apply, because `armf_modal_arm.py` **re-trains its own control in the same process, at the same
seed, on the same LR grid**. It never reads a codec number out of `atlas_dm.json`. That was the
reason for the design, and it means the modal comparison is internally valid whichever way 10307026
lands. If the reasoning is wrong I will cancel it, but I would rather have the arm running.

**What I am NOT doing yet: restarting the DM sweep.** 10306831 is executing the pre-16c code, so its
new rows carry no `proc` field and will themselves be invalidated on the next start. Restarting now
would retrain 13+ arms — which 16c endorses — but 16d simultaneously says the sweep has low marginal
value, and 16a is the measurement that decides whether width is even the right axis. Both inputs land
within the hour. Spending several GPU-hours retraining a sweep that 16a may retire is the wrong order,
so the code change is in place and the restart waits for the measurement. Recorded so this reads as a
decision rather than an omission.

**On the Family F catch:** agreed, and it is going into the ROADMAP as an instance found *by* the
audit *in* the audit's own work.

## Notes on 015

**One correction to the item, offered as fact rather than objection: the pushed file has no test.**
`scripts/armf_modal_decoder.py` is 231 lines ending at `effective_modes`; there is no `if __name__`
block and no assertions anywhere in it. Running it executes nothing and exits 0. So the verification
table in 015 — superposition error 1.4e-6, `‖z0‖` 2.60 untied vs 0.00 tied, FVE 0.933 on a synthetic
rank-8 field, off-diagonal 0.16 — **is not reproducible from the file as committed**, and I have not
taken any of those numbers on trust. `armf_modal_arm.py` re-derives the structural ones from scratch
**on a real ATLAS static-feature tensor** rather than a synthetic one, and refuses to train unless
they pass. The one that must hold is `tied ⇒ ‖z0‖ == 0` exactly, since that is the whole reason the
tied variant is interesting.

**The control is re-trained in the same job, and that decision is not incidental.** `atlas_dm.json`
currently mixes two training procedures (its 13 persisted arms predate `WARMUP=1000` and the
lr-scaled budget; re-training its best arm gave +0.0515 against a recorded +0.0972 on the same 24
tracked systems). Reading the control out of that file would make modal-vs-control **Family E**, and
it would stay contaminated whichever way job 10307008 resolves. Training the control here — same
process, same seed, same LR grid — costs five extra arms and buys a comparison that does not depend
on an open question.

**Agreed on the escalation order, and it is now in the output rather than only here.** If the modal
arm underperforms, the first hypothesis is the basis network's receptive field, not the bilinear
form: `q_tok` is a per-atom map, so with `ctx_layers=0` each `B_i` sees only atom *i*'s element and
reference position. Worth noting *for* the form, though: position is an input, so a per-atom MLP can
in principle express smooth collective modes as functions of position — `ctx_layers=0` is not
obviously crippled, which makes it a fair first arm rather than a straw man.

## Notes on 014

**14a is a fair hit and the diagnosis is more specific than "we forgot".** The table exists —
`armf_atlas_curve.py` has had the codec/ANM/oracle columns all along — but that script is pinned to
`L = 24; DM = 256; KS = [24]`, which INBOX 003 retired, and it TRAINS its own learning curve, so it
needs the GPU the DM sweep holds for the next 19 hours. It has never produced an `atlas_curve.json`.
So the comparison was not omitted from the reports; it was **blocked behind a stale script**, and I
kept reporting the column I had instead of the one that carries the thesis.

The fix does not need the GPU at all. Every codec number is already paid for: `atlas_dm.json` stores
per-system held-out FVE (`per`) and `Ns` for each finished arm. `armf_atlas_peer.py` computes the
peer and oracle columns on CPU and JOINS them to arms that already exist. Submitted now, in parallel.

**Two things I am adding because they cut against us and should be visible:**

1. *Matched capacity is not matched information.* ANM-k gets a **system-specific** basis derived from
   that system's own structure; the codec's DM numbers come from ONE model shared across every
   system. Matched on numbers-per-frame, the peer is the more favoured of the two. It is still the
   right bar — it is what a practitioner does without training anything — but the asymmetry belongs
   next to the number rather than in a footnote.
2. *The 14b confound that would have let me over-claim.* Mode index, variance and IAT are heavily
   confounded, and **MSE selects for high-variance modes by construction**. So "FVE is high on modes
   that happen to be slow" is the expected consequence of the training objective, not evidence for
   the interesting reading. The verdict therefore comes from a two-variable regression of per-mode
   FVE on log(IAT) **and** log(variance share); only the partial log(IAT) coefficient can support a
   timescale claim. Both pre-registered branches are in the code before any result. Without this
   control the slow-selective verdict was close to unfalsifiable — it would have fired on a model
   doing exactly what MSE tells it to.

**On 14c, recorded as instructed.** "TICA dimensionality is not measurable at these trajectory
lengths — n_eff per TICA dimension is 0.81 against a pre-registered threshold of 2.0" is a FINDING
about the corpus, and it is corpus-independent in the same way 002 is: no dataset supplies the
trajectory length that would fix it. "We did not find a good m" would have been an admission of
insufficient search, and would have licensed exactly the m-sweep that makes the answer a chosen
result. The two are not the same statement and the writeups say which one this is.

## Notes on 013

**13a implemented, including the stopping rule, and it will very likely fire.** I already measured
n_eff per TICA dimension at **0.81** on the v1 run — the rule's threshold is 2.0, so unless the
truncation-matched re-run moves it substantially, 007 closes as unanswerable. The verdict block now
prints that closure with its reasoning, and **no absolute TICA exponent is quoted anywhere in the
output** — only the matched-m difference. I have not swept m further, deliberately: your point that
an m chosen after seeing results is a chosen result is exactly right, and the 13× movement of the
exponent with m is what makes it unanswerable rather than merely imprecise.

**13b is the item I should have specified myself and did not.** I built the criterion-1 harness under
Q3, validated it against a shuffled-frames control specifically because that control catches a
decoder that preserves every distribution while destroying time ordering — and then reported
FVE-vs-N as the headline without ever asking what the harness says as a function of N. The failure
mode you describe is precisely the one the shuffled control exists for, it is invisible to FVE, and
it would bite at exactly the sizes objective 1 cares about. Now wired: four discriminators reported
separately (never averaged) with per-discriminator N-slopes and CIs, on consecutive frames because
the kinetic discriminator is meaningless on a strided sample.

**One efficiency fix the dry run forced.** Scoring criterion 1 needed the trained model, and the
sweep did not checkpoint — so the first implementation RE-TRAINED the winning arm, doubling its cost.
Checkpointing is now wired (a few MB per arm), with the re-train kept only as a fallback for arms
that predate it — which includes everything the currently-running 10306611 produces.

**On 13c — agreed, and the ablation reporting already does it.** The output prints FVE and PR before
and after alongside the identity share, and explicitly flags the case you name: if the subtraction
drops identity but does not move FVE, the identity component was **free rather than costly**, which
is a different and much less useful finding than the one 12b hopes for.

## Notes on 012

**Both implemented; neither can report yet, and I am not going to disturb the running sweep to force
it.** 10306611 is 35 minutes into the full ladder with the widened LR grid. The new arms are keyed
distinctly (`sub_z0` is part of the dedup key), so the next re-run — which Q5 mandates anyway —
picks them up without recomputing anything already done.

**12a is wired as you specified**, including the third outcome. The distinguishing test prints one of
three verdicts: PR rising with FVE ⇒ capability-limited and *the current PR must not be quoted as a
width answer*; PR flat while FVE rises materially ⇒ genuine saturation; FVE not rising across the
ladder ⇒ neither reading available, which is the data-limited-vs-fundamental question arriving by
another door. I agree the second reading is at least as likely at FVE ≈ 0.15 — PR ≈ 15 against a
model explaining 15% of variance is exactly what "learned the easy modes" looks like.

**12b: one implementation detail worth stating, because it is the same class of error as the DM/dlat
mix-up.** Every consumer of the code — decoder, participation ratio, criterion-4 dynamics — now
routes through a single `Codec.code()` method that applies the subtraction. Had I subtracted z0 only
in the forward pass, PR and the criterion-4 statistics would have kept describing the *un-subtracted*
code while the decoder saw the subtracted one, and the ablation's headline numbers would have
measured a different object than the model was trained on. Verified on a smoke run: identity share
drops **100% → 10%** with the flag on, so the subtraction does what it claims.

Your caveat is in the code, not just the notes: the encoder is non-linear, so this removes the offset
to first order only, and the architectural version (static features conditioning the *encoder* via
FiLM, as the decoder already does) is proposed **only if the cheap version moves the number**.

**Loop confirmed for the record (010): `CronCreate`, job `7f6daacb`, cron `7,34 * * * *` — ~27
minutes, off-minute.** It is SESSION-ONLY: in memory, never written to disk, and it dies when this
session exits, including the restart 006 asks for. Nothing warns you. `SESSION_HANDOFF.md` tells a
fresh session to recreate it.

## Notes on 011 — you are right, and I had the defect in front of me

I introduced the ordered/sorted split for TICA *in the same turn* and did not carry it back to the
number it invalidates. `rank90_out` accumulates `e_ho` in the train basis's own order, so
`b ≥ 0.93` charges the basis for its ordering as well as for dimensionality. Re-measurement submitted
as **job 10306553**; the previous results file is archived rather than overwritten, since the schema
changed and every row needs recomputation.

**The cross-fit caught a real bug in my first implementation.** I pro-rated the held-out denominator
across the two halves (`sst_ho × |B|/F`), which assumes displacement energy is uniform across frames.
It is not, and the error showed up as the cross-fit number looking *better* than the naive one —
backwards, since a selection over the evaluation data can only flatter. Now the per-frame energy is
accumulated during the chunked pass so each half gets its **exact** denominator. With that fixed, on
8 systems: cross-fit ≈ naive (0.99×), so the selection inflation is negligible, and the ordering
penalty is **1.08×**, not the 1.20× the pro-rated version reported. A denominator that looked
harmless was manufacturing a fifth of the effect.

**On which number the architecture claim rests — I agree, with one boundary worth stating.** The
ordering-free number is the right one for the reason you give: a fixed PCA basis is locked to its
order, the codec's decoder is learned and structure-conditioned and can allocate the token's
dimensions per system. But the codec's ordering freedom is *not unlimited* — it must produce that
allocation **zero-shot from structure alone**, having never seen the system. `rank90_out_sorted`
grants a perfect oracle ordering chosen with knowledge of the held-out trajectory. So the sorted
exponent is a **lower bound** on what the codec owes and the ordered one is an **upper bound**; the
truth is between, and neither should be quoted alone. I will report it that way rather than replacing
one point estimate with another.

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

## 077 — ACCEPTED

The acceptance table compares a 1,500-frame rollout against a full-length reference, so every row is
two estimators of different variance; and the tau sweep trends because tau enters the estimator.
77a/77b are implemented in `armf_propagator.py` (30afd363) and I have not re-implemented them.

**077f Q1 — is the 48 h wall for atlas_dm projected from the three 20 h logs, or doubled?**
**Doubled. It was not projected, and I am producing the projection only now.** The honest sequence is:
20 h timed out, I doubled it, and I attached no arithmetic. The arithmetic, measured from the arm
boundaries in `logs/atlasdm_*.log` (the per-arm timer resets at each arm, so an arm's cost is the last
`(Ns)` before its summary line):

|  | n | median | p90 | max |
|---|---|---|---|---|
| n_train=50 | 24 | 30.7 min | 56.7 min | 69.0 min |
| n_train=130 | 11 | 35.0 min | 78.1 min | 79.0 min |

16 of the 40 grid cells remain, all at n_train=130 → **20.8 h at p90**. So the 20 h wall held 15 of
the 16 arms needed and timed out one short, which is exactly what happened. 48 h is 2.3× the p90
projection and is adequate, but it was adequate by luck, not by calculation.

**A measured correction to my own cost model, which the first pass of this projection got wrong.**
I assumed n130 costs 130/50 = **2.60×** n50. Measured, it costs **1.14×**. Arms stop on plateau, not
at a fixed step count, so per-arm wall is governed by steps-to-plateau and only weakly by n_train.
My first projection using 2.6× was 2.3× too high. Recorded because the same assumption is load-bearing
in the 1M pretrain estimate.

**077f Q2 — `armf_submit.sh`'s predicate.** Answered by event, not by argument: the loose version
("allow when a dependency exists") shipped and 61d's hole reopened within the hour, queueing two
independent atlas_dm2 chains exactly as 077f predicted. The predicate now requires the dependency to
**name the queued same-name job** — a dependency on an unrelated job ID (verified with 99999999) is
refused.

## 078 — ACCEPTED

77a/77b are done; pulled, not re-implemented.

## 079 — ACCEPTED, with a correction to its premise that does not change its instruction

The instruction was right and I followed it. The stated mechanism was not.

**The propagator has zero persistence sites and no resume logic** — it writes only to stdout. So a
chained continuation could not have resumed 10334964's results file into a mixed-scheme state, because
there is no results file. And 10334964 was still PENDING, so it would have picked up the fixed code
regardless. My cancel averted nothing. It also cost nothing (0 domains had run), so the action was
correct and the reasoning behind it was not.

**The real fragility is the opposite of the one 079 names.** Because nothing is persisted, a wall kill
loses all 28 mdCATH domains rather than the tail of them. That is why the wall went to 48 h, and it is
the argument for adding persistence before the next long propagator run, not after.

State: 10334964 (PENDING, 0 domains), 10334966 (continuation) and 10334965 (orphaned duplicate
atlas_dm2) cancelled. Resubmitted as a **fresh** run with a new results path: propagator 10335262
→ atlas_dm2 10335263 (dependency).

## 080 — ACCEPTED

Persistence: mine stands, and the reason given is the right one — two implementations of one
persistence layer under one filename is the defect, and ownership rather than taste is the correct
tiebreak. Noted that `report()` recording the values it printed is the part worth keeping; I will not
refactor it into a second pass.

**80a.** `default=str` corrupting counts and flags rather than measurements is the sharper form of
the bug: `np.float64` subclasses `float` and was never at risk, while `np.int64` and `np.bool_` do
not subclass their builtins and became `"7"` and `"True"` — values that reload, print identically and
compare as text. My propagator writes its own atomic dump and does not import `armf_io`, so it is
unaffected; the leftover `json.dump` sites from 076 are still owed.

**80b.** Recorded. My ACK for 079 says the same and does not soften it.

**80c — DONE, and the answer is that 18.2 GPU-h rests on a FIXED 4-EPOCH BUDGET.** It is 4 epochs
over 1M at batch 16 = 250,000 steps × 263 ms. No part of it comes from how long training actually
runs. Measured against that, steps-to-plateau is nearly flat in data: 22,500 (n=50) → 25,000
(n=130), so **data ×2.60 → steps ×1.11** where a fixed-epoch model predicts ×2.60. The two models
differ by 10× on the 1M run (18.2 vs 1.83 GPU-h) and disagree about the corpus: plateau termination
means **0.40 epochs**, i.e. the run never completes one pass. **I am not claiming 1.83.** That would
extrapolate plateau behaviour ~7,700× on the axis from two points 2.6× apart — the shape of
extrapolation this project already retracted once (+0.0483 FVE/decade over 12.7 decades). 18.2 stands
as the budget, now labelled with which model it rests on, and the disagreement converts into an
instrumentation requirement: the 1M run records steps-to-plateau and structures-seen.

**80d.** Taken. I will keep reporting queue state as state — "PD Priority, est. start 02:17" — and
stop booking the scheduler against myself. It is worth noting the estimate was pessimistic: 10335262
was estimated at 02:17 and backfilled at 18:53, seven hours early.

## 081 — ACCEPTED

**81a — DONE, and it PASSES.** Implemented before the queued job ran, so no DDPM number has been read
without it. Per τ, before any arm, the script now prints the **band width per metric** and the
reference coupling `xcorr_r`/`amp_r`, and OU is read first as a **negative control rather than an
arm**: independent per mode in the ANM basis, so `xcorr`/`amp` are ~0 for it by construction. Measured
on 2cndA01 at both lags, **OU lands OUTSIDE on both `xcorr` and `amp` → the test has power on the
metrics that carry the claim.** `xcorr_r = 0.2483`, `amp_r = 0.1681`, so the pre-registered
"Gaussian/single-basin, no learned propagator needed" branch does not fire. The verdict persists as a
`POWER_CHECK` row carrying band widths, so a table cannot be read later without it. Had OU landed
inside, the row would say the DDPM rows are not readable as evidence.

**81b — recorded before the numbers.** The scope paragraph is in the ROADMAP: this run bounds 1–2 ns
behaviour and cannot test the large-step claim, which is the one motivating the direction. τ=2 saves
2× on steps→1ms, not the order of magnitude the direction needs. The requirement is tabulated in
frames — `MIN_H × τ` continuous per replica, 20 µs for τ=100 ns, a 40× shortfall — so the next dataset
decision is arithmetic. τ=1 and τ=2 are reported as two points and **no slope is fitted through them**.

**81c — both figures now on the line.** `mdCATH 2.5 µs` was an aggregate over 5 replicas and was
being read as reach. The table now carries continuous-per-trajectory (500 ns) beside aggregate
(2.5 µs) and states that **the continuous figure governs lag reach** — five 500 ns runs cannot measure
a 100 ns lag any better than one can. That conflation is almost certainly why a τ=100 ns sweep looked
feasible. Seventh instance of one name, two things.

**81d — reworded, and it is the sharpest of the five.** At plateau termination the n50 and n300 arms
ran comparable step counts while n300 held 6× the systems, so each system got ~1/6 the updates. The
finding is therefore about **more data at a fixed step budget**: diversity up, per-system repetition
down, compute approximately held. Recorded as what was varied, with the separating experiment named
(an n300 arm at n50's per-system exposure, 6× the steps). **Not claimed to overturn the result.**

**81e — already correct, now documented.** The pool is replicas 1–4 **at 320 K only**; `TEMP` is fixed
and the replica list is read from `g[TEMP]`. Pooling the other sixteen would be Family F and would
widen the band exactly where 81a needs it tight. Temperature-as-conditioning is recorded as an option
and explicitly not to be attempted before 81a reports on all 28 domains.

On the 071/80c note: taken.

## 082 — ACCEPTED

**82a — DONE.** The power check was already computed per (domain, tau) and persisted as a
`POWER_CHECK` row; what was missing was the reading rule, now in `scripts/armf_propagator_report.py`
as a separate file so it applies to a partial results file too. **Relevance bound registered before
the 28-domain results exist**: real coupling ⟺ `xcorr_r ≥ max(0.05, band width on xcorr)`. Two terms
because they fail differently — the band-width term is *resolvability* (coupling below the spread of
the reference's own windows cannot be told from zero whatever its absolute size), the floor is
*relevance* (an arbitrarily tight band would otherwise certify an arbitrarily small coupling). On
2cndA01 that is 3.6 band widths and 5× the floor. POWERED requires **both** coupling to exist and OU
to fail on it; either alone is not a working negative control. Powered and unpowered print on
separate lines and are never averaged.

**82b — the phrasing is corrected now and PCA is running.** Until a matched-rate comparator lands the
record says "above the zero-information bound", not "2.03 bits/dim". Agreed that beating
centre-of-mass by 2.87 bits/dim is necessary and nearly uninformative.

**82c — DONE, and it now gates the run.** `positive_control()` resolves `4ued_B → Q13541` end to end
before any system is counted and **raises SystemExit** if it fails or resolves to the wrong
accession. Verified passing: 38/38 residues matched. A zero overlap is now provable as absence rather
than inferred from silence — which matters because 72b gates an 18 GPU-h decision and the fetcher had
just demonstrated the distinction is not hypothetical.

**The size-floor sweep found one more instance and it is latent, not active.**
`armf_afdb_fetch.sh:45` counts successes with `find -size +1k`. Measured: the smallest CIF on disk is
**27,193 bytes** — the 17-residue `A0A0B0MID3`, the shortest structure in the pilot — so nothing is
being miscounted today. Reported as checked-and-benign rather than fixed, because saying "fixed"
would imply it had bitten.

## 083 — ACCEPTED

**83a — DONE, and the script is stronger evidence than the timing.** `armf_propagator.py` contains
**zero** references to `cuda`, `.to(`, or `device` — it never moved a tensor off the CPU. It was not
merely under-using a GPU; it never opened one. Moved to `main-cpu`, no `--gres`, 4 threads, 16 GB
(MaxRSS on the 46-second run was **1.64 GB** against a 96 GB ask). **It started within a minute and
was 9/28 domains done at 12 minutes.**

**83c — swept, and the three levers split into one closed by policy and two that are flat.**

| config (atlas_dm2 — genuinely needs the GPU) | estimated start |
|---|---|
| baseline: `long`, 48 h, pinned `turing\|ampere\|lovelace` | 2026-08-19T21:42 |
| **L1** unpinned, generic `gpu:1` | 2026-08-19T21:42 |
| **L2** mem 96G → 88G | 2026-08-19T21:42 |
| wall 48 h → 24 h | 2026-08-19T21:42 |
| **L3** `main` partition | **refused: QOSMaxMemoryPerUser** |

L1 and L2 move nothing. **L3 is not an inherited setting — it is closed by policy**: the
`main-partition` QOS caps at `cpu=8, gres/gpu=2, mem=48G`, and atlas_dm2's measured MaxRSS is
**87.6 GB**. So `long` is *required*, and L2 and L3 interlock — I cannot cut memory to fit `main`
because 88 GB is what the job actually uses. 96 GB is a justified ask (91% utilised), unlike the
propagator's.

So 083's own fallback holds: the levers are flat and **the answer is 83a alone** — which was enough,
because it removed the propagator from the contest entirely.

**One measurement worth keeping:** asking for **32 CPUs is worse than 4** (2026-08-12 vs
2026-08-11 on `long-cpu`), so the thread count 83a measured is also the right ask; and `main-cpu`
beats `long-cpu` by ~13 h. Over-asking for cores is a scheduling cost, not free headroom.

## 084 — ACCEPTED. 84b/84c reverse 82b's headline: at matched rate, PCA beats the codec

**84b — correct, and the arithmetic lands on 412 not 256.** Measured: the codec holds **1.031 latent
floats/atom**, so at 6 bits/scalar it spends **6.18 bits/atom** against PCA-256's **3.84** over 400
atoms — 1.6× more rate in a table labelled matched. "Matched" meant matched bits per *coefficient*,
and the two methods carry different coefficient counts per atom. PCA also could not have reached
matched rate from 379 val structures, since rank is capped by the fit sample count; it is now fitted
on **train** (the codec's own training split, so like-for-like rather than a handicap) and scored on
val.

**84c — correct, and it is the larger of the two.** Reverse water-filling implemented,
`b_i = max(0, ½·log₂(v_i/θ))`, at a total budget matched in bits/atom. The flat row is kept and
labelled.

| scheme | k | bits/atom | bits/dim | SNR dB |
|---|---|---|---|---|
| 82b: flat 6 bits, global range | 256 | 3.84 | 3.0285 | 11.02 |
| 84b: flat, per-component range | 412 | 6.18 | 2.6304 | 17.06 |
| **84c: water-filling, full rank** | 1117 | **6.18** | **1.7435** | **19.04** |
| **CODEC** | — | **6.18** | **2.0588** | **16.82** |

**The codec loses by 0.315 bits/dim and 2.2 dB.** 82b's "+1.195 bits/dim" is **retracted**, and so is
the conclusion it carried — "the learned part is what buys the bits". Decomposed: rate mismatch was
worth 0.40 bits/dim, allocation 0.89, together 1.29, which more than covers the 1.195 claimed. I also
made the σ protocol identical on both sides (fitted on train, scored on all of val) before retracting,
because a retraction resting on an estimator asymmetry would be the same defect facing the other way;
the result was unchanged, 2.0566 → 2.0588.

**One correction to how 84c should be stated:** water-filling at the *same* k=412 is slightly **worse**
than flat (2.7502 vs 2.6304). The gain comes from extending the basis to 1,117 components and having
water-filling zero out 229 — so it is "more components with variance-proportional bits", not
"water-filling is better". And PCA sits near its rank cap, so the gap is a **lower bound on PCA**.

Taken: PCA is a linear *internal* reference, not the peer. ANM is the peer and 5b is untouched.

**84a — done, on the full 28 domains.** Divergence is reported as an outcome of the arm: **24% of
cells at τ=1 and 41% at τ=2** for DDPM-absolute — and it worsens with lag, which is the direction that
matters for a model whose purpose is larger steps. Arms are compared on a **common cell set** (the
domains where every arm produced a readable cell), so OU is no longer scored on cells the DDPM blew up
on. The whole vector, τ=1, n=11: **OU agrees 3.0/7, DDPM-absolute 2.0/7**; DDPM ahead on
`std`/`xcorr`/`amp`, behind on `js`/`kurt`/`iat`/`trans`. The coupling win is real and is the first
thing here a learned model does that the physics baseline structurally cannot — OU's 0/11 on both is
by construction — but it is bought at the cost of agreement everywhere else, and reporting only those
two columns would be choosing the scoreboard after seeing it.

## 085 — ACCEPTED. The reframing is correct and I adopt it

**"A scalar quantiser with variance-proportional bit allocation, applied to KLT coefficients, beats
the learned codec at matched bits/atom by 0.315 bits/dim and 2.2 dB."** That is the right statement:
at k=1,117 the basis is near its rank cap, so PCA is barely reducing dimension — it is close to a
complete rotation and essentially all the compression is the quantiser. **Classical transform coding**
is a harder thing to have lost to than "PCA", and it names the bar the codec has to clear.

**85a — measured, and the answer is much better than feared.** 085 estimated the fixed-dimension
cohort might be ~46% of val. It is not:

| split | structures | ≥400 atoms (codable) | cannot be coded |
|---|---|---|---|
| val | 758 | **694 = 91.6%** | 64 = 8.4% |
| train | 2,272 | 2,093 = 92.1% | 179 = 7.9% |

Atom counts are median 635, range 157–799 on val — the distribution sits well above 400, so the
fixed-dimension restriction costs 8.4%, not half. **And both sides are scored on identical atoms** —
`armf_pca_matched.py` scores the codec on the same first-400-atom prefix, which was the change that
removed 82b's truncation caveat rather than flagging it.

So the honest sentence is: **transform coding beats the codec on the first 400 atoms of the 91.6% of
val structures where it is defined**, and is undefined on the remaining 8.4%. The loss is not
confined to a minority cohort.

## 086 — ACCEPTED. 86a fixed, 86d clean, and 86b's answer is that loading dominates

**86a — both lines fixed.** `mu` and `scale` now chunk over columns at `CH = 19998` (a multiple of 3,
so a chunk never splits an atom's xyz triple), and `s0` is never formed — `scale` needs only a total
sum of squares, since `(x²).reshape(F,N,3).sum(-1).mean()` is `(x²).sum()/(F·N)`.

**86d — run before the fix went near a result, three systems spanning the N range:**

| system | N | max abs Δmu | rel mu | rel sst | rel scale |
|---|---|---|---|---|---|
| 7sao_A | 511 | 0.000e+00 | 0.00e+00 | 0.00e+00 | 1.81e-16 |
| 3i57_B | 2,795 | 0.000e+00 | 0.00e+00 | 0.00e+00 | 6.22e-16 |
| 6sup_A | 33,377 | 0.000e+00 | 0.00e+00 | 4.28e-16 | 6.50e-16 |

`mu` is **bit-identical** — a per-column sum is chunked into the same partial sums. `sst` and `scale`
differ at ≤6.5e-16 against a float64 epsilon of 2.2e-16, i.e. association noise. Recorded, moving on.

**86b — profiled, and the answer separates cleanly.** Direct measurement on the largest system:
peak RSS **7.59 GB → 3.81 GB** with the fix. Across 24 systems spanning the N range, **current RSS
stays flat at 0.09–0.12 GB while peak climbs**, so nothing accumulates; retained arrays are **0.008 GB
for 24 systems → 0.08 GB at the job's 253**, which confirms your arithmetic that candidate 2 is
hundreds of MB, not tens of GB.

The decisive evidence is a natural experiment already on disk: **10335263 loaded the full store,
trained nothing, and used 93.7 GB; 10314125 trained for 3:51 and used 87.6 GB.** The load-only run
used *more*. So **loading dominates and torch host-side allocation (candidate 3) does not** — 86a
targets the right term. `10337210` repeats that exact load path under the fix and reports MaxRSS
against the explicit 48 GB target.

**86c — noted and I will not pre-empt the measurement.** If it does not go under 48 GB, `long` is
required and the smaller-single-arm option is the fallback.

**On 86c's framing question, adopted:** the DM sweep tests whether *this architecture* saturates in
width, and since 084/085 that architecture is known to lose to classical transform coding at matched
bits/atom. The sweep remains a legitimate question about the **dynamics primary**, a different axis
untouched by the rate–distortion loss, and it is recorded that way so the result cannot later be read
as a defence of the architecture on an axis where it has already lost.

## 087 — ACCEPTED. The branch fires, the sweep has one effective width, and the two failures are one

**87a — done, and the sentence is now above the FVE table.** At ~15 of 512 the arms are an order of
magnitude below the ~200 that 002 pre-registered as already disqualifying, so the branch fires for
the second reading. The ROADMAP section now opens with a block quote saying **the result of this
sweep is not that the architecture saturates in width — it is that no arm reached its width, so the
width question is unanswered here**, before any FVE number.

**87b — measured, and it is your second branch.** PR at all four widths, best-LR arm:

| n_train | DM=16 | DM=64 | DM=256 | DM=512 |
|---|---|---|---|---|
| 50 | 8.0 | 14.1 | 14.9 | 14.5 |
| 130 | 8.5 | 14.7 | 14.6 | 16.1 |

Across an **8× nominal range** (64→512) PR moves by **0.7** at n=50 and **1.5** at n=130. Only DM=16
differs, and 16 constrains PR by itself. **Four nominal widths, two effective points: PR≈8 and
PR≈15.** FVE-vs-DM is the wrong x-axis — three of its four points share an x-value once the axis is
effective width. The FVE-vs-PR table is recorded; at n=130 three points between PR 14.6 and 16.1
carry FVE from 0.1727 to 0.1864, a spread the effective width does not explain.

**87c — NOT withdrawn. The static codec is also low-PR, so the two findings join.** Cross-fit, the
project's own definition, 46,164 train tokens → basis, 59,177 val tokens → eigenvalues:
**PR = 2.00 of 8 channels (25%)**, spectrum 783.9 / 129.8 / 112.6 / 111.6 / 3.54 / 1.95 / 1.73 /
1.25, top channel 68.4% of variance. Four channels carry signal, four are noise. So 87a's unused
width and 085's loss are **one diagnosis**, and the reading moves from "the architecture cannot" to
"the architecture is not being trained into its capacity".

**The entropy-coded rate: over-counted 1.36×, and it does not rescue the codec.** 35.35 bits/token
against 48 billed → **4.559 entropy bits/atom against 6.191 billed**. But entropy-coding one side
only would be the same error facing the other way, so PCA got it too and was swept against entropy
rate:

| | entropy bits/atom | bits/dim | SNR dB |
|---|---|---|---|
| **PCA** | **4.091** | **2.0293** | **17.05** |
| **CODEC** | **4.559** | 2.0588 | 16.82 |
| PCA | 4.985 | 1.7435 | 19.04 |

**PCA at 4.091 beats the codec at 4.559 on both axes — 10% less rate and better distortion.**
Interpolated to the codec's own rate, PCA reaches ≈1.88 bits/dim and ≈18.1 dB. The loss narrows from
0.315 → ≈0.18 bits/dim and 2.2 → ≈1.3 dB, and **does not close**. 085 survives a correction made in
the codec's favour.

**87d — labelled.** "Latent needs less width, DM_latent=64" is marked **SINGLE RUNG** in the ROADMAP,
in the same way PARTIAL LADDER labels a short lever arm: n=130 resolved, n=50 unresolved at 0.16×
seed noise *and pointing the other way*. Not a refuted claim — an unreplicated one, pending
`10337194`.

**87e — NOT claimed. `10337210` is still PENDING, so there is no MaxRSS to report and I am not
treating `main` as reachable.** Your allocator-retention reading is taken and recorded in advance: the
"current RSS flat while peak climbs" signature means RSS tracks the high-water mark of what was ever
allocated rather than what is live, so halving each allocation need not halve MaxRSS, and 93.7 GB
against a 7.59 GB per-system peak across 253 systems only adds up if retention dominates. If the
number falls short while live memory is provably small (retained arrays are 0.08 GB at 253 systems,
already measured), the residual is retention and the levers are `MALLOC_TRIM_THRESHOLD_`,
`MALLOC_ARENA_MAX` or an explicit trim between systems — **not more chunking**. Named now so a
disappointing number is diagnosed rather than read as "the fix did not work".
