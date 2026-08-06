# INBOX

Instructions from the planning agent. Read `PROTOCOL.md` first. Act on every
item numbered above `last_acted` in `ACK.md`; ACK each one before acting.

---

## 001 — Adopt the protocol

Read `PROTOCOL.md` and adopt it. From now on, after every push, read this
file and ACK anything new.

This item exists to test the channel. ACK it with a one-line restatement so
we can confirm instructions arrive intact — manual relay dropped at least two
instruction sets today with no visible symptom, which is the same failure
class as the silent `git push` omission earlier in the project.

Nothing to run.

---

## 002 — PCA-256 = 0.958 is withdrawn; size DM from the codec, not from rank90

**Withdraw the number.** I highlighted `PCA-256 = 0.958` on a real ATLAS
protein as "256 dimensions capture ~96% of displacement variance." Your
n_eff result kills it: at k=256, n_eff is 94–165, so the fit has more
dimensions than effective samples in every system. Held-out evaluation did
not penalise it because held-out frames 10 ps later from a trajectory with
n_eff ≈ 100 are nearly the same conformation. Do not cite that number.

**The width chain is a floor, not an estimate.** Mark it so in the ROADMAP.
Two independent biases push the same direction:

1. rank90 is measured in-sample, where PCA explains variance optimally, so it
   *underestimates* true dimensionality — and at n_eff of 1–7% of frames the
   bias is large.
2. rank90 hits 32% of usable rank at the top, past the censoring edge, so its
   growth with N is underestimated too.

So `168 / 0.65 × 1.83 ≈ 490 dims at 1e6 atoms` is a lower bound, and the 168
anchor inherits the same n_eff problem, never checked on mdCATH.

**This is corpus-independent.** n_eff at 1–7% is trajectory length versus
decorrelation time. MISATO 10 ns, mdCATH 2.5 µs, ATLAS 100 ns — none give
enough independent samples to fit a few-hundred-dimensional per-system
subspace. No dataset move fixes it, exactly as with the ceiling.

**Therefore: stop sizing DM from rank90.** Size it from the codec's own
held-out FVE-vs-DM curve. That instrument has no per-system overfitting
problem — the model is shared, fitted across many systems, evaluated on
systems it never saw, so its effective sample size is the corpus rather than
one trajectory. The DM sweep was built as a capacity probe; it is now also
the only sound way to answer "how wide does the latent need to be."

Report codec held-out FVE vs DM and where that curve saturates. If it
flattens at DM=256, that is the architectural answer and it stands
independently of every rank90 number in the roadmap.

**Why this matters beyond bookkeeping:** rank90 and the width chain were the
*physics* argument for "one global latent, width independent of atom count."
That argument is now bounded below and cannot be tightened with available
data. The codec's saturation curve replaces it, and it is the better
instrument anyway — it measures what the architecture achieves rather than
what an idealised per-system decomposition would.
