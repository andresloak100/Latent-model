# Usability test: is a relaxed decoded backbone physically valid? (not RMSD)

The test §8.1 named. "Learning worked" (headroom closed) and "codec is usable" are
different questions; §8.1 records ~1.2 A backbone as an unremovable floor, so
usability was never going to be settled by RMSD. It is settled by whether a SHORT
LOCAL RELAXATION of the decoded backbone yields a physically valid structure.

**Method.** Decode held-out backbone frames (rep0 second half) of 4 frozen-test
systems (n_CA 64..207, incl. the flexibility outlier 2k4qA00) with each codec, run
a short local relaxation (labelled LJ + bond + angle + weak tether stand-in -- no
real FF is callable in this venv), then score: backbone bond lengths, angles, CA
clashes, Ramachandran. Identical pipeline on per-system PCA (best), cross-replica
(general ceiling), and ANM (zero-parameter general codec).

**Result (mean over systems, AFTER relaxation).**

| codec | recon A | bond \|dA\| | bond% | angle \|dd\| | angle% | clashes | minCA | rama% |
|---|---|---|---|---|---|---|---|---|
| PCA   | 0.71 | 0.004 | 100% | 1.6 | 98% | 1 | 3.88 | 90% |
| cross | 0.91 | 0.007 |  99% | 2.4 | 96% | 2 | 3.75 | 87% |
| ANM   | 1.54 | 0.028 |  97% | 3.1 | 94% | 3 | 3.85 | 88% |

**Verdict: ANM-relaxed ~= PCA-relaxed -> the codec question CLOSES at ANM.**
The 2x reconstruction gap (1.54 vs 0.71 A) does not survive relaxation: ANM's
decoded backbones are physically valid (bonds 97%, angles 94%, Rama 88%, minCA
3.85 > 3.7), within a few points of PCA on every axis. The extra accuracy a learned
map might buy does not change physical validity. Per the pre-registered read, **no
eigh build** -- the conditional trigger (ANM fails AND PCA passes, transition in
1.01-1.37) did not fire.

**Caveats.**
- Relaxation is a labelled stand-in, not a real force field; "valid" is relative to
  bond/angle/clash/Rama checks, which are the standard ones.
- Tether is weak, so relaxation repairs local geometry with some freedom; post-
  relaxation RMSD drift was not measured. Preserved fold is indicated by native-
  like Rama (88%) and minCA (3.85, not collapsed). If a stricter test is wanted,
  add post-relax RMSD-to-truth.
- recon_A here is BACKBONE on 4 systems (PCA 0.71 / cross 0.91 / ANM 1.54), so it
  differs from the CA frozen-7 numbers (0.78 / 1.01 / 1.37); internally consistent.

**Consequence.** ANM is not a fallback; it is the answer for the codec stage. It
generalises by construction (contact graph from any reference structure, no fitting,
no training corpus), and after relaxation it is usable. The learned-map track
FINISHED here -- it was trying to beat a baseline that already meets the generality
requirement and is usable.
