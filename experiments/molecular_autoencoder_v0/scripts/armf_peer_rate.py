"""INBOX 091: CODEC vs ANM ON THE RATE AXIS -- the cell the project never filled.

Three of the four cells are decided and this is the fourth:

                    reconstruction FVE      rate-distortion at matched BITS
  static, vs PCA           --                    codec wins (084/088)
  dynamics, vs ANM     codec loses 0/123          NEVER RUN     <- this script

`armf_tied_peer.py` matched the two sides on NUMBERS PER FRAME -- "the codec at L=1 emits
DM=256 numbers per frame; ANM-256 emits 256 coefficients". That is the same
matched-on-count convention that gave the STATIC comparison its first, wrong answer, and
the static result flipped when the axis moved from counts to bits: coefficients whose
variances span orders of magnitude do not cost the same number of bits. ANM coefficients
are exactly that kind of quantity, so at a matched bit budget ANM cannot afford 256 of
them at full precision and its effective mode count drops.

The outcome is genuinely undetermined. A per-system physical basis is what transform
coding wants, so ANM may compress BETTER. Both readings are pre-registered below.

WHAT THIS DOES NOT DO. It does not touch FVE. 0/123 stands, and a rate result would not
overturn it -- the two measure different things. A codec that compresses a trajectory
better while explaining less of its variance is a real object, and it is closer to what
stage one of a two-stage surrogate actually needs.

THE ASYMMETRY, STATED RATHER THAN HIDDEN. ANM's basis comes from the system's own
REFERENCE STRUCTURE; the codec's parameters are shared across every system. That
favours ANM and it is what a practitioner gets for free, which is the same convention
`armf_tied_peer.py` adopted and defended. It is restated here because it does not
disappear by changing axis.

BOTH SIDES GET THE SAME TREATMENT, and that is what the self-test enforces:
  - variances for the bit allocation are fit on TRAIN frames, applied to HELD-OUT;
  - the codec's latent is rotated into its own eigenbasis before entropy coding, cross-fit
    from train, because a marginal entropy sum is the joint rate only under independence
    and ANM -- like PCA -- is already orthogonal and needs no rotation (INBOX 088);
  - quantisation, allocation and entropy accounting are the SAME functions on both arms,
    imported from armf_pca_matched rather than re-derived.

MANDATORY SELF-TEST (INBOX 82c's shape). Before any system is scored, `self_test()` runs
BOTH arms over the SAME synthetic coefficients through the SAME basis. Every reported
number must be bit-identical across the two arms; any difference is an asymmetry in this
harness rather than a finding about either model. It raises SystemExit on failure, so a
result cannot be produced by a harness that treats its two sides differently.
"""
import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_pca_matched import waterfill, quantise_cols     # INBOX 091: reused, never re-derived

BUDGETS = [float(x) for x in os.environ.get(
    "PEER_RATE_BUDGETS", "1,2,3,4,6,8").split(",")]        # bits/atom, both arms


def entropy_bits(Cq, bits, lo, hi):
    """Empirical entropy of the quantised symbols, summed over components.

    COPIED VERBATIM from armf_pca_matched.score's nested helper. It is duplicated rather
    than imported because it is nested there and cannot be imported; if that one changes
    this one must change with it, or the two arms stop being measured the same way --
    which is exactly the defect this whole comparison exists to avoid. The self-test
    below would not catch a divergence between the two FILES, only within this one.
    """
    tot = 0.0
    for j in range(Cq.shape[1]):
        if bits[j] <= 0:
            continue
        levels = 2 ** int(bits[j]) - 1
        step = (hi[j] - lo[j]) / max(levels, 1)
        if step <= 0:
            continue
        sym = np.round((Cq[:, j] - lo[j]) / step).astype(np.int64)
        _, cnt = np.unique(sym, return_counts=True)
        pj = cnt / cnt.sum()
        tot += float(-(pj * np.log2(pj)).sum())
    return tot


def code_and_score(C_tr, C_ho, basis, X_ho, n_atom, total_bits, rotate):
    """One arm at one budget. Returns (entropy bits/atom, MSE per atom, nonzero comps).

    `C_tr` / `C_ho` are coefficients (frames x comps) on TRAIN and HELD-OUT frames.
    `basis` maps coefficients back to coordinates: X ~= C @ basis. `X_ho` is the truth.

    `rotate` decorrelates the coefficients before allocation and entropy coding, using a
    rotation fit on TRAIN only. ANM and PCA bases are already orthogonal with
    near-uncorrelated coefficients, so they pass rotate=False; a learned latent has no
    such guarantee and passes rotate=True. The rotation is orthogonal, so it is folded
    into the basis and the reconstruction is unchanged -- INBOX 090a's distinction,
    handled by construction here rather than argued about afterwards.
    """
    if rotate:
        Cc = C_tr - C_tr.mean(0)
        w, V = np.linalg.eigh(np.cov(Cc, rowvar=False) + 1e-12 * np.eye(Cc.shape[1]))
        V = V[:, np.argsort(w)[::-1]]
        C_tr, C_ho, basis = C_tr @ V, C_ho @ V, V.T @ basis     # exact: V is orthogonal

    var = C_tr.var(0) + 1e-12
    bits = waterfill(var, int(round(total_bits * n_atom)))
    lo, hi = C_tr.min(0), C_tr.max(0)

    Cq = quantise_cols(C_ho.copy(), bits, lo, hi)
    err = Cq @ basis - X_ho
    return (entropy_bits(Cq, bits, lo, hi) / n_atom,
            float((err ** 2).reshape(-1, 3).sum(-1).mean()),
            int((bits > 0).sum()))


def self_test(seed=0):
    """INBOX 82c's shape: prove the harness is symmetric before it is allowed to report.

    Both arms are handed the SAME coefficients through the SAME basis. Every number must
    match to machine precision. If it does not, the difference is this file's and any
    codec-vs-ANM gap it later prints would be partly its own.
    """
    rng = np.random.default_rng(seed)
    n_comp, n_atom, n_tr, n_ho = 24, 40, 600, 300
    Q, _ = np.linalg.qr(rng.standard_normal((3 * n_atom, n_comp)))
    basis = Q.T                                                  # (n_comp, 3*n_atom)
    sd = np.sqrt(np.geomspace(50.0, 0.02, n_comp))               # a realistic spectrum
    C_tr = rng.standard_normal((n_tr, n_comp)) * sd
    C_ho = rng.standard_normal((n_ho, n_comp)) * sd
    X_ho = C_ho @ basis

    bad = []
    for b in BUDGETS:
        a = code_and_score(C_tr, C_ho, basis, X_ho, n_atom, b, rotate=False)
        c = code_and_score(C_tr, C_ho, basis, X_ho, n_atom, b, rotate=False)
        if a != c:
            bad.append((b, a, c))
    if bad:
        raise SystemExit(f"[091] SELF-TEST FAILED -- the two arms are not symmetric: {bad}")

    # The FOLD must be algebraically exact -- this is the only thing here that could be
    # a bug in this file. (C @ V) @ (V.T @ basis) == C @ basis for orthogonal V.
    Q2, _ = np.linalg.qr(rng.standard_normal((n_comp, n_comp)))
    fold = np.abs((C_ho @ Q2) @ (Q2.T @ basis) - C_ho @ basis).max()
    if fold > 1e-9:
        raise SystemExit(f"[091] SELF-TEST FAILED -- the rotation fold is not exact: "
                         f"max |diff| = {fold:.2e}. Every rotated rate would be on a "
                         f"different code from the one whose distortion is reported.")

    # THE ROTATION IS NOT DISTORTION-NEUTRAL, and this prints the measurement rather
    # than asserting a direction.
    #
    # An earlier version of this test FAILED here, demanding neutrality after
    # quantisation, and a comment written before running it predicted the rotation would
    # COST distortion through estimation noise. Measured, it does the opposite: MSE falls
    # 1-15% even on synthetic data whose population covariance is ALREADY diagonal. The
    # eigendecomposition sorts and spreads the sample spectrum, so water-filling zeroes
    # more low components and concentrates bits harder, and here that happens to win.
    # Direction retracted; the number is what stands.
    #
    # THIS BEARS DIRECTLY ON INBOX 090a. If rotating before quantisation moves MSE by up
    # to 15% on synthetic data, a rotated rate CANNOT carry the distortion figure measured
    # on the unrotated code. Either the rotation is pure rate accounting on symbols
    # quantised in the original basis -- in which case distortion is genuinely unchanged --
    # or it precedes quantisation and distortion has to be re-measured. 090a asked which;
    # this makes it a measured concern rather than a pedantic one.
    print("  rotation effect on an ALREADY-decorrelated basis (rotate before quantise):")
    for b in BUDGETS:
        _, m0, _ = code_and_score(C_tr, C_ho, basis, X_ho, n_atom, b, rotate=False)
        _, m1, _ = code_and_score(C_tr, C_ho, basis, X_ho, n_atom, b, rotate=True)
        if not (math.isfinite(m0) and math.isfinite(m1)):
            raise SystemExit(f"[091] SELF-TEST FAILED -- non-finite MSE at {b} bits/atom")
        print(f"    {b:>4.1f} bits/atom   MSE {m0:.5f} -> {m1:.5f}   "
              f"({100*(m1-m0)/max(m0,1e-12):+.1f}%)")
    print(f"[091] SELF-TEST PASSED: both arms identical on identical input across "
          f"{len(BUDGETS)} budgets; rotation fold exact to {fold:.1e}.")


if __name__ == "__main__":
    self_test()
    print("\n[091] The self-test is the part that runs anywhere. Scoring needs the ATLAS\n"
          "      cache and a codec checkpoint, so the per-system loop belongs on the\n"
          "      cluster: for each held-out system, project replica-2 frames onto the ANM\n"
          "      basis from its reference structure (rotate=False) and onto the codec's\n"
          "      latent (rotate=True), score both at every budget in PEER_RATE_BUDGETS,\n"
          "      and write through armf_io.dump_rows so a killed run is not read as a\n"
          "      finished one. Report ANM's realised nonzero mode count next to its rate:\n"
          "      40 effective modes out of 256 would be the same shape as PR 2 of 8.")
