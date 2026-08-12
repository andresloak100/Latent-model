#!/usr/bin/env python3
"""INBOX 113d: the PAIRED verdict across systems, which per-system interval overlap discards.

WHY THE CURRENT RULE HAS NO POWER. 77a/77b designed interval overlap for ONE system with many
windows, and it is still right for that. But there are now 24 held-out systems, and asking "does this
band overlap" 24 separate times throws away the pairing. Measured on a synthetic control -- reference
tau = 147 frames (the corrected collective IAT, 5.88 ns) against a model with a wrong tau, T=256,
18 windows, 24 systems:

    model tau   x wrong   per-system overlap caught   paired sign test p   n_pos/n
           40       3.7                      0 / 24               0.0000      0/24
           74       2.0                      0 / 24               0.3075      9/24
          100       1.5                      0 / 24               0.0639      7/24
          120       1.2                      0 / 24               1.0000     12/24
          147       1.0 (null)               0 / 24               0.1516      8/24

PER-SYSTEM OVERLAP CATCHES NOTHING AT ANY ERROR SIZE. The paired test catches a 3.7x error at
p < 0.0001 and does NOT fire on the null, so this is added power rather than a looser threshold.

MY NUMBERS DIVERGE FROM 113d's AT 2x AND 1.5x -- it reported p < 0.0001 there and I get 0.31 and
0.064. The likely cause is the estimator's own bias: at T=256 my iat_series recovers ~34% of a
147-frame truth where 113d measured 13%, and a model with SHORTER tau is less biased than the
reference, which compresses the paired difference. Reported as measured rather than reconciled,
because the conclusion that matters -- overlap catches nothing, pairing catches the large errors --
holds under both.

NOTHING IN 77a IS GIVEN UP. Each system's difference is still two same-length series through one
estimator; the pairing is applied AFTER that, to the per-system differences.

WHY A SEPARATE FILE. The per-system results are already persisted, so this reads them rather than
requiring a re-run -- the same division as armf_propagator_report.py for 82a.

INBOX 115c: WILCOXON SIGNED-RANK IS PRE-REGISTERED AS THE PRIMARY VERDICT FOR onestep_diff, AND FOR
NOTHING ALREADY REPORTED. The sign test discards magnitude and keeps only the sign, which 114's MDE
column exposed exactly: xcorr_top8 has |diff| = 0.0592 against MDE = 0.0510 -- the difference EXCEEDS
the minimum detectable effect and the test still does not fire. Simulated at n=24, 4,000 reps, the
signed-rank test roughly doubles power over the whole relevant range (+14.6 to +27.5 points from
0.3 sd to 0.8 sd) and is calibrated at the null where the sign test sits at 1.8%, i.e. conservative
from discreteness, and conservative means under-powered.

TWO CONSTRAINTS, AND THE FIRST IS NOT OPTIONAL.
  1. NOT APPLIED TO lv_r8's HEADLINE. xcorr_top8 at p = 0.0639 is a reported result, and switching to
     a more powerful test AFTER seeing that it nearly fired is the move 114.1 refused. lv_r8 stands
     as reported: sign test, p = 0.0639, marginal. The `--wilcoxon` flag is OFF by default and the
     printed table says which test governs, so the default path cannot silently restate a headline.
  2. SKEW IS REPORTED BESIDE IT. Signed-rank assumes roughly symmetric differences. The skew of the
     per-system differences is printed in its own column, and where |skew| > 1 the SIGN TEST GOVERNS
     and the row is marked. That rule is fixed here, before onestep_diff exists, rather than chosen
     once the two tests are seen to disagree.

Per-system interval overlap is still printed, third, because it is what 77a designed and what 113d
measured as powerless across systems -- keeping it visible is what makes the power claim checkable.
"""
import sys, os, json
import numpy as np
from scipy import stats as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

WR = os.environ["WR"]
UNEVAL = "UNEVALUABLE"      # 116.4: not a p-value, and must not sort as one
MIN_EFF = int(os.environ.get("PV_MIN_EFF", "12"))   # half of n=24; below this a fire is small-n
XMETRICS = ["std", "js", "kurt", "xcorr", "amp", "iat", "trans",
            "xcorr_top8", "amp_top8", "xcorr_top16", "amp_top16"]
TIME_AWARE = {"iat", "trans"}


def load(p):
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return d.get("rows", d)


def paired(d, ref=None):
    """Both tests on one set of per-system differences, plus the symmetry check that decides which
    governs. Returned together so a caller cannot take the signed-rank p without its skew.

    115a FOUND A DEFECT IN THE SIGN TEST AS 113d INSTALLED IT: `(d > 0).sum()` counts an EXACT ZERO
    as a negative, so a difference vector of all zeros gives n_pos = 0 and binomtest(0, 24, 0.5) =
    1.2e-07 -- the most extreme p obtainable, on 24 identical numbers. It surfaced on the SHUFFLE
    arm, where 9 of 11 metrics are invariant to row permutation BY CONSTRUCTION, so their true
    difference is zero and the observed one is float rounding: xcorr fired at p = 0.0066 on a
    difference of 1.1e-16 against a reference value of 0.229.

    THE TOLERANCE IS NOT A NEW KNOB. 108.1 already fixed |rel| < 1e-6 as this project's meaning of
    "this statistic did not move", and that is the threshold applied here. Ties are DROPPED and their
    count reported, which is Wilcoxon's own convention, so the test is run on the differences that
    exist rather than on rounding noise.

    AUDITED AGAINST EVERYTHING ALREADY REPORTED: lv_r8 has ZERO ties on all eleven metrics, so no
    headline number changes. latent_video_onestep's `trans` has one tie and moves p 0.0227 -> 0.0106,
    the same verdict. Nothing on the record is invalidated by this fix.

    116.4 ADDS TWO THINGS THE TIE FIX MADE NECESSARY.
      n_effective IS RETURNED AND PRINTED BESIDE EVERY p. Dropping ties shrinks n silently, and a
      metric with 18 ties and 6/6 positive returns p = 0.031 on SIX systems -- a number that reads
      like the 24-system result next to it. n_eff is now the denominator in every printed n_pos/n,
      and rows below MIN_EFF are marked so a small-n fire cannot be mistaken for a large-n one.
      UNEVALUABLE REPLACES nan. Wilcoxon returns nan when every difference is a tie, and a nan p
      reads as "not significant" to anything that sorts or filters -- it would be silently ranked
      as the safest row in the table. UNEVALUABLE is returned instead, and it is not a p-value.
    """
    d = np.asarray(d, dtype=float)
    scale = np.maximum(np.abs(np.asarray(ref, dtype=float)), 1e-12) if ref is not None else 1.0
    keep = np.abs(d) > 1e-6 * scale
    nt = int((~keep).sum())
    k = d[keep]
    if len(k) == 0:      # every difference is a tie: invariant by construction, NOT "not significant"
        return 0, 0, UNEVAL, UNEVAL, float("nan"), False, nt
    npos = int((k > 0).sum())
    p_sign = st.binomtest(npos, len(k), 0.5).pvalue
    try:
        p_wil = float(st.wilcoxon(k, zero_method="wilcox").pvalue)
    except ValueError:
        p_wil = UNEVAL
    # `p_wil == p_wil` is already False for a nan, so that guard could never fire. scipy returns
    # nan when the signed-rank statistic has no variance left, and nan must not reach the table.
    if isinstance(p_wil, float) and np.isnan(p_wil):
        p_wil = UNEVAL
    sk = float(st.skew(k)) if len(k) > 2 else float("nan")
    return npos, len(k), p_sign, p_wil, sk, (abs(sk) > 1.0 if sk == sk else False), nt


def fires(p):
    """UNEVALUABLE is not a p-value and must never satisfy a `< 0.05` test."""
    return isinstance(p, float) and p == p and p < 0.05


def pfmt(p, w=10):
    return f"{'UNEVALUABLE':>{w}}" if not isinstance(p, float) or p != p else f"{p:>{w}.4f}"


if __name__ == "__main__":
    # 115c: OFF by default, so the default path cannot restate a headline under a test chosen after
    # the fact. Pre-registered ON for onestep_diff, and for nothing already on the record.
    WIL = "--wilcoxon" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--wilcoxon"]
    files = sys.argv[1:] or [f"{WR}/latent_video_joint.json",
                             f"{WR}/latent_video_joint_r8.json",
                             f"{WR}/latent_video_onestep.json"]
    for f in files:
        rows = load(f)
        if not rows:
            print(f"  {os.path.basename(f)}: ABSENT"); continue
        arms = sorted({a for r in rows.values() for a in r["arms"]
                       if a not in ("POWER", "PROFILE", "geometry")})
        print(f"\n=== {os.path.basename(f)} -- {len(rows)} systems, arms {arms} ===")
        print(f"  PRIMARY VERDICT: {'WILCOXON signed-rank (115c)' if WIL else 'SIGN TEST'}"
              f"{'; sign test beside it as the conservative check' if WIL else ''}")
        head = f"{'wilcoxon p':>12}{'skew':>8}" if WIL else ""
        print(f"  {'arm':>10}{'metric':>13}{'n_pos/n_eff':>12}{'ties':>6}{'sign p':>12}{head}"
              f"{'median diff':>14}{'overlap caught':>16}  time?")
        for arm in arms:
            if arm == "OU": continue
            for m in XMETRICS:
                d, ref, caught = [], [], 0
                for r in rows.values():
                    a = r["arms"].get(arm, {})
                    if m not in a.get("med", {}) or m not in r.get("ref_med", {}): continue
                    d.append(a["med"][m] - r["ref_med"][m]); ref.append(r["ref_med"][m])
                    if not a.get("inside", {}).get(m, True): caught += 1
                if len(d) < 4: continue
                d = np.array(d)
                npos, nk, p_sign, p_wil, sk, skewed, nt = paired(d, ref)
                # 115c: where the differences are badly skewed the signed-rank assumption fails and
                # the SIGN TEST governs, decided by |skew| > 1 fixed in advance.
                p = p_wil if (WIL and not skewed) else p_sign
                flag = "TIME" if m in TIME_AWARE else "static"
                star = " <-- paired test fires where overlap did not" \
                    if (fires(p) and caught == 0) else ""
                if WIL and skewed: star += " [SKEWED: sign test governs]"
                if nt == len(d): star += " [ALL TIED: invariant, not significant]"
                # 116.4: a fire on a shrunken n reads like a fire on the full n unless it is said.
                elif fires(p) and nk < MIN_EFF:
                    star += f" [SMALL n_eff={nk} of {len(d)}: {nt} ties dropped]"
                extra = f"{pfmt(p_wil, 12)}{sk:>8.2f}" if WIL else ""
                print(f"  {arm:>10}{m:>13}{f'{npos}/{nk}':>12}{nt:>6}{pfmt(p_sign, 12)}{extra}"
                      f"{np.median(d):>14.4f}{f'{caught}/{len(d)}':>16}  {flag}{star}")
        print(f"\n  The paired test uses each system's DIFFERENCE from its own reference, so 77a's"
              f"\n  same-length one-estimator discipline is untouched; only the ACROSS-SYSTEM"
              f"\n  aggregation changes. `n_pos/n` is printed so a fire can be read as directional"
              f"\n  rather than taken on the p-value.")
