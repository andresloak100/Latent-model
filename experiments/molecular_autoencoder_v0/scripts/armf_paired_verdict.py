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
XMETRICS = ["std", "js", "kurt", "xcorr", "amp", "iat", "trans",
            "xcorr_top8", "amp_top8", "xcorr_top16", "amp_top16"]
TIME_AWARE = {"iat", "trans"}


def load(p):
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return d.get("rows", d)


def paired(d):
    """Both tests on one set of per-system differences, plus the symmetry check that decides which
    governs. Returned together so a caller cannot take the signed-rank p without its skew."""
    npos = int((d > 0).sum())
    p_sign = st.binomtest(npos, len(d), 0.5).pvalue
    try:
        p_wil = float(st.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:                     # all differences zero
        p_wil = 1.0
    sk = float(st.skew(d))
    return npos, p_sign, p_wil, sk, (abs(sk) > 1.0)


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
        print(f"  {'arm':>10}{'metric':>13}{'n_pos/n':>10}{'sign p':>10}{head}"
              f"{'median diff':>14}{'overlap caught':>16}  time?")
        for arm in arms:
            if arm == "OU": continue
            for m in XMETRICS:
                d, caught = [], 0
                for r in rows.values():
                    a = r["arms"].get(arm, {})
                    if m not in a.get("med", {}) or m not in r.get("ref_med", {}): continue
                    d.append(a["med"][m] - r["ref_med"][m])
                    if not a.get("inside", {}).get(m, True): caught += 1
                if len(d) < 4: continue
                d = np.array(d)
                npos, p_sign, p_wil, sk, skewed = paired(d)
                # 115c: where the differences are badly skewed the signed-rank assumption fails and
                # the SIGN TEST governs, decided by |skew| > 1 fixed in advance.
                p = p_wil if (WIL and not skewed) else p_sign
                flag = "TIME" if m in TIME_AWARE else "static"
                star = " <-- paired test fires where overlap did not" \
                    if (p < 0.05 and caught == 0) else ""
                if WIL and skewed: star += " [SKEWED: sign test governs]"
                extra = f"{p_wil:>12.4f}{sk:>8.2f}" if WIL else ""
                print(f"  {arm:>10}{m:>13}{f'{npos}/{len(d)}':>10}{p_sign:>10.4f}{extra}"
                      f"{np.median(d):>14.4f}{f'{caught}/{len(d)}':>16}  {flag}{star}")
        print(f"\n  The paired test uses each system's DIFFERENCE from its own reference, so 77a's"
              f"\n  same-length one-estimator discipline is untouched; only the ACROSS-SYSTEM"
              f"\n  aggregation changes. `n_pos/n` is printed so a fire can be read as directional"
              f"\n  rather than taken on the p-value.")
