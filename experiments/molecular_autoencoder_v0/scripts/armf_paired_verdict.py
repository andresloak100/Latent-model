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


if __name__ == "__main__":
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
        print(f"  {'arm':>10}{'metric':>13}{'n_pos/n':>10}{'sign p':>10}{'median diff':>14}"
              f"{'overlap caught':>16}  time?")
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
                d = np.array(d); npos = int((d > 0).sum())
                p = st.binomtest(npos, len(d), 0.5).pvalue
                flag = "TIME" if m in TIME_AWARE else "static"
                star = " <-- paired test fires where overlap did not" \
                    if (p < 0.05 and caught == 0) else ""
                print(f"  {arm:>10}{m:>13}{f'{npos}/{len(d)}':>10}{p:>10.4f}"
                      f"{np.median(d):>14.4f}{f'{caught}/{len(d)}':>16}  {flag}{star}")
        print(f"\n  The paired test uses each system's DIFFERENCE from its own reference, so 77a's"
              f"\n  same-length one-estimator discipline is untouched; only the ACROSS-SYSTEM"
              f"\n  aggregation changes. `n_pos/n` is printed so a fire can be read as directional"
              f"\n  rather than taken on the p-value.")
