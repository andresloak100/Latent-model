#!/usr/bin/env python3
"""INBOX 52d: does any run NAME cover two different trainings?

complex_d8 did. outputs/cluster/complex_d8 ends at epoch 899 / 63,000 steps / rmsd 4.799;
$WR/results/complex_d8 ends at 3456 / 241,990 / 6.722. Same name, two models, and every number
computed from "complex_d8" was silently one or the other.

WHY A NAME AUDIT AND NOT A HASH. The provenance block carries sha256 for exactly this, but 61 of the
64 committed run directories have NO final.pt -- there is nothing to hash, so a collision there is
undetectable by hashing. train_log.json is present for all 64, and its final row is a training
FINGERPRINT that needs no weights. One detected collision out of a possible 61 was never evidence
that there was only one.

No inference, no GPU, no checkpoint load: a directory walk over two trees.
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CL = os.path.join(REPO, "outputs", "cluster")
WRR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/results"
FIELDS = ("epoch", "steps", "elapsed_s", "rmsd")


def fingerprint(d):
    """Final training row: the identity of the run that wrote this directory."""
    p = os.path.join(d, "train_log.json")
    if not os.path.exists(p):
        return None
    try:
        log = (json.load(open(p)) or {}).get("log")
    except Exception:
        return None
    if not log or not isinstance(log[-1], dict):
        return None
    r = log[-1]
    return tuple(round(float(r[k]), 3) if isinstance(r.get(k), (int, float)) else r.get(k)
                 for k in FIELDS)


def main():
    if not os.path.isdir(CL):
        print(f"no {CL}"); return 1
    coll, compared, nockpt, unparsed = [], 0, 0, 0
    for name in sorted(os.listdir(CL)):
        a = os.path.join(CL, name)
        if not os.path.isdir(a):
            continue
        has = os.path.exists(os.path.join(a, "final.pt"))
        nockpt += (not has)
        fa = fingerprint(a)
        if fa is None:
            unparsed += 1
            continue
        b = os.path.join(WRR, name)
        fb = fingerprint(b) if os.path.isdir(b) else None
        if fb is None:
            continue
        compared += 1
        if fa != fb:
            coll.append((name, fa, fb, has))
    print(f"[collision-audit] {compared} run names compared against {WRR}")
    print(f"[collision-audit] {nockpt} of them have NO final.pt -- a sha256 cannot detect a "
          f"collision there, which is why this exists")
    if unparsed:
        print(f"[collision-audit] {unparsed} directories had no parseable train_log.json")
    print(f"\n[collision-audit] NAME COLLISIONS: {len(coll)}")
    for n, fa, fb, has in coll:
        print(f"  {n}   (final.pt committed: {has})")
        print(f"    committed : epoch={fa[0]} steps={fa[1]} elapsed_s={fa[2]} rmsd={fa[3]}")
        print(f"    $WR       : epoch={fb[0]} steps={fb[1]} elapsed_s={fb[2]} rmsd={fb[3]}")
    if not coll:
        print("  none -- every compared name refers to one training in both trees.")
    return 1 if coll else 0


if __name__ == "__main__":
    sys.exit(main())
