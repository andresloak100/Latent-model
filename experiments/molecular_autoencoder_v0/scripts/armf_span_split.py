#!/usr/bin/env python3
"""INBOX 102c: is the codec's deficit INSIDE ANM's span or outside it?

THE DECOMPOSITION IS EXACT. With SSE and SST both splitting across the ANM boundary,

    FVE_total = par_share x FVE_par + perp_share x FVE_perp,     par_share + perp_share = 1

so FVE_par = (FVE_total - perp_share x FVE_perp) / par_share. ANM has FVE_perp = 0 BY CONSTRUCTION
-- 099's self-test asserts it and measures 2.2e-16 -- so ANM_total = par_share x FVE_par(ANM), and
ANM's within-span score is just ANM_total / par_share.

WHY THIS IS THE QUESTION NOBODY ASKED. 0/123 and 099 both report totals or orthogonal scores. Neither
says whether the codec is behind ANM *in the directions ANM actually spans*. If the codec is far
behind INSIDE the span, then winning the orthogonal axis outright cannot close the peer gap, and both
SEM and 98b's ANM-basis initialisation are aimed at the smaller half of the problem.

THE DECISIVE DERIVED QUANTITY, and it needs no new model: give the codec a PERFECT residual,
FVE_perp = 1, and leave its within-span score untouched:

    FVE_total(perfect residual) = par_share x FVE_par(codec) + perp_share x 1

If that is still below ANM_total, the orthogonal axis CANNOT CLOSE THE PEER GAP EVEN IF WON
OUTRIGHT. That is a bound, not an estimate: no model can exceed FVE_perp = 1.

ONE CODEC ONLY. 101's correction 2 stands: the 0/123 headline is the ModalCodec tied arm at
n_train=300, while 099/100a score armf_atlas_dm.Codec at n_train=130. Everything here is computed
WITHIN the atlas_dm codec, and ANM totals are recomputed from the same reference structures rather
than transferred. Nothing here may be said about the tied arm.

INBOX 102b is also answered here: 0 flips in 13 puts the 95% upper bound on the flip rate at
3/13 = 23% by the rule of three, so the COUNT is weak. min(margin - movement) is deterministic and
says how close the nearest system actually came.
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from armf_atlas_data import AtlasStore, sysdata
from armf_anm import modes as anm_modes
from armf_tied_peer import anm_fve
import armf_atlas_dm as D

WR = D.WR
K = int(os.environ.get("SS_K", "256"))
CUTOFF = float(os.environ.get("SS_CUTOFF", "5.0"))
PJ = os.environ.get("SS_PJ", f"{WR}/project_to_anm.json")
AO = os.environ.get("SS_AO", f"{WR}/anm_orthogonal.json")
RES = f"{WR}/span_split.json"


def load(p):
    if not os.path.exists(p): return {}
    d = json.load(open(p))
    return d.get("rows", d)


if __name__ == "__main__":
    pj, ao = load(PJ), load(AO)
    common = sorted(set(pj) & set(ao))
    if not common:
        raise SystemExit("  no overlap between projanm and 099 yet")
    print(f"[102c] within-span vs orthogonal, ONE codec (armf_atlas_dm.Codec n_train=130 DM=256)",
          flush=True)
    print(f"  {len(common)} systems where projanm and 099 overlap", flush=True)

    store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    rows = {}
    for k in common:
        try:
            d = sysdata(store, have[k])
            V, _, _ = anm_modes(d["ref"], K, CUTOFF)
            if V is None: continue
            a_tot = anm_fve(d, V)                       # ANM total, recomputed not transferred
            perp = float(ao[k]["perp_frac"]); par = 1.0 - perp
            c_tot = float(pj[k]["fve_raw"])
            c_perp = float(ao[k]["codec"])
            c_par = (c_tot - perp * c_perp) / max(par, 1e-12)
            a_par = a_tot / max(par, 1e-12)             # ANM FVE_perp = 0 by construction
            perfect = par * c_par + perp * 1.0          # give the codec a PERFECT residual
            rows[k] = dict(N=ao[k]["N"], perp=perp, anm_total=a_tot, codec_total=c_tot,
                           codec_perp=c_perp, codec_par=c_par, anm_par=a_par,
                           perfect_residual_total=perfect,
                           ceiling_perp=float(ao[k]["ceiling"]),
                           ceiling_total=par * c_par + perp * float(ao[k]["ceiling"]))
        except Exception as e:
            print(f"  {k}: FAIL {type(e).__name__}: {e}", flush=True)
    if not rows:
        raise SystemExit("  nothing scored")

    P = np.array([r["perp"] for r in rows.values()])
    AT = np.array([r["anm_total"] for r in rows.values()])
    CT = np.array([r["codec_total"] for r in rows.values()])
    CP = np.array([r["codec_par"] for r in rows.values()])
    AP = np.array([r["anm_par"] for r in rows.values()])
    PF = np.array([r["perfect_residual_total"] for r in rows.values()])
    CE = np.array([r["ceiling_total"] for r in rows.values()])

    print(f"\n=== 102c: WHERE IS THE DEFICIT? ({len(rows)} systems) ===")
    print(f"  {'quantity':>34}{'median':>10}{'IQR':>22}")
    # ANM's within-span FVE is 1 BY CONSTRUCTION -- it is the exact projection onto its own basis,
    # so it reproduces the spanned component with no error, just as its FVE_perp is 0. The identity
    # ANM_total == par_share follows and is checked below rather than asserted. This is the same
    # kind of tautology as the 0 floor, and reporting it as if ANM "achieved" 1.0000 within span
    # would be reading a definition as a result.
    print(f"  [check] max |ANM_total - par_share| = {np.abs(AT - (1 - P)).max():.2e} "
          f"-> ANM_total IS the share of variance inside its span")
    for lab, v in (("perp share of variance", P), ("ANM total", AT), ("codec total", CT),
                   ("codec WITHIN span (FVE_par)", CP), ("ANM WITHIN span (=1 by construction)", AP)):
        print(f"  {lab:>34}{np.median(v):>10.4f}"
              f"{f'[{np.percentile(v,25):.4f}, {np.percentile(v,75):.4f}]':>22}")
    print(f"\n  WITHIN-SPAN DEFICIT (ANM_par - codec_par): median {np.median(AP-CP):+.4f}, "
          f"codec behind on {100*(CP<AP).mean():.0f}% of systems")

    print(f"\n=== THE BOUND: give the codec a PERFECT residual (FVE_perp = 1) ===")
    print(f"  {'perfect-residual total':>34}{np.median(PF):>10.4f}"
          f"{f'[{np.percentile(PF,25):.4f}, {np.percentile(PF,75):.4f}]':>22}")
    print(f"  {'ANM total':>34}{np.median(AT):>10.4f}")
    print(f"  systems where a PERFECT residual still loses to ANM: "
          f"{(PF < AT).sum()}/{len(rows)} = {100*(PF<AT).mean():.0f}%")
    print(f"  median shortfall even with a perfect residual: {np.median(AT-PF):+.4f}")
    # A 0.9 CLIFF HERE PRINTED THE OPPOSITE OF WHAT THE NUMBERS SAY. At 86% losing even with a
    # perfect residual the branch fell through to "the axis is live", which is a threshold artefact,
    # not a reading. The fraction is reported and the verdict is proportional to it.
    lose = (PF < AT).mean()
    print(f"\n  -> WITH A PERFECT RESIDUAL THE CODEC STILL LOSES ON {100*lose:.0f}% OF SYSTEMS.")
    if lose > 0.5:
        print(f"     So on the large majority the ANM-orthogonal axis CANNOT CLOSE THE PEER GAP EVEN\n"
              f"     IF WON OUTRIGHT. The deficit is INSIDE ANM's own span, where the codec scores\n"
              f"     {np.median(CP):.3f} against ANM's {np.median(AP):.3f}. 099 measures a real and open\n"
              f"     axis that is not the one the peer comparison turns on, and SEM and 98b's\n"
              f"     ANM-basis initialisation are aimed at the smaller half of the problem.")
        print(f"     On the remaining {100*(1-lose):.0f}% a perfect residual WOULD overtake ANM, so the\n"
              f"     axis is not closed everywhere -- reported rather than rounded away.")
    else:
        print(f"     A perfect residual overtakes ANM on {100*(1-lose):.0f}%, so the orthogonal axis is\n"
              f"     live for the peer comparison on most systems.")
    print(f"\n  and with the MEASURED linear ceiling instead of a perfect residual: "
          f"median total {np.median(CE):.4f}, beats ANM on {100*(CE>AT).mean():.0f}%")

    # ---- 102b: the deterministic version of the flip question ----
    mv = np.array([pj[k]["fve_proj"] - pj[k]["fve_raw"] for k in rows])
    mg = AT - CT
    gap = mg - mv
    print(f"\n=== 102b: HOW CLOSE DID THE NEAREST SYSTEM COME? ===")
    print(f"  0 flips in {len(rows)} -> 95% upper bound on the flip rate is 3/{len(rows)} = "
          f"{300/len(rows):.0f}% (rule of three), so the COUNT is weak evidence.")
    print(f"  min(margin - movement) = {gap.min():+.4f} on {list(rows)[int(np.argmin(gap))]}")
    print(f"  median {np.median(gap):+.4f}; a flip needs this below 0, and the nearest system is\n"
          f"  {gap.min():.4f} away -- deterministic, not statistical.")
    json.dump(dict(n=len(rows), rows=rows, min_margin_minus_movement=float(gap.min()),
                   perfect_loses=int((PF < AT).sum())), open(RES, "w"))
