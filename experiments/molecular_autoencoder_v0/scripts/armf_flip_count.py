#!/usr/bin/env python3
"""INBOX 101b: how many of the systems the codec loses does the ANM projection FLIP?

101a's identity is exact and I have verified it:  dFVE_total = -FVE_perp x residual_share.
    1j8e_A  -(-0.1580) x 0.214 = +0.0338   measured +0.0338
    1fd3_A  -(+0.0066) x 0.384 = -0.0025   measured -0.0025

ONE CORRECTION TO 101b, AND IT CHANGES WHAT IS REDUNDANT. 101b says 10343261 recomputes a quantity
099 already determines, so the flip count is arithmetic over 099's output. The identity is right but
099 does not carry `codec_total` -- its rows hold sst_total, sst_perp, perp_frac, and the three
ORTHOGONAL scores. dFVE is computable from 099; the ABSOLUTE total it must be added to is not. So
10343261 is not redundant: it is the only run producing codec_total for this checkpoint, and the flip
count needs both.

A SECOND AND LARGER ONE. The 0/123 headline comes from tied_peer_n300.json -- the ModalCodec TIED arm
at n_train=300 (ANM median +0.6619, codec +0.1747, codec wins 0.0%). 099 and 100a score
L1_n130_dm256_dl256_lr0.0001_s0_z0.pt, which is armf_atlas_dm.Codec at n_train=130. DIFFERENT MODELS.
Asking "how many of the 123 does the projection flip" about the 0/123 codec using 099's FVE_perp
would be one name, two things across two codecs -- the ninth instance, in the answer to a question
about the eighth.

So this reports the flip count for THE MODEL ACTUALLY MEASURED, and states plainly that the same
question about the tied arm requires re-running 099 against that checkpoint. ANM's total FVE is
model-independent and is recomputed here from the same reference structures rather than transferred,
with the transfer checked against tied_peer as a cross-check rather than assumed.
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from armf_atlas_data import AtlasStore, sysdata, ho_frames
from armf_anm import modes as anm_modes
from armf_tied_peer import anm_fve
import armf_atlas_dm as D

WR = D.WR
K = int(os.environ.get("FC_K", "256"))
CUTOFF = float(os.environ.get("FC_CUTOFF", "5.0"))
PJ = os.environ.get("FC_PJ", f"{WR}/project_to_anm.json")
AO = os.environ.get("FC_AO", f"{WR}/anm_orthogonal.json")
RES = f"{WR}/flip_count.json"


def load(p):
    if not os.path.exists(p): return {}
    d = json.load(open(p))
    return d.get("rows", d)


if __name__ == "__main__":
    pj, ao = load(PJ), load(AO)
    common = sorted(set(pj) & set(ao))
    print(f"[101b] projanm has {len(pj)} systems, 099 has {len(ao)}, overlap {len(common)}", flush=True)
    if not common:
        raise SystemExit("  no overlap yet -- both runs must reach the same systems first")

    # --- 101a identity, checked on every overlapping system rather than the two already seen ---
    err = []
    for k in common:
        pred = -ao[k]["codec"] * ao[k]["perp_frac"]
        err.append(abs(pred - (pj[k]["fve_proj"] - pj[k]["fve_raw"])))
    err = np.array(err)
    print(f"  101a identity across {len(common)} systems: max |predicted - measured| = {err.max():.2e}, "
          f"median {np.median(err):.2e}")
    print(f"  -> {'IDENTITY HOLDS; the two harnesses agree' if err.max() < 1e-6 else 'IDENTITY VIOLATED -- one harness is wrong'}",
          flush=True)

    store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    tp = load(f"{WR}/tied_peer_n300.json")
    rows, t0 = {}, time.time()
    for i, k in enumerate(common, 1):
        try:
            d = sysdata(store, have[k])
            V, _, _ = anm_modes(d["ref"], K, CUTOFF)
            a = anm_fve(d, V) if V is not None else float("nan")
            rows[k] = dict(N=pj[k]["N"], anm=a, codec=pj[k]["fve_raw"], proj=pj[k]["fve_proj"],
                           delta=pj[k]["fve_proj"] - pj[k]["fve_raw"],
                           anm_tied_peer=tp.get(k, {}).get("anm"))
        except Exception as e:
            print(f"  {k}: FAIL {type(e).__name__}: {e}", flush=True)
    if not rows:
        raise SystemExit("  nothing scored")
    A = np.array([r["anm"] for r in rows.values()])
    C = np.array([r["codec"] for r in rows.values()])
    P = np.array([r["proj"] for r in rows.values()])
    ref = np.array([r["anm_tied_peer"] if r["anm_tied_peer"] is not None else np.nan
                    for r in rows.values()])
    m = ~np.isnan(ref)
    if m.any():
        print(f"\n  ANM cross-check vs tied_peer_n300 on {m.sum()} systems: "
              f"max |diff| {np.abs(A[m]-ref[m]).max():.2e} -> "
              f"{'same ANM, transfer valid' if np.abs(A[m]-ref[m]).max() < 1e-6 else 'DIFFERENT ANM -- do not transfer'}")
    lost = C < A
    flip = (C < A) & (P > A)
    print(f"\n=== 101b: FLIP COUNT ({len(rows)} systems, atlas_dm Codec n_train=130 DM=256) ===")
    print(f"  codec loses to ANM on {lost.sum()}/{len(rows)}")
    print(f"  the ANM projection FLIPS {flip.sum()} of them ({100*flip.sum()/max(lost.sum(),1):.1f}%)")
    marg = A[lost] - C[lost]
    dl = (P - C)[lost]
    print(f"\n  margin to ANM on the lost systems : median {np.median(marg):+.4f}, "
          f"IQR [{np.percentile(marg,25):+.4f}, {np.percentile(marg,75):+.4f}]")
    print(f"  what the projection moves         : median {np.median(dl):+.4f}, "
          f"IQR [{np.percentile(dl,25):+.4f}, {np.percentile(dl,75):+.4f}]")
    print(f"  ratio (movement / margin)         : median {np.median(dl)/max(np.median(marg),1e-9):.3f}")
    print(f"  projection helps on {100*(dl>0).mean():.1f}% of the lost systems, hurts on "
          f"{100*(dl<0).mean():.1f}%")
    if flip.sum() == 0:
        print(f"\n  -> ZERO FLIPS. The projection is a real and almost uniform improvement, and it is\n"
              f"     {np.median(dl)/max(np.median(marg),1e-9):.0%} of the median margin -- it moves the codec but does not\n"
              f"     reach ANM on a single system. Reported plainly rather than as a near-miss.")
    else:
        print(f"\n  -> {flip.sum()} SYSTEMS FLIP at zero training cost, using only the reference structure.")
    json.dump(dict(n=len(rows), lost=int(lost.sum()), flipped=int(flip.sum()),
                   median_margin=float(np.median(marg)), median_delta=float(np.median(dl)),
                   identity_max_err=float(err.max()), rows=rows), open(RES, "w"))
