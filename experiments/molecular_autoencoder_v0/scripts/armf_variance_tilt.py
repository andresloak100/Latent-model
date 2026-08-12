#!/usr/bin/env python3
"""INBOX 114 item 2: does ONE variance tilt explain all four observations?

FOUR OBSERVATIONS FROM lv_r8, each currently separate:
    xcorr    below reference,  p = 0.0003
    trans    above reference,  p = 0.0227, +80.4
    CA span  above the rank-64 reconstruction, 3.69 vs 3.566 A
    std      matched, 12/24, p = 1.0000

A VARIANCE TILT from slow modes toward fast modes produces all four and keeps the 64-mode mean
constant:
    fast ANM modes are less mutually coupled  -> weight shifted to them LOWERS xcorr
    fast modes cross basin thresholds more often -> trans RISES
    fast modes carry local deformation        -> consecutive-CA spacing INFLATES
    std is X.std(0).mean(), a MEAN over 64 modes -> BLIND BY CONSTRUCTION to a tilt that moves
                                                    variance between modes without changing the total

PREDICTION, stated before the numbers: deficit in the slow block, excess in the fast block, sum
unchanged. If the blocks are flat the hypothesis is DEAD and the CA anomaly stays open. The same
breakdown is run on iat, because a tilt should also SHORTEN the generated iat -- and the full
eleven-metric table already shows iat at 7/24 with a negative median difference, which is that
prediction's direction.

Modes arrive slowest-first from anm_modes, so [:8] is the slowest block.
"""
import sys, os, json, time
import numpy as np, torch
from scipy import stats as st
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE + "/.."
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from molae.latent_video import LatentVideoConfig, LatentVideoDiffusion
from armf_atlas_data import AtlasStore, sysdata
from armf_anm import modes as anm_modes
from armf_propagator import iat_series
import armf_atlas_dm as D

WR = D.WR
K, RGRP = 64, int(os.environ.get("VT_R", "8"))
DLAT = K // RGRP
T = int(os.environ.get("VT_T", "256"))
NEVAL = int(os.environ.get("VT_NEVAL", "24"))
KEVAL = int(os.environ.get("VT_KEVAL", "18"))
CUTOFF = 5.0
CKPT = os.environ.get("VT_CKPT", f"{WR}/latent_video_ckpt_joint.pt")
BLOCKS = [("slow 0-7", 0, 8), ("mid 8-47", 8, 48), ("fast 48-63", 48, 64)]
RES = f"{WR}/variance_tilt.json"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho = [p for p in man["heldout"] if p in have][:NEVAL]
    mdl = LatentVideoDiffusion(LatentVideoConfig(latent_dim=DLAT, d_model=256, depth=6,
                                                 max_frames_hint=T)).to(dev)
    mdl.load_state_dict(torch.load(CKPT, map_location=dev)); mdl.eval()
    print(f"[114.2] variance tilt: R={RGRP}xD={DLAT}, T={T}, {len(ho)} systems, ckpt "
          f"{os.path.basename(CKPT)}", flush=True)
    rng = np.random.default_rng(0)
    rows = {}
    for i, p in enumerate(ho, 1):
        try:
            d = sysdata(store, have[p])
            V, _, _ = anm_modes(d["ref"], K, CUTOFF)
            if V is None: continue
            a = np.load(d["path"], mmap_mode="r")
            C0 = (np.asarray(a[0]).astype(np.float64).reshape(d["F"], -1) - d["mu"]) @ V
            sd = C0.std(0) + 1e-8
            ref = np.concatenate([((np.asarray(a[r]).astype(np.float64).reshape(d["F"], -1)
                                    - d["mu"]) @ V) / sd for r in (1, 2)])
            nseg = (len(ref) - 1) // T
            st_ = rng.permutation(nseg)[:min(KEVAL, nseg)] * T
            R = np.stack([ref[s:s + T] for s in st_])
            with torch.no_grad():
                G = mdl.sample((len(R), T, RGRP, DLAT), dev, steps=50).cpu().numpy()
            G = G.reshape(len(R), T, K)
            rows[p] = dict(N=d["N"], blocks={}, iat={})
            for lab, lo, hi in BLOCKS:
                rs = float(np.mean([w[:, lo:hi].std(0).mean() for w in R]))
                gs = float(np.mean([w[:, lo:hi].std(0).mean() for w in G]))
                ri = float(np.median([iat_series(w[:, k]) for w in R
                                      for k in range(lo, hi, max(1, (hi - lo) // 4))]))
                gi = float(np.median([iat_series(w[:, k]) for w in G
                                      for k in range(lo, hi, max(1, (hi - lo) // 4))]))
                rows[p]["blocks"][lab] = dict(ref=rs, gen=gs, diff=gs - rs)
                rows[p]["iat"][lab] = dict(ref=ri, gen=gi, diff=gi - ri)
            tot_r = float(np.mean([w.std(0).mean() for w in R]))
            tot_g = float(np.mean([w.std(0).mean() for w in G]))
            rows[p]["total"] = dict(ref=tot_r, gen=tot_g, diff=tot_g - tot_r)
            print(f"  [{i}/{len(ho)}] {p:10s} " + "  ".join(
                f"{lab.split()[0]} {rows[p]['blocks'][lab]['diff']:+.4f}" for lab, _, _ in BLOCKS)
                + f"   total {tot_g - tot_r:+.4f}", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(ho)}] {p}: FAIL {type(e).__name__}: {e}", flush=True)
        json.dump(rows, open(RES, "w"))

    if not rows: raise SystemExit("  nothing scored")
    print(f"\n=== 114.2 VERDICT ({len(rows)} systems) ===")
    print(f"  {'block':>12}{'n_pos/n':>10}{'sign p':>10}{'median diff':>14}  prediction")
    for lab, _, _ in BLOCKS:
        dd = np.array([r["blocks"][lab]["diff"] for r in rows.values()])
        npos = int((dd > 0).sum()); p = st.binomtest(npos, len(dd), 0.5).pvalue
        pred = "DEFICIT expected" if lab.startswith("slow") else \
               ("EXCESS expected" if lab.startswith("fast") else "")
        print(f"  {lab:>12}{f'{npos}/{len(dd)}':>10}{p:>10.4f}{np.median(dd):>14.4f}  {pred}")
    tt = np.array([r["total"]["diff"] for r in rows.values()])
    print(f"  {'TOTAL':>12}{f'{int((tt>0).sum())}/{len(tt)}':>10}"
          f"{st.binomtest(int((tt>0).sum()),len(tt),0.5).pvalue:>10.4f}{np.median(tt):>14.4f}"
          f"  unchanged expected")
    print(f"\n  {'iat block':>12}{'n_pos/n':>10}{'sign p':>10}{'median diff':>14}")
    for lab, _, _ in BLOCKS:
        dd = np.array([r["iat"][lab]["diff"] for r in rows.values()])
        npos = int((dd > 0).sum())
        print(f"  {lab:>12}{f'{npos}/{len(dd)}':>10}"
              f"{st.binomtest(npos,len(dd),0.5).pvalue:>10.4f}{np.median(dd):>14.4f}")
    slow = np.array([r["blocks"][BLOCKS[0][0]]["diff"] for r in rows.values()])
    fast = np.array([r["blocks"][BLOCKS[2][0]]["diff"] for r in rows.values()])
    tilt = (np.median(slow) < 0) and (np.median(fast) > 0)
    if tilt:
        print("\n  -> TILT CONFIRMED: deficit in slow, excess in fast, total unchanged. The four")
        print("     observations collapse into ONE defect, and per-mode loss weighting is the remedy.")
    else:
        print("\n  -> NO TILT: the blocks do not show the predicted pattern, so the hypothesis is")
        print("     DEAD and the CA anomaly stays open. Reported as a failed hypothesis, which is")
        print("     what it is -- 114d was proposed as one that CAN fail.")
