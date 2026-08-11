#!/usr/bin/env python3
"""INBOX 100a: is the codec's loss MISSING output, or SPURIOUS output?

A NEGATIVE ORTHOGONAL FVE HAS A MECHANICAL READING. For a prediction p uncorrelated with the residual
r, FVE_perp = -||p||^2 / ||r||^2. So 4ued_B's -1.2753 does not say the codec fails to predict the
residual; it says the codec emits 1.28x MORE energy into ANM-orthogonal directions than the true
motion contains there. Spurious output, not missing output, and a different defect with a different
fix.

THE TEST IS FREE AND IT IS A REAL INTERVENTION, NOT A DIAGNOSTIC. Project the codec's reconstruction
onto the ANM span, discard the orthogonal component, and re-score TOTAL FVE on the same held-out
frames with the same sst. The projection uses only the reference structure -- ANM is zero-shot -- so
this is something a deployed codec could actually do.

  TOTAL FVE IMPROVES  the codec has been losing to ANM partly through its OWN spurious
                      high-frequency content, and a post-hoc projection improves the headline at zero
                      training cost
  TOTAL FVE DOES NOT  the spurious energy is incidental and the loss is entirely about what the codec
                      MISSES, which closes this reading

Reported per system and never pooled with the sign, because a projection that helps badly-behaved
systems and hurts well-behaved ones would average to nothing while being two findings.
"""
import sys, os, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from armf_atlas_data import AtlasStore, sysdata, ho_frames
from armf_anm import modes as anm_modes
import armf_atlas_dm as D
from armf_atlas_dm import Codec
import armf_io

WR = D.WR
DM = int(os.environ.get("PJ_DM", "256"))
NT = int(os.environ.get("PJ_NTRAIN", "130"))
LR = os.environ.get("PJ_LR", "0.0001")
SEED = int(os.environ.get("PJ_SEED", "0"))
CUTOFF = float(os.environ.get("PJ_CUTOFF", "5.0"))
CHUNK = 8
RES = os.environ.get("PJ_RES", f"{WR}/project_to_anm.json")
CKPT = f"{WR}/atlas_dm_ckpt/L1_n{NT}_dm{DM}_dl{DM}_lr{LR}_s{SEED}_z0.pt"
dev = D.dev


@torch.no_grad()
def fve_pair(mdl, d, V):
    """TOTAL FVE of the raw codec and of the ANM-projected codec, in ONE pass over the same frames
    and against the SAME sst -- so the two numbers cannot drift apart through the estimator."""
    st = torch.tensor(d["stat"], device=dev)
    sse_raw = sse_prj = 0.0
    e_par = e_perp = 0.0
    for s in range(0, d["F"], CHUNK):
        e = min(s + CHUNK, d["F"])
        t = ho_frames(d, s, e)                                  # centred, (k, N, 3)
        dw = torch.tensor((t / d["scale"]).astype(np.float32), device=dev)
        p = mdl(st.unsqueeze(0).expand(e - s, -1, -1), dw).cpu().numpy().astype(np.float64) * d["scale"]
        P = p.reshape(e - s, -1)
        Ppar = (P @ V) @ V.T                                    # keep only the ANM span
        T = t.reshape(e - s, -1)
        sse_raw += float(((T - P) ** 2).sum())
        sse_prj += float(((T - Ppar) ** 2).sum())
        e_par += float((Ppar ** 2).sum()); e_perp += float(((P - Ppar) ** 2).sum())
    sst = d["sst"] + 1e-12
    return (1 - sse_raw / sst, 1 - sse_prj / sst, e_perp / (e_par + e_perp + 1e-12))


if __name__ == "__main__":
    if not os.path.exists(CKPT):
        raise SystemExit(f"  no checkpoint at {CKPT}")
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho = [p for p in man["heldout"] if p in have]
    ho.sort(key=lambda p: store.meta[have[p]]["atoms"])
    print(f"[100a] spurious or missing? {len(ho)} held-out systems, ANM K={DM} cutoff={CUTOFF}",
          flush=True)
    mdl, rows, failed, t0 = None, {}, 0, time.time()
    for i, pdb in enumerate(ho, 1):
        try:
            d = sysdata(store, have[pdb])
            if d is None: failed += 1; continue
            V, _, _ = anm_modes(d["ref"], DM, CUTOFF)
            if V is None: failed += 1; continue
            if mdl is None:
                mdl = Codec(d["stat"].shape[1], 1, DM).to(dev)
                mdl.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False))
                mdl.eval()
            raw, prj, frac = fve_pair(mdl, d, V)
            rows[pdb] = dict(N=d["N"], fve_raw=raw, fve_proj=prj, delta=prj - raw,
                             spurious_frac=frac)
            print(f"  [{i}/{len(ho)}] {pdb:10s} N={d['N']:>6}  raw {raw:+.4f} -> projected {prj:+.4f}"
                  f"  ({prj-raw:+.4f})   {100*frac:4.1f}% of output energy was ANM-orthogonal"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(ho)}] {pdb}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(ho), complete=(i == len(ho)), n_failed=failed,
                          dm=DM, ckpt=os.path.basename(CKPT))
    if not rows:
        raise SystemExit("\n  NO system scored. Reported as absent, not as a null.")
    R = np.array([r["fve_raw"] for r in rows.values()])
    P = np.array([r["fve_proj"] for r in rows.values()])
    Dl = P - R
    F = np.array([r["spurious_frac"] for r in rows.values()])
    print(f"\n=== 100a: TOTAL FVE, raw codec vs ANM-projected codec ({len(rows)} systems) ===")
    print(f"  {'':>14}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}")
    for lab, v in (("raw", R), ("projected", P), ("delta", Dl)):
        print(f"  {lab:>14}{np.median(v):>10.4f}{v.mean():>10.4f}{v.min():>10.4f}{v.max():>10.4f}")
    print(f"\n  projection HELPS on {100*(Dl > 0).mean():.1f}% of systems, hurts on "
          f"{100*(Dl < 0).mean():.1f}%")
    print(f"  ANM-orthogonal share of the codec's OUTPUT energy: median {100*np.median(F):.1f}%, "
          f"range {100*F.min():.1f}-{100*F.max():.1f}%")
    if np.median(Dl) > 0.005:
        print(f"  -> SPURIOUS OUTPUT IS PART OF THE LOSS. Discarding the ANM-orthogonal component of\n"
              f"     the codec's own prediction raises total FVE by {np.median(Dl):+.4f} at ZERO\n"
              f"     training cost, using only the reference structure.")
    else:
        print(f"  -> THE SPURIOUS ENERGY IS INCIDENTAL. Removing it does not raise total FVE, so the\n"
              f"     loss is about what the codec MISSES, not what it adds. This reading closes.")
