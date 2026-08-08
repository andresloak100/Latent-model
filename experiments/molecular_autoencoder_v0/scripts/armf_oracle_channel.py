#!/usr/bin/env python3
"""INBOX 53b/54d/54e/59e: the ORACLE upper bound on the sparse event channel.

Every negative result in the ATLAS line measures x_i(t) = f(S_i, g_t, e_t) with the sparse event
channel e_t ABSENT. 41c has now closed the global-latent-alone path measurably: no win at any rung,
95% upper bound on the win rate 2.4% (rule of three, 0/123), and the only improvement from 6x the
data was repairing the large-system tail from harmful to useless. The channel is the one component
of section 7 never built.

WHY AN ORACLE. Give the decoder the EXACT displacements of the top-K atoms by residual magnitude. A
learned channel cannot beat one told exactly which atoms matter and how they move, so this is a
STRICT UPPER BOUND -- decisive in the negative at zero training cost, and bounded in what a positive
may claim:
    oracle fails to close the gap -> the branch is DEAD; no learned channel does better
    oracle closes the gap         -> the branch stays alive and NOTHING MORE

FALSIFIER (53b, corrected by 54d): at K=74 -- budget-matched to DM=256 once the index cost
log2(N)/32 floats per atom is charged, which K=85 did not -- if the oracle recovers less than 25% of
the codec-to-ANM gap, the channel is refused as a path to a peer win and the item closes.

54e: report K100, the K at which it closes 100%. Closing 25% or 50% still loses to ANM, so neither is
a peer win; the decision-relevant number is how sparse the channel must be NOT TO LOSE. Also sweep
K=ALL as a Family B ceiling check -- a full oracle must recover essentially all the residual by
construction, and if it does not, every smaller K is uninterpretable.

59e: report per QUARTILE, never pooled. 41c established the residual is quartile-dependent.

SCOPE (53b's Family D limit): this asks whether the architecture can REPRESENT the residual at a
sparse budget. It says nothing about whether events can be GENERATED at sampling time.
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ho_frames
from armf_anm import modes as anm_modes
import armf_atlas_dm as D
import armf_stamp as STAMP
from armf_modal_decoder import ModalCodec
from armf_tied_peer import ckpt_path, anm_fve

WR = D.WR
DM, SEED, CUTOFF = 256, 0, 5.0
ARM_LR = 3e-5
NTRAIN = int(os.environ.get("PEER_NTRAIN", "300"))
RES = f"{WR}/oracle_channel_n{NTRAIN}.json"
KS = [0, 1, 4, 16, 64, 74, 256, 1024, -1]
dev = D.dev


@torch.no_grad()
def oracle_fve(mdl, d, ks, chunk=8):
    """SSE per K on the SAME frames and SAME sst fve_model uses -- a modification of that function,
    not a second implementation. The correction is exact substitution: for the top-K atoms by
    per-frame residual magnitude the reconstruction is replaced by the truth, which is the most any
    sparse channel of budget K could do."""
    st = torch.tensor(d["stat"], device=dev)
    sse = {k: 0.0 for k in ks}
    for s in range(0, d["F"], chunk):
        e = min(s + chunk, d["F"])
        t = ho_frames(d, s, e)
        dw = torch.tensor((t / d["scale"]).astype(np.float32), device=dev)
        p = mdl(st.unsqueeze(0).expand(e - s, -1, -1), dw).cpu().numpy().astype(np.float64) * d["scale"]
        base = ((t - p) ** 2).sum(-1)
        order = np.argsort(-base, axis=1)
        tot = base.sum(1)
        for k in ks:
            if k == 0:
                sse[0] += float(tot.sum()); continue
            kk = d["N"] if k < 0 else min(k, d["N"])
            removed = np.take_along_axis(base, order[:, :kk], axis=1).sum(1)
            sse[k] += float((tot - removed).sum())
    return {k: 1.0 - sse[k] / (d["sst"] + 1e-12) for k in ks}


if __name__ == "__main__":
    print(f"[oracle] sparse event channel ORACLE. arm=tied n_train={NTRAIN} DM={DM} cutoff={CUTOFF}",
          flush=True)
    print(f"[oracle] K sweep {KS} (-1 = all atoms); falsifier at K=74 (54d budget-matched)", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    HO = [x for x in (sysdata(store, have[p]) for p in man["heldout"] if p in have) if x is not None]
    HO.sort(key=lambda z: z["N"])
    print(f"  {len(HO)} held-out systems, N {HO[0]['N']}-{HO[-1]['N']}", flush=True)

    cp = ckpt_path("tied", DM, ARM_LR, SEED, NTRAIN)
    print(f"  loading tied arm trained at n_train={NTRAIN}: {os.path.basename(cp)}", flush=True)
    FS = HO[0]["stat"].shape[1]
    mdl = ModalCodec(FS, 1, DM, tie_encoder=True).to(dev)
    mdl.load_state_dict(torch.load(cp, map_location=dev, weights_only=False)); mdl.eval()

    # n_train IS in the stamp: 41c's near-miss was an n300 run reusing n50 results because it was not.
    ST = STAMP.stamp(dict(dm=DM, lr=ARM_LR, seed=SEED, cutoff=CUTOFF, arm="tied",
                          n_train=NTRAIN, ks=str(KS)), ModalCodec, D.fve_model)
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    STAMP.report(list(res.values()), ST, "systems")
    res = {k: v for k, v in res.items() if STAMP.same_stamp(v, ST)}

    t0 = time.time()
    for i, d in enumerate(HO):
        if d["pdb"] in res: continue
        try:
            o = oracle_fve(mdl, d, KS)
            V, _, _ = anm_modes(d["ref"], DM, CUTOFF)
            a = anm_fve(d, V) if V is not None else float("nan")
            res[d["pdb"]] = dict(N=d["N"], anm=a, oracle={str(k): o[k] for k in KS}, stamp=ST)
            json.dump(res, open(RES, "w"))
            print(f"    [{i+1}/{len(HO)}] {d['pdb']} N={d['N']:>6}  K=0 {o[0]:+.4f}  "
                  f"K=74 {o[74]:+.4f}  K=all {o[-1]:+.4f}  ANM {a:+.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"    {d['pdb']} N={d['N']}: FAIL {type(e).__name__}: {e}", flush=True)

    if not res: raise SystemExit
    N = np.array([v["N"] for v in res.values()])
    q = np.quantile(N, [.25, .5, .75]); lab = np.digitize(N, q)
    anm = np.array([v["anm"] for v in res.values()])
    print(f"\n=== ORACLE SPARSE CHANNEL: gap closure per quartile (59e: never pooled) ===", flush=True)
    full = np.array([v["oracle"]["-1"] for v in res.values()])
    ok = np.median(full) > 0.999
    print(f"  FAMILY B CEILING CHECK: K=all median FVE {np.median(full):+.6f} (must be ~1.0 by "
          f"construction) -> {'HARNESS OK' if ok else 'HARNESS BROKEN -- every smaller K UNREADABLE'}",
          flush=True)
    print("  " + f"{'quartile':>9}{'n':>4}"
          + "".join(f"{('K='+('all' if k < 0 else str(k))):>11}" for k in KS), flush=True)
    for qi in range(4):
        m = lab == qi
        row = f"  {'Q'+str(qi+1):>9}{m.sum():>4}"
        for k in KS:
            row += f"{np.median(np.array([r['oracle'][str(k)] for r in res.values()])[m]):>+11.4f}"
        print(row, flush=True)
        base = np.median(np.array([r["oracle"]["0"] for r in res.values()])[m])
        k74 = np.median(np.array([r["oracle"]["74"] for r in res.values()])[m])
        gap = np.median(anm[m]) - base
        frac = (k74 - base) / gap if gap > 0 else float("nan")
        print(f"            ANM median {np.median(anm[m]):+.4f}; gap {gap:+.4f}; K=74 closes "
              f"{100*frac:.1f}% -> {'REFUSED (<25%)' if frac < 0.25 else 'not refused'}", flush=True)
        k100 = None
        for k in [k_ for k_ in KS if k_ > 0]:
            if np.median(np.array([r["oracle"][str(k)] for r in res.values()])[m]) >= np.median(anm[m]):
                k100 = k; break
        print(f"            K100 (stops losing to ANM): "
              + (f"{k100} atoms" if k100 else ">1024 atoms -- NOT SPARSE, the cost argument dies"),
              flush=True)
