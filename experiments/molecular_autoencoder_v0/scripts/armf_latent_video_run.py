#!/usr/bin/env python3
"""THE END-TO-END GENERATIVE CHAIN, on real molecular trajectories, for the first time.

    real trajectory -> fixed collective latent sequence -> JOINT SEGMENT DIFFUSION
                    -> generated latent trajectory -> atom trajectory

WHY THIS, AND WHY NOW. An external read of this project landed a prioritisation criticism that is
correct: enormous effort has gone into reconstruction codecs and statistical instruments, while
`molae/latent_video.py` -- the whole-segment generative module, the analogue of a video model -- has
only ever been unit-tested. The "diffusion results" on record are three disconnected things: one
diffuses PCA latents, one propagates ANM coordinates one step at a time, and this module has never
seen real data. So the chain above has never run.

WHY A FIXED PHYSICS BASIS RATHER THAN THE LEARNED CODEC, and this is my own result arguing against
my own prior work. 102c decomposed FVE across the ANM boundary within one codec: the codec scores
0.2997 INSIDE ANM's span against ANM's 1.0000, it is behind on 100% of systems, and giving it a
PERFECT ANM-orthogonal residual still loses on 86%. The learned basis is not the differentiator. The
differentiator this project can still own is the LEARNED GENERATOR, so the basis is handed to physics
and the generator is what gets trained.

CORPUS: ATLAS, not mdCATH, and the reason is measured rather than preferred. 104's feasibility check
chose mdCATH under a T=16 budget. 105 requires T >= MIN_H = 200 for the statistics to mean anything,
and at T=256 the disjoint-segment budget inverts:

    mdCATH  T=16 -> 4,340 segments      T=256 ->   140   (1 window per 500-frame replica)
    ATLAS   T=16 -> 28,080 segments     T=256 -> 1,620

mdCATH's 500-frame replicas yield ONE disjoint window each at T=256, so the corpus that was abundant
at T=16 is exhausted at the length the estimator requires.

ANM specifically, not tICA: it is zero-shot from the reference structure, so the basis costs nothing
on a protein never seen, and it is the peer the dynamics line has been scored against all along.
Coefficients are whitened by TRAIN statistics, so mode k means "the k-th slowest ANM mode" in
comparable units across proteins -- which is what makes a single cross-system generator meaningful
at all.

THE QUESTION THIS ANSWERS, which one-step propagation cannot: does modelling a WHOLE SEGMENT jointly
buy anything over generating one step at a time? The one-step DDPM propagator is on record as
reaching cross-mode coupling OU structurally cannot, at the cost of agreement elsewhere and 24-41%
divergence. Joint generation is scored against the SAME acceptance test so the two are comparable.

THE SCORING IS IMPORTED, NOT REBUILT: stats_of, band, consistent, METRICS and MIN_H come from
armf_propagator and carry 77a's matched-length discipline. `ref_windows` is NOT imported -- the
header used to claim it was, while `segments()` below is a local reimplementation for a different
data layout. Corrected rather than quietly aligned, because a header that overstates reuse is how a
second implementation hides.

PRE-REGISTERED READINGS, before any number exists:
  JOINT INSIDE THE BAND WHERE ONE-STEP IS NOT  joint trajectory generation buys something real, and
                                               the generative direction has its first positive result
  JOINT AND ONE-STEP INDISTINGUISHABLE         segment modelling is not the missing piece; the
                                               bottleneck is elsewhere and the next experiment is not
                                               a bigger generator
  JOINT WORSE                                  the segment model is harder to fit at this data scale,
                                               which is a statement about the corpus, not the idea

101d's constraint is carried: state IDENTITY and assignment are scorable, OCCUPANCY and TRANSITION
RATES are determined to only ~61% at the median, so `trans` is reported with that bound attached and
never as a rate claim.
"""
import sys, os, json, time, math
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE + "/.."
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from molae.latent_video import LatentVideoConfig, LatentVideoDiffusion
from armf_atlas_data import AtlasStore, sysdata
from armf_anm import modes as anm_modes
from armf_propagator import stats_of, band, consistent, METRICS, MIN_H
import armf_atlas_dm as D
import armf_io

WR = D.WR
K = int(os.environ.get("LV_K", "64"))          # ANM modes = R x Dlat
# INBOX 104b: THE TOKEN AXIS IS A STATED CHOICE, NOT A SILENT ONE. The mode axis becomes the
# "spatial" axis: (B, T, R=64, latent_dim=1), one token per ANM mode, which is what 104 specifies.
# My first run used R=8 x D=8 -- eight consecutive modes bundled per token -- without saying so, and
# that is a different inductive bias: factorised attention over MODES is not attention over groups of
# modes, and if one underperforms that is a finding about the token axis rather than about joint
# generation. LV_R=8 reproduces the bundled variant as a one-variable ablation.
RGRP = int(os.environ.get("LV_R", "64"))
DLAT = K // RGRP
# INBOX 105. T=32 WAS UNREADABLE AND THE GUARD DID NOT COME ACROSS WITH THE ESTIMATOR.
# armf_propagator:290 sets MIN_H = 200, "below this a series cannot carry these statistics at all",
# and lines 415/424 REFUSE to score below it, writing UNEVALUABLE. This file imported
# stats_of/band/consistent, set T=32, and scored. The estimator came across; the guard that says
# when it is meaningless did not.
#
# Why it matters here specifically: OU is independent per mode BY CONSTRUCTION so its true xcorr and
# amp are exactly 0, but the ESTIMATOR floor is ~0.8/sqrt(T) and I reproduced it --
#   T=32  a1=0.90 -> xcorr 0.3023  amp 0.2265      T=256 a1=0.90 -> 0.1498 / 0.0978
#   T=32  a1=0.99 -> xcorr 0.4002  amp 0.3407      T=512 a1=0.90 -> 0.1073 / 0.0719
# Bias still cancels (77a holds); what dies is POWER. consistent() is interval overlap, so below the
# floor EVERY arm passes including OU, and a good JOINT result would be unreadable too.
#
# The floor depends on T and on the modes' own a1, NOT on K -- so trimming modes does not help. My
# sweep adds one the item did not: at T=256 with a1=0.99 the floor is still 0.3177. So raising T is
# necessary and not sufficient, and the power check below is BLOCKING rather than advisory.
T = int(os.environ.get("LV_T", "256"))        # frames per segment; ATLAS gives 2245 starts at 256
NTRAIN = int(os.environ.get("LV_NTRAIN", "60"))
NEVAL = int(os.environ.get("LV_NEVAL", "24"))
STEPS = int(os.environ.get("LV_STEPS", "20000"))
BATCH = int(os.environ.get("LV_BATCH", "32"))
LR = float(os.environ.get("LV_LR", "3e-4"))
CUTOFF = float(os.environ.get("LV_CUTOFF", "5.0"))
KEVAL = int(os.environ.get("LV_KEVAL", "32"))  # segments per side, per system
SAMPLE_STEPS = int(os.environ.get("LV_SAMPLE_STEPS", "50"))
RES = os.environ.get("LV_RES", f"{WR}/latent_video.json")
CKPT = f"{WR}/latent_video_ckpt.pt"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def coeffs(d, rep, V, mu, sd):
    """Whitened ANM coefficients for one replica: (F, K). The basis is zero-shot from the reference
    structure and the whitening is TRAIN-only, so nothing from the scored replica enters either."""
    a = np.load(d["path"], mmap_mode="r")
    X = np.asarray(a[rep]).astype(np.float64).reshape(d["F"], -1) - mu
    return ((X @ V) / sd).astype(np.float32)


def prepare(store, have, ids, n):
    """One pass: ANM basis per system, whitened coefficients for train and held-out replicas."""
    out = []
    for p in ids[:n]:
        try:
            d = sysdata(store, have[p])
            if d is None: continue
            V, _, _ = anm_modes(d["ref"], K, CUTOFF)
            if V is None: continue
            a = np.load(d["path"], mmap_mode="r")
            tr = np.concatenate([np.asarray(a[r]).astype(np.float64).reshape(d["F"], -1)
                                 for r in (0, 1)]) - d["mu"]
            C_tr = tr @ V
            sd = C_tr.std(0) + 1e-8
            out.append(dict(pdb=p, N=d["N"], F=d["F"], V=V, mu=d["mu"], sd=sd, scale=d["scale"],
                            ref=d["ref"], C_tr=(C_tr / sd).astype(np.float32),
                            C_ho=coeffs(d, 2, V, d["mu"], sd)))
        except Exception as e:
            print(f"    {p}: prepare FAIL {type(e).__name__}: {e}", flush=True)
    return out


def segments(C, T, k, rng, disjoint=False):
    """k contiguous segments of length T, as (k, T, R, Dlat).

    INBOX 105: with `disjoint`, starts are drawn WITHOUT overlap. Overlapping reference windows share
    frames, so band()'s range is narrower than independent draws would give -- which moves
    consistent() STRICTER, the opposite direction to the power loss above. Two biases in opposite
    directions do not cancel to anything known unless both are measured, so the reference side is
    drawn disjoint and the realised count is reported rather than assumed.
    """
    if len(C) <= T: return None
    if disjoint:
        n = (len(C) - 1) // T
        if n < 2: return None
        st = rng.permutation(n)[:min(k, n)] * T
    else:
        st = rng.integers(0, len(C) - T, size=k)
    return np.stack([C[i:i + T] for i in st]).reshape(len(st), T, RGRP, DLAT)


def ou_segments(C_tr, C0, T, k, rng):
    """OU at lag 1 in the whitened ANM basis, fitted on TRAIN. The physics null: independent per
    mode by construction, so xcorr and amp are ~0 for it and it is a NEGATIVE CONTROL on those two
    exactly as 81a established."""
    a1 = (C_tr[:-1] * C_tr[1:]).mean(0) / ((C_tr[:-1] ** 2).mean(0) + 1e-9)
    a1 = np.clip(a1, -0.99, 0.99)
    s = np.sqrt(np.clip(1 - a1 ** 2, 1e-6, None))
    out = []
    for j in range(k):
        x = C0[rng.integers(0, len(C0))].copy(); roll = [x.copy()]
        for _ in range(T - 1):
            x = a1 * x + s * rng.standard_normal(len(a1)).astype(np.float32); roll.append(x.copy())
        out.append(np.array(roll))
    return np.stack(out).reshape(k, T, RGRP, DLAT)


def bond_stats(sysd, C_seg):
    """ATOM-LEVEL GEOMETRY, which no latent-space metric can express. Coefficients are decoded to
    coordinates through the same ANM basis and consecutive-CA distances are measured -- a generated
    trajectory can match every latent statistic and still produce broken chemistry."""
    X = (C_seg.reshape(-1, K) * sysd["sd"]) @ sysd["V"].T + sysd["mu"]
    P = X.reshape(len(X), -1, 3)
    step = max(1, P.shape[1] // 200)
    Q = P[:, ::step]
    dd = np.linalg.norm(Q[:, 1:] - Q[:, :-1], axis=-1)
    return float(dd.mean()), float(dd.std())


if __name__ == "__main__":
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    tr_ids = [p for p in man["train_ordered"] if p in have]
    ho_ids = [p for p in man["heldout"] if p in have]
    print(f"[latent-video] END-TO-END on real trajectories. device={dev}", flush=True)
    print(f"  ANM K={K} -> segments ({T}, {RGRP}, {DLAT}); train {NTRAIN} systems, "
          f"eval {NEVAL} held-out", flush=True)
    TR = prepare(store, have, tr_ids, NTRAIN)
    HO = prepare(store, have, ho_ids, NEVAL)
    print(f"  prepared {len(TR)} train / {len(HO)} held-out systems", flush=True)
    if not TR or not HO:
        raise SystemExit("  nothing prepared")

    cfg = LatentVideoConfig(latent_dim=DLAT, d_model=256, depth=6, max_frames_hint=T)
    mdl = LatentVideoDiffusion(cfg).to(dev)
    print(f"  SegmentDiT parameters: {mdl.num_parameters():,}", flush=True)
    opt = torch.optim.Adam(mdl.parameters(), LR)
    rng = np.random.default_rng(0)
    t0, log = time.time(), []
    for st in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, st / 1000)
        sysd = TR[rng.integers(len(TR))]
        seg = segments(sysd["C_tr"], T, BATCH, rng)
        if seg is None: continue
        z1 = torch.tensor(seg, device=dev)
        opt.zero_grad()
        loss, _ = mdl.training_loss(z1)
        loss.backward(); opt.step()
        if st % 2000 == 0:
            log.append(dict(step=st, loss=float(loss.detach()), secs=time.time() - t0))
            print(f"    step {st:>6}: flow MSE {float(loss.detach()):.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save(mdl.state_dict(), CKPT)

    print(f"\n=== ACCEPTANCE TEST: joint segment diffusion vs OU vs the reference band ===",
          flush=True)
    print(f"  imported from armf_propagator: same estimator on every arm, matched segment length "
          f"{T} (77a)", flush=True)
    mdl.eval()
    rows = {}
    for i, sysd in enumerate(HO, 1):
        try:
            if T < MIN_H:
                print(f"  [{i}/{len(HO)}] {sysd['pdb']}: UNEVALUABLE -- T={T} < MIN_H={MIN_H}; "
                      f"the estimator cannot carry these statistics at this length", flush=True)
                continue
            refw = segments(sysd["C_ho"], T, KEVAL, rng, disjoint=True)
            if refw is None: continue
            # MATCH n ON BOTH SIDES. 2,501 held-out frames give only (2501-1)//256 = 9 DISJOINT
            # windows, not the 32 requested, and `consistent()` compares two SPREADS -- an interval
            # from 9 draws against one from 32 is not the same estimator on both sides, which is the
            # asymmetry 77a exists to remove. The generated side is therefore drawn at the reference
            # count, and that count is reported rather than assumed.
            keval = len(refw)
            pool = sysd["C_ho"]
            # ONE computation of the reference statistics. The first of the two was dead -- it built
            # top2/thr inline, then both were rebuilt and rs recomputed, so the first result was
            # discarded unused. Removed rather than left as a second definition of the same thing.
            top2 = np.argsort(pool.std(0))[::-1][:2]
            thr = np.median(pool[:, top2], 0)
            rs = [stats_of(w.reshape(T, K), top2, thr, pool) for w in refw]
            with torch.no_grad():
                gen = mdl.sample((keval, T, RGRP, DLAT), dev, steps=SAMPLE_STEPS).cpu().numpy()
            ou = ou_segments(sysd["C_tr"], sysd["C_ho"], T, keval, rng)
            arms = {}
            for name, S in (("OU", ou), ("JOINT-diffusion", gen)):
                ss = [stats_of(w.reshape(T, K), top2, thr, pool) for w in S]
                inside = {k: bool(consistent(ss, rs, k)) for k in METRICS}
                arms[name] = dict(agree=int(sum(inside.values())), inside=inside,
                                  med={k: float(band(ss, k)[0]) for k in METRICS})
            # INBOX 81a/105: THE POWER CHECK IS BLOCKING. OU is wrong on xcorr and amp BY
            # CONSTRUCTION, so if it lands INSIDE the band on both, the band cannot reject a model
            # that is wrong by construction and NOTHING about JOINT may be read from this system --
            # including a good result.
            o_in = arms["OU"]["inside"]
            powered = not (o_in["xcorr"] and o_in["amp"])
            arms["POWER"] = dict(ou_inside_xcorr=o_in["xcorr"], ou_inside_amp=o_in["amp"],
                                 powered=powered, n_refw=int(len(refw)),
                                 band_xcorr=float(band(rs, "xcorr")[2] - band(rs, "xcorr")[1]))
            bm, bs = bond_stats(sysd, gen); rm, rs_ = bond_stats(sysd, refw)
            arms["geometry"] = dict(gen_bond_mean=bm, gen_bond_sd=bs,
                                    ref_bond_mean=rm, ref_bond_sd=rs_)
            rows[sysd["pdb"]] = dict(N=sysd["N"], arms=arms,
                                     ref_med={k: float(band(rs, k)[0]) for k in METRICS})
            j = arms["JOINT-diffusion"]; o = arms["OU"]
            print(f"  [{i}/{len(HO)}] {sysd['pdb']:10s} N={sysd['N']:>6} "
                  f"{'POWERED' if powered else 'NO POWER'}  "
                  f"JOINT {j['agree']}/7  OU {o['agree']}/7   "
                  f"xcorr {'IN' if j['inside']['xcorr'] else '--'}/"
                  f"{'IN' if o['inside']['xcorr'] else '--'}  "
                  f"bond {bm:.2f}A vs ref {rm:.2f}A  n={keval}  ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"  [{i}/{len(HO)}] {sysd['pdb']}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(HO), complete=(i == len(HO)), n_failed=0,
                          K=K, T=T, steps=STEPS, train_log=log)

    if not rows:
        raise SystemExit("\n  NO system scored.")
    print(f"\n=== RESULT ({len(rows)} held-out systems) ===")
    print(f"  {'arm':>18}{'median agree':>14}" + "".join(f"{k:>9}" for k in METRICS))
    pw = {k: v for k, v in rows.items() if v["arms"]["POWER"]["powered"]}
    print(f"  POWERED systems: {len(pw)}/{len(rows)}  "
          f"(OU outside the band on xcorr AND amp -- 81a)")
    if not pw:
        print(f"  -> NO SYSTEM IS POWERED. Nothing about JOINT is readable from this run, including\n"
              f"     a good result. The honest output is the band width, not a table.")
    for arm in ("OU", "JOINT-diffusion"):
        src = pw or {}
        if not src: break
        ag = np.array([r["arms"][arm]["agree"] for r in src.values()])
        cells = "".join(f"{100*np.mean([r['arms'][arm]['inside'][k] for r in src.values()]):>8.0f}%"
                        for k in METRICS)
        print(f"  {arm:>18}{np.median(ag):>14.1f}{cells}")
    print(f"  (percentages are the share of systems whose generated segments land INSIDE the band "
          f"real segments make)")
    bg = np.array([r["arms"]["geometry"]["gen_bond_mean"] for r in rows.values()])
    br = np.array([r["arms"]["geometry"]["ref_bond_mean"] for r in rows.values()])
    print(f"\n  ATOM-LEVEL GEOMETRY: generated consecutive-CA spacing {np.median(bg):.2f} A vs "
          f"reference {np.median(br):.2f} A  ({100*abs(np.median(bg)-np.median(br))/np.median(br):.1f}% off)")
    jo = np.array([r["arms"]["JOINT-diffusion"]["agree"] for r in rows.values()])
    ou_ = np.array([r["arms"]["OU"]["agree"] for r in rows.values()])
    print(f"\n  joint beats OU on {100*(jo>ou_).mean():.0f}% of systems, ties {100*(jo==ou_).mean():.0f}%")
    print(f"  NOTE (101d): `trans` is a transition RATE and this corpus determines occupancy to only\n"
          f"  ~61% at the median, so it is reported as band-membership and never as a rate claim.")
