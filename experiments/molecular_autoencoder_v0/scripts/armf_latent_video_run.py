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

ANM specifically, not tICA: it is zero-shot from the reference structure, so the basis costs nothing
on a protein never seen, and it is the peer the dynamics line has been scored against all along.
Coefficients are whitened by TRAIN statistics, so mode k means "the k-th slowest ANM mode" in
comparable units across proteins -- which is what makes a single cross-system generator meaningful
at all.

THE QUESTION THIS ANSWERS, which one-step propagation cannot: does modelling a WHOLE SEGMENT jointly
buy anything over generating one step at a time? The one-step DDPM propagator is on record as
reaching cross-mode coupling OU structurally cannot, at the cost of agreement elsewhere and 24-41%
divergence. Joint generation is scored against the SAME acceptance test so the two are comparable.

THE ACCEPTANCE TEST IS IMPORTED, NOT REBUILT. armf_propagator.stats_of / band / consistent /
ref_windows carry 77a's matched-length matched-spacing discipline and 81a's power check. A second
implementation of these would be the tenth "one name, two things" in a project that has caught nine.

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
from armf_propagator import stats_of, band, consistent, METRICS
import armf_atlas_dm as D
import armf_io

WR = D.WR
K = int(os.environ.get("LV_K", "64"))          # ANM modes = R x Dlat
RGRP = int(os.environ.get("LV_R", "8"))
DLAT = K // RGRP
T = int(os.environ.get("LV_T", "32"))          # frames per segment
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


def segments(C, T, k, rng):
    """k random contiguous segments of length T, as (k, T, R, Dlat)."""
    if len(C) <= T: return None
    s = rng.integers(0, len(C) - T, size=k)
    return np.stack([C[i:i + T] for i in s]).reshape(k, T, RGRP, DLAT)


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
            refw = segments(sysd["C_ho"], T, KEVAL, rng)
            if refw is None: continue
            pool = sysd["C_ho"]
            rs = [stats_of(w.reshape(T, K), np.argsort(pool[:len(pool)//2].std(0))[::-1][:2],
                           np.median(pool[:, np.argsort(pool.std(0))[::-1][:2]], 0), pool)
                  for w in refw]
            top2 = np.argsort(pool.std(0))[::-1][:2]
            thr = np.median(pool[:, top2], 0)
            rs = [stats_of(w.reshape(T, K), top2, thr, pool) for w in refw]
            with torch.no_grad():
                gen = mdl.sample((KEVAL, T, RGRP, DLAT), dev, steps=SAMPLE_STEPS).cpu().numpy()
            ou = ou_segments(sysd["C_tr"], sysd["C_ho"], T, KEVAL, rng)
            arms = {}
            for name, S in (("OU", ou), ("JOINT-diffusion", gen)):
                ss = [stats_of(w.reshape(T, K), top2, thr, pool) for w in S]
                inside = {k: bool(consistent(ss, rs, k)) for k in METRICS}
                arms[name] = dict(agree=int(sum(inside.values())), inside=inside,
                                  med={k: float(band(ss, k)[0]) for k in METRICS})
            bm, bs = bond_stats(sysd, gen); rm, rs_ = bond_stats(sysd, refw)
            arms["geometry"] = dict(gen_bond_mean=bm, gen_bond_sd=bs,
                                    ref_bond_mean=rm, ref_bond_sd=rs_)
            rows[sysd["pdb"]] = dict(N=sysd["N"], arms=arms,
                                     ref_med={k: float(band(rs, k)[0]) for k in METRICS})
            j = arms["JOINT-diffusion"]; o = arms["OU"]
            print(f"  [{i}/{len(HO)}] {sysd['pdb']:10s} N={sysd['N']:>6}  "
                  f"JOINT {j['agree']}/7  OU {o['agree']}/7   "
                  f"xcorr {'IN' if j['inside']['xcorr'] else '--'}/"
                  f"{'IN' if o['inside']['xcorr'] else '--'}  "
                  f"bond {bm:.2f}A vs ref {rm:.2f}A  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(HO)}] {sysd['pdb']}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(HO), complete=(i == len(HO)), n_failed=0,
                          K=K, T=T, steps=STEPS, train_log=log)

    if not rows:
        raise SystemExit("\n  NO system scored.")
    print(f"\n=== RESULT ({len(rows)} held-out systems) ===")
    print(f"  {'arm':>18}{'median agree':>14}" + "".join(f"{k:>9}" for k in METRICS))
    for arm in ("OU", "JOINT-diffusion"):
        ag = np.array([r["arms"][arm]["agree"] for r in rows.values()])
        cells = "".join(f"{100*np.mean([r['arms'][arm]['inside'][k] for r in rows.values()]):>8.0f}%"
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
