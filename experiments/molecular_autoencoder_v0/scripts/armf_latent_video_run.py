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
from armf_propagator import stats_of, band, consistent, METRICS, MIN_H, offdiag
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
# OOM at batch 32. The temporal attention is B*H*R*T^2 = 4.29 GB per LAYER at T=256, R=64, and
# R=64 makes that term 64x what R=8 would -- the cost 106 named, paid in memory rather than time.
# Batch 8 is ~8.1 GB across six layers, which fits with headroom on a 46 GB device. Chosen by
# arithmetic rather than by halving until it ran.
BATCH = int(os.environ.get("LV_BATCH", "8"))
ACCUM = int(os.environ.get("LV_ACCUM", "4"))   # 107b: effective batch = BATCH * ACCUM = 32
# INBOX 109. THE MATCHED-BUDGET ONE-STEP ARM LIVES IN THIS FILE, not a second one. The
# pre-registered question is joint vs one-step; 107b established samples-seen must match; and the
# existing propagator numbers were trained at an UNRECORDED budget, so they cannot answer it. Putting
# the one-step arm behind an env var rather than in its own script means both arms get the identical
# acceptance test, the identical top-8 M-gate, the identical feasibility-first output and the
# identical samples-seen accounting -- one instrument, which is what 77a is about.
ARM = os.environ.get("LV_ARM", "joint")        # joint | onestep
LR = float(os.environ.get("LV_LR", "3e-4"))
CUTOFF = float(os.environ.get("LV_CUTOFF", "5.0"))
KEVAL = int(os.environ.get("LV_KEVAL", "32"))  # segments per side, per system
SAMPLE_STEPS = int(os.environ.get("LV_SAMPLE_STEPS", "50"))
RES = os.environ.get("LV_RES", f"{WR}/latent_video_{os.environ.get('LV_ARM','joint')}.json")
# 116: THE CHECKPOINT NAME DID NOT ENCODE R, so the R=64 run was about to overwrite lv_r8's --
# the checkpoint behind the headline result. RES was overridden per-arm and CKPT was not, which
# is the same "one name, two things" in a filename. R is in the name now, and the rescued
# lv_r8 file is saved as latent_video_ckpt_joint_r8.pt.
CKPT = f"{WR}/latent_video_ckpt_{os.environ.get('LV_ARM','joint')}_r{RGRP}.pt"
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
            # INBOX 106b.2/106c. sd COMES FROM REPLICA 0 ALONE. It used to come from replicas 0+1 of
            # EVERY system including the held-out ones, so the frames that set the normalisation
            # overlapped the frames that were scored. Held-out is by SYSTEM, so the generator saw
            # none of these replicas -- but the whitening did, and that is a leak in the PIPELINE
            # claim even though the generator is clean.
            r0 = np.asarray(a[0]).astype(np.float64).reshape(d["F"], -1) - d["mu"]
            C0 = r0 @ V
            sd = C0.std(0) + 1e-8
            # The reference band is drawn from replicas 1 AND 2 -- 18 disjoint windows at T=256
            # instead of 9, with no overlap against the frames that set sd. Windows still lie inside
            # ONE replica, so ref_windows' seam rule holds and no window crosses a join.
            C_ref = np.stack([coeffs(d, 1, V, d["mu"], sd), coeffs(d, 2, V, d["mu"], sd)])
            lam = None
            try:
                _, lam = anm_modes(d["ref"], K, CUTOFF)[0], None
            except Exception:
                pass
            out.append(dict(pdb=p, N=d["N"], F=d["F"], V=V, mu=d["mu"], sd=sd, scale=d["scale"],
                            ref=d["ref"], C_tr=(C0 / sd).astype(np.float32), C_ref=C_ref,
                            C_ho=coeffs(d, 2, V, d["mu"], sd)))
        except Exception as e:
            print(f"    {p}: prepare FAIL {type(e).__name__}: {e}", flush=True)
    return out


TOPM = [int(x) for x in os.environ.get("LV_TOPM", "8,16").split(",")]
XMETRICS = list(METRICS) + [f"{m}_top{M}" for M in TOPM for m in ("xcorr", "amp")]


def stats_ext(X, top2, thr, pool):
    """INBOX 107a. THE POOLED STATISTIC CANNOT EXPRESS THE EFFECT THE EXPERIMENT EXISTS TO DETECT.

    `offdiag` is the MEAN |correlation| over all 64*63/2 = 2,016 mode pairs, while real collective
    coupling lives in the slowest handful. Reproduced at T=256, a1=0.876, K=18:

        scored on   independent band        coupled-8 @0.6 band      verdict
        all 64      [0.1275, 0.1431]        [0.1310, 0.1440]         OVERLAP -- undetected
        top 16      [0.1117, 0.1517]        [0.1505, 0.2316]         marginal (bands touch)
        top  8      [0.0923, 0.1478]        [0.2344, 0.4771]         DETECTED

    Making the eight slowest modes almost completely dependent -- stronger than any protein --
    moves the pooled statistic less than the band width. Family D on the METRIC, not the estimator,
    which is why the estimator recalibration came back clean. Not fixable by raising T or K: the
    dilution is in the aggregation.

    One refinement on 107a: top-16 is MARGINAL in my reproduction (0.1517 against 0.1505) where the
    item reported it detected, so top-8 carries the discrimination and top-16 is reported beside it
    rather than made the sole primary. Pooled-64 stays because the propagator's existing numbers
    are on it and dropping it would break comparability.

    The modes arrive slowest-first from anm_modes, so [:M] IS the M slowest.
    """
    d = dict(stats_of(X, top2, thr, pool))
    for M in TOPM:
        Y = X[:, :M]
        d[f"xcorr_top{M}"] = offdiag(np.corrcoef(Y.T))
        d[f"amp_top{M}"] = offdiag(np.corrcoef(np.abs(Y).T))
    return d


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


def ca_index(sysd):
    """CA atom indices from ATLAS's own published topology, by ATOM NAME.

    INBOX 112 item 4. The previous version took `step = P.shape[1] // 200` over the ALL-ATOM array
    and measured every N//200-th atom. Two defects, and the reference value was the tell: consecutive
    CA is 3.80 +/- 0.03 A and it was reporting 2.70 A, which is what an arbitrary every-k-th-atom
    spacing gives. `step` varies with N, so it was a DIFFERENT PHYSICAL DISTANCE on every system --
    Family F -- and a distance between atoms six apart in file order is conformation-dependent and
    cannot detect broken chemistry, which is the one job the guard exists for.
    """
    p = f"{WR}/atlas_topo/{sysd['pdb']}.pdb"
    if not os.path.exists(p):
        return None
    at = [l for l in open(p, errors="ignore").read().splitlines()
          if l.startswith(("ATOM", "HETATM"))]
    if len(at) != sysd["N"]:
        return None                      # refuse rather than reindex
    return np.array([i for i, l in enumerate(at) if l[12:16].strip() == "CA"])


def bond_stats(sysd, C_seg, ca=None):
    """Consecutive-CA distance. A generated trajectory can match every latent statistic and still
    produce broken chemistry, so this is the geometry guard -- and it only works on real CA pairs."""
    X = (C_seg.reshape(-1, K) * sysd["sd"]) @ sysd["V"].T + sysd["mu"]
    P = X.reshape(len(X), -1, 3)
    if ca is None or len(ca) < 2:
        return float("nan"), float("nan")
    Q = P[:, ca]
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
    if ARM == "onestep":
        # A one-step conditional propagator on the SAME coordinates: predict z_{t+1} from z_t, then
        # roll out T frames autoregressively. Width matched to the segment model's d_model so the
        # comparison is of the MODELLING CHOICE (whole segment vs one step) rather than of capacity.
        class OneStep(torch.nn.Module):
            def __init__(self, k, d=256):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(k, d), torch.nn.SiLU(),
                    torch.nn.Linear(d, d), torch.nn.SiLU(),
                    torch.nn.Linear(d, d), torch.nn.SiLU(),
                    torch.nn.Linear(d, 2 * k))
                self.k = k
            def forward(self, z):
                h = self.net(z)
                return h[..., :self.k], h[..., self.k:]        # mean, log-sigma of the step
            def loss(self, seg):
                x = seg.reshape(seg.shape[0], seg.shape[1], -1)
                mu, ls = self(x[:, :-1])
                ls = ls.clamp(-6, 3)
                return (((x[:, 1:] - mu) ** 2) / (2 * ls.exp() ** 2) + ls).mean()
            @torch.no_grad()
            def rollout(self, z0, T):
                out = [z0]
                for _ in range(T - 1):
                    mu, ls = self(out[-1])
                    out.append(mu + ls.clamp(-6, 3).exp() * torch.randn_like(mu))
                return torch.stack(out, 1)
        mdl = OneStep(K).to(dev)
    else:
        mdl = LatentVideoDiffusion(cfg).to(dev)
    print(f"  ARM={ARM}  parameters: {sum(p.numel() for p in mdl.parameters()):,}", flush=True)
    opt = torch.optim.Adam(mdl.parameters(), LR)
    rng = np.random.default_rng(0)
    t0, log, seen = time.time(), [], 0   # 108: `seen` was never initialised -- the run died on it
    for st in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, st / 1000)
        # INBOX 107b. GRADIENT ACCUMULATION restores the optimisation budget the memory fix cut.
        # Batch 32 -> 8 was correct arithmetic, but LV_STEPS stayed at 20,000, so the rerun would
        # have seen ONE QUARTER of the samples -- and pre-registered reading 3, "joint worse ->
        # harder to fit at this data scale", could not then be told apart from "trained on a quarter
        # as much". Family E introduced by a memory fix. ACCUM micro-steps at BATCH give effective
        # batch 32 at unchanged memory.
        opt.zero_grad()
        loss = None                      # INBOX 111: `loss` is bound INSIDE the accumulation loop
        for _ in range(ACCUM):           # and read at the logging line. Every `continue` in that
                                         # loop leaves it unbound -- dormant at T=256 because
                                         # segments() never returns None there, live the moment T
                                         # rises past a replica length. Same shape as `seen`.
            sysd = TR[rng.integers(len(TR))]
            seg = segments(sysd["C_tr"], T, BATCH, rng)
            if seg is None: continue
            z1 = torch.tensor(seg, device=dev)
            loss = mdl.loss(z1) if ARM == "onestep" else mdl.training_loss(z1)[0]
            (loss / ACCUM).backward()
            seen += BATCH * T
        opt.step()
        if st % 2000 == 0:
            if loss is None:
                print(f"    step {st:>6}: every micro-step was skipped -- no segment available at "
                      f"T={T}. Reported rather than logged as a loss.", flush=True)
                continue
            log.append(dict(step=st, loss=float(loss.detach()), samples_seen=seen,
                            eff_batch=BATCH * ACCUM, secs=time.time() - t0))
            print(f"    step {st:>6}: flow MSE {float(loss.detach()):.5f}  "
                  f"frames seen {seen:,}  ({time.time()-t0:.0f}s)", flush=True)
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
            # 106b.2: reference windows from BOTH held-out replicas, disjoint within each.
            parts = [segments(c, T, KEVAL, rng, disjoint=True) for c in sysd["C_ref"]]
            parts = [x for x in parts if x is not None]
            if not parts: continue
            refw = np.concatenate(parts)
            # MATCH n ON BOTH SIDES. 2,501 held-out frames give only (2501-1)//256 = 9 DISJOINT
            # windows, not the 32 requested, and `consistent()` compares two SPREADS -- an interval
            # from 9 draws against one from 32 is not the same estimator on both sides, which is the
            # asymmetry 77a exists to remove. The generated side is therefore drawn at the reference
            # count, and that count is reported rather than assumed.
            keval = len(refw)
            pool = np.concatenate(list(sysd["C_ref"]))
            # ONE computation of the reference statistics. The first of the two was dead -- it built
            # top2/thr inline, then both were rebuilt and rs recomputed, so the first result was
            # discarded unused. Removed rather than left as a second definition of the same thing.
            top2 = np.argsort(pool.std(0))[::-1][:2]
            thr = np.median(pool[:, top2], 0)
            rs = [stats_ext(w.reshape(T, K), top2, thr, pool) for w in refw]
            with torch.no_grad():
                if ARM == "onestep":
                    st0 = torch.tensor(np.stack([pool[rng.integers(len(pool) - 1)]
                                                 for _ in range(keval)]), device=dev,
                                       dtype=torch.float32)
                    gen = mdl.rollout(st0, T).cpu().numpy().reshape(keval, T, RGRP, DLAT)
                else:
                    gen = mdl.sample((keval, T, RGRP, DLAT), dev,
                                     steps=SAMPLE_STEPS).cpu().numpy()
            ou = ou_segments(sysd["C_tr"], sysd["C_ho"], T, keval, rng)
            arms = {}
            # INBOX 112 item 2. THE SHUFFLE ARM: reference windows with their ROWS PERMUTED.
            # Same frames, same marginals, same instantaneous mode covariance -- time destroyed. It
            # is the ceiling a purely STATIC model reaches, and this test has never had one: OU is
            # the null for coupling, the shuffle is the null for TIME. Expected 9/11, failing only
            # iat and trans, because 9 of 11 metrics are invariant to row permutation (verified:
            # corrcoef, |c*x|, moments and histograms are all order-blind).
            #   SHUFFLE fails iat/trans -> the test HAS dynamic discrimination and JOINT's margin
            #                              over it is the real temporal result
            #   SHUFFLE PASSES iat/trans -> the test has NO dynamic discrimination at T=256 and
            #                              nothing about temporal modelling is readable, whatever
            #                              JOINT scored
            shuf = np.stack([w.reshape(T, K)[rng.permutation(T)].reshape(T, RGRP, DLAT)
                             for w in refw])
            for name, S in (("OU", ou), ("SHUFFLE", shuf), (ARM.upper(), gen)):
                ss = [stats_ext(w.reshape(T, K), top2, thr, pool) for w in S]
                inside = {k: bool(consistent(ss, rs, k)) for k in XMETRICS}
                arms[name] = dict(agree=int(sum(inside.values())), inside=inside,
                                  med={k: float(band(ss, k)[0]) for k in XMETRICS})
            # INBOX 81a/105: THE POWER CHECK IS BLOCKING. OU is wrong on xcorr and amp BY
            # CONSTRUCTION, so if it lands INSIDE the band on both, the band cannot reject a model
            # that is wrong by construction and NOTHING about JOINT may be read from this system --
            # including a good result.
            o_in = arms["OU"]["inside"]
            # 107a: the power check now rides on the TOP-8 coupling, which is the statistic that can
            # actually see the effect. Pooled-64 is recorded but cannot gate anything, because OU
            # passing there is uninformative by construction.
            key_x, key_a = f"xcorr_top{TOPM[0]}", f"amp_top{TOPM[0]}"
            powered = not (o_in[key_x] and o_in[key_a])
            arms["POWER"] = dict(ou_inside_xcorr=o_in["xcorr"], ou_inside_amp=o_in["amp"],
                                 powered=powered, n_refw=int(len(refw)),
                                 band_xcorr=float(band(rs, "xcorr")[2] - band(rs, "xcorr")[1]))
            cai = ca_index(sysd)
            bm, bs = bond_stats(sysd, gen, cai)
            rm, rs_ = bond_stats(sysd, refw, cai)
            arms["geometry"] = dict(gen_bond_mean=bm, gen_bond_sd=bs,
                                    ref_bond_mean=rm, ref_bond_sd=rs_,
                                    n_ca=int(len(cai)) if cai is not None else 0)
            rows[sysd["pdb"]] = dict(N=sysd["N"], arms=arms,
                                     ref_med={k: float(band(rs, k)[0]) for k in XMETRICS})
            j = arms[ARM.upper()]; o = arms["OU"]
            print(f"  [{i}/{len(HO)}] {sysd['pdb']:10s} N={sysd['N']:>6} "
                  f"{'POWERED' if powered else 'NO POWER'}  "
                  f"JOINT {j['agree']}/{len(XMETRICS)}  OU {o['agree']}/{len(XMETRICS)}   "
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
    print(f"  {'arm':>18}{'median agree':>14}" + "".join(f"{k:>13}" for k in XMETRICS))
    pw = {k: v for k, v in rows.items() if v["arms"]["POWER"]["powered"]}
    # INBOX 112 item 3. OU's number on the POWERED subset is the SELECTION RESTATED: a system is
    # powered BECAUSE OU failed there. So OU is quoted over ALL systems, and only JOINT and SHUFFLE
    # are quoted on the powered subset, where the question is whether the test could discriminate.
    print(f"\n  OU over ALL {len(rows)} systems (the powered subset is a selection ON OU and quoting"
          f" it there\n  would be circular):")
    for k in ("xcorr", "xcorr_top8", "amp_top8", "iat", "trans"):
        v = 100 * np.mean([r["arms"]["OU"]["inside"][k] for r in rows.values()])
        print(f"    {k:>12}: inside on {v:5.1f}% of all systems")
    print(f"  POWERED systems: {len(pw)}/{len(rows)}  "
          f"(OU outside the band on xcorr_top8 AND amp_top8 -- 81a)")
    if pw:
        # 112 item 3: 0 failures in n gives a 95% upper bound of 3/n by the rule of three.
        j_all = sum(1 for r in pw.values()
                    if r["arms"][ARM.upper()]["inside"][f"xcorr_top{TOPM[0]}"])
        lo = 100 * max(0.0, 1 - 3.0 / len(pw)) if j_all == len(pw) else 100 * j_all / len(pw)
        print(f"  {ARM.upper()} inside on xcorr_top{TOPM[0]}: {j_all}/{len(pw)} powered systems"
              + (f", 95% CI [{lo:.0f}%, 100%] by the rule of three" if j_all == len(pw) else ""))
    if not pw:
        print(f"  -> NO SYSTEM IS POWERED. Nothing about JOINT is readable from this run, including\n"
              f"     a good result. The honest output is the band width, not a table.")
    for arm in ("OU", "SHUFFLE", ARM.upper()):
        src = pw or {}
        if not src: break
        ag = np.array([r["arms"][arm]["agree"] for r in src.values()])
        cells = "".join(f"{100*np.mean([r['arms'][arm]['inside'][k] for r in src.values()]):>12.0f}%"
                        for k in XMETRICS)
        print(f"  {arm:>18}{np.median(ag):>14.1f}{cells}")
    print(f"  (percentages are the share of systems whose generated segments land INSIDE the band "
          f"real segments make)")
    # 107b: samples-seen for EVERY arm, because joint-vs-one-step is only meaningful at matched
    # budget and nothing recorded it before.
    print(f"\n  SAMPLES SEEN (107b): {ARM.upper()} {seen:,} frames over {STEPS:,} steps at effective batch "
          f"{BATCH*ACCUM}\n"
          f"    OU is fitted in closed form from the train replica -- no optimisation budget, so the\n"
          f"    comparison is not budget-matched to it and is not claimed to be. The one-step\n"
          f"    propagator's budget is in its own results file and must be quoted beside any\n"
          f"    joint-vs-one-step statement.")
    bg = np.array([r["arms"]["geometry"]["gen_bond_mean"] for r in rows.values()])
    br = np.array([r["arms"]["geometry"]["ref_bond_mean"] for r in rows.values()])
    print(f"\n  ATOM-LEVEL GEOMETRY: generated consecutive-CA spacing {np.median(bg):.2f} A vs "
          f"reference {np.median(br):.2f} A  ({100*abs(np.median(bg)-np.median(br))/np.median(br):.1f}% off)")
    jo = np.array([r["arms"][ARM.upper()]["agree"] for r in rows.values()])
    ou_ = np.array([r["arms"]["OU"]["agree"] for r in rows.values()])
    print(f"\n  {ARM} beats OU on {100*(jo>ou_).mean():.0f}% of systems, ties {100*(jo==ou_).mean():.0f}%")
    print(f"  NOTE (101d): `trans` is a transition RATE and this corpus determines occupancy to only\n"
          f"  ~61% at the median, so it is reported as band-membership and never as a rate claim.")
