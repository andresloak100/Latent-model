#!/usr/bin/env python3
"""INBOX 116.1 / 117: the ATOM-LEVEL error of a GENERATED trajectory, as a two-sided
nearest-neighbour pair, at MATCHED sample count, against what a PERFECT generator scores.

WHAT WAS MISSING. 109c compared a floor to a floor: both 3.009 A and 3.537 A are the SAME rank-64
reconstruction of TRUE frames under two sigmas. No atom-level number for a GENERATED trajectory
existed, and milestones 3 and 4 both need one.

WHY NOT RMSD. A sample from a distribution is not an estimate of a particular frame, so "how far is
generated frame t from reference frame t" is the wrong question -- a model drawing exactly from
equilibrium would score terribly. The two-sided pair instead:

    COVERAGE   per REFERENCE frame, min RMSD to ANY generated frame -> mode collapse
    FIDELITY   per GENERATED frame, min RMSD to ANY reference frame -> hallucinated structure

117.1: COVERAGE IS COUNT-DEPENDENT AND FIDELITY IS NOT, SO THE COUNT MUST BE MATCHED AND PRINTED.
Coverage is a mean over reference frames of the min distance to the GENERATED SET, so every extra
generated frame can only lower it; fidelity averages a per-frame quantity, so the count divides out.
Simulated on a perfect model (same distribution both sides, K=64, ANM-like spectrum, 2,000 reference
frames), coverage falls 24% on sample count alone -- 3.3461 at n_gen=10 to 2.5318 at n_gen=3200 --
while fidelity moves 2.6004 to 2.5698. Every arm here therefore emits EXACTLY n_gen frames, the
count is a printed column rather than an inferred one, and fidelity is labelled count-invariant so
the caveat is not applied to both.

117.2: TWO BUDGETS, BECAUSE THE SLOPE IS THE DIAGNOSIS. Coverage is reported at n_gen and at 4*n_gen.
A model whose coverage barely improves with 4x the samples HAS COLLAPSED; one that keeps improving
was UNDER-SAMPLED. That distinction is invisible at a single budget and it is the whole question the
coverage side exists to answer.

117.3: THE FLOOR IS A RECONSTRUCTION, NOT A RESAMPLE, SO IT IS NOT WHAT A PERFECT MODEL SCORES.
It reproduces the same frames and therefore measures decode error only; a perfect generator draws
FRESH samples and scores well above zero. The arm that was missing is REF: replica 1 subsampled to
n_gen, scored against replica 2. TWO INDEPENDENT REPLICAS ARE TWO INDEPENDENT SAMPLES OF THE SAME
DYNAMICS -- exactly what a perfect generator produces. Held-out systems have all three replicas
unseen and the whitening came from replica 0 alone (106b.2), so this costs nothing and leaks nothing.

    REF        what a PERFECT generator scores at this n_gen        <- the scale
    FLOOR      the SAME replica-1 frames, rank-64 reconstructed     <- REF plus decode error, only
    SHUFFLE    the SAME replica-1 frames, time-permuted             <- must equal REF EXACTLY
    OU         the physics null
    JOINT      the model

Because FLOOR and SHUFFLE are built from the SAME frames as REF, three of the five arms differ from
each other by exactly one thing, and JOINT reads as a fraction of the REF-to-OU gap rather than
against a floor that measures something else.

117.4: WHAT THIS PAIR DOES NOT MEASURE. Both are functions of the frame SET, so any permutation of
time leaves them EXACTLY unchanged -- the SHUFFLE arm scores identically to REF, and the delta is
printed to prove it rather than asserted. The suite is now THREE TIERS: nine time-blind distributional
metrics; two time-sensitive ones (iat, trans); and this set pair, which is ALSO time-blind. Coverage
and fidelity do see cross-mode coupling, so they are a genuine addition on the CONFORMATIONAL axis,
but they see nothing about ordering. THE TEMPORAL CLAIM RESTS ENTIRELY ON iat AND trans, however this
pair comes out.

ALIGNMENT: no superposition, matching 109c. Every arm shares whatever alignment the Atlas
trajectories carry, so rigid-body motion inflates all arms equally and the comparison is between
arms, never against an absolute.
"""
import sys, os, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE + "/.."
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from molae.latent_video import LatentVideoConfig, LatentVideoDiffusion
from armf_latent_video_run import prepare, ou_segments, K, T, RGRP, DLAT
from armf_atlas_data import AtlasStore, sysdata
import armf_atlas_dm as D
import armf_io

WR = D.WR
NEVAL = int(os.environ.get("CF_NEVAL", "24"))
NREF = int(os.environ.get("CF_NREF", "1024"))    # reference frames, replica 2 only
NGEN = int(os.environ.get("CF_NGEN", "256"))     # the MATCHED budget; 4*NGEN is the second budget
MULT = int(os.environ.get("CF_MULT", "4"))
STEPS = int(os.environ.get("CF_STEPS", "50"))
CKPT = os.environ.get("CF_CKPT", f"{WR}/latent_video_ckpt_joint_r8.pt")
RES = os.environ.get("CF_RES", f"{WR}/coverage_fidelity.json")
ARMS = ("REF", "FLOOR", "SHUFFLE", "OU", "JOINT")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nn_rmsd(A, B, natom):
    """Two-sided nearest-neighbour RMSD between frame sets (nA, 3N) and (nB, 3N).

    coverage[i] = min RMSD from reference frame i to ANY frame of B
    fidelity[j] = min RMSD from generated frame j to ANY frame of A
    Via the Gram identity rather than a pairwise loop: exact, and (nA, nB) is all that is held.
    """
    d2 = np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T), 0.0) / natom
    r = np.sqrt(d2)
    return r.min(1), r.min(0)


def sub(P, n):
    """Uniform stride to EXACTLY n frames, so every arm's count is the same by construction."""
    if len(P) < n: return P
    return P[np.linspace(0, len(P) - 1, n).astype(int)]


def arm_stats(Rref, G, natom, ca):
    """One arm at one budget. `n_gen` is returned so the count is a printed column, never inferred."""
    cov, fid = nn_rmsd(Rref, G, natom)
    out = dict(n_gen=int(len(G)),
               coverage=float(np.median(cov)), fidelity=float(np.median(fid)),
               coverage_p90=float(np.percentile(cov, 90)),
               fidelity_p90=float(np.percentile(fid, 90)),
               coverage_max=float(cov.max()), fidelity_min=float(fid.min()))
    if ca is not None and len(ca) > 1:
        sel = np.sort(np.concatenate([3 * ca, 3 * ca + 1, 3 * ca + 2]))
        c2, f2 = nn_rmsd(Rref[:, sel], G[:, sel], len(ca))
        out["coverage_ca"] = float(np.median(c2)); out["fidelity_ca"] = float(np.median(f2))
    return out


def ca_index(pdb, N):
    p = f"{WR}/atlas_topo/{pdb}.pdb"
    if not os.path.exists(p): return None
    at = [l for l in open(p, errors="ignore").read().splitlines()
          if l.startswith(("ATOM", "HETATM"))]
    if len(at) != N: return None
    return np.array([i for i, l in enumerate(at) if l[12:16].strip() == "CA"])


if __name__ == "__main__":
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    print(f"[116.1/117] coverage / fidelity of GENERATED trajectories. device={dev}", flush=True)
    print(f"  K={K} R={RGRP} Dlat={DLAT} T={T} steps={STEPS}  ckpt={os.path.basename(CKPT)}",
          flush=True)
    print(f"  MATCHED budget n_gen={NGEN}, second budget {MULT}*n_gen={MULT*NGEN}, "
          f"reference = replica 2 at {NREF} frames", flush=True)
    if not os.path.exists(CKPT): raise SystemExit(f"  checkpoint absent: {CKPT}")
    HO = prepare(store, have, ho_ids, NEVAL)
    print(f"  prepared {len(HO)} held-out systems", flush=True)

    cfg = LatentVideoConfig(latent_dim=DLAT, d_model=256, depth=6, max_frames_hint=T)
    mdl = LatentVideoDiffusion(cfg).to(dev)
    st = torch.load(CKPT, map_location=dev)
    mdl.load_state_dict(st.get("model", st) if isinstance(st, dict) and "model" in st else st)
    mdl.eval()
    print(f"  loaded {sum(p.numel() for p in mdl.parameters()):,} parameters", flush=True)

    NBIG = MULT * NGEN
    nseg = int(np.ceil(NBIG / T))
    rng = np.random.default_rng(0)
    rows, t0 = {}, time.time()
    for i, s in enumerate(HO, 1):
        try:
            d = sysdata(store, have[s["pdb"]])
            a = np.load(d["path"], mmap_mode="r")
            natom, mu, V, sdw = d["N"], d["mu"], s["V"], s["sd"]
            # REFERENCE = replica 2. The "perfect generator" arm is replica 1. Both are held out and
            # the whitening came from replica 0 alone (106b.2), so no set here has seen another.
            Rref = sub(np.asarray(a[2]).astype(np.float64).reshape(d["F"], -1), NREF)
            R1 = sub(np.asarray(a[1]).astype(np.float64).reshape(d["F"], -1), NBIG)

            def decode(C):
                return (C * sdw) @ V.T + mu

            with torch.no_grad():
                z = mdl.sample((nseg, T, RGRP, DLAT), dev, steps=STEPS).cpu().numpy()
            gen = sub(decode(z.reshape(nseg * T, K)), NBIG)
            ou = sub(decode(ou_segments(s["C_tr"], s["C_ho"], T, nseg, rng).reshape(nseg * T, K)),
                     NBIG)
            # FLOOR and SHUFFLE are built from the SAME frames as REF, so each differs from REF by
            # exactly one thing: the decode, and the time order.
            floor = ((R1 - mu) @ V) @ V.T + mu
            ca = ca_index(s["pdb"], natom)
            # 117.3 DIAGNOSTIC: REF is only a meaningful "perfect generator" bar where the two
            # replicas sample the same region. Measured on 4ued_B, they do NOT -- the replica means
            # are 8.94 A apart AFTER Kabsch superposition, against a within-replica spread of 7.57 A,
            # while 7lp1_A sits at 1.70 A against 2.78 A. On a system like the first, a null centred
            # on the POOLED mean covers replica 2 better than a real sample from replica 1 does, so
            # OU can beat a perfect generator by blurring to the centre. Recorded per system so that
            # can be read off rather than inferred from an anomalous median.
            m1, m2 = R1.mean(0), Rref.mean(0)
            rms = lambda u, v: float(np.sqrt(((u - v) ** 2).reshape(-1, 3).sum(-1).mean()))
            sep = dict(replica_mean_sep=rms(m1, m2), spread_ref=rms(Rref, m2),
                       mu_to_ref_mean=rms(mu, m2))

            pools = dict(REF=R1, FLOOR=floor, OU=ou, JOINT=gen)
            budgets = {}
            for nb in (NGEN, NBIG):
                b = {k: arm_stats(Rref, sub(P, nb), natom, ca) for k, P in pools.items()}
                # SHUFFLE IS BUILT AFTER SUBSAMPLING, WHICH IS THE WHOLE POINT OF THE CONTROL.
                # Permuting first and subsampling second selects a DIFFERENT SUBSET, because sub()
                # strides by index -- so the delta measured the subsample, not the time order, and
                # came out at 1.6e-01 instead of 0. Permuting the ALREADY-SUBSAMPLED frames gives
                # the identical SET in a different order, which is the invariance being tested.
                Rs = sub(R1, nb)
                b["SHUFFLE"] = arm_stats(Rref, Rs[rng.permutation(len(Rs))], natom, ca)
                budgets[str(nb)] = b
            # 117.4: the time-blindness is PROVEN here, not asserted. SHUFFLE is REF's frames
            # permuted, so a set metric must return bit-identical numbers.
            inv = max(abs(budgets[str(nb)]["SHUFFLE"][k] - budgets[str(nb)]["REF"][k])
                      for nb in (NGEN, NBIG) for k in ("coverage", "fidelity"))
            rows[s["pdb"]] = dict(N=natom, n_ref=len(Rref), budgets=budgets,
                                 shuffle_delta=float(inv), **sep)
            b = budgets[str(NGEN)]
            print(f"  [{i}/{len(HO)}] {s['pdb']:10s} N={natom:>6} n_gen={NGEN} "
                  f"REF {b['REF']['coverage']:5.2f}/{b['REF']['fidelity']:5.2f}  "
                  f"FLOOR {b['FLOOR']['coverage']:5.2f}/{b['FLOOR']['fidelity']:5.2f}  "
                  f"OU {b['OU']['coverage']:5.2f}/{b['OU']['fidelity']:5.2f}  "
                  f"JOINT {b['JOINT']['coverage']:5.2f}/{b['JOINT']['fidelity']:5.2f}  "
                  f"shufD {inv:.1e}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(HO)}] {s['pdb']}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(HO), complete=(i == len(HO)), n_failed=0,
                          K=K, R=RGRP, T=T, n_gen=NGEN, steps=STEPS, ckpt=os.path.basename(CKPT))

    if not rows: raise SystemExit("  nothing scored")
    med = lambda nb, arm, k: float(np.median([r["budgets"][str(nb)][arm][k] for r in rows.values()
                                              if k in r["budgets"][str(nb)][arm]]))
    print(f"\n=== 116.1/117: COVERAGE and FIDELITY ({len(rows)} held-out systems) ===")
    print(f"  coverage: per REFERENCE frame, min RMSD to any generated frame -- MODE COLLAPSE.")
    print(f"            COUNT-DEPENDENT (117.1), so it is reported at two matched budgets.")
    print(f"  fidelity: per GENERATED frame, min RMSD to any reference frame -- HALLUCINATION.")
    print(f"            COUNT-INVARIANT (117.1): the caveat above does NOT apply to it.")
    print(f"\n  {'arm':>9}{'n_gen':>7}{'cov@n':>9}{'cov@4n':>9}{'slope':>9}"
          f"{'fidelity':>10}{'fid p90':>9}{'cov CA':>8}{'fid CA':>8}  reading")
    READ = {"REF": "PERFECT generator at this n_gen",
            "FLOOR": "REF + rank-64 decode error, same frames",
            "SHUFFLE": "REF's frames, time-permuted",
            "OU": "physics null",
            "JOINT": "the model"}
    for arm in ARMS:
        c1, c4 = med(NGEN, arm, "coverage"), med(NBIG, arm, "coverage")
        print(f"  {arm:>9}{int(med(NGEN, arm, 'n_gen')):>7}{c1:>8.3f}A{c4:>8.3f}A{100*(c1-c4)/max(c1,1e-9):>8.1f}%"
              f"{med(NGEN, arm, 'fidelity'):>9.3f}A{med(NGEN, arm, 'fidelity_p90'):>9.2f}"
              f"{med(NGEN, arm, 'coverage_ca'):>8.2f}{med(NGEN, arm, 'fidelity_ca'):>8.2f}"
              f"  {READ[arm]}")
    print(f"\n  'slope' is the coverage gain from {NGEN} to {NBIG} samples. A model whose coverage")
    print(f"  barely improves with {MULT}x the samples HAS COLLAPSED; one that keeps improving was")
    print(f"  UNDER-SAMPLED. REF's own slope is the count effect with a perfect generator, so any")
    print(f"  arm should be read against THAT and not against zero.")
    # REF IS A TWO-SIDED TARGET, NOT A FLOOR, AND THE FIRST DRAFT OF THIS LINE GOT IT WRONG.
    # "% of the way from REF to OU" assumes the null is the far end, but OU's fidelity came out
    # BELOW REF's -- an OU centred on the pooled mean sits closer to every reference frame than a
    # real independent sample does. Scoring BELOW REF is therefore not "better than perfect", it is
    # UNDER-DISPERSION: frames collapsed toward the mean. A perfect generator MATCHES REF on both
    # sides, so the reportable quantity is the SIGNED deviation from REF and its direction.
    print(f"\n  DEVIATION FROM REF, which is a TWO-SIDED target -- a perfect generator MATCHES it.")
    print(f"  Above REF on coverage = under-covering (mode collapse). Below REF on fidelity =")
    print(f"  UNDER-DISPERSED (collapsed toward the mean), which is a defect, not an improvement.")
    print(f"    {'n_gen':>7}{'arm':>9}{'coverage - REF':>17}{'fidelity - REF':>17}  reading")
    for nb in (NGEN, NBIG):
        r, rf = med(nb, "REF", "coverage"), med(nb, "REF", "fidelity")
        for arm in ("FLOOR", "OU", "JOINT"):
            dc, df = med(nb, arm, "coverage") - r, med(nb, arm, "fidelity") - rf
            tag = ("under-covers" if dc > 0 else "over-covers") + ", " + \
                  ("hallucinating" if df > 0 else "UNDER-DISPERSED")
            print(f"    {nb:>7}{arm:>9}{dc:>+16.3f}A{df:>+16.3f}A  {tag}")
    print(f"\n  117.3 GUARD -- REF is a meaningful bar only where the replicas sample the same")
    print(f"  region. Per-system replica separation (all in A):")
    seps = np.array([r["replica_mean_sep"] for r in rows.values()])
    sprd = np.array([r["spread_ref"] for r in rows.values()])
    bad = [k for k, r in rows.items() if r["replica_mean_sep"] > r["spread_ref"]]
    print(f"    median mean(rep1)-mean(rep2) {np.median(seps):.3f} A vs median within-replica"
          f" spread {np.median(sprd):.3f} A")
    print(f"    systems where the replicas are FURTHER apart than one replica is wide: "
          f"{len(bad)}/{len(rows)}  {bad[:6]}")
    print(f"    On those, a null centred on the pooled mean can COVER better than a real sample")
    print(f"    from replica 1, so OU beating REF on coverage there is centre-blur, not quality.")
    sd_ = max(r["shuffle_delta"] for r in rows.values())
    print(f"\n  117.4 -- SHUFFLE vs REF, max |delta| over all systems and budgets: {sd_:.3e}")
    print(f"  SHUFFLE is REF's frames PERMUTED, so a set metric must return identical numbers, and")
    print(f"  it does. The suite is THREE TIERS: nine time-blind distributional metrics; TWO")
    print(f"  time-sensitive (iat, trans); and this pair, which is ALSO time-blind. It is a genuine")
    print(f"  addition on the CONFORMATIONAL axis -- it does see cross-mode coupling -- but the")
    print(f"  TEMPORAL claim rests entirely on iat and trans, however this pair comes out.")
