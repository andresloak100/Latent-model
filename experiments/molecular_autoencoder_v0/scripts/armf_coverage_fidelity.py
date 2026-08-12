#!/usr/bin/env python3
"""INBOX 116 item 1: the ATOM-LEVEL error of a GENERATED trajectory, as a two-sided nearest-neighbour
pair rather than as an RMSD to a matched frame.

WHAT WAS MISSING. 109c compared a floor to a floor: both 3.009 A and 3.537 A are the SAME rank-64
reconstruction of TRUE frames under two sigmas. That is what 109c asked for and the sigma conclusion
stands untouched -- but no atom-level number for a GENERATED trajectory exists on this record, and
milestones 3 and 4 both need one.

WHY NOT RMSD. RMSD is a reconstruction metric and a generator is not a reconstructor. "How far is
generated frame t from reference frame t" is the wrong question: a sample from a distribution is not
an estimate of a particular frame, and a perfect model -- one drawing exactly from the equilibrium
ensemble -- would score terribly, because its frame t is an independent draw. So the pair:

    COVERAGE   for each REFERENCE frame, min RMSD to ANY generated frame
               -> did the model reach the conformations the protein actually visits?
               -> poor coverage is MODE COLLAPSE
    FIDELITY   for each GENERATED frame, min RMSD to ANY reference frame
               -> is everything it produced a real conformation?
               -> poor fidelity is HALLUCINATED STRUCTURE

Different problems, so both medians AND both distributions are reported. One number cannot be
substituted for the other, and a model can be good at either alone.

THESE METRICS ARE BLIND TO TIME BY CONSTRUCTION, AND THAT IS MEASURED HERE, NOT ARGUED. Both are
functions of the SET of frames, so any permutation of time leaves them exactly unchanged. The
SHUFFLE arm therefore scores 0.000 A on both -- a PERFECT score for a model with no dynamics at all.
It is included precisely so that number is on the page: coverage and fidelity answer milestone 3/4's
GEOMETRY question and say nothing whatever about dynamics, which is what `trans` and `iat` are for
(115a). Reading good coverage as a dynamics result would be the eleventh instance of one name for
two things.

THE FLOOR IS COMPUTED THE SAME WAY, WHICH IS WHAT SEPARATES THE DECODE FROM THE MODEL. The FLOOR arm
replaces generated frames with the rank-64 RECONSTRUCTION of the reference frames, and runs through
the identical nearest-neighbour code. Whatever the ANM truncation costs appears in both arms, so the
model's own contribution is the gap. Without it, a 3 A coverage would be indistinguishable from a
decode that cannot do better than 3 A on true frames -- which 115d showed is exactly the case.

ALIGNMENT: no superposition is applied, matching 109c. Every arm including the floor shares whatever
alignment convention the Atlas trajectories carry, so rigid-body motion inflates all arms equally and
the floor absorbs it. The comparison is between arms, never against an absolute.
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
NREF = int(os.environ.get("CF_NREF", "512"))     # reference frames drawn from replicas 1 and 2
NGEN = int(os.environ.get("CF_NGEN", "512"))     # generated frames, from NSEG independent segments
NSEG = int(os.environ.get("CF_NSEG", "8"))
STEPS = int(os.environ.get("CF_STEPS", "50"))
CKPT = os.environ.get("CF_CKPT", f"{WR}/latent_video_ckpt_joint_r8.pt")
RES = os.environ.get("CF_RES", f"{WR}/coverage_fidelity.json")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nn_rmsd(A, B, natom):
    """Two-sided nearest-neighbour RMSD between two frame sets, (nA, 3N) and (nB, 3N).

    Returns (coverage, fidelity) as per-frame vectors: coverage[i] is the min RMSD from reference
    frame i to ANY frame of B; fidelity[j] is the min from generated frame j to ANY frame of A.
    Computed via the Gram identity rather than a pairwise loop -- exact, and the (nA, nB) matrix is
    the only thing held.
    """
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0) / natom
    r = np.sqrt(d2)
    return r.min(1), r.min(0)


def arm_stats(Rtrue, G, natom, ca):
    cov, fid = nn_rmsd(Rtrue, G, natom)
    out = dict(coverage=float(np.median(cov)), fidelity=float(np.median(fid)),
               coverage_p90=float(np.percentile(cov, 90)),
               fidelity_p90=float(np.percentile(fid, 90)),
               coverage_min=float(cov.min()), fidelity_min=float(fid.min()),
               coverage_max=float(cov.max()), fidelity_max=float(fid.max()))
    if ca is not None and len(ca) > 1:
        sel = np.concatenate([3 * ca, 3 * ca + 1, 3 * ca + 2])
        sel.sort()
        c2, f2 = nn_rmsd(Rtrue[:, sel], G[:, sel], len(ca))
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
    print(f"[116.1] coverage / fidelity of GENERATED trajectories. device={dev}", flush=True)
    print(f"  K={K} R={RGRP} Dlat={DLAT} T={T}  ckpt={os.path.basename(CKPT)}", flush=True)
    if not os.path.exists(CKPT):
        raise SystemExit(f"  checkpoint absent: {CKPT}")
    HO = prepare(store, have, ho_ids, NEVAL)
    print(f"  prepared {len(HO)} held-out systems", flush=True)

    cfg = LatentVideoConfig(latent_dim=DLAT, d_model=256, depth=6, max_frames_hint=T)
    mdl = LatentVideoDiffusion(cfg).to(dev)
    sd_state = torch.load(CKPT, map_location=dev)
    mdl.load_state_dict(sd_state.get("model", sd_state) if isinstance(sd_state, dict)
                        and "model" in sd_state else sd_state)
    mdl.eval()
    print(f"  loaded {sum(p.numel() for p in mdl.parameters()):,} parameters", flush=True)

    rng = np.random.default_rng(0)
    rows, t0 = {}, time.time()
    for i, s in enumerate(HO, 1):
        try:
            d = sysdata(store, have[s["pdb"]])
            a = np.load(d["path"], mmap_mode="r")
            # REFERENCE FRAMES: replicas 1 and 2, the same replicas the acceptance test draws its
            # band from, and disjoint from replica 0 which set the whitening (106b.2).
            per = max(1, NREF // 2)
            Rtrue = np.concatenate([
                np.asarray(a[r][np.arange(0, d["F"], max(1, d["F"] // per))[:per]])
                .astype(np.float64).reshape(-1, 3 * d["N"]) for r in (1, 2)])
            natom, mu, V, sdw = d["N"], d["mu"], s["V"], s["sd"]

            def decode(C):                       # whitened coefficients -> atom coordinates
                return (C * sdw) @ V.T + mu

            with torch.no_grad():
                z = mdl.sample((NSEG, T, RGRP, DLAT), dev, steps=STEPS).cpu().numpy()
            z = z.reshape(NSEG * T, K)
            z = z[np.arange(0, len(z), max(1, len(z) // NGEN))[:NGEN]]
            ou = ou_segments(s["C_tr"], s["C_ho"], T, NSEG, rng).reshape(NSEG * T, K)
            ou = ou[np.arange(0, len(ou), max(1, len(ou) // NGEN))[:NGEN]]
            ca = ca_index(s["pdb"], natom)

            arms = {}
            # THE FLOOR: rank-64 reconstruction of the reference frames, through the identical
            # nearest-neighbour code, so the decode is separated from the model.
            arms["FLOOR"] = arm_stats(Rtrue, ((Rtrue - mu) @ V) @ V.T + mu, natom, ca)
            # THE TIME-BLINDNESS CONTROL: the reference frames themselves, permuted. A SET metric
            # cannot see this, so it must score exactly 0.000 on both sides.
            arms["SHUFFLE"] = arm_stats(Rtrue, Rtrue[rng.permutation(len(Rtrue))], natom, ca)
            arms["OU"] = arm_stats(Rtrue, decode(ou), natom, ca)
            arms["JOINT"] = arm_stats(Rtrue, decode(z), natom, ca)
            rows[s["pdb"]] = dict(N=natom, n_ref=len(Rtrue), n_gen=len(z), arms=arms)
            f, j = arms["FLOOR"], arms["JOINT"]
            print(f"  [{i}/{len(HO)}] {s['pdb']:10s} N={natom:>6}  "
                  f"FLOOR cov {f['coverage']:5.2f} fid {f['fidelity']:5.2f} | "
                  f"JOINT cov {j['coverage']:5.2f} fid {j['fidelity']:5.2f} | "
                  f"SHUFFLE cov {arms['SHUFFLE']['coverage']:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"  [{i}/{len(HO)}] {s['pdb']}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(HO), complete=(i == len(HO)), n_failed=0,
                          K=K, R=RGRP, T=T, ckpt=os.path.basename(CKPT))

    if not rows: raise SystemExit("  nothing scored")
    print(f"\n=== 116.1: COVERAGE and FIDELITY ({len(rows)} held-out systems) ===")
    print(f"  coverage = for each REFERENCE frame, min RMSD to any generated frame (mode collapse)")
    print(f"  fidelity = for each GENERATED frame, min RMSD to any reference frame (hallucination)")
    print(f"  {'arm':>9}{'coverage':>11}{'p90':>8}{'fidelity':>11}{'p90':>8}"
          f"{'cov CA':>9}{'fid CA':>9}{'vs FLOOR cov':>14}{'vs FLOOR fid':>14}")
    base = {}
    for arm in ("FLOOR", "SHUFFLE", "OU", "JOINT"):
        g = lambda k: np.array([r["arms"][arm][k] for r in rows.values() if k in r["arms"][arm]])
        if not len(g("coverage")): continue
        c, f = float(np.median(g("coverage"))), float(np.median(g("fidelity")))
        if arm == "FLOOR": base = dict(c=c, f=f)
        ca_c = g("coverage_ca"); ca_f = g("fidelity_ca")
        dc = f"{c-base['c']:+13.3f}" if base else " " * 14
        df = f"{f-base['f']:+13.3f}" if base else " " * 14
        print(f"  {arm:>9}{c:>10.3f}A{float(np.median(g('coverage_p90'))):>8.2f}"
              f"{f:>10.3f}A{float(np.median(g('fidelity_p90'))):>8.2f}"
              f"{(float(np.median(ca_c)) if len(ca_c) else float('nan')):>9.2f}"
              f"{(float(np.median(ca_f)) if len(ca_f) else float('nan')):>9.2f}{dc} {df}")
    sh = np.array([r["arms"]["SHUFFLE"]["coverage"] for r in rows.values()])
    print(f"\n  THE SHUFFLE ROW IS THE POINT OF THE CONTROL: max over systems "
          f"{sh.max():.2e} A on coverage.")
    print(f"  A model with NO DYNAMICS AT ALL scores perfectly here, because coverage and fidelity")
    print(f"  are functions of the frame SET. They answer milestone 3/4's geometry question and say")
    print(f"  nothing about dynamics -- `trans` and `iat` (115a) are the instruments for that.")
