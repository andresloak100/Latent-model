#!/usr/bin/env python3
"""INBOX 72b: is an AFDB model a DRAW from the ensemble, or an estimate of its CENTRE?

The staged video/image recipe treats 1M AFDB structures as still frames. A still frame is a sample
from the same distribution as the moving frames. A predicted structure is not: structure predictors
are mode-seeking, so a model is closer to an estimate of the ensemble's centre than to a typical
thermally populated conformation.

IF THAT HOLDS AT SCALE, the pretrain teaches the manifold of MEAN structures, and the fluctuation
directions -- the one thing the dynamics primary is measured on, since FVE in armf_tied_peer.py is
computed about mu and a static predictor scores exactly zero there -- are what the corpus does not
contain. The pretrain can still be worth doing for geometry, packing and secondary structure. What
changes is what it may be CLAIMED to have taught, and that has to be settled before the result
exists rather than after.

WHY A PROJECTION AND NOT AN RMSD. A scalar distance-to-mean cannot separate "sits at the centre"
from "sits a typical distance away in an unusual direction" -- Family D at the metric. The AFDB model
is projected onto the system's own principal fluctuation modes and its position is reported as a
percentile of the frames' own projections, per mode.

THE THREE GUARDS 72b REQUIRES:
  JOIN       ATLAS is PDB-numbered, AFDB is UniProt-numbered. This goes through ATLAS's OWN
             published per-residue correspondence table (`*_corresp.tsv`, column UnP_num), not
             through residue index and not through a sequence alignment I would have to trust. The
             matched-residue count is reported per system. A silent numbering offset here produces a
             clean-looking number that means nothing -- the shape 27d caught (Family F).
  OVERLAP    the size distribution of ATLAS-with-AFDB against all of ATLAS. If the systems that have
             an AFDB entry are systematically larger or better studied, the conclusion is
             conditioned on that (Family A).
  SPREAD     systems whose fluctuation is too small to resolve a position are FLAGGED, not averaged
             in. A rigid protein puts everything near its mean and would manufacture the conclusion
             (Family B).

VERDICT. "Indistinguishable from a typical frame" is a null and goes through null_verdict with a
relevance bound; on this overlap size that null is weak and says so.
"""
import sys, os, json, math, zipfile, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from armf_stamp import null_verdict

WR = os.environ["WR"]
CACHE = f"{WR}/atlas_cache"
TOPO = f"{WR}/atlas_topo"
AFDB = f"{WR}/atlas_afdb"
RES = os.environ.get("ENS_RES", f"{WR}/afdb_ensemble.json")
NMODE = 10
MIN_RMSF = 0.5            # A; below this the ensemble cannot resolve a position (Family B flag)
TIMEOUT = 180


def fetch(url, dest, timeout=TIMEOUT, minbytes=0):
    """Via curl, not urllib. This venv's Python has no HTTPS support at all -- urlopen raises
    "unknown url type: https" -- which is the same missing-SSL that killed pip here. A urllib
    fetcher fails on every URL while looking like a network fault."""
    if os.path.exists(dest) and os.path.getsize(dest) > minbytes:
        return True
    r = subprocess.run(["curl", "-sS", "-L", "--max-time", str(timeout), "-o", dest + ".tmp",
                        "-w", "%{http_code}", url], capture_output=True, text=True)
    ok = r.returncode == 0 and r.stdout.strip().startswith("2") and os.path.exists(dest + ".tmp") \
        and os.path.getsize(dest + ".tmp") > minbytes
    # `minbytes` defaults to 0 and each caller sets its own. A blanket 1 KB floor rejected every
    # SIFTS mapping -- they are a few hundred bytes -- so a valid response was being discarded as a
    # network failure, and the run reported "no UniProt accession for this chain" for entries whose
    # accession was sitting in the reply. Size is not a validity test; the HTTP status is.
    if not ok:
        if os.path.exists(dest + ".tmp"):
            os.remove(dest + ".tmp")
        return False
    os.replace(dest + ".tmp", dest)
    return True


def atlas_topology(sysid):
    """ATLAS's own .pdb (the structure the MD was launched from, so its atom ORDER is the
    trajectory's) and its own residue correspondence table. Taken from the `analysis` endpoint,
    31 MB, rather than `protein`, 292 MB -- both carry the same two files."""
    os.makedirs(TOPO, exist_ok=True)
    pdbf, corf = f"{TOPO}/{sysid}.pdb", f"{TOPO}/{sysid}_corresp.tsv"
    if os.path.exists(pdbf) and os.path.exists(corf):
        return open(pdbf, errors="ignore").read(), open(corf, errors="ignore").read()
    zp = f"{TOPO}/{sysid}.zip"
    if not fetch(f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{sysid}", zp,
                 minbytes=1 << 20):
        return None, None
    try:
        z = zipfile.ZipFile(zp)
        pdb = z.read(f"{sysid}.pdb").decode(errors="ignore")
        cor = z.read(f"{sysid}_corresp.tsv").decode(errors="ignore")
    except Exception:
        return None, None
    finally:
        pass
    open(pdbf, "w").write(pdb); open(corf, "w").write(cor)
    os.remove(zp)                                     # 31 MB x 125 is not worth keeping
    return pdb, cor


def uniprot_acc(pdb4, chain):
    """PDBe SIFTS. Returns the accession whose mapping names THIS chain -- an entry with several
    UniProt segments would otherwise silently hand back the first one."""
    os.makedirs(AFDB, exist_ok=True)
    cf = f"{AFDB}/{pdb4}_sifts.json"
    if not fetch(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb4}", cf):
        return None
    try:
        d = json.load(open(cf))[pdb4]["UniProt"]
    except Exception:
        return None
    for acc, v in d.items():
        for m in v.get("mappings", []):
            if m.get("chain_id") == chain:
                return acc
    return None


def afdb_ca(acc):
    """CA coordinates of the AFDB model, keyed by UniProt residue number (label_seq_id), which is
    what the correspondence table's UnP_num is expressed in."""
    os.makedirs(AFDB, exist_ok=True)
    f = f"{AFDB}/{acc}.cif"
    if not fetch(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.cif", f):
        return None
    out = {}
    for ln in open(f, errors="ignore"):
        if not ln.startswith("ATOM"):
            continue
        p = ln.split()
        if len(p) > 18 and p[3] == "CA":
            try:
                out[int(p[8])] = (float(p[10]), float(p[11]), float(p[12]))
            except (ValueError, IndexError):
                continue
    return out or None


def kabsch_to(P, Q):
    """Rotate+translate P onto Q. The AFDB model arrives in its own frame, and an unaligned
    comparison would measure the frame rather than the conformation."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (P - pc) @ R.T + qc


def analyse(sysid):
    pdb4, chain = sysid.split("_")[0], sysid.split("_")[1]
    pdb, cor = atlas_topology(sysid)
    if pdb is None:
        return {"sys": sysid, "skip": "no ATLAS topology"}
    atoms = [l for l in pdb.splitlines() if l.startswith(("ATOM", "HETATM"))]
    ca_idx = [i for i, l in enumerate(atoms) if l[12:16].strip() == "CA"]

    npy = f"{CACHE}/{sysid}.npy"
    if not os.path.exists(npy):
        return {"sys": sysid, "skip": "not in ATLAS cache"}
    a = np.load(npy, mmap_mode="r")                    # (R, F, N, 3)
    if a.shape[0] < 3 or a.shape[2] != len(atoms):
        return {"sys": sysid, "skip": f"topology/trajectory mismatch "
                                      f"{len(atoms)} vs {a.shape[2]} atoms"}

    # UnP_num per CA, in the order the CAs appear in the topology. Rows flagged no_ca have no
    # CA atom at all and must not consume a slot in that ordering.
    lines = [l.split("\t") for l in cor.strip().splitlines()]
    hdr = lines[0]
    ix = {k: hdr.index(k) for k in ("UnP_num", "no_ca") if k in hdr}
    if "UnP_num" not in ix:
        return {"sys": sysid, "skip": "corresp.tsv has no UnP_num column"}
    unp = [r[ix["UnP_num"]] for r in lines[1:]
           if not ("no_ca" in ix and r[ix["no_ca"]].strip() == "1")]
    if len(unp) != len(ca_idx):
        return {"sys": sysid, "skip": f"corresp rows {len(unp)} != CA count {len(ca_idx)}"}

    acc = uniprot_acc(pdb4, chain)
    if acc is None:
        return {"sys": sysid, "skip": "no UniProt accession for this chain"}
    model = afdb_ca(acc)
    if model is None:
        return {"sys": sysid, "skip": f"no AFDB model for {acc}"}

    pairs = [(k, int(u)) for k, u in enumerate(unp) if u.strip().isdigit() and int(u) in model]
    if len(pairs) < 20:
        return {"sys": sysid, "skip": f"only {len(pairs)} residues matched", "acc": acc}
    keep_ca = np.array([ca_idx[k] for k, _ in pairs])
    afd = np.array([model[u] for _, u in pairs], np.float64)

    # mu over TRAIN replicas 0+1, matching the peer harness; frames scored on replica 2.
    F = a.shape[1]
    tr = np.concatenate([np.asarray(a[r][:, keep_ca, :]).astype(np.float64) for r in (0, 1)])
    ho = np.asarray(a[2][:, keep_ca, :]).astype(np.float64)
    mu = tr.mean(0)
    tr_al = np.array([kabsch_to(f, mu) for f in tr])
    ho_al = np.array([kabsch_to(f, mu) for f in ho])
    mu = tr_al.mean(0)
    rmsf = float(np.sqrt(((tr_al - mu) ** 2).sum(-1).mean(0).mean()))

    D = (tr_al - mu).reshape(len(tr_al), -1)
    # modes from the TRAIN replicas; the AFDB model and the held-out frames are both scored on a
    # basis neither of them defined.
    _, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    V = Vt[:NMODE]
    afd_al = kabsch_to(afd, mu)
    da = (afd_al - mu).ravel()
    dh = (ho_al - mu).reshape(len(ho_al), -1)

    pa = V @ da                                        # (NMODE,)
    ph = dh @ V.T                                      # (F, NMODE)
    # Where does the model sit on each mode, as a percentile of the frames' own projections?
    pct_mode = [float((ph[:, m] < pa[m]).mean() * 100) for m in range(NMODE)]
    # Two-sided centrality: 0 = dead centre of the frame distribution, 100 = beyond every frame.
    cen_mode = [float((np.abs(ph[:, m]) < abs(pa[m])).mean() * 100) for m in range(NMODE)]

    na = float(np.linalg.norm(da)); nh = np.linalg.norm(dh, axis=1)
    pct_dist = float((nh < na).mean() * 100)
    return {
        "sys": sysid, "acc": acc, "n_matched": len(pairs), "n_ca": len(ca_idx),
        "rmsf": rmsf, "low_spread": bool(rmsf < MIN_RMSF),
        "pct_dist": pct_dist,
        "dist_model": na / math.sqrt(len(pairs)),
        "dist_frames_med": float(np.median(nh)) / math.sqrt(len(pairs)),
        "pct_mode": pct_mode, "cen_mode": cen_mode,
        "var_frac": (S[:NMODE] ** 2 / (S ** 2).sum()).tolist(),
    }


if __name__ == "__main__":
    man = json.load(open(f"{WR}/atlas_manifest.json"))
    ho = man["heldout"]
    print(f"[72b] AFDB model vs the ATLAS ensemble it should be a draw from: {len(ho)} held-out "
          f"systems", flush=True)
    print(f"  join = ATLAS's OWN corresp.tsv (UnP_num), not residue index, not my alignment",
          flush=True)
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    for i, s in enumerate(ho):
        if s in res:
            continue
        try:
            r = analyse(s)
        except Exception as e:
            r = {"sys": s, "skip": f"{type(e).__name__}: {e}"}
        res[s] = r
        tmp = RES + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, RES)
        if "skip" in r:
            print(f"  [{i+1}/{len(ho)}] {s:10s} SKIP {r['skip']}", flush=True)
        else:
            print(f"  [{i+1}/{len(ho)}] {s:10s} {r['acc']:10s} matched {r['n_matched']:>4}/"
                  f"{r['n_ca']:<4} rmsf {r['rmsf']:5.2f}  dist pct {r['pct_dist']:5.1f}"
                  f"{'  LOW-SPREAD' if r['low_spread'] else ''}", flush=True)

    ok = [r for r in res.values() if "skip" not in r]
    if not ok:
        raise SystemExit("\n  NO system produced a measurement. Reported as absent, not as a null.")
    used = [r for r in ok if not r["low_spread"]]
    print(f"\n=== 72b: IS THE AFDB MODEL A DRAW FROM THE ENSEMBLE? ===", flush=True)
    print(f"  usable {len(ok)}/{len(res)} systems; {len(ok)-len(used)} FLAGGED low-spread "
          f"(rmsf < {MIN_RMSF} A) and excluded from the verdict, not averaged in", flush=True)
    print(f"  matched residues per system: median {np.median([r['n_matched'] for r in ok]):.0f}, "
          f"min {min(r['n_matched'] for r in ok)}, "
          f"as a fraction of CA {np.median([r['n_matched']/r['n_ca'] for r in ok]):.2f}", flush=True)

    # OVERLAP GUARD (Family A): is the set that HAS an AFDB entry a biased slice of ATLAS?
    allmeta = [json.load(open(f"{CACHE}/{s}.json"))["length"]
               for s in ho if os.path.exists(f"{CACHE}/{s}.json")]
    gotlen = [json.load(open(f"{CACHE}/{r['sys']}.json"))["length"] for r in ok
              if os.path.exists(f"{CACHE}/{r['sys']}.json")]
    if allmeta and gotlen:
        from scipy import stats as _st
        ks = _st.ks_2samp(gotlen, allmeta)
        print(f"  OVERLAP GUARD: length median {np.median(gotlen):.0f} (with AFDB) vs "
              f"{np.median(allmeta):.0f} (all held-out), KS p={ks.pvalue:.3f} -> "
              f"{'NOT a biased slice' if ks.pvalue > 0.05 else 'BIASED slice; the verdict is conditioned on it'}",
              flush=True)

    pd = np.array([r["pct_dist"] for r in used])
    print(f"\n  DISTANCE-TO-MEAN percentile among the frames' own distances")
    print(f"    median {np.median(pd):.1f}  mean {pd.mean():.1f}  "
          f"[{np.percentile(pd,25):.1f}, {np.percentile(pd,75):.1f}]")
    print(f"    a DRAW from the ensemble sits at 50. Below 50 = closer to the mean than a typical "
          f"frame.")
    print(f"    fraction of systems below the 25th percentile of their own frames: "
          f"{(pd < 25).mean():.1%}  ({int((pd<25).sum())}/{len(pd)})", flush=True)

    cen = np.array([r["cen_mode"] for r in used])          # (S, NMODE)
    print(f"\n  PER-MODE CENTRALITY -- percentile of |projection| among the frames' own")
    print(f"    {'mode':>5}{'median':>9}{'mean':>8}{'var frac':>10}")
    vf = np.array([r["var_frac"] for r in used])
    for m in range(NMODE):
        print(f"    {m+1:>5}{np.median(cen[:,m]):>9.1f}{cen[:,m].mean():>8.1f}"
              f"{np.median(vf[:,m]):>10.3f}")
    print(f"    A mode-seeking predictor sits LOW on every mode. A draw sits near 50 on all of "
          f"them.\n    This is the measurement a scalar RMSD-to-mean cannot make: it separates "
          f"'at the centre'\n    from 'a typical distance away in an unusual direction'.", flush=True)

    # VERDICT through null_verdict with a relevance bound (72b's own instruction).
    est = float(np.median(pd) - 50.0)
    sd = float(pd.std(ddof=1)) if len(pd) > 1 else float("nan")
    hw = 1.96 * sd / math.sqrt(len(pd)) if len(pd) > 1 else float("nan")
    print(f"\n  VERDICT on 'the AFDB model is a typical frame' (percentile == 50):")
    print(f"    estimate {est:+.1f} percentile points, 95% CI +/-{hw:.1f}, n={len(pd)}")
    print("    " + null_verdict(est, hw, 10.0, "departure from a typical frame").replace("\n", "\n    "),
          flush=True)
    json.dump(res, open(RES, "w"))
