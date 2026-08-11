#!/usr/bin/env python3
"""INBOX 100c: what is the ANM-orthogonal residual MADE OF?

THIS DECIDES 98b AND THE SEM ARM ON MEASUREMENT RATHER THAN ARGUMENT, and it needs no training.

The open question is whether a SHARED model can reach the residual at all. A basis cannot transfer
between systems of different N, so the question has to be asked in a SIZE-INDEPENDENT form: not
"which directions" but "which atoms". Atom class transfers; a 598-atom system and a 33,377-atom system
both have backbones, side chains, buried cores and exposed surfaces.

  CONCENTRATED ON A CHEMICAL CLASS      there is shared structure. A model that sees atom identity --
                                        which this architecture does, via `stat` -- has something
                                        learnable, and SEM is motivated on evidence rather than by
                                        analogy to image models.
  SPREAD OVER BACKBONE, SYSTEM-SPECIFIC no shared latent generalises it, and the axis is closed for
                                        this architecture class regardless of the prior.

HOW THE CLASSES ARE OBTAINED. ATLAS publishes the .pdb the MD was launched from, and 72b already
downloaded all 125 -- so atom names and residue types come from the SOURCE rather than from a guess
about atom ordering. Burial is computed from the reference structure as a neighbour count, which is
a coordination number, not SASA: it is monotone with burial, needs no extra dependency, and is
reported as what it is.

The energy is measured in the SAME residual the 099 harness scores in, through the SAME projector, so
"the residual" means one thing across both files.
"""
import sys, os, json, time, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import armf_anm_orthogonal as AO
from armf_atlas_data import AtlasStore, sysdata
from armf_anm import modes as anm_modes
import armf_atlas_dm as D
import armf_io

WR = D.WR
TOPO = f"{WR}/atlas_topo"
K_ANM = int(os.environ.get("RC_K", "256"))
CUTOFF = float(os.environ.get("RC_CUTOFF", "5.0"))
NFRAME = int(os.environ.get("RC_NFRAME", "400"))
BURIED_R = float(os.environ.get("RC_BURIED_R", "10.0"))
BURIED_Q = float(os.environ.get("RC_BURIED_Q", "0.5"))
RES = os.environ.get("RC_RES", f"{WR}/residual_composition.json")
BB = {"N", "CA", "C", "O", "OXT"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"}


def topology(sysid, n_expect):
    """Atom names and residue names in the trajectory's own atom order. Refuses on a count
    mismatch rather than reindexing -- a silent offset here would assign every label to the wrong
    atom and still produce a clean-looking table."""
    p = f"{TOPO}/{sysid}.pdb"
    if not os.path.exists(p): return None
    at = [l for l in open(p, errors="ignore").read().splitlines()
          if l.startswith(("ATOM", "HETATM"))]
    if len(at) != n_expect: return None
    return ([l[12:16].strip() for l in at], [l[17:20].strip() for l in at])


if __name__ == "__main__":
    store = AtlasStore(f"{WR}/atlas_cache")
    man = json.load(open(D.MAN))
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho = [p for p in man["heldout"] if p in have]
    ho.sort(key=lambda p: store.meta[have[p]]["atoms"])
    print(f"[100c] what is the residual made of? {len(ho)} systems, ANM K={K_ANM}", flush=True)
    rows, failed, t0 = {}, 0, time.time()
    for i, pdb in enumerate(ho, 1):
        try:
            d = sysdata(store, have[pdb])
            if d is None: failed += 1; continue
            top = topology(pdb, d["N"])
            if top is None:
                failed += 1
                print(f"  [{i}/{len(ho)}] {pdb}: SKIP -- topology absent or atom count mismatch",
                      flush=True)
                continue
            names, resn = top
            V, _, _ = anm_modes(d["ref"], K_ANM, CUTOFF)
            if V is None: failed += 1; continue
            a = np.load(d["path"], mmap_mode="r")
            step = max(1, d["F"] // NFRAME)
            idx = np.arange(0, d["F"], step)[:NFRAME]
            X = np.asarray(a[2][idx]).astype(np.float64).reshape(len(idx), -1)
            R = AO._perp(X - d["mu"], V).reshape(len(idx), d["N"], 3)
            e_atom = (R ** 2).sum(-1).mean(0)                    # residual energy per atom
            tot = float(e_atom.sum()) + 1e-12

            ref = d["ref"]
            dd = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
            nb = (dd < BURIED_R).sum(1) - 1                      # coordination number
            thr = np.quantile(nb, BURIED_Q)
            is_bb = np.array([n in BB for n in names])
            is_bur = nb >= thr
            frac = dict(
                backbone=float(e_atom[is_bb].sum() / tot),
                sidechain=float(e_atom[~is_bb].sum() / tot),
                buried=float(e_atom[is_bur].sum() / tot),
                exposed=float(e_atom[~is_bur].sum() / tot),
                sidechain_exposed=float(e_atom[(~is_bb) & (~is_bur)].sum() / tot),
                hydrophobic=float(e_atom[np.array([r in HYDROPHOBIC for r in resn])].sum() / tot))
            share = dict(backbone=float(is_bb.mean()), buried=float(is_bur.mean()))
            byres = collections.defaultdict(float)
            for r_, e_ in zip(resn, e_atom): byres[r_] += float(e_)
            byres = {k: v / tot for k, v in byres.items()}
            cnt = collections.Counter(resn)
            enrich = {k: (byres[k] / max(cnt[k] / len(resn), 1e-9)) for k in byres}
            rows[pdb] = dict(N=d["N"], frac=frac, atom_share=share, by_res=byres, enrich=enrich)
            print(f"  [{i}/{len(ho)}] {pdb:10s} N={d['N']:>6}  backbone {100*frac['backbone']:4.1f}% "
                  f"(of {100*share['backbone']:4.1f}% of atoms)  sidechain {100*frac['sidechain']:4.1f}%"
                  f"  buried {100*frac['buried']:4.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(ho)}] {pdb}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(ho), complete=(i == len(ho)), n_failed=failed,
                          k_anm=K_ANM, buried_r=BURIED_R, buried_q=BURIED_Q)
    if not rows:
        raise SystemExit("\n  NO system scored. Reported as absent, not as a null.")
    print(f"\n=== 100c: RESIDUAL COMPOSITION ACROSS {len(rows)} SYSTEMS ===")
    print(f"  {'class':>20}{'median share of residual':>26}{'IQR':>20}{'median share of atoms':>24}")
    for k in ("backbone", "sidechain", "buried", "exposed", "sidechain_exposed", "hydrophobic"):
        v = np.array([r["frac"][k] for r in rows.values()])
        base = ("backbone" if k == "backbone" else "buried" if k == "buried" else None)
        b = f"{100*np.median([r['atom_share'][base] for r in rows.values()]):.1f}%" if base else ""
        print(f"  {k:>20}{100*np.median(v):>25.1f}%"
              f"{f'[{100*np.percentile(v,25):.1f}, {100*np.percentile(v,75):.1f}]':>20}{b:>24}")
    bb = np.array([r["frac"]["backbone"] for r in rows.values()])
    bs = np.array([r["atom_share"]["backbone"] for r in rows.values()])
    lift = bb / np.clip(bs, 1e-9, None)
    print(f"\n  BACKBONE ENRICHMENT (share of residual / share of atoms): median {np.median(lift):.2f}x")
    print(f"  spread of the backbone share ACROSS systems: sd {100*bb.std():.1f} points "
          f"(range {100*bb.min():.1f}-{100*bb.max():.1f}%)")
    allres = sorted({k for r in rows.values() for k in r["enrich"]})
    med = {k: np.median([r["enrich"].get(k, np.nan) for r in rows.values()]) for k in allres}
    top = sorted(med, key=lambda k: -med[k])[:6]
    print(f"  most enriched residue types (energy share / count share, median across systems):")
    print("    " + "  ".join(f"{k} {med[k]:.2f}x" for k in top))
    conc = np.median([max(r["frac"]["backbone"], r["frac"]["sidechain"]) for r in rows.values()])
    if np.median(lift) > 1.15 or conc > 0.75:
        print(f"\n  -> CONCENTRATED ON A CHEMICAL CLASS. There is shared, size-independent structure\n"
              f"     for a model that sees atom identity, so SEM is motivated on evidence.")
    else:
        print(f"\n  -> SPREAD roughly in proportion to atom counts. No chemical class carries the\n"
              f"     residual, so a shared latent has no size-independent handle on it.")
