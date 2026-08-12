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
from armf_propagator import iat_series
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
    # INBOX 107c. THREE SWEEPS HAVE NOW TRUNCATED ON THE LARGEST SYSTEMS AND THAT IS ONE DEFECT.
    # Ascending order was chosen so a kill truncates VISIBLY, which was right -- but it means every
    # truncation removes the SAME END, and exposed-surface fraction falls with N by surface-to-volume,
    # which is the regressor the enrichment is defined against. A DESCENDING pass truncates the other
    # way, so the union of the two resolves the N trend instead of both passes being small-end.
    ho.sort(key=lambda p: store.meta[have[p]]["atoms"],
            reverse=bool(os.environ.get("RC_DESC")))
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
            # CHUNKED. The pairwise (N,N,3) array is 27 GB at N=33,377 and OOM-killed the run at
            # 118/125 -- the same class of allocation the 099 harness needed a frame-space route
            # for, and it fails on the LARGEST systems, which is an N-correlated exclusion.
            nb = np.zeros(len(ref), np.int64)
            for c0 in range(0, len(ref), 2048):
                c1 = min(c0 + 2048, len(ref))
                dchunk = np.linalg.norm(ref[c0:c1, None, :] - ref[None, :, :], axis=-1)
                nb[c0:c1] = (dchunk < BURIED_R).sum(1) - 1       # coordination number
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
            # INBOX 101e. THE POPULATION SHARE OF EVERY CLASS, not just backbone and buried.
            # Side chains are ~60% of heavy atoms, so "60% of the residual is side-chain" is the
            # NULL, not a finding. The quantity that carries information is ENRICHMENT -- residual
            # share divided by population share -- and without it a flat result reads as
            # concentration.
            is_hyd = np.array([r_ in HYDROPHOBIC for r_ in resn])
            share = dict(backbone=float(is_bb.mean()), sidechain=float((~is_bb).mean()),
                         buried=float(is_bur.mean()), exposed=float((~is_bur).mean()),
                         sidechain_exposed=float(((~is_bb) & (~is_bur)).mean()),
                         hydrophobic=float(is_hyd.mean()))
            byres = collections.defaultdict(float)
            for r_, e_ in zip(resn, e_atom): byres[r_] += float(e_)
            byres = {k: v / tot for k, v in byres.items()}
            cnt = collections.Counter(resn)
            enrich = {k: (byres[k] / max(cnt[k] / len(resn), 1e-9)) for k in byres}
            # INBOX 106d. THE ONE TEST THAT GATES SEM. "92% linearly predictable" is a SPATIAL
            # measure from cross-fit residual PCA -- it says the residual has shared spatial
            # structure and NOTHING about whether that structure carries dynamical information.
            # Exposed side chains are where rotameric flipping and thermal jitter live, and those
            # are fast and close to uncorrelated in time. Family D on the motivating claim rather
            # than on the measurement.
            #   IAT ~ 1 frame  -> jitter; a GENERATIVE DYNAMICS model gains nothing from capturing
            #                     it, and SEM should be argued on the static axis only
            #   IAT ~ the collective modes' -> slow structured motion ANM misses, which is a far
            #                     stronger case for SEM than the enrichment number alone
            # Per atom: the residual's own dominant direction gives a scalar series, so the IAT is
            # of the motion rather than of an arbitrary Cartesian axis.
            iat_cls = {}
            sel = np.arange(0, d["N"], max(1, d["N"] // 400))     # subsample atoms, not frames
            ia = np.full(d["N"], np.nan)
            for j in sel:
                ri = R[:, j, :] - R[:, j, :].mean(0)
                _, _, Vt3 = np.linalg.svd(ri, full_matrices=False)   # ONE decomposition, not two
                ia[j] = iat_series(ri @ Vt3[0])
            m = ~np.isnan(ia)
            for lab, msk in (("backbone", is_bb), ("sidechain", ~is_bb),
                             ("buried", is_bur), ("exposed", ~is_bur),
                             ("sidechain_exposed", (~is_bb) & (~is_bur))):
                mm = m & msk
                iat_cls[lab] = float(np.median(ia[mm])) if mm.any() else float("nan")
            # The COLLECTIVE-MODE reference: IAT of the ANM coefficients on the same frames.
            Cc = (X - d["mu"]) @ V
            iat_cls["collective_modes"] = float(np.median(
                [iat_series(Cc[:, k]) for k in range(min(16, Cc.shape[1]))]))
            iat_cls["n_atoms_scored"] = int(m.sum())
            rows[pdb] = dict(N=d["N"], frac=frac, atom_share=share, by_res=byres, enrich=enrich,
                             iat=iat_cls,
                             res_present={k: int(v) for k, v in cnt.items()})
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
    print(f"  {'class':>20}{'residual share':>16}{'atom share':>13}{'ENRICHMENT':>13}{'IQR of enrich':>18}")
    for k in ("backbone", "sidechain", "buried", "exposed", "sidechain_exposed", "hydrophobic"):
        v = np.array([r["frac"][k] for r in rows.values()])
        a_ = np.array([r["atom_share"][k] for r in rows.values() if k in r["atom_share"]])
        if len(a_) != len(v):
            print(f"  {k:>20}{100*np.median(v):>15.1f}%{'-':>13}{'-':>13}{'-':>18}")
            continue
        en = v / np.clip(a_, 1e-9, None)
        print(f"  {k:>20}{100*np.median(v):>15.1f}%{100*np.median(a_):>12.1f}%{np.median(en):>13.2f}x"
              f"{f'[{np.percentile(en,25):.2f}, {np.percentile(en,75):.2f}]':>18}")
    print(f"  ENRICHMENT is the column that carries information: 1.00x means the class holds exactly\n"
          f"  its population share of the residual, which is the null. 101e.")
    bb = np.array([r["frac"]["backbone"] for r in rows.values()])
    bs = np.array([r["atom_share"]["backbone"] for r in rows.values()])
    lift = bb / np.clip(bs, 1e-9, None)
    print(f"\n  BACKBONE ENRICHMENT (share of residual / share of atoms): median {np.median(lift):.2f}x")
    print(f"  spread of the backbone share ACROSS systems: sd {100*bb.std():.1f} points "
          f"(range {100*bb.min():.1f}-{100*bb.max():.1f}%)")
    # 106e: a plain median over NaN printed nan wherever ANY system lacked a residue type. The
    # count of systems carrying each type is reported BESIDE the enrichment rather than the
    # absent systems being silently dropped -- a printed nan beats a nan a nanmedian absorbs.
    allres = sorted({k for r in rows.values() for k in r["enrich"]})
    med, npres = {}, {}
    for k in allres:
        v = [r["enrich"][k] for r in rows.values() if k in r["enrich"]]
        med[k] = float(np.median(v)) if v else float("nan")
        npres[k] = len(v)
    top = sorted([k for k in allres if npres[k] >= 0.5 * len(rows)], key=lambda k: -med[k])[:6]
    print(f"  most enriched residue types (energy share / count share), median across the systems")
    print(f"  that CONTAIN the type, with that count:")
    print("    " + "  ".join(f"{k} {med[k]:.2f}x (n={npres[k]})" for k in top))
    rare = [k for k in allres if npres[k] < 0.5 * len(rows)]
    if rare:
        print(f"    excluded as present in <50% of systems: "
              + ", ".join(f"{k} (n={npres[k]})" for k in sorted(rare, key=lambda k: npres[k])))

    print(f"\n=== 106d: IS THE RESIDUAL SLOW, OR IS IT JITTER? (integrated autocorrelation time) ===")
    print(f"  {'class':>20}{'median IAT (frames)':>22}{'IQR':>18}")
    for k in ("backbone", "sidechain", "buried", "exposed", "sidechain_exposed",
              "collective_modes"):
        v = np.array([r["iat"][k] for r in rows.values() if k in r.get("iat", {})], float)
        v = v[~np.isnan(v)]
        if not len(v): continue
        print(f"  {k:>20}{np.median(v):>22.2f}"
              f"{f'[{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}]':>18}")
    se = np.array([r["iat"]["sidechain_exposed"] for r in rows.values() if "iat" in r], float)
    cm = np.array([r["iat"]["collective_modes"] for r in rows.values() if "iat" in r], float)
    ok = ~np.isnan(se) & ~np.isnan(cm)
    if ok.any():
        ratio = np.median(se[ok] / np.clip(cm[ok], 1e-9, None))
        print(f"\n  exposed-side-chain residual IAT / collective-mode IAT: median {ratio:.3f}")
        if np.median(se[ok]) < 2.0:
            print(f"  -> IAT ~ {np.median(se[ok]):.2f} frames: THIS IS JITTER. A generative dynamics\n"
                  f"     model gains nothing from capturing it, and the 1.54x enrichment motivates\n"
                  f"     SEM on the STATIC axis only.")
        elif ratio > 0.3:
            print(f"  -> the residual is SLOW, comparable to the collective modes. That is structured\n"
                  f"     motion ANM misses, and a far stronger case for SEM than enrichment alone.")
        else:
            print(f"  -> the residual is FASTER than the collective modes by {1/ratio:.1f}x but not\n"
                  f"     jitter. Intermediate: reported as such rather than pushed to either branch.")
    conc = np.median([max(r["frac"]["backbone"], r["frac"]["sidechain"]) for r in rows.values()])
    if np.median(lift) > 1.15 or conc > 0.75:
        print(f"\n  -> CONCENTRATED ON A CHEMICAL CLASS. There is shared, size-independent structure\n"
              f"     for a model that sees atom identity, so SEM is motivated on evidence.")
    else:
        print(f"\n  -> SPREAD roughly in proportion to atom counts. No chemical class carries the\n"
              f"     residual, so a shared latent has no size-independent handle on it.")
