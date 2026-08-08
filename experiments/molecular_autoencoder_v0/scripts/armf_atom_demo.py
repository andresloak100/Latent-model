#!/usr/bin/env python3
"""INBOX 045/046: an atom-to-atom demo of the DIRECT per-residue codec.

WHAT THIS IS. Atoms in -> latent -> atoms out, on held-out structures, with the geometry metrics that
say whether the reconstruction is chemically real rather than merely close: all-atom and backbone
RMSD, chirality violation, contact F1, and the latent's actual size in scalars.

WHAT IT IS NOT, and why that has to be structural rather than a caption. This codec's latent SCALES
WITH THE STRUCTURE -- 8 floats per residue, so ~1,600 scalars for a 200-residue protein. It is NOT
the fixed-size compression the project is about, and it makes no claim about dynamics. The fixed-size
line loses to zero-shot ANM on 100% of 123 ATLAS systems.

Three times in three days a number correct in its own context has been joined to one where it meant
something different (41a tied-vs-untied, 42b ladder-vs-14a, 46a single-chain-vs-complex). Each time
the scope was RECONSTRUCTABLE rather than ATTACHED. So this script does not accept a caption. Every
number it emits carries provenance GENERATED from the run that produced it -- arm, encoder type,
splits file, processed dir, checkpoint identity, held-out count -- and the report groups by that
provenance and refuses to put two groups in one table. A hand-written "this is not the fixed-size
codec" is correct the day it is written and drifts the first time the table is copied. A generated
`direct / single-chain / heldout n=...` beside each number cannot.

Usage (one invocation per ARM -- never merge arms in one run):
    python scripts/armf_atom_demo.py --config <cfg> --ckpt <ckpt> --out <dir> --arm single-chain
    python scripts/armf_atom_demo.py --config <cfg> --ckpt <ckpt> --out <dir> --arm complex
    python scripts/armf_atom_demo.py --report <dir>        # build REPORT.md from whatever ran
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig            # noqa: E402
from molae.dataset import ProteinStructureDataset, collate_fn   # noqa: E402
from molae.model import MolecularAutoencoder         # noqa: E402,F401
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.metrics import build_topology_info, compute_all_metrics  # noqa: E402
from molae.pdb_io import write_pdb                   # noqa: E402
from molae import utils                              # noqa: E402


# --------------------------------------------------------------------------------------- provenance
def _git_sha():
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip() != ""
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _file_id(p: Path):
    """Identity of the weights, not just their path. Two runs pointing at the same filename after a
    retrain are not the same run, and 043 was exactly a stale file sitting at an expected path."""
    if not p.exists():
        return {"path": str(p), "present": False}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "present": True, "bytes": p.stat().st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)),
            "sha256": h.hexdigest()[:16]}


def provenance(cfg, cfg_path, ckpt_path, model, arm, n_heldout):
    """The scope label, DERIVED. Nothing here is passed in by hand except `arm`, and `arm` only names
    the invocation -- every field that determines what the numbers mean is read from the run."""
    return {
        "arm": arm,
        "encoder_type": getattr(cfg.model, "encoder_type", "?"),
        "model_class": type(model).__name__,
        "latent_scaling": _latent_scaling_kind(model),
        "splits_file": str(getattr(cfg.data, "splits_file", "?")),
        "processed_dir": str(getattr(cfg.data, "processed_dir", "?")),
        "n_heldout_evaluated": n_heldout,
        "config": str(cfg_path),
        "checkpoint": _file_id(Path(ckpt_path)),
        "git": _git_sha(),
        "torch": torch.__version__,
    }


def provenance_key(pv):
    """What makes two runs comparable. Differ on ANY of these and they do not belong in one table --
    this is the check that makes 46a structurally impossible rather than a thing to remember."""
    return (pv["arm"], pv["encoder_type"], pv["splits_file"], pv["processed_dir"])


def _latent_scaling_kind(model):
    if hasattr(model, "latent_floats_for_atoms"):
        return "per-atom (scales with atom count)"
    if hasattr(model, "latent_floats_for"):
        return "per-residue (scales with residue count)"
    return "fixed (independent of structure size)"


def latent_floats_for(model, n_atoms, n_res, default):
    """Mirrors scripts/eval.py exactly, including its warning: per-atom latents scale with atoms,
    per-residue with residues, fixed bottlenecks with neither, and getting it wrong silently inflates
    the reported compression ratio -- it once did, by ~60x. Kept in the same three-branch order so the
    two cannot diverge; do not 'simplify' this."""
    if hasattr(model, "latent_floats_for_atoms"):
        return int(model.latent_floats_for_atoms(int(n_atoms)))
    if hasattr(model, "latent_floats_for"):
        return int(model.latent_floats_for(int(n_res)))
    return int(default)


def load_with_posenc_shim(ckpt_path: Path, model, device):
    """molae/scaling.py:73 promises "every existing checkpoint loads and every prior result
    reproduces". It does not: PositionEncoding wrapped a bare nn.Embedding as `self.table`, so a
    checkpoint holding `...res_pos_emb.weight` cannot load into `...res_pos_emb.table.weight`. Every
    direct checkpoint predating that refactor is affected, including the one behind the 0.79 A result.

    The remap is EXACT for any structure within `max_positions`: same tensor, same lookup. Beyond it
    the new path CLAMPS where the old one would have raised, so a silently-clamped structure would be
    a different computation wearing the old number's label. That condition is asserted per structure
    in run_arm() rather than assumed, and the fact that a shim was used is recorded in provenance --
    a caveat that only exists in a commit message is one the report cannot show."""
    # weights_only=False matches molae/utils.load_checkpoint; these are our own files under $SCRATCH.
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    want = set(model.state_dict().keys())
    fixed, moved = {}, []
    for k, v in sd.items():
        if k not in want and k.endswith(".weight"):
            cand = k[: -len(".weight")] + ".table.weight"
            if cand in want:
                fixed[cand] = v; moved.append((k, cand)); continue
        fixed[k] = v
    # Reuse the project's own vocabulary-extension path rather than a second one of my own. But note
    # what it does: it RANDOMLY INITIALISES the new rows. Any structure that actually indexes into
    # them is being reconstructed with untrained embeddings, so its number is not this checkpoint's
    # result. The grown extents are returned so run_arm() can refuse such a structure outright.
    grown = {}
    for k_, old_, new_ in (utils.grow_embedding_rows(fixed, model) or []):
        print(f"  [ckpt] grew {k_} {old_} -> {new_} (new rows RANDOM, not trained)", flush=True)
        grown[k_] = int(old_[0] if isinstance(old_, (tuple, list)) else old_)
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint still does not fit after shim: missing={list(missing)[:4]} "
                         f"unexpected={list(unexpected)[:4]}")
    for a, b in moved:
        print(f"  [shim] {a} -> {b}", flush=True)
    return [f"{a} -> {b}" for a, b in moved], grown


def posenc_limit(model):
    """Smallest max_positions over any PositionEncoding in the model, or None."""
    lims = [m.max_positions for m in model.modules()
            if type(m).__name__ == "PositionEncoding" and getattr(m, "table", None) is not None]
    return min(lims) if lims else None


# ------------------------------------------------------------------------------------ conformers
def spread(coords_list):
    """Mean pairwise RMSD across an ensemble, after superposition. The quantity that must not collapse
    through the round trip."""
    from molae.alignment import kabsch_rmsd_numpy
    n = len(coords_list)
    if n < 2:
        return float("nan")
    vals = [kabsch_rmsd_numpy(coords_list[i], coords_list[j])
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


# ------------------------------------------------------------------------------------------- run
def run_arm(args):
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(args.device)
    utils.set_seed(getattr(cfg.train, "seed", 0))

    splits = utils.load_json(ROOT / cfg.data.splits_file)
    keys = list(splits.get("val") or splits.get("train") or [])
    processed = ROOT / cfg.data.processed_dir
    paths = [processed / f"{k}.npz" for k in keys if (processed / f"{k}.npz").exists()]
    if not paths:
        raise SystemExit(f"no held-out structures found under {processed}")
    ds = ProteinStructureDataset(paths)

    model = make_autoencoder(cfg.model).to(device)
    remapped, grown = load_with_posenc_shim(Path(args.ckpt), model, device)
    model.eval()

    # Span small -> large by atom count, so the table is not silently a small-structure table. 28a
    # caught a comparison made against an all-N median; picking the easy end is the same error.
    sizes = [(i, int(ds[i]["n_atoms"])) for i in range(len(ds))]
    sizes.sort(key=lambda t: t[1])
    pick = sorted({int(round(x)) for x in np.linspace(0, len(sizes) - 1, min(args.n, len(sizes)))})
    chosen = [sizes[i][0] for i in pick]

    out = Path(args.out); (out / "structures").mkdir(parents=True, exist_ok=True)
    lim = posenc_limit(model)
    pv = provenance(cfg, args.config, args.ckpt, model, args.arm, len(ds))
    pv["checkpoint_key_remap"] = remapped or "none"
    pv["posenc_max_positions"] = lim
    pv["vocab_grown"] = {k_: v_ for k_, v_ in grown.items()} or "none"
    print(f"[demo] arm={pv['arm']}  encoder={pv['encoder_type']}  latent={pv['latent_scaling']}")
    print(f"[demo] held-out pool {len(ds)}; evaluating {len(chosen)} spanning "
          f"{sizes[pick[0]][1]}-{sizes[pick[-1]][1]} atoms", flush=True)

    # INBOX 47a. The demo's structures are chosen to SPAN the size range, not drawn at random, so
    # their median is NOT an estimate of the set's median -- it over-weights both extremes and the
    # small end reconstructs worst. The headline must therefore be the FULL held-out distribution,
    # with these as illustrations beneath it. Read from the run's own metrics.json when present, so
    # the headline is the recorded evaluation rather than a re-derivation of it.
    full = None
    mpath = Path(args.ckpt).parent / "metrics.json"
    if mpath.exists():
        try:
            md = json.loads(mpath.read_text())
            ps = md.get("per_structure") or md.get("structures") or []
            aa = np.array([m["all_atom_rmsd"] for m in ps], float)
            nr = np.array([m.get("n_residues", np.nan) for m in ps], float)
            if aa.size:
                full = {"n": int(aa.size), "mean": round(float(aa.mean()), 4),
                        "median": round(float(np.median(aa)), 4),
                        "sd": round(float(aa.std(ddof=1)), 4),
                        "min": round(float(aa.min()), 3), "max": round(float(aa.max()), 3),
                        "q25": round(float(np.percentile(aa, 25)), 3),
                        "q75": round(float(np.percentile(aa, 75)), 3),
                        "frac_above_2A": round(float((aa > 2).mean()), 4),
                        "n_above_2A": int((aa > 2).sum()),
                        "residues_min": None if not np.isfinite(nr).any() else int(np.nanmin(nr)),
                        "residues_max": None if not np.isfinite(nr).any() else int(np.nanmax(nr)),
                        "source": str(mpath)}
                # INBOX 49c: one median over what may be two regimes describes neither. Band by
                # residue count and report where the rolling median crosses 2 A and 5 A. Computed
                # from the recorded full evaluation, so this is a summary and not a new inference.
                cl_ = np.array([m.get("clashes_per_1000_atoms", np.nan) for m in ps], float)
                okm = np.isfinite(nr) & np.isfinite(aa)
                if okm.sum() > 8:
                    nr_, aa_, cl2 = nr[okm], aa[okm], cl_[okm]
                    order = np.argsort(nr_); nr_, aa_, cl2 = nr_[order], aa_[order], cl2[order]
                    bands, edges = [], [0, 50, 100, 150, 200, 300, 500, 10 ** 9]
                    for a_, b_ in zip(edges[:-1], edges[1:]):
                        mb = (nr_ >= a_) & (nr_ < b_)
                        if mb.sum():
                            bands.append({"lo": int(a_), "hi": int(min(b_, nr_.max())),
                                          "n": int(mb.sum()),
                                          "median_rmsd": round(float(np.median(aa_[mb])), 2),
                                          "median_clashes": (None if not np.isfinite(cl2[mb]).any()
                                                             else round(float(np.nanmedian(cl2[mb])), 1))})
                    def _cross(th):
                        for i_ in range(len(nr_)):
                            if np.median(aa_[i_:]) > th:
                                return int(nr_[i_])
                        return None
                    full["bands"] = bands
                    full["cross_2A"] = _cross(2.0)
                    full["cross_5A"] = _cross(5.0)
                    full["spearman_res_rmsd"] = round(float(stats.spearmanr(nr_, aa_).statistic), 3)
                print(f"[demo] full held-out set (recorded): n={full['n']} mean {full['mean']} "
                      f"median {full['median']} sd {full['sd']}", flush=True)
        except Exception as e:
            print(f"[demo] could not read {mpath}: {type(e).__name__}: {e}", flush=True)

    # INBOX 47b. The clamp guard excludes on RESIDUE COUNT, the same axis the metric varies along, so
    # the excluded set is not random with respect to what is reported -- Family A in a demo, where
    # there are no error bars to be suspicious of. Count it and print it whether or not it is zero:
    # "none were skipped" closes the question permanently and costs one line.
    skipped = []
    if lim is not None:
        for i in range(len(ds)):
            nr_i = int(ds[i]["res_pos"].numpy().max()) + 1
            if nr_i > lim:
                skipped.append((ds[i]["pdb_id"], nr_i))
    print(f"[demo] 47b exclusion audit: {len(skipped)} of {len(ds)} held-out structures exceed "
          f"max_positions={lim}"
          + (f" (residues {min(x[1] for x in skipped)}-{max(x[1] for x in skipped)})" if skipped
             else " -- the exclusion is EMPTY, so it cannot bias the reported numbers"), flush=True)

    rows = []
    with torch.no_grad():
        for idx in chosen:
            s = ds[idx]
            gb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
            t0 = time.time()
            preds, _z = model(gb)
            dt = time.time() - t0
            na = int(s["n_atoms"])
            pred = preds[0, :na].cpu().numpy().astype(np.float64)
            targ = gb["coords"][0, :na].cpu().numpy().astype(np.float64)
            topo = build_topology_info(s["atom_name_idx"].numpy(), s["res_pos"].numpy(),
                                       s["bonds"].numpy(), s["element_symbol"])
            n_res = int(s["res_pos"].numpy().max()) + 1
            # A structure indexing into a randomly-initialised row is not this checkpoint's result.
            for fld, key in (("element_idx", "encoder.feat.elem_emb.weight"),
                             ("residue_idx", "encoder.feat.res_emb.weight")):
                cap = grown.get(key)
                if cap is not None and fld in s and int(s[fld].max()) >= cap:
                    raise SystemExit(
                        f"{s['pdb_id']}: {fld} max {int(s[fld].max())} indexes into rows the "
                        f"checkpoint never trained (original vocabulary {cap}). Those embeddings are "
                        f"randomly initialised, so this structure cannot be reported under this "
                        f"checkpoint's label.")
            if lim is not None and n_res > lim:
                raise SystemExit(
                    f"{s['pdb_id']}: {n_res} residues exceeds PositionEncoding max_positions={lim}. "
                    f"The current code would CLAMP here where the checkpoint's original code would "
                    f"not, so this is a different computation and must not be reported under the "
                    f"old result's label.")
            lf = latent_floats_for(model, na, n_res, getattr(model, "latent_floats", 0))
            m = compute_all_metrics(pred, targ, topo, latent_floats=lf)
            m.update(pdb_id=s["pdb_id"], n_atoms=na, n_residues=n_res,
                     latent_floats=lf, floats_per_residue=round(lf / max(n_res, 1), 3),
                     compression_x=round((na * 3) / max(lf, 1), 2), infer_s=round(dt, 3))
            rows.append(m)

            stem = out / "structures" / f"{s['pdb_id']}"
            rs = s["res_seq"].numpy() if "res_seq" in s else np.arange(na)
            write_pdb(f"{stem}_true.pdb", targ, s["residue_idx"].numpy(),
                      s["atom_name_idx"].numpy(), rs, s["element_symbol"], chain_id="A")
            write_pdb(f"{stem}_recon.pdb", pred, s["residue_idx"].numpy(),
                      s["atom_name_idx"].numpy(), rs, s["element_symbol"], chain_id="A")
            # per-atom error, so a viewer can colour by it rather than trusting one scalar
            np.save(f"{stem}_per_atom_error.npy",
                    np.linalg.norm(pred - targ, axis=-1).astype(np.float32))
            print(f"  {s['pdb_id']:>8}  {na:>6} atoms  {n_res:>4} res  "
                  f"all-atom {m['all_atom_rmsd']:.3f} A  backbone {m['backbone_rmsd']:.3f} A  "
                  f"latent {lf} floats ({m['compression_x']}x)", flush=True)

    payload = {"provenance": pv, "structures": rows, "full_heldout": full,
               "excluded_over_max_positions": [{"pdb_id": a_, "n_residues": b_} for a_, b_ in skipped]}

    # --- conformational spread through the round trip, if an ensemble was named ---
    if args.ensemble:
        ens = [k for k in args.ensemble.split(",") if (processed / f"{k}.npz").exists()]
        if len(ens) >= 2:
            eds = ProteinStructureDataset([processed / f"{k}.npz" for k in ens])
            tr, rc = [], []
            with torch.no_grad():
                for i in range(len(eds)):
                    s = eds[i]
                    gb = {k: (v.to(device) if torch.is_tensor(v) else v)
                          for k, v in collate_fn([s]).items()}
                    p, _ = model(gb)
                    na = int(s["n_atoms"])
                    tr.append(gb["coords"][0, :na].cpu().numpy().astype(np.float64))
                    rc.append(p[0, :na].cpu().numpy().astype(np.float64))
            st, sr = spread(tr), spread(rc)
            payload["conformers"] = {
                "members": ens, "spread_true_A": round(st, 4), "spread_recon_A": round(sr, 4),
                "ratio": round(sr / st, 4) if st else None,
                "spread_lost_pct": round(100 * (1 - sr / st), 1) if st else None,
            }
            print(f"  ensemble {len(ens)}: spread {st:.3f} -> {sr:.3f} A, "
                  f"{100*(1-sr/st):.1f}% lost", flush=True)
        else:
            payload["conformers"] = {"members": ens, "note": "fewer than 2 members present"}

    (out / f"run_{args.arm}.json").write_text(json.dumps(payload, indent=2))
    print(f"[demo] wrote {out}/run_{args.arm}.json", flush=True)


# ---------------------------------------------------------------------------------------- report
def build_report(out: Path):
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    if not runs:
        raise SystemExit(f"no run_*.json under {out}")
    groups = {}
    for r in runs:
        groups.setdefault(provenance_key(r["provenance"]), []).append(r)

    L = ["# Atom-to-atom reconstruction — direct per-residue codec", ""]
    L += [f"Generated {time.strftime('%Y-%m-%d %H:%M')} · every number below carries the provenance of "
          f"the run that produced it.", ""]
    if len(groups) > 1:
        L += ["> **These are different arms and are reported at different scales.** They are in "
              "separate sections", "> because they are not comparable; a number from one section does "
              "not describe the other.", ""]

    for key, rs in groups.items():
        for r in rs:
            pv, rows = r["provenance"], r["structures"]
            ck = pv["checkpoint"]
            L += [f"## Arm: `{pv['arm']}`", ""]
            L += ["| provenance | |", "|---|---|",
                  f"| encoder | `{pv['encoder_type']}` ({pv['model_class']}) |",
                  f"| latent | **{pv['latent_scaling']}** |",
                  f"| splits | `{pv['splits_file']}` |",
                  f"| data | `{pv['processed_dir']}` |",
                  f"| held-out pool | {pv['n_heldout_evaluated']} structures |",
                  f"| checkpoint | `{Path(ck['path']).name}` sha256 `{ck.get('sha256','?')}` "
                  f"({ck.get('mtime','?')}) |",
                  f"| code | git `{pv['git']}`, torch {pv['torch']} |"]
            rm = pv.get("checkpoint_key_remap", "none")
            if rm != "none":
                L += [f"| checkpoint shim | **{len(rm)} key(s) remapped** to load under current code "
                      f"(`{rm[0]}`); exact for structures within max_positions="
                      f"{pv.get('posenc_max_positions')}, which is asserted per structure |"]
            vg = pv.get("vocab_grown", "none")
            if vg != "none":
                L += [f"| vocabulary | **{len(vg)} embedding(s) extended** since training; new rows "
                      f"are randomly initialised, so any structure indexing them is refused rather "
                      f"than reported |"]
            L += [""]
            # INBOX 49b: clashes/1000 atoms is computed and stored and was in NEITHER table, and it
            # is the column that says whether a structure is physically usable. chirality 0.0000 sat
            # on every row including one at 15.5 A, so a reader saw a stereochemistry check passing
            # and reasonably inferred sound geometry. Chirality survives the bottleneck; validity
            # does not.
            L += ["| structure | atoms | res | all-atom Å | backbone Å | chirality | clashes/1k | "
                  "contact F1 | latent floats | ×compression |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for m in rows:
                L.append(f"| `{m['pdb_id']}` | {m['n_atoms']} | {m['n_residues']} | "
                         f"**{m['all_atom_rmsd']:.2f}** | {m['backbone_rmsd']:.2f} | "
                         f"{m.get('chirality_violation_rate', float('nan')):.4f} | "
                         f"{m.get('clashes_per_1000_atoms', float('nan')):,.1f} | "
                         f"{m.get('contact_f1', float('nan')):.3f} | {m['latent_floats']} | "
                         f"{m['compression_x']}× |")
            aa = [m["all_atom_rmsd"] for m in rows]
            fu = r.get("full_heldout")
            if fu:
                L += ["", f"### Headline — the full held-out set (n={fu['n']})", "",
                      f"| | all-atom Å |", "|---|---|",
                      f"| mean | **{fu['mean']:.2f}** |",
                      f"| median | **{fu['median']:.2f}** |",
                      f"| sd | {fu['sd']:.2f} |",
                      f"| quartiles | {fu['q25']:.2f} / {fu['median']:.2f} / {fu['q75']:.2f} |",
                      f"| range | {fu['min']:.2f} – {fu['max']:.2f} |",
                      f"| above 2 Å | {fu['n_above_2A']} of {fu['n']} ({100*fu['frac_above_2A']:.1f}%) |",
                      ""]
                if fu.get("residues_min") is not None:
                    L += [f"That set is **{fu['residues_min']}–{fu['residues_max']} residues**. The "
                          f"headline describes structures of that size, not proteins in general.", ""]
                # INBOX 49a: this sentence was boilerplate emitted per arm, true for single-chain
                # (0.79 < 0.84) and FALSE for complex (5.88 vs 2.49). It asserted "left-skewed, mean
                # below median" while its own table two lines up said otherwise, and contradicted the
                # Scope block at the foot. Derive the direction from the two numbers; the consequence
                # for a reader flips with it.
                below = fu["mean"] < fu["median"]
                L += [f"The commonly quoted **{fu['mean']:.2f} Å is the mean**; the median is "
                      f"**{fu['median']:.2f} Å**. The mean sits "
                      + (f"**below** the median, so the distribution is **left-skewed** and quoting "
                         f"the mean alone **understates** the typical error."
                         if below else
                         f"**{fu['mean']/max(fu['median'],1e-9):.1f}× the median**, so the "
                         f"distribution is **right-skewed** with a long tail and quoting the mean "
                         f"alone **overstates** the typical error.") + " Quote both.", ""]
                bd = fu.get("bands")
                if not bd and fu.get("source") and os.path.exists(fu["source"]):
                    try:
                        _m = json.load(open(fu["source"]))
                        _per = _m.get("per_structure") or _m.get("structures") or []
                        _rr = [(x["n_residues"], x["all_atom_rmsd"]) for x in _per
                               if x.get("n_residues") and x.get("all_atom_rmsd") is not None]
                        if _rr:
                            _R = np.array([v[0] for v in _rr]); _A = np.array([v[1] for v in _rr])
                            bd = []
                            for _lo, _hi in ((0, 150), (150, 300), (300, 10 ** 6)):
                                _sel = (_R >= _lo) & (_R < _hi)
                                if _sel.sum():
                                    bd.append({"label": f"{_lo}-{_hi if _hi < 10**6 else '+'}",
                                               "n": int(_sel.sum()),
                                               "median_rmsd": float(np.median(_A[_sel])),
                                               "median_clashes": None})
                            _o = np.argsort(_R); _Rs, _As = _R[_o], _A[_o]
                            _cross = {}
                            for _t in (2.0, 5.0):
                                _c = None
                                for _i in range(len(_Rs)):
                                    _l, _h = max(0, _i - 7), min(len(_Rs), _i + 8)
                                    if np.median(_As[_l:_h]) > _t: _c = int(_Rs[_i]); break
                                _cross[_t] = _c
                            L += ["", f"**Size trend (49c), from {len(_rr)} held-out structures in the "
                                      f"source metrics -- no re-run.** all-atom spans "
                                      f"{_A.min():.2f}-{_A.max():.2f} A over {_R.min()}-{_R.max()} "
                                      f"residues. The running median crosses **2 A at ~"
                                      f"{_cross[2.0]} residues** and **5 A at ~{_cross[5.0]} "
                                      f"residues**.", ""]
                            if _A.min() > 2.0:
                                L += [f"Note the minimum is **{_A.min():.2f} A**: this is not a tight "
                                      f"cluster that reconstructs plus a tail that does not -- "
                                      f"**nothing in this arm is below 2 A**, and the trend is "
                                      f"monotone degradation from already-poor.", ""]
                    except Exception as _e:
                        L += ["", f"*(49c size trend unavailable: {type(_e).__name__})*", ""]
                if bd:
                    L += ["", f"**Size regimes (49c).** One median over the whole set describes "
                              f"neither end of it:", "",
                          "| residues | n | median all-atom Å | median clashes/1k |",
                          "|---|---|---|---|"]
                    for b_ in bd:
                        L.append(f"| {b_['lo']}–{b_['hi']} | {b_['n']} | {b_['median_rmsd']:.2f} | "
                                 + (f"{b_['median_clashes']:,.0f} |" if b_["median_clashes"] is not None
                                    else "— |"))
                    # A threshold that is never crossed is a RESULT, not missing data. Rendering
                    # None into the sentence printed "crosses 2 A at >=None residues" on the
                    # single-chain arm -- which never crosses 2 A, the strongest thing that arm has to
                    # say -- and read as a broken field. Same class as 49a: the sentence has to follow
                    # the number instead of assuming one exists.
                    c2, c5 = fu.get("cross_2A"), fu.get("cross_5A")
                    def _cross(v, thr):
                        return (f"crosses **{thr} Å** at ≥{v} residues" if v is not None else
                                f"**never crosses {thr} Å** across {fu['residues_min']}–"
                                f"{fu['residues_max']} residues")
                    L += ["", f"Rolling median {_cross(c2, 2)}; {_cross(c5, 5)}. "
                              f"Spearman(residues, RMSD) = "
                              f"**{fu.get('spearman_res_rmsd')}**.", ""]
                L += [f"The {len(aa)} structures below are chosen to **span the size range**, not "
                      f"drawn at random, so their median ({np.median(aa):.2f} Å) is an illustration "
                      f"and not an estimate of the set's.", ""]
            else:
                L += ["", f"Median all-atom **{np.median(aa):.2f} Å**, mean {np.mean(aa):.2f} Å, "
                          f"worst {max(aa):.2f} Å over {len(aa)} structures "
                          f"({sum(1 for v in aa if v > 2.0)} above 2 Å).", ""]
            ex = r.get("excluded_over_max_positions", [])
            L += [f"**Exclusion audit (47b):** {len(ex)} structure(s) skipped for exceeding "
                  f"`max_positions`"
                  + ("" if not ex else f" — residues {min(e['n_residues'] for e in ex)}–"
                                       f"{max(e['n_residues'] for e in ex)}")
                  + (". The exclusion is empty, so it cannot bias these numbers." if not ex
                     else ". This exclusion is on the size axis and is **not** random with respect to "
                          "the reported metric."), ""]
            L += [f"Latent is **{pv['latent_scaling']}** — "
                  f"{rows[0]['floats_per_residue']} floats per residue, so it grows with the "
                  f"structure. This is **not** the fixed-size codec.", ""]
            c = r.get("conformers")
            if c and c.get("ratio") is not None:
                L += ["### Conformers", "",
                      f"Ensemble of {len(c['members'])}: spread **{c['spread_true_A']} Å → "
                      f"{c['spread_recon_A']} Å**, ratio {c['ratio']} — "
                      f"**{c['spread_lost_pct']}% of the spread is lost** through the round trip. "
                      f"Conformers do not collapse; they are not preserved intact either.", ""]
    # The NMR result comes from latent_suitability.py, the canonical script for it -- reimplementing
    # ensemble parsing here would be a second implementation of a measured quantity. Ingest its JSON
    # instead, and carry the entry/conformer COUNT beside the ratio: the ratio is sample-dependent
    # (0.831 on 3 entries vs 0.9324 on 11), and a ratio without its n is how the 3-entry read got
    # published as a finding.
    nm = sorted(out.glob("nmr_suitability*.json"))
    if nm:
        best, bd = None, -1
        for q in nm:
            try:
                dd = json.loads(q.read_text())
                if int(dd.get("n_entries", 0)) > bd:
                    bd, best = int(dd.get("n_entries", 0)), (dd, q)
            except Exception:
                pass
        if best:
            dd, q = best
            st_, sr_ = dd["mean_spread_true_conformers"], dd["mean_spread_reconstructions"]
            rr, re_ = dd["mean_resolution_ratio"], dd["mean_recon_rmsd"]
            L += ["---", "", "## Conformers (NMR ensembles)", "",
                  f"Source `{q.name}` — **{dd['n_entries']} entries, {dd['n_conformers']} "
                  f"conformers**.", "",
                  "| | |", "|---|---|",
                  f"| true conformational spread | **{st_:.3f} Å** |",
                  f"| spread after the round trip | **{sr_:.3f} Å** |",
                  f"| ratio | **{rr:.3f}** — {100*(1-rr):.1f}% of the spread is lost |",
                  f"| mean reconstruction error | {re_:.3f} Å |", "",
                  f"Conformers **do not collapse**: {100*(1-rr):.1f}% of the spread is lost, not all "
                  f"of it. And the reconstruction error ({re_:.2f} Å) is "
                  + (f"**well below** the true spread ({st_:.2f} Å), so the codec resolves "
                     f"conformers rather than blurring them together."
                     if re_ < st_ else
                     f"**larger** than the true spread ({st_:.2f} Å) — on this sample the codec "
                     f"cannot tell conformers apart, which is the condition that matters more than "
                     f"the ratio.") , "",
                  f"The ratio is **sample-dependent** — it is {rr:.3f} here and 0.868 in §5 on a "
                  f"different entry count — so it is quoted with its n, never alone.", ""]
    L += ["---", "", "## Scope", "",
          "- The latent **scales with residue count**. This is not the fixed-size compression the "
          "project is about.",
          "- No claim is made about **dynamics**. On the ATLAS line the fixed-size codec loses to "
          "zero-shot ANM on **100% of 123 systems**.",
          "- Numbers from different arms appear in **separate sections at their own scales** and are "
          "not comparable.",
          "- Both headline means hide their distributions: the single-chain mean sits *below* its "
          "median (left skew), the complex mean is *2.4x* its median (long right tail). Medians and "
          "ranges are given above; quote both.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    print(f"[demo] wrote {out}/REPORT.md  ({len(groups)} provenance group(s), {len(runs)} run(s))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--ckpt")
    ap.add_argument("--out", default=None); ap.add_argument("--arm", default="single-chain")
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--device", default="cpu")
    ap.add_argument("--ensemble", default="")
    ap.add_argument("--report", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.report:
        return build_report(Path(a.report))
    if not (a.config and a.ckpt and a.out):
        raise SystemExit("need --config, --ckpt and --out (or --report DIR, or --selftest)")
    run_arm(a)


def selftest():
    """Exercises the parts that do not need a GPU or the dataset: the provenance key -- which is the
    guard that makes 46a impossible -- and the report builder's refusal to merge groups."""
    import tempfile
    a = {"arm": "single-chain", "encoder_type": "direct", "splits_file": "s_small.json",
         "processed_dir": "processed_small", "model_class": "M", "latent_scaling": "per-residue",
         "n_heldout_evaluated": 10, "config": "c", "checkpoint": {"path": "x", "sha256": "ab"},
         "git": "g", "torch": "t"}
    b = dict(a, arm="complex", processed_dir="processed_complex", splits_file="s_complex.json")
    assert provenance_key(a) != provenance_key(b), "different arms must not share a key"
    assert provenance_key(a) == provenance_key(dict(a)), "same run must share a key"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        mk = lambda pv, rmsd: {"provenance": pv, "structures": [
            {"pdb_id": "T", "n_atoms": 100, "n_residues": 12, "all_atom_rmsd": rmsd,
             "backbone_rmsd": rmsd * 0.7, "chirality_violation_rate": 0.0002, "contact_f1": 0.96,
             "latent_floats": 96, "floats_per_residue": 8.0, "compression_x": 3.1}]}
        (d / "run_single-chain.json").write_text(json.dumps(mk(a, 0.79)))
        (d / "run_complex.json").write_text(json.dumps(mk(b, 5.7)))
        build_report(d)
        txt = (d / "REPORT.md").read_text()
        assert "different arms and are reported at different scales" in txt, "missing separation banner"
        assert txt.count("## Arm:") == 2, "arms must get their own sections"
        # Order is alphabetical and arbitrary; asserting it was MY assumption, not a property of the
        # report. The property that matters is that each section contains only its own numbers.
        secs = {}
        for chunk in txt.split("## Arm: ")[1:]:
            secs[chunk.split("`")[1]] = chunk
        assert set(secs) == {"single-chain", "complex"}, f"sections {sorted(secs)}"
        assert "0.79" in secs["single-chain"] and "5.70" not in secs["single-chain"], \
            "a complex number leaked into the single-chain section"
        assert "5.70" in secs["complex"] and "0.79" not in secs["complex"], \
            "a sub-angstrom number leaked into the complex section"
        print("  selftest: provenance key separates arms")
        print("  selftest: report emits one section per arm, with the separation banner")
        print("  selftest: no cross-arm number leaked between sections")
    print("[demo] selftest PASSED")


if __name__ == "__main__":
    main()
