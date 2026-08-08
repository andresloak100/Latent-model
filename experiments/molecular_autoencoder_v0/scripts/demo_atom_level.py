#!/usr/bin/env python3
"""INBOX 045/046: an atom-to-atom demo of the DIRECT per-residue codec, captioned so it cannot be
over-read.

NO SCIENTIFIC CONTENT. This produces no new measurement and settles no open question. If it competes
with the ladder, 41c or atlas_dm for a GPU slot, they win.

WHY THE PROVENANCE IS GENERATED, NOT WRITTEN BY HAND (46a). Three times in three days a number
correct in its own context was joined to one where it meant something different -- 41a joined a
tied-LABELLED slope to the tied peer gap (the slope was untied), 42b joined the ladder's FVE to 14a's
ANM across harnesses, and 045's own table put complex_d8's 5.5-5.9 A beside the single-chain 0.79 A.
Same failure, three axes. A hand-written caption would not have caught any of them. A provenance
string BUILT FROM THE LOADED CHECKPOINT makes a different arm visibly a different arm -- the PARTIAL
tag's principle applied to provenance instead of completeness.

WHY resolve_ckpt() RAISES (45c). INBOX 043 found armf_tied_peer.py about to load a stale n50
checkpoint and report it as an n300 result, because the read path had no discriminator and a
plausible file sat at it. A demo is MORE exposed to that than an experiment: it is read by people who
cannot check what produced it, and it outlives the session that made it.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

DEFAULT_RUN = "ladder_direct3m_n2272"   # 0.79 A all-atom / 0.51 A backbone (docs/CORPORA.md)
RUNS_DIR = ROOT / "outputs" / "cluster"

# INBOX 45b. Not footnotes: printed at the top of stdout, written to limits.md, and appended to the
# summary table. A demo that omits these is the thing every guard here exists to prevent.
LIMITS = [
 ("THE LATENT SCALES WITH RESIDUE COUNT",
  "latent_floats_for(n_res) = latent_dim * n_res = 8 * n_res, so ~1,600 scalars at 200 residues. "
  "This is NOT the fixed-size codec. The fixed-size Perceiver is dead in direction -- its gap to the "
  "direct codec WIDENS with data."),
 ("NO CLAIM ABOUT DYNAMICS",
  "Static reconstruction only. On dynamics the ATLAS line loses to zero-shot ANM on 100% of 123 "
  "held-out systems at every cutoff (5/7/10 A), with FVE_perp ~ 0. Nothing here bears on that."),
 ("NO CLAIM ABOUT 1M ATOMS",
  "The largest system ever run in this project is 33,377 atoms, and two caps block scaling above "
  "~8k. Quality here is measured on 20-110 residue single chains."),
]

def sh(cmd, default="unknown"):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10, cwd=str(ROOT)).stdout.strip() or default
    except Exception:
        return default

def resolve_ckpt(run_name):
    """Explicit, or an exception. Never a neighbour."""
    d = RUNS_DIR / run_name
    cp, cfg = d / "final.pt", d / "config.yaml"
    if not cp.exists() or not cfg.exists():
        avail = sorted(p.name for p in RUNS_DIR.iterdir() if (p / "final.pt").exists()) \
                if RUNS_DIR.is_dir() else []
        raise FileNotFoundError(
            f"no checkpoint+config at {d}. Refusing to fall back to any other run -- a checkpoint the "
            f"caller did not name must never be loaded (INBOX 43b). Runs with a final.pt: {avail}")
    h = hashlib.sha256()
    with open(cp, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
    return dict(run=run_name, ckpt=str(cp), config=str(cfg), sha256=h.hexdigest()[:12],
                mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(cp.stat().st_mtime)),
                git=sh("git rev-parse --short HEAD"), dirty=bool(sh("git status --porcelain", "")))

def provenance_string(prov, meta, n_heldout):
    return (f"direct-per-residue / {meta['corpus']} / held-out n={n_heldout} / "
            f"n_train={meta['n_train']} / ckpt {prov['run']}@{prov['sha256']} / "
            f"git {prov['git']}{'+dirty' if prov['dirty'] else ''}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--n-structures", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo_atom_level"))
    ap.add_argument("--validate", action="store_true",
                    help="resolve checkpoint/config/splits and exit. No GPU, no inference.")
    a = ap.parse_args()

    print("=" * 100)
    print("ATOM-LEVEL DEMO -- DIRECT PER-RESIDUE CODEC. No new measurement; settles no open question.")
    print("=" * 100)
    for i, (t, b) in enumerate(LIMITS, 1):
        print(f"  LIMIT {i}. {t}")
        for j in range(0, len(b), 92): print(f"          {b[j:j+92]}")
    print("=" * 100, flush=True)

    prov = resolve_ckpt(a.run)
    print(f"  checkpoint : {prov['ckpt']}")
    print(f"  sha256[:12]: {prov['sha256']}   written {prov['mtime']}")
    print(f"  git        : {prov['git']}{'  (DIRTY TREE)' if prov['dirty'] else ''}", flush=True)

    import yaml
    raw = yaml.safe_load(open(prov["config"]))
    splits_file = ROOT / raw["data"]["splits_file"]
    splits = json.load(open(splits_file))
    heldout = list(splits.get("val") or [])
    meta = dict(n_train=len(splits.get("train") or []), corpus="single-chain 20-110 res")
    pstr = provenance_string(prov, meta, len(heldout))
    print(f"  splits     : {splits_file.name}  train={meta['n_train']} held-out={len(heldout)}")
    print(f"  PROVENANCE : {pstr}", flush=True)

    if a.validate:
        print("\n  --validate: everything resolves. Not running inference (needs a GPU slot).")
        print(f"  would reconstruct {a.n_structures} held-out structures into {a.out}")
        return 0

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "limits.md").write_text("# What this demo does NOT show\n\n" +
        "".join(f"## {i}. {t}\n\n{b}\n\n" for i, (t, b) in enumerate(LIMITS, 1)) +
        f"\n---\n\nProvenance: `{pstr}`\n")

    import torch
    from molae.config import ExperimentConfig
    from molae.dataset import ProteinStructureDataset, collate_fn
    from molae.model_equivariant import make_autoencoder
    from molae.metrics import build_topology_info, compute_all_metrics
    from molae.pdb_io import write_pdb

    cfg = ExperimentConfig.from_yaml(Path(prov["config"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_autoencoder(cfg.model).to(dev)
    model.load_state_dict(torch.load(prov["ckpt"], map_location=dev), strict=False)
    model.eval()

    ds = ProteinStructureDataset(ROOT / cfg.data.processed_dir, heldout, cfg.data)
    order = sorted(range(len(ds)), key=lambda i: ds[i]["n_atoms"])   # span small -> large
    pick = [order[j] for j in np.linspace(0, len(order) - 1,
                                          min(a.n_structures, len(order))).astype(int)]
    rows = []
    for i in pick:
        s = ds[i]
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in collate_fn([s]).items()}
        with torch.no_grad():
            pred = model(batch)["coords"][0].cpu().numpy()
        target = np.asarray(s["coords"])
        n_res = int(s.get("n_residues", 0)) or max(int(s["n_atoms"]) // 8, 1)
        lf = model.latent_floats_for(n_res) if hasattr(model, "latent_floats_for") else model.latent_floats
        topo = build_topology_info(s)
        m = compute_all_metrics(pred, target, topo, latent_floats=lf)
        pid = s["pdb_id"]
        write_pdb(out / f"{pid}_true.pdb", target, topo)
        write_pdb(out / f"{pid}_pred.pdb", pred, topo)
        err = np.linalg.norm(pred - target, axis=-1)   # per-atom, so it can be COLOURED not summarised
        np.savetxt(out / f"{pid}_per_atom_error.csv", err, fmt="%.4f",
                   header=f"{pid} per-atom |err| Angstrom")
        rows.append(dict(pdb_id=pid, n_atoms=int(s["n_atoms"]), n_res=n_res, latent_scalars=int(lf),
                         all_atom_rmsd=float(m["all_atom_rmsd"]),
                         backbone_rmsd=float(m.get("backbone_rmsd", float("nan"))),
                         chirality=float(m.get("chirality_violation_rate", float("nan"))),
                         contact_f1=float(m.get("contact_f1", float("nan"))),
                         err_median=float(np.median(err)), err_p95=float(np.percentile(err, 95)),
                         err_max=float(err.max())))
        print(f"    {pid:>6}  {int(s['n_atoms']):>5} atoms  {n_res:>4} res  latent {int(lf):>5} "
              f"scalars  all-atom {m['all_atom_rmsd']:.3f} A", flush=True)

    aa = np.array([r["all_atom_rmsd"] for r in rows])   # 28e: median, mean AND tail together
    tbl = ["# Atom-level demo -- direct per-residue codec", "", f"`{pstr}`", "",
           "Every number below carries that provenance line. It is GENERATED from the loaded "
           "checkpoint, not written by hand (INBOX 46a).", "",
           "| PDB | atoms | res | latent scalars | all-atom RMSD (A) | backbone RMSD (A) | chirality "
           "| contact F1 | err median | err p95 | err max |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tbl.append(f"| {r['pdb_id']} | {r['n_atoms']} | {r['n_res']} | {r['latent_scalars']} | "
                   f"{r['all_atom_rmsd']:.3f} | {r['backbone_rmsd']:.3f} | {r['chirality']:.5f} | "
                   f"{r['contact_f1']:.3f} | {r['err_median']:.3f} | {r['err_p95']:.3f} | "
                   f"{r['err_max']:.3f} |")
    tbl += ["", f"median all-atom {np.median(aa):.3f} A, mean {aa.mean():.3f} A, worst "
                f"{aa.max():.3f} A over {len(aa)} structures.", "", "## What this does NOT show", ""]
    tbl += [f"**{i}. {t}** {b}\n" for i, (t, b) in enumerate(LIMITS, 1)]
    (out / "summary.md").write_text("\n".join(tbl) + "\n")
    json.dump(dict(provenance=pstr, checkpoint=prov, rows=rows),
              open(out / "summary.json", "w"), indent=2)
    print(f"\n  wrote {len(rows)} structures + summary.md + summary.json + limits.md to {out}")
    print(f"  median all-atom {np.median(aa):.3f} A, mean {aa.mean():.3f} A, worst {aa.max():.3f} A")
    return 0

if __name__ == "__main__":
    sys.exit(main())
