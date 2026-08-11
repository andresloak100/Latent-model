#!/usr/bin/env python3
"""INBOX 93d: re-split the STATIC corpus at 30% identity and re-score, because `auto` chose `exact`.

THE ASYMMETRY. `prepare_dataset.split_dataset` picks `"similarity" if n <= 400 else "exact"`. The
static corpus is 3,030 structures, so `auto` silently selected **exact-sequence dedup** -- which does
not separate homologs. A val structure 90% identical to a training structure is a legitimate member
of the val set under that rule.

So the project applies a STRICTER standard to its pretraining corpus than to its own headline
evaluation: the AFDB gate is MMseqs2 at 30% identity, deliberately chosen and self-tested, while the
set that produces the 0.8357 A reconstruction number is split by exact match. That is not defensible,
and it was invisible because the threshold lives behind an `auto` that changes rule at n=400.

This does NOT invalidate the reconstruction number and nothing here claims it does. It makes the
number's GENERALISATION claim unmeasured, in the same way and for the same reason the AFDB leak was
before 072a.

WHY MMseqs2 AND NOT similarity_split. `similarity_split` is quadratic, which is exactly why `auto`
avoids it above 400. MMseqs2 is not, and it is already installed, validated and used by the AFDB gate
-- so this reuses the tool the project already trusts rather than introducing a second definition of
"30% identity".

WHAT IS REPORTED: the old and new numbers SIDE BY SIDE. If they agree the concern closes permanently.
If they do not, better to know it from us.
"""
import sys, os, json, glob, subprocess, math
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE + "/.."
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from molae.config import ExperimentConfig
from molae.dataset import ProteinStructureDataset, collate_fn
from molae.model_equivariant import make_autoencoder
from molae.metrics import all_atom_rmsd
from molae import utils

WR = os.environ["WR"]
CFG = os.environ.get("LIK_CFG", f"{WR}/results/ladder_direct3m_n2272/config.yaml")
CKPT = os.environ.get("LIK_CKPT", f"{WR}/results/ladder_direct3m_n2272/final.pt")
MMSEQS = f"{WR}/mmseqs/bin/mmseqs"
MIN_ID = 0.30
TMP = f"{WR}/resplit30"
RES = f"{WR}/resplit30.json"
dev = torch.device("cpu")


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"  FAILED: {cmd}\n{r.stderr[-1500:]}")
    return r.stdout


@torch.no_grad()
def score(model, paths, label):
    """Median all-atom Kabsch-aligned RMSD -- the project's own metric, not a second one."""
    ds = ProteinStructureDataset(paths)
    out = []
    for i in range(len(ds)):
        s = ds[i]; na = int(s["n_atoms"])
        gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
        pred, _ = model(gb)
        out.append(float(all_atom_rmsd(pred[0, :na].cpu().numpy().astype(np.float64),
                                       gb["coords"][0, :na].cpu().numpy().astype(np.float64))))
    a = np.array(out)
    print(f"  {label:<34} n={len(a):>5}  median {np.median(a):.4f} A  mean {a.mean():.4f}  "
          f"p90 {np.percentile(a,90):.4f}", flush=True)
    return a


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    cfg = ExperimentConfig.from_yaml(CFG)
    proc = f"{ROOT}/{cfg.data.processed_dir}"
    sp = utils.load_json(f"{ROOT}/{cfg.data.splits_file}")
    old_tr = [k for k in (sp.get("train") or []) if os.path.exists(f"{proc}/{k}.npz")]
    old_va = [k for k in (sp.get("val") or []) if os.path.exists(f"{proc}/{k}.npz")]
    print(f"[93d] existing split (method=exact, chosen by `auto` at n>400): "
          f"{len(old_tr)} train / {len(old_va)} val", flush=True)

    seqs = {}
    for k in old_tr + old_va:
        d = np.load(f"{proc}/{k}.npz", allow_pickle=True)
        s = str(d["sequence"]) if "sequence" in d.files else ""
        if s: seqs[k] = s
    with open(f"{TMP}/all.fasta", "w") as fh:
        for k, s in seqs.items(): fh.write(f">{k}\n{s}\n")
    print(f"  {len(seqs)} sequences extracted", flush=True)

    # Cluster at 30% identity. Every member of a cluster goes to the SAME side, which is what
    # exact-match dedup fails to do.
    run(f"{MMSEQS} easy-cluster {TMP}/all.fasta {TMP}/clu {TMP}/mm --min-seq-id {MIN_ID} "
        f"-c 0.5 --cov-mode 0 --threads 8 >/dev/null 2>&1")
    clu = {}
    for line in open(f"{TMP}/clu_cluster.tsv"):
        rep, mem = line.rstrip("\n").split("\t")[:2]
        clu.setdefault(rep, []).append(mem)
    print(f"  {len(clu)} clusters at {MIN_ID:.0%} identity from {len(seqs)} sequences "
          f"-- {len(seqs)-len(clu)} sequences share a cluster with another", flush=True)

    # A DEFECT OF MINE, CAUGHT BY THE SIGN OF THE ANSWER. My first version re-split from scratch and
    # re-scored the EXISTING checkpoint on the new val set. That set contained 577 of 760 structures
    # (76%) drawn from the OLD TRAIN set -- data the model was trained on -- so it measured
    # MEMORISATION and reported 0.2879 A against 0.8357, a 65% "improvement" from a STRICTER split.
    # Leakage cannot make a held-out set easier; the sign was the tell. The checkpoint's training set
    # is fixed, so no re-split can be scored under it without retraining.
    #
    # WHAT CAN BE MEASURED WITHOUT RETRAINING is the leakage already present: partition the OLD val
    # set by whether each structure has a >=30% identity match in the OLD TRAIN set, and compare the
    # codec on the two halves. That is the quantity 93d is about -- whether the headline number rests
    # on homologs the split rule failed to separate.
    with open(f"{TMP}/train.fasta", "w") as fh:
        for k in old_tr:
            if k in seqs: fh.write(f">{k}\n{seqs[k]}\n")
    with open(f"{TMP}/val.fasta", "w") as fh:
        for k in old_va:
            if k in seqs: fh.write(f">{k}\n{seqs[k]}\n")
    run(f"{MMSEQS} easy-search {TMP}/val.fasta {TMP}/train.fasta {TMP}/hits.m8 {TMP}/mm2 "
        f"--min-seq-id {MIN_ID} -c 0.5 --cov-mode 0 -s 5.7 --threads 8 "
        f"--format-output query,target,fident >/dev/null 2>&1")
    contaminated = {}
    for line in open(f"{TMP}/hits.m8"):
        q, t, fid = line.split("\t")[:3]
        if q != t:
            contaminated[q] = max(contaminated.get(q, 0.0), float(fid))
    clean = [k for k in old_va if k not in contaminated]
    dirty = [k for k in old_va if k in contaminated]
    print(f"\n=== 93d: HOW MUCH OF THE HELD-OUT SET HAS A TRAINING HOMOLOG ===", flush=True)
    print(f"  {len(dirty)} of {len(old_va)} val structures ({100*len(dirty)/len(old_va):.1f}%) have a "
          f">={MIN_ID:.0%} identity match in TRAIN", flush=True)
    if dirty:
        f = np.array(list(contaminated.values()))
        print(f"  identity to the best training match: median {np.median(f):.1%}, "
              f"max {f.max():.1%}, and {(f>=0.9).sum()} at >=90%", flush=True)

    model = make_autoencoder(cfg.model).to(dev)
    utils.load_checkpoint(CKPT, model, None, map_location=dev); model.eval()
    print(f"\n  THE SAME CHECKPOINT ON THE TWO HALVES OF ITS OWN HELD-OUT SET:", flush=True)
    a_old = score(model, [f"{proc}/{k}.npz" for k in old_va], "ALL val (the 0.8357 number)")
    a_dirty = score(model, [f"{proc}/{k}.npz" for k in dirty], "val WITH a training homolog") \
        if dirty else np.array([])
    a_new = score(model, [f"{proc}/{k}.npz" for k in clean], "val with NO training homolog") \
        if clean else np.array([])
    if len(a_new) and len(a_dirty):
        d = float(np.median(a_new) - np.median(a_dirty))
        print(f"\n  homolog-free minus homolog-bearing: {d:+.4f} A "
              f"({100*d/max(np.median(a_dirty),1e-9):+.1f}%)", flush=True)
        print(f"  -> {'the concern CLOSES: separating homologs does not move the number' if abs(d) < 0.05 else 'the headline number IS easier on structures with training homologs'}",
              flush=True)
    else:
        a_new = a_new if len(a_new) else a_old
        d = 0.0
        print(f"\n  one half is empty -- reported as absent rather than as a null", flush=True)
    moved = len(dirty)
    json.dump(dict(all_median=float(np.median(a_old)),
                   clean_median=float(np.median(a_new)) if len(a_new) else None,
                   dirty_median=float(np.median(a_dirty)) if len(a_dirty) else None,
                   n_all=len(a_old), n_clean=len(clean), n_dirty=len(dirty),
                   n_clusters=len(clu), min_seq_id=MIN_ID), open(RES, "w"))
