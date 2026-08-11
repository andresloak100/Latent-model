#!/usr/bin/env python3
"""INBOX 87c: the participation ratio of the STATIC codec, and its ENTROPY-CODED rate.

87c poses this as a question, not a conclusion, and it is right to: the 3% PR was measured on the
ATLAS dynamics arms, and the 82b/085 rate-distortion numbers come from the section-5 static codec on
758 val structures. Different codecs, so nothing transfers. This measures the static one.

IF IT IS ALSO LOW-PR, the two findings join into one diagnosis. A code concentrating its variance in
a handful of directions being beaten by a KLT with 1,117 components and variance-proportional bits is
close to expected, and the reading changes from "the architecture cannot" -- a dead end -- to "the
architecture is not being trained into its capacity", which is an optimisation and regularisation
problem and is addressable. If it is NOT low-PR, they are separate failures and 87c is withdrawn.

PR IS THE PROJECT'S OWN DEFINITION, CROSS-FIT, copied from armf_atlas_dm.participation_ratio rather
than re-derived: basis from TRAIN latents, eigenvalues from VAL latents projected onto that fixed
basis, PR = (sum lam)^2 / sum lam^2. An in-sample PR would repeat the error retracted for rank90 --
an eigenbasis fitted and evaluated on the same samples explains their variance optimally by
construction. Re-deriving it would also be a sixth "one name, two things" in a project that has
caught five.

THE ENTROPY-CODED RATE, which runs in the CODEC'S FAVOUR. The rate is currently billed as
(latent scalars x bits per scalar), i.e. every dimension charged the full 6 bits. A near-constant
dimension costs a uniform quantiser 6 bits and an entropy coder close to 0. If the code is low-PR,
most dimensions are near-constant and the reported rate is OVER-counted. This is the mirror image of
the float-counting error 66c already caught -- same class, opposite sign, and never checked in this
direction.
"""
import sys, os, json, math
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE + "/.."
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from molae.config import ExperimentConfig
from molae.dataset import ProteinStructureDataset, collate_fn
from molae.model_equivariant import make_autoencoder
from molae import utils

WR = os.environ["WR"]
CFG = os.environ.get("LIK_CFG", f"{WR}/results/ladder_direct3m_n2272/config.yaml")
CKPT = os.environ.get("LIK_CKPT", f"{WR}/results/ladder_direct3m_n2272/final.pt")
NTRAIN = int(os.environ.get("PR_NTRAIN", "600"))
NVAL = int(os.environ.get("PR_NVAL", "758"))
BITS = 6
dev = torch.device("cpu")


def latents(model, ds, n):
    """Token latents as observations in R^d, plus atoms per structure for the rate."""
    Z, na = [], []
    with torch.no_grad():
        for i in range(min(len(ds), n)):
            s = ds[i]
            gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
            _, z = model(gb)
            Z.append(z[0].cpu().numpy().astype(np.float64))     # (tokens, d)
            na.append(int(s["n_atoms"]))
    return np.concatenate(Z), na


if __name__ == "__main__":
    cfg = ExperimentConfig.from_yaml(CFG)
    sp = utils.load_json(f"{ROOT}/{cfg.data.splits_file}"); proc = f"{ROOT}/{cfg.data.processed_dir}"
    def mk(split, n):
        ks = [k for k in (sp.get(split) or []) if os.path.exists(f"{proc}/{k}.npz")][:n]
        return ProteinStructureDataset([f"{proc}/{k}.npz" for k in ks])
    model = make_autoencoder(cfg.model).to(dev)
    utils.load_checkpoint(CKPT, model, None, map_location=dev); model.eval()

    A, _ = latents(model, mk("train", NTRAIN), NTRAIN)
    B, na_v = latents(model, mk("val", NVAL), NVAL)
    d = A.shape[1]
    print(f"[87c] static codec latent: {d} channels per token; "
          f"{len(A):,} train tokens, {len(B):,} val tokens", flush=True)

    _, U = np.linalg.eigh(A.T @ A / len(A))                     # basis from TRAIN
    lam = np.clip(((B @ U) ** 2).mean(0), 0, None)              # variance of VAL on that basis
    pr = float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-30))
    print(f"\n=== 87c.1 PARTICIPATION RATIO (cross-fit, the project's own definition) ===")
    print(f"  PR = {pr:.2f} of {d} channels = {100*pr/d:.0f}% of the width", flush=True)
    print(f"  eigenvalue spectrum (VAL variance on the TRAIN basis), largest first:")
    print("    " + "  ".join(f"{v:.4g}" for v in np.sort(lam)[::-1]), flush=True)
    print(f"  share of variance in the top channel: {np.sort(lam)[::-1][0]/lam.sum():.1%}; "
          f"top 2: {np.sort(lam)[::-1][:2].sum()/lam.sum():.1%}", flush=True)
    verdict = ("LOW-PR -- joins 87a's finding: the same diagnosis, not two failures"
               if pr / d < 0.5 else
               "NOT low-PR -- the two findings are SEPARATE and 87c is withdrawn")
    print(f"  -> {verdict}", flush=True)

    print(f"\n=== 87c.2 ENTROPY-CODED RATE vs the billed rate ===")
    # Quantise exactly as the RD curve does: uniform, over the per-tensor range, BITS levels.
    lo, hi = B.min(), B.max()
    step = (hi - lo) / (2 ** BITS - 1)
    q = np.round((B - lo) / step).astype(np.int64)
    ent = []
    for j in range(d):
        _, cnt = np.unique(q[:, j], return_counts=True)
        p = cnt / cnt.sum()
        ent.append(float(-(p * np.log2(p)).sum()))
    tok_billed = d * BITS
    tok_entropy = float(np.sum(ent))
    fl_per_atom = float(np.median([len(B) / sum(na_v) for _ in (0,)]))  # tokens*d per atom, below
    tokens_per_atom = len(B) / sum(na_v)
    print(f"  per-channel entropy (bits): " + "  ".join(f"{e:.2f}" for e in ent), flush=True)
    print(f"  per token: billed {tok_billed} bits, entropy-coded {tok_entropy:.2f} bits "
          f"({100*tok_entropy/tok_billed:.0f}% of billed)", flush=True)
    print(f"  tokens per atom {tokens_per_atom:.4f}", flush=True)
    print(f"  BILLED RATE        {tok_billed*tokens_per_atom:.3f} bits/atom", flush=True)
    print(f"  ENTROPY-CODED RATE {tok_entropy*tokens_per_atom:.3f} bits/atom", flush=True)
    print(f"  -> the rate is over-counted by {tok_billed/max(tok_entropy,1e-9):.2f}x. "
          f"This runs in the CODEC'S FAVOUR and is the mirror of 66c's float-counting error.",
          flush=True)
    json.dump(dict(pr=pr, d=int(d), lam=lam.tolist(), entropy_per_channel=ent,
                   billed_bits_per_atom=tok_billed*tokens_per_atom,
                   entropy_bits_per_atom=tok_entropy*tokens_per_atom,
                   tokens_per_atom=tokens_per_atom),
              open(f"{WR}/codec_pr.json", "w"))
