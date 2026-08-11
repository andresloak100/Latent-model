#!/usr/bin/env python3
"""INBOX 84b/84c: PCA against the codec at MATCHED BITS PER ATOM, with bits allocated properly.

TWO DEFECTS IN 82b's COMPARISON, both of which flattered the codec.

84b -- "MATCHED RATE" MEANT MATCHED BITS PER COEFFICIENT, NOT PER ATOM. 82b quantised both sides to
6 bits per coefficient and called that matched. But the two methods carry different numbers of
coefficients per atom, so equal bits/coefficient is not equal bits/atom -- and bits/atom is the axis
66c and 071 spent four items establishing as the honest one. Measured here: the codec holds 1.025
latent floats per atom, so at 6 bits it spends 6.15 bits/atom, while PCA-256 over 400 atoms spends
1536/400 = 3.84. The codec had 1.6x MORE RATE in a table labelled matched. To match, PCA needs
2459 bits per structure = 410 components at 6 flat bits, not 256. Eighth instance of one name, two
things, and this one sat in the headline comparison.

  Rank is capped by the number of structures the PCA is FITTED on, and 379 val structures cannot
  support 410 components -- which is why 82b could not reach matched rate even in principle. PCA is
  therefore fitted on the TRAIN split (2,272 structures). That is also the fairer comparison: the
  codec was trained on train, so a linear reference fitted on train and scored on val is the
  like-for-like control, not a handicap.

84c -- FLAT SIX BITS ON EVERY COMPONENT IS A KNOWN-SUBOPTIMAL ALLOCATION. PCA coefficient variances
span the eigenvalue spectrum by orders of magnitude. Rate-distortion theory says bits follow
variance, by reverse water-filling: b_i = max(0, 0.5*log2(v_i/theta)), so components below the water
level get ZERO bits rather than six. Under a flat allocation each extra low-variance component costs
6 bits and buys almost nothing -- which is exactly the saturation 82b measured (3.2559 -> 3.2891 ->
3.2821 from k=64 to k=256) and attributed to PCA. It is a property of the allocation.

THREE ROWS, so the two effects are separable rather than confounded:
    flat 6 bits, GLOBAL range      -- what 82b did, kept and labelled
    flat 6 bits, per-component range -- isolates the range convention
    water-filling, per-component range -- the proper allocation

AND THE TRUNCATION CAVEAT IS REMOVED, not merely flagged. 82b scored PCA on the first 400 atoms and
the codec on all atoms. Here BOTH are scored on the same first-NATOM atoms of the same structures.
The codec's rate stays its true whole-structure rate (its latent encodes every atom), and the scored
atoms are a subset of that -- so if anything this now runs against the codec, which is the direction
a caveat should run.

PCA is a LINEAR INTERNAL reference, not the peer. ANM is the peer, and 5b's "no surviving
peer-comparison win" is untouched by anything here.
"""
import sys, os, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE + "/.."
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from molae.config import ExperimentConfig
from molae.dataset import ProteinStructureDataset, collate_fn
from molae.model_equivariant import make_autoencoder
from molae import utils
import torch

WR = os.environ["WR"]
CFG = os.environ.get("LIK_CFG", f"{WR}/results/ladder_direct3m_n2272/config.yaml")
CKPT = os.environ.get("LIK_CKPT", f"{WR}/results/ladder_direct3m_n2272/final.pt")
NATOM = int(os.environ.get("PCA_NATOM", "400"))
NFIT = int(os.environ.get("PCA_NFIT", "1200"))     # train structures for the PCA fit
NVAL = int(os.environ.get("PCA_NVAL", "400"))      # val structures for scoring
RES = f"{WR}/pca_matched.json"
dev = torch.device("cpu")


def nll_bits(err, sigma):
    n = 0.5 * math.log(2 * math.pi * sigma ** 2) + float((err ** 2).mean()) / (2 * sigma ** 2)
    return n / math.log(2)


def waterfill(var, total_bits):
    """Reverse water-filling: b_i = max(0, 0.5*log2(v_i/theta)), theta set so sum(b_i) = total_bits.

    Components below the water level get ZERO bits. That is the whole point -- a flat allocation
    spends six bits on a component carrying almost no variance, which is why adding components under
    a flat budget stops buying anything."""
    lo, hi = var.min() * 1e-12, var.max()
    for _ in range(200):
        th = math.sqrt(lo * hi) if lo > 0 else hi / 2
        b = np.maximum(0.0, 0.5 * np.log2(var / th))
        if b.sum() > total_bits: lo = th
        else: hi = th
    b = np.maximum(0.0, 0.5 * np.log2(var / th))
    bi = np.floor(b).astype(int)                 # integer bits; the remainder is handed out to the
    rem = int(total_bits - bi.sum())             # highest-variance components, largest first
    if rem > 0:
        order = np.argsort(-(b - bi))
        bi[order[:min(rem, len(bi))]] += 1
    return bi


def quantise_cols(C, bits, lo, hi):
    """Per-column uniform scalar quantisation at the column's own bit count. Zero bits = the column
    is not transmitted at all, so it decodes to the training mean (0 in centred coordinates)."""
    out = np.zeros_like(C)
    for j in range(C.shape[1]):
        if bits[j] <= 0:
            continue
        levels = 2 ** int(bits[j]) - 1
        step = (hi[j] - lo[j]) / max(levels, 1)
        if step <= 0:
            out[:, j] = C[:, j]; continue
        out[:, j] = np.round((C[:, j] - lo[j]) / step) * step + lo[j]
    return out


def load(split, n):
    cfg = ExperimentConfig.from_yaml(CFG)
    sp = utils.load_json(f"{ROOT}/{cfg.data.splits_file}")
    proc = f"{ROOT}/{cfg.data.processed_dir}"
    keys = [k for k in (sp.get(split) or []) if os.path.exists(f"{proc}/{k}.npz")][:n]
    return cfg, ProteinStructureDataset([f"{proc}/{k}.npz" for k in keys])


if __name__ == "__main__":
    cfg, ds_fit = load("train", NFIT)
    _, ds_val = load("val", NVAL)
    print(f"[84b/84c] PCA fitted on TRAIN, scored on VAL, both on the first {NATOM} atoms",
          flush=True)

    def coords(ds):
        X, na_list = [], []
        for i in range(len(ds)):
            s = ds[i]; na = int(s["n_atoms"])
            if na < NATOM: continue
            X.append(np.asarray(s["coords"])[:NATOM].ravel()); na_list.append(na)
        return np.array(X, np.float64), na_list

    Xf, _ = coords(ds_fit)
    Xv, na_v = coords(ds_val)
    print(f"  fit {len(Xf)} train structures (rank cap {len(Xf)-1}), score {len(Xv)} val structures",
          flush=True)

    mu = Xf.mean(0)
    U, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    var = (S ** 2) / max(len(Xf) - 1, 1)
    print(f"  eigenvalue spectrum spans {var.max()/max(var.min(),1e-30):.2e}x "
          f"-- which is why a flat allocation cannot be right", flush=True)

    # THE CODEC, scored on the SAME atoms, at its own measured rate.
    model = make_autoencoder(cfg.model).to(dev)
    utils.load_checkpoint(CKPT, model, None, map_location=dev); model.eval()
    cerr, cfl = [], []
    with torch.no_grad():
        for i in range(len(ds_val)):
            s = ds_val[i]; na = int(s["n_atoms"])
            if na < NATOM: continue
            gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
            pred, z = model(gb)
            p = pred[0, :NATOM].cpu().numpy().astype(np.float64)
            t = gb["coords"][0, :NATOM].cpu().numpy().astype(np.float64)
            cerr.append((p - t).ravel()); cfl.append(int(np.prod(z.shape)) / na)
    ce = np.concatenate(cerr)
    fl_per_atom = float(np.median(cfl))
    CODEC_BPA = fl_per_atom * 6.0
    # IDENTICAL PROTOCOL ON BOTH SIDES. PCA's sigma is fitted on the TRAIN structures it was fitted
    # on and scored on ALL of val; the codec's must be too, or the two bits/dim are not comparable
    # and the comparison is Family F in the estimator -- which is exactly what 77a caught in the
    # propagator. The codec's sigma therefore comes from its reconstruction error on the SAME train
    # structures, and it is scored on the SAME full val set. This is a stricter test of the codec
    # than the disjoint-val-half protocol, because train error is smaller than val error, so the
    # fitted sigma is smaller and the val NLL it produces is larger.
    terr = []
    with torch.no_grad():
        for i in range(len(ds_fit)):
            s = ds_fit[i]; na = int(s["n_atoms"])
            if na < NATOM: continue
            gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
            pr, _ = model(gb)
            terr.append((pr[0, :NATOM].cpu().numpy().astype(np.float64)
                         - gb["coords"][0, :NATOM].cpu().numpy().astype(np.float64)).ravel())
    sig_c = float(np.sqrt((np.concatenate(terr) ** 2).mean()))
    cod_bits = nll_bits(ce, sig_c)
    print(f"  codec sigma fitted on {len(terr)} TRAIN structures = {sig_c:.4f} A "
          f"(same protocol as PCA's)", flush=True)
    sig_pow = float(((Xv.reshape(len(Xv), NATOM, 3)
                      - Xv.reshape(len(Xv), NATOM, 3).mean(1, keepdims=True)) ** 2).sum(-1).mean())
    cod_mse = float((ce ** 2).reshape(-1, 3).sum(-1).mean())
    cod_db = 10 * math.log10(sig_pow / cod_mse)
    print(f"\n  CODEC on these atoms: {fl_per_atom:.3f} latent floats/atom -> "
          f"{CODEC_BPA:.2f} bits/atom at 6 bits/scalar", flush=True)
    print(f"    {cod_bits:.4f} bits/dim   SNR {cod_db:.2f} dB", flush=True)

    budget = CODEC_BPA * NATOM
    k_flat = int(round(budget / 6))
    print(f"\n  MATCHED BUDGET = {CODEC_BPA:.2f} bits/atom x {NATOM} atoms = {budget:.0f} bits"
          f"  ->  {k_flat} components at 6 flat bits (82b used 256)", flush=True)

    Cf = (Xf - mu) @ Vt.T
    Cv = (Xv - mu) @ Vt.T
    rows = []
    print(f"\n  {'scheme':<40}{'k':>6}{'bits/atom':>11}{'bits/dim':>10}{'SNR dB':>9}{'nonzero':>9}")

    def score(label, kk, bits, lo, hi, note=""):
        Cq = quantise_cols(Cv[:, :kk].copy(), bits, lo, hi)
        rec = Cq @ Vt[:kk] + mu
        err = (rec - Xv)
        Cqf = quantise_cols(Cf[:, :kk].copy(), bits, lo, hi)
        sg = float(np.sqrt(((Cqf @ Vt[:kk] + mu - Xf) ** 2).mean()))
        b = nll_bits(err.ravel(), sg)
        mse = float((err ** 2).reshape(-1, 3).sum(-1).mean())
        db = 10 * math.log10(sig_pow / mse)
        bpa = float(bits.sum()) / NATOM
        nz = int((bits > 0).sum())
        print(f"  {label:<40}{kk:>6}{bpa:>11.2f}{b:>10.4f}{db:>9.2f}{nz:>9}  {note}", flush=True)
        rows.append(dict(scheme=label, k=int(kk), bits_per_atom=bpa, bits_per_dim=b, snr_db=db,
                         nonzero=nz))
        return db, b

    # ROW 1: what 82b did -- flat 6 bits, GLOBAL range, k=256. Kept and labelled.
    for kk in (64, 256):
        lo = np.full(kk, Cf[:, :kk].min()); hi = np.full(kk, Cf[:, :kk].max())
        score(f"82b: flat 6 bits, GLOBAL range", kk, np.full(kk, 6), lo, hi,
              "<- NOT rate-matched" if kk == 256 else "")
    # ROW 2: flat 6 bits at the MATCHED k, per-component range -- isolates the range convention.
    lo = Cf[:, :k_flat].min(0); hi = Cf[:, :k_flat].max(0)
    score(f"84b: flat 6 bits, per-component range", k_flat, np.full(k_flat, 6), lo, hi,
          "<- rate-matched")
    # ROW 3: reverse water-filling at the same total budget, over the full available rank.
    for kmax in (k_flat, min(len(Xf) - 1, Vt.shape[0])):
        b_wf = waterfill(var[:kmax], budget)
        lo = Cf[:, :kmax].min(0); hi = Cf[:, :kmax].max(0)
        score(f"84c: WATER-FILLING, per-component range", kmax, b_wf, lo, hi,
              "<- rate-matched, proper allocation")

    best = min(r["bits_per_dim"] for r in rows if abs(r["bits_per_atom"] - CODEC_BPA) < 0.5) \
        if any(abs(r["bits_per_atom"] - CODEC_BPA) < 0.5 for r in rows) else None
    print(f"\n  CODEC {cod_bits:.4f} bits/dim, {cod_db:.2f} dB at {CODEC_BPA:.2f} bits/atom")
    if best is not None:
        print(f"  BEST RATE-MATCHED PCA {best:.4f} bits/dim  ->  codec advantage "
              f"{best - cod_bits:+.4f} bits/dim", flush=True)
    json.dump(dict(codec=dict(bits_per_dim=cod_bits, snr_db=cod_db, bits_per_atom=CODEC_BPA),
                   pca=rows, natom=NATOM, n_fit=len(Xf), n_val=len(Xv)), open(RES, "w"))
    print(f"  -> {RES}", flush=True)
