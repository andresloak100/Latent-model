#!/usr/bin/env python3
"""INBOX 090a: the codec's OWN rate-distortion point, with the latent actually quantised.

TWO DEFECTS IN THE 084/085/087/088 COMPARISON, BOTH MINE, BOTH FOUND BY 090a's QUESTION.

1. THE 088 ROTATION PRECEDES QUANTISATION. `ent(Rx)` quantises its argument, and I passed the
   ROTATED latent. So the 3.221 bits/atom is the rate of a code quantised in the eigenbasis -- but
   the distortion it was paired with, 2.0588 bits/dim, came from a code that was never rotated. A
   rotated rate cannot carry an unrotated code's distortion. 092b measured the size of that on
   synthetic data: rotating before quantising moves MSE by 1-15%.

2. WORSE, AND NOT WHAT 090a ASKED ABOUT: the codec's distortion was measured with the latent NEVER
   QUANTISED AT ALL. armf_pca_matched.py does `pred, z = model(gb)` and then scores `pred - coords`;
   `z` is not quantised anywhere on that path. So the codec was billed a 6-bit rate while being
   scored at full float precision, while PCA's distortion WAS measured on quantised coefficients
   (quantise_cols -> reconstruct -> error). The two sides were never comparable on the distortion
   axis. That is Family F in the comparison itself, it has been there since 084, and it runs in the
   codec's favour.

WHAT THIS MEASURES INSTEAD. One path, end to end, with nothing assumed:

    z -> centre (train mu) -> rotate (train eigenbasis U) -> QUANTISE -> dequantise
      -> unrotate -> uncentre -> decode -> distortion

and the rate is the entropy of the symbols that were actually transmitted. Rotation is orthogonal so
the decoder undoes it exactly; the only loss is the quantiser's, which is the point.

The UNROTATED arm runs at the identical bit budget so the rotation's effect on DISTORTION is
isolated rather than assumed -- 092b predicted 1-15% and retracted its own sign, so it is measured
here rather than argued.
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
NATOM = int(os.environ.get("PCA_NATOM", "400"))
NFIT = int(os.environ.get("RD_NFIT", "600"))
NVAL = int(os.environ.get("RD_NVAL", "400"))
BUDGETS = [3, 4, 5, 6, 8]
# INBOX 094. THE FLAT SWEEP ABOVE IS 84c's HANDICAP APPLIED TO THE OTHER ARM. 84c established that a
# flat bits-per-component allocation makes a spread-variance code look bad, and correcting it is what
# flipped the static comparison in the first place -- then 090a swept the CODEC flat while PCA's arm
# used waterfill. The codec's spectrum spans 627x (AM 143.3, GM 19.28), so optimal allocation is worth
# up to 8.71 dB against the 2.1 dB gap measured -- 4.1x the effect. That is a HIGH-RATE GAUSSIAN BOUND
# IN LATENT SPACE and the decoder is nonlinear, so it caps the headroom rather than predicting the
# outcome. The flat rows are kept and labelled, exactly as 84c kept PCA's.
WF_BUDGETS = [float(x) for x in os.environ.get("RD_WF", "2.6,4.2,5.9,7.6").split(",")]
dev = torch.device("cpu")


def nll_bits(err, sigma):
    n = 0.5 * math.log(2 * math.pi * sigma ** 2) + float((err ** 2).mean()) / (2 * sigma ** 2)
    return n / math.log(2)


def ent_bits_cols(q):
    """Entropy of the ACTUALLY TRANSMITTED symbols, per column, summed. Marginal per column, which
    is the joint rate only if the columns are independent -- which is precisely why the rotation is
    applied first (088). After an orthogonal rotation into the eigenbasis the residual off-diagonal
    correlation was measured at 0.0194, so the sum is the joint rate to that accuracy."""
    tot = 0.0
    for j in range(q.shape[1]):
        _, c = np.unique(q[:, j], return_counts=True)
        p = c / c.sum()
        tot += float(-(p * np.log2(p)).sum())
    return tot


def quant(X, bits, lo, hi):
    """Uniform scalar quantisation per column at `bits`, returning symbols AND the dequantised
    values -- the rate comes from the first and the distortion from the second, so they cannot
    drift apart."""
    step = (hi - lo) / (2 ** bits - 1)
    step = np.where(step <= 0, 1.0, step)
    q = np.round((X - lo) / step)
    return q.astype(np.int64), q * step + lo


if __name__ == "__main__":
    cfg = ExperimentConfig.from_yaml(CFG)
    sp = utils.load_json(f"{ROOT}/{cfg.data.splits_file}"); proc = f"{ROOT}/{cfg.data.processed_dir}"
    def mk(split, n):
        ks = [k for k in (sp.get(split) or []) if os.path.exists(f"{proc}/{k}.npz")][:n]
        return ProteinStructureDataset([f"{proc}/{k}.npz" for k in ks])
    model = make_autoencoder(cfg.model).to(dev)
    utils.load_checkpoint(CKPT, model, None, map_location=dev); model.eval()
    if not hasattr(model, "decode"):
        raise SystemExit("  model exposes no decode(z, batch); cannot re-decode a quantised latent")

    # TRAIN latents -> centre and eigenbasis. Cross-fit, the same discipline PCA's basis got.
    A = []
    with torch.no_grad():
        ds = mk("train", NFIT)
        for i in range(len(ds)):
            s = ds[i]
            if int(s["n_atoms"]) < NATOM: continue
            gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
            _, z = model(gb); A.append(z[0].cpu().numpy().astype(np.float64))
    A = np.concatenate(A); mu = A.mean(0)
    _, U = np.linalg.eigh((A - mu).T @ (A - mu) / len(A))
    Ac = (A - mu) @ U
    lo_r, hi_r = Ac.min(0), Ac.max(0)                 # ranges from TRAIN, never from the scored set
    lo_p, hi_p = (A - mu).min(0), (A - mu).max(0)
    print(f"[090a] codec RD with the latent ACTUALLY quantised. basis+ranges from {len(A):,} train "
          f"tokens", flush=True)

    ds = mk("val", NVAL)
    rows = []
    for bits in BUDGETS:
        errs = {"rot": [], "raw": []}
        syms = {"rot": [], "raw": []}
        sig = {"rot": 0.0, "raw": 0.0}
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]; na = int(s["n_atoms"])
                if na < NATOM: continue
                gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
                _, z = model(gb)
                zn = z[0].cpu().numpy().astype(np.float64)
                t = gb["coords"][0, :NATOM].cpu().numpy().astype(np.float64)
                for arm in ("rot", "raw"):
                    if arm == "rot":
                        c = (zn - mu) @ U
                        q, dq = quant(c, bits, lo_r, hi_r)
                        back = dq @ U.T + mu                    # orthogonal: exact undo
                    else:
                        c = zn - mu
                        q, dq = quant(c, bits, lo_p, hi_p)
                        back = dq + mu
                    zq = torch.tensor(back, dtype=z.dtype).unsqueeze(0)
                    p = model.decode(zq, gb)[0, :NATOM].cpu().numpy().astype(np.float64)
                    errs[arm].append((p - t).ravel()); syms[arm].append(q)
        out = {}
        for arm in ("rot", "raw"):
            e = np.concatenate(errs[arm]); S = np.concatenate(syms[arm])
            sg = float(np.sqrt((e ** 2).mean()))
            bpa = ent_bits_cols(S) * (len(S) / (len(errs[arm]) * NATOM))
            mse = float((e ** 2).reshape(-1, 3).sum(-1).mean())
            out[arm] = dict(bits=bits, sigma=sg, entropy_bits_per_atom=bpa,
                            billed_bits_per_atom=S.shape[1] * bits * len(S)
                            / (len(errs[arm]) * NATOM), mse=mse)
        rows.append(out)
        r, w = out["rot"], out["raw"]
        print(f"  {bits} bits/component:  ROTATED entropy {r['entropy_bits_per_atom']:.3f} b/atom, "
              f"sigma {r['sigma']:.4f} A   |   UNROTATED entropy {w['entropy_bits_per_atom']:.3f}, "
              f"sigma {w['sigma']:.4f} A   |   rotation moves MSE "
              f"{100*(r['mse']-w['mse'])/w['mse']:+.1f}%", flush=True)
    # ---- 094: the same codec, WATER-FILLED at matched bits/atom ----
    from armf_pca_matched import waterfill, quantise_cols
    print(f"\n  094: WATER-FILLED codec arm at matched bits/atom (flat rows above kept, labelled)",
          flush=True)
    print(f"  {'budget b/atom':>14}{'entropy b/atom':>16}{'sigma A':>10}{'bits/dim':>10}{'nonzero':>9}")
    wf_rows = []
    var = Ac.var(0) + 1e-12
    for b in WF_BUDGETS:
        errs, syms = [], []
        with torch.no_grad():
            for i in range(len(ds)):
                s_ = ds[i]; na = int(s_["n_atoms"])
                if na < NATOM: continue
                gb = {k: (v.to(dev) if torch.is_tensor(v) else v)
                      for k, v in collate_fn([s_]).items()}
                _, z = model(gb)
                zn = z[0].cpu().numpy().astype(np.float64)
                t = gb["coords"][0, :NATOM].cpu().numpy().astype(np.float64)
                c = (zn - mu) @ U
                bits = waterfill(var, int(round(b * NATOM)))
                dq = quantise_cols(c.copy(), bits, lo_r, hi_r)
                zq = torch.tensor(dq @ U.T + mu, dtype=z.dtype).unsqueeze(0)
                p = model.decode(zq, gb)[0, :NATOM].cpu().numpy().astype(np.float64)
                errs.append((p - t).ravel())
                syms.append(np.stack([np.round((dq[:, j] - lo_r[j]) /
                            max((hi_r[j]-lo_r[j])/max(2**int(bits[j])-1, 1), 1e-30))
                            if bits[j] > 0 else np.zeros(len(dq))
                            for j in range(dq.shape[1])], 1).astype(np.int64))
        e = np.concatenate(errs); S = np.concatenate(syms)
        sg = float(np.sqrt((e ** 2).mean()))
        bpa = ent_bits_cols(S) * (len(S) / (len(errs) * NATOM))
        bd = math.log2(sg * math.sqrt(2 * math.pi * math.e))
        nz = int((waterfill(var, int(round(b * NATOM))) > 0).sum())
        print(f"  {b:>14.2f}{bpa:>16.3f}{sg:>10.4f}{bd:>10.4f}{nz:>9}", flush=True)
        wf_rows.append(dict(budget=b, entropy_bits_per_atom=bpa, sigma=sg, bits_per_dim=bd,
                            nonzero=nz))
    json.dump(dict(flat=rows, waterfilled=wf_rows), open(f"{WR}/codec_rd.json", "w"))
    print(f"\n  -> {WR}/codec_rd.json", flush=True)
