#!/usr/bin/env python3
"""INBOX 66c/70a: a LIKELIHOOD SURROGATE and a real rate-distortion curve.

WHY. molae/ has no kl, elbo, log_prob, nll, logvar, reparam or variational anywhere, and
`compression_ratio` counts FLOATS. Floats are not a rate: a rate is bits at a stated quantisation, and
without one the codec cannot be compared to any published codec, nor can anyone say whether this
latent is fit to diffuse in. Both are measurable on the deterministic checkpoints already on disk --
no retraining, no GPU, no new corpus.

WHAT IS AND IS NOT MEASURED. The decoder is deterministic, so there is no likelihood to read off it.
The standard surrogate is a Gaussian decoder: treat the output as the mean and fit one sigma, then
NLL = 0.5*log(2*pi*sigma^2) + (x-mu)^2/(2*sigma^2) per coordinate. That is a SURROGATE and it is
stated as one -- it prices the reconstruction error as a likelihood, and says nothing about whether
the latent is distributed in a way a diffusion model could sample.

SIGMA IS FIT ON A DISJOINT HALF. Fitting sigma on the same residuals it then scores is optimistic by
construction -- the MLE sigma minimises exactly that NLL. Structures are split in half: sigma from
one, NLL reported on the other. Both numbers print, so the size of the optimism is visible rather
than argued about."""
import sys, os, json, glob, math
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE + "/.."
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from molae.config import ExperimentConfig
from molae.dataset import ProteinStructureDataset, collate_fn
from molae.model_equivariant import make_autoencoder
from molae.metrics import all_atom_rmsd            # KABSCH-ALIGNED, the project's own metric
from molae import utils

WR = os.environ["WR"]
CFG = os.environ.get("LIK_CFG", f"{WR}/results/ladder_direct3m_n2272/config.yaml")
CKPT = os.environ.get("LIK_CKPT", f"{WR}/results/ladder_direct3m_n2272/final.pt")
NSTRUCT = int(os.environ.get("LIK_N", "120"))
BITS = [2, 3, 4, 6, 8, 12, 16, 32]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantise(z, bits):
    """Uniform scalar quantisation over the per-tensor range. 32 bits == passthrough (no quantisation),
    which is the rate the project has been implicitly reporting when it counts floats."""
    if bits >= 32: return z, 32.0
    lo, hi = z.min(), z.max()
    levels = 2 ** bits - 1
    step = (hi - lo) / max(levels, 1)
    q = torch.round((z - lo) / step) * step + lo
    return q, float(bits)


@torch.no_grad()
def run():
    cfg = ExperimentConfig.from_yaml(CFG)
    splits = utils.load_json(f"{ROOT}/{cfg.data.splits_file}")
    keys = list(splits.get("val") or [])
    proc = f"{ROOT}/{cfg.data.processed_dir}"
    paths = [f"{proc}/{k}.npz" for k in keys if os.path.exists(f"{proc}/{k}.npz")][:NSTRUCT]
    ds = ProteinStructureDataset(paths)
    model = make_autoencoder(cfg.model).to(dev)
    utils.load_checkpoint(CKPT, model, None, map_location=dev); model.eval()
    print(f"[likelihood] {len(ds)} held-out structures, device={dev}", flush=True)

    half = len(ds) // 2
    res_fit, res_eval, dims_fit, dims_eval = [], [], 0, 0
    rows = []
    for i in range(len(ds)):
        s = ds[i]
        gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn([s]).items()}
        na = int(s["n_atoms"]); nres = int(s["res_pos"].numpy().max()) + 1
        pred, z = model(gb)
        p = pred[0, :na].cpu().numpy().astype(np.float64)
        t = gb["coords"][0, :na].cpu().numpy().astype(np.float64)
        e = (p - t).ravel()
        (res_fit if i < half else res_eval).append(e)
        if i < half: dims_fit += e.size
        else: dims_eval += e.size
        # rate-distortion: quantise the latent, decode, measure RMSD
        if i >= half:
            row = {"n_atoms": na, "n_res": nres}
            for b in BITS:
                zq, bb = quantise(z, b)
                pq = model.decode(zq, gb) if hasattr(model, "decode") else None
                if pq is None:
                    row[str(b)] = None; continue
                pq = pq[0, :na].cpu().numpy().astype(np.float64)
                # ALIGNED, matching molae.metrics.all_atom_rmsd (= kabsch_rmsd_numpy). Raw and
                # aligned RMSD differ by ~3x here (1.589 vs 0.487 on one structure), so reporting raw
                # against a recorded aligned number would be one-name-two-things in the distortion
                # axis. The NLL below stays on RAW residuals deliberately: alignment is a per-structure
                # rigid fit, i.e. extra free parameters the model does not have, so a likelihood must
                # not be paid that discount.
                row[str(b)] = float(all_atom_rmsd(pq, t))
            zfl = int(np.prod(z.shape))
            row["latent_floats"] = zfl
            row["bits_per_atom"] = {str(b): zfl * b / na for b in BITS}
            rows.append(row)

    ef = np.concatenate(res_fit); ee = np.concatenate(res_eval)
    sig_fit = float(np.sqrt((ef ** 2).mean()))
    def nll(err, sigma):
        return 0.5 * math.log(2 * math.pi * sigma ** 2) + float((err ** 2).mean()) / (2 * sigma ** 2)
    n_in = nll(ef, sig_fit); n_out = nll(ee, sig_fit)
    sig_self = float(np.sqrt((ee ** 2).mean())); n_self = nll(ee, sig_self)
    print(f"\n=== GAUSSIAN-DECODER SURROGATE (66c) ===", flush=True)
    print(f"  sigma fitted on {dims_fit:,} coords from the first half: {sig_fit:.4f} A", flush=True)
    print(f"  NLL on the FIT half   {n_in:8.4f} nats/dim = {n_in/math.log(2):7.4f} bits/dim", flush=True)
    print(f"  NLL on the HELD half  {n_out:8.4f} nats/dim = {n_out/math.log(2):7.4f} bits/dim  "
          f"<- the honest one", flush=True)
    print(f"  (sigma refit on the held half would give {n_self/math.log(2):.4f} bits/dim -- the "
          f"optimism from fitting and scoring on one set is {(n_out-n_self)/math.log(2):+.4f} bits/dim)",
          flush=True)
    json.dump({"sigma": sig_fit, "bits_per_dim_held": n_out / math.log(2),
               "bits_per_dim_fit": n_in / math.log(2), "rows": rows},
              open(f"{WR}/likelihood.json", "w"))

    if rows and rows[0].get("2") is not None:
        print(f"\n=== RATE-DISTORTION, rate in BITS at uniform scalar quantisation (66c) ===", flush=True)
        print(f"  {'bits/scalar':>12}{'bits/atom':>12}{'median aligned RMSD A':>24}", flush=True)
        for b in BITS:
            d = [r[str(b)] for r in rows if r.get(str(b)) is not None]
            bpa = float(np.median([r["bits_per_atom"][str(b)] for r in rows]))
            if d: print(f"  {b:>12}{bpa:>12.2f}{np.median(d):>24.3f}", flush=True)
    else:
        print(f"\n  RATE-DISTORTION NOT COMPUTED: the model exposes no decode(z, batch) entry point, "
              f"so the latent cannot be re-decoded after quantisation without a forward pass that "
              f"re-encodes. Reported as absent rather than approximated.", flush=True)


if __name__ == "__main__":
    run()
