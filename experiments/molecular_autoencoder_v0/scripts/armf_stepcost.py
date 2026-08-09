#!/usr/bin/env python3
"""INBOX 68a: MEASURE ms/step against structure size, do not scale it.

067 priced 1M structures at 150 ms/step -- measured at the OLD distribution (median 81 residues).
068 argues AFDB's median of 277 makes that optimistic by >10x via O(R^2) attention, giving ~122 GPU-h
as a FLOOR. Both are extrapolations from an assumed exponent.

This measures the exponent instead. processed_big spans 22-385 residues (median 183) and already
brackets AFDB's median of 277, so the scaling can be fitted on data already on disk -- no AFDB
preprocessing needed for this question. Fit log(ms/step) on log(R), then evaluate the fitted law at
AFDB's actual length distribution rather than at its median, since E[R^a] != (E[R])^a for a != 1 and
the distribution is right-skewed (068's own point, applied to the fit rather than to a guess)."""
import sys, os, json, time, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, ROOT)
from molae.config import ExperimentConfig
from molae.dataset import ProteinStructureDataset, collate_fn
from molae.model_equivariant import make_autoencoder
from molae import utils

CFG = os.environ.get("STEPCOST_CFG",
                     f"{os.environ['WR']}/results/ladder_direct3m_n2272/config.yaml")
BANDS = [(20, 60), (60, 100), (100, 150), (150, 220), (220, 300), (300, 400)]
BATCH = int(os.environ.get("STEPCOST_BATCH", "16"))
NSTEP = int(os.environ.get("STEPCOST_STEPS", "30"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bench(ds, idx, batch, nstep, model, opt):
    """Forward+backward, timed. Warmup excluded; CUDA synchronised so the timing is real."""
    if len(idx) < batch: return None
    times = []
    for s in range(nstep + 3):
        pick = [idx[(s * batch + j) % len(idx)] for j in range(batch)]
        gb = {k: (v.to(dev) if torch.is_tensor(v) else v)
              for k, v in collate_fn([ds[i] for i in pick]).items()}
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad()
        pred, _ = model(gb)
        loss = ((pred - gb["coords"]) ** 2).mean()
        loss.backward(); opt.step()
        if dev.type == "cuda": torch.cuda.synchronize()
        if s >= 3: times.append(time.time() - t0)     # drop warmup
    return float(np.median(times)) * 1000.0


if __name__ == "__main__":
    cfg = ExperimentConfig.from_yaml(CFG)
    proc = os.environ.get("STEPCOST_PROC", f"{ROOT}/data/processed_big")
    paths = sorted(glob.glob(f"{proc}/*.npz"))
    print(f"[stepcost] {len(paths)} structures from {os.path.basename(proc)}, "
          f"batch={BATCH}, {NSTEP} timed steps/band, device={dev}", flush=True)
    ds = ProteinStructureDataset(paths)
    R = np.array([int(ds[i]["res_pos"].max()) + 1 for i in range(len(ds))])
    print(f"  residues {R.min()}-{R.max()}, median {int(np.median(R))}", flush=True)

    model = make_autoencoder(cfg.model).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-4)
    print(f"\n  {'band':>12}{'n':>6}{'medR':>7}{'ms/step':>10}{'ms/struct':>11}", flush=True)
    rows = []
    for lo, hi in BANDS:
        idx = [i for i in range(len(ds)) if lo <= R[i] < hi]
        ms = bench(ds, idx, BATCH, NSTEP, model, opt)
        if ms is None:
            print(f"  {f'{lo}-{hi}':>12}{len(idx):>6}{'-':>7}{'too few':>10}", flush=True); continue
        mr = float(np.median(R[idx]))
        rows.append((mr, ms, len(idx)))
        print(f"  {f'{lo}-{hi}':>12}{len(idx):>6}{mr:>7.0f}{ms:>10.1f}{ms/BATCH:>11.2f}", flush=True)

    if len(rows) >= 3:
        r = np.array([x[0] for x in rows]); m = np.array([x[1] for x in rows])
        sl = np.polyfit(np.log(r), np.log(m), 1)
        print(f"\n  FITTED EXPONENT a in ms ~ R^a : {sl[0]:.2f}", flush=True)
        print(f"    (068 assumed a=2 from O(R^2) attention; 067 implicitly assumed a=0)", flush=True)
        # evaluate over AFDB's ACTUAL distribution, not its median
        idxf = f"{os.environ['WR']}/afdb_lengths.npy"
        if os.path.exists(idxf):
            L = np.load(idxf)
            pred_ms = np.exp(sl[1]) * np.mean(L.astype(float) ** sl[0])
            base = np.exp(sl[1]) * np.median(R) ** sl[0]
            print(f"    AFDB E[R^a] over n={L.size}: predicted {pred_ms:.0f} ms/step "
                  f"({pred_ms/base:.1f}x the current-distribution cost)", flush=True)
            for ep in (4,):
                steps = ep * 1_000_000 / BATCH
                print(f"    {ep} epochs over 1M at batch {BATCH}: {steps:,.0f} steps -> "
                      f"{steps*pred_ms/3.6e6:.1f} GPU-h", flush=True)
        json.dump({"rows": rows, "exponent": float(sl[0]), "batch": BATCH},
                  open(f"{os.environ['WR']}/stepcost.json", "w"))
