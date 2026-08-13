#!/usr/bin/env python3
"""The two scaling sweeps, run sequentially in one job.

    parameters   width and depth together, at fixed latent
    latent L     at fixed width and depth

Both pass through a shared centre configuration, so the two curves are commensurable. Every
configuration sees the SAME number of training tokens, the SAME data in the SAME order, and the SAME
held-out set -- so a difference between rows is a difference in capacity and not in budget or in
data. The seed is fixed and printed.

Two baselines are computed on the identical held-out crops and printed beside every row:

    CENTROID     predict the crop centroid for every atom  -- the zero-information floor
    MEAN_SHAPE   predict the training mean position for each atom INDEX -- what is achievable with
                 no per-structure information at all

A model that does not clearly beat MEAN_SHAPE has learned nothing structure-specific, whatever its
RMSD looks like in isolation. That comparison is the reason both baselines exist.
"""
import os, sys, json, time, math
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from model import StructAE, count_params

SHARD = os.environ.get("SAE_SHARD", "/network/scratch/j/jacob-junqi.tian/static-ae-v1/shard")
RES = os.environ.get("SAE_RES", "/network/scratch/j/jacob-junqi.tian/static-ae-v1/results.json")
STEPS = int(os.environ.get("SAE_STEPS", "6000"))
BATCH = int(os.environ.get("SAE_BATCH", "64"))
LR = float(os.environ.get("SAE_LR", "3e-4"))
NHELD = int(os.environ.get("SAE_NHELD", "4096"))
SEED = int(os.environ.get("SAE_SEED", "0"))
ONLY = os.environ.get("SAE_ONLY", "")           # optional substring filter over config tags
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The shared centre appears in BOTH sweeps and is trained once.
CENTRE = dict(d_model=256, depth=6, latent=128)
PARAM_SWEEP = [dict(d_model=128, depth=4, latent=128),
               dict(d_model=256, depth=6, latent=128),
               dict(d_model=384, depth=8, latent=128),
               dict(d_model=512, depth=12, latent=128)]
LATENT_SWEEP = [dict(d_model=256, depth=6, latent=L) for L in (8, 32, 128, 512, 2048)]


def tag(c):
    return f"d{c['d_model']}_L{c['depth']}_z{c['latent']}"


def rmsd(pred, true):
    """Angstrom RMSD per structure: sqrt(mean over atoms of squared 3-D distance). No superposition."""
    return torch.sqrt(((pred - true) ** 2).sum(-1).mean(-1))


def evaluate(mdl, E, X, bs=256):
    mdl.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(E), bs):
            e = torch.from_numpy(np.ascontiguousarray(E[i:i + bs])).long().to(dev)
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(dev)
            x = x - x.mean(1, keepdim=True)
            out.append(rmsd(mdl(e, x), x).cpu())
    mdl.train()
    return torch.cat(out).numpy()


def train_one(cfg, Etr, Xtr, Eho, Xho, na, mean_shape):
    torch.manual_seed(SEED)
    mdl = StructAE(na=na, **cfg).to(dev)
    npar = count_params(mdl)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
    # The SAME data order for every configuration, so budget is held exactly constant.
    rng = np.random.default_rng(SEED)
    order = rng.integers(0, len(Etr), size=(STEPS, BATCH))
    t0, loss_hist = time.time(), []
    for s in range(STEPS):
        idx = np.sort(order[s])
        e = torch.from_numpy(np.ascontiguousarray(Etr[idx])).long().to(dev)
        x = torch.from_numpy(np.ascontiguousarray(Xtr[idx])).float().to(dev)
        x = x - x.mean(1, keepdim=True)
        loss = ((mdl(e, x) - x) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step(); sched.step()
        loss_hist.append(float(loss))
        if (s + 1) % max(1, STEPS // 6) == 0:
            print(f"      step {s+1}/{STEPS}  train MSE {np.mean(loss_hist[-200:]):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    r = evaluate(mdl, Eho, Xho)
    return dict(cfg=cfg, tag=tag(cfg), params=npar, steps=STEPS, batch=BATCH,
                tokens=STEPS * BATCH * na,
                rmsd_median=float(np.median(r)), rmsd_mean=float(r.mean()),
                rmsd_p10=float(np.percentile(r, 10)), rmsd_p90=float(np.percentile(r, 90)),
                train_mse_final=float(np.mean(loss_hist[-200:])),
                seconds=float(time.time() - t0))


if __name__ == "__main__":
    meta = json.load(open(SHARD + "_meta.json"))
    na, n = meta["NA"], meta["n"]
    E = np.load(SHARD + "_elem.npy", mmap_mode="r")
    X = np.load(SHARD + "_xyz.npy", mmap_mode="r")
    E, X = E[:n], X[:n]
    print(f"[ladder] shard {n:,} structures x {na} atoms   device={dev}", flush=True)
    print(f"  steps={STEPS} batch={BATCH} lr={LR} seed={SEED}  "
          f"tokens per config = {STEPS*BATCH*na:,}", flush=True)
    # Held-out is the LAST NHELD rows, and the shard order was shuffled at build time with a
    # recorded seed, so this is a random split without a second shuffle to get wrong.
    Eho, Xho = np.array(E[-NHELD:]), np.array(X[-NHELD:])
    Etr, Xtr = E[:-NHELD], X[:-NHELD]
    print(f"  train {len(Etr):,}   held-out {len(Eho):,}", flush=True)

    Xho_c = Xho - Xho.mean(1, keepdims=True)
    # MEAN_SHAPE uses TRAIN structures only, so the baseline never sees held-out geometry.
    ns = min(20000, len(Xtr))
    sample = np.array(Xtr[:ns])
    mean_shape = (sample - sample.mean(1, keepdims=True)).mean(0)          # (na, 3)
    base = {
        "CENTROID": float(np.median(np.sqrt((Xho_c ** 2).sum(-1).mean(-1)))),
        "MEAN_SHAPE": float(np.median(np.sqrt(((Xho_c - mean_shape) ** 2).sum(-1).mean(-1)))),
    }
    print(f"  BASELINES on the identical held-out crops:", flush=True)
    for k, v in base.items():
        print(f"    {k:>10}  median RMSD {v:7.3f} A", flush=True)

    seen, rows = set(), []
    if os.path.exists(RES):
        old = json.load(open(RES))
        rows = old.get("rows", [])
        seen = {r["tag"] for r in rows}
        print(f"  resuming: {len(seen)} configurations already done", flush=True)

    plan = [("params", c) for c in PARAM_SWEEP] + \
           [("latent", c) for c in LATENT_SWEEP if tag(c) != tag(CENTRE)]
    for i, (sweep, cfg) in enumerate(plan, 1):
        t = tag(cfg)
        if t in seen or (ONLY and ONLY not in t):
            print(f"  [{i}/{len(plan)}] {t}: skip", flush=True)
            continue
        print(f"  [{i}/{len(plan)}] {sweep} sweep -- {t}", flush=True)
        try:
            r = train_one(cfg, Etr, Xtr, Eho, Xho, na, mean_shape)
        except Exception as ex:
            print(f"      FAIL {type(ex).__name__}: {ex}", flush=True)
            continue
        r["sweep"] = sweep
        rows.append(r); seen.add(t)
        print(f"      {t}: {r['params']:>12,} params  held-out median RMSD "
              f"{r['rmsd_median']:7.3f} A   ({r['seconds']:.0f}s)", flush=True)
        json.dump(dict(baselines=base, na=na, n=n, steps=STEPS, batch=BATCH, lr=LR, seed=SEED,
                       rows=rows), open(RES, "w"), indent=1)

    print(f"\n=== STATIC AUTOENCODER SCALING ({len(rows)} configurations) ===")
    print(f"  held-out {NHELD:,} structures, {na} atoms each, no superposition, "
          f"{STEPS*BATCH*na:,} training tokens each")
    for k, v in base.items():
        print(f"  baseline {k:>10}: {v:7.3f} A")
    for sweep in ("params", "latent"):
        rs = sorted([r for r in rows if r.get("sweep") == sweep] +
                    ([r for r in rows if r["tag"] == tag(CENTRE) and r.get("sweep") != sweep]
                     if sweep == "latent" else []),
                    key=lambda r: (r["params"] if sweep == "params" else r["cfg"]["latent"]))
        if not rs: continue
        print(f"\n  --- {sweep.upper()} SWEEP ---")
        print(f"  {'config':>18}{'params':>13}{'latent':>8}{'RMSD':>9}{'vs prev':>9}"
              f"{'vs MEAN_SHAPE':>15}")
        prev = None
        for r in rs:
            d = f"{100*(prev-r['rmsd_median'])/prev:>7.1f}%" if prev else " " * 9
            frac = r["rmsd_median"] / base["MEAN_SHAPE"]
            print(f"  {r['tag']:>18}{r['params']:>13,}{r['cfg']['latent']:>8}"
                  f"{r['rmsd_median']:>8.3f}A{d}{frac:>14.3f}x")
            prev = r["rmsd_median"]
    print(f"\n  Read against SPEC.md's four pre-registered outcomes: clean scaling, wall,")
    print(f"  latent-bound, parameter-bound. 'vs MEAN_SHAPE' below 1.0 is the minimum bar --")
    print(f"  above it the model has not beaten a per-index average that uses no input at all.")
