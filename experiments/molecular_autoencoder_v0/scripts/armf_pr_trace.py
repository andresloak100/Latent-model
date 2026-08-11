#!/usr/bin/env python3
"""INBOX 88d: is the latent still SPREADING when the stopping rule fires?

THE QUESTION. 87b measured PR ~15 at DM=64, 256 and 512 alike -- four nominal widths, one effective
width. 87c measured PR = 2.00 of 8 on the static codec. 085 measured a loss to classical transform
coding. And 81d showed "more data hurts" is a statement about a FIXED STEP BUDGET, because arms stop
on plateau at a roughly fixed step count regardless of dataset size. If PR is still climbing when the
plateau rule fires, those are not four findings -- they are one stopping rule, and the architecture
has never been observed at its capacity.

DIRECT EVIDENCE THAT THIS IS LIVE: the n50/DM16 seed-1 arm from 10337194 hit its cap at 54,772 steps
STILL IMPROVING and was correctly voided. At least one arm was limited by the step budget rather than
by the architecture.

WHY A SEPARATE SCRIPT. armf_atlas_dm.train() has neither the train nor the held-out systems in scope
at the eval point, so tracing PR inside it would mean threading state through the shared training
path that every arm uses. This imports Codec, participation_ratio, fve_model and the data path from
that module rather than re-deriving them -- a second Codec or a second PR under one name is the
defect this project has caught six times.

WHAT IS DEFEATED, AND WHY THAT AND NOT THE CAP. The existing n130/DM512 arm stopped at 35,000 by
PLATEAU, not by the cap, so raising the cap alone would change nothing. The plateau rule is the thing
under test, so it is disabled and the arm runs to the script's own 90,000 ceiling with PR logged at
every EVAL_EVERY.

READING, PRE-REGISTERED:
  PR still rising at the cap        -> the stopping rule, not the architecture, set the effective
                                       width. 87a/87b/87c/085 become one diagnosis and it is
                                       addressable.
  PR flat long before the cap       -> the architecture reaches ~15 and stays there. The width
                                       finding stands on its own and the stopping rule is exonerated.
"""
import sys, os, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import armf_atlas_dm as D
from armf_atlas_dm import Codec, participation_ratio, fve_model, train_frames
from armf_atlas_data import AtlasStore, sysdata

WR = D.WR
DM = int(os.environ.get("PRT_DM", "512"))
NTRAIN = int(os.environ.get("PRT_NTRAIN", "130"))
LR = float(os.environ.get("PRT_LR", "3e-5"))
MAXSTEPS = int(os.environ.get("PRT_MAXSTEPS", "90000"))
EVAL_EVERY = int(os.environ.get("PRT_EVAL", "2500"))
SEED = int(os.environ.get("PRT_SEED", "0"))
RES = os.environ.get("PRT_RES", f"{WR}/pr_trace_dm{DM}_n{NTRAIN}.json")
dev = D.dev


if __name__ == "__main__":
    print(f"[88d] PR trace: DM={DM} n_train={NTRAIN} lr={LR:g} to {MAXSTEPS} steps, "
          f"PLATEAU RULE DISABLED, PR every {EVAL_EVERY}", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have][:D.NHO_EVAL if hasattr(D, "NHO_EVAL") else 24]
    tr_ids = [p for p in man["train_ordered"] if p in have][:NTRAIN]
    HOt = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    tr = [x for x in (sysdata(store, have[p]) for p in tr_ids) if x is not None]
    print(f"  {len(tr)} train systems, {len(HOt)} held-out for the plateau signal", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED + 1)
    mdl = Codec(tr[0]["stat"].shape[1], 1, DM).to(dev)
    opt = torch.optim.Adam(mdl.parameters(), LR)
    WARMUP = D.WARMUP if hasattr(D, "WARMUP") else 1000
    FPS = D.FRAMES_PER_STEP
    trace, t0 = [], time.time()
    for st in range(1, MAXSTEPS + 1):
        for g_ in opt.param_groups:
            g_["lr"] = LR * min(1.0, st / WARMUP)
        d = tr[np.random.randint(len(tr))]
        idx = np.random.randint(0, 2 * d["F"], FPS)
        S = torch.tensor(d["stat"], device=dev).unsqueeze(0).expand(FPS, -1, -1)
        Dw = torch.tensor(train_frames(d, idx).reshape(FPS, d["N"], 3) / d["scale"], device=dev)
        opt.zero_grad(); ((mdl(S, Dw) - Dw) ** 2).mean().backward(); opt.step()
        if st % EVAL_EVERY == 0:
            mdl.eval()
            v = float(np.mean([fve_model(mdl, x) for x in HOt]))
            p = participation_ratio(mdl, tr[:min(len(tr), len(HOt))], HOt, DM)
            mdl.train()
            trace.append(dict(step=st, fve=v, pr=p["pr"], pr_frac=p["pr_frac"],
                              ident_frac=p["ident_frac"], censored=p["censored"],
                              secs=time.time() - t0))
            print(f"    step {st:>6}: FVE {v:+.4f}  PR {p['pr']:5.1f}/{DM} "
                  f"({100*p['pr']/DM:4.1f}%)  ident {p['ident_frac']:.0%}"
                  f"{'  [PR CENSORED: n_obs < 2*DM]' if p['censored'] else ''}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            tmp = RES + ".tmp"
            json.dump(dict(dm=DM, n_train=NTRAIN, lr=LR, seed=SEED, maxsteps=MAXSTEPS,
                           plateau_disabled=True, trace=trace), open(tmp, "w"))
            os.replace(tmp, RES)

    if len(trace) >= 8:
        pr = np.array([t["pr"] for t in trace]); fv = np.array([t["fve"] for t in trace])
        q = len(trace) // 4
        print(f"\n=== 88d VERDICT ===", flush=True)
        print(f"  PR   first quarter {pr[:q].mean():.1f} -> last quarter {pr[-q:].mean():.1f}",
              flush=True)
        print(f"  FVE  first quarter {fv[:q].mean():+.4f} -> last quarter {fv[-q:].mean():+.4f}",
              flush=True)
        rise = (pr[-q:].mean() - pr[-2*q:-q].mean()) / max(pr[-2*q:-q].mean(), 1e-9)
        print(f"  PR change over the FINAL half: {100*rise:+.1f}%", flush=True)
        print(f"  -> {'PR IS STILL RISING AT THE CAP: the stopping rule set the effective width, not the architecture' if rise > 0.02 else 'PR IS FLAT AT THE CAP: the architecture reaches this width and stays; the stopping rule is exonerated'}",
              flush=True)
