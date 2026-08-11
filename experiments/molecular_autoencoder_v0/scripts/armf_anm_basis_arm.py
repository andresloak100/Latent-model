#!/usr/bin/env python3
"""INBOX 98b: give the codec ANM's basis and re-run the peer comparison.

THE QUESTION, AND WHY IT IS WORTH ONE TRAINING RUN. 0/123 says the codec loses to ANM. It has never
said WHY. If the binding constraint is BASIS QUALITY -- the learned modes are simply worse directions
than ANM's -- then handing the codec ANM's directions should close the gap, and a representation
change (a simplex prior, a different bottleneck) is worth pursuing. If it does not close the gap, the
basis is not the constraint, and a prior on the basis will not help either. **That closes the SEM
direction for the cost of one training run rather than a project.**

WHAT IS CHANGED, AND ONLY THIS. ModalCodec.basis_of(stat) -> (B,N,3,C) predicts modes from structure
alone; ANM modes are (3N,K) and reshape to (N,3,K) on the same atoms. A SUBSPACE-ALIGNMENT penalty
pulls the learned basis toward the ANM span. The architecture, the data, the optimiser and the
evaluation are untouched, so a difference is attributable to the basis and to nothing else.

The penalty is on the SUBSPACE, not the individual vectors: ||B_perp||^2 where B_perp is the learned
basis with its ANM-span component removed, normalised by ||B||^2. Mode ORDER and sign are arbitrary
in both bases, so penalising vector-by-vector would punish a correct basis for a permutation.

PRE-REGISTERED, AND 97b MAKES THE FAILURE BRANCH LIVE. 97b retracted the "not trained into its
capacity" reading: PR rose 4.5 -> 28.2 while FVE PEAKED AT PR 16.9 AND DECLINED (slope -0.00211/PR,
r=-0.797). Better-organised did not mean better. So:

  ALIGNMENT UP AND FVE UP        basis quality was the binding constraint; the representation
                                 direction is worth building
  ALIGNMENT UP AND FVE FLAT/DOWN **FAILURE CONDITION, DECLARED IN ADVANCE.** "Better organised and no
                                 better on any measured axis" is the outcome 97b says is available,
                                 and it closes the direction rather than motivating a richer prior
  ALIGNMENT FLAT                 the penalty did not bind; the run says nothing and must not be read
                                 as evidence either way -- reported as a failed intervention

The alignment fraction is logged every eval so the third branch is distinguishable from the second,
which a final-loss number alone could not do.
"""
import sys, os, json, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import armf_atlas_dm as D
from armf_atlas_data import AtlasStore, sysdata
from armf_anm import modes as anm_modes
from armf_modal_decoder import ModalCodec
import armf_io

WR = D.WR
DM = int(os.environ.get("AB_DM", "256"))
NT = int(os.environ.get("AB_NTRAIN", "130"))
LR = float(os.environ.get("AB_LR", "1e-4"))
LAM = float(os.environ.get("AB_LAM", "1.0"))      # 0 = the control arm
STEPS = int(os.environ.get("AB_STEPS", "30000"))
EVERY = int(os.environ.get("AB_EVERY", "2500"))
CUTOFF = float(os.environ.get("AB_CUTOFF", "5.0"))
SEED = int(os.environ.get("AB_SEED", "0"))
FPS = D.FRAMES_PER_STEP
RES = os.environ.get("AB_RES", f"{WR}/anm_basis_arm_lam{LAM:g}.json")
dev = D.dev


def anm_span(d, k):
    """ANM modes for one system as (N,3,k), orthonormal, from the REFERENCE structure only."""
    V, _, _ = anm_modes(d["ref"], k, CUTOFF)
    if V is None: return None
    return torch.tensor(V.reshape(d["N"], 3, -1), dtype=torch.float32, device=dev)


def align_loss(Bm, A):
    """Fraction of the learned basis's energy OUTSIDE the ANM span. Subspace-level, so a permutation
    or sign flip of the modes costs nothing -- both bases order modes arbitrarily."""
    B, N, _, C = Bm.shape
    X = Bm.reshape(B, N * 3, C)
    Q = A.reshape(1, N * 3, -1).expand(B, -1, -1)
    Qo, _ = torch.linalg.qr(Q)                       # orthonormal basis of the ANM span
    proj = Qo @ (Qo.transpose(1, 2) @ X)
    num = ((X - proj) ** 2).sum()
    den = (X ** 2).sum() + 1e-12
    return num / den


if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED + 1)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    tr_ids = [p for p in man["train_ordered"] if p in have][:NT]
    ho_ids = [p for p in man["heldout"] if p in have]
    tr = [x for x in (sysdata(store, have[p]) for p in tr_ids) if x is not None]
    HOt = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None][:24]
    print(f"[98b] ANM-basis arm: lambda={LAM:g} DM={DM} n_train={len(tr)} steps={STEPS}", flush=True)
    print(f"  lambda=0 is the control; the ONLY difference between arms is the alignment penalty",
          flush=True)

    A = {}
    for d in tr:
        a = anm_span(d, DM)
        if a is not None: A[d["pdb"]] = a
    print(f"  ANM span precomputed for {len(A)}/{len(tr)} train systems", flush=True)

    mdl = ModalCodec(tr[0]["stat"].shape[1], 1, DM, tie_encoder=True).to(dev)
    opt = torch.optim.Adam(mdl.parameters(), LR)
    trace, t0 = [], time.time()
    for st in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, st / 1000)
        d = tr[np.random.randint(len(tr))]
        idx = np.random.randint(0, 2 * d["F"], FPS)
        S = torch.tensor(d["stat"], device=dev).unsqueeze(0).expand(FPS, -1, -1)
        Dw = torch.tensor(D.train_frames(d, idx).reshape(FPS, d["N"], 3) / d["scale"], device=dev)
        opt.zero_grad()
        rec = ((mdl(S, Dw) - Dw) ** 2).mean()
        pen = torch.zeros((), device=dev)
        if LAM > 0 and d["pdb"] in A:
            pen = align_loss(mdl.basis_of(S[:1]), A[d["pdb"]])
        (rec + LAM * pen).backward(); opt.step()
        if st % EVERY == 0:
            mdl.eval()
            with torch.no_grad():
                v = float(np.mean([D.fve_model(mdl, x) for x in HOt]))
                offs = []
                for dd in tr[:8]:
                    if dd["pdb"] in A:
                        SS = torch.tensor(dd["stat"], device=dev).unsqueeze(0)
                        offs.append(float(align_loss(mdl.basis_of(SS), A[dd["pdb"]])))
            mdl.train()
            off = float(np.mean(offs)) if offs else float("nan")
            trace.append(dict(step=st, fve=v, off_span=off, aligned=1.0 - off,
                              rec=float(rec), secs=time.time() - t0))
            print(f"    step {st:>6}: FVE {v:+.4f}  ALIGNED {100*(1-off):5.1f}% of basis energy "
                  f"inside the ANM span  ({time.time()-t0:.0f}s)", flush=True)
            armf_io.dump_rows(RES, trace, n_expected=STEPS // EVERY,
                              complete=(st == STEPS), n_failed=0, lam=LAM, dm=DM, n_train=len(tr))
    if trace:
        a0, a1 = trace[0]["aligned"], trace[-1]["aligned"]
        f0 = max(t["fve"] for t in trace[:max(1, len(trace)//4)])
        f1 = max(t["fve"] for t in trace[-max(1, len(trace)//4):])
        print(f"\n=== 98b (lambda={LAM:g}) ===")
        print(f"  alignment {100*a0:.1f}% -> {100*a1:.1f}%   best FVE {f0:+.4f} -> {f1:+.4f}")
        if a1 - a0 < 0.05:
            print("  -> THE PENALTY DID NOT BIND. This run says nothing about basis quality either\n"
                  "     way and must not be read as evidence; reported as a failed intervention.")
        elif f1 > f0 + 0.01:
            print("  -> ALIGNMENT UP AND FVE UP: basis quality was the binding constraint.")
        else:
            print("  -> ALIGNMENT UP, FVE FLAT/DOWN. This is the PRE-REGISTERED FAILURE CONDITION:\n"
                  "     better organised and no better on any measured axis. A prior on the basis\n"
                  "     will not help, and the representation direction closes here.")
