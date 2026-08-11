#!/usr/bin/env python3
"""INBOX 091/092a: the per-system loop for armf_peer_rate.py -- ANM vs the codec at MATCHED RATE.

THE HARNESS IS NOT MINE AND IS NOT DUPLICATED HERE. armf_peer_rate.code_and_score does the coding,
allocation, quantisation, entropy and scoring for BOTH arms through one path, and its self_test()
raises SystemExit unless the two arms are bit-identical on identical input. This file is the data
plumbing only: it decides what coefficients and what basis each arm gets, and nothing about how they
are scored. That division is deliberate -- we collided last time both sides wrote one component.

WHAT EACH ARM IS
  ANM    the zero-shot physics peer. Modes from the system's own REFERENCE structure, so it sees no
         trajectory at all. Already orthogonal with near-uncorrelated coefficients -> rotate=False.
  CODEC  the trained latent. No orthogonality guarantee and no decorrelation term anywhere in the
         model -- no KL, no VQ -- and 87c measured PR = 2.00 of 8 channels on the static codec, so
         its channels are correlated by measurement, not by assumption -> rotate=True.

A SCOPE LIMIT I AM STATING RATHER THAN HIDING. code_and_score reconstructs linearly (`Cq @ basis`),
which ANM is by construction and the codec is NOT -- its decoder is a network. The codec arm
therefore uses a LEAST-SQUARES READOUT from latent to displacement, fitted on TRAIN frames of the
same system. So this compares the two REPRESENTATIONS under a common linear decoder. It is the only
way both arms pass through the identical scoring path, which is the property the self-test exists to
guarantee, and breaking that symmetry to give the codec its nonlinear decoder would reintroduce
exactly the asymmetry 090a just caught in my own rate accounting. A codec win here is a statement
about the latent; the nonlinear decoder is owed separately.

PRE-REGISTERED, BEFORE THE NUMBERS EXIST (both readings, so neither is chosen after the fact):
  CODEC WINS at matched rate -> the first peer win on the dynamics task in this project. It would
        NOT overturn 0/123: FVE and rate-distortion measure different things, and a representation
        that is cheaper per bit can still explain less variance. Saying so is the point, not a hedge.
  ANM WINS  -> the loss is now measured on BOTH axes, reconstruction and rate, and is that much more
        solid. This is the more likely outcome given 0/123 and it is worth having either way.

ALSO REPORTED, per 092a: ANM's REALISED NONZERO MODE COUNT beside its rate. Water-filling gives
zero bits to modes below the water level, so a 256-mode basis spending bits on 40 is the same shape
as PR = 2 of 8 -- an allocated width that is not the used width. If both sides show it, "the latent
is under-used" stops being a fact about the codec.
"""
import sys, os, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import armf_peer_rate as PR
from armf_peer_rate import code_and_score, self_test, BUDGETS
from armf_atlas_data import AtlasStore, sysdata, ho_frames
from armf_anm import modes as anm_modes
import armf_atlas_dm as D
from armf_atlas_dm import Codec
import armf_io

WR = D.WR
DM = int(os.environ.get("PEER_DM", "256"))
LR = os.environ.get("PEER_LR", "0.0001")
NT = int(os.environ.get("PEER_NTRAIN", "130"))
SEED = int(os.environ.get("PEER_SEED", "0"))
CUTOFF = float(os.environ.get("PEER_CUTOFF", "5.0"))
NFRAME = int(os.environ.get("PEER_NFRAME", "400"))     # frames per side, strided
RES = os.environ.get("PEER_RATE_RES", f"{WR}/peer_rate.json")
CKPT = f"{WR}/atlas_dm_ckpt/L1_n{NT}_dm{DM}_dl{DM}_lr{LR}_s{SEED}_z0.pt"
dev = D.dev


@torch.no_grad()
def codec_coeffs(mdl, d, frames):
    """Latent per frame. The Codec takes (static features, displacement) and returns the code."""
    st = torch.tensor(d["stat"], device=dev)
    out = []
    for s0 in range(0, len(frames), 8):
        blk = frames[s0:s0 + 8]
        dw = torch.tensor((blk / d["scale"]).astype(np.float32), device=dev)
        z = mdl.encode(st.unsqueeze(0).expand(len(blk), -1, -1), dw) \
            if hasattr(mdl, "encode") else None
        if z is None:
            raise SystemExit("  Codec exposes no encode(); cannot obtain latent coefficients")
        out.append(z.reshape(len(blk), -1).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


if __name__ == "__main__":
    self_test()                                  # 82c's shape: prove symmetry before scoring anything
    if not os.path.exists(CKPT):
        raise SystemExit(f"  no checkpoint at {CKPT}")
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    print(f"\n[091] ANM vs codec at matched rate, {len(ho_ids)} held-out systems, "
          f"budgets {BUDGETS} bits/atom", flush=True)

    mdl = None
    rows, failed = {}, 0
    t0 = time.time()
    for i, pdb in enumerate(ho_ids, 1):
        try:
            d = sysdata(store, have[pdb])
            if d is None:
                failed += 1; continue
            if mdl is None:
                mdl = Codec(d["stat"].shape[1], 1, DM).to(dev)
                mdl.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False))
                mdl.eval()
            N, F = d["N"], d["F"]
            stride = max(1, F // NFRAME)
            idx = np.arange(0, F, stride)[:NFRAME]
            X = ho_frames(d, 0, F)[idx].reshape(len(idx), -1)      # held-out replica, displacements
            half = len(X) // 2
            X_tr, X_ho = X[:half], X[half:]

            # ---- ANM arm: basis from the REFERENCE structure only, zero-shot ----
            V, _, _ = anm_modes(d["ref"], DM, CUTOFF)
            if V is None:
                failed += 1; continue
            B_anm = V.T                                            # (comps, 3N)
            C_anm_tr, C_anm_ho = X_tr @ B_anm.T, X_ho @ B_anm.T    # orthonormal -> projection

            # ---- CODEC arm: latent, with a least-squares readout as its linear basis ----
            Z = codec_coeffs(mdl, d, X.reshape(len(X), N, 3))
            Z_tr, Z_ho = Z[:half], Z[half:]
            B_cod, *_ = np.linalg.lstsq(Z_tr, X_tr, rcond=None)     # (comps, 3N), fitted on TRAIN

            row = {"N": N, "n_frames": len(X), "dm": DM, "anm_modes": int(B_anm.shape[0]),
                   "codec_comps": int(Z.shape[1]), "arms": {}}
            for b in BUDGETS:
                r_a = code_and_score(C_anm_tr, C_anm_ho, B_anm, X_ho, N, b, rotate=False)
                r_c = code_and_score(Z_tr, Z_ho, B_cod, X_ho, N, b, rotate=True)
                row["arms"][str(b)] = {
                    "anm":   dict(rate=r_a[0], mse=r_a[1], nonzero=r_a[2]),
                    "codec": dict(rate=r_c[0], mse=r_c[1], nonzero=r_c[2])}
            rows[pdb] = row
            a1 = row["arms"][str(BUDGETS[len(BUDGETS)//2])]
            print(f"  [{i}/{len(ho_ids)}] {pdb:10s} N={N:>6}  @{BUDGETS[len(BUDGETS)//2]:g} b/atom  "
                  f"ANM mse {a1['anm']['mse']:.4f} ({a1['anm']['nonzero']}/{DM} modes)  "
                  f"CODEC mse {a1['codec']['mse']:.4f} ({a1['codec']['nonzero']} comps)  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(ho_ids)}] {pdb}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(ho_ids), complete=(i == len(ho_ids)),
                          n_failed=failed, dm=DM, budgets=BUDGETS, ckpt=os.path.basename(CKPT))

    if not rows:
        raise SystemExit("\n  NO system scored. Reported as absent, not as a null.")
    print(f"\n=== 091: ANM vs CODEC at matched rate ({len(rows)} systems, {failed} failed) ===")
    print(f"  {'bits/atom':>10}{'ANM mse':>11}{'CODEC mse':>11}{'codec wins':>12}"
          f"{'ANM modes used':>16}")
    for b in BUDGETS:
        A = np.array([r["arms"][str(b)]["anm"]["mse"] for r in rows.values()])
        C = np.array([r["arms"][str(b)]["codec"]["mse"] for r in rows.values()])
        nz = np.array([r["arms"][str(b)]["anm"]["nonzero"] for r in rows.values()])
        print(f"  {b:>10.1f}{np.median(A):>11.4f}{np.median(C):>11.4f}"
              f"{(C < A).mean():>11.1%}{f'{np.median(nz):.0f}/{DM}':>16}")
    print(f"\n  A codec win is the first peer win on the dynamics task and does NOT overturn 0/123:\n"
          f"  FVE and rate-distortion measure different things. An ANM win means the loss is\n"
          f"  measured on both axes.", flush=True)
