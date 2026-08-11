#!/usr/bin/env python3
"""INBOX 099: the per-system loop for armf_anm_orthogonal.py -- does the codec explain variance ANM
CANNOT reach?

WHY THIS IS A DIFFERENT QUESTION FROM 0/123. That result says the codec explains less TOTAL variance
than ANM. It has never said whether the codec explains variance ANM cannot reach. One number cannot
separate those claims, and the second is the one a peer win would have to be made of.

THE MACHINERY IS NOT MINE AND IS NOT DUPLICATED. armf_anm_orthogonal.py owns the projector, the
residual energy, the scoring and the ceiling basis; its self_test() asserts that an ANM reconstruction
scores EXACTLY zero inside r = d - dVV' and RAISES rather than warns, because a drifting projector
makes every number from the file meaningless. This file decides only what data each arm sees.

THREE ROWS PER SYSTEM, ALWAYS TOGETHER (099's instruction, and it is Family B):
  ANM       exactly 0 by arithmetic -- the FLOOR, and a live check that the projector still holds
  CEILING   PCA fitted INSIDE the residual on TRAIN replicas 0+1, scored on held-out replica 2
  CODEC     the trained arm
A bare codec percentage without the ceiling is a count against an unknown maximum: the residual is
local, high-frequency and partly thermal, so some of it is unpredictable by anything.

TWO CONTRACT DETAILS, both of which would have produced silently wrong numbers:
  - `frames(a,b)` must return RAW coordinates. armf_atlas_data.ho_frames already subtracts mu, and
    residual_energy/orthogonal_fve subtract it again, so handing them ho_frames would double-centre.
    This reads the memmap directly.
  - residual_pca_basis accumulates a (3N,3N) Gram: 80 GB at N=33,377. For large systems the basis is
    obtained from the FRAME-space Gram instead, which spans the identical subspace, and the two are
    VERIFIED against each other on a small system rather than asserted equal.
"""
import sys, os, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import armf_anm_orthogonal as AO
from armf_anm_orthogonal import residual_energy, orthogonal_fve, residual_pca_basis, self_test
from armf_atlas_data import AtlasStore, sysdata
from armf_anm import modes as anm_modes
import armf_atlas_dm as D
from armf_atlas_dm import Codec
import armf_io

WR = D.WR
K_ANM = int(os.environ.get("AO_K", "256"))
K_CEIL = int(os.environ.get("AO_KCEIL", "256"))
CUTOFF = float(os.environ.get("AO_CUTOFF", "5.0"))
NTRF = int(os.environ.get("AO_NTRAIN_FRAMES", "600"))
DM = int(os.environ.get("AO_DM", "256"))
LR = os.environ.get("AO_LR", "0.0001")
NT = int(os.environ.get("AO_NTRAIN", "130"))
SEED = int(os.environ.get("AO_SEED", "0"))
GRAM_MAX = int(os.environ.get("AO_GRAM_MAX", "30000"))   # 3N above this -> frame-space route
RES = os.environ.get("AO_RES", f"{WR}/anm_orthogonal.json")
CKPT = f"{WR}/atlas_dm_ckpt/L1_n{NT}_dm{DM}_dl{DM}_lr{LR}_s{SEED}_z0.pt"
dev = D.dev


class RawFrames:
    """`frames(a,b)` over RAW coordinates of one replica -- no centring, per the machinery's
    contract. Memmapped; nothing large retained."""
    def __init__(self, path, rep, n):
        self.path, self.rep, self.n = path, rep, n
    def __call__(self, a, b):
        arr = np.load(self.path, mmap_mode="r")
        return np.asarray(arr[self.rep, a:b]).astype(np.float64).reshape(b - a, -1)


class StridedTrain:
    """TRAIN frames (replicas 0+1) at a stride, raw. Used only to FIT the ceiling basis."""
    def __init__(self, path, F, n_want):
        self.path, self.F = path, F
        step = max(1, (2 * F) // n_want)
        self.idx = np.arange(0, 2 * F, step)[:n_want]
        self.n = len(self.idx)
    def __call__(self, a, b):
        arr = np.load(self.path, mmap_mode="r")
        sel = self.idx[a:b]
        return np.stack([np.asarray(arr[t // self.F, t % self.F]).ravel() for t in sel]).astype(np.float64)


def ceiling_basis_framespace(tr, V, mu, k):
    """Top-k PCA directions inside the ANM-orthogonal subspace, via the FRAME-space Gram.

    residual_pca_basis forms a (3N,3N) coordinate Gram, which is 80 GB at N=33,377. R R' is (F,F)
    instead and spans the identical right-singular subspace. Verified numerically against
    residual_pca_basis on a small system below, not asserted."""
    R = np.concatenate([AO._perp(tr(s, min(s + AO.CHUNK, tr.n)) - mu, V)
                        for s in range(0, tr.n, AO.CHUNK)])
    G = R @ R.T
    w, U = np.linalg.eigh(G)
    o = np.argsort(w)[::-1][:k]
    w, U = np.clip(w[o], 1e-12, None), U[:, o]
    B = (R.T @ U) / np.sqrt(w)
    return AO._perp(B.T, V).T


if __name__ == "__main__":
    self_test()
    if not os.path.exists(CKPT):
        raise SystemExit(f"  no checkpoint at {CKPT}")
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho = [p for p in man["heldout"] if p in have]
    ho.sort(key=lambda p: store.meta[have[p]]["atoms"])       # ascending N, as the peer harness does
    print(f"\n[099] variance ANM CANNOT reach: {len(ho)} held-out systems, ascending N", flush=True)
    print(f"  ANM K={K_ANM} cutoff={CUTOFF}; ceiling k={K_CEIL} fitted on {NTRF} train frames",
          flush=True)

    mdl, rows, failed, verified = None, {}, 0, False
    t0 = time.time()
    for i, pdb in enumerate(ho, 1):
        try:
            d = sysdata(store, have[pdb])
            if d is None:
                failed += 1; continue
            N, F, mu = d["N"], d["F"], d["mu"]
            V, _, _ = anm_modes(d["ref"], K_ANM, CUTOFF)
            if V is None:
                failed += 1; continue
            hof = RawFrames(d["path"], 2, F)
            trf = StridedTrain(d["path"], F, NTRF)
            sst_perp = residual_energy(hof, V, mu)
            tot = float(d["sst"])
            if mdl is None:
                mdl = Codec(d["stat"].shape[1], 1, DM).to(dev)
                mdl.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False))
                mdl.eval()

            # ---- ANM arm: the floor. Must come out EXACTLY 0; if it does not, the projector moved.
            def pred_anm(a, b, V=V, mu=mu, hof=hof):
                X = hof(a, b) - mu
                return mu + (X @ V) @ V.T
            f_anm = orthogonal_fve(hof, pred_anm, V, mu, sst_perp)

            # ---- CEILING: PCA inside the residual, fitted on TRAIN replicas only.
            if 3 * N <= GRAM_MAX:
                B = residual_pca_basis(trf, V, mu, K_CEIL)
                if not verified:                     # verify the frame-space route once, on a small system
                    B2 = ceiling_basis_framespace(trf, V, mu, K_CEIL)
                    ov = np.abs(np.linalg.svd(B.T @ B2, compute_uv=False))
                    print(f"  [verify] frame-space vs coordinate-space ceiling basis on {pdb}: "
                          f"min principal cosine {ov.min():.6f}, mean {ov.mean():.6f} "
                          f"({'SAME SUBSPACE' if ov.min() > 0.999 else 'DIFFERENT -- do not use'})",
                          flush=True)
                    verified = True
            else:
                B = ceiling_basis_framespace(trf, V, mu, K_CEIL)

            def pred_ceil(a, b, B=B, mu=mu, hof=hof):
                X = hof(a, b) - mu
                return mu + (X @ B) @ B.T
            f_ceil = orthogonal_fve(hof, pred_ceil, V, mu, sst_perp)

            # ---- CODEC: the trained arm, in raw coordinates.
            @torch.no_grad()
            def pred_cod(a, b, d=d, mu=mu, hof=hof):
                X = (hof(a, b) - mu).reshape(b - a, d["N"], 3)
                st = torch.tensor(d["stat"], device=dev).unsqueeze(0).expand(b - a, -1, -1)
                dw = torch.tensor((X / d["scale"]).astype(np.float32), device=dev)
                p = mdl(st, dw).cpu().numpy().astype(np.float64) * d["scale"]
                return mu + p.reshape(b - a, -1)
            f_cod = orthogonal_fve(hof, pred_cod, V, mu, sst_perp)

            rows[pdb] = dict(N=N, F=F, sst_total=tot, sst_perp=sst_perp,
                             perp_frac=sst_perp / (tot + 1e-12),
                             anm=f_anm, ceiling=f_ceil, codec=f_cod,
                             k_anm=K_ANM, k_ceil=K_CEIL,
                             basis_route="gram" if 3 * N <= GRAM_MAX else "framespace")
            print(f"  [{i}/{len(ho)}] {pdb:10s} N={N:>6}  perp {100*sst_perp/(tot+1e-12):4.1f}% of "
                  f"variance   ANM {f_anm:+.2e}  CEILING {f_ceil:+.4f}  CODEC {f_cod:+.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(ho)}] {pdb}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(ho), complete=(i == len(ho)),
                          n_failed=failed, k_anm=K_ANM, k_ceil=K_CEIL, ckpt=os.path.basename(CKPT))

    if not rows:
        raise SystemExit("\n  NO system scored. Reported as absent, not as a null.")
    A = np.array([r["anm"] for r in rows.values()])
    C = np.array([r["ceiling"] for r in rows.values()])
    M = np.array([r["codec"] for r in rows.values()])
    P = np.array([r["perp_frac"] for r in rows.values()])
    print(f"\n=== 099: FVE INSIDE THE ANM-ORTHOGONAL SUBSPACE ({len(rows)} systems, {failed} failed) ===")
    print(f"  the residual carries {100*np.median(P):.1f}% of held-out variance (median), "
          f"range {100*P.min():.1f}-{100*P.max():.1f}%")
    print(f"  {'arm':>10}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}")
    for lab, v in (("ANM", A), ("CEILING", C), ("CODEC", M)):
        print(f"  {lab:>10}{np.median(v):>10.4f}{v.mean():>10.4f}{v.min():>10.4f}{v.max():>10.4f}")
    print(f"\n  ANM max |deviation from exactly 0|: {np.abs(A).max():.3e}  "
          f"({'projector holds' if np.abs(A).max() < 1e-10 else 'PROJECTOR DRIFTED -- numbers void'})")
    print(f"  codec beats the ceiling on {100*(M > C).mean():.1f}% of systems; "
          f"codec > 0 on {100*(M > 0).mean():.1f}%")
    if np.median(M) > np.median(C):
        print("  -> CODEC ABOVE CEILING: a peer win on an axis with an exact floor and a measured ceiling")
    elif np.median(C) < 0.02 and np.median(M) < 0.02:
        print("  -> BOTH ~0: the axis is CLOSED. Nothing linear or learned reaches this residual,\n"
              "     which is learned for free and is a real result about the data, not the model.")
    else:
        print("  -> CODEC BELOW CEILING: a cleaner statement of the failure than 0/123, because the\n"
              "     floor is exact and the ceiling is measured rather than assumed.")
