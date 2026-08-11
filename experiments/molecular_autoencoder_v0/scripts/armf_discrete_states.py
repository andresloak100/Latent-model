#!/usr/bin/env python3
"""INBOX 98c: do DISCRETE conformational states exist in this corpus at all?

WHY THIS COMES BEFORE THE ARCHITECTURE. A grouped-simplex latent encodes "choose among alternatives".
If the trajectories contain no alternatives to choose among, the prior has nothing to represent and
the direction closes before a line of it is written. Everything this project has measured points that
way and none of it was aimed at this question: replicas sit at 1.177x separation, n_eff is 1-7%,
reachable lags are 1-2 ns, and armf_propagator.basins() is a MEDIAN SPLIT on the top two modes -- an
IMPOSED partition, not discovered states. A median split always returns four occupied quadrants
whether or not any state exists, so it can never answer this.

THE STATE COUNT IS CHOSEN BY THE DATA, not by me. tICA gives the slow collective coordinates -- the
ones a state must be slow along to be a state at all -- and k-means over a range of k is scored by
the GAP against a phase-randomised surrogate that preserves each coordinate's power spectrum and
therefore its autocorrelation, destroying only the joint structure. A unimodal diffusive walk in a
single basin produces well-separated k-means clusters too; the surrogate is what distinguishes real
metastability from the clustering algorithm doing its job on noise.

TWO PRE-REGISTERED READINGS:
  states exist   silhouette exceeds the surrogate at some k>1 AND the implied transitions are rarer
                 than the lag -- a simplex prior has something to encode
  no states      the corpus is one diffusive basin at these lags; a simplex prior encodes a choice
                 that is not there, and 98b's basis question is the live one instead

REPORTED EITHER WAY: the timescale separation. Metastability is not just "clusters exist" but "the
system stays". A partition with transitions every few frames is a fast coordinate cut in half.
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from armf_atlas_data import AtlasStore, sysdata
import armf_atlas_dm as D
import armf_io

WR = D.WR
NSYS = int(os.environ.get("DS_NSYS", "40"))
NTICA = int(os.environ.get("DS_NTICA", "4"))
LAG = int(os.environ.get("DS_LAG", "25"))          # frames; 40 ps/frame -> 1 ns
KS = [int(x) for x in os.environ.get("DS_KS", "2,3,4,5,6").split(",")]
NSURR = int(os.environ.get("DS_NSURR", "20"))
NMODE = int(os.environ.get("DS_NMODE", "32"))
RES = os.environ.get("DS_RES", f"{WR}/discrete_states.json")


def tica(X, lag, k):
    """Slow collective coordinates. Solves C(lag) v = lam C(0) v in a whitened basis -- the
    coordinates along which the system decorrelates most slowly, which is what a state must be."""
    Xc = X - X.mean(0)
    C0 = np.cov(Xc, rowvar=False) + 1e-9 * np.eye(Xc.shape[1])
    Ct = (Xc[:-lag].T @ Xc[lag:]) / max(len(Xc) - lag, 1)
    Ct = 0.5 * (Ct + Ct.T)
    w, U = np.linalg.eigh(C0)
    Wh = U @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-12, None))) @ U.T
    lam, V = np.linalg.eigh(Wh @ Ct @ Wh)
    o = np.argsort(lam)[::-1][:k]
    return Xc @ (Wh @ V[:, o]), lam[o]


def kmeans(X, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all(): break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any(): C[j] = X[m].mean(0)
    return lab


def silhouette(X, lab, rng, n=1500):
    """Mean silhouette on a subsample -- O(n^2) exactly, so the subsample is the point."""
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Y, L = X[idx], lab[idx]
    if len(np.unique(L)) < 2: return 0.0
    Dm = np.sqrt(((Y[:, None, :] - Y[None]) ** 2).sum(-1))
    out = []
    for i in range(len(Y)):
        same = L == L[i]; same[i] = False
        if not same.any(): continue
        a = Dm[i, same].mean()
        b = min(Dm[i, L == j].mean() for j in np.unique(L) if j != L[i])
        out.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(out)) if out else 0.0


def phase_randomise(X, rng):
    """Surrogate preserving each coordinate's POWER SPECTRUM -- hence its autocorrelation and
    marginal variance -- while destroying joint and higher-order structure. The null is 'a coloured
    Gaussian walk with the same slowness', which is exactly what a single diffusive basin looks
    like."""
    F = np.fft.rfft(X, axis=0)
    ph = rng.uniform(0, 2 * np.pi, F.shape[0])[:, None]
    ph[0] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=X.shape[0], axis=0)


if __name__ == "__main__":
    store = AtlasStore(f"{WR}/atlas_cache")
    man = json.load(open(D.MAN))
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ids = [p for p in man["heldout"] if p in have]
    ids.sort(key=lambda p: store.meta[have[p]]["atoms"])
    ids = ids[:NSYS]
    print(f"[98c] do discrete conformational states exist? {len(ids)} systems, tICA lag {LAG} frames "
          f"(~{LAG*0.04:.1f} ns), k in {KS}, {NSURR} phase-randomised surrogates", flush=True)
    rng = np.random.default_rng(0)
    rows, failed = {}, 0
    t0 = time.time()
    for i, pdb in enumerate(ids, 1):
        try:
            d = sysdata(store, have[pdb])
            if d is None: failed += 1; continue
            a = np.load(d["path"], mmap_mode="r")
            X = np.asarray(a[2]).astype(np.float64).reshape(d["F"], -1) - d["mu"]
            # project onto the system's own leading PCA modes first: tICA on 3N is rank-starved
            Xc = X - X.mean(0)
            G = Xc @ Xc.T
            w, U = np.linalg.eigh(G)
            o = np.argsort(w)[::-1][:NMODE]
            P = U[:, o] * np.sqrt(np.clip(w[o], 1e-12, None))
            Y, lam = tica(P, LAG, NTICA)
            its = -LAG / np.log(np.clip(lam, 1e-9, 1 - 1e-9))     # implied timescales, frames
            best = {}
            for k in KS:
                lab = kmeans(Y, k, seed=0)
                s = silhouette(Y, lab, rng)
                sur = [silhouette(tica(phase_randomise(P, rng), LAG, NTICA)[0],
                                  kmeans(tica(phase_randomise(P, rng), LAG, NTICA)[0], k, seed=0),
                                  rng) for _ in range(max(3, NSURR // len(KS)))]
                trans = float((lab[1:] != lab[:-1]).mean())
                dwell = 1.0 / max(trans, 1e-9)
                best[k] = dict(sil=s, sur_mean=float(np.mean(sur)), sur_sd=float(np.std(sur)),
                               z=float((s - np.mean(sur)) / max(np.std(sur), 1e-9)),
                               trans_per_frame=trans, dwell_frames=dwell)
            rows[pdb] = dict(N=d["N"], F=d["F"], its_frames=its.tolist(),
                             its_ns=(its * 0.04).tolist(), ks=best)
            k_best = max(best, key=lambda k: best[k]["z"])
            b = best[k_best]
            print(f"  [{i}/{len(ids)}] {pdb:10s} N={d['N']:>6} slowest ITS {its[0]*0.04:7.2f} ns  "
                  f"best k={k_best} sil {b['sil']:+.3f} vs surrogate {b['sur_mean']:+.3f} "
                  f"(z={b['z']:+.1f})  dwell {b['dwell_frames']:.1f} fr  ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(ids)}] {pdb}: FAIL {type(e).__name__}: {e}", flush=True)
        armf_io.dump_rows(RES, rows, n_expected=len(ids), complete=(i == len(ids)),
                          n_failed=failed, lag=LAG, ks=KS, n_tica=NTICA)

    if not rows:
        raise SystemExit("\n  NO system scored. Reported as absent, not as a null.")
    print(f"\n=== 98c: DO DISCRETE STATES EXIST? ({len(rows)} systems, {failed} failed) ===")
    print(f"  {'k':>4}{'median sil':>12}{'median surrogate':>18}{'median z':>10}"
          f"{'systems z>2':>13}{'median dwell (fr)':>19}")
    for k in KS:
        s = np.array([r["ks"][str(k)]["sil"] if str(k) in r["ks"] else r["ks"][k]["sil"]
                      for r in rows.values()])
        su = np.array([(r["ks"][str(k)] if str(k) in r["ks"] else r["ks"][k])["sur_mean"]
                       for r in rows.values()])
        z = np.array([(r["ks"][str(k)] if str(k) in r["ks"] else r["ks"][k])["z"]
                      for r in rows.values()])
        dw = np.array([(r["ks"][str(k)] if str(k) in r["ks"] else r["ks"][k])["dwell_frames"]
                       for r in rows.values()])
        print(f"  {k:>4}{np.median(s):>12.3f}{np.median(su):>18.3f}{np.median(z):>10.1f}"
              f"{(z>2).sum():>9}/{len(z):<3}{np.median(dw):>19.1f}")
    its0 = np.array([r["its_ns"][0] for r in rows.values()])
    print(f"\n  slowest implied timescale: median {np.median(its0):.2f} ns, "
          f"range {its0.min():.2f}-{its0.max():.2f} ns, against a {LAG*0.04:.1f} ns lag")
    zbest = np.array([max((r["ks"][kk]["z"] for kk in r["ks"])) for r in rows.values()])
    frac = (zbest > 2).mean()
    print(f"  systems with ANY k beating the surrogate at z>2: {100*frac:.0f}%")
    if frac < 0.25:
        print("  -> NO DISCRETE STATES at these lags. A simplex prior would encode a choice that is\n"
              "     not in the data, and 98b's basis question is the live one instead.")
    else:
        print("  -> STATES PRESENT in a substantial fraction. A simplex prior has something to encode;\n"
              "     the dwell time says whether they are metastable or a fast coordinate cut in half.")
