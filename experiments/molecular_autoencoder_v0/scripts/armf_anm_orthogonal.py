"""INBOX 099: FVE in the subspace ANM CANNOT REACH.

THE QUESTION THE EXISTING METRIC CANNOT ASK. `armf_tied_peer.py` reports FVE about `mu`
and the codec loses 0/123. That says the codec explains less TOTAL variance than ANM. It
does not say whether the codec explains variance ANM **cannot**, and those are different
claims that one number cannot separate.

Split the held-out displacement into the part ANM spans and the part it does not:

    d                held-out displacement about mu, (F, 3N)
    V                ANM modes from the system's REFERENCE structure, (3N, K), orthonormal
    d_par = dVV'     what ANM can reach
    r     = d - dVV' what it cannot          <- everything below lives here

`orthogonal_fve` scores a predictor inside `r` only.

WHY THIS AXIS IS SELF-CALIBRATING. For orthonormal V, an ANM reconstruction projected into
the residual is `dVV'(I - VV') = dVV' - dVV'VV' = 0`, because `V'V = I`. So **ANM scores
EXACTLY 0 here, by arithmetic and not by luck**, and so does any model that has merely
learned ANM. Only structure ANM cannot reach scores at all. `self_test` asserts this to
machine precision; if it ever fails, the projector is wrong and every number from this
file is meaningless.

WHY IT NEEDS A CEILING (INBOX 99a). `r` is local, high-frequency and partly thermal, and
some of it is not predictable by anything. A bare "the codec explains 3% of it" is a count
against an unknown maximum -- Family B. Report three rows, always together:

    ANM                    0.000   exact, the floor
    PCA on the residual        ?   cross-fit from TRAIN frames, scored on held-out, the
                                   linear ceiling
    codec                      ?

Codec above the linear ceiling is a peer win on an axis with an exact floor and a measured
ceiling. Codec below it is a cleaner statement of the failure than 0/123. Both near zero
means the axis is closed, which is worth knowing before an architecture is built to attack
it.

The PCA basis MUST be cross-fit. Fitting it on the frames it is scored on hands it the
answer; this project has already paid for that once, at +0.041 bits/atom in 088.

MEMORY -- AND A SENTENCE IN THIS DOCSTRING THAT WAS FALSE WHEN IT WAS WRITTEN.

`orthogonal_fve` and `residual_energy` do stream: only (chunk, 3N) and (3N, K) are ever
resident. `residual_pca_basis` DOES NOT. It accumulates `R.T @ R`, which is (3N, 3N) --
80 GB at N=33,377 -- and an earlier version of this docstring asserted that nothing of
that shape is ever formed, three lines above the code that forms it.

The cluster agent hit it and routed large systems through a frame-space Gram instead
(`R @ R.T`, (chunk, chunk), eigenvectors mapped back), verifying the two agree at minimum
principal cosine 1.000000 rather than assuming they do.

Same defect as claiming `git rm --cached` "keeps them on disk": a description asserting a
property the code does not have. Left corrected in place rather than quietly rewritten,
because the false sentence is the part worth remembering.
"""
import numpy as np

CHUNK = 256          # frames per block; (CHUNK, 3N) float64 is 205 MB at N=33,377


def _perp(X, V):
    """X - (X V) V' : the component of each row orthogonal to span(V). Never forms VV'."""
    return X - (X @ V) @ V.T


def residual_energy(frames, V, mu, chunk=CHUNK):
    """Total energy of the held-out displacement that ANM cannot reach.

    `frames(a, b)` returns rows [a, b) of the raw held-out coordinates, (b-a, 3N).
    Streamed, so the caller can hand it a memmap slice rather than an array.
    """
    tot = 0.0
    for s in range(0, frames.n, chunk):
        e = min(s + chunk, frames.n)
        tot += float((_perp(frames(s, e) - mu, V) ** 2).sum())
    return tot


def orthogonal_fve(frames, predict, V, mu, sst_perp=None, chunk=CHUNK):
    """FVE of `predict` inside the ANM-orthogonal subspace.

    `predict(a, b)` returns the model's reconstruction for rows [a, b), same shape and
    units as the frames. Both sides are projected with the SAME `_perp`, which is what
    makes ANM score exactly zero rather than approximately.

    Returns `1 - ||r - p_perp||^2 / ||r||^2`. Unbounded below, like every FVE here: a
    predictor that adds energy in directions ANM cannot reach scores negative, and that
    is information rather than a failure of the metric.
    """
    if sst_perp is None:
        sst_perp = residual_energy(frames, V, mu, chunk)
    sse = 0.0
    for s in range(0, frames.n, chunk):
        e = min(s + chunk, frames.n)
        r = _perp(frames(s, e) - mu, V)
        p = _perp(predict(s, e) - mu, V)
        sse += float(((r - p) ** 2).sum())
    return 1.0 - sse / (sst_perp + 1e-12)


def residual_pca_basis(train_frames, V, mu, k, chunk=CHUNK):
    """The linear CEILING: PCA fitted INSIDE the ANM-orthogonal subspace, on TRAIN frames.

    Cross-fit by construction -- this only ever sees train frames, and the caller scores
    the returned basis on held-out ones through `orthogonal_fve`. Returns a (3N, k) basis
    already orthogonal to V, so a reconstruction built from it lives entirely in `r`.
    """
    acc = None
    for s in range(0, train_frames.n, chunk):
        e = min(s + chunk, train_frames.n)
        R = _perp(train_frames(s, e) - mu, V)
        acc = R.T @ R if acc is None else acc + R.T @ R
    w, U = np.linalg.eigh(acc)
    B = U[:, np.argsort(w)[::-1][:k]]
    return _perp(B.T, V).T          # re-orthogonalise against V; eigh drifts at float64


class Frames:
    """Minimal adapter so `frames(a, b)` works over an array, a memmap or a generator."""

    def __init__(self, arr):
        self.arr, self.n = arr, len(arr)

    def __call__(self, a, b):
        return np.asarray(self.arr[a:b]).astype(np.float64)


def self_test(seed=0):
    """INBOX 82c's shape. The first assertion is the one that matters.

    If an ANM reconstruction does not score EXACTLY zero, the projector is wrong and every
    number this file produces is meaningless -- so it raises rather than warns.
    """
    rng = np.random.default_rng(seed)
    F, N, K = 400, 60, 12
    D3 = 3 * N
    Q, _ = np.linalg.qr(rng.standard_normal((D3, K)))          # ANM stand-in, orthonormal
    coef = rng.standard_normal((F, K)) * np.sqrt(np.geomspace(40, 1, K))
    resid = rng.standard_normal((F, D3)) * 0.35
    d = coef @ Q.T + _perp(resid, Q)                           # spanned part + residual
    mu = np.zeros(D3)
    fr = Frames(d)
    sst = residual_energy(fr, Q, mu)
    ok = []

    # 1. ANM scores EXACTLY zero. Arithmetic, not luck.
    anm = Frames((d @ Q) @ Q.T)
    v = orthogonal_fve(fr, lambda a, b: anm(a, b), Q, mu, sst)
    assert abs(v) < 1e-12, f"ANM scored {v:.3e}, must be 0 -- projector is wrong"
    ok.append(f"1. ANM reconstruction        {v:+.2e}   exact zero")

    # 2. a perfect predictor scores 1
    v = orthogonal_fve(fr, lambda a, b: fr(a, b), Q, mu, sst)
    assert abs(v - 1) < 1e-12, f"perfect predictor scored {v}"
    ok.append(f"2. perfect predictor         {v:+.4f}   exact one")

    # 3. ANM plus arbitrary energy INSIDE its own span still scores zero -- the metric
    #    cannot be gamed by spending capacity where ANM already is
    extra = Frames((d @ Q) @ Q.T + (rng.standard_normal((F, K)) @ Q.T))
    v = orthogonal_fve(fr, lambda a, b: extra(a, b), Q, mu, sst)
    assert abs(v) < 1e-12, f"in-span noise leaked into the residual: {v:.3e}"
    ok.append(f"3. ANM + in-span noise       {v:+.2e}   still zero")

    # 4. the cross-fit ceiling is a real ceiling: fitted on the first half, scored on the
    #    second, it explains some of the residual and less than 1
    half = F // 2
    B = residual_pca_basis(Frames(d[:half]), Q, mu, k=8)
    ho = Frames(d[half:])
    rec = Frames((d[half:] @ B) @ B.T)
    v = orthogonal_fve(ho, lambda a, b: rec(a, b), Q, mu)
    assert 0.0 < v < 1.0, f"cross-fit PCA ceiling out of range: {v}"
    ok.append(f"4. cross-fit PCA-on-residual {v:+.4f}   a real ceiling, 0 < v < 1")

    # 5. and the ceiling basis is itself orthogonal to ANM, so it cannot borrow ANM's span
    leak = float(np.abs(B.T @ Q).max())
    assert leak < 1e-8, f"ceiling basis overlaps ANM by {leak:.2e}"
    ok.append(f"5. ceiling basis vs ANM      {leak:.1e}   orthogonal")

    print("[099] SELF-TEST PASSED")
    for line in ok:
        print("   ", line)


if __name__ == "__main__":
    self_test()
    print("\n[099] Machinery only -- no data plumbing, deliberately. The per-system loop\n"
          "      needs the ATLAS cache and a checkpoint, so it belongs on the cluster:\n"
          "      for each held-out system take V from armf_anm.modes(ref, K, cutoff),\n"
          "      mu and the frames from sysdata, and report three rows per system --\n"
          "      ANM (exactly 0), residual-PCA cross-fit from replicas 0+1 (the ceiling),\n"
          "      and the codec. Write through armf_io.dump_rows. Sort systems ASCENDING\n"
          "      IN N as the peer harness does, so a killed run is size-truncated in the\n"
          "      way 74b's envelope already makes visible.")
