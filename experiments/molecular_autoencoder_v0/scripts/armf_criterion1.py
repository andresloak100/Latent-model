"""CRITERION 1: DOES A DECODED TRAJECTORY RETAIN THE DYNAMICS? (STANDING QUEUE Q3)

INBOX 004b makes "retains dynamical information" the FIRST success criterion, ahead of per-frame FVE,
and says so in the sharpest possible form: *a codec with mediocre FVE that preserves autocorrelation
structure is worth more than one with better FVE that flattens it.* A criterion that cannot be
measured is not a criterion, so this harness is built and smoke-tested BEFORE the DM sweep produces a
live arm rather than after.

The propagator already has this test (`armf_ensemble_validate.py`). What changes for a codec is what
"generated" means: the propagator ROLLS OUT a trajectory, whereas the codec DECODES the reference
frames one at a time. So the question is not "did it invent plausible dynamics" but "did passing the
trajectory through the bottleneck destroy the dynamics that were already there" -- which is the
failure that would make the latent useless to stage 2 while looking fine on FVE.

FOUR DISCRIMINATORS, in a fixed mode basis fitted on the REFERENCE:
  1. MARGINAL STD RATIO per mode (decoded/reference). Variance collapse: a decoder that regresses
     toward the mean scores well on MSE and lands near 0 here.
  2. INTEGRATED AUTOCORRELATION TIME per mode, both trajectories. THE KINETIC TEST. Right marginals
     with wrong decay means the distribution survived and the dynamics did not.
  3. CROSS-MODE COUPLING: correlation between mode pairs, reference vs decoded. A decoder can get
     every marginal right and still scramble which modes move together.
  4. 2D FREE ENERGY on the top-2 reference modes: Jensen-Shannon divergence plus basin coverage.
     Missing basins and filled barriers are visible nowhere else.

WHY THE SHUFFLED CONTROL IS THE RIGHT NEGATIVE. Shuffling the reference frames preserves EVERY
marginal exactly -- same std, same free-energy surface, same cross-mode correlations -- and destroys
only the time ordering. So it passes discriminators 1, 3 and 4 and must fail 2. A harness that
"passes" shuffled frames is measuring distribution, not dynamics, and would certify a latent the
propagator cannot use. That is the whole point of building it against a known-bad input first.

REPORTING RULE (standing): report FRACTIONS PER DISCRIMINATOR, never a mean across them. A mean lets
a catastrophic kinetic failure hide behind three passing distributional checks -- which is exactly
the case this harness exists to catch."""
import sys, os, numpy as np, warnings
warnings.filterwarnings("ignore")

IAT_CUT = 0.05
STD_TOL = (0.75, 1.30)     # per-mode marginal std ratio counts as retained inside this band
IAT_TOL = 2.0              # per-mode IAT within this factor counts as retained
CORR_TOL = 0.15            # max |delta| in cross-mode correlation
JS_TOL = 0.10              # Jensen-Shannon divergence on the 2D free-energy projection


def iat(x, cutoff=IAT_CUT, maxlag=None):
    """tau_int = 1 + 2*sum rho(k), truncated at rho < cutoff. Same convention as the n_eff work."""
    x = np.asarray(x, float); x = x - x.mean()
    v = float((x * x).mean())
    if v < 1e-14: return 1.0
    n = len(x); maxlag = maxlag or max(2, n // 2)
    t = 1.0
    for k in range(1, maxlag):
        r = float((x[:-k] * x[k:]).mean()) / v
        if r < cutoff: break
        t += 2.0 * r
    return max(t, 1.0)


def js_2d(a, b, bins=32):
    """Jensen-Shannon divergence of two 2D histograms on a COMMON grid (base-2, so in [0, 1])."""
    lo = np.minimum(a.min(0), b.min(0)); hi = np.maximum(a.max(0), b.max(0))
    rng = [[lo[0], hi[0]], [lo[1], hi[1]]]
    P, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=rng)
    Q, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=bins, range=rng)
    P = P.ravel() / max(P.sum(), 1e-12); Q = Q.ravel() / max(Q.sum(), 1e-12)
    M = 0.5 * (P + Q)
    def kl(p, m):
        g = p > 0
        return float((p[g] * np.log2(p[g] / m[g])).sum())
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def basis_from_reference(ref, k=10):
    """Top-k PCA modes of the REFERENCE trajectory. Both trajectories are scored in this ONE basis:
    a basis refitted per trajectory would let a decoder that rotated the dynamics into different
    directions still score perfectly, which is the Family-D failure of measuring in a moving frame."""
    X = ref.reshape(len(ref), -1)
    X = X - X.mean(0)
    G = X @ X.T
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]
    w = np.clip(w[o], 0, None); U = U[:, o]
    k = min(k, int((w > w[0] * 1e-12).sum()))
    s = np.sqrt(w[:k]) + 1e-12
    V = (X.T @ U[:, :k]) / s                       # (D, k), frame-space trick: no (D,D) ever formed
    return V, X.mean(0)


def project(traj, V, mu):
    return (traj.reshape(len(traj), -1) - mu) @ V


def acceptance(ref, dec, k=10, label=""):
    """ref, dec: (F, N, 3) reference and decoded trajectories, SAME frames in the SAME order."""
    V, mu = basis_from_reference(ref, k)
    A = project(ref, V, mu); B = project(dec, V, mu)
    kk = A.shape[1]

    sa, sb = A.std(0), B.std(0)
    ratio = sb / (sa + 1e-12)
    ta = np.array([iat(A[:, i]) for i in range(kk)])
    tb = np.array([iat(B[:, i]) for i in range(kk)])
    trat = tb / (ta + 1e-12)

    Ca = np.corrcoef(A.T); Cb = np.corrcoef(B.T)
    iu = np.triu_indices(kk, 1)
    dcorr = np.abs(Ca[iu] - Cb[iu])

    js = js_2d(A[:, :2], B[:, :2])

    d = dict(label=label, k=kk,
             std_frac=float(np.mean((ratio > STD_TOL[0]) & (ratio < STD_TOL[1]))),
             std_med=float(np.median(ratio)),
             iat_frac=float(np.mean((trat > 1 / IAT_TOL) & (trat < IAT_TOL))),
             iat_med=float(np.median(trat)),
             iat_ref_med=float(np.median(ta)), iat_dec_med=float(np.median(tb)),
             corr_frac=float(np.mean(dcorr < CORR_TOL)), corr_med=float(np.median(dcorr)),
             js=float(js))
    d["pass_std"] = d["std_frac"] >= 0.80
    d["pass_iat"] = d["iat_frac"] >= 0.80
    d["pass_corr"] = d["corr_frac"] >= 0.80
    d["pass_js"] = d["js"] <= JS_TOL
    d["n_pass"] = int(d["pass_std"] + d["pass_iat"] + d["pass_corr"] + d["pass_js"])
    return d


def report(d):
    print(f"  [{d['label']}] k={d['k']} modes", flush=True)
    print(f"    1 marginal std ratio   {d['std_frac']*100:>5.0f}% of modes retained   "
          f"median {d['std_med']:.3f}   {'PASS' if d['pass_std'] else 'FAIL'}")
    print(f"    2 IAT ratio (KINETIC)  {d['iat_frac']*100:>5.0f}% of modes retained   "
          f"median {d['iat_med']:.3f}  (ref {d['iat_ref_med']:.1f} -> dec {d['iat_dec_med']:.1f} "
          f"frames)   {'PASS' if d['pass_iat'] else 'FAIL'}")
    print(f"    3 cross-mode coupling  {d['corr_frac']*100:>5.0f}% of pairs within {CORR_TOL}   "
          f"median |d| {d['corr_med']:.3f}   {'PASS' if d['pass_corr'] else 'FAIL'}")
    print(f"    4 2D free energy JS    {d['js']:.4f} (tol {JS_TOL})   "
          f"{'PASS' if d['pass_js'] else 'FAIL'}")
    print(f"    -> {d['n_pass']}/4 discriminators passed", flush=True)


if __name__ == "__main__":
    # SELF-TEST against a trivially-passing and a trivially-failing input, so the harness is known to
    # discriminate BEFORE it is ever pointed at a codec arm.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from armf_atlas_data import AtlasStore
    WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
    store = AtlasStore(f"{WR}/atlas_cache")
    m = sorted(store.meta, key=lambda z: z["atoms"])[0]
    a = np.load(m["path"], mmap_mode="r")
    ref = np.asarray(a[2, :800]).astype(np.float64)
    print(f"[criterion-1 self-test] {m['pdb']} N={m['atoms']} F={len(ref)}", flush=True)
    rng = np.random.default_rng(0)

    print("\n=== POSITIVE CONTROL: decoded == reference. Must pass 4/4. ===", flush=True)
    d_id = acceptance(ref, ref.copy(), label="identity"); report(d_id)

    print("\n=== NEGATIVE CONTROL: frames SHUFFLED. Marginals, couplings and the free-energy")
    print("    surface are preserved EXACTLY -- only time ordering is destroyed. So 1, 3 and 4 must")
    print("    PASS and 2 (the kinetic test) must FAIL. A harness that passes this is measuring")
    print("    distribution, not dynamics, and would certify a latent stage 2 cannot use. ===", flush=True)
    d_sh = acceptance(ref, ref[rng.permutation(len(ref))], label="shuffled"); report(d_sh)

    print("\n=== NEGATIVE CONTROL: variance collapse (decoded = 0.3x deviation from the mean).")
    print("    An MSE-minimising decoder that regresses toward the mean looks like this. ===", flush=True)
    mu = ref.mean(0)
    d_vc = acceptance(ref, mu + 0.3 * (ref - mu), label="collapsed"); report(d_vc)

    print(f"\n=== VERDICT ===", flush=True)
    ok = (d_id["n_pass"] == 4 and d_sh["pass_std"] and not d_sh["pass_iat"]
          and not d_vc["pass_std"])
    print(f"  identity 4/4                      : {'YES' if d_id['n_pass']==4 else 'NO'}")
    print(f"  shuffled keeps marginals (disc 1) : {'YES' if d_sh['pass_std'] else 'NO'}")
    print(f"  shuffled FAILS the kinetic test   : {'YES' if not d_sh['pass_iat'] else 'NO  <-- the'}"
          f"{'' if not d_sh['pass_iat'] else ' harness cannot tell dynamics from distribution'}")
    print(f"  variance collapse caught (disc 1) : {'YES' if not d_vc['pass_std'] else 'NO'}")
    print(f"  -> HARNESS {'DISCRIMINATES; ready for a live codec arm' if ok else 'IS NOT SOUND -- do not use'}",
          flush=True)
