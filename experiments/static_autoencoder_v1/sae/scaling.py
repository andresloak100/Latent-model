"""Fit  L(N) = A * N^(-alpha) + L_inf  and report both parameters with intervals.

L_inf IS THE POINT OF THE STUDY. alpha says how fast scale helps; L_inf says whether scale
arrives anywhere useful. A steep alpha over a floor that is already too high is not a reason to
buy compute.

Intervals come from a NON-PARAMETRIC BOOTSTRAP over the sweep points rather than from the
covariance of the fit, because the fit is nonlinear, the residuals are not Gaussian, and there
are only a handful of points. A bootstrap over four or five points is itself weak, and
`n_points` is returned so the reader can discount accordingly rather than being handed an
interval with no denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import curve_fit


def power_law(N, A, alpha, L_inf):
    return A * np.power(N, -alpha) + L_inf


@dataclass
class ScalingFit:
    A: float
    alpha: float
    L_inf: float
    alpha_lo: float
    alpha_hi: float
    L_inf_lo: float
    L_inf_hi: float
    n_points: int
    r2: float
    converged: bool
    # Predicted loss at 10x the largest measured N, with interval. IDENTIFIABLE EVEN WHEN A AND
    # L_inf ARE NOT -- see `degenerate` below.
    L_extrap_10x: float = float("nan")
    L_extrap_10x_lo: float = float("nan")
    L_extrap_10x_hi: float = float("nan")
    # True when alpha is indistinguishable from zero. Then A*N^-alpha is effectively constant,
    # A and L_inf trade off freely, and L_inf ALONE IS MEANINGLESS -- the fit will happily put
    # the whole level into A and report L_inf = 0. Caught by measurement, not by reasoning: a
    # self-test with truth (alpha=0.9, L_inf=0.40) on an already-saturated grid returned
    # alpha=0.008, L_inf=0.0000 with a tight interval before this flag existed.
    degenerate: bool = False
    note: str = ""

    def as_dict(self):
        return asdict(self)


def fit_scaling(N, L, n_boot: int = 2000, seed: int = 0) -> ScalingFit:
    """N: the scaled quantity (parameters, or latent size). L: held-out loss at each point."""
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)
    ok = np.isfinite(N) & np.isfinite(L)
    N, L = N[ok], L[ok]
    if len(N) < 4:
        return ScalingFit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                          len(N), np.nan, False, note=
                          "fewer than 4 usable points: a three-parameter fit is not identifiable")

    def _fit(n, l):
        p0 = [max(l.max() - l.min(), 1e-6) * n.min() ** 0.5, 0.5, max(l.min() * 0.5, 0.0)]
        return curve_fit(power_law, n, l, p0=p0, maxfev=200_000,
                         bounds=([0, 0, 0], [np.inf, 5.0, max(l.max(), 1e-9)]))[0]

    try:
        A, alpha, L_inf = _fit(N, L)
    except Exception as exc:
        return ScalingFit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                          len(N), np.nan, False,
                          note=f"fit failed: {type(exc).__name__}: {exc}")

    pred = power_law(N, A, alpha, L_inf)
    ss_res = float(((L - pred) ** 2).sum())
    ss_tot = float(((L - L.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    a_s, l_s, boot_params = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), len(N))
        if len(np.unique(idx)) < 4:
            continue
        try:
            pa, a, li = _fit(N[idx], L[idx])
            a_s.append(a); l_s.append(li); boot_params.append((pa, a, li))
        except Exception:
            continue

    note = "" if len(a_s) >= 0.5 * n_boot else \
        f"only {len(a_s)}/{n_boot} bootstrap resamples converged; intervals are unreliable"
    q = (lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))) if a_s \
        else (lambda v: (float("nan"), float("nan")))
    a_lo, a_hi = q(a_s) if a_s else (np.nan, np.nan)
    l_lo, l_hi = q(l_s) if l_s else (np.nan, np.nan)

    n_ex = 10.0 * N.max()
    e_s = [power_law(n_ex, *p) for p in boot_params] if boot_params else []
    e_lo, e_hi = (float(np.percentile(e_s, 2.5)), float(np.percentile(e_s, 97.5))) if e_s \
        else (float("nan"), float("nan"))

    # DEGENERACY IS TESTED DIRECTLY, not through a bootstrap quantile. The first version keyed
    # on alpha_hi < 0.02, and on the very case it exists for only 125/300 resamples converged --
    # so the quantile was noise and the flag stayed off while L_inf came back at 0.045 against a
    # truth of 0.40. The direct test: how much of the observed variation does the power-law term
    # actually produce across the measured range? If that span is no larger than the fit's own
    # residual scatter, A and L_inf are trading off freely and neither means anything alone.
    span = float(A * (N.min() ** -alpha - N.max() ** -alpha))
    resid_rms = float(np.sqrt(ss_res / len(N)))
    degenerate = bool(span <= 2.0 * resid_rms or (np.isfinite(a_hi) and a_hi < 0.02))
    if degenerate:
        note = (note + "; " if note else "") + (
            "alpha is indistinguishable from zero, so A and L_inf are NOT separately "
            "identifiable -- read L_extrap_10x, not L_inf")

    return ScalingFit(float(A), float(alpha), float(L_inf), a_lo, a_hi, l_lo, l_hi,
                      len(N), r2, True, float(power_law(n_ex, A, alpha, L_inf)), e_lo, e_hi,
                      degenerate, note)


def reading(fit_params: ScalingFit, fit_latent: ScalingFit, L_target: float | None = None) -> str:
    """Map the two fits onto the readings pre-registered in BRIEF.md.

    A branch on a hard threshold is how an arbitrary cliff turns a measurement into a category,
    so the thresholds here are stated in the returned string rather than hidden in the code, and
    the verdict is proportional to what was measured.
    """
    parts = []
    for name, f in (("parameters", fit_params), ("latent size", fit_latent)):
        if not f.converged:
            parts.append(f"{name}: NOT EVALUABLE -- {f.note}")
            continue
        span = "saturating" if f.alpha_hi < 0.05 else \
               "improving" if f.alpha_lo > 0.05 else "indeterminate"
        floor = (f"L_inf NOT IDENTIFIABLE (alpha~0); L at 10x largest N = {f.L_extrap_10x:.4g} "
                 f"[{f.L_extrap_10x_lo:.4g}, {f.L_extrap_10x_hi:.4g}]") if f.degenerate else \
                (f"L_inf={f.L_inf:.4g} [{f.L_inf_lo:.4g}, {f.L_inf_hi:.4g}]")
        parts.append(
            f"{name}: alpha={f.alpha:.3f} [{f.alpha_lo:.3f}, {f.alpha_hi:.3f}] ({span}), "
            f"{floor}, n={f.n_points}, R2={f.r2:.3f}"
        )
    if L_target is not None and fit_params.converged:
        parts.append(
            f"target L={L_target:.4g} is "
            f"{'REACHABLE' if fit_params.L_inf < L_target else 'BELOW THE FLOOR'} "
            f"by the parameter-axis fit (L_inf={fit_params.L_inf:.4g})"
        )
    return "\n".join(parts)
