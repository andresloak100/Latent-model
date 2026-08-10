"""STEP 3: learned propagator vs OU, strengthened acceptance test, lag-time sweep.

Acceptance metrics (gen vs reference), the discriminators OU cannot pass by design:
  marginals   std ratio, per-mode JS, excess kurtosis (Gaussian=0)
  kinetics    IAT ratio (tau-aware), basin-transition RATE per 1000 steps
  COUPLING    cross-mode linear corr (off-diag) and amplitude coupling corr(|c_i|,|c_j|)
              -- OU in the ANM basis is independent per mode, so these are ~0 for OU;
              any nonzero reference coupling is unreachable by OU, reachable by a learned model.

Sweep tau = 1/10/50/100 ns; DDPM with FiLM per-layer conditioning; delta-vs-absolute
ablation; OU at matched lag; step-1 conditioning guard reported per DDPM; steps-to-1ms.

HOW THE COMPARISON IS MADE (INBOX 77a/77b -- this replaced an earlier scheme).
Earlier, the generated statistics came from ONE rollout of 1,500 steps and the
reference statistics came from the WHOLE trajectory. None of these estimators is
length-neutral, so the two sides carried different bias and every gap read as the
model failing. Worse, the generated autocorrelation time is capped by the rollout
length while a rollout step advances by tau -- so the cap loosened as tau grew, and
the tau sweep, which is the axis this experiment exists to measure, would have shown
a trend for a model that had not changed.

Now: K rollouts and K tau-strided reference windows of the SAME length, every one
through the same `stats_of`. Estimator bias is common to both sides and cancels. The
verdict per metric is whether the two spreads OVERLAP -- measured on a synthetic
control where both sides are the same process, interval-overlap gives 0-3.6% false
misses against 7-14% for testing the generated median against the reference band.
`H/iat_r` is printed per row and a lag whose trajectory is too short to evaluate is
reported as unevaluable rather than scored.
"""
import argparse, math, numpy as np, torch, torch.nn as nn, h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
BB = ["N", "CA", "C", "O"]; TEMP, R0 = "320", "0"
L, Tdiff, kT = 64, 100, 0.593
# WIDENED from 2 domains to all 28 present on disk. Two domains cannot separate "the propagator
# works" from "it works on these two" -- the same n-too-small problem 22a/24c kept hitting, and
# the generative axis is the only untested direction left (ROADMAP 5b), so it should not be
# decided on n=2. Resumable per domain, so a time-limit kill costs the current domain only.
USE = ['2cndA01', '2e2dC02', '2f93B00', '2fm7A00', '2h7tA01', '2k4qA00', '2lklA01', '2lt5A00', '2m1xA00', '2p9xA00', '2yx1A01', '2z1kA02', '3a5zD02', '3a9lA00', '3g7dA03', '3h7lB02', '3jamY00', '3jvvA01', '3k30A03', '4ccdA02', '4cd8A00', '4e6sA00', '4ifdE00', '4lrzE01', '4n9jA02', '4qdcA02', '4qiwK00', '4xk8d00']; TAUS = [1, 10, 50, 100]
torch.manual_seed(0); np.random.seed(0)


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    a = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in a])


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P[mask].mean(0); Pc = P[mask] - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


def anm(xyz, K, cutoff=10.0):
    n = len(xyz); H = np.zeros((3 * n, 3 * n)); ar = np.arange(3); I, J, U = [], [], []
    for i in range(n):
        dv = xyz - xyz[i]; dd = np.linalg.norm(dv, axis=1)
        for j in np.where((dd < cutoff) & (dd > 1e-6))[0]:
            if j <= i: continue
            I.append(i); J.append(j); U.append(dv[j] / dd[j])
    I, J, U = np.array(I), np.array(J), np.array(U); b = U[:, :, None] * U[:, None, :]
    def sc(a, c, V):
        r = (3*a[:, None, None] + ar[None, :, None]) * np.ones((1, 1, 3), int)
        co = (3*c[:, None, None] + ar[None, None, :]) * np.ones((1, 3, 1), int)
        np.add.at(H, (r.ravel(), co.ravel()), V.ravel())
    sc(I, I, b); sc(J, J, b); sc(I, J, -b); sc(J, I, -b)
    w, V = np.linalg.eigh(H); return V[:, 6:6 + K], w[6:6 + K]


# ---------------- strengthened metrics ----------------
def offdiag(C):
    n = C.shape[0]; return float(np.abs(C[~np.eye(n, dtype=bool)]).mean())

def excess_kurt(X):
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9); return float((((Xc ** 4).mean(0)) - 3).mean())

def marg_js(gen, ref, bins=30):
    js = []
    for i in range(gen.shape[1]):
        lo = min(gen[:, i].min(), ref[:, i].min()); hi = max(gen[:, i].max(), ref[:, i].max())
        e = np.linspace(lo, hi, bins + 1)
        pg = np.histogram(gen[:, i], e)[0] / len(gen) + 1e-12; pr = np.histogram(ref[:, i], e)[0] / len(ref) + 1e-12
        m = 0.5 * (pg + pr); js.append(0.5 * (pg * np.log(pg / m)).sum() + 0.5 * (pr * np.log(pr / m)).sum())
    return float(np.mean(js))

def iat_series(x, maxlag=400):
    x = x - x.mean(); v = (x * x).mean()
    if v < 1e-12: return 0.0
    acf = np.correlate(x, x, "full")[len(x) - 1:] / (v * len(x)); t = 1.0
    for k in range(1, min(len(acf), maxlag)):
        if acf[k] < 0.05: break
        t += 2 * acf[k]
    return t

def iat_ref_tau(Z, tau, maxk=200):                               # ref IAT in tau-units from stride-1 pairs
    out = []
    for i in range(Z.shape[1]):
        x = Z[:, i] - Z[:, i].mean(); v = (x * x).mean()
        if v < 1e-12: out.append(0.0); continue
        t = 1.0
        for k in range(1, maxk):
            lag = k * tau
            if lag >= len(x): break
            ac = (x[:-lag] * x[lag:]).mean() / v
            if ac < 0.05: break
            t += 2 * ac
        out.append(t)
    return np.mean(out)

def basins(X, thr):                                             # top-2-mode median quadrants
    return 2 * (X[:, 0] > thr[0]).astype(int) + (X[:, 1] > thr[1]).astype(int)

def trans_rate(labels):
    return float((labels[1:] != labels[:-1]).mean() * 1000)

def stats_of(X, top2, thr, ref_pool):
    """INBOX 77a. EVERY statistic from ONE series through ONE estimator.

    The version this replaces computed the generated statistics from a 1,500-step
    rollout and the reference statistics from the entire trajectory -- tens of
    thousands of frames. Length is not a neutral parameter for any of these
    estimators. An integrated autocorrelation time cannot be resolved much above a
    tenth of the series it is measured on; excess kurtosis over 1,500 CORRELATED
    points has an effective sample size of 1500/iat, which on this project's own
    n_eff numbers is a few dozen; a transition rate over 1,500 opportunities has a
    relative error the reference side simply does not have.

    So the two sides carried different estimator bias, and every gap read as the
    model failing -- Family F arriving through the estimator rather than through a
    join. Both sides now come through this function, on series of the SAME length and
    the SAME time-spacing (see `ref_windows`), so the bias is common and cancels.

    `js` is measured against a common pool for both sides, which keeps it comparable
    for the same reason: what matters is gen-vs-pool set beside refwindow-vs-pool.
    """
    m = {}
    m["std"] = float(X.std(0).mean())
    m["js"] = marg_js(X, ref_pool)
    m["kurt"] = excess_kurt(X)
    m["xcorr"] = offdiag(np.corrcoef(X.T))
    m["amp"] = offdiag(np.corrcoef(np.abs(X).T))
    m["iat"] = float(np.mean([iat_series(X[:, i]) for i in range(X.shape[1])]))
    m["trans"] = trans_rate(basins(X[:, top2], thr))
    return m


METRICS = ("std", "js", "kurt", "xcorr", "amp", "iat", "trans")


def ref_windows(Z, H, tau, K, rng):
    """INBOX 77a. K reference slices matched to an H-step rollout at lag `tau`.

    A rollout step ADVANCES BY TAU, so H steps span `H*tau` frames of trajectory.
    The comparable slice of reference is therefore tau-STRIDED, not tau consecutive
    frames -- matching only the count would compare 1,500 rollout steps covering
    150,000 frames against 1,500 frames covering 1,500. These windows match on all
    three of count, spacing and time-span, which is what makes any estimator applied
    to both unbiased relative to the other.

    Returns [] when the trajectory cannot supply even one window, which is a real
    limit at large tau and is reported rather than silently worked around.
    """
    span = H * tau
    if len(Z) <= span + 1:
        return []
    starts = rng.integers(0, len(Z) - span, size=K)
    return [Z[s:s + span:tau][:H] for s in starts]


def band(rows, key):
    """median and [min, max] of one statistic across a set of series."""
    v = np.array([r[key] for r in rows])
    return float(np.median(v)), float(v.min()), float(v.max())


def consistent(gen_rows, ref_rows, key):
    """Do the two spreads OVERLAP? -- the verdict, and it is deliberately not
    'is the generated median inside the reference band'.

    Measured on a synthetic control where both sides are draws from the SAME process,
    so every verdict should be 'consistent' and any miss is the test's own error:

        point-in-band      7-14% false misses
        interval overlap   0-3.6% false misses

    The point test is mis-calibrated because reference windows come from ONE
    trajectory and are correlated with each other, so their min-max is narrower than
    K independent draws would give, while the generated rollouts are genuinely
    independent. Comparing a point against that band therefore fails a correct model
    roughly one time in ten. Comparing the two intervals accounts for spread on both
    sides and holds up whether the windows overlap or not and across K.
    """
    _, glo, ghi = band(gen_rows, key)
    _, rlo, rhi = band(ref_rows, key)
    return (glo <= rhi) and (ghi >= rlo)


# ---------------- FiLM denoiser (per-layer conditioning) ----------------
class FiLMDenoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__()
        self.inp = nn.Linear(L + 16, h); self.h1 = nn.Linear(h, h); self.h2 = nn.Linear(h, h); self.out = nn.Linear(h, L)
        self.f0 = nn.Linear(L, 2 * h); self.f1 = nn.Linear(L, 2 * h); self.f2 = nn.Linear(L, 2 * h)
        for f in (self.f0, self.f1, self.f2):                     # zero-init FiLM -> identity at start (stable)
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
    def temb(self, t):
        f = torch.exp(torch.arange(8) * (-math.log(1e4) / 8)); a = t[:, None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)
    def film(self, lin, x, cond):
        g, b = lin(cond).chunk(2, -1); return torch.nn.functional.gelu(x * (1 + g) + b)
    def forward(self, zt, cond, t):
        x = self.film(self.f0, self.inp(torch.cat([zt, self.temb(t)], -1)), cond)
        x = self.film(self.f1, self.h1(x), cond); x = self.film(self.f2, self.h2(x), cond)
        return self.out(x)


betas = torch.linspace(1e-4, 0.02, Tdiff); ac = torch.cumprod(1 - betas, 0)


def train_ddpm(Zn, h, tau, param):
    m = FiLMDenoiser(L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    zc = Zn[:h - tau]; tgt = (Zn[tau:h] - Zn[:h - tau]) if param == "delta" else Zn[tau:h]
    for ep in range(1500):
        i = torch.randint(0, zc.shape[0], (min(128, zc.shape[0]),))
        c, x0 = zc[i], tgt[i]; k = torch.randint(0, Tdiff, (len(i),)); noise = torch.randn_like(x0)
        xk = ac[k].sqrt()[:, None] * x0 + (1 - ac[k]).sqrt()[:, None] * noise
        opt.zero_grad(); ((m(xk, c, k) - noise) ** 2).mean().backward(); opt.step()
    return m


@torch.no_grad()
def ddpm_step(m, cond):
    x = torch.randn(cond.shape[0], L)
    for k in reversed(range(Tdiff)):
        eps = m(x, cond, torch.full((cond.shape[0],), k)); x0 = (x - (1 - ac[k]).sqrt() * eps) / ac[k].sqrt()
        x = ac[k - 1].sqrt() * x0 + (1 - ac[k - 1]).sqrt() * torch.randn_like(x) if k > 0 else x0
    return x


def rollout_ddpm(m, z0, H, param):
    """INBOX 77a. BATCHED: z0 is (K, L) and the return is (K, H+1, L).

    K independent rollouts, not one. The version this replaces produced a single
    trajectory per configuration, so all 28 x 4 x 2 = 224 cells of the table were
    n=1 -- one roll of the dice, no spread. Across 224 cells some will beat OU by
    chance, and nothing in the output could tell those from a real effect.

    Costs almost nothing: `ddpm_step` is already batched over its conditioning, so K
    rollouts run as one batch of K through the SAME number of sequential denoiser
    calls. The wall-clock difference is a wider matmul, not K times the work.
    """
    z = z0.clone(); out = [z.clone()]
    for _ in range(H):
        s = ddpm_step(m, z); z = (z + s) if param == "delta" else s
        z = torch.nan_to_num(z, nan=0.0).clamp(-12, 12)          # guard blow-up (whitened space)
        out.append(z.clone())
    return torch.stack(out, 1).numpy()                            # (K, H+1, L)


ap = argparse.ArgumentParser(); ap.add_argument("--H", type=int, default=1500)
ap.add_argument("--K", type=int, default=32, help="INBOX 77a: rollouts AND reference windows per cell")
args = ap.parse_args()
K_EVAL = args.K          # n per cell on BOTH sides -- was 1 on the generated side.
                         # 32 rather than 8: the verdict is interval-overlap so it is not
                         # very K-sensitive, but the SPREAD it prints is, and a spread from
                         # 8 draws is not worth reading.
MIN_H = 200              # below this a series cannot carry these statistics at all
COVER_MIN = 20.0         # H/iat_r below this means iat is ceiling-limited (INBOX 77b)
SEED_BASE = 20250810
print(f"[propagator] tau sweep {TAUS}, FiLM cond, delta/absolute, vs OU. H={args.H} K={K_EVAL}")
print(f"  INBOX 77a/77b: generated and reference statistics now come from the SAME estimator on\n"
      f"  series of the SAME length and tau-spacing, K={K_EVAL} per side. A model is scored by whether\n"
      f"  it lands INSIDE the band real trajectory slices make (* = inside), not by beating a number.")
for dom in USE:
    with h5py.File(f"{DATA}/mdcath_dataset_{dom}.h5", "r") as f:
        g = f[dom]; z = np.array(g["z"]); N = len(z); nm = parse_names(g, N)
        heavy = z != 1; bb = np.isin(nm[heavy], BB); ca = nm[heavy] == "CA"
        coords = g[TEMP][R0]["coords"][:].astype(np.float64)[:, heavy, :][:, bb, :]
    al = kabsch(coords, ca[bb]); d = (al - al[0]).reshape(len(al), -1); T = d.shape[0]; h = int(T * 0.8)
    mean = d[:h].mean(0); modes, lam = anm(al[0], L); B = modes[:, :L].T; lam = lam[:L]
    Z = (d - mean) @ B.T; zmu = Z[:h].mean(0); sd = np.sqrt(0.593 / np.clip(lam, 1e-8, None))
    Zn_all = ((Z - zmu) / sd).astype(np.float32); ref_full = Z
    # INBOX 77c. Basins are defined on the TRAINING span only. Everything else here
    # already restricted correctly -- mean, zmu, v, a1, gamma are all fit on [:h] --
    # and this was the one target that was selected using frames the model never saw.
    top2 = np.argsort(Z[:h].std(0))[::-1][:2]; thr = np.median(Z[:h, top2], 0)
    # INBOX 77c. `ref_full` is used as the comparison pool on the argument that the
    # trajectory is stationary. That is the assumption this project has the most
    # evidence against: ATLAS replicas sit only 1.18x further apart than frames
    # within one replica, and n_eff runs 1-7%. Measure it instead of asserting it --
    # if the two halves disagree, the pool is contaminated by the training portion.
    _hf = len(Z) // 2
    stat_js = marg_js(Z[:_hf], Z[_hf:])
    print(f"  stationarity: JS(first half || second half) = {stat_js:.4f}"
          f"{'   <- halves DISAGREE; ref pool is not a clean equilibrium sample' if stat_js > 0.05 else ''}")
    v = kT / np.clip(lam, 1e-8, None); v *= (Z[:h].var(0).sum() / v.sum())
    a1 = (Z[:h - 1] * Z[1:h]).mean(0) / ((Z[:h - 1] ** 2).mean(0) + 1e-9)
    gamma = float(np.median((-lam / np.log(np.clip(a1, 0.02, 0.98)))[lam > 0]))
    print(f"\n=== {dom} (nCA {ca.sum()}) ===")
    for tau in TAUS:
        steps1ms = int(1e6 / tau)
        rng = np.random.default_rng(SEED_BASE + tau)

        # INBOX 77b. THE ROLLOUT LENGTH IS A CEILING AND IT MOVES WITH TAU.
        # A rollout step advances by tau, so H steps need H*tau frames of reference
        # to compare against. At large tau the trajectory cannot supply that, and the
        # honest response is to shorten BOTH sides together and say by how much --
        # not to leave the generated side capped while the reference side is not.
        H_use = min(args.H, (len(Z) - 1) // tau)
        REFW = ref_windows(Z, H_use, tau, K_EVAL, rng)
        iat_r_full = iat_ref_tau(ref_full, tau)
        cover = H_use / max(iat_r_full, 1e-9)
        if H_use < MIN_H or not REFW:
            print(f"  tau={tau:<4} UNEVALUABLE: H_use={H_use} over {len(Z)} frames at stride {tau} "
                  f"({len(REFW)} reference windows). Not reported -- at this lag the trajectory is "
                  f"too short to estimate these statistics on either side.")
            continue
        rs = [stats_of(w, top2, thr, ref_full) for w in REFW]
        flag = "" if cover >= COVER_MIN else f"  <- CEILING: H/iat_r={cover:.1f} < {COVER_MIN}, iat is not resolvable here"
        print(f"  tau={tau:<4} H={H_use}  refwindows={len(REFW)}  H/iat_r={cover:.1f}"
              f"  steps->1ms={steps1ms}{flag}")
        print(f"    {'model':14s}" + "".join(f"{k:>16s}" for k in METRICS))
        # The reference band is the NULL: what these statistics do across real slices
        # of trajectory measured exactly the way the generated side is measured. A
        # model is not asked to beat it, it is asked to land inside it.
        print(f"    {'REFERENCE':14s}" + "".join(
            f"{band(rs,k)[0]:.2f}[{band(rs,k)[1]:.2f},{band(rs,k)[2]:.2f}]".rjust(16) for k in METRICS))

        def report(name, series_list, extra=""):
            ss = [stats_of(s, top2, thr, ref_full) for s in series_list]
            cells, agree = [], 0
            for k in METRICS:
                med, lo, hi = band(ss, k)
                ok = consistent(ss, rs, k)
                agree += ok
                cells.append(f"{med:.2f}[{lo:.2f},{hi:.2f}]{'*' if ok else ''}".rjust(16))
            print(f"    {name:14s}" + "".join(cells) + f"   {agree}/{len(METRICS)} consistent {extra}")

        # OU at lag tau -- K independent realisations, same length as the reference
        # windows and as the DDPM rollouts. Every arm is now n=K, not n=1.
        ou = []
        for _ in range(K_EVAL):
            a_ou = np.exp(-lam * tau / gamma); x = Z[h].copy(); roll = [x.copy()]
            for _ in range(H_use):
                x = a_ou * x + np.sqrt(v * (1 - a_ou ** 2)) * rng.standard_normal(L); roll.append(x.copy())
            ou.append(np.array(roll)[:H_use])
        report("OU", ou)

        for param in ("absolute", "delta"):
            m = train_ddpm(torch.tensor(Zn_all), h, tau, param)
            # K rollouts from K DIFFERENT held-out start frames: this varies the
            # generative draw and the start point together, which is the sensitivity
            # a single rollout from a single frame cannot show.
            st = np.linspace(h, len(Zn_all) - 1, K_EVAL).astype(int)
            gen = rollout_ddpm(m, torch.tensor(Zn_all[st]), H_use, param) * sd + zmu
            # step-1 cond guard: a_ddpm vs a_ref at this lag
            idx = rng.choice(h - tau, min(150, h - tau), replace=False)
            with torch.no_grad():
                g1 = ddpm_step(m, torch.tensor(Zn_all[idx])).numpy()
            g1 = (Zn_all[idx] + g1) if param == "delta" else g1
            cnd = Zn_all[idx]
            add = ((cnd - cnd.mean(0)) * (g1 - g1.mean(0))).mean(0) / (((cnd - cnd.mean(0)) ** 2).mean(0) + 1e-9)
            aref_tau = (Zn_all[:h - tau] * Zn_all[tau:h]).mean(0) / ((Zn_all[:h - tau] ** 2).mean(0) + 1e-9)
            guard = f"cg{abs(add).mean()/max(abs(aref_tau).mean(),1e-6):.2f}"
            report("DDPM-" + param, [gen[k, :H_use] for k in range(gen.shape[0])], guard)
print("\n  discriminators: OU has xcorr~0, amp~0, kurt~0 BY CONSTRUCTION. If ref xcorr/amp/kurt ~0 too")
print("  -> task is Gaussian/single-basin at this lag, no learned propagator needed (pre-registered).")
print("  A learned win = matching ref xcorr/amp/kurt/trans that OU misses. cg = conditioning guard (a_ddpm/a_ref).")
