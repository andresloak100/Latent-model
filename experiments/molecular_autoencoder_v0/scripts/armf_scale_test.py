"""INBOX 29b: IS THE TIED COLLAPSE A WRONG MAGNITUDE OR A WRONG DIRECTION? Closed form, no retraining.

THE IDENTITY (verified numerically before use, not assumed). For a reconstruction `r` and a true
displacement `d`, the optimal per-system rescale is `a* = <r,d>/<r,r>`, and

    FVE at a*  ==  cos^2(r, d)          exactly

so `cos^2` is FVE with ALL magnitude error removed -- a pure DIRECTION measure -- and `cos^2 - FVE`
is exactly the portion of the error attributable to scale alone. `cos^2 >= FVE` always, with equality
only when the scale is already optimal. Checked: a 3x over-scaled reconstruction gives raw FVE
-3.0000 and cos^2 +1.0000, agreeing with the closed form to 0.00e+00.

WHY IT DECIDES SOMETHING. The tied arm is 2.5-2.7x the control on Q1/Q2/Q3 and inverts at Q4
(-0.2791). That is the shape of an architecture that is uniformly better and then BREAKS, not one
that is worse at large N. Under a pure over-scale by `k`, `FVE = 1 - (k-1)^2` exactly, so the
measured values imply k ~ 2.13 at the Q4 median and k ~ 4.01 at the worst system -- against
sqrt(N_max/N_Q1) = 4.82 predicted by the `||B||_F ~ sqrt(N)` synthesis hypothesis. Close enough to
test rather than admire.

    cos^2(Q4) ~ cos^2(Q1)   -> the SUBSPACE is right at every N and only the MAGNITUDE is wrong.
                               A calibration defect, plausibly removable.
    cos^2(Q4) << cos^2(Q1)  -> the DIRECTION degrades too. Structural; no rescale saves it, and
                               ||B||_F is at most part of the story.

AND THE MEDIATOR, in the 26a pattern rather than the downstream one: the over-scale hypothesis
predicts `a* ~ 1/sqrt(N)`, i.e. a slope of -0.5 for log10(a*) on log10(N). That is a sharper test
than the quartile comparison because it names an exponent. 26a is the precedent for measuring the
stated cause and letting it fail: no fix is proposed here, whichever branch fires."""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ho_frames
import armf_atlas_dm as D
import armf_io as IO
import armf_stamp as STAMP
from armf_modal_decoder import ModalCodec

WR = D.WR
CKPT = f"{WR}/modal_arm_ckpt"
# n_train of the checkpoints this script reads. Explicit because the directory now
# holds several (INBOX 43a); override to point at another rung.
NTRAIN = int(os.environ.get("PEER_NTRAIN", "50"))

def ckpt_path(kind, dm, lr, seed, n_train):
    """INBOX 43a/43b. One directory, two naming conventions, no discriminator -- armf_modal_arm.py
    wrote {kind}_dm_lr_s.pt at n_train=50 and the ladder writes ..._n{N}.pt for three rungs. The peer
    script read the n-less name, and a REAL n50 file sat at that path, so the n300 re-run would have
    loaded the n50 model and reported it as the lifted result: the old answer wearing the new run's
    label. Every checkpoint now names its n_train, and a caller that asks for one it cannot have gets
    an exception rather than a neighbour."""
    p = f"{CKPT}/{kind}_dm{dm}_lr{lr:g}_s{seed}_n{n_train}.pt"
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no checkpoint at {p}. Refusing to fall back to any other file -- a checkpoint the "
            f"caller did not name must never be loaded (INBOX 43b). Available: "
            f"{sorted(os.listdir(CKPT)) if os.path.isdir(CKPT) else '<no dir>'}")
    return p

RES = f"{WR}/scale_test.json"
DM, SEED = 256, 0
ARMS = [("tied", 3e-5), ("control", 3e-4), ("untied", 3e-4)]
# FULL window, not a subsample: with fewer frames the denominator is that window's own sst rather
# than d['sst'], so the raw FVE would not line up with the recorded per-system values the quartile
# medians are built from -- a denominator mismatch of exactly the kind this project keeps catching.
# Verified: at nf = d['F'] the pooled FVE matches fve_model to 2.2e-15.
NFRAME = 10**9
dev = D.dev


def build(kind, Fs):
    if kind == "control":
        return D.Codec(Fs, 1, DM).to(dev)
    return ModalCodec(Fs, 1, DM, tie_encoder=(kind == "tied")).to(dev)


@torch.no_grad()
def scale_stats(mdl, d, nf=NFRAME, chunk=8):
    """Pooled over frames, on the SAME mu-centred displacement `fve_model` uses:
    raw FVE, cos^2(r,d), and the optimal rescale a* = <r,d>/<r,r>."""
    F = min(nf, d["F"])
    st = torch.tensor(d["stat"], device=dev)
    rd = rr = dd = 0.0
    sse = 0.0
    for s0 in range(0, F, chunk):
        e0 = min(s0 + chunk, F)
        t = ho_frames(d, s0, e0)                                  # (b, N, 3), mu-centred
        dw = torch.tensor((t / d["scale"]).astype(np.float32), device=dev)
        p = mdl(st.unsqueeze(0).expand(e0 - s0, -1, -1), dw).cpu().numpy().astype(np.float64) \
            * d["scale"]
        rd += float((p * t).sum()); rr += float((p * p).sum()); dd += float((t * t).sum())
        sse += float(((t - p) ** 2).sum())
    fve = 1.0 - sse / (dd + 1e-12)
    cos2 = (rd * rd) / ((rr * dd) + 1e-30)
    astar = rd / (rr + 1e-30)
    return dict(fve=float(fve), cos2=float(cos2), astar=float(astar), N=d["N"])


if __name__ == "__main__":
    print(f"[scale-test] INBOX 29b: magnitude or direction? cos^2 = FVE with all scale error "
          f"removed. {NFRAME} frames/system, existing checkpoints, no retraining.", flush=True)
    # the identity, re-verified in the job's own output so the number has a stated provenance
    rngc = np.random.default_rng(0); dv = rngc.normal(size=4000); rv = 3 * dv
    _raw = 1 - ((dv - rv) ** 2).sum() / (dv ** 2).sum()
    _a = (rv @ dv) / (rv @ rv); _opt = 1 - ((dv - _a * rv) ** 2).sum() / (dv ** 2).sum()
    _c2 = (rv @ dv) ** 2 / ((rv @ rv) * (dv @ dv))
    print(f"  [identity check] 3x over-scale: raw FVE {_raw:+.4f}, FVE at a* {_opt:+.4f}, "
          f"cos^2 {_c2:+.4f}, |diff| {abs(_opt-_c2):.1e}", flush=True)

    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    HO.sort(key=lambda d: d["N"])
    Fs = HO[0]["stat"].shape[1]
    allN = np.array([d["N"] for d in HO], float)
    q = np.quantile(allN, [0.25, 0.5, 0.75])
    bands = [(0, q[0], "Q1"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1e18, "Q4")]
    print(f"  {len(HO)} systems, quartile boundaries {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}", flush=True)

    ST = STAMP.stamp(dict(dm=DM, seed=SEED, nframe=NFRAME, arms=str(ARMS)), ModalCodec, scale_stats)
    # INBOX 74b: RESUME read -- accepts a partial file by design. The analysis side
    # must not; see the complete=True write after the arms loop.
    res, _was_complete, _ = IO.load_partial(RES)
    if not isinstance(res, dict): res = {}
    res = {k: v for k, v in res.items() if v and STAMP.same_stamp(list(v.values())[0], ST)} \
        if res else {}
    # The unit of work is the (arm, system) CELL, not the arm, so conservation is
    # counted in cells. Arms skipped for a missing checkpoint are accounted as failed
    # rather than quietly shrinking the denominator.
    CELLS = len(HO) * len(ARMS)
    cells = lambda r: sum(len(v) for v in r.values())
    nfail = 0

    for kind, lr in ARMS:
        cp = ckpt_path(kind, DM, lr, SEED, NTRAIN)
        if not os.path.exists(cp):
            print(f"  {kind} lr{lr:g}: no checkpoint -- skipped", flush=True)
            nfail += len(HO); continue
        if len(res.get(kind, {})) >= len(HO): continue
        mdl = build(kind, Fs)
        mdl.load_state_dict(torch.load(cp, map_location=dev)); mdl.eval()
        rows, t0 = res.get(kind, {}), time.time()
        res[kind] = rows
        for i, d in enumerate(HO):
            if d["pdb"] in rows: continue
            try:
                r = scale_stats(mdl, d); r["stamp"] = ST; rows[d["pdb"]] = r
            except Exception as e:
                print(f"    {d['pdb']} N={d['N']}: FAIL {type(e).__name__}: {e}", flush=True)
                nfail += 1; continue
            # PER-SYSTEM checkpoint: the full window makes each arm a multi-hour unit, and a
            # preemption that loses a whole arm is a whole arm redone. Ascending N also means a
            # partial arm is a SIZE-TRUNCATED sample, which the 18d coverage_by call below makes
            # visible -- but only to a run that REACHES it. INBOX 74b: a killed run never does, and
            # the file it leaves is valid, parseable, and missing its large-N tail with nothing on
            # disk saying so. Partial-by-default here; the declaration goes on after the arms loop.
            IO.dump_rows(RES, res, n_expected=CELLS, n_present=cells(res), n_failed=nfail)
            if (i + 1) % 20 == 0:
                print(f"    {kind} {i+1}/{len(HO)} ({(time.time()-t0)/60:.0f} min)", flush=True)
        print(f"  {kind}: {len(rows)}/{len(HO)} systems ({(time.time()-t0)/60:.0f} min)", flush=True)
        del mdl

    # INBOX 74b: every arm has run to the end of HO, so the file may now be analysed.
    IO.dump_rows(RES, res, n_expected=CELLS, n_present=cells(res), n_failed=nfail, complete=True)

    for kind, _ in ARMS:
        if kind not in res: continue
        got = {r["N"] for r in res[kind].values()}
        STAMP.coverage_by(sorted(got), sorted(set(allN.astype(int)) - got), f"N ({kind})")
    print(f"\n=== 29b: RAW FVE vs cos^2 BY QUARTILE (cos^2 = FVE with scale error removed) ===",
          flush=True)
    print(f"    {'arm':>9}{'band':>5}{'n':>5}{'raw FVE':>10}{'cos^2':>10}{'scale cost':>12}"
          f"{'a* median':>11}{'implied k':>11}", flush=True)
    summary = {}
    for kind, _ in ARMS:
        if kind not in res: continue
        v = res[kind]
        N = np.array([r["N"] for r in v.values()], float)
        F_ = np.array([r["fve"] for r in v.values()], float)
        C_ = np.array([r["cos2"] for r in v.values()], float)
        A_ = np.array([r["astar"] for r in v.values()], float)
        summary[kind] = {}
        for lo, hi, lab in bands:
            m = (N >= lo) & (N < hi)
            if m.sum() < 3: continue
            mf, mc, ma = float(np.median(F_[m])), float(np.median(C_[m])), float(np.median(A_[m]))
            summary[kind][lab] = (mf, mc, ma)
            print(f"    {kind:>9}{lab:>5}{int(m.sum()):>5}{mf:>+10.4f}{mc:>+10.4f}"
                  f"{mc-mf:>+12.4f}{ma:>11.3f}{(1/ma if ma > 1e-6 else float('nan')):>11.2f}",
                  flush=True)

    print(f"\n=== THE MEDIATOR (29a): does a* scale as 1/sqrt(N)? ===", flush=True)
    print(f"  the ||B||_F ~ sqrt(N) synthesis hypothesis predicts slope -0.5 for log10(a*) vs log10(N)",
          flush=True)
    for kind, _ in ARMS:
        if kind not in res: continue
        v = res[kind]
        N = np.array([r["N"] for r in v.values()], float)
        A_ = np.array([r["astar"] for r in v.values()], float)
        g = np.isfinite(A_) & (A_ > 0)
        if g.sum() < 10: continue
        lr_ = stats.linregress(np.log10(N[g]), np.log10(A_[g]))
        h = stats.t.ppf(0.975, int(g.sum()) - 2) * lr_.stderr
        has = (lr_.slope - h) <= -0.5 <= (lr_.slope + h)
        print(f"    {kind:>9}: slope {lr_.slope:+.4f} +/- {h:.4f}  R^2 {lr_.rvalue**2:.3f}  -> "
              f"{'CONSISTENT with -0.5' if has else 'NOT -0.5'}", flush=True)

    print(f"\n=== VERDICT (29b, pre-registered) ===", flush=True)
    t = summary.get("tied", {})
    if "Q1" in t and "Q4" in t:
        c1, c4 = t["Q1"][1], t["Q4"][1]
        print(f"  tied cos^2: Q1 {c1:+.4f} -> Q4 {c4:+.4f}   (raw FVE {t['Q1'][0]:+.4f} -> "
              f"{t['Q4'][0]:+.4f})", flush=True)
        if c4 >= 0.7 * c1:
            print(f"  => THE SUBSPACE IS RIGHT AT EVERY N AND ONLY THE MAGNITUDE IS WRONG. The Q4")
            print(f"     collapse is a CALIBRATION defect: cos^2 holds at {c4:+.4f} while raw FVE is")
            print(f"     {t['Q4'][0]:+.4f}, so {t['Q4'][1]-t['Q4'][0]:+.4f} of the loss is scale alone.")
            print(f"     Per 29b no fix is proposed here -- the branch is stated, not acted on.",
                  flush=True)
        else:
            print(f"  => THE DIRECTION DEGRADES TOO ({c4:+.4f} against {c1:+.4f} at Q1). No rescale")
            print(f"     saves it, the failure is STRUCTURAL, and ||B||_F ~ sqrt(N) is at most part")
            print(f"     of the story.", flush=True)
    print(f"\n  NOTE: cos^2 is a DIRECTION measure and is NOT a claim about achievable FVE -- a", flush=True)
    print(f"  per-system rescale is not available zero-shot, so it is a diagnostic, not a result.",
          flush=True)
