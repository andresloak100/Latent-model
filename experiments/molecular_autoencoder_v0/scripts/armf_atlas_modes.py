"""INBOX 14b: MODE-RESOLVED FVE. Not how much is wrong -- WHERE the error lives.

FVE ~ 0.155 is one number hiding which part of the dynamics survives the bottleneck. Decomposing it
separates two readings with completely different consequences:

  SLOW-SELECTIVE  FVE high on slow/collective modes, near zero on fast ones. The codec is keeping
                  what matters and discarding thermal noise. This would EXPLAIN the flat FVE-vs-N:
                  collective content is low-dimensional and genuinely does not grow with N, while the
                  discarded high-rank remainder is what grows as N^0.93. Plain MSE would then be
                  DEMONSTRABLY the wrong objective rather than merely suspected of it.
  UNIFORM         FVE roughly flat across timescales. The model is uniformly weak, the flat slope
                  carries no architectural information, and only the ladder can change the picture.

THE BASIS IS THE REFERENCE'S, FOR BOTH TRAJECTORIES -- the same convention as the criterion-1
harness. A per-trajectory basis would let a decoder that ROTATED the dynamics into other directions
score perfectly; that is a Family D failure, a measurement structurally unable to express the defect
it is named for. Here the basis is a fixed coordinate system, not a fit being scored, so using the
reference's own frames for it is correct rather than in-sample flattery: both the true and the
reconstructed displacement are projected through the SAME fixed matrix.

THE CONTROL THAT DECIDES WHETHER THIS MEASURES ANYTHING (and it is not optional).
Mode index, variance and IAT are heavily confounded: low-index modes carry the most variance AND are
usually the slowest. MSE minimises squared error, so it favours HIGH-VARIANCE modes BY CONSTRUCTION.
Finding "FVE is high on modes that are slow" is therefore NOT evidence for the slow-selective
reading -- it is the expected consequence of training on MSE, arriving in different clothes.
So the verdict is taken from a TWO-VARIABLE regression of per-mode FVE on log(IAT) AND log(variance
share) together. Only the log(IAT) coefficient, holding variance fixed, can support the interesting
reading. The one-variable version is reported alongside precisely so the difference is visible.

PRE-REGISTERED, WRITTEN BEFORE RESULTS:
  SLOW-SELECTIVE requires BOTH
     (i)  FVE(slowest IAT tercile) - FVE(fastest IAT tercile) > 0.10, CI across systems excluding 0
     (ii) the PARTIAL log(IAT) coefficient (variance held fixed) positive with CI excluding 0
  UNIFORM   requires the tercile difference CI to include 0 AND |difference| < 0.05
  otherwise INTERMEDIATE -- report the numbers, force no reading.
If (i) holds and (ii) does not, the honest statement is "the codec captures HIGH-VARIANCE modes,
which is what MSE selects for" -- NOT a timescale claim."""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ho_frames
import armf_atlas_dm as D
from armf_criterion1 import iat
import armf_stamp as STAMP

WR = D.WR
RES = f"{WR}/atlas_modes.json"
NMODE = 30           # modes resolved per system
NFRAME = 2000        # CONSECUTIVE held-out frames -- IAT is meaningless on a strided sample
NSYS = 24            # N-stratified held-out systems
ANM_K = [6, 16]      # INBOX 25a: matched to the codec's realised rank, and to the ladder
ANM_CUTOFF = 5.0     # the cutoff selected on TRAINING systems by armf_atlas_peer.py
dev = D.dev


def mode_table(mdl, d, k=NMODE, nf=NFRAME, chunk=8):
    """Per-mode FVE, variance share and reference IAT, in the reference's own PCA basis."""
    F = min(nf, d["F"])
    ref = ho_frames(d, 0, F).reshape(F, d["N"], 3)
    st = torch.tensor(d["stat"], device=dev)
    dec = np.empty_like(ref)
    with torch.no_grad():
        for s0 in range(0, F, chunk):
            e0 = min(s0 + chunk, F)
            dw = torch.tensor((ref[s0:e0] / d["scale"]).astype(np.float32), device=dev)
            dec[s0:e0] = mdl(st.unsqueeze(0).expand(e0 - s0, -1, -1), dw).cpu().numpy() * d["scale"]

    X = ref.reshape(F, -1)                       # already mu-centred by ho_frames
    Y = dec.reshape(F, -1)
    G = X @ X.T
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]
    w = np.clip(w[o], 0, None); U = U[:, o]
    kk = min(k, int((w > w[0] * 1e-12).sum()))
    s = np.sqrt(w[:kk]) + 1e-12
    A = U[:, :kk] * s                            # true projection: X @ V, without ever forming V
    B = (Y @ X.T) @ (U[:, :kk] / s)              # decoded projection through the SAME fixed basis

    num = ((A - B) ** 2).sum(0)                  # per-mode residual energy
    den = (A ** 2).sum(0)
    fve = 1.0 - num / (den + 1e-12)

    # INBOX 16a: THE REALISED RANK OF THE DECODER'S OUTPUT.
    # Per-mode FVE going negative past mode ~6 SUGGESTS the outputs live in a low-dimensional
    # subspace; this MEASURES that subspace instead of inferring it. Both matrices are centred on
    # their OWN frame-mean, because the question is how many directions the output actually varies
    # along -- a constant offset is not a mode. The data's rank on the SAME frames is computed the
    # same way so the two numbers are comparable.
    def rank_at(M, frac):
        Mc = M - M.mean(0)
        w = np.clip(np.linalg.eigvalsh(Mc @ Mc.T), 0, None)[::-1]
        t = w.sum()
        if t <= 0: return 0
        return int(np.searchsorted(np.cumsum(w) / t, frac) + 1)

    # INBOX 25a-1: FVE ON THE ANM-ORTHOGONAL RESIDUAL -- the direct test of whether the codec is
    # more than a collective-mode model. MD displacement variance is dominated by a few slow
    # collective modes, so aggregate FVE cannot separate "encodes the dynamic state" from "encodes
    # the top six modes": a model reproducing only those scores well BY CONSTRUCTION. 14b's
    # mode-resolved FVE says where the captured variance sits; it does not say whether the codec
    # captures anything THE PEER DOES NOT.
    #   FVE_perp = 1 - (||E||^2 - ||E V||^2) / (sst - ||ho V||^2),  E = true - decoded
    # No (F, 3N) residual is ever formed -- only the two projected norms.
    orth = {}
    try:
        from armf_anm import modes as anm_modes
        E = X - Y
        e2 = float((E ** 2).sum()); h2 = float((X ** 2).sum())
        for kk_ in ANM_K:
            V, _, _ = anm_modes(d["ref"], kk_, ANM_CUTOFF)
            if V is None: continue
            # NAMES DELIBERATELY DISTINCT: `num`/`den` already exist in this function as the
            # PER-MODE arrays, and shadowing them here made the return statement iterate a float.
            onum = e2 - float(((E @ V) ** 2).sum())
            oden = h2 - float(((X @ V) ** 2).sum())
            orth[f"fve_perp_anm{kk_}"] = (float(1.0 - onum / (oden + 1e-12))
                                          if oden > 1e-9 else float("nan"))
            orth[f"anm{kk_}_share"] = float(1.0 - oden / (h2 + 1e-12))
    except Exception as e:
        orth["orth_err"] = f"{type(e).__name__}: {e}"

    # INBOX 25a-2: THE PER-ATOM ERROR DISTRIBUTION, NOT ITS MEAN. A summary model is right on the
    # mobile core and wrong in the tail, and a mean hides exactly that. Each atom's error is
    # normalised by ITS OWN motion amplitude, so a quiet atom is not credited for being quiet.
    ea = np.sqrt(((ref - dec) ** 2).sum(-1).mean(0))         # per-atom RMS error, (N,)
    aa = np.sqrt((ref ** 2).sum(-1).mean(0)) + 1e-12         # per-atom RMS amplitude, (N,)
    rat = ea / aa
    return dict(pdb=d["pdb"], N=d["N"], F=F,
                rank90_out=rank_at(Y, 0.90), rank99_out=rank_at(Y, 0.99),
                rank90_data=rank_at(X, 0.90),
                peratom_med=float(np.median(rat)), peratom_p90=float(np.percentile(rat, 90)),
                peratom_p99=float(np.percentile(rat, 99)),
                **orth,
                fve=[float(v) for v in fve],
                # INBOX 16b: report the residual AGAINST THE PREDICT-ZERO BASELINE explicitly.
                # Under MSE a capacity-limited model will optimally PUSH error into low-variance
                # modes to buy a better fit on high-variance ones, so a ratio above 1.0 is the
                # signature of a restricted function class -- not a bug to hunt. Making it a
                # visible number beats leaving it implied by a minus sign on FVE.
                err_ratio=[float(v) for v in num / (den + 1e-12)],
                var=[float(v) for v in den / (den.sum() + 1e-12)],
                var_abs=[float(v) for v in den],
                tau=[float(iat(A[:, i])) for i in range(kk)],
                # absolute scale, so "FVE 0.15" is anchored to something physical
                rmsd_ref=float(np.sqrt(((ref - dec) ** 2).sum(-1).mean())),
                amp_ref=float(np.sqrt((ref ** 2).sum(-1).mean())))


def regress(x, y):
    lr = stats.linregress(x, y)
    return lr.slope, stats.t.ppf(0.975, max(len(x) - 2, 1)) * lr.stderr


def partial(logtau, logvar, fve):
    """OLS of FVE on [log IAT, log variance share]. The log(IAT) coefficient HOLDING VARIANCE FIXED
    is the only one that can support a timescale claim, because MSE already selects for variance."""
    Xd = np.column_stack([np.ones_like(logtau), logtau, logvar])
    n, p = Xd.shape
    if n <= p: return (np.nan,) * 4
    beta, *_ = np.linalg.lstsq(Xd, fve, rcond=None)
    r = fve - Xd @ beta
    s2 = float(r @ r) / (n - p)
    cov = s2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = stats.t.ppf(0.975, n - p)
    return float(beta[1]), float(t * se[1]), float(beta[2]), float(t * se[2])


if __name__ == "__main__":
    print(f"[atlas-modes] 14b: WHERE does the codec's error live? {NMODE} modes, "
          f"{NFRAME} consecutive frames, {NSYS} N-stratified held-out systems.", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]

    rows = json.load(open(D.RES)) if os.path.exists(D.RES) else []
    cand = [r for r in rows if r.get("L") == 1 and r.get("arch") == "network" and not r["improving"]]
    if not cand:
        print("  no CONVERGED L=1 network arm in atlas_dm.json -- 14b needs a trained arm. "
              "Re-run once the sweep records one.", flush=True); raise SystemExit
    b = max(cand, key=lambda r: r["fve"])
    print(f"  BEST ARM: n_train={b['n_train']} DM={b['dm']} lr={b['lr']:g} seed={b['seed']}  "
          f"held-out FVE {b['fve']:+.4f}  PR {b['pr']:.1f}  N-slope {b['nslope']:+.4f}"
          f"+/-{b['nslope_ci']:.4f}", flush=True)

    HOall = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    HOall.sort(key=lambda d: d["N"])
    HO = [HOall[i] for i in np.linspace(0, len(HOall) - 1, min(NSYS, len(HOall))).astype(int)]
    print(f"  N-stratified sample: {len(HO)} systems, N {HO[0]['N']}-{HO[-1]['N']}", flush=True)

    cp = (f"{D.CKPT}/L1_n{b['n_train']}_dm{b['dm']}_dl{b['dlat']}_lr{b['lr']:g}"
          f"_s{b['seed']}_z{int(b.get('sub_z0', False))}.pt")
    mdl = D.Codec(HO[0]["stat"].shape[1], 1, b["dm"], dlat=b["dlat"],
                  sub_z0=bool(b.get("sub_z0", False))).to(dev)
    if os.path.exists(cp):
        mdl.load_state_dict(torch.load(cp, map_location=dev))
        print(f"  loaded checkpoint {os.path.basename(cp)}", flush=True)
    else:
        # Arms finished before checkpointing was wired have no state to load. Re-training the SAME
        # (n_train, DM, lr, seed) is the honest recovery; it is one arm, not the ladder.
        print(f"  no checkpoint for this arm (it predates checkpointing) -- RE-TRAINING it. "
              f"Same n_train/DM/lr/seed, so it is the same arm, but seed-level reproduction is not "
              f"guaranteed bit-for-bit; the FVE it reaches is printed for comparison.", flush=True)
        TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:b["n_train"]]) if x is not None]
        HOt = [HOall[i] for i in np.linspace(0, len(HOall) - 1, 24).astype(int)]
        torch.manual_seed(b["seed"]); np.random.seed(b["seed"] + 1)
        mdl, hist, used, stopped, improving = D.train(TR, HOt, b["dm"], b["lr"], "14b-retrain", 1,
                                                      dlat=b["dlat"])
        os.makedirs(D.CKPT, exist_ok=True); torch.save(mdl.state_dict(), cp)
        # COMPARE LIKE WITH LIKE. `b["fve"]` is the mean over ALL held-out systems; what a re-run
        # produces here is the mean over the 24 N-STRATIFIED TRACK systems, and those two differ
        # SYSTEMATICALLY -- for the recorded DM=256 arms, 0.1453/0.1553/0.1497 over all 123 against
        # 0.0969/0.0913/0.0958 over the 24, a consistent ~1.6x. The stratified sample deliberately
        # over-weights the large end, so it is the harder set, not a noisier estimate of the same
        # thing. Checking a re-run against `fve` would therefore report a divergence that is really
        # a change of denominator -- the Family F shape (comparator computed on different data),
        # committed against my own arm.
        got = float(np.mean([D.fve_model(mdl, x) for x in HOt]))
        ref = b.get("best_track", float("nan"))
        agree = ("consistent" if abs(got - ref) < 0.03 else
                 "*** DIVERGED -- treat this as a RE-RUN, not a reproduction of the recorded arm ***")
        print(f"  re-trained: tracked FVE {got:+.4f} vs the recorded arm's BEST TRACKED {ref:+.4f} "
              f"on the same 24 systems ({agree})", flush=True)
        print(f"  (the recorded arm's all-{b['nho']}-system FVE was {b['fve']:+.4f}; that is a "
              f"DIFFERENT denominator and is not the comparison to make here)", flush=True)
        del TR
    mdl.eval()

    # INBOX 26d/26c made the REPORT change while the per-system computation stayed identical, and a
    # re-run recomputed everything to print different summary statistics. Resume on the 18a stamp so
    # a reporting change costs nothing -- the same reason the coverage analysis was kept
    # reporting-only in 23a.
    ST = STAMP.stamp(dict(nmode=NMODE, nframe=NFRAME, nsys=NSYS, anm_k=str(ANM_K),
                          anm_cutoff=ANM_CUTOFF, arm=f"{b['n_train']}_{b['dm']}_{b['lr']:g}_{b['seed']}"),
                     mode_table, D.Codec)
    prev = []
    if os.path.exists(RES):
        try:
            pj = json.load(open(RES))
            prev = [r for r in pj.get("sys", []) if STAMP.same_stamp(r, ST)]
        except Exception:
            prev = []
    if prev:
        print(f"  [resume] {len(prev)} systems already scored under this stamp -- recomputing only "
              f"the rest, then re-reporting.", flush=True)
    have_pdb = {r["pdb"] for r in prev}
    out = list(prev)
    for i, d in enumerate(HO):
        if d["pdb"] in have_pdb: continue
        t0 = time.time()
        try:
            r = mode_table(mdl, d)
        except Exception as e:
            print(f"    {d['pdb']} N={d['N']}: FAIL {type(e).__name__}: {e}", flush=True); continue
        r["stamp"] = ST
        out.append(r)
        print(f"    {i+1}/{len(HO)} {d['pdb']} N={d['N']:>6}  FVE mode1 {r['fve'][0]:+.3f}  "
              f"mode10 {r['fve'][9] if len(r['fve']) > 9 else float('nan'):+.3f}  "
              f"tau1 {r['tau'][0]:.0f} fr  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(arm={k: v for k, v in b.items() if k not in ("per", "Ns")}, sys=out),
                  open(RES, "w"))
    if not out:
        print("  nothing scored."); raise SystemExit

    # ---------------- REPORT ----------------
    K = min(len(r["fve"]) for r in out)
    Fv = np.array([r["fve"][:K] for r in out])           # (systems, modes)
    Tv = np.array([r["tau"][:K] for r in out])
    Vv = np.array([r["var"][:K] for r in out])

    print(f"\n=== ABSOLUTE SCALE FIRST (FVE is a ratio and hides magnitude) ===", flush=True)
    rm = np.array([r["rmsd_ref"] for r in out]); am = np.array([r["amp_ref"] for r in out])
    print(f"  reconstruction RMSD  median {np.median(rm):.3f} A   range {rm.min():.3f}-{rm.max():.3f} A")
    print(f"  displacement RMS     median {np.median(am):.3f} A   range {am.min():.3f}-{am.max():.3f} A")
    print(f"  i.e. the residual is {100*np.median(rm)/np.median(am):.0f}% of the motion's own "
          f"amplitude, in absolute Angstroms.", flush=True)

    print(f"\n=== 16a: THE REALISED RANK OF THE DECODER'S OUTPUT ===", flush=True)
    print(f"  SVD of the RECONSTRUCTED displacement per system: how many directions does the output")
    print(f"  actually vary along? This separates three hypotheses the DM sweep cannot.", flush=True)
    r90 = np.array([r["rank90_out"] for r in out], float)
    r99 = np.array([r["rank99_out"] for r in out], float)
    rdat = np.array([r["rank90_data"] for r in out], float)
    dm_arm = b["dm"]; pr_arm = b.get("pr", float("nan"))
    print(f"    realised rank90 of the RECONSTRUCTION : median {np.median(r90):.0f}   "
          f"range {r90.min():.0f}-{r90.max():.0f}")
    print(f"    realised rank99 of the RECONSTRUCTION : median {np.median(r99):.0f}   "
          f"range {r99.min():.0f}-{r99.max():.0f}")
    print(f"    rank90 of the DATA on the same frames : median {np.median(rdat):.0f}   "
          f"range {rdat.min():.0f}-{rdat.max():.0f}")
    print(f"    for comparison: DM = {dm_arm}, latent PR = {pr_arm:.1f}", flush=True)
    med = float(np.median(r90))
    # INBOX 19a: print what this verdict would have been across EVERY free choice it contains,
    # unconditionally and whether or not anything looks wrong. The first version of this branch used
    # a hand-picked constant that sat 0.7% from the measurement and printed the OPPOSITE conclusion;
    # sensitivity is what turns "the numbers were right and the reading was wrong" from something
    # caught by review into something the output cannot hide.
    STAMP.verdict_sensitivity(
        lambda v, t: "DECODER-limited" if v < t else "not decoder-limited",
        {"realised rank90": med, "realised rank99": float(np.median(r99))},
        0.6 * pr_arm, scales=(0.5, 1.0, 2.0), label="16a realised rank")
    # THRESHOLD CORRECTED AFTER THE FIRST RUN, and the correction is recorded because it changed the
    # printed verdict. The original test was `med <= max(3.0, 2*PR/5)`, an invented constant that
    # evaluated to 5.96 against a measured median of 6.0 -- so it selected "realised rank ~ PR" on a
    # margin of 0.04, at rounding scale, when the measured ratio is 6/14.9 = 0.40. The question 16a
    # actually poses is whether the OUTPUT rank is materially BELOW the latent's effective rank, so
    # the test is now that ratio directly, with no free constant tuned to nothing.
    if med < 0.6 * pr_arm:
        print(f"    => THE DECODER CANNOT CONVERT LATENT DIMENSIONS INTO OUTPUT MODES. Realised rank")
        print(f"       {med:.0f} against DM={dm_arm} and PR={pr_arm:.1f}: the binding constraint is the")
        print(f"       DECODER'S FUNCTION CLASS -- not capacity, not data, not DM. More width and more")
        print(f"       LR arms have low marginal value; changing the decoder is the lever (INBOX 015).",
              flush=True)
    elif med <= 1.6 * pr_arm:
        print(f"    => REALISED RANK ~ PR ({med:.0f} vs {pr_arm:.1f}). The LATENT is the limit, so")
        print(f"       capacity work is the right lever and the DM sweep is answering a real question.",
              flush=True)
        print(f"       (ratio {med/max(pr_arm,1e-9):.2f}; this branch requires the output rank to be")
        print(f"        COMPARABLE to the latent's effective rank, not merely below it.)", flush=True)
    elif med >= 0.5 * dm_arm:
        print(f"    => REALISED RANK ~ DM ({med:.0f} of {dm_arm}). The decoder is EXPRESSIVE; the")
        print(f"       problem is upstream, in the encoder or the objective, and 015 is unnecessary.",
              flush=True)
    else:
        print(f"    => INTERMEDIATE: {med:.0f}, between PR ({pr_arm:.1f}) and DM ({dm_arm}). Report the")
        print(f"       number; no single hypothesis is selected.", flush=True)
    print(f"    (the DATA needs {np.median(rdat):.0f} directions on these frames, so the reconstruction "
          f"spans {100*np.median(r90)/max(np.median(rdat), 1):.0f}% of what the motion does)", flush=True)

    # ---------------- INBOX 25a REPORT ----------------
    print(f"\n=== 25a-1: IS THE CODEC MORE THAN A COLLECTIVE-MODE MODEL? ===", flush=True)
    print(f"  FVE on the ANM-ORTHOGONAL RESIDUAL: of the motion the zero-shot peer does NOT span,")
    print(f"  how much does the codec explain? Aggregate FVE cannot answer this -- a model that")
    print(f"  reproduces only the top collective modes scores well on it by construction.", flush=True)
    print(f"  INBOX 26d: '~ 0' is decided as a DISTRIBUTION, not a point -- median, IQR and the")
    print(f"  fraction of systems above zero. ZERO IS THE CONSTRUCTED BOUNDARY, not a chosen")
    print(f"  threshold: a model reproducing the ANM subspace exactly and nothing else gives")
    print(f"  FVE_perp = 0 identically. So this stays inside 21b's threshold-free requirement.",
          flush=True)
    for kk_ in ANM_K:
        v = np.array([r[f"fve_perp_anm{kk_}"] for r in out if f"fve_perp_anm{kk_}" in r
                      and np.isfinite(r[f"fve_perp_anm{kk_}"])], float)
        sh = np.array([r[f"anm{kk_}_share"] for r in out if f"anm{kk_}_share" in r], float)
        if not len(v): 
            print(f"    ANM-{kk_}: not computed"); continue
        Nv = np.log10(np.array([r["N"] for r in out if f"fve_perp_anm{kk_}" in r
                                and np.isfinite(r[f"fve_perp_anm{kk_}"])], float))
        sl, h = regress(Nv, v)
        print(f"    ANM-{kk_} spans {100*np.median(sh):.0f}% of the motion; on the REMAINDER the codec")
        q1, q3 = np.percentile(v, [25, 75])
        print(f"      FVE_perp median {np.median(v):+.4f}   IQR [{q1:+.4f}, {q3:+.4f}]   "
              f"range [{v.min():+.3f}, {v.max():+.3f}]")
        print(f"      ABOVE ZERO on {100*np.mean(v > 0):.0f}% of {len(v)} systems"
              f"   (n>0 = {int((v > 0).sum())})")
        print(f"      vs log10(N): {sl:+.4f} +/- {h:.4f}", flush=True)

    print(f"\n=== 25a-2: PER-ATOM ERROR, NORMALISED BY EACH ATOM'S OWN AMPLITUDE ===", flush=True)
    pm = np.array([r["peratom_med"] for r in out], float)
    p9 = np.array([r["peratom_p90"] for r in out], float)
    p99 = np.array([r["peratom_p99"] for r in out], float)
    print(f"  median atom   {np.median(pm):.3f}   p90 atom {np.median(p9):.3f}   "
          f"p99 atom {np.median(p99):.3f}   (1.0 = error equals the atom's own motion)")
    print(f"  tail width p90/median: {np.median(p9/pm):.2f}x -- a mean would have hidden this",
          flush=True)

    # INBOX 26c: a metric with no recorded FLOOR invites reading a small positive value as a result.
    # The untrained value is computed here, on ONE system, and printed beside the trained numbers.
    try:
        m0 = D.Codec(HO[0]["stat"].shape[1], 1, b["dm"], dlat=b["dlat"]).to(dev); m0.eval()
        r0 = mode_table(m0, HO[len(HO) // 2])
        print(f"\n  [26c NULL] same metrics on an UNTRAINED model, {r0['pdb']}: "
              f"FVE_perp(ANM-{ANM_K[0]}) {r0.get(f'fve_perp_anm{ANM_K[0]}', float('nan')):+.3f}   "
              f"per-atom med {r0['peratom_med']:.3f} p90 {r0['peratom_p90']:.3f}   "
              f"realised rank90 {r0['rank90_out']}", flush=True)
        print(f"  [26c NULL] an untrained decoder SHOULD sit well below zero on FVE_perp and above "
              f"1.0 per-atom; that is the floor the trained numbers are read against.", flush=True)
        del m0
    except Exception as e:
        print(f"  [26c NULL] untrained reference FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"\n=== 25a JOINT READING (pre-registered in INBOX 025) ===", flush=True)
    k0 = ANM_K[0]
    vv = np.array([r.get(f"fve_perp_anm{k0}", np.nan) for r in out], float)
    vv = vv[np.isfinite(vv)]
    wide = float(np.median(p9 / pm)) > 1.5
    if len(vv) and abs(np.median(vv)) < 0.05:
        print(f"  FVE_perp ~ 0 with a {'wide' if wide else 'narrow'} per-atom tail =>")
        print(f"  COLLECTIVE-MODE MODEL. Real, and publishable as such -- but it is NOT an answer to")
        print(f"  'can one token encode the dynamic state', and the honest headline changes.",
              flush=True)
    elif len(vv) and np.median(vv) > 0:
        print(f"  FVE_perp > 0 (median {np.median(vv):+.4f}) => the latent carries content the peer")
        print(f"  does NOT span. Whether it is width-limited is the DM axis: rising with DM and a")
        print(f"  narrowing tail is the result that would validate the foundation; flat in DM means")
        print(f"  it captures something beyond ANM but the DM axis is answering the wrong question.",
              flush=True)
    else:
        print(f"  FVE_perp negative (median {np.median(vv) if len(vv) else float('nan'):+.4f}): the")
        print(f"  codec ADDS error on the ANM-orthogonal part. Report as such.", flush=True)

    print(f"\n=== FVE BY MODE INDEX (median across {len(out)} systems) ===", flush=True)
    print(f"  'err vs zero' is per-mode SSE divided by the SSE of PREDICTING ZERO (INBOX 16b): above")
    print(f"  1.0 the decoder INJECTS error into that mode. Under MSE that is the CORRECT move for a")
    print(f"  capacity-limited model -- error is pushed where it is cheap to buy fit on the modes that")
    print(f"  dominate the loss -- so it is the signature of a restricted function class, NOT a bug.",
          flush=True)
    Ev = np.array([r["err_ratio"][:K] for r in out])
    print(f"    {'mode':>5}{'FVE':>9}{'err vs zero':>13}{'var share':>11}{'IAT (frames)':>14}")
    for i in list(range(min(10, K))) + [j for j in (14, 19, 24, 29) if j < K]:
        print(f"    {i+1:>5}{np.median(Fv[:, i]):>9.4f}{np.median(Ev[:, i]):>13.3f}"
              f"{100*np.median(Vv[:, i]):>10.2f}%{np.median(Tv[:, i]):>14.0f}", flush=True)
    inj = float(np.mean(np.median(Ev, 0) > 1.0))
    print(f"    -> the decoder injects error into {100*inj:.0f}% of the {K} resolved modes", flush=True)

    print(f"\n=== FVE BY TIMESCALE (modes binned by their OWN reference IAT, per system) ===",
          flush=True)
    ter = np.zeros((len(out), 3))
    for si in range(len(out)):
        q = np.percentile(Tv[si], [33.3, 66.7])
        for bi, m in enumerate([Tv[si] <= q[0], (Tv[si] > q[0]) & (Tv[si] <= q[1]), Tv[si] > q[1]]):
            ter[si, bi] = Fv[si, m].mean() if m.any() else np.nan
    names = ["FAST   (bottom IAT tercile)", "MEDIUM (middle tercile)   ", "SLOW   (top IAT tercile)  "]
    for bi in range(3):
        v = ter[:, bi][np.isfinite(ter[:, bi])]
        print(f"    {names[bi]}  mean FVE {v.mean():+.4f}   median {np.median(v):+.4f}", flush=True)
    dif = ter[:, 2] - ter[:, 0]; dif = dif[np.isfinite(dif)]
    hw = stats.t.ppf(0.975, len(dif) - 1) * dif.std(ddof=1) / np.sqrt(len(dif))
    print(f"    SLOW - FAST = {dif.mean():+.4f} +/- {hw:.4f} (95% CI across systems)", flush=True)

    print(f"\n=== THE CONTROL: is it TIMESCALE, or just VARIANCE (which MSE selects for by "
          f"construction)? ===", flush=True)
    s1, s1h, s2, s2h = [], [], [], []
    for si in range(len(out)):
        lt = np.log10(np.clip(Tv[si], 1e-6, None)); lv = np.log10(np.clip(Vv[si], 1e-12, None))
        a, ah = regress(lt, Fv[si])
        pa, pah, pb, pbh = partial(lt, lv, Fv[si])
        s1.append(a); s1h.append(ah); s2.append(pa); s2h.append(pah)
    s1 = np.array(s1); s2 = np.array(s2)
    h1 = stats.t.ppf(0.975, len(s1) - 1) * s1.std(ddof=1) / np.sqrt(len(s1))
    h2 = stats.t.ppf(0.975, len(s2) - 1) * s2.std(ddof=1) / np.sqrt(len(s2))
    print(f"  FVE ~ log(IAT)                       coefficient {s1.mean():+.4f} +/- {h1:.4f}")
    print(f"  FVE ~ log(IAT) + log(var share)      PARTIAL log(IAT) {s2.mean():+.4f} +/- {h2:.4f}")
    print(f"  (both are means of per-system fits with a CI across systems, so a few systems cannot")
    print(f"   carry the result)", flush=True)

    print(f"\n=== PRE-REGISTERED VERDICT ===", flush=True)
    tercile_ok = (dif.mean() > 0.10) and (dif.mean() - hw > 0)
    partial_ok = (s2.mean() - h2 > 0)
    uniform = (abs(dif.mean()) < 0.05) and (dif.mean() - hw < 0 < dif.mean() + hw)
    if tercile_ok and partial_ok:
        print(f"  SLOW-SELECTIVE. The codec keeps slow/collective content and discards fast motion,")
        print(f"  AND that survives holding variance share fixed -- so it is a timescale effect, not")
        print(f"  MSE's variance preference relabelled. This EXPLAINS the flat FVE-vs-N: collective")
        print(f"  content is low-dimensional and does not grow with N, while the discarded remainder")
        print(f"  is what grows as N^0.93. It also makes plain MSE demonstrably the wrong objective.")
        print(f"  DO NOT change the loss on this alone -- 007's instruction stands -- but this is the")
        print(f"  evidence that would justify designing the replacement.", flush=True)
    elif tercile_ok and not partial_ok:
        print(f"  VARIANCE-SELECTIVE, NOT TIMESCALE-SELECTIVE. FVE is higher on slow modes, but the")
        print(f"  effect does NOT survive holding variance share fixed. Slow modes are the")
        print(f"  high-variance ones and MSE selects for variance by construction, so this is the")
        print(f"  training objective doing exactly what it says. It is NOT evidence that the codec")
        print(f"  preferentially captures slow dynamics, and must not be reported as such.", flush=True)
    elif uniform:
        print(f"  UNIFORM. FVE is flat across timescales, so the model is uniformly weak: the flat")
        print(f"  FVE-vs-N slope carries NO architectural information, and only the n_train ladder")
        print(f"  can change the picture. Read alongside 14a -- a flat slope on a model far from the")
        print(f"  achievable is consistent with uniform weakness, which this would confirm directly.",
              flush=True)
    else:
        print(f"  INTERMEDIATE: SLOW-FAST {dif.mean():+.4f} +/- {hw:.4f}, partial log(IAT) "
              f"{s2.mean():+.4f} +/- {h2:.4f}. Neither pre-registered branch fires; the numbers")
        print(f"  stand as measured and no reading is forced onto them.", flush=True)

    print(f"\n=== DOES THE PICTURE CHANGE WITH N? ===", flush=True)
    Nv = np.log10(np.array([r["N"] for r in out], float))
    for lab, y in (("SLOW-tercile FVE", ter[:, 2]), ("FAST-tercile FVE", ter[:, 0]),
                   ("SLOW - FAST", ter[:, 2] - ter[:, 0])):
        m = np.isfinite(y)
        if m.sum() < 4: continue
        sl, h = regress(Nv[m], y[m])
        print(f"  {lab:>18} vs log10(N): {sl:+.4f} +/- {h:.4f}  "
              f"{'flat' if abs(sl) < h else '*** MOVES WITH N ***'}", flush=True)
