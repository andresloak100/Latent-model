"""INBOX 015: run the MODAL DECODER as an arm, against a control trained IN THE SAME JOB.

    disp_i = B_i(structure) @ z          -- linear in z, per-atom modes from structure alone

THREE DECODERS, ONE PROCEDURE, ONE JOB:
    control  armf_atlas_dm.Codec              the current attention/FiLM decoder
    untied   ModalCodec(tie_encoder=False)    modal decoder, attention encoder
    tied     ModalCodec(tie_encoder=True)     analysis/synthesis pair, z0 == 0 by construction

WHY THE CONTROL IS RE-TRAINED HERE RATHER THAN READ OUT OF atlas_dm.json. That file currently MIXES
TWO TRAINING PROCEDURES: its 13 persisted arms predate `WARMUP = 1000` and the lr-scaled step
budget, completed arms skip on re-run, and re-training its best arm did NOT reproduce it (+0.0515
against a recorded +0.0972 on the same 24 tracked systems). Job 10307008 is deciding whether that is
the procedure or seed noise. Either way, a modal-vs-control comparison that read the control off
that file would be FAMILY E -- the comparator separated from its peer by an unswept hyperparameter --
and would stay contaminated whichever way 10307008 lands. Training the control HERE, in the same
process, at the same seed, on the same LR grid, makes this comparison internally valid regardless.
That costs five extra arms and buys a result that does not depend on an open question.

LR IS SWEPT PER DECODER (FAMILY E, as 015 instructs). A bilinear decoder is a different optimisation
problem, and the DM=256 "collapse" that was retracted under INBOX 012 was exactly an unswept LR. A
modal arm that lost at the control's best LR would tell us nothing about the ARCHITECTURE.

WHAT THIS FILE VERIFIES BEFORE IT TRAINS ANYTHING. 015 reports a local test table, but the pushed
`armf_modal_decoder.py` contains NO `__main__` and no test code, so none of those numbers are
reproducible from the file as committed. They are therefore re-derived here from scratch, ON REAL
ATLAS TENSORS rather than synthetic ones, and the arms do not start unless the checks pass. The
claim that matters most is `tied => ||z0|| == 0` EXACTLY, because that is the architectural version
of the 012b identity ablation and the whole reason the tied variant is interesting."""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata
import armf_atlas_dm as D
from armf_modal_decoder import ModalCodec, basis_orthogonality, effective_modes

WR = D.WR
RES = f"{WR}/modal_arm.json"
CKPT = f"{WR}/modal_arm_ckpt"
DM = 256                       # the sweep's best width
NTR = 50
SEED = 0
LRS = D.LRS                    # same grid as the DM sweep
dev = D.dev

VARIANTS = ["control", "untied", "tied"]


def make(kind):
    """Factory installed into armf_atlas_dm.Codec so `train()` -- the EXACT training loop the sweep
    uses -- is reused rather than reimplemented. Duplicating that loop would risk procedure drift,
    which is the very defect this file is written around."""
    base = D.Codec if kind == "control" else None
    if kind == "control":
        return lambda Fs, L, dm, dlat=None, sub_z0=False: base(Fs, L, dm, dlat=dlat, sub_z0=sub_z0)
    tie = (kind == "tied")
    return lambda Fs, L, dm, dlat=None, sub_z0=False: ModalCodec(
        Fs, L, dm, dlat=dlat, sub_z0=sub_z0, tie_encoder=tie)


def selftest(stat_np):
    """Re-derive 015's claims on a REAL ATLAS static-feature tensor. Returns True iff all pass."""
    print("\n=== SELF-TEST (015's table is NOT in the pushed file; re-derived here) ===", flush=True)
    Fs = stat_np.shape[1]
    st = torch.tensor(stat_np, device=dev).unsqueeze(0)
    B, N = 1, st.shape[1]
    ok = True

    for tie in (False, True):
        for dl in (None, 16):
            m = ModalCodec(Fs, 1, DM, dlat=dl, tie_encoder=tie).to(dev)
            disp = torch.randn(B, N, 3, device=dev) * 0.1
            out = m(st, disp)
            good = tuple(out.shape) == (B, N, 3)
            out.pow(2).mean().backward()
            grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            print(f"  forward+backward tie={tie!s:>5} dlat={str(dl):>4}: shape {tuple(out.shape)} "
                  f"{'OK' if good else 'WRONG'}, finite grads {grad}", flush=True)
            ok &= good and grad

    # decode must be EXACTLY linear in z -- that is the entire architectural claim
    m = ModalCodec(Fs, 1, DM, tie_encoder=False).to(dev)
    z1 = torch.randn(B, 1, DM, device=dev); z2 = torch.randn(B, 1, DM, device=dev)
    with torch.no_grad():
        lhs = m.decode(st, 2.5 * z1 - 1.5 * z2)
        rhs = 2.5 * m.decode(st, z1) - 1.5 * m.decode(st, z2)
        err = float((lhs - rhs).abs().max())
    lin = err < 1e-4
    print(f"  decode linear in z (superposition): max err {err:.2e}  {'OK' if lin else 'FAIL'}",
          flush=True)
    ok &= lin

    # THE claim: identity offset is architecturally impossible when tied
    with torch.no_grad():
        mu = ModalCodec(Fs, 1, DM, tie_encoder=False).to(dev)
        mt = ModalCodec(Fs, 1, DM, tie_encoder=True).to(dev)
        nu = float(mu.z0(st).norm()); nt = float(mt.z0(st).norm())
    print(f"  ||z0|| untied {nu:.4f} (the leak, at random init)   tied {nt:.2e} "
          f"{'EXACTLY ZERO -- identity cannot occupy the code' if nt < 1e-10 else '*** NONZERO ***'}",
          flush=True)
    ok &= (nt < 1e-10)

    with torch.no_grad():
        _, off = basis_orthogonality(mt, st)
        em = effective_modes(mt, st)
    print(f"  basis at random init: off-diagonal {float(off[0]):.4f}, effective modes "
          f"{int(em[0])}/{DM}", flush=True)
    print(f"  -> SELF-TEST {'PASSED' if ok else 'FAILED -- NOT TRAINING'}", flush=True)
    return ok


if __name__ == "__main__":
    print(f"[modal-arm] 015: modal decoder vs control, DM={DM}, n_train={NTR}, LR grid {LRS}, "
          f"seed {SEED}. Control trained HERE so the comparison is procedure-matched.", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]

    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    _ord = sorted(range(len(HO)), key=lambda i: HO[i]["N"])
    HOt = [HO[_ord[i]] for i in np.linspace(0, len(HO) - 1, min(D.NHO_TRACK, len(HO))).astype(int)]
    TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:NTR]) if x is not None]
    print(f"  {len(TR)} train / {len(HOt)} tracked / {len(HO)} held-out, N "
          f"{min(x['N'] for x in HO)}-{max(x['N'] for x in HO)}", flush=True)

    if not selftest(TR[0]["stat"]):
        raise SystemExit(1)

    rows = json.load(open(RES)) if os.path.exists(RES) else []
    done = {(r["variant"], r["lr"]) for r in rows}
    orig = D.Codec

    for kind in VARIANTS:
        print(f"\n=== {kind.upper()} ===", flush=True)
        for lr in LRS:
            if (kind, lr) in done: continue
            D.Codec = make(kind)
            try:
                torch.manual_seed(SEED); np.random.seed(SEED + 1)
                t0 = time.time()
                mdl, hist, used, stopped, improving = D.train(
                    TR, HOt, DM, lr, f"{kind} lr{lr:g}", 1)
                mdl.eval()
                per = [D.fve_model(mdl, x) for x in HO]
                fve = float(np.mean(per))
                p = D.participation_ratio(mdl, TR[:min(len(TR), len(HO))], HO, DM)
                p.update(D.latent_dynamics(mdl, HOt[:8], DM))
                p["z0_frac"] = D.identity_offset(mdl, HOt[:12])
                lr_ = stats.linregress(np.log10([x["N"] for x in HO]), per)
                hw = stats.t.ppf(0.975, len(per) - 2) * lr_.stderr
                rec = dict(variant=kind, lr=lr, dm=DM, n_train=NTR, seed=SEED, fve=fve,
                           med=float(np.median(per)), steps=used, stopped=stopped,
                           improving=improving, best_track=max(h[1] for h in hist),
                           tail_track=float(np.mean([h[1] for h in hist[-5:]])),
                           nslope=float(lr_.slope), nslope_ci=float(hw), nho=len(HO),
                           per=[float(v) for v in per], Ns=[int(x["N"]) for x in HO],
                           secs=time.time() - t0, **p)
                if kind != "control":
                    with torch.no_grad():
                        offs, ems = [], []
                        for d in HOt[:8]:
                            s1 = torch.tensor(d["stat"], device=dev).unsqueeze(0)
                            offs.append(float(basis_orthogonality(mdl, s1)[1][0]))
                            ems.append(int(effective_modes(mdl, s1)[0]))
                    rec["basis_off"] = float(np.median(offs))
                    rec["eff_modes"] = float(np.median(ems))
                rows.append(rec); json.dump(rows, open(RES, "w")); done.add((kind, lr))
                os.makedirs(CKPT, exist_ok=True)
                torch.save(mdl.state_dict(), f"{CKPT}/{kind}_dm{DM}_lr{lr:g}_s{SEED}.pt")
                extra = ("" if kind == "control" else
                         f"  basis-off {rec['basis_off']:.3f}  eff-modes {rec['eff_modes']:.0f}/{DM}")
                print(f"    {kind} lr{lr:g}: FVE {fve:+.4f}  PR {p['pr']:.1f}/{DM}  "
                      f"identity {100*p['ident_frac']:.0f}% (z0 {100*p['z0_frac']:.0f}%)  "
                      f"N-slope {lr_.slope:+.4f}+/-{hw:.4f}  steps {used} ({stopped})"
                      f"{'  *** VOID: STILL IMPROVING ***' if improving else ''}{extra}", flush=True)
            except Exception as e:
                print(f"    {kind} lr{lr:g}: FAIL {type(e).__name__}: {e}", flush=True)
            finally:
                D.Codec = orig

    if not rows: raise SystemExit
    # ---------------- REPORT ----------------
    print(f"\n=== 015: MODAL DECODER vs CONTROL, procedure-matched, best LR per variant ===",
          flush=True)
    print(f"    {'variant':>9}{'bestLR':>9}{'FVE':>9}{'PR':>8}{'ident':>8}{'z0':>7}"
          f"{'N-slope':>18}{'eff-modes':>11}{'flag':>10}", flush=True)
    best = {}
    for k in VARIANTS:
        c = [r for r in rows if r["variant"] == k and not r["improving"]]
        if not c:
            allk = [r for r in rows if r["variant"] == k]
            print(f"    {k:>9}: every arm VOID (still improving at the step cap) -- no verdict for "
                  f"this variant. Not a null result." if allk else
                  f"    {k:>9}: no arms.", flush=True)
            continue
        b = max(c, key=lambda r: r["fve"]); best[k] = b
        print(f"    {k:>9}{b['lr']:>9.0e}{b['fve']:>9.4f}{b['pr']:>8.1f}"
              f"{100*b['ident_frac']:>7.0f}%{100*b['z0_frac']:>6.0f}%"
              f"{b['nslope']:>+11.4f}+/-{b['nslope_ci']:.4f}"
              f"{b.get('eff_modes', float('nan')):>11.0f}"
              f"{'low-PR' if b['pr_frac'] < 0.5 else '':>10}", flush=True)

    if "control" in best:
        cb = best["control"]
        print(f"\n=== VERDICT ===", flush=True)
        for k in ("untied", "tied"):
            if k not in best: continue
            b = best[k]
            gap = b["fve"] - cb["fve"]
            # paired over the SAME held-out systems, which is the powerful comparison
            g = np.array(b["per"]) - np.array(cb["per"])
            hw = stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
            print(f"  {k:>7} - control: {gap:+.4f}   paired {g.mean():+.4f} +/- {hw:.4f}   "
                  f"wins on {100*(g > 0).mean():.0f}% of {len(g)} systems  -> "
                  f"{'MODAL WINS' if g.mean() - hw > 0 else ('MODAL LOSES' if g.mean() + hw < 0 else 'INDISTINGUISHABLE')}",
                  flush=True)
        if "tied" in best:
            print(f"  tied identity share {100*best['tied']['ident_frac']:.0f}% with z0 "
                  f"{100*best['tied']['z0_frac']:.0f}% -- the 012b offset removed EXACTLY rather than "
                  f"to first order.", flush=True)
        em = [best[k].get("eff_modes") for k in ("untied", "tied") if k in best]
        em = [e for e in em if e]
        if em:
            print(f"  effective modes {min(em):.0f}-{max(em):.0f} of {DM}. This is the ARCHITECTURE's")
            print(f"  rank, not a per-system PCA fit, so it carries neither the in-sample bias nor the")
            print(f"  n_eff limit that made rank90 a floor. Well below {DM} => the token is")
            print(f"  over-provisioned whatever the training curve says.", flush=True)
        print(f"\n  IF THE MODAL ARM UNDERPERFORMS, the pre-registered first hypothesis (015) is NOT")
        print(f"  that bilinearity is wrong -- it is that the basis network cannot see enough")
        print(f"  structure, since q_tok is a per-atom map and collective modes are nonlocal.")
        print(f"  ESCALATE ctx_layers>0 (k-NN message passing) BEFORE abandoning the form.", flush=True)
