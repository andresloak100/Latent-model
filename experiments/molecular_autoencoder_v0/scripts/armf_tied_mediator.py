"""INBOX 26a: MEASURE THE MEDIATOR, not the downstream effect. Forward passes only, no training.

THE CLAIM UNDER TEST. The tied encoder is

    z = einsum('bnic,bni->bc', B, disp) * coef_scale / N**0.5

and `1/sqrt(N)` is the normaliser for a sum with RANDOM signs. Collective modes are COHERENT, so the
analysis sum grows like N, leaving a residual `sqrt(N)` drift in code magnitude -- 7.5x across
ATLAS's 56x range -- which a decoder LINEAR in z cannot absorb.

That story is about `||z||`, not about FVE. So it is tested on `||z||`:

    per held-out system, regress  log10( ||z|| / ||disp|| )  on  log10(N)
        slope ~ +0.5  -> the mediator exists, the mechanism is real
        slope ~  0    -> it does not, and the mechanism is dead whatever the remaining arms show

DECISIVE FROM ONE ARM, because it measures the STATED CAUSE rather than the downstream effect.
Replication across the other three tied arms would test whether the SLOPE is reproducible; this tests
whether the mechanism exists at all. Different questions, and this one is free.

NORMALISED BY ||disp||, deliberately. Raw `||z||` would rise with N simply because a bigger system has
more total motion -- a Family A confound sitting inside the confirmation, where it would be hardest to
see. The ratio removes it.

CONTROL ARMS, because a mechanism that is not SPECIFIC is not established. The same regression runs on
the UNTIED and CONTROL checkpoints, which have no such normaliser: untied encodes by attention,
control is the FiLM decoder. If they show the same +0.5 the explanation is not the tied normaliser
and the diagnosis fails even if the tied slope matches.

INBOX 26b -- WHY THE NETWORK CANNOT LEARN AROUND IT, which is the first objection a reader raises.
The basis `B` comes from `q_tok`, a PER-ATOM map that never sees N, and `coef_scale` is a single
global parameter of shape (ncoef,) shared across every system. There is no path by which either could
scale itself per system, so the drift cannot be absorbed during training -- it survives it. That is
also why measuring a TRAINED checkpoint is the right test rather than an objection to it."""
import sys, os, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ho_frames
import armf_atlas_dm as D
import armf_stamp as STAMP
from armf_modal_decoder import ModalCodec

WR = D.WR
CKPT = f"{WR}/modal_arm_ckpt"
RES = f"{WR}/tied_mediator.json"
DM, SEED = 256, 0
NFRAME = 48
ARMS = [("tied", 3e-5), ("untied", 3e-5), ("control", 3e-4)]   # control at its own best LR
PREDICTED = 0.5


def build(kind):
    if kind == "control":
        return D.Codec(FS, 1, DM).to(dev)
    return ModalCodec(FS, 1, DM, tie_encoder=(kind == "tied")).to(dev)


@torch.no_grad()
def ratio_for(mdl, d, nf=NFRAME):
    """Returns (||z||/||disp||, ||decoded||/||disp||), medians over frames, on the SCALED
    displacement the model receives.

    THE SECOND RATIO IS THE SYNTHESIS SIDE, added after 26a refuted the analysis-side story. 26a
    showed the tied encoder's code magnitude is nearly FLAT in N (-0.099), so the analysis normaliser
    is doing roughly the right thing. But the tied arm collapses at large N anyway -- median FVE
    +0.296 on the smallest quartile against -0.279 on the largest. The reconstruction is `B @ z`, and
    `B` is built per atom, so ||B||_F grows like sqrt(N): with ||z||/||disp|| flat, ||B z||/||disp||
    would grow like sqrt(N) and the OUTPUT would be over-scaled at large N. That is a DIFFERENT
    mechanism from the one just refuted, on the other half of the analysis/synthesis pair, and it is
    measured the same way rather than asserted."""
    F = min(nf, d["F"])
    ref = ho_frames(d, 0, F).reshape(F, d["N"], 3) / d["scale"]
    st = torch.tensor(d["stat"], device=dev)
    # element budget on the (chunk, N, 3, ncoef) basis tensor -- the modal decoder materialises it,
    # and at N=33,377 a careless chunk is 6.6 GB. Bound it rather than discover it on the biggest
    # system, which is the N-correlated failure this project keeps designing out.
    chunk = max(1, min(32, int(2e8 // max(d["N"] * 3 * DM, 1))))
    rs, os_ = [], []
    for s0 in range(0, F, chunk):
        e0 = min(s0 + chunk, F)
        dw = torch.tensor(ref[s0:e0].astype(np.float32), device=dev)
        S = st.unsqueeze(0).expand(e0 - s0, -1, -1)
        z = mdl.code(S, dw)
        out = mdl.decode(S, z)                      # the SYNTHESIS side
        zn = z.reshape(e0 - s0, -1).norm(dim=1)
        dn = dw.reshape(e0 - s0, -1).norm(dim=1) + 1e-12
        on = out.reshape(e0 - s0, -1).norm(dim=1)
        rs.append((zn / dn).cpu().numpy()); os_.append((on / dn).cpu().numpy())
    return float(np.median(np.concatenate(rs))), float(np.median(np.concatenate(os_)))


if __name__ == "__main__":
    dev = D.dev
    print(f"[tied-mediator] INBOX 26a: does ||z||/||disp|| scale as N^0.5? Forward passes only, "
          f"{NFRAME} frames/system, arms {[a for a, _ in ARMS]}", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    HO.sort(key=lambda d: d["N"])
    FS = HO[0]["stat"].shape[1]
    globals()["FS"] = FS
    print(f"  {len(HO)} held-out systems, N {HO[0]['N']}-{HO[-1]['N']} "
          f"({np.log10(HO[-1]['N']/HO[0]['N']):.2f} decades)", flush=True)

    out = json.load(open(RES)) if os.path.exists(RES) else {}
    for kind, lr in ARMS:
        cp = f"{CKPT}/{kind}_dm{DM}_lr{lr:g}_s{SEED}.pt"
        if not os.path.exists(cp):
            print(f"  {kind}: no checkpoint at lr{lr:g} -- skipped", flush=True); continue
        if kind in out: continue
        mdl = build(kind)
        mdl.load_state_dict(torch.load(cp, map_location=dev)); mdl.eval()
        rr, NN = [], []
        for i, d in enumerate(HO):
            try:
                rr.append(ratio_for(mdl, d)); NN.append(d["N"])
            except Exception as e:
                print(f"    {d['pdb']} N={d['N']}: FAIL {type(e).__name__}: {e}", flush=True)
        out[kind] = dict(lr=lr, N=NN, ratio=[a for a, _ in rr], out_ratio=[b for _, b in rr])
        json.dump(out, open(RES, "w"))
        print(f"  {kind}: scored {len(rr)}/{len(HO)} systems", flush=True)
        del mdl
        torch.cuda.empty_cache() if dev == "cuda" else None

    print(f"\n=== 26a: log10(||z||/||disp||) vs log10(N) ===", flush=True)
    print(f"  mechanism predicts slope +{PREDICTED:.1f} for TIED and ~0 for the others.", flush=True)
    print(f"    {'arm':>9}{'field':>12}{'n':>5}{'slope':>10}{'95% CI':>20}{'R^2':>8}   verdict",
          flush=True)
    res = {}
    for kind, _ in ARMS:
        if kind not in out: continue
        for fld, lab in (("ratio", "||z||/||d||"), ("out_ratio", "||dec||/||d||")):
            if fld not in out[kind]: continue
            N2 = np.array(out[kind]["N"], float); r2 = np.array(out[kind][fld], float)
            g2 = np.isfinite(r2) & (r2 > 0)
            if g2.sum() < 10: continue
            l2 = stats.linregress(np.log10(N2[g2]), np.log10(r2[g2]))
            h2 = stats.t.ppf(0.975, int(g2.sum()) - 2) * l2.stderr
            print(f"    {kind:>9}{lab:>12}{int(g2.sum()):>5}{l2.slope:>+10.4f}"
                  f"{f'[{l2.slope-h2:+.3f}, {l2.slope+h2:+.3f}]':>20}{l2.rvalue**2:>8.3f}",
                  flush=True)
    for kind, _ in ARMS:
        if kind not in out: continue
        N = np.array(out[kind]["N"], float); r = np.array(out[kind]["ratio"], float)
        g = np.isfinite(r) & (r > 0)
        if g.sum() < 10: continue
        lr_ = stats.linregress(np.log10(N[g]), np.log10(r[g]))
        h = stats.t.ppf(0.975, int(g.sum()) - 2) * lr_.stderr
        lo, hi = lr_.slope - h, lr_.slope + h
        has_half = lo <= PREDICTED <= hi
        excl_zero = lo > 0 or hi < 0
        v = ("CONSISTENT with N^0.5" if has_half and excl_zero else
             "flat (mechanism absent)" if not excl_zero else
             f"nonzero but NOT {PREDICTED:.1f}")
        res[kind] = (lr_.slope, h, has_half, excl_zero)
        print(f"    {kind:>9}{int(g.sum()):>5}{lr_.slope:>+10.4f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{lr_.rvalue**2:>8.3f}   {v}", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if "tied" in res:
        sl, h, has_half, excl = res["tied"]
        others = [k for k in res if k != "tied"]
        others_flat = all(not res[k][3] for k in others) if others else None
        if has_half and excl:
            if others_flat:
                print(f"  THE MEDIATOR EXISTS AND IS SPECIFIC TO THE TIED ARM. ||z||/||disp|| scales")
                print(f"  as N^{sl:+.3f} (CI contains +0.5) while every control arm is flat. The")
                print(f"  1/sqrt(N) normaliser divides a COHERENT sum by the incoherent normaliser,")
                print(f"  and per INBOX 26b the network cannot compensate: B comes from q_tok, a")
                print(f"  per-atom map that never sees N, and coef_scale is one global vector. The")
                print(f"  diagnosis is complete WITHOUT the other three arms.", flush=True)
            else:
                print(f"  Tied matches N^0.5 BUT a control arm is also non-flat, so the drift is NOT")
                print(f"  specific to the normaliser. The mechanism is NOT established -- a")
                print(f"  non-specific effect cannot identify a cause.", flush=True)
        elif not excl:
            print(f"  THE MEDIATOR IS ABSENT. ||z||/||disp|| is flat in N ({sl:+.4f} +/- {h:.4f}),")
            print(f"  so the sqrt(N) story is DEAD regardless of what the remaining tied arms show.")
            print(f"  The -0.65 FVE slope needs a different explanation, and the honest state is that")
            print(f"  it is one arm at one seed with no mechanism behind it.", flush=True)
        else:
            print(f"  Non-zero but not +0.5 ({sl:+.4f} +/- {h:.4f}). The code magnitude does drift")
            print(f"  with N, but not at the exponent the stated mechanism predicts -- report the")
            print(f"  number and do not claim the mechanism.", flush=True)
    print(f"\n  NOTE: this measures whether the STATED CAUSE exists, on one arm. Whether the -0.65 FVE")
    print(f"  SLOPE reproduces is a different question and still needs the remaining tied arms.",
          flush=True)
