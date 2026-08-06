"""HOW WIDE DOES THE LATENT NEED TO BE? Sized from the CODEC, not from rank90. (INBOX 002)

rank90 is retired as a sizing instrument: it is measured in-sample (the in-sample rank90 modes
explain a median of 74.7% of held-out variance, not 90%, and the shortfall GROWS with N) and it is
censored at large N (32.3% of usable rank at the top). Both biases push the same way, so the width
chain `168 / 0.65 x 1.83 ~= 490` is a FLOOR. And it is corpus-independent -- MISATO 10 ns, mdCATH
2.5 us, ATLAS 100 ns; none supplies enough independent samples to fit a few-hundred-dimensional
PER-SYSTEM subspace, so no dataset move fixes it.

THE CODEC'S OWN CURVE HAS NO SUCH PROBLEM. The model is SHARED, fitted across many systems, and
evaluated on SYSTEMS IT NEVER SAW, so its effective sample size is the CORPUS, not one trajectory.
Report held-out FVE vs DM and where it saturates. That answer stands independently of every rank90
number in the roadmap.

FOUR CONTROLS, WITHOUT WHICH A FLAT CURVE IS UNINTERPRETABLE
------------------------------------------------------------
The prior DM sweep on mdCATH produced DM=256 and DM=512 arms that COLLAPSED TO CONSTANT OUTPUT
(G1 cos = 1.0000, G4 base -0.000) on 21 training domains and were declared VOID. "Flat curve" and
"arm never trained" look identical in an FVE table. So:

1. PARTICIPATION RATIO of the learned code. `PR = (sum lam)^2 / sum lam^2` on the DM-by-DM covariance
   of the latent channels over held-out frames and systems. PR ~ DM => the arm genuinely used its
   width, and a flat FVE curve IS saturation. PR << DM => capacity that failed to train, which is a
   DIFFERENT finding and must not be reported as the first. This is the control that distinguishes
   "512 is unnecessary" from "512 was never trained".

2. LEARNING-RATE SWEEP PER ARM. One LR across a 32x width range is FAMILY E -- a comparator
   handicapped by an unswept hyperparameter. Wide models generally want a smaller LR; the mdCATH
   collapse is exactly what too-large an LR does to a wide FiLM decoder. Each DM gets the best of
   LRS and the winning LR is reported, so no arm loses on a hyperparameter rather than on capacity.

3. TRAIN TO PLATEAU, NOT TO A FIXED STEP COUNT. Wider arms have more parameters and would be
   undertrained at a fixed budget, which manufactures a flat curve. Every arm runs to its own
   plateau under a common generous ceiling; steps-used is reported, and any arm STILL IMPROVING at
   MAXSTEPS is flagged -- that flag makes the run FAMILY D (a measurement structurally unable to
   express "more width would help") and voids the saturation read for that arm.

4. TWO TRAINING-SET SIZES. If the saturating DM MOVES with n_train, the curve is data-limited, not
   width-limited, and the saturation point is a property of the corpus rather than the architecture.
   If it is stable, the width answer is real. This is the control that decides whether the number
   generalises at all.

PRE-REGISTERED READ (recorded before results exist)
---------------------------------------------------
- FVE rises with DM then flattens; the flattening DM is the width answer.
- Flat + PR ~ DM                        => genuine saturation. Report as the architectural answer.
- Flat + PR << DM                       => the arm failed to train. Report as such, NOT as saturation.
- Still improving at MAXSTEPS           => VOID for that arm (Family D).
- Saturating DM moves with n_train      => data-limited; report the dependence, not a single number.
SCOPE LIMIT, stated before the result: this is measured on ATLAS single chains, N 598-33,377, 100 ns,
apo. It licenses a width statement for THAT regime. It does NOT license the 1e6-atom extrapolation
the width chain used to provide -- that extrapolation is dropped, not transferred."""
import sys, os, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ho_frames, train_frames
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
MAN = f"{WR}/atlas_manifest.json"; RES = f"{WR}/atlas_dm.json"
DMS = [16, 64, 256, 512]
LRS = [3e-4, 1e-3, 3e-3]
# INBOX 003: L=1 IS THE ARCHITECTURE, not one option in a sweep. One latent token per frame
# regardless of whether the system has 1e3, 3e4 or 1e6 atoms; DM is the ONLY capacity knob.
# L=12/24 are DEMOTED TO DIAGNOSTICS -- they exist solely to localise any degradation with N:
#   present at L=12/24 and ABSENT at L=1  => slot ASSIGNMENT (addressing) is the cause
#   present at L=1 too                    => the POOLING/BROADCAST pathway is the cause
# They are NOT candidate designs. A better number at L=12 is not a recommendation, and results are
# never averaged across L.
# (L=1 was formerly untestable -- softmax over a single key is identically 1.0, so every atom got the
# same vector. The FiLM path fixes that: per-atom variation comes from the static-feature query, and
# the frame-specific conformational signal arrives as a global scale+shift. That is the design.)
L_PRIMARY = 1
L_DIAG = [12, 24]
# CONTROL 4 ladder. Filtered at runtime to what is cached, so low arms run while the cache is still
# building and the results file ACCUMULATES across re-runs (completed (n,dm,lr) combos are skipped).
# The ladder is the control: a saturating DM that moves across a 12x range of corpus size is
# data-limited, not width-limited.
NTRAIN = [50, 130, 300, 600]
FRAMES_PER_STEP = 8; MAXSTEPS = 30000; EVAL_EVERY = 2500; PATIENCE = 4; COMPETENCE = 0.05
NHO_TRACK = 24                      # held-out SYSTEMS used for the plateau signal
np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Codec(nn.Module):
    def __init__(self, Fs, L, dm, heads=4):
        super().__init__()
        self.lat = nn.Parameter(torch.randn(L, dm) * 0.02)
        self.in_tok = nn.Linear(Fs + 3, dm); self.q_tok = nn.Linear(Fs, dm)
        self.enc = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.sa = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.dec = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.l1, self.l2, self.l3, self.l4 = (nn.LayerNorm(dm) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(dm, dm*2), nn.GELU(), nn.Linear(dm*2, dm))
        self.dff = nn.Sequential(nn.Linear(dm, dm*2), nn.GELU(), nn.Linear(dm*2, dm))
        self.film = nn.Linear(dm, 2*dm); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.out = nn.Linear(dm, 3)

    def encode(self, stat, disp):
        tok = self.in_tok(torch.cat([stat, disp], -1))
        lat = self.lat.unsqueeze(0).expand(stat.shape[0], -1, -1)
        lat = self.l1(lat + self.enc(lat, tok, tok)[0]); lat = self.l2(lat + self.sa(lat, lat, lat)[0])
        return self.l3(lat + self.ff(lat))

    def decode(self, stat, lat):
        q = self.q_tok(stat); a = self.dec(q, lat, lat)[0]
        g, b = self.film(lat.mean(1, keepdim=True)).chunk(2, -1)
        h = self.l4(q + a) * (1 + g) + b
        return self.out(h + self.dff(h))

    def forward(self, stat, disp): return self.decode(stat, self.encode(stat, disp))


@torch.no_grad()
def fve_model(mdl, d, chunk=8):
    st = torch.tensor(d["stat"], device=dev); sse = 0.0
    for s in range(0, d["F"], chunk):
        e = min(s + chunk, d["F"])
        t = ho_frames(d, s, e)
        dw = torch.tensor((t / d["scale"]).astype(np.float32), device=dev)
        p = mdl(st.unsqueeze(0).expand(e - s, -1, -1), dw).cpu().numpy().astype(np.float64) * d["scale"]
        sse += float(((t - p) ** 2).sum())
    return 1 - sse / (d["sst"] + 1e-12)


@torch.no_grad()
def participation_ratio(mdl, HO, dm, nf=64):
    """CONTROL 1. How many of the DM channels does the learned code actually use?

    Second-moment matrix of the DM-dimensional latent channel vectors, pooled over latent slots,
    held-out frames and held-out SYSTEMS. PR = (sum lam)^2 / sum lam^2 in [1, DM]. Mean-centred, so a
    code that is constant across frames (the mdCATH collapse mode) scores ~1 rather than ~DM."""
    C = np.zeros((dm, dm)); tot = 0; acc = np.zeros(dm)
    for d in HO:
        st = torch.tensor(d["stat"], device=dev)
        idx = np.linspace(0, d["F"] - 1, min(nf, d["F"])).astype(int)
        t = np.concatenate([ho_frames(d, i, i + 1) for i in idx], 0).reshape(len(idx), d["N"], 3)
        dw = torch.tensor((t / d["scale"]).astype(np.float32), device=dev)
        z = mdl.encode(st.unsqueeze(0).expand(len(idx), -1, -1), dw)   # (F, L, dm)
        Z = z.reshape(-1, dm).double().cpu().numpy()
        C += Z.T @ Z; acc += Z.sum(0); tot += len(Z)
    mu = acc / max(tot, 1)
    C = C / max(tot, 1) - np.outer(mu, mu)                              # centred second moment
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    return float(s1 * s1 / (s2 + 1e-30)), float(lam.max() / (s1 + 1e-30))


def train(tr, HOt, dm, lr, tag, L):
    mdl = Codec(tr[0]["stat"].shape[1], L, dm).to(dev)
    opt = torch.optim.Adam(mdl.parameters(), lr); hist = []; t0 = time.time()
    used = MAXSTEPS; stopped = "maxsteps"
    for st in range(1, MAXSTEPS + 1):
        d = tr[np.random.randint(len(tr))]
        idx = np.random.randint(0, 2 * d["F"], FRAMES_PER_STEP)
        S = torch.tensor(d["stat"], device=dev).unsqueeze(0).expand(FRAMES_PER_STEP, -1, -1)
        D = torch.tensor(train_frames(d, idx).reshape(FRAMES_PER_STEP, d["N"], 3) / d["scale"], device=dev)
        opt.zero_grad(); ((mdl(S, D) - D) ** 2).mean().backward(); opt.step()
        if st % EVAL_EVERY == 0:
            mdl.eval(); v = float(np.mean([fve_model(mdl, x) for x in HOt])); mdl.train()
            hist.append((st, v))
            print(f"      {tag} step {st:>6}: held-out FVE {v:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
            comp = any(h[1] > COMPETENCE for h in hist)          # competence gate
            if comp and len(hist) > PATIENCE and \
               max(h[1] for h in hist[-PATIENCE:]) <= max(h[1] for h in hist[:-PATIENCE]) * 1.01:
                used, stopped = st, "plateau"; break
    # CONTROL 3: an arm still climbing at MAXSTEPS was never given the chance to show its capacity.
    improving = False
    if stopped == "maxsteps" and len(hist) > PATIENCE:
        improving = max(h[1] for h in hist[-PATIENCE:]) > max(h[1] for h in hist[:-PATIENCE]) * 1.01
    return mdl, hist, used, stopped, improving


if __name__ == "__main__":
    print(f"[atlas-dm] device={dev}  L={L}  DM sweep {DMS}  LR sweep {LRS}  n_train {NTRAIN}", flush=True)
    man = json.load(open(MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]
    print(f"  cached: {len(ho_ids)} held-out SYSTEMS / {len(tr_ids)} train systems", flush=True)
    if len(tr_ids) < NTRAIN[0] or len(ho_ids) < 20:
        print(f"  cache too small (need >={NTRAIN[0]} train, >=20 held-out) -- rerun later"); raise SystemExit
    NT = [n for n in NTRAIN if n <= len(tr_ids)]
    print(f"  realised n_train arms {NT} of ladder {NTRAIN} "
          f"({len(NTRAIN)-len(NT)} awaiting cache; re-run to fill them in)", flush=True)

    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    store.conservation_report(expected=man["heldout"])
    HOt = HO[::max(1, len(HO) // NHO_TRACK)][:NHO_TRACK]
    Ns = [x["N"] for x in HO]
    print(f"  held-out N {min(Ns)}-{max(Ns)}, {np.log10(max(Ns)/min(Ns)):.2f} decades; "
          f"plateau signal on {len(HOt)} systems, final eval on all {len(HO)}", flush=True)

    rows = json.load(open(RES)) if os.path.exists(RES) else []
    done = {(r["L"], r["n_train"], r["dm"], r["lr"]) for r in rows}

    def run(TR, n, dm, lr, Lv):
        if (Lv, n, dm, lr) in done: return None
        tag = f"L{Lv} n{n} DM{dm} lr{lr:g}"
        try:
            mdl, hist, used, stopped, improving = train(TR, HOt, dm, lr, tag, Lv)
            mdl.eval()
            per = [fve_model(mdl, x) for x in HO]             # per-system, for the N-slope
            fve = float(np.mean(per))
            pr, top1 = participation_ratio(mdl, HOt, dm)
            lr_ = stats.linregress(np.log10([x["N"] for x in HO]), per)
            hw = stats.t.ppf(0.975, len(per) - 2) * lr_.stderr
            rec = dict(L=Lv, n_train=n, dm=dm, lr=lr, fve=fve, med=float(np.median(per)),
                       pr=pr, pr_frac=pr / dm, top1_var=top1, steps=used, stopped=stopped,
                       improving=improving, best_track=max(h[1] for h in hist), nho=len(HO),
                       nslope=float(lr_.slope), nslope_ci=float(hw),
                       per=[float(v) for v in per], Ns=[int(x["N"]) for x in HO])
            rows.append(rec); json.dump(rows, open(RES, "w")); done.add((Lv, n, dm, lr))
            print(f"    {tag}: FVE {fve:+.4f}  PR {pr:.1f}/{dm} ({100*pr/dm:.0f}%)  "
                  f"N-slope {lr_.slope:+.4f}+/-{hw:.4f}  steps {used} ({stopped})"
                  f"{'  *** STILL IMPROVING -> VOID ***' if improving else ''}", flush=True)
            return rec
        except Exception as e:
            print(f"    {tag}: FAIL {type(e).__name__}: {e}", flush=True)
            return None

    for n in NT:
        TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:n]) if x is not None]
        print(f"\n=== n_train={n} ({len(TR)} loaded) -- L={L_PRIMARY} IS THE DESIGN POINT ===", flush=True)
        for dm in DMS:
            for lr in LRS:                                   # CONTROL 2: no arm loses on the LR
                run(TR, n, dm, lr, L_PRIMARY)
        # ADDRESSING DIAGNOSTIC ONLY -- NOT candidate designs.
        # Run TWO WAYS, because total latent capacity is L x DM: at the same DM, L=12 carries 12x the
        # capacity of L=1, so a slope difference there could be CAPACITY rather than ADDRESSING.
        #   (a) same-DM       -- more total capacity, isolates slots-plus-capacity
        #   (b) capacity-matched (DM/L) -- same L x DM budget, isolates SLOTS ALONE
        # Addressing is implicated only if the N-slope difference survives BOTH. If it appears only
        # in (a), the cause is capacity and the addressing reading would be Family D -- a measurement
        # unable to separate the effect it names from the one it holds fixed.
        c1 = [r for r in rows if r["L"] == L_PRIMARY and r["n_train"] == n]
        if c1:
            b = max(c1, key=lambda r: r["fve"])
            print(f"  --- addressing diagnostic, L=1 winner DM={b['dm']} lr={b['lr']:g} ---", flush=True)
            for Lv in L_DIAG:
                run(TR, n, b["dm"], b["lr"], Lv)                      # (a) same DM
                dmm = max(16, (b["dm"] // Lv) // 8 * 8)               # (b) capacity-matched
                if dmm != b["dm"]: run(TR, n, dmm, b["lr"], Lv)

    if not rows: raise SystemExit

    def saturating_dm(n, Lv):
        prev = None; s = None
        for dm in DMS:
            c = [r for r in rows if r["L"] == Lv and r["n_train"] == n and r["dm"] == dm]
            if not c: continue
            b = max(c, key=lambda r: r["fve"])
            if prev is not None and s is None and b["fve"] <= prev * 1.02: s = dm
            prev = max(prev, b["fve"]) if prev is not None else b["fve"]
        return s

    print(f"\n=== L=1: HELD-OUT FVE vs DM (best LR per arm; {rows[0]['nho']} UNSEEN SYSTEMS) ===",
          flush=True)
    print(f"  ONE latent token per frame, any N. DM is the only capacity knob.", flush=True)
    for n in sorted({r["n_train"] for r in rows if r["L"] == L_PRIMARY}):
        print(f"  n_train={n}")
        print(f"    {'DM':>5}{'FVE':>9}{'median':>9}{'bestLR':>9}{'PR':>8}{'PR/DM':>7}"
              f"{'N-slope':>18}{'steps':>7}{'flag':>9}")
        for dm in DMS:
            c = [r for r in rows if r["L"] == L_PRIMARY and r["n_train"] == n and r["dm"] == dm]
            if not c: continue
            b = max(c, key=lambda r: r["fve"])
            flag = "VOID" if b["improving"] else ("low-PR" if b["pr_frac"] < 0.5 else "")
            print(f"    {dm:>5}{b['fve']:>9.4f}{b['med']:>9.4f}{b['lr']:>9.0e}{b['pr']:>8.1f}"
                  f"{100*b['pr_frac']:>6.0f}%{b['nslope']:>+11.4f}+/-{b['nslope_ci']:.4f}"
                  f"{b['steps']:>7}{flag:>9}", flush=True)
        s = saturating_dm(n, L_PRIMARY)
        print(f"    -> saturates at DM={s}" if s else "    -> NO saturation within the swept range")

    print(f"\n=== THE HEADLINE QUESTION: at L=1, does codec-vs-N hold FLAT from ~600 to ~33,500 atoms? ===",
          flush=True)
    for n in sorted({r["n_train"] for r in rows if r["L"] == L_PRIMARY}):
        c = [r for r in rows if r["L"] == L_PRIMARY and r["n_train"] == n]
        if not c: continue
        b = max(c, key=lambda r: r["fve"])
        flat = abs(b["nslope"]) < b["nslope_ci"]
        print(f"  n_train={n} DM={b['dm']}: FVE-vs-log10(N) slope {b['nslope']:+.4f} +/- "
              f"{b['nslope_ci']:.4f}  -> {'FLAT (CI spans 0)' if flat else '*** DEGRADES WITH N ***'}",
              flush=True)

    print(f"\n=== ADDRESSING DIAGNOSTIC (L=12/24 are NOT candidate designs) ===", flush=True)
    print(f"  Localises any N-degradation: present at L=12/24 and ABSENT at L=1 => slot ASSIGNMENT.")
    print(f"  Present at L=1 too => the POOLING/BROADCAST pathway. Never averaged across L.")
    print(f"  Reported at same-DM AND capacity-matched (L x DM held fixed), because a slope gap that")
    print(f"  appears only at same-DM is CAPACITY, not addressing.", flush=True)
    print(f"    {'L':>4}{'n_train':>9}{'DM':>6}{'LxDM':>7}{'FVE':>9}{'N-slope':>18}{'arm':>20}")
    for n in sorted({r["n_train"] for r in rows}):
        base = [r for r in rows if r["L"] == L_PRIMARY and r["n_train"] == n]
        bdm = max(base, key=lambda r: r["fve"])["dm"] if base else None
        for Lv in [L_PRIMARY] + L_DIAG:
            for r in sorted([x for x in rows if x["L"] == Lv and x["n_train"] == n],
                            key=lambda x: -x["fve"])[:2 if Lv != L_PRIMARY else 1]:
                arm = ("THE DESIGN" if Lv == L_PRIMARY else
                       ("same-DM" if r["dm"] == bdm else "capacity-matched"))
                print(f"    {Lv:>4}{n:>9}{r['dm']:>6}{Lv*r['dm']:>7}{r['fve']:>9.4f}"
                      f"{r['nslope']:>+11.4f}+/-{r['nslope_ci']:.4f}{arm:>20}", flush=True)

    sats = {n: saturating_dm(n, L_PRIMARY) for n in sorted({r["n_train"] for r in rows if r["L"] == L_PRIMARY})}
    print(f"\n=== CONTROL 4: does the saturating DM move with corpus size? {sats} ===", flush=True)
    vals = [v for v in sats.values() if v]
    if len(vals) > 1 and len(set(vals)) > 1:
        print("  IT MOVES -> the curve is DATA-limited, not width-limited. The saturation point is a")
        print("  property of the corpus, not the architecture. Report the dependence, not a number.")
    elif len(vals) > 1:
        print("  STABLE across corpus size -> the width answer is a property of the architecture.")
    else:
        print("  only one n_train arm produced a saturation point -- control 4 is UNTESTED.")
    print("\n  SCOPE: ATLAS single chains, N 598-33,377, 100 ns, apo. Licenses a width statement for")
    print("  THAT regime only. The 1e6-atom extrapolation the width chain used to provide is DROPPED,")
    print("  not transferred -- see ROADMAP 'WIDTH CHAIN IS A FLOOR'.", flush=True)

    best = max([r for r in rows if r["L"] == L_PRIMARY], key=lambda r: r["fve"], default=None)
    if best:
        dm = best["dm"]
        print(f"\n=== THE COMPRESSION CLAIM (objective 4, in one line) ===", flush=True)
        print(f"  ONE latent token of width DM={dm} per frame, for a system of ANY size.")
        print(f"  At N=1e6 atoms that is 3,000,000 coordinates -> {dm} numbers, a "
              f"{3e6/dm:,.0f}x reduction,")
        print(f"  and the {dm} DOES NOT GROW WITH N. Measured here at N={min(best['Ns'])}"
              f"-{max(best['Ns'])} ({max(best['Ns'])/min(best['Ns']):.0f}x); the 1e6 figure is the "
              f"ARCHITECTURAL claim (token count is N-independent by construction), NOT a measured")
        print(f"  reconstruction quality at that size.", flush=True)
