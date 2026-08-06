"""LEARNING CURVE IN TRAINING-SET SIZE -- the experiment that distinguishes the two hypotheses.

Every peer/oracle comparison this project has run was at n_train of 21-207 systems, and every one
lost. Two hypotheses:
  (a) DATA-LIMITED  -- a shared codec needs many systems to learn a structure->dynamics map, and
                       21-207 is far too few.
  (b) NOT DATA-LIMITED -- more systems will not move it, and the cause must then be DIAGNOSED.
A SINGLE run at n=700 cannot distinguish them. A CURVE can.

  climbing and crossing ANM      -> (a), and the curve says what n is needed
  climbing but flattening below  -> (a) partially, with a ceiling; extrapolate the crossing and
                                    decide whether that n is reachable
  FLAT across a 14x range in n   -> (b): RUN THE DIAGNOSIS FAN-OUT. Do not infer a cause from
                                    flatness alone -- at least six mechanisms produce it.

ANM IS A DIAGNOSTIC, NOT THE BAR (INBOX 004b). It is the honest zero-shot comparator and it answers
one question: does the learned map carry information a physics prior does not? It is NOT the success
criterion. ANM is a fixed-topology structural prior with no generator; it cannot produce a
programmable latent dynamics system, which is the entire point of the project. A one-token codec that
loses to ANM on per-frame reconstruction is NOT thereby dead.
THE ACTUAL SUCCESS CRITERIA, in reporting order:
  1. RETAINS DYNAMICAL INFORMATION -- not just per-frame FVE. Decoded trajectories must preserve the
     DYNAMICS: per-mode marginal std ratio, integrated autocorrelation time, cross-mode coupling and
     the 2D free-energy projection, via the ensemble acceptance test already built for the
     propagator. A codec with mediocre FVE that PRESERVES autocorrelation structure is worth more
     than one with better FVE that flattens it.
  2. GENERALISES TO UNSEEN SYSTEMS -- held-out systems, not held-out frames. Already the protocol.
  3. SCALES WITH N -- the codec-vs-N trend at L=1 from 600 to 33,500 atoms.
  4. SUPPORTS THE DOWNSTREAM GENERATOR -- measurable NOW: the latent time-series' autocorrelation
     time, its frame-to-frame smoothness, and whether an AR(1)/OU fit in latent space gives stable
     rollouts. A latent that reconstructs well but JUMPS between frames is useless to stage 2, and
     that is far better discovered now than after the propagator is built.

DESIGN
- n_train in {50, 100, 200, 400, 700}; the SAME held-out set throughout; same architecture, same L.
- Training sets are NESTED PREFIXES of one deterministic shuffle (see atlas_manifest.json), so the
  curve is not confounded by WHICH systems were drawn -- only by HOW MANY.
- Every comparator is computed IN THIS SCRIPT over the SAME held-out rows in the SAME run. No
  baseline number is a constant. (FAMILY F: the ligand win died because a baseline was a hardcoded
  literal from a different 12-molecule sample.)
- ANM cutoff SWEPT on TRAINING systems only, winner applied unchanged to held-out (FAMILY E), from
  {5, 7} A. NOT 10 A: measured infeasible at the ATLAS top (shift-invert factorisation OOMs even with
  128 GB at N=33,377) and measured WEAKER on proteins (ANM-5 beat ANM-10 by +0.135 FVE). The
  strongest cutoff and the tractable one coincide.
- ENTRY CONDITION: the PCA-baseline-vs-N guard runs FIRST. Flat -> the ceiling is sound and
  everything downstream is trustworthy. Sloping -> subsample deeper and re-check before reading.
- CONSERVATION OF n on the data pipeline; realised vs selected atom range printed.

IF THE CURVE IS FLAT: run the DIAGNOSIS FAN-OUT, do not propose a design (INBOX 004a/004c). The
previous version of this file pre-committed to one diagnosis -- an ANM-basis decoder with a learned
residual -- and printed it as the recorded next step. That has been DELETED. It named a cause for a
result nobody had seen, and the cure it named risked an open-ended ANM optimisation loop. Six
mechanisms produce a flat curve, each with a distinguishing test; the verdict block enumerates them
and the run reports which one the evidence supports BEFORE any redesign is proposed."""
import os, sys, json, time, math, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
from scipy import stats
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import (AtlasStore, pca_ceiling_framespace, sysdata, ho_frames,
                             ho_cols, anm_fve_streamed, train_frames, ceiling_chunked)
from armf_anm import modes as anm_modes
MAN = f"{WR}/atlas_manifest.json"; RES = f"{WR}/atlas_curve.json"
CURVE = [50, 100, 200, 400, 700]
L = 24; DM = 256; KS = [24]
CUTOFFS = (5.0, 7.0)
FRAMES_PER_STEP = 8; MAXSTEPS = 40000; EVAL_EVERY = 2000; PATIENCE = 5; COMPETENCE = 0.05
np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Codec(nn.Module):
    """Same architecture at every n_train -- only the training-set SIZE varies."""
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
    for s in range(0, d["F"], chunk):                      # held-out = replica 2, an INDEPENDENT run
        e = min(s + chunk, d["F"])
        t = ho_frames(d, s, e)
        dw = torch.tensor((t / d["scale"]).astype(np.float32), device=dev)
        p = mdl(st.unsqueeze(0).expand(e - s, -1, -1), dw).cpu().numpy().astype(np.float64) * d["scale"]
        sse += float(((t - p) ** 2).sum())
    return 1 - sse / (d["sst"] + 1e-12)


def train(tr, n, ho_track):
    mdl = Codec(tr[0]["stat"].shape[1], L, DM).to(dev)
    opt = torch.optim.Adam(mdl.parameters(), 1e-3); hist = []
    for st in range(1, MAXSTEPS + 1):
        d = tr[np.random.randint(len(tr))]
        idx = np.random.randint(0, 2 * d["F"], FRAMES_PER_STEP)     # replicas 0+1
        S = torch.tensor(d["stat"], device=dev).unsqueeze(0).expand(FRAMES_PER_STEP, -1, -1)
        D = torch.tensor(train_frames(d, idx).reshape(FRAMES_PER_STEP, d["N"], 3) / d["scale"], device=dev)
        opt.zero_grad(); ((mdl(S, D) - D) ** 2).mean().backward(); opt.step()
        if st % EVAL_EVERY == 0:
            mdl.eval(); v = float(np.mean([fve_model(mdl, x) for x in ho_track])); mdl.train()
            hist.append((st, v))
            print(f"      n={n} step {st:>6}: held-out FVE {v:+.4f}", flush=True)
            comp = any(h[1] > COMPETENCE for h in hist)          # competence gate
            if comp and len(hist) > PATIENCE and max(h[1] for h in hist[-PATIENCE:]) <= max(h[1] for h in hist[:-PATIENCE]) * 1.01:
                print(f"      early stop at {st} (competent, no improvement for {PATIENCE} evals)", flush=True); break
    return mdl, hist


print(f"[atlas-curve] device={dev}  L={L} DM={DM}  n_train curve {CURVE}", flush=True)
man = json.load(open(MAN))
store = AtlasStore(f"{WR}/atlas_cache")
have = {m["pdb"]: i for i, m in enumerate(store.meta)}
ho_ids = [p for p in man["heldout"] if p in have]
tr_ids = [p for p in man["train_ordered"] if p in have]
print(f"  manifest: {len(man['heldout'])} held-out / {len(man['train_ordered'])} train pool", flush=True)
print(f"  cached:   {len(ho_ids)} held-out / {len(tr_ids)} train pool", flush=True)
if len(tr_ids) < CURVE[0] or len(ho_ids) < 20:
    print("  cache still building -- rerun when it has enough"); raise SystemExit

HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
store.conservation_report()
print(f"  held-out atom range {min(x['N'] for x in HO)}-{max(x['N'] for x in HO)}, "
      f"log10 span {np.log10(max(x['N'] for x in HO)/min(x['N'] for x in HO)):.2f} decades", flush=True)

# ---------------- ENTRY CONDITION: PCA-baseline-vs-N guard ----------------
print("\n=== GUARD (entry condition): does the PCA baseline degrade with N on ATLAS? ===", flush=True)
pc = {}
for d in HO:
    pc[d["pdb"]] = ceiling_chunked(d, KS)
x = np.log10([d["N"] for d in HO]); y = np.array([pc[d["pdb"]][KS[0]] for d in HO])
lr = stats.linregress(x, y); hw = stats.t.ppf(0.975, len(x) - 2) * lr.stderr
span = x.max() - x.min()
print(f"  PCA-{KS[0]} vs log10(N): slope {lr.slope:+.4f} +/- {hw:.4f}  p={lr.pvalue:.3f}  n={len(HO)}")
print(f"  realised log-N span {span:.2f} decades" + ("" if span >= 1.2 else "   <-- UNDER 1.2: guard UNTESTED"))
ok = abs(lr.slope) < hw
GUARD = dict(slope=float(lr.slope), hw=float(hw), p=float(lr.pvalue), n=len(HO), span=float(span), flat=bool(ok))
print(f"  -> {'FLAT: ceiling sound' if ok else 'SLOPING: the CEILING DEGRADES WITH N on ATLAS too'}")
print(f"  FAMILY C: this null permits a slope up to {abs(lr.slope)+hw:+.4f}, i.e. a ceiling change of "
      f"{(abs(lr.slope)+hw)*span:+.3f} FVE across the range.", flush=True)
# THE GUARD DOES NOT BLOCK. The PRIMARY comparison -- codec vs ANM -- is two ZERO-SHOT methods scored
# on the same held-out frames: no ceiling, no denominator, immune to this entirely. A failed guard
# damages ONLY the secondary oracle-fraction number.
if not ok:
    print("  *** GUARD FAILED -- but the run PROCEEDS. ***", flush=True)
    print("  PRIMARY (codec vs ANM, and the n_train curve) is CEILING-FREE and unaffected.", flush=True)
    print("  SECONDARY (oracle-fraction) is reported WITH this slope attached so the bias is visible.", flush=True)
    print(f"  BIAS DIRECTION: the ceiling DEGRADES with N (slope {lr.slope:+.4f}), so the oracle-fraction", flush=True)
    print("  looks BETTER at high N than it truly is -- any high-N oracle-fraction is an UPPER BOUND,", flush=True)
    print("  not a flattering artifact to mistake for success.", flush=True)

# ---------------- ANM peer: cutoff swept on TRAINING systems only ----------------
print("\n=== PEER: ANM cutoff swept on TRAINING systems (FAMILY E) ===", flush=True)
tr_probe = [x for x in (sysdata(store, have[p]) for p in tr_ids[:30]) if x is not None]
best_c, means = None, {}
for c in CUTOFFS:
    v = []
    for d in tr_probe:
        V, att, dt = anm_modes(d["ref"], KS[0], c)
        if V is not None: v.append(anm_fve_streamed(d, V))
    means[c] = float(np.mean(v)) if v else float('nan')
    print(f"  cutoff {c:>4} A: mean TRAIN FVE {means[c]:.4f} (n={len(v)})", flush=True)
best_c = max((c for c in means if np.isfinite(means[c])), key=lambda c: means[c])
print(f"  SELECTED {best_c} A on training; applied unchanged to held-out.", flush=True)
anm_ho = {}
for d in HO:
    V, att, dt = anm_modes(d["ref"], KS[0], best_c)
    anm_ho[d["pdb"]] = anm_fve_streamed(d, V) if V is not None else np.nan
cov = np.mean([np.isfinite(v) for v in anm_ho.values()])
print(f"  ANM computable on {cov*100:.0f}% of held-out systems", flush=True)

# ---------------- THE CURVE ----------------
print(f"\n=== PRIMARY (CEILING-FREE): learning curve, codec vs ANM -- both zero-shot, same held-out ===", flush=True)
print(f"  oracle-fraction is SECONDARY and carries the guard slope {GUARD['slope']:+.4f} +/- {GUARD['hw']:.4f}"
      + ("" if GUARD['flat'] else "  (ceiling degrades with N -> high-N oracle-fraction is an UPPER BOUND)"), flush=True)
res = {}
track = HO[:8]
print(f"  {'n_train':>8}{'codec FVE':>12}{'ANM FVE':>10}{'codec-ANM':>12}{'frac codec>ANM':>16}{'PCA oracle':>12}{'% of oracle':>13}")
for n in [c for c in CURVE if c <= len(tr_ids)]:
    TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:n]) if x is not None]
    mdl, hist = train(TR, n, track)
    mdl.eval()
    cf = np.array([fve_model(mdl, d) for d in HO])
    af = np.array([anm_ho[d["pdb"]] for d in HO])
    of = np.array([pc[d["pdb"]][KS[0]] for d in HO])
    m = np.isfinite(af)
    res[n] = dict(guard=GUARD, codec=float(np.mean(cf)), anm=float(np.nanmean(af)),
                  diff=float(np.mean(cf[m] - af[m])), frac=float(np.mean(cf[m] > af[m])),
                  oracle=float(np.mean(of)), pct=float(np.mean(cf / np.maximum(of, 1e-9))),
                  n_used=len(TR))
    json.dump(res, open(RES, "w"))
    r = res[n]
    print(f"  {n:>8}{r['codec']:>12.4f}{r['anm']:>10.4f}{r['diff']:>+12.4f}{r['frac']*100:>15.0f}%"
          f"{r['oracle']:>12.4f}{r['pct']*100:>12.0f}%", flush=True)

ns = sorted(res); cd = [res[n]["codec"] for n in ns]; df = [res[n]["diff"] for n in ns]
if len(ns) >= 3:
    lr2 = stats.linregress(np.log10(ns), cd); h2 = stats.t.ppf(0.975, len(ns)-2)*lr2.stderr
    print(f"\n  codec FVE vs log10(n_train): slope {lr2.slope:+.4f} +/- {h2:.4f}  p={lr2.pvalue:.3f}")
    climbing = lr2.slope - h2 > 0
    crossed = any(d > 0 for d in df)
    if climbing and crossed: v = "(a) DATA-LIMITED -- climbing AND crossing ANM"
    elif climbing: v = "(a) PARTIAL -- still climbing; ANM crossing is a diagnostic, not a gate"
    else: v = "(b) FLAT across the n range -- RUN THE DIAGNOSIS FAN-OUT (below). Do NOT infer a cause."
    print(f"  VERDICT: {v}")
    if not climbing:
        # INBOX 004a. This branch used to print the ANM-basis-decoder design as the recorded next
        # step. DELETED. It pre-committed to ONE diagnosis of a result nobody had seen yet, and the
        # diagnosis it named would have pulled the project into an open-ended ANM optimisation loop.
        # A flat curve licenses a fan-out, not a design.
        print("  -> A FLAT CURVE HAS AT LEAST SIX CAUSES. Each has a distinguishing test; run them")
        print("     BEFORE proposing any redesign, and report which hypothesis the evidence supports:")
        for h, t in (
            ("decoder capacity",       "raise decoder depth/width at fixed L=1, DM. FVE rises => decoder-limited"),
            ("one-token info limit",   "the DM sweep itself. FVE still rising at DM=512 => information-limited"),
            ("conditioning",           "enrich static features (local frames, neighbour geometry). FVE rises => conditioning-limited"),
            ("objective mismatch",     "geometry-aware loss (pairwise-distance / per-mode weighted). Dynamical fidelity improves while MSE does not => the LOSS was wrong"),
            ("representation",         "local-frame / internal-coordinate target instead of Cartesian displacement"),
            ("encoder pooling",        "G6 permutation + effective rank of the latent across systems. Realised rank << DM => the encoder is not filling the token"),
        ):
            print(f"       - {h:<22} {t}")
        print("     ANM is a DIAGNOSTIC, not the bar: it says whether the learned map carries")
        print("     information a physics prior does not. It cannot produce a programmable latent")
        print("     dynamics system, so failing to beat it is NOT a death certificate for the codec.")
    elif not crossed:
        tgt = np.interp(0, df, np.log10(ns)) if min(df) < 0 < max(df) else None
        print(f"  -> extrapolated crossing at n_train ~ {10**np.interp(0,[df[0],df[-1]],[np.log10(ns[0]),np.log10(ns[-1])]):.0f}"
              if df[-1] > df[0] else "  -> gap not closing; extrapolation not meaningful")


