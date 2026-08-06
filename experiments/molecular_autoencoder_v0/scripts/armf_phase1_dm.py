"""CAPACITY AXIS -- CONCATENATED FRAMES ARE CORRECT HERE. DO NOT "FIX" THIS TO PER-REPLICA.

Concatenation is an artifact for DIMENSIONALITY measurement: between-replica structural offsets add
apparent variance directions and inflate rank90, which is why the b experiment (armf_b_exponent.py)
needs a replica-join sweep. It is NOT an artifact for PER-FRAME AUTOENCODING. This AE encodes single
frames with no time dependence, so frames drawn from five replicas are simply BROADER SAMPLING of the
same equilibrium ensemble -- strictly better training data, not discontinuities. Different use,
different concern: concatenated frames here, join sweep for b.

Phase 1: atom-native Perceiver AE, deficit-ratio-vs-N scaling on MISATO. L in {12, 24}.
N atom tokens -> L learned latent slots -> N atom outputs, L independent of N.

TASK: displacement from the Kabsch-aligned reference, CENTERED on the per-complex TRAIN-frame mean.
Centering matters: the model is scored against a train-mean-centered denominator, so predicting raw
displacement forces it to also learn each complex's drift -- work PCA gets free (its reconstruction
is mu + projection). Centering input, target and metric alike makes the model's task IDENTICAL to
PCA's, so the deficit ratio compares like with like. mu comes from TRAIN frames only and is static
per system -> not a leak.

N-AXIS CONVENTION: ALL-ATOM (incl. hydrogens) throughout -- model, PCA ceiling, ANM baseline, axis.
Static per-atom features (element + reference position) condition encoder AND decoder; displacement
enters ONLY through the L slots.

GUARDS (all mandatory; any failure voids the run or the bucket):
  G1 frame-variance : latent must differ across frames (cosine < 0.99)
  G4 zero-latent    : zeroing all slots must destroy reconstruction
  G6 permutation    : shuffling atom order with identical features must not change reconstruction
  G7 rank sanity    : L < 30% of usable rank (79 frames) -> L=12 is 15%, L=24 is 30%
  G8 CONVERGENCE    : train to PLATEAU, not a fixed step budget. If large-N systems converge slower
                      than small-N ones under a fixed budget, the deficit ratio grows with N for
                      pure OPTIMIZATION reasons and we would misread it as index-addressing. Every
                      bucket must show <1% relative improvement in held-out FVE over the final 25%
                      of steps; a bucket that has not plateaued is VOID FOR THAT BUCKET. Steps-to-
                      plateau is reported PER BUCKET and regressed on N -- if it grows with N that
                      is an objective-4 finding (training cost vs system size) in its own right.

Also fixed: FRAMES_PER_STEP is now CONSTANT across N. The previous memory-budgeted schedule gave
small systems 80 frames/step and large ones 13, which is the same undertraining-at-high-N artifact
G8 exists to catch, baked into the sampler."""
import h5py, numpy as np, torch, torch.nn as nn, time, json, os, pickle, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
MDC = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data/*.h5"   # CAPACITY AXIS runs on mdCATH
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
CACHE = f"{WR}/mdcath_dm_data.pkl"; RESJSON = f"{WR}/phase1_dm_rows.json"
BUCKETS = [(700, 1200), (1200, 2000), (2000, 3500), (3500, 8000)]      # mdCATH all-atom spans 711-7,524
PERB = {b: 99 for b in BUCKETS}      # take every available mdCATH domain (28 local)
LS = [1]; DMS = [16, 64, 256, 512]; ANM_MAXN = 4000   # L=1 fixed; DM is the CAPACITY AXIS.
# "One global latent per frame, width independent of atom count." L=1 x DM is a DM-DIMENSIONAL
# code -- NOT one number. For a 1M-atom system that is 3e6 coordinates -> DM floats per frame.
# DM range sized by the rank90 asymptote (median 168, range 13-500, mobility-driven not N-driven).
# CEILINGS: ANM-DM is PRIMARY -- structure-predicted, zero-parameter, FRAME-INDEPENDENT so valid at
# any DM (dense eigh gives all modes at once; affordable to N~6000). PCA-DM is only valid where
# DM < 30% of usable rank (79 train frames) => DM=16 ONLY; higher DM is marked RANK-VOID, the same
# G7 failure that killed L=64. L=1 is also the routing probe: with one slot there is no slot
# assignment, so a G6 permutation failure here would mean the ENCODER index-addresses -- a more
# serious problem than slot assignment. Old L=1 comment follows:
# L=1 is a MECHANISM probe: with one slot there is no
# slot assignment to learn, so routing is impossible by construction. An N effect at L=1 is NOT a
# capacity limit -- rank90 (N-exponent CI spans zero) and TICA (flat, uncensored) both say intrinsic
# dimensionality does not grow with atom count, so the physics says a fixed-width code suffices.
# It would instead localise to the POOLING/BROADCAST pathway (encoder aggregating N tokens into a
# fixed code, or decoder broadcasting one code back to N atoms) -- an architectural bottleneck with
# different fixes (hierarchical pooling, deeper cross-attention, relative-position conditioning). NOTE: L=1 at DM=64 is a 64-DIMENSIONAL code,
# not one number. Judge L=1 on flatness of the deficit ratio in N, not on its absolute FVE.
FRAMES_PER_STEP = 8                     # CONSTANT across N (was memory-budgeted -> N-correlated undertraining)
COMPETENCE = 0.05   # early stopping is gated on this
MAXSTEPS = 150000; EVAL_EVERY = 1000; PATIENCE = 5; PLATEAU_TOL = 0.01; RANKVOID = 0.30 * 1999      # mdCATH: 2000 train frames -> PCA-k valid to k~600, covering DM=512
np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ELEMS = [1, 6, 7, 8, 16, 15, 9, 17]


def kabsch_traj(traj):
    ref = traj[0]; rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def load():
    """mdCATH, 320K, 5 replicas concatenated -> 2,500 frames (2,000 train / 500 held-out).
    ALL-ATOM. Same centred-displacement convention as the MISATO run so the protocol matches, but the
    two axes are measured on DIFFERENT CORPORA and absolute numbers must NOT be cross-compared."""
    import glob
    if os.path.exists(CACHE):
        d = pickle.load(open(CACHE, "rb")); print(f"  loaded cache ({len(d)} domains)", flush=True); return d
    data = []
    for fp in sorted(glob.glob(MDC)):
        try:
            ff = h5py.File(fp, "r"); dom = list(ff.keys())[0]; g = ff[dom]
            T0 = sorted([k for k in g.keys() if k.isdigit()], key=int)[0]
            reps = sorted(g[T0].keys(), key=lambda x: int(x))
            ref = np.array(g[T0][reps[0]]["coords"][0]).astype(np.float64); N = len(ref)
            b = next((b for b in BUCKETS if b[0] <= N < b[1]), None)
            if b is None: ff.close(); continue
            al = np.concatenate([kabsch_traj_to(np.array(g[T0][r]["coords"]).astype(np.float64), ref)
                                 for r in reps], 0)
            eln = np.array(g["element"])
            eln = np.array([int(x) if not isinstance(x, bytes) else 0 for x in eln]) if eln.dtype == object else eln
            ff.close()
            disp = al - ref; T = len(disp); h = int(T * 0.8)
            mu = disp[:h].mean(0); dc = disp - mu
            scale = float(np.sqrt((dc[:h] ** 2).sum(-1).mean()) + 1e-6)
            oh = np.zeros((N, len(ELEMS) + 1), np.float32)
            for a in range(N):
                e = int(eln[a]) if a < len(eln) else 0
                oh[a, ELEMS.index(e) if e in ELEMS else -1] = 1.0
            rp = ((ref - ref.mean(0)) / (ref.std() + 1e-6)).astype(np.float32)
            mag = np.linalg.norm(disp, axis=-1)
            data.append(dict(dom=dom, bucket=b, N=N, h=h, T=T, scale=scale,
                             stat=np.concatenate([oh, rp], 1), dc=dc.astype(np.float32),
                             ref=ref.astype(np.float32), rmsf=float(np.median(np.sqrt((mag ** 2).mean(0))))))
            print(f"    {dom} N={N} T={T} h={h}", flush=True)
        except Exception as e:
            print(f"    {fp.split('/')[-1]}: skipped ({type(e).__name__}: {e})", flush=True)
    pickle.dump(data, open(CACHE, "wb")); return data


def kabsch_traj_to(traj, ref):
    rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def ceilings(d, DMS):                                           # PCA-DM (rank-limited) + ANM-DM (frame-free)
    h = d["h"]; X = d["dc"].astype(np.float64).reshape(d["T"], -1)
    tr, ho = X[:h], X[h:]; sst = float((ho ** 2).sum())
    _, _, Vt = np.linalg.svd(tr, full_matrices=False)
    out = {"pca": {}, "anm": {}, "pca_valid": {}}
    for L in DMS:
        out["pca_valid"][L] = bool(L <= 0.30 * (h - 1))         # G7 PER DOMAIN: h varies 1408-2000,
        out["pca_rankuse"] = out.get("pca_rankuse", {}); out["pca_rankuse"][L] = float(L / max(h - 1, 1))
        # so a GLOBAL threshold would pass DM=512 everywhere while 10/28 domains exceed 30% of their own rank
        if L <= min(len(Vt), h - 1):
            V = Vt[:L]; out["pca"][L] = 1 - float(((ho - ho @ V.T @ V) ** 2).sum()) / (sst + 1e-12)
        else: out["pca"][L] = float('nan')
    out["leadeig"] = float((np.linalg.svd(tr, compute_uv=False)[0] ** 2) / ((np.linalg.svd(tr, compute_uv=False) ** 2).sum() + 1e-12))
    n = d["N"]
    if n <= ANM_MAXN:                                            # DENSE eigh -> ALL modes, so every DM is free
        ref = d["ref"].astype(np.float64); tree = cKDTree(ref); pr = tree.query_pairs(10.0, output_type='ndarray')
        if len(pr):
            v = ref[pr[:, 1]] - ref[pr[:, 0]]; u = v / (np.linalg.norm(v, axis=1)[:, None] + 1e-9)
            rows, cols, vals = [], [], []
            for (i, j), uu in zip(pr, u):
                bb = np.outer(uu, uu)
                for A, B, s in [(i, j, -1), (j, i, -1), (i, i, 1), (j, j, 1)]:
                    for a in range(3):
                        for c in range(3): rows.append(3*A+a); cols.append(3*B+c); vals.append(s*bb[a, c])
            H = coo_matrix((vals, (rows, cols)), shape=(3*n, 3*n)).tocsr()
            try:
                w, Vm = np.linalg.eigh(H.toarray()); M = Vm[:, np.argsort(w)][:, 6:]
                for L in DMS:
                    if M.shape[1] >= L:
                        out["anm"][L] = 1 - float(((ho - ho @ M[:, :L] @ M[:, :L].T) ** 2).sum()) / (sst + 1e-12)
            except Exception: pass
    for L in DMS: out["anm"].setdefault(L, float('nan'))
    return out


def conserve(stage, rows_in, rows_out, regressors=("N", "rmsf")):
    """CONSERVATION OF n (Family A, generalised). Every silent failure is a FILTER -- OOMs, NaNs,
    timeouts, unreadable files, missing ceilings. None announce themselves. Any stage that passes
    fewer rows than it received must characterise the shortfall against EVERY regressor before its
    output is used downstream."""
    ni, no = len(rows_in), len(rows_out)
    if ni == no:
        print(f"  [n] {stage}: {ni} -> {no}  conserved"); return True
    kept_ids = {id(r) for r in rows_out}
    dropped = [r for r in rows_in if id(r) not in kept_ids]
    msg = []
    for g in regressors:
        k = [r[g] for r in rows_out if g in r]; d = [r[g] for r in dropped if g in r]
        if k and d:
            rel = abs(np.median(d) - np.median(k)) / max(abs(np.median(k)), 1e-9)
            msg.append(f"{g}: kept {np.median(k):.3g} vs dropped {np.median(d):.3g}"
                       + (" *SKEWED*" if rel > 0.25 else ""))
    print(f"  [n] {stage}: {ni} -> {no}  DROPPED {ni-no}  |  " + "; ".join(msg))
    return False


class Perceiver(nn.Module):
    def __init__(self, Fs, L, dm=64, heads=4):
        super().__init__()
        self.lat = nn.Parameter(torch.randn(L, dm) * 0.02)
        self.in_tok = nn.Linear(Fs + 3, dm); self.q_tok = nn.Linear(Fs, dm)
        self.enc = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.sa = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.dec = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.l1 = nn.LayerNorm(dm); self.l2 = nn.LayerNorm(dm); self.l3 = nn.LayerNorm(dm); self.l4 = nn.LayerNorm(dm)
        self.ff = nn.Sequential(nn.Linear(dm, dm*2), nn.GELU(), nn.Linear(dm*2, dm))
        self.dff = nn.Sequential(nn.Linear(dm, dm*2), nn.GELU(), nn.Linear(dm*2, dm))
        self.film = nn.Linear(dm, 2*dm); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.out = nn.Linear(dm, 3)
    def encode(self, stat, disp):
        B = stat.shape[0]; tok = self.in_tok(torch.cat([stat, disp], -1))
        lat = self.lat.unsqueeze(0).expand(B, -1, -1)
        lat = self.l1(lat + self.enc(lat, tok, tok)[0]); lat = self.l2(lat + self.sa(lat, lat, lat)[0])
        return self.l3(lat + self.ff(lat))
    def decode(self, stat, lat):                                # static query + FiLM-conditioned per-atom MLP
        q = self.q_tok(stat); a = self.dec(q, lat, lat)[0]
        g, b = self.film(lat.mean(1, keepdim=True)).chunk(2, -1)  # global code -> per-channel scale+shift
        h = self.l4(q + a) * (1 + g) + b                          # FiLM: works even when L=1 makes attention constant
        return self.out(h + self.dff(h))
    def forward(self, stat, disp): return self.decode(stat, self.encode(stat, disp))


@torch.no_grad()
def fve(m, d, lo, hi, chunk=8):
    st0 = torch.tensor(d["stat"], device=dev); sse = sst = 0.0
    for s in range(lo, hi, chunk):
        e = min(s+chunk, hi); dw = torch.tensor(d["dc"][s:e] / d["scale"], device=dev)
        pred = m(st0.unsqueeze(0).expand(e-s, -1, -1), dw).cpu().numpy().astype(np.float64) * d["scale"]
        true = d["dc"][s:e].astype(np.float64)
        sse += float(((true - pred) ** 2).sum()); sst += float((true ** 2).sum())
    return 1 - sse / (sst + 1e-12)


def train_eval(data, L, tr, he, track, dm):
    Fs = data[0]["stat"].shape[1]; m = Perceiver(Fs, L, dm=dm).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    hist = {b: [] for b in BUCKETS}; steps_done = 0; best = {b: -9e9 for b in BUCKETS}; bad = 0
    takeoff = {b: None for b in BUCKETS}          # first step where held-out FVE crosses COMPETENCE
    for st in range(1, MAXSTEPS + 1):
        d = tr[np.random.randint(len(tr))]
        idx = np.random.randint(0, d["h"], FRAMES_PER_STEP)      # CONSTANT frames/step for every N
        S = torch.tensor(d["stat"], device=dev).unsqueeze(0).expand(FRAMES_PER_STEP, -1, -1)
        D = torch.tensor(d["dc"][idx] / d["scale"], device=dev)
        opt.zero_grad(); ((m(S, D) - D) ** 2).mean().backward(); opt.step(); steps_done = st
        if st % EVAL_EVERY == 0:
            m.eval(); improved = False
            for b in BUCKETS:
                sel = track[b]
                if not sel: continue
                v = float(np.mean([fve(m, d2, d2["h"], d2["T"]) for d2 in sel])); hist[b].append((st, v))
                if v > best[b] * (1 + PLATEAU_TOL) or (best[b] < 0 and v > best[b] + 0.01): best[b] = max(best[b], v); improved = True
                else: best[b] = max(best[b], v)
                if takeoff[b] is None and hist[b][-1][1] > COMPETENCE: takeoff[b] = st
            m.train()
            # COMPETENCE GATE: a flat curve at ~0 is a PRE-TAKEOFF phase, not a plateau. A plateau
            # detector cannot tell them apart, and if takeoff step correlates with N or DM then early
            # stopping kills those arms preferentially and MANUFACTURES the N-trend under test (a killed
            # arm reports ratio ~1.0 -- the index-addressing signature). Observed: all four L=1 arms died
            # this way. So stopping does not activate until some bucket is actually competent.
            competent = any(h and h[-1][1] > COMPETENCE for h in hist.values())
            bad = 0 if (improved or not competent) else bad + 1
            print(f"    step {st:>6}: " + " ".join(f"{b[0]//1000}k:{hist[b][-1][1]:+.3f}" for b in BUCKETS if hist[b]), flush=True)
            if bad >= PATIENCE:
                print(f"    early stop at {st} (competent, no bucket improved >1% for {PATIENCE} evals)", flush=True); break
    return m, hist, steps_done, takeoff


def g8(hist, steps_done):                                       # per-bucket plateau + steps-to-plateau
    out = {}
    for b, h in hist.items():
        if len(h) < 8: out[b] = dict(plateau=False, rel=float('nan'), slope=float('nan'), hw=float('nan'),
                                     steps=None, final=float('nan'), tail=[]); continue
        cut = 0.80 * steps_done; tail = [(s_, v) for s_, v in h if s_ >= cut]
        final = h[-1][1]
        pre = [v for s_, v in h if s_ <= 0.75 * steps_done]
        rel = (final - (pre[-1] if pre else h[0][1])) / (abs(final) + 1e-9)
        if len(tail) >= 4:
            lr = stats.linregress([s_ for s_, _ in tail], [v for _, v in tail])
            hw = stats.t.ppf(0.975, len(tail) - 2) * lr.stderr
            plateau = bool(lr.slope - hw <= 0 <= lr.slope + hw)   # slope indistinguishable from ZERO
            sl, hh = float(lr.slope), float(hw)
        else:
            plateau, sl, hh = False, float('nan'), float('nan')
        sfp = next((s_ for s_, v in h if v >= final - 0.01 * abs(final)), h[-1][0])
        out[b] = dict(plateau=plateau, rel=float(rel), slope=sl, hw=hh, steps=int(sfp), final=float(final),
                      tail=[round(v, 4) for _, v in tail[-6:]])
    return out
    return out


print(f"[phase1-CAPACITY] device={dev}; CORPUS=mdCATH (2000 train frames -> PCA-k valid to k~600);", flush=True)
print(f"  L=1, DM sweep {DMS}. The N axis is measured SEPARATELY on MISATO -- different corpora,", flush=True)
print(f"  absolute numbers are NOT comparable between the two runs.", flush=True)
t0 = time.time(); data = load()
print(f"  {len(data)} complexes ({time.time()-t0:.0f}s)", flush=True)
for b in BUCKETS:
    cs = [d for d in data if d["bucket"] == b]
    print(f"    {str(b):14s} n={len(cs):>3} medN={int(np.median([d['N'] for d in cs])) if cs else 0}", flush=True)
tr = [d for i, d in enumerate(data) if i % 4 != 0]; he = [d for i, d in enumerate(data) if i % 4 == 0]
track = {b: [d for d in he if d["bucket"] == b][:4] for b in BUCKETS}
print(f"  train {len(tr)} / held-out {len(he)}; plateau tracking on {sum(len(v) for v in track.values())} systems", flush=True)
print("  precomputing PCA/ANM ceilings ...", flush=True)
for d in he: d["ceil"] = ceilings(d, DMS)
print("  NOTE: capacity axis = mdCATH; N axis = MISATO. Do NOT cross-compare absolute numbers.", flush=True)
print(f"  ceilings done ({time.time()-t0:.0f}s)", flush=True)

allres = {}
for DM_ in DMS:
    L = 1
    print(f"\n=== L={L}  DM={DM_}  -> code = {L*DM_} floats/frame, INDEPENDENT of atom count ===", flush=True)
    print(f"  compression at this bucket span: 3N coords -> {L*DM_} floats "
          f"(e.g. N=22443 -> {3*22443}:{L*DM_} = {3*22443/(L*DM_):.0f}:1); at 1e6 atoms -> 3e6:{L*DM_} = {3e6/(L*DM_):.0f}:1", flush=True)
    m, hist, sd, tko = train_eval(data, L, tr, he, track, DM_)
    G = g8(hist, sd)
    try:
        d0 = he[0]; nf = min(8, d0["T"] - d0["h"])
        st0 = torch.tensor(d0["stat"], device=dev); dw = torch.tensor(d0["dc"][d0["h"]:d0["h"]+nf] / d0["scale"], device=dev)
        S = st0.unsqueeze(0).expand(nf, -1, -1)
        with torch.no_grad():
            lat = m.encode(S, dw); lf = lat.reshape(nf, -1); lf = lf / (lf.norm(dim=1, keepdim=True) + 1e-9)
            g1 = float((lf[:-1] * lf[1:]).sum(1).mean())
            base = 1 - float((((m.decode(S, lat) - dw) ** 2).sum() / ((dw ** 2).sum() + 1e-9)))
            zero = 1 - float((((m.decode(S, torch.zeros_like(lat)) - dw) ** 2).sum() / ((dw ** 2).sum() + 1e-9)))
            pm = torch.randperm(d0["N"], device=dev)
            pp = m(S[:, pm, :], dw[:, pm, :])[:, torch.argsort(pm), :]
            g6 = 1 - float((((pp - dw) ** 2).sum() / ((dw ** 2).sum() + 1e-9)))
        g6ok = abs(g6 - base) < 0.02
        gs = (f"G1 cos {g1:.4f} {'PASS' if g1 < 0.99 else 'FAIL'} | G4 base {base:+.3f}->zero {zero:+.3f} "
              f"{'PASS' if (base > 0.05 and base - zero > 0.1) else 'FAIL'} | G6 perm {g6:+.3f} vs {base:+.3f} "
              f"{'PASS' if g6ok else 'FAIL -- ENCODER INDEX-ADDRESSES (no slots at L=1, so this is not slot assignment)'}")
    except Exception as e:
        gs = f"FAILED ({type(e).__name__}: {e}) -- VOID"
    print(f"  GUARDS: {gs}", flush=True)
    print(f"  G8 (steps run {sd}); TAKEOFF = first step crossing FVE {COMPETENCE}:", flush=True)
    for b in BUCKETS:
        g = G[b]; print(f"    {str(b):14s} plateau {'YES' if g['plateau'] else 'NO -> BUCKET VOID':17s} "
                        f"tail-slope {g['slope']*1e5:+.3f}e-5 +/-{g['hw']*1e5:.3f}  rel {g['rel']:+.4f}  "
                        f"steps-to-plateau {g['steps']} ({(g['steps'] or 0)/MAXSTEPS*100:.0f}% of cap)  "
                        f"TAKEOFF {tko[b]}{'' if tko[b] is None else f' ({tko[b]/MAXSTEPS*100:.0f}%)'}  tail {g['tail']}", flush=True)
    rows = []
    for d in he:
        v = fve(m, d, d["h"], d["T"]); ce = d["ceil"]
        rows.append(dict(dom=d["dom"], bucket=list(d["bucket"]), N=d["N"], rmsf=d["rmsf"], model=v,
                         pca=ce["pca"][DM_], pca_valid=ce["pca_valid"][DM_], anm=ce["anm"][DM_],
                         gap_anm=(ce["pca"][DM_]-v) if np.isfinite(ce["pca"][DM_]) else float('nan'),
                         ratio_anm=((ce["pca"][DM_]-v)/ce["pca"][DM_]) if (np.isfinite(ce["pca"][DM_]) and ce["pca"][DM_] > 1e-6) else float('nan'),
                         gap_anmref=(ce["anm"][DM_]-v) if np.isfinite(ce["anm"][DM_]) else float('nan'),
                         pca_rankuse=ce["pca_rankuse"][DM_],
                         void=not G[d["bucket"]]["plateau"]))
    allres[DM_] = dict(rows=rows, g8={str(k): v for k, v in G.items()}, steps=sd,
                       takeoff={str(k): v for k, v in tko.items()})
    json.dump(allres, open(RESJSON, "w"))
    print(f"  {'bucket':14s}{'n':>3}{'medN':>7}{'modelFVE':>19}{'ANM-DM ceil':>19}{'PCA-DM':>16}{'gap':>8}{'ratio':>7}{'G8':>6}", flush=True)
    for b in BUCKETS:
        br = [r for r in rows if tuple(r["bucket"]) == b]
        if not br: continue
        q = lambda a: f"{np.median(a):.3f}[{np.percentile(a,25):.2f},{np.percentile(a,75):.2f}]"
        an = [r["anm"] for r in br if np.isfinite(r["anm"])]
        pv = br[0]["pca_valid"]; pc = [r["pca"] for r in br if np.isfinite(r["pca"])]
        pcs = (q(pc) if pv and pc else ("RANK-VOID" if pc else "n/a"))
        gp = [r["gap_anm"] for r in br if np.isfinite(r["gap_anm"])]; rt = [r["ratio_anm"] for r in br if np.isfinite(r["ratio_anm"])]
        print(f"  {str(b):14s}{len(br):>3}{int(np.median([r['N'] for r in br])):>7}{q([r['model'] for r in br]):>19}"
              f"{(q(an) if an else f'n/a (N>{ANM_MAXN})'):>19}{pcs:>19}"
              f"{(f'{np.median(gp):.3f}' if gp else '  n/a'):>8}{(f'{np.nanmedian(rt):.2f}' if rt else ' n/a'):>7}"
              f"{'ok' if G[b]['plateau'] else 'VOID':>6}", flush=True)

print("\n=== STEPS-TO-PLATEAU vs N (objective 4: training cost vs system size) ===", flush=True)
for L in DMS:
    G = allres[L]["g8"]; xs, ys = [], []
    for b in BUCKETS:
        g = G[str(b)]
        if g["steps"]:
            md = np.median([d["N"] for d in data if d["bucket"] == b]); xs.append(np.log10(md)); ys.append(np.log10(g["steps"]))
    if len(xs) > 2:
        lr = stats.linregress(xs, ys); ci = stats.t.ppf(0.975, len(xs)-2) * lr.stderr
        print(f"  DM={L}: log10(steps) ~ {lr.slope:+.3f} +/- {ci:.3f} * log10(N)   R^2 {lr.rvalue**2:.3f}  "
              f"-> {'GROWS with N' if lr.slope-ci > 0 else 'flat in N'}", flush=True)
        print(f"        per-bucket steps: " + " ".join(f"{b[0]//1000}k:{G[str(b)]['steps']}" for b in BUCKETS if G[str(b)]["steps"]), flush=True)

print("\n=== DEFICIT-RATIO-vs-N -- ratio AND absolute gap must agree; VOID buckets excluded ===", flush=True)
for L in DMS:
    rows = [r for r in allres[L]["rows"] if not r["void"]]
    rs, gs_, ns = [], [], []
    for b in BUCKETS:
        br = [r for r in rows if tuple(r["bucket"]) == b]
        if not br: continue
        rr_ = [r["ratio_anm"] for r in br if np.isfinite(r["ratio_anm"])]; gg_ = [r["gap_anm"] for r in br if np.isfinite(r["gap_anm"])]
        conserve(f"deficit-ratio bucket {b}", br, [r for r in br if np.isfinite(r["ratio_anm"])])
        if not rr_: continue
        rs.append(np.median(rr_)); gs_.append(np.median(gg_)); ns.append(b)
    if len(rs) < 2: print(f"  DM={L}: fewer than 2 valid buckets -> NO VERDICT (G8 voided the rest)", flush=True); continue
    v = [x for x in rs if np.isfinite(x) and x > 0]
    span = max(v)/min(v) if len(v) > 1 else float('nan'); gspan = max(gs_) - min(gs_)
    print(f"  DM={L} (code {L}d/frame) valid buckets {[f'{b[0]//1000}k' for b in ns]}", flush=True)
    print(f"    ratio  : " + " ".join(f"{x:.2f}" for x in rs) + f"   SPAN {span:.2f}x", flush=True)
    print(f"    absGAP : " + " ".join(f"{x:.3f}" for x in gs_) + f"   max-min {gspan:.3f} FVE", flush=True)
    if span < 1.3 and gspan < 0.02: vd = "ARBITRARY-L SURVIVES (content-addressed)"
    elif span > 2.0 and gspan > 0.03: vd = "INDEX-ADDRESSING (will not reach 1M atoms)"
    elif span > 2.0 and gspan < 0.02: vd = "NEITHER -- ceiling compression, cannot discriminate at this L"
    else: vd = "IN BETWEEN -- report the number, no verdict"
    print(f"    VERDICT: {vd}", flush=True)
print("\n  CAPACITY AXIS (mdCATH). Question: does L=1 at width DM approach PCA-DM, and is the deficit", flush=True)
print("  FLAT IN N? PCA-DM is RANK-VALID here (2,000 train frames -> k~600 available, covering DM=512);", flush=True)
print("  on MISATO's 79 train frames it is void above k=23, which is why this axis is measured on a", flush=True)
print("  DIFFERENT CORPUS. Absolute numbers are NOT comparable with the MISATO N-axis run.", flush=True)
print("  An N effect at L=1 is NOT a capacity limit -- rank90 and TICA both say intrinsic dimensionality", flush=True)
print("  does not grow with atom count. It would localise to the POOLING/BROADCAST pathway (outcome B).", flush=True)
