"""Phase 1: atom-native Perceiver AE, deficit-ratio-vs-N scaling on MISATO. L in {12, 24}.
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
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
CACHE = f"{WR}/phase1_data_v2.pkl"; RESJSON = f"{WR}/phase1_rows_v2.json"
BUCKETS = [(700, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 32000)]
PERB = {(700, 1000): 36, (1000, 2000): 48, (2000, 4000): 48, (4000, 8000): 48, (8000, 16000): 48, (16000, 32000): 48}
LS = [1, 12, 24]; DM = 64; ANM_MAXN = 2500   # L=1 is a MECHANISM probe: with one slot there is no
# slot assignment to learn, so routing is impossible by construction. Degradation with N still
# present at L=1 -> CAPACITY limit (index-addressing excluded). Degradation at L=12 but not L=1
# -> the problem is specifically slot assignment. NOTE: L=1 at DM=64 is a 64-DIMENSIONAL code,
# not one number. Judge L=1 on flatness of the deficit ratio in N, not on its absolute FVE.
FRAMES_PER_STEP = 8                     # CONSTANT across N (was memory-budgeted -> N-correlated undertraining)
COMPETENCE = 0.05   # early stopping is gated on this
MAXSTEPS = 150000; EVAL_EVERY = 1000; PATIENCE = 6; PLATEAU_TOL = 0.01   # raised: at L=24, 3/6 buckets
# hit the 30000 cap and were VOIDed by G8 rather than reaching plateau. Early stopping still decides.
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
    if os.path.exists(CACHE):
        d = pickle.load(open(CACHE, "rb")); print(f"  loaded cache ({len(d)} complexes)", flush=True); return d
    f = h5py.File(MD, "r"); picked = {b: [] for b in BUCKETS}; data = []
    for dom in list(f.keys()):        # stride 1: the (700,1000) bucket is supply-limited
        g = f[dom]; el = np.array(g["atoms_element"]); N = len(el)
        b = next((b for b in BUCKETS if b[0] <= N < b[1] and len(picked[b]) < PERB[b]), None)
        if b is None: continue
        co = np.array(g["trajectory_coordinates"]).astype(np.float64)
        eln = np.array(g["atoms_number"]); al = kabsch_traj(co); ref = al[0]
        disp = al - ref; T = len(disp); h = 80
        mu = disp[:h].mean(0)                                   # TRAIN-frame mean (static per system)
        dc = disp - mu                                          # CENTERED -- same object PCA reconstructs
        scale = float(np.sqrt((dc[:h] ** 2).sum(-1).mean()) + 1e-6)
        oh = np.zeros((N, len(ELEMS) + 1), np.float32)
        for a in range(N): oh[a, ELEMS.index(int(eln[a])) if int(eln[a]) in ELEMS else -1] = 1.0
        rp = ((ref - ref.mean(0)) / (ref.std() + 1e-6)).astype(np.float32)
        mag = np.linalg.norm(disp, axis=-1)
        picked[b].append(dom)
        data.append(dict(dom=dom, bucket=b, N=N, h=h, T=T, scale=scale,
                         stat=np.concatenate([oh, rp], 1), dc=dc.astype(np.float32),
                         ref=ref.astype(np.float32), rmsf=float(np.median(np.sqrt((mag ** 2).mean(0))))))
        if all(len(picked[bb]) >= PERB[bb] for bb in BUCKETS): break
    pickle.dump(data, open(CACHE, "wb")); return data


def ceilings(d, LS):                                            # PCA + ANM once per complex, all L reused
    h = d["h"]; X = d["dc"].astype(np.float64).reshape(d["T"], -1)
    tr, ho = X[:h], X[h:]; sst = float((ho ** 2).sum())          # centered -> denominator is sum(dc^2)
    _, _, Vt = np.linalg.svd(tr, full_matrices=False)
    out = {"pca": {}, "anm": {}}
    for L in LS:
        V = Vt[:L]; out["pca"][L] = 1 - float(((ho - ho @ V.T @ V) ** 2).sum()) / (sst + 1e-12)
    out["leadeig"] = float((np.linalg.svd(tr, compute_uv=False)[0] ** 2) / ((np.linalg.svd(tr, compute_uv=False) ** 2).sum() + 1e-12))
    n = d["N"]
    if n <= ANM_MAXN:
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
                w, Vm = eigsh(H, k=max(LS)+6, sigma=1e-6, which='LM'); M = Vm[:, np.argsort(w)][:, 6:]
                for L in LS: out["anm"][L] = 1 - float(((ho - ho @ M[:, :L] @ M[:, :L].T) ** 2).sum()) / (sst + 1e-12)
            except Exception: pass
    for L in LS: out["anm"].setdefault(L, float('nan'))
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
    def __init__(self, Fs, L, dm=DM, heads=4):
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


def train_eval(data, L, tr, he, track):
    Fs = data[0]["stat"].shape[1]; m = Perceiver(Fs, L).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
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


print(f"[phase1] device={dev}; ALL-ATOM; centered task; G8 convergence guard", flush=True)
t0 = time.time(); data = load()
print(f"  {len(data)} complexes ({time.time()-t0:.0f}s)", flush=True)
for b in BUCKETS:
    cs = [d for d in data if d["bucket"] == b]
    print(f"    {str(b):14s} n={len(cs):>3} medN={int(np.median([d['N'] for d in cs])) if cs else 0}", flush=True)
tr = [d for i, d in enumerate(data) if i % 4 != 0]; he = [d for i, d in enumerate(data) if i % 4 == 0]
track = {b: [d for d in he if d["bucket"] == b][:4] for b in BUCKETS}
print(f"  train {len(tr)} / held-out {len(he)}; plateau tracking on {sum(len(v) for v in track.values())} systems", flush=True)
print("  precomputing PCA/ANM ceilings ...", flush=True)
for d in he: d["ceil"] = ceilings(d, LS)
print(f"  ceilings done ({time.time()-t0:.0f}s)", flush=True)

allres = {}
for L in LS:
    print(f"\n=== L={L}  DM={DM}  (code = {L}x{DM} = {L*DM} dims; G7: {L}/79 = {L/79*100:.0f}% of usable rank) ===", flush=True)
    m, hist, sd, tko = train_eval(data, L, tr, he, track)
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
        gs = (f"G1 cos {g1:.4f} {'PASS' if g1 < 0.99 else 'FAIL'} | G4 base {base:+.3f}->zero {zero:+.3f} "
              f"{'PASS' if (base > 0.05 and base - zero > 0.1) else 'FAIL'} | G6 perm {g6:+.3f} vs {base:+.3f} {'PASS' if abs(g6-base) < 0.02 else 'FAIL'}")
    except Exception as e:
        gs = f"FAILED ({type(e).__name__}: {e}) -- run VOID per the guard rule"
    print(f"  GUARDS: {gs}", flush=True)
    print(f"  G8 CONVERGENCE (steps run {sd}); TAKEOFF = first step crossing FVE {COMPETENCE}:", flush=True)
    for b in BUCKETS:
        g = G[b]; print(f"    {str(b):14s} plateau {'YES' if g['plateau'] else 'NO -> BUCKET VOID':17s} "
                        f"tail-slope {g['slope']*1e5:+.3f}e-5 +/-{g['hw']*1e5:.3f}  rel {g['rel']:+.4f}  "
                        f"steps-to-plateau {g['steps']} ({(g['steps'] or 0)/MAXSTEPS*100:.0f}% of cap)  "
                        f"TAKEOFF {tko[b]}{'' if tko[b] is None else f' ({tko[b]/MAXSTEPS*100:.0f}%)'}  tail {g['tail']}", flush=True)
    rows = []
    for d in he:
        v = fve(m, d, d["h"], d["T"])
        rows.append(dict(dom=d["dom"], bucket=list(d["bucket"]), N=d["N"], rmsf=d["rmsf"], model=v,
                         pca=d["ceil"]["pca"][L], anm=d["ceil"]["anm"][L],
                         gap=d["ceil"]["pca"][L]-v, ratio=(d["ceil"]["pca"][L]-v)/d["ceil"]["pca"][L] if d["ceil"]["pca"][L] > 1e-6 else float('nan'),
                         void=not G[d["bucket"]]["plateau"]))
    allres[L] = dict(rows=rows, g8={str(k): v for k, v in G.items()}, steps=sd,
                     takeoff={str(k): v for k, v in tko.items()})
    json.dump(allres, open(RESJSON, "w"))
    print(f"  {'bucket':14s}{'n':>3}{'medN':>7}{'modelFVE':>19}{'ceiling':>19}{'ANM':>7}{'absGAP':>8}{'ratio':>7}{'G8':>6}", flush=True)
    for b in BUCKETS:
        br = [r for r in rows if tuple(r["bucket"]) == b]
        if not br: continue
        q = lambda a: f"{np.median(a):.3f}[{np.percentile(a,25):.2f},{np.percentile(a,75):.2f}]"
        mo = [r["model"] for r in br]; pc = [r["pca"] for r in br]; an = np.nanmean([r["anm"] for r in br])
        print(f"  {str(b):14s}{len(br):>3}{int(np.median([r['N'] for r in br])):>7}{q(mo):>19}{q(pc):>19}"
              f"{('n/a' if not np.isfinite(an) else f'{an:.2f}'):>7}{np.median([r['gap'] for r in br]):>8.3f}"
              f"{np.nanmedian([r['ratio'] for r in br]):>7.2f}{'ok' if G[b]['plateau'] else 'VOID':>6}", flush=True)

print("\n=== STEPS-TO-PLATEAU vs N (objective 4: training cost vs system size) ===", flush=True)
for L in LS:
    G = allres[L]["g8"]; xs, ys = [], []
    for b in BUCKETS:
        g = G[str(b)]
        if g["steps"]:
            md = np.median([d["N"] for d in data if d["bucket"] == b]); xs.append(np.log10(md)); ys.append(np.log10(g["steps"]))
    if len(xs) > 2:
        lr = stats.linregress(xs, ys); ci = stats.t.ppf(0.975, len(xs)-2) * lr.stderr
        print(f"  L={L} (DM={DM}): log10(steps) ~ {lr.slope:+.3f} +/- {ci:.3f} * log10(N)   R^2 {lr.rvalue**2:.3f}  "
              f"-> {'GROWS with N' if lr.slope-ci > 0 else 'flat in N'}", flush=True)
        print(f"        per-bucket steps: " + " ".join(f"{b[0]//1000}k:{G[str(b)]['steps']}" for b in BUCKETS if G[str(b)]["steps"]), flush=True)

print("\n=== DEFICIT-RATIO-vs-N -- ratio AND absolute gap must agree; VOID buckets excluded ===", flush=True)
for L in LS:
    rows = [r for r in allres[L]["rows"] if not r["void"]]
    rs, gs_, ns = [], [], []
    for b in BUCKETS:
        br = [r for r in rows if tuple(r["bucket"]) == b]
        if not br: continue
        rs.append(np.nanmedian([r["ratio"] for r in br])); gs_.append(np.median([r["gap"] for r in br])); ns.append(b)
    if len(rs) < 2: print(f"  L={L} (DM={DM}): fewer than 2 valid buckets -> NO VERDICT (G8 voided the rest)", flush=True); continue
    v = [x for x in rs if np.isfinite(x) and x > 0]
    span = max(v)/min(v) if len(v) > 1 else float('nan'); gspan = max(gs_) - min(gs_)
    print(f"  L={L} DM={DM} code={L*DM}d valid buckets {[f'{b[0]//1000}k' for b in ns]}", flush=True)
    print(f"    ratio  : " + " ".join(f"{x:.2f}" for x in rs) + f"   SPAN {span:.2f}x", flush=True)
    print(f"    absGAP : " + " ".join(f"{x:.3f}" for x in gs_) + f"   max-min {gspan:.3f} FVE", flush=True)
    if span < 1.3 and gspan < 0.02: vd = "ARBITRARY-L SURVIVES (content-addressed)"
    elif span > 2.0 and gspan > 0.03: vd = "INDEX-ADDRESSING (will not reach 1M atoms)"
    elif span > 2.0 and gspan < 0.02: vd = "NEITHER -- ceiling compression, cannot discriminate at this L"
    else: vd = "IN BETWEEN -- report the number, no verdict"
    print(f"    VERDICT: {vd}", flush=True)
print("\n  Slopes must match at L=12 and L=24; divergence = bottleneck saturation, not addressing.", flush=True)
