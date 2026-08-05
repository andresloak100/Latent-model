"""Phase 1: atom-native Perceiver AE, deficit-ratio-vs-N scaling on MISATO. L in {12, 24} (two L
values guard the low-L bottleneck-saturation confound). N atom tokens -> L learned latent slots ->
N atom outputs, L independent of N. Task: displacement from the Kabsch-aligned reference (NOT
absolute coords). Static per-atom features (element + reference position) condition encoder AND
decoder; DISPLACEMENT enters only through the L slots.

N-AXIS CONVENTION: ALL-ATOM (including hydrogens) throughout -- model input, PCA ceiling, ANM
baseline, and the N-axis are counted identically (what the model consumes; what the 1M-atom
objective counts). MISATO all-atom spans 717-26556 (median 5149). Empirical per-complex whitening.
Shared model across six buckets (700-1k/1-2k/2-4k/4-8k/8-16k/16-32k), held-out complexes,
temporal split (80/20). PERB=25 on the top two buckets (power where the verdict rests), 12 below.

Per bucket: model FVE at L, per-system PCA-L ceiling (temporal split, held-out), ANM-L baseline
(all-atom; skipped >10k atoms for eigensolver cost), DEFICIT RATIO = (ceiling-model)/ceiling. All
FVE are train-mean-centered and identical across model/PCA/ANM. Pre-registered: ratio changes <1.3x
across N -> content-addressed, arbitrary-L survives; >2x -> index-addressing, report as the answer;
slope must match at L=12 and L=24 (else bottleneck saturation). GUARDS: G1 frame-variance,
G4 zero-latent, G6 permutation, G7 rank (L/79 = 15%/30%, clean)."""
import h5py, numpy as np, torch, torch.nn as nn, time
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
BUCKETS = [(700, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 32000)]
PERB = {(700, 1000): 12, (1000, 2000): 12, (2000, 4000): 12, (4000, 8000): 12, (8000, 16000): 25, (16000, 32000): 25}
LS = [12, 24]; DM = 64; EPOCHS = 400; ANM_MAXN = 10000; np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ELEMS = [1, 6, 7, 8, 16, 15, 9, 17]                             # all-atom: H included


def fbud(N): return int(max(4, min(80, 300000 // max(N, 1))))   # frames/chunk, bounds GPU memory at large N


def kabsch_traj(traj):
    ref = traj[0]; rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def pca_fve(train, ho, mu, L):
    _, _, Vt = np.linalg.svd(train - mu, full_matrices=False); V = Vt[:L]
    rec = mu + (ho - mu) @ V.T @ V; return 1 - ((ho - rec) ** 2).sum() / (((ho - mu) ** 2).sum() + 1e-12)


def anm_fve(ref, ho, mu, L, cutoff=10.0):                       # all-atom sparse ANM, L slowest modes, held-out FVE
    n = len(ref)
    if n > ANM_MAXN: return float('nan')                        # eigensolver cost guard (report NaN, honest)
    tree = cKDTree(ref); pairs = tree.query_pairs(cutoff, output_type='ndarray')
    if len(pairs) == 0: return float('nan')
    d = ref[pairs[:, 1]] - ref[pairs[:, 0]]; dn = np.linalg.norm(d, axis=1) + 1e-9; u = d / dn[:, None]
    rows, cols, vals = [], [], []
    for (i, j), uu in zip(pairs, u):
        b = np.outer(uu, uu)
        for A, Bc, s in [(i, j, -1), (j, i, -1), (i, i, 1), (j, j, 1)]:
            for a in range(3):
                for c in range(3):
                    rows.append(3*A+a); cols.append(3*Bc+c); vals.append(s*b[a, c])
    H = coo_matrix((vals, (rows, cols)), shape=(3*n, 3*n)).tocsr()
    try:
        w, Vm = eigsh(H, k=L+6, sigma=1e-6, which='LM'); idx = np.argsort(w); M = Vm[:, idx][:, 6:6+L]
    except Exception:
        return float('nan')
    rec = mu + (ho - mu) @ M @ M.T; return 1 - ((ho - rec) ** 2).sum() / (((ho - mu) ** 2).sum() + 1e-12)


def load():
    f = h5py.File(MD, "r"); allk = list(f.keys()); picked = {b: [] for b in BUCKETS}; data = []
    for dom in allk[::2]:
        g = f[dom]; el = np.array(g["atoms_element"]); N = len(el)                # ALL-ATOM count
        b = next((b for b in BUCKETS if b[0] <= N < b[1] and len(picked[b]) < PERB[b]), None)
        if b is None: continue
        co = np.array(g["trajectory_coordinates"]).astype(np.float64)             # all atoms incl H
        eln = np.array(g["atoms_number"]); al = kabsch_traj(co); ref = al[0]
        disp = (al - ref); T = len(disp); h = 80
        scale = np.sqrt((disp[:h] ** 2).sum(-1).mean()) + 1e-6                    # empirical per-complex whitening
        dw = (disp / scale).astype(np.float32)
        oh = np.zeros((N, len(ELEMS)+1), np.float32)
        for a in range(N): oh[a, ELEMS.index(int(eln[a])) if int(eln[a]) in ELEMS else -1] = 1.0
        rp = ((ref - ref.mean(0)) / (ref.std() + 1e-6)).astype(np.float32)
        stat = np.concatenate([oh, rp], 1)
        picked[b].append(dom)
        data.append(dict(dom=dom, bucket=b, N=N, dw=dw, stat=stat, h=h, T=T, ref=ref,
                         disp=disp.astype(np.float64), scale=scale))
        if all(len(picked[b]) >= PERB[b] for b in BUCKETS): break
    return data


class Perceiver(nn.Module):
    def __init__(self, Fs, L, dm=DM, heads=4):
        super().__init__()
        self.lat = nn.Parameter(torch.randn(L, dm) * 0.02)
        self.in_tok = nn.Linear(Fs + 3, dm); self.q_tok = nn.Linear(Fs, dm)
        self.enc = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.self_at = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.dec = nn.MultiheadAttention(dm, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dm); self.ln2 = nn.LayerNorm(dm); self.ln3 = nn.LayerNorm(dm)
        self.ff = nn.Sequential(nn.Linear(dm, dm*2), nn.GELU(), nn.Linear(dm*2, dm)); self.out = nn.Linear(dm, 3)
    def encode(self, stat, disp):
        B = stat.shape[0]; tok = self.in_tok(torch.cat([stat, disp], -1)); lat = self.lat.unsqueeze(0).expand(B, -1, -1)
        lat = self.ln1(lat + self.enc(lat, tok, tok)[0]); lat = self.ln2(lat + self.self_at(lat, lat, lat)[0])
        return self.ln3(lat + self.ff(lat))
    def decode(self, stat, lat): return self.out(self.dec(self.q_tok(stat), lat, lat)[0])
    def forward(self, stat, disp): return self.decode(stat, self.encode(stat, disp))


def model_fve(m, d, lo, hi):                                    # train-mean-centered FVE over frames [lo,hi), chunked
    N = d["N"]; fb = fbud(N); mu = d["disp"][:d["h"]].mean(0).reshape(1, N, 3)
    stat0 = torch.tensor(d["stat"], device=dev); sse = sst = 0.0
    for s in range(lo, hi, fb):
        e = min(s+fb, hi); dw = torch.tensor(d["dw"][s:e], device=dev)
        st = stat0.unsqueeze(0).expand(e-s, -1, -1)
        with torch.no_grad(): pred = (m(st, dw).cpu().numpy()) * d["scale"]
        true = d["disp"][s:e]; sse += ((true - pred) ** 2).sum(); sst += ((true - mu) ** 2).sum()
    return 1 - sse / (sst + 1e-12)


def train_eval(data, L):
    tr = [d for i, d in enumerate(data) if i % 4 != 0]; he = [d for i, d in enumerate(data) if i % 4 == 0]
    Fs = data[0]["stat"].shape[1]; m = Perceiver(Fs, L).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(EPOCHS):
        d = tr[np.random.randint(len(tr))]; h = d["h"]; fb = fbud(d["N"]); idx = np.random.randint(0, h, fb)
        stat = torch.tensor(d["stat"], device=dev).unsqueeze(0).expand(fb, -1, -1)
        disp = torch.tensor(d["dw"][idx], device=dev)
        opt.zero_grad(); pred = m(stat, disp); (((pred-disp)**2).mean()).backward(); opt.step()
    m.eval(); rows = []; tr_fve = np.mean([model_fve(m, d, 0, d["h"]) for d in tr[:8]])   # train-FVE adequacy check
    for d in he:
        h = d["h"]; mo = model_fve(m, d, h, d["T"])
        trd = d["disp"][:h].reshape(h, -1); hod = d["disp"][h:].reshape(d["T"]-h, -1); mu = trd.mean(0)
        pca = pca_fve(trd, hod, mu, L); anm = anm_fve(d["ref"], hod, mu, L)
        rows.append(dict(dom=d["dom"], bucket=d["bucket"], N=d["N"], model=mo, pca=pca, anm=anm))
    return m, tr, he, rows, tr_fve


def guards(m, data, L):
    d = data[0]; h = d["h"]; N = d["N"]; fb = fbud(N)
    stat0 = torch.tensor(d["stat"], device=dev); dwh = torch.tensor(d["dw"][h:h+fb], device=dev)
    st = stat0.unsqueeze(0).expand(fb, -1, -1)
    with torch.no_grad():
        lat = m.encode(st, dwh); lf = lat.reshape(fb, -1); lf = lf / (lf.norm(dim=1, keepdim=True)+1e-9)
        g1 = (lf[:-1]*lf[1:]).sum(1).mean().item()
        base = 1 - (((m.decode(st, lat)-dwh)**2).sum() / ((dwh**2).sum()+1e-9)).item()
        zero = 1 - (((m.decode(st, torch.zeros_like(lat))-dwh)**2).sum() / ((dwh**2).sum()+1e-9)).item()
        perm = torch.randperm(N, device=dev)
        pp = m(st[:, perm, :], dwh[:, perm, :])[:, torch.argsort(perm), :]
        g6 = 1 - (((pp-dwh)**2).sum() / ((dwh**2).sum()+1e-9)).item()
    return dict(g1=g1, base=base, zero=zero, g6=g6)


print(f"[phase1-misato] device={dev}; ALL-ATOM convention; loading ...", flush=True); t0 = time.time()
data = load()
print(f"  {len(data)} complexes ({time.time()-t0:.0f}s). PER-BUCKET COUNTS:", flush=True)
for b in BUCKETS:
    cs = [d for d in data if d["bucket"] == b]
    print(f"    {str(b):14s} n={len(cs):>3}  medN={int(np.median([d['N'] for d in cs])) if cs else 0}", flush=True)
allrows = {}
for L in LS:
    m, tr, he, rows, trf = train_eval(data, L); g = guards(m, data, L); allrows[L] = rows
    print(f"\n=== L={L} (G7: {L}/79 = {L/79*100:.0f}% of usable rank, clean) | train-FVE {trf:.2f} (adequacy) ===", flush=True)
    print(f"  GUARDS: G1 frame-cos {g['g1']:.3f} (<0.99 pass) | G4 base {g['base']:.2f}->zero {g['zero']:.2f} (must collapse) | "
          f"G6 perm {g['g6']:.2f} vs base {g['base']:.2f} (must match)", flush=True)
    print(f"  {'bucket':14s}{'nsys':>5}{'medN':>7}{'model':>7}{'PCA':>6}{'ANM':>6}{'deficit':>9}", flush=True)
    for b in BUCKETS:
        br = [r for r in rows if r["bucket"] == b]
        if not br: continue
        mo = np.mean([r["model"] for r in br]); pc = np.mean([r["pca"] for r in br]); an = np.nanmean([r["anm"] for r in br])
        dr = (pc-mo)/pc if pc > 0 else float('nan')
        print(f"  {str(b):14s}{len(br):>5}{int(np.median([r['N'] for r in br])):>7}{mo:>7.2f}{pc:>6.2f}{an:>6.2f}{dr:>9.2f}", flush=True)

print("\n=== DEFICIT-RATIO-vs-N (the experiment) ===", flush=True)
for L in LS:
    rows = allrows[L]; drs = []
    for b in BUCKETS:
        br = [r for r in rows if r["bucket"] == b]
        if not br: continue
        mo = np.mean([r["model"] for r in br]); pc = np.mean([r["pca"] for r in br]); drs.append((pc-mo)/pc if pc > 0 else np.nan)
    v = [x for x in drs if np.isfinite(x) and x > 0]; span = max(v)/min(v) if len(v) > 1 else float('nan')
    print(f"  L={L}: " + " ".join(f"{x:.2f}" for x in drs) + f"  | span {span:.2f}x", flush=True)
print("  read: span <1.3x -> content-addressed, arbitrary-L survives. >2x -> index-addressing. slopes must match across L.", flush=True)
