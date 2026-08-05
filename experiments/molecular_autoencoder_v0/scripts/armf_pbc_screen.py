"""MISATO artifact screen + censored intrinsic-dimensionality exponent.

SCREEN: the two high-displacement signatures are different failures and need separating.
  1CBR: medRMSF 5.05A (whole structure inflated)  -> genuinely floppy, KEEP
  1IL3: medRMSF 1.12A with a fat tail             -> few atoms far = unwrapped PBC, DROP
Decisive test: MAX PER-ATOM DISPLACEMENT BETWEEN CONSECUTIVE FRAMES. Real dynamics cannot move an
atom more than ~2-3A in one saved interval; a PBC wrap is a single box-sized jump. Smooth
accumulation = floppy (keep); discontinuous jump = unwrapped (drop).
Drop counts are reported PER BUCKET: if exclusions correlate with N they bias the trend; if uniform
they are only noise.

FIT: rank90 vs N per-system (not bucket means) with 95% CI -- the MISATO (censored) exponent to
compare against mdCATH's uncensored one."""
import h5py, numpy as np, time
from scipy import stats
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
BUCKETS = [(700, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 32000)]
PERB = 8; JUMP = 10.0; np.random.seed(0)


def kabsch_traj(traj):
    ref = traj[0]; rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


f = h5py.File(MD, "r"); picked = {b: [] for b in BUCKETS}; rows = []; t0 = time.time()
print(f"[pbc-screen] MISATO, ALL-ATOM. consecutive-frame jump test (>{JUMP}A = unwrapped PBC)", flush=True)
print(f"  {'dom':8s}{'N':>7}{'bucket':>15}{'maxConsec':>10}{'maxTotal':>9}{'medRMSF':>9}{'rank90':>7}  class", flush=True)
for dom in list(f.keys())[::2]:
    g = f[dom]; N = g["atoms_element"].shape[0]
    b = next((b for b in BUCKETS if b[0] <= N < b[1] and len(picked[b]) < PERB), None)
    if b is None: continue
    co = np.array(g["trajectory_coordinates"]).astype(np.float64)
    al = kabsch_traj(co); disp = al - al[0]
    cons = float(np.linalg.norm(np.diff(al, axis=0), axis=-1).max())     # decisive test
    mag = np.linalg.norm(disp, axis=-1); mx = float(mag.max()); rmsf = float(np.median(np.sqrt((mag**2).mean(0))))
    d = disp[:80].reshape(80, -1); mu = d.mean(0)
    S = np.linalg.svd(d - mu, compute_uv=False); cum = np.cumsum(S**2)/((S**2).sum()+1e-12)
    r90 = int(np.searchsorted(cum, 0.90) + 1)
    unwrapped = cons > JUMP
    cls = "UNWRAPPED-drop" if unwrapped else ("floppy-keep" if rmsf > 3.0 else "clean")
    picked[b].append(dom); rows.append(dict(dom=dom, N=N, bucket=b, cons=cons, mx=mx, rmsf=rmsf, r90=r90, drop=unwrapped))
    print(f"  {dom:8s}{N:>7}{str(b):>15}{cons:>10.1f}{mx:>9.1f}{rmsf:>9.2f}{r90:>7}  {cls}", flush=True)
    if all(len(picked[bb]) >= PERB for bb in BUCKETS): break
print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== DROP COUNTS PER BUCKET (do exclusions correlate with N?) ===", flush=True)
for b in BUCKETS:
    br = [r for r in rows if r["bucket"] == b]
    if not br: continue
    nd = sum(r["drop"] for r in br); nf_ = sum(1 for r in br if (not r["drop"]) and r["rmsf"] > 3.0)
    print(f"  {str(b):15s} n={len(br):>3}  dropped(unwrapped) {nd}/{len(br)} ({nd/len(br)*100:>3.0f}%)  floppy-kept {nf_}", flush=True)
dc = [(np.median([r["N"] for r in rows if r["bucket"] == b]), sum(r["drop"] for r in rows if r["bucket"] == b)/max(1, len([r for r in rows if r["bucket"] == b]))) for b in BUCKETS if any(r["bucket"] == b for r in rows)]
if len(dc) > 2:
    rr = stats.linregress(np.log10([x for x, _ in dc]), [y for _, y in dc])
    print(f"  drop-rate vs log10(N): slope {rr.slope:+.3f}, p={rr.pvalue:.2f} -> "
          f"{'CORRELATED WITH N (biases the trend)' if rr.pvalue < 0.05 else 'uniform in N (noise only)'}", flush=True)

clean = [r for r in rows if not r["drop"]]
print(f"\n=== CENSORED EXPONENT (MISATO, 79 frames), clean n={len(clean)} ===", flush=True)
x = np.log10([r["N"] for r in clean]); y = np.log10([r["r90"] for r in clean])
r = stats.linregress(x, y); ci = stats.t.ppf(0.975, len(x)-2)*r.stderr
print(f"  rank90 ~ N^b :  b = {r.slope:.3f} +/- {ci:.3f}   R^2 {r.rvalue**2:.3f}   "
      f"rank90 range {min(rr_['r90'] for rr_ in clean)}-{max(rr_['r90'] for rr_ in clean)} of 79 available", flush=True)
a = np.median(y - r.slope*x)
print(f"  1e6-atom extrapolation: rank90 ~ {10**(a+r.slope*6):.0f}  [95% CI {10**(a+(r.slope-ci)*6):.0f} - {10**(a+(r.slope+ci)*6):.0f}]", flush=True)
print(f"  CENSORING: max rank90 is {max(rr_['r90'] for rr_ in clean)/79*100:.0f}% of the 79-frame cap -> this b is a LOWER BOUND.", flush=True)
