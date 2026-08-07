"""Phase 1 G7 rank-sanity PRE-CHECK (gates the full run). MISATO has exactly 100 frames/complex;
an L=64 PCA ceiling fit on ~80 train frames sits at 64/79 of the frame rank. G7: "if frames < ~200
per system the ceiling is the artifact." Measure it before spending a GPU run.

Per complex (bucketed by HEAVY atoms; total-atom buckets starve the small ones): Kabsch-align heavy
atoms, displacement from frame-0 reference, TEMPORAL split (80 train / 20 held-out, no leak). Report
PCA train vs HELD-OUT fractional variance explained (FVE) at L=8/16/32/64, effective signal rank
(#modes for 90% train var), and the train-vs-heldout gap at L=64. If PCA-64 train FVE ~= 1
(saturated) or held-out FVE is inflated by rank, the L=64 ceiling is void on MISATO."""
import h5py, numpy as np
MD = "/network/scratch/j/jacob-junqi.tian/datasets/misato/MD.hdf5"
BUCKETS = [(500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]; PER = 4; LS = [8, 16, 32, 64]
np.random.seed(0)


def kabsch_traj(traj):                                          # align every frame to frame 0 (all heavy atoms)
    ref = traj[0]; rc = ref.mean(0); Q = ref - rc; out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pc = P.mean(0); Pc = P - pc
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pc) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rc
    return out


def fve(train, ho, L):                                          # top-L train-PCA modes, held-out FVE (temporal split)
    mu = train.mean(0); _, _, Vt = np.linalg.svd(train - mu, full_matrices=False); V = Vt[:L]
    def r(X): rec = mu + (X - mu) @ V.T @ V; return 1 - ((X - rec) ** 2).sum() / (((X - mu) ** 2).sum() + 1e-12)
    return r(train), r(ho)


f = h5py.File(MD, "r"); allk = list(f.keys())
picked = {b: [] for b in BUCKETS}
for dom in allk[::7]:
    g = f[dom]; el = np.array(g["atoms_element"]); nh = int((el != 1).sum())
    for b in BUCKETS:
        if b[0] <= nh < b[1] and len(picked[b]) < PER: picked[b].append(dom)
    if all(len(v) >= PER for v in picked.values()): break

print("[G7 pre-check] frames/complex = 100 (temporal split 80/20). PCA held-out FVE by L:")
print(f"  {'bucket':11s}{'dom':8s}{'nHvy':>6}{'rank90':>7}  " + "  ".join(f"L{L}(tr/ho)" for L in LS) + "   L64gap")
for b in BUCKETS:
    rows = []
    for dom in picked[b]:
        g = f[dom]; el = np.array(g["atoms_element"]); hv = el != 1
        co = np.array(g["trajectory_coordinates"])[:, hv, :].astype(np.float64); nh = int(hv.sum())
        al = kabsch_traj(co); d = (al - al[0]).reshape(len(al), -1); h = 80
        tr, ho = d[:h], d[h:]
        mu = tr.mean(0); s = np.linalg.svd(tr - mu, compute_uv=False); cum = np.cumsum(s**2) / (s**2).sum()
        rank90 = int(np.searchsorted(cum, 0.90) + 1)
        cells = [fve(tr, ho, L) for L in LS]; gap = cells[-1][0] - cells[-1][1]
        rows.append((cells, rank90)); cs = "  ".join(f"{c[0]:.2f}/{c[1]:.2f}" for c in cells)
        print(f"  {str(b):11s}{dom:8s}{nh:>6}{rank90:>7}  {cs}   {gap:+.2f}")
    A = np.array([[c[1] for c in r[0]] for r in rows])          # held-out FVE per L, per complex
    print(f"    -> bucket mean held-out FVE: " + "  ".join(f"L{LS[j]} {A[:,j].mean():.2f}" for j in range(len(LS))) + f"   mean rank90 {np.mean([r[1] for r in rows]):.0f}")

print("\n  G7 verdict guide: PCA-64 train FVE ~= 1.0 AND large train-vs-heldout gap -> rank-saturated,")
print("  ceiling is the artifact (void the L=64 read).")
print("  *** THE SECOND HALF OF THIS GUIDE IS RETIRED (INBOX 002). It used to read 'rank90 << 64 ->")
print("  signal is low-rank, L=64 over-provisioned'. That inference RUNS THE FLOOR BACKWARDS: the")
print("  rank90 printed above is IN-SAMPLE, and an in-sample rank90 is a LOWER BOUND on the true")
print("  dimensionality (out of sample the same modes cover 77.1% of held-out variance, and the")
print("  out-of-sample count is 3.49x larger). A small FLOOR licenses NO conclusion that the signal")
print("  is low-rank, and therefore none that L=64 is over-provisioned. Read the rank90 column as a")
print("  floor and take the width answer from the codec's own saturation curve instead. ***")
