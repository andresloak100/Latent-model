"""mdCATH matched-window tier table (Part 1). Within ONE trajectory (320 K,
replica 0), compare a SHORT window (A, ~50 ns) at multiple offsets to a LONG
window (B, ~500 ns), everything else matched. Leak-free split PCA (fit first half,
eval second) for CA / backbone / all-atom / side-chain at L=8/16/32. DECISION IS ON
ABSOLUTE ANGSTROM (B's null >> A's); report both % and A and both nulls."""
import glob, sys, numpy as np
from pathlib import Path
import h5py
DATA = "/network/scratch/j/jacob-junqi.tian/datasets/mdcath/data"
Ls = [8, 16, 32]
WINDOWS = {"A@0": list(range(0, 50)), "A@200": list(range(200, 250)),
           "A@400": list(range(400, 450)), "B@0-499": list(np.linspace(0, 499, 50).astype(int)),
           "C@all": list(range(500))}
# C = full trajectory (1 ns stride, 250/250 split, rank<=249) breaks the span/stride
# coupling (same span as B, same stride as A) AND gives the leak-free L=64 test.
# A/B are 50 frames -> 25 fit -> rank<=24, so their "L=32" is rank-capped at 24
# (matched across A and B, so the A-vs-B comparison stays fair; L=8/16 are clean).
WL = {"C@all": [8, 16, 32, 64]}


def parse_names(g, N):
    pp = g["pdbProteinAtoms"][()]; pp = pp.decode() if isinstance(pp, bytes) else str(pp)
    atoms = [ln for ln in pp.splitlines() if ln.startswith(("ATOM", "HETATM"))][:N]
    return np.array([ln[12:16].strip() for ln in atoms])


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pcen = P[mask].mean(0); Pc = P[mask] - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q); d = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pcen) @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T + rcen
    return out


def split_pca(d, L):
    T = d.shape[0]; h = T // 2; npera = d.shape[1]
    X = d.reshape(T, -1); tr, ev = X[:h], X[h:]; mean = tr.mean(0)
    _, _, Vt = np.linalg.svd(tr - mean, full_matrices=False)
    B = Vt[:min(L, Vt.shape[0])]
    rec = (ev - mean) @ B.T @ B + mean
    resid = np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * npera))
    null = np.sqrt((ev ** 2).sum() / (ev.shape[0] * npera))
    return 100 * (null - resid) / null, resid, null


agg = {}   # (window, set, L) -> {"red":[], "res":[], "null":[]}
def add(w, s, L, r, res, nl):
    agg.setdefault((w, s, L), {"red": [], "res": [], "null": []})
    agg[(w, s, L)]["red"].append(r); agg[(w, s, L)]["res"].append(res); agg[(w, s, L)]["null"].append(nl)

files = sorted(glob.glob(f"{DATA}/*.h5"))
print(f"[mdcath tiers] {len(files)} domains")
for fp in files:
    try:
        f = h5py.File(fp, "r")
    except Exception as e:
        print(f"  skip {Path(fp).name}: {type(e).__name__} (corrupt/partial)"); continue
    with f:
        dom = list(f.keys())[0]; g = f[dom]
        if "320" not in g or "0" not in g["320"]:
            continue
        z = np.array(g["z"]); N = len(z); heavy = z != 1
        names = parse_names(g, N)
        if len(names) != N:
            print(f"  skip {dom}: name/z mismatch"); continue
        nmh = names[heavy]
        sets = {"CA": nmh == "CA", "backbone": np.isin(nmh, ["N", "CA", "C", "O"]),
                "all-atom": np.ones(heavy.sum(), bool), "side-chain": ~np.isin(nmh, ["N", "CA", "C", "O"])}
        ca_align = nmh == "CA"
        if ca_align.sum() < 3:
            continue
        coords = g["320"]["0"]["coords"][:].astype(np.float64)     # (500, N, 3)
        for w, frames in WINDOWS.items():
            sub = coords[frames][:, heavy, :]
            d = kabsch(sub, ca_align)
            d = d - d[0]
            for sname, m in sets.items():
                if m.sum() < 12:
                    continue
                for L in WL.get(w, Ls):
                    r, res, nl = split_pca(d[:, m, :], L)
                    add(w, sname, L, r, res, nl)

g = lambda w, s, L: np.mean(agg[(w, s, L)]["res"])
gr = lambda w, s, L: np.mean(agg[(w, s, L)]["red"])
gn = lambda w, s, L: np.mean(agg[(w, s, L)]["null"])

print("\n=== A/B matched-window tiers (50 frames, 25/25 split; L=32 rank-capped ~24): red% (abs A) ===")
for s in ("all-atom", "CA", "backbone", "side-chain"):
    print(f"\n  {s}:")
    for w in ("A@0", "A@200", "A@400", "B@0-499"):
        if (w, s, 8) not in agg:
            continue
        cells = " ".join(f"L{L}: {gr(w,s,L):2.0f}% ({g(w,s,L):.2f}A)" for L in Ls)
        print(f"    {w:9s} null={gn(w,s,8):.2f}A | {cells}")

print("\n=== A-vs-B DECISION (absolute A at matched L, locked) ===")
for s in ("all-atom", "CA"):
    for L in (16, 32):
        a, b = g("A@0", s, L), g("B@0-499", s, L)
        print(f"  {s:9s} L{L}: A@0 {a:.2f}A (null {gn('A@0',s,L):.2f}) vs B {b:.2f}A (null {gn('B@0-499',s,L):.2f}) "
              f"-> B {'BETTER' if b < a else 'WORSE/EQUAL'}; Bnull/Anull={gn('B@0-499',s,L)/gn('A@0',s,L):.1f}x")
    off = [g(w, s, 16) for w in ("A@0", "A@200", "A@400")]
    print(f"  {s:9s} A-offset spread (L16 resid A, 0/200/400 ns): {off[0]:.2f}/{off[1]:.2f}/{off[2]:.2f}")

print("\n=== WINDOW C (full traj, leak-free L=64) -- reported SEPARATELY, NOT rank-matched to A/B ===")
for s in ("all-atom", "CA", "backbone", "side-chain"):
    if ("C@all", s, 8) not in agg:
        continue
    cells = " ".join(f"L{L}: {gr('C@all',s,L):2.0f}% ({g('C@all',s,L):.2f}A)" for L in [8, 16, 32, 64])
    print(f"  {s:11s} null={gn('C@all',s,8):.2f}A | {cells}")
aa64 = gr("C@all", "all-atom", 64)
print(f"  L=64 all-atom = {aa64:.0f}% vs PREDICTED 43-44%: "
      f"{'MATCHES (saturation holds, L=64 call stands)' if abs(aa64-43.5)<6 else 'DIVERGES -> saturation argument wrong, revisit L=64'}")
