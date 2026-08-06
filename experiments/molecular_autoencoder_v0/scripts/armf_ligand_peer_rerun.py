"""RE-RUN of the project's one prior peer win, against baselines actually made strong.

Claim on record: "the learned graph codec beats Cartesian ANM on ligands" (held-out RMSD 0.64 A vs
0.74 A at L=8). That used a SINGLE UNSWEPT 8 A cutoff on molecules of 8-90 heavy atoms spanning
~10 A -- near-complete connectivity. FAMILY E: a comparator handicapped by an unswept
hyperparameter, so the comparison could not fail and therefore proved nothing.

 1. ANM cutoff SWEPT over {3,4,5,6,8} A, selected on TRAINING molecules only by mean training RMSD,
    winner applied unchanged to held-out; all curves reported. ANM-8 kept as the labelled CONTINUITY
    baseline, never pooled.
 2. SECOND PEER: bond-graph ENM -- a distance cutoff may be the wrong network at this scale. Springs
    on bonded pairs plus 1-3 neighbours, 1-3 weight also swept on training data. MISATO stores NO
    bond orders (bonds perceived by distance <1.8 A), so weighting is TOPOLOGICAL rather than by bond
    order -- substitution stated, not hidden.
 3. FAMILY B, never applied to this arm: an 8-atom molecule has only 3N-6 = 18 modes, so L=8 is 44%
    of available rank. L/(3N-6) printed per molecule, flagged past 30%.

THE BAR: beating the BEST of {swept ANM, bond-graph ENM} is citable; beating only ANM-8 is not.
Protocol matched to armf_graph_codec.py: same cache, same i%4 split, same centred-displacement
projection, same Kabsch RMSD on the second half of frames."""
import pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
CACHE = f"{WR}/graph_codec_cache.pkl"
LS = [4, 8, 16]
CUTOFFS = [3.0, 4.0, 5.0, 6.0, 8.0]
CONTINUITY = 8.0
W13 = [0.0, 0.1, 0.3, 0.6, 1.0]
CODEC = {4: 0.80, 8: 0.64, 16: 0.44}


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc); d = np.sign(np.linalg.det(Vt.T @ U.T))
    return np.sqrt((((Pc @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T) - Qc) ** 2).sum() / len(P))


def hess_from_pairs(ref, pairs, wts):
    n = len(ref); H = np.zeros((3*n, 3*n))
    for (i, j), w in zip(pairs, wts):
        d = ref[j] - ref[i]; L = np.linalg.norm(d)
        if L < 1e-9: continue
        u = d / L; b = w * np.outer(u, u)
        H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
        H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    return H


def anm_pairs(ref, cutoff):
    pr = cKDTree(ref).query_pairs(cutoff, output_type='ndarray')
    return [tuple(p) for p in pr], np.ones(len(pr))


def bondgraph_pairs(ref, edges, w13):
    n = len(ref); adj = [set() for _ in range(n)]
    for a, b in zip(edges[0], edges[1]):
        if a != b: adj[a].add(b); adj[b].add(a)
    p12 = {(min(a, b), max(a, b)) for a in range(n) for b in adj[a]}
    p13 = set()
    for a in range(n):
        for b in adj[a]:
            for c in adj[b]:
                if c != a and (min(a, c), max(a, c)) not in p12: p13.add((min(a, c), max(a, c)))
    pairs = sorted(p12) + sorted(p13)
    wts = np.concatenate([np.ones(len(p12)), np.full(len(p13), w13)])
    return pairs, wts


def rmsd_for(d, pairs, wts, L):
    co = d["co"].astype(np.float64); T = len(co); h = T // 2; n = co.shape[1]
    if not pairs: return np.nan
    H = hess_from_pairs(co[0], pairs, wts)
    w, V = np.linalg.eigh(H); M = V[:, 6:6+L]
    if M.shape[1] < L: return np.nan
    dc = (co - co[0]).reshape(T, -1); mc = dc[:h].mean(0)
    rec = (dc[h:] - mc) @ M @ M.T
    return float(np.mean([kabsch_rmsd(co[0] + (mc + rec[k]).reshape(n, 3), co[h+k]) for k in range(T-h)]))


data = pickle.load(open(CACHE, "rb"))
tr = [d for i, d in enumerate(data) if i % 4 != 0]
he = [d for i, d in enumerate(data) if i % 4 == 0]
print(f"[ligand-peer-rerun] {len(data)} ligands: {len(tr)} train / {len(he)} held-out", flush=True)

print("\n=== FAMILY B: is L past 30% of available rank? (3N-6 non-trivial modes) ===", flush=True)
nn = np.array([d["n"] for d in he])
print(f"  held-out heavy atoms: min {nn.min()} median {int(np.median(nn))} max {nn.max()}", flush=True)
for L in LS:
    fr = L / (3*nn - 6)
    print(f"  L={L:>2}: rank usage median {np.median(fr)*100:>5.1f}%  worst {fr.max()*100:>5.1f}%  "
          f"past 30%: {int((fr > 0.30).sum())}/{len(nn)}"
          + ("   <-- BOUND for those" if (fr > 0.30).any() else ""), flush=True)

print("\n=== PEER 1: ANM cutoff SWEPT on TRAINING molecules (held-out never seen) ===", flush=True)
print(f"  {'cutoff':>8}" + "".join(f"{('train L='+str(L)):>14}" for L in LS), flush=True)
tr_anm = {}
for c in CUTOFFS:
    row = [np.nanmean([rmsd_for(d, *anm_pairs(d["co"][0].astype(np.float64), c), L) for d in tr]) for L in LS]
    tr_anm[c] = row
    print(f"  {c:>8.1f}" + "".join(f"{x:>14.4f}" for x in row), flush=True)
best_anm = {L: min(CUTOFFS, key=lambda c: tr_anm[c][i]) for i, L in enumerate(LS)}
print(f"  SELECTED: " + "  ".join(f"L={L}: {best_anm[L]:.0f} A" for L in LS), flush=True)

print("\n=== PEER 2: BOND-GRAPH ENM (1-2 + 1-3), weight SWEPT on TRAINING molecules ===", flush=True)
print("  NOTE: no bond orders in MISATO -> TOPOLOGICAL weighting (1-2=1.0, 1-3=w), stated not hidden.", flush=True)
print(f"  {'w13':>8}" + "".join(f"{('train L='+str(L)):>14}" for L in LS), flush=True)
tr_bg = {}
for w in W13:
    row = [np.nanmean([rmsd_for(d, *bondgraph_pairs(d["co"][0].astype(np.float64), d["edges"], w), L) for d in tr]) for L in LS]
    tr_bg[w] = row
    print(f"  {w:>8.1f}" + "".join(f"{x:>14.4f}" for x in row), flush=True)
best_bg = {L: min(W13, key=lambda w: tr_bg[w][i]) for i, L in enumerate(LS)}
print(f"  SELECTED: " + "  ".join(f"L={L}: w13={best_bg[L]:.1f}" for L in LS), flush=True)

print("\n=== HELD-OUT (training-selected settings applied unchanged); lower RMSD is better ===", flush=True)
print(f"  {'L':>3}{'codec':>9}{'ANMswept':>11}{'(cut)':>7}{'bondgraph':>11}{'(w13)':>7}"
      f"{'ANM-8cont':>11}{'VERDICT':>30}", flush=True)
for L in LS:
    a = np.nanmean([rmsd_for(d, *anm_pairs(d["co"][0].astype(np.float64), best_anm[L]), L) for d in he])
    b = np.nanmean([rmsd_for(d, *bondgraph_pairs(d["co"][0].astype(np.float64), d["edges"], best_bg[L]), L) for d in he])
    c8 = np.nanmean([rmsd_for(d, *anm_pairs(d["co"][0].astype(np.float64), CONTINUITY), L) for d in he])
    bp = min(a, b); cd = CODEC[L]
    v = "CODEC WINS vs best peer" if cd < bp else f"CODEC LOSES to {'ANM-'+str(int(best_anm[L])) if a<b else 'bondgraph'}"
    print(f"  {L:>3}{cd:>9.3f}{a:>11.3f}{best_anm[L]:>7.0f}{b:>11.3f}{best_bg[L]:>7.1f}{c8:>11.3f}{v:>30}", flush=True)
print("\n  BAR: beating the BEST of {swept ANM, bond-graph ENM} is citable; beating only ANM-8 is not.", flush=True)
