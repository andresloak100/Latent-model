"""INBOX 30a: the ZERO-SHOT N-only correction, reported beside the oracle upper bound.

30a names a hole I opened in 29b. `a* = <r,d>/<r,r>` is fitted AGAINST THE HELD-OUT TARGET, so
"FVE after optimal rescaling" is an ORACLE quantity in exactly the sense the oracle-fraction
secondary is permanently caveated for. Put in a table beside honest FVE values it will be read as
performance -- including by us.

THE LEGITIMATE COUNTERPART: N IS AN INPUT. A correction that is a fixed function of N alone,
    c(N) = sqrt(N_ref / N),   N_ref FROZEN from the TRAINING distribution,
uses nothing from the target and is available ZERO-SHOT on an unseen system. If the sqrt(N)
mechanism is real it should recover most of what the oracle rescale recovers, and THAT number is
reportable. The gap between them is the part of the scale error N does not explain.

NO RE-RUN NEEDED. With u = <r,d>/<d,d> and v = <r,r>/<d,d>, FVE at any fixed scale c is
    FVE(c) = 2*c*u - c^2*v
and the stored fields give u = cos2/astar, v = cos2/astar^2. Verified against the stored values:
FVE(1) reproduces the raw FVE to 2.8e-16 and FVE(a*) reproduces cos^2 to 5.6e-17.

FIELD NAMES ARE THE LABEL (30a): `fve_oracle_rescaled`, never `fve_corrected`. No model achieves it."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import stats
from armf_atlas_data import AtlasStore
import armf_atlas_dm as D

WR = D.WR
NTR = 50

man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
have = {m["pdb"]: i for i, m in enumerate(store.meta)}
tr_ids = [p for p in man["train_ordered"] if p in have][:NTR]
# N_ref FROZEN FROM THE TRAINING DISTRIBUTION -- nothing from any target system enters it.
N_ref = float(np.median([store.meta[have[p]]["atoms"] for p in tr_ids]))
res = json.load(open(f"{WR}/scale_test.json"))
allN = np.array(sorted(v["N"] for v in res["tied"].values()), float)
q = np.quantile(allN, [0.25, 0.5, 0.75])
bands = [(0, q[0], "Q1"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1e18, "Q4")]

print(f"[n-scale] INBOX 30a: N_ref = median N of the {len(tr_ids)} TRAINING systems = {N_ref:.0f}")
print(f"  c(N) = sqrt(N_ref/N); uses N only, so it is available ZERO-SHOT on an unseen system.\n")
print(f"    {'arm':>9}{'band':>5}{'n':>4}{'raw FVE':>11}{'N-only (REPORTABLE)':>22}"
      f"{'oracle (UPPER BOUND)':>23}{'N-only recovers':>17}")
for kind in ("tied", "control", "untied"):
    if kind not in res: continue
    v = res[kind]
    N = np.array([r["N"] for r in v.values()], float)
    f = np.array([r["fve"] for r in v.values()], float)
    c2 = np.array([r["cos2"] for r in v.values()], float)
    a = np.array([r["astar"] for r in v.values()], float)
    u = c2 / a; vv = c2 / a**2
    c = np.sqrt(N_ref / N)
    f_n = 2 * c * u - c**2 * vv          # zero-shot, N only
    for lo, hi, lab in bands:
        m = (N >= lo) & (N < hi)
        if m.sum() < 3: continue
        r0, rn, ro = np.median(f[m]), np.median(f_n[m]), np.median(c2[m])
        rec = (rn - r0) / (ro - r0) if abs(ro - r0) > 1e-9 else float("nan")
        print(f"    {kind:>9}{lab:>5}{int(m.sum()):>4}{r0:>+11.4f}{rn:>+22.4f}{ro:>+23.4f}"
              f"{100*rec:>16.0f}%")
print(f"\n  'N-only recovers' = (N-only - raw) / (oracle - raw): the share of the recoverable scale")
print(f"  error that a FIXED FUNCTION OF N alone captures. 100% would mean N explains all of it.")
print(f"  The oracle column is an UPPER BOUND that NO MODEL ACHIEVES -- it is fitted against the")
print(f"  held-out target, and is printed only to bound the N-only column.")
