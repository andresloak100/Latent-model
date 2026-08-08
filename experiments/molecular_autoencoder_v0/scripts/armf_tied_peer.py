"""INBOX 28b: ON THE SMALL SYSTEMS, DOES THE TIED ARM BEAT THE ZERO-SHOT PEER?

THE STANDING GAP. The project has no surviving peer-comparison win. 14a measured the *control* codec
losing to ANM on 0% of 123 systems, at every width -- and the peer is ANM, not the internal control.
So "tied beats the control on small systems" (Q1 median +0.2956 vs +0.1108, matched quartiles per
28a) is a win over an internal baseline, and TIED vs ANM IS UNMEASURED AT EVERY N.

    If tied clears ANM on the smallest quartile, that is the project's FIRST PEER WIN.
    If it does not, tied is the best of several architectures that all lose to a zero-cost
    physics baseline -- a much smaller claim, to be worded as one.

BOTH SIDES IN ONE PASS, ON THE SAME FRAMES. The tied FVE and the ANM FVE are computed here, per
system, from the same `ho_frames` with the same `mu` and the same `sst`. ANM's values do already
exist in `atlas_peer.json` and ANM does not depend on any checkpoint -- but the modal artefacts key
by position in the held-out order while the peer keys by pdb, and reconciling two orders is exactly
the Family F join that 27d caught one level up. Recomputing costs ~40 minutes of eigensolves and
removes the question entirely.

MATCHED CAPACITY. The codec at L=1 emits DM=256 numbers per frame; ANM-256 emits 256 coefficients.
Same convention as 14a, and the same caveat: ANM gets a SYSTEM-SPECIFIC basis from that system's own
structure while the codec's 256 numbers come from one model shared across every system. Matched on
numbers-per-frame the peer is the more favoured of the two. Reported anyway, because it is what a
practitioner gets for free.

ASCENDING N, so the Q1 systems -- the ones 28b actually asks about -- land first and the answer is
available before the large-N tail finishes."""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ho_cols
from armf_anm import modes as anm_modes
import armf_atlas_dm as D
import armf_stamp as STAMP
from armf_modal_decoder import ModalCodec

WR = D.WR
CKPT = f"{WR}/modal_arm_ckpt"
# n_train of the checkpoints this script reads. Explicit because the directory now
# holds several (INBOX 43a); override to point at another rung.
NTRAIN = int(os.environ.get("PEER_NTRAIN", "50"))

def ckpt_path(kind, dm, lr, seed, n_train):
    """INBOX 43a/43b. One directory, two naming conventions, no discriminator -- armf_modal_arm.py
    wrote {kind}_dm_lr_s.pt at n_train=50 and the ladder writes ..._n{N}.pt for three rungs. The peer
    script read the n-less name, and a REAL n50 file sat at that path, so the n300 re-run would have
    loaded the n50 model and reported it as the lifted result: the old answer wearing the new run's
    label. Every checkpoint now names its n_train, and a caller that asks for one it cannot have gets
    an exception rather than a neighbour."""
    p = f"{CKPT}/{kind}_dm{dm}_lr{lr:g}_s{seed}_n{n_train}.pt"
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no checkpoint at {p}. Refusing to fall back to any other file -- a checkpoint the "
            f"caller did not name must never be loaded (INBOX 43b). Available: "
            f"{sorted(os.listdir(CKPT)) if os.path.isdir(CKPT) else '<no dir>'}")
    return p

# results are keyed by n_train too, so two rungs cannot share a file even if the stamp were wrong
RES = f"{WR}/tied_peer.json" if int(os.environ.get("PEER_NTRAIN", "50")) == 50 \
      else f"{WR}/tied_peer_n{int(os.environ['PEER_NTRAIN'])}.json"
DM, SEED = 256, 0
ARM_LR = 3e-5          # the tied arm with the best Q1 median (+0.2956), per 28a
CUTOFF = 5.0           # selected on TRAINING systems by armf_atlas_peer.py
CHUNK = 20000
dev = D.dev


def anm_fve(d, V):
    """Streamed ||ho V||^2 / sst -- (F, 3N) is 1.0 GB at N=33,377 and is never materialised."""
    acc = np.zeros((d["F"], V.shape[1]))
    for c0 in range(0, 3 * d["N"], CHUNK):
        c1 = min(c0 + CHUNK, 3 * d["N"])
        acc += ho_cols(d, c0, c1) @ V[c0:c1]
    return float((acc ** 2).sum() / (d["sst"] + 1e-12))


if __name__ == "__main__":
    print(f"[tied-peer] INBOX 28b: does the TIED arm beat zero-shot ANM-{DM}? Same pass, same "
          f"frames, ascending N so Q1 lands first.", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    HO.sort(key=lambda d: d["N"])
    FS = HO[0]["stat"].shape[1]
    print(f"  {len(HO)} held-out systems, N {HO[0]['N']}-{HO[-1]['N']}", flush=True)

    cp = ckpt_path("tied", DM, ARM_LR, SEED, NTRAIN)
    print(f"  loading tied arm trained at n_train={NTRAIN}: {os.path.basename(cp)}", flush=True)
    mdl = ModalCodec(FS, 1, DM, tie_encoder=True).to(dev)
    mdl.load_state_dict(torch.load(cp, map_location=dev)); mdl.eval()

    # INBOX 043 ONE LAYER UP. 043 stopped an n300 run loading the n50 CHECKPOINT. This stamp then let
    # an n300 run reuse the n50 RESULTS: it keyed on dm/lr/seed/cutoff/arm and NOT on n_train, so job
    # 10317061 loaded tied_..._n300.pt correctly, matched all 123 stored n50 systems, computed nothing,
    # and reprinted the n50 numbers under an n300 heading. The checkpoint path and the results key have
    # to carry the same identity or the guard only moves the failure.
    ST = STAMP.stamp(dict(dm=DM, lr=ARM_LR, seed=SEED, cutoff=CUTOFF, arm="tied", n_train=NTRAIN),
                     ModalCodec, D.fve_model)
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    STAMP.report(list(res.values()), ST, "systems")
    res = {k: v for k, v in res.items() if STAMP.same_stamp(v, ST)}

    t0 = time.time()
    for i, d in enumerate(HO):
        if d["pdb"] in res: continue
        try:
            c = float(D.fve_model(mdl, d))
            V, _, _ = anm_modes(d["ref"], DM, CUTOFF)
            a = anm_fve(d, V) if V is not None else float("nan")
            res[d["pdb"]] = dict(N=d["N"], codec=c, anm=a, stamp=ST)
        except Exception as e:
            print(f"    {d['pdb']} N={d['N']}: FAIL {type(e).__name__}: {e}", flush=True)
            continue
        json.dump(res, open(RES, "w"))
        if (i + 1) % 10 == 0 or i < 5:
            print(f"  {i+1}/{len(HO)} {d['pdb']} N={d['N']}  tied {c:+.4f}  ANM {a:+.4f}  "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)

    good = {k: v for k, v in res.items() if np.isfinite(v.get("anm", np.nan))}
    if len(good) < 12:
        print(f"  only {len(good)} systems scored -- too few to report."); raise SystemExit
    N = np.array([v["N"] for v in good.values()], float)
    C = np.array([v["codec"] for v in good.values()], float)
    A = np.array([v["anm"] for v in good.values()], float)

    # QUARTILE BOUNDARIES FROM THE FULL HELD-OUT SET, not from whatever has been scored so far --
    # otherwise a partial run silently redefines "Q1" (the INBOX 18d failure, one level over).
    allN = np.array([d["N"] for d in HO], float)
    q = np.quantile(allN, [0.25, 0.5, 0.75])
    bands = [(0, q[0], "Q1"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1e18, "Q4")]
    print(f"\n=== 28b: TIED vs ZERO-SHOT ANM-{DM}, matched capacity, same frames ===", flush=True)
    print(f"  quartile boundaries from ALL {len(allN)} held-out systems: "
          f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}", flush=True)
    print(f"    {'band':>5}{'n':>5}{'tied med':>11}{'ANM med':>10}{'gap med':>10}"
          f"{'tied>ANM':>10}", flush=True)
    q1 = None
    for lo, hi, lab in bands:
        m = (N >= lo) & (N < hi)
        if m.sum() < 3: continue
        g = C[m] - A[m]
        row = (int(m.sum()), float(np.median(C[m])), float(np.median(A[m])),
               float(np.median(g)), float(np.mean(g > 0)))
        if lab == "Q1": q1 = row
        print(f"    {lab:>5}{row[0]:>5}{row[1]:>+11.4f}{row[2]:>+10.4f}{row[3]:>+10.4f}"
              f"{100*row[4]:>9.0f}%", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if q1 is None:
        print("  Q1 not yet covered.", flush=True)
    else:
        n1, tm, am, gm, fr = q1
        if gm > 0 and fr > 0.5:
            print(f"  ON THE SMALLEST QUARTILE THE TIED ARM BEATS THE ZERO-SHOT PEER: median tied")
            print(f"  {tm:+.4f} vs ANM {am:+.4f}, gap {gm:+.4f}, winning on {100*fr:.0f}% of {n1}")
            print(f"  systems. THAT IS THE PROJECT'S FIRST PEER WIN, and it is confined to Q1 --")
            print(f"  state the N range it holds over, never as a general claim.", flush=True)
        else:
            print(f"  NO PEER WIN EVEN ON Q1: median tied {tm:+.4f} vs ANM {am:+.4f}, gap {gm:+.4f},")
            print(f"  winning on {100*fr:.0f}% of {n1} systems. So tied is the best of several")
            print(f"  architectures that ALL lose to a zero-cost physics baseline -- a much smaller")
            print(f"  claim than 'the strongest arm measured', and it must be worded as one.",
                  flush=True)
    print(f"\n  CAVEAT that travels with any tied result (INBOX 28e): this arm's worst system is")
    print(f"  FVE {C.min():+.3f} and {100*np.mean(C < -0.5):.0f}% of systems are below -0.5. Best on")
    print(f"  the typical system AND catastrophic on a minority is a different operating point, not")
    print(f"  a uniformly better model.", flush=True)
