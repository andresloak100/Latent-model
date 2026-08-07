"""INBOX 22a, RUNG 1: can MESSAGE PASSING learn the nonlocal coupling ANM gets for free?

WHERE THIS SITS. 14a measured the codec losing to ANM at every width on 0% of 123 systems, and 17c
split the gap 65% BASIS QUALITY / 35% mode count -- with ANM-6 beating the codec at MATCHED RANK SIX.
So the deficit is not mainly too few directions; the directions are worse ones. The first modal arms
sharpen that: `untied` reaches PR 69.1/256 and identity 24% (control ~15-20 and 64-92%) with 56
effective basis modes, and still scores WORSE (+0.0996 / +0.0563 vs +0.1346). More directions, worse
ones -- rank and quality are empirically decoupled, so widening rank alone is not a path.

THE PRE-REGISTERED DIAGNOSIS. ANM's basis is NONLOCAL: a Hessian coupling neighbours within 5 A,
whose LOW modes span the whole structure. `ModalCodec` at `ctx_layers=0` builds `B_i` from atom i's
own element and reference position ALONE. This file adds the message passing that 015 and 17b both
said to try BEFORE abandoning the bilinear form.

AND THE HONEST LIMIT, STATED BEFORE THE RESULT (INBOX 22a). Two to four k-NN hops buy a receptive
field of roughly 10-15 A. The modes that matter are GLOBAL. So rung 1 may well stall, and the
measurement that makes it interpretable is the RECEPTIVE FIELD ITSELF -- reported in Angstroms, from
the real reference coordinates, rather than assumed from the hop count. If it stalls, rung 2
(Laplacian eigenvector positional encoding) is nonlocal BY CONSTRUCTION; rung 3 is to report that a
learned structure->basis map does not reach a physics-derived one at matched rank, and NOT to drift
into optimising ANM variants, which is the loop 004b exists to prevent.

WHY k-NN IS COMPUTED INSIDE THE MODEL. `armf_atlas_dm.train()` calls `mdl(stat, disp)` and knows
nothing about graphs. Threading `knn` through it would mean forking the training loop -- and a forked
loop is procedure drift, which is the defect this project spent the night measuring. Instead the
graph is derived from the REFERENCE POSITIONS ALREADY INSIDE `stat` (its last three columns) and
cached per system, so the training procedure is bit-identical to every other arm and the comparison
stays internally valid."""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata
import armf_atlas_dm as D
import armf_stamp as STAMP
from armf_modal_decoder import ModalCodec, basis_orthogonality, effective_modes

WR = D.WR
RES = f"{WR}/modal_ctx.json"
CKPT = f"{WR}/modal_ctx_ckpt"
DM, NTR, SEED = 256, 50, 0
KNN = 16                       # neighbours per atom
# INBOX 22a rung 1 as specified is {2, 4}. THE MEASURED RECEPTIVE FIELD FORCED A THIRD SETTING:
# the all-atom 16-NN radius is only 2.6 A -- atoms are densely packed, so sixteen neighbours barely
# leave the residue -- giving reach ~5.3 A at 2 hops and ~10.6 A at 4, at or BELOW the low end of
# 22a's ~10-15 A estimate. A stall at 10.6 A would then be AMBIGUOUS between "message passing cannot
# learn nonlocal coupling" and "it was never given the reach", which is a Family D hole: a
# measurement structurally unable to express the effect it is named for. ctx=8 (~21 A) is the
# DISAMBIGUATION arm -- not a fourth rung, and not a result to quote on its own.
CTX = [2, 4, 8]
LRS = D.LRS
dev = D.dev


class GraphModalCodec(ModalCodec):
    """ModalCodec whose basis network sees a NEIGHBOURHOOD, with the graph derived from `stat`.

    The k-NN indices are built from the reference positions carried in the last three columns of
    `stat` and cached per system, so `forward(stat, disp)` keeps the signature `train()` expects."""

    def __init__(self, *a, knn_k=KNN, **kw):
        super().__init__(*a, **kw)
        self.knn_k = knn_k
        self._cache = {}

    def _knn(self, stat):
        N = stat.shape[1]
        pos = stat[0, :, -3:]
        key = (N, float(pos[0, 0]), float(pos[N // 2, 1]), float(pos[-1, 2]))
        g = self._cache.get(key)
        if g is not None and g.shape[0] == stat.shape[0]:
            return g
        k = min(self.knn_k + 1, N)
        idx = torch.empty(N, k - 1, dtype=torch.long, device=stat.device)
        # chunked over rows: a full (N,N) distance matrix is 4 GB at N=33,377 and would OOM on the
        # LARGEST systems first -- the N-correlated failure this project keeps having to design out.
        step = max(1, min(2048, int(4e7 // max(N, 1))))
        for i in range(0, N, step):
            d2 = torch.cdist(pos[i:i + step], pos)
            idx[i:i + step] = d2.topk(k, largest=False).indices[:, 1:]
        g = idx.unsqueeze(0).expand(stat.shape[0], -1, -1)
        self._cache = {key: g}                       # one system at a time; never grows
        return g

    def encode(self, stat, disp, knn=None):
        return super().encode(stat, disp, self._knn(stat) if self.ctx_layers else None)

    def decode(self, stat, lat, knn=None):
        return super().decode(stat, lat, self._knn(stat) if self.ctx_layers else None)

    def z0(self, stat, knn=None):
        return super().z0(stat, self._knn(stat) if self.ctx_layers else None)

    def code(self, stat, disp, knn=None):
        z = self.encode(stat, disp)
        return z - self.z0(stat) if self.sub_z0 else z

    def forward(self, stat, disp, knn=None):
        return self.decode(stat, self.code(stat, disp))


def receptive_field(d, k=KNN, hops=2):
    """Report the reach in ANGSTROMS from the REAL reference coordinates -- stated, not assumed.
    Mean k-NN radius x hops is an upper bound on how far information can travel in the basis net."""
    from scipy.spatial import cKDTree
    ref = d["ref"]
    n = min(len(ref), 4000)
    sel = np.linspace(0, len(ref) - 1, n).astype(int)
    dd, _ = cKDTree(ref).query(ref[sel], k=min(k + 1, len(ref)))
    r = float(np.mean(dd[:, 1:]))
    return r, r * hops



def diameter(d, kind="2Rg"):
    """Structure extent in ANGSTROMS. `kind` is stated in the output, never left implicit:
      2Rg  -- twice the radius of gyration; robust, defined for every system
      maxd -- max pairwise reference distance on a subsample; the true extent, noisier
    INBOX 23a: absolute reach is the wrong axis. The same 21 A covers most of a 598-atom domain and a
    fifth of a 33,377-atom one, so what determines whether message passing can express a COLLECTIVE
    mode is reach RELATIVE TO THE STRUCTURE'S OWN EXTENT."""
    ref = d["ref"]
    if kind == "2Rg":
        c = ref.mean(0)
        return float(2.0 * np.sqrt(((ref - c) ** 2).sum(1).mean()))
    sel = ref[np.linspace(0, len(ref) - 1, min(len(ref), 1500)).astype(int)]
    from scipy.spatial.distance import pdist
    return float(pdist(sel).max())


def coverage_report(rows, HO, hop_radius, log=print):
    """INBOX 23a/23b: regress BASIS QUALITY on reach/diameter, not on N.

    This is what converts the ctx sweep from a hyperparameter search into a MECHANISM TEST. Without
    the coverage column, a stall at any ctx is the same ambiguity one rung further along."""
    try:
        pj = json.load(open(f"{WR}/atlas_peer.json")).get("sys", {})
    except Exception:
        log("  23a: atlas_peer.json unavailable -- basis quality needs the PCA curves. Sequenced, "
            "not skipped."); return
    def pca_at(pdb, k):
        cs = pj.get(pdb, {}).get("pca_out_cs")
        return float(cs[min(int(k), len(cs)) - 1]) if cs else None
    diam = {d["pdb"]: diameter(d, "2Rg") for d in HO}
    log(f"\n=== 23a: BASIS QUALITY vs REACH/DIAMETER (diameter = 2*Rg, stated) ===")
    log(f"  structure 2*Rg spans {min(diam.values()):.0f}-{max(diam.values()):.0f} A over "
        f"{len(diam)} held-out systems")
    log(f"    {'ctx':>5}{'reach A':>9}{'cov med':>9}{'cov range':>14}{'r':>7}"
        f"{'slope(bq~cov)':>16}{'slope(bq~logN)':>17}{'partial cov':>14}")
    for r in sorted(rows, key=lambda z: (z["ctx"], -z["fve"])):
        if r.get("_reported"): continue
        rr = r.get("realised_rank")
        if not rr: continue
        r["_reported"] = True
        reach = hop_radius * r["ctx"]
        bq, cov, lgn = [], [], []
        for pdb, n, f in zip([d["pdb"] for d in HO], r["Ns"], r["per"]):
            p_ = pca_at(pdb, rr)
            if p_ is None: continue
            bq.append(f - p_); cov.append(reach / diam[pdb]); lgn.append(np.log10(n))
        if len(bq) < 10: continue
        bq = np.array(bq); cov = np.array(cov); lgn = np.array(lgn)
        s1 = stats.linregress(cov, bq); h1 = stats.t.ppf(0.975, len(bq) - 2) * s1.stderr
        s2 = stats.linregress(lgn, bq); h2 = stats.t.ppf(0.975, len(bq) - 2) * s2.stderr
        # PARTIAL: does coverage survive controlling for N? (23b's third branch)
        X = np.column_stack([np.ones_like(cov), cov, lgn])
        beta, *_ = np.linalg.lstsq(X, bq, rcond=None)
        res = bq - X @ beta
        s2e = float(res @ res) / max(len(bq) - 3, 1)
        se = np.sqrt(np.clip(np.diag(s2e * np.linalg.pinv(X.T @ X)), 0, None))
        hp = stats.t.ppf(0.975, max(len(bq) - 3, 1)) * se[1]
        log(f"    {r['ctx']:>5}{reach:>9.1f}{np.median(cov):>9.2f}"
            f"{f'{cov.min():.2f}-{cov.max():.2f}':>14}{rr:>7.0f}"
            f"{s1.slope:>+11.4f}+/-{h1:.4f}{s2.slope:>+12.4f}+/-{h2:.4f}"
            f"{beta[1]:>+9.4f}+/-{hp:.4f}")
    log(f"  READ (INBOX 23b): basis quality tracking coverage, with N adding nothing once coverage is")
    log(f"  controlled => THE RECEPTIVE FIELD IS THE MECHANISM, and rung 2 (nonlocality without")
    log(f"  depth) is the evidence-directed step. Flat in coverage while still below ANM at matched")
    log(f"  rank => reach is NOT the constraint and rung 2 would treat the wrong cause. Tracking N")
    log(f"  even after controlling for coverage => a size effect independent of receptive field,")
    log(f"  which is a third finding and must not be folded into either.")
    log(f"  INBOX 23c: spanning a 120 A structure at {hop_radius:.1f} A per hop needs ~{120/hop_radius:.0f}")
    log(f"  layers, which over-smoothing makes unusable long before it is affordable. So a stall at")
    log(f"  LOW coverage is STRUCTURAL, not a disappointment -- the informative quantity is the")
    log(f"  CROSSOVER, and that is the number rung 2 has to beat.")


if __name__ == "__main__":
    print(f"[modal-ctx] INBOX 22a RUNG 1: does message passing buy the nonlocal coupling ANM has? "
          f"ctx_layers {CTX}, k={KNN}, DM={DM}, n_train={NTR}, LR grid {LRS}", flush=True)
    man = json.load(open(D.MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]
    HO = [x for x in (sysdata(store, have[p]) for p in ho_ids) if x is not None]
    _ord = sorted(range(len(HO)), key=lambda i: HO[i]["N"])
    HOt = [HO[_ord[i]] for i in np.linspace(0, len(HO) - 1, min(D.NHO_TRACK, len(HO))).astype(int)]
    TR = [x for x in (sysdata(store, have[p]) for p in tr_ids[:NTR]) if x is not None]
    print(f"  {len(TR)} train / {len(HOt)} tracked / {len(HO)} held-out", flush=True)

    # THE RECEPTIVE FIELD, IN ANGSTROMS, BEFORE ANY ARM RUNS -- so "still local" is a measurement.
    print(f"\n=== RECEPTIVE FIELD (INBOX 22a: state the reach, do not assume it) ===", flush=True)
    for h in CTX:
        rs = [receptive_field(d, KNN, h) for d in HOt[:8]]
        r1 = float(np.median([a for a, _ in rs])); rh = float(np.median([b for _, b in rs]))
        print(f"  ctx_layers={h}: mean {KNN}-NN radius {r1:.1f} A -> reach ~{rh:.1f} A over {h} hops",
              flush=True)
    Ns = [d["N"] for d in HO]
    print(f"  for scale, held-out systems span N {min(Ns)}-{max(Ns)}; the modes that matter are")
    print(f"  GLOBAL, so a 10-15 A reach may well be insufficient. That is the point of measuring it.",
          flush=True)

    ST = STAMP.stamp(dict(dm=DM, ntr=NTR, seed=SEED, knn=KNN, ctx=str(CTX), lrs=str(LRS),
                          warmup=D.WARMUP, maxsteps=D.MAXSTEPS),
                     GraphModalCodec, D.train)
    rows = json.load(open(RES)) if os.path.exists(RES) else []
    STAMP.report(rows, ST, "arms")
    rows = [r for r in rows if STAMP.same_stamp(r, ST)]
    done = {(r["ctx"], r["lr"]) for r in rows}
    orig = D.Codec

    for ctx in CTX:
        print(f"\n=== ctx_layers={ctx} ===", flush=True)
        for lr in LRS:                                   # Family E at every rung, as 22a instructs
            if (ctx, lr) in done: continue
            D.Codec = (lambda c: (lambda Fs, L, dm, dlat=None, sub_z0=False:
                                  GraphModalCodec(Fs, L, dm, dlat=dlat, sub_z0=sub_z0,
                                                  ctx_layers=c, tie_encoder=False)))(ctx)
            try:
                torch.manual_seed(SEED); np.random.seed(SEED + 1)
                t0 = time.time()
                mdl, hist, used, stopped, improving = D.train(TR, HOt, DM, lr, f"ctx{ctx} lr{lr:g}", 1)
                mdl.eval()
                per = [D.fve_model(mdl, x) for x in HO]
                fve = float(np.mean(per))
                p = D.participation_ratio(mdl, TR[:min(len(TR), len(HO))], HO, DM)
                p["z0_frac"] = D.identity_offset(mdl, HOt[:12])
                lr_ = stats.linregress(np.log10([x["N"] for x in HO]), per)
                hw = stats.t.ppf(0.975, len(per) - 2) * lr_.stderr
                with torch.no_grad():
                    s1 = torch.tensor(HOt[0]["stat"], device=dev).unsqueeze(0)
                    off = float(basis_orthogonality(mdl, s1)[1][0])
                    em = int(effective_modes(mdl, s1)[0])
                rec = dict(ctx=ctx, lr=lr, dm=DM, n_train=NTR, seed=SEED, stamp=ST, fve=fve,
                           steps=used, stopped=stopped, improving=improving,
                           best_track=max(h[1] for h in hist), nslope=float(lr_.slope),
                           nslope_ci=float(hw), nho=len(HO), basis_off=off, eff_modes=em,
                           per=[float(v) for v in per], Ns=[int(x["N"]) for x in HO],
                           secs=time.time() - t0, **p)
                rows.append(rec); json.dump(rows, open(RES, "w")); done.add((ctx, lr))
                os.makedirs(CKPT, exist_ok=True)
                torch.save(mdl.state_dict(), f"{CKPT}/ctx{ctx}_lr{lr:g}_s{SEED}.pt")
                print(f"    ctx{ctx} lr{lr:g}: FVE {fve:+.4f}  PR {p['pr']:.1f}/{DM}  "
                      f"identity {100*p['ident_frac']:.0f}%  eff-modes {em}/{DM}  "
                      f"basis-off {off:.3f}  N-slope {lr_.slope:+.4f}+/-{hw:.4f}  "
                      f"steps {used} ({stopped})"
                      f"{'  *** VOID ***' if improving else ''}  [{time.time()-t0:.0f}s]", flush=True)
            except Exception as e:
                print(f"    ctx{ctx} lr{lr:g}: FAIL {type(e).__name__}: {e}", flush=True)
            finally:
                D.Codec = orig

    if not rows: raise SystemExit
    # realised rank per arm (INBOX 17b convention, imported so it cannot drift)
    try:
        import armf_atlas_modes as M
        SUB = [HO[i] for i in np.linspace(0, len(HO) - 1, min(16, len(HO))).astype(int)]
        for r in rows:
            if r.get("realised_rank") or r["improving"]: continue
            cp = f"{CKPT}/ctx{r['ctx']}_lr{r['lr']:g}_s{SEED}.pt"
            if not os.path.exists(cp): continue
            D.Codec = (lambda c: (lambda Fs, L, dm, dlat=None, sub_z0=False:
                                  GraphModalCodec(Fs, L, dm, dlat=dlat, sub_z0=sub_z0,
                                                  ctx_layers=c, tie_encoder=False)))(r["ctx"])
            m2 = D.Codec(HO[0]["stat"].shape[1], 1, DM).to(dev); D.Codec = orig
            m2.load_state_dict(torch.load(cp, map_location=dev)); m2.eval()
            rk = []
            for d in SUB:
                try: rk.append(M.mode_table(m2, d)["rank90_out"])
                except Exception: pass
            if rk: r["realised_rank"] = float(np.median(rk))
        json.dump(rows, open(RES, "w"))
    except Exception as e:
        print(f"  realised-rank scoring FAILED: {type(e).__name__}: {e}", flush=True)
    hop_r = float(np.median([receptive_field(d, KNN, 1)[0] for d in HOt[:8]]))
    coverage_report(rows, HO, hop_r)

    print(f"\n=== RUNG 1 SUMMARY (best LR per ctx) ===", flush=True)
    print(f"    {'ctx':>5}{'bestLR':>9}{'FVE':>9}{'PR':>8}{'ident':>8}{'eff-modes':>11}"
          f"{'basis-off':>11}", flush=True)
    for ctx in CTX:
        c = [r for r in rows if r["ctx"] == ctx and not r["improving"]]
        if not c: continue
        b = max(c, key=lambda r: r["fve"])
        print(f"    {ctx:>5}{b['lr']:>9.0e}{b['fve']:>9.4f}{b['pr']:>8.1f}"
              f"{100*b['ident_frac']:>7.0f}%{b['eff_modes']:>11}{b['basis_off']:>11.3f}", flush=True)
    print(f"\n  Compare (same procedure, same seed, same data, from modal_arm.json):")
    print(f"    control (attention decoder) best FVE +0.1346;  untied modal ctx=0 best +0.0996")
    print(f"  READ: FVE rising with ctx and the 17c basis-quality gap CLOSING => message passing is")
    print(f"  buying the nonlocality, and rung 2 is unnecessary. FVE flat in ctx => the receptive")
    print(f"  field printed above is the limit, and rung 2 (Laplacian eigenvector features, nonlocal")
    print(f"  BY CONSTRUCTION) is the next rung -- NOT an ANM optimisation loop (004b).", flush=True)
