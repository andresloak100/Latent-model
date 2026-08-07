"""INBOX 14a: THE PRIMARY COMPARISON. Codec vs ANM vs the per-system PCA oracle, same held-out frames.

The standing frame since 004b is that the primary result is **codec vs ANM, both zero-shot on the
same held-out frames** -- ceiling-free, immune to the failed guard, and the thing the thesis rests
on. Every ATLAS report so far has given CODEC FVE ALONE. The comparison the project exists to make
has never appeared in an output anybody can read. 14a is right, and this file is the fix.

WHY A NEW FILE RATHER THAN RUNNING armf_atlas_curve.py. That script does contain the table, but it
(a) is pinned to `L = 24; DM = 256; KS = [24]`, which INBOX 003 retired -- L=1 is the architecture,
not a swept variable -- and (b) TRAINS its own learning curve, so it needs the GPU that the DM sweep
occupies for the next 19 hours. Every codec number needed here ALREADY EXISTS: `atlas_dm.json`
records `per` (per-system held-out FVE) and `Ns` for every finished arm. So the peer columns are a
pure CPU measurement that can run NOW, in parallel, and be JOINED to arms already paid for.

MATCHED CAPACITY. The codec at L=1 emits DM numbers per frame. ANM-k and PCA-k emit k coefficients
per frame. So k = DM is the matched comparison, and it is computed at every DM the sweep uses.
STATED PLAINLY, because it cuts against us: ANM-k gets a SYSTEM-SPECIFIC basis derived from that
system's own structure, while the codec's DM numbers come from ONE model shared across all systems.
Matched on numbers-per-frame, ANM is the more favoured of the two. It is still the right bar --
it is what a practitioner would actually do without training anything -- but "matched capacity"
must not be read as "matched information".

THE THREE COLUMNS, AND WHAT EACH IS FOR.
  codec FVE     what we have.
  ANM-k FVE     THE PEER. Zero-shot exactly like the codec: needs the reference structure and no
                trajectory. Cutoff swept on TRAINING systems only and applied unchanged (Family E),
                because a codec that beats a deliberately-weak ANM has won nothing.
  PCA-k FVE     THE ORACLE, never a bar. It fits 3N x k free parameters to the target system's own
                trajectory -- 1.6M parameters at k=16, N=33,377 -- which is not something any
                deployed method has. Reported only as a FRACTION, per 014.

TWO PCA ORACLES, because the rank90 retraction applies here too. `oracle_in` fits and evaluates on
the SAME replica-2 frames and is therefore flattered by construction -- it is reported only because
earlier numbers (PCA-16 ~ 0.53) were computed that way and the continuity matters. `oracle_out`
fits the basis on replicas 0+1 and evaluates on replica 2, which is the honest within-system oracle
and the one to quote. Quoting the in-sample one as "the achievable" would repeat the exact error
retracted for rank90.

EVERYTHING IS mu-CENTRED, with mu the replica-0+1 mean -- the same origin the codec reconstructs
displacements from, and the same `sst` in every denominator. Three methods scored against three
different centres would not be a comparison.

ONE STREAMED PASS PER SYSTEM over column chunks accumulates the ANM projection, the cross-replica
Gram and the held-out/train cross-product together. (F, 3N) is 1.0 GB at N=33,377 and is never
materialised -- the failure mode that would kill the LARGEST systems first, i.e. an N-correlated
exclusion truncating the very axis under test."""
import sys, os, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armf_atlas_data import AtlasStore, sysdata, ho_cols
from armf_anm import modes as anm_modes, CUTOFF_SWEEP
import armf_stamp as STAMP

WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
MAN = f"{WR}/atlas_manifest.json"
DMRES = f"{WR}/atlas_dm.json"
RES = f"{WR}/atlas_peer.json"
MODRES = f"{WR}/atlas_modes.json"
KS = [16, 64, 256]                 # matched to the DM sweep's widths
KMAX = max(KS)
CUT_K = 16                         # k at which the ANM cutoff is selected -- FIXED, so that adding
                                   # a k to the ladder cannot silently change which cutoff wins
CUT_PROBE = 24                     # training systems for the cutoff sweep (Family E)
CHUNK = 20000                      # columns per streamed block
RANK_K_DEFAULT = 6                 # INBOX 16a: the codec's measured realised rank (median)


def tr_cols(d, c0, c1):
    """TRAIN-replica (0+1) column slice, mu-centred. Mirrors ho_cols; nothing large retained."""
    a = np.load(d["path"], mmap_mode="r")
    return np.concatenate([np.asarray(a[r]).reshape(d["F"], -1)[:, c0:c1].astype(np.float64)
                           for r in (0, 1)], 0) - d["mu"][c0:c1]


def peer_one(d, V):
    """ONE streamed pass. Returns ANM prefix-FVE at every k, and both PCA oracles at every k.

    Orthonormal modes make FVE_k a PREFIX SUM of per-mode projected energy, so the k-ladder costs
    nothing beyond the KMAX solve -- no separate eigensolve per k."""
    F, N3 = d["F"], 3 * d["N"]
    kk = V.shape[1] if V is not None else 0
    P = np.zeros((F, kk)) if kk else None      # ho @ V           (F, KMAX)
    G = np.zeros((2 * F, 2 * F))               # Xtr @ Xtr^T      frame-space Gram
    M = np.zeros((F, 2 * F))                   # ho  @ Xtr^T
    Q = np.zeros((F, F))                       # ho  @ ho^T       for the in-sample oracle
    for c0 in range(0, N3, CHUNK):
        c1 = min(c0 + CHUNK, N3)
        h = ho_cols(d, c0, c1); x = tr_cols(d, c0, c1)
        if kk: P += h @ V[c0:c1]
        G += x @ x.T; M += h @ x.T; Q += h @ h.T
        del h, x
    sst = d["sst"]
    out = {}

    # THE FULL CUMULATIVE CURVE IS STORED, not just the ladder points. INBOX 17c needs FVE at
    # k = the codec's REALISED RANK, a number that comes from a different job and can change; storing
    # only the pre-chosen ladder would force a full recomputation every time that k moved. Modes are
    # orthonormal, so the curve is a prefix sum and costs nothing extra.
    if kk:
        e = (P ** 2).sum(0)                                        # per-mode projected energy
        cs = np.cumsum(e) / (sst + 1e-12)
        out["anm_cs"] = [float(v) for v in cs]
        for k in KS: out[f"anm{k}"] = float(cs[min(k, kk) - 1])

    # ORACLE, OUT-OF-SAMPLE: basis from replicas 0+1, evaluated on replica 2.
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]
    w = np.clip(w[o], 0, None); U = U[:, o]
    good = w > w[0] * 1e-12
    s = np.sqrt(w[good]) + 1e-12
    Pj = (M @ U[:, good]) / s                                       # = ho @ V_pca, without forming V
    cs = np.cumsum((Pj ** 2).sum(0)) / (sst + 1e-12)
    out["pca_out_cs"] = [float(v) for v in cs[:KMAX]]
    for k in KS: out[f"pca_out{k}"] = float(cs[min(k, len(cs)) - 1])

    # ORACLE, IN-SAMPLE: fit and evaluate on the same replica-2 frames. FLATTERED BY CONSTRUCTION;
    # kept only because the earlier PCA-16 ~ 0.53 figure was computed this way.
    wq = np.sort(np.clip(np.linalg.eigvalsh(Q), 0, None))[::-1]
    cs = np.cumsum(wq) / (sst + 1e-12)
    out["pca_in_cs"] = [float(v) for v in cs[:KMAX]]
    for k in KS: out[f"pca_in{k}"] = float(cs[min(k, len(cs)) - 1])
    return out


def select_cutoff(TR, log=print):
    """FAMILY E. Sweep the cutoff on TRAINING systems only; apply the winner unchanged to held-out.
    Selecting it on held-out would make the peer an oracle; not sweeping it at all would make the
    peer weak, and a win over a weak peer is not a win."""
    log(f"\n=== PEER CUTOFF SWEPT ON {len(TR)} TRAINING SYSTEMS (held-out never seen) ===")
    means = {}
    for c in CUTOFF_SWEEP:
        v = []
        for d in TR:
            V, _, _ = anm_modes(d["ref"], CUT_K, c)
            if V is None: continue
            P = np.zeros((d["F"], V.shape[1]))
            for c0 in range(0, 3 * d["N"], CHUNK):
                c1 = min(c0 + CHUNK, 3 * d["N"])
                P += ho_cols(d, c0, c1) @ V[c0:c1]
            v.append(float((P ** 2).sum() / (d["sst"] + 1e-12)))
        means[c] = float(np.mean(v)) if v else float("nan")
        log(f"  cutoff {c:>4} A: mean TRAIN FVE at k={CUT_K}  {means[c]:.4f}  (n={len(v)})")
    best = max((c for c in means if np.isfinite(means[c])), key=lambda c: means[c])
    log(f"  -> SELECTED {best} A on TRAINING systems, applied unchanged to all held-out systems.")
    return best, means


if __name__ == "__main__":
    print(f"[atlas-peer] 14a: codec vs ANM vs per-system PCA oracle. k ladder {KS}", flush=True)
    man = json.load(open(MAN)); store = AtlasStore(f"{WR}/atlas_cache")
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho_ids = [p for p in man["heldout"] if p in have]
    tr_ids = [p for p in man["train_ordered"] if p in have]
    print(f"  {len(ho_ids)} held-out / {len(tr_ids)} train systems cached", flush=True)

    res = json.load(open(RES)) if os.path.exists(RES) else {}
    res.setdefault("sys", {})
    if "cutoff" not in res:
        TRp = [x for x in (sysdata(store, have[p]) for p in tr_ids[:CUT_PROBE]) if x is not None]
        best, means = select_cutoff(TRp)
        res["cutoff"] = best; res["cutoff_train_fve"] = {str(k): v for k, v in means.items()}
        res.setdefault("sys", {}); json.dump(res, open(RES, "w"))
        del TRp
    best = res["cutoff"]
    # INBOX 18a: stamp each system row, and PURGE rows written by a different configuration rather
    # than resuming over them. This is precisely the 10307102 case: 49 systems stored by an earlier
    # build had no cumulative curves, would never have been recomputed, and would have silently
    # dropped out of the matched-rank decomposition.
    ST = STAMP.stamp(dict(ks=str(KS), kmax=KMAX, cut_k=CUT_K, cutoff=best, chunk=CHUNK),
                     peer_one, tr_cols)
    stale = [p for p, v in res["sys"].items() if not STAMP.same_stamp(v, ST) and "err" not in v]
    STAMP.report(list(res["sys"].values()), ST, "system rows")
    if stale:
        print(f"  [stamp] purging {len(stale)} system rows written by a different configuration "
              f"so they are RECOMPUTED rather than silently excluded.", flush=True)
        for p in stale: res["sys"].pop(p, None)
        json.dump(res, open(RES, "w"))
    print(f"\n=== held-out peer columns at cutoff {best} A (resuming: "
          f"{len(res.get('sys', {}))} systems already done) ===", flush=True)

    # ASCENDING N. If the job is preempted or the wall clock runs out, what survives is a
    # SIZE-TRUNCATED sample -- so the truncation is visible in the coverage table rather than
    # silently reweighting the very axis under test.
    order = sorted(ho_ids, key=lambda p: store.meta[have[p]]["atoms"])
    # INBOX 18d: a resume path IS an exclusion filter when the work is ordered by a regressor.
    STAMP.coverage_by([store.meta[have[p]]["atoms"] for p in order if p in res["sys"]],
                      [store.meta[have[p]]["atoms"] for p in order if p not in res["sys"]], "N")
    t00 = time.time()
    for i, p in enumerate(order):
        if p in res.get("sys", {}): continue
        d = sysdata(store, have[p])
        if d is None:
            res["sys"][p] = dict(N=store.meta[have[p]]["atoms"], err="sysdata failed"); continue
        t0 = time.time()
        V, att, dt = anm_modes(d["ref"], KMAX, best)
        try:
            row = peer_one(d, V)
        except Exception as e:
            row = dict(err=f"{type(e).__name__}: {e}")
        row["N"] = d["N"]; row["anm_ok"] = V is not None; row["anm_att"] = att
        row["stamp"] = ST
        row["secs"] = time.time() - t0
        res["sys"][p] = row
        json.dump(res, open(RES, "w"))
        if (i + 1) % 5 == 0 or i < 3:
            print(f"  {i+1}/{len(order)} {p} N={d['N']} anm{KS[0]}={row.get(f'anm{KS[0]}', float('nan')):.4f} "
                  f"pca_out{KS[0]}={row.get(f'pca_out{KS[0]}', float('nan')):.4f} "
                  f"({row['secs']:.0f}s, {(time.time()-t00)/60:.0f} min elapsed)", flush=True)
        del d, V

    # ---------------- REPORT ----------------
    S = {p: r for p, r in res["sys"].items() if "err" not in r}
    print(f"\n=== COVERAGE (a peer missing at the top of the N range is a FAMILY A exclusion) ===",
          flush=True)
    allN = np.array([res["sys"][p].get("N", 0) for p in res["sys"]], float)
    ok = np.array([p in S and S[p].get("anm_ok", False) for p in res["sys"]])
    if len(allN):
        qs = np.percentile(allN, [0, 33, 67, 100])
        for a, b in zip(qs[:-1], qs[1:]):
            m = (allN >= a) & (allN <= b)
            print(f"  N {int(a):>6}-{int(b):>6}: ANM computable on {ok[m].sum()}/{m.sum()}", flush=True)
    print(f"  overall {ok.sum()}/{len(ok)} systems", flush=True)
    if ok.sum() < len(ok):
        bad = sorted([res["sys"][p].get("N", 0) for p in res["sys"] if not (p in S and S[p].get("anm_ok"))])
        print(f"  *** {len(bad)} systems have NO peer value, N = {bad[:8]}{'...' if len(bad) > 8 else ''}. "
              f"If these skew large the comparison is TRUNCATED at the end that matters. ***", flush=True)

    rows = json.load(open(DMRES)) if os.path.exists(DMRES) else []
    net = [r for r in rows if r.get("arch", "network") == "network" and r.get("L") == 1]
    if not net:
        print("\n  no finished L=1 codec arms yet -- peer columns are computed and stored; re-run to "
              "join them. The peer table below stands on its own.", flush=True)

    # The codec's `per` is indexed by the SAME construction used here (manifest held-out order,
    # filtered by cache membership), so alignment is by position -- but VERIFIED against the stored
    # Ns rather than assumed, because the cache grew between runs.
    # Reconstruct the codec's held-out ORDER exactly: manifest order, minus systems whose sysdata
    # load failed -- which is the same filter `armf_atlas_dm.py` applies when it builds HO. Using the
    # unfiltered id list instead would shift every position by one after the first failure and
    # silently pair each codec value with the WRONG system's peer value.
    cur = [(p, res["sys"][p]["N"]) for p in ho_ids
           if p in res["sys"] and "N" in res["sys"][p] and res["sys"][p].get("err") != "sysdata failed"]
    print(f"\n=== 14a: THE PRIMARY COMPARISON, per arm, same held-out frames, matched capacity k=DM ===",
          flush=True)
    print(f"  ANM is ZERO-SHOT like the codec. PCA is an ORACLE fitted to the target's own")
    print(f"  trajectory (3N x k free parameters) and is a FRACTION, never a bar.", flush=True)
    print(f"    {'n_train':>8}{'DM=k':>6}{'codec':>9}{'ANM-k':>9}{'codec-ANM':>11}"
          f"{'frac>ANM':>10}{'PCAout-k':>10}{'% oracle':>10}", flush=True)
    joined = []
    for r in sorted(net, key=lambda r: (r["n_train"], r["dm"])):
        k = r["dm"]
        if k not in KS:
            # NEVER SILENT. An arm dropped without a word would leave the primary table looking
            # complete while the widest arm -- possibly the best one -- had no peer at all.
            print(f"    n{r['n_train']} DM{k}: NO MATCHED PEER COLUMN. The k ladder is {KS} (capped "
                  f"at {KMAX} because ANM cost grows steeply: 101 s at N=15,673 for k=256). This arm "
                  f"scored FVE {r['fve']:+.4f} and is ABSENT from the comparison below -- if it is "
                  f"the best arm, EXTEND THE LADDER before quoting any codec-vs-ANM verdict.",
                  flush=True)
            continue
        Ns = r.get("Ns", [])
        if len(Ns) != len(cur) or [n for _, n in cur] != list(Ns):
            print(f"    n{r['n_train']} DM{k}: SKIPPED -- this arm scored {len(Ns)} systems but the "
                  f"cache now holds {len(cur)}; positional join is unsafe and an N-mismatched join "
                  f"would be a Family F error (comparator on different data).", flush=True)
            continue
        cf, af, of, Nv = [], [], [], []
        for (p, n), c in zip(cur, r["per"]):
            s = S.get(p)
            if not s or not s.get("anm_ok"): continue
            cf.append(c); af.append(s[f"anm{k}"]); of.append(s[f"pca_out{k}"]); Nv.append(n)
        if len(cf) < 10:
            print(f"    n{r['n_train']} DM{k}: only {len(cf)} systems have BOTH a codec and a peer "
                  f"value -- too few to report. Not a null result; the peer pass is incomplete.",
                  flush=True)
            continue
        cf = np.array(cf); af = np.array(af); of = np.array(of); Nv = np.array(Nv, float)
        gap = cf - af
        joined.append(dict(n_train=r["n_train"], dm=k, n=len(cf), codec=float(cf.mean()),
                           anm=float(af.mean()), gap=float(gap.mean()),
                           frac=float((gap > 0).mean()), oracle=float(of.mean()),
                           gap_per=gap, Ns=Nv, codec_per=cf, anm_per=af))
        print(f"    {r['n_train']:>8}{k:>6}{cf.mean():>9.4f}{af.mean():>9.4f}{gap.mean():>+11.4f}"
              f"{100*(gap > 0).mean():>9.0f}%{of.mean():>10.4f}"
              f"{100*cf.mean()/(of.mean()+1e-12):>9.0f}%", flush=True)

    if joined:
        print(f"\n=== WHERE THE CODEC ACTUALLY SITS ===", flush=True)
        b = max(joined, key=lambda j: j["codec"])
        print(f"  Best arm: n_train={b['n_train']} DM={b['dm']} on {b['n']} held-out systems.")
        print(f"    codec {b['codec']:+.4f}   ANM-{b['dm']} {b['anm']:+.4f}   "
              f"signed gap {b['gap']:+.4f}   codec beats ANM on {100*b['frac']:.0f}% of systems")
        print(f"    per-system PCA-{b['dm']} oracle {b['oracle']:.4f}; the codec reaches "
              f"{100*b['codec']/(b['oracle']+1e-12):.0f}% of it.", flush=True)
        if b["gap"] < 0:
            print(f"  => THE CODEC LOSES TO THE ZERO-SHOT PEER at matched capacity. Per 004b this is")
            print(f"     NOT fatal -- ANM has no generator and cannot be the product -- but it must be")
            print(f"     stated in front of every FVE-vs-N claim, and a FLAT SLOPE ON A MODEL THIS FAR")
            print(f"     FROM THE ACHIEVABLE IS CONSISTENT WITH UNIFORM WEAKNESS, not with the")
            print(f"     architecture holding up.", flush=True)
        else:
            print(f"  => the codec beats the zero-shot peer at matched capacity.", flush=True)

        print(f"\n=== IS THE GAP ITSELF FLAT IN N? (the ceiling-free version of the headline) ===",
              flush=True)
        print(f"  Codec FVE vs N is confounded by the ceiling degrading with N. The codec-ANM GAP is")
        print(f"  not: both terms are scored on the same frames with the same denominator, so a")
        print(f"  shared N-dependent ceiling cancels. This is the slope to quote.", flush=True)
        for j in joined:
            lr = stats.linregress(np.log10(j["Ns"]), j["gap_per"])
            hw = stats.t.ppf(0.975, j["n"] - 2) * lr.stderr
            lc = stats.linregress(np.log10(j["Ns"]), j["codec_per"])
            hc = stats.t.ppf(0.975, j["n"] - 2) * lc.stderr
            print(f"    n{j['n_train']:>4} DM{j['dm']:>4}: gap-vs-log10(N) {lr.slope:+.4f} +/- {hw:.4f}"
                  f"   (codec alone {lc.slope:+.4f} +/- {hc:.4f})   "
                  f"{'FLAT' if abs(lr.slope) < hw else '*** MOVES WITH N ***'}", flush=True)

    # ---------------- INBOX 17c: WHERE DOES THE GAP LIVE? ----------------
    # "codec 0.155 vs PCA-16 ~ 0.53" compares SIX realised directions against SIXTEEN, so it fuses
    # two different deficits. Splitting them at matched rank says which fix is worth making:
    #   codec vs PCA-r / ANM-r at r = the codec's OWN realised rank -> BASIS QUALITY at matched count
    #   PCA-r vs PCA-16                                             -> what the missing MODES are worth
    RANK_K = RANK_K_DEFAULT
    try:                       # take the realised rank from the 16a measurement, not a constant
        md = json.load(open(MODRES))
        rr = [s_["rank90_out"] for s_ in md.get("sys", []) if "rank90_out" in s_]
        if rr: RANK_K = max(1, int(round(float(np.median(rr)))))
        print(f"\n  (realised rank read from atlas_modes.json: median {RANK_K} across {len(rr)} systems)",
              flush=True)
    except Exception:
        print(f"\n  (atlas_modes.json unreadable -- falling back to the recorded realised rank "
              f"{RANK_K}; re-run after 16a to pick it up automatically)", flush=True)

    def at_k(p, field, k):
        cs = S[p].get(field)
        if not cs: return None
        return float(cs[min(k, len(cs)) - 1])

    print(f"\n=== 17c: IS THE GAP MODE COUNT, OR BASIS QUALITY? (matched rank r={RANK_K}) ===",
          flush=True)
    print(f"  The codec realises {RANK_K} directions (16a). Comparing it to a 16-mode fit charges it")
    print(f"  for BOTH deficits at once. At matched rank the two separate.", flush=True)
    if joined:
        b = max(joined, key=lambda j: j["codec"])
        # EVERY MEAN BELOW IS OVER THE SAME SYSTEMS. The codec value is carried alongside each row
        # rather than reused from `b["codec"]`, because a system whose cumulative curve is missing
        # would drop from the peer means but not from the codec's -- means over different row sets,
        # which is exactly the FAMILY F failure this file's own header warns about.
        pdbs = [p for (p, _) in cur if p in S and S[p].get("anm_ok")]
        cmap = dict(zip([p for (p, _) in cur if p in S and S[p].get("anm_ok")], b["codec_per"]))
        rows_k = [(p, at_k(p, "anm_cs", RANK_K), at_k(p, "pca_out_cs", RANK_K),
                   at_k(p, "pca_out_cs", 16), cmap.get(p)) for p in pdbs]
        rows_k = [r for r in rows_k if all(v is not None for v in r[1:])]
        if rows_k:
            a_r = np.array([r[1] for r in rows_k]); p_r = np.array([r[2] for r in rows_k])
            p_16 = np.array([r[3] for r in rows_k]); c_r = np.array([r[4] for r in rows_k])
            if len(rows_k) != len(pdbs):
                print(f"    ({len(pdbs) - len(rows_k)} of {len(pdbs)} systems dropped for a missing "
                      f"curve; ALL columns below are means over the SAME {len(rows_k)} systems)",
                      flush=True)
            b = dict(b); b["codec"] = float(c_r.mean())        # codec mean on the matched set
            print(f"    {'quantity':>26}{'FVE':>10}   (n={len(rows_k)} systems)")
            print(f"    {'codec (realises ' + str(RANK_K) + ')':>26}{b['codec']:>10.4f}")
            print(f"    {'ANM-' + str(RANK_K) + ' (zero-shot peer)':>26}{a_r.mean():>10.4f}")
            print(f"    {'PCA-' + str(RANK_K) + ' (oracle)':>26}{p_r.mean():>10.4f}")
            print(f"    {'PCA-16 (oracle)':>26}{p_16.mean():>10.4f}", flush=True)
            print(f"\n    BASIS QUALITY at matched rank {RANK_K}:")
            print(f"      codec - PCA-{RANK_K}  = {b['codec'] - p_r.mean():+.4f}   "
                  f"(how much worse a zero-shot structure-derived basis is than one fitted to the")
            print(f"      {'':>{len(str(RANK_K))}}          target's own trajectory, holding the number of directions FIXED)")
            print(f"      codec - ANM-{RANK_K}  = {b['codec'] - a_r.mean():+.4f}   "
                  f"(the LIKE-FOR-LIKE number: both zero-shot, both {RANK_K} directions)")
            print(f"    MODE COUNT:")
            print(f"      PCA-16 - PCA-{RANK_K} = {p_16.mean() - p_r.mean():+.4f}   "
                  f"(what the missing directions are worth)", flush=True)
            bq = p_r.mean() - b["codec"]; mc = p_16.mean() - p_r.mean()
            tot = bq + mc
            if tot > 1e-9:
                print(f"\n    => of the {tot:.4f} between the codec and PCA-16, "
                      f"{100*bq/tot:.0f}% is BASIS QUALITY and {100*mc/tot:.0f}% is MODE COUNT.")
                if bq > mc:
                    print(f"       BASIS QUALITY DOMINATES. INBOX 015's modal decoder widens the")
                    print(f"       realised rank, which addresses the SMALLER half. A better basis --")
                    print(f"       not merely more modes -- is where the larger deficit is.", flush=True)
                else:
                    print(f"       MODE COUNT DOMINATES, so raising the realised rank (015) attacks")
                    print(f"       the larger half.", flush=True)

    print(f"\n=== THE ORACLE LADDER (fraction, never a bar) ===", flush=True)
    print(f"    {'k':>5}{'ANM-k':>10}{'PCA out':>10}{'PCA in':>10}{'in/out':>9}", flush=True)
    for k in KS:
        a = np.array([S[p][f"anm{k}"] for p in S if S[p].get("anm_ok")], float)
        po = np.array([S[p][f"pca_out{k}"] for p in S if f"pca_out{k}" in S[p]], float)
        pi = np.array([S[p][f"pca_in{k}"] for p in S if f"pca_in{k}" in S[p]], float)
        if not len(po): continue
        print(f"    {k:>5}{a.mean() if len(a) else float('nan'):>10.4f}{po.mean():>10.4f}"
              f"{pi.mean():>10.4f}{pi.mean()/(po.mean()+1e-12):>9.2f}x", flush=True)
    print(f"  in/out > 1 is the SAME in-sample flattery retracted for rank90: fitting and scoring a")
    print(f"  k-dim basis on the same frames explains them optimally by construction. Quote PCA out.",
          flush=True)
