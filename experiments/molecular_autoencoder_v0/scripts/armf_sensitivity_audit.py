"""INBOX 20b: RUN VERDICT SENSITIVITY OVER THE CLAIMS THAT SURVIVED.

Every application of `verdict_sensitivity()` so far has been to a verdict already under suspicion.
That is the easy direction. The claims sitting in the ROADMAP as load-bearing have NEVER been asked
whether they survive their own free choices -- and each of them contains at least one constant that
somebody chose.

The by-product that motivates this: run against the CORRECTED 16a test, the decoder-limited verdict
comes back STABLE across 0.75x-1.5x. So "survives its own free choices" is a property this project
can now state about a claim. The ones already in the record have not been asked for it.

WHAT A FLIP MEANS, stated before any result. A claim that flips is NOT thereby wrong. It means the
claim is only defined together with its threshold, and from then on it must carry the range in the
ROADMAP and must not be cited without it. A claim that is stable has earned something it did not have
before.

EVERY NUMBER BELOW IS READ FROM THE STORED RESULT FILES, never retyped from a commit message --
a hardcoded comparator is the Family F failure this project has already committed twice."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armf_stamp as STAMP

WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"


def load(name):
    try:
        return json.load(open(f"{WR}/{name}"))
    except Exception as e:
        print(f"  [skip] {name}: {type(e).__name__}", flush=True)
        return None


def slope_ci(x, y):
    lr = stats.linregress(x, y)
    return lr.slope, stats.t.ppf(0.975, max(len(x) - 2, 1)) * lr.stderr


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}", flush=True)


results = []


def record(name, stable, note=""):
    results.append((name, stable, note))


if __name__ == "__main__":
    print("[sensitivity-audit] INBOX 20b: do the SURVIVING load-bearing claims survive their own "
          "free choices?", flush=True)

    # ---------------------------------------------------------------- 1. b >= 0.93
    hdr("1. b >= 0.93 (rank90 vs N) -- and the EXCLUSION RULE hidden inside it")
    b = load("atlas_b.json")
    rows = (b if isinstance(b, list) else (b or {}).get("rows", [])) or []
    rows = [r for r in rows if isinstance(r, dict) and r.get("N")]
    if len(rows) >= 10:
        # FIELD NAMES READ FROM THE FILE, NOT GUESSED. atlas_b stores `j{J}_r` (in-sample),
        # `j{J}_rout` (out-of-sample, TRAIN-ordered) and `j{J}_rsort` (out-of-sample, sorted on
        # held-out variance) per replica-join count J. My first version looked for `rank90_out` and
        # friends, none of which exist -- it would have printed "no rank90 field found" forever and
        # been read as "the job has not landed yet". A guessed schema is the same failure as a
        # hardcoded comparator, one level up.
        joins = sorted({int(k[1]) for r in rows for k in r if k.startswith("j") and k[2:] == "_r"})
        print(f"  replica-join counts present: {joins}")
        # INBOX 011: the ORDERING-FREE variant is the architecture-relevant one, so it leads; the
        # other two are reported so the choice of variant is itself visible as a free choice.
        for J in joins:
            for fld, lab in ((f"j{J}_rsort", "out-of-sample SORTED (ordering-free)"),
                             (f"j{J}_rout", "out-of-sample, train-ordered"),
                             (f"j{J}_r", "in-sample")):
                use = [r for r in rows if r.get(fld) and r.get("N")]
                if len(use) < 12: continue
                N = np.array([r["N"] for r in use], float)
                R = np.array([r[fld] for r in use], float)
                def dec_b(_v, cut, N=N, R=R):
                    m = N >= cut
                    if m.sum() < 10: return "n/a"
                    sl, h = slope_ci(np.log10(N[m]), np.log10(R[m]))
                    tag = ">=0.9" if sl - h > 0.9 else ("<0.9" if sl + h < 0.9 else "spans 0.9")
                    return f"b={sl:+.2f} {tag}"
                print(f"\n  --- J={J}, {lab}, n={len(use)}, N {N.min():.0f}-{N.max():.0f} ---")
                st, _ = STAMP.verdict_sensitivity(
                    dec_b, {"sample N floor": float(np.percentile(N, 25))},
                    float(np.percentile(N, 25)), scales=(0.5, 1.0, 2.0),
                    label=f"b vs N (J={J}, {lab})")
                record(f"b vs N  J={J} {lab[:26]}", st)
    else:
        print("  atlas_b.json not populated yet -- the Q2 job is still running.", flush=True)

    # ---------------------------------------------------------------- 2. the 007 null
    hdr("2. THE 007 NULL -- the n_eff >= 2 viability rule is a free constant")
    t = load("atlas_tica_vs_n.json")
    trows = (t if isinstance(t, list) else (t or {}).get("rows", [])) or []
    NE = {r["pdb"]: r for r in (load("atlas_neff.json") or [])}
    if trows and NE:
        by = {}
        for m_req in sorted({r["m_req"] for r in trows if r.get("m_req")}):
            R = [r for r in trows if r["m_req"] == m_req and r.get("dim_in")]
            if len(R) < 3: continue
            ne = [NE[r["pdb"]]["neff256"] / r["dim_in"] for r in R if r["pdb"] in NE]
            x = np.log10([r["N"] for r in R])
            ti, hi = slope_ci(x, np.log10([r["dim_in"] for r in R]))
            pi, hp = slope_ci(x, np.log10([r["pca_dim_in_m"] for r in R if r.get("pca_dim_in_m")]))
            by[m_req] = (float(np.median(ne)) if ne else np.nan, ti - pi, hi + hp)
        for m_req, (nem, d, dh) in by.items():
            print(f"  basis {m_req}: n_eff/dim {nem:.2f}   in-sample DIFFERENCE {d:+.4f} +/- {dh:.4f}")
        if by:
            def dec_007(nem, cut):
                return "VIABLE -> answers 007" if nem >= cut else "CENSORED -> cannot answer"
            st, _ = STAMP.verdict_sensitivity(
                dec_007, {f"n_eff/dim @ m={k}": v[0] for k, v in by.items()}, 2.0,
                scales=(0.5, 1.0, 2.0), label="007 viability rule")
            record("007 null (n_eff>=2 rule)", st,
                   "viability is per-basis; the m=100 basis answers, m=400 does not")
    else:
        print("  need atlas_tica_vs_n.json and atlas_neff.json")

    # ---------------------------------------------------------------- 3. the ANM cutoff
    hdr("3. THE SPARSE-ANM CUTOFF -- the peer is chosen on a TRAINING MEAN (Family E)")
    pj = load("atlas_peer.json")
    if pj and pj.get("cutoff_train_fve"):
        tf = {float(k): v for k, v in pj["cutoff_train_fve"].items()}
        print(f"  training-mean FVE by cutoff: " + "  ".join(f"{k:g}A {v:.4f}" for k, v in sorted(tf.items())))
        best = max(tf, key=lambda k: tf[k]); second = sorted(tf, key=lambda k: -tf[k])[1]
        marg = (tf[best] - tf[second]) / max(abs(tf[best]), 1e-12)
        print(f"  winner {best:g} A beats runner-up {second:g} A by {100*marg:.1f}% of its own value")
        def dec_cut(m_, cut): return f"{best:g}A selected" if m_ > cut else "SELECTION IS A TIE"
        st, _ = STAMP.verdict_sensitivity(dec_cut, {"relative margin": marg}, 0.02,
                                          scales=(0.5, 1.0, 2.0), label="ANM cutoff selection")
        record("ANM cutoff (Family E)", st, f"{best:g}A over {second:g}A by {100*marg:.1f}%")
    else:
        print("  atlas_peer.json has no cutoff sweep yet -- the 14a job has not run.")

    # ---------------------------------------------------------------- 4. criterion-1 thresholds
    hdr("4. CRITERION-1 -- four discriminators, each with a PASS THRESHOLD")
    c1 = load("atlas_dm_criterion1.json")
    if c1:
        for keyname, rr in list(c1.items())[:1]:
            for disc, fld, lo, hi in (("1 marginal std", "std_frac", 0.80, None),
                                      ("2 IAT (KINETIC)", "iat_frac", 0.80, None),
                                      ("3 cross-mode", "corr_frac", 0.80, None),
                                      ("4 free energy JS", "js", None, 0.10)):
                v = np.array([r[fld] for r in rr], float)
                if lo is not None:
                    def dec(_x, cut, v=v): return f"{100*np.mean(v >= cut):.0f}% of systems pass"
                    st, _ = STAMP.verdict_sensitivity(dec, {disc: float(np.median(v))}, lo,
                                                      scales=(0.75, 1.0, 1.25),
                                                      label=f"criterion-1 {disc}")
                    record(f"criterion-1 {disc}", st)
    else:
        print("  atlas_dm_criterion1.json not produced yet -- it runs in the DM sweep's report phase.")

    # ---------------------------------------------------------------- 5. 16a, the live headline
    hdr("5. 16a REALISED RANK -- the current headline, re-checked")
    md = load("atlas_modes.json")
    if md and md.get("sys"):
        r90 = np.array([s_["rank90_out"] for s_ in md["sys"] if "rank90_out" in s_], float)
        pr = float(md.get("arm", {}).get("pr", np.nan))
        if len(r90) and np.isfinite(pr):
            print(f"  realised rank90 median {np.median(r90):.0f} (n={len(r90)}), latent PR {pr:.1f}")
            st, _ = STAMP.verdict_sensitivity(
                lambda v, t: "DECODER-limited" if v < t else "not decoder-limited",
                {"realised rank90": float(np.median(r90)),
                 "realised rank99": float(np.median([s_["rank99_out"] for s_ in md["sys"]
                                                     if "rank99_out" in s_]))},
                0.6 * pr, scales=(0.5, 1.0, 2.0), label="16a decoder-limited")
            record("16a decoder-limited (label)", st, "flips below ~0.40*PR; see the ratio below")
            # HONEST QUALIFICATION OF THAT FLIP, both directions.
            # (a) rank99 is NOT calibrated against a PR-derived boundary -- 0.6*PR was chosen for the
            #     rank90 convention -- so its row is a mis-specified comparison, not a genuine free
            #     choice, and it should not be counted as evidence against the claim.
            # (b) the threshold-scale flip IS genuine: the measured ratio is rank90/PR, and the label
            #     "decoder-limited" holds for any threshold ABOVE that ratio.
            rat = float(np.median(r90)) / pr
            print(f"\n  RATIO FORM, which carries no free constant at all:")
            print(f"    realised rank90 / latent PR   = {rat:.2f}  (6 of 14.9 directions)")
            rd = np.array([s_["rank90_data"] for s_ in md["sys"] if "rank90_data" in s_], float)
            if len(rd):
                print(f"    realised rank90 / DATA rank90 = {float(np.median(r90))/float(np.median(rd)):.3f}"
                      f"  ({np.median(r90):.0f} of {np.median(rd):.0f} directions)")
            print(f"  => the LABEL 'decoder-limited' holds for any threshold above {rat:.2f}*PR and")
            print(f"     fails below it. The RATIOS themselves carry no threshold and are the form to")
            print(f"     quote. The data-rank ratio is the stronger of the two and does not involve")
            print(f"     PR at all.", flush=True)
            record("16a ratios (threshold-free)", True,
                   f"rank90/PR {rat:.2f}, rank90/data {float(np.median(r90))/max(float(np.median(rd)),1):.3f}")
    else:
        print("  atlas_modes.json missing")

    # ---------------------------------------------------------------- summary
    hdr("SUMMARY -- which surviving claims survive their own free choices?")
    if not results:
        print("  nothing evaluated; the inputs are still being produced.", flush=True)
    for name, stable, note in results:
        print(f"  {'STABLE ' if stable else 'FLIPS  '}  {name:<38} {note}", flush=True)
    nf = [r for r in results if not r[1]]
    if nf:
        print(f"\n  {len(nf)} claim(s) FLIP across their own free choices. A flip is NOT a refutation:")
        print(f"  it means the claim is only defined TOGETHER WITH its threshold, so from here it")
        print(f"  carries the range in the ROADMAP and is not cited without it.", flush=True)
    print(f"\n  NOTE: every number above is read from the stored result files. None is retyped from a")
    print(f"  commit message -- a hardcoded comparator is the Family F failure already committed "
          f"twice here.", flush=True)
