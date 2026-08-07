"""INBOX 26c: THE TEST THAT WOULD HAVE CAUGHT BOTH FAILURES -- it calls the CALLER.

TWICE IN ONE SESSION the same gap produced a job that failed on every system:
  * `FVE_perp` was validated standalone against a dense projection-matrix computation to 1.1e-16,
    and submitted without ever calling `mode_table()`. It failed because the new block shadowed two
    variables the same function returns.
  * the ctx message-passing path was unit-tested at N=400 / B=1, a size where its O(B*N^2*dm) term is
    41 MB, and OOMed at 45 GiB on the first real step.

Both times the tests covered the KERNEL and nothing called the CALLER. "Run the function next time"
does not survive a third instance, so this is the structural version: for every metric-producing
entry point, invoke IT -- not its mathematics -- on the smallest real fixture available, and do it
before submitting. Cheap enough to be habitual, which is the only property that matters.

    python armf_smoke.py            # all registered entry points
    python armf_smoke.py modes      # one
"""
import sys, os, json, traceback, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"


def _tiny_system():
    from armf_atlas_data import AtlasStore, sysdata
    import armf_atlas_dm as D
    store = AtlasStore(f"{WR}/atlas_cache")
    man = json.load(open(D.MAN))
    have = {m["pdb"]: i for i, m in enumerate(store.meta)}
    ho = [p for p in man["heldout"] if p in have]
    smallest = min(ho, key=lambda z: store.meta[have[z]]["atoms"])
    return sysdata(store, have[smallest])


def smoke_modes():
    """armf_atlas_modes.mode_table -- the caller that consumes FVE_perp, per-atom and rank."""
    import armf_atlas_modes as M, armf_atlas_dm as D
    d = _tiny_system()
    m = D.Codec(d["stat"].shape[1], 1, 64).to(M.dev); m.eval()
    r = M.mode_table(m, d, nf=40)
    assert len(r["fve"]) == len(r["err_ratio"]) > 0, "per-mode arrays broken"
    for k in ("rank90_out", "rank90_data", "peratom_med", "peratom_p90"):
        assert k in r and np.isfinite(r[k]), f"missing/NaN {k}"
    assert any(k.startswith("fve_perp_anm") for k in r), "FVE_perp absent"
    return f"{len(r['fve'])} modes, rank90_out={r['rank90_out']}, peratom_med={r['peratom_med']:.3f}"


def smoke_mediator():
    """armf_tied_mediator.ratio_for -- the caller that produces the 26a mediator ratio."""
    import armf_tied_mediator as T, armf_atlas_dm as D
    from armf_modal_decoder import ModalCodec
    d = _tiny_system()
    T.dev = D.dev
    m = ModalCodec(d["stat"].shape[1], 1, 64, tie_encoder=True).to(D.dev); m.eval()
    v = T.ratio_for(m, d, nf=6)
    # ratio_for returns (analysis, synthesis) since the 26a follow-up; assert the CONTRACT, not just
    # a value -- this test caught the tuple change the moment it was made, which is the point of it.
    assert isinstance(v, tuple) and len(v) == 2, f"expected (z_ratio, out_ratio), got {type(v)}"
    zr, orr = v
    assert np.isfinite(zr) and zr > 0, f"bad z ratio {zr}"
    assert np.isfinite(orr) and orr >= 0, f"bad decoded ratio {orr}"
    return f"||z||/||disp|| = {zr:.4f}, ||dec||/||disp|| = {orr:.4f}"


def smoke_modal_ctx():
    """armf_modal_ctx.GraphModalCodec forward AT A SIZE WHERE THE QUADRATIC TERM WOULD SHOW."""
    import armf_modal_ctx as C
    Fs, N, B, dm = 12, 3000, 8, 128
    stat1 = torch.randn(1, N, Fs); stat1[0, :, -3:] = torch.randn(N, 3) * 10
    stat = stat1.expand(B, -1, -1)
    m = C.GraphModalCodec(Fs, 1, dm, ctx_layers=4, tie_encoder=False)
    o = m(stat, torch.randn(B, N, 3) * 0.1)
    assert tuple(o.shape) == (B, N, 3)
    return f"B={B} N={N} ctx=4 forward OK (old path would need {B*N*N*dm*4/1e9:.0f} GB)"


def smoke_peer_report():
    """armf_atlas_peer.report_cutoffs -- a branch no truncated run reaches."""
    import armf_atlas_peer as P
    n = 30
    pdbs = [f"s{i}" for i in range(n)]
    cur = list(zip(pdbs, [600 * 1.2 ** i for i in range(n)]))
    S = {p: {"anm_ok": True, **{f"anm256_c{c:g}": 0.6 for c in P.CUTOFF_SWEEP}} for p in pdbs}
    joined = [dict(codec=0.15, dm=256, n_train=50, codec_per=[0.15] * n)]
    v = P.report_cutoffs(joined, S, cur, log=lambda *a: None)
    assert len(v) == len(P.CUTOFF_SWEEP), "not all cutoffs reported"
    return f"{len(v)} cutoffs reported"


TESTS = {"modes": smoke_modes, "mediator": smoke_mediator,
         "modal_ctx": smoke_modal_ctx, "peer_report": smoke_peer_report}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TESTS)
    bad = 0
    for name in want:
        fn = TESTS.get(name)
        if fn is None:
            print(f"  {name:<12} NO SUCH TEST"); bad += 1; continue
        try:
            print(f"  {name:<12} PASS  {fn()}", flush=True)
        except Exception:
            bad += 1
            print(f"  {name:<12} FAIL", flush=True)
            traceback.print_exc()
    print(f"\n  {len(want)-bad}/{len(want)} entry points OK")
    sys.exit(1 if bad else 0)
