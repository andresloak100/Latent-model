#!/usr/bin/env python3
"""INBOX 82a: read the propagator sweep WITHOUT pooling powered and unpowered domains.

WHY A SEPARATE FILE FROM THE PRODUCER. The power check is per domain and per tau, and its point is
that a table averaging across it is not readable. In the producer the split would exist only if the
producer finished; here the same rule applies to a partial results file, which is what a resumable
28-domain run mostly has.

THE ARGUMENT, kept next to the code implementing it. OU is a negative control because it is
independent per mode in the ANM basis, so xcorr and amp are ~0 for it BY CONSTRUCTION. That makes it
WRONG only on a domain whose reference actually HAS coupling. On a rigid domain with xcorr_r near
zero OU is RIGHT, the negative control quietly stops being one, and its passing means nothing.
Pooling such a domain into "DDPM consistent on 6 of 7 across 28 domains" averages domains where the
test can discriminate with domains where it cannot -- the dilution error caught in 071 (0.62 dB/bit
pooled against 4.42 while actually coding), and 59e's shape, already forced out of the oracle sweep
once.

RELEVANCE BOUND, REGISTERED BEFORE THE 28-DOMAIN RESULTS EXIST (82a asked for it while free):

    real coupling  <=>  xcorr_r >= max(ABS_FLOOR, one reference band width on xcorr)

Two terms because they fail differently. The band-width term is RESOLVABILITY: coupling below the
spread of the reference's own windows cannot be told from zero by this instrument whatever its
absolute size. The absolute floor is RELEVANCE: an arbitrarily tight band would otherwise certify an
arbitrarily small coupling as real. On 2cndA01, xcorr_r = 0.2483 against a band width of 0.068 --
3.6 band widths and 5x the floor -- so the threshold is not chosen to fit.

A domain failing the bound is NOT a failed power check. It is the pre-registered Gaussian branch
firing FOR THAT DOMAIN: the task is Gaussian/single-basin at that lag and no learned propagator is
needed there. A result about the system, reported as one.
"""
import sys, os, json
import numpy as np

WR = os.environ["WR"]
RES = os.environ.get("PROP_RES", f"{WR}/propagator_tauK.json")
ABS_FLOOR = 0.05
METRICS = ["std", "js", "kurt", "xcorr", "amp", "iat", "trans"]


def classify(rows):
    """Per (domain, tau): has the reference resolvable, relevant coupling, and did OU fail on it?"""
    out = {}
    for r in rows:
        if r.get("model") != "POWER_CHECK":
            continue
        bw = r.get("band_width", {}).get("xcorr")
        xr = r.get("xcorr_r")
        if bw is None or xr is None:
            continue
        thr = max(ABS_FLOOR, bw)
        coupled = xr >= thr
        # POWERED needs BOTH: coupling for the control to be wrong about, AND OU actually failing.
        # Either alone is not a working negative control.
        out[(r["dom"], r["tau"])] = dict(xcorr_r=xr, band_xcorr=bw, thr=thr, coupled=bool(coupled),
                                         has_power=bool(r.get("has_power")),
                                         powered=bool(coupled and r.get("has_power")))
    return out


if __name__ == "__main__":
    if not os.path.exists(RES):
        raise SystemExit(f"  no results at {RES}")
    d = json.load(open(RES))
    rows = [r for v in d.values() for r in v]
    cls = classify(rows)
    taus = sorted({t for _, t in cls})
    print(f"[82a] propagator sweep read WITHOUT pooling. {len(d)} domains persisted, "
          f"{len(cls)} (domain,tau) cells carrying a power check.", flush=True)
    print(f"  relevance bound registered in advance: real coupling <=> xcorr_r >= "
          f"max({ABS_FLOOR}, band width on xcorr)", flush=True)

    print(f"\n=== HOW MUCH OF mdCATH CAN ADDRESS THE COUPLING CLAIM AT ALL (82a.2) ===")
    print(f"  {'tau':>5}{'cells':>7}{'coupled':>9}{'OU fails':>10}{'POWERED':>9}{'Gaussian':>10}")
    powered = {}
    for t in taus:
        cs = [v for (dm, tt), v in cls.items() if tt == t]
        powered[t] = {dm for (dm, tt), v in cls.items() if tt == t and v["powered"]}
        print(f"  {t:>5}{len(cs):>7}{sum(v['coupled'] for v in cs):>9}"
              f"{sum(v['has_power'] for v in cs):>10}{len(powered[t]):>9}"
              f"{sum(not v['coupled'] for v in cs):>10}")
    print(f"  'Gaussian' = reference coupling below the bound -> the task is Gaussian/single-basin "
          f"at that\n  lag for that domain and NO LEARNED PROPAGATOR IS NEEDED there. A result about "
          f"the system,\n  not a defect in the test.", flush=True)

    print(f"\n=== ARMS, SPLIT BY POWER, NEVER POOLED (82a.3) ===")
    for t in taus:
        for model in ("OU", "DDPM-absolute", "DDPM-delta"):
            for lab, want in (("POWERED  ", True), ("unpowered", False)):
                sel = [r for r in rows if r.get("model") == model and r.get("tau") == t
                       and ((r["dom"] in powered[t]) == want)]
                if not sel:
                    continue
                div = sum(1 for r in sel if r.get("cg_diverged"))
                use = [r for r in sel if not r.get("cg_diverged")]
                if not use:
                    print(f"  tau={t} {model:<14} {lab} n={len(sel):>3}  ALL {div} DIVERGED "
                          f"(cg>=1e3) -- no readable cell", flush=True)
                    continue
                ag = np.array([r["agree"] for r in use])
                per = {k: sum(1 for r in use if r["metrics"][k]["inside"]) for k in METRICS}
                extra = f" (+{div} diverged, excluded)" if div else ""
                print(f"  tau={t} {model:<14} {lab} n={len(use):>3}{extra}"
                      f"  agree median {np.median(ag):.1f}/7", flush=True)
                print(f"        inside: " + "  ".join(f"{k} {per[k]}/{len(use)}" for k in METRICS),
                      flush=True)
        print()

    if not any(powered.values()):
        print("  NO DOMAIN IS POWERED AT ANY LAG. Nothing about any DDPM is readable from this "
              "sweep;\n  the honest output is the band width, not a table -- 81a's own branch.",
              flush=True)
    else:
        n_un = sum(len({dm for (dm, tt) in cls if tt == t}) - len(powered[t]) for t in taus)
        if n_un:
            print(f"  {n_un} (domain,tau) cells are UNPOWERED. A DDPM result there is not a weaker "
                  f"finding,\n  it is not a finding, and it sits on its own line above rather than "
                  f"being averaged in.", flush=True)
