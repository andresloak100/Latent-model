"""INBOX 18a: STAMP EVERY ARM AT WRITE TIME, so the next unreproducible result names its own cause.

WHY THIS EXISTS. Three DM-sweep arms (`atlas_dm.json` rows 6/9/10) turned out to be unreproducible:
LEGACY hyperparameters in the current codebase give all-123 FVE +0.1352 stopping at 20,000 steps,
where the stored rows recorded +0.1553 at 15,000. Three candidate causes were checked and excluded --
model-init RNG (`down`/`up` are None at `dlat == dm`, so initialisation is bit-identical), the
N-stratified tracked set (predates those arms), and training-set membership (the cache fills
`heldout + train_ordered` in manifest order, so `tr_ids[:50]` was the same 50 systems at 271 cached as
at 697). The residual cause is STILL UNIDENTIFIED after a full investigation, and the only evidence
that dated the rows at all was the incidental absence of a `z0_frac` field.

A per-arm stamp would have named it in seconds. That is the whole argument: hunting one instance has
poor expected value, and the structural fix catches the instances nobody predicted.

WHAT IS HASHED, AND WHY NOT THE REPO SHA. Keying invalidation on the repository HEAD would retrain
every arm on every commit, including commits that touch only documentation -- so the gate would be
turned off within a day, which is worse than not having it. Instead `code` is a NORMALISED hash of the
source of the objects actually responsible for a result (the model class, the training loop): parsed
to an AST with DOCSTRINGS STRIPPED and dumped, so it is invariant to comments, docstrings, blank lines
and formatting, and changes exactly when behaviour can change. The repo SHA is recorded ALONGSIDE for
forensics, never used for gating.

USE:
    ST = stamp(dict(warmup=WARMUP, maxsteps=MAXSTEPS, ...), Codec, train)
    rec["stamp"] = ST
    ...
    if not same_stamp(row, ST): -> recompute, or exclude from the table
"""
import os, sys, json, hashlib, subprocess, ast, inspect

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _sha(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def code_hash(*objs):
    """Normalised hash of the SOURCE of each object. Invariant to comments, docstrings and
    formatting; changes when behaviour can. Falls back to the raw source if parsing fails --
    a coarser hash is still better than none, and the fallback is visible in the value."""
    parts = []
    for o in objs:
        try:
            src = inspect.getsource(o)
        except Exception:
            parts.append(f"<nosource:{getattr(o, '__name__', o)}>"); continue
        try:
            parts.append(ast.dump(_strip_docstrings(ast.parse(src.lstrip()))))
        except Exception:
            parts.append("RAW:" + src)
    return _sha("\n".join(parts))


def git_info():
    def run(*a):
        try:
            return subprocess.run(a, cwd=_REPO, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    sha = run("git", "rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(run("git", "status", "--porcelain"))
    return sha, dirty


def stamp(config, *code_objs):
    """config: the EFFECTIVE hyperparameters actually used -- not the ones nominally set.
    Passing the module constants when a caller has overridden them at runtime is the failure this
    field exists to catch, so callers should build it from the values they pass to the trainer."""
    sha, dirty = git_info()
    try:
        import torch
        tv = torch.__version__
        dv = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = torch.cuda.get_device_name(0) if dv == "cuda" else ""
    except Exception:
        tv, dv, gpu = "", "", ""
    cfg = json.dumps({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                      for k, v in sorted(config.items())}, sort_keys=True)
    return dict(git=sha, dirty=dirty, cfg=_sha(cfg), cfg_raw=cfg,
                code=code_hash(*code_objs) if code_objs else "",
                torch=tv, dev=dv, gpu=gpu)


def key(st):
    """The part of a stamp that GATES comparison: effective config + normalised code.
    `git`/`dirty` are forensic only -- gating on them would retrain on documentation commits."""
    return (st.get("cfg", ""), st.get("code", ""))


def same_stamp(row, st):
    rs = row.get("stamp")
    return bool(rs) and key(rs) == key(st)


def report(rows, st, label="rows", log=print):
    """Print how many stored rows match the current stamp, and refuse silence about the rest."""
    same = [r for r in rows if same_stamp(r, st)]
    diff = [r for r in rows if not same_stamp(r, st)]
    log(f"  [stamp] git {st['git']}{'+dirty' if st['dirty'] else ''}  cfg {st['cfg']}  "
        f"code {st['code']}  torch {st['torch']} {st['dev']}")
    if diff:
        seen = {}
        for r in diff:
            k = tuple(key(r["stamp"])) if r.get("stamp") else ("<unstamped>", "")
            seen[k] = seen.get(k, 0) + 1
        log(f"  [stamp] {len(same)}/{len(rows)} stored {label} match the current stamp; "
            f"{len(diff)} DO NOT and are EXCLUDED from every table:")
        for k, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            log(f"            {n:>4} with cfg/code {k[0]}/{k[1]}")
        log(f"  [stamp] INBOX 18a: arms that cannot be reproduced by the current configuration are "
            f"not evidence. Re-run to regenerate them.")
    else:
        log(f"  [stamp] all {len(rows)} stored {label} match the current stamp.")
    return same, diff


def coverage_by(stored_vals, missing_vals, name="N", log=print):
    """INBOX 18d: A RESUME PATH IS AN EXCLUSION FILTER.

    When work is ordered by a regressor -- the peer loop runs ASCENDING IN N by design -- a partial
    run leaves a SIZE-BIASED sample stored, and skip-if-exists then reports on whatever the previous
    run happened not to reach. That is a Family A exclusion arriving through INFRASTRUCTURE rather
    than through any analysis choice, and it is invisible in the analysis code. It already nearly
    happened here: 49 stored systems spanning N 598-2,308, the smallest 40%.

    So before resuming, put the regressor distribution of STORED against MISSING on the record."""
    import numpy as np
    s = np.asarray(sorted(stored_vals), float); m = np.asarray(sorted(missing_vals), float)
    if not len(s) or not len(m):
        return False
    ms, mm = float(np.median(s)), float(np.median(m))
    allv = np.concatenate([s, m]); rel = abs(ms - mm) / max(float(np.median(allv)), 1e-9)
    log(f"  [18d] resume coverage by {name}: stored n={len(s)} median {ms:.0f} "
        f"[{s.min():.0f}-{s.max():.0f}]   missing n={len(m)} median {mm:.0f} "
        f"[{m.min():.0f}-{m.max():.0f}]   relative median gap {rel:.2f}")
    # TAIL TRUNCATION, added after the median test PASSED on a genuinely truncated sample.
    # atlas_b at 706/841 had stored median 2,804 against missing median 2,693 -- a 0.04 gap, "not
    # skewed" -- while the stored MAXIMUM was 15,673 and the missing maximum 33,377. The entire top
    # of the range was absent and a median comparison cannot see it. For a slope-vs-N fit that is the
    # WORST case, not a mild one: leverage lives at the extremes, so losing the top shortens the
    # fitted range and can move the exponent by more than any interior shift.
    trunc = False
    if m.max() > s.max():
        lost = np.log10(m.max() / max(s.max(), 1e-9))
        span = np.log10(max(allv.max(), 1e-9) / max(allv.min(), 1e-9))
        log(f"  [18d] TAIL: stored max {s.max():.0f} vs missing max {m.max():.0f} -- the top "
            f"{lost:.2f} of {span:.2f} decades ({100*lost/max(span,1e-9):.0f}% of the log-range) is "
            f"NOT in the stored sample")
        if lost / max(span, 1e-9) > 0.10:
            log(f"  [18d] *** THE TOP OF THE {name} RANGE IS MISSING. Any slope fitted now is fitted "
                f"on a TRUNCATED range, and leverage lives at the extremes -- this is the Family A "
                f"pattern that matters, and a median test cannot see it. ***")
            trunc = True
    if trunc or rel > 0.25:
        log(f"  [18d] *** THE STORED SAMPLE IS SKEWED IN {name}. Completing this run fixes it; "
            f"REPORTING BEFORE IT COMPLETES WOULD NOT. Any table produced from a partial run is a "
            f"{name}-correlated exclusion of the axis under test. ***")
        return True
    return False


# ---------------------------------------------------------------------------
# INBOX 19a: VERDICT SENSITIVITY, PRINTED UNCONDITIONALLY.
#
# FAMILY G's real statement, in the form the evidence supports:
#   every instance was a verdict AUTOMATED TO GUARD AGAINST BIAS, and the automation moved the bias
#   from the conclusion into the THRESHOLD, where it is harder to see.
# Naming the metric does not close that hole -- the next instance will have a defensible metric and
# an arbitrary constant. What closes it is showing what the verdict WOULD have been across the
# plausible range of every free choice the verdict contains, whether or not anything looks wrong.
#
# All three instances from 2026-08-06/07 fail this on sight:
#   16a   threshold 5.96 against a measured 6.0  -> flips at any nearby threshold
#   007   pooled n_eff 2.045 against 2.0         -> flips when assessed PER BASIS
#   proc  gap>spread on best_track               -> flips when decided on full_fve
# ---------------------------------------------------------------------------


def verdict_sensitivity(decide, metrics, threshold, scales=(0.5, 1.0, 2.0), log=print,
                        label="verdict"):
    """decide(metric_value, threshold_value) -> a short verdict string.

    metrics:   {name: measured_value} -- EVERY metric this verdict could defensibly have used.
    threshold: the chosen constant.
    Returns (stable: bool, grid). Prints the full grid and, when the verdict is not stable across
    the range, says plainly that it is a measurement plus an opinion."""
    names = list(metrics)
    grid = {(m, s): str(decide(metrics[m], threshold * s)) for m in names for s in scales}
    vals = sorted({v for v in grid.values()})
    stable = len(vals) == 1
    w = max([len(m) for m in names] + [12])
    log(f"  [19a sensitivity] {label}: what would have been concluded across every free choice")
    log(f"    {'metric':<{w}}" + "".join(f"{'x' + format(s, 'g'):>26}" for s in scales))
    for m in names:
        log(f"    {m:<{w}}" + "".join(f"{grid[(m, s)]:>26}" for s in scales))
    for m in names:
        v = metrics[m]
        marg = abs(v - threshold) / max(abs(threshold), 1e-12)
        flag = "  <-- WITHIN 10% OF THE BOUNDARY" if marg < 0.10 else ""
        log(f"    margin, {m}: measured {v:+.4f} vs boundary {threshold:+.4f} -> {100*marg:.0f}%{flag}")
    if stable:
        log(f"    -> STABLE across all {len(grid)} combinations. The verdict survives its own free "
            f"choices.")
    else:
        log(f"    -> *** THE VERDICT FLIPS across the plausible range ({len(vals)} distinct outcomes). "
            f"IT IS NOT A VERDICT: it is a MEASUREMENT PLUS AN OPINION, and must be read as one. ***")
    return stable, grid


# ---------------------------------------------------------------------------
# FAMILY C, AS A FUNCTION: "fails to reject zero" is not "is zero".
#
# Written after the same inference was found compiled into three separate verdict branches:
#     armf_atlas_modes  27b  "the code decay adds nothing once N is held -- they are SEPARATE"
#     armf_tied_mediator     "flat (mechanism absent)"
#     armf_modal_ctx    23b  "reach is NOT the constraint"
# each firing on `abs(estimate) <= halfwidth` alone. That is believing an underpowered null, and in
# the 27b case it fired where the non-significant coefficient had the LARGER point estimate.
#
# The correct form was already in armf_tica_vs_n's pre-registered read, which required the CI to
# INCLUDE zero *and* EXCLUDE the effect that would have mattered. Rejecting the alternative is what
# licenses a null; failing to reject zero licenses nothing. So every caller must now NAME the
# effect size it would care about, and gets one of three answers rather than two.
# ---------------------------------------------------------------------------


def null_verdict(estimate, halfwidth, relevant, label="effect"):
    """Three-way read. `relevant` = the smallest effect that would change the conclusion.

    EXCLUDES_ZERO  the CI excludes 0                      -> a real effect
    EQUIVALENT     the CI excludes +/-`relevant`          -> a genuine null, bounded by `relevant`
    NOT_RESOLVABLE the CI contains both 0 and `relevant`  -> underpowered; NOTHING is licensed
    """
    lo, hi = estimate - halfwidth, estimate + halfwidth
    ratio = abs(estimate) / max(halfwidth, 1e-30)
    if lo > 0 or hi < 0:
        return "EXCLUDES_ZERO", ratio, (lo, hi)
    if abs(relevant) > 0 and lo > -abs(relevant) and hi < abs(relevant):
        return "EQUIVALENT", ratio, (lo, hi)
    return "NOT_RESOLVABLE", ratio, (lo, hi)


def null_report(estimate, halfwidth, relevant, label="effect", log=print):
    """Print the three-way read with the words that match it. Returns the verdict string."""
    v, ratio, (lo, hi) = null_verdict(estimate, halfwidth, relevant, label)
    log(f"    {label}: {estimate:+.4f} +/- {halfwidth:.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
        f"|effect|/half-width {ratio:.2f}")
    if v == "EXCLUDES_ZERO":
        log(f"      -> REAL EFFECT (CI excludes zero).")
    elif v == "EQUIVALENT":
        log(f"      -> BOUNDED NULL: the CI also excludes +/-{abs(relevant):.4f}, the smallest effect "
            f"that would matter, so 'no effect' is licensed AT THAT BOUND.")
    else:
        log(f"      -> NOT RESOLVABLE. The CI contains BOTH zero AND {abs(relevant):+.4f}, so it is "
            f"consistent with no effect AND with one that matters.")
        log(f"         'No effect' is NOT licensed -- that would be believing an underpowered null.")
    return v
