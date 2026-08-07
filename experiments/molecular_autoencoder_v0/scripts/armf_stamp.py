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
    if rel > 0.25:
        log(f"  [18d] *** THE STORED SAMPLE IS SKEWED IN {name}. Completing this run fixes it; "
            f"REPORTING BEFORE IT COMPLETES WOULD NOT. Any table produced from a partial run is a "
            f"{name}-correlated exclusion of the axis under test. ***")
        return True
    return False
