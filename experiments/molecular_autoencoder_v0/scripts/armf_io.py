"""INBOX 74b: results files that declare whether they are FINISHED.

THE FAILURE THIS EXISTS TO PREVENT. Nearly every sweep in this project writes its
results incrementally -- `rows.append(rec); json.dump(rows, open(OUT, "w"))` inside
the loop -- so that a run which dies still leaves what it had. That is the right
instinct and it has a sharp edge.

The corrupt file is the SAFE case. A kill landing inside the write leaves invalid
JSON, and `json.load` raises: the failure is loud and nobody acts on it.

The dangerous case is the file written *between* items. It is valid, it parses, it
looks finished, and it holds a PREFIX of the work. And the work is ordered: at
`armf_tied_peer.py:98` the held-out systems are sorted ascending in N, deliberately,
so the small-system answer lands before the large-N tail finishes. A job killed
mid-run therefore leaves a results file whose missing rows are exactly the large
systems -- an exclusion perfectly correlated with the regressor, which is FAMILY A,
manufactured by the scheduler rather than by any filter in the analysis.

`os.replace` does not fix this. Atomicity guarantees a complete WRITE; it says
nothing about a complete RUN. Both are needed and they are different guarantees.

THE CONTRACT.
  - a producer calls `dump_rows(path, rows, n_expected)` every iteration, and
    `dump_rows(..., complete=True)` on the final one;
  - an ANALYSIS calls `load_complete(path)`, which raises unless the file says it
    finished -- the way `ckpt_path()` raises rather than resolving a neighbour;
  - a RESUME calls `load_partial(path)`, which accepts anything and reports what it
    got. A partial file is legitimate input to a resume and never to an analysis,
    and only the caller knows which it is doing.

BACKWARD COMPATIBILITY. Every results file written before this module is a bare
list or dict with no declaration, and there are 237 of them in the repo alone. Those
cannot be proven complete OR incomplete, so `load_complete` refuses them by default
and takes `allow_undeclared=True` to read one anyway -- loudly, naming the risk. The
point is that a legacy read becomes visible, not that it becomes impossible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ENVELOPE_VERSION = 1
_KEYS = ("armf_io_version", "complete", "n_expected", "n_present", "rows")


def _jsonable(o):
    """Numpy scalars and arrays become JSON numbers. Anything else RAISES.

    `json.dump(..., default=str)` is the obvious thing to write and it is a trap in a
    results file -- but NOT for the type you would first suspect. Measured, because I
    guessed wrong about which types are affected:

        np.float64   subclasses float -> json serialises it directly, never reaching
                     the fallback. It was always safe.
        np.int64     does NOT subclass int   -> reaches the fallback
        np.bool_     does NOT subclass bool  -> reaches the fallback
        np.ndarray   reaches the fallback

    So under `default=str` a COUNT or a FLAG became the string "7" or "True", which
    reloads without error, prints identically, and then compares as text. That is the
    same shape as every other defect on this record: a value silently the wrong KIND
    of thing while looking right. These convert; anything genuinely unserialisable
    raises at write time, which is loud, immediate and fixable in a line.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(
        f"{type(o).__name__} is not JSON-serialisable and armf_io will not coerce it to a string. "
        f"Convert it at the call site so the results file holds a number rather than text that "
        f"looks like one. Value: {o!r}")


def _write_atomic(path, payload):
    """Temp file + os.replace: the destination is the old file or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, default=_jsonable)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def dump_rows(path, rows, n_expected, complete=False, n_failed=0, n_present=None, **meta):
    """Write `rows` under a declaration of whether the run finished.

    `rows` may be a list or a dict keyed by item id -- both are preserved as-is, so
    a caller can switch to this without changing how it accumulates.

    `n_failed` matters and is not cosmetic. Every sweep here skips items that raise
    (`armf_tied_peer.py` prints the failure and continues), so a run that reaches the
    end of its work list legitimately holds FEWER rows than it expected. Without a
    failure count, "finished with 120 of 123" is indistinguishable from "killed at
    120 of 123" -- which is the exact confusion this module exists to remove. With
    it, the reader can check the project's own conservation rule: every item is
    either present or explicitly accounted for.

    `n_present` overrides the default `len(rows)`, for producers whose results are
    NESTED rather than flat -- `armf_scale_test.py` keys by arm and then by system,
    so `len(rows)` counts arms while the unit of work is the (arm, system) cell.
    Passing the cell count keeps the conservation check meaningful instead of
    comparing two different units, which would be the same "one name, two things"
    confusion this envelope exists to prevent.
    """
    payload = {
        "armf_io_version": ENVELOPE_VERSION,
        "complete": bool(complete),
        "n_expected": int(n_expected),
        "n_present": int(len(rows) if n_present is None else n_present),
        "n_failed": int(n_failed),
        # INBOX 73b -- a requeued run is a second producer under one name.
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "rows": rows,
    }
    payload.update({k: v for k, v in meta.items() if k not in _KEYS})
    _write_atomic(path, payload)


def _read(path):
    with open(path) as f:
        return json.load(f)


def is_enveloped(obj):
    return isinstance(obj, dict) and "armf_io_version" in obj and "rows" in obj


def load_complete(path, allow_undeclared=False):
    """Return the rows, or raise if the file does not declare that it finished.

    Raises rather than warning, on purpose. A warning printed into a log nobody
    reads is how 21c/27a/27b went unread, and this failure produces a NUMBER rather
    than an error, which makes it worse than the ones a warning was enough for.
    """
    path = Path(path)
    obj = _read(path)

    if not is_enveloped(obj):
        if allow_undeclared:
            n = len(obj)
            print(f"[armf_io] {path.name} predates the completeness envelope. Read anyway on the "
                  f"caller's say-so: {n} rows, completeness UNKNOWN. If this file was produced by "
                  f"a job that was killed, the missing rows are the LARGE systems (INBOX 74b).")
            return obj
        raise ValueError(
            f"{path} has no completeness declaration -- it predates INBOX 74b, so it cannot be "
            f"shown to be a finished run. A results file truncated by a scheduler kill is valid "
            f"JSON and is missing its large-N tail, which is a Family A exclusion. Re-run the "
            f"producer, or pass allow_undeclared=True to accept the risk explicitly.")

    if not obj.get("complete"):
        raise ValueError(
            f"{path} declares complete=false: {obj.get('n_present')} of {obj.get('n_expected')} "
            f"rows. This is a PARTIAL run and must not be analysed -- the work is ordered, so the "
            f"missing rows are not a random subset. Use load_partial() if you are resuming.")

    # CONSERVATION OF n. A finished run may hold fewer rows than it expected, because
    # items that raise are skipped and logged. What it may NOT do is lose track of
    # them: present + failed must account for expected. A shortfall that is not
    # explained by the failure count means rows went missing without anything
    # recording it, and a silently short file is the whole hazard here.
    n_exp = obj.get("n_expected")
    n_got = obj.get("n_present")
    n_bad = obj.get("n_failed", 0) or 0
    if n_exp is not None and n_got is not None and n_got + n_bad != n_exp:
        raise ValueError(
            f"{path} says complete=true but {n_got} present + {n_bad} failed does not account for "
            f"{n_exp} expected -- {n_exp - n_got - n_bad} rows are unaccounted for. The producer's "
            f"own count disagrees with its own verdict; do not trust either.")
    if n_bad:
        print(f"[armf_io] {path.name}: {n_got} rows, {n_bad} items failed and were logged. The "
              f"analysis is on {n_got} of {n_exp} -- check the failures are not ordered.")

    rc = obj.get("slurm_restart_count") or 0
    if rc:
        print(f"[armf_io] {path.name} was produced by a job requeued {rc}x -- it is not a single "
              f"uninterrupted run (INBOX 73b).")
    return obj["rows"]




def load_partial(path):
    """For RESUME paths only. Returns (rows, complete, n_expected).

    Accepts anything, including a legacy bare file and a file that does not parse --
    a resume that cannot read the previous attempt should start over, not crash.
    """
    path = Path(path)
    if not path.exists():
        return ([], False, None)
    try:
        obj = _read(path)
    except json.JSONDecodeError as exc:
        print(f"[armf_io] {path.name} does not parse ({exc}); starting from nothing. This is the "
              f"expected shape of a kill landing inside a write.")
        return ([], False, None)
    if not is_enveloped(obj):
        return (obj, False, None)
    return (obj["rows"], bool(obj.get("complete")), obj.get("n_expected"))
