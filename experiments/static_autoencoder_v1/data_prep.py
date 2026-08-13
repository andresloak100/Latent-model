#!/usr/bin/env python3
"""AFDB mmCIF -> a flat coordinate shard. Parsing only; no featurisation of any kind.

Each structure contributes one contiguous crop of NA atoms: the element symbol as a small integer
and the raw (x, y, z) in Angstroms, exactly as the file gives them. Nothing is aligned, rotated,
scaled, or reordered. Structures with fewer than NA atoms are skipped rather than padded, so every
row in the shard is a full-length crop and no mask logic can silently change what is measured.

The crop start is drawn once per structure with a fixed seed and stored, so the shard is a fixed
dataset rather than an augmentation stream -- a scaling comparison needs every configuration to see
the identical data.
"""
import os, sys, glob, json, time
import numpy as np

OUT = os.environ.get("SAE_SHARD", "/network/scratch/j/jacob-junqi.tian/static-ae-v1/shard")
SRC = os.environ.get("SAE_SRC", "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/afdb_raw")
NA = int(os.environ.get("SAE_NA", "256"))          # context window, in atoms
NSTRUCT = int(os.environ.get("SAE_NSTRUCT", "200000"))
SEED = int(os.environ.get("SAE_SEED", "0"))

# Element symbols are read from the file and mapped to ids by first appearance order, capped. This
# is an inventory of what the data contains, not a chemistry-derived ordering.
ELEMENTS = ["C", "N", "O", "S", "P", "SE", "FE", "ZN", "MG", "CA", "NA", "CL", "K", "MN", "CU"]
EL2I = {e: i + 1 for i, e in enumerate(ELEMENTS)}   # 0 reserved for unknown/pad


def parse_cif(path):
    """Return (elem_ids int8 [n], xyz float32 [n, 3]) from an mmCIF _atom_site loop.

    The loop header is read to find the columns rather than assuming a fixed layout, because column
    order is not guaranteed by the format even within one source.
    """
    cols, in_loop, rows = {}, False, []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, {}
                cols[line.strip().split(".", 1)[1]] = len(cols)
                continue
            if in_loop:
                if line.startswith(("ATOM", "HETATM")):
                    rows.append(line.split())
                elif rows:
                    break                          # loop ended
                elif line.startswith("#") or line.strip() == "":
                    continue
                else:
                    break
    need = ("type_symbol", "Cartn_x", "Cartn_y", "Cartn_z")
    if not rows or any(k not in cols for k in need):
        return None, None
    ci = [cols[k] for k in need]
    n = len(rows)
    el = np.zeros(n, dtype=np.int8)
    xyz = np.empty((n, 3), dtype=np.float32)
    for i, r in enumerate(rows):
        if len(r) <= ci[3]:
            return None, None
        el[i] = EL2I.get(r[ci[0]].strip().upper(), 0)
        xyz[i] = (float(r[ci[1]]), float(r[ci[2]]), float(r[ci[3]]))
    return el, xyz


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.cif")))
    print(f"[data_prep] {len(files):,} mmCIF files at {SRC}", flush=True)
    print(f"  NA={NA} atoms per crop, target {NSTRUCT:,} structures, seed={SEED}", flush=True)
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)

    cap = min(NSTRUCT, len(files))
    E = np.lib.format.open_memmap(OUT + "_elem.npy", mode="w+", dtype=np.int8, shape=(cap, NA))
    X = np.lib.format.open_memmap(OUT + "_xyz.npy", mode="w+", dtype=np.float32, shape=(cap, NA, 3))
    kept, seen, short, bad, t0 = 0, 0, 0, 0, time.time()
    names = []
    for p in files:
        if kept >= cap:
            break
        seen += 1
        try:
            el, xyz = parse_cif(p)
        except Exception:
            bad += 1
            continue
        if el is None:
            bad += 1
            continue
        if len(el) < NA:
            short += 1
            continue
        s = int(rng.integers(0, len(el) - NA + 1))
        E[kept] = el[s:s + NA]
        X[kept] = xyz[s:s + NA]
        names.append(os.path.basename(p)[:-4])
        kept += 1
        if kept % 20000 == 0:
            print(f"  {kept:,}/{cap:,} kept  ({seen:,} seen, {short:,} short, {bad:,} unparsed)  "
                  f"{time.time()-t0:.0f}s", flush=True)
    E.flush(); X.flush()
    meta = dict(n=kept, NA=NA, seed=SEED, seen=seen, short=short, unparsed=bad,
                elements=ELEMENTS, src=SRC, names_head=names[:20])
    json.dump(meta, open(OUT + "_meta.json", "w"), indent=1)
    np.save(OUT + "_names.npy", np.array(names))
    print(f"[data_prep] kept {kept:,} of {seen:,} seen "
          f"({short:,} shorter than {NA} atoms, {bad:,} unparsed) in {time.time()-t0:.0f}s",
          flush=True)
    if kept < cap:
        print(f"  NOTE: shard is {kept:,} rows, short of the {cap:,} requested.", flush=True)
