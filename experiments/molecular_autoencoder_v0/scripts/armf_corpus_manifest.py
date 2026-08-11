#!/usr/bin/env python3
"""INBOX 97a: a per-structure CHECKSUM manifest, so a rebuild can be VERIFIED rather than assumed.

WHY. `git rm --cached` keeps files on disk ONLY in the tree where the command runs. Every other
working tree that rebases past the commit LOSES them -- which is what happened to
data/processed_small, all 3,030 structures the static evaluation reads, recovered only because they
were still tracked in the parent commit.

AND "REGENERABLE" IS NOT "REPRODUCIBLY REGENERABLE". prepare_dataset.py rebuilds from data/raw*/,
also untracked, so regeneration means re-downloading from the PDB -- and PDB entries are obsoleted
and remediated. A structure fetched today may differ from the one that produced a published number.

The splits say WHICH STRUCTURES. Nothing said WHICH BYTES. This does.

The manifest is kilobytes, versioned, and lets a rebuild be checked structure by structure rather
than trusted. It records the content hash of the .npz AND the shape fields a silent remediation would
move, so a mismatch says what changed rather than only that something did.
"""
import sys, os, json, hashlib, glob, time
import numpy as np

WR = os.environ.get("WR", "")
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."


def digest(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()


def entry(path):
    """Content hash plus the fields a silent PDB remediation would move. A hash alone says
    'different'; these say 'different HOW', which is the difference between a scare and a diagnosis."""
    d = np.load(path, allow_pickle=True)
    n_atoms = int(np.asarray(d["coords"]).shape[0])
    n_res = int(np.asarray(d["res_pos"]).max()) + 1 if "res_pos" in d.files else -1
    seq = str(d["sequence"]) if "sequence" in d.files else ""
    return dict(sha256=digest(path), bytes=os.path.getsize(path), n_atoms=n_atoms, n_res=n_res,
                seq_sha256=hashlib.sha256(seq.encode()).hexdigest()[:16], seq_len=len(seq))


def build(corpus_dir, out):
    paths = sorted(glob.glob(f"{corpus_dir}/*.npz"))
    if not paths:
        raise SystemExit(f"  no .npz under {corpus_dir}")
    man, t0 = {}, time.time()
    for i, p in enumerate(paths, 1):
        man[os.path.basename(p)[:-4]] = entry(p)
        if i % 500 == 0:
            print(f"    {i}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    payload = dict(corpus=os.path.relpath(corpus_dir, ROOT), n=len(man),
                   tool="armf_corpus_manifest.py", entries=man)
    tmp = out + ".tmp"
    json.dump(payload, open(tmp, "w"), indent=0, sort_keys=True)
    os.replace(tmp, out)
    print(f"  {len(man)} structures -> {out} ({os.path.getsize(out)/1e6:.2f} MB)", flush=True)
    return payload


def verify(corpus_dir, man_path):
    man = json.load(open(man_path))["entries"]
    present = {os.path.basename(p)[:-4] for p in glob.glob(f"{corpus_dir}/*.npz")}
    missing = sorted(set(man) - present)
    extra = sorted(present - set(man))
    changed, moved = [], []
    for k in sorted(set(man) & present):
        e = entry(f"{corpus_dir}/{k}.npz")
        if e["sha256"] != man[k]["sha256"]:
            changed.append(k)
            if (e["n_atoms"], e["n_res"], e["seq_sha256"]) != \
               (man[k]["n_atoms"], man[k]["n_res"], man[k]["seq_sha256"]):
                moved.append((k, man[k], e))
    print(f"  VERIFY {corpus_dir}")
    print(f"    {len(present)} on disk, {len(man)} in manifest")
    print(f"    missing {len(missing)}   unexpected {len(extra)}   byte-changed {len(changed)}")
    print(f"    of the changed, {len(moved)} ALSO changed atoms/residues/sequence -- "
          f"those are remediations, not re-serialisations")
    for k, old, new in moved[:5]:
        print(f"      {k}: atoms {old['n_atoms']}->{new['n_atoms']}  res {old['n_res']}->{new['n_res']}  "
              f"seqlen {old['seq_len']}->{new['seq_len']}")
    ok = not missing and not changed
    print(f"    -> {'IDENTICAL to the manifest' if ok else 'DOES NOT MATCH the manifest'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: armf_corpus_manifest.py build|verify <corpus_dir> [manifest.json]")
    mode, corpus = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else \
        f"{ROOT}/data/manifests/{os.path.basename(corpus.rstrip('/'))}.sha256.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if mode == "build": build(corpus, out)
    elif mode == "verify": raise SystemExit(0 if verify(corpus, out) else 1)
    else: raise SystemExit(f"unknown mode {mode}")
