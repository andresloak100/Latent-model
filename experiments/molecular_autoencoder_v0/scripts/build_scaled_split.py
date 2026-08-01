#!/usr/bin/env python3
"""Build a grown corpus's split against a FIXED inherited validation set.

A data-scaling ladder answers "does more training data help?" only if the
evaluation set is the same one the baseline was measured on and nothing in the
new training data leaks into it. Growing a corpus threatens both, and the
second failure is silent:

  * Taking ``val`` from the new corpus's own split changes what is being
    measured, so the ladder cannot be compared to the baseline at all. Here
    ``val`` is copied verbatim from the reference splits and a missing key is
    fatal.

  * Taking ``train = everything - val`` removes only exact KEY collisions. The
    PDB is heavily redundant -- the same protein is deposited many times under
    different ids -- so a 10x larger draw pulls in structures whose sequence is
    identical to a validation structure under a different id. Those are not
    new information; they are the answer key. A ladder built that way improves
    with size because it is memorising val's homologs, which looks exactly
    like the result we are hoping for.

Exact-sequence exclusion is applied always (matching the reference corpus's own
``exact_sequence_split``). Near-duplicate exclusion is opt-in but always
MEASURED, because "how much near-duplicate leakage would we have had" is the
number that says whether exact matching was sufficient at this scale.

Usage:
    python scripts/build_scaled_split.py \
        --manifest data/processed_complex_scaled/manifest.json \
        --reference-splits data/splits_complex.json \
        --out data/splits_complex_scaled.json
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae import utils  # noqa: E402


def manifest_key(record) -> str:
    """The key prepare_dataset.py derives for the .npz filename."""
    return f"{record['pdb_id']}_{record['chain_id'].replace(',', '-')}"


def sequences_of(record):
    """Every kept chain's one-letter sequence, as a set."""
    seqs = record.get("per_chain_sequence") or []
    if isinstance(seqs, str):
        seqs = [seqs]
    return {s for s in seqs if s}


def kmers(seq: str, k: int = 8):
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def exact_leaks(candidates, val_sequences):
    """Keys sharing at least one chain sequence verbatim with a val structure."""
    return {key for key, seqs in candidates.items() if seqs & val_sequences}


def near_duplicates(candidates, val_sequences, threshold, k=8, prefilter=0.3):
    """(key, best_ratio) for candidates similar to any val chain sequence.

    A k-mer Jaccard prefilter narrows the O(N*M) difflib comparison to pairs
    that could plausibly clear the threshold; difflib then decides. The
    prefilter is deliberately loose -- it may pass pairs that fail, but must
    not drop pairs that would have passed.
    """
    val_list = [(s, kmers(s, k)) for s in val_sequences]
    hits = []
    for key, seqs in candidates.items():
        best = 0.0
        for s in seqs:
            ks = kmers(s, k)
            for vs, kv in val_list:
                if jaccard(ks, kv) < prefilter:
                    continue
                r = difflib.SequenceMatcher(None, s, vs).ratio()
                if r > best:
                    best = r
        if best >= threshold:
            hits.append((key, best))
    return sorted(hits, key=lambda t: -t[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="manifest.json of the GROWN corpus")
    ap.add_argument("--reference-splits", required=True,
                    help="splits json whose val keys are inherited verbatim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exempt-reference-train", action="store_true", default=True,
                    help="keep the reference training set intact (default). Its "
                         "members are exempt from per-chain exclusion so the "
                         "anchor rung still reproduces the baseline's exact "
                         "training set; how many WOULD have been excluded is "
                         "reported as the baseline's own leakage.")
    ap.add_argument("--no-exempt-reference-train", dest="exempt_reference_train",
                    action="store_false",
                    help="apply per-chain exclusion to the reference training "
                         "set too. Produces a cleaner corpus but no anchor.")
    ap.add_argument("--report-similarity", type=float, default=0.9,
                    help="measure (do not exclude) near-duplicates at this ratio")
    ap.add_argument("--exclude-similarity", type=float, default=None,
                    help="also EXCLUDE train candidates this similar to any val "
                         "chain. Off by default: the reference corpus used exact "
                         "matching, and changing the leakage rule mid-ladder "
                         "would confound the comparison it exists to enable.")
    args = ap.parse_args()

    manifest = utils.load_json(ROOT / args.manifest if not Path(args.manifest).is_absolute()
                               else args.manifest)
    reference = utils.load_json(ROOT / args.reference_splits
                                if not Path(args.reference_splits).is_absolute()
                                else args.reference_splits)

    records = {manifest_key(r): r for r in manifest["kept"]}
    val_keys = list(reference["val"])

    missing = [k for k in val_keys if k not in records]
    if missing:
        raise SystemExit(
            f"[split] {len(missing)} of {len(val_keys)} reference val keys are "
            f"absent from the grown corpus, e.g. {missing[:5]}. The evaluation "
            f"set must be intact or the ladder is not comparable to the "
            f"baseline. Check the key format and that the ids were pinned."
        )

    val_sequences = set()
    for k in val_keys:
        val_sequences |= sequences_of(records[k])

    val_set = set(val_keys)
    candidates = {k: sequences_of(r) for k, r in records.items() if k not in val_set}

    # The reference training set is exempt from the exclusion below, and the
    # reason is not leniency. The reference split clustered on the CONCATENATED
    # structure sequence; this script clusters per CHAIN, which is strictly
    # stronger -- a complex whose chain A matches a val complex's chain A can
    # pass the old rule and fail this one. Applying the new rule to the old
    # training set would silently delete members of it, and the rung that is
    # supposed to reproduce the baseline would no longer be the baseline's
    # training set, destroying the anchor.
    #
    # So the old set is kept intact and the count of its members that WOULD
    # have been excluded is reported instead. That count is not bookkeeping:
    # it measures how much per-chain leakage the baseline number itself was
    # measured under.
    exempt = set(reference.get("train", [])) if args.exempt_reference_train else set()
    leaked_all = exact_leaks(candidates, val_sequences)
    leaked_baseline = leaked_all & exempt
    leaked = leaked_all - exempt
    train_keys = sorted(k for k in candidates if k not in leaked)

    report_thr = args.report_similarity
    near = near_duplicates({k: candidates[k] for k in train_keys},
                           val_sequences, report_thr) if report_thr else []
    near_baseline = [(k, r) for k, r in near if k in exempt]

    excluded_near = []
    if args.exclude_similarity is not None:
        # Exempt the reference training set here too, for the same reason the
        # exact rule exempts it: pruning it breaks the anchor.
        #
        # This is what makes exclusion the right choice rather than a tradeoff.
        # Near-duplicate leakage that GROWS with corpus size is fatal to a data
        # ladder -- the slope then mixes data volume with rising leakage, and
        # that is the confound the whole exercise exists to avoid. Excluding it
        # from the added data only, while leaving the reference set intact,
        # holds leakage CONSTANT at the baseline's own level across every rung.
        # Constant leakage does not bend a slope; rising leakage does.
        excluded_near = [k for k, r in near_duplicates(
            {k: candidates[k] for k in train_keys if k not in exempt},
            val_sequences, args.exclude_similarity)]
        train_keys = sorted(set(train_keys) - set(excluded_near))

    splits = {
        "train": train_keys,
        "val": val_keys,
        "inherited_val_from": str(args.reference_splits),
        "n_corpus": len(records),
        "n_exact_sequence_leaks_excluded": len(leaked),
        "n_reference_train_leaks_exempted": len(leaked_baseline),
        "reference_train_leaks": sorted(leaked_baseline),
        "exempt_reference_train": bool(args.exempt_reference_train),
        "n_near_duplicates_at_report_threshold": len(near),
        "report_similarity": report_thr,
        "exclude_similarity": args.exclude_similarity,
        "n_near_duplicates_excluded": len(excluded_near),
        "n_reference_train_near_duplicates": len(near_baseline),
    }
    utils.save_json(splits, ROOT / args.out if not Path(args.out).is_absolute() else args.out)

    print(f"[split] corpus {len(records)} structures")
    print(f"[split] val    {len(val_keys)} (inherited verbatim, all present)")
    print(f"[split] train  {len(train_keys)}")
    print(f"[split] excluded {len(leaked)} exact-sequence leaks into val")
    if args.exempt_reference_train:
        n_ref = len(set(reference.get("train", [])))
        if leaked_baseline:
            print(f"[split] {len(leaked_baseline)} of the {n_ref} reference "
                  f"training structures share a chain with val and were KEPT "
                  f"so the anchor rung stays intact -- the baseline number was "
                  f"measured under that much per-chain leakage:")
            for k in sorted(leaked_baseline)[:10]:
                print(f"           {k}")
        else:
            print(f"[split] reference training set ({n_ref}) is clean under the "
                  f"stricter per-chain rule too")
    if report_thr:
        print(f"[split] {len(near)} remaining train structures are >={report_thr} "
              f"similar to a val chain "
              f"({'excluded' if args.exclude_similarity is not None else 'NOT excluded'})")
        for key, r in near[:10]:
            print(f"           {key}  ratio {r:.3f}")
    if args.exclude_similarity is not None:
        print(f"[split] excluded {len(excluded_near)} near-duplicates at "
              f">={args.exclude_similarity} from the ADDED data; the reference "
              f"training set keeps its own {len(near_baseline)}, so leakage is "
              f"constant across rungs rather than rising with corpus size")
    print(f"[split] wrote {args.out}")


if __name__ == "__main__":
    main()
