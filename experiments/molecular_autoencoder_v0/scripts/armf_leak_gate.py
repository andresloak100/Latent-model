#!/usr/bin/env python3
"""SEQUENCE-LEAKAGE GATE for the 1M AFDB draw.

THE RISK. AlphaFold DB covers essentially all of UniProt, so a 1M draw will contain predicted
structures of the HELD-OUT ATLAS proteins unless they are removed. Pretraining on them would
invalidate every zero-shot claim on the record -- 0/123 at every cutoff, the 2.4% win-rate bound, the
whole peer comparison. This gate runs BEFORE the corpus is built, not after.

WHAT IS EXCLUDED AND WHAT IS NOT, stated rather than implied:
  * EXCLUDED: any AFDB entry above 30% identity to ANY held-out ATLAS sequence.
  * NOT EXCLUDED: the TRAINING systems. Pretraining on the training distribution is the point, and
    removing them would defeat the purpose of the pretrain. Said explicitly because a reader seeing
    a leakage gate will reasonably assume it removed both.

QUERY-SIDE COMPLETENESS. The query set is ALL CHAINS of all 125 held-out entries (156 sequences), not
one chain each. Over-inclusion on the query side can only remove more AFDB entries, which is the safe
direction; under-inclusion leaves a held-out protein unscreened, which is the failure the gate exists
to prevent. Two defects were caught building it: fragile chain-matching lost 24 of 125 entries, and a
file without a trailing newline made `while read` silently drop the last one (6sup_A, which had zero
sequences in the query set until it was noticed)."""
import sys, os, subprocess, json, argparse

WR = os.environ.get("WR", "/network/scratch/j/jacob-junqi.tian/latent-model-workspace")
MMSEQS = f"{WR}/mmseqs/bin/mmseqs"
HELDOUT = f"{WR}/heldout_atlas.fasta"
MIN_ID = float(os.environ.get("LEAK_MIN_ID", "0.30"))


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def gate(candidate_fasta, out_keep, tmp=None):
    tmp = tmp or f"{WR}/leak_tmp"
    os.makedirs(tmp, exist_ok=True)
    nq = sum(1 for l in open(HELDOUT) if l.startswith(">"))
    nc = sum(1 for l in open(candidate_fasta) if l.startswith(">"))
    print(f"[leak-gate] {nc:,} candidates vs {nq} held-out sequences "
          f"(all chains of 125 entries), threshold {MIN_ID:.0%} identity", flush=True)
    run(f"{MMSEQS} easy-search {candidate_fasta} {HELDOUT} {tmp}/hits.m8 {tmp}/mm "
        f"--min-seq-id {MIN_ID} -c 0.5 --cov-mode 0 -s 5.7 --threads 8 --format-output "
        f"'query,target,fident,alnlen,evalue,bits' -v 1")
    hit = {}
    for line in open(f"{tmp}/hits.m8"):
        p = line.split("\t")
        if len(p) >= 3:
            q, t, f = p[0], p[1], float(p[2])
            if f >= MIN_ID and (q not in hit or f > hit[q][1]): hit[q] = (t, f)
    ids = [l[1:].split()[0] for l in open(candidate_fasta) if l.startswith(">")]
    keep = [i for i in ids if i not in hit]
    with open(out_keep, "w") as fh: fh.write("\n".join(keep) + "\n")
    print(f"[leak-gate] REMOVED {len(hit):,} of {nc:,} ({100*len(hit)/max(nc,1):.3f}%) above "
          f"{MIN_ID:.0%} identity to a held-out ATLAS sequence", flush=True)
    print(f"[leak-gate] KEPT    {len(keep):,} -> {out_keep}", flush=True)
    print(f"[leak-gate] TRAINING systems were NOT excluded -- pretraining on the training "
          f"distribution is the point, and that is a decision, not an oversight.", flush=True)
    if hit:
        top = sorted(hit.items(), key=lambda kv: -kv[1][1])[:5]
        print(f"[leak-gate] highest-identity removals:", flush=True)
        for q, (t, f) in top:
            print(f"    {q}  ->  {t}  {100*f:.1f}%", flush=True)
    json.dump({"threshold": MIN_ID, "n_candidates": nc, "n_removed": len(hit),
               "n_kept": len(keep), "n_query_seqs": nq,
               "removed": {q: {"target": t, "fident": f} for q, (t, f) in hit.items()}},
              open(f"{WR}/leak_gate_report.json", "w"))
    return len(hit), len(keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="FASTA of the AFDB draw")
    ap.add_argument("--out", default=f"{WR}/afdb_keep_ids.txt")
    a = ap.parse_args()
    if not os.path.exists(MMSEQS): raise SystemExit(f"mmseqs not at {MMSEQS}")
    if not os.path.exists(HELDOUT): raise SystemExit(f"held-out fasta not at {HELDOUT}")
    gate(a.candidates, a.out)
