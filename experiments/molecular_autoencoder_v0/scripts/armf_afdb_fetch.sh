#!/bin/bash
# AFDB acquisition: draw -> download -> extract sequences. Gating happens after, on the sequences.
#
# ORDER, and why. Gating BEFORE download would need sequences we do not have (UniProt's .fasta serves
# 0 bytes for deprecated TrEMBL accessions, which is much of AFDB). The CIF carries the sequence, and
# the discard fraction is minuscule -- 125 held-out proteins against ~200M in AFDB -- so downloading
# first wastes ~nothing. The gate's job is to keep leakage out of TRAINING, not to save bandwidth.
#
# VERSION IS v6. Every AF-*-model_v4.cif 404s with an S3 NoSuchKey; the index's last column is the
# version and it is 6. A fetcher written against the assumed v4 fails on every structure while looking
# like a network fault.
set -uo pipefail
WR=/network/scratch/j/jacob-junqi.tian/latent-model-workspace
N=${1:-2000}; P=${2:-32}
OUT=$WR/afdb_raw; mkdir -p "$OUT"
IDX=$WR/afdb_accessions_sample.csv
shuf -n "$N" "$IDX" > "$OUT/draw.csv"
echo "[afdb] drawing $N accessions, $P workers"
t0=$(date +%s)
fetch1() {
  IFS=, read -r acc _ len id ver <<< "$1"
  o="$2/${acc}.cif"
  [ -s "$o" ] || curl -sS --max-time 45 -o "$o" \
      "https://alphafold.ebi.ac.uk/files/${id}-model_v${ver}.cif" 2>/dev/null
}
export -f fetch1
cat "$OUT/draw.csv" | xargs -P "$P" -I{} bash -c 'fetch1 "$@"' _ {} "$OUT"
t1=$(date +%s)
got=$(find "$OUT" -name '*.cif' -size +1k | wc -l)
echo "[afdb] $got/$N structures in $((t1-t0))s = $(echo "scale=1; $got/($t1-$t0+1)" | bc) struct/s"
echo "[afdb] raw bytes: $(du -sh $OUT | cut -f1)"
