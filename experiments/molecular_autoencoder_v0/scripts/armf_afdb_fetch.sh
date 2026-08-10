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
# THE DRAW MUST COVER THE WHOLE INDEX, NOT A PREFIX.
# The pilot drew from `curl -r 0-2000000` -- a byte-range PREFIX over 0.028% of an 8.7 GB index --
# and then shuffled inside it, which is uniform over a non-uniform pool. Measured, the prefix is
# indistinguishable from the index through 75% (KS p 0.68-0.92) but the final ~1% is a distinct,
# longer population (median 366 vs 277 residues, p < 1e-300). So the pilot's rates hold for ~99% of
# AFDB and not for the tail, and the fix costs one 8.7 GB download.
IDXURL=https://ftp.ebi.ac.uk/pub/databases/alphafold/accession_ids.csv
IDX=$WR/afdb_accessions_full.csv
NEED=8702185935
have=$(stat -c%s "$IDX" 2>/dev/null || echo 0)
if [ "$have" -lt "$NEED" ]; then
  echo "[afdb] index $have/$NEED bytes; resuming download"
  curl -sS -C - -o "$IDX" "$IDXURL" || { echo "[afdb] index download FAILED"; exit 1; }
fi
got=$(stat -c%s "$IDX")
[ "$got" -eq "$NEED" ] || { echo "[afdb] index truncated: $got != $NEED -- REFUSING to draw from a partial index"; exit 1; }
echo "[afdb] index complete: $got bytes, $(wc -l < "$IDX") accessions"
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
