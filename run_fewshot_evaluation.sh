#!/usr/bin/env bash
set -euo pipefail

# Official-style BiomedCoOp comparison: single-label accuracy across K shots.
for shots in 1 2 4 8 16; do
  for seed in 1 2 3; do
    python train_biomedcoop.py \
      --task single \
      --shots "${shots}" \
      --seed "${seed}"
  done
done

# Native PTB-XL multi-label comparison.
for shots in 1 2 4 8 16; do
  for seed in 1 2 3; do
    python train_biomedcoop.py \
      --task multi \
      --shots "${shots}" \
      --seed "${seed}" \
      --threshold-mode per-class
  done
done

python aggregate_results.py \
  --pattern 'work/results/biomedcoop_*.json' \
  --output work/results/biomedcoop_summary.csv
