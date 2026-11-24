#!/bin/bash
# FVD of one model's test-split generations, as reported in the paper.
#
#   bash fvd/run_fvd.sh output/<experiment_name>/duet_2048
#
# The argument is the folder of video_N_side_by_side.mp4 files written by
#   python src/inference.py ... ~dataset@eval_datasets.<each eval> +dataset@eval_datasets.duet_2048=duet_2048
# (see fvd/README.md). Each video holds the ground truth and the generation for both
# players; the script extracts I3D features for all four quadrants and prints the FVD of
# the generated clips against the ground-truth clips, both players pooled.
set -euo pipefail

VIDEOS=${1:?usage: run_fvd.sh VIDEO_DIR [FEATURES_DIR]}
FEATS=${2:-$VIDEOS/fvd_features}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
I3D=${I3D:-$HERE/models/i3d_torchscript.pt}  # downloaded on first use

mkdir -p "$FEATS"
for q in 0 1 2 3; do
    out="$FEATS/q$q.npz"
    [[ -s "$out" ]] && { echo "[skip] $out exists"; continue; }
    "$PY" "$HERE/compute_features.py" --data_dir "$VIDEOS" --quadrant $q --output "$out" \
        --i3d_weights "$I3D" --sample_mode random_window --num_frames 16 --seed 42 \
        --batch_size 32 --num_workers 8
done

# Quadrants 0 and 2 are the ground truth of player 1 and 2; 1 and 3 the generations.
"$PY" "$HERE/compute_fvd.py" --features1 "$FEATS/q0.npz" "$FEATS/q2.npz" \
                             --features2 "$FEATS/q1.npz" "$FEATS/q3.npz"
