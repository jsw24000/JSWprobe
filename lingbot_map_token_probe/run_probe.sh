#!/usr/bin/env bash
set -euo pipefail

LINGBOT_ROOT=${LINGBOT_ROOT:-../lingbot-map}
DATA_ROOT=${DATA_ROOT:-/disk1/3dsm/S2VGGT}
SCENE=${SCENE:-scene0000_00}
NUM_FRAMES=${NUM_FRAMES:-12}
FRAME_STRIDE=${FRAME_STRIDE:-10}
MODEL_PATH=${MODEL_PATH:-$LINGBOT_ROOT/checkpoints/lingbot-map.pt}
PYTHON=${PYTHON:-python}

echo "[probe] Lingbot root: $LINGBOT_ROOT"
echo "[probe] Data root request: $DATA_ROOT"
echo "[probe] Scene: $SCENE"

$PYTHON scripts/find_scannet_scenes.py --data-root "$DATA_ROOT"

$PYTHON scripts/extract_tokens.py \
  --lingbot-root "$LINGBOT_ROOT" \
  --data-root "$DATA_ROOT" \
  --scene "$SCENE" \
  --num-frames "$NUM_FRAMES" \
  --frame-stride "$FRAME_STRIDE" \
  --model-path "$MODEL_PATH" \
  --output "outputs/$SCENE"

$PYTHON scripts/visualize_pca_tokens.py \
  --input "outputs/$SCENE" \
  --output "outputs/$SCENE/pca_vis" \
  --fit-scope scene

$PYTHON scripts/visualize_correspondence.py \
  --input "outputs/$SCENE" \
  --stage auto \
  --src-frame 0 \
  --tgt-frame 5 \
  --output "outputs/$SCENE/correspondence_vis"

echo "[probe] Done. Summary: outputs/$SCENE/summary.md"
