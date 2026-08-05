#!/bin/bash
# ============================================================
# Download and extract ESC-50, GSC v1, and GSC v2 for Nape fine-tuning.
# ============================================================
#
# Outputs under $DATA_ROOT:
#   esc50/    - ESC-50: 2000 clips, 50 classes, 5-fold CV
#   gsc_v1/   - Google Speech Commands v1: ~65K clips, 30 classes, 1s
#   gsc_v2/   - Google Speech Commands v2: ~106K clips, 35 classes, 1s
#
# Each has wavs/, manifest.json (or per-split manifest.json), classes.json.
#
# Usage:
#   bash download_finetune_datasets.sh [DATA_ROOT]
#
#   DATA_ROOT defaults to /ucappell/datasets
#
# Individual datasets can also be run standalone by calling extract_esc50.py
# or extract_gsc.py directly; see each script's --help for flags.

set -e

DATA_ROOT="${1:-/ucappell/datasets}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  NAPE fine-tuning datasets"
echo "  DATA_ROOT: $DATA_ROOT"
echo "============================================"

# ---- ESC-50 ----
echo ""
echo "[1/3] ESC-50"
echo "----"
python "$SCRIPT_DIR/extract_esc50.py" \
    --cache_dir "$DATA_ROOT/esc50_hf" \
    --output_dir "$DATA_ROOT/esc50" \
    --sample_rate 16000

# ---- GSC v1 (30 classes) ----
echo ""
echo "[2/3] Google Speech Commands v1 (30 classes)"
echo "----"
python "$SCRIPT_DIR/extract_gsc.py" \
    --version v0.01 \
    --cache_dir "$DATA_ROOT/gsc_hf" \
    --output_dir "$DATA_ROOT/gsc_v1" \
    --sample_rate 16000

# ---- GSC v2 (35 classes) ----
echo ""
echo "[3/3] Google Speech Commands v2 (35 classes)"
echo "----"
python "$SCRIPT_DIR/extract_gsc.py" \
    --version v0.02 \
    --cache_dir "$DATA_ROOT/gsc_hf" \
    --output_dir "$DATA_ROOT/gsc_v2" \
    --sample_rate 16000

echo ""
echo "============================================"
echo "  All datasets extracted to $DATA_ROOT"
echo "============================================"
ls -la "$DATA_ROOT/esc50" "$DATA_ROOT/gsc_v1" "$DATA_ROOT/gsc_v2" 2>/dev/null || true
