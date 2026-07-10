#!/bin/bash
# Preprocess raw FashionIQ data into M-BEIR format.
# Run on the headnode (no GPU needed). Safe to re-run.
#
# Prerequisites:
#   1. scripts/download_fashioniq.sh must have completed successfully.
#   2. Images in: mbeir_data/mbeir_images/fashioniq_images/
#   3. Captions in: mbeir_data/src_data/fashioniq/captions/cap.{dress|shirt|toptee}.{train|val}.json
#
# Outputs (in src_data/fashioniq/):
#   mbeir_fashioniq_cand_pool.jsonl
#   mbeir_fashioniq_new_train.jsonl  (original train, minus 1700 held-out for val)
#   mbeir_fashioniq_new_val.jsonl    (1700 items from original train)
#   mbeir_fashioniq_new_test.jsonl   (original val set, used as test)
#
# After this, run scripts/setup_fashioniq_mbeir_dirs.sh to symlink into the expected layout.

set -e

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
PYTHON="/home/tcetoje/miniconda3/envs/genius2/bin/python"

export PYTHONPATH="$GENIR_DIR/src"

cd "$GENIR_DIR/src/data/preprocessing"

echo "Running FashionIQ preprocessor..."
$PYTHON fashioniq_data_preprocessor.py \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --fashioniq_images_dir "mbeir_images/fashioniq_images/" \
    --fashioniq_dir "src_data/fashioniq" \
    --enable_candidate_pool \
    --enable_mbeir_conversion \
    --split_train_into_val_and_val_into_test

echo ""
echo "Preprocessing complete. Outputs in $MBEIR_DATA_DIR/src_data/fashioniq/"
ls -lh "$MBEIR_DATA_DIR/src_data/fashioniq/"*.jsonl 2>/dev/null
echo ""
echo "Next: run scripts/setup_fashioniq_mbeir_dirs.sh"
