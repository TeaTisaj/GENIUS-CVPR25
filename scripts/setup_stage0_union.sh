#!/bin/bash
# Build combined COCO+FashionIQ "union" JSONL files for Stage 0 feature extraction.
# Run on the headnode after FashionIQ preprocessing is complete.
# Creates:
#   cand_pool/global/mbeir_coco_fashioniq_train_cand_pool.jsonl
#   cand_pool/global/mbeir_coco_fashioniq_val_cand_pool.jsonl
#   query/union_train/mbeir_coco_fashioniq_up_train.jsonl
#   query/union_val/mbeir_coco_fashioniq_val.jsonl

set -e

MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
LOCAL="$MBEIR_DATA_DIR/cand_pool/local"
SRC_COCO="$MBEIR_DATA_DIR/src_data/mscoco"
SRC_FIQ="$MBEIR_DATA_DIR/src_data/fashioniq"

mkdir -p "$MBEIR_DATA_DIR/cand_pool/global"
mkdir -p "$MBEIR_DATA_DIR/query/union_train"
mkdir -p "$MBEIR_DATA_DIR/query/union_val"

echo "Building combined candidate pool (train)..."
cat "$LOCAL/mbeir_mscoco_task0_cand_pool.jsonl" \
    "$LOCAL/mbeir_mscoco_task3_cand_pool.jsonl" \
    "$LOCAL/mbeir_fashioniq_task7_cand_pool.jsonl" \
    > "$MBEIR_DATA_DIR/cand_pool/global/mbeir_coco_fashioniq_train_cand_pool.jsonl"
wc -l "$MBEIR_DATA_DIR/cand_pool/global/mbeir_coco_fashioniq_train_cand_pool.jsonl"

echo "Building combined candidate pool (val — same candidates)..."
cp "$MBEIR_DATA_DIR/cand_pool/global/mbeir_coco_fashioniq_train_cand_pool.jsonl" \
   "$MBEIR_DATA_DIR/cand_pool/global/mbeir_coco_fashioniq_val_cand_pool.jsonl"

echo "Building combined train queries..."
cat "$SRC_COCO/mbeir_mscoco_train.jsonl" \
    "$SRC_FIQ/mbeir_fashioniq_new_train.jsonl" \
    > "$MBEIR_DATA_DIR/query/union_train/mbeir_coco_fashioniq_up_train.jsonl"
wc -l "$MBEIR_DATA_DIR/query/union_train/mbeir_coco_fashioniq_up_train.jsonl"

echo "Building combined val queries..."
cat "$MBEIR_DATA_DIR/query/val/mbeir_mscoco_task0_val.jsonl" \
    "$MBEIR_DATA_DIR/query/val/mbeir_mscoco_task3_val.jsonl" \
    "$SRC_FIQ/mbeir_fashioniq_new_val.jsonl" \
    > "$MBEIR_DATA_DIR/query/union_val/mbeir_coco_fashioniq_val.jsonl"
wc -l "$MBEIR_DATA_DIR/query/union_val/mbeir_coco_fashioniq_val.jsonl"

echo ""
echo "Done. Summary:"
ls -lh "$MBEIR_DATA_DIR/cand_pool/global/"
ls -lh "$MBEIR_DATA_DIR/query/union_train/"
ls -lh "$MBEIR_DATA_DIR/query/union_val/"
echo ""
echo "Next: sbatch scripts/slurm_stage0_coco_fashioniq.sh"
