#!/bin/bash
# Symlink preprocessed FashionIQ files into the M-BEIR directory layout expected by GENIUS.
# Run on the headnode after scripts/preprocess_fashioniq.sh has completed.
# Safe to re-run — existing symlinks are skipped.
#
# Source files (from preprocessor with --split_train_into_val_and_val_into_test):
#   src_data/fashioniq/mbeir_fashioniq_cand_pool.jsonl
#   src_data/fashioniq/mbeir_fashioniq_new_train.jsonl
#   src_data/fashioniq/mbeir_fashioniq_new_val.jsonl
#   src_data/fashioniq/mbeir_fashioniq_new_test.jsonl
#
# Target layout (used by retriever):
#   cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl
#   query/train/mbeir_fashioniq_task7_train.jsonl
#   query/val/mbeir_fashioniq_task7_val.jsonl
#   query/test/mbeir_fashioniq_task7_test.jsonl

set -e

MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
SRC="$MBEIR_DATA_DIR/src_data/fashioniq"

mkdir -p "$MBEIR_DATA_DIR/cand_pool/local"
mkdir -p "$MBEIR_DATA_DIR/query/train"
mkdir -p "$MBEIR_DATA_DIR/query/val"
mkdir -p "$MBEIR_DATA_DIR/query/test"

link_if_missing() {
    local src="$1"
    local dst="$2"
    if [ ! -e "$dst" ]; then
        if [ -f "$src" ]; then
            ln -s "$src" "$dst"
            echo "  Linked: $(basename $dst)"
        else
            echo "  WARNING: source not found: $src"
        fi
    else
        echo "  Already exists: $(basename $dst)"
    fi
}

echo "Linking candidate pool..."
link_if_missing \
    "$SRC/mbeir_fashioniq_cand_pool.jsonl" \
    "$MBEIR_DATA_DIR/cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl"

echo "Linking query splits..."
link_if_missing \
    "$SRC/mbeir_fashioniq_new_train.jsonl" \
    "$MBEIR_DATA_DIR/query/train/mbeir_fashioniq_task7_train.jsonl"

link_if_missing \
    "$SRC/mbeir_fashioniq_new_val.jsonl" \
    "$MBEIR_DATA_DIR/query/val/mbeir_fashioniq_task7_val.jsonl"

link_if_missing \
    "$SRC/mbeir_fashioniq_new_test.jsonl" \
    "$MBEIR_DATA_DIR/query/test/mbeir_fashioniq_task7_test.jsonl"

echo ""
echo "Setup complete. Current layout:"
find "$MBEIR_DATA_DIR/cand_pool" "$MBEIR_DATA_DIR/query" -name "*fashioniq*" 2>/dev/null | sort
echo ""
echo "Next: run scripts/setup_eval_prerequisites.sh (generates qrels), then sbatch slurm_feat_extract_fashioniq.sh"
