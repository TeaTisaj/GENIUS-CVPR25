#!/bin/bash
# One-time setup: create the M-BEIR directory structure expected by GENIUS
# and symlink COCO JSONL files into the right locations.
# Safe to re-run — symlinks are skipped if they already exist.
# Run on the headnode (no GPU needed).

MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COCO_SRC="$MBEIR_DATA_DIR/src_data/mscoco"

# ── Candidate pool ───────────────────────────────────────────────────────────
# The extractor expects:
#   {mbeir_data_dir}/cand_pool/local/mbeir_{task_name}_cand_pool.jsonl
# We use the unified cand pool for both COCO tasks (it covers all modalities).
mkdir -p "$MBEIR_DATA_DIR/cand_pool/local"

for task in mscoco_task0 mscoco_task3; do
    target="$MBEIR_DATA_DIR/cand_pool/local/mbeir_${task}_cand_pool.jsonl"
    if [ ! -e "$target" ]; then
        ln -s "$COCO_SRC/mbeir_mscoco_cand_pool.jsonl" "$target"
        echo "Created: $target"
    else
        echo "Already exists: $target"
    fi
done

# ── Test queries ─────────────────────────────────────────────────────────────
# The eval script expects:
#   {mbeir_data_dir}/query/test/mbeir_{dataset}_{split}.jsonl
mkdir -p "$MBEIR_DATA_DIR/query/test"
mkdir -p "$MBEIR_DATA_DIR/query/val"

# task0 = text → image
for split in test val; do
    target="$MBEIR_DATA_DIR/query/$split/mbeir_mscoco_task0_$split.jsonl"
    if [ ! -e "$target" ]; then
        ln -s "$COCO_SRC/mbeir_mscoco_$split.jsonl" "$target"
        echo "Created: $target"
    else
        echo "Already exists: $target"
    fi
done

# task3 = image → text
for split in test val; do
    target="$MBEIR_DATA_DIR/query/$split/mbeir_mscoco_task3_$split.jsonl"
    src="$COCO_SRC/mbeir_mscoco_txt_$split.jsonl"
    if [ -f "$src" ] && [ ! -e "$target" ]; then
        ln -s "$src" "$target"
        echo "Created: $target"
    fi
done

# ── Instructions ─────────────────────────────────────────────────────────────
# Point the mbeir_data instructions dir at the one shipped with the repo.
GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
if [ ! -e "$MBEIR_DATA_DIR/instructions" ]; then
    if [ -d "$GENIR_DIR/instructions" ]; then
        ln -s "$GENIR_DIR/instructions" "$MBEIR_DATA_DIR/instructions"
        echo "Linked instructions dir"
    else
        echo "WARNING: $GENIR_DIR/instructions not found — locate query_instructions.tsv manually"
    fi
else
    echo "Already exists: $MBEIR_DATA_DIR/instructions"
fi

echo ""
echo "Setup complete. Directory layout:"
find "$MBEIR_DATA_DIR/cand_pool" "$MBEIR_DATA_DIR/query" -name "*.jsonl" 2>/dev/null | sort
