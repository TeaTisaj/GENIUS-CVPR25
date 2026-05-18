#!/bin/bash
# One-time setup for the GENIUS eval step on COCO.
# Run on the headnode (no GPU needed) before submitting slurm_eval_coco.sh.
# Safe to re-run — existing files are skipped.

set -e

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
PYTHON="/home/tcetoje/miniconda3/envs/genius2/bin/python"

# ── 1. Compile trie_cpp ───────────────────────────────────────────────────────
# The retriever imports trie_cpp at module level, so this must exist before eval.
TRIE_DIR="$GENIR_DIR/src/models/generative_retriever"
SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TRIE_SO="$TRIE_DIR/trie_cpp${SUFFIX}"

if [ ! -f "$TRIE_SO" ]; then
    echo "Compiling trie_cpp..."
    PYBIND_INCLUDES=$($PYTHON -m pybind11 --includes)
    c++ -O3 -Wall -shared -std=c++17 -fPIC \
        $PYBIND_INCLUDES \
        "$TRIE_DIR/trie_cpp.cpp" \
        -o "$TRIE_SO"
    echo "  Compiled: $TRIE_SO"
else
    echo "Already compiled: $TRIE_SO"
fi

# ── 2. Symlinks — _test-suffixed cand pool files ─────────────────────────────
# mbeir_generative_retriever.py appends _test to mscoco_task0/task3 automatically
# (EXCEPTIONAL_CAND_POOLS). Both the JSONL and the pre-extracted .pt files need
# the _test-suffixed name.

CAND_LOCAL="$MBEIR_DATA_DIR/cand_pool/local"
EMBED_DIR="$GENIR_DIR/extracted_embed/CLIP_SF/cand"

for task in mscoco_task0 mscoco_task3; do
    # JSONL symlink
    src="$CAND_LOCAL/mbeir_${task}_cand_pool.jsonl"
    dst="$CAND_LOCAL/mbeir_${task}_test_cand_pool.jsonl"
    if [ ! -e "$dst" ]; then
        ln -s "$src" "$dst" && echo "  Linked: $(basename $dst)"
    fi

    # Pre-extracted embedding .pt symlink
    src="$EMBED_DIR/cand_pool_${task}_IT_dict.pt"
    dst="$EMBED_DIR/cand_pool_${task}_test_IT_dict.pt"
    if [ ! -e "$dst" ]; then
        ln -s "$src" "$dst" && echo "  Linked: $(basename $dst)"
    fi
done

# ── 3. Generate qrel files ────────────────────────────────────────────────────
# Qrel format (TREC): qid Q0 did relevance task_id
# One line per (query, positive_candidate) pair; relevance=1.

mkdir -p "$MBEIR_DATA_DIR/qrels/test"

$PYTHON - <<'PYEOF'
import json, os, pathlib

mbeir_data_dir = pathlib.Path("/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")

tasks = [
    # (query_jsonl_relpath, task_id, qrel_name)
    ("query/test/mbeir_mscoco_task0_test.jsonl", 0, "mscoco_task0"),
    ("query/test/mbeir_mscoco_task3_test.jsonl", 3, "mscoco_task3"),
]

for rel_path, task_id, qrel_name in tasks:
    qrel_path = mbeir_data_dir / "qrels" / "test" / f"mbeir_{qrel_name}_test_qrels.txt"
    if qrel_path.exists():
        print(f"  Already exists: {qrel_path.name}")
        continue

    src = mbeir_data_dir / rel_path
    if not src.exists():
        print(f"  WARNING: query file not found: {src}")
        continue

    lines = []
    with open(src) as f:
        for row in f:
            entry = json.loads(row)
            qid = entry["qid"]
            for did in entry.get("pos_cand_list", []):
                lines.append(f"{qid} Q0 {did} 1 {task_id}\n")

    with open(qrel_path, "w") as f:
        f.writelines(lines)
    print(f"  Generated {len(lines)} qrel lines → {qrel_path.name}")
PYEOF

# ── 4. Query instructions TSV ─────────────────────────────────────────────────
# Format (tab-separated, header skipped):
#   query_modality  cand_modality  task  dataset_id  prompt1  prompt2 ...
# Key used by dataset loader: "{dataset_id}, {query_modality}, {cand_modality}"
# COCO = dataset_id 9.

INST_DIR="$MBEIR_DATA_DIR/instructions"
INST_FILE="$INST_DIR/query_instructions.tsv"
mkdir -p "$INST_DIR"

if [ ! -f "$INST_FILE" ]; then
    cat > "$INST_FILE" <<'TSV'
query_modality	cand_modality	task	dataset_id	prompt1	prompt2	prompt3
text	image	text_to_image	9	Retrieve an image that matches the given text description.	Find the image described by the query.	Given this text query, retrieve the corresponding image.
image	text	image_to_text	9	Retrieve a text caption that describes the given image.	Find a textual description for the given image.	Given this image, retrieve text that describes it.
TSV
    echo "  Created: $INST_FILE"
else
    echo "  Already exists: $INST_FILE"
fi

# Create symlink from genir_dir so the path config.mbeir_data_dir/instructions resolves
if [ ! -e "$MBEIR_DATA_DIR/instructions" ]; then
    ln -s "$INST_DIR" "$MBEIR_DATA_DIR/instructions" 2>/dev/null || true
fi

echo ""
echo "Setup complete. Summary:"
echo "  trie_cpp .so : $TRIE_SO"
echo "  qrel files   : $(ls $MBEIR_DATA_DIR/qrels/test/*.txt 2>/dev/null | wc -l) file(s)"
echo "  instructions : $INST_FILE"
