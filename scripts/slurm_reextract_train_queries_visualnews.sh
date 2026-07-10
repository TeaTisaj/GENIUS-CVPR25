#!/bin/bash
#SBATCH --job-name=genius-reextract-train-queries-visualnews
#SBATCH --output=/home/tcetoje/logs/genius_reextract_train_queries_visualnews_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_reextract_train_queries_visualnews_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=01:30:00
#SBATCH --mem=64G

# BUG FIX: the original train-pool extraction (335036) only embedded task0
# (T->I) queries, but Stage-2 training needs BOTH task0+task3 queries (it
# trains on the combined file, unlike Stage-1 which legitimately only uses
# task0) -- caused a KeyError crash in Stage-2 (job 335041) for task3 rows
# with no embedding. This re-extracts query embeddings ONLY (--query_only,
# candidate pool embeddings from 335036 are untouched and still correct) from
# the combined query file, overwriting query_SFpretrained_instruction_IT_dict.pt
# in train_visualnews/ with all 199,903 queries. Safe for the already-completed
# Stage-1 runs (335039/335040): Stage-1 only ever looked up task0 qids from
# this dict, so a superset of embeddings changes nothing for them.
# 2 GPUs required, not 1: clip_feature_extration_train.py's query-embedding
# loop only assigns q_image_mask/q_text_mask inside `if world_size > 1`.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
CONFIG_PATH="$SRC/feature_extraction/config_train_visualnews.yaml"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node:        $(hostname)"
echo "Config:      $CONFIG_PATH"

cd "$SRC/feature_extraction"
python3 -m torch.distributed.run --nproc_per_node=2 --master_port 29520 clip_feature_extration_train.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --query_only

echo "Extraction complete."
