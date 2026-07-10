#!/bin/bash
#SBATCH --job-name=genius-extract-test-queries-visualnews
#SBATCH --output=/home/tcetoje/logs/genius_extract_test_queries_visualnews_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_extract_test_queries_visualnews_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=01:00:00
#SBATCH --mem=48G

# Extracts instructed CLIP-SF embeddings for VisualNews's held-out test-split
# queries (task0 T->I + task3 I->T, merged, 39,996 queries). Mirrors
# slurm_extract_test_query_embeds.sh (COCO) exactly, including the
# NPROC=2-not-1 requirement: clip_feature_extration_train.py's query-embedding
# loop only assigns q_image_mask/q_text_mask inside an
# `if utils.get_world_size() > 1:` branch (a real, pre-existing bug in that
# shared script), so 1 GPU fails with UnboundLocalError. Output goes to
# extracted_embed/CLIP_SF/test/ (shared across datasets, distinct filenames
# per dataset's config -- verify no overwrite before trusting this if that
# assumption ever changes).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
CONFIG_PATH="$SRC/feature_extraction/config_test_visualnews_queries.yaml"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node:        $(hostname)"
echo "Config:      $CONFIG_PATH"

cd "$SRC/feature_extraction"
python3 -m torch.distributed.run --nproc_per_node=2 clip_feature_extration_train.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --query_only

echo "Extraction complete."
