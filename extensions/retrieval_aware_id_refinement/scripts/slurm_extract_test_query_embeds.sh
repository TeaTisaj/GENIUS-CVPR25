#!/bin/bash
#SBATCH --job-name=genius-extract-test-queries
#SBATCH --output=/home/tcetoje/logs/genius_extract_test_queries_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_extract_test_queries_%x_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=01:00:00
#SBATCH --mem=48G

# Component 9, Step 2 (correcting the Criterion-5 probe methodology): extracts
# instructed CLIP-SF embeddings for the official, held-out MSCOCO test-split
# queries (task0 T->I + task3 I->T, merged, 29,809 queries) -- these do not
# exist anywhere on disk yet (verified in Step 1's pre-flight check). Output
# goes to extracted_embed/CLIP_SF/test/ (a NEW directory -- the output
# filename is hard-coded by clip_feature_extration_train.py, so saving under
# train/ would silently overwrite the existing train query dict).
#
# Uses 2 GPUs (--nproc_per_node=2), not 1: clip_feature_extration_train.py's
# query-embedding loop only assigns q_image_mask/q_text_mask inside an
# `if utils.get_world_size() > 1:` branch (a real, pre-existing bug in that
# shared script -- confirmed via a failed 1-GPU run, UnboundLocalError on
# q_image_mask). Every other caller of this script in the repo always uses
# NPROC>=4, so this bug was never hit before. Fix here is to match that
# established convention (world_size>1), not patch the shared extraction
# script for one caller.
#
# Usage: sbatch scripts/slurm_extract_test_query_embeds.sh

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
CONFIG_PATH="$SRC/feature_extraction/config_test_coco_queries.yaml"

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
