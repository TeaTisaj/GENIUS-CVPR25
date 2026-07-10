#!/bin/bash
#SBATCH --job-name=genius-stage0-query
#SBATCH --output=/home/tcetoje/logs/genius_stage0_query_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_stage0_query_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=08:00:00
#SBATCH --mem=128G

# Stage 0 (query only): re-extract corrupted query embeddings for full M-BEIR union.
# Pool file already OK: extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt
# Output: extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_train.yaml"
NPROC=4

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"
echo "MBEIR data: $MBEIR_DATA_DIR"
echo "Job ID:     $SLURM_JOB_ID"

QUERY_JSONL="$MBEIR_DATA_DIR/query/union_train/mbeir_union_up_train.jsonl"
if [ ! -f "$QUERY_JSONL" ]; then
    echo "ERROR: $QUERY_JSONL not found."
    exit 1
fi
echo "Union query JSONL: $(wc -l < $QUERY_JSONL) entries"

# Delete corrupted file before re-extracting
rm -f "$GENIR_DIR/extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt"
echo "Removed corrupted query file."

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 1326 \
    "$MODEL_DIR/clip_feature_extration_train.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --query_only

echo ""
echo "Query extraction complete. Output:"
ls -lh "$GENIR_DIR/extracted_embed/CLIP_SF/train/"
