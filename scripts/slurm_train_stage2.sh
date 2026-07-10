#!/bin/bash
#SBATCH --job-name=genius-stage2
#SBATCH --output=/home/tcetoje/logs/genius_stage2_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_stage2_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Stage 2: Train T5-small Generative Retriever on COCO + FashionIQ.
# Prerequisite: scripts/slurm_stage0_coco_fashioniq.sh must have completed.
# Checkpoint saved to: checkpoint/GENIUS_t5small/Large/Instruct/CocoFashionIQ/

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/train/inbatch/inbatch_coco_fashioniq.yaml"
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

# Verify Stage 0 outputs exist
POOL_PT="$GENIR_DIR/extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt"
QUERY_PT="$GENIR_DIR/extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt"
if [ ! -f "$POOL_PT" ] || [ ! -f "$QUERY_PT" ]; then
    echo "ERROR: Stage 0 embeddings not found. Run slurm_stage0_coco_fashioniq.sh first."
    echo "  Missing: $POOL_PT"
    echo "  Missing: $QUERY_PT"
    exit 1
fi
echo "Stage 0 embeddings: OK"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --master_port 1325 \
    --nproc_per_node=$NPROC \
    "$MODEL_DIR/train.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Stage 2 training complete."
echo "Checkpoint: $GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/CocoFashionIQ/"
