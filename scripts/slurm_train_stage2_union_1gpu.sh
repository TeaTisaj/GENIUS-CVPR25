#!/bin/bash
#SBATCH --job-name=genius-stage2-union
#SBATCH --output=/home/tcetoje/logs/genius_stage2_union_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_stage2_union_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=96:00:00
#SBATCH --mem=128G

# Stage 2: Train T5-small Generative Retriever on FULL M-BEIR union, 1 GPU.
# 4-GPU run OOM'd (4 ranks × 19 GB embeddings > 128 GB RAM).
# Uses official rq_clip_large.pth as frozen quantizer.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/train/inbatch/inbatch.yaml"
NPROC=1

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"
echo "MBEIR data: $MBEIR_DATA_DIR"
echo "Job ID:     $SLURM_JOB_ID"

POOL_PT="$GENIR_DIR/extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt"
QUERY_PT="$GENIR_DIR/extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt"
if [ ! -f "$POOL_PT" ] || [ ! -f "$QUERY_PT" ]; then
    echo "ERROR: Stage 0 embeddings not found."
    exit 1
fi
echo "Stage 0 embeddings: OK"

RQ_CKPT="$GENIR_DIR/checkpoint/rq_clip_large.pth"
if [ ! -f "$RQ_CKPT" ]; then
    echo "ERROR: rq_clip_large.pth not found at $RQ_CKPT"
    exit 1
fi
echo "Quantizer (frozen):  $RQ_CKPT"

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
ls -lh "$GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/InBatch/" 2>/dev/null | sort
