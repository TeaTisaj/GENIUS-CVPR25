#!/bin/bash
#SBATCH --job-name=genius-train-biomedclip
#SBATCH --output=/home/tcetoje/logs/genius_train_biomedclip_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_train_biomedclip_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# Stage 2 — T5-small generator training for GENIUS-CXR.
# Prerequisite: Stage 1 checkpoint must exist.
# Before running: set codebook_config.quantizer_path in inbatch_biomedclip.yaml
#   to the Stage 1 best.pth, e.g.:
#   checkpoint/rq_biomedclip/Large/Instruct/InBatch/best.pth

# ── Paths ────────────────────────────────────────────────────────────────────
GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/train/inbatch/inbatch_biomedclip.yaml"
NPROC=4

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Node:       $(hostname)"
echo "PYTHONPATH: $PYTHONPATH"
echo "GENIR_DIR:  $GENIR_DIR"
echo "MBEIR_DATA: $MBEIR_DATA_DIR"
echo "Config:     $CONFIG_PATH"

# ── Patch instruct flag ───────────────────────────────────────────────────────
cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

# ── Run generator training ───────────────────────────────────────────────────
cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29512 \
    train.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Done. Checkpoint saved to: $GENIR_DIR/checkpoint/GENIUS_biomedclip/Large/Instruct/InBatch/"
