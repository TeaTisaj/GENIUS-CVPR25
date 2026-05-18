#!/bin/bash
#SBATCH --job-name=rq-train-biomedclip
#SBATCH --output=/home/tcetoje/logs/rq_train_biomedclip_%j.log
#SBATCH --error=/home/tcetoje/logs/rq_train_biomedclip_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# Stage 1 — Residual Quantization training on BiomedCLIP (512-dim) embeddings.
# Prerequisite: Stage 0 (slurm_biomedclip_feat_extract.sh) must have completed.
# Checkpoint saved to: checkpoint/rq_biomedclip/Large/Instruct/InBatch/

# ── Paths ────────────────────────────────────────────────────────────────────
GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/residual_quantization"
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

# ── Run RQ training ──────────────────────────────────────────────────────────
cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29511 \
    train.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Done. Checkpoint saved to: $GENIR_DIR/checkpoint/rq_biomedclip/Large/Instruct/InBatch/"
echo "Set codebook_config.quantizer_path in inbatch_biomedclip.yaml (Stage 2) to best.pth before running generator training."
