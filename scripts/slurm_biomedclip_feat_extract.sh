#!/bin/bash
#SBATCH --job-name=biomedclip-feat
#SBATCH --output=/home/tcetoje/logs/biomedclip_feat_%j.log
#SBATCH --error=/home/tcetoje/logs/biomedclip_feat_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=06:00:00
#SBATCH --mem=64G

# Stage 0 — BiomedCLIP feature extraction for MIMIC-CXR candidates and queries.
# Produces .pt embedding files under extracted_embed/BiomedCLIP/ consumed by Stage 1.
# BiomedCLIP weights are downloaded from HuggingFace on first run (~1 GB cache).

# ── Paths ────────────────────────────────────────────────────────────────────
GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_biomedclip_cand.yaml"
COMMON_DIR="$SRC/common"
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

# ── Run feature extraction ───────────────────────────────────────────────────
cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29510 \
    biomedclip_feature_extraction.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Done. Embeddings saved to: $GENIR_DIR/extracted_embed/BiomedCLIP/"
