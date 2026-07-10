#!/bin/bash
#SBATCH --job-name=genius-feat-visualnews
#SBATCH --output=/home/tcetoje/logs/genius_feat_visualnews_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_feat_visualnews_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # torchrun manages the 4 worker processes
#SBATCH --cpus-per-task=32               # 4 GPUs × 8 workers each
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# ── Paths ────────────────────────────────────────────────────────────────────
GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_cand_visualnews.yaml"
NPROC=4

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Node:         $(hostname)"
echo "PYTHONPATH:   $PYTHONPATH"
echo "GENIR_DIR:    $GENIR_DIR"
echo "MBEIR_DATA:   $MBEIR_DATA_DIR"
echo "Config:       $CONFIG_PATH"

# ── Patch instruct flag in config ────────────────────────────────────────────
cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

# ── Run feature extraction ───────────────────────────────────────────────────
cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29509 \
    "$MODEL_DIR/clip_feature_extraction_cand.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo "Done. Embeddings saved to: $GENIR_DIR/extracted_embed/CLIP_SF/cand/cand_pool_visualnews_task0_IT_dict.pt"
