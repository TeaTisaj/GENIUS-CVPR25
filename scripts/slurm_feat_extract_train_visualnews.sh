#!/bin/bash
#SBATCH --job-name=genius-feat-train-visualnews
#SBATCH --output=/home/tcetoje/logs/genius_feat_train_visualnews_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_feat_train_visualnews_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=64G

# Stage-0 train-side (query+candidate) feature extraction for VisualNews, the
# 3rd-dataset ECIR full-paper extension. Mirrors slurm_feat_extract_train_fashioniq.sh
# exactly: only task0 (100,000 image candidates + 99,903 text queries) is used for
# Stage-1 RQ training, matching COCO/FashionIQ's own convention (Stage-1 is trained
# contrastively on task0 T->I pairs only; task3 candidates are quantized later via
# the trained RQ's inference() at gen-code/eval time, not seen during Stage-1
# training itself). model.emb_save_path is a DISTINCT directory
# (extracted_embed/CLIP_SF/train_visualnews) since clip_feature_extration_train.py
# writes fixed filenames into whatever emb_save_path is given.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_train_visualnews.yaml"
NPROC=4

export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Node:         $(hostname)"
echo "Config:       $CONFIG_PATH"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

mkdir -p "$GENIR_DIR/extracted_embed/CLIP_SF/train_visualnews"

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29505 \
    "$MODEL_DIR/clip_feature_extration_train.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: clip_feature_extration_train.py failed with exit code $STATUS"
    exit $STATUS
fi

echo "Done. Embeddings saved to: $GENIR_DIR/extracted_embed/CLIP_SF/train_visualnews/"
