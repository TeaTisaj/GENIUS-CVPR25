#!/bin/bash
#SBATCH --job-name=genius-feat-train-fashioniq
#SBATCH --output=/home/tcetoje/logs/genius_feat_train_fashioniq_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_feat_train_fashioniq_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Stage-0 train-side (query+candidate) feature extraction for FashionIQ, RQ3 prep.
# Mirrors scripts/slurm_feat_extract_fashioniq.sh (cand-side, already done/verified complete)
# but runs clip_feature_extration_train.py against config_train_fashioniq.yaml. That config's
# model.emb_save_path is extracted_embed/CLIP_SF/train_fashioniq -- a DISTINCT directory from
# COCO's extracted_embed/CLIP_SF/train/, since clip_feature_extration_train.py saves to a fixed
# filename pair (pool_SFpretrained_IT_dict.pt, query_SFpretrained_instruction_IT_dict.pt) inside
# whatever emb_save_path is given -- reusing COCO's directory would silently overwrite its dicts.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_train_fashioniq.yaml"
NPROC=4

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

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

mkdir -p "$GENIR_DIR/extracted_embed/CLIP_SF/train_fashioniq"

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29504 \
    "$MODEL_DIR/clip_feature_extration_train.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: clip_feature_extration_train.py failed with exit code $STATUS"
    exit $STATUS
fi

echo "Done. Embeddings saved to: $GENIR_DIR/extracted_embed/CLIP_SF/train_fashioniq/"
