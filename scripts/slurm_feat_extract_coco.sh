#!/bin/bash
#SBATCH --job-name=genius-feat-coco
#SBATCH --output=/home/tcetoje/logs/genius_feat_coco_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_feat_coco_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# Stage 0 candidate extraction for COCO (task0 + task3, train + test pools).
# Required before COCO eval (reranking uses these embeddings).
# Outputs: extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task{0,3}{,_test}_IT_dict.pt

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_cand_coco.yaml"
NPROC=4

export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"
echo "MBEIR data: $MBEIR_DATA_DIR"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29505 \
    "$MODEL_DIR/clip_feature_extraction_cand.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo "Done. Embeddings in: $GENIR_DIR/extracted_embed/CLIP_SF/cand/"
ls -lh "$GENIR_DIR/extracted_embed/CLIP_SF/cand/"*mscoco* 2>/dev/null
