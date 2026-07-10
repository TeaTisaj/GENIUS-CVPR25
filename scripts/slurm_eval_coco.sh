#!/bin/bash
#SBATCH --job-name=genius-eval-coco
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# Evaluate trained GENIUS on COCO (task0: T→I, task3: I→T).
# Prerequisites:
#   1. slurm_train_stage2.sh completed
#   2. slurm_feat_extract_coco.sh completed (needed for reranking)
#   3. Update ckpt_name in config_eval_coco.yaml to your trained checkpoint filename
#
# Compare results against GENIUS paper Table 3:
#   COCO T→I  R@1=40.1  R@5=66.2  R@10=75.8
#   COCO I→T  R@1=46.1  R@5=74.0  R@10=82.7  (with reranking)

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_coco.yaml"
NPROC=4

export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Verify checkpoint is set
CKPT_NAME=$(grep 'ckpt_name' "$CONFIG_PATH" | head -1 | awk '{print $2}')
if [[ "$CKPT_NAME" == *"PLACEHOLDER"* ]]; then
    echo "ERROR: Update ckpt_name in $CONFIG_PATH before running eval."
    echo "  Checkpoint dir: $GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/CocoFashionIQ/"
    ls "$GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/CocoFashionIQ/" 2>/dev/null || echo "  (dir not yet created)"
    exit 1
fi

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"
echo "Checkpoint: $CKPT_NAME"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29506 \
    mbeir_generative_retriever.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Results: $GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/CocoFashionIQ/"
