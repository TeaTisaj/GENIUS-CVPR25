#!/bin/bash
#SBATCH --job-name=genius-eval-coco-only-1gpu
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_only_1gpu_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_only_1gpu_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --exclude=ilps-cn120

# Evaluate the CocoOnly1GPU checkpoint (job 328629: identical 30-epoch/lr=1e-4/batch=256
# recipe as the 4-GPU "by-the-book" run, but trained on 1 GPU -> ~25,000 optimizer steps
# instead of ~6,240, final train loss ~3.35 vs 4.12). This is the direct test of whether
# the lower training loss translates into better recall (Phase B paradox resolution).
# Reference numbers (T->I, COCO test):
#   4-GPU "by-the-book" 30ep (epoch_25): R@1=5.76% (no-rerank) / 11.47% (rerank)
#   Pretrained HF general:               R@1=14.2%
#   Locally trained 200-epoch union:     R@1=11.6%
#   Paper Table 3 (COCO-specific):       R@1=40.1% (46.1 reranked)
# This script runs WITHOUT rerank first, then WITH rerank (both passes from one job).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_coco_only_1gpu.yaml"
CKPT_DIR="$GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly1GPU"
NPROC=4

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"
echo "Job ID:     $SLURM_JOB_ID"

# Auto-select the latest epoch checkpoint if epoch_25 is missing
CKPT_NAME=$(grep 'ckpt_name' "$CONFIG_PATH" | head -1 | awk '{print $2}')
if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
    echo "WARNING: $CKPT_DIR/$CKPT_NAME not found. Auto-selecting latest checkpoint..."
    LATEST=$(ls -t "$CKPT_DIR"/*.pth 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No checkpoints found in $CKPT_DIR"
        exit 1
    fi
    CKPT_NAME=$(basename "$LATEST")
    echo "Using checkpoint: $CKPT_NAME"
    sed -i "s|ckpt_name:.*|ckpt_name: $CKPT_NAME|" "$CONFIG_PATH"
fi
echo "Checkpoint: $CKPT_DIR/$CKPT_NAME"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

cd "$MODEL_DIR"
# Run without rerank
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29509 \
    "$SRC/common/mbeir_generative_retriever.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "=== No-rerank results ==="
RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/CocoOnly1GPU"
cat "$RESULTS_DIR"/*.tsv 2>/dev/null || echo "(no .tsv results found)"

# Run with rerank
sed -i 's/rerank: false/rerank: true/' "$CONFIG_PATH"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29510 \
    "$SRC/common/mbeir_generative_retriever.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "=== With-rerank results ==="
cat "$RESULTS_DIR"/*.tsv 2>/dev/null || echo "(no .tsv results found)"

# Restore rerank: false in config
sed -i 's/rerank: true/rerank: false/' "$CONFIG_PATH"
echo ""
echo "Eval complete. Results: $RESULTS_DIR"
