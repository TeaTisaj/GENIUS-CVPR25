#!/bin/bash
#SBATCH --job-name=genius-eval-coco-only
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_only_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_only_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# Evaluate COCO-only T5 model.
# Expected results (GENIUS paper Table 3, COCO-specific model):
#   T→I  R@1=40.1  R@5=66.2  R@10=75.8  (no rerank)
#   I→T  R@1=46.1  R@5=74.0  R@10=82.7  (with rerank)
# This script runs WITHOUT rerank first; edit config_eval_coco_only.yaml to set rerank: true for rerank run.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_coco_only.yaml"
CKPT_DIR="$GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly"
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
    --master_port 29507 \
    "$SRC/common/mbeir_generative_retriever.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "=== No-rerank results ==="
RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/CocoOnly"
cat "$RESULTS_DIR"/*.tsv 2>/dev/null || echo "(no .tsv results found)"

# Run with rerank
sed -i 's/rerank: false/rerank: true/' "$CONFIG_PATH"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29508 \
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
