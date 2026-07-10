#!/bin/bash
#SBATCH --job-name=genius-eval-fashioniq-cfiq
#SBATCH --output=/home/tcetoje/logs/genius_eval_fashioniq_cfiq_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_fashioniq_cfiq_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# ── Paths ────────────────────────────────────────────────────────────────────
GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_fashioniq_coco_fashioniq.yaml"
NPROC=1

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "Node:         $(hostname)"
echo "PYTHONPATH:   $PYTHONPATH"
echo "GENIR_DIR:    $GENIR_DIR"
echo "MBEIR_DATA:   $MBEIR_DATA_DIR"
echo "Config:       $CONFIG_PATH"

# ── Patch instruct flag ───────────────────────────────────────────────────────
cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

# ── Run eval ─────────────────────────────────────────────────────────────────
cd "$COMMON_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29505 \
    mbeir_generative_retriever.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Results saved to: $GENIR_DIR/retrieval_results/"
echo "TSV summary:      $GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/CocoFashionIQ/final_tsv/"
