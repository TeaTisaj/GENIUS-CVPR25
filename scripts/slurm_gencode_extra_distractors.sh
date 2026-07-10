#!/bin/bash
#SBATCH --job-name=genius-gencode-extra
#SBATCH --output=/home/tcetoje/logs/genius_gencode_extra_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_gencode_extra_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G

# Stage-1 quantizer-only RQ encode for Fashion200K/NIGHTS/CIRR candidate pools
# (GENIUS Phase 2 candidate-pool composition study). No beam search / T5
# decoding involved -- reuses the existing eval entrypoint, which generates
# cand-pool codes via the frozen RQ quantizer whenever enable_retrieve is
# false for all query splits.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_gencode_extra_distractors.yaml"
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

# ── Run gen-code ─────────────────────────────────────────────────────────────
cd "$COMMON_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29508 \
    mbeir_generative_retriever.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Codes saved to: $GENIR_DIR/gen_code/GENIUS_t5small/Large/Instruct/InBatch/cand_pool/"
