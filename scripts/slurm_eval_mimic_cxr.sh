#!/bin/bash
#SBATCH --job-name=genius-eval-mimic-cxr
#SBATCH --output=/home/tcetoje/logs/genius_eval_mimic_cxr_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_mimic_cxr_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# Evaluation — GENIUS-CXR on MIMIC-CXR test split.
# Before running:
#   1. Set model.ckpt_config.ckpt_name in config_eval_mimic_cxr.yaml to Stage 2 checkpoint filename.
#   2. Set codebook_config.quantizer_path to Stage 1 checkpoint path.
#   3. Compile C++ trie if not done yet (see comment below).

# Compile trie once (skip if trie_cpp*.so already exists):
#   cd $GENIR_DIR/src/models/generative_retriever
#   c++ -O3 -Wall -shared -std=c++17 -fPIC \
#       $(python3 -m pybind11 --includes) \
#       trie_cpp.cpp -o trie_cpp$(python3-config --extension-suffix)

# ── Paths ────────────────────────────────────────────────────────────────────
GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
CONFIG_PATH="$SRC/models/generative_retriever/configs_scripts/large/eval/inbatch/config_eval_mimic_cxr.yaml"
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

# ── Run evaluation ───────────────────────────────────────────────────────────
cd "$COMMON_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29513 \
    mbeir_generative_retriever.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Results saved to: $GENIR_DIR/retrieval_results/"
echo "TSV summary:      $GENIR_DIR/retrieval_results/GENIUS_biomedclip/Large/Instruct/InBatch/final_tsv/"
