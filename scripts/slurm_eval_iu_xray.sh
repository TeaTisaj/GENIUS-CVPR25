#!/bin/bash
#SBATCH --job-name=eval-iu
#SBATCH --output=/home/tcetoje/logs/eval_iu_%j.log
#SBATCH --error=/home/tcetoje/logs/eval_iu_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Eval — GENIUS IU X-Ray smoke test.
# Prerequisite: Stage 2 done; set model.ckpt_config.ckpt_name in config_eval_iu_xray.yaml.
# C++ trie must be compiled: see CLAUDE.md Inference / Evaluation section.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_iu_xray.yaml"
NPROC=2

export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29514 \
    eval.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Results in: $GENIR_DIR/retrieval_results/"
