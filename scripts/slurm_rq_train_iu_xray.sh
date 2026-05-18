#!/bin/bash
#SBATCH --job-name=rq-train-iu
#SBATCH --output=/home/tcetoje/logs/rq_train_iu_%j.log
#SBATCH --error=/home/tcetoje/logs/rq_train_iu_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=04:00:00
#SBATCH --mem=32G

# Stage 1 — RQ training on IU X-Ray BiomedCLIP embeddings (smoke test).
# Prerequisite: slurm_biomedclip_feat_extract_iu_xray.sh must have completed.
# Checkpoint: checkpoint/rq_biomedclip_iu_xray/Large/Instruct/InBatch/

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/residual_quantization"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/train/inbatch/inbatch_iu_xray.yaml"
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
    --master_port 29512 \
    train.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Done. Set quantizer_path in inbatch_iu_xray.yaml (Stage 2) to best.pth before continuing."
