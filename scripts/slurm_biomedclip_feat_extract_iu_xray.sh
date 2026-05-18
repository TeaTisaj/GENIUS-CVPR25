#!/bin/bash
#SBATCH --job-name=biomedclip-feat-iu
#SBATCH --output=/home/tcetoje/logs/biomedclip_feat_iu_%j.log
#SBATCH --error=/home/tcetoje/logs/biomedclip_feat_iu_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Stage 0 — BiomedCLIP feature extraction for IU X-Ray (smoke test).
# IU X-Ray is ~7 k images; 2 GPUs and 2 h is generous.
# Produces .pt files under extracted_embed/BiomedCLIP/ for Stage 1.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_biomedclip_cand_iu_xray.yaml"
COMMON_DIR="$SRC/common"
NPROC=2

export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

echo "Node:       $(hostname)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Config:     $CONFIG_PATH"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port 29511 \
    biomedclip_feature_extraction.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo "Done. Embeddings: $GENIR_DIR/extracted_embed/BiomedCLIP/"
echo "Next: sbatch scripts/slurm_rq_train_biomedclip.sh  (with inbatch_iu_xray.yaml)"
