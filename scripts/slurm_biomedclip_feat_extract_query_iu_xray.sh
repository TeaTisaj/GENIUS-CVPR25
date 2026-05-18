#!/bin/bash
#SBATCH --job-name=biomedclip-query-iu
#SBATCH --output=/home/tcetoje/logs/biomedclip_query_iu_%j.log
#SBATCH --error=/home/tcetoje/logs/biomedclip_query_iu_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=01:00:00
#SBATCH --mem=32G

# Stage 0b — BiomedCLIP query embedding extraction for IU X-Ray (train + val).
# Prerequisite: cand pool embeddings must exist (slurm_biomedclip_feat_extract_iu_xray.sh).
# Produces extracted_embed/BiomedCLIP/train/train_iu_xray_IT_dict.pt
#              extracted_embed/BiomedCLIP/train/val_iu_xray_IT_dict.pt

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_PATH="$MODEL_DIR/config_biomedclip_query_iu_xray.yaml"
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
    --master_port 29512 \
    biomedclip_feature_extraction.py \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo "Done. Query embeddings: $GENIR_DIR/extracted_embed/BiomedCLIP/train/"
echo "Next: sbatch scripts/slurm_rq_train_iu_xray.sh"
