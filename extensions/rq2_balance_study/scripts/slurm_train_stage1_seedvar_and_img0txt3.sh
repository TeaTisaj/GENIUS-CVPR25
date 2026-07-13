#!/bin/bash
#SBATCH --job-name=genius-s1-seedvar-img0txt3
#SBATCH --output=/home/tcetoje/logs/genius_stage1_seedvar_img0txt3_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_stage1_seedvar_img0txt3_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=02:00:00
#SBATCH --mem=128G

# ECIR audit Round 5 fix plan, Phase 1 + Phase 2 (2026-07-13): Stage-1 training for
#   - 2 new Stage-1 seeds (7, 13) x {vanilla, strong} on MSCOCO, to measure tokenizer-level
#     seed variance (blocking item 1: the headline T->I claim currently rests on n=1 Stage-1
#     tokenizer per condition).
#   - img0txt3 (lambda_img=0, lambda_txt=3), the missing diagnostic cell in the
#     direction-aware-lambda grid (high-impact item 5).
# Each run is ~12 min per prior job logs (332120/332127) -- COCO Stage-1 only trains the RQ
# codebook/combiner on frozen CLIP-SF embeddings, not the vision/text backbones.
# Vanilla's new configs are repointed to fnwi_fs (local disk has only 17GB free, see
# feedback_disk_full_checkpoint_redirect.md); strong/img0txt3 already target fnwi_fs.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/residual_quantization"
CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
NPROC=4

CONFIGS="inbatch_coco_vanilla_s1seed7 inbatch_coco_vanilla_s1seed13 inbatch_coco_strong_s1seed7 inbatch_coco_strong_s1seed13 inbatch_coco_img0txt3"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

for CFG in $CONFIGS; do
    echo ""
    echo "##### Stage-1 training: $CFG #####"

    CONFIG="$CONFIG_DIR/${CFG}.yaml"

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$CONFIG" \
        --enable_instruct True

    cd "$MODEL_DIR"
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29900 + RANDOM % 1000)) \
        train.py \
        --config_path "$CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "ERROR: Stage-1 training for $CFG failed with exit code $STATUS"
        exit $STATUS
    fi
    echo "##### Done: $CFG #####"
done

echo ""
echo "=== All 5 Stage-1 runs (seed-variance x4 + img0txt3 x1) complete. ==="
