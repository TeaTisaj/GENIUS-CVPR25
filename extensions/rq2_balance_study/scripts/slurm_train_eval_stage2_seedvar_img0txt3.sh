#!/bin/bash
#SBATCH --job-name=genius-s2-seedvar-img0txt3
#SBATCH --output=/home/tcetoje/logs/genius_stage2_seedvar_img0txt3_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_stage2_seedvar_img0txt3_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=08:00:00
#SBATCH --mem=128G

# ECIR audit Round 5 fix plan, Phase 1 + Phase 2 (2026-07-13): Stage-2 train+eval on top of
# the 5 new Stage-1 checkpoints from job 335469 (verified COMPLETED, all 5 checkpoints present
# on fnwi_fs). Phase 1 (4 combos): MSCOCO vanilla/strong at 2 new Stage-1 seeds (7, 13), each
# with ONE Stage-2 seed (2023) on top -- gives n=3 independent tokenizers per condition
# (existing seed=2023 + these 2), the load-bearing check for whether the headline T->I claim
# survives Stage-1 reseeding. Phase 2 (3 combos): img0txt3 (the missing direction-aware-lambda
# diagnostic cell) at 3 Stage-2 seeds (2023, 7, 13), matching img3txt0/img3txt0p3's existing
# convention. Train immediately followed by eval for each combo (not all-train-then-all-eval)
# so a later combo's crash doesn't block already-trained earlier combos from being evaluated.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
TRAIN_DIR="$SRC/models/generative_retriever"
TRAIN_CFG_DIR="$TRAIN_DIR/configs_scripts/large/train/inbatch"
EVAL_CFG_DIR="$TRAIN_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

CONFIGS="vanilla_s1seed7 vanilla_s1seed13 strong_s1seed7 strong_s1seed13 img0txt3 img0txt3_seed7 img0txt3_seed13"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_stage1_seedvar_and_img0txt3"
mkdir -p "$SWEEP_DIR"

for CFG in $CONFIGS; do
    TRAIN_CONFIG="$TRAIN_CFG_DIR/inbatch_coco_only_${CFG}.yaml"
    EVAL_CONFIG="$EVAL_CFG_DIR/config_eval_coco_only_${CFG}.yaml"

    echo ""
    echo "##### Stage-2 TRAIN: $CFG #####"
    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TRAIN_CONFIG" \
        --enable_instruct True

    cd "$TRAIN_DIR"
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29600 + RANDOM % 1000)) \
        train.py \
        --config_path "$TRAIN_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    TRAIN_STATUS=$?

    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "ERROR: Stage-2 training for $CFG failed with exit code $TRAIN_STATUS -- skipping its eval"
        continue
    fi
    echo "##### Stage-2 train done: $CFG #####"

    echo ""
    echo "##### Stage-2 EVAL: $CFG #####"
    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$EVAL_CONFIG" \
        --enable_instruct True

    EXP_NAME=$(grep -A2 "^experiment:" "$EVAL_CONFIG" | grep "exp_name:" | awk '{print $2}')
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29650 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$EVAL_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    EVAL_STATUS=$?

    if [ $EVAL_STATUS -ne 0 ]; then
        echo "ERROR: Stage-2 eval for $CFG failed with exit code $EVAL_STATUS"
        continue
    fi

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$SWEEP_DIR/${CFG}.tsv"
        echo "Saved results for $CFG -> $SWEEP_DIR/${CFG}.tsv"
    else
        echo "WARNING: no results tsv found for $CFG"
    fi
    echo "##### Done: $CFG #####"
done

echo ""
echo "=== All 7 Stage-2 train+eval combos (Phase 1 x4 + Phase 2 x3) complete. ==="
