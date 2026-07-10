#!/bin/bash
#SBATCH --job-name=genius-diraware-seedvar-retrain
#SBATCH --output=/home/tcetoje/logs/genius_diraware_seedvar_retrain_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_diraware_seedvar_retrain_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=200G

# All 4 img3txt0/img3txt0p3 seed7/13 eval runs (job 334982) crashed with the
# known "unstable-codebook beam-search decode fragility" signature (CUDA
# device-side assert / prefix_allowed_tokens_fn empty-list) -- see
# feedback_beam_search_decode_fragility.md: this hit FashionIQ's weak/medium
# checkpoints before, and the established fix is RETRAIN FROM SCRATCH (a
# fresh, cuDNN-non-deterministic rerun of the identical config), not
# retry-eval-on-the-same-checkpoint (which fails again deterministically-ish
# on a different batch). Training itself (334981) succeeded cleanly with no
# errors -- only the eval/decode step crashed -- so this reruns Stage-2
# training fresh for all 4 combos (overwriting the unstable checkpoints),
# then evaluates each immediately after, continuing past any single-item
# failure rather than hard-exiting (in case one combo needs a second retry).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
TRAIN_CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

COMBOS=(
    "img3txt0:7"
    "img3txt0:13"
    "img3txt0p3:7"
    "img3txt0p3:13"
)

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_direction_aware_seed_variance"
mkdir -p "$SWEEP_DIR"

for ENTRY in "${COMBOS[@]}"; do
    VARIANT="${ENTRY%%:*}"
    SEED="${ENTRY##*:}"
    TRAIN_CONFIG="$TRAIN_CONFIG_DIR/inbatch_coco_only_${VARIANT}_seed${SEED}.yaml"
    EVAL_CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}_seed${SEED}.yaml"
    SEED_EXP_NAME="CocoOnly$(python3 -c "print('${VARIANT}'.capitalize())")Seed${SEED}"

    echo ""
    echo "##### RETRAIN: $VARIANT seed=$SEED #####"

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TRAIN_CONFIG" \
        --enable_instruct True

    cd "$MODEL_DIR"
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29700 + RANDOM % 1000)) \
        train.py \
        --config_path "$TRAIN_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    TRAIN_STATUS=$?

    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "ERROR: retrain for $VARIANT seed=$SEED failed with exit code $TRAIN_STATUS -- skipping its eval"
        continue
    fi
    echo "##### Retrain done: $VARIANT seed=$SEED, now evaluating #####"

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$EVAL_CONFIG" \
        --enable_instruct True

    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${SEED_EXP_NAME}/final_tsv"
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29750 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$EVAL_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    EVAL_STATUS=$?

    if [ $EVAL_STATUS -ne 0 ]; then
        echo "ERROR: eval for $VARIANT seed=$SEED failed again (exit $EVAL_STATUS) -- may need a second retrain attempt"
        continue
    fi

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$SWEEP_DIR/${VARIANT}_seed${SEED}.tsv"
        echo "Saved results for $VARIANT seed=$SEED -> $SWEEP_DIR/${VARIANT}_seed${SEED}.tsv"
    else
        echo "WARNING: no results tsv found for $VARIANT seed=$SEED despite exit code 0"
    fi
done

echo ""
echo "=== Retrain-and-eval pass complete. Check each combo's outcome above before trusting the sweep dir. ==="
