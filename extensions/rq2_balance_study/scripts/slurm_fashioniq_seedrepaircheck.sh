#!/bin/bash
#SBATCH --job-name=genius-fiq-seedrepaircheck
#SBATCH --output=/home/tcetoje/logs/genius_fiq_seedrepaircheck_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_fiq_seedrepaircheck_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=01:00:00
#SBATCH --mem=128G

# FashionIQ weak/medium seed-variance confound fix (2026-07-03): the existing seed-variance
# table's reused seed=2023 point for weak/medium was trained on the ORIGINAL (now-lost) Stage-1
# quantizer, while seed7/13 were trained on the REPAIRED one -- and those quantizers differ ~50%
# in level-1 code concentration for weak/medium specifically (see project_rq2_balance_regularization.md
# item 10). This retrains a clean seed=2023-equivalent point for weak/medium ONLY (vanilla had
# tight repair-parity, strong's quantizer was never touched -- both already clean), on the
# REPAIRED quantizer, under a fresh exp_name (*SeedRepairCheck) so the original confounded
# checkpoints/results are left untouched on disk for audit. This replaces only the seed2023
# column for weak/medium in the seed-variance aggregation, not the whole table.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
TRAIN_CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
EVAL_CONFIG_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

VARIANTS="weak medium"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/fashioniq_seed_variance"
mkdir -p "$SWEEP_DIR"

for VARIANT in $VARIANTS; do
    echo ""
    echo "##### Training: $VARIANT seedrepaircheck (repaired quantizer, seed=2023) #####"

    TRAIN_CONFIG="$TRAIN_CONFIG_DIR/inbatch_fashioniq_only_${VARIANT}_seedrepaircheck.yaml"
    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$TRAIN_CONFIG" \
        --enable_instruct True

    cd "$MODEL_DIR"
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29970 + RANDOM % 1000)) \
        train.py \
        --config_path "$TRAIN_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "ERROR: seedrepaircheck training for $VARIANT failed with exit code $STATUS"
        exit $STATUS
    fi
    echo "##### Done training: $VARIANT seedrepaircheck #####"
done

for VARIANT in $VARIANTS; do
    CAP=$(python3 -c "print('${VARIANT}'.capitalize())")
    EXP_NAME="FashioniqOnly${CAP}SeedRepairCheck"
    EVAL_CONFIG="$EVAL_CONFIG_DIR/config_eval_fashioniq_only_${VARIANT}_seedrepaircheck.yaml"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

    echo ""
    echo "##### Evaluating: $VARIANT seedrepaircheck (exp_name=$EXP_NAME) #####"

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$EVAL_CONFIG" \
        --enable_instruct True

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29980 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$EVAL_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "ERROR: seedrepaircheck eval for $VARIANT failed with exit code $STATUS"
        exit $STATUS
    fi

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$SWEEP_DIR/${VARIANT}_seedrepaircheck.tsv"
        echo "Saved results for $VARIANT seedrepaircheck -> $SWEEP_DIR/${VARIANT}_seedrepaircheck.tsv"
    else
        echo "WARNING: no results tsv found for $VARIANT seedrepaircheck"
    fi
done

echo ""
echo "=== FashionIQ weak/medium seed-repair-check (train+eval) complete. ==="
