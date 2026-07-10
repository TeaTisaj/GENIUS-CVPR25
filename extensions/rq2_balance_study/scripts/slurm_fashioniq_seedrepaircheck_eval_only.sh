#!/bin/bash
#SBATCH --job-name=genius-fiq-seedrepaircheck-eval
#SBATCH --output=/home/tcetoje/logs/genius_fiq_seedrepaircheck_eval_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_fiq_seedrepaircheck_eval_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=00:30:00
#SBATCH --mem=128G

# Eval-only retry (2026-07-03): job 333704's weak eval failed with "prefix_allowed_tokens_fn
# returned an empty list for batch ID 106" (unsatisfiable beam constraint) after training
# completed fine for both weak/medium (loss curves healthy, both checkpoints saved through
# epoch_20). Medium's eval never ran because the script exits on first failure. This reruns
# eval only for both, using the already-trained checkpoints -- no retraining needed unless
# weak fails again identically (which would indicate a real, reproducible bad model, not a
# transient glitch).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
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
    CAP=$(python3 -c "print('${VARIANT}'.capitalize())")
    EXP_NAME="FashioniqOnly${CAP}SeedRepairCheck"
    EVAL_CONFIG="$EVAL_CONFIG_DIR/config_eval_fashioniq_only_${VARIANT}_seedrepaircheck.yaml"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

    echo ""
    echo "##### Evaluating (retry): $VARIANT seedrepaircheck (exp_name=$EXP_NAME) #####"

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$EVAL_CONFIG" \
        --enable_instruct True

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29990 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$EVAL_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "ERROR: seedrepaircheck eval retry for $VARIANT failed with exit code $STATUS"
        continue
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
echo "=== Eval retry pass complete (check per-variant status above -- this script does not hard-exit on failure so both variants get a chance). ==="
