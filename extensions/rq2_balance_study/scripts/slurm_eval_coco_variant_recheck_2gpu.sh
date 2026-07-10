#!/bin/bash
#SBATCH --job-name=genius-eval-coco-variant-recheck
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_variant_recheck_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_variant_recheck_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --exclude=ilps-cn117,ilps-cn118,ilps-cn120

# 2-GPU variant of slurm_eval_coco_variant_recheck.sh (2026-07-02), submitted to fit into
# cn120's 2 free A6000 GPUs instead of waiting indefinitely for 4 free ones elsewhere. Safe to
# reduce GPU count here because this is pure eval/inference (frozen checkpoint, deterministic
# beam search + trie lookup) -- world size only affects wall-clock time, not results. This is
# NOT safe for training jobs (e.g. slurm_repair_fashioniq_stage1.sh), where GPU count changes
# the effective global batch size and would compromise the parity check that job feeds into.
# Time limit doubled (4h->6h with margin) since NPROC halved roughly doubles wall time.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=2

VARIANTS="vanilla weak medium strong"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

RECHECK_DIR="$GENIR_DIR/retrieval_results/coco_pool_fix_recheck"
mkdir -p "$RECHECK_DIR"

# Preflight: this cluster shows a transient NFS attribute-cache race right after a job
# starts on a fresh node -- a file that unquestionably exists (confirmed via `stat` from
# the headnode) gets a fast FileNotFoundError from the compute node for the first ~60-90s.
# Hit 3 times today across 2 different (otherwise healthy) nodes. Retry `ls` on the target
# path before launching python, so a transient miss doesn't waste a whole job attempt.
wait_for_file() {
    local path="$1"
    for i in $(seq 1 60); do
        if [ -f "$path" ]; then
            echo "Preflight: $path visible after $((i - 1)) retries."
            return 0
        fi
        sleep 5
    done
    echo "Preflight WARNING: $path still not visible after 300s -- proceeding anyway, will likely fail."
    return 1
}

for VARIANT in $VARIANTS; do
    echo ""
    echo "##### Variant: $VARIANT #####"

    CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}.yaml"
    EXP_NAME="CocoOnly$(python3 -c "print('${VARIANT}'.capitalize())")"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"
    CKPT_PATH="$MBEIR_DATA_DIR/genius_checkpoints_stage2/GENIUS_t5small/Large/Instruct/${EXP_NAME}/genius_t5small_epoch_25.pth"

    wait_for_file "$CKPT_PATH"

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$CONFIG" \
        --enable_instruct True

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29700 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "ERROR: eval for $VARIANT failed with exit code $STATUS"
        exit $STATUS
    fi

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$RECHECK_DIR/${VARIANT}.tsv"
        echo "Saved corrected results for $VARIANT -> $RECHECK_DIR/${VARIANT}.tsv"
    else
        echo "WARNING: no results tsv found for $VARIANT"
    fi
done

echo ""
echo "=== All 4 variant re-evals complete. ==="
