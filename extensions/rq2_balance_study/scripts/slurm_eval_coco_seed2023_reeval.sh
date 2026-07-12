#!/bin/bash
#SBATCH --job-name=genius-eval-coco-seed2023-reeval
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_seed2023_reeval_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_seed2023_reeval_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=02:00:00
#SBATCH --mem=128G

# ECIR-paper Round 3, item 4 (2026-07-11): re-runs the seed=2023 COCO eval with the CURRENT
# eval configs/pipeline, to check whether the R@5/R@10 truncation anomaly documented in
# table2_retrieval_performance.csv (R@5/R@10 nearly flat over R@1, unlike seeds 7/13/21/42's
# normal ~2-2.5x growth -- see paper Section 8) was an eval-run-specific artifact or is
# checkpoint-intrinsic. Uses the EXISTING, UNMODIFIED base configs (no seed suffix) --
# these already point at the correct checkpoint (CocoOnly{Vanilla,Weak,Medium,Strong} at
# /fnwi_fs/.../genius_checkpoints_stage2/, ckpt_name=genius_t5small_epoch_25.pth, confirmed
# present) with no changes needed. Do NOT overwrite table2_retrieval_performance.csv; this
# writes to a separate directory for comparison first.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

VARIANTS="vanilla weak medium strong"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_seed2023_reeval"
mkdir -p "$SWEEP_DIR"

for VARIANT in $VARIANTS; do
    EXP_NAME="CocoOnly$(python3 -c "print('${VARIANT}'.capitalize())")"
    CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}.yaml"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

    echo ""
    echo "=== $VARIANT seed=2023 re-eval (exp_name=$EXP_NAME) ==="

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$CONFIG" \
        --enable_instruct True

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29990 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$SWEEP_DIR/${VARIANT}_seed2023_reeval.tsv"
        echo "Saved results for $VARIANT -> $SWEEP_DIR/${VARIANT}_seed2023_reeval.tsv"
    else
        echo "WARNING: no results tsv found for $VARIANT"
    fi
done

echo ""
echo "=== All 4 COCO seed=2023 re-evals complete. ==="
