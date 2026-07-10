#!/bin/bash
#SBATCH --job-name=genius-eval-coco-variant-recheck
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_variant_recheck_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_variant_recheck_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --exclude=ilps-cn116,ilps-cn117,ilps-cn118

# Corrected re-eval of the 4 RQ1-3 COCO variants (Table 2) after fixing the
# COCO test-split data bugs: (1) cand_pool/local/mbeir_mscoco_task{0,3}_test
# symlinks were pointing at the full 713K non-split pool instead of the
# official 5K-image/24.8K-text split; (2) query/qrels files for task0/task3
# were scrambled between the two tasks. Both fixed in-place (data dir, no repo
# code changes); qrels regenerated from the corrected query files. This script
# only re-runs eval at the already-chosen epoch_25 checkpoint per variant --
# no retraining, no epoch sweep (that already served its purpose).

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

RECHECK_DIR="$GENIR_DIR/retrieval_results/coco_pool_fix_recheck"
mkdir -p "$RECHECK_DIR"

for VARIANT in $VARIANTS; do
    echo ""
    echo "##### Variant: $VARIANT #####"

    CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}.yaml"
    EXP_NAME="CocoOnly$(python3 -c "print('${VARIANT}'.capitalize())")"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

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
