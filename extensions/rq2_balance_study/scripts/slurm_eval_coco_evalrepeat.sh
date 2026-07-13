#!/bin/bash
#SBATCH --job-name=genius-eval-coco-evalrepeat
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_evalrepeat_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_evalrepeat_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=02:00:00
#SBATCH --mem=128G

# ECIR audit Round 5, Phase 0b (2026-07-13, blocking item 3): a second independent
# eval-only repeat on the EXISTING, unmodified seed=2023 checkpoints for vanilla/strong/
# img3txt0p3, to establish an eval-repeat variance floor. This is repeat #2 -- repeat #1
# already exists at retrieval_results/coco_seed2023_reeval/{vanilla,strong}_seed2023_reeval.tsv
# (from the 2026-07-11 seed=2023 correction work), so together with the ORIGINAL eval numbers
# already in table2_retrieval_performance.csv, this gives 3 independent eval points per
# checkpoint. Uses the EXISTING, UNMODIFIED base eval configs -- no training, no checkpoint
# changes. Writes to a separate directory, does not overwrite any existing result file.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

# variant_name:config_file_basename:exp_name
COMBOS="vanilla:config_eval_coco_only_vanilla:CocoOnlyVanilla strong:config_eval_coco_only_strong:CocoOnlyStrong img3txt0p3:config_eval_coco_only_img3txt0p3:CocoOnlyImg3txt0p3"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_eval_repeat_check"
mkdir -p "$SWEEP_DIR"

for COMBO in $COMBOS; do
    VARIANT=$(echo "$COMBO" | cut -d: -f1)
    CFG_BASE=$(echo "$COMBO" | cut -d: -f2)
    EXP_NAME=$(echo "$COMBO" | cut -d: -f3)

    CONFIG="$EVAL_DIR/${CFG_BASE}.yaml"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

    echo ""
    echo "=== $VARIANT eval-repeat #2 (exp_name=$EXP_NAME) ==="

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$CONFIG" \
        --enable_instruct True

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29850 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$SWEEP_DIR/${VARIANT}_evalrepeatB.tsv"
        echo "Saved results for $VARIANT -> $SWEEP_DIR/${VARIANT}_evalrepeatB.tsv"
    else
        echo "WARNING: no results tsv found for $VARIANT"
    fi
done

echo ""
echo "=== All 3 eval-repeat runs complete. ==="
