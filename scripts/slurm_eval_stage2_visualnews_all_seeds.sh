#!/bin/bash
#SBATCH --job-name=genius-visualnews-eval-allseeds
#SBATCH --output=/home/tcetoje/logs/genius_visualnews_eval_allseeds_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_visualnews_eval_allseeds_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# VisualNews eval, 2 variants x 3 seeds = 6 runs, mirroring
# slurm_eval_coco_seed_variance.sh's loop pattern. Each run evaluates both
# task0 (T->I) and task3 (I->T) via the trie+beam-search pipeline. Depends on
# slurm_train_stage2_visualnews_all_seeds.sh having completed -- submit with
# --dependency=afterok:<that job's ID>.
#
# PRE-FLIGHT REMINDER (this project has hit this bug class before -- see
# feedback_trie_cache_staleness.md): grep this log for "Save" (fresh trie)
# vs "Loaded" (stale, reused) before trusting any number below.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

CONFIGS=(
    "config_eval_visualnews_only_vanilla.yaml:VisualnewsOnlyVanilla"
    "config_eval_visualnews_only_vanilla_seed7.yaml:VisualnewsOnlyVanillaSeed7"
    "config_eval_visualnews_only_vanilla_seed13.yaml:VisualnewsOnlyVanillaSeed13"
    "config_eval_visualnews_only_strong.yaml:VisualnewsOnlyStrong"
    "config_eval_visualnews_only_strong_seed7.yaml:VisualnewsOnlyStrongSeed7"
    "config_eval_visualnews_only_strong_seed13.yaml:VisualnewsOnlyStrongSeed13"
)

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/visualnews_all_seeds"
mkdir -p "$SWEEP_DIR"

for ENTRY in "${CONFIGS[@]}"; do
    CONFIG_NAME="${ENTRY%%:*}"
    EXP_NAME="${ENTRY##*:}"
    CONFIG="$EVAL_DIR/$CONFIG_NAME"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

    echo ""
    echo "=== $EXP_NAME ==="

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
        cp "$LATEST_TSV" "$SWEEP_DIR/${EXP_NAME}.tsv"
        echo "Saved results for $EXP_NAME -> $SWEEP_DIR/${EXP_NAME}.tsv"
    else
        echo "WARNING: no results tsv found for $EXP_NAME"
    fi
done

echo ""
echo "=== All 6 VisualNews evals complete. ==="
