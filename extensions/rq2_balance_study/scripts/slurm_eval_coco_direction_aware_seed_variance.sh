#!/bin/bash
#SBATCH --job-name=genius-eval-coco-diraware-seedvar
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_diraware_seedvar_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_diraware_seedvar_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=02:00:00
#SBATCH --mem=128G

# Direction-aware lambda seed-variance eval (2026-07-10): mirrors
# slurm_eval_coco_seed_variance.sh. Evaluates the 4 (variant, seed) checkpoints trained by
# slurm_train_coco_direction_aware_seed_variance.sh (seeds 7, 13; seed=2023 is already known
# from retrieval_results/coco_epoch_sweep_{img3txt0,img3txt0p3}/coco_recall_by_epoch.csv,
# epoch 25 row, and is NOT re-evaluated here). Headline metric is T->I Recall@1 (matches the
# rest of the study's convention); I->T is evaluated too (base configs cover both tasks).
#
# PRE-FLIGHT REMINDER (this project has hit this bug class twice -- see
# feedback_trie_cache_staleness.md): after this runs, grep the log for "Save" (fresh trie)
# vs "Loaded" (stale, reused) before trusting any number below, and confirm the eval config's
# cand_pool_dir_name/test_dir_name point at the 5K/24.8K test splits, not the 713K train pool
# (they should already, since these configs were copied from the already-verified
# config_eval_coco_only_img3txt0{,p3}.yaml -- this is a sanity check, not an expected fix).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

VARIANTS="img3txt0 img3txt0p3"
SEEDS="7 13"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_direction_aware_seed_variance"
mkdir -p "$SWEEP_DIR"

for VARIANT in $VARIANTS; do
    for SEED in $SEEDS; do
        SEED_EXP_NAME="CocoOnly$(python3 -c "print('${VARIANT}'.capitalize())")Seed${SEED}"
        CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}_seed${SEED}.yaml"
        RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${SEED_EXP_NAME}/final_tsv"

        echo ""
        echo "=== $VARIANT seed=$SEED (exp_name=$SEED_EXP_NAME) ==="

        cd "$COMMON_DIR"
        python config_updater.py \
            --update_mbeir_yaml_instruct_status \
            --mbeir_yaml_file_path "$CONFIG" \
            --enable_instruct True

        python -m torch.distributed.run \
            --nproc_per_node=$NPROC \
            --master_port $((29960 + RANDOM % 1000)) \
            "$COMMON_DIR/mbeir_generative_retriever.py" \
            --config_path "$CONFIG" \
            --genir_dir "$GENIR_DIR" \
            --mbeir_data_dir "$MBEIR_DATA_DIR"

        LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
        if [ -n "$LATEST_TSV" ]; then
            cp "$LATEST_TSV" "$SWEEP_DIR/${VARIANT}_seed${SEED}.tsv"
            echo "Saved results for $VARIANT seed=$SEED -> $SWEEP_DIR/${VARIANT}_seed${SEED}.tsv"
        else
            echo "WARNING: no results tsv found for $VARIANT seed=$SEED"
        fi
    done
done

echo ""
echo "=== All 4 COCO direction-aware seed-variance evals complete. ==="
