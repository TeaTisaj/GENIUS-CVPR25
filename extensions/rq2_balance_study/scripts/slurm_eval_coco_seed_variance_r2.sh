#!/bin/bash
#SBATCH --job-name=genius-eval-coco-seed-variance-r2
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_seed_variance_r2_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_seed_variance_r2_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=128G

# COCO seed-variance eval (2026-07-03): mirrors slurm_eval_fashioniq_seed_variance.sh. Evaluates
# the 8 (variant, seed) COCO Stage-2 seed-variant checkpoints (seeds 7, 13; the real study's
# seed=2023 result is already known from table2_retrieval_performance.csv and is NOT re-evaluated
# here), all at ckpt_name=genius_t5small_epoch_25.pth (COCO's best epoch is uniform across
# variants, unlike FashionIQ's per-variant split). Headline metric is T->I Recall@1 (matches
# Table 2/3's existing convention); I->T is evaluated too (base configs cover both tasks) but is
# near-zero for 3 of 4 variants and not the ranking-robustness metric being tested here.


# Round 2 (2026-07-10, ECIR audit B1 follow-up): evaluates the 4 new (variant, seed)
# checkpoints from slurm_train_coco_seed_variance_r2.sh (vanilla/strong x seed{21,42}).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

VARIANTS="vanilla strong"
SEEDS="21 42"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_seed_variance"
mkdir -p "$SWEEP_DIR"

for VARIANT in $VARIANTS; do
    EXP_NAME="CocoOnly$(python3 -c "print('${VARIANT}'.capitalize())")"
    for SEED in $SEEDS; do
        CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}_seed${SEED}.yaml"
        SEED_EXP_NAME="${EXP_NAME}Seed${SEED}"
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
            --master_port $((29950 + RANDOM % 1000)) \
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
echo "=== All 8 COCO seed-variance evals complete. ==="
