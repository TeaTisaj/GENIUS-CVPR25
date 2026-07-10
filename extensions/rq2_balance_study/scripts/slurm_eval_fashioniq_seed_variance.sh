#!/bin/bash
#SBATCH --job-name=genius-eval-fashioniq-seed-variance
#SBATCH --output=/home/tcetoje/logs/genius_eval_fashioniq_seed_variance_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_fashioniq_seed_variance_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=128G

# Seed-variance check (RQ3 robustness addition): evaluates the 8 (variant, seed) FashionIQ
# Stage-2 seed-variant checkpoints (seeds 7, 13; the real study's seed=2023 result is
# already known from table2_retrieval_performance.csv and is NOT re-evaluated here), each
# at that variant's already-known best epoch (vanilla:15, weak/medium/strong:10). Mirrors
# slurm_eval_fashioniq_variant_sweep.sh's structure. Every one of these 8 evals is a
# first-eval-per-exp_name (brand-new checkpoint dir, no cached candidate codes), so all 8
# eval configs were generated with --gen_cand_codes already set -- no need to special-case
# "first iteration" the way the epoch sweep does.
#
# GATED: do not submit this (or the 8 training jobs it depends on) until Yubao has
# responded to the question already sent about this addition.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

VARIANTS="vanilla weak medium strong"
SEEDS="7 13"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/fashioniq_seed_variance"
mkdir -p "$SWEEP_DIR"

for VARIANT in $VARIANTS; do
    EXP_NAME="FashioniqOnly$(python3 -c "print('${VARIANT}'.capitalize())")"
    for SEED in $SEEDS; do
        CONFIG="$EVAL_DIR/config_eval_fashioniq_only_${VARIANT}_seed${SEED}.yaml"
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
            --master_port $((29900 + RANDOM % 1000)) \
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
echo "=== All 8 seed-variance evals complete. ==="
