#!/bin/bash
#SBATCH --job-name=genius-coco-stage2-seedvar-r2
#SBATCH --output=/home/tcetoje/logs/genius_coco_stage2_seedvar_r2_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_coco_stage2_seedvar_r2_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=200G

# COCO seed-variance check (2026-07-03): mirrors slurm_train_fashioniq_seed_variance.sh. COCO's
# reported ranking (vanilla -> weak up -> medium down -> strong up +50.9%) rests on a single run
# per variant, same gap the FashionIQ seed check already found and fixed for that dataset. Trains
# 2 extra Stage-2 seeds (7, 13) per balance variant; Stage-1 RQ (quantizer_path) is held fixed per
# variant, matching the seed=2023 runs already reported. COCO's Stage-1 checkpoints were never
# overwritten (unlike FashionIQ's), so no repair/parity-check gate applies here.
# Empirical timing check (job 332905/332906/332873/332874, 2026-06-30): a single COCO Stage-2
# variant (30 epochs, 4 GPU) takes ~17-25 min, so 8 sequential runs ~= 3h -- 5h limit for margin.
# 4 variants x 2 seeds = 8 sequential training runs.


# Round 2 (2026-07-10, ECIR audit B1 follow-up): 2 extra seeds (21, 42) for vanilla/strong
# only, to strengthen the vanilla-vs-strong T->I comparison flagged as not statistically
# significant (Welch p~=0.2 on the existing 3-seed data) by a strict pre-submission audit.
# weak/medium are intentionally excluded -- B1 only concerns the vanilla/strong headline gap.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
NPROC=4

VARIANTS="vanilla strong"
SEEDS="21 42"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

for VARIANT in $VARIANTS; do
    for SEED in $SEEDS; do
        echo ""
        echo "##### COCO Stage-2 seed-variance: $VARIANT seed=$SEED #####"

        CONFIG="$CONFIG_DIR/inbatch_coco_only_${VARIANT}_seed${SEED}.yaml"
        if [ ! -f "$CONFIG" ]; then
            echo "ERROR: missing config $CONFIG"
            exit 1
        fi

        cd "$COMMON_DIR"
        python config_updater.py \
            --update_mbeir_yaml_instruct_status \
            --mbeir_yaml_file_path "$CONFIG" \
            --enable_instruct True

        cd "$MODEL_DIR"
        python -m torch.distributed.run \
            --nproc_per_node=$NPROC \
            --master_port $((29900 + RANDOM % 1000)) \
            train.py \
            --config_path "$CONFIG" \
            --genir_dir "$GENIR_DIR" \
            --mbeir_data_dir "$MBEIR_DATA_DIR"
        STATUS=$?

        if [ $STATUS -ne 0 ]; then
            echo "ERROR: Stage-2 seed-variance training for $VARIANT seed=$SEED failed with exit code $STATUS"
            exit $STATUS
        fi
        echo "##### Done: $VARIANT seed=$SEED #####"
    done
done

echo ""
echo "=== COCO Stage-2 seed-variance training (8 runs) complete. ==="
