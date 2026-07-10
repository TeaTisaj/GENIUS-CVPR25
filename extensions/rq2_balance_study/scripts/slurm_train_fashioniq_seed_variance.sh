#!/bin/bash
#SBATCH --job-name=genius-fiq-stage2-seedvar
#SBATCH --output=/home/tcetoje/logs/genius_fiq_stage2_seedvar_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_fiq_stage2_seedvar_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --exclude=ilps-cn117,ilps-cn118

# Yubao's seed-variance ask (2026-07-02): FashionIQ IT->I R@1 is tiny (0.27-0.55%), so train
# 2 extra Stage-2 seeds (7, 13) per balance variant to see whether the vanilla-wins ranking
# survives seed noise. Only Stage-2/T5 varies by seed -- Stage-1 RQ (quantizer_path) is held
# fixed per variant, matching the seed=2023 runs already reported. DO NOT RUN this until the
# FashionIQ Stage-1 checkpoint-overwrite repair (slurm_repair_fashioniq_stage1.sh) has been
# parity-verified against the archived table1_id_structure.csv -- these configs' quantizer_path
# points straight at the checkpoints that were accidentally overwritten on 2026-07-01.
# 4 variants x 2 seeds = 8 sequential training runs.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
NPROC=4

VARIANTS="vanilla weak medium strong"
SEEDS="7 13"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

for VARIANT in $VARIANTS; do
    for SEED in $SEEDS; do
        echo ""
        echo "##### FashionIQ Stage-2 seed-variance: $VARIANT seed=$SEED #####"

        CONFIG="$CONFIG_DIR/inbatch_fashioniq_only_${VARIANT}_seed${SEED}.yaml"
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
echo "=== FashionIQ Stage-2 seed-variance training (8 runs) complete. ==="
