#!/bin/bash
#SBATCH --job-name=genius-coco-stage2-diraware-seedvar
#SBATCH --output=/home/tcetoje/logs/genius_coco_stage2_diraware_seedvar_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_coco_stage2_diraware_seedvar_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=200G

# Direction-aware lambda seed-variance check (2026-07-10): mirrors
# slurm_train_coco_seed_variance.sh exactly, but for the two direction-aware balance
# variants (img3txt0: lambda_img=3.0/lambda_txt=0.0; img3txt0p3: lambda_img=3.0/lambda_txt=0.3)
# instead of the original vanilla/weak/medium/strong four. These two variants currently have
# only n=1 (seed=2023, already reported in rq1-rq3_final_tables.md: T->I 7.97%/5.45%), while
# every other headline number in the study has a 3-seed check -- this closes that one gap
# before the result goes into the ECIR paper draft. Stage-1 RQ (quantizer_path) is held fixed
# per variant, matching the seed=2023 runs already reported -- only Stage-2 (T5) seed varies,
# identical convention to the original seed-variance study.
# Timing: COCO Stage-2 (30 epochs, 4 GPU) took ~17-25 min per run in the original seed-variance
# study (job 332905/332906/332873/332874) -- 4 sequential runs here, 3h limit for margin.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
NPROC=4

VARIANTS="img3txt0 img3txt0p3"
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
        echo "##### COCO direction-aware Stage-2 seed-variance: $VARIANT seed=$SEED #####"

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
            echo "ERROR: Stage-2 direction-aware seed-variance training for $VARIANT seed=$SEED failed with exit code $STATUS"
            exit $STATUS
        fi
        echo "##### Done: $VARIANT seed=$SEED #####"
    done
done

echo ""
echo "=== All 4 COCO direction-aware Stage-2 seed-variance training runs complete. ==="
