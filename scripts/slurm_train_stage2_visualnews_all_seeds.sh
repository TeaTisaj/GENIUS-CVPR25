#!/bin/bash
#SBATCH --job-name=genius-visualnews-stage2-allseeds
#SBATCH --output=/home/tcetoje/logs/genius_visualnews_stage2_allseeds_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_visualnews_stage2_allseeds_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=06:00:00
#SBATCH --mem=200G

# VisualNews Stage-2 (T5 generative retriever), 2 variants (vanilla/strong)
# x 3 seeds (2023/7/13) = 6 sequential runs, mirroring
# slurm_train_coco_seed_variance.sh's loop pattern exactly. Depends on both
# VisualNews Stage-1 jobs (335039 vanilla, 335040 strong) having completed --
# submit this with --dependency=afterok:335039:335040.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
NPROC=4

CONFIGS=(
    "inbatch_visualnews_only_vanilla.yaml"
    "inbatch_visualnews_only_vanilla_seed7.yaml"
    "inbatch_visualnews_only_vanilla_seed13.yaml"
    "inbatch_visualnews_only_strong.yaml"
    "inbatch_visualnews_only_strong_seed7.yaml"
    "inbatch_visualnews_only_strong_seed13.yaml"
)

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

for CONFIG_NAME in "${CONFIGS[@]}"; do
    echo ""
    echo "##### VisualNews Stage-2: $CONFIG_NAME #####"

    CONFIG="$CONFIG_DIR/$CONFIG_NAME"
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
        --master_port $((29800 + RANDOM % 1000)) \
        train.py \
        --config_path "$CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "ERROR: Stage-2 training for $CONFIG_NAME failed with exit code $STATUS"
        exit $STATUS
    fi
    echo "##### Done: $CONFIG_NAME #####"
done

echo ""
echo "=== All 6 VisualNews Stage-2 training runs complete. ==="
