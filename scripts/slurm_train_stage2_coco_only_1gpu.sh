#!/bin/bash
#SBATCH --job-name=genius-stage2-coco-1gpu
#SBATCH --output=/home/tcetoje/logs/genius_stage2_coco_1gpu_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_stage2_coco_1gpu_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --exclude=ilps-cn120

# Diagnostic: COCO-only Stage 2, SAME nominal recipe (30 epochs, lr=1e-4, batch=256/GPU)
# but on a SINGLE GPU instead of 4. Goal: test whether "30 epochs" on 1 GPU yields
# ~4x more optimizer steps (len(train_loader) ~833/epoch x 30 ~= 25k steps vs ~6.2k
# on 4 GPUs) and a much lower converged loss (~2.0 like the 200-epoch run reached
# around its own ~25k-step mark), which would explain the Phase B paradox.
# Checkpoint dir: checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly1GPU/

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
CONFIG_PATH="$MODEL_DIR/configs_scripts/large/train/inbatch/inbatch_coco_only_1gpu.yaml"
NPROC=1

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"
echo "MBEIR data: $MBEIR_DATA_DIR"
echo "Job ID:     $SLURM_JOB_ID"

# Verify Stage 0 outputs exist
POOL_PT="$GENIR_DIR/extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt"
QUERY_PT="$GENIR_DIR/extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt"
if [ ! -f "$POOL_PT" ] || [ ! -f "$QUERY_PT" ]; then
    echo "ERROR: Stage 0 embeddings not found."
    exit 1
fi

# Verify COCO-only train JSONL exists
TRAIN_JSONL="$MBEIR_DATA_DIR/query/union_train/mbeir_coco_only_train.jsonl"
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "ERROR: COCO-only train JSONL not found: $TRAIN_JSONL"
    exit 1
fi
echo "COCO-only train queries: $(wc -l < $TRAIN_JSONL)"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --master_port 1327 \
    --nproc_per_node=$NPROC \
    "$MODEL_DIR/train.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Training complete."
echo "Checkpoints: $GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly1GPU/"
ls "$GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly1GPU/" 2>/dev/null | sort
