#!/bin/bash
#SBATCH --job-name=genius-mine-hardneg
#SBATCH --output=/home/tcetoje/logs/genius_mine_hardneg_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_mine_hardneg_%x_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# Component 1 (retrieval-aware RQ refinement feasibility study): offline
# hard-negative mining Slurm wrapper. Uses 1 GPU for the teacher
# RQ.inference() forward passes over the COCO train candidate pool + the 20K
# feasibility query subsample -- this is a short, forward-only job (per the
# plan's "CPU-or-1-GPU, short"); the gpu partition/1-GPU request here is for
# speed, not a hard requirement -- --mode random needs no GPU/teacher at all
# and could run on --partition=cpu instead if the gpu queue is busy.
#
# Usage:
#   sbatch --job-name=genius-mine-vanilla-teacher \
#       scripts/slurm_mine_hard_negatives.sh \
#       hard \
#       checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth \
#       query/union_train/mbeir_coco_only_train_feasibility20k.jsonl \
#       extensions/retrieval_aware_id_refinement/hard_negatives_coco_train_vanilla_teacher.pt \
#       vanilla_teacher_train
#
#   sbatch --job-name=genius-mine-random \
#       scripts/slurm_mine_hard_negatives.sh \
#       random \
#       "" \
#       query/union_train/mbeir_coco_only_train_feasibility20k.jsonl \
#       extensions/retrieval_aware_id_refinement/hard_negatives_coco_train_random.pt \
#       random_control_train

MODE="$1"               # "hard" or "random"
TEACHER_CKPT="$2"       # relative to genir_dir, or absolute; ignored for random mode
QUERY_JSONL_PATH="$3"   # relative to mbeir_data_dir
OUT_PATH="$4"           # relative to genir_dir
CELL_NAME="$5"

if [ -z "$MODE" ] || [ -z "$QUERY_JSONL_PATH" ] || [ -z "$OUT_PATH" ] || [ -z "$CELL_NAME" ]; then
    echo "ERROR: usage: sbatch slurm_mine_hard_negatives.sh <hard|random> <teacher_ckpt_or_empty> <query_jsonl_path> <out_path> <cell_name>"
    exit 1
fi

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
EXT_DIR="$GENIR_DIR/extensions/retrieval_aware_id_refinement"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node:        $(hostname)"
echo "Mode:        $MODE"
echo "Teacher:     $TEACHER_CKPT"
echo "Query JSONL: $QUERY_JSONL_PATH"
echo "Out path:    $OUT_PATH"
echo "Cell name:   $CELL_NAME"

cd "$EXT_DIR"
if [ "$MODE" == "hard" ]; then
    python mine_hard_negatives.py \
        --mode hard \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --teacher_ckpt "$TEACHER_CKPT" \
        --query_jsonl_path "$QUERY_JSONL_PATH" \
        --out_path "$OUT_PATH" \
        --cell_name "$CELL_NAME" \
        --k 20
else
    python mine_hard_negatives.py \
        --mode random \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR" \
        --query_jsonl_path "$QUERY_JSONL_PATH" \
        --out_path "$OUT_PATH" \
        --cell_name "$CELL_NAME" \
        --k 20
fi

echo ""
echo "Hard-negative mining for '$CELL_NAME' complete."
