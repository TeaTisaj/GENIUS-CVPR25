#!/bin/bash
#SBATCH --job-name=genius-coco-cand-codes
#SBATCH --output=/home/tcetoje/logs/genius_coco_cand_codes_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_coco_cand_codes_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# ECIR audit Round 5, item 7 (tie-break sensitivity): compute candidate_id -> full 8-level
# semantic code for MSCOCO task0 candidates under vanilla/strong Stage-1 checkpoints.
# CPU-only, no training, small inference pass -- see feedback_slurm_partitions.md.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

cd "$GENIR_DIR/extensions/rq2_balance_study/analysis"
python compute_coco_candidate_codes.py --genir_dir "$GENIR_DIR"

echo "=== Done ==="
