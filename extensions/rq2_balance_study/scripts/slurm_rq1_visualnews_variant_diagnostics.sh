#!/bin/bash
#SBATCH --job-name=genius-rq1-visualnews-diag
#SBATCH --output=/home/tcetoje/logs/genius_rq1_visualnews_diag_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_rq1_visualnews_diag_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# ECIR paper Section 7 / RQ1 (2026-07-13, high-impact audit item 6): VisualNews is the
# dataset where the T->I effect reverses sign, yet RQ1's ID-structure table (Section 4)
# only ever covered COCO/FashionIQ. Reuses rq1_coco_variant_diagnostics.py unchanged,
# pointed at VisualNews's vanilla/strong Stage-1 checkpoints via the new "visualnews"
# entry in DATASET_CONFIGS. No GPU needed.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC/common:$SRC"

echo "Node: $(hostname)"

cd "$ANALYSIS_DIR"
python rq1_coco_variant_diagnostics.py --dataset visualnews --genir_dir "$GENIR_DIR"

echo ""
echo "RQ1 VisualNews variant diagnostics complete."
