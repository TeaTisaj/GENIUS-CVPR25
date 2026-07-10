#!/bin/bash
#SBATCH --job-name=genius-rq1-coco-diag
#SBATCH --output=/home/tcetoje/logs/genius_rq1_coco_diag_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_rq1_coco_diag_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Step A of the RQ1+RQ2 COCO-completion plan: ID-property + reconstruction
# diagnostic for the 4 balance-regularization Stage-1 quantizers. No GPU
# needed -- runs on the cpu partition.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC/common:$SRC"

echo "Node: $(hostname)"
which python
python -c "import torch; print('torch', torch.__version__)"

cd "$ANALYSIS_DIR"
python rq1_coco_variant_diagnostics.py --genir_dir "$GENIR_DIR"

echo ""
echo "RQ1 COCO variant diagnostics complete."
