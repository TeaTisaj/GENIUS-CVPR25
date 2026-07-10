#!/bin/bash
#SBATCH --job-name=genius-rq1-coco-sub74k-diag
#SBATCH --output=/home/tcetoje/logs/genius_rq1_coco_sub74k_diag_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_rq1_coco_sub74k_diag_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Pool-size confound check (RQ3 robustness addition): re-run the same ID-property +
# reconstruction diagnostic as slurm_rq1_coco_variant_diagnostics.sh, but on a random
# 74,380-candidate subsample of COCO's pool (matching FashionIQ's exact pool size) to
# isolate how much of COCO's higher utilization/collision (vs FashionIQ) is a pool-size
# artifact rather than a genuine domain difference. No retraining -- reuses the existing
# 4 COCO Stage-1 checkpoints. Writes to a NEW --out_root, never touching the real
# rq_analysis_coco_variants/ results. No GPU needed -- runs on the cpu partition.

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
python rq1_coco_variant_diagnostics.py --genir_dir "$GENIR_DIR" \
    --dataset coco \
    --subsample_size 74380 \
    --subsample_seed 74 \
    --out_root /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_subsampled74k_variants
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: rq1_coco_variant_diagnostics.py failed with exit code $STATUS"
    exit $STATUS
fi

echo ""
echo "RQ1 COCO subsampled-74K pool-size confound diagnostics complete."
