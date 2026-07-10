#!/bin/bash
#SBATCH --job-name=genius-finalize-coco
#SBATCH --output=/home/tcetoje/logs/genius_finalize_coco_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_finalize_coco_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# Re-run COCO Step D + rebuild cross-dataset Table 4 after the corrected epoch sweep
# (job 333474, correct 5K/24.8K test splits) completes.
# Submit with: sbatch --dependency=afterok:333474 slurm_finalize_coco.sh

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "=== Step D: summarize COCO variant study (corrected data) ==="
python "$GENIR_DIR/extensions/rq2_balance_study/scripts/summarize_coco_variant_study.py" \
    --dataset coco --genir_dir "$GENIR_DIR"

echo ""
echo "=== Step E: rebuild cross-dataset Table 4 with corrected COCO numbers ==="
python "$GENIR_DIR/extensions/rq2_balance_study/analysis/build_table4_cross_dataset.py" \
    --coco_dir /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants \
    --fashioniq_dir /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_fashioniq_variants \
    --out_dir /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_cross_dataset

echo ""
echo "=== Finalize complete. ==="
