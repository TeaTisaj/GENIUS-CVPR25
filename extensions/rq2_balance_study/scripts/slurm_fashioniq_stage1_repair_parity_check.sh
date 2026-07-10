#!/bin/bash
#SBATCH --job-name=genius-fiq-repair-parity
#SBATCH --output=/home/tcetoje/logs/genius_fiq_repair_parity_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_fiq_repair_parity_%j.log
#SBATCH --partition=cpu
#SBATCH --nodelist=ilps-cn002
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Runs Step A diagnostics on the REPAIRED FashionIQ Stage-1 checkpoints (after
# slurm_repair_fashioniq_stage1.sh retrains vanilla/weak/medium), writing to a SEPARATE
# out_root so the archived pre-overwrite table1_id_structure.csv
# (rq_analysis_fashioniq_variants/tables/table1_id_structure_ARCHIVED_pre_overwrite_20260701.csv)
# is never touched. Submit with: sbatch --dependency=afterok:<repair_job_id> <this script>

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC/common:$SRC"

echo "Node: $(hostname)"

cd "$ANALYSIS_DIR"
python rq1_coco_variant_diagnostics.py --genir_dir "$GENIR_DIR" --dataset fashioniq \
    --out_root /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_fashioniq_variants_repair_check

echo ""
echo "=== Parity check diagnostics complete. Compare against the ARCHIVED pre-overwrite table: ==="
echo "New:      /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_fashioniq_variants_repair_check/tables/table1_id_structure.csv"
echo "Archived: /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_fashioniq_variants/tables/table1_id_structure_ARCHIVED_pre_overwrite_20260701.csv"
