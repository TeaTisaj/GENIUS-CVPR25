#!/bin/bash
#SBATCH --job-name=genius-finalize-fashioniq
#SBATCH --output=/home/tcetoje/logs/genius_finalize_fashioniq_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_finalize_fashioniq_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# Step D + Step E: build FashionIQ's Table 1/2/3/correlations (Step D), then the
# cross-dataset Table 4 (Step E), once the real 4-variant Stage-2 + eval sweep are done.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "=== Step D: summarize FashionIQ variant study ==="
python "$GENIR_DIR/extensions/rq2_balance_study/scripts/summarize_coco_variant_study.py" \
    --dataset fashioniq --genir_dir "$GENIR_DIR"

echo ""
echo "=== Step E: build cross-dataset Table 4 ==="
python "$GENIR_DIR/extensions/rq2_balance_study/analysis/build_table4_cross_dataset.py" \
    --coco_dir /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants \
    --fashioniq_dir /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_fashioniq_variants \
    --out_dir /fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_cross_dataset

echo ""
echo "=== Finalize complete. ==="
