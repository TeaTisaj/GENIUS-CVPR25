#!/bin/bash
#SBATCH --job-name=rq-analysis
#SBATCH --output=/home/tcetoje/logs/rq_analysis_%j.log
#SBATCH --error=/home/tcetoje/logs/rq_analysis_%j.log
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# RQ semantic-ID diagnostic analysis (Yubao's Extension 2 task, see
# rq_code_analysis.md). All four analyses read precomputed RQ codes already
# produced by Phase 1/2 eval jobs against the frozen official quantizer -- no
# model loading needed, so this runs entirely on the cpu partition.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
OUT_ROOT="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis"

export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

cd "$SRC/common"

echo "=== Analysis 1: codebook utilization ==="
python rq_analysis_codebook_utilization.py --genir_dir "$GENIR_DIR" --out_root "$OUT_ROOT"

echo "=== Analysis 2: prefix collision (extended phase2_prefix_overlap.py) ==="
python phase2_prefix_overlap.py --genir_dir "$GENIR_DIR" --rq_analysis_out_root "$OUT_ROOT"

echo "=== Analysis 4: dataset dependence ==="
python rq_analysis_dataset_dependence.py --genir_dir "$GENIR_DIR" --out_root "$OUT_ROOT"

echo "=== Analysis 3: cross-modal alignment ==="
python rq_analysis_cross_modal.py --genir_dir "$GENIR_DIR" --mbeir_data_dir "$MBEIR_DATA_DIR" --out_root "$OUT_ROOT"

echo "All RQ analysis outputs written under $OUT_ROOT"
