#!/bin/bash
#SBATCH --job-name=genius-anisotropy-test
#SBATCH --output=/home/tcetoje/logs/genius_anisotropy_test_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_anisotropy_test_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Mechanism forensics (Fable's ladder item 3): text-embedding anisotropy
# hypothesis test, COCO only (where the I->T collapse is documented).
# Analysis-only, no new training. sbatch --partition=cpu.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC/common:$SRC"

echo "Node: $(hostname)"

cd "$ANALYSIS_DIR"
python test_anisotropy_hypothesis.py --genir_dir "$GENIR_DIR"
