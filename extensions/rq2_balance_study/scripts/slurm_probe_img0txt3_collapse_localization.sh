#!/bin/bash
#SBATCH --job-name=genius-probe-img0txt3-localize
#SBATCH --output=/home/tcetoje/logs/genius_probe_img0txt3_localize_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_probe_img0txt3_localize_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Same tokenizer-level dense-KNN probe already run on img3txt0/img3txt0p3
# (probe_rq_only_collapse_localization.py), pointed at img0txt3's Stage-1
# checkpoint so it gets the same corroboration standard.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node: $(hostname)"

cd "$ANALYSIS_DIR"
python probe_img0txt3_collapse_localization.py
