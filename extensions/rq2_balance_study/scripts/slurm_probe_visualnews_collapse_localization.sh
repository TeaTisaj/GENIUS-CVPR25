#!/bin/bash
#SBATCH --job-name=genius-probe-collapse-localize-visualnews
#SBATCH --output=/home/tcetoje/logs/genius_probe_collapse_localize_visualnews_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_probe_collapse_localize_visualnews_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# ECIR-paper Round 3, item 3: same decoder-free dense-KNN probe used for MSCOCO
# (Fable's ladder, Component 9), now on VisualNews vanilla/strong Stage-1
# checkpoints. sbatch --partition=cpu, never bare nohup on the headnode (see
# [[feedback_slurm_partitions]]). Depends on job 335180 (VisualNews test-query
# embedding extraction) having completed first.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node: $(hostname)"

cd "$ANALYSIS_DIR"
python probe_visualnews_collapse_localization.py
