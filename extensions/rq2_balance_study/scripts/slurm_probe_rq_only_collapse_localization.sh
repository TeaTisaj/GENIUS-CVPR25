#!/bin/bash
#SBATCH --job-name=genius-probe-collapse-localize
#SBATCH --output=/home/tcetoje/logs/genius_probe_collapse_localize_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_probe_collapse_localize_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Mechanism forensics step 1 (Fable's ladder): localize the I->T collapse to
# tokenizer vs. decoder by running the already-verified RQ-only dense-KNN
# probe (no T5, no beam search) on the four balance-regularization Stage-1
# checkpoints. sbatch --partition=cpu, never bare nohup on the headnode
# (see [[feedback_slurm_partitions]]).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node: $(hostname)"

cd "$ANALYSIS_DIR"
python probe_rq_only_collapse_localization.py
