#!/bin/bash
#SBATCH --job-name=genius-hourglass-measure
#SBATCH --output=/home/tcetoje/logs/genius_hourglass_measure_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_hourglass_measure_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Related-work engagement (Fable's Gap-5): measure the Kuai et al. "hourglass"
# phenomenon (intermediate RQ levels concentrating relative to boundary
# levels) across the balance-regularization lambda sweep, both datasets.
# Analysis-only, no new training. sbatch --partition=cpu (see
# [[feedback_slurm_partitions]]).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
ANALYSIS_DIR="$GENIR_DIR/extensions/rq2_balance_study/analysis"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC/common:$SRC"

echo "Node: $(hostname)"

cd "$ANALYSIS_DIR"
python measure_hourglass_phenomenon.py \
    --genir_dir "$GENIR_DIR" \
    --out_csv "/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_hourglass/hourglass_per_level.csv"
