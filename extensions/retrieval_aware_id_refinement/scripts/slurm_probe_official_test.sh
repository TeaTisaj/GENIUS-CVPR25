#!/bin/bash
#SBATCH --job-name=genius-probe-official-test
#SBATCH --output=/home/tcetoje/logs/genius_probe_official_test_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_probe_official_test_%x_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Component 9, Step 5: runs probe_dense_recall_official_test.py (the
# corrected, modality-matched + official-test-split Criterion-5 probe) via
# sbatch --partition=cpu, never bare nohup on the headnode (see
# [[feedback_slurm_partitions]]). Pool sizes are now 5K/24.8K instead of the
# old 615K mixed pool, so this should run much faster per-checkpoint than
# prior full-pool jobs.
#
# Usage:
#   sbatch scripts/slurm_probe_official_test.sh                  # all checkpoints
#   sbatch scripts/slurm_probe_official_test.sh --only Teacher_CocoVanilla

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
EXT_DIR="$GENIR_DIR/extensions/retrieval_aware_id_refinement"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node: $(hostname)"
echo "Args: $@"

cd "$EXT_DIR"
python probe_dense_recall_official_test.py "$@"
