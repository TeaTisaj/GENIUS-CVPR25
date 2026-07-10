#!/bin/bash
#SBATCH --job-name=genius-tie-check
#SBATCH --output=/home/tcetoje/logs/genius_tie_check_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_tie_check_%x_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
EXT_DIR="$GENIR_DIR/extensions/retrieval_aware_id_refinement"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node: $(hostname)"

cd "$EXT_DIR"
python _check_tie_hypothesis.py
