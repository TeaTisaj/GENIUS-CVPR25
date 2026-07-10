#!/bin/bash
#SBATCH --job-name=genius-leakage-check
#SBATCH --output=/home/tcetoje/logs/genius_leakage_check_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_leakage_check_%x_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
EXT_DIR="$GENIR_DIR/extensions/retrieval_aware_id_refinement"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"

cd "$EXT_DIR"
python _check_train_test_leakage.py
