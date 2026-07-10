#!/bin/bash
#SBATCH --job-name=genius-extract-visualnews-splits
#SBATCH --output=/home/tcetoje/logs/genius_extract_visualnews_splits_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_extract_visualnews_splits_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# 3rd-dataset stand-up (ECIR full-paper extension): extract VisualNews-only
# local candidate pools + train/test query splits from the already-downloaded
# M-BEIR union files. Pure JSON filtering, no GPU needed -- sbatch
# --partition=cpu per [[feedback_slurm_partitions]], not run on the headnode
# since mbeir_union_up_train.jsonl is ~650MB.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

cd "$SRC/data/preprocessing"
python extract_visualnews_local_splits.py --mbeir_data_dir "$MBEIR_DATA_DIR" --no_overwrite_task0
