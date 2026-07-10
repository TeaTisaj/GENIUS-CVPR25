#!/bin/bash
#SBATCH --job-name=extract-union-subset
#SBATCH --output=/home/tcetoje/logs/extract_union_subset_%j.log
#SBATCH --error=/home/tcetoje/logs/extract_union_subset_%j.log
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Filter the M-BEIR union train candidate pool into per-dataset local JSONLs for
# Fashion200K / NIGHTS / CIRR (GENIUS Phase 2 candidate-pool composition study).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"

export PATH="/home/tcetoje/miniconda3/bin:$PATH"
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

cd "$SRC/data/preprocessing"
python extract_union_subset_to_local_pool.py --mbeir_data_dir "$MBEIR_DATA_DIR"
