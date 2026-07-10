#!/bin/bash
#SBATCH --job-name=genius-gate-img3txt
#SBATCH --output=/home/tcetoje/logs/genius_gate_img3txt_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_gate_img3txt_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Part 3 step 2 of the direction-aware-lambda plan: codebook-health gate on the
# new CocoImg3txt0/CocoImg3txt0p3 Stage-1 checkpoints before committing Stage-2
# compute. No GPU needed -- runs on the cpu partition.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
EXT_DIR="$GENIR_DIR/extensions/rq2_balance_study"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC/common:$SRC"

echo "Node: $(hostname)"
which python
python -c "import torch; print('torch', torch.__version__)"

cd "$EXT_DIR"
python gate_check_img3txt_codebook_health.py --genir_dir "$GENIR_DIR"

echo ""
echo "Codebook-health gate complete."
