#!/bin/bash
#SBATCH --job-name=genius-verify-reimpl
#SBATCH --output=/home/tcetoje/logs/genius_verify_reimpl_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_verify_reimpl_%x_%j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Component 9, Step 6.2: independent re-implementation check. Runs the
# from-scratch numpy probe (verify_independent_reimpl.py) on the teacher +
# one method + one control checkpoint, to confirm the numbers reported by
# check_feasibility_criteria.py's dense_recall_at_k_probe (job 334749) are
# not an artifact of a bug replicated by construction (since a from-scratch
# reimplementation can't inherit the same mistake).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
EXT_DIR="$GENIR_DIR/extensions/retrieval_aware_id_refinement"
BASE="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_retrieval_aware_feasibility/rq_clip_large/Large/Instruct"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"

echo "Node: $(hostname)"

cd "$EXT_DIR"

python verify_independent_reimpl.py --label Teacher_CocoVanilla \
    --ckpt "$GENIR_DIR/checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth"

python verify_independent_reimpl.py --label Method_Frozen_lambda3.0 \
    --ckpt "$BASE/FeasibilityVxVFreezeCodebook/rq_clip_large_epoch_10.pth"

python verify_independent_reimpl.py --label Control_Frozen_lambda3.0_NoAMP \
    --ckpt "$BASE/FeasibilityControlNoNewLossesFreezeCodebookNoAMP/rq_clip_large_epoch_10.pth"

echo "ALL DONE"
