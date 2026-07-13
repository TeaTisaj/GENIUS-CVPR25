#!/bin/bash
#SBATCH --job-name=genius-eval-prepatch-decode-diag
#SBATCH --output=/home/tcetoje/logs/genius_eval_prepatch_decode_diag_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_prepatch_decode_diag_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# ECIR audit Round 5, Phase 0c-2 (2026-07-13, blocking item 2): the paper's Section 8 claims
# the img3txt0/img3txt0p3 constrained-decoding dead-end crash was "structural" (12/12 failures
# across 3 retrain rounds) and was "fixed" by commit b56eeaf's fallback patch to
# prefix_allowed_tokens_fn -- but the patched reruns never actually triggered the fallback
# (0/4), which is suspicious for a claimed-deterministic bug on byte-identical checkpoints.
# This reruns the PRE-PATCH retriever.py (via PYTHONPATH pointing at a symlink shadow-src
# tree with only retriever.py swapped to git rev b56eeaf~1's version -- see
# /tmp/.../scratchpad/prepatch_diag) against the SAME already-trained epoch_25.pth
# checkpoints, 2 independent reps per (variant, seed), to see whether the crash recurs
# every time (supports "structural") or only sometimes (supports the fp16/cudnn
# nondeterminism hypothesis -- see git diff notes and log analysis in
# project_ecir_paper_strict_audit.md Round 5). Uses distinct *_prepatchdiag exp_names with
# a hardcoded (not template-derived) ckpt_dir pointing at the ORIGINAL checkpoints, so this
# cannot corrupt or overwrite the already-reported 3-seed results.
#
# Does NOT modify the main repo's retriever.py at any point -- only PYTHONPATH changes.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
# Must be on the shared NFS home mount, NOT /tmp -- /tmp is local to whichever machine
# submits the job and is invisible to the compute node the job actually lands on (this
# broke the first attempt of this diagnostic, job 335471: FileNotFoundError on
# src/common/mbeir_generative_retriever.py, 19s crash, before ever reaching decode).
SHADOW_SRC="/home/tcetoje/prepatch_diag_shadow_src/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SHADOW_SRC/common"
EVAL_DIR="$GENIR_DIR/src/models/generative_retriever/configs_scripts/large/eval/inbatch"
NPROC=4
REPS="A B"

CONFIGS="img3txt0_seed7 img3txt0_seed13 img3txt0p3_seed7 img3txt0p3_seed13"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SHADOW_SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SUMMARY="/home/tcetoje/logs/prepatch_decode_diag_summary.txt"
: > "$SUMMARY"

for CFG in $CONFIGS; do
    for REP in $REPS; do
        CONFIG="$EVAL_DIR/config_eval_coco_only_${CFG}_prepatchdiag.yaml"

        echo ""
        echo "##### PRE-PATCH decode diagnostic: $CFG rep=$REP #####"

        cd "$COMMON_DIR"
        python config_updater.py \
            --update_mbeir_yaml_instruct_status \
            --mbeir_yaml_file_path "$CONFIG" \
            --enable_instruct True

        python -m torch.distributed.run \
            --nproc_per_node=$NPROC \
            --master_port $((29700 + RANDOM % 1000)) \
            "$COMMON_DIR/mbeir_generative_retriever.py" \
            --config_path "$CONFIG" \
            --genir_dir "$GENIR_DIR" \
            --mbeir_data_dir "$MBEIR_DATA_DIR"
        STATUS=$?

        if [ $STATUS -eq 0 ]; then
            echo "$CFG rep=$REP: SUCCESS (exit 0)" | tee -a "$SUMMARY"
        else
            echo "$CFG rep=$REP: CRASHED (exit $STATUS)" | tee -a "$SUMMARY"
        fi
    done
done

echo ""
echo "=== Pre-patch decode diagnostic complete. Summary: ==="
cat "$SUMMARY"
