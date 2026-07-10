#!/bin/bash
#SBATCH --job-name=genius-fiq-stage1-repair
#SBATCH --output=/home/tcetoje/logs/genius_fiq_stage1_repair_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_fiq_stage1_repair_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --exclude=ilps-cn117,ilps-cn118,ilps-cn120

# Repair job (2026-07-02): FashionIQ Stage-1 checkpoints for Vanilla/Weak/Medium were
# accidentally overwritten on 2026-07-01 20:21-20:48 by stray job resubmissions
# (333465-467) during an unrelated Stage-0 embeddings-path relocation. The already-reported
# Stage-2/eval results are unaffected (computed before the overwrite), but the on-disk
# quantizers no longer match what those Stage-2 checkpoints were trained against.
# This retrains the 3 affected variants with the exact same config/seed used originally,
# so the new checkpoints can be parity-checked (Step A diagnostics) against the archived
# table1_id_structure.csv before being trusted for the seed-variance follow-up.
# Strong is NOT retrained -- its original Stage-1 checkpoint survived the overwrite untouched.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/residual_quantization"
CONFIG_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
NPROC=4

VARIANTS="vanilla weak medium"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Preflight: this cluster shows a transient NFS attribute-cache race right after a job
# starts on a fresh node -- a file that unquestionably exists (confirmed via `stat` from
# the headnode) gets a fast FileNotFoundError from the compute node for the first ~60-90s.
# Hit repeatedly today (cn116 x2, cn120 x1) across otherwise-healthy nodes. Retry before
# launching python so a transient miss doesn't waste a whole job attempt.
wait_for_file() {
    local path="$1"
    for i in $(seq 1 60); do
        if [ -f "$path" ]; then
            echo "Preflight: $path visible after $((i - 1)) retries."
            return 0
        fi
        sleep 5
    done
    echo "Preflight WARNING: $path still not visible after 300s -- proceeding anyway, will likely fail."
    return 1
}

wait_for_file "$GENIR_DIR/extracted_embed/CLIP_SF/train_fashioniq/pool_SFpretrained_IT_dict.pt"
wait_for_file "$GENIR_DIR/extracted_embed/CLIP_SF/train_fashioniq/query_SFpretrained_instruction_IT_dict.pt"
wait_for_file "$GENIR_DIR/checkpoint/CLIP_SF/clip_sf_large.pth"

for VARIANT in $VARIANTS; do
    echo ""
    echo "##### Repairing Stage-1 FashionIQ: $VARIANT #####"

    CONFIG="$CONFIG_DIR/inbatch_fashioniq_${VARIANT}.yaml"

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$CONFIG" \
        --enable_instruct True

    cd "$MODEL_DIR"
    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29800 + RANDOM % 1000)) \
        train.py \
        --config_path "$CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "ERROR: Stage-1 repair training for $VARIANT failed with exit code $STATUS"
        exit $STATUS
    fi
    echo "##### Done: $VARIANT #####"
done

echo ""
echo "=== FashionIQ Stage-1 repair (vanilla/weak/medium) complete. ==="
