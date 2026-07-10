#!/bin/bash
#SBATCH --job-name=genius-eval-fashioniq-calib-sweep
#SBATCH --output=/home/tcetoje/logs/genius_eval_fashioniq_calib_sweep_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_fashioniq_calib_sweep_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=03:00:00
#SBATCH --mem=128G

# Calibration sweep per Yubao's guidance (2026-07-01): find where FashionIQ vanilla
# Recall stabilizes before committing weak/medium/strong to a fixed epoch budget.
# Dense checkpoints every 5 epochs up to 60 (vs the full variant sweep's every-20-up-to-200).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

EPOCHS="5 10 15 20 25 30 35 40 45 50 55 60"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_CONFIG="$EVAL_DIR/config_eval_fashioniq_calib_vanilla.yaml"
SWEEP_CONFIG="$EVAL_DIR/config_eval_fashioniq_calib_vanilla_sweep_tmp.yaml"
EXP_NAME="FashioniqCalibVanilla"
RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"
CKPT_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage2/GENIUS_t5small/Large/Instruct/${EXP_NAME}"
SWEEP_DIR="$GENIR_DIR/retrieval_results/fashioniq_calib_sweep_vanilla"
mkdir -p "$SWEEP_DIR"

FIRST=1
for EPOCH in $EPOCHS; do
    CKPT_NAME="genius_t5small_epoch_${EPOCH}.pth"
    if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
        echo "Skipping calib_vanilla epoch $EPOCH: checkpoint not found at $CKPT_DIR/$CKPT_NAME"
        continue
    fi

    echo ""
    echo "=== calib_vanilla epoch $EPOCH ($CKPT_NAME) ==="

    GEN_CAND_FLAG=""
    if [ "$FIRST" -eq 1 ]; then
        GEN_CAND_FLAG="--gen_cand_codes"
        FIRST=0
    fi

    python "$GENIR_DIR/scripts/make_sweep_eval_config.py" \
        --base_config "$BASE_CONFIG" \
        --out_config "$SWEEP_CONFIG" \
        --ckpt_name "$CKPT_NAME" \
        $GEN_CAND_FLAG

    cd "$COMMON_DIR"
    python config_updater.py \
        --update_mbeir_yaml_instruct_status \
        --mbeir_yaml_file_path "$SWEEP_CONFIG" \
        --enable_instruct True

    python -m torch.distributed.run \
        --nproc_per_node=$NPROC \
        --master_port $((29800 + RANDOM % 1000)) \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$SWEEP_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$SWEEP_DIR/epoch_${EPOCH}.tsv"
        echo "Saved results for calib_vanilla epoch $EPOCH -> $SWEEP_DIR/epoch_${EPOCH}.tsv"
    else
        echo "WARNING: no results tsv found for calib_vanilla epoch $EPOCH"
    fi
done

echo "=== Calibration sweep complete. Building summary ==="
python "$GENIR_DIR/scripts/summarize_fashioniq_sweep.py" --sweep_dir "$SWEEP_DIR"
