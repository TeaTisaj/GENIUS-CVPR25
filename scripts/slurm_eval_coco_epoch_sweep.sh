#!/bin/bash
#SBATCH --job-name=genius-eval-coco-sweep
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_sweep_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_sweep_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=08:00:00
#SBATCH --mem=128G

# Evaluate COCO T->I/I->T (no rerank) for every saved epoch checkpoint of the
# full-union Stage 2 run (checkpoint/GENIUS_t5small/Large/Instruct/InBatch/),
# so we can see which epoch actually gives the best COCO recall instead of
# just trusting the last saved checkpoint (epoch_95). Candidate-pool codes
# are reused from cache (frozen RQ quantizer -> same codes for every epoch).

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
BASE_CONFIG="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_coco.yaml"
CKPT_DIR="$GENIR_DIR/checkpoint/GENIUS_t5small/Large/Instruct/InBatch"
RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/InBatch/final_tsv"
SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_epoch_sweep"
SWEEP_CONFIG="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_coco_sweep_tmp.yaml"
NPROC=4

mkdir -p "$SWEEP_DIR"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

EPOCHS="5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95"
FIRST=1

for EPOCH in $EPOCHS; do
    CKPT_NAME="genius_t5small_epoch_${EPOCH}.pth"
    if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
        echo "Skipping epoch $EPOCH: checkpoint not found at $CKPT_DIR/$CKPT_NAME"
        continue
    fi

    echo ""
    echo "=== Epoch $EPOCH ($CKPT_NAME) ==="

    GEN_CAND_FLAG=""
    if [ "$FIRST" -eq 1 ]; then
        # Force a fresh candidate-pool gen_code pass once, in case the cache is stale.
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
        --master_port 29509 \
        "$COMMON_DIR/mbeir_generative_retriever.py" \
        --config_path "$SWEEP_CONFIG" \
        --genir_dir "$GENIR_DIR" \
        --mbeir_data_dir "$MBEIR_DATA_DIR"

    LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
    if [ -n "$LATEST_TSV" ]; then
        cp "$LATEST_TSV" "$SWEEP_DIR/epoch_${EPOCH}.tsv"
        echo "Saved results for epoch $EPOCH -> $SWEEP_DIR/epoch_${EPOCH}.tsv"
    else
        echo "WARNING: no results tsv found for epoch $EPOCH"
    fi
done

echo ""
echo "=== Sweep complete. Building summary ==="
python "$GENIR_DIR/scripts/summarize_coco_sweep.py" --sweep_dir "$SWEEP_DIR"
