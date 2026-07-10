#!/bin/bash
#SBATCH --job-name=genius-eval-coco-variant-sweep
#SBATCH --output=/home/tcetoje/logs/genius_eval_coco_variant_sweep_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_eval_coco_variant_sweep_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=08:00:00
#SBATCH --mem=128G

# Step C of the RQ1+RQ2 COCO-completion plan: real trie-constrained-beam-search
# COCO Recall@1/5/10 eval, for every saved Stage-2 epoch checkpoint, for each
# of the 4 balance-regularization variants. Generalizes
# slurm_eval_coco_epoch_sweep.sh (which sweeps epochs for a single quantizer)
# into a double loop: outer over variant (each has its own frozen RQ
# quantizer, so candidate codes differ across variants and must be
# regenerated once per variant), inner over saved epochs.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

EPOCHS="5 10 15 20 25"
# Optional $1: space-separated subset of variants (e.g. "vanilla medium strong"
# to run Step C for already-finished variants without waiting on a still-
# training one). Defaults to all 4 if no arg given.
VARIANTS="${1:-vanilla weak medium strong}"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

for VARIANT in $VARIANTS; do
    echo ""
    echo "##### Variant: $VARIANT #####"

    BASE_CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}.yaml"
    SWEEP_CONFIG="$EVAL_DIR/config_eval_coco_only_${VARIANT}_sweep_tmp.yaml"
    EXP_NAME="CocoOnly$(python3 -c "print('${VARIANT}'.capitalize())")"
    RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"
    CKPT_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage2/GENIUS_t5small/Large/Instruct/${EXP_NAME}"
    SWEEP_DIR="$GENIR_DIR/retrieval_results/coco_epoch_sweep_${VARIANT}"
    mkdir -p "$SWEEP_DIR"

    FIRST=1
    for EPOCH in $EPOCHS; do
        CKPT_NAME="genius_t5small_epoch_${EPOCH}.pth"
        if [ ! -f "$CKPT_DIR/$CKPT_NAME" ]; then
            echo "Skipping $VARIANT epoch $EPOCH: checkpoint not found at $CKPT_DIR/$CKPT_NAME"
            continue
        fi

        echo ""
        echo "=== $VARIANT epoch $EPOCH ($CKPT_NAME) ==="

        GEN_CAND_FLAG=""
        if [ "$FIRST" -eq 1 ]; then
            # Force a fresh candidate-pool gen_code pass once per variant: the
            # frozen RQ quantizer differs across variants, so codes differ too
            # (unlike the single-quantizer sweep script, where codes are
            # identical across all T5 epochs and only need generating once
            # globally).
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
            --master_port $((29600 + RANDOM % 1000)) \
            "$COMMON_DIR/mbeir_generative_retriever.py" \
            --config_path "$SWEEP_CONFIG" \
            --genir_dir "$GENIR_DIR" \
            --mbeir_data_dir "$MBEIR_DATA_DIR"

        LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
        if [ -n "$LATEST_TSV" ]; then
            cp "$LATEST_TSV" "$SWEEP_DIR/epoch_${EPOCH}.tsv"
            echo "Saved results for $VARIANT epoch $EPOCH -> $SWEEP_DIR/epoch_${EPOCH}.tsv"
        else
            echo "WARNING: no results tsv found for $VARIANT epoch $EPOCH"
        fi
    done

    echo "=== $VARIANT sweep complete. Building per-variant summary ==="
    python "$GENIR_DIR/scripts/summarize_coco_sweep.py" --sweep_dir "$SWEEP_DIR"
done

echo ""
echo "=== All variant sweeps complete. ==="
