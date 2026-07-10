#!/bin/bash
#SBATCH --job-name=genius-fiq-medium-srchk-eval
#SBATCH --output=/home/tcetoje/logs/genius_fiq_medium_srchk_eval_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_fiq_medium_srchk_eval_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=00:30:00
#SBATCH --mem=128G

# Isolated retry (2026-07-03): medium's eval crashed with "CUDA error: illegal memory access"
# in job 333708, immediately after weak's eval crashed with an unhandled ValueError in the same
# job/GPU allocation -- very likely leftover corrupted CUDA/NCCL state from weak's crash, not a
# real problem with medium's (healthy-trained) checkpoint. Running medium alone, freshly
# allocated, to confirm.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
EVAL_CONFIG_DIR="$MODEL_DIR/configs_scripts/large/eval/inbatch"
NPROC=4

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/fashioniq_seed_variance"
mkdir -p "$SWEEP_DIR"

EXP_NAME="FashioniqOnlyMediumSeedRepairCheck"
EVAL_CONFIG="$EVAL_CONFIG_DIR/config_eval_fashioniq_only_medium_seedrepaircheck.yaml"
RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$EVAL_CONFIG" \
    --enable_instruct True

python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port $((29995 + RANDOM % 1000)) \
    "$COMMON_DIR/mbeir_generative_retriever.py" \
    --config_path "$EVAL_CONFIG" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: isolated medium seedrepaircheck eval failed with exit code $STATUS"
    exit $STATUS
fi

LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
if [ -n "$LATEST_TSV" ]; then
    cp "$LATEST_TSV" "$SWEEP_DIR/medium_seedrepaircheck.tsv"
    echo "Saved results for medium seedrepaircheck -> $SWEEP_DIR/medium_seedrepaircheck.tsv"
else
    echo "WARNING: no results tsv found for medium seedrepaircheck"
fi
