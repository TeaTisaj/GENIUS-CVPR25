#!/bin/bash
#SBATCH --job-name=genius-fiq-weak-srchk-retrain
#SBATCH --output=/home/tcetoje/logs/genius_fiq_weak_srchk_retrain_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_fiq_weak_srchk_retrain_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=00:45:00
#SBATCH --mem=128G

# Retrain-from-scratch (2026-07-03): weak's seedrepaircheck checkpoint (repaired quantizer,
# seed=2023) has now twice produced an unsatisfiable constrained-beam-search error at eval time
# (batch 106, then batch 227 on retry) -- training itself was healthy both times (loss 13.8->2.15),
# but the resulting T5 model apparently generates out-of-trie sequences for a non-trivial number
# of candidates. Since seed7/13 on the SAME repaired quantizer evaluated fine (job 333674), this
# looks specific to this training run, not the quantizer. DDP/cuDNN training is not perfectly
# bit-reproducible even with a fixed seed (established this session) -- retraining from scratch
# may converge to a usably different model. If THIS also fails at eval, stop and escalate rather
# than retry again blindly.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
TRAIN_CONFIG="$MODEL_DIR/configs_scripts/large/train/inbatch/inbatch_fashioniq_only_weak_seedrepaircheck.yaml"
EVAL_CONFIG="$MODEL_DIR/configs_scripts/large/eval/inbatch/config_eval_fashioniq_only_weak_seedrepaircheck.yaml"
NPROC=4

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

SWEEP_DIR="$GENIR_DIR/retrieval_results/fashioniq_seed_variance"
mkdir -p "$SWEEP_DIR"

echo "##### Retraining: weak seedrepaircheck (fresh DDP run) #####"
cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$TRAIN_CONFIG" \
    --enable_instruct True

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port $((29996 + RANDOM % 1000)) \
    train.py \
    --config_path "$TRAIN_CONFIG" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: weak seedrepaircheck retrain failed with exit code $STATUS"
    exit $STATUS
fi
echo "##### Retrain complete, evaluating #####"

EXP_NAME="FashioniqOnlyWeakSeedRepairCheck"
RESULTS_DIR="$GENIR_DIR/retrieval_results/GENIUS_t5small/Large/Instruct/${EXP_NAME}/final_tsv"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$EVAL_CONFIG" \
    --enable_instruct True

python -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --master_port $((29997 + RANDOM % 1000)) \
    "$COMMON_DIR/mbeir_generative_retriever.py" \
    --config_path "$EVAL_CONFIG" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "ERROR: weak seedrepaircheck eval (post-retrain) failed with exit code $STATUS -- STOP, do not retry again blindly, escalate to user."
    exit $STATUS
fi

LATEST_TSV=$(ls -t "$RESULTS_DIR"/*.tsv 2>/dev/null | head -1)
if [ -n "$LATEST_TSV" ]; then
    cp "$LATEST_TSV" "$SWEEP_DIR/weak_seedrepaircheck.tsv"
    echo "Saved results for weak seedrepaircheck -> $SWEEP_DIR/weak_seedrepaircheck.tsv"
else
    echo "WARNING: no results tsv found for weak seedrepaircheck"
fi
