#!/bin/bash
#SBATCH --job-name=genius-retrieval-aware-feasibility
#SBATCH --output=/home/tcetoje/logs/genius_retrieval_aware_feasibility_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_retrieval_aware_feasibility_%x_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G

# Component 7 (retrieval-aware RQ refinement feasibility study): Slurm wrapper
# for one feasibility-study training cell.
#
# HARD 1-GPU REQUIREMENT, not just a sizing default: engine.py's
# train_one_epoch_retrieval_aware asserts utils.get_world_size() == 1 --
# running this with --gres requesting >1 GPU (or --nproc_per_node>1) will
# fail that assertion loudly (by design -- see Component 6's docstring for
# why the manually-injected projected gradient would silently diverge across
# ranks on >1 GPU). Do NOT override --gres/NPROC upward for this script, unlike
# the general-purpose scripts/slurm_train_stage1_param.sh.
#
# Expects Component 1's hard-negative .pt sidecars for this cell to already
# exist (via slurm_mine_hard_negatives.sh) at the paths referenced by the
# chosen config's retrieval_aware_config.hard_neg_path/val_hard_neg_path, and
# the 20K/2K feasibility query subsample files to already exist (via
# subsample_coco_train_queries.py).
#
# Usage:
#   sbatch --job-name=genius-feasibility-vxv \
#       scripts/slurm_train_feasibility_cell.sh inbatch_coco_feasibility_vxv.yaml

CONFIG_NAME="$1"   # e.g. inbatch_coco_feasibility_vxv.yaml, relative to configs/

if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: usage: sbatch slurm_train_feasibility_cell.sh <config_name.yaml>"
    echo "  <config_name.yaml> must exist under extensions/retrieval_aware_id_refinement/configs/"
    exit 1
fi

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
MODEL_DIR="$SRC/models/residual_quantization"
EXT_DIR="$GENIR_DIR/extensions/retrieval_aware_id_refinement"
CONFIG_PATH="$EXT_DIR/configs/$CONFIG_NAME"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found at $CONFIG_PATH"
    exit 1
fi

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false

# Hard-coded to 1, unlike slurm_train_stage1_param.sh's SLURM_GPUS_ON_NODE
# auto-detection -- this study's training step is 1-GPU-only by design (see
# header comment above).
NPROC=1

echo "Node:       $(hostname)"
echo "Config:     $CONFIG_PATH"
echo "Job ID:     $SLURM_JOB_ID"
echo "NPROC:      $NPROC (hard-coded -- this study is 1-GPU-only)"

# Sanity-check the retrieval_aware_config fields this study needs actually
# exist in the chosen config before spending any GPU time.
python3 -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$CONFIG_PATH')
assert 'retrieval_aware_config' in c, 'missing retrieval_aware_config block'
assert 'warm_start_rq_ckpt' in c.model, 'missing model.warm_start_rq_ckpt'
print('Config sanity check OK. retrieval_aware_config.enabled =', c.retrieval_aware_config.enabled)
"
if [ $? -ne 0 ]; then
    echo "ERROR: config sanity check failed."
    exit 1
fi

# Verify the Stage-0 embedding dicts and the warm-start checkpoint exist before
# spending any GPU time (mirrors scripts/slurm_train_stage1_param.sh's existing
# Stage-0 check).
POOL_REL=$(python3 -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG_PATH'); print(c.codebook_config.pool_path)")
QUERY_REL=$(python3 -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG_PATH'); print(c.codebook_config.query_path)")
WARM_START_REL=$(python3 -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG_PATH'); print(c.model.warm_start_rq_ckpt)")
POOL_PT=$(python3 -c "import os; print(os.path.realpath('$GENIR_DIR/$POOL_REL'))")
QUERY_PT=$(python3 -c "import os; print(os.path.realpath('$GENIR_DIR/$QUERY_REL'))")
if [[ "$WARM_START_REL" == /* ]]; then
    WARM_START_PT="$WARM_START_REL"
else
    WARM_START_PT="$GENIR_DIR/$WARM_START_REL"
fi
if [ ! -f "$POOL_PT" ] || [ ! -f "$QUERY_PT" ]; then
    echo "ERROR: Stage 0 embeddings not found. Missing: $POOL_PT or $QUERY_PT"
    exit 1
fi
if [ ! -f "$WARM_START_PT" ]; then
    echo "ERROR: warm_start_rq_ckpt not found at $WARM_START_PT"
    exit 1
fi
echo "Stage 0 embeddings + warm-start checkpoint: OK"

cd "$SRC/common"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "Master port: $MASTER_PORT"

cd "$MODEL_DIR"
python -m torch.distributed.run \
    --master_port "$MASTER_PORT" \
    --nproc_per_node="$NPROC" \
    "$MODEL_DIR/train.py" \
    --config_path "$CONFIG_PATH" \
    --genir_dir "$GENIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"

echo ""
echo "Feasibility cell '$CONFIG_NAME' training complete."
