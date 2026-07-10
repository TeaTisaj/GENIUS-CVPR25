#!/bin/bash
#SBATCH --job-name=genius-stage2
#SBATCH --output=/home/tcetoje/logs/genius_stage2_%x_%j.log
#SBATCH --error=/home/tcetoje/logs/genius_stage2_%x_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --time=12:00:00
#SBATCH --mem=200G

# Parameterized Stage-2 (T5 generative retriever) training launcher, mirroring
# scripts/slurm_train_stage1_param.sh. Replaces the hardcoded
# slurm_train_stage2_coco_only.sh so the same script can launch any of the
# 4 COCO balance-regularization variants by passing a different config.
#
# Usage: sbatch [--job-name=... --gres=... --time=... --mem=... --cpus-per-task=...] \
#          scripts/slurm_train_stage2_param.sh <config_path_relative_to_inbatch_dir> [run_label]
#
#   sbatch --job-name=genius-stage2-coco-vanilla \
#       scripts/slurm_train_stage2_param.sh inbatch_coco_only_vanilla.yaml

CONFIG_REL_PATH="$1"
RUN_LABEL="${2:-$(basename "$CONFIG_REL_PATH" .yaml)}"

if [ -z "$CONFIG_REL_PATH" ]; then
    echo "ERROR: usage: sbatch slurm_train_stage2_param.sh <config_path> [run_label]"
    echo "  <config_path> is relative to src/models/generative_retriever/configs_scripts/large/train/inbatch/"
    exit 1
fi

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
SRC="$GENIR_DIR/src"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
COMMON_DIR="$SRC/common"
MODEL_DIR="$SRC/models/generative_retriever"
INBATCH_DIR="$MODEL_DIR/configs_scripts/large/train/inbatch"
CONFIG_PATH="$INBATCH_DIR/$CONFIG_REL_PATH"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found at $CONFIG_PATH"
    exit 1
fi

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
export PATH="/home/tcetoje/miniconda3/envs/genius2/bin:$PATH"
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false

# Same reasoning as slurm_train_stage1_param.sh: derive NPROC from Slurm's own
# allocation variable rather than trusting nvidia-smi -L, which may list every
# physical GPU on a shared node regardless of this job's actual --gres grant.
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NPROC="$SLURM_GPUS_ON_NODE"
    echo "NPROC derived from \$SLURM_GPUS_ON_NODE"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NPROC=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
    echo "NPROC derived from \$CUDA_VISIBLE_DEVICES (Slurm-exported)"
else
    NPROC=$(nvidia-smi -L | wc -l)
    echo "WARNING: neither \$SLURM_GPUS_ON_NODE nor \$CUDA_VISIBLE_DEVICES was set;" \
         "falling back to 'nvidia-smi -L', which may report the node's full" \
         "physical GPU count rather than this job's actual allocation. Verify" \
         "NPROC below matches the --gres request before trusting this run."
fi
if [ -z "$NPROC" ] || [ "$NPROC" -lt 1 ]; then
    echo "ERROR: could not determine a valid GPU count for this job (NPROC=$NPROC)."
    exit 1
fi

echo "Node:       $(hostname)"
echo "Run label:  $RUN_LABEL"
echo "Config:     $CONFIG_PATH"
echo "Job ID:     $SLURM_JOB_ID"
echo "NPROC (GPUs visible to job): $NPROC"

# Derived from the config being submitted (codebook_config.pool_path/query_path), not
# hardcoded to COCO's paths -- a hardcoded check would pass vacuously for any other
# dataset's config whose Stage-0 dicts live elsewhere, silently skipping the check it's
# meant to perform (see RQ3 planning notes: a FashionIQ config's Stage-0 dicts live under
# extracted_embed/CLIP_SF/train_fashioniq/, not .../train/).
POOL_REL=$(python3 -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG_PATH'); print(c.codebook_config.pool_path)")
QUERY_REL=$(python3 -c "from omegaconf import OmegaConf; c=OmegaConf.load('$CONFIG_PATH'); print(c.codebook_config.query_path)")
# Resolve symlinks so the -f check works even when extracted_embed/ is a symlink
# to an NFS path (e.g. /fnwi_fs/...) whose target may not be visible via the
# symlink path on all cluster nodes even though the target itself is accessible.
POOL_PT=$(python3 -c "import os; print(os.path.realpath('$GENIR_DIR/$POOL_REL'))")
QUERY_PT=$(python3 -c "import os; print(os.path.realpath('$GENIR_DIR/$QUERY_REL'))")
if [ ! -f "$POOL_PT" ] || [ ! -f "$QUERY_PT" ]; then
    echo "ERROR: Stage 0 embeddings not found."
    echo "  Missing: $POOL_PT"
    echo "  Missing: $QUERY_PT"
    exit 1
fi
echo "Stage 0 embeddings: OK ($POOL_PT, $QUERY_PT)"

cd "$COMMON_DIR"
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path "$CONFIG_PATH" \
    --enable_instruct True

# Per-job master_port: multiple jobs from this script can land on the same
# shared node at once, and a fixed port causes EADDRINUSE for all but the
# first job to bind it.
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
echo "Run '$RUN_LABEL' complete."
