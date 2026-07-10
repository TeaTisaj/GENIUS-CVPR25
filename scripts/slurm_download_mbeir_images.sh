#!/bin/bash
#SBATCH --job-name=download-mbeir-images
#SBATCH --output=/home/tcetoje/logs/download_mbeir_images_%j.log
#SBATCH --error=/home/tcetoje/logs/download_mbeir_images_%j.log
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Download the full M-BEIR image archive from HuggingFace (TIGER-Lab/M-BEIR).
#
# The archive is split into 4 parts (~169 GB total compressed, ~250 GB extracted).
# Parts are downloaded to mbeir_data/tmp_image_download/, then combined and
# extracted with --skip-old-files so existing fashioniq/mscoco images are kept.
#
# Missing datasets this provides:
#   webqa (309K images), edis (200K), visualnews (100K), fashion200k (49K),
#   oven (33K), nights (32K), cirr (16K)
#
# After this job completes, resubmit slurm_stage0_union.sh.

set -e

MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
PYTHON="/home/tcetoje/miniconda3/envs/genius2/bin/python"
TMP_DIR="$MBEIR_DATA_DIR/tmp_image_download"

echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "MBEIR dir:  $MBEIR_DATA_DIR"
echo "Temp dir:   $TMP_DIR"
echo ""

mkdir -p "$TMP_DIR"
export TMP_DIR

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2

# ── Step 1: Download the 4 split parts ───────────────────────────────────────
$PYTHON - <<'PYEOF'
import os
import sys
from huggingface_hub import hf_hub_download

REPO_ID = "TIGER-Lab/M-BEIR"
TMP_DIR = os.environ["TMP_DIR"]

parts = [f"mbeir_images.tar.gz.part-{i:02d}" for i in range(4)]

for part in parts:
    local_path = os.path.join(TMP_DIR, part)
    if os.path.exists(local_path):
        size_gb = os.path.getsize(local_path) / 1e9
        print(f"  SKIP ({size_gb:.1f} GB already present): {part}", flush=True)
        continue
    print(f"  Downloading {part} (~48 GB) ...", flush=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=part,
        repo_type="dataset",
        local_dir=TMP_DIR,
        local_dir_use_symlinks=False,
    )
    size_gb = os.path.getsize(local_path) / 1e9
    print(f"    -> {size_gb:.1f} GB written", flush=True)

print("All parts downloaded.", flush=True)
PYEOF

echo ""
echo "── Step 2: Verify parts ──────────────────────────────────────────────────"
for i in 00 01 02 03; do
    f="$TMP_DIR/mbeir_images.tar.gz.part-$i"
    size=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "  part-$i: $size"
done

echo ""
echo "── Step 3: Combine and extract (skip existing files) ────────────────────"
echo "  Extracting to $MBEIR_DATA_DIR ..."
cat "$TMP_DIR/mbeir_images.tar.gz.part-00" \
    "$TMP_DIR/mbeir_images.tar.gz.part-01" \
    "$TMP_DIR/mbeir_images.tar.gz.part-02" \
    "$TMP_DIR/mbeir_images.tar.gz.part-03" \
  | tar xz --skip-old-files -C "$MBEIR_DATA_DIR"
echo "  Extraction complete."

echo ""
echo "── Step 4: Verify image directories ─────────────────────────────────────"
for ds in webqa_images edis_images visualnews_images fashion200k_images \
           oven_images nights_images cirr_images fashioniq_images mscoco_images; do
    dir="$MBEIR_DATA_DIR/mbeir_images/$ds"
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        echo "  $count files  $ds"
    else
        echo "  MISSING       $ds"
    fi
done

echo ""
echo "── Step 5: Clean up temp parts ──────────────────────────────────────────"
rm -f "$TMP_DIR/mbeir_images.tar.gz.part-"*
rmdir "$TMP_DIR" 2>/dev/null || true
echo "  Temp files removed."

echo ""
echo "Done. Next step: sbatch scripts/slurm_stage0_union.sh"
