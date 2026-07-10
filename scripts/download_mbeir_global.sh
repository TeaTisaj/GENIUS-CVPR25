#!/bin/bash
# Download M-BEIR global/union JSONL files from HuggingFace (TIGER-Lab/M-BEIR).
# These are needed for Stage 2 (Generator) training.
#
# Files downloaded (~1.26 GB total):
#   cand_pool/global/mbeir_union_train_cand_pool.jsonl   478 MB
#   cand_pool/global/mbeir_union_val_cand_pool.jsonl      48 MB
#   query/union_train/mbeir_union_up_train.jsonl         649 MB
#   query/union_val/mbeir_union_val.jsonl                 83 MB
#
# Run on the headnode (CPU only). Safe to re-run — existing files are skipped.
# Estimated time: 5–15 min depending on network.

#SBATCH --job-name=download_mbeir_global
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/tcetoje/logs/download_mbeir_global_%j.out

set -e

MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
PYTHON="/home/tcetoje/miniconda3/envs/genius2/bin/python"

echo "Downloading M-BEIR global/union files from HuggingFace..."
echo "Destination: $MBEIR_DATA_DIR"
echo ""

$PYTHON - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download

REPO_ID = "TIGER-Lab/M-BEIR"
LOCAL_DIR = os.environ.get("MBEIR_DATA_DIR", "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")

# HF repo paths match expected local layout under MBEIR_DATA_DIR
hf_paths = [
    "cand_pool/global/mbeir_union_train_cand_pool.jsonl",
    "cand_pool/global/mbeir_union_val_cand_pool.jsonl",
    "query/union_train/mbeir_union_up_train.jsonl",
    "query/union_val/mbeir_union_val.jsonl",
]

for hf_path in hf_paths:
    local_path = os.path.join(LOCAL_DIR, hf_path)
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"  SKIP ({size_mb:.0f} MB, already exists): {local_path}")
        continue

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  Downloading: {hf_path} ...", flush=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=hf_path,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"    -> {size_mb:.0f} MB written", flush=True)

print("")
print("Summary:")
for hf_path in hf_paths:
    local_path = os.path.join(LOCAL_DIR, hf_path)
    status = f"{os.path.getsize(local_path)/1e6:.0f} MB" if os.path.exists(local_path) else "MISSING"
    print(f"  {status:>10}  {local_path}")
PYEOF

echo ""
echo "Next steps:"
echo "  1. Re-run Stage 0 feature extraction on the union training data"
echo "     (the existing embeddings only cover COCO)."
echo "  2. Then submit Stage 2 training: sbatch scripts/slurm_train_stage2.sh"
