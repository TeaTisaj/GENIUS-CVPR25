#!/bin/bash
# Preprocessing for IU X-Ray (Open-i Indiana University) — no GPU required.
# Run directly on the headnode (not via sbatch).
#
# Expected IU X-Ray layout in IU_XRAY_DIR:
#   indiana_projections.csv
#   indiana_reports.csv
#   images/  (PNG files)
#
# After running, symlink or copy images to mbeir_data_dir/mbeir_images/iu_xray_images/
# before running feature extraction.

GENIR_DIR="/home/tcetoje/GENIUS-CVPR25"
MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
IU_XRAY_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/iu_xray"   # adjust to actual download path

export PYTHONPATH="$GENIR_DIR/src"
PYTHON="/home/tcetoje/miniconda3/envs/genius2/bin/python"

cd "$GENIR_DIR/src/data/preprocessing"

$PYTHON iu_xray.py \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --iu_xray_dir    "$IU_XRAY_DIR" \
    --enable_candidate_pool \
    --enable_mbeir_conversion

echo ""
echo "Outputs written to:"
echo "  $MBEIR_DATA_DIR/cand_pool/local/mbeir_iu_xray_cand_pool.jsonl"
echo "  $MBEIR_DATA_DIR/query/{train,val,test}/mbeir_iu_xray_{train,val,test}.jsonl"
echo ""
echo "Next: symlink images then run scripts/slurm_biomedclip_feat_extract_iu_xray.sh"
