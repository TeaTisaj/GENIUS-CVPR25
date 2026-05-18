#!/bin/bash
# Download IU X-Ray (Open-i Indiana University) from Kaggle.
# Prerequisite: ~/.kaggle/kaggle.json must exist with valid credentials.
# Dataset: raddar/chest-xrays-indiana-university
#   Contains: indiana_reports.csv, indiana_projections.csv, images/ (PNG)

IU_XRAY_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/iu_xray"

source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2

echo "Downloading IU X-Ray dataset to $IU_XRAY_DIR ..."
mkdir -p "$IU_XRAY_DIR"
cd "$IU_XRAY_DIR"

kaggle datasets download raddar/chest-xrays-indiana-university --unzip

echo ""
echo "Downloaded files:"
ls -lh "$IU_XRAY_DIR"
echo ""
echo "Next steps:"
echo "  1. Symlink images to mbeir_data:"
echo "     ln -s $IU_XRAY_DIR/images /fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/mbeir_images/iu_xray_images"
echo "  2. Run preprocessing:"
echo "     bash /home/tcetoje/GENIUS-CVPR25/scripts/preprocess_iu_xray.sh"
