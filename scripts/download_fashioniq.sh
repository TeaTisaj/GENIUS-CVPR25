#!/bin/bash
# Download raw FashionIQ data: captions (from GitHub) + images (Amazon ASIN lookup).
# Run on the headnode (no GPU). Safe to re-run — existing files are skipped.
#
# FashionIQ contains three clothing categories: dress, shirt, toptee.
# Caption splits: train + val (no public test captions).
# Images are Amazon product images identified by ASIN code.
#
# Steps performed:
#   1. Download caption JSON files from the official GitHub repo.
#   2. Extract all unique image IDs from the captions.
#   3. Download images from the Amazon CDN using the ASIN codes.
#
# If the Amazon CDN links fail (they occasionally go down), see:
#   https://github.com/XiaoxiaoGuo/fashion-iq
# for alternative hosting or contact the dataset authors.

set -e

MBEIR_DATA_DIR="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
FASHIONIQ_SRC="$MBEIR_DATA_DIR/src_data/fashioniq"
FASHIONIQ_IMAGES="$MBEIR_DATA_DIR/mbeir_images/fashioniq_images"
CAPTIONS_DIR="$FASHIONIQ_SRC/captions"

mkdir -p "$CAPTIONS_DIR"
mkdir -p "$FASHIONIQ_IMAGES"

# ── 1. Caption JSON files ─────────────────────────────────────────────────────
# Official GitHub raw URLs for caption files.
GITHUB_RAW="https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions"

for cat in dress shirt toptee; do
    for split in train val; do
        fname="cap.${cat}.${split}.json"
        dest="$CAPTIONS_DIR/$fname"
        if [ ! -f "$dest" ]; then
            echo "Downloading $fname ..."
            wget -q "$GITHUB_RAW/$fname" -O "$dest"
        else
            echo "Already exists: $fname"
        fi
    done
done

echo ""
echo "Caption files:"
ls -lh "$CAPTIONS_DIR/"

# ── 2. Extract unique image IDs ───────────────────────────────────────────────
echo ""
echo "Extracting unique image IDs from captions..."

PYTHON="/home/tcetoje/miniconda3/envs/genius2/bin/python"
IMAGE_ID_LIST="$FASHIONIQ_SRC/image_ids.txt"

$PYTHON - <<PYEOF
import json, glob, os

caption_dir = "$CAPTIONS_DIR"
out_path = "$IMAGE_ID_LIST"

ids = set()
for fp in glob.glob(os.path.join(caption_dir, "cap.*.json")):
    with open(fp) as f:
        data = json.load(f)
    for entry in data:
        ids.add(entry["candidate"])
        ids.add(entry["target"])

ids = sorted(ids)
with open(out_path, "w") as f:
    f.write("\n".join(ids) + "\n")
print(f"  Found {len(ids)} unique image IDs → {out_path}")
PYEOF

# ── 3. Download images ────────────────────────────────────────────────────────
# Amazon CDN URL pattern for product images.
echo ""
echo "Downloading images (Amazon CDN) ..."

AMAZON_URL="https://images-na.ssl-images-amazon.com/images/I"

total=0
skipped=0
failed=0

while IFS= read -r asin; do
    dest="$FASHIONIQ_IMAGES/${asin}.jpg"
    if [ -f "$dest" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Amazon stores images under the ASIN with a suffix; try common patterns.
    url="${AMAZON_URL}/${asin}.jpg"
    if wget -q --timeout=15 --tries=3 "$url" -O "$dest" 2>/dev/null; then
        total=$((total + 1))
    else
        rm -f "$dest"
        failed=$((failed + 1))
        echo "  WARN: failed to download $asin"
    fi
done < "$IMAGE_ID_LIST"

echo ""
echo "Image download summary:"
echo "  Downloaded : $total"
echo "  Skipped    : $skipped (already existed)"
echo "  Failed     : $failed"
echo ""
echo "Images saved to: $FASHIONIQ_IMAGES"
echo "Next: run scripts/preprocess_fashioniq.sh"
