"""
Download FashionIQ test split from MrZilinXiao/mbeir_test_fashioniq_task7 on HuggingFace.
Extracts images and writes M-BEIR JSONL files directly (data is already in M-BEIR format).

Usage:
    python scripts/download_fashioniq_hf.py
"""

import os
import json
import pathlib
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "MrZilinXiao/mbeir_test_fashioniq_task7"
MBEIR_DATA_DIR = pathlib.Path("/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
FASHIONIQ_SRC  = MBEIR_DATA_DIR / "src_data" / "fashioniq"
IMAGE_DIR      = MBEIR_DATA_DIR / "mbeir_images" / "fashioniq_images"
CAND_POOL_DIR  = MBEIR_DATA_DIR / "cand_pool" / "local"
QUERY_DIR      = MBEIR_DATA_DIR / "query" / "test"

for d in [FASHIONIQ_SRC, IMAGE_DIR, CAND_POOL_DIR, QUERY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def save_image(img_field, img_path_rel):
    """Write image bytes to disk if not already present."""
    # img_field is a dict {'bytes': b'...', 'path': 'filename.jpg'} or similar
    if img_field is None:
        return False
    raw = img_field.get("bytes") if isinstance(img_field, dict) else img_field
    if not raw:
        return False
    dest = MBEIR_DATA_DIR / img_path_rel
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    return True


# ── 1. Corpus (candidate pool) ────────────────────────────────────────────────
print("=== Extracting corpus (candidate pool) ===")
corpus_shards = [
    "corpus/test-00000-of-00004.parquet",
    "corpus/test-00001-of-00004.parquet",
    "corpus/test-00002-of-00004.parquet",
    "corpus/test-00003-of-00004.parquet",
]

cand_pool_path = FASHIONIQ_SRC / "mbeir_fashioniq_cand_pool.jsonl"
n_cands = 0
n_img_ok = 0

with open(cand_pool_path, "w") as f_out:
    for shard in corpus_shards:
        print(f"  Downloading {shard} ...")
        local = hf_hub_download(REPO, shard, repo_type="dataset")
        table  = pq.read_table(local)
        rows   = table.to_pydict()
        n_rows = len(rows["did"])

        for i in range(n_rows):
            img_path = rows["img_path"][i]   # relative path, e.g. mbeir_images/fashioniq_images/B002ZPH5NI.jpg
            img_field = rows["img"][i]

            ok = save_image(img_field, img_path)
            if ok:
                n_img_ok += 1

            entry = {
                "txt":         rows["txt"][i],
                "img_path":    img_path,
                "modality":    rows["modality"][i],
                "did":         rows["did"][i],
                "src_content": rows["src_content"][i],
            }
            f_out.write(json.dumps(entry) + "\n")
            n_cands += 1

        print(f"    → {n_rows} entries processed")

print(f"Candidate pool: {n_cands} entries, {n_img_ok} images saved → {cand_pool_path}")


# ── 2. Test queries ───────────────────────────────────────────────────────────
print("\n=== Extracting test queries ===")
query_shard = "queries/test-00000-of-00001.parquet"
local = hf_hub_download(REPO, query_shard, repo_type="dataset")
table = pq.read_table(local)
rows  = table.to_pydict()
n_rows = len(rows["qid"])

query_out = FASHIONIQ_SRC / "mbeir_fashioniq_new_test.jsonl"
n_q_img = 0

with open(query_out, "w") as f_out:
    for i in range(n_rows):
        img_path = rows["query_img_path"][i]
        img_field = rows["query_img"][i]

        ok = save_image(img_field, img_path)
        if ok:
            n_q_img += 1

        entry = {
            "qid":              rows["qid"][i],
            "query_txt":        rows["query_txt"][i],
            "query_img_path":   img_path,
            "query_modality":   rows["query_modality"][i],
            "query_src_content": rows["query_src_content"][i],
            "pos_cand_list":    rows["pos_cand_list"][i],
            "neg_cand_list":    rows["neg_cand_list"][i],
        }
        f_out.write(json.dumps(entry) + "\n")

print(f"Test queries: {n_rows} entries, {n_q_img} query images saved → {query_out}")


# ── 3. Symlinks into expected M-BEIR layout ───────────────────────────────────
print("\n=== Creating symlinks ===")

def symlink(src, dst):
    if not dst.exists():
        dst.symlink_to(src)
        print(f"  Linked: {dst.name}")
    else:
        print(f"  Already exists: {dst.name}")

# Candidate pool symlink (fashioniq uses a single pool for all tasks)
symlink(cand_pool_path,
        CAND_POOL_DIR / "mbeir_fashioniq_task7_cand_pool.jsonl")

# Test query symlink
symlink(query_out,
        QUERY_DIR / "mbeir_fashioniq_task7_test.jsonl")

print("\nDone. Summary:")
print(f"  Candidate pool  : {n_cands} entries")
print(f"  Test queries    : {n_rows} entries")
print(f"  Images saved    : {n_img_ok + n_q_img} total")
print(f"\nNext: bash scripts/setup_eval_prerequisites.sh")
print(f"      sbatch scripts/slurm_feat_extract_fashioniq.sh")
