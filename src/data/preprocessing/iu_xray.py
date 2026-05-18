"""
iu_xray.py

Converts IU X-Ray (Open-i Indiana University) to M-BEIR JSONL format for GENIUS retrieval.

Expected directory layout:
  <iu_xray_dir>/
    indiana_projections.csv   (uid, filename, projection)
    indiana_reports.csv       (uid, filename, projection, Problems, MeSH,
                               impression, findings, indication, comparison, tags)
    images/
      {filename}              PNG images (one frontal + one lateral per study)

Output (written under mbeir_data_dir):
  cand_pool/local/mbeir_iu_xray_cand_pool.jsonl
  query/train/mbeir_iu_xray_train.jsonl
  query/val/mbeir_iu_xray_val.jsonl
  query/test/mbeir_iu_xray_test.jsonl

Retrieval task: image -> image,text
  Query    : frontal CXR image of a study
  Candidate: same study — frontal image + findings + impression
  Split    : deterministic hash-based 80/10/10 by uid
             (IU X-Ray has no official split; patient IDs not exposed in CSV)

Usage:
  python iu_xray.py \\
      --mbeir_data_dir /path/to/mbeir_data \\
      --iu_xray_dir    /path/to/iu-xray \\
      --enable_candidate_pool \\
      --enable_mbeir_conversion

Note: images must be placed (or symlinked) under
  <mbeir_data_dir>/mbeir_images/iu_xray_images/
before running feature extraction.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from utils import (
    count_entries_in_file,
    format_string,
    get_dataset_id,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_cand_pool_stats,
    print_mbeir_format_dataset_stats,
    save_list_as_jsonl,
)

IU_XRAY_DATASET_ID = get_dataset_id("IU_XRAY")
assert IU_XRAY_DATASET_ID is not None, "IU_XRAY not found in DATASET_IDS — add it to utils.py"

# Projection label used in indiana_projections.csv for frontal images.
FRONTAL_LABEL = "Frontal"


# ---------------------------------------------------------------------------
# Split assignment (deterministic, reproducible, no official split exists)
# ---------------------------------------------------------------------------

def assign_split(uid):
    """Hash uid → train/val/test at roughly 80/10/10."""
    h = int(hashlib.md5(str(uid).encode()).hexdigest(), 16) % 10
    if h < 2:
        return "test"
    if h < 4:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# Report text helpers
# ---------------------------------------------------------------------------

def build_report_text(findings, impression):
    parts = []
    if findings and str(findings).strip() and str(findings).strip().lower() not in ("nan", "none", ""):
        parts.append(f"FINDINGS: {str(findings).strip()}")
    if impression and str(impression).strip() and str(impression).strip().lower() not in ("nan", "none", ""):
        parts.append(f"IMPRESSION: {str(impression).strip()}")
    return " ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Path helpers (image paths are relative to mbeir_data_dir)
# ---------------------------------------------------------------------------

def image_rel_path(filename):
    return os.path.join("mbeir_images", "iu_xray_images", filename)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_projections_csv(iu_xray_dir):
    """
    Returns {uid: [(filename, projection), ...]}
    uid values are strings.
    """
    path = os.path.join(iu_xray_dir, "indiana_projections.csv")
    study_images = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["uid"].strip()
            study_images.setdefault(uid, []).append(
                (row["filename"].strip(), row["projection"].strip())
            )
    return study_images


def load_reports_csv(iu_xray_dir):
    """
    Returns {uid: {"findings": ..., "impression": ...}}
    uid values are strings.
    """
    path = os.path.join(iu_xray_dir, "indiana_reports.csv")
    reports = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["uid"].strip()
            reports[uid] = {
                "findings": row.get("findings", ""),
                "impression": row.get("impression", ""),
            }
    return reports


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def select_frontal_image(images):
    """Return filename of the frontal image, or None."""
    for filename, projection in images:
        if projection.strip().lower() == FRONTAL_LABEL.lower():
            return filename
    return None


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def generate_iu_xray_candidate_pool(
    iu_xray_dir,
    mbeir_data_dir,
    output_path,
    study_images,
    reports,
):
    """
    Write candidate pool JSONL. Returns {uid: did}.
    Studies without a frontal image or with an empty report are skipped.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    document_id = 1
    uid_to_did = {}
    skipped = {"no_frontal": 0, "empty_report": 0}

    with open(output_path, "w") as f:
        for uid in sorted(study_images.keys(), key=lambda x: int(x) if x.isdigit() else x):
            frontal = select_frontal_image(study_images[uid])
            if frontal is None:
                skipped["no_frontal"] += 1
                continue

            report = reports.get(uid, {})
            report_text = format_string(
                build_report_text(report.get("findings"), report.get("impression"))
            )
            if not report_text or len(report_text) < 10:
                skipped["empty_report"] += 1
                continue

            did = f"{IU_XRAY_DATASET_ID}:{document_id}"
            entry = {
                "did": did,
                "txt": report_text,
                "img_path": image_rel_path(frontal),
                "modality": "image,text",
                "src_content": json.dumps({"uid": uid, "filename": frontal}),
            }
            f.write(json.dumps(entry) + "\n")
            uid_to_did[uid] = did
            document_id += 1

    print(f"\nCandidate pool → {output_path}")
    print(f"  Candidates written : {document_id - 1}")
    print(f"  Skipped no-frontal : {skipped['no_frontal']}")
    print(f"  Skipped empty rpt  : {skipped['empty_report']}")

    return uid_to_did


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def generate_iu_xray_queries(
    split_name,
    output_path,
    study_images,
    uid_to_did,
):
    """Write query JSONL for one split (train / val / test)."""
    query_id = 1
    entries = []

    for uid, did in uid_to_did.items():
        if assign_split(uid) != split_name:
            continue

        frontal = select_frontal_image(study_images[uid])
        if frontal is None:
            continue

        entries.append({
            "qid": f"{IU_XRAY_DATASET_ID}:{query_id}",
            "query_txt": None,
            "query_img_path": image_rel_path(frontal),
            "query_modality": "image",
            "query_src_content": None,
            "pos_cand_list": [did],
            "neg_cand_list": [],
        })
        query_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_list_as_jsonl(entries, output_path)
    print(f"Query {split_name:5s} → {output_path}  ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert IU X-Ray to M-BEIR JSONL format for GENIUS retrieval."
    )
    parser.add_argument("--mbeir_data_dir", type=str, required=True,
                        help="Root of the M-BEIR data directory.")
    parser.add_argument("--iu_xray_dir", type=str, required=True,
                        help="Root of the downloaded IU X-Ray dataset.")
    parser.add_argument("--enable_candidate_pool", action="store_true",
                        help="Generate candidate pool JSONL.")
    parser.add_argument("--enable_mbeir_conversion", action="store_true",
                        help="Generate train/val/test query JSONL files.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Loading IU X-Ray CSV files...")
    study_images = load_projections_csv(args.iu_xray_dir)
    reports = load_reports_csv(args.iu_xray_dir)
    print(f"  Studies in projections CSV : {len(study_images)}")
    print(f"  Studies in reports CSV     : {len(reports)}")

    cand_pool_path = os.path.join(
        args.mbeir_data_dir, "cand_pool", "local", "mbeir_iu_xray_cand_pool.jsonl"
    )

    uid_to_did = None

    if args.enable_candidate_pool:
        print("\n--- Generating candidate pool ---")
        uid_to_did = generate_iu_xray_candidate_pool(
            iu_xray_dir=args.iu_xray_dir,
            mbeir_data_dir=args.mbeir_data_dir,
            output_path=cand_pool_path,
            study_images=study_images,
            reports=reports,
        )
        print_mbeir_format_cand_pool_stats(cand_pool_path)

    if args.enable_mbeir_conversion:
        print("\n--- Generating query files ---")

        if uid_to_did is None:
            if not os.path.exists(cand_pool_path):
                raise FileNotFoundError(
                    f"Candidate pool not found at {cand_pool_path}. "
                    "Run with --enable_candidate_pool first."
                )
            pool = load_mbeir_format_pool_file_as_dict(
                cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            uid_to_did = {}
            for did, entry in pool.items():
                src = json.loads(entry["src_content"])
                uid_to_did[src["uid"]] = did

        for split_name in ("train", "val", "test"):
            query_path = os.path.join(
                args.mbeir_data_dir, "query", split_name,
                f"mbeir_iu_xray_{split_name}.jsonl",
            )
            generate_iu_xray_queries(
                split_name=split_name,
                output_path=query_path,
                study_images=study_images,
                uid_to_did=uid_to_did,
            )

        print("\n--- Query stats (train) ---")
        train_query_path = os.path.join(
            args.mbeir_data_dir, "query", "train", "mbeir_iu_xray_train.jsonl"
        )
        total, data = count_entries_in_file(train_query_path)
        print(f"Total train queries: {total}")
        cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            cand_pool_path, doc_key_to_content=True, key_type="did"
        )
        print_mbeir_format_dataset_stats(data, cand_pool_dict)


if __name__ == "__main__":
    main()
