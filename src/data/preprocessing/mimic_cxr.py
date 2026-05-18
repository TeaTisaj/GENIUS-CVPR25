"""
mimic_cxr.py

Converts MIMIC-CXR-JPG to M-BEIR JSONL format for GENIUS CXR retrieval.

Expected MIMIC-CXR-JPG directory layout:
  <mimic_cxr_dir>/
    mimic-cxr-2.0.0-split.csv        (dicom_id, subject_id, study_id, split)
    mimic-cxr-2.0.0-metadata.csv     (dicom_id, subject_id, study_id, ViewPosition, ...)
    mimic-cxr-2.0.0-chexpert.csv     (subject_id, study_id, {14 pathology labels})
    files/
      p{pid_prefix}/
        p{subject_id}/
          s{study_id}.txt             (radiology report)
          s{study_id}/
            {dicom_id}.jpg            (one image per projection)

Output (written under mbeir_data_dir):
  cand_pool/local/mbeir_mimic_cxr_cand_pool.jsonl
  query/train/mbeir_mimic_cxr_train.jsonl
  query/val/mbeir_mimic_cxr_val.jsonl
  query/test/mbeir_mimic_cxr_test.jsonl

Retrieval task: image → image,text
  Query    : frontal CXR image of a study (PA preferred over AP)
  Candidate: same study — frontal CXR image + findings + impression text
  Split    : patient-level (mimic-cxr-2.0.0-split.csv), no patient leakage across splits

CheXpert labels are stored in candidate src_content for future disease-aware training.

Usage:
  python mimic_cxr.py \\
      --mbeir_data_dir /path/to/mbeir_data \\
      --mimic_cxr_dir  /path/to/mimic-cxr-jpg \\
      --enable_candidate_pool \\
      --enable_mbeir_conversion
"""

import argparse
import csv
import json
import os
import re
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

MIMIC_CXR_DATASET_ID = get_dataset_id("MIMIC_CXR")
assert MIMIC_CXR_DATASET_ID is not None, "MIMIC_CXR not found in DATASET_IDS — add it to utils.py"

# PA preferred; AP as fallback. Laterals are excluded.
FRONTAL_VIEW_PRIORITY = ["PA", "AP"]

# Matches section headers like "FINDINGS:", "IMPRESSION :", "findings:" etc.
# Captures everything until the next known section header or end of file.
_SECTION_HEADER = r"(?:FINDINGS|IMPRESSION|EXAMINATION|INDICATION|TECHNIQUE|COMPARISON|NOTIFICATION|RECOMMENDATION|CONCLUSION|HISTORY)"
SECTION_RE = re.compile(
    rf"(?:^|\n)({_SECTION_HEADER})\s*:?\s*\n(.*?)(?=\n{_SECTION_HEADER}\s*:?|\Z)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def parse_report(report_path):
    """
    Extract FINDINGS and IMPRESSION text from a MIMIC-CXR .txt report.
    Returns (findings, impression) — either may be None.
    """
    try:
        text = Path(report_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None, None

    sections = {}
    for match in SECTION_RE.finditer(text):
        key = match.group(1).upper()
        val = match.group(2).strip()
        if val and key not in sections:
            sections[key] = val

    return sections.get("FINDINGS"), sections.get("IMPRESSION")


def build_report_text(findings, impression):
    """Concatenate findings and impression into a single string."""
    parts = []
    if findings:
        parts.append(f"FINDINGS: {findings}")
    if impression:
        parts.append(f"IMPRESSION: {impression}")
    return " ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Path helpers (all image paths are relative to mbeir_data_dir)
# ---------------------------------------------------------------------------

def image_rel_path(subject_id, study_id, dicom_id):
    pid_prefix = f"p{str(subject_id)[:2]}"
    return os.path.join(
        "mbeir_images", "mimic_cxr_images",
        pid_prefix, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg",
    )


def report_abs_path(mimic_cxr_dir, subject_id, study_id):
    pid_prefix = f"p{str(subject_id)[:2]}"
    return os.path.join(
        mimic_cxr_dir, "files",
        pid_prefix, f"p{subject_id}", f"s{study_id}.txt",
    )


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_split_csv(mimic_cxr_dir):
    """
    Returns:
      study_split   : {study_id: split}    split in {"train","validate","test"}
      study_subject : {study_id: subject_id}
    """
    path = os.path.join(mimic_cxr_dir, "mimic-cxr-2.0.0-split.csv")
    study_split, study_subject = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["study_id"]
            study_split[sid] = row["split"]
            study_subject[sid] = row["subject_id"]
    return study_split, study_subject


def load_metadata_csv(mimic_cxr_dir):
    """Returns {study_id: [(dicom_id, ViewPosition), ...]}"""
    path = os.path.join(mimic_cxr_dir, "mimic-cxr-2.0.0-metadata.csv")
    study_dicoms = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["study_id"]
            study_dicoms.setdefault(sid, []).append(
                (row["dicom_id"], row.get("ViewPosition", "").strip())
            )
    return study_dicoms


def load_chexpert_csv(mimic_cxr_dir):
    """Returns {study_id: {label: raw_value, ...}} — raw values are "1","0","-1","" """
    path = os.path.join(mimic_cxr_dir, "mimic-cxr-2.0.0-chexpert.csv")
    labels = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        pathology_cols = [c for c in reader.fieldnames if c not in ("subject_id", "study_id")]
        for row in reader:
            labels[row["study_id"]] = {col: row[col] for col in pathology_cols}
    return labels


# ---------------------------------------------------------------------------
# View selection
# ---------------------------------------------------------------------------

def select_frontal_dicom(dicoms):
    """
    Pick the best frontal-view DICOM from a study.
    Priority: PA > AP. Returns dicom_id or None if no frontal view exists.
    """
    by_view = {}
    for dicom_id, view in dicoms:
        by_view.setdefault(view.upper(), []).append(dicom_id)

    for view in FRONTAL_VIEW_PRIORITY:
        if view in by_view:
            return by_view[view][0]
    return None


# ---------------------------------------------------------------------------
# Candidate pool generation
# ---------------------------------------------------------------------------

def generate_mimic_cxr_candidate_pool(
    mimic_cxr_dir,
    mbeir_data_dir,
    output_path,
    study_split,
    study_subject,
    study_dicoms,
    chexpert_labels,
):
    """
    Write candidate pool JSONL.

    Each candidate is one study:
      img_path : frontal CXR (relative to mbeir_data_dir)
      txt      : FINDINGS + IMPRESSION concatenated
      modality : "image,text"
      src_content contains study_id, subject_id, dicom_id, and CheXpert labels
                  (kept for disease-aware hard negative mining later)

    Only studies with a frontal view AND non-empty report are included.
    Returns study_to_did {study_id: did} for building query files.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    document_id = 1
    study_to_did = {}
    skipped = {"no_frontal": 0, "no_report": 0, "empty_report": 0}

    with open(output_path, "w") as f:
        for study_id in sorted(study_subject.keys()):
            subject_id = study_subject[study_id]
            dicoms = study_dicoms.get(study_id, [])
            frontal_dicom = select_frontal_dicom(dicoms)

            if frontal_dicom is None:
                skipped["no_frontal"] += 1
                continue

            findings, impression = parse_report(
                report_abs_path(mimic_cxr_dir, subject_id, study_id)
            )
            if findings is None and impression is None:
                skipped["no_report"] += 1
                continue

            report_text = format_string(build_report_text(findings, impression))
            if not report_text or len(report_text) < 10:
                skipped["empty_report"] += 1
                continue

            did = f"{MIMIC_CXR_DATASET_ID}:{document_id}"
            entry = {
                "did": did,
                "txt": report_text,
                "img_path": image_rel_path(subject_id, study_id, frontal_dicom),
                "modality": "image,text",
                "src_content": json.dumps({
                    "study_id": study_id,
                    "subject_id": subject_id,
                    "dicom_id": frontal_dicom,
                    "chexpert_labels": chexpert_labels.get(study_id, {}),
                }),
            }
            f.write(json.dumps(entry) + "\n")
            study_to_did[study_id] = did
            document_id += 1

    print(f"\nCandidate pool → {output_path}")
    print(f"  Candidates written : {document_id - 1}")
    print(f"  Skipped no-frontal : {skipped['no_frontal']}")
    print(f"  Skipped no-report  : {skipped['no_report']}")
    print(f"  Skipped empty      : {skipped['empty_report']}")

    return study_to_did


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def generate_mimic_cxr_queries(
    split_name,
    output_path,
    study_split,
    study_subject,
    study_dicoms,
    study_to_did,
):
    """
    Write query JSONL for one split (train / val / test).

    Query modality: "image" (frontal CXR).
    Positive candidate: the study's own entry in the candidate pool.
    Studies absent from study_to_did (no frontal / bad report) are skipped.
    """
    # split.csv uses "validate", not "val"
    split_csv_key = "validate" if split_name == "val" else split_name

    query_id = 1
    entries = []

    for study_id, split in study_split.items():
        if split != split_csv_key:
            continue
        if study_id not in study_to_did:
            continue

        subject_id = study_subject[study_id]
        frontal_dicom = select_frontal_dicom(study_dicoms.get(study_id, []))
        if frontal_dicom is None:
            continue

        entries.append({
            "qid": f"{MIMIC_CXR_DATASET_ID}:{query_id}",
            "query_txt": None,
            "query_img_path": image_rel_path(subject_id, study_id, frontal_dicom),
            "query_modality": "image",
            "query_src_content": None,
            "pos_cand_list": [study_to_did[study_id]],
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
        description="Convert MIMIC-CXR-JPG to M-BEIR JSONL format for GENIUS retrieval."
    )
    parser.add_argument("--mbeir_data_dir", type=str, required=True,
                        help="Root of the M-BEIR data directory.")
    parser.add_argument("--mimic_cxr_dir", type=str, required=True,
                        help="Root of the downloaded MIMIC-CXR-JPG dataset.")
    parser.add_argument("--enable_candidate_pool", action="store_true",
                        help="Generate candidate pool JSONL.")
    parser.add_argument("--enable_mbeir_conversion", action="store_true",
                        help="Generate train/val/test query JSONL files.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Loading MIMIC-CXR-JPG CSV files...")
    study_split, study_subject = load_split_csv(args.mimic_cxr_dir)
    study_dicoms = load_metadata_csv(args.mimic_cxr_dir)
    chexpert_labels = load_chexpert_csv(args.mimic_cxr_dir)
    print(f"  Studies in split CSV  : {len(study_split)}")
    print(f"  Studies in metadata   : {len(study_dicoms)}")
    print(f"  Studies in CheXpert   : {len(chexpert_labels)}")

    cand_pool_path = os.path.join(
        args.mbeir_data_dir, "cand_pool", "local", "mbeir_mimic_cxr_cand_pool.jsonl"
    )

    study_to_did = None

    if args.enable_candidate_pool:
        print("\n--- Generating candidate pool ---")
        study_to_did = generate_mimic_cxr_candidate_pool(
            mimic_cxr_dir=args.mimic_cxr_dir,
            mbeir_data_dir=args.mbeir_data_dir,
            output_path=cand_pool_path,
            study_split=study_split,
            study_subject=study_subject,
            study_dicoms=study_dicoms,
            chexpert_labels=chexpert_labels,
        )
        print_mbeir_format_cand_pool_stats(cand_pool_path)

    if args.enable_mbeir_conversion:
        print("\n--- Generating query files ---")

        if study_to_did is None:
            # Rebuild study_to_did from existing candidate pool on disk
            if not os.path.exists(cand_pool_path):
                raise FileNotFoundError(
                    f"Candidate pool not found at {cand_pool_path}. "
                    "Run with --enable_candidate_pool first."
                )
            pool = load_mbeir_format_pool_file_as_dict(
                cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            study_to_did = {}
            for did, entry in pool.items():
                src = json.loads(entry["src_content"])
                study_to_did[src["study_id"]] = did

        for split_name in ("train", "val", "test"):
            query_path = os.path.join(
                args.mbeir_data_dir, "query", split_name,
                f"mbeir_mimic_cxr_{split_name}.jsonl",
            )
            generate_mimic_cxr_queries(
                split_name=split_name,
                output_path=query_path,
                study_split=study_split,
                study_subject=study_subject,
                study_dicoms=study_dicoms,
                study_to_did=study_to_did,
            )

        print("\n--- Query stats (train) ---")
        train_query_path = os.path.join(
            args.mbeir_data_dir, "query", "train", "mbeir_mimic_cxr_train.jsonl"
        )
        total, data = count_entries_in_file(train_query_path)
        print(f"Total train queries: {total}")
        cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            cand_pool_path, doc_key_to_content=True, key_type="did"
        )
        print_mbeir_format_dataset_stats(data, cand_pool_dict)


if __name__ == "__main__":
    main()
