"""
split_mscoco_train_task0.py

Module description:
    The per-task MSCOCO val/test query splits (mbeir_mscoco_task0_{val,test}.jsonl,
    text->image) already exist under query/{val,test}/, but the train split was
    never produced standalone -- only merged into the 16-dataset union train file.
    This filters the raw src_data/mscoco/mbeir_mscoco_train.jsonl (which mixes
    task0 text->image and task3 image->text queries) down to task0-only, matching
    the same query_modality == "text" filter already used for the val/test splits.

Usage:
    python split_mscoco_train_task0.py --mbeir_data_dir /path/to/mbeir_data
"""

import os
import argparse

from utils import load_jsonl_as_list, save_list_as_jsonl

MSCOCO_QUERY_MODALITY_TEXT = "text"


def main(mbeir_data_dir):
    src_path = os.path.join(mbeir_data_dir, "src_data", "mscoco", "mbeir_mscoco_train.jsonl")
    dst_dir = os.path.join(mbeir_data_dir, "query", "train")
    dst_path = os.path.join(dst_dir, "mbeir_mscoco_task0_train.jsonl")

    entries = load_jsonl_as_list(src_path)
    task0_entries = [e for e in entries if e.get("query_modality") == MSCOCO_QUERY_MODALITY_TEXT]

    os.makedirs(dst_dir, exist_ok=True)
    save_list_as_jsonl(task0_entries, dst_path)

    print(f"Filtered {len(task0_entries)} / {len(entries)} entries with "
          f"query_modality == '{MSCOCO_QUERY_MODALITY_TEXT}' from {src_path}")
    print(f"Saved to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbeir_data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.mbeir_data_dir)
