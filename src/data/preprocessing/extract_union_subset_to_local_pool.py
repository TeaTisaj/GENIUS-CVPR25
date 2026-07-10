"""
One-off utility: extract per-dataset candidate-pool JSONLs for datasets that only
exist inside the M-BEIR union pool (no local/ JSONL has ever been generated for
them), by filtering the union train candidate pool on dataset id.

Used for GENIUS Phase 2 (candidate-pool composition study) to obtain local pools
for Fashion200K / NIGHTS / CIRR, which Stage 0 feature extraction needs as input.
"""
import argparse
import os

from utils import DATASET_IDS, load_jsonl_as_list, save_list_as_jsonl

# (dataset_name, local pool file name) — file names match the existing
# cand_pools_name_to_embed / cand_pools_name_to_gen_code conventions used
# elsewhere in this repo (e.g. mbeir_fashioniq_task7_cand_pool.jsonl).
TARGETS = [
    ("Fashion200K", "fashion200k_task0"),
    ("NIGHTS", "nights_task4"),
    ("CIRR", "cirr_task7"),
    ("VisualNews", "visualnews_task0"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbeir_data_dir", required=True)
    parser.add_argument(
        "--union_pool_path",
        default="cand_pool/global/mbeir_union_train_cand_pool.jsonl",
        help="Relative to --mbeir_data_dir",
    )
    parser.add_argument("--out_dir", default="cand_pool/local")
    args = parser.parse_args()

    union_path = os.path.join(args.mbeir_data_dir, args.union_pool_path)
    out_dir = os.path.join(args.mbeir_data_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading union pool from {union_path} ...")
    union_data = load_jsonl_as_list(union_path)
    print(f"Loaded {len(union_data)} entries.")

    by_dataset_id = {}
    for entry in union_data:
        dataset_id = int(entry["did"].split(":")[0])
        by_dataset_id.setdefault(dataset_id, []).append(entry)

    for dataset_name, local_name in TARGETS:
        dataset_id = DATASET_IDS[dataset_name]
        entries = by_dataset_id.get(dataset_id, [])
        out_path = os.path.join(out_dir, f"mbeir_{local_name}_cand_pool.jsonl")
        save_list_as_jsonl(entries, out_path)
        n_image = sum(1 for e in entries if e.get("modality") == "image")
        n_text = sum(1 for e in entries if e.get("modality") == "text")
        print(
            f"{dataset_name} (id={dataset_id}): wrote {len(entries)} entries "
            f"({n_image} image, {n_text} text) -> {out_path}"
        )


if __name__ == "__main__":
    main()
