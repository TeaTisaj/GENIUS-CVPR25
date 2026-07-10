"""
One-off data-quality fix for Step B: jobs 332151-332154 all crashed within minutes with a
KeyError in MBEIRDictInstructioneDataset.__getitem__ (mbeir_dataset.py:482) while looking up a
training query's positive candidate in the Stage-0 embeddings dict
(extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt).

Root cause (measured, not assumed): of the 615,538 distinct positive candidate dids referenced by
mbeir_coco_only_train.jsonl, 2,102 (0.34%) are missing from that embeddings dict's id_to_index.
The dict's mtime (2026-06-15 23:40) is AFTER the one precedent run that trained cleanly for all 30
epochs (job 329776, finished 16:44 the same day) -- the file was very likely regenerated/replaced
after that run, dropping a small number of COCO candidates that used to be present. 4-GPU DDP runs
hit one of these missing dids almost immediately because DistributedSampler shards differently
than the 1-GPU run did; the 1-GPU run got lucky across its particular shuffle order.

Fix (user-chosen): drop training queries whose positive candidate(s) are missing, rather than
re-extracting Stage-0 embeddings. This is a strict subset filter -- no query gets a positive
added or changed, just dropped if unusable. ~0.34% of training signal lost, not the 0.34% of all
candidates (a query is dropped only if it has a positive in this missing set; some dids may not
even be referenced by any query, so the dropped-query count can be smaller).

Run with: PYTHONPATH=<repo>/src/data/preprocessing python filter_missing_positive_queries.py
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "data", "preprocessing"))
from utils import hash_did  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", default="/home/tcetoje/GENIUS-CVPR25")
    parser.add_argument("--mbeir_data_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
    parser.add_argument("--pool_dict_rel", default="extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt")
    parser.add_argument("--query_rel", default="query/union_train/mbeir_coco_only_train.jsonl")
    parser.add_argument("--out_name", default="mbeir_coco_only_train_filtered.jsonl")
    args = parser.parse_args()

    pool_dict_path = os.path.join(args.genir_dir, args.pool_dict_rel)
    query_path = os.path.join(args.mbeir_data_dir, args.query_rel)
    out_path = os.path.join(os.path.dirname(query_path), args.out_name)

    print(f"Loading embeddings dict id_to_index from {pool_dict_path} ...")
    pool_dict = torch.load(pool_dict_path, map_location="cpu", weights_only=False, mmap=True)
    id_to_index = pool_dict["id_to_index"]
    print(f"  {len(id_to_index)} candidate embeddings available")

    n_total, n_dropped = 0, 0
    missing_dids_seen = set()
    with open(query_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            n_total += 1
            q = json.loads(line)
            pos_cand_list = q.get("pos_cand_list", [])
            missing = [did for did in pos_cand_list if hash_did(did) not in id_to_index]
            if missing:
                n_dropped += 1
                missing_dids_seen.update(missing)
                continue
            f_out.write(line)

    print(f"Total queries: {n_total}")
    print(f"Dropped queries (>=1 positive missing from embeddings dict): {n_dropped} "
          f"({100 * n_dropped / n_total:.4f}%)")
    print(f"Distinct missing candidate dids encountered: {len(missing_dids_seen)}")
    print(f"Filtered query file written to {out_path}")


if __name__ == "__main__":
    main()
