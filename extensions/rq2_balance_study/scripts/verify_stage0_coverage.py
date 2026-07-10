"""
Generalized Stage-0 embedding coverage check: confirms every did/qid referenced in a cand-pool or
query JSONL is actually present in the corresponding extracted embeddings dict's id_to_index.

Built for RQ3 (FashionIQ) after a real incident on COCO (see [[feedback_slurm_completed_not_success]]
in project memory, and project_rq2_balance_regularization.md): 0.34% of COCO's training positive
candidates were silently missing from their embeddings dict, crashing Stage-2 training with a
KeyError minutes in. This script makes that check explicit and mandatory before trusting a Stage-0
extraction, instead of discovering gaps reactively after a crash.

Run with: PYTHONPATH=<repo>/src/data/preprocessing python verify_stage0_coverage.py \
    --jsonl <cand_pool_or_query.jsonl> --dict_pt <embeddings.pt> --id_field did|qid
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "data", "preprocessing"))
from utils import hash_did, hash_qid  # noqa: E402

HASH_FN = {"did": hash_did, "qid": hash_qid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="cand_pool or query JSONL to check coverage for")
    parser.add_argument("--dict_pt", required=True, help="extracted embeddings .pt file with an id_to_index key")
    parser.add_argument("--id_field", choices=["did", "qid"], default="did")
    args = parser.parse_args()

    hash_fn = HASH_FN[args.id_field]

    print(f"Loading {args.dict_pt} ...")
    d = torch.load(args.dict_pt, map_location="cpu", weights_only=False, mmap=True)
    id_to_index = d["id_to_index"]
    print(f"  {len(id_to_index)} entries in id_to_index")

    n_total, n_missing = 0, 0
    missing_ids = []
    with open(args.jsonl) as f:
        for line in f:
            entry_id = json.loads(line)[args.id_field]
            n_total += 1
            if hash_fn(entry_id) not in id_to_index:
                n_missing += 1
                missing_ids.append(entry_id)

    coverage_pct = 100 * (n_total - n_missing) / n_total if n_total else 0.0
    print(f"\n{args.jsonl}")
    print(f"  total {args.id_field}s: {n_total}")
    print(f"  missing from {args.dict_pt}: {n_missing}")
    print(f"  coverage: {coverage_pct:.4f}%")
    if missing_ids:
        print(f"  sample missing: {missing_ids[:10]}")

    if n_missing > 0:
        print("\nFAIL: coverage is not 100%. Do not proceed to Stage-1/Stage-2 training against "
              "this dict until this is resolved.")
        sys.exit(1)
    print("\nPASS: 100% coverage.")


if __name__ == "__main__":
    main()
