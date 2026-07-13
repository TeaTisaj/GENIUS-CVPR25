"""
ECIR audit Round 5, item 7: MSCOCO-strong has 69.7% full-ID collision yet the best-in-table
T->I R@1. retriever.py's non-rerank retrieval path (retrieve_indices_for_query in
mbeir_generative_retriever.py) has NO similarity-based ranking within a code's candidate
group -- ties are broken purely by candidate-pool file order (create_hash_map's insertion
order). This script asks: how much of the reported T->I R@1 gain is that arbitrary ordering,
vs. genuine discrimination?

Method (no query-side re-decoding needed): for each test query, the run file's rank-1
candidate id already tells us which candidate the model's top beam happened to land on
first. Since retrieve_indices_for_query's first block of returned candidates is exactly
cand_pool_hash_map[top_beam_code] in candidate-pool order, we recover the FULL tie group for
that top-predicted code by looking up (from compute_coco_candidate_codes.py's id->code map)
every candidate sharing that code -- not just the <=k shown in the run file. Then:

  - current R@1: unchanged, whatever the pipeline already reported.
  - expected R@1 (random tie-break): if gold is in the top-1 code's full group, 1/|group|
    (any member equally likely to land first under a random within-group shuffle); else 0.
  - adversarial R@1: 1 only if group size == 1 (gold forced to rank 1); else 0.

Run with: PYTHONPATH=<repo>/src python tie_break_sensitivity.py --genir_dir <repo>
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from data.preprocessing.utils import hash_did


def load_qrels(path):
    """qrels doc-ids are raw 'dataset:local' strings (e.g. '9:29820'); run files and the
    candidate-code npz both key on hash_did(did) (e.g. 90029820) -- hash here so they match."""
    qrel = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _, did, rel = line.strip().split()[:4]
            if int(rel) > 0:
                qrel[qid].append(hash_did(did))
    return qrel


def load_run_rank1(path):
    """qid -> rank-1 candidate id, from a TREC-format run file (col1=qid, col3=did, col4=rank)."""
    rank1 = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, did, rank = parts[0], int(parts[2]), int(parts[3])
            if rank == 1:
                rank1[qid] = did
    return rank1


def build_code_groups(ids, codes):
    """candidate_id -> tuple(code) map, and code -> list[candidate_id] group map."""
    id_to_code = {int(i): tuple(c) for i, c in zip(ids, codes)}
    code_to_group = defaultdict(list)
    for i, c in id_to_code.items():
        code_to_group[tuple(c)].append(i)
    return id_to_code, code_to_group


def analyze(variant, codes_npz_path, run_file_path, qrels_path):
    data = np.load(codes_npz_path)
    id_to_code, code_to_group = build_code_groups(data["ids"], data["codes"])

    qrel = load_qrels(qrels_path)
    rank1 = load_run_rank1(run_file_path)

    n_total = 0
    n_current_hit = 0
    expected_hits = 0.0
    n_adversarial_hit = 0
    n_gold_not_in_pool_codes = 0
    group_sizes_when_gold_present = []

    for qid, gold_list in qrel.items():
        if qid not in rank1:
            continue
        n_total += 1
        r1_id = rank1[qid]
        current_hit = 1 if r1_id in gold_list else 0
        n_current_hit += current_hit

        if r1_id not in id_to_code:
            # Shouldn't happen (rank-1 must be a real candidate), but guard anyway.
            expected_hits += current_hit
            n_adversarial_hit += current_hit
            continue

        top_code = id_to_code[r1_id]
        group = code_to_group[top_code]
        group_gold_members = [g for g in gold_list if g in set(group)]

        if group_gold_members:
            group_sizes_when_gold_present.append(len(group))
            expected_hits += 1.0 / len(group)
            if len(group) == 1:
                n_adversarial_hit += 1
        else:
            n_gold_not_in_pool_codes += 1 if not any(g in id_to_code for g in gold_list) else 0
            # gold isn't reachable via the top-1 code's group either way -> 0 contribution,
            # consistent with current_hit (which must also be 0 here).

    current_r1 = n_current_hit / n_total
    expected_r1 = expected_hits / n_total
    adversarial_r1 = n_adversarial_hit / n_total

    return {
        "variant": variant,
        "n_queries": n_total,
        "current_R@1_pct": round(100 * current_r1, 3),
        "expected_random_tiebreak_R@1_pct": round(100 * expected_r1, 3),
        "adversarial_worst_case_R@1_pct": round(100 * adversarial_r1, 3),
        "mean_top1_group_size_when_gold_present": round(
            float(np.mean(group_sizes_when_gold_present)), 2
        ) if group_sizes_when_gold_present else None,
        "n_queries_with_gold_in_top1_group": len(group_sizes_when_gold_present),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
    args = parser.parse_args()

    codes_dir = os.path.join(args.genir_dir, "extensions/rq2_balance_study/analysis/coco_candidate_codes")
    qrels_path = os.path.join(args.mbeir_data_dir, "qrels/test/mbeir_mscoco_task0_test_qrels.txt")

    run_files = {
        "vanilla": os.path.join(
            args.genir_dir,
            "retrieval_results/GENIUS_t5small/Large/Instruct/CocoOnlyVanilla/run_files/"
            "mbeir_mscoco_task0_single_pool_test_run.txt",
        ),
        "strong": os.path.join(
            args.genir_dir,
            "retrieval_results/GENIUS_t5small/Large/Instruct/CocoOnlyStrong/run_files/"
            "mbeir_mscoco_task0_single_pool_test_run.txt",
        ),
    }

    rows = []
    for variant, run_file in run_files.items():
        codes_npz = os.path.join(codes_dir, f"{variant}_candidate_codes.npz")
        if not os.path.exists(codes_npz):
            print(f"SKIP {variant}: missing {codes_npz}")
            continue
        if not os.path.exists(run_file):
            print(f"SKIP {variant}: missing {run_file}")
            continue
        result = analyze(variant, codes_npz, run_file, qrels_path)
        rows.append(result)
        print(f"\n=== {variant} ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

    out_csv = os.path.join(
        args.genir_dir, "extensions/rq2_balance_study/analysis/table_tie_break_sensitivity.csv"
    )
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
