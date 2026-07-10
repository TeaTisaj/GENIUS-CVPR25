"""
Component 9 (correcting the Criterion-5 probe methodology) -- Step 1.

Pre-flight verification of the official, held-out MSCOCO test split before
building anything on top of it. Run this BEFORE Step 2 (query embedding
extraction). All assertions are meant to fail loudly and early -- see
~/.claude/plans/so-this-is-the-swirling-quiche.md, Component 9, for why each
check exists (the qid-collision trap in particular: M-BEIR restarts qid
numbering per split, so train qid "9:1" and test qid "9:1" are different,
unrelated queries -- looking up test qids in the train query embedding dict
would report false 100% "coverage" while silently returning the wrong
embeddings).
"""
import os
import random
import sys

sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/src")

import torch

from data.preprocessing.utils import hash_did, hash_qid, load_jsonl_as_list

MBEIR_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"

TASK0_TEST_QUERY = os.path.join(MBEIR_DIR, "query/test/mbeir_mscoco_task0_test.jsonl")
TASK3_TEST_QUERY = os.path.join(MBEIR_DIR, "query/test/mbeir_mscoco_task3_test.jsonl")
TASK0_TEST_CAND = os.path.join(MBEIR_DIR, "cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl")
TASK3_TEST_CAND = os.path.join(MBEIR_DIR, "cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl")
FULL_LOCAL_POOL = os.path.join(MBEIR_DIR, "cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl")
TASK0_TEST_CAND_DICT = os.path.join(GENIR_DIR, "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_test_IT_dict.pt")
TASK3_TEST_CAND_DICT = os.path.join(GENIR_DIR, "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task3_test_IT_dict.pt")
TASK0_TRAIN_QUERY = os.path.join(MBEIR_DIR, "query/train/mbeir_mscoco_task0_train.jsonl")

random.seed(2023)


def check(label, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise AssertionError(label)


def main():
    print("=== Step 1: pre-flight verification of the official MSCOCO test split ===\n")

    task0_q = load_jsonl_as_list(TASK0_TEST_QUERY)
    task3_q = load_jsonl_as_list(TASK3_TEST_QUERY)
    task0_c = load_jsonl_as_list(TASK0_TEST_CAND)
    task3_c = load_jsonl_as_list(TASK3_TEST_CAND)

    print(f"task0 (T->I) test queries: {len(task0_q)}")
    print(f"task3 (I->T) test queries: {len(task3_q)}")
    print(f"task0 (T->I) test cand pool (expect image-only): {len(task0_c)}")
    print(f"task3 (I->T) test cand pool (expect text-only): {len(task3_c)}\n")

    check("task0 test query count == 24809", len(task0_q) == 24809)
    check("task3 test query count == 5000", len(task3_q) == 5000)
    check("task0 test cand pool count == 5000", len(task0_c) == 5000)
    check("task3 test cand pool count == 24809", len(task3_c) == 24809)

    # --- Modality homogeneity ---
    check("task0 test queries 100% query_modality=='text'",
          all(q["query_modality"] == "text" for q in task0_q))
    check("task3 test queries 100% query_modality=='image'",
          all(q["query_modality"] == "image" for q in task3_q))
    check("task0 test cand pool 100% modality=='image'",
          all(c["modality"] == "image" for c in task0_c))
    check("task3 test cand pool 100% modality=='text'",
          all(c["modality"] == "text" for c in task3_c))

    # --- qid disjointness between task0-test and task3-test (needed before merging in Step 2) ---
    task0_qids = {q["qid"] for q in task0_q}
    task3_qids = {q["qid"] for q in task3_q}
    overlap = task0_qids & task3_qids
    check(f"task0/task3 test qids disjoint (overlap={len(overlap)})", len(overlap) == 0)

    # --- Candidate coverage by content, not just key presence ---
    for cand_data, cand_dict_path, label in [
        (task0_c, TASK0_TEST_CAND_DICT, "task0"),
        (task3_c, TASK3_TEST_CAND_DICT, "task3"),
    ]:
        cand_dict = torch.load(cand_dict_path, map_location="cpu", weights_only=False)
        id_to_index = cand_dict["id_to_index"]
        sample_key = next(iter(id_to_index.keys()))
        print(f"{label} cand dict id_to_index key type: {type(sample_key).__name__}")
        n_covered = sum(1 for c in cand_data if hash_did(c["did"]) in id_to_index)
        check(f"{label} test cand pool 100% covered by its dict ({n_covered}/{len(cand_data)})",
              n_covered == len(cand_data))

    # --- Did-content identity: same did -> same content, test pool vs full local pool ---
    full_pool = load_jsonl_as_list(FULL_LOCAL_POOL)
    full_pool_by_did = {c["did"]: c for c in full_pool}
    for cand_data, label in [(task0_c, "task0"), (task3_c, "task3")]:
        sample = random.sample(cand_data, min(25, len(cand_data)))
        n_match = 0
        for c in sample:
            full_c = full_pool_by_did.get(c["did"])
            if full_c is not None and full_c.get("txt") == c.get("txt") and full_c.get("img_path") == c.get("img_path"):
                n_match += 1
        check(f"{label}: 25 random test-pool dids match full local pool content ({n_match}/{len(sample)})",
              n_match == len(sample))

    # --- The qid-collision trap: train qid "9:1" != test qid "9:1" ---
    train_q = load_jsonl_as_list(TASK0_TRAIN_QUERY)
    train_by_qid = {q["qid"]: q for q in train_q}
    task0_by_qid = {q["qid"]: q for q in task0_q}
    shared_qid = "9:1"
    train_txt = train_by_qid.get(shared_qid, {}).get("query_txt")
    test_txt = task0_by_qid.get(shared_qid, {}).get("query_txt")
    print(f"\nqid-collision trap check: train '{shared_qid}' = {train_txt!r}")
    print(f"                           test  '{shared_qid}' = {test_txt!r}")
    check("train qid '9:1' and test qid '9:1' are DIFFERENT queries (collision trap confirmed)",
          train_txt is not None and test_txt is not None and train_txt != test_txt)
    print("\n=> Regression guard: never look up test qids in the TRAIN query embedding dict "
          "(extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt) -- "
          "it will silently return unrelated train-split embeddings for the same-looking qid.")

    print("\nALL PRE-FLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
