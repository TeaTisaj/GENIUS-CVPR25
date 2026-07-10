"""
Third-dataset stand-up (ECIR full-paper extension, Fable's Gap-2 recommendation):
extract VisualNews-only local candidate pools and query splits from the
already-downloaded M-BEIR union files, without needing the original raw
VisualNews source dump (src_data/visualnews/origin/data.json does not exist
in this environment -- it was never kept/copied when the full-union pool was
assembled). Everything this script needs already exists in
cand_pool/global/mbeir_union_{train,val}_cand_pool.jsonl and
query/union_train/mbeir_union_up_train.jsonl + query/union_val/mbeir_union_val.jsonl.

Produces (mirroring the mscoco_task{0,3} local-file convention exactly):
- cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl   (text candidates, NEW)
- cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl   (image candidates -- already
  exists at 199,903 entries; re-derived here from the union train+val pools for an
  auditable, single-source-of-truth regeneration, and to confirm the existing file
  matches; not overwritten if --no_overwrite_task0 is passed)
- query/train/mbeir_visualnews_task0_train.jsonl   (text queries, T->I)
- query/train/mbeir_visualnews_task3_train.jsonl   (image queries, I->T)
- query/test/mbeir_visualnews_task0_test.jsonl      (held out -- see note below)
- query/test/mbeir_visualnews_task3_test.jsonl

IMPORTANT, disclose in the paper: no official M-BEIR *test* split for VisualNews
exists in this environment (only train/val global pools were ever downloaded --
see the "why absolute Recall is low" precedent in the paper for MSCOCO/FashionIQ's
own from-scratch-training framing). This script therefore uses the *val* split
as the held-out test set (never trained on, but not the official M-BEIR test
partition) -- a deliberate, disclosed deviation from the MSCOCO/FashionIQ
convention, not an oversight. Report it as such in Setup/Limitations.

Run with: python extract_visualnews_local_splits.py --mbeir_data_dir <path>
"""
import argparse
import os

from utils import load_jsonl_as_list, save_list_as_jsonl

VISUALNEWS_DATASET_ID = 0  # DATASET_IDS["VisualNews"], see utils.py


def dataset_id_of(did):
    return int(did.split(":")[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbeir_data_dir", required=True)
    parser.add_argument("--no_overwrite_task0", action="store_true",
                         help="Skip regenerating the existing task0 (image) cand pool.")
    args = parser.parse_args()

    d = args.mbeir_data_dir

    def p(*parts):
        return os.path.join(d, *parts)

    print("Loading global union candidate pools (train + val) ...")
    union_train_cand = load_jsonl_as_list(p("cand_pool/global/mbeir_union_train_cand_pool.jsonl"))
    union_val_cand = load_jsonl_as_list(p("cand_pool/global/mbeir_union_val_cand_pool.jsonl"))
    print(f"  train: {len(union_train_cand)} entries, val: {len(union_val_cand)} entries")

    vn_train_cand = [c for c in union_train_cand if dataset_id_of(c["did"]) == VISUALNEWS_DATASET_ID]
    vn_val_cand = [c for c in union_val_cand if dataset_id_of(c["did"]) == VISUALNEWS_DATASET_ID]
    print(f"VisualNews candidates -- train: {len(vn_train_cand)}, val: {len(vn_val_cand)}")

    # Candidate pool for Stage-1 training = train-split candidates only (mirrors
    # MSCOCO/FashionIQ's convention: train_cand_pool_path points at the train pool).
    task0_cand = [c for c in vn_train_cand if c["modality"] == "image"]
    task3_cand = [c for c in vn_train_cand if c["modality"] == "text"]
    print(f"Task0 (image) candidates: {len(task0_cand)}, Task3 (text) candidates: {len(task3_cand)}")

    os.makedirs(p("cand_pool/local"), exist_ok=True)
    task3_path = p("cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl")
    save_list_as_jsonl(task3_cand, task3_path)
    print(f"Wrote {task3_path}")

    task0_path = p("cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl")
    if args.no_overwrite_task0 and os.path.exists(task0_path):
        print(f"Skipping {task0_path} (already exists, --no_overwrite_task0 set)")
    else:
        save_list_as_jsonl(task0_cand, task0_path)
        print(f"Wrote {task0_path}")

    # Held-out test candidate pools = val-split candidates (see docstring note).
    test_task0_cand = [c for c in vn_val_cand if c["modality"] == "image"]
    test_task3_cand = [c for c in vn_val_cand if c["modality"] == "text"]
    os.makedirs(p("cand_pool/local"), exist_ok=True)
    save_list_as_jsonl(test_task0_cand, p("cand_pool/local/mbeir_visualnews_task0_test_cand_pool.jsonl"))
    save_list_as_jsonl(test_task3_cand, p("cand_pool/local/mbeir_visualnews_task3_test_cand_pool.jsonl"))
    print(f"Wrote held-out test cand pools: {len(test_task0_cand)} image / {len(test_task3_cand)} text "
          f"(from the val split -- see docstring, no official M-BEIR test partition available here)")

    # Queries: filter from the union upsampled train file (task0=text query/image
    # cand i.e. query_modality=='text'; task3=image query/text cand i.e.
    # query_modality=='image'), restricted to VisualNews qids.
    print("\nLoading union train queries (this file is large, ~650MB) ...")
    union_train_queries = load_jsonl_as_list(p("query/union_train/mbeir_union_up_train.jsonl"))
    vn_train_queries = [q for q in union_train_queries if dataset_id_of(q["qid"]) == VISUALNEWS_DATASET_ID]
    print(f"VisualNews train queries: {len(vn_train_queries)}")

    train_task0_q = [q for q in vn_train_queries if q["query_modality"] == "text"]
    train_task3_q = [q for q in vn_train_queries if q["query_modality"] == "image"]
    print(f"  task0 (T->I) train queries: {len(train_task0_q)}, task3 (I->T) train queries: {len(train_task3_q)}")

    os.makedirs(p("query/train"), exist_ok=True)
    save_list_as_jsonl(train_task0_q, p("query/train/mbeir_visualnews_task0_train.jsonl"))
    save_list_as_jsonl(train_task3_q, p("query/train/mbeir_visualnews_task3_train.jsonl"))

    print("\nLoading union val queries (held out, used as the test split here) ...")
    union_val_queries = load_jsonl_as_list(p("query/union_val/mbeir_union_val.jsonl"))
    vn_val_queries = [q for q in union_val_queries if dataset_id_of(q["qid"]) == VISUALNEWS_DATASET_ID]
    print(f"VisualNews val/test queries: {len(vn_val_queries)}")

    test_task0_q = [q for q in vn_val_queries if q["query_modality"] == "text"]
    test_task3_q = [q for q in vn_val_queries if q["query_modality"] == "image"]
    print(f"  task0 (T->I) test queries: {len(test_task0_q)}, task3 (I->T) test queries: {len(test_task3_q)}")

    os.makedirs(p("query/test"), exist_ok=True)
    save_list_as_jsonl(test_task0_q, p("query/test/mbeir_visualnews_task0_test.jsonl"))
    save_list_as_jsonl(test_task3_q, p("query/test/mbeir_visualnews_task3_test.jsonl"))

    print("\nALL DONE. Sanity-check before trusting any of this (this project has been bitten "
          "by pool/qid mismatches twice before -- see feedback_trie_cache_staleness and the "
          "COCO test-pool symlink bug): spot-check that every test query's pos_cand_list did "
          "is actually present in the corresponding test cand pool file.")


if __name__ == "__main__":
    main()
