"""
GENIUS Phase 2 — candidate-pool composition study, Deliverables 1a/1b/1c.

Pure CPU/numpy: reuses the precomputed query codes (already beam-decoded by the
trained T5 decoder) and precomputed candidate-pool RQ codes (quantizer-only,
no beam search) that already exist under gen_code/.../{test,cand_pool}/. Only
candidate-pool *composition* changes across 1a/1b/1c -- nothing here re-runs
the model.

Run with PYTHONPATH=<repo>/src, e.g.:
    PYTHONPATH=src python src/common/phase2_pool_eval.py --deliverable all \
        --genir_dir /home/tcetoje/GENIUS-CVPR25 \
        --mbeir_data_dir /fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data
"""
import argparse
import csv
import os

import numpy as np

from mbeir_generative_retriever import (
    create_hash_map,
    create_id_to_index_map,
    retrieve_indices_for_query,
    compute_recall_at_k,
    load_qrel,
)
from data.preprocessing.utils import unhash_did, unhash_qid, hash_did
from phase2_pool_utils import load_cand_pool


QUERY_NAME = "mscoco_task0_test"
# "several additional image datasets" step (mentor's progression, step 3).
SEVERAL_ADDITIONAL_NAMES = ["fashioniq_task7", "fashion200k_task0", "nights_task4", "cirr_task7"]
# True full-union *image-modality* pool: adds VisualNews, the only one of the
# remaining M-BEIR datasets (VisualNews/WebQA/EDIS/OVEN/INFOSEEK) that actually
# contains pure image-modality candidates -- WebQA/EDIS/OVEN/INFOSEEK are
# text-only or composite (image,text) candidates (verified directly against
# cand_pool/global/mbeir_union_train_cand_pool.jsonl), so they contribute zero
# additional candidates to an image-modality-filtered T->I pool and were
# intentionally not extracted (would have cost significant GPU time, WebQA
# alone is COCO-sized, for no effect on this analysis).
FULL_UNION_IMAGE_NAMES = SEVERAL_ADDITIONAL_NAMES + ["visualnews_task0"]
CROSS_DATASET_NAMES = FULL_UNION_IMAGE_NAMES  # used as the 1c cross-dataset distractor pool
IMAGE_ONLY_FILTER_NAMES = {"fashion200k_task0", "visualnews_task0"}  # pools that mix modalities and need filtering

POOL_VARIANTS_1B = [
    ("coco_only", [QUERY_NAME], set()),
    ("coco_fashioniq", [QUERY_NAME, "fashioniq_task7"], set()),
    ("coco_several_additional", [QUERY_NAME] + SEVERAL_ADDITIONAL_NAMES, IMAGE_ONLY_FILTER_NAMES),
    ("coco_full_union_image", [QUERY_NAME] + FULL_UNION_IMAGE_NAMES, IMAGE_ONLY_FILTER_NAMES),
]


def build_named_pool(gen_code_dir, names, image_only_names=()):
    codes_list, ids_list = [], []
    for name in names:
        codes, ids = load_cand_pool(gen_code_dir, name)
        if name in image_only_names:
            mask = codes[:, 0] == 0
            codes, ids = codes[mask], ids[mask]
        codes_list.append(codes)
        ids_list.append(ids)
    codes = np.concatenate(codes_list, axis=0)
    ids = np.concatenate(ids_list, axis=0)
    assert len(ids) == len(codes)
    assert len(set(ids.tolist())) == len(ids), "Duplicate ids across concatenated pools."
    return codes, ids


def evaluate_pool(query_codes, query_ids, cand_codes, cand_ids, qrel, k_values, query_indices=None):
    hash_map = create_hash_map(cand_codes, cand_ids)
    k_max = max(k_values)
    recalls = {k: [] for k in k_values}
    indices = range(len(query_ids)) if query_indices is None else query_indices
    for i in indices:
        retrieved = retrieve_indices_for_query(query_codes[i], hash_map, k_max)
        retrieved_unhashed = [unhash_did(idx) for idx in retrieved]
        qid = unhash_qid(int(query_ids[i]))
        relevant_docs = qrel.get(qid)
        if relevant_docs is None:
            continue
        for k in k_values:
            recalls[k].append(compute_recall_at_k(relevant_docs, retrieved_unhashed, k))
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in recalls.items()}


def load_query_data(gen_code_dir):
    query_codes = np.load(os.path.join(gen_code_dir, "test", f"mbeir_{QUERY_NAME}_codes.npy"))
    query_ids = np.load(os.path.join(gen_code_dir, "test", f"mbeir_{QUERY_NAME}_ids.npy"))
    return query_codes, query_ids


def run_1a_1b(gen_code_dir, query_codes, query_ids, qrel, writer, deliverables_wanted):
    for variant_name, names, image_only_names in POOL_VARIANTS_1B:
        deliverable = "1a" if variant_name == "coco_only" else "1b"
        if deliverable not in deliverables_wanted:
            continue
        codes, ids = build_named_pool(gen_code_dir, names, image_only_names)
        recalls = evaluate_pool(query_codes, query_ids, codes, ids, qrel, k_values=[1, 5, 10])
        for k, value in recalls.items():
            writer.writerow([deliverable, variant_name, len(ids), "-", "-", f"Recall@{k}", round(value, 4)])
            print(f"[{deliverable}] {variant_name} (n={len(ids)}): Recall@{k} = {value:.4f}")


def run_1c(gen_code_dir, query_codes, query_ids, qrel, writer, query_subset_seed, distractor_seeds, pool_sizes, out_dir):
    import random

    all_qids = sorted(qrel.keys())
    rng = random.Random(query_subset_seed)
    n_subset = min(2000, len(all_qids))
    selected_qids = sorted(rng.sample(all_qids, n_subset))

    subset_path = os.path.join(out_dir, f"coco_test_query_subset_seed{query_subset_seed}.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(subset_path, "w") as f:
        f.write("\n".join(selected_qids) + "\n")
    print(f"1c: persisted fixed {n_subset}-query subset to {subset_path}")

    qid_to_index = {unhash_qid(int(qid_hashed)): i for i, qid_hashed in enumerate(query_ids)}
    selected_indices = [qid_to_index[qid] for qid in selected_qids if qid in qid_to_index]
    assert len(selected_indices) == n_subset, "Some selected qids missing from query codes file."

    relevant_set = set()
    for qid in selected_qids:
        relevant_set.update(qrel[qid])
    relevant_hashed_ids = np.array([hash_did(d) for d in relevant_set], dtype=np.int64)
    print(f"1c: {n_subset} queries -> {len(relevant_set)} unique relevant COCO targets.")

    # qrels mix modalities (a caption query's "relevant docs" include both the
    # target image AND sibling caption ids describing the same image), so the
    # guaranteed-included relevant set must be looked up against the FULL
    # (unfiltered) COCO pool, matching the original eval pipeline's behavior.
    coco_codes, coco_ids = load_cand_pool(gen_code_dir, QUERY_NAME)
    coco_id_to_idx = create_id_to_index_map(coco_ids.tolist())
    missing = [h for h in relevant_hashed_ids if h not in coco_id_to_idx]
    assert not missing, f"{len(missing)} relevant targets not found in COCO candidate pool."
    relevant_codes = coco_codes[[coco_id_to_idx[h] for h in relevant_hashed_ids]]

    # Distractor sampling, however, should only draw from COCO *image*
    # candidates -- these are the actual "same-dataset, visually similar"
    # stress-test distractors the mentor's directive is about; COCO caption
    # candidates would almost never collide with a T->I query's predicted
    # (image-modality) code anyway, per the Deliverable 2 finding below.
    img_mask = coco_codes[:, 0] == 0
    coco_img_codes, coco_img_ids = coco_codes[img_mask], coco_ids[img_mask]
    avail_mask = ~np.isin(coco_img_ids, relevant_hashed_ids)
    avail_coco_ids = coco_img_ids[avail_mask]
    avail_coco_codes = coco_img_codes[avail_mask]

    cross_codes, cross_ids = build_named_pool(gen_code_dir, CROSS_DATASET_NAMES, IMAGE_ONLY_FILTER_NAMES)
    print(f"1c: cross-dataset distractor pool size = {len(cross_ids)}")

    for size in pool_sizes:
        n_distractors = size - len(relevant_set)
        assert n_distractors > 0, f"pool size {size} too small for {len(relevant_set)} relevant targets"
        for dist_type in ("within", "cross", "mixed"):
            for seed in distractor_seeds:
                rng_np = np.random.RandomState(seed)
                if dist_type == "within":
                    idx = rng_np.choice(len(avail_coco_ids), size=n_distractors, replace=False)
                    d_codes, d_ids = avail_coco_codes[idx], avail_coco_ids[idx]
                elif dist_type == "cross":
                    idx = rng_np.choice(len(cross_ids), size=n_distractors, replace=False)
                    d_codes, d_ids = cross_codes[idx], cross_ids[idx]
                else:  # mixed
                    n_half1 = n_distractors // 2
                    n_half2 = n_distractors - n_half1
                    idx1 = rng_np.choice(len(avail_coco_ids), size=n_half1, replace=False)
                    rng_np2 = np.random.RandomState(seed + 1000)
                    idx2 = rng_np2.choice(len(cross_ids), size=n_half2, replace=False)
                    d_codes = np.concatenate([avail_coco_codes[idx1], cross_codes[idx2]], axis=0)
                    d_ids = np.concatenate([avail_coco_ids[idx1], cross_ids[idx2]], axis=0)

                pool_codes = np.concatenate([relevant_codes, d_codes], axis=0)
                pool_ids = np.concatenate([relevant_hashed_ids, d_ids], axis=0)
                assert len(pool_ids) == size, f"pool size mismatch: {len(pool_ids)} != {size}"
                assert set(relevant_hashed_ids.tolist()).issubset(set(pool_ids.tolist()))

                recalls = evaluate_pool(
                    query_codes, query_ids, pool_codes, pool_ids, qrel,
                    k_values=[1, 5, 10], query_indices=selected_indices,
                )
                for k, value in recalls.items():
                    writer.writerow(["1c", "controlled", size, dist_type, seed, f"Recall@{k}", round(value, 4)])
                print(f"[1c] size={size} type={dist_type} seed={seed}: " +
                      " ".join(f"R@{k}={v:.4f}" for k, v in recalls.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deliverable", choices=["1a", "1b", "1c", "all"], default="all")
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", required=True)
    parser.add_argument(
        "--gen_code_dir",
        default="gen_code/GENIUS_t5small/Large/Instruct/InBatch",
        help="Relative to --genir_dir",
    )
    parser.add_argument("--query_subset_seed", type=int, default=2023)
    parser.add_argument("--distractor_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--pool_sizes", type=int, nargs="+", default=[10000, 50000, 100000])
    parser.add_argument("--out", default="retrieval_results/phase2/pool_composition_results.tsv")
    args = parser.parse_args()

    gen_code_dir = os.path.join(args.genir_dir, args.gen_code_dir)
    qrel_path = os.path.join(args.mbeir_data_dir, "qrels", "test", f"mbeir_{QUERY_NAME}_qrels.txt")
    out_path = os.path.join(args.genir_dir, args.out)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    query_codes, query_ids = load_query_data(gen_code_dir)
    qrel, _ = load_qrel(qrel_path)

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["deliverable", "pool_variant", "pool_size", "distractor_type", "seed", "metric", "value"])

        if args.deliverable in ("1a", "1b", "all"):
            wanted = {"1a", "1b"} if args.deliverable == "all" else {args.deliverable}
            run_1a_1b(gen_code_dir, query_codes, query_ids, qrel, writer, wanted)
        if args.deliverable in ("1c", "all"):
            run_1c(
                gen_code_dir, query_codes, query_ids, qrel, writer,
                args.query_subset_seed, args.distractor_seeds, args.pool_sizes, out_dir,
            )

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
