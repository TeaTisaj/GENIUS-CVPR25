"""
Part 4 (Yubao's baseline-comparability answer): raw CLIP-SF dense-retrieval
baseline -- plain cosine-similarity KNN over the frozen CLIP-SF score-fusion
embeddings, with NO residual quantization and NO T5 generation anywhere in
the loop.

Adapted from `check_feasibility_criteria.py::dense_recall_at_k_probe` (via
`probe_dense_recall_official_test.py`'s existing plumbing), with the RQ step
removed entirely: instead of loading an RQ checkpoint and scoring
`model.inference(...)["quant"]` (the post-quantization reconstruction), this
script scores the raw, mask-weighted, L2-normalized CLIP-SF embeddings
(`cand_dict["img"]`/`["text"]`) directly. Every other guard from the
reference implementation is kept as-is because it is correctness-critical:
  - the corrected `OFFICIAL_TEST_DIRECTIONS` (modality-matched pools only,
    never a mixed image+text pool)
  - the train-vs-test query-dict refusal (M-BEIR restarts qid numbering per
    split; looking up test qids in the train dict silently returns unrelated
    train-split queries with the same-looking qid)
  - the per-direction candidate/query modality assertions
  - the 100%-coverage assertions for both candidates and queries
  - the final top-k / Recall@1/5/10 computation and printing

This answers Yubao's "how does this compare to other retrieval baselines"
question: on the exact same corrected, official COCO test pools already used
for the real pipeline's reference numbers (T->I 14.08%, I->T 9.44%,
Recall@1, trie + T5 beam search -- see
/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants/tables/table2_retrieval_performance.csv),
this script reports what plain dense CLIP-SF retrieval gets with zero
discretization loss and zero generative-decoding loss.

CPU-only by design (embeddings are ~2GB each on disk, score matmuls are at
most ~25K x 25K -- seconds on CPU). No GPU, no Slurm needed; run directly on
the headnode.

Usage:
  python dense_clip_sf_raw_baseline.py [--k_list 1 5 10]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data.preprocessing.utils import hash_did, hash_qid, load_jsonl_as_list  # noqa: E402

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
MBEIR_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"

# [Component 9] The official, held-out MSCOCO test split -- same query/pool
# convention as the real GENIUS pipeline's published reference numbers for
# these checkpoints (T->I 14.08%, I->T 9.44%, see table2_retrieval_performance.csv).
# All paths are relative to MBEIR_DIR except cand_dict, which is relative to
# GENIR_DIR (matches this project's existing extracted_embed/ convention).
# Copied verbatim from check_feasibility_criteria.py's OFFICIAL_TEST_DIRECTIONS.
OFFICIAL_TEST_DIRECTIONS = [
    dict(
        name="T2I_task0",
        query_jsonl="query/test/mbeir_mscoco_task0_test.jsonl",
        cand_pool_jsonl="cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl",
        cand_dict="extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_test_IT_dict.pt",
        expected_query_modality="text",
        expected_cand_modality="image",
    ),
    dict(
        name="I2T_task3",
        query_jsonl="query/test/mbeir_mscoco_task3_test.jsonl",
        cand_pool_jsonl="cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl",
        cand_dict="extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task3_test_IT_dict.pt",
        expected_query_modality="image",
        expected_cand_modality="text",
    ),
]

DEFAULT_TEST_QUERY_DICT = "extracted_embed/CLIP_SF/test/query_SFpretrained_instruction_IT_dict.pt"


def _load_cached(cache, path):
    if cache is None:
        return torch.load(path, map_location="cpu", weights_only=False)
    if path not in cache:
        cache[path] = torch.load(path, map_location="cpu", weights_only=False)
    return cache[path]


def _load_jsonl_cached(cache, path):
    if cache is None:
        return load_jsonl_as_list(path)
    key = ("jsonl", path)
    if key not in cache:
        cache[key] = load_jsonl_as_list(path)
    return cache[key]


def _fuse_and_normalize(d, idxs):
    """Raw, mask-weighted, L2-normalized CLIP-SF score-fusion embedding --
    the un-quantized counterpart of what RQ.inference()["quant"] would have
    reconstructed. On COCO each direction's pool/query side is effectively
    single-modality (T->I candidates are pure images, I->T candidates are
    pure text, and vice-versa for queries) so this reduces to a plain
    normalized CLIP-SF embedding in practice -- but the general mask-weighted
    fusion formula is implemented (not special-cased to COCO's pattern) so it
    stays correct if a mixed-modality row ever appears.
    """
    emb = d["img"][idxs].float() * d["img_mask"][idxs].float().unsqueeze(-1) \
        + d["text"][idxs].float() * d["text_mask"][idxs].float().unsqueeze(-1)
    return F.normalize(emb, dim=-1)


@torch.no_grad()
def dense_clip_sf_raw_baseline(genir_dir, mbeir_data_dir,
                                query_dict_rel=DEFAULT_TEST_QUERY_DICT,
                                directions=None, k_list=(1, 5, 10),
                                data_cache=None, force_query_dict=False):
    """Raw CLIP-SF dense-retrieval baseline: brute-force cosine similarity
    over mask-weighted, L2-normalized CLIP-SF score-fusion embeddings, with
    NO residual quantization and NO T5 generation anywhere in the pipeline.

    Scores each direction in `directions` (default OFFICIAL_TEST_DIRECTIONS)
    against its OWN modality-matched candidate pool -- never a mixed pool --
    using the official, held-out MSCOCO test queries. Every query/candidate's
    modality is asserted against what that direction expects; a mismatch
    means the wrong file was passed and this raises loudly rather than
    silently mixing modalities.

    `query_dict_rel` MUST be the official test-split query embedding dict
    (default `extracted_embed/CLIP_SF/test/query_SFpretrained_instruction_IT_dict.pt`)
    -- NEVER the train dict (`extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt`).
    M-BEIR restarts qid numbering per split, so the train dict would report
    false "100% coverage" while silently returning embeddings for unrelated
    train-split queries with the same qid. This function refuses to run
    against a `/train/` query dict unless `force_query_dict=True` is passed
    explicitly.

    `data_cache`: an optional dict, reused across calls so the ~2GB candidate
    dicts and the query dict are each loaded from disk only once per process.
    """
    if directions is None:
        directions = OFFICIAL_TEST_DIRECTIONS

    def resolve_genir(p):
        return p if os.path.isabs(p) else os.path.join(genir_dir, p)

    def resolve_mbeir(p):
        return p if os.path.isabs(p) else os.path.join(mbeir_data_dir, p)

    query_dict_path = resolve_genir(query_dict_rel)
    if "/train/" in query_dict_rel.replace("\\", "/") and not force_query_dict:
        raise ValueError(
            f"Refusing to score against a train-split query embedding dict ({query_dict_rel}). "
            "M-BEIR restarts qid numbering per split -- looking up official test qids in the "
            "train dict silently returns unrelated train-split queries with the same-looking "
            "qid. Pass force_query_dict=True only if you have deliberately verified this is safe."
        )

    query_dict = _load_cached(data_cache, query_dict_path)
    qid_to_index = query_dict["id_to_index"]

    results = {}
    for d in directions:
        cand_data = _load_jsonl_cached(data_cache, resolve_mbeir(d["cand_pool_jsonl"]))
        bad_cand_modality = [c["did"] for c in cand_data if c.get("modality") != d["expected_cand_modality"]]
        if bad_cand_modality:
            raise ValueError(
                f"[{d['name']}] {len(bad_cand_modality)} candidates in {d['cand_pool_jsonl']} "
                f"do not have modality=='{d['expected_cand_modality']}' -- wrong pool file, "
                "or a mixed pool slipped back in."
            )

        cand_dict = _load_cached(data_cache, resolve_genir(d["cand_dict"]))
        did_to_index = cand_dict["id_to_index"]
        n_covered = sum(1 for c in cand_data if hash_did(c["did"]) in did_to_index)
        if n_covered != len(cand_data):
            raise ValueError(
                f"[{d['name']}] candidate coverage {n_covered}/{len(cand_data)} < 100% -- "
                "the official test pool should be fully covered by its own extracted dict; "
                "this means the wrong dict was passed."
            )

        cand_h_dids = [hash_did(c["did"]) for c in cand_data]
        c_idxs = torch.tensor([did_to_index[h] for h in cand_h_dids], dtype=torch.long)
        cand_emb = _fuse_and_normalize(cand_dict, c_idxs)

        query_data = _load_jsonl_cached(data_cache, resolve_mbeir(d["query_jsonl"]))
        bad_query_modality = [q["qid"] for q in query_data if q.get("query_modality") != d["expected_query_modality"]]
        if bad_query_modality:
            raise ValueError(
                f"[{d['name']}] {len(bad_query_modality)} queries in {d['query_jsonl']} "
                f"do not have query_modality=='{d['expected_query_modality']}'."
            )

        h_qids = [hash_qid(q["qid"]) for q in query_data]
        n_qcov = sum(1 for h in h_qids if h in qid_to_index)
        if n_qcov != len(h_qids):
            raise ValueError(
                f"[{d['name']}] query coverage {n_qcov}/{len(h_qids)} < 100% in {query_dict_rel} -- "
                "the official test queries should be fully covered; wrong query dict passed, "
                "or the extraction wasn't (re)run."
            )
        q_idxs = torch.tensor([qid_to_index[h] for h in h_qids], dtype=torch.long)
        q_emb = _fuse_and_normalize(query_dict, q_idxs)

        sims = q_emb @ cand_emb.T  # [n_q, n_cand]
        max_k = max(k_list)
        topk_idx = sims.topk(k=min(max_k, sims.shape[1]), dim=-1).indices.cpu()

        recalls = {k: 0 for k in k_list}
        for i, q in enumerate(query_data):
            pos_h_dids = {hash_did(dd) for dd in q.get("pos_cand_list", [])}
            retrieved_h_dids = [cand_h_dids[j] for j in topk_idx[i].tolist()]
            for k in k_list:
                if any(h in pos_h_dids for h in retrieved_h_dids[:k]):
                    recalls[k] += 1
        n = len(query_data)
        print(f"  [{d['name']}] n_query={n}  n_cand={len(cand_data)}  "
              f"hits@1={recalls[1]}  R@1={100.0*recalls[1]/n:.2f}%  "
              f"R@5={100.0*recalls[5]/n:.2f}%  R@10={100.0*recalls[10]/n:.2f}%")
        results[d["name"]] = {k: recalls[k] / n for k in k_list}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--genir_dir", default=GENIR_DIR)
    parser.add_argument("--mbeir_data_dir", default=MBEIR_DIR)
    parser.add_argument("--test_query_dict_path", default=DEFAULT_TEST_QUERY_DICT)
    args = parser.parse_args()

    print("Raw CLIP-SF dense-retrieval baseline (no RQ, no T5) on the official "
          "held-out MSCOCO test split:")
    results = dense_clip_sf_raw_baseline(
        genir_dir=args.genir_dir, mbeir_data_dir=args.mbeir_data_dir,
        query_dict_rel=args.test_query_dict_path, k_list=tuple(args.k_list),
    )

    print("\n=== Summary table (markdown) ===")
    print("| Direction | R@1 | R@5 | R@10 |")
    print("|---|---|---|---|")
    for direction, recalls in results.items():
        print(f"| {direction} | {100*recalls[1]:.2f}% | {100*recalls[5]:.2f}% | {100*recalls[10]:.2f}% |")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
