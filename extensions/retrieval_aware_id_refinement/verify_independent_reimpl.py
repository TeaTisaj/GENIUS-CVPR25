"""
Component 9, Step 6.2 -- independent re-implementation of the corrected
dense Recall@K probe (`dense_recall_at_k_probe` in
check_feasibility_criteria.py), sharing nothing with it except the `RQ`
model class and the trivial, deterministic `hash_did`/`hash_qid` encoding
(dataset_id * upper_bound + within_id -- not scoring logic, not worth
re-deriving). Everything else -- jsonl loading, modality checks, top-k
selection, positive matching, recall aggregation -- is written from scratch
using plain Python + numpy instead of the original's torch vectorized path,
specifically to catch a bug that would be replicated by copy-pasting the
same (possibly wrong) code twice.

Must agree with check_feasibility_criteria.py's numbers to within +/-0.1pp
on R@1/R@5/R@10 for every checkpoint/direction it's run against, per the
Component 9 plan's Step 6.2 verification gate.

Usage:
  python verify_independent_reimpl.py --label Teacher_CocoVanilla \
      --ckpt /home/tcetoje/GENIUS-CVPR25/checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.residual_quantization.residual_quantization import RQ  # noqa: E402

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
MBEIR_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"

QUERY_DICT_PATH = os.path.join(GENIR_DIR, "extracted_embed/CLIP_SF/test/query_SFpretrained_instruction_IT_dict.pt")

# Must match src/data/preprocessing/utils.py's hash_qid/hash_did constants
# exactly (verified by reading that file directly) -- these are shared,
# trivial encoding constants, not part of the scoring logic under test here.
DATASET_QUERY_NUM_UPPER_BOUND = 500_000
DATASET_CAN_NUM_UPPER_BOUND = 10_000_000


def my_hash(id_str, upper_bound):
    a, b = id_str.split(":")
    return int(a) * upper_bound + int(b)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


DIRECTIONS = [
    dict(
        name="T2I_task0",
        query_jsonl=os.path.join(MBEIR_DIR, "query/test/mbeir_mscoco_task0_test.jsonl"),
        cand_pool_jsonl=os.path.join(MBEIR_DIR, "cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl"),
        cand_dict=os.path.join(GENIR_DIR, "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_test_IT_dict.pt"),
        q_modality="text",
        c_modality="image",
    ),
    dict(
        name="I2T_task3",
        query_jsonl=os.path.join(MBEIR_DIR, "query/test/mbeir_mscoco_task3_test.jsonl"),
        cand_pool_jsonl=os.path.join(MBEIR_DIR, "cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl"),
        cand_dict=os.path.join(GENIR_DIR, "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task3_test_IT_dict.pt"),
        q_modality="image",
        c_modality="text",
    ),
]


def run_direction(model, d, query_dict):
    cand_rows = read_jsonl(d["cand_pool_jsonl"])
    for c in cand_rows:
        assert c["modality"] == d["c_modality"], f"bad candidate modality in {d['name']}"

    cand_dict = torch.load(d["cand_dict"], map_location="cpu", weights_only=False)
    did_to_index = cand_dict["id_to_index"]
    cand_h = [my_hash(c["did"], DATASET_CAN_NUM_UPPER_BOUND) for c in cand_rows]
    for h in cand_h:
        assert h in did_to_index, f"candidate not covered by dict in {d['name']}"
    c_idx = np.array([did_to_index[h] for h in cand_h], dtype=np.int64)

    with torch.no_grad():
        c_idx_t = torch.from_numpy(c_idx)
        cand_out = model.inference(
            cand_dict["img"][c_idx_t].float(), cand_dict["text"][c_idx_t].float(),
            cand_dict["img_mask"][c_idx_t].float().unsqueeze(-1),
            cand_dict["text_mask"][c_idx_t].float().unsqueeze(-1),
        )
    cand_vecs = cand_out["quant"].numpy().astype(np.float64)
    cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)

    query_rows = read_jsonl(d["query_jsonl"])
    for q in query_rows:
        assert q["query_modality"] == d["q_modality"], f"bad query modality in {d['name']}"

    qid_to_index = query_dict["id_to_index"]
    q_h = [my_hash(q["qid"], DATASET_QUERY_NUM_UPPER_BOUND) for q in query_rows]
    for h in q_h:
        assert h in qid_to_index, f"query not covered by query dict in {d['name']}"
    q_idx = np.array([qid_to_index[h] for h in q_h], dtype=np.int64)

    with torch.no_grad():
        q_idx_t = torch.from_numpy(q_idx)
        q_out = model.inference(
            query_dict["img"][q_idx_t].float(), query_dict["text"][q_idx_t].float(),
            query_dict["img_mask"][q_idx_t].float().unsqueeze(-1),
            query_dict["text_mask"][q_idx_t].float().unsqueeze(-1),
        )
    q_vecs = q_out["quant"].numpy().astype(np.float64)
    q_vecs = q_vecs / np.linalg.norm(q_vecs, axis=1, keepdims=True)

    sims = q_vecs @ cand_vecs.T  # [n_q, n_cand], plain numpy matmul

    k_list = (1, 5, 10)
    max_k = max(k_list)
    hits = {k: 0 for k in k_list}
    n = len(query_rows)
    for i, q in enumerate(query_rows):
        pos_h = {my_hash(pc, DATASET_CAN_NUM_UPPER_BOUND) for pc in q.get("pos_cand_list", [])}
        row = sims[i]
        top_idx = np.argpartition(-row, max_k - 1)[:max_k]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        retrieved_h = [cand_h[j] for j in top_idx.tolist()]
        for k in k_list:
            if any(h in pos_h for h in retrieved_h[:k]):
                hits[k] += 1

    recalls = {k: hits[k] / n for k in k_list}
    print(f"  [{d['name']}] n_query={n}  n_cand={len(cand_rows)}  hits@1={hits[1]}  "
          f"R@1={100*recalls[1]:.2f}%  R@5={100*recalls[5]:.2f}%  R@10={100*recalls[10]:.2f}%")
    return recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    print(f"\n=== [independent reimpl] {args.label} ===")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = RQ(config=ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    query_dict = torch.load(QUERY_DICT_PATH, map_location="cpu", weights_only=False)

    for d in DIRECTIONS:
        run_direction(model, d, query_dict)


if __name__ == "__main__":
    main()
