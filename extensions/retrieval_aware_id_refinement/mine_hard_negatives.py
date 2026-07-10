"""
Component 1 (retrieval-aware RQ refinement feasibility study, see
`.claude/plans/so-this-is-the-swirling-quiche.md`) -- offline hard-negative
mining against a frozen Stage-1 RQ "teacher" checkpoint.

**Deliberate simplification, stated explicitly per the plan's Fable-fix note on
this component:** this mines negatives in the teacher's *embedding space*
(cosine similarity between `RQ.inference(...)['encode']` vectors), not by
running the frozen *generative retriever's* actual constrained beam search over
the trie (which is what the mentor's point 2 literally asks for -- "retrieve
top-ranked non-relevant candidates" from the retriever's own ranking). Embedding
nearest-neighbor negatives are cheaper (no beam search / trie build) but are a
materially different, likely easier, negative distribution than what the
retriever actually confuses at inference time. This is acceptable for a
*feasibility* pass but must not be silently presented as the literal point-2
mining procedure -- see the plan file's Component 1 section for the full
discussion, including why neither CocoVanilla nor CocoStrong perfectly satisfies
"reasonable retrieval performance in both directions" (the COCO I->T collapse).

Two modes:
  --mode hard    (default) frozen-teacher cosine-similarity nearest-neighbor mining.
  --mode random  the required random-negatives control (Component 7): same K per
                 query, sampled uniformly at random from the COCO candidate
                 pool (excluding positives), no teacher/embeddings needed at all.

**[Correction, found during implementation]** An earlier draft of this script
(and the plan it implements) used
`cand_pool/global/mbeir_coco_fashioniq_train_cand_pool.jsonl` as the candidate
pool -- that file is a *different, non-matching* candidate universe from what
this project's actual Stage-1 training configs use: it has 1,427,358
MSCOCO-tagged entries, of which the embed dict (`pool_SFpretrained_IT_dict.pt`)
only covers ~43% (confirmed by direct `hash_did` lookup), which crashes
mining with a `KeyError` partway through a real run. The actual Stage-1
configs (`inbatch_coco_{vanilla,strong}.yaml`) use
`cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl` instead (713,679
entries, already COCO-only -- no dataset-id filtering needed), which the same
embed dict covers at 86.2% (615,538/713,679) -- a normal/expected coverage
level for this project, not a new bug. This script now defaults to the local
pool and filters the candidate list to the intersection with
`pool_dict['id_to_index']`'s keys, printing an explicit coverage report
rather than assuming every JSONL entry has an embedding.

Usage (hard mode):
  python mine_hard_negatives.py \
      --genir_dir <repo> --mbeir_data_dir <mbeir_data_dir> \
      --teacher_ckpt <path to rq_clip_large_epoch_140.pth> \
      --query_jsonl_path query/union_train/mbeir_coco_only_train_feasibility20k.jsonl \
      --out_path extensions/retrieval_aware_id_refinement/hard_negatives_coco_train_vanilla.pt \
      --cell_name vanilla_teacher

Usage (random control):
  python mine_hard_negatives.py --mode random --genir_dir <repo> --mbeir_data_dir <mbeir_data_dir> \
      --query_jsonl_path query/union_train/mbeir_coco_only_train_feasibility20k.jsonl \
      --out_path extensions/retrieval_aware_id_refinement/hard_negatives_coco_train_random.pt \
      --cell_name random_control
"""
import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

# Allow running this script directly (PYTHONPATH may only contain src/).
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data.preprocessing.utils import (  # noqa: E402
    hash_did,
    hash_qid,
    load_jsonl_as_list,
)
from models.residual_quantization.residual_quantization import RQ  # noqa: E402


def resolve(genir_dir, rel_or_abs_path):
    return rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(genir_dir, rel_or_abs_path)


def filter_to_embed_dict_coverage(cand_data, did_to_index, cell_name):
    """Filter cand_data (list of cand_pool JSONL entries) down to the
    intersection with pool_dict['id_to_index']'s keys, printing an explicit
    coverage report. [Correction, found during implementation] Never assume
    every JSONL entry has a matching embedding -- confirmed by direct lookup
    that even the correct local COCO pool (`cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl`,
    713,679 entries) is only 86.2% covered by `pool_SFpretrained_IT_dict.pt`
    (615,538/713,679), a normal/expected gap for this project, not a bug to
    silently work around by crashing."""
    covered = [c for c in cand_data if hash_did(c["did"]) in did_to_index]
    print(f"[{cell_name}] Candidate embed-dict coverage: {len(covered)}/{len(cand_data)} "
          f"({100.0 * len(covered) / max(len(cand_data), 1):.1f}%) -- mining only over the covered subset.")
    return covered


def load_teacher(genir_dir, teacher_ckpt_path, device):
    ckpt_path = resolve(genir_dir, teacher_ckpt_path)
    assert os.path.exists(ckpt_path), f"Teacher checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RQ(config=ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def encode_all(model, ids, id_to_index, emb_dict, device, chunk_size, desc):
    """ids: list of hashed ids (int). Returns encode matrix [len(ids), dim] (float32, on CPU)."""
    out_chunks = []
    n = len(ids)
    for start in range(0, n, chunk_size):
        chunk_ids = ids[start:start + chunk_size]
        idxs = torch.tensor([id_to_index[i] for i in chunk_ids], dtype=torch.long)
        img_emb = emb_dict["img"][idxs].float().to(device)
        txt_emb = emb_dict["text"][idxs].float().to(device)
        img_mask = emb_dict["img_mask"][idxs].float().unsqueeze(-1).to(device)
        txt_mask = emb_dict["text_mask"][idxs].float().unsqueeze(-1).to(device)
        out = model.inference(img_emb, txt_emb, img_mask, txt_mask)
        encode = out["encode"]
        if encode.dim() == 1:
            encode = encode.unsqueeze(0)
        out_chunks.append(encode.cpu())
        if (start // chunk_size) % 10 == 0:
            print(f"  [{desc}] encoded {min(start + chunk_size, n)}/{n}")
    return torch.cat(out_chunks, dim=0)


def mine_hard(args, device):
    query_data = load_jsonl_as_list(resolve(args.mbeir_data_dir, args.query_jsonl_path))
    cand_data = load_jsonl_as_list(resolve(args.mbeir_data_dir, args.cand_pool_jsonl_path))
    print(f"Candidate pool ({args.cand_pool_jsonl_path}): {len(cand_data)} entries "
          f"(already COCO-only -- no dataset-id filtering needed).")

    pool_dict = torch.load(resolve(args.genir_dir, args.pool_dict_path), map_location="cpu", weights_only=False)
    query_dict = torch.load(resolve(args.genir_dir, args.query_dict_path), map_location="cpu", weights_only=False)
    did_to_index = pool_dict["id_to_index"]
    qid_to_index = query_dict["id_to_index"]

    cand_data_filtered = filter_to_embed_dict_coverage(cand_data, did_to_index, args.cell_name)

    model = load_teacher(args.genir_dir, args.teacher_ckpt, device)

    cand_h_dids = [hash_did(c["did"]) for c in cand_data_filtered]
    cand_encode = encode_all(model, cand_h_dids, did_to_index, pool_dict, device, args.chunk_size, "cand")
    cand_encode = F.normalize(cand_encode, dim=-1)

    q_h_qids = [hash_qid(q["qid"]) for q in query_data]
    q_encode = encode_all(model, q_h_qids, qid_to_index, query_dict, device, args.chunk_size, "query")
    q_encode = F.normalize(q_encode, dim=-1)

    cand_encode_gpu = cand_encode.to(device)
    hard_neg_dict = {}
    all_sims_of_topk = []

    for start in range(0, len(query_data), args.query_chunk_size):
        chunk = query_data[start:start + args.query_chunk_size]
        chunk_q_encode = q_encode[start:start + args.query_chunk_size].to(device)
        sims = chunk_q_encode @ cand_encode_gpu.T  # [chunk, n_cand]

        for i, q in enumerate(chunk):
            h_qid = hash_qid(q["qid"])
            pos_h_dids = {hash_did(d) for d in q.get("pos_cand_list", [])}
            row = sims[i].clone()
            for j, h_did in enumerate(cand_h_dids):
                if h_did in pos_h_dids:
                    row[j] = -1e9  # exclude positives from negative mining
            topk_vals, topk_idx = torch.topk(row, k=min(args.k, row.shape[0]))
            hard_neg_dict[h_qid] = [cand_h_dids[j] for j in topk_idx.tolist()]
            all_sims_of_topk.extend(topk_vals.tolist())

        if (start // args.query_chunk_size) % 5 == 0:
            print(f"  mined negatives for {min(start + args.query_chunk_size, len(query_data))}/{len(query_data)} queries")

    # Sanity check (Component 1 / Verification item 4): mean/median cosine similarity
    # of mined hard negatives vs. the positive, plus spot-printed human-readable triples.
    import statistics
    print(f"\n[{args.cell_name}] Hard-negative cosine similarity to query: "
          f"mean={statistics.mean(all_sims_of_topk):.4f} median={statistics.median(all_sims_of_topk):.4f}")

    did_to_entry = {c["did"]: c for c in cand_data_filtered}
    rng = random.Random(2023)
    sample_qs = rng.sample(query_data, min(5, len(query_data)))
    print(f"\n[{args.cell_name}] 5 random (query, positive, top-3 hard-negative) text triples:")
    for q in sample_qs:
        h_qid = hash_qid(q["qid"])
        pos_did = q.get("pos_cand_list", [None])[0]
        pos_txt = did_to_entry.get(pos_did, {}).get("txt", "<no local entry>")
        neg_h_dids = hard_neg_dict[h_qid][:3]
        h_did_to_did = {hash_did(c["did"]): c["did"] for c in cand_data_filtered}
        neg_txts = [did_to_entry.get(h_did_to_did.get(h), {}).get("txt", "?") for h in neg_h_dids]
        print(f"  query_txt={q.get('query_txt', '')!r}")
        print(f"    positive_txt={pos_txt!r}")
        for t in neg_txts:
            print(f"    hard_neg_txt={t!r}")

    return hard_neg_dict


def mine_random(args):
    query_data = load_jsonl_as_list(resolve(args.mbeir_data_dir, args.query_jsonl_path))
    cand_data = load_jsonl_as_list(resolve(args.mbeir_data_dir, args.cand_pool_jsonl_path))
    print(f"Candidate pool ({args.cand_pool_jsonl_path}): {len(cand_data)} entries "
          f"(already COCO-only -- no dataset-id filtering needed).")

    # Coverage filtering is REQUIRED here too, not just cosmetic: downstream
    # training (HardNegativeAugmentedDataset.__getitem__) looks up each mined
    # negative's h_did in this SAME pool_dict's did_to_index -- a negative
    # sampled from an uncovered did would KeyError at training time, not just
    # here. Also ensures the random control draws from the identical candidate
    # universe as the hard-negative cells, for a fair comparison.
    pool_dict = torch.load(resolve(args.genir_dir, args.pool_dict_path), map_location="cpu", weights_only=False)
    did_to_index = pool_dict["id_to_index"]
    cand_data_filtered = filter_to_embed_dict_coverage(cand_data, did_to_index, args.cell_name)
    cand_h_dids_all = [hash_did(c["did"]) for c in cand_data_filtered]

    rng = random.Random(args.seed)
    hard_neg_dict = {}
    for q in query_data:
        h_qid = hash_qid(q["qid"])
        pos_h_dids = {hash_did(d) for d in q.get("pos_cand_list", [])}
        candidates = [h for h in cand_h_dids_all if h not in pos_h_dids]
        k = min(args.k, len(candidates))
        hard_neg_dict[h_qid] = rng.sample(candidates, k)

    print(f"[{args.cell_name}] Random-negative control: sampled {args.k} random negatives "
          f"(seed={args.seed}) for {len(query_data)} queries from a pool of "
          f"{len(cand_h_dids_all)} COCO candidates.")
    return hard_neg_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", required=True)
    parser.add_argument("--mode", choices=["hard", "random"], default="hard")
    parser.add_argument("--teacher_ckpt", default=None, help="Required if --mode hard")
    parser.add_argument("--query_jsonl_path", required=True,
                         help="Relative to --mbeir_data_dir, e.g. the 20K feasibility subsample.")
    parser.add_argument("--cand_pool_jsonl_path", default="cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl",
                         help="[Correction, found during implementation] the LOCAL COCO pool "
                              "(713,679 entries, already COCO-only), matching the actual Stage-1 "
                              "training configs (inbatch_coco_{vanilla,strong}.yaml) -- NOT the "
                              "global union pool, which the embed dict only covers at ~43%.")
    parser.add_argument("--pool_dict_path", default="extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt")
    parser.add_argument("--query_dict_path", default="extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt")
    parser.add_argument("--k", type=int, default=20, help="Number of hard negatives per query.")
    parser.add_argument("--chunk_size", type=int, default=8192, help="Encoding chunk size.")
    parser.add_argument("--query_chunk_size", type=int, default=2048, help="Similarity-matmul chunk size (query side).")
    parser.add_argument("--seed", type=int, default=2023, help="Only used by --mode random.")
    parser.add_argument("--out_path", required=True, help="Where to save the {h_qid: [h_did,...]} .pt sidecar.")
    parser.add_argument("--cell_name", required=True, help="Label for this mining run, used only in printed logs.")
    args = parser.parse_args()

    if args.mode == "hard":
        assert args.teacher_ckpt is not None, "--teacher_ckpt is required for --mode hard"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hard_neg_dict = mine_hard(args, device)
    else:
        hard_neg_dict = mine_random(args)

    out_path = resolve(args.genir_dir, args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(hard_neg_dict, out_path)
    print(f"\nSaved {len(hard_neg_dict)} queries' hard negatives to {out_path}")


if __name__ == "__main__":
    main()
