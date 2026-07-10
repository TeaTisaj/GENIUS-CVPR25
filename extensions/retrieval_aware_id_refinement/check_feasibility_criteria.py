"""
Component 7 (retrieval-aware RQ refinement feasibility study) -- diagnostics
script. Parses a feasibility-study training log (produced by
`train_one_epoch_retrieval_aware`, engine.py) and an optional pair of
checkpoints (before/after refinement), and reports the mentor's point-8
feasibility checks, corrected/expanded per the plan's Component 7:

  1. Hard negatives meaningful: the hard-negative cell's L_ret trajectory
     compared against the random-negatives control's (--control_log_path),
     not a cosine-similarity spot-check in isolation.
  2. L_ret decreases over training: printed per epoch from the log.
  3. lambda_l behavior: this implementation used Component 4's "preferred"
     validation-guided-perturbation option (see RQ.__init__'s docstring in
     residual_quantization.py), so this reports the LOGGED PERTURBATION
     TRAJECTORY and flags whether it is still moving at the last epoch --
     explicitly NOT calling this "learned via backprop" (it never was, and
     could not be, per that docstring's causal-mechanism explanation).
  4. Gradient conflicts occur: fraction of (level, step) pairs where
     projection actually triggered (grad_conflict_frac, logged per epoch,
     averaged from the per-step per-level cosine-sign check in engine.py).
  5. Dense Recall@K probe: before/after, using the refined RQ's `quant`
     output directly (no beam search, no T5) -- compared against the
     continued-training-no-new-losses control's checkpoint
     (--no_new_losses_ckpt), not just against the original teacher.

Per [[feedback_slurm_completed_not_success]]: this script ALWAYS checks the
log for a Python traceback first and refuses to report criteria 1-4 from a
log that contains one (Slurm COMPLETED does not mean the inner Python
succeeded).

[Component 9, 2026-07-08 -- Criterion 5 methodology correction] The original
version of `dense_recall_at_k_probe` scored BOTH directions against one
mixed image+text candidate pool, using validation queries carved from
COCO's TRAINING query split. A supervisor's skepticism about an implausible
"97.7% T->I / 0.0% I->T" starting-checkpoint result led to an audit that
found no code bug, but two real methodology problems: (1) mixing modalities
in the scored pool, when the real GENIUS pipeline scores each direction
against a modality-matched pool only (image-only for T->I, text-only for
I->T); (2) those train-derived validation queries are largely memorized by
Stage-1 for the text (T->I) direction specifically (the teacher checkpoint
was contrastively trained on ~100K of them), while the image (I->T)
direction had zero training exposure -- an apples-to-oranges comparison.
`dense_recall_at_k_probe` now scores exclusively against the official,
held-out MSCOCO test split (`OFFICIAL_TEST_DIRECTIONS` below), matching the
query/pool convention used by this checkpoint's already-published real-
pipeline reference numbers (T->I 14.08%, I->T 9.44%, trie + T5 beam search --
still a different metric from this dense-KNN proxy, so exact agreement is
not expected, but the two numbers should no longer show a wild asymmetry).
The old mixed-pool/train-query behavior is entirely removed, not kept as an
option -- see ~/.claude/plans/so-this-is-the-swirling-quiche.md, Component 9,
for the full investigation.

Usage (process-metric criteria only, from logs):
  python check_feasibility_criteria.py \
      --log_path <path to this cell's train.log> \
      --control_log_path <path to the random-negatives control's train.log>

Usage (adding the dense Recall@K probe, criterion 5):
  python check_feasibility_criteria.py \
      --log_path ... --control_log_path ... \
      --genir_dir <repo> --mbeir_data_dir <mbeir_data_dir> \
      --before_ckpt <warm-start/teacher checkpoint> \
      --after_ckpt <this cell's final trained checkpoint> \
      --no_new_losses_ckpt <continued-training-no-new-losses control's final checkpoint>
  (query/pool sources default to OFFICIAL_TEST_DIRECTIONS + the official test
  query embedding dict extracted at extracted_embed/CLIP_SF/test/ -- override
  the latter with --test_query_dict_path only if you know what you're doing;
  to change the directions/pools themselves, edit OFFICIAL_TEST_DIRECTIONS or
  pass a custom `directions=` list to dense_recall_at_k_probe directly.)
"""
import argparse
import ast
import os
import re
import sys
import torch
import torch.nn.functional as F

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data.preprocessing.utils import hash_did, hash_qid, load_jsonl_as_list  # noqa: E402
from models.residual_quantization.residual_quantization import RQ  # noqa: E402

LAMBDA_LINE = re.compile(r"\[step (\d+)\] lambda_l=(\[.*\])")
AVG_STATS_LINE = re.compile(r"^Averaged stats:\s*(.*)$")
KV_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*): (-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def log_has_traceback(log_path):
    """[[feedback_slurm_completed_not_success]]: Slurm COMPLETED doesn't mean
    the inner Python succeeded -- always grep the log for a real traceback
    before trusting anything else in it."""
    with open(log_path) as f:
        text = f.read()
    return "Traceback (most recent call last)" in text


def parse_lambda_trajectory(log_path):
    steps, lambdas = [], []
    with open(log_path) as f:
        for line in f:
            m = LAMBDA_LINE.search(line)
            if m:
                steps.append(int(m.group(1)))
                lambdas.append(ast.literal_eval(m.group(2)))
    return steps, lambdas


def parse_epoch_averages(log_path):
    epochs = []
    with open(log_path) as f:
        for line in f:
            m = AVG_STATS_LINE.match(line.strip())
            if m:
                kvs = {k: float(v) for k, v in KV_PATTERN.findall(m.group(1))}
                epochs.append(kvs)
    return epochs


def report_criterion_1_hard_vs_random(main_epochs, control_epochs):
    print("\n[Criterion 1] Hard negatives meaningful (hard-cell L_ret vs. random-negatives control L_ret):")
    if not main_epochs or not control_epochs:
        print("  INSUFFICIENT DATA -- one of the two logs has no parsed epoch-average lines.")
        return
    main_final = main_epochs[-1].get("L_ret")
    control_final = control_epochs[-1].get("L_ret")
    print(f"  main cell final-epoch L_ret:    {main_final}")
    print(f"  random-control final-epoch L_ret: {control_final}")
    if main_final is not None and control_final is not None:
        meaningful = main_final < control_final
        print(f"  -> hard negatives {'ARE' if meaningful else 'are NOT'} more effective than random "
              f"negatives on this objective (lower final L_ret = more effective).")


def report_criterion_2_l_ret_decreases(main_epochs):
    print("\n[Criterion 2] L_ret decreases over training:")
    l_ret_traj = [e.get("L_ret") for e in main_epochs if "L_ret" in e]
    print(f"  L_ret per epoch: {l_ret_traj}")
    if len(l_ret_traj) >= 2:
        print(f"  first={l_ret_traj[0]:.5f}  last={l_ret_traj[-1]:.5f}  "
              f"decreased={l_ret_traj[-1] < l_ret_traj[0]}")
    else:
        print("  INSUFFICIENT DATA (need >= 2 epochs).")


def report_criterion_3_lambda_l(log_path):
    print("\n[Criterion 3] lambda_l behavior:")
    print("  NOTE: this implementation uses Component 4's 'preferred, cheaper "
          "alternative' -- a VALIDATION-GUIDED PERTURBATION rule, not literal "
          "backprop (see RQ.__init__'s docstring in residual_quantization.py for "
          "why: a_l has no direct effect on L_ret within a single frozen-encoder "
          "forward pass). Reporting the logged perturbation trajectory below, "
          "NOT a 'learned via backprop' claim.")
    steps, lambdas = parse_lambda_trajectory(log_path)
    if not lambdas:
        print("  INSUFFICIENT DATA -- no '[step N] lambda_l=[...]' lines found.")
        return
    print(f"  {len(lambdas)} logged lambda_l snapshots, first={lambdas[0]}, last={lambdas[-1]}")
    if len(lambdas) >= 5:
        import statistics
        last_5 = lambdas[-5:]
        per_level_std = [statistics.pstdev([snap[l] for snap in last_5]) for l in range(len(last_5[0]))]
        print(f"  per-level stddev over last 5 snapshots: {[round(s, 5) for s in per_level_std]}")
        still_moving = any(s > 1e-3 for s in per_level_std)
        print(f"  still moving at the end of training: {still_moving}")


def report_criterion_4_grad_conflicts(main_epochs):
    print("\n[Criterion 4] Gradient conflicts occur in practice:")
    conflict_traj = [e.get("grad_conflict_frac") for e in main_epochs if "grad_conflict_frac" in e]
    cos_traj = [e.get("grad_cos_mean") for e in main_epochs if "grad_cos_mean" in e]
    print(f"  grad_conflict_frac per epoch: {conflict_traj}")
    print(f"  grad_cos_mean per epoch:      {cos_traj}")
    if conflict_traj:
        mean_conflict = sum(conflict_traj) / len(conflict_traj)
        print(f"  mean conflict fraction across epochs: {mean_conflict:.4f}")
        print(f"  conflicts occur in practice: {mean_conflict > 0.0}")
    else:
        print("  INSUFFICIENT DATA.")


# [Component 9] The official, held-out MSCOCO test split -- same query/pool
# convention as the real GENIUS pipeline's published reference numbers for
# these checkpoints (T->I 14.08%, I->T 9.44%, see table2_retrieval_performance.csv).
# All paths are relative to mbeir_data_dir except cand_dict, which is relative
# to genir_dir (matches this project's existing extracted_embed/ convention).
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


@torch.no_grad()
def dense_recall_at_k_probe(genir_dir, mbeir_data_dir, ckpt_path,
                             query_dict_rel=DEFAULT_TEST_QUERY_DICT,
                             directions=None, k_list=(1, 5, 10), device="cpu",
                             data_cache=None, force_query_dict=False):
    """Criterion 5: dense-embedding Recall@K probe. Uses the RQ's `quant`
    (post-quantization reconstruction) output directly -- no beam search, no
    trie, no T5 -- scoring by brute-force cosine similarity, mirroring the
    scoring logic in src/common/mbeir_generative_retriever.py's
    `compute_cosine_similarities`/`retrieve_and_rank_for_query` (simplified
    here since every candidate is eligible -- no code-hash-map restriction --
    which IS the point of this cheap proxy).

    [Component 9] Scores each direction in `directions` (default
    OFFICIAL_TEST_DIRECTIONS) against its OWN modality-matched candidate
    pool -- never a mixed pool -- using the official, held-out MSCOCO test
    queries. Every query/candidate's modality is asserted against what that
    direction expects; a mismatch means the wrong file was passed and this
    raises loudly rather than silently mixing modalities again.

    `query_dict_rel` MUST be the official test-split query embedding dict
    (default `extracted_embed/CLIP_SF/test/query_SFpretrained_instruction_IT_dict.pt`,
    produced by Component 9 Step 2) -- NEVER the train dict
    (`extracted_embed/CLIP_SF/train/query_SFpretrained_instruction_IT_dict.pt`).
    M-BEIR restarts qid numbering per split, so the train dict would report
    false "100% coverage" while silently returning embeddings for unrelated
    train-split queries with the same qid. This function refuses to run
    against a `/train/` query dict unless `force_query_dict=True` is passed
    explicitly.

    `data_cache`: an optional dict, reused across calls (e.g. before/after/
    control, or across many checkpoints) so the ~2.2GB candidate dicts and
    the query dict are each loaded from disk only once per process.
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
            "qid (see Component 9's 'qid-collision trap' in "
            "~/.claude/plans/so-this-is-the-swirling-quiche.md). "
            "Pass force_query_dict=True only if you have deliberately verified this is safe."
        )

    ckpt = torch.load(resolve_genir(ckpt_path), map_location="cpu", weights_only=False)
    model = RQ(config=ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)

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
        cand_out = model.inference(
            cand_dict["img"][c_idxs].float().to(device), cand_dict["text"][c_idxs].float().to(device),
            cand_dict["img_mask"][c_idxs].float().unsqueeze(-1).to(device),
            cand_dict["text_mask"][c_idxs].float().unsqueeze(-1).to(device),
        )
        cand_quant = F.normalize(cand_out["quant"], dim=-1)

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
                "or Step 2's extraction wasn't (re)run."
            )
        q_idxs = torch.tensor([qid_to_index[h] for h in h_qids], dtype=torch.long)
        q_out = model.inference(
            query_dict["img"][q_idxs].float().to(device), query_dict["text"][q_idxs].float().to(device),
            query_dict["img_mask"][q_idxs].float().unsqueeze(-1).to(device),
            query_dict["text_mask"][q_idxs].float().unsqueeze(-1).to(device),
        )
        q_quant = F.normalize(q_out["quant"], dim=-1)

        sims = q_quant @ cand_quant.T  # [n_q, n_cand]
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


def report_criterion_5_recall(args):
    print("\n[Criterion 5] Dense Recall@K probe on the official held-out MSCOCO test "
          "split (before vs. after refinement, compared against the "
          "continued-training-no-new-losses control):")
    if not (args.before_ckpt and args.after_ckpt):
        print("  SKIPPED -- --before_ckpt/--after_ckpt not provided.")
        return

    data_cache = {}
    common_kwargs = dict(
        genir_dir=args.genir_dir, mbeir_data_dir=args.mbeir_data_dir,
        query_dict_rel=args.test_query_dict_path, data_cache=data_cache,
    )
    print("  --- BEFORE (teacher/warm-start checkpoint) ---")
    before = dense_recall_at_k_probe(ckpt_path=args.before_ckpt, **common_kwargs)
    print("  --- AFTER (this cell's trained checkpoint) ---")
    after = dense_recall_at_k_probe(ckpt_path=args.after_ckpt, **common_kwargs)
    print(f"  BEFORE: {before}")
    print(f"  AFTER:  {after}")

    if args.no_new_losses_ckpt:
        print("  --- CONTROL (continued-training-no-new-losses) ---")
        control = dense_recall_at_k_probe(ckpt_path=args.no_new_losses_ckpt, **common_kwargs)
        print(f"  CONTROL: {control}")
    else:
        print("  --no_new_losses_ckpt not provided; skipping the control comparison "
              "(criterion 5 is only fully satisfied when this is included).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True, help="This cell's train.log.")
    parser.add_argument("--control_log_path", default=None,
                         help="The random-negatives control's train.log (criterion 1).")
    parser.add_argument("--genir_dir", default=None)
    parser.add_argument("--mbeir_data_dir", default=None)
    parser.add_argument("--before_ckpt", default=None)
    parser.add_argument("--after_ckpt", default=None)
    parser.add_argument("--no_new_losses_ckpt", default=None)
    parser.add_argument("--test_query_dict_path", default=DEFAULT_TEST_QUERY_DICT,
                         help="[Component 9] Official held-out test-split query embedding dict "
                              "(extracted_embed/CLIP_SF/test/..., produced by Step 2's GPU "
                              "extraction job) -- NEVER the train dict, see "
                              "dense_recall_at_k_probe's docstring for why.")
    args = parser.parse_args()

    print(f"Checking log: {args.log_path}")
    if log_has_traceback(args.log_path):
        print("FATAL: this log contains a Python traceback. Slurm COMPLETED does "
              "not mean the inner Python succeeded -- fix the underlying error "
              "before trusting ANY of the criteria below "
              "[[feedback_slurm_completed_not_success]].")
        sys.exit(1)

    main_epochs = parse_epoch_averages(args.log_path)
    control_epochs = parse_epoch_averages(args.control_log_path) if args.control_log_path else []
    if args.control_log_path:
        if log_has_traceback(args.control_log_path):
            print("FATAL: the control log contains a Python traceback.")
            sys.exit(1)

    report_criterion_1_hard_vs_random(main_epochs, control_epochs)
    report_criterion_2_l_ret_decreases(main_epochs)
    report_criterion_3_lambda_l(args.log_path)
    report_criterion_4_grad_conflicts(main_epochs)
    report_criterion_5_recall(args)


if __name__ == "__main__":
    main()
