"""
Component 7 (retrieval-aware RQ refinement feasibility study) -- query
subsampling.

Data scope per the plan: reuse the FULL COCO train candidate pool (shrinking
the pool would make hard-negative mining trivial and uninformative), but
subsample the QUERIES to keep the feasibility run cheap. Draws a fixed-seed
20,000-query subsample from `mbeir_coco_only_train_filtered.jsonl`, seed 2023
(matching this project's established seed convention).

[Deliberate addition beyond the plan's literal text, required to make
Component 4 implementable] The plan's Component 4 "preferred" option needs a
held-out VALIDATION batch to evaluate `L_ret` on (a "validation retrieval
objective for weight learning", mentor's point 5). The plan says to reuse
"the existing val_loader machinery" -- but that machinery
(`config.evaluator.enable_eval` / `DatasetType.IN_BATCH_VAL`) yields
CLIP-encoded `(q_emb, p_emb)` pairs, a batch format incompatible with
`compute_single_batch`'s `(query, pool, instruct, h_qid[, hard_neg_pool])`
dict-batch convention that `L_ret` depends on. At the time this script was
written, the real, separate, OFFICIAL held-out COCO test query split
(`query/test/mbeir_mscoco_task{0,3}_test.jsonl`) also had no CLIP-SF query
embedding dict extracted for it, so it could not be used here without a new
Stage-0 extraction run.

[Component 9, 2026-07-08 update] That query-side extraction has since been
done (see ~/.claude/plans/so-this-is-the-swirling-quiche.md, Component 9,
Step 2) -- `extracted_embed/CLIP_SF/test/query_SFpretrained_instruction_IT_dict.pt`
now exists and covers all 29,809 official test queries. The claim above ("no
embedding dict... cannot be used") is now stale for the query side, and was
already inaccurate for the candidate side even at the time it was written
(`cand_pool_mscoco_task{0,3}_test_IT_dict.pt` are, and were, full-coverage
symlinks into the existing local-pool candidate dict). This script's
train-derived 18K/2K split is NOT being replaced, however: this is Component
9's fix for the REPORTED Criterion-5 recall metric only
(`dense_recall_at_k_probe` in `check_feasibility_criteria.py`, which now
scores exclusively against the official test split). The val slice produced
by THIS script remains legitimate for exactly one purpose -- the fast,
in-distribution, training-time `balance_level_logits` perturbation signal in
`perturb_and_maybe_update_balance_logits` (engine.py), where being
train-derived and partially memorized is irrelevant (it's a cheap directional
proxy consumed only during optimization, never reported as a result). It must
never again be used as, or reported as, a generalization/recall metric --
that is precisely the mistake Component 9 corrected.

CRITICAL, permanent trap: M-BEIR restarts qid numbering per split, so a qid
like `9:1` refers to a DIFFERENT, unrelated query in the train split than in
the test split (verified directly: train `9:1` is a teddy-bear-shop caption,
test `9:1` is a man-on-a-moped caption). Never look up a qid from this
script's train-derived output files in the official TEST query embedding
dict (or vice versa) -- matching qids across splits will silently return the
wrong embedding rather than raising an error.

This script draws TWO disjoint slices from the same fixed-seed 20,000-query
draw out of the train-split query file: a training slice (`--n_val` fewer
than `--n_total`) and a held-out validation slice (`--n_val`), both written
as separate JSONL files, both still covered by the existing TRAIN Stage-0
embedding dicts (`extracted_embed/CLIP_SF/train/...`) -- never the test
dict, per the qid-collision warning above.
"""
import argparse
import os
import random
import sys

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data.preprocessing.utils import load_jsonl_as_list, save_list_as_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbeir_data_dir", required=True)
    parser.add_argument(
        "--input_query_path",
        default="query/union_train/mbeir_coco_only_train_filtered.jsonl",
        help="Relative to --mbeir_data_dir.",
    )
    parser.add_argument(
        "--out_train_path",
        default="query/union_train/mbeir_coco_only_train_feasibility20k.jsonl",
        help="Relative to --mbeir_data_dir.",
    )
    parser.add_argument(
        "--out_val_path",
        default="query/union_train/mbeir_coco_only_train_feasibility20k_val.jsonl",
        help="Relative to --mbeir_data_dir. Held-out slice for Component 4's "
             "validation-guided balance_level_logits perturbation (see module docstring).",
    )
    parser.add_argument("--n_total", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=2000,
                         help="Drawn from within --n_total, not in addition to it "
                              "(so --n_total=20000 --n_val=2000 gives an 18000/2000 split).")
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    assert args.n_val < args.n_total, "--n_val must be smaller than --n_total."

    input_path = os.path.join(args.mbeir_data_dir, args.input_query_path)
    print(f"Loading full query set from {input_path} ...")
    full_data = load_jsonl_as_list(input_path)
    print(f"Loaded {len(full_data)} queries.")

    rng = random.Random(args.seed)
    indices = list(range(len(full_data)))
    rng.shuffle(indices)
    subsample_indices = indices[: args.n_total]
    val_indices = set(subsample_indices[: args.n_val])
    train_indices = subsample_indices[args.n_val:]

    train_data = [full_data[i] for i in train_indices]
    val_data = [full_data[i] for i in subsample_indices if i in val_indices]

    out_train_path = os.path.join(args.mbeir_data_dir, args.out_train_path)
    out_val_path = os.path.join(args.mbeir_data_dir, args.out_val_path)
    os.makedirs(os.path.dirname(out_train_path), exist_ok=True)
    save_list_as_jsonl(train_data, out_train_path)
    save_list_as_jsonl(val_data, out_val_path)

    print(f"Wrote {len(train_data)} training queries to {out_train_path}")
    print(f"Wrote {len(val_data)} held-out validation queries to {out_val_path}")
    print(f"(seed={args.seed}, n_total={args.n_total}, n_val={args.n_val}, disjoint by construction)")


if __name__ == "__main__":
    main()
