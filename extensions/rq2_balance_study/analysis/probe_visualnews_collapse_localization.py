"""
ECIR-paper Round 3, item 3: extends the decoder-free probe (Component 9's
dense_recall_at_k_probe, reused unchanged) to VisualNews vanilla/strong Stage-1
checkpoints. Mirrors probe_rq_only_collapse_localization.py's MSCOCO probe exactly,
with a VisualNews-specific `directions` list -- deliberately NOT edited into the
MSCOCO-hardcoded OFFICIAL_TEST_DIRECTIONS in check_feasibility_criteria.py, since
that list backs already-published, cited paper numbers and must stay untouched.

Purpose: (1) makes literal the paper's intro claim that the tokenizer-level
localization "replicates across two broad-domain corpora" (previously only tested
on MSCOCO); (2) tests whether VisualNews's T->I DROP under strong lambda (the
paper's cleanest negative gain-side finding, seed-stable at the decoder level) is
also visible at the tokenizer level with no T5/beam search involved -- new
information not previously available.

Prerequisite: VisualNews test-query CLIP-SF embeddings
(extracted_embed/CLIP_SF/test_visualnews/query_SFpretrained_instruction_IT_dict.pt),
produced by scripts/slurm_extract_test_query_embeds_visualnews.sh (job 335180).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python probe_visualnews_collapse_localization.py
"""
import os
import sys

sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/src")
sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/extensions/retrieval_aware_id_refinement")

from check_feasibility_criteria import dense_recall_at_k_probe  # noqa: E402

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
MBEIR_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"

STAGE1_CKPT_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct"
CHECKPOINTS = [
    (f"{STAGE1_CKPT_DIR}/VisualnewsVanilla/rq_clip_large_epoch_140.pth", "vanilla"),
    (f"{STAGE1_CKPT_DIR}/VisualnewsStrong/rq_clip_large_epoch_140.pth", "strong"),
]

# VisualNews-specific directions, mirroring OFFICIAL_TEST_DIRECTIONS's structure
# (check_feasibility_criteria.py) with mscoco -> visualnews. Query JSONLs and cand
# pool JSONLs confirmed present on disk before this script was written; cand dicts
# confirmed present at extracted_embed/CLIP_SF/cand/.
VISUALNEWS_TEST_DIRECTIONS = [
    dict(
        name="T2I_task0",
        query_jsonl="query/test/mbeir_visualnews_task0_test.jsonl",
        cand_pool_jsonl="cand_pool/local/mbeir_visualnews_task0_test_cand_pool.jsonl",
        cand_dict="extracted_embed/CLIP_SF/cand/cand_pool_visualnews_task0_test_IT_dict.pt",
        expected_query_modality="text",
        expected_cand_modality="image",
    ),
    dict(
        name="I2T_task3",
        query_jsonl="query/test/mbeir_visualnews_task3_test.jsonl",
        cand_pool_jsonl="cand_pool/local/mbeir_visualnews_task3_test_cand_pool.jsonl",
        cand_dict="extracted_embed/CLIP_SF/cand/cand_pool_visualnews_task3_test_IT_dict.pt",
        expected_query_modality="image",
        expected_cand_modality="text",
    ),
]

VISUALNEWS_QUERY_DICT = "extracted_embed/CLIP_SF/test_visualnews/query_SFpretrained_instruction_IT_dict.pt"

# Real-pipeline reference (T5 + trie + beam search), for comparison -- from the
# paper's Section 7, Table 5 (3-seed means).
REAL_PIPELINE_REFERENCE = {
    "vanilla": {"T->I": 20.07, "I->T": 2.88},
    "strong": {"T->I": 9.43, "I->T": 0.01},
}


def main():
    data_cache = {}
    rows = []
    for path, label in CHECKPOINTS:
        print(f"\n=== {label} ===")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        query_dict_path = os.path.join(GENIR_DIR, VISUALNEWS_QUERY_DICT)
        if not os.path.exists(query_dict_path):
            print(f"  MISSING QUERY DICT: {query_dict_path} -- run "
                  f"scripts/slurm_extract_test_query_embeds_visualnews.sh first.")
            continue
        results = dense_recall_at_k_probe(
            genir_dir=GENIR_DIR, mbeir_data_dir=MBEIR_DIR, ckpt_path=path,
            query_dict_rel=VISUALNEWS_QUERY_DICT,
            directions=VISUALNEWS_TEST_DIRECTIONS,
            data_cache=data_cache,
        )
        for direction, recalls in results.items():
            rows.append((label, direction, recalls))

    if not rows:
        print("\nNO RESULTS -- check prerequisites above.")
        return

    print("\n=== VisualNews RQ-only (no T5) dense-probe Recall@K ===")
    print("| Variant | Direction | R@1 | R@5 | R@10 |")
    print("|---|---|---|---|---|")
    for label, direction, recalls in rows:
        print(f"| {label} | {direction} | {100*recalls[1]:.2f}% | {100*recalls[5]:.2f}% | {100*recalls[10]:.2f}% |")

    print("\n=== Real-pipeline reference (T5 + trie + beam search, 3-seed mean, for comparison) ===")
    print("| Variant | T->I R@1 | I->T R@1 |")
    print("|---|---|---|")
    for label, vals in REAL_PIPELINE_REFERENCE.items():
        print(f"| {label} | {vals['T->I']:.2f}% | {vals['I->T']:.2f}% |")

    print("\nInterpretation: (1) if I->T collapses ALREADY in the RQ-only probe for "
          "strong (comparable to vanilla's collapse-free band), the I->T collapse "
          "localizes to the tokenizer on VisualNews too, mirroring MSCOCO. "
          "(2) Separately, check whether dense T->I ALSO drops for strong relative "
          "to vanilla (mirroring the real pipeline's clean T->I drop) -- if so, "
          "that drop is tokenizer-level too, not decoder-introduced; if dense T->I "
          "does NOT drop while the real pipeline's does, the drop is decoder-level, "
          "a genuinely different mechanism from the I->T collapse. Report whichever "
          "outcome the numbers actually show.")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
