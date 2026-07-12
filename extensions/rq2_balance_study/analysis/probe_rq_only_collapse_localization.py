"""
Mechanism forensics, step 1 (Fable's ladder): localize the I->T collapse to the
tokenizer (Stage 1, RQ) vs. the decoder (Stage 2, T5 + beam search).

Reuses `dense_recall_at_k_probe` from the retrieval-aware-RQ-refinement study
(already independently verified there: modality-matched pools, official
held-out MSCOCO test split, no train/test qid-collision trap) unchanged --
this is the same "RQ-only, no T5" dense-KNN proxy, just pointed at the four
balance-regularization Stage-1 checkpoints (CocoVanilla/Weak/Medium/Strong)
instead of the feasibility study's checkpoints. No new training.

Logic: the real GENIUS pipeline (T5 + trie + beam search) gives these four
checkpoints I->T Recall@1 of 9.44% (vanilla) vs. <=0.04% (weak/medium/strong)
-- see rq1-rq3_final_tables.md. If this cheap RQ-only probe ALSO shows
weak/medium/strong collapsing on I->T (near the teacher/vanilla's dense-probe
band), the collapse is already present at the tokenizer level, before any T5
training -- i.e. the regularized text-candidate codes are non-discriminative
on their own. If instead the RQ-only probe shows all four checkpoints with
comparable I->T recall, the collapse must be introduced downstream, by
Stage-2 T5 training or by beam-search decoding specifically.

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python probe_rq_only_collapse_localization.py
"""
import os
import sys

sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/src")
sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/extensions/retrieval_aware_id_refinement")

from check_feasibility_criteria import dense_recall_at_k_probe  # noqa: E402

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
MBEIR_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"

MBEIR_STAGE1_CKPT_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct"
CHECKPOINTS = [
    (f"{GENIR_DIR}/checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth", "vanilla"),
    (f"{MBEIR_STAGE1_CKPT_DIR}/CocoWeak/rq_clip_large_epoch_140.pth", "weak"),
    (f"{MBEIR_STAGE1_CKPT_DIR}/CocoMedium/rq_clip_large_epoch_140.pth", "medium"),
    (f"{MBEIR_STAGE1_CKPT_DIR}/CocoStrong/rq_clip_large_epoch_140.pth", "strong"),
    # Added 2026-07-10 for the ECIR audit's Section 6 single-seed mitigation (H1): the
    # direction-aware-lambda Stage-1 checkpoints exist on disk independent of the Stage-2
    # decode crashes that blocked a 3-seed retrain of Section 6's real-pipeline numbers.
    # This probe needs no Stage-2 seed at all, so it corroborates (or contradicts) the
    # single-seed real-pipeline result at the tokenizer level.
    (f"{MBEIR_STAGE1_CKPT_DIR}/CocoImg3txt0/rq_clip_large_epoch_140.pth", "img3txt0"),
    (f"{MBEIR_STAGE1_CKPT_DIR}/CocoImg3txt0p3/rq_clip_large_epoch_140.pth", "img3txt0p3"),
]

# Real-pipeline reference (T5 + trie + beam search), for comparison -- from
# rq1-rq3_final_tables.md Table 2, 3-seed means (vanilla/weak/medium/strong) and the
# direction-aware-lambda single-seed table (img3txt0/img3txt0p3, Section 6 of the paper).
REAL_PIPELINE_REFERENCE = {
    "vanilla": {"T->I": 18.52, "I->T": 9.44},
    "weak": {"T->I": 14.13, "I->T": 0.04},
    "medium": {"T->I": 10.36, "I->T": 0.02},
    "strong": {"T->I": 22.60, "I->T": 0.04},
    "img3txt0": {"T->I": 7.97, "I->T": 0.12},
    "img3txt0p3": {"T->I": 5.45, "I->T": 0.08},
}


def main():
    data_cache = {}
    rows = []
    for path, label in CHECKPOINTS:
        print(f"\n=== {label} ===")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        results = dense_recall_at_k_probe(
            genir_dir=GENIR_DIR, mbeir_data_dir=MBEIR_DIR, ckpt_path=path,
            data_cache=data_cache,
        )
        for direction, recalls in results.items():
            rows.append((label, direction, recalls))

    print("\n=== RQ-only (no T5) dense-probe Recall@K ===")
    print("| Variant | Direction | R@1 | R@5 | R@10 |")
    print("|---|---|---|---|---|")
    for label, direction, recalls in rows:
        print(f"| {label} | {direction} | {100*recalls[1]:.2f}% | {100*recalls[5]:.2f}% | {100*recalls[10]:.2f}% |")

    print("\n=== Real-pipeline reference (T5 + trie + beam search, for comparison) ===")
    print("| Variant | T->I R@1 | I->T R@1 |")
    print("|---|---|---|")
    for label, vals in REAL_PIPELINE_REFERENCE.items():
        print(f"| {label} | {vals['T->I']:.2f}% | {vals['I->T']:.2f}% |")

    print("\nInterpretation: if I->T collapses ALREADY in the RQ-only probe for "
          "weak/medium/strong (comparable to vanilla's collapse-free band), the "
          "collapse is a tokenizer property. If the RQ-only probe shows all four "
          "in a similar I->T band (no collapse), the collapse must be introduced "
          "by Stage-2 T5 training or beam-search decoding.")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
