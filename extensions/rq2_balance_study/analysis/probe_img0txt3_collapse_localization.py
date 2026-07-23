"""
Decoder-free (no T5, no beam search) probe for img0txt3's Stage-1 checkpoint only, added
alongside probe_rq_only_collapse_localization.py (which already covers vanilla/weak/medium/
strong/img3txt0/img3txt0p3) so img0txt3 gets the same tokenizer-level corroboration standard
as the other direction-aware-lambda cells, rather than being decoder-level-only.

Run with: python probe_img0txt3_collapse_localization.py
"""
import os
import sys

sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/src")
sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/extensions/retrieval_aware_id_refinement")

from check_feasibility_criteria import dense_recall_at_k_probe  # noqa: E402

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
MBEIR_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
CKPT_PATH = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoImg0txt3/rq_clip_large_epoch_140.pth"


def main():
    print(f"=== img0txt3 ===")
    print(f"Checkpoint: {CKPT_PATH}")
    assert os.path.exists(CKPT_PATH), f"MISSING: {CKPT_PATH}"

    results = dense_recall_at_k_probe(
        genir_dir=GENIR_DIR, mbeir_data_dir=MBEIR_DIR, ckpt_path=CKPT_PATH, data_cache={},
    )
    print("\n| Direction | R@1 | R@5 | R@10 |")
    print("|---|---|---|---|")
    for direction, recalls in results.items():
        print(f"| {direction} | {100*recalls[1]:.2f}% | {100*recalls[5]:.2f}% | {100*recalls[10]:.2f}% |")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
