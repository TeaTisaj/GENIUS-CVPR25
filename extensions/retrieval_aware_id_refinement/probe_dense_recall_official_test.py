"""
Component 9, Step 4: batch re-probe of every already-trained feasibility-study
checkpoint (plus the teacher) using the corrected `dense_recall_at_k_probe`
(modality-matched pools + official held-out MSCOCO test split). No retraining
-- this is purely a re-evaluation of existing checkpoints with the fixed
methodology. Supersedes the earlier scratch prototype
`_tmp_probe_corrected_pools.py` (deleted in this same change).

Usage:
  python probe_dense_recall_official_test.py [--only <label>]
"""
import argparse
import os
import sys

sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/src")

from check_feasibility_criteria import dense_recall_at_k_probe  # noqa: E402

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
MBEIR_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
BASE = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_retrieval_aware_feasibility/rq_clip_large/Large/Instruct"

CHECKPOINTS = [
    (f"{GENIR_DIR}/checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth", "Teacher_CocoVanilla"),
    (f"{BASE}/FeasibilityVxV/rq_clip_large_epoch_10.pth", "Method_EMAlive_lambda3.0"),
    (f"{BASE}/FeasibilityControlNoNewLosses/rq_clip_large_epoch_10.pth", "Control_EMAlive_lambda3.0"),
    (f"{BASE}/FeasibilityVxVFreezeCodebook/rq_clip_large_epoch_10.pth", "Method_Frozen_lambda3.0"),
    (f"{BASE}/FeasibilityControlNoNewLossesFreezeCodebook/rq_clip_large_epoch_10.pth", "Control_Frozen_lambda3.0_AMP"),
    (f"{BASE}/FeasibilityControlNoNewLossesFreezeCodebookNoAMP/rq_clip_large_epoch_10.pth", "Control_Frozen_lambda3.0_NoAMP"),
    (f"{BASE}/FeasibilityVxVFreezeCodebookLowLambda/rq_clip_large_epoch_5.pth", "Method_Frozen_lambda0.3_ep5"),
    (f"{BASE}/FeasibilityVxVFreezeCodebookLowLambda/rq_clip_large_epoch_10.pth", "Method_Frozen_lambda0.3_ep10"),
    (f"{BASE}/FeasibilityControlNoNewLossesFreezeCodebookLowLambda/rq_clip_large_epoch_10.pth", "Control_Frozen_lambda0.3"),
    (f"{BASE}/FeasibilityVxVFreezeCodebookLambda003/rq_clip_large_epoch_10.pth", "Method_Frozen_lambda0.03"),
    (f"{BASE}/FeasibilityControlNoNewLossesFreezeCodebookNoBalance/rq_clip_large_epoch_10.pth", "Control_Frozen_NoBalance"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Run only the checkpoint with this label.")
    args = parser.parse_args()

    data_cache = {}
    rows = []
    for path, label in CHECKPOINTS:
        if args.only and label != args.only:
            continue
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

    print("\n=== Summary table (markdown) ===")
    print("| Checkpoint | Direction | R@1 | R@5 | R@10 |")
    print("|---|---|---|---|---|")
    for label, direction, recalls in rows:
        print(f"| {label} | {direction} | {100*recalls[1]:.2f}% | {100*recalls[5]:.2f}% | {100*recalls[10]:.2f}% |")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
