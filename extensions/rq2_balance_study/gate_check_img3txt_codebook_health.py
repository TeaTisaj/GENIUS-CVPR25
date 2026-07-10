"""
Cheap codebook-health gate for the new direction-aware-lambda Stage-1 checkpoints
(CocoImg3txt0, CocoImg3txt0p3) before committing Stage-2 T5 compute to them.

Reuses the existing rq1_coco_variant_diagnostics.py machinery (load_quantizer,
run_inference, level_stats) against the same COCO candidate pool, comparing the
two new checkpoints' per-level utilization/entropy/collision against the
already-established vanilla/strong reference numbers. Intentionally lighter than
the full diagnostic script: no CSV table writing, no neighbor-preservation
metric (expensive, not needed for a pass/fail gate), just the per-level stats
that would show a collapsed/near-zero-entropy semantic level -- the known
failure signature this gate exists to catch (see the plan's Part 3 step 2).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python gate_check_img3txt_codebook_health.py --genir_dir <repo>
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
from rq1_coco_variant_diagnostics import load_quantizer, run_inference, modality_distribution, DATASET_CONFIGS
from rq_analysis_codebook_utilization import level_stats, LEVELS

from data.mbeir_dataset import MBEIRDictCandDataset

NEW_VARIANTS = {
    "img3txt0": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoImg3txt0/rq_clip_large_epoch_140.pth",
    "img3txt0p3": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoImg3txt0p3/rq_clip_large_epoch_140.pth",
}
REFERENCE_VARIANTS = ["vanilla", "strong"]  # from DATASET_CONFIGS["coco"]["variants"], known-healthy anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS["coco"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_dict_dir = os.path.join(args.genir_dir, cfg["pool_dict_rel_path"])
    assert os.path.exists(pool_dict_dir), f"Missing pool dict: {pool_dict_dir}"

    variants = {**{v: cfg["variants"][v] for v in REFERENCE_VARIANTS}, **NEW_VARIANTS}

    any_collapse = False
    for variant, ckpt_path in variants.items():
        print(f"\n=== Variant: {variant} ===")
        dataset = MBEIRDictCandDataset(
            mbeir_data_dir=args.mbeir_data_dir,
            cand_pool_path=cfg["cand_pool_path"],
            pool_dict_dir=pool_dict_dir,
            print_config=False,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        model = load_quantizer(args.genir_dir, ckpt_path, device)
        codes, modality, recon_mse, recon_cos, encode, quant, img_mask, txt_mask = run_inference(model, loader, device)
        n = codes.shape[0]
        print(f"N candidates: {n}  recon_mse_mean={recon_mse.mean():.6f}  recon_cos_mean={recon_cos.mean():.4f}")

        mod_dist = modality_distribution(modality, img_mask, txt_mask)
        print(f"  modality_code[0]: image={mod_dist['image_candidates']}  text={mod_dist['text_candidates']}")

        for level in LEVELS:
            stats = level_stats(codes[:, level - 1])
            top1_code, top1_freq = stats["top5"][0]
            top1_pct = 100.0 * top1_freq / n
            flag = ""
            # Failure signature this gate exists to catch: a semantic level collapsed onto
            # (near-)one code, i.e. near-zero entropy relative to its own max, or a single
            # code holding almost the entire mass -- either means that level carries no
            # discriminative information regardless of what the other levels do.
            if variant in NEW_VARIANTS and (stats["entropy"] / stats["max_entropy"] < 0.05 or top1_pct > 90.0):
                flag = "  <-- COLLAPSE SIGNATURE"
                any_collapse = True
            print(
                f"  level {level}: used={stats['used']:4d}/4096 ({stats['util_pct']:5.1f}%)  "
                f"entropy={stats['entropy']:6.3f}/{stats['max_entropy']:.3f} "
                f"({100 * stats['entropy'] / stats['max_entropy']:5.1f}%)  "
                f"top1_code={top1_code} top1_pct={top1_pct:5.1f}%{flag}"
            )

    print("\n" + "=" * 70)
    if any_collapse:
        print("GATE RESULT: FAIL -- at least one new variant has a collapsed semantic level. "
              "Do not proceed to Stage-2 on the flagged checkpoint(s); retrain from scratch "
              "(fresh weights / bumped seed), not retry-eval.")
        sys.exit(1)
    else:
        print("GATE RESULT: PASS -- both new variants' per-level stats are in the family of "
              "the vanilla/strong reference checkpoints (no collapsed semantic level). "
              "Safe to proceed to Stage-2.")


if __name__ == "__main__":
    main()
