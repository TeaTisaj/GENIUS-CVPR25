"""
Mechanism forensics / related-work engagement (Fable's Gap-5 item): measure
whether this study's checkpoints reproduce, contradict, or modulate the
"hourglass" phenomenon reported by Kuai et al. 2024 (intermediate RQ levels
becoming disproportionately concentrated relative to the boundary levels),
across the balance-regularization lambda sweep on both datasets.

Reuses existing, already-verified machinery unchanged:
- `load_quantizer`, `run_inference` from rq1_coco_variant_diagnostics.py
- `level_stats` (per-level used-code count, entropy, top-1 concentration)
  from rq_analysis_codebook_utilization.py, the same function the
  direction-aware-lambda codebook-health gate already used successfully.

No new training. Analysis-only, CPU-friendly (falls back to GPU if available,
but the RQ forward pass is cheap -- no gradients).

Output: one row per (dataset, variant, level) with utilization/entropy/top1,
plus a printed "hourglass score" per (dataset, variant) = mean relative
entropy of levels 2-7 minus mean relative entropy of levels {1, 8} (negative
= genuine hourglass shape: middle levels more concentrated than the boundary
levels; near-zero or positive = no hourglass shape).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python measure_hourglass_phenomenon.py --genir_dir <repo>
"""
import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from rq1_coco_variant_diagnostics import load_quantizer, run_inference, DATASET_CONFIGS  # noqa: E402
from rq_analysis_codebook_utilization import level_stats, LEVELS  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from data.mbeir_dataset import MBEIRDictCandDataset  # noqa: E402

BOUNDARY_LEVELS = {1, 8}
MIDDLE_LEVELS = {2, 3, 4, 5, 6, 7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
    parser.add_argument("--datasets", nargs="+", default=["coco", "fashioniq"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    summary_rows = []

    for dataset_name in args.datasets:
        cfg = DATASET_CONFIGS[dataset_name]
        pool_dict_dir = os.path.join(args.genir_dir, cfg["pool_dict_rel_path"])
        assert os.path.exists(pool_dict_dir), f"Missing pool dict: {pool_dict_dir}"

        for variant, ckpt_path in cfg["variants"].items():
            print(f"\n=== {dataset_name} / {variant} ===")
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

            rel_entropy_by_level = {}
            for level in LEVELS:
                stats = level_stats(codes[:, level - 1])
                rel_entropy = stats["entropy"] / stats["max_entropy"]
                rel_entropy_by_level[level] = rel_entropy
                rows.append({
                    "dataset": dataset_name, "variant": variant, "level": level,
                    "used": stats["used"], "util_pct": stats["util_pct"],
                    "entropy": stats["entropy"], "max_entropy": stats["max_entropy"],
                    "rel_entropy": rel_entropy, "top1_pct": 100.0 * stats["top5"][0][1] / n,
                })
                print(f"  level {level}: rel_entropy={rel_entropy:.3f}  top1_pct={100.0 * stats['top5'][0][1] / n:5.1f}%")

            mid_mean = sum(rel_entropy_by_level[l] for l in MIDDLE_LEVELS) / len(MIDDLE_LEVELS)
            boundary_mean = sum(rel_entropy_by_level[l] for l in BOUNDARY_LEVELS) / len(BOUNDARY_LEVELS)
            hourglass_score = mid_mean - boundary_mean
            summary_rows.append({
                "dataset": dataset_name, "variant": variant,
                "mid_levels_mean_rel_entropy": round(mid_mean, 4),
                "boundary_levels_mean_rel_entropy": round(boundary_mean, 4),
                "hourglass_score": round(hourglass_score, 4),
            })
            print(f"  --> hourglass_score (mid - boundary rel. entropy) = {hourglass_score:+.4f} "
                  f"({'HOURGLASS SHAPE' if hourglass_score < -0.02 else 'no hourglass shape'})")

    print("\n=== Summary (markdown) ===")
    print("| Dataset | Variant | Mid-levels rel. entropy | Boundary rel. entropy | Hourglass score |")
    print("|---|---|---|---|---|")
    for r in summary_rows:
        print(f"| {r['dataset']} | {r['variant']} | {r['mid_levels_mean_rel_entropy']:.3f} | "
              f"{r['boundary_levels_mean_rel_entropy']:.3f} | {r['hourglass_score']:+.3f} |")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-level rows written to {args.out_csv}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
