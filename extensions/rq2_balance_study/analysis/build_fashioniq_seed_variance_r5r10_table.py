"""
ECIR-paper Round 3, item 2: FashionIQ R@5/R@10 seed-variance table (companion to the
existing R@1-only table_seed_variance.csv). Seed points per variant:
  - vanilla/strong: seed2023 from table2_retrieval_performance.csv (Stage-1 quantizer
    never overwritten, no confound -- see project_rq2_balance_regularization.md item 10)
    plus seed7/seed13 from retrieval_results/fashioniq_seed_variance/.
  - weak/medium: seed2023-equivalent from *_seedrepaircheck.tsv (the ORIGINAL seed2023
    Stage-1 checkpoint was overwritten and later repaired; the repaircheck run uses the
    same repaired quantizer as seed7/13, avoiding the confound patch_fashioniq_seed_
    variance_confound.py already fixed for the R@1-only table) plus seed7/seed13.
Do NOT use table2_retrieval_performance.csv's weak/medium rows directly -- those are the
pre-repair (confounded) checkpoint's numbers and are inconsistent with the R@1 already
published in the paper's Table 2 (0.27%/0.21% for weak/medium, which match the repaired
seedrepaircheck values, not the confounded ones).

CAVEAT: n=3 is a lightweight descriptive check, not a statistical test -- report mean/std
descriptively only, consistent with this project's established convention (see
build_coco_seed_variance_table.py).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python build_fashioniq_seed_variance_r5r10_table.py
"""
import argparse
import csv
import os
import statistics

TASK_LABEL = "image,text -> image"
VARIANTS = ["vanilla", "weak", "medium", "strong"]
CONFOUNDED_SEED2023 = {"weak", "medium"}  # need *_seedrepaircheck.tsv, not table2_retrieval_performance.csv


def load_table2_r1_r5_r10(table2_csv):
    out = {}
    with open(table2_csv, newline="") as f:
        for row in csv.DictReader(f):
            out[row["method"]] = {
                "r1": float(row["IT->I Recall@1"]),
                "r5": float(row["IT->I Recall@5"]),
                "r10": float(row["IT->I Recall@10"]),
            }
    return out


def load_tsv_r1_r5_r10(path):
    vals = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Task"] != TASK_LABEL:
                continue
            if row["Metric"] == "Recall@1":
                vals["r1"] = float(row["Value"]) * 100
            elif row["Metric"] == "Recall@5":
                vals["r5"] = float(row["Value"]) * 100
            elif row["Metric"] == "Recall@10":
                vals["r10"] = float(row["Value"]) * 100
    if len(vals) != 3:
        raise ValueError(f"Expected R@1/5/10 in {path}, got {vals}")
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table2_csv", required=True,
                         help="rq_analysis_fashioniq_variants/tables/table2_retrieval_performance.csv")
    parser.add_argument("--sweep_dir", required=True,
                         help="retrieval_results/fashioniq_seed_variance/")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    table2 = load_table2_r1_r5_r10(args.table2_csv)

    rows = []
    for variant in VARIANTS:
        if variant in CONFOUNDED_SEED2023:
            seed2023 = load_tsv_r1_r5_r10(os.path.join(args.sweep_dir, f"{variant}_seedrepaircheck.tsv"))
        else:
            seed2023 = table2[variant]

        per_seed = {"seed2023": seed2023}
        for seed in args.seeds:
            per_seed[f"seed{seed}"] = load_tsv_r1_r5_r10(
                os.path.join(args.sweep_dir, f"{variant}_seed{seed}.tsv")
            )

        row = {"variant": variant}
        for metric in ("r1", "r5", "r10"):
            vals = [v[metric] for v in per_seed.values()]
            row[f"{metric}_mean"] = round(statistics.mean(vals), 4)
            row[f"{metric}_std"] = round(statistics.stdev(vals), 4)
            for label, v in per_seed.items():
                row[f"{metric}_{label}"] = v[metric]
        rows.append(row)
        print(f"{variant}: R@1={row['r1_mean']:.3f}+-{row['r1_std']:.3f}  "
              f"R@5={row['r5_mean']:.3f}+-{row['r5_std']:.3f}  "
              f"R@10={row['r10_mean']:.3f}+-{row['r10_std']:.3f}  "
              f"(n={len(per_seed)}, descriptive only -- not a statistical test)")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
