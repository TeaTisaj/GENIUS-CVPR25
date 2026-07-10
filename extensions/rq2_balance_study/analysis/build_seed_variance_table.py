"""
Seed-variance check (RQ3 robustness addition): compares FashionIQ Recall@1 across 3
training seeds (2023 [real study, reused -- NOT retrained here], 7, 13) per variant, to
check whether the "vanilla wins" ranking survives training-seed noise given how tiny the
absolute Recall@1 numbers are (0.27-0.55%).

CAVEAT -- READ BEFORE EXTENDING THIS SCRIPT: n=3 is a lightweight descriptive sanity
check, not a statistical test. Do NOT add a p-value, confidence interval, or significance
claim here -- report mean/std descriptively only, consistent with how the mentor already
asked the RQ1/RQ2 correlation results to be framed (initial trend, not a strong
conclusion). Also: manually eyeball the 3 raw values per variant before trusting mean/std
-- with n=3, one anomalous run can dominate both.

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python build_seed_variance_table.py
"""
import argparse
import csv
import os
import statistics

VARIANTS = ["vanilla", "weak", "medium", "strong"]
HEADLINE_COLUMN = "IT->I Recall@1"
TASK_LABEL = "image,text -> image"


def load_real_r1(table2_csv):
    """seed=2023 Recall@1 per variant, from the already-completed real study. Not retrained."""
    r1 = {}
    with open(table2_csv, newline="") as f:
        for row in csv.DictReader(f):
            r1[row["method"]] = float(row[HEADLINE_COLUMN])
    return r1


def load_seed_variant_r1(sweep_dir, variant, seed):
    path = os.path.join(sweep_dir, f"{variant}_seed{seed}.tsv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Task"] == TASK_LABEL and row["Metric"] == "Recall@1":
                return float(row["Value"]) * 100
    raise ValueError(f"No Recall@1 row found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table2_csv", required=True,
                         help="Real study's table2_retrieval_performance.csv (FashionIQ)")
    parser.add_argument("--sweep_dir", required=True,
                         help="retrieval_results/fashioniq_seed_variance/ (from "
                              "slurm_eval_fashioniq_seed_variance.sh)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    real_r1 = load_real_r1(args.table2_csv)

    rows = []
    for variant in VARIANTS:
        values = {"seed2023": real_r1[variant]}
        for seed in args.seeds:
            values[f"seed{seed}"] = load_seed_variant_r1(args.sweep_dir, variant, seed)

        all_vals = list(values.values())
        row = {"variant": variant, **{f"r1_{k}": v for k, v in values.items()}}
        row["r1_mean"] = round(statistics.mean(all_vals), 4)
        row["r1_std"] = round(statistics.stdev(all_vals), 4)  # sample std, ddof=1, n=3
        rows.append(row)
        print(f"{variant}: {values} -> mean={row['r1_mean']:.3f}  std={row['r1_std']:.3f}  "
              f"(n={len(all_vals)}, descriptive only -- not a statistical test)")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSeed-variance table written to {args.out_csv}")


if __name__ == "__main__":
    main()
