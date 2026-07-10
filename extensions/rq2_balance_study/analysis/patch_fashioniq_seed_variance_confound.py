"""
One-off patch (2026-07-03): the existing FashionIQ table_seed_variance.csv's seed2023 point for
weak/medium was trained on the ORIGINAL (now-lost) Stage-1 quantizer, while seed7/13 used the
REPAIRED one -- see project_rq2_balance_regularization.md item 10. This replaces ONLY weak/medium's
seed2023 value with a freshly-trained clean point (same quantizer as seed7/13, from
slurm_fashioniq_seedrepaircheck.sh), and recomputes their mean/std. vanilla/strong rows are
rewritten byte-identical to before (no confound there -- vanilla had tight repair-parity, strong's
quantizer was never touched).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python patch_fashioniq_seed_variance_confound.py
"""
import argparse
import csv
import os
import statistics

TASK_LABEL = "image,text -> image"


def load_seedrepaircheck_r1(sweep_dir, variant):
    path = os.path.join(sweep_dir, f"{variant}_seedrepaircheck.tsv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Task"] == TASK_LABEL and row["Metric"] == "Recall@1":
                return float(row["Value"]) * 100
    raise ValueError(f"No Recall@1 row found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True, help="existing table_seed_variance.csv (FashionIQ)")
    parser.add_argument("--sweep_dir", required=True,
                         help="retrieval_results/fashioniq_seed_variance/ (has *_seedrepaircheck.tsv)")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        variant = row["variant"]
        if variant not in ("weak", "medium"):
            continue  # vanilla/strong untouched -- no confound there

        old_seed2023 = float(row["r1_seed2023"])
        new_seed2023 = load_seedrepaircheck_r1(args.sweep_dir, variant)
        row["r1_seed2023"] = new_seed2023

        all_vals = [new_seed2023, float(row["r1_seed7"]), float(row["r1_seed13"])]
        row["r1_mean"] = round(statistics.mean(all_vals), 4)
        row["r1_std"] = round(statistics.stdev(all_vals), 4)

        print(f"{variant}: seed2023 {old_seed2023:.3f} (confounded, original quantizer) -> "
              f"{new_seed2023:.3f} (clean, repaired quantizer). "
              f"New mean={row['r1_mean']:.3f} std={row['r1_std']:.3f}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCorrected seed-variance table written to {args.out_csv}")


if __name__ == "__main__":
    main()
