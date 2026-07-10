"""
Seed-variance check for the direction-aware lambda follow-up (img3txt0, img3txt0p3),
mirroring build_coco_seed_variance_table.py's approach for the original 4 balance
variants. Compares COCO T->I Recall@1 across 3 training seeds (2023 [real study,
reused -- NOT retrained here], 7, 13) per variant, to check whether the reported
"both variants underperform vanilla and strong on T->I" finding
(rq1-rq3_final_tables.md, "Direction-Aware Lambda Follow-up") survives training-seed
noise -- this was the one gap Fable's ECIR go/no-go review flagged as the single
mandatory addition before this result goes into the paper.

CAVEAT -- read before extending this script: n=3 is a lightweight descriptive sanity
check, not a statistical test, consistent with every other seed-variance table in this
study. Do NOT add a p-value, confidence interval, or significance claim here.

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python build_direction_aware_seed_variance_table.py
"""
import argparse
import csv
import os
import statistics

VARIANTS = ["img3txt0", "img3txt0p3"]
HEADLINE_COLUMN = "T->I Recall@1"
TASK_LABEL = "text -> image"


def load_seed2023_r1(epoch_sweep_root, variant, best_epoch=25):
    """seed=2023 Recall@1 for a variant, from the already-completed real study's per-epoch
    sweep (retrieval_results/coco_epoch_sweep_<variant>/coco_recall_by_epoch.csv). Not
    retrained here -- this is the same number already reported in rq1-rq3_final_tables.md."""
    path = os.path.join(epoch_sweep_root, f"coco_epoch_sweep_{variant}", "coco_recall_by_epoch.csv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == best_epoch:
                return float(row[HEADLINE_COLUMN])
    raise ValueError(f"No epoch={best_epoch} row found in {path}")


def load_seed_variant_r1(sweep_dir, variant, seed):
    path = os.path.join(sweep_dir, f"{variant}_seed{seed}.tsv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Task"] == TASK_LABEL and row["Metric"] == "Recall@1":
                return float(row["Value"]) * 100
    raise ValueError(f"No Recall@1 row found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_root", required=True,
                         help="repo's retrieval_results/ dir (contains coco_epoch_sweep_img3txt0{,p3}/)")
    parser.add_argument("--sweep_dir", required=True,
                         help="retrieval_results/coco_direction_aware_seed_variance/ (from "
                              "slurm_eval_coco_direction_aware_seed_variance.sh)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--best_epoch", type=int, default=25)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    rows = []
    for variant in VARIANTS:
        values = {"seed2023": load_seed2023_r1(args.retrieval_results_root, variant, args.best_epoch)}
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
