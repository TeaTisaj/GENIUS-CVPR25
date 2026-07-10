"""
Collect per-epoch COCO eval tsv files (epoch_<N>.tsv, written by
slurm_eval_coco_epoch_sweep.sh) into one table: epoch vs T->I / I->T Recall@1/5/10.
"""
import argparse
import csv
import glob
import os
import re

TASKS = {"text -> image": "T->I", "image -> text": "I->T"}
METRICS = ["Recall@1", "Recall@5", "Recall@10"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True)
    args = parser.parse_args()

    rows_by_epoch = {}
    for path in glob.glob(os.path.join(args.sweep_dir, "epoch_*.tsv")):
        match = re.search(r"epoch_(\d+)\.tsv$", path)
        if not match:
            continue
        epoch = int(match.group(1))
        values = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                task = TASKS.get(row["Task"])
                if task and row["Metric"] in METRICS:
                    values[f"{task} {row['Metric']}"] = float(row["Value"]) * 100
        rows_by_epoch[epoch] = values

    columns = [f"{t} {m}" for t in TASKS.values() for m in METRICS]
    out_csv = os.path.join(args.sweep_dir, "coco_recall_by_epoch.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + columns)
        for epoch in sorted(rows_by_epoch):
            row = rows_by_epoch[epoch]
            writer.writerow([epoch] + [row.get(c, "") for c in columns])

    print(f"{'epoch':>6} | " + " | ".join(f"{c:>14}" for c in columns))
    for epoch in sorted(rows_by_epoch):
        row = rows_by_epoch[epoch]
        print(f"{epoch:>6} | " + " | ".join(f"{row.get(c, float('nan')):>14.2f}" for c in columns))
    print(f"\nSaved {out_csv}")
