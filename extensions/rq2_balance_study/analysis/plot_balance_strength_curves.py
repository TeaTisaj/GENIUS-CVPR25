"""
RQ2 "Outputs" (rq-new-plan.txt) asks for three plots: balance strength vs.
utilization, vs. collision-rate, vs. reconstruction-error. Pure plotting from
Step A's existing CSVs (table1_id_structure.csv, table_collision_recon_summary.csv)
-- no rerun of the diagnostic needed.

Run with: python plot_balance_strength_curves.py [--tables_dir ...] [--figures_dir ...]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA = {"vanilla": 0.0, "weak": 0.3, "medium": 1.0, "strong": 3.0}
VARIANTS = ["vanilla", "weak", "medium", "strong"]


def load_mean_utilization(tables_dir):
    """Averages utilization_pct across all 8 RQ levels, per variant."""
    sums, counts = {}, {}
    with open(os.path.join(tables_dir, "table1_id_structure.csv")) as f:
        for row in csv.DictReader(f):
            v = row["variant"]
            sums[v] = sums.get(v, 0.0) + float(row["utilization_pct"])
            counts[v] = counts.get(v, 0) + 1
    return {v: sums[v] / counts[v] for v in sums}


def load_collision_recon(tables_dir):
    out = {}
    with open(os.path.join(tables_dir, "table_collision_recon_summary.csv")) as f:
        for row in csv.DictReader(f):
            out[row["variant"]] = row
    return out


def plot_curve(x, y, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, marker="o")
    for xi, yi, v in zip(x, y, VARIANTS):
        ax.annotate(v, (xi, yi), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants/tables")
    parser.add_argument("--figures_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants/figures")
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    mean_util = load_mean_utilization(args.tables_dir)
    collision_recon = load_collision_recon(args.tables_dir)

    lam = [LAMBDA[v] for v in VARIANTS]
    util = [mean_util[v] for v in VARIANTS]
    collision = [float(collision_recon[v]["full_id_collision_rate"]) * 100 for v in VARIANTS]
    recon_mse = [float(collision_recon[v]["recon_mse_mean"]) for v in VARIANTS]

    plot_curve(lam, util, "balance loss strength (lambda)", "mean utilization (%, avg over 8 RQ levels)",
               "Balance strength vs. codebook utilization",
               os.path.join(args.figures_dir, "balance_vs_utilization.png"))
    plot_curve(lam, collision, "balance loss strength (lambda)", "full-ID collision rate (%)",
               "Balance strength vs. collision rate",
               os.path.join(args.figures_dir, "balance_vs_collision.png"))
    plot_curve(lam, recon_mse, "balance loss strength (lambda)", "reconstruction MSE",
               "Balance strength vs. reconstruction error",
               os.path.join(args.figures_dir, "balance_vs_reconstruction.png"))

    print(f"Wrote 3 plots to {args.figures_dir}")


if __name__ == "__main__":
    main()
