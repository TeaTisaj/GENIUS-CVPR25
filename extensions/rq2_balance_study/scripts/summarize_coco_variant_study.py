"""
Step D of the RQ1+RQ2 COCO-completion plan: join the per-variant ID-property/
reconstruction diagnostics (Step A, extensions/rq2_balance_study/analysis/
rq1_coco_variant_diagnostics.py) with the per-variant best-epoch Recall@1/5/10
(Step C, scripts/slurm_eval_coco_variant_sweep.sh + summarize_coco_sweep.py)
into rq-new-plan.txt's Table 1 (ID structure, one row per method) / Table 2
(retrieval performance) / Table 3 (balance trade-off) + the correlation table
(RQ1 "Outputs") + all 4 RQ2 "Outputs" curves (balance vs. utilization/
collision/reconstruction live in plot_balance_strength_curves.py since they
don't need Recall; balance vs. Recall lives here since it does) + the 3 RQ1
"Outputs" scatter plots (utilization/collision/reconstruction vs. Recall).

"Best epoch" per variant is selected by max T->I Recall@1 (the paper's
headline metric) among the swept epochs (5,10,15,20,25) -- a single, stated,
falsifiable rule rather than hand-picking.

Run with: PYTHONPATH=<repo>/src python summarize_coco_variant_study.py --genir_dir <repo>
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["vanilla", "weak", "medium", "strong"]
LAMBDA = {"vanilla": 0.0, "weak": 0.3, "medium": 1.0, "strong": 3.0}

# Per-dataset display label, sweep-dir prefix, recall-CSV filename, recall columns, and headline
# (best-epoch-selection) metric. FashionIQ task 7 is single-direction ("image,text -> image",
# IT->I) unlike COCO's two-direction T->I/I->T pair -- see summarize_fashioniq_sweep.py.
DATASET_CONFIGS = {
    "coco": {
        "label": "MSCOCO",
        "sweep_dir_prefix": "coco_epoch_sweep_",
        "recall_csv_name": "coco_recall_by_epoch.csv",
        "recall_columns": ["T->I Recall@1", "T->I Recall@5", "T->I Recall@10",
                            "I->T Recall@1", "I->T Recall@5", "I->T Recall@10"],
        "headline_metric": "T->I Recall@1",
    },
    "fashioniq": {
        "label": "FashionIQ",
        "sweep_dir_prefix": "fashioniq_epoch_sweep_",
        "recall_csv_name": "fashioniq_recall_by_epoch.csv",
        "recall_columns": ["IT->I Recall@1", "IT->I Recall@5", "IT->I Recall@10"],
        "headline_metric": "IT->I Recall@1",
    },
}

# RQ1 "Outputs" (rq-new-plan.txt) asks for a correlation table linking ID
# metrics to Recall. With N=4 (one point per balance variant) a correlation
# coefficient is descriptive at best -- there is no statistical power at
# df=2, so this deliberately reports Pearson r / Spearman rho only, never a
# p-value, to avoid implying a hypothesis test that 4 points cannot support.
def correlation_pairs(headline_metric):
    return [
        ("utilization_pct_mean", headline_metric, "utilization_vs_recall"),
        ("full_id_collision_rate", headline_metric, "collision_vs_recall"),
        ("recon_mse_mean", headline_metric, "reconstruction_vs_recall"),
    ]


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def pearson_r(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x, y):
    return pearson_r(_rank(np.asarray(x, dtype=float)), _rank(np.asarray(y, dtype=float)))


def correlation_table(table3_rows, tables_out, headline_metric):
    """Writes correlation_table.csv: Pearson r + Spearman rho for the 3 ID-metric-vs-Recall
    pairs rq-new-plan.txt's RQ1 "Outputs" asks for ("correlation table for each dataset" --
    distinct from section 5's "Table 1: ID structure", hence the non-table1 filename). N=4
    (one point per balance variant) -- descriptive only, no p-value (false precision at df=2)."""
    n = len(table3_rows)
    rows = []
    for x_key, y_key, label in correlation_pairs(headline_metric):
        x = [float(r[x_key]) for r in table3_rows]
        y = [float(r[y_key]) for r in table3_rows]
        rows.append({
            "pair": label,
            "x_metric": x_key,
            "y_metric": y_key,
            "n": n,
            "pearson_r": round(pearson_r(x, y), 4),
            "spearman_rho": round(spearman_rho(x, y), 4),
        })

    out_csv = os.path.join(tables_out, "correlation_table.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "x_metric", "y_metric", "n", "pearson_r", "spearman_rho"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Correlation table written to {out_csv}")
    print(f"NOTE: n={n} (one point per balance variant) -- descriptive only, no p-value reported "
          f"(no statistical power at this sample size).")
    return rows


def load_recall_by_epoch(sweep_dir, recall_csv_name):
    path = os.path.join(sweep_dir, recall_csv_name)
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def best_epoch_row(rows, headline_metric):
    """Select the row with max value of the dataset's headline metric."""
    valid = [r for r in rows if r.get(headline_metric)]
    return max(valid, key=lambda r: float(r[headline_metric]))


def load_id_structure(tables_dir):
    path = os.path.join(tables_dir, "table1_id_structure.csv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_collision_recon_summary(tables_dir):
    path = os.path.join(tables_dir, "table_collision_recon_summary.csv")
    by_variant = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_variant[row["variant"]] = row
    return by_variant


def mean_utilization(id_structure_rows, variant):
    vals = [float(r["utilization_pct"]) for r in id_structure_rows if r["variant"] == variant]
    return sum(vals) / len(vals)


def mean_entropy(id_structure_rows, variant):
    vals = [float(r["entropy_bits"]) for r in id_structure_rows if r["variant"] == variant]
    return sum(vals) / len(vals)


def build_table1_id_structure_summary(id_structure_rows, collision_recon, tables_out, dataset_label):
    """rq-new-plan.txt section 5's literal "Table 1: ID structure" -- one row per (Dataset,
    Method) with Utilization/Entropy/Collision/Mean bucket/Reconstruction error. Distinct from
    Step A's table1_id_structure.csv (one row per (variant, RQ level), which separately satisfies
    RQ1's "used-code percentage at each RQ level" measurement) -- this is the paper-table
    aggregation, so it gets its own filename to avoid confusing the two."""
    rows = []
    for variant in VARIANTS:
        recon = collision_recon[variant]
        rows.append({
            "dataset": dataset_label,
            "method": variant,
            "utilization_pct_mean": round(mean_utilization(id_structure_rows, variant), 3),
            "entropy_bits_mean": round(mean_entropy(id_structure_rows, variant), 4),
            "full_id_collision_rate": recon["full_id_collision_rate"],
            "mean_bucket_size_depth8": recon["bucket_mean_depth8"],
            "recon_mse_mean": recon["recon_mse_mean"],
        })
    out_csv = os.path.join(tables_out, "table1_id_structure_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Table 1 (ID structure summary) written to {out_csv}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="coco")
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--rq1_tables_dir", default=None, help="Defaults to <dataset>-specific rq_analysis tables dir")
    parser.add_argument("--sweep_root", default=None, help="Defaults to <genir_dir>/retrieval_results")
    parser.add_argument("--out_dir", default=None, help="Defaults to <dataset>-specific rq_analysis dir")
    args = parser.parse_args()

    ds = DATASET_CONFIGS[args.dataset]
    out_dir = args.out_dir or f"/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_{args.dataset}_variants"
    rq1_tables_dir = args.rq1_tables_dir or os.path.join(out_dir, "tables")

    sweep_root = args.sweep_root or os.path.join(args.genir_dir, "retrieval_results")
    tables_out = os.path.join(out_dir, "tables")
    figures_out = os.path.join(out_dir, "figures")
    os.makedirs(tables_out, exist_ok=True)
    os.makedirs(figures_out, exist_ok=True)

    id_structure_rows = load_id_structure(rq1_tables_dir)
    collision_recon = load_collision_recon_summary(rq1_tables_dir)

    build_table1_id_structure_summary(id_structure_rows, collision_recon, tables_out, ds["label"])

    headline_metric = ds["headline_metric"]
    table2_rows = []
    table3_rows = []
    for variant in VARIANTS:
        sweep_dir = os.path.join(sweep_root, f"{ds['sweep_dir_prefix']}{variant}")
        recall_rows = load_recall_by_epoch(sweep_dir, ds["recall_csv_name"])
        best = best_epoch_row(recall_rows, headline_metric)
        best_epoch = best["epoch"]

        table2_rows.append({
            "dataset": ds["label"],
            "method": variant,
            "best_epoch": best_epoch,
            **{col: best.get(col, "") for col in ds["recall_columns"]},
        })

        recon = collision_recon[variant]
        table3_rows.append({
            "dataset": ds["label"],
            "method": variant,
            "balance_lambda": LAMBDA[variant],
            "utilization_pct_mean": round(mean_utilization(id_structure_rows, variant), 3),
            "full_id_collision_rate": recon["full_id_collision_rate"],
            "recon_mse_mean": recon["recon_mse_mean"],
            "recon_cos_sim_mean": recon["recon_cos_sim_mean"],
            # RQ2 "Semantic fidelity" has 3 sub-metrics (recon error, cosine sim, neighbor
            # preservation) -- all 3 belong in the trade-off table, not just the first two.
            "neighbor_preservation_at_10": recon["neighbor_preservation_at_10"],
            "best_epoch": best_epoch,
            headline_metric: best.get(headline_metric, ""),
        })

    table2_csv = os.path.join(tables_out, "table2_retrieval_performance.csv")
    with open(table2_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table2_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table2_rows)
    print(f"Table 2 written to {table2_csv}")

    table3_csv = os.path.join(tables_out, "table3_balance_tradeoff.csv")
    with open(table3_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table3_rows)
    print(f"Table 3 written to {table3_csv}")

    correlation_table(table3_rows, tables_out, headline_metric)

    # Scatter plots: one point per variant, labeled.
    def scatter(x_key, y_key, x_label, y_label, out_name):
        fig, ax = plt.subplots(figsize=(6, 5))
        xs = [float(r[x_key]) for r in table3_rows]
        ys = [float(r[y_key]) for r in table3_rows]
        ax.scatter(xs, ys)
        for r, x, y in zip(table3_rows, xs, ys):
            ax.annotate(r["method"], (x, y), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs {x_label} ({ds['label']}, 4 balance settings)")
        fig.tight_layout()
        out_path = os.path.join(figures_out, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Figure written to {out_path}")

    scatter("utilization_pct_mean", headline_metric, "Mean codebook utilization (%)", headline_metric, "utilization_vs_recall.png")
    scatter("full_id_collision_rate", headline_metric, "Full-ID collision rate", headline_metric, "collision_vs_recall.png")
    scatter("recon_mse_mean", headline_metric, "Reconstruction MSE", headline_metric, "reconstruction_vs_recall.png")

    # RQ2 "Outputs" lists 4 curves, not 3: balance-vs-utilization/collision/reconstruction
    # (plot_balance_strength_curves.py, no Recall dependency) PLUS balance-vs-Recall (this one,
    # which needs Step C's Recall data and so lives here instead).
    fig, ax = plt.subplots(figsize=(5, 4))
    lam = [float(r["balance_lambda"]) for r in table3_rows]
    recall = [float(r[headline_metric]) for r in table3_rows]
    ax.plot(lam, recall, marker="o")
    for r, x, y in zip(table3_rows, lam, recall):
        ax.annotate(r["method"], (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("balance loss strength (lambda)")
    ax.set_ylabel(headline_metric)
    ax.set_title(f"Balance strength vs. Recall ({ds['label']})")
    fig.tight_layout()
    out_path = os.path.join(figures_out, "balance_vs_recall.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure written to {out_path}")


if __name__ == "__main__":
    main()
