"""
RQ3's final deliverable: rq-new-plan.txt section 5's "Table 4: Cross-dataset comparison"
(Property / MSCOCO / FashionIQ / Main difference). No COCO precedent needed one (RQ1/RQ2 never
compare across datasets) -- this is RQ3-specific.

Per rq-new-plan.txt's explicit RQ3 instruction: "Do not directly compare their absolute Recall
values because their retrieval tasks differ. Compare instead: relative improvement over vanilla
RQ; ID-level structural differences; optimal balance strength; relationships between ID metrics
and retrieval performance." This script never puts an absolute-Recall row in the table -- only
relative-improvement-over-vanilla and optimal-balance-strength, both computed per dataset
independently, never compared as raw numbers across datasets.

Reads each dataset's already-built Step D outputs (table1_id_structure_summary.csv,
table2_retrieval_performance.csv, table3_balance_tradeoff.csv) -- run
summarize_coco_variant_study.py --dataset {coco,fashioniq} for both datasets first.

Run with: PYTHONPATH=<repo>/src python build_table4_cross_dataset.py \
    --coco_dir /fnwi_fs/.../rq_analysis_coco_variants \
    --fashioniq_dir /fnwi_fs/.../rq_analysis_fashioniq_variants \
    --out_dir /fnwi_fs/.../rq_analysis_cross_dataset
"""
import argparse
import csv
import os

VARIANTS = ["vanilla", "weak", "medium", "strong"]
LAMBDA = {"vanilla": 0.0, "weak": 0.3, "medium": 1.0, "strong": 3.0}

HEADLINE_METRIC = {"MSCOCO": "T->I Recall@1", "FashionIQ": "IT->I Recall@1"}


def load_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_table1(out_dir):
    rows = load_csv_rows(os.path.join(out_dir, "tables", "table1_id_structure_summary.csv"))
    return {r["method"]: r for r in rows}


def load_table3(out_dir):
    rows = load_csv_rows(os.path.join(out_dir, "tables", "table3_balance_tradeoff.csv"))
    return {r["method"]: r for r in rows}


def optimal_balance_strength(table3_by_method, dataset_label):
    """argmax variant by the dataset's own headline Recall metric -- per-dataset, never compared
    as an absolute value across datasets, only the resulting variant NAME and its lambda."""
    metric = HEADLINE_METRIC[dataset_label]
    best_variant = max(table3_by_method, key=lambda v: float(table3_by_method[v][metric]))
    return best_variant, LAMBDA[best_variant]


def relative_recall_improvement(table3_by_method, dataset_label):
    """(best_variant_recall - vanilla_recall) / vanilla_recall -- the one metric rq-new-plan.txt
    explicitly says IS comparable across datasets, unlike absolute Recall."""
    metric = HEADLINE_METRIC[dataset_label]
    vanilla_recall = float(table3_by_method["vanilla"][metric])
    best_variant, _ = optimal_balance_strength(table3_by_method, dataset_label)
    best_recall = float(table3_by_method[best_variant][metric])
    if vanilla_recall == 0:
        return float("nan")
    return (best_recall - vanilla_recall) / vanilla_recall


def mean_property_across_variants(table_by_method, field):
    vals = [float(table_by_method[v][field]) for v in VARIANTS]
    return sum(vals) / len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants")
    parser.add_argument("--fashioniq_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_fashioniq_variants")
    parser.add_argument("--out_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_cross_dataset")
    args = parser.parse_args()

    tables_out = os.path.join(args.out_dir, "tables")
    os.makedirs(tables_out, exist_ok=True)

    coco_t1 = load_table1(args.coco_dir)
    coco_t3 = load_table3(args.coco_dir)
    fiq_t1 = load_table1(args.fashioniq_dir)
    fiq_t3 = load_table3(args.fashioniq_dir)

    coco_best_variant, coco_best_lambda = optimal_balance_strength(coco_t3, "MSCOCO")
    fiq_best_variant, fiq_best_lambda = optimal_balance_strength(fiq_t3, "FashionIQ")
    coco_rel_improvement = relative_recall_improvement(coco_t3, "MSCOCO")
    fiq_rel_improvement = relative_recall_improvement(fiq_t3, "FashionIQ")

    rows = [
        {
            "property": "mean utilization (%, avg over 4 variants)",
            "MSCOCO": round(mean_property_across_variants(coco_t1, "utilization_pct_mean"), 3),
            "FashionIQ": round(mean_property_across_variants(fiq_t1, "utilization_pct_mean"), 3),
        },
        {
            "property": "mean entropy (bits, avg over 4 variants)",
            "MSCOCO": round(mean_property_across_variants(coco_t1, "entropy_bits_mean"), 4),
            "FashionIQ": round(mean_property_across_variants(fiq_t1, "entropy_bits_mean"), 4),
        },
        {
            "property": "mean prefix-bucket size (depth 8, avg over 4 variants)",
            "MSCOCO": round(mean_property_across_variants(coco_t1, "mean_bucket_size_depth8"), 3),
            "FashionIQ": round(mean_property_across_variants(fiq_t1, "mean_bucket_size_depth8"), 3),
        },
        {
            "property": "mean full-ID collision rate (avg over 4 variants)",
            "MSCOCO": round(mean_property_across_variants(coco_t1, "full_id_collision_rate"), 6),
            "FashionIQ": round(mean_property_across_variants(fiq_t1, "full_id_collision_rate"), 6),
        },
        {
            "property": "mean reconstruction MSE (avg over 4 variants)",
            "MSCOCO": round(mean_property_across_variants(coco_t1, "recon_mse_mean"), 6),
            "FashionIQ": round(mean_property_across_variants(fiq_t1, "recon_mse_mean"), 6),
        },
        {
            "property": "optimal balance strength (variant, lambda)",
            "MSCOCO": f"{coco_best_variant} (lambda={coco_best_lambda})",
            "FashionIQ": f"{fiq_best_variant} (lambda={fiq_best_lambda})",
        },
        {
            "property": "relative Recall improvement over vanilla (best variant)",
            "MSCOCO": f"{coco_rel_improvement:+.1%}",
            "FashionIQ": f"{fiq_rel_improvement:+.1%}",
        },
    ]
    for row in rows:
        a, b = row["MSCOCO"], row["FashionIQ"]
        row["main_difference"] = "same" if a == b else f"MSCOCO={a} vs FashionIQ={b}"

    out_csv = os.path.join(tables_out, "table4_cross_dataset.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "MSCOCO", "FashionIQ", "main_difference"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Table 4 (cross-dataset comparison) written to {out_csv}")
    for row in rows:
        print(f"  {row['property']}: MSCOCO={row['MSCOCO']}  FashionIQ={row['FashionIQ']}")


if __name__ == "__main__":
    main()
