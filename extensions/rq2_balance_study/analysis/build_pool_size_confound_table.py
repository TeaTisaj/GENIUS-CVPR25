"""
Pool-size confound check (RQ3 robustness addition): builds a 3-way comparison table
(COCO-full vs COCO-subsampled-74K vs FashionIQ-real) to isolate how much of COCO's higher
utilization/collision (vs FashionIQ, Table 4) is a pool-size artifact rather than a
genuine broad-domain-vs-fine-grained property.

Deliberately standalone, not routed through build_table4_cross_dataset.py: that script
compares real cross-dataset *retrieval* numbers, and the subsampled-COCO run produces no
new Recall data (no retraining occurs) -- forcing it in there would be a category error.

utilization_pct_mean / entropy_bits_mean are computed here using the exact same formula as
summarize_coco_variant_study.py's mean_utilization()/mean_entropy() (simple arithmetic mean
of the per-level utilization_pct/entropy_bits columns across all 8 RQ levels, filtered by
variant) -- verified by reading that script directly, not guessed, so these numbers are
directly comparable to the ones already reported in Table 4.

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python build_pool_size_confound_table.py
"""
import argparse
import csv
import os

VARIANTS = ["vanilla", "weak", "medium", "strong"]


def load_id_structure(tables_dir):
    path = os.path.join(tables_dir, "table1_id_structure.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


def build_rows(tables_dir, source_label):
    id_structure_rows = load_id_structure(tables_dir)
    collision_recon = load_collision_recon_summary(tables_dir)
    rows = []
    for variant in VARIANTS:
        recon = collision_recon[variant]
        rows.append({
            "source": source_label,
            "variant": variant,
            "n_candidates": recon["n_candidates"],
            "utilization_pct_mean": round(mean_utilization(id_structure_rows, variant), 3),
            "entropy_bits_mean": round(mean_entropy(id_structure_rows, variant), 4),
            "full_id_collision_rate": recon["full_id_collision_rate"],
            "recon_mse_mean": recon["recon_mse_mean"],
            "recon_cos_sim_mean": recon["recon_cos_sim_mean"],
            "neighbor_preservation_at_10": recon["neighbor_preservation_at_10"],
            "bucket_mean_depth1": recon["bucket_mean_depth1"],
            "bucket_mean_depth8": recon["bucket_mean_depth8"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_full_dir", required=True,
                         help="e.g. /fnwi_fs/.../rq_analysis_coco_variants")
    parser.add_argument("--coco_subsampled_dir", required=True,
                         help="e.g. /fnwi_fs/.../rq_analysis_coco_subsampled74k_variants")
    parser.add_argument("--fashioniq_dir", required=True,
                         help="e.g. /fnwi_fs/.../rq_analysis_fashioniq_variants")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    all_rows = []
    all_rows += build_rows(os.path.join(args.coco_full_dir, "tables"), "COCO-full (N=713679)")
    all_rows += build_rows(os.path.join(args.coco_subsampled_dir, "tables"), "COCO-subsampled74K")
    all_rows += build_rows(os.path.join(args.fashioniq_dir, "tables"), "FashionIQ-real")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Pool-size confound table ({len(all_rows)} rows) written to {args.out_csv}")


if __name__ == "__main__":
    main()
