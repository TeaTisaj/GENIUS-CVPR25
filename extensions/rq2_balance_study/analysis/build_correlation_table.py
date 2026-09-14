"""
Recomputes the ID-structure-vs-Recall correlations (RQ1) from source, for MSCOCO and
FashionIQ.

WHY THIS SCRIPT EXISTS (2026-09-04): the r-values previously reported in the paper
(MSCOCO collision r=+0.79, FashionIQ collision r=-0.70, FashionIQ utilization r=+0.90)
were computed ad hoc, persisted only as `correlation_table.csv` inside
the rq_analysis_* output dirs, and then NEVER recomputed after the seed-2023 re-eval
correction (2026-07-11) changed MSCOCO's headline Recall vector (vanilla 14.08 -> 20.86).
The stale values survived four audit rounds and two independent fact-check passes,
because every check compared the draft against the *previous draft* rather than against
source data. Correct values from current data are MSCOCO collision r=+0.47, FashionIQ
collision r=-0.99, FashionIQ utilization r=+0.97.

This script exists so those numbers are never again produced by hand. It reads only
source artifacts and derives the Recall vector the paper actually reports:
  * MSCOCO vanilla/strong -- 5-seed mean (seeds 2023[re-eval]/7/13/21/42)
  * MSCOCO weak/medium    -- original 3-seed mean, seed-2023 NOT re-evaluated
  * FashionIQ             -- 3-seed mean, from the study's own seed-variance table
The vanilla/strong seed-2023 points MUST come from the re-eval TSVs
(`coco_seed2023_reeval/`); weak/medium deliberately keep their original seed-2023
values. See the COCO_REEVAL_VARIANTS comment below for why that asymmetry is correct.

Run with: python build_correlation_table.py [--write]
  (without --write it only reports; with --write it refreshes correlation_table.csv
   in each rq_analysis dir and archives the stale file alongside it.)
"""
import argparse
import csv
import os
import shutil
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ANALYSIS_ROOT = os.environ.get("RQ_ANALYSIS_ROOT", "/fnwi_fs/ivi/irlab/personal/tcetoje")
RESULTS_ROOT = os.path.join(REPO, "retrieval_results")

VARIANTS = ["vanilla", "weak", "medium", "strong"]
# MSCOCO seed sets, per the paper's Table 1 caption (vanilla/strong n=5, others n=3).
#
# IMPORTANT ASYMMETRY (documented decision, not an oversight -- see
# SEED2023_CORRECTION_NOTE.md): the 2026-07-11 seed-2023 re-eval was adopted for
# vanilla and strong ONLY, because only those two had a cluster-outlier seed-2023
# point. weak/medium "are left as originally reported" per that note, so their 3-seed
# means still incorporate the ORIGINAL seed-2023 R@1 (weak 14.74, medium 11.15) rather
# than the re-eval values (13.62 / 9.85). Recomputing weak/medium from the re-eval TSVs
# instead shifts MSCOCO's collision-vs-Recall r from +0.47 to +0.46 -- small, but it
# would silently disagree with the published table, so we read those two straight from
# the study's own seed-variance table to stay consistent with what the paper reports.
COCO_REEVAL_VARIANTS = ["vanilla", "strong"]
COCO_REEVAL_SEEDS = ["2023", "7", "13", "21", "42"]


def read_recall_at1(path, task_id=0):
    """Pull Recall@1 (as a percentage) for one task from a results TSV."""
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["TaskID"]) == task_id and row["Metric"] == "Recall@1":
                return float(row["Value"]) * 100
    raise KeyError(f"Recall@1 for task {task_id} not found in {path}")


def coco_recall_means(tables_dir):
    """MSCOCO T->I Recall@1 per variant, matching the vector the paper reports."""
    means = {}
    # vanilla/strong: 5-seed mean, seed-2023 taken from the corrected re-eval.
    for variant in COCO_REEVAL_VARIANTS:
        vals = []
        for seed in COCO_REEVAL_SEEDS:
            if seed == "2023":
                path = os.path.join(RESULTS_ROOT, "coco_seed2023_reeval", f"{variant}_seed2023_reeval.tsv")
            else:
                path = os.path.join(RESULTS_ROOT, "coco_seed_variance", f"{variant}_seed{seed}.tsv")
            vals.append(read_recall_at1(path, task_id=0))
        means[variant] = sum(vals) / len(vals)
    # weak/medium: original 3-seed means, per the correction note's stated scope.
    with open(os.path.join(tables_dir, "table_seed_variance.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["variant"] in ("weak", "medium"):
                means[row["variant"]] = float(row["r1_mean"])
    return means


def fashioniq_recall_means(tables_dir):
    """FashionIQ (image,text)->image Recall@1, 3-seed means from the study's own table."""
    means = {}
    with open(os.path.join(tables_dir, "table_seed_variance.csv"), newline="") as f:
        for row in csv.DictReader(f):
            means[row["variant"]] = float(row["r1_mean"])
    return means


def id_structure(tables_dir):
    """Mean codebook utilization across the 8 RQ levels, plus collision rate and recon MSE."""
    util = {}
    with open(os.path.join(tables_dir, "table1_id_structure.csv"), newline="") as f:
        for row in csv.DictReader(f):
            util.setdefault(row["variant"], []).append(float(row["utilization_pct"]))
    util = {k: sum(v) / len(v) for k, v in util.items()}

    collision, mse = {}, {}
    with open(os.path.join(tables_dir, "table_collision_recon_summary.csv"), newline="") as f:
        for row in csv.DictReader(f):
            collision[row["variant"]] = float(row["full_id_collision_rate"]) * 100
            mse[row["variant"]] = float(row["recon_mse_mean"])
    return util, collision, mse


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) ** 0.5) * (sum((b - my) ** 2 for b in ys) ** 0.5)
    return num / den


def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    for pos, i in enumerate(order):
        ranks[i] = pos + 1.0
    return ranks


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def build(dataset, tables_dir, recall_means, recall_label):
    util, collision, mse = id_structure(tables_dir)
    recall = [recall_means[v] for v in VARIANTS]
    pairs = [
        ("utilization_vs_recall", "utilization_pct_mean", [util[v] for v in VARIANTS]),
        ("collision_vs_recall", "full_id_collision_rate", [collision[v] for v in VARIANTS]),
        ("reconstruction_vs_recall", "recon_mse_mean", [mse[v] for v in VARIANTS]),
    ]
    rows = []
    for name, metric, xs in pairs:
        rows.append({
            "pair": name,
            "x_metric": metric,
            "y_metric": recall_label,
            "n": len(VARIANTS),
            "pearson_r": round(pearson(xs, recall), 4),
            "spearman_rho": round(spearman(xs, recall), 4),
        })
    print(f"\n{dataset}  (Recall@1 per variant: " +
          ", ".join(f"{v}={recall_means[v]:.2f}" for v in VARIANTS) + ")")
    for r in rows:
        print(f"   {r['pair']:26} pearson r={r['pearson_r']:+.4f}  spearman rho={r['spearman_rho']:+.4f}")
    return rows


def write_table(tables_dir, rows):
    out = os.path.join(tables_dir, "correlation_table.csv")
    if os.path.exists(out):
        archived = os.path.join(tables_dir, f"correlation_table_STALE_ARCHIVED_{date.today():%Y%m%d}.csv")
        shutil.copy2(out, archived)
        print(f"   archived superseded table -> {os.path.basename(archived)}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "x_metric", "y_metric", "n", "pearson_r", "spearman_rho"])
        w.writeheader()
        w.writerows(rows)
    print(f"   wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="refresh correlation_table.csv in each rq_analysis dir (archives the old one)")
    args = ap.parse_args()

    coco_dir = os.path.join(ANALYSIS_ROOT, "rq_analysis_coco_variants", "tables")
    fiq_dir = os.path.join(ANALYSIS_ROOT, "rq_analysis_fashioniq_variants", "tables")

    coco_rows = build("MSCOCO", coco_dir, coco_recall_means(coco_dir), "T->I Recall@1")
    fiq_rows = build("FashionIQ", fiq_dir, fashioniq_recall_means(fiq_dir), "IT->I Recall@1")

    if args.write:
        write_table(coco_dir, coco_rows)
        write_table(fiq_dir, fiq_rows)
    else:
        print("\n(report only; pass --write to refresh the on-disk correlation_table.csv files)")


if __name__ == "__main__":
    main()
