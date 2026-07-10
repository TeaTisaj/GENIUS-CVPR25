"""
GENIUS Phase 2 — Deliverable 2: RQ code prefix-overlap analysis.

For each RQ level 1..8, groups candidates by their code prefix (digits 1..L)
using a vectorized numpy grouping (np.unique with return_inverse/return_counts
-- equivalent to the create_hash_map grouping pattern used elsewhere in this
repo, just O(n) instead of O(n^2) pairwise comparison) and reports collision
statistics, separately for COCO-only, cross-dataset-only, and a combined pool
(where collisions are further split into same-dataset vs cross-dataset).

Run with PYTHONPATH=<repo>/src.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_pool_utils import load_cand_pool
from data.preprocessing.utils import DATASET_CAN_NUM_UPPER_BOUND
from rq_analysis_utils import output_dirs

QUERY_NAME = "mscoco_task0_test"
# True full-union image-modality cross-dataset pool (see phase2_pool_eval.py
# for why WebQA/EDIS/OVEN/INFOSEEK are excluded -- they contain zero
# image-modality candidates).
CROSS_DATASET_NAMES = ["fashioniq_task7", "fashion200k_task0", "nights_task4", "cirr_task7", "visualnews_task0"]
IMAGE_ONLY_FILTER_NAMES = {"fashion200k_task0", "visualnews_task0"}
LEVELS = range(1, 9)
# Subset called out explicitly in the Analysis 2 assignment (LEVELS above is a superset).
REQUESTED_LEVELS = [1, 2, 3, 4, 8]


def load_image_only(gen_code_dir, name, filter_modality):
    codes, ids = load_cand_pool(gen_code_dir, name)
    if filter_modality:
        mask = codes[:, 0] == 0
        codes, ids = codes[mask], ids[mask]
    return codes, ids


def build_pools(gen_code_dir):
    coco_codes, coco_ids = load_image_only(gen_code_dir, QUERY_NAME, filter_modality=True)

    cross_codes_list, cross_ids_list = [], []
    for name in CROSS_DATASET_NAMES:
        codes, ids = load_image_only(gen_code_dir, name, filter_modality=name in IMAGE_ONLY_FILTER_NAMES)
        cross_codes_list.append(codes)
        cross_ids_list.append(ids)
    cross_codes = np.concatenate(cross_codes_list, axis=0)
    cross_ids = np.concatenate(cross_ids_list, axis=0)

    combined_codes = np.concatenate([coco_codes, cross_codes], axis=0)
    combined_ids = np.concatenate([coco_ids, cross_ids], axis=0)
    return (coco_codes, coco_ids), (cross_codes, cross_ids), (combined_codes, combined_ids)


def basic_collision_stats(codes, level):
    n = len(codes)
    prefixes = codes[:, 1 : 1 + level]
    uniq, inverse, counts = np.unique(prefixes, axis=0, return_inverse=True, return_counts=True)
    n_groups = len(uniq)
    mean_size = float(counts.mean()) if n_groups else 0.0
    median_size = float(np.median(counts)) if n_groups else 0.0
    max_size = int(counts.max()) if n_groups else 0
    collided = counts > 1
    collision_rate = float(counts[collided].sum()) / n if n else 0.0
    return n_groups, mean_size, median_size, max_size, collision_rate


def theoretical_collision_rate(n, level):
    """Expected fraction of items landing in a bucket of size > 1 if codes were
    assigned uniformly at random over the level's full prefix capacity
    (4096**level), i.e. the birthday-problem baseline: P(collision) ~ 1 -
    exp(-(n-1)/K). Used to separate "capacity exhausted" from "capacity unused
    but codes collapsed" in the discriminability-gain figure."""
    capacity = float(4096 ** level)
    return 1.0 - np.exp(-(n - 1) / capacity)


def bucket_sizes(codes, level):
    """Returns the array of group sizes (one entry per unique prefix) at this level."""
    prefixes = codes[:, 1 : 1 + level]
    _, counts = np.unique(prefixes, axis=0, return_counts=True)
    return counts


def top_crowded_prefixes(codes, ids, level, top_n=10):
    """Returns the top_n largest prefix groups: (prefix tuple, size, {dataset_name: count})."""
    prefixes = codes[:, 1 : 1 + level]
    uniq, inverse, counts = np.unique(prefixes, axis=0, return_inverse=True, return_counts=True)
    dataset_ids = ids // DATASET_CAN_NUM_UPPER_BOUND
    order = np.argsort(-counts)[:top_n]
    rows = []
    for group_idx in order:
        member_mask = inverse == group_idx
        member_datasets, member_counts = np.unique(dataset_ids[member_mask], return_counts=True)
        breakdown = {int(d): int(c) for d, c in zip(member_datasets, member_counts)}
        rows.append((tuple(int(x) for x in uniq[group_idx]), int(counts[group_idx]), breakdown))
    return rows


def combined_collision_stats(codes, ids, level):
    n = len(codes)
    prefixes = codes[:, 1 : 1 + level]
    uniq, inverse, counts = np.unique(prefixes, axis=0, return_inverse=True, return_counts=True)
    n_groups = len(uniq)
    mean_size = float(counts.mean()) if n_groups else 0.0
    median_size = float(np.median(counts)) if n_groups else 0.0
    max_size = int(counts.max()) if n_groups else 0

    dataset_ids = ids // DATASET_CAN_NUM_UPPER_BOUND
    order = np.argsort(inverse, kind="stable")
    sorted_tags = dataset_ids[order]
    start_idx = np.concatenate(([0], np.cumsum(counts)[:-1]))
    group_min = np.minimum.reduceat(sorted_tags, start_idx)
    group_max = np.maximum.reduceat(sorted_tags, start_idx)
    group_same_dataset = group_min == group_max

    collided = counts > 1
    same_and_collided = collided & group_same_dataset
    cross = ~group_same_dataset  # implies size > 1 by construction

    collision_rate = float(counts[collided].sum()) / n if n else 0.0
    same_dataset_rate = float(counts[same_and_collided].sum()) / n if n else 0.0
    cross_dataset_rate = float(counts[cross].sum()) / n if n else 0.0
    return n_groups, mean_size, median_size, max_size, collision_rate, same_dataset_rate, cross_dataset_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument(
        "--gen_code_dir",
        default="gen_code/GENIUS_t5small/Large/Instruct/InBatch",
        help="Relative to --genir_dir",
    )
    parser.add_argument("--out", default="retrieval_results/phase2/prefix_overlap_by_level.tsv")
    parser.add_argument("--join_with_1c_results", default=None)
    parser.add_argument("--rq_analysis_out_root", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis")
    args = parser.parse_args()

    gen_code_dir = os.path.join(args.genir_dir, args.gen_code_dir)
    out_path = os.path.join(args.genir_dir, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    (coco_codes, coco_ids), (cross_codes, cross_ids), (combined_codes, combined_ids) = build_pools(gen_code_dir)
    print(f"COCO image candidates: {len(coco_ids)}; cross-dataset candidates: {len(cross_ids)}; combined: {len(combined_ids)}")

    rows = [
        [
            "pool_variant", "level", "num_groups", "mean_group_size", "median_group_size", "max_group_size",
            "collision_rate", "same_dataset_collision_rate", "cross_dataset_collision_rate",
        ]
    ]
    bucket_size_by_level = {"coco": {}, "cross": {}}
    for level in LEVELS:
        n_groups, mean_size, median_size, max_size, collision_rate = basic_collision_stats(coco_codes, level)
        rows.append(["coco", level, n_groups, round(mean_size, 4), round(median_size, 4), max_size, round(collision_rate, 4), "", ""])
        print(f"[coco]   level={level} groups={n_groups} mean_size={mean_size:.2f} median_size={median_size:.1f} max={max_size} collision_rate={collision_rate:.4f}")
        bucket_size_by_level["coco"][level] = bucket_sizes(coco_codes, level)

        n_groups, mean_size, median_size, max_size, collision_rate = basic_collision_stats(cross_codes, level)
        rows.append(["cross", level, n_groups, round(mean_size, 4), round(median_size, 4), max_size, round(collision_rate, 4), "", ""])
        print(f"[cross]  level={level} groups={n_groups} mean_size={mean_size:.2f} median_size={median_size:.1f} max={max_size} collision_rate={collision_rate:.4f}")
        bucket_size_by_level["cross"][level] = bucket_sizes(cross_codes, level)

        n_groups, mean_size, median_size, max_size, collision_rate, same_rate, cross_rate = combined_collision_stats(combined_codes, combined_ids, level)
        rows.append(["combined", level, n_groups, round(mean_size, 4), round(median_size, 4), max_size, round(collision_rate, 4), round(same_rate, 4), round(cross_rate, 4)])
        print(f"[combined] level={level} groups={n_groups} mean_size={mean_size:.2f} median_size={median_size:.1f} max={max_size} "
              f"collision_rate={collision_rate:.4f} same_dataset={same_rate:.4f} cross_dataset={cross_rate:.4f}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    print(f"Results written to {out_path}")

    if args.join_with_1c_results:
        join_path = os.path.join(args.genir_dir, args.join_with_1c_results) if not os.path.isabs(args.join_with_1c_results) else args.join_with_1c_results
        print_join_summary(out_path, join_path)

    # --- Analysis 2 deliverables: table2/figures into the shared rq_analysis output tree ---
    dirs = output_dirs(args.rq_analysis_out_root)

    table2_path = os.path.join(dirs["tables"], "table2_collisions.csv")
    with open(table2_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0])
        writer.writerows(r for r in rows[1:] if r[1] in REQUESTED_LEVELS)
    print(f"Table2 (requested-levels subset) written to {table2_path}")

    plot_bucket_boxplot(bucket_size_by_level, os.path.join(dirs["figures"], "analysis2_bucket_dist.png"))
    plot_discriminability_gain(coco_codes, cross_codes, os.path.join(dirs["figures"], "analysis2_discriminability_gain.png"))

    top_crowded_rows = [["level", "prefix", "bucket_size", "dataset_breakdown"]]
    for level in (3,):
        for prefix, size, breakdown in top_crowded_prefixes(combined_codes, combined_ids, level, top_n=10):
            top_crowded_rows.append([level, str(prefix), size, str(breakdown)])
    top_crowded_path = os.path.join(dirs["tables"], "analysis2_top_crowded.csv")
    with open(top_crowded_path, "w", newline="") as f:
        csv.writer(f).writerows(top_crowded_rows)
    print(f"Top-crowded-prefixes table written to {top_crowded_path}")


def plot_bucket_boxplot(bucket_size_by_level, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions, labels, data = [], [], []
    for i, level in enumerate(REQUESTED_LEVELS):
        for j, variant in enumerate(("coco", "cross")):
            data.append(bucket_size_by_level[variant][level])
            positions.append(i * 3 + j)
            labels.append(f"L{level}\n{variant}")
    ax.boxplot(data, positions=positions, labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("prefix-bucket size (log scale)")
    ax.set_title("Prefix-bucket size distribution: COCO vs. cross-dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_discriminability_gain(coco_codes, cross_codes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for codes, label, marker in ((coco_codes, "COCO", "o"), (cross_codes, "cross-dataset", "s")):
        n = len(codes)
        observed = [1 - basic_collision_stats(codes, level)[4] for level in LEVELS]
        theoretical = [1 - theoretical_collision_rate(n, level) for level in LEVELS]
        ax.plot(list(LEVELS), observed, marker=marker, label=f"{label} (observed)")
        ax.plot(list(LEVELS), theoretical, linestyle="--", marker=marker, alpha=0.5, label=f"{label} (uniform-random baseline)")
    ax.set_xlabel("prefix length (RQ level)")
    ax.set_ylabel("fraction of items with a unique prefix (1 - collision rate)")
    ax.set_title("Discriminability gain vs. prefix length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_join_summary(prefix_overlap_path, results_1c_path):
    """Qualitative side-by-side summary: per-level COCO collision rate vs the
    1c within-vs-cross recall gap at matched pool size. Not a statistical
    correlation -- units don't naturally support one (8 discrete levels vs
    3x3 pool-size/distractor-type cells)."""
    coco_collision_by_level = {}
    with open(prefix_overlap_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["pool_variant"] == "coco":
                coco_collision_by_level[int(row["level"])] = float(row["collision_rate"])

    recall_by_cell = {}  # (size, type) -> {metric: value averaged over seeds}
    with open(results_1c_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["deliverable"] != "1c":
                continue
            key = (int(row["pool_size"]), row["distractor_type"], row["metric"])
            recall_by_cell.setdefault(key, []).append(float(row["value"]))

    print("\n=== Qualitative summary: COCO prefix collision rate vs 1c within/cross recall gap ===")
    print(f"{'RQ level':>8} {'COCO collision rate':>20}")
    for level, rate in sorted(coco_collision_by_level.items()):
        print(f"{level:>8} {rate:>20.4f}")

    sizes = sorted(set(k[0] for k in recall_by_cell))
    for size in sizes:
        for metric in sorted(set(k[2] for k in recall_by_cell if k[0] == size)):
            within = recall_by_cell.get((size, "within", metric))
            cross = recall_by_cell.get((size, "cross", metric))
            if within is None or cross is None:
                continue
            within_mean = sum(within) / len(within)
            cross_mean = sum(cross) / len(cross)
            gap = cross_mean - within_mean
            print(f"pool_size={size} {metric}: within={within_mean:.4f} cross={cross_mean:.4f} (cross-within gap={gap:+.4f})")


if __name__ == "__main__":
    main()
