"""
RQ semantic-ID diagnostic, Analysis 1 -- Codebook utilization.

For each of the 8 semantic RQ levels (modality token at index 0 excluded),
reports how much of the 4096-code vocabulary is actually used: utilization
rate, entropy, perplexity (2^entropy, an "effective vocabulary size"), a
fitted Zipf exponent for the rank-frequency curve, and the top-5 codes. Stats
are reported both pooled (all 6 datasets combined) and per-dataset, so
collapse can be distinguished from dataset-specific behavior (feeds Analysis 4).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python rq_analysis_codebook_utilization.py --genir_dir <repo>
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rq_analysis_utils import GEN_CODE_DIR_DEFAULT, POOL_TO_DATASET_LABEL, load_all_pools, output_dirs, semantic_codes, split_by_modality

LEVELS = range(1, 9)
VOCAB_SIZE = 4096


def level_stats(level_codes):
    """level_codes: 1D array of code values in [0, VOCAB_SIZE)."""
    values, counts = np.unique(level_codes, return_counts=True)
    n = len(level_codes)
    used = len(values)
    util_pct = 100.0 * used / VOCAB_SIZE
    unused_pct = 100.0 - util_pct
    probs = counts / n
    entropy = float(-np.sum(probs * np.log2(probs)))
    perplexity = float(2 ** entropy)
    order = np.argsort(-counts)
    top5 = [(int(values[i]), int(counts[i])) for i in order[:5]]

    # Zipf exponent: fit log(freq) = -alpha * log(rank) + c on the sorted
    # frequency curve (skip rank 0 contribution to intercept only).
    sorted_counts = counts[order].astype(float)
    ranks = np.arange(1, len(sorted_counts) + 1, dtype=float)
    if len(ranks) > 1:
        slope, _ = np.polyfit(np.log(ranks), np.log(sorted_counts), 1)
        zipf_exp = float(-slope)
    else:
        zipf_exp = float("nan")

    return {
        "used": used,
        "util_pct": util_pct,
        "unused_pct": unused_pct,
        "entropy": entropy,
        "max_entropy": float(np.log2(VOCAB_SIZE)),
        "perplexity": perplexity,
        "zipf_exp": zipf_exp,
        "top5": top5,
        "rank_freq": sorted_counts,
    }


def write_table(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scope", "level", "used_codes", "utilization_pct", "unused_pct",
            "entropy_bits", "max_entropy_bits", "normalized_entropy", "perplexity", "zipf_exponent", "top1_code", "top1_freq",
        ])
        for row in rows:
            writer.writerow(row)


def plot_rank_frequency(per_level_rank_freq, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for level, ax in zip(LEVELS, axes.flat):
        freq = per_level_rank_freq[level]
        ranks = np.arange(1, len(freq) + 1)
        ax.loglog(ranks, freq, marker=".", linestyle="none", markersize=3)
        ax.set_title(f"Level {level}")
        ax.set_xlabel("code rank")
        ax.set_ylabel("frequency")
    fig.suptitle("Codebook rank-frequency (pooled across datasets), log-log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_code_heatmap(pools, levels=(1, 4, 8), top_n=256, out_path=None):
    """Dataset x code-value heatmap of relative deviation from the uniform
    baseline (1/VOCAB_SIZE), one subplot per level -- same style as the
    semantic-ID codebook-distribution figures used in related RQ-VAE/TIGER-
    style papers: red = a dataset over-concentrates on that code value, white
    = near-uniform, blue = avoided. Columns are the top_n globally most
    frequent codes (pooled across datasets) at that level, ordered by
    descending pooled frequency, so the same column means the same code
    across all subplots/datasets within a level."""
    labels = list(pools.keys())
    baseline = 1.0 / VOCAB_SIZE

    fig, axes = plt.subplots(len(levels), 1, figsize=(14, 3.2 * len(levels)))
    if len(levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        pooled_col = np.concatenate([sem[:, level - 1] for sem in pools.values()])
        top_values, top_counts = np.unique(pooled_col, return_counts=True)
        order = np.argsort(-top_counts)[:top_n]
        top_codes = top_values[order]

        matrix = np.zeros((len(labels), len(top_codes)))
        for i, label in enumerate(labels):
            col = pools[label][:, level - 1]
            n = len(col)
            values, counts = np.unique(col, return_counts=True)
            freq = dict(zip(values.tolist(), counts.tolist()))
            proportions = np.array([freq.get(c, 0) / n for c in top_codes])
            matrix[i] = (proportions - baseline) / baseline

        vmax = np.percentile(np.abs(matrix), 99) or 1.0
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"code value (top {top_n} most frequent, level {level}, ranked by pooled frequency)")
        ax.set_title(f"Level {level}: relative deviation from uniform baseline (1/{VOCAB_SIZE})")
        fig.colorbar(im, ax=ax, label="relative deviation", shrink=0.8)

    fig.suptitle("Dataset x semantic-code-value distribution (red = over-concentration, blue = avoidance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--gen_code_dir", default=GEN_CODE_DIR_DEFAULT, help="Relative to --genir_dir")
    parser.add_argument("--out_root", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis")
    args = parser.parse_args()

    gen_code_dir = os.path.join(args.genir_dir, args.gen_code_dir)
    dirs = output_dirs(args.out_root)

    pools = load_all_pools(gen_code_dir)
    pooled_codes = np.concatenate([semantic_codes(c) for c, _ in pools.values()], axis=0)
    print(f"Pooled candidates across {len(pools)} datasets: {len(pooled_codes)}")

    rows = []
    per_level_rank_freq = {}
    for level in LEVELS:
        stats = level_stats(pooled_codes[:, level - 1])
        per_level_rank_freq[level] = stats["rank_freq"]
        top1_code, top1_freq = stats["top5"][0]
        rows.append([
            "pooled", level, stats["used"], round(stats["util_pct"], 3), round(stats["unused_pct"], 3),
            round(stats["entropy"], 4), round(stats["max_entropy"], 4), round(stats["entropy"] / stats["max_entropy"], 4),
            round(stats["perplexity"], 1), round(stats["zipf_exp"], 4), top1_code, top1_freq,
        ])
        print(f"[pooled] level={level} used={stats['used']}/{VOCAB_SIZE} ({stats['util_pct']:.1f}%) "
              f"entropy={stats['entropy']:.2f}/{stats['max_entropy']:.1f} bits "
              f"perplexity={stats['perplexity']:.0f} zipf_exp={stats['zipf_exp']:.2f} top5={stats['top5']}")

    for name, (codes, _) in pools.items():
        label = POOL_TO_DATASET_LABEL[name]
        sem = semantic_codes(codes)
        for level in LEVELS:
            stats = level_stats(sem[:, level - 1])
            top1_code, top1_freq = stats["top5"][0]
            rows.append([
                label, level, stats["used"], round(stats["util_pct"], 3), round(stats["unused_pct"], 3),
                round(stats["entropy"], 4), round(stats["max_entropy"], 4), round(stats["entropy"] / stats["max_entropy"], 4),
                round(stats["perplexity"], 1), round(stats["zipf_exp"], 4), top1_code, top1_freq,
            ])

    # Supplementary MSCOCO modality breakdown: the main "MSCOCO" row above is
    # spec-compliant (rq_code_analysis.md requires "images + captions"), but
    # MSCOCO is the only dataset here that's not pure-image, so these two
    # extra rows let a reader see how much of "MSCOCO" behavior is driven by
    # the image/caption split vs. genuine within-modality diversity. Additive
    # only -- the pooled stat and the main per-dataset rows above are untouched.
    mscoco_codes, _ = pools["mscoco_task0_test"]
    mscoco_split = split_by_modality(mscoco_codes, np.arange(len(mscoco_codes)))
    mscoco_modality_sem = {}
    for sub_label, (sub_codes, _) in (("MSCOCO (image-only)", mscoco_split["image"]), ("MSCOCO (caption-only)", mscoco_split["caption"])):
        sem = semantic_codes(sub_codes)
        if len(sem) == 0:
            print(f"Warning: {sub_label} has 0 candidates, skipping.")
            continue
        mscoco_modality_sem[sub_label] = sem
        for level in LEVELS:
            stats = level_stats(sem[:, level - 1])
            top1_code, top1_freq = stats["top5"][0]
            rows.append([
                sub_label, level, stats["used"], round(stats["util_pct"], 3), round(stats["unused_pct"], 3),
                round(stats["entropy"], 4), round(stats["max_entropy"], 4), round(stats["entropy"] / stats["max_entropy"], 4),
                round(stats["perplexity"], 1), round(stats["zipf_exp"], 4), top1_code, top1_freq,
            ])
        print(f"[{sub_label}] N={len(sem)}")

    out_csv = os.path.join(dirs["tables"], "table1_utilization.csv")
    write_table(rows, out_csv)
    print(f"Table written to {out_csv}")

    out_fig = os.path.join(dirs["figures"], "analysis1_rank_freq.png")
    plot_rank_frequency(per_level_rank_freq, out_fig)
    print(f"Figure written to {out_fig}")

    pools_sem = {POOL_TO_DATASET_LABEL[name]: semantic_codes(codes) for name, (codes, _) in pools.items()}
    pools_sem.update(mscoco_modality_sem)
    heatmap_fig = os.path.join(dirs["figures"], "analysis1_dataset_code_heatmap.png")
    plot_dataset_code_heatmap(pools_sem, levels=(1, 4, 8), out_path=heatmap_fig)
    print(f"Dataset x code-value heatmap written to {heatmap_fig}")


if __name__ == "__main__":
    main()
