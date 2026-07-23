"""
ECIR audit Round 5, blocking item 1 (2026-07-13): every headline number in this study was a
mean over Stage-2 (decoder) seeds, holding the Stage-1 RQ tokenizer fixed at n=1 per
condition -- but the paper's own dense probe shows both the I->T collapse and the T->I gain
live in the tokenizer, not the decoder. This script adds the missing axis: 2 NEW independently
trained Stage-1 tokenizers (seeds 7, 13) for MSCOCO vanilla/strong, each with ONE Stage-2 seed
(2023) on top, giving n=3 independent tokenizers per condition (the existing seed=2023
tokenizer + these 2). Mirrors build_coco_seed_variance_table.py's descriptive-only framing
(n=3 is a lightweight check, not a full inferential test) -- report mean/std descriptively,
same convention as this project's other n=3/n=4 checks.

Run with: python build_coco_stage1_seed_variance_table.py --table2_csv <path> --sweep_dir <path> --out_csv <path>
"""
import argparse
import csv
import os
import statistics

TASK_LABELS = {"T->I": "text -> image", "I->T": "image -> text"}


def load_real_r1(table2_csv, variant):
    """seed=2023 T->I/I->T Recall@1 for the ORIGINAL (already 5-Stage-2-seed-verified)
    Stage-1 tokenizer, from the real study's table2 CSV."""
    with open(table2_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == variant:
                return float(row["T->I Recall@1"]), float(row["I->T Recall@1"])
    raise ValueError(f"variant {variant} not found in {table2_csv}")


def load_tsv_r1(sweep_dir, filename):
    path = os.path.join(sweep_dir, filename)
    ti, it = None, None
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Metric"] != "Recall@1":
                continue
            if row["Task"] == TASK_LABELS["T->I"]:
                ti = float(row["Value"]) * 100
            elif row["Task"] == TASK_LABELS["I->T"]:
                it = float(row["Value"]) * 100
    if ti is None or it is None:
        raise ValueError(f"Missing Recall@1 row(s) in {path}")
    return ti, it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table2_csv", required=True)
    parser.add_argument("--sweep_dir", required=True,
                         help="retrieval_results/coco_stage1_seedvar_and_img0txt3/")
    parser.add_argument("--s1_seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    rows = []
    for variant in ["vanilla", "strong"]:
        real_ti, real_it = load_real_r1(args.table2_csv, variant)
        ti_vals = {"2023": real_ti}
        it_vals = {"2023": real_it}
        for seed in args.s1_seeds:
            ti, it = load_tsv_r1(args.sweep_dir, f"{variant}_s1seed{seed}.tsv")
            ti_vals[str(seed)] = ti
            it_vals[str(seed)] = it

        ti_list = list(ti_vals.values())
        it_list = list(it_vals.values())
        rows.append({
            "variant": variant,
            "n_stage1_tokenizers": len(ti_list),
            "stage1_seeds": ",".join(ti_vals.keys()),
            "T->I_per_tokenizer": ";".join(f"{v:.2f}" for v in ti_list),
            "T->I_mean": round(statistics.mean(ti_list), 3),
            "T->I_std": round(statistics.stdev(ti_list), 3),
            "I->T_per_tokenizer": ";".join(f"{v:.2f}" for v in it_list),
            "I->T_mean": round(statistics.mean(it_list), 3),
            "I->T_std": round(statistics.stdev(it_list), 3),
        })
        print(f"{variant}: T->I {ti_vals}  mean={statistics.mean(ti_list):.2f} std={statistics.stdev(ti_list):.2f}")
        print(f"{variant}: I->T {it_vals}  mean={statistics.mean(it_list):.2f} std={statistics.stdev(it_list):.2f}")

    v_ti = [float(x) for x in rows[0]["T->I_per_tokenizer"].split(";")]
    s_ti = [float(x) for x in rows[1]["T->I_per_tokenizer"].split(";")]
    print(f"\nNon-overlapping check: min(strong)={min(s_ti):.2f} vs max(vanilla)={max(v_ti):.2f} "
          f"-> {'NON-OVERLAPPING' if min(s_ti) > max(v_ti) else 'OVERLAPPING'}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
