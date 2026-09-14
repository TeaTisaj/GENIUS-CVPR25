"""
ECIR short paper (2026-09-14 senior-review revision): recompute every number in the
paper's main table (Table 1) and its significance tests directly from the raw per-run
retrieval TSVs and ID-structure CSVs, instead of copying them from earlier drafts.

Seed points per variant (decoder seeds; all on the seed-2023 Stage-1 tokenizer):
  - MSCOCO vanilla/strong: 2023 (re-evaluated run) + 7, 13, 21, 42
  - MSCOCO weak/medium:    2023 (re-evaluated run) + 7, 13
  - VisualNews vanilla/strong: 2023, 7, 13
  - FashionIQ vanilla/strong: 2023 (table2_retrieval_performance.csv) + 7, 13
  - FashionIQ weak/medium: seed-2023-equivalent *_seedrepaircheck.tsv + 7, 13
    (see build_fashioniq_seed_variance_r5r10_table.py for why)
Tokenizer replication (MSCOCO vanilla/strong): Stage-1 seeds 2023, 7, 13, each with
decoder seed 2023.

Metrics: R@1 for every direction, plus the M-BEIR standard metric for the primary
direction (R@5 for MSCOCO/VisualNews text->image, R@10 for FashionIQ).

Run with: /home/tcetoje/miniconda3/envs/genius2/bin/python build_shortpaper_table1.py
"""
import csv
import os
import statistics

from scipy import stats

RESULTS = "/home/tcetoje/GENIUS-CVPR25/retrieval_results"
ANALYSIS = "/fnwi_fs/ivi/irlab/personal/tcetoje"
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_shortpaper_table1.csv")

T2I, I2T, IT2I = "text -> image", "image -> text", "image,text -> image"


def read_tsv(path):
    """Return {(task, metric): value_in_percent} for one retrieval run."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["Task"], row["Metric"])] = float(row["Value"]) * 100
    return out


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def coco_runs(variant):
    runs = [read_tsv(f"{RESULTS}/coco_seed2023_reeval/{variant}_seed2023_reeval.tsv")]
    seeds = [7, 13, 21, 42] if variant in ("vanilla", "strong") else [7, 13]
    runs += [read_tsv(f"{RESULTS}/coco_seed_variance/{variant}_seed{s}.tsv") for s in seeds]
    return runs


def visualnews_runs(variant):
    base = f"VisualnewsOnly{variant.capitalize()}"
    return [read_tsv(f"{RESULTS}/visualnews_all_seeds/{base}{suffix}.tsv")
            for suffix in ("", "Seed7", "Seed13")]


def fashioniq_runs(variant):
    d = f"{RESULTS}/fashioniq_seed_variance"
    if variant in ("weak", "medium"):
        runs = [read_tsv(f"{d}/{variant}_seedrepaircheck.tsv")]
    else:
        with open(f"{ANALYSIS}/rq_analysis_fashioniq_variants/tables/table2_retrieval_performance.csv") as f:
            row = next(r for r in csv.DictReader(f) if r["method"] == variant)
        runs = [{(IT2I, "Recall@1"): float(row["IT->I Recall@1"]),
                 (IT2I, "Recall@10"): float(row["IT->I Recall@10"])}]
    runs += [read_tsv(f"{d}/{variant}_seed{s}.tsv") for s in (7, 13)]
    return runs


def id_structure(dataset_dir, variant):
    """Mean per-level hard-assignment entropy (levels 1-8, bits) and full-ID collision (%)."""
    tables = f"{ANALYSIS}/{dataset_dir}/tables"
    with open(f"{tables}/table1_id_structure.csv") as f:
        ent = [float(r["entropy_bits"]) for r in csv.DictReader(f)
               if r["variant"] == variant and 1 <= int(r["level"]) <= 8]
    assert len(ent) == 8, (dataset_dir, variant, len(ent))
    with open(f"{tables}/table_collision_recon_summary.csv") as f:
        coll = next(float(r["full_id_collision_rate"]) for r in csv.DictReader(f) if r["variant"] == variant)
    return statistics.mean(ent), coll * 100


def main():
    rows = []

    def add(dataset, variant, runs, prim_task, prim_std_metric, id_dir, has_b):
        a1 = [r[(prim_task, "Recall@1")] for r in runs]
        ak = [r[(prim_task, prim_std_metric)] for r in runs]
        b1 = [r[(I2T, "Recall@1")] for r in runs] if has_b else None
        ent, coll = id_structure(id_dir, variant)
        row = {"dataset": dataset, "variant": variant, "n_seeds": len(runs),
               "entropy_bits": round(ent, 2), "collision_pct": round(coll, 1),
               "dirA_R1_mean": round(mean_std(a1)[0], 2), "dirA_R1_std": round(mean_std(a1)[1], 2),
               "dirA_std_metric": prim_std_metric,
               "dirA_Rk_mean": round(mean_std(ak)[0], 2), "dirA_Rk_std": round(mean_std(ak)[1], 2),
               "dirB_R1_mean": round(mean_std(b1)[0], 2) if has_b else "",
               "dirB_R1_std": round(mean_std(b1)[1], 2) if has_b else "",
               "dirA_R1_per_seed": ";".join(f"{v:.2f}" for v in a1)}
        rows.append(row)
        return a1

    coco = {v: add("MSCOCO", v, coco_runs(v), T2I, "Recall@5", "rq_analysis_coco_variants", True)
            for v in ("vanilla", "weak", "medium", "strong")}
    vn = {v: add("VisualNews", v, visualnews_runs(v), T2I, "Recall@5", "rq_analysis_visualnews_variants", True)
          for v in ("vanilla", "strong")}
    for v in ("vanilla", "weak", "medium", "strong"):
        add("FashionIQ", v, fashioniq_runs(v), IT2I, "Recall@10", "rq_analysis_fashioniq_variants", False)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)

    print("\n--- Significance / effect sizes ---")
    t = stats.ttest_ind(coco["strong"], coco["vanilla"], equal_var=False)
    print(f"MSCOCO T->I decoder seeds (n=5 vs 5, one tokenizer): t={t.statistic:.2f} p={t.pvalue:.2e} "
          f"rel={100*(statistics.mean(coco['strong'])/statistics.mean(coco['vanilla'])-1):+.1f}%")
    tok = {v: [read_tsv(f"{RESULTS}/coco_seed2023_reeval/{v}_seed2023_reeval.tsv")[(T2I, "Recall@1")]]
           + [read_tsv(f"{RESULTS}/coco_stage1_seedvar_and_img0txt3/{v}_s1seed{s}.tsv")[(T2I, "Recall@1")]
              for s in (7, 13)] for v in ("vanilla", "strong")}
    t = stats.ttest_ind(tok["strong"], tok["vanilla"], equal_var=False)
    print(f"MSCOCO T->I tokenizers (n=3 vs 3): vanilla {tok['vanilla']} mean {mean_std(tok['vanilla'])} | "
          f"strong {tok['strong']} mean {mean_std(tok['strong'])} | t={t.statistic:.2f} df={t.df:.2f} "
          f"p={t.pvalue:.3f} rel={100*(statistics.mean(tok['strong'])/statistics.mean(tok['vanilla'])-1):+.1f}%")
    print(f"  every strong tokenizer > every vanilla tokenizer: {min(tok['strong']) > max(tok['vanilla'])}")
    t = stats.ttest_ind(vn["strong"], vn["vanilla"], equal_var=False)
    print(f"VisualNews T->I decoder seeds (n=3 vs 3): p={t.pvalue:.2e} "
          f"rel={100*(statistics.mean(vn['strong'])/statistics.mean(vn['vanilla'])-1):+.1f}%")
    for v in ("weak", "medium"):
        print(f"MSCOCO {v} vs vanilla T->I rel: "
              f"{100*(statistics.mean(coco[v])/statistics.mean(coco['vanilla'])-1):+.1f}%")

    print("\n--- Chance-level R@1 (mean positives / pool size) ---")
    print(f"MSCOCO I->T: {100*4.9978/24809:.3f}%  MSCOCO T->I: {100*1/5000:.3f}%  "
          f"VisualNews I->T: {100*1/19996:.4f}%  FashionIQ R@10: {100*10*1.0018/74380:.3f}%")


if __name__ == "__main__":
    main()
