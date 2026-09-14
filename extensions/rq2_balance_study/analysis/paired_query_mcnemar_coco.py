"""
ECIR short paper (2026-09-14 senior-review revision, plan item A5, optional): paired
per-query test of MSCOCO text->image Recall@1, vanilla vs. strong, on the seed-2023
tokenizer/decoder runs whose TREC run files are kept on disk (the same runs used by
tie_break_sensitivity.py; their R@1 is 20.86% / 23.33%).

McNemar's exact test on the discordant pairs: b = queries only vanilla gets right at rank 1,
c = queries only strong gets right. Under H0 (no difference), b ~ Binomial(b + c, 0.5).

This complements, and does not replace, the tokenizer-level Welch test in
build_shortpaper_table1.py: a per-query test shows the difference is larger than
query-sampling noise for one fixed pair of models, not that it is robust to retraining.

Run with: PYTHONPATH=<repo>/src /home/tcetoje/miniconda3/envs/genius2/bin/python paired_query_mcnemar_coco.py
"""
import os
import sys
from collections import defaultdict

from scipy.stats import binomtest

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
sys.path.insert(0, os.path.join(GENIR_DIR, "src"))
from data.preprocessing.utils import hash_did  # noqa: E402

QRELS = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/qrels/test/mbeir_mscoco_task0_test_qrels.txt"
RUN = os.path.join(GENIR_DIR, "retrieval_results/GENIUS_t5small/Large/Instruct/{}/run_files/"
                   "mbeir_mscoco_task0_single_pool_test_run.txt")


def load_qrels(path):
    qrel = defaultdict(set)
    with open(path) as f:
        for line in f:
            qid, _, did, rel = line.strip().split()[:4]
            if int(rel) > 0:
                qrel[qid].add(hash_did(did))
    return qrel


def load_rank1(path):
    rank1 = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[3]) == 1:
                rank1[parts[0]] = int(parts[2])
    return rank1


def main():
    qrel = load_qrels(QRELS)
    runs = {v: load_rank1(RUN.format(d)) for v, d in (("vanilla", "CocoOnlyVanilla"), ("strong", "CocoOnlyStrong"))}
    qids = sorted(qrel)
    hits = {v: [runs[v].get(q) in qrel[q] for q in qids] for v in runs}
    n = len(qids)
    b = sum(hv and not hs for hv, hs in zip(hits["vanilla"], hits["strong"]))
    c = sum(hs and not hv for hv, hs in zip(hits["vanilla"], hits["strong"]))
    print(f"n_queries={n}  vanilla R@1={100*sum(hits['vanilla'])/n:.2f}%  strong R@1={100*sum(hits['strong'])/n:.2f}%")
    print(f"discordant: vanilla-only={b}  strong-only={c}")
    res = binomtest(b, b + c, 0.5)
    print(f"McNemar exact two-sided p={res.pvalue:.3e}")


if __name__ == "__main__":
    main()
