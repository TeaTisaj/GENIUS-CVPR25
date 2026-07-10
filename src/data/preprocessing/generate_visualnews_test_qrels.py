"""
Generates TREC-style qrels files for VisualNews's held-out test queries
(task0 T->I, task3 I->T), in the exact format `load_qrel` in
src/common/mbeir_generative_retriever.py expects:
  <qid> Q0 <did> <relevance_score> <task_id>
one line per (query, positive candidate) pair, mirroring
qrels/test/mbeir_mscoco_task{0,3}_test_qrels.txt exactly.

Run with: python generate_visualnews_test_qrels.py --mbeir_data_dir <path>
"""
import argparse
import os

from utils import load_jsonl_as_list


def write_qrels(queries, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_lines = 0
    with open(out_path, "w") as f:
        for q in queries:
            for did in q["pos_cand_list"]:
                f.write(f"{q['qid']} Q0 {did} 1 {q['task_id']}\n")
                n_lines += 1
    print(f"Wrote {n_lines} qrel lines for {len(queries)} queries -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbeir_data_dir", required=True)
    args = parser.parse_args()
    d = args.mbeir_data_dir

    task0_queries = load_jsonl_as_list(os.path.join(d, "query/test/mbeir_visualnews_task0_test.jsonl"))
    task3_queries = load_jsonl_as_list(os.path.join(d, "query/test/mbeir_visualnews_task3_test.jsonl"))

    write_qrels(task0_queries, os.path.join(d, "qrels/test/mbeir_visualnews_task0_test_qrels.txt"))
    write_qrels(task3_queries, os.path.join(d, "qrels/test/mbeir_visualnews_task3_test_qrels.txt"))

    print("ALL DONE")


if __name__ == "__main__":
    main()
