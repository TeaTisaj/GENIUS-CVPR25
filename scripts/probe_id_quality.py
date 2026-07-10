"""Qualitative probe: compare T5-generated beam codes vs the gold candidate's RQ code
for a sample of COCO test (text->image) queries, using the gen_code outputs already
saved by eval job 328642 (CocoOnly1GPU / epoch_25).

Answers: is the model "close but imprecise" (beams share a long prefix with gold /
land on valid trie paths) or "generating a different ID space" (wrong modality code,
no overlap with gold at all)?
"""
import sys
import numpy as np

sys.path.insert(0, "/home/tcetoje/GENIUS-CVPR25/src")
from data.preprocessing.utils import unhash_qid, unhash_did, hash_did

GENIR_DIR = "/home/tcetoje/GENIUS-CVPR25"
MBEIR_DATA_DIR = "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data"
GC = f"{GENIR_DIR}/gen_code/GENIUS_t5small/Large/Instruct/CocoOnly1GPU"
N_SAMPLES = 20
SEED = 0

# --- load qrels (qid -> [gold did, ...]) ---
qrel_path = f"{MBEIR_DATA_DIR}/qrels/test/mbeir_mscoco_task0_test_qrels.txt"
qrel = {}
with open(qrel_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
            qid, _, did, _, _ = parts[:5]
            qrel.setdefault(qid, []).append(did)

# --- load generated codes/ids ---
q_codes = np.load(f"{GC}/test/mbeir_mscoco_task0_test_codes.npy")       # (n_q, n_beams, 9)
q_ids = np.load(f"{GC}/test/mbeir_mscoco_task0_test_ids.npy", allow_pickle=True)
cand_codes = np.load(f"{GC}/cand_pool/mbeir_mscoco_task0_test_cand_pool_codes.npy")  # (n_cand, 9)
cand_ids = np.load(f"{GC}/cand_pool/mbeir_mscoco_task0_test_cand_pool_ids.npy", allow_pickle=True)

cand_id_to_code = {int(i): tuple(c.tolist()) for i, c in zip(cand_ids, cand_codes)}
cand_code_to_ids = {}
for i, c in zip(cand_ids, cand_codes):
    cand_code_to_ids.setdefault(tuple(c.tolist()), []).append(int(i))

q_id_to_index = {int(i): idx for idx, i in enumerate(q_ids)}

rng = np.random.default_rng(SEED)
all_qids = list(qrel.keys())
sample_qids = rng.choice(all_qids, size=min(N_SAMPLES, len(all_qids)), replace=False)

def longest_common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n

print(f"{'qid':<10} {'gold_did':<10} mod_ok  exact  best_lcp  gold_in_beams  gold_code")
print("-" * 100)

n_modality_ok = 0
n_exact = 0
n_gold_in_beams = 0
lcp_values = []

for qid in sample_qids:
    gold_dids = qrel[qid]
    gold_did = gold_dids[0]
    h_gold = hash_did(gold_did)
    if h_gold not in cand_id_to_code:
        print(f"{qid:<10} {gold_did:<10} -- gold doc not found in cand pool codes (skipped) --")
        continue
    gold_code = cand_id_to_code[h_gold]

    h_qid = hash_did  # placeholder, real hash below
    from data.preprocessing.utils import hash_qid
    h_q = hash_qid(qid)
    if h_q not in q_id_to_index:
        print(f"{qid:<10} {gold_did:<10} -- query id not found in generated codes (skipped) --")
        continue
    beams = q_codes[q_id_to_index[h_q]]  # (n_beams, 9)

    modality_ok = sum(1 for b in beams if b[0] == gold_code[0])
    exact = any(tuple(b.tolist()) == gold_code for b in beams)
    lcps = [longest_common_prefix(b.tolist(), gold_code) for b in beams]
    best_lcp = max(lcps)
    gold_in_beams = exact  # exact match is the only way the trie would retrieve the gold doc

    n_modality_ok += 1 if beams[0][0] == gold_code[0] else 0
    n_exact += 1 if exact else 0
    n_gold_in_beams += 1 if gold_in_beams else 0
    lcp_values.append(best_lcp)

    print(f"{qid:<10} {gold_did:<10} {beams[0][0]==gold_code[0]!s:<7} {exact!s:<6} {best_lcp:<9} {gold_in_beams!s:<14} {gold_code}")
    print(f"{'':<10} {'':<10} top beam: {tuple(beams[0].tolist())}")

n = len(sample_qids)
print("-" * 100)
print(f"Top-beam modality-code correct: {n_modality_ok}/{n}  ({100*n_modality_ok/n:.0f}%)")
print(f"Exact full-code match in any of 50 beams: {n_exact}/{n} ({100*n_exact/n:.0f}%)")
print(f"Mean of best longest-common-prefix (out of 9 tokens incl. modality): {np.mean(lcp_values):.2f}")
print(f"LCP distribution: {sorted(lcp_values)}")
