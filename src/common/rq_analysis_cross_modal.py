"""
RQ semantic-ID diagnostic, Analysis 3 -- Cross-modal semantic alignment.

Verified (see project memory / plan) that no model inference is needed:
- MSCOCO's task0/task3 cand_pool codes are the SAME shared pool (images +
  captions both already quantized by the frozen rq_clip_large.pth).
- Every GT caption (query_txt in mbeir_mscoco_task0_test.jsonl) appears
  verbatim as a separate text-modality candidate in the cand_pool JSONL
  (confirmed 100% match rate on the 24,809 text-modality queries) -- so
  matched-pair codes are a pure lookup, no fresh encoding required.
- Hard-negative mining uses the raw (pre-quantization) CLIP-SF embeddings
  already extracted in extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_IT_dict.pt
  ({'img','text','img_mask','text_mask','id_to_index'}, 768-dim, float16).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python rq_analysis_cross_modal.py --genir_dir <repo> --mbeir_data_dir <mbeir_data>
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

from data.preprocessing.utils import hash_did
from phase2_pool_utils import load_cand_pool
from rq_analysis_utils import GEN_CODE_DIR_DEFAULT, hamming_distance, output_dirs, per_level_agreement, shared_prefix_length

RNG_SEED = 2026
N_PAIRS = 5000
HARDNEG_BATCH = 500


def build_code_lookup(gen_code_dir):
    codes, ids = load_cand_pool(gen_code_dir, "mscoco_task0_test")
    order = np.argsort(ids)
    return ids[order], codes[order]


def lookup_codes(sorted_ids, sorted_codes, query_ids):
    pos = np.searchsorted(sorted_ids, query_ids)
    found = (pos < len(sorted_ids)) & (sorted_ids[np.clip(pos, 0, len(sorted_ids) - 1)] == query_ids)
    return pos, found


def load_qrels_and_queries(mbeir_data_dir):
    qrel_path = os.path.join(mbeir_data_dir, "qrels/test/mbeir_mscoco_task0_test_qrels.txt")
    qid_to_image_did = {}
    with open(qrel_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, did, _, _ = parts[:5]
                qid_to_image_did.setdefault(qid, did)

    query_path = os.path.join(mbeir_data_dir, "query/test/mbeir_mscoco_task0_test.jsonl")
    qid_to_caption = {}
    with open(query_path) as f:
        for line in f:
            d = json.loads(line)
            if d["query_modality"] == "text":
                qid_to_caption[d["qid"]] = d["query_txt"]

    cand_path = os.path.join(mbeir_data_dir, "cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl")
    caption_text_to_did = {}
    with open(cand_path) as f:
        for line in f:
            d = json.loads(line)
            if d["modality"] == "text":
                caption_text_to_did[d["txt"]] = d["did"]

    return qid_to_image_did, qid_to_caption, caption_text_to_did


def build_matched_table(qid_to_image_did, qid_to_caption, caption_text_to_did):
    """Returns arrays of (image_hashed_id, caption_hashed_id) for every qid with
    both a GT image and a resolvable caption did, plus a reverse map
    image_hashed_id -> set of all GT caption_hashed_ids (for hard-negative
    exclusion), plus its inverse caption_hashed_id -> image_hashed_id (for
    enforcing "different image" when sampling Random pairs, per the spec)."""
    image_ids, caption_ids = [], []
    image_to_gt_captions = {}
    for qid, caption in qid_to_caption.items():
        image_did = qid_to_image_did.get(qid)
        caption_did = caption_text_to_did.get(caption)
        if image_did is None or caption_did is None:
            continue
        img_h = hash_did(image_did)
        cap_h = hash_did(caption_did)
        image_ids.append(img_h)
        caption_ids.append(cap_h)
        image_to_gt_captions.setdefault(img_h, set()).add(cap_h)
    caption_to_image = {cap_h: img_h for img_h, caps in image_to_gt_captions.items() for cap_h in caps}
    return np.array(image_ids), np.array(caption_ids), image_to_gt_captions, caption_to_image


def sample_random_captions(image_ids, caption_ids_all, caption_to_image, rng, max_retries=50):
    """Sample one caption per entry in image_ids, uniformly at random from
    caption_ids_all, rejecting (and redrawing) any draw whose true GT image
    matches that entry's sampled image -- per the spec: "Random pairs: COCO
    image + randomly sampled caption from a different image." A draw that's
    a different array index but the SAME image (common, since a single
    image's ~5 GT captions are typically adjacent in caption_ids_all) would
    silently be a disguised Matched pair, not a Random one."""
    n = len(image_ids)
    pool_size = len(caption_ids_all)
    chosen_idx = rng.integers(0, pool_size, size=n)
    pending = np.arange(n)

    def find_bad(positions):
        candidate_caption_ids = caption_ids_all[chosen_idx[positions]]
        candidate_image_of_caption = np.array([caption_to_image.get(int(c), -1) for c in candidate_caption_ids])
        return positions[candidate_image_of_caption == image_ids[positions]]

    pending = find_bad(pending)
    for _ in range(max_retries):
        if len(pending) == 0:
            break
        chosen_idx[pending] = rng.integers(0, pool_size, size=len(pending))
        pending = find_bad(pending)
    if len(pending) > 0:
        print(f"Warning: sample_random_captions still has {len(pending)} unresolved same-image collisions "
              f"after {max_retries} retries; accepting them as-is (same-image collision probability is "
              f"~0.001% per draw, so this should be rare/never).")
    return caption_ids_all[chosen_idx]


def mine_hard_negatives(pt_path, sampled_image_hashed_ids, image_to_gt_captions, rng):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    id_to_index = data["id_to_index"]
    img_emb = data["img"].float()
    text_emb = data["text"].float()
    text_mask = data["text_mask"].bool()

    caption_row_idx = torch.nonzero(text_mask, as_tuple=True)[0]
    caption_embs = text_emb[caption_row_idx]
    caption_embs = caption_embs / caption_embs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    index_to_id = {v: k for k, v in id_to_index.items()}
    caption_hashed_ids = np.array([index_to_id[int(i)] for i in caption_row_idx])
    caption_id_to_col = {h: i for i, h in enumerate(caption_hashed_ids)}

    hard_neg_caption_ids = np.zeros(len(sampled_image_hashed_ids), dtype=np.int64)
    for start in range(0, len(sampled_image_hashed_ids), HARDNEG_BATCH):
        batch_ids = sampled_image_hashed_ids[start : start + HARDNEG_BATCH]
        rows = [id_to_index[int(h)] for h in batch_ids]
        batch_img = img_emb[rows]
        batch_img = batch_img / batch_img.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = batch_img @ caption_embs.T  # [batch, n_captions]
        for i, img_h in enumerate(batch_ids):
            for gt_cap in image_to_gt_captions.get(int(img_h), ()):
                col = caption_id_to_col.get(gt_cap)
                if col is not None:
                    sim[i, col] = -1.0
        top_idx = sim.argmax(dim=1).numpy()
        hard_neg_caption_ids[start : start + HARDNEG_BATCH] = caption_hashed_ids[top_idx]
    return hard_neg_caption_ids


def pair_metrics(codes_a, codes_b):
    spl = shared_prefix_length(codes_a, codes_b)
    ham = hamming_distance(codes_a, codes_b)
    agree = per_level_agreement(codes_a, codes_b)
    identical = (ham == 0).mean()
    return spl, ham, agree, identical


def write_significance_table(spl_by_type, out_path):
    """Mann-Whitney U test on shared-prefix-length distributions, one-sided
    (a > b), plus the common-language effect size (probability of
    superiority = U / (n_a * n_b)): for a random pair from each group, the
    probability group a's shared-prefix-length exceeds group b's."""
    import csv
    comparisons = [("Matched", "Random"), ("Hard-negative", "Matched")]
    rows = [["comparison", "U_statistic", "p_value", "prob_superiority", "median_a", "median_b"]]
    for a, b in comparisons:
        x, y = spl_by_type[a], spl_by_type[b]
        u_stat, p_value = mannwhitneyu(x, y, alternative="greater")
        prob_superiority = u_stat / (len(x) * len(y))
        rows.append([f"{a} > {b}", round(float(u_stat), 1), f"{p_value:.3e}", round(float(prob_superiority), 4),
                     float(np.median(x)), float(np.median(y))])
        print(f"[significance] {a} > {b}: U={u_stat:.1f} p={p_value:.3e} P(superiority)={prob_superiority:.4f}")
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", required=True)
    parser.add_argument("--gen_code_dir", default=GEN_CODE_DIR_DEFAULT)
    parser.add_argument("--out_root", default="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis")
    args = parser.parse_args()

    gen_code_dir = os.path.join(args.genir_dir, args.gen_code_dir)
    dirs = output_dirs(args.out_root)
    rng = np.random.default_rng(RNG_SEED)

    sorted_ids, sorted_codes = build_code_lookup(gen_code_dir)
    qid_to_image_did, qid_to_caption, caption_text_to_did = load_qrels_and_queries(args.mbeir_data_dir)
    image_ids_all, caption_ids_all, image_to_gt_captions, caption_to_image = build_matched_table(qid_to_image_did, qid_to_caption, caption_text_to_did)
    print(f"Resolvable matched pairs: {len(image_ids_all)} (out of {len(qid_to_caption)} text queries)")

    n = min(N_PAIRS, len(image_ids_all))
    sample_idx = rng.choice(len(image_ids_all), size=n, replace=False)
    image_ids = image_ids_all[sample_idx]
    matched_caption_ids = caption_ids_all[sample_idx]

    # random pairs: uniformly sampled caption from a DIFFERENT image (per spec),
    # enforced via rejection sampling against the true GT image-of-caption map.
    random_caption_ids = sample_random_captions(image_ids, caption_ids_all, caption_to_image, rng)

    pt_path = os.path.join(args.genir_dir, "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_IT_dict.pt")
    hardneg_caption_ids = mine_hard_negatives(pt_path, image_ids, image_to_gt_captions, rng)

    img_pos, _ = lookup_codes(sorted_ids, sorted_codes, image_ids)
    img_codes = sorted_codes[img_pos][:, 1:]

    table3_rows = [["pair_type", "mean_shared_prefix", "std_shared_prefix", "mean_hamming", "std_hamming", "pct_identical"]]
    agreement_by_type = {}
    spl_by_type = {}

    for label, caption_ids in (("Matched", matched_caption_ids), ("Random", random_caption_ids), ("Hard-negative", hardneg_caption_ids)):
        cap_pos, _ = lookup_codes(sorted_ids, sorted_codes, caption_ids)
        cap_codes = sorted_codes[cap_pos][:, 1:]
        spl, ham, agree, identical = pair_metrics(img_codes, cap_codes)
        table3_rows.append([label, round(spl.mean(), 4), round(spl.std(), 4), round(ham.mean(), 4), round(ham.std(), 4), round(100 * identical, 4)])
        agreement_by_type[label] = agree
        spl_by_type[label] = spl
        print(f"[{label}] mean_shared_prefix={spl.mean():.3f} mean_hamming={ham.mean():.3f} pct_identical={100*identical:.3f}%")

    table3_path = os.path.join(dirs["tables"], "table3_alignment.csv")
    with open(table3_path, "w") as f:
        import csv
        csv.writer(f).writerows(table3_rows)
    print(f"Table3 written to {table3_path}")

    sig_path = os.path.join(dirs["tables"], "table3_significance.csv")
    write_significance_table(spl_by_type, sig_path)
    print(f"Significance table written to {sig_path}")

    plot_agreement_bars(agreement_by_type, os.path.join(dirs["figures"], "analysis3_agreement_bars.png"))
    plot_prefix_violin(spl_by_type, os.path.join(dirs["figures"], "analysis3_prefix_violin.png"))
    plot_pca_scatter(pt_path, image_ids, matched_caption_ids, sorted_ids, sorted_codes,
                      os.path.join(dirs["figures"], "analysis3_pca_scatter.png"), rng)


def plot_agreement_bars(agreement_by_type, out_path):
    levels = np.arange(1, 9)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, agree) in enumerate(agreement_by_type.items()):
        ax.bar(levels + (i - 1) * width, agree, width=width, label=label)
    ax.set_xlabel("RQ semantic level")
    ax.set_ylabel("code-agreement probability")
    ax.set_title("Cross-modal code agreement by level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prefix_violin(spl_by_type, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(spl_by_type.keys())
    ax.violinplot([spl_by_type[l] for l in labels], showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("shared-prefix length (out of 8 semantic levels)")
    ax.set_title("Shared-prefix-length distribution by pair type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(pt_path, image_ids, caption_ids, sorted_ids, sorted_codes, out_path, rng, n_sample=3000):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    id_to_index = data["id_to_index"]
    img_emb = data["img"].float().numpy()
    text_emb = data["text"].float().numpy()

    n = min(n_sample, len(image_ids))
    idx = rng.choice(len(image_ids), size=n, replace=False)
    img_rows = np.array([id_to_index[int(h)] for h in image_ids[idx]])
    cap_rows = np.array([id_to_index[int(h)] for h in caption_ids[idx]])

    embs = np.concatenate([img_emb[img_rows], text_emb[cap_rows]], axis=0)
    modality = np.array(["image"] * n + ["caption"] * n)

    pos_img, _ = lookup_codes(sorted_ids, sorted_codes, image_ids[idx])
    pos_cap, _ = lookup_codes(sorted_ids, sorted_codes, caption_ids[idx])
    level1_code = np.concatenate([sorted_codes[pos_img][:, 1], sorted_codes[pos_cap][:, 1]])

    coords = PCA(n_components=2).fit_transform(embs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m, marker in (("image", "o"), ("caption", "^")):
        mask = modality == m
        axes[0].scatter(coords[mask, 0], coords[mask, 1], marker=marker, alpha=0.4, s=8, label=m)
    axes[0].set_title("PCA of raw CLIP-SF embeddings, colored by modality")
    axes[0].legend()

    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=level1_code, cmap="viridis", s=8, alpha=0.6)
    axes[1].set_title("Same PCA, colored by level-1 semantic code\n(code index, not an ordinal quantity)")
    fig.colorbar(sc, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
