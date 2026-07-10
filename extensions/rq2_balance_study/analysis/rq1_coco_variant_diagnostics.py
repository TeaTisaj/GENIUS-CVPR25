"""
RQ1 ID-property + reconstruction diagnostic for the COCO balance-regularization
matrix (vanilla/weak/medium/strong Stage-1 quantizers). See rq-new-plan.txt
RQ1/RQ2 and the plan doc for context. Unlike rq_analysis_codebook_utilization.py
(which analyzes the released official quantizer's codes across 6 datasets),
this script generates fresh codes for 4 NEW COCO-only quantizers against the
COCO-local candidate pool Stage-1 itself trained on, and additionally computes
reconstruction MSE / cosine similarity (not needed by the official-quantizer
analysis, which never has access to a comparison point across balance strengths).

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python rq1_coco_variant_diagnostics.py --genir_dir <repo>
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rq_analysis_codebook_utilization import level_stats, LEVELS, VOCAB_SIZE
from rq_analysis_utils import output_dirs

from data.mbeir_dataset import MBEIRDictCandDataset
from models.residual_quantization.residual_quantization import RQ

# Per-dataset variant checkpoints + candidate pool/embeddings paths. --dataset selects one of
# these at runtime (default "coco", preserving the original single-dataset behavior). Added for
# RQ3 (FashionIQ) without changing COCO's already-verified call signature.
DATASET_CONFIGS = {
    "coco": {
        "variants": {
            "vanilla": "checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth",
            "weak": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoWeak/rq_clip_large_epoch_140.pth",
            "medium": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoMedium/rq_clip_large_epoch_140.pth",
            "strong": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoStrong/rq_clip_large_epoch_140.pth",
        },
        "cand_pool_path": "cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl",
        "pool_dict_rel_path": "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_IT_dict.pt",
    },
    "fashioniq": {
        "variants": {
            "vanilla": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/FashioniqVanilla/rq_clip_large_epoch_170.pth",
            "weak": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/FashioniqWeak/rq_clip_large_epoch_170.pth",
            "medium": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/FashioniqMedium/rq_clip_large_epoch_170.pth",
            "strong": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/FashioniqStrong/rq_clip_large_epoch_170.pth",
        },
        "cand_pool_path": "cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl",
        "pool_dict_rel_path": "extracted_embed/CLIP_SF/cand/cand_pool_fashioniq_task7_IT_dict.pt",
    },
}


def resolve_checkpoint_path(genir_dir, ckpt_path):
    return ckpt_path if os.path.isabs(ckpt_path) else os.path.join(genir_dir, ckpt_path)


def load_quantizer(genir_dir, ckpt_path, device):
    ckpt = torch.load(resolve_checkpoint_path(genir_dir, ckpt_path), map_location="cpu", weights_only=False)
    model = RQ(config=ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    """Returns semantic_codes [N, 8], modality_code [N], recon_mse [N], recon_cos_sim [N],
    encode [N, D], quant [N, D] (all numpy). encode/quant are kept (not just their per-batch
    MSE/cos-sim) because the neighbor-preservation metric needs the actual vectors, not just
    a pointwise distance. modality_code (code[0]) is returned separately from the semantic
    levels: it is never touched by balance_loss's diversity regularization (only levels 1-8
    are), so its distribution is a diagnostic in its own right (e.g. to check whether a
    variant collapses image/text onto the same modality code) rather than an ID-structure
    metric to fold into the level 1-8 aggregates."""
    all_codes, all_modality, all_mse, all_cos, all_encode, all_quant = [], [], [], [], [], []
    all_img_mask, all_txt_mask = [], []
    for pool, id_list in loader:
        img_mask = pool["img_mask"].view(-1).unsqueeze(-1).float().to(device, non_blocking=True)
        txt_mask = pool["txt_mask"].view(-1).unsqueeze(-1).float().to(device, non_blocking=True)
        img_emb = pool["img_emb"].view(-1, pool["img_emb"].size(-1)).float().to(device, non_blocking=True)
        txt_emb = pool["txt_emb"].view(-1, pool["txt_emb"].size(-1)).float().to(device, non_blocking=True)

        output = model.inference(img_emb, txt_emb, img_mask, txt_mask)
        code, quant, encode = output["code"], output["quant"], output["encode"]
        if code.ndim == 1:  # batch size 1 squeeze edge case (residual_quantization.py:555-557)
            code, quant, encode = code.unsqueeze(0), quant.unsqueeze(0), encode.unsqueeze(0)

        mse = ((encode - quant) ** 2).mean(dim=-1)
        cos = F.cosine_similarity(encode, quant, dim=-1)

        all_codes.append(code[:, 1:].cpu().numpy())  # drop modality token, keep 8 semantic levels
        all_modality.append(code[:, 0].cpu().numpy())
        all_mse.append(mse.cpu().numpy())
        all_cos.append(cos.cpu().numpy())
        all_encode.append(encode.cpu().numpy())
        all_quant.append(quant.cpu().numpy())
        all_img_mask.append(img_mask.view(-1).cpu().numpy())
        all_txt_mask.append(txt_mask.view(-1).cpu().numpy())

    return (
        np.concatenate(all_codes, axis=0),
        np.concatenate(all_modality),
        np.concatenate(all_mse),
        np.concatenate(all_cos),
        np.concatenate(all_encode, axis=0),
        np.concatenate(all_quant, axis=0),
        np.concatenate(all_img_mask),
        np.concatenate(all_txt_mask),
    )


def modality_distribution(modality_codes, img_mask, txt_mask):
    """Value-count breakdown of the modality-indicator level (code[0]), split by the
    candidate's true modality (image vs. text) so collapse/inversion patterns are visible
    (e.g. a variant assigning code[0]=0 to 100% of BOTH image and text candidates, vs. one
    that assigns 0/1 consistently but swapped relative to the 0=image/1=text convention)."""
    def counts_for(mask):
        vals, cnts = np.unique(modality_codes[mask], return_counts=True)
        total = cnts.sum()
        return {int(v): (int(c), round(100.0 * c / total, 2)) for v, c in zip(vals, cnts)}

    img_sel = img_mask.astype(bool)
    txt_sel = txt_mask.astype(bool) & ~img_sel  # MBEIR candidates are image-only or text-only here
    return {
        "image_candidates": counts_for(img_sel) if img_sel.any() else {},
        "text_candidates": counts_for(txt_sel) if txt_sel.any() else {},
    }


def neighbor_preservation_at_k(encode, quant, k=10, n_queries=5000, n_corpus=100_000, seed=42):
    """RQ2 'semantic fidelity: preservation of local embedding neighbours'. For a sample of
    query candidates, compares their top-k nearest neighbors (by cosine similarity) within a
    random comparison subsample, once in the original (pre-quantization) embedding space and
    once in the reconstructed (post-quantization) space. Score per query = |intersection| / k;
    returned value is the mean over all sampled queries.

    Deliberately compares against a random ~100K-candidate subsample, not the full pool: this is
    a local-structure metric, not a full-corpus one, and the full pool would need a much larger
    (and unnecessary) similarity matrix for the same signal.
    """
    rng = np.random.default_rng(seed)
    n = encode.shape[0]
    query_idx = rng.choice(n, size=min(n_queries, n), replace=False)
    corpus_idx = rng.choice(n, size=min(n_corpus, n), replace=False)

    def normalized(x):
        x = torch.from_numpy(x)
        return F.normalize(x, dim=-1)

    enc_q, enc_c = normalized(encode[query_idx]), normalized(encode[corpus_idx])
    qnt_q, qnt_c = normalized(quant[query_idx]), normalized(quant[corpus_idx])

    enc_topk = torch.topk(enc_q @ enc_c.T, k=k, dim=-1).indices
    qnt_topk = torch.topk(qnt_q @ qnt_c.T, k=k, dim=-1).indices

    overlaps = []
    for i in range(enc_topk.shape[0]):
        overlap = len(set(enc_topk[i].tolist()) & set(qnt_topk[i].tolist()))
        overlaps.append(overlap / k)
    return float(np.mean(overlaps))


def prefix_collision_rate(codes, depth):
    """Fraction of candidates whose depth-`depth` ID prefix is shared with >=1 other candidate."""
    prefixes = codes[:, :depth]
    _, inverse, counts = np.unique(prefixes, axis=0, return_inverse=True, return_counts=True)
    group_size = counts[inverse]
    return float((group_size > 1).mean())


def full_id_collision_rate(codes):
    _, counts = np.unique(codes, axis=0, return_counts=True)
    n = codes.shape[0]
    n_unique_rows = len(counts)
    n_colliding = n - (counts == 1).sum()
    return float(n_colliding / n), n_unique_rows


def bucket_size_stats(codes, depth):
    prefixes = codes[:, :depth]
    _, counts = np.unique(prefixes, axis=0, return_counts=True)
    return {"mean": float(counts.mean()), "median": float(np.median(counts)), "max": int(counts.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="coco")
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
    parser.add_argument("--out_root", default=None, help="Defaults to <dataset>-specific rq_analysis dir")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subsample_size", type=int, default=None,
                         help="If set, subsample the candidate pool to this many items "
                              "(drawn once, before the variant loop, reused for all "
                              "variants in this run) before computing ID-structure stats. "
                              "Default None preserves full-pool behavior unchanged.")
    parser.add_argument("--subsample_seed", type=int, default=74,
                         help="Seed for the one-time subsample draw. Deliberately distinct "
                              "from the unrelated hardcoded seed=42 in "
                              "neighbor_preservation_at_k (different purpose, independent "
                              "RNG instance, kept distinct for writeup clarity).")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    out_root = args.out_root or f"/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_{args.dataset}_variants"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = output_dirs(out_root)

    pool_dict_dir = os.path.join(args.genir_dir, cfg["pool_dict_rel_path"])
    assert os.path.exists(pool_dict_dir), f"Missing pool dict: {pool_dict_dir}"

    # Pool-size confound check (RQ3 robustness addition): subsample the candidate pool down
    # to --subsample_size items, drawn ONCE here (before the variant loop) and reused
    # identically for every variant. All variants share the same underlying candidate pool
    # (same cand_pool_path), differing only in which RQ checkpoint quantizes them -- drawing
    # a fresh random subsample per variant would add avoidable between-variant noise
    # unrelated to balance-regularization strength, undermining the point of the check.
    # Counting JSONL lines (not constructing a dataset) avoids loading the full multi-GB
    # Stage-0 pool-embedding dict just to learn N.
    subsample_indices = None
    if args.subsample_size is not None:
        cand_pool_full_path = os.path.join(args.mbeir_data_dir, cfg["cand_pool_path"])
        with open(cand_pool_full_path) as f:
            full_n = sum(1 for _ in f)
        subsample_indices = np.random.default_rng(args.subsample_seed).choice(
            full_n, size=args.subsample_size, replace=False
        )
        print(f"Pool-size confound check: subsampling {args.subsample_size}/{full_n} candidates "
              f"(seed={args.subsample_seed}), applied identically across all variants.")

    structure_rows = []
    summary_rows = []
    modality_rows = []

    for variant, ckpt_path in cfg["variants"].items():
        print(f"\n=== Variant: {variant} ===")
        dataset = MBEIRDictCandDataset(
            mbeir_data_dir=args.mbeir_data_dir,
            cand_pool_path=cfg["cand_pool_path"],
            pool_dict_dir=pool_dict_dir,
            print_config=True,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        model = load_quantizer(args.genir_dir, ckpt_path, device)
        codes, modality, recon_mse, recon_cos, encode, quant, img_mask, txt_mask = run_inference(model, loader, device)

        if subsample_indices is not None:
            codes = codes[subsample_indices]
            modality = modality[subsample_indices]
            recon_mse = recon_mse[subsample_indices]
            recon_cos = recon_cos[subsample_indices]
            encode = encode[subsample_indices]
            quant = quant[subsample_indices]
            img_mask = img_mask[subsample_indices]
            txt_mask = txt_mask[subsample_indices]

        n = codes.shape[0]
        print(f"N candidates: {n}")

        mod_dist = modality_distribution(modality, img_mask, txt_mask)
        print(f"  modality_code[0] distribution: image={mod_dist['image_candidates']}  text={mod_dist['text_candidates']}")
        for cand_type, dist in mod_dist.items():
            for code_val, (count, pct) in sorted(dist.items()):
                modality_rows.append([variant, cand_type, code_val, count, pct])

        neighbor_preservation = neighbor_preservation_at_k(encode, quant)
        print(f"  neighbor_preservation_at_10={neighbor_preservation:.4f} (N=4 across variants; descriptive only)")

        for level in LEVELS:
            stats = level_stats(codes[:, level - 1])
            top1_code, top1_freq = stats["top5"][0]
            structure_rows.append([
                variant, level, stats["used"], round(stats["util_pct"], 3), round(stats["unused_pct"], 3),
                round(stats["entropy"], 4), round(stats["max_entropy"], 4),
                round(stats["entropy"] / stats["max_entropy"], 4),
                round(stats["perplexity"], 1), round(stats["zipf_exp"], 4), top1_code, top1_freq,
            ])

        full_collision, n_unique = full_id_collision_rate(codes)
        row = {
            "variant": variant,
            "n_candidates": n,
            "n_unique_full_ids": n_unique,
            "full_id_collision_rate": round(full_collision, 6),
            "recon_mse_mean": round(float(recon_mse.mean()), 6),
            "recon_cos_sim_mean": round(float(recon_cos.mean()), 6),
            "neighbor_preservation_at_10": round(neighbor_preservation, 6),
        }
        for depth in (1, 2, 4, 8):
            row[f"collision_rate_depth{depth}"] = round(prefix_collision_rate(codes, depth), 6)
            bucket = bucket_size_stats(codes, depth)
            row[f"bucket_mean_depth{depth}"] = round(bucket["mean"], 3)
            row[f"bucket_median_depth{depth}"] = bucket["median"]
            row[f"bucket_max_depth{depth}"] = bucket["max"]
        summary_rows.append(row)
        print(f"  full_id_collision_rate={full_collision:.4f}  recon_mse={recon_mse.mean():.6f}  recon_cos_sim={recon_cos.mean():.4f}")

        del model
        torch.cuda.empty_cache()

    structure_csv = os.path.join(dirs["tables"], "table1_id_structure.csv")
    with open(structure_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "level", "used_codes", "utilization_pct", "unused_pct",
            "entropy_bits", "max_entropy_bits", "normalized_entropy", "perplexity",
            "zipf_exponent", "top1_code", "top1_freq",
        ])
        writer.writerows(structure_rows)
    print(f"\nTable 1 written to {structure_csv}")

    summary_csv = os.path.join(dirs["tables"], "table_collision_recon_summary.csv")
    fieldnames = list(summary_rows[0].keys())
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Collision/reconstruction summary written to {summary_csv}")

    modality_csv = os.path.join(dirs["tables"], "table_modality_distribution.csv")
    with open(modality_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "candidate_type", "modality_code_value", "count", "pct"])
        writer.writerows(modality_rows)
    print(f"Modality-code[0] distribution written to {modality_csv}")


if __name__ == "__main__":
    main()
