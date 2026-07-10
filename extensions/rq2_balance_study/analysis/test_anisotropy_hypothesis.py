"""
Mechanism forensics, Fable's ladder item 3: test whether text-embedding
anisotropy interacts badly with per-sample entropy pressure. CLIP text
embeddings are known to occupy a narrower cone of representation space than
image embeddings (higher average pairwise cosine similarity = more
anisotropic); per-sample entropy pressure pushes every embedding toward
equidistance from all codes, which may shatter an already-tight text cluster
far more than a naturally-spread image cluster.

Measures, per COCO variant (vanilla/weak/medium/strong), on the fused
Combiner output ("encode", the space the RQ actually quantizes -- not raw
CLIP):
1. Anisotropy: mean pairwise cosine similarity within a random sample of
   image-only encode vectors vs. text-only encode vectors (sampled to avoid
   O(n^2) on ~700K candidates).
2. Per-modality level-1 code entropy: relative entropy of the level-1 code
   distribution, computed SEPARATELY for image rows and text rows (extends
   the existing level_stats machinery, which was previously only ever
   computed on the full, mixed-modality candidate set).

No new training. Reuses load_quantizer/run_inference unchanged.

Run with: PYTHONPATH=<repo>/src/common:<repo>/src python test_anisotropy_hypothesis.py --genir_dir <repo>
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from rq1_coco_variant_diagnostics import load_quantizer, run_inference, DATASET_CONFIGS  # noqa: E402
from rq_analysis_codebook_utilization import level_stats  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from data.mbeir_dataset import MBEIRDictCandDataset  # noqa: E402


def mean_pairwise_cosine(x, n_sample=3000, seed=42):
    """x: [N, D] tensor. Samples n_sample rows, returns mean off-diagonal
    pairwise cosine similarity -- higher = more anisotropic (tightly
    clustered in one direction of the embedding space)."""
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    idx = torch.randperm(n, generator=g)[:min(n_sample, n)]
    sample = F.normalize(x[idx], dim=-1)
    sim = sample @ sample.T
    n_s = sim.shape[0]
    off_diag_sum = sim.sum() - torch.diagonal(sim).sum()
    return float(off_diag_sum / (n_s * (n_s - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS["coco"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_dict_dir = os.path.join(args.genir_dir, cfg["pool_dict_rel_path"])

    rows = []
    for variant, ckpt_path in cfg["variants"].items():
        print(f"\n=== coco / {variant} ===")
        dataset = MBEIRDictCandDataset(
            mbeir_data_dir=args.mbeir_data_dir,
            cand_pool_path=cfg["cand_pool_path"],
            pool_dict_dir=pool_dict_dir,
            print_config=False,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        model = load_quantizer(args.genir_dir, ckpt_path, device)
        codes, modality, recon_mse, recon_cos, encode, quant, img_mask, txt_mask = run_inference(model, loader, device)

        encode_t = torch.from_numpy(encode)
        img_mask_bool = img_mask.astype(bool)
        txt_mask_bool = txt_mask.astype(bool)

        img_encode = encode_t[img_mask_bool]
        txt_encode = encode_t[txt_mask_bool]
        img_aniso = mean_pairwise_cosine(img_encode)
        txt_aniso = mean_pairwise_cosine(txt_encode)
        print(f"  anisotropy (mean pairwise cos sim): image={img_aniso:.4f}  text={txt_aniso:.4f}  "
              f"(higher = more tightly clustered / anisotropic)")

        level1_codes = codes[:, 0]
        img_level1_stats = level_stats(level1_codes[img_mask_bool])
        txt_level1_stats = level_stats(level1_codes[txt_mask_bool])
        img_rel_entropy = img_level1_stats["entropy"] / img_level1_stats["max_entropy"]
        txt_rel_entropy = txt_level1_stats["entropy"] / txt_level1_stats["max_entropy"]
        print(f"  level-1 rel. entropy: image={img_rel_entropy:.4f}  text={txt_rel_entropy:.4f}")

        rows.append({
            "variant": variant, "img_anisotropy": round(img_aniso, 4), "txt_anisotropy": round(txt_aniso, 4),
            "img_level1_rel_entropy": round(img_rel_entropy, 4), "txt_level1_rel_entropy": round(txt_rel_entropy, 4),
        })

    print("\n=== Summary (markdown) ===")
    print("| Variant | Image anisotropy | Text anisotropy | Image L1 rel.entropy | Text L1 rel.entropy |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['img_anisotropy']:.3f} | {r['txt_anisotropy']:.3f} | "
              f"{r['img_level1_rel_entropy']:.3f} | {r['txt_level1_rel_entropy']:.3f} |")

    print("\nInterpretation: if text_anisotropy > image_anisotropy for vanilla (text embeddings "
          "naturally more tightly clustered), and text L1 rel.entropy drops faster than image's "
          "as lambda increases, that supports the hypothesis that per-sample entropy pressure "
          "disproportionately damages the already-tighter text cluster.")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
