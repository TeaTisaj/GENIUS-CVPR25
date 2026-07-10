"""
Component 8 (retrieval-aware RQ refinement feasibility study) -- clarifies
what the CURRENT balance regularization actually measures (mentor's point 1,
"clarify the current balance regularization" -- silently dropped from every
prior draft of the plan this extends; see the plan file's Component 8 section
for the full finding this script's output is meant to support).

**The finding this script demonstrates:** `vector_quantize_pytorch`'s
diversity/"balance" loss computes
    prob = (distances * temperature).softmax(dim=-1)
    avg_prob = reduce(prob, '... n l -> n l', 'mean')
    codebook_diversity_loss = -entropy(avg_prob).mean()
(`vector_quantize_pytorch.py:1243-1246`). The `'...'` dims collapsed by that
`reduce(..., 'mean')` are ordinarily the BATCH dimension -- which is what
"codebook usage balance" is supposed to mean (are all codes used roughly
equally ACROSS THE CORPUS?). But this project always calls
`self.residual_rq(encode_feature.unsqueeze(0), ...)`
(`residual_quantization.py`, both the original call site and this study's new
`_quantize_with_breakdown`) -- an explicit `.unsqueeze(0)` that puts every
query+pool embedding into the SEQUENCE dimension `n`, with the batch dim
reduced to a singleton. Under this calling convention, `avg_prob` is NOT a
batch-usage histogram at all -- it degenerates to each INDIVIDUAL sample's own
softmax-assignment distribution, and the "diversity loss" becomes, per
sample, a push toward higher-entropy (more equidistant) placement relative to
the codebook's Voronoi cells -- a materially different regularizer than
"encourage broad code usage across the corpus."

This script computes and reports, per RQ semantic level, for each of the
COCO vanilla/weak/medium/strong Stage-1 checkpoints (`rq2_balance_study`):
  (a) BATCH-USAGE entropy: build a histogram of which HARD code (`Is[:, level]`,
      the actual assigned code index) each embedding in a large batch picks;
      Shannon entropy (bits) over that histogram. This is what "balance
      regularization" was assumed to optimize.
  (b) MEAN PER-SAMPLE assignment-softmax entropy: what the library, called
      this way, actually optimizes -- computed as `-diversity_losses[level]`
      from `RQ._quantize_with_breakdown` (Component 3), which is exactly
      `entropy(avg_prob).mean()` per the library's own formula above (i.e.
      `codebook_diversity_loss = -that value`).

...and reports the Pearson correlation between (a) and (b) across the 4
checkpoints x 8 semantic levels, to show whether these two quantities are
correlated, anti-correlated, or independent.

**Implementation note on getting real (non-zero) `diversity_losses` for every
checkpoint, including `vanilla` (trained with `balance_loss_config.enabled:
false`, i.e. `codebook_diversity_loss_weight == 0` at construction time,
which gates the library's `has_codebook_diversity_loss` flag to False and
would otherwise make `loss_breakdown.codebook_diversity` a constant zero for
that checkpoint's OWN config):** this script constructs a temporary `RQ` shell
per checkpoint using the checkpoint's own `codebook_config` but FORCING
`balance_loss_config.enabled=True` (an arbitrary positive `lambda`, e.g. 1.0,
and `temperature=100.0` matching the library/training default) purely so the
underlying `VectorQuantize` layers compute a real (b) value regardless of
what lambda the checkpoint was originally trained with -- `codebook_diversity_loss_weight`/
`temperature` are plain constructor hyperparameters, not stored buffers, so
`load_state_dict(strict=True)` from the real checkpoint still faithfully
restores every trained buffer/parameter (encoder weights, `_codebook.embed`,
`cluster_size`, etc.) into this shell.

**Note this script runs `_quantize_with_breakdown` in `.train()` mode** (the
library's loss computations, including `codebook_diversity_loss`, are
skipped entirely in `.eval()` mode, per Component 3's own finding) -- this
DOES trigger a one-time EMA codebook mutation in the in-memory temporary
model, which is harmless here since this script never saves the model back
to disk and constructs a fresh temporary shell per checkpoint.

No GPU training required -- this only loads existing checkpoints and runs a
few forward passes. Runs on CPU or GPU (auto-detected).

Usage:
  python clarify_balance_regularization.py --genir_dir <repo> \
      --mbeir_data_dir <mbeir_data_dir> [--n_samples 4096]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data.mbeir_dataset import MBEIRDictCandDataset  # noqa: E402
from models.residual_quantization.residual_quantization import RQ  # noqa: E402

# Same 4 COCO Stage-1 checkpoints used by
# extensions/rq2_balance_study/analysis/rq1_coco_variant_diagnostics.py
# (hardcoded independently here, not imported, to keep this script standalone
# and not coupled to that extension's internal module layout).
VARIANT_CHECKPOINTS = {
    "vanilla": "checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth",
    "weak": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoWeak/rq_clip_large_epoch_140.pth",
    "medium": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoMedium/rq_clip_large_epoch_140.pth",
    "strong": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoStrong/rq_clip_large_epoch_140.pth",
}
CAND_POOL_JSONL = "cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl"
POOL_DICT_REL_PATH = "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_IT_dict.pt"


def resolve(genir_dir, p):
    return p if os.path.isabs(p) else os.path.join(genir_dir, p)


def batch_usage_entropy_bits(codes_1d, vocab_size):
    """(a) BATCH-USAGE entropy: histogram of hard code assignments across the
    batch, Shannon entropy in bits."""
    counts = np.bincount(codes_1d, minlength=vocab_size).astype(np.float64)
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def load_shell_with_forced_diversity(genir_dir, ckpt_path, forced_lambda=1.0, forced_temp=100.0):
    ckpt = torch.load(resolve(genir_dir, ckpt_path), map_location="cpu", weights_only=False)
    shell_config = OmegaConf.create(OmegaConf.to_container(ckpt["config"], resolve=True))
    shell_config.balance_loss_config = {"enabled": True, "lambda": forced_lambda, "temperature": forced_temp}
    model = RQ(config=shell_config)
    model.load_state_dict(ckpt["model"], strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=4096,
                         help="Number of COCO candidates to sample for this analysis.")
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_dict_dir = resolve(args.genir_dir, POOL_DICT_REL_PATH)
    dataset = MBEIRDictCandDataset(
        mbeir_data_dir=args.mbeir_data_dir,
        cand_pool_path=CAND_POOL_JSONL,
        pool_dict_dir=pool_dict_dir,
        print_config=False,
    )

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)

    pool_list = [dataset[i][0] for i in indices]
    img_emb = torch.stack([p["img_emb"] for p in pool_list]).float()
    txt_emb = torch.stack([p["txt_emb"] for p in pool_list]).float()
    img_mask = torch.stack([p["img_mask"] for p in pool_list]).float().unsqueeze(-1)
    txt_mask = torch.stack([p["txt_mask"] for p in pool_list]).float().unsqueeze(-1)

    all_rows = []  # (variant, level, batch_usage_entropy, mean_per_sample_entropy)

    for variant, ckpt_rel_path in VARIANT_CHECKPOINTS.items():
        ckpt_path = resolve(args.genir_dir, ckpt_rel_path)
        if not os.path.exists(ckpt_path):
            print(f"[{variant}] checkpoint not found at {ckpt_path}, skipping.")
            continue

        print(f"\n=== Variant: {variant} ===")
        model = load_shell_with_forced_diversity(args.genir_dir, ckpt_rel_path)
        model.to(device)
        model.train()  # loss computation (incl. codebook_diversity_loss) is train()-only

        # torch.no_grad(): this is a read-only diagnostic, never backpropagated.
        # Values are unaffected (the straight-through-estimator wrapping that
        # `input_requires_grad` gates only changes the backward graph, not the
        # forward VALUE of `quantize`).
        with torch.no_grad():
            encode_feature = model.encoder(
                img_emb.to(device), txt_emb.to(device), img_mask.to(device), txt_mask.to(device)
            )
            encode_feature = F.normalize(encode_feature)

            _quant, Is, _commit, diversity_losses, _level_quantized = model._quantize_with_breakdown(
                encode_feature.unsqueeze(0)
            )
        Is = Is.squeeze(0).cpu().numpy()

        num_semantic_levels = model.num_semantic_levels
        start_level = 1 if model.modality_index else 0
        for l in range(num_semantic_levels):
            level_idx = start_level + l
            codes_1d = Is[:, level_idx].astype(np.int64)
            usage_entropy = batch_usage_entropy_bits(codes_1d, model.codebook_vocab)
            mean_per_sample_entropy = -diversity_losses[level_idx].item()  # -codebook_diversity_loss = entropy(avg_prob).mean()
            print(f"  level={l + 1}: batch_usage_entropy={usage_entropy:.4f} bits, "
                  f"mean_per_sample_assignment_entropy={mean_per_sample_entropy:.4f}")
            all_rows.append((variant, l + 1, usage_entropy, mean_per_sample_entropy))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(all_rows) >= 2:
        usage_vals = np.array([r[2] for r in all_rows])
        sample_vals = np.array([r[3] for r in all_rows])
        if usage_vals.std() > 0 and sample_vals.std() > 0:
            corr = float(np.corrcoef(usage_vals, sample_vals)[0, 1])
        else:
            corr = float("nan")
        print(f"\nPearson correlation between batch-usage entropy and mean "
              f"per-sample assignment entropy across {len(all_rows)} "
              f"(variant, level) pairs: r={corr:.4f}")
        print("(A value far from +1 indicates these are NOT interchangeable "
              "quantities -- i.e. that 'balance regularization', as actually "
              "computed under this project's unsqueeze(0) calling convention, "
              "is not equivalent to encouraging broad code usage across the "
              "corpus, which is what its name assumes.)")
    else:
        print("\nFewer than 2 (variant, level) rows collected -- cannot compute a correlation. "
              "Check that at least 2 of the 4 checkpoint paths above exist on disk.")


if __name__ == "__main__":
    main()
