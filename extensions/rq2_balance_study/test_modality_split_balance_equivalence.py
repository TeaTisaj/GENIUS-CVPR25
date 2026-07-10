"""
RQ2 direction-aware balance study -- equivalence unit test for
`RQ._quantize_with_breakdown_modality_split` vs. the original fused
`self.residual_rq(x, ...)` codebook-diversity path.

Protocol mirrors `extensions/retrieval_aware_id_refinement/test_quantize_with_breakdown_equivalence.py`:
build ONE tiny synthetic RQ (uninitialized codebooks -- kmeans_init has not run
yet), `copy.deepcopy` it into independent copies BEFORE any forward call so all
copies start bit-identical, and set the SAME `torch.manual_seed` immediately
before each top-level forward so kmeans-init and any stochastic sampling draw
identical random numbers in the same order.

The modality-split model is produced by deepcopy'ing the fused base and setting
its split attributes directly (`balance_split_enabled` + `balance_lambda_img/
txt`). This keeps the codebooks bit-identical between the fused and split copies
(the only difference the split path introduces is HOW the diversity term is
recomposed from the per-level breakdown -- the library's fused
`codebook_diversity_loss_weight` does not affect codebook init or EMA state, so
whether it is 0.03 or 0.0 is irrelevant to the codebook forward/EMA).

Checks:
  1. quantized_out / Is match between the fused path and the split path with
     lambda_img == lambda_txt == lambda_shared.
  2. (load-bearing) fused rq_loss.mean() == split recomposed
     (commit + weighted_div).mean() to atol=1e-6, proving equal per-row weights
     exactly recompose the library's fused scalar.
  3. post-step EMA buffers (`_codebook.embed`, `_codebook.cluster_size`)
     identical across the fused and split copies -- no side-effect divergence
     from the extra hook/computation.
  4. an asymmetric case (lambda_img != lambda_txt) matches an INDEPENDENTLY
     computed per-modality reference inside this test, and differs meaningfully
     from the symmetric case.

Run with:
  PYTHONPATH=<repo>/src python test_modality_split_balance_equivalence.py
"""
import copy
import os
import sys

import torch
from einops import reduce
from omegaconf import OmegaConf

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.residual_quantization.residual_quantization import RQ  # noqa: E402
# The independent reference below intentionally re-imports the library's own
# entropy helper directly, so the check is written against the library, not
# against our implementation's re-export.
from vector_quantize_pytorch.vector_quantize_pytorch import entropy as ref_entropy  # noqa: E402

SEED = 2023
FEATURE_DIM = 32
CODEBOOK_VOCAB = 16
CODEBOOK_LEVEL = 3      # + 1 modality level = 4 total layers
BATCH_ROWS = 24
TEMPERATURE = 100.0
LAMBDA_SHARED = 3.0     # YAML-scale; RQ pre-divides by 1e2 -> 0.03 internally


def build_fused_config():
    """Fused (shared-lambda) config -- gives self.balance_lambda == 0.03."""
    return OmegaConf.create({
        "codebook_config": {
            "codebook_vocab": CODEBOOK_VOCAB,
            "codebook_level": CODEBOOK_LEVEL,
        },
        "balance_loss_config": {
            "enabled": True,
            "lambda": LAMBDA_SHARED,
            "temperature": TEMPERATURE,
        },
        "retrieval_aware_config": {"enabled": False},
    })


def make_masks(n_rows):
    """Per-row modality masks with a mix of img-only, txt-only, and multimodal
    rows (every row has >= 1 modality, required for the split path to match the
    fused path in the symmetric case)."""
    img_mask = torch.zeros(n_rows, 1)
    txt_mask = torch.zeros(n_rows, 1)
    for i in range(n_rows):
        r = i % 3
        if r == 0:      # image-only
            img_mask[i, 0] = 1.0
        elif r == 1:    # text-only
            txt_mask[i, 0] = 1.0
        else:           # multimodal
            img_mask[i, 0] = 1.0
            txt_mask[i, 0] = 1.0
    return img_mask, txt_mask


def as_split(model, lam_img, lam_txt):
    """Turn a deepcopy of the fused base into a split-mode model in place."""
    model.balance_split_enabled = True
    model.balance_lambda_img = lam_img / 1e2
    model.balance_lambda_txt = lam_txt / 1e2
    return model


def independent_reference_div(model, x, img_mask, txt_mask, lam_img, lam_txt):
    """Independently recompute per-level modality-weighted diversity losses.

    Written from scratch against the library (own hook, own entropy import, own
    explicit per-modality-class weighting via boolean masks -- NOT the
    implementation's `w` vector formula), so a bug shared with the main method
    would not hide here. Runs its OWN forward loop on `model` (a fresh deepcopy),
    so `model` must be seeded identically to the method-under-test's copy.
    """
    im = img_mask.reshape(-1)
    tm = txt_mask.reshape(-1)
    img_only = (im > 0) & (tm == 0)
    txt_only = (tm > 0) & (im == 0)
    both = (im > 0) & (tm > 0)
    w_ref = torch.zeros(im.shape[0])
    w_ref[img_only] = lam_img / 1e2
    w_ref[txt_only] = lam_txt / 1e2
    w_ref[both] = (lam_img / 1e2 + lam_txt / 1e2) / 2.0

    residual = x
    per_level = []
    for level_idx, vq in enumerate(model.residual_rq.layers):
        captured = {}

        def _cap(module, inp, out, _s=captured):
            _s["distances"] = out[2]

        h = vq._codebook.register_forward_hook(_cap)
        try:
            quantized, _ei, _fl, _lb = vq(residual, return_loss_breakdown=True)
        finally:
            h.remove()

        if model.modality_index and level_idx == 0:
            per_level.append(torch.zeros((), dtype=x.dtype))
        else:
            dist = captured["distances"]
            prob = (dist * TEMPERATURE).softmax(dim=-1)
            avg_prob = reduce(prob, "... n l -> n l", "mean")
            ent = ref_entropy(avg_prob)
            per_level.append(-(w_ref * ent).mean())

        residual = residual - quantized.detach()
    return per_level


def main():
    config = build_fused_config()
    base_model = RQ(config=config, feature_dim=FEATURE_DIM)
    base_model.train()

    img_mask, txt_mask = make_masks(BATCH_ROWS)

    model_fused = copy.deepcopy(base_model)
    model_split = as_split(copy.deepcopy(base_model), LAMBDA_SHARED, LAMBDA_SHARED)

    torch.manual_seed(SEED)
    x = torch.randn(BATCH_ROWS, FEATURE_DIM)

    # --- Fused path (library's own diversity) ---
    torch.manual_seed(SEED)
    quant_f, Is_f, rq_loss_f = model_fused.residual_rq(
        x.unsqueeze(0), rand_quantize_dropout_fixed_seed=2023
    )
    quant_f = quant_f.squeeze(0)
    Is_f = Is_f.squeeze(0)

    # --- Split path (symmetric lambda == shared lambda) ---
    torch.manual_seed(SEED)
    quant_s, Is_s, commit_s, wdiv_s = model_split._quantize_with_breakdown_modality_split(
        x.unsqueeze(0), img_mask, txt_mask
    )
    quant_s = quant_s.squeeze(0)
    Is_s = Is_s.squeeze(0)

    # --- Check 1: quantized_out / Is ---
    quant_match = torch.allclose(quant_f, quant_s, atol=1e-5)
    is_match = torch.equal(Is_f, Is_s)
    print(f"[check 1] quantized_out match: {quant_match} "
          f"(max abs diff: {(quant_f - quant_s).abs().max().item():.3e})")
    print(f"[check 1] Is (code indices) match: {is_match}")

    # --- Check 2: fused rq_loss.mean() vs. split recomposed (commit + weighted_div).mean() ---
    fused_mean = rq_loss_f.mean()
    recomposed = (torch.stack(commit_s) + torch.stack(wdiv_s)).mean()
    loss_match = torch.allclose(fused_mean, recomposed, atol=1e-6)
    print(f"[check 2] fused rq_loss.mean()={fused_mean.item():.8f} vs. "
          f"recomposed={recomposed.item():.8f} -> match (atol=1e-6): {loss_match} "
          f"(abs diff: {(fused_mean - recomposed).abs().item():.3e})")

    # --- Check 3: post-step EMA buffers identical across fused & split copies ---
    ema_match = True
    for level_idx, (la, lb) in enumerate(
        zip(model_fused.residual_rq.layers, model_split.residual_rq.layers)
    ):
        embed_ok = torch.allclose(la._codebook.embed, lb._codebook.embed, atol=1e-6)
        cluster_ok = torch.allclose(la._codebook.cluster_size, lb._codebook.cluster_size, atol=1e-6)
        print(f"[check 3] level {level_idx}: embed match={embed_ok}, cluster_size match={cluster_ok}")
        ema_match = ema_match and embed_ok and cluster_ok

    # --- Check 4: asymmetric (lambda_img != lambda_txt) vs. independent reference ---
    LAM_IMG, LAM_TXT = 3.0, 0.0
    model_asym = as_split(copy.deepcopy(base_model), LAM_IMG, LAM_TXT)
    model_ref = copy.deepcopy(base_model)  # independent copy for the reference loop

    torch.manual_seed(SEED)
    _q_a, _Is_a, _commit_a, wdiv_asym = model_asym._quantize_with_breakdown_modality_split(
        x.unsqueeze(0), img_mask, txt_mask
    )

    torch.manual_seed(SEED)
    wdiv_ref = independent_reference_div(
        model_ref, x.unsqueeze(0), img_mask, txt_mask, LAM_IMG, LAM_TXT
    )

    asym_match = True
    for level_idx, (a, r) in enumerate(zip(wdiv_asym, wdiv_ref)):
        ok = torch.allclose(a, r, atol=1e-6)
        asym_match = asym_match and ok
    print(f"[check 4] asymmetric method vs. independent reference match (atol=1e-6): {asym_match}")

    # asymmetric must differ meaningfully from the symmetric case (txt-only rows
    # get lambda 0.0 instead of 0.03, so the diversity loss must change).
    sym_vec = torch.stack(wdiv_s)
    asym_vec = torch.stack([a.detach() for a in wdiv_asym])
    max_gap = (sym_vec - asym_vec).abs().max().item()
    asym_differs = max_gap > 1e-4
    print(f"[check 4] asymmetric differs from symmetric: {asym_differs} "
          f"(max abs gap across levels: {max_gap:.3e})")

    all_pass = (quant_match and is_match and loss_match and ema_match
                and asym_match and asym_differs)
    print()
    print(f"ALL CHECKS PASS: {all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
