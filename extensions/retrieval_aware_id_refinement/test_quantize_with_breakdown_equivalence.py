"""
Component 3 (retrieval-aware RQ refinement feasibility study) -- equivalence
unit test for `RQ._quantize_with_breakdown` vs. the original
`self.residual_rq(x, ...)` call.

**Corrected protocol (per the plan's Component 7 fix to this test):** a single
call in `.train()` mode mutates the EMA codebooks (and can expire/reinitialize
codes), so calling the old path then the new path on the SAME live module
would make the second call see a different model -- a spurious failure even
if the implementation is correct. Running in `.eval()` mode avoids that
mutation but also skips the library's OWN loss computation entirely (guarded
by `if self.training:` in `vector_quantize_pytorch`), so nothing about the
loss path would be exercised either way.

This test instead:
  1. Constructs ONE small RQ instance (uninitialized codebooks -- kmeans_init
     has not run yet).
  2. `copy.deepcopy`s it into two copies, A and B, BEFORE any forward call, so
     both start from bit-identical (uninitialized) state.
  3. Sets the SAME `torch.manual_seed` immediately before each top-level
     forward call (old path on A, new path on B). Both paths call the
     identical per-layer `VectorQuantize.forward(residual, ...)` with
     identical kwargs (mask=None, indices=None, sample_codebook_temp=None,
     freeze_codebook=default, codebook_transform_fn=None, topk=None) on
     identical residual inputs at every one of the 9 layers, so this makes
     kmeans-init (which triggers on the first forward regardless of
     train/eval) and any other stochastic sampling draw IDENTICAL random
     numbers in the same order in both copies.
  4. Compares `quantized_out`/`Is` (should match exactly).
  5. Compares the fused-loss-vs-recomposed-loss: `self.residual_rq(x)`'s
     returned `all_losses.mean()` vs. `torch.stack(commit_losses +
     diversity_losses-weighted-appropriately)`... actually the library's fused
     per-layer `loss` already includes commitment + `codebook_diversity_loss *
     codebook_diversity_loss_weight` (see `vector_quantize_pytorch.py:1248`),
     so the correct comparison is: fused `all_losses.mean()` (old path) vs.
     `(torch.stack(commit_losses) * commitment_weight + torch.stack(diversity_losses)
     * codebook_diversity_loss_weight).mean()` (new path, recomposed) -- both
     BEFORE any `1e2 *` scaling. Since this project's `VectorQuantize` layers
     use the default `commitment_weight=1.0`, the recomposition simplifies to
     `(stack(commit_losses) + stack(diversity_losses) * balance_lambda).mean()`.
  6. Compares each copy's post-step EMA buffer states (`_codebook.embed`,
     `_codebook.cluster_size`) per layer, to prove the manual loop has the
     same SIDE EFFECTS, not just the same forward outputs.

No real checkpoint is needed or used -- this constructs a tiny synthetic RQ
directly (small feature_dim / codebook_vocab / codebook_level) so the whole
test runs in a few seconds on CPU.

Run with:
  PYTHONPATH=<repo>/src python test_quantize_with_breakdown_equivalence.py
"""
import copy
import os
import sys

import torch
from omegaconf import OmegaConf

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.residual_quantization.residual_quantization import RQ  # noqa: E402

SEED = 2023
FEATURE_DIM = 32
CODEBOOK_VOCAB = 16  # small vocab so kmeans_init (which needs >= codebook_size samples
                      # ideally) is well-conditioned at this test's small batch size
CODEBOOK_LEVEL = 3   # + 1 modality level = 4 total layers, small but non-trivial
BATCH_ROWS = 24       # 2*bs rows fed into residual_rq, matching the unsqueeze(0) convention


def build_test_config():
    return OmegaConf.create({
        "codebook_config": {
            "codebook_vocab": CODEBOOK_VOCAB,
            "codebook_level": CODEBOOK_LEVEL,
        },
        "balance_loss_config": {
            "enabled": True,   # need codebook_diversity_loss_weight > 0 so
            "lambda": 3.0,     # loss_breakdown.codebook_diversity is non-zero,
            "temperature": 100.0,  # exercising the same code path Component 4/8 rely on.
        },
        "retrieval_aware_config": {
            "enabled": False,  # irrelevant to this test -- _quantize_with_breakdown
                                # is a plain method, callable regardless of this flag.
        },
    })


def main():
    config = build_test_config()
    base_model = RQ(config=config, feature_dim=FEATURE_DIM)
    base_model.train()

    model_a = copy.deepcopy(base_model)
    model_b = copy.deepcopy(base_model)

    torch.manual_seed(SEED)
    x = torch.randn(BATCH_ROWS, FEATURE_DIM)

    # --- Old path (self.residual_rq) on copy A ---
    torch.manual_seed(SEED)
    quant_a, Is_a, rq_loss_a = model_a.residual_rq(x.unsqueeze(0), rand_quantize_dropout_fixed_seed=2023)
    quant_a = quant_a.squeeze(0)
    Is_a = Is_a.squeeze(0)

    # --- New path (_quantize_with_breakdown) on copy B ---
    torch.manual_seed(SEED)
    quant_b, Is_b, commit_losses_b, diversity_losses_b, level_quantized_b = model_b._quantize_with_breakdown(
        x.unsqueeze(0)
    )
    quant_b = quant_b.squeeze(0)
    Is_b = Is_b.squeeze(0)

    # --- 1. Compare quantized_out / Is ---
    quant_match = torch.allclose(quant_a, quant_b, atol=1e-5)
    is_match = torch.equal(Is_a, Is_b)
    print(f"[check 1] quantized_out match: {quant_match} (max abs diff: "
          f"{(quant_a - quant_b).abs().max().item():.3e})")
    print(f"[check 1] Is (code indices) match: {is_match}")

    # --- 2. Compare fused loss (old) vs. recomposed loss (new), before 1e2* scaling ---
    commitment_weight = model_b.residual_rq.layers[1].commitment_weight  # semantic layer, not modality
    diversity_weight = model_b.balance_lambda
    recomposed = (
        torch.stack(commit_losses_b) * commitment_weight
        + torch.stack(diversity_losses_b) * diversity_weight
    ).mean()
    fused = rq_loss_a.mean()
    loss_match = torch.allclose(fused, recomposed, atol=1e-4)
    print(f"[check 2] fused rq_loss.mean()={fused.item():.6f} vs. recomposed={recomposed.item():.6f} "
          f"-> match: {loss_match}")

    # --- 3. Compare post-step EMA buffer states across the two copies ---
    ema_match = True
    for level_idx, (layer_a, layer_b) in enumerate(zip(model_a.residual_rq.layers, model_b.residual_rq.layers)):
        embed_a = layer_a._codebook.embed
        embed_b = layer_b._codebook.embed
        cluster_a = layer_a._codebook.cluster_size
        cluster_b = layer_b._codebook.cluster_size
        embed_ok = torch.allclose(embed_a, embed_b, atol=1e-5)
        cluster_ok = torch.allclose(cluster_a, cluster_b, atol=1e-5)
        print(f"[check 3] level {level_idx}: embed match={embed_ok}, cluster_size match={cluster_ok}")
        ema_match = ema_match and embed_ok and cluster_ok

    all_pass = quant_match and is_match and loss_match and ema_match
    print()
    print(f"ALL CHECKS PASS: {all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
