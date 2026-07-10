"""
Residual Quantization model implementation.
"""

# Standard library
import os
import math
import time
import copy
import string
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Dict, List, Union

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from einops import rearrange, repeat, reduce, pack, unpack
from torch.nn.utils import weight_norm
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
# Import the library's OWN entropy helper (not a reimplementation): its exact eps
# clamping is load-bearing for numerical equivalence with the library's fused
# codebook-diversity loss in the modality-split path below.
from vector_quantize_pytorch.vector_quantize_pytorch import entropy as _vq_entropy

# Local modules
from models.uniir_clip import utils
from models.residual_quantization.loss import ClipLoss 

# ================ Loss Functions ================

class ContrastiveLoss(nn.Module):
    """Contrastive loss for feature learning.
    
    This loss function computes the contrastive loss between two sets of embeddings,
    encouraging similar items to be close and dissimilar items to be far apart in the embedding space.
    
    Args:
        temperature: Temperature parameter for softmax scaling
        metric: Similarity metric ('cos' for cosine similarity or 'euclid' for Euclidean distance)
        gather: Whether to gather embeddings from all GPUs in distributed training
    """
    def __init__(self, temperature: float = 0.01, metric: str = 'cos', gather: bool = True):
        super().__init__()
        self.temperature = temperature
        self.metric = metric
        self.gather = gather
        
    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        """Generate ground truth labels for contrastive learning.
        
        Args:
            device: Device to create labels on
            num_logits: Number of logits (batch size)
            
        Returns:
            Tensor of labels where each item is matched with itself
        """
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def forward(self, x: torch.Tensor, y: torch.Tensor, temp: Optional[float] = None) -> torch.Tensor:
        """Compute contrastive loss between two sets of embeddings.
        
        Args:
            x: First set of embeddings
            y: Second set of embeddings
            temp: Optional temperature parameter (overrides default if provided)
            
        Returns:
            Contrastive loss value
        """
        if temp is None:
            temp = self.temperature

        # Gather embeddings from all GPUs if in distributed training
        if utils.get_world_size() > 1 and self.gather:
            x = torch.cat(utils.GatherLayer.apply(x), dim=0)
            y = torch.cat(utils.GatherLayer.apply(y), dim=0)

        labels = self.get_ground_truth(x.device, x.shape[0])

        # Compute similarity based on chosen metric
        if self.metric == 'cos':
            logits_per_x = F.linear(F.normalize(x), F.normalize(y))
        elif self.metric == 'euclid':
            logits_per_x = -torch.cdist(x,y) ** 2
        else:
            raise ValueError(f'Invalid metric: {self.metric}')

        # Scale logits and compute bidirectional loss
        logits_per_x = logits_per_x / temp
        logits_per_y = logits_per_x.T

        total_loss = (F.cross_entropy(logits_per_x, labels)
                     + F.cross_entropy(logits_per_y, labels)) / 2

        return total_loss

    def forward_explicit_neg(self, q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                              temp: Optional[float] = None) -> torch.Tensor:
        """Retrieval-aware RQ refinement feasibility study, Component 5 (L_ret).

        Explicit-negative-set contrastive loss (mentor's L_ret), as opposed to
        this class's default `forward`, which uses in-batch negatives with a
        diagonal-positive assumption:

            -log( exp(sim(q,pos)/T) / (exp(sim(q,pos)/T) + sum_i exp(sim(q,neg_i)/T)) )

        Implemented as cross-entropy with the positive placed at logit index 0,
        numerically identical to the explicit formula above.

        Args:
            q: [bs, d] query representations.
            pos: [bs, d] positive representations.
            neg: [bs, K, d] explicit hard/random negative representations.
            temp: optional temperature override.

        Returns:
            Scalar loss.
        """
        if temp is None:
            temp = self.temperature

        assert self.metric == 'cos', "forward_explicit_neg only supports metric='cos'."

        q = F.normalize(q, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        pos_sim = (q * pos).sum(dim=-1) / temp  # [bs]
        neg_sim = torch.einsum('bd,bkd->bk', q, neg) / temp  # [bs, K]

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [bs, K+1], positive at index 0
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances between two sets of vectors.
    
    Args:
        x: First set of vectors
        y: Second set of vectors
        
    Returns:
        Matrix of pairwise distances
    """
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min=0).sqrt()

# ================ Neural Network Modules ================

class Combiner(nn.Module):
    """Combiner module for fusing textual and visual information.
    
    This module combines CLIP image and text features through a series of transformations
    and a dynamic weighting mechanism to create a unified representation. (https://github.com/ABaldrati/CLIP4Cir)
    
    Args:
        clip_feature_dim: CLIP input feature dimension
        projection_dim: Dimension for projected features
        hidden_dim: Hidden layer dimension
        drop_rate: Dropout rate
    """
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int, drop_rate=0):
        super(Combiner, self).__init__()
        # Projection layers for image and text features
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)

        # Feature combination layers
        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        # Dynamic weighting mechanism
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.out_projection_layer = nn.Identity()

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
                img_masks: Optional[torch.Tensor] = None, txt_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine image and text features.
        
        Args:
            image_features: CLIP image features
            text_features: CLIP text features
            img_masks: Optional mask for image features
            txt_masks: Optional mask for text features
            
        Returns:
            Combined features
        """
        # Apply masks if provided
        if img_masks is not None:
            image_features = image_features * img_masks
        if txt_masks is not None:
            text_features = text_features * txt_masks

        # Project features
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features))) 

        # Combine features
        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        
        # Compute dynamic weights
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        
        # Combine with original features
        output = (self.output_layer(combined_features) + 
                 dynamic_scalar * text_features + 
                 (1 - dynamic_scalar) * image_features)
        
        return self.out_projection_layer(output)

class RQ(nn.Module):
    """Residual Quantization.
    
    Args:
        config: Model configuration
        feature_dim: Feature dimension
        clip_model: CLIP model for feature extraction
        unique_code: Whether to ensure unique codes
        modality_index: Whether to use modality-specific indexing
    """
    def __init__(self,
                 config,
                 feature_dim=768,
                 clip_model=None,
                 unique_code=False,
                 modality_index=True):
        super().__init__()
        # Initialize CLIP model if provided
        if clip_model is not None:
            self.clip_model = clip_model
            for _, param in self.clip_model.named_parameters():
                param.requires_grad = False
            self.clip_model.eval()

        # Initialize model parameters
        self.iter = 0
        self.unique_code = unique_code
        self.modality_index = modality_index
        
        # Initialize components
        self.encoder = Combiner(feature_dim, 2560, 5120)
        self.critic = nn.MSELoss(reduction='mean')
        self.contra_loss = ContrastiveLoss(temperature=0.01)

        # Set up codebook configuration
        self.codebook_vocab = config.codebook_config.codebook_vocab
        self.codebook_level = (config.codebook_config.codebook_level + 1
                             if self.modality_index
                             else config.codebook_config.codebook_level)
        self.level_indicators = list(string.ascii_lowercase[:self.codebook_level])

        # Optional soft code-balance regularization on the semantic levels only
        # (the modality level's skew is intentional, not collapse, so it is
        # built separately below via self.vq and never receives this kwarg).
        # Uses vector_quantize_pytorch's built-in codebook_diversity_loss:
        # a softmax-over-scaled-distances entropy bonus that is differentiable
        # through the encoder even though the codebook itself is EMA-updated
        # (learnable_codebook=False below) rather than learned via backprop.
        balance_loss_config = config.get("balance_loss_config", {})
        balance_enabled = balance_loss_config.get("enabled", False)
        balance_lambda_raw = balance_loss_config.get("lambda", 0.0) if balance_enabled else 0.0
        # compute_single_batch applies a blanket `1e2 *` to the whole rq_loss
        # (commitment + this diversity term combined) below. Pre-divide by that
        # same factor here so balance_loss_config.lambda is interpretable on
        # the same O(1-10) scale as cl_loss/mse_loss/accuracy, not 100x larger
        # for an identical-looking config value.
        self.balance_lambda = balance_lambda_raw / 1e2
        self.balance_temperature = balance_loss_config.get("temperature", 100.0)

        # Direction-aware (modality-split) balance regularization (RQ2 study).
        # The established, mentor-reviewed finding is that T->I and I->T retrieval
        # react in OPPOSITE directions to the shared balance strength: strong
        # balance helps T->I but destroys I->T. lambda_img / lambda_txt let a
        # single model apply a different balance strength to image-target vs
        # text-target rows so both directions can be tuned independently.
        lambda_img = balance_loss_config.get("lambda_img", None)
        lambda_txt = balance_loss_config.get("lambda_txt", None)
        self.balance_split_enabled = balance_enabled and (lambda_img is not None or lambda_txt is not None)
        if self.balance_split_enabled:
            # A nonzero shared `lambda` is ALSO fed to ResidualVQ below as
            # codebook_diversity_loss_weight (self.balance_lambda), so the
            # library's own fused diversity term would then fire on top of our
            # manual per-row one -- double-counting the same entropy. Split mode
            # must therefore run with the shared lambda absent/zero, which makes
            # self.balance_lambda == 0.0 and disables the library's fused path
            # (has_codebook_diversity_loss is False for every layer).
            assert self.balance_lambda == 0.0, (
                "balance_loss_config: cannot set a nonzero shared `lambda` together with "
                "lambda_img/lambda_txt. The shared lambda is passed to ResidualVQ's "
                "codebook_diversity_loss_weight and would double-count the diversity term "
                "against the manual modality-split path. Remove the `lambda` key (or set it "
                "to 0.0) when using lambda_img/lambda_txt."
            )
        # Same /1e2 pre-division convention as self.balance_lambda so YAML values
        # stay on the same O(1-10) scale (compute_single_batch multiplies the whole
        # rq_loss by 1e2 downstream).
        self.balance_lambda_img = float(lambda_img or 0.0) / 1e2
        self.balance_lambda_txt = float(lambda_txt or 0.0) / 1e2

        # ============================================================
        # Retrieval-aware RQ refinement feasibility study (Components 3-5).
        # See extensions/retrieval_aware_id_refinement/ and
        # .claude/plans/so-this-is-the-swirling-quiche.md. Fully additive:
        # when retrieval_aware_config.enabled is false (the default for every
        # config outside this study), none of this branch is exercised and
        # compute_single_batch's original code path runs byte-for-byte
        # unchanged (see Component 3's equivalence requirement).
        # ============================================================
        retrieval_aware_config = config.get("retrieval_aware_config", {})
        self.retrieval_aware_enabled = bool(retrieval_aware_config.get("enabled", False))
        # balance_split_enabled (lambda_img/lambda_txt) and retrieval_aware_enabled
        # are separate, mutually-exclusive dispatch branches in compute_single_batch,
        # each recomposing rq_loss its own way; they must never both be active.
        assert not (self.balance_split_enabled and self.retrieval_aware_enabled), (
            "balance_split_enabled (lambda_img/lambda_txt) and retrieval_aware_enabled are "
            "mutually exclusive code paths; enable at most one of them."
        )
        self.hard_neg_num = int(retrieval_aware_config.get("hard_neg_num", 20))
        self.retrieval_aware_beta = float(retrieval_aware_config.get("beta", 1.0))
        self.prefix_omega_decay = float(retrieval_aware_config.get("prefix_omega_decay", 0.7))
        # [Added post-feasibility-run, per real-cluster finding] Criterion 5's dense
        # Recall@K probe showed a severe T->I collapse (~98%->~5-7%) in BOTH the full
        # method AND its continued-training-no-new-losses control equally -- consistent
        # with EMA codebook churn on the narrow 18K-query training subset (rare codes
        # falling below `threshold_ema_dead_code` and getting randomly reassigned),
        # not a property of the retrieval-aware method itself. `freeze_codebook=True`
        # (passed through to each level's `VectorQuantize.forward`, gates the
        # `self.training and ema_update and not freeze_codebook` check at
        # vector_quantize_pytorch.py:746) stops EMA codebook updates entirely while
        # still allowing gradient flow to the encoder (commitment/diversity/L_ret all
        # still backprop normally) -- isolating whether the method's signal survives
        # once codebook churn is removed as a confound.
        self.freeze_codebook = bool(retrieval_aware_config.get("freeze_codebook", False))

        # Component 4 [Fable fix]: self.balance_lambda above (balance_lambda_raw / 1e2)
        # exists ONLY to gate/scale the library's OWN fused per-layer diversity term
        # (used by the ORIGINAL, non-retrieval-aware global-lambda mechanism, and here
        # also just to keep `codebook_diversity_loss_weight > 0` so the library actually
        # computes `loss_breakdown.codebook_diversity` at all -- see VectorQuantize's
        # `has_codebook_diversity_loss` gate). It is NOT the layer-wise budget Lambda used
        # below. Reusing self.balance_lambda as Lambda would silently run this study at
        # Lambda=0.03 instead of the intended Lambda=3.0 (100x too small, no error, just a
        # quietly-degenerate feasibility study). Lambda is instead taken directly,
        # undivided, from the same YAML field:
        self.balance_budget = float(balance_loss_config.get("lambda", 0.0))  # Lambda, undivided
        if self.retrieval_aware_enabled:
            assert self.balance_budget > 0, (
                "retrieval_aware_config.enabled=True requires a non-zero "
                "balance_loss_config.lambda (the layer-wise budget Lambda) -- with "
                "Lambda<=0, lambda_l is always all-zeros regardless of "
                "balance_level_logits, and every one of the mentor's feasibility "
                "criteria would trivially and misleadingly 'pass'. Set "
                "balance_loss_config.enabled=true and lambda > 0 (e.g. 3.0, matching "
                "the existing 'strong' variant) explicitly in the feasibility config."
            )

        # Component 4 (Method 1): layer-wise learnable balance weights a_l, one per
        # semantic level (the modality level, index 0, is excluded -- its skew is
        # intentional, not collapse, exactly as for self.balance_lambda above).
        #
        # [Fable-fix-driven design decision, see this repo's task report for full
        # rationale] Of the plan's two explicitly-offered "preferred" sub-options for
        # learning a_l from a VALIDATION retrieval objective (mentor's point 5) --
        # (a) a literal DARTS-style one-step-lookahead second-order backprop, or
        # (b) a cheaper validation-guided perturbation ("perturb the logits, evaluate
        # a held-out batch, keep the change only if it helps") -- this implementation
        # uses (b). Both are valid instances of the plan's "Preferred, faithful to the
        # mentor's spec" option (not the "explicit fallback" of a frozen, unlearned
        # prior); (a) was judged too risky to get right without live GPU debugging
        # (constructing a differentiable one-step optimizer lookahead blind is exactly
        # the kind of thing the plan's own Fable-fix review process was created to catch
        # mistakes in), while (b) is a well-defined, easily-unit-testable zeroth-order
        # update rule that still uses genuine validation L_ret feedback, not the
        # training objective itself (which the plan shows collapses to a degenerate
        # one-hot allocation if used directly).
        #
        # Consequently, `balance_level_logits` is registered as a BUFFER (not an
        # `nn.Parameter`): it is NEVER intended to receive an autograd gradient
        # through the normal backward pass (there is none to receive here -- L_bal
        # is never part of the single real `.backward()` call, only used via
        # `torch.autograd.grad` in the Component 6 gradient-projection step, which
        # differentiates w.r.t. the encoder, not w.r.t. a_l), and buffers are never
        # returned by `named_parameters()` at all, so it is automatically excluded
        # from every optimizer parameter group in train.py's `filter_parameters`
        # -- see engine.py's `perturb_and_maybe_update_balance_logits` for the
        # actual update rule, called from `train_one_epoch_retrieval_aware`.
        #
        # [Bug found and fixed during this task's own local verification, not
        # anticipated by the plan text] `persistent=False` is required here, not
        # just a style choice: a PERSISTENT buffer (or an `nn.Parameter`, as this
        # was first implemented) would appear in `state_dict()`, and since this
        # attribute is new, `model.load_state_dict(old_checkpoint, strict=True)`
        # would then raise `Missing key(s): "balance_level_logits"` for every
        # checkpoint saved before this change existed -- breaking `strict=True`
        # loads throughout this repo wherever an RQ checkpoint is loaded (this
        # extension's own `mine_hard_negatives.py`/`clarify_balance_regularization.py`/
        # the new warm-start loader in `train.py`, but also any other, unrelated
        # pre-existing script that loads an RQ checkpoint with `strict=True`).
        # `persistent=False` excludes it from `state_dict()` entirely, so old
        # checkpoints keep loading exactly as before. This also means a
        # retrieval-aware run's OWN saved checkpoints do not carry the learned
        # lambda_l forward either -- by design: the plan's own diagnostics
        # (`check_feasibility_criteria.py`) source the learned lambda_l
        # trajectory from the printed/logged values (Component 4's explicit
        # "log lambda_l ... every print_freq steps" requirement), not from
        # reloading a checkpoint file.
        num_semantic_levels = self.codebook_level - 1 if self.modality_index else self.codebook_level
        self.num_semantic_levels = num_semantic_levels
        self.register_buffer(
            'balance_level_logits', torch.zeros(num_semantic_levels), persistent=False
        )  # a_l

        # Initialize Residual Vector Quantization
        if self.modality_index:
            self.vq = VectorQuantize(
                dim=feature_dim,
                codebook_dim=feature_dim,  
                codebook_size=3, 
                kmeans_init=True,
                kmeans_iters=1000, 
                learnable_codebook=False,
                ema_update=True, 
                threshold_ema_dead_code=0,
                decay=0.9
            )

        self.residual_rq = ResidualVQ(
            dim=feature_dim,
            codebook_dim=feature_dim,
            num_quantizers=self.codebook_level,
            codebook_size=self.codebook_vocab,
            kmeans_init=True,
            kmeans_iters=1000,
            learnable_codebook=False,
            ema_update=True,
            threshold_ema_dead_code=2,
            decay=0.9,
            codebook_diversity_loss_weight=self.balance_lambda,
            codebook_diversity_temperature=self.balance_temperature,
        )

        if self.modality_index:
            self.residual_rq.layers[0] = self.vq 

        # Initialize tracking variables
        self.gather_embedding = True
        self.is_history = {}
        self.codebook_reset()
        
    def get_img_preprocess_fn(self):
        """Get image preprocessing function from CLIP model."""
        return self.clip_model.get_img_preprocess_fn()
    
    def get_tokenizer(self):
        """Get tokenizer from CLIP model."""
        return self.clip_model.get_tokenizer()

    def codebook_reset(self):
        """Reset codebook tracking variables."""
        self.codebag = {}
        self.num_collision = 0

    def collision_update(self, codes: torch.Tensor, id_list: List[int]):
        """Update collision tracking for codes.
        
        Args:
            codes: Generated codes
            id_list: List of IDs
        """
        for i in range(len(codes)):
            code = str(tuple(codes[i].cpu().numpy().tolist()))
            id = str(id_list[i])
            
            if code in self.codebag:
                if self.codebag[code] != id:
                    self.num_collision += 1
                    self.codebag[code] = id
            else:
                self.codebag[code] = id

    def transform_row(self, row: List[int], separator: str = '') -> str:
        """Transform a row of codes into token format.
        
        Args:
            row: List of code values
            separator: Separator string between codes
            
        Returns:
            String representation of codes
        """
        transformed_row = []
        for l, level_indicator in enumerate(self.level_indicators):
            new_value = row[l]
            transformed_row.append(f"<{level_indicator}{new_value}>")
        return separator.join(transformed_row)

    def _quantize_with_breakdown(self, x: torch.Tensor):
        """Retrieval-aware RQ refinement feasibility study, Component 3.

        A unified manual per-level quantization loop exposing a clean per-level
        loss breakdown, used (instead of `self.residual_rq(x, ...)`) only when
        `self.retrieval_aware_enabled`.

        Why this exists: `ResidualVQ.forward` (vector_quantize_pytorch's
        `residual_vq.py`) calls each level's `VectorQuantize.forward` WITHOUT
        `return_loss_breakdown=True`, so it only ever returns a single FUSED
        `loss` per level (commitment + codebook-diversity summed together) --
        enough to reproduce today's `rq_loss.mean()` (L_RQ), but not enough to
        isolate a clean `L_bal^(l)` for Component 6's gradient-conflict
        projection, which needs the diversity term on its own.

        This bypasses `self.residual_rq.__call__` and calls each level's
        `VectorQuantize.forward(..., return_loss_breakdown=True)` directly,
        replicating `ResidualVQ.forward`'s residual-accumulation loop manually.
        This is a faithful replication, not an approximation: this project's
        `ResidualVQ` is constructed with `quantize_dropout` left at its default
        (False) and `quant_grad_frac` left at its default (0.), so neither the
        "should_quantize_dropout" branch nor `frac_gradient`'s partial
        straight-through blending (`frac_gradient(t, 0.) == t.detach()`,
        confirmed by reading `vector_quantize_pytorch/residual_vq.py`) ever
        activates here -- the loop below is exactly what `ResidualVQ.forward`
        does in this project's configuration, just unrolled with per-level
        loss access. `project_in`/`project_out` are `nn.Identity()` in this
        project's configuration too (dim == codebook_dim == feature_dim, so
        `requires_projection` is False in `ResidualVQ.__init__`), so they are
        correctly omitted here exactly as the original call site never uses
        them either.

        Numerically identical to `self.residual_rq(x)` for `quantized_out`/`Is`
        (verified by
        `extensions/retrieval_aware_id_refinement/test_quantize_with_breakdown_equivalence.py`);
        when `retrieval_aware_enabled` is False, this method is never called and
        `compute_single_batch`'s original path is completely untouched.

        Args:
            x: input of shape `(1, N, dim)` -- call with the SAME
                `encode_feature.unsqueeze(0)` shape convention the original
                `self.residual_rq(encode_feature.unsqueeze(0), ...)` call site
                uses; squeeze(0) the `quantized_out`/`Is` outputs afterward the
                same way the original call site does (skipping the unsqueeze
                would silently change how the EMA codebook's internal batch-dim
                bookkeeping treats the input).

        Returns:
            quantized_out: [1, N, dim] summed reconstruction (same shape/values
                as `self.residual_rq(x)`'s first return value).
            Is: [1, N, codebook_level] stacked per-level code indices.
            commit_losses: list of `codebook_level` scalar commitment-loss
                tensors (one per level), each still attached to the autograd
                graph through the encoder.
            diversity_losses: list of `codebook_level` scalar
                codebook-diversity-loss tensors (one per level); components
                where `has_codebook_diversity_loss` is False (i.e.
                `codebook_diversity_loss_weight <= 0` for that layer) will be
                the library's constant `self.zero` buffer (no gradient).
            level_quantized: list of `codebook_level` `[1, N, dim]` tensors,
                each level's OWN contribution to the sum (NOT yet
                accumulated) -- used by Component 5 to build prefix-sum
                embeddings without re-deriving them from `quantized_out`.
        """
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices, commit_losses, diversity_losses, level_quantized = [], [], [], []
        for level_idx, vq in enumerate(self.residual_rq.layers):
            # NOTE: rand_quantize_dropout_fixed_seed is a ResidualVQ.forward-only
            # kwarg, not accepted by the per-layer VectorQuantize.forward -- do not
            # pass it here. Irrelevant anyway since quantize_dropout=False in this
            # project's ResidualVQ config (see docstring above).
            # NOTE: return_loss_breakdown=True yields a 4-tuple (quantize, embed_ind,
            # loss, loss_breakdown), not a 3-tuple.
            # freeze_codebook=self.freeze_codebook: see RQ.__init__'s comment on
            # self.freeze_codebook (post-feasibility-run addition) -- stops EMA
            # codebook updates (vector_quantize_pytorch.py:746's
            # `self.training and ema_update and not freeze_codebook` gate) while
            # leaving gradient flow to the encoder untouched.
            quantized, embed_indices, _fused_loss, loss_breakdown = vq(
                residual, return_loss_breakdown=True, freeze_codebook=self.freeze_codebook
            )
            commit_losses.append(loss_breakdown.commitment)
            diversity_losses.append(loss_breakdown.codebook_diversity)
            level_quantized.append(quantized)  # this level's OWN contribution (not yet summed)
            residual = residual - quantized.detach()  # quant_grad_frac=0. => frac_gradient(t,0)==t.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(embed_indices)
        Is = torch.stack(all_indices, dim=-1)
        return quantized_out, Is, commit_losses, diversity_losses, level_quantized

    def _quantize_with_breakdown_modality_split(self, x: torch.Tensor,
                                                img_mask: torch.Tensor,
                                                txt_mask: torch.Tensor):
        """Direction-aware (modality-split) balance regularization (RQ2 study).

        Same per-level residual-accumulation loop as `_quantize_with_breakdown`,
        but instead of the library's single fused codebook-diversity term (one
        shared lambda for every row), it recomputes that term MANUALLY, per row,
        from the raw `distances` tensor so each row can be weighted by a
        modality-dependent lambda -- lambda_img for image-target rows, lambda_txt
        for text-target rows.

        Why a forward hook: the raw `distances` are exposed neither by
        `VectorQuantize.forward`'s return value nor by its `loss_breakdown`
        (`loss_breakdown.codebook_diversity` is the already-reduced, batch-wide
        scalar, not split by modality). They are only available as the third
        element of `vq._codebook`'s forward output
        (`(quantize, embed_ind, dist)`, vector_quantize_pytorch.py:754), so we
        capture them via a forward hook on that submodule and replay the
        library's exact entropy math (vector_quantize_pytorch.py:1244-1246). When
        lambda_img == lambda_txt == the shared lambda, this recomposes the fused
        scalar EXACTLY (proven by the equivalence test), because the library's
        `avg_prob = reduce(prob, '... n l -> n l', 'mean')` keeps entropy
        per-row under the batch-of-1 convention, so `lambda * mean_n(-ent_n)` and
        `mean_n(w_n * (-ent_n))` coincide when `w_n == lambda`.

        Args:
            x: [1, N, dim] -- same `encode_feature.unsqueeze(0)` convention as
                `_quantize_with_breakdown`.
            img_mask, txt_mask: per-row modality masks, each reshapeable to N rows
                (e.g. the [N, 1] masks built in `compute_single_batch`). Not
                mutually exclusive: a multimodal row is both image and text.

        Returns:
            quantized_out: [1, N, dim] summed reconstruction (same as
                `_quantize_with_breakdown`).
            Is: [1, N, codebook_level] stacked per-level code indices.
            commit_losses: list of `codebook_level` scalar commitment losses.
            weighted_div_losses: list of `codebook_level` scalar modality-weighted
                diversity losses (`-(w * ent).mean()`); level 0 (the modality
                level) is a hard zero, matching that level being built with no
                diversity kwarg at all -- its skew is intentional, not collapse.
        """
        im = img_mask.reshape(-1).float()
        tm = txt_mask.reshape(-1).float()
        n_rows = im.shape[0]
        # Per-row balance weight. Masks are not mutually exclusive, so a
        # multimodal row gets the MEAN of the two lambdas. clamp(min=1) only
        # guards a hypothetical all-zero-mask row from a 0/0 -- such a row has a
        # zero numerator and so contributes nothing regardless.
        w = (im * self.balance_lambda_img + tm * self.balance_lambda_txt) / (im + tm).clamp(min=1.0)

        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices, commit_losses, weighted_div_losses = [], [], []
        for level_idx, vq in enumerate(self.residual_rq.layers):
            captured = {}

            def _capture_distances(module, inp, out, _store=captured):
                # _codebook.forward returns (quantize, embed_ind, dist); grab the
                # LAST call's distances (this project sets no
                # in_place_codebook_optimizer, so _codebook is called exactly once
                # per VectorQuantize.forward).
                _store['distances'] = out[2]

            handle = vq._codebook.register_forward_hook(_capture_distances)
            try:
                quantized, embed_indices, _fused_loss, loss_breakdown = vq(
                    residual, return_loss_breakdown=True, freeze_codebook=self.freeze_codebook
                )
            finally:
                # Always detach the hook, even if vq(...) raises, so we never leak
                # a hook onto a shared submodule across calls.
                handle.remove()

            commit_losses.append(loss_breakdown.commitment)

            if self.modality_index and level_idx == 0:
                # The modality level never receives balance regularization,
                # mirroring the fused path where self.vq is built with no
                # diversity kwarg. Append a zero of the right device/dtype.
                weighted_div_losses.append(torch.zeros((), device=x.device, dtype=x.dtype))
            else:
                distances = captured['distances']
                # Exact replay of vector_quantize_pytorch.py:1244-1246. entropy()
                # is the library's own helper (its eps clamping is load-bearing for
                # exact equivalence); reduce() uses the identical einops pattern.
                prob = (distances * self.balance_temperature).softmax(dim=-1)
                avg_prob = reduce(prob, '... n l -> n l', 'mean')
                ent = _vq_entropy(avg_prob)  # [N]: per-row entropy under batch-of-1
                # If the [1, N, dim] batch-of-1 convention ever breaks upstream,
                # `ent` would not be one-per-row and the modality weighting would
                # be silently misaligned. This assertion is what catches that.
                assert ent.shape[0] == n_rows, (
                    f"modality-split balance: per-row entropy has {ent.shape[0]} rows but "
                    f"expected {n_rows} (the [1, N, dim] batch-of-1 convention is broken)."
                )
                weighted_div_losses.append(-(w * ent).mean())

            residual = residual - quantized.detach()  # quant_grad_frac=0. => detach STE residual
            quantized_out = quantized_out + quantized
            all_indices.append(embed_indices)

        Is = torch.stack(all_indices, dim=-1)
        return quantized_out, Is, commit_losses, weighted_div_losses

    def _compute_retrieval_aware_L_ret(self, level_quantized: List[torch.Tensor], bs: int,
                                        hard_neg_pool: Dict, gpu_id) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Retrieval-aware RQ refinement feasibility study, Component 5.

        `L_ret` (+ prefix-depth version): a retrieval-aware contrastive loss
        that makes the query closer to the positive ID representation than to
        explicit hard/random-negative ID representations, computed at every
        prefix depth `l = 1..num_semantic_levels` (mentor's point 4) and
        combined with geometric-decay weights `omega_l` favoring earlier levels
        (mentor's explicit instruction: "give earlier levels larger weights
        because early prefixes determine pruning during constrained decoding").

        Hard negatives arrive as the SEPARATE `hard_neg_pool` batch key built by
        `HardNegativeAugmentedDataset`/`hard_negative_collate_fn` (Component 2's
        corrected design) -- encoded here through their OWN
        `self.encoder`/`_quantize_with_breakdown` forward call, kept entirely
        distinct from the `q_emb`/`p_emb` used by `mse_loss`/`cl_loss`/accuracy.

        Args:
            level_quantized: per-level `[2*bs, dim]` tensors (query+pos
                concatenated, same layout as `q_emb`/`p_emb` elsewhere in
                `compute_single_batch`) from the MAIN `_quantize_with_breakdown`
                call already performed on the query+pool batch.
            bs: number of queries (== number of positives) in this batch.
            hard_neg_pool: dict with `img_emb`/`txt_emb`/`img_mask`/`txt_mask`,
                each of shape `[bs * hard_neg_num, ...]` (flattened by
                `hard_negative_collate_fn`).
            gpu_id: device to move `hard_neg_pool` tensors to.

        Returns:
            L_ret: scalar, `sum_l omega_l * L_ret^(l)`.
            L_ret_per_level: list of `num_semantic_levels` scalar tensors
                (`omega_l * L_ret^(l)`, already weighted), for Component 6 to
                `torch.autograd.grad` per level.
        """
        hn_img_mask = hard_neg_pool['img_mask'].reshape(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        hn_txt_mask = hard_neg_pool['txt_mask'].reshape(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        hn_img_emb = hard_neg_pool['img_emb'].reshape(-1, hard_neg_pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        hn_txt_emb = hard_neg_pool['txt_emb'].reshape(-1, hard_neg_pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)

        # Explicit, asserted slice boundary (Component 2's corrected design) --
        # never inferred from shape alone without checking it first.
        expected_n = bs * self.hard_neg_num
        assert hn_img_emb.shape[0] == expected_n, (
            f"hard_neg_pool batch size {hn_img_emb.shape[0]} != bs*hard_neg_num "
            f"({bs}*{self.hard_neg_num}={expected_n}); Component 2 dataset/collate wiring mismatch."
        )

        hn_encode_feature = self.encoder(hn_img_emb, hn_txt_emb, hn_img_mask, hn_txt_mask)
        hn_encode_feature = F.normalize(hn_encode_feature)

        neg_quant, _neg_Is, _neg_commit, _neg_diversity, neg_level_quantized = self._quantize_with_breakdown(
            hn_encode_feature.unsqueeze(0)
        )
        neg_quant = neg_quant.squeeze(0)
        neg_level_quantized = [lv.squeeze(0) for lv in neg_level_quantized]
        assert neg_quant.shape[0] == expected_n, (
            f"neg_quant.shape[0]={neg_quant.shape[0]} != bs*hard_neg_num={expected_n}"
        )

        q_level = [lv[:bs] for lv in level_quantized]          # queries, per level, [bs, dim]
        pos_level = [lv[bs:2 * bs] for lv in level_quantized]  # positives, per level, [bs, dim]
        neg_level = [lv.view(bs, self.hard_neg_num, -1) for lv in neg_level_quantized]  # [bs, K, dim]

        start_level = 1 if self.modality_index else 0  # skip modality level (index 0) in prefix sums
        num_levels = self.num_semantic_levels

        # omega_l: geometric decay, normalized to sum to 1, earlier levels weighted
        # larger (mentor's point 4, literal instruction).
        raw_omega = torch.tensor(
            [self.prefix_omega_decay ** l for l in range(num_levels)],
            device=q_level[0].device, dtype=q_level[0].dtype,
        )
        omega_l = raw_omega / raw_omega.sum()

        L_ret_per_level = []
        q_prefix = torch.zeros_like(q_level[0])
        pos_prefix = torch.zeros_like(pos_level[0])
        neg_prefix = torch.zeros_like(neg_level[0])
        for l in range(num_levels):
            level_idx = start_level + l
            q_prefix = q_prefix + q_level[level_idx]
            pos_prefix = pos_prefix + pos_level[level_idx]
            neg_prefix = neg_prefix + neg_level[level_idx]

            q_n = F.normalize(q_prefix, dim=-1)
            pos_n = F.normalize(pos_prefix, dim=-1)
            neg_n = F.normalize(neg_prefix, dim=-1)

            L_ret_l = self.contra_loss.forward_explicit_neg(q_n, pos_n, neg_n)
            L_ret_per_level.append(omega_l[l] * L_ret_l)

        L_ret = torch.stack(L_ret_per_level).sum()
        return L_ret, L_ret_per_level

    def compute_single_batch(self, batch: Tuple, logit_scale: Optional[float] = None) -> Dict:
        """Compute loss and metrics for a single batch.

        Args:
            batch: Input batch containing query, pool, instructions, and IDs
            logit_scale: Optional scaling factor for logits

        Returns:
            Dictionary containing loss values and metrics
        """
        # Unpack batch data. The retrieval-aware feasibility study (Components
        # 2-6, extensions/retrieval_aware_id_refinement/) adds a 5th element,
        # `hard_neg_pool`, via HardNegativeAugmentedDataset + its custom collate
        # function. Every other training config in this repo still yields the
        # original 4-tuple; nothing below this unpack differs for that case
        # (Component 2's "existing losses stay byte-for-byte unchanged"
        # requirement).
        hard_neg_pool = None
        if len(batch) == 5:
            query, pool, instruct, h_qid, hard_neg_pool = batch
        else:
            query, pool, instruct, h_qid = batch
        gpu_id = utils.get_rank()

        # Prepare masks and embeddings
        h_qid = h_qid.view(-1)
        q_img_mask = query['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        q_txt_mask = query['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_img_mask = pool['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_txt_mask = pool['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)

        q_img_emb = query['img_emb'].view(-1, query['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        q_txt_emb = query['txt_emb'].view(-1, query['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_img_emb = pool['img_emb'].view(-1, pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_txt_emb = pool['txt_emb'].view(-1, pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)

        bs = len(q_img_emb)
        device = q_img_emb.device

        # Concatenate query and pool embeddings
        all_img_emb = torch.cat((q_img_emb, p_img_emb), dim=0)
        all_txt_emb = torch.cat((q_txt_emb, p_txt_emb), dim=0)
        all_img_mask = torch.cat((q_img_mask, p_img_mask), dim=0)
        all_txt_mask = torch.cat((q_txt_mask, p_txt_mask), dim=0)

        # Combine features
        encode_feature = self.encoder(all_img_emb, all_txt_emb, all_img_mask, all_txt_mask)
        encode_feature = F.normalize(encode_feature)
        q_emb, p_emb = encode_feature[:bs], encode_feature[bs:]

        # Generate hash IDs for checking uniqueness
        qid_list = [hash(q_emb[i]) for i in range(len(q_emb))]
        p_did_list = [hash(p_emb[i]) for i in range(len(p_emb))]
        all_id_list = qid_list + p_did_list

        # Perform residual quantization.
        # Component 3: when the retrieval-aware study is enabled, use the manual
        # per-level breakdown loop (needed to isolate L_bal^(l) for Component 6's
        # gradient projection, and level_quantized for Component 5's prefix-depth
        # L_ret) INSTEAD OF self.residual_rq(...)'s fused-loss call -- never both,
        # to avoid double-applying the EMA codebook update for the same step. The
        # two paths are numerically equivalent for quant/Is (see
        # extensions/retrieval_aware_id_refinement/test_quantize_with_breakdown_equivalence.py);
        # this branch changes nothing about this method's behavior when
        # retrieval_aware_enabled is False (the default for every config outside
        # this study).
        level_quantized = None
        commit_losses = None
        diversity_losses = None
        if self.retrieval_aware_enabled:
            quant, Is, commit_losses, diversity_losses, level_quantized = self._quantize_with_breakdown(
                encode_feature.unsqueeze(0)
            )
            quant = quant.squeeze(0)
            Is = Is.squeeze(0)
            level_quantized = [lv.squeeze(0) for lv in level_quantized]
            # Component 4 [Fable fix]: L_RQ in the retrieval-aware path is PURE
            # commitment loss (the codebook-diversity term is handled entirely
            # separately, as L_bal, via Component 6's gradient projection -- it
            # must NOT also be fused into L_RQ here, or it would be counted twice:
            # once (correctly, layer-wise) via L_bal, and once (incorrectly,
            # un-layer-wise) if it leaked back into L_RQ through rq_loss.mean()).
            rq_loss = torch.stack(commit_losses)
        elif self.balance_split_enabled:
            # Direction-aware balance regularization (RQ2 modality-split study).
            # Reuse the already-computed per-sample modality masks (all_img_mask/
            # all_txt_mask) -- no new plumbing. rq_loss is recomposed as
            # commitment + modality-weighted diversity, which equals the fused
            # path exactly when lambda_img == lambda_txt (equivalence test). This
            # assumes commitment_weight == 1.0 (the library default for this
            # project's VectorQuantize construction, documented in the existing
            # equivalence test), so no per-term scaling is applied.
            quant, Is, commit_losses, weighted_div_losses = self._quantize_with_breakdown_modality_split(
                encode_feature.unsqueeze(0), all_img_mask, all_txt_mask
            )
            quant = quant.squeeze(0)
            Is = Is.squeeze(0)
            rq_loss = torch.stack(commit_losses) + torch.stack(weighted_div_losses)
        else:
            # freeze_codebook is an orthogonal knob to retrieval_aware_enabled
            # (used by the continued-training-no-new-losses control cell to
            # test the EMA-churn hypothesis in isolation from the retrieval-aware
            # losses) -- ResidualVQ.forward accepts it directly, so it must be
            # threaded through here too, not just inside _quantize_with_breakdown.
            quant, Is, rq_loss = self.residual_rq(
                encode_feature.unsqueeze(0), rand_quantize_dropout_fixed_seed=2023,
                freeze_codebook=self.freeze_codebook,
            )
            quant = quant.squeeze(0)
            Is = Is.squeeze(0)

        quant = F.normalize(quant)
        q_decode, p_decode = quant[:bs], quant[bs:]

        # Gather embeddings for compute contrastive score if in distributed training
        if self.gather_embedding:
            dist.barrier()
            all_p_emb = torch.cat(utils.GatherLayer.apply(p_emb), dim=0)
            all_p_decode = torch.cat(utils.GatherLayer.apply(p_decode), dim=0)
            all_h_qid = torch.cat(utils.GatherLayer.apply(h_qid), dim=0)
        else:
            all_p_emb = p_emb
            all_p_decode = p_decode
            all_h_qid = h_qid

        # Compute similarity scores
        score_org = torch.matmul(q_emb, all_p_emb.t()) # [bs, bs]
        score = torch.matmul(q_emb, all_p_decode.t()) # [bs, bs]
        sim_targets = h_qid

        # Compute accuracy
        _, max_idxs_org = torch.max(score_org, 1)
        max_idxs_org = all_h_qid[max_idxs_org]
        _, max_idxs = torch.max(score, 1)
        max_idxs = all_h_qid[max_idxs]
        accuracy_org = (max_idxs_org == sim_targets).sum() / bs
        accuracy = (max_idxs == sim_targets).sum() / bs
        
        # Compute losses
        rq_loss = 1e2 * rq_loss.mean()
        mse_loss = 1e2 * self.critic(q_decode, p_decode)
        cl_loss = self.contra_loss(q_emb, p_emb)

        # Compute perplexity
        perplexity = 0
        if Is.dim() != 1:
            for i in range(self.codebook_level):
                encodings = F.one_hot(Is[:, i].long(), self.codebook_vocab).type(quant.dtype)
                avg_probs = torch.mean(encodings, dim=0)
                perplexity_per_level = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-6)))
                perplexity += perplexity_per_level
            perplexity /= self.codebook_level
            
        loss = cl_loss + rq_loss + mse_loss

        # ============================================================
        # Retrieval-aware extras (Components 4-5). L_bal is computed here
        # per-level but is deliberately NOT added into `loss` above -- per the
        # mentor's point 6/7 and Component 6, it is applied to the encoder via
        # gradient-conflict projection in engine.py's
        # train_one_epoch_retrieval_aware, not through this method's ordinary
        # backward path. L_ret, by contrast, IS added into `loss` (mentor's
        # formula: L = L_RQ + beta*L_ret + sum_l lambda_l*L_bal^(l) -- L_ret
        # shares the normal backward with L_RQ; only the L_bal term is
        # projected).
        # ============================================================
        L_ret = None
        L_ret_per_level = None
        L_bal_per_level = None
        lambda_l = None
        if self.retrieval_aware_enabled and hard_neg_pool is not None:
            lambda_l = self.balance_budget * F.softmax(self.balance_level_logits, dim=0)  # [num_semantic_levels]
            diversity_losses_semantic = diversity_losses[1:] if self.modality_index else diversity_losses
            L_bal_per_level = [
                lambda_l[l] * diversity_losses_semantic[l] for l in range(self.num_semantic_levels)
            ]

            L_ret, L_ret_per_level = self._compute_retrieval_aware_L_ret(
                level_quantized, bs, hard_neg_pool, gpu_id
            )
            loss = loss + self.retrieval_aware_beta * L_ret

        self.collision_update(Is, all_id_list)

        if self.iter % 200 == 0:
            print('Query ID: ' + str(self.transform_row(Is[0])))
            print('Doc ID: ' + str(self.transform_row(Is[bs])))

        self.iter += 1

        # Prepare outputs
        outputs = {
            'Is': Is,
            'org_feateture': F.normalize(encode_feature),
            'quant': F.normalize(quant),
            'loss': loss,
            'rq_loss': rq_loss,
            'cl_loss': cl_loss,
            'mse_loss': mse_loss,
            'num_collision': self.num_collision,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'accuracy_org': accuracy_org
        }

        if self.retrieval_aware_enabled:
            outputs.update({
                'L_ret': L_ret if L_ret is not None else torch.zeros((), device=device),
                'L_ret_per_level': L_ret_per_level,
                'L_bal_per_level': L_bal_per_level,
                'lambda_l': lambda_l,
                'commit_losses': commit_losses,
                'diversity_losses': diversity_losses,
                'level_quantized': level_quantized,
            })

        return outputs

    def encode_mbeir_batch(self, batch: Dict, code_output: bool = False, 
                          encode_output: bool = False) -> Tuple:
        """Encode a batch of MBEIR data.
        
        Args:
            batch: Input batch
            code_output: Whether to return code output
            encode_output: Whether to return encoded output
            
        Returns:
            Tuple containing outputs and IDs
        """
        # Get hashed id_list
        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        # Compute embeddings
        img_emb, txt_emb = self.clip_model.encode_multimodal_input(
            batch["image_batched"],
            batch["txt_batched"],
        )
        img_mask = batch["image_mask_batched"].unsqueeze(-1)
        txt_mask = batch["txt_mask_batched"].unsqueeze(-1)

        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
        
        # Get outputs based on requested types
        if code_output and encode_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)
            return output['code'], output['encode'], id_list
        elif code_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['code']
        else:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['quant']
        return output, id_list

    def encode_extracted_batch(self, batch: Tuple, code_output: bool = False,
                             encode_output: bool = False, recon_output: bool = False) -> Tuple:
        """Encode a batch of extracted features.

        Args:
            batch: Input batch containing pool and IDs
            code_output: Whether to return code output
            encode_output: Whether to return encoded (rerank) output
            recon_output: Whether to return the decoded/reconstructed ('quant')
                embedding alongside the code, for reconstruction-quality
                diagnostics (MSE / cosine similarity vs. the original embedding)

        Returns:
            Tuple containing outputs and IDs
        """
        pool, id_list = batch
        gpu_id = utils.get_rank()

        # Prepare masks and embeddings
        img_mask = pool['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        txt_mask = pool['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        img_emb = pool['img_emb'].view(-1, pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        txt_emb = pool['txt_emb'].view(-1, pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)

        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."

        # Get outputs based on requested types
        if code_output and recon_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)
            return output['code'], output['quant'], id_list
        elif code_output and encode_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)
            return output['code'], output['rerank'], id_list
        elif code_output:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['code']
        else:
            output = self.inference(img_emb, txt_emb, img_mask, txt_mask)['quant']
        return output, id_list
    
    @torch.no_grad()
    def inference(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, 
                 img_mask: Optional[torch.Tensor] = None, 
                 txt_mask: Optional[torch.Tensor] = None) -> Dict:
        """Perform inference on input embeddings.
        
        Args:
            img_emb: Image embeddings
            txt_emb: Text embeddings
            img_mask: Optional mask for image features
            txt_mask: Optional mask for text features
            
        Returns:
            Dictionary containing various outputs
        """
        # Encode features
        encode_feature = self.encoder(img_emb, txt_emb, img_mask, txt_mask)
        encode = F.normalize(encode_feature)
        
        # Perform quantization
        quant, Is, _ = self.residual_rq(encode)
        quant = F.normalize(quant)
        rerank_feature = F.normalize(img_emb * img_mask + txt_emb * txt_mask)

        # Handle single sample case
        if quant.shape[0] == 1:
            quant = quant.squeeze(0)
            Is = Is.squeeze(0)

        # Handle unique code tracking
        if self.unique_code:
            updated_Is = []
            for i in range(Is.shape[0]):
                is_key = tuple(Is[i].cpu().numpy().tolist())
                if is_key in self.is_history:
                    stored_encode = self.is_history[is_key]['encode'].to(encode.device)
                    if not torch.allclose(stored_encode, encode[i], atol=1e-5):
                        self.is_history[is_key]['count'] += 1
                    count = self.is_history[is_key]['count']
                else:
                    self.is_history[is_key] = {'encode': encode[i].detach().cpu(), 'count': 0}
                    count = 0   
                
                updated_Is.append(torch.cat([Is[i], torch.tensor([count], device=Is.device)]))
            Is = torch.stack(updated_Is)

        # Prepare outputs
        outputs = {
            'code': Is,
            'quant': quant,
            'encode': encode,
            'rerank': rerank_feature
        }
        return outputs
    
    def forward(self,
                input: Optional[Tuple] = None,
                evaluation: bool = False,
                encode_mbeir_batch: bool = True,
                code_output: bool = False,
                encode_output: bool = False,
                recon_output: bool = False,
                logit_scale: Optional[float] = None) -> Union[Dict, Tuple]:
        """Forward pass of the model.

        Args:
            input: Input batch
            evaluation: Whether in evaluation mode
            encode_mbeir_batch: Whether to encode MBEIR batch
            code_output: Whether to return code output
            encode_output: Whether to return encoded output
            recon_output: Whether to return the decoded/reconstructed ('quant')
                embedding instead of the raw ('rerank') one (extracted-batch path only)
            logit_scale: Optional scaling factor for logits

        Returns:
            Model outputs
        """
        if evaluation:
            if encode_mbeir_batch:
                return self.encode_mbeir_batch(input, code_output=code_output, encode_output=encode_output)
            else:
                return self.encode_extracted_batch(input, code_output=code_output, encode_output=encode_output,
                                                    recon_output=recon_output)
        return self.compute_single_batch(input, logit_scale=logit_scale)