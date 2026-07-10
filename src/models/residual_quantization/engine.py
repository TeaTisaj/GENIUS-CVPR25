"""
Training and evaluation engine for Residual Quantization model.
"""

# Standard library
import gc

# Third-party
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Local modules
from models.uniir_clip import utils

# ================ Training Functions ================

def train_one_epoch(model, clip_model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config, feature_dict=None, use_amp=True):
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        clip_model: CLIP model for feature extraction
        data_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        epoch: Current epoch number
        gpu_id: GPU device ID
        scheduler: Learning rate scheduler
        global_step: Global training step counter
        scaler: Gradient scaler for mixed precision training
        config: Configuration object
        feature_dict: Optional dictionary of pre-computed features
        use_amp: if False, skip autocast/GradScaler entirely and run this exact
            loop in fp32 -- added to isolate the AMP-vs-fp32 variable from
            train_one_epoch_retrieval_aware's fp32 loop when comparing the two
            (see [[project_retrieval_aware_rq_refinement]]); default True keeps
            every other config's behavior byte-identical.

    Returns:
        Dictionary of training metrics
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # Initialize metrics
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("accuracy_org", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("cl_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("mse_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("rq_loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("num_collision", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("perplexity", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    header = f"Train Epoch: [{epoch}]"
    print_freq = config.trainer_config.print_freq
    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0

    model.module.codebook_reset()
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(batch)
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # Optimizer step with gradient scaling
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            model.zero_grad()
            scheduler.step()
            accumulation_counter = 0

        # Update metrics
        metric_logger.update(accuracy=outputs["accuracy"].item())
        metric_logger.update(accuracy_org=outputs["accuracy_org"].item())
        metric_logger.update(loss=loss.item() * accumulation_steps)
        metric_logger.update(cl_loss=outputs["cl_loss"].item() * accumulation_steps)
        metric_logger.update(mse_loss=outputs["mse_loss"].item() * accumulation_steps)
        metric_logger.update(rq_loss=outputs["rq_loss"].item() * accumulation_steps)
        metric_logger.update(num_collision=outputs["num_collision"])
        metric_logger.update(perplexity=outputs["perplexity"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_e2e(model, clip_model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config, 
                    discriminator=None, opt_disc=None, scaler_disc=None, scheduler_disc=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("cl_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("mse_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("rq_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("gan_loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("num_collision", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("perplexity", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0

    model.train()
    model.module.codebook_reset()

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # autocast for mixed precision
        with autocast(enabled=False):
            outputs = model(batch, logit_scale=clip_model.get_logit_scale())
            loss = outputs["loss"]
            
            if discriminator is not None:
                disc_factor = adopt_weight(1, epoch * len(data_loader)+i, threshold=500)

                disc_real = discriminator(outputs["org_feateture"].detach().clone())
                disc_fake = discriminator(outputs["decode_feature"].detach().clone())
                g_loss = -disc_factor * 0.01 * torch.mean(disc_fake)
                outputs['cl_loss'] += g_loss * 0
                loss += g_loss * 0
                
                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = 0.5 * disc_factor * (d_loss_real + d_loss_fake)  * 0

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        loss = loss / accumulation_steps

        # Use scaler for backward
        scaler.scale(loss).backward(retain_graph=True)
        if discriminator is not None:
            scaler_disc.scale(gan_loss).backward()
        
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        scaler_disc.unscale_(opt_disc)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if discriminator is not None:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            scaler_disc.step(opt_disc)
            scaler_disc.update()

            optimizer.zero_grad()
            opt_disc.zero_grad()
            
            scheduler.step()
            scheduler_disc.step()
            accumulation_counter = 0

        metric_logger.update(accuracy=outputs["accuracy"].item())  # We scale back the loss for logging.
        metric_logger.update(loss=loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(cl_loss=outputs["cl_loss"].item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(mse_loss=outputs["mse_loss"].item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(rq_loss=outputs["rq_loss"].item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(gan_loss=gan_loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(num_collision=outputs["num_collision"])
        metric_logger.update(perplexity=outputs["perplexity"])
        metric_logger.update(lr=optimizer.param_groups[1]["lr"])  # TODO: might need to loop through all param groups

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# ================ Retrieval-Aware RQ Refinement Feasibility Study (Component 6) ================
#
# See extensions/retrieval_aware_id_refinement/ and
# .claude/plans/so-this-is-the-swirling-quiche.md. `train_one_epoch_retrieval_aware`
# below is used ONLY by this study's Slurm scripts
# (extensions/retrieval_aware_id_refinement/scripts/); `train_one_epoch` above
# is completely untouched and remains the training loop for every other
# experiment in this repo.


def _move_batch_to_device(batch, gpu_id):
    """[Bug found and fixed during real-cluster verification, not anticipated
    by the plan text] `RQ.compute_single_batch` only ever explicitly moves the
    embedding/mask tensors (query['img_emb'], etc.) to `gpu_id` -- `h_qid`
    (and `instruct`) are left on whatever device the DataLoader produced them
    on (CPU). The main training path never notices this because
    `DistributedDataParallel(model, device_ids=[gpu_id])`'s forward silently
    scatters every input tensor to that device before calling the wrapped
    module -- an easy-to-miss DDP behavior this whole codebase has always
    implicitly relied on. `perturb_and_maybe_update_balance_logits` below calls
    `model_without_ddp(val_batch)` -- the RAW module, bypassing DDP's forward
    entirely -- so that automatic transfer does not happen, and `h_qid`
    reaches `all_h_qid = torch.cat(utils.GatherLayer.apply(h_qid), dim=0)`
    still on CPU, which NCCL's all_gather cannot handle
    ("RuntimeError: No backend type associated with device type cpu"),
    confirmed by reproducing this exact crash on a real feasibility run
    before adding this fix. Recursively moves every tensor leaf in the
    (query_dict, pool_dict, instruct, h_qid[, hard_neg_pool_dict]) batch
    structure; a no-op for any tensor already on the right device.
    """
    if torch.is_tensor(batch):
        return batch.to(gpu_id, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, gpu_id) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_move_batch_to_device(v, gpu_id) for v in batch]
        return type(batch)(moved)
    return batch


def perturb_and_maybe_update_balance_logits(model, val_batch, gpu_id, state, sigma=0.1):
    """Component 4 (Method 1) -- the 'preferred, cheaper alternative' to a
    literal DARTS-style one-step lookahead for learning `a_l`
    (`balance_level_logits`), using a validation retrieval objective (mentor's
    point 5), as opposed to the plan's 'explicit fallback' of a frozen,
    unlearned prior. See `RQ.__init__`'s docstring in residual_quantization.py
    for the full rationale for choosing this over the DARTS-style option.

    **Causal note, important for correctly reading this function:** `a_l` has
    NO direct effect on `L_ret` within a single frozen-encoder forward pass --
    `L_ret`'s formula never references `balance_level_logits`/`lambda_l`, only
    `L_bal` does. `a_l`'s only effect on retrieval quality is INDIRECT,
    mediated by training dynamics: `a_l` sets `lambda_l`, which scales
    `L_bal^(l)`, which (via this file's gradient-conflict projection, below)
    shapes the encoder's parameter updates over SUBSEQUENT training steps.
    Consequently this function does NOT compare validation L_ret before/after
    perturbing `a_l` within the same step -- that comparison would be causally
    vacuous, since the encoder has not been updated yet at that point, so
    `L_ret` cannot possibly depend on `a_l` there. Instead, this evaluates
    validation L_ret once every call (the caller decides the interval, e.g.
    every `print_freq` or a dedicated `perturb_every` steps) and compares it to
    the validation L_ret measured at the END of the PREVIOUS window -- an
    interval over which the CURRENT `a_l` value was actually in effect (via the
    projected encoder updates it influenced during that window). This is a
    coarser-grained, noisier signal than a clean one-step lookahead would give,
    but it is causally meaningful and simple enough to implement and verify
    without live GPU debugging, in the spirit of evolution-strategies-style
    zeroth-order optimization.

    `a_l` is updated by directly mutating `.data`; it is NEVER touched via
    `.backward()`/`optimizer.step()` (it has `requires_grad=False` -- see
    `RQ.__init__`), so this function is the ONLY place its value changes.

    Args:
        model: DDP-wrapped RQ model, currently in train mode.
        val_batch: one held-out validation batch (same 5-tuple format the
            training batches use, from a separate HardNegativeAugmentedDataset
            instance built over a held-out query slice).
        gpu_id: device to move `val_batch` to before the raw (non-DDP-wrapped)
            forward call -- see `_move_batch_to_device`'s docstring above for
            why this is required here specifically (DDP's forward normally
            does this automatically, but `model_without_ddp(...)` bypasses
            that).
        state: a plain dict, created once by the caller before the training
            loop starts and passed back in on every call, with keys
            'last_val_L_ret' (float or None) and 'last_perturbation' (tensor
            or None). Mutated in place.
        sigma: perturbation standard deviation (logit-scale units).

    Returns:
        val_L_ret (float): the just-measured validation L_ret, for logging.
    """
    model_without_ddp = model.module
    was_training = model_without_ddp.training
    # eval() freezes the EMA codebook (no update happens for this diagnostic
    # forward pass) and, since we use torch.no_grad() below, we don't need the
    # straight-through-estimator gradient path that eval() mode disables --
    # this measurement is forward-value-only.
    model_without_ddp.eval()
    val_batch = _move_batch_to_device(val_batch, gpu_id)
    try:
        with torch.no_grad():
            outputs = model_without_ddp(val_batch)
            val_L_ret = outputs['L_ret'].item()
    finally:
        if was_training:
            model_without_ddp.train()

    a_l = model_without_ddp.balance_level_logits
    if state['last_val_L_ret'] is not None and state['last_perturbation'] is not None:
        improved = val_L_ret < state['last_val_L_ret']
        with torch.no_grad():
            if improved:
                a_l.data.add_(state['last_perturbation'])  # reinforce: keep going the same direction
            else:
                a_l.data.sub_(state['last_perturbation'])  # revert: that direction did not help

    new_perturbation = torch.randn_like(a_l.data) * sigma
    with torch.no_grad():
        a_l.data.add_(new_perturbation)
    state['last_perturbation'] = new_perturbation
    state['last_val_L_ret'] = val_L_ret
    return val_L_ret


def train_one_epoch_retrieval_aware(model, data_loader, optimizer, epoch, gpu_id, scheduler,
                                     global_step, config, val_loader=None, perturbation_state=None):
    """Component 6: gradient-conflict-projection training step for the
    retrieval-aware RQ refinement feasibility study.

    [Fable fix, disqualifying bug in the naive version of this design] Runs
    ENTIRELY in fp32 -- NO `autocast`, NO `GradScaler` anywhere in this
    function. `train_one_epoch`'s AMP path scales `loss` by ~65536x before
    `.backward()`, then `scaler.step(optimizer)` unscales EVERY gradient in
    `.grad` (dividing by that same factor) before applying the update. The
    manually-injected projected balance gradient below is computed via plain,
    unscaled `torch.autograd.grad` calls; if this ran under AMP, that
    already-unscaled component would get divided by ~65536 a SECOND time by
    `scaler.step`, shrinking it to numerical noise -- Method 2 (gradient
    projection) would silently do nothing while logging plausible numbers.
    Running this whole function in fp32 removes that entire class of silent
    corruption, which matters here because trusting the gradient-cosine
    diagnostic (feasibility criterion 4) is the actual point of this study.

    [Fable fix] 1-GPU-only by design. `train.py` wraps the model in
    `DDP(..., find_unused_parameters=True)` whenever distributed mode is on,
    which includes this project's standard single-node `torchrun` launch even
    for 1-GPU jobs. On 1 GPU, `torch.autograd.grad(..., inputs=...)` does not
    trigger DDP's reducer/all-reduce hooks, so the scheme below is safe as
    written. On >1 GPU it would be SILENTLY WRONG: the real `.backward()`
    call's gradients get all-reduced across ranks as usual, but the manually
    injected projected balance component is computed per-rank via
    `torch.autograd.grad` and is NEVER synced -- each rank would apply a
    different update to nominally-shared parameters, a silent divergence with
    no error message. Hence the hard assert below; do not remove it or reuse
    this function unmodified for a multi-GPU scale-up without adding an
    explicit all-reduce on the manually injected gradient component first.

    [Deliberate, conservative interpretation of the plan's "one real backward
    for L_RQ + beta*L_ret"] The mentor's abstract formula is
    `L = L_RQ + beta*L_ret + sum_l lambda_l*L_bal^(l)`, and the mentor
    describes `L_RQ` as preserving "quantization AND RECONSTRUCTION"
    (`the-newest-rq-plan-3-7-26.md` point 3). In this codebase's concrete
    implementation, "reconstruction" is `mse_loss` and the in-batch alignment
    term is `cl_loss` -- both already part of `compute_single_batch`'s
    existing, essential (non-retrieval-aware) training objective, entirely
    separate in code from `rq_loss` (pure commitment loss, in the
    retrieval-aware path) and from the NEW `L_bal`/`L_ret` terms. Reading
    Component 6's "one real backward for L_RQ + beta*L_ret" as literally
    excluding `cl_loss`/`mse_loss` would silently drop the two loss terms this
    codebase already relies on to keep the quantizer non-degenerate -- an
    unrelated, obviously-harmful confound with nothing to do with testing
    retrieval-aware refinement. This function therefore backpropagates
    `outputs['loss']` as already assembled by `compute_single_batch`
    (`cl_loss + rq_loss + mse_loss + beta*L_ret`) as the single real backward
    pass, and separately, additively injects the projected `L_bal` gradient
    onto the encoder's `.grad` afterward -- matching the mentor's literal
    formula's STRUCTURE (base objective plus a separately-handled balance
    term) while preserving this codebase's existing non-degenerate training
    signal.

    Args:
        model: DDP-wrapped RQ model with `retrieval_aware_config.enabled: true`.
        data_loader: training DataLoader yielding HardNegativeAugmentedDataset
            5-tuples.
        optimizer, epoch, gpu_id, scheduler, global_step, config: as in
            `train_one_epoch`.
        val_loader: DataLoader over a held-out query slice (same 5-tuple
            format), used by Component 4's validation-guided perturbation of
            `a_l`. If None, `a_l` stays at its initial value (Component 4's
            explicit-fallback-equivalent behavior for a single run, logged as
            such) -- required to be non-None per this study's design, but not
            hard-asserted so this function still runs for a quick
            `L_ret`/gradient-projection-only smoke test without a val split.
        perturbation_state: mutable dict for
            `perturb_and_maybe_update_balance_logits`, created once by the
            caller (`train.py`) before the epoch loop and threaded across
            epochs; if None, a fresh one is created (meaning cross-epoch state
            is lost -- caller should pass a persistent dict for a real run).

    Returns:
        Dictionary of training metrics (same convention as `train_one_epoch`).
    """
    assert utils.get_world_size() == 1, (
        "train_one_epoch_retrieval_aware is 1-GPU-only: the manually injected "
        "projected balance gradient is computed per-rank via torch.autograd.grad "
        "and is never all-reduced, so >1 GPU would silently diverge across ranks. "
        "See this function's docstring before attempting a multi-GPU scale-up."
    )
    assert config.trainer_config.gradient_accumulation_steps == 1, (
        "train_one_epoch_retrieval_aware does not support gradient accumulation "
        "steps > 1: combining PCGrad-style manual gradient injection with "
        "accumulation adds bookkeeping complexity with no benefit at this "
        "feasibility study's scale. Set trainer_config.gradient_accumulation_steps: 1."
    )

    model.train()
    model_without_ddp = model.module
    assert model_without_ddp.retrieval_aware_enabled, (
        "train_one_epoch_retrieval_aware requires retrieval_aware_config.enabled: true."
    )

    if perturbation_state is None:
        perturbation_state = {'last_val_L_ret': None, 'last_perturbation': None}

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("accuracy_org", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("cl_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("mse_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("rq_loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("L_ret", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("val_L_ret", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("grad_cos_mean", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("grad_conflict_frac", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("num_collision", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("perplexity", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    header = f"Train Epoch (retrieval-aware): [{epoch}]"
    print_freq = config.trainer_config.print_freq

    model_without_ddp.codebook_reset()

    val_iter = iter(val_loader) if val_loader is not None else None

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # No autocast: this entire function runs in fp32 (see docstring).
        outputs = model(batch)
        loss = outputs["loss"]

        encoder_params = list(model_without_ddp.encoder.parameters())
        lambda_l = outputs["lambda_l"]
        L_bal_per_level = outputs["L_bal_per_level"]
        L_ret_per_level = outputs["L_ret_per_level"]
        num_levels = model_without_ddp.num_semantic_levels

        # Accumulator for the projected L_bal gradient, one slot per encoder
        # parameter, summed across all `num_levels` semantic levels.
        projected_accum = [torch.zeros_like(p) for p in encoder_params]
        cos_sims = []
        conflict_count = 0
        total_pairs = 0

        for l in range(num_levels):
            optimizer.zero_grad()
            g_bal_l = torch.autograd.grad(L_bal_per_level[l], encoder_params, retain_graph=True, allow_unused=True)

            optimizer.zero_grad()
            g_ret_l = torch.autograd.grad(L_ret_per_level[l], encoder_params, retain_graph=True, allow_unused=True)

            for p_idx, (gb, gr) in enumerate(zip(g_bal_l, g_ret_l)):
                if gb is None:
                    continue
                if gr is not None:
                    total_pairs += 1
                    gb_flat, gr_flat = gb.flatten(), gr.flatten()
                    dot = (gb_flat * gr_flat).sum()
                    cos = dot / (gb_flat.norm() * gr_flat.norm()).clamp_min(1e-12)
                    cos_sims.append(cos.item())
                    # PCGrad-style projection (mentor's point 6, exact formula):
                    # if the dot product is negative, remove the conflicting
                    # component; otherwise keep the original balancing gradient.
                    if dot < 0:
                        conflict_count += 1
                        gr_sq = (gr_flat * gr_flat).sum().clamp_min(1e-12)
                        gb = gb - (dot / gr_sq) * gr
                projected_accum[p_idx] = projected_accum[p_idx] + gb

        # ONE real backward for the base objective (see docstring for the
        # deliberate, conservative reading of "L_RQ + beta*L_ret" as this
        # codebase's outputs['loss'] = cl_loss + rq_loss + mse_loss + beta*L_ret).
        optimizer.zero_grad()
        loss.backward()

        # Manually add the accumulated projected balance gradient onto the
        # encoder's .grad before optimizer.step() (mentor's point 7: "final
        # generator update combines L_RQ, L_ret, and the projected balancing
        # gradients").
        for p, proj_g in zip(encoder_params, projected_accum):
            if p.grad is None:
                p.grad = proj_g.clone()
            else:
                p.grad = p.grad + proj_g

        optimizer.step()
        scheduler.step()
        global_step += 1

        grad_cos_mean = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
        grad_conflict_frac = conflict_count / total_pairs if total_pairs > 0 else 0.0

        val_L_ret = perturbation_state.get('last_val_L_ret', 0.0) or 0.0
        if val_loader is not None and (i % print_freq == 0):
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            val_L_ret = perturb_and_maybe_update_balance_logits(model, val_batch, gpu_id, perturbation_state)

        if i % print_freq == 0:
            print(f"  [step {global_step}] lambda_l={lambda_l.detach().cpu().tolist()}")

        metric_logger.update(accuracy=outputs["accuracy"].item())
        metric_logger.update(accuracy_org=outputs["accuracy_org"].item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(cl_loss=outputs["cl_loss"].item())
        metric_logger.update(mse_loss=outputs["mse_loss"].item())
        metric_logger.update(rq_loss=outputs["rq_loss"].item())
        metric_logger.update(L_ret=outputs["L_ret"].item())
        metric_logger.update(val_L_ret=val_L_ret)
        metric_logger.update(grad_cos_mean=grad_cos_mean)
        metric_logger.update(grad_conflict_frac=grad_conflict_frac)
        metric_logger.update(num_collision=outputs["num_collision"])
        metric_logger.update(perplexity=outputs["perplexity"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # NOTE: parenthesize the multiplication before .cpu().tolist() -- without
    # the parens, method-chaining precedence applies .cpu().tolist() to the
    # softmax result FIRST (producing a plain Python list), and then attempts
    # `float * list`, raising "can't multiply sequence by non-int of type
    # 'float'" (caught by this task's own end-to-end synthetic-batch test).
    lambda_l_final = (
        model_without_ddp.balance_budget
        * F.softmax(model_without_ddp.balance_level_logits.detach(), dim=0)
    ).cpu().tolist()
    stats["balance_lambda_l_learned_perturbation"] = lambda_l_final
    return stats


# ================ Evaluation Functions ================

@torch.no_grad()
def eval_engine(model, clip_model, data_loader, gpu_id, config):
    """Evaluate the model.
    
    Args:
        model: The model to evaluate
        clip_model: CLIP model for feature extraction
        data_loader: DataLoader for evaluation data
        gpu_id: GPU device ID
        config: Configuration object
        
    Returns:
        Dictionary of evaluation metrics
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # Initialize metrics
    metric_logger.add_meter("accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("cl_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("mse_loss", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("rq_loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("num_collision", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("perplexity", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    
    header = "Test:"
    print_freq = config.evaluator.print_freq

    model.eval()
    model.module.codebook_reset()
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move batch to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(gpu_id, non_blocking=True)

        # Forward pass with mixed precision
        with autocast():
            q_emb, p_emb = clip_model.get_embedding(batch)
            outputs = model((q_emb, p_emb))
            loss = outputs["loss"]

        # Update metrics
        metric_logger.update(accuracy=outputs["accuracy"].item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(cl_loss=outputs["cl_loss"].item())
        metric_logger.update(mse_loss=outputs["mse_loss"].item())
        metric_logger.update(rq_loss=outputs["rq_loss"].item())
        metric_logger.update(num_collision=outputs["num_collision"])
        metric_logger.update(perplexity=outputs["perplexity"])

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
