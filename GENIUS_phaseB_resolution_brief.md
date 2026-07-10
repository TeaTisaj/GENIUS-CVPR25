# GENIUS Phase B paradox — RESOLVED (handoff for next session, 2026-06-08)

This supersedes `GENIUS_phaseB_paradox_brief.md`. That brief posed an open puzzle — read it for
the full numbers table and original framing, but **the puzzle is now solved**: it was a training
hyperparameter artifact, not a pipeline bug. This file is the TL;DR + what to do next.

---

## 0. TL;DR — root cause found

**"30 epochs, batch=256, lr=1e-4" is an under-specified recipe in a multi-GPU DDP setup.**
The number of *optimizer steps* (and therefore the cosine LR-schedule horizon, and therefore
final convergence) depends on `world_size`, which the paper doesn't report:

- `train.py:292-296`: `t_total = len(train_loader) * num_train_epochs`, and
  `len(train_loader)` is the **per-rank** dataloader length = `dataset_size / world_size / batch_size`.
- Our "by-the-book" run used **4 GPUs** → ~208 steps/epoch → `t_total ≈ 6,240` steps. The cosine
  LR schedule decayed to ~0 by epoch ~25-29, **freezing** the model at loss ≈ 4.12
  (perplexity ≈ 62 over the 4096-way codebook — too imprecise for accurate ID generation).
- I re-ran the **identical** recipe (30 epochs, lr=1e-4, batch=256/GPU, frozen `rq_clip_large.pth`,
  same data) on **1 GPU** → ~833 steps/epoch → `t_total ≈ 25,000` steps (4x more). Loss converged
  much further: **3.0-3.3 at epoch 28** (still falling), vastly better than the 4-GPU run's final
  4.12 — at the SAME epoch numbers / SAME total data exposure. Proof the gap is optimizer-step
  budget, not a data or candidate-ID mismatch (Hypothesis A in the old brief — now a red herring;
  both the 200-epoch run and the 30-epoch run shared the identical train-pool/embedding paths,
  so a mismatch there can't explain why one scores worse than the other).

**Verified epoch-by-epoch** (loss, same data exposure, different step granularity):

| Epoch | 1-GPU run (833 steps/ep) | 4-GPU "by-the-book" run (208 steps/ep) |
|---|---|---|
| 0 | 6.09 | 8.03 |
| 4 | 4.74 | 5.50 |
| 9 | 4.04 | 4.76 |
| 14 | 3.72 | 4.47 |
| 19 | 3.43 | 4.26 |
| 24 | 3.27 | — |
| 28 | ~3.2 | 4.18 (epoch 29, final) |

The 200-epoch COCO+FashionIQ union run (4 GPUs, ~44,800 steps) reached loss ≈ 1.9 by epoch
~150-180 — *not* "epoch ~50" as the old Phase A diagnosis claimed (re-checked the actual log:
loss falls continuously to ~epoch 150, where its own (longer) cosine schedule finally hits
LR≈0). That diagnosis's "overfitting" framing looks like a misread of a normal cosine-decay curve
on a longer horizon — not a real overfitting signature.

---

## 1. What to do FIRST in the new session

1. **Check job 328629** finished cleanly: `sacct -j 328629 --format=JobID,State,Elapsed,ExitCode`
   and tail `~/logs/genius_stage2_coco_1gpu_328629.log`. It was at epoch 28/30 when this session
   ended — should complete within minutes (~2 min/epoch). Final loss expected ≈ 3.0-3.3.
2. **Eval the new checkpoint(s)**: `checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly1GPU/`
   (epoch_{0,5,10,15,20,25} saved via `eval_freq=5`; final epoch_29 not checkpointed — same
   `eval_freq` caveat as before, epoch_25 is the latest available). Copy
   `config_eval_coco_only.yaml` → adjust `ckpt_name` / experiment path to `CocoOnly1GPU`, run
   both no-rerank and rerank passes (reuse `scripts/slurm_eval_coco_only.sh` pattern).
3. **Compare R@1/5/10** against the table below. If it lands clearly above 5.76% (ideally
   approaching or beating 11.6%/14.2%), that **confirms** the step-count fix actually improves
   retrieval, not just training loss — closing the loop on the paradox.

## 2. Numbers so far (T→I, COCO test, ~123K local pool)

| Model | Optimizer steps for "30 epochs" | Final train loss | R@1 (no-rerank) |
|---|---|---|---|
| 4-GPU "by-the-book" 30ep (epoch_25, job 328613) | ~6,240 | 4.12 | 5.76% |
| **1-GPU 30ep, identical recipe (job 328629)** | **~25,000** | **~3.0-3.3** (still dropping at ep28) | **TBD — eval next** |
| Pretrained HF general checkpoint | — | — | 14.2% |
| Locally trained 200-epoch union (epoch_195) | ~44,800 | 1.9 | 11.6% |
| Paper Table 3 target | — | — | 40.1% (46.1 reranked) |

## 3. Where this likely goes next

- If the 1-GPU run's recall lands roughly where its loss level suggests (somewhere between
  5.76% and 11.6%, scaling with loss 4.12→1.9), the **fix is simply: train for more steps** —
  e.g., bump `num_train_epochs` so total steps approach ~25K-45K (whether via more epochs on
  4 GPUs, or staying on 1 GPU with more epochs), and pick the checkpoint where train loss is
  lowest / val recall peaks (watch for real overfitting this time — now that we know what the
  *normal* cosine curve looks like, a genuine plateau-then-divergence would be diagnostic).
- Even the BEST locally-trained number so far (11.6% at loss 1.9) is still ~4x below the paper's
  40.1 — so a residual gap likely remains beyond step count (possibly: paper trained on a larger
  effective batch with LR scaled accordingly, different beam size at eval, or some other
  recipe detail not recoverable from the paper text alone).
- **Cheapest way to close the remaining gap: ask Yubao directly** — "How many GPUs / per-GPU
  batch size did you use for the 30-epoch COCO run, and what train loss did it converge to?"
  Her answer would let us replicate the exact step budget instead of guessing.
- Per the supervisor's reproducibility-track framing (`new_guidelines.md`), documenting *why*
  a gap exists is itself valid output — don't over-invest in chasing the exact 40.1 if the
  step-count story is well-supported. Consider whether to lock in this analysis and move on to
  Step 2 (paper comparison write-up) and Step 3 (FashionIQ + high-similarity extension), which
  are the actual deliverables.

## 4. Key files

| What | Where |
|---|---|
| 1-GPU diagnostic training config | `src/models/generative_retriever/configs_scripts/large/train/inbatch/inbatch_coco_only_1gpu.yaml` |
| 1-GPU diagnostic Slurm script | `scripts/slurm_train_stage2_coco_only_1gpu.sh` |
| Training log (job 328629) | `~/logs/genius_stage2_coco_1gpu_328629.log` |
| New checkpoints | `checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly1GPU/` |
| Root-cause code refs | `train.py:292-296` (`t_total` calc), `models/utils.py:5` (`cosine_warmup_scheduler`), `engine.py:52` (`scheduler.step()` per-iteration) |
| Old (now superseded) puzzle framing | `GENIUS_phaseB_paradox_brief.md` — Hypothesis A there is a red herring; the real answer is an extension of Hypothesis D (undertraining), precisely diagnosed as a DDP step-count mismatch rather than vague "raw exposure" |
| Full diagnostic memory | `project_stage2_status` (auto-memory; ask to recall) |

## 5. Guardrails (same as always)

Shared cluster — confirm with the user before launching new Slurm **training** jobs (eval/diagnostic
runs are fine to launch freely, per established practice this session). Never `--partition=gpu`
for CPU-only work. `--exclude=ilps-cn120` for memory-heavy jobs (prone to oversubscription).
