# GENIUS Phase 1 — Reproduction Pipeline, Results, and Gap Analysis

This is the Deliverable-0 writeup required by the mentor before Phase 2
(`GENIUS_phase2_pool_composition.md`) begins. It consolidates what's already
scattered across `GENIUS_reproduction_brief.md` (paper targets) and this
session's eval/training logs (actual numbers).

## 1. Pipeline summary

Three frozen/trained stages (see `CLAUDE.md` "Architecture" for the full
description):
- **Stage 0 — CLIP-SF encoder** (frozen): UniIR's CLIP ViT-L/14 score-fusion
  model, `checkpoint/CLIP_SF/clip_sf_large.pth`. Produces 768-dim joint
  image/text embeddings for every query and candidate.
- **Stage 1 — Residual Quantization** (frozen, official checkpoint
  `checkpoint/rq_clip_large.pth`, no retraining — per Yubao Tang's advice).
  Combiner fuses image+text embeddings, RQ quantizes into a 9-digit discrete
  code: digit 0 = modality (confirmed empirically this session: 0 = image,
  1 = text), digits 1–8 = 8 levels of semantic codebook (vocab 4096 each).
- **Stage 2 — T5-small generative decoder** (trained from scratch): an
  `embed_projector` maps the 768-dim fused embedding into 30 prefix tokens;
  T5-small is trained to generate the target's 9-digit code sequence.
  Training uses Beta-distribution query/target embedding mixing.
- **Inference**: trie-constrained beam search (beam size 50, `trie_cpp`) over
  the candidate pool's valid code sequences, with optional embedding-based
  re-ranking (`GENIUS_R` variant).

## 2. Training configurations actually used

Two Stage 2 models were trained this session, both reusing the same frozen
Stage 0/1:

| | "Sanity" model | Full-union model (current) |
|---|---|---|
| Training data | COCO + FashionIQ only | Full M-BEIR union (16 dataset/tasks, ~1.3M queries/epoch) |
| Epochs | 31 (sanity run) | 100 |
| Steps/epoch | 833 | 5,206 |
| Total steps | ~25,800 | 520,600 |
| Hardware | 1× A6000 | 1× A6000 |
| `train_batch_size` | 256 | 256 |
| `hard_neg_num` / `in_batch_neg_num` | 0 / 0 | 0 / 0 |
| `evaluator.enable_eval` | false | false |
| LR schedule | cosine, hit ~0 early (truncated) | cosine, fully decayed by ~epoch 95 |
| Checkpoint used for eval | mid-training snapshot | `genius_t5small_epoch_95.pth` (last saved, **not** validated as "best" — no eval during training) |

## 3. Results vs paper

Paper targets (Table 1/3 of the GENIUS paper, COCO-only train/eval and
Fashion task-specific pools, already transcribed in
`GENIUS_reproduction_brief.md` §3):

| Task | Metric | Paper (GENIUS) | Paper (GENIUS_R) |
|---|---|---|---|
| COCO T→I | R@1 / R@5 / R@10 | 40.1 / 66.2 / 75.8 | 46.1 / 74.0 / 82.7 |
| COCO I→T | R@5 (task-specific pool) | 83.2 | 91.1 |
| FashionIQ (qi,qt)→ci | R@10 | 13.1 | 19.2 |

Our actual numbers:

| Model | Task | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|---|
| Sanity (31 ep, COCO+FashionIQ) | COCO T→I | 20.61% | — | 42.22% | — | — |
| Sanity (31 ep, COCO+FashionIQ) | FashionIQ | — | — | 18.58% | — | — |
| **Full-union (100 ep, job 329846)** | **COCO T→I** (job 329954) | **8.10%** | 21.14% | 29.42% | — | — |
| Full-union (100 ep) | COCO I→T (job 329954) | 6.38% | 18.12% | 25.83% | — | — |
| Full-union (100 ep) | FashionIQ (job 329965) | — | — | 18.33% | 22.16% | 25.43% |

Observations:
- FashionIQ recall is close to the paper's R@10=19.2% in both models
  (18.58% and 18.33%) — this part of the pipeline reproduces reasonably well.
- COCO is far below paper (8.10% vs 40.1% R@1) for the full-union model, and
  notably **worse than our own earlier 31-epoch sanity model** (20.61%)
  despite having a complete (non-truncated) LR schedule and 20× more training
  steps. This is the main open gap.

## 4. Gap analysis — why is full-union COCO so much lower?

Ranked by how directly we've confirmed each factor this session:

1. **Capacity dilution across 16 dataset/tasks.** The full-union model must
   represent a far more diverse query/target distribution than the 2-dataset
   sanity model in the same T5-small capacity and the same 9-digit RQ code
   space — shared across all 16 dataset/tasks rather than specialized for
   COCO. Training-time metrics (`Level123_acc` plateauing ~0.57, train R@1
   ~0.32 by epoch 99) show the model is still improving slowly when the LR
   schedule ends, suggesting the model hasn't saturated for the harder task,
   not that it diverged.
2. **No validation-based checkpoint selection.** `evaluator.enable_eval:
   false` for both training runs means epoch_95 is simply the last saved
   checkpoint, never confirmed as the best for COCO specifically. An earlier
   epoch might generalize better before any late-stage overfitting to the
   union's most frequent datasets.
3. **No hard or in-batch negatives.** `hard_neg_num`/`in_batch_neg_num` are
   both 0 in both training configs — the decoder never sees explicit hard
   negatives during Stage 2 training, which likely matters more at the
   full-union scale (more semantically similar distractors across datasets)
   than for a 2-dataset run.
4. **Effective batch size.** Both runs use `train_batch_size: 256` on a
   single GPU (forced after the earlier 4-GPU run OOM'd on embedding
   gathering) — the paper's setup likely uses a larger effective batch via
   multi-GPU gradient/embedding gathering, which affects the in-batch
   negative pool size implicitly used by the contrastive-style training
   objective.
5. **A separate, now-fixed, evaluation pipeline bug** (not a training issue,
   but worth recording): the `rerank=true` eval code path
   (`generate_codes_for_dataset` in `src/common/mbeir_generative_retriever.py`)
   was missing `dist.barrier()` synchronization around its `dist.gather()`
   calls, unlike the `rerank=false` path. This caused the FashionIQ eval
   (job 329955, 4 GPUs) to desync and crash via a 1-hour NCCL collective
   timeout, producing zero metrics. Fixed by adding the missing barriers;
   confirmed working via job 329965 (1 GPU rerun, completed in ~10 min). This
   bug did not affect the COCO numbers above (COCO eval uses `rerank: false`,
   which already had the barriers), but is recorded here since it could
   silently corrupt any future `rerank: true` multi-GPU eval if not carried
   forward.

None of the above is confirmed as *the* root cause yet — items 1–4 are
plausible contributors based on the training logs, not proven via ablation.
A clean next step (out of scope for Phase 2, noted for later) would be to
rerun Stage 2 with `evaluator.enable_eval: true` to track per-epoch COCO
validation recall and see whether an earlier checkpoint would have scored
higher, isolating item 2 from items 1/3/4.

## 5. Segue into Phase 2

The full-union model (job 329846, checkpoint `genius_t5small_epoch_95.pth`)
is the frozen retriever Phase 2 will reuse as-is — Phase 2 does not retrain
anything, and does not aim to close the gap documented above. Instead, Phase
2 (`GENIUS_phase2_pool_composition.md`) asks a different question: given this
trained model, how does retrieval quality change as the *candidate pool*
composition changes — does adding more same-dataset (COCO) candidates hurt
more than adding cross-dataset candidates, because of higher RQ code-prefix
overlap among same-dataset items? See that document and
`/home/tcetoje/.claude/plans/okej-a-ona-je-tingly-mountain.md` for the
detailed experiment plan.
