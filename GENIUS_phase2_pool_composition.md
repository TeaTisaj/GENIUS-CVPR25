# GENIUS Phase 2 — Candidate-Pool Composition Study (mentor directive, 2026-06-17)

**Status: NOT STARTED.** This is the next phase, to begin only after Phase 1
(reproduction pipeline — see `GENIUS_reproduction_brief.md`, `genius-pipeline.md`,
and the "Project Plan" section in `CLAUDE.md`) is finalized and documented.

Do not mix work for this phase with the Phase 1 training/eval jobs currently
running (full-union Stage 2 model, job 329846, evaluated by jobs 329954/329955).
Phase 2 *consumes* the finished Phase 1 model as input — it does not retrain it.

This document supersedes the vaguer "Extension 1 / Extension 2" sketch under
**Step 3: Core Extensions** in `CLAUDE.md` with the mentor's concrete instructions.

---

## Motivation (mentor's framing, verbatim)

> GENIUS is designed for universal retrieval over a large and heterogeneous candidate
> pool. However, our preliminary results suggest that its performance may depend
> strongly on candidate-pool composition. In particular, candidates from the same
> dataset may share more similar RQ prefixes and therefore be more difficult to
> distinguish than candidates from different datasets. We would therefore like to
> study whether GENIUS remains effective for fine-grained within-dataset retrieval
> as the candidate pool changes.

## Deliverable 0 — Document Phase 1 (prerequisite, do this first)

Briefly document:
- The completed reproduction pipeline (Stage 0 → 1 → 2, per-dataset vs full-union training)
- Main results table (COCO T→I/I→T, FashionIQ, with/without rerank, vs paper numbers)
- Possible reasons for the remaining gap from the paper (training steps/epochs,
  effective batch size 256 vs paper's likely 1024 with 4 GPUs, LR schedule length, etc.)

This writeup is a prerequisite deliverable for the mentor, separate from the
Phase 2 experiments below.

## Deliverable 1 — COCO T→I retrieval under different candidate-pool settings

Uses the **trained full-union Stage 2 model** (checkpoint:
`checkpoint/GENIUS_t5small/Large/Instruct/InBatch/genius_t5small_epoch_95.pth`,
pending final eval as of this writing) as a frozen retriever — no retraining in this phase.

### 1a. COCO-local pool
Evaluate COCO T→I using only the complete COCO candidate pool (current eval setup,
`mscoco_task0_test`).

### 1b. Progressively expanded pools
Start from COCO-local and gradually add image candidates from other M-BEIR datasets:
1. COCO only
2. COCO + FashionIQ
3. COCO + several additional image datasets (pick a handful, e.g. Fashion200K, CIRR, NIGHTS)
4. Full M-BEIR union pool

Goal: see how retrieval performance changes as the pool grows larger and more diverse.

### 1c. Controlled candidate-composition experiment
- Fix a subset of COCO test queries.
- Build candidate pools of **equal size** that always contain all relevant COCO targets,
  varying only the distractors:
  - **within-dataset distractors**: sampled only from COCO
  - **cross-dataset distractors**: sampled from non-COCO image datasets
  - **mixed distractors**: 50% COCO / 50% other datasets
- Test at multiple fixed pool sizes: **10K, 50K, 100K** candidates.
- Same queries and same candidate count across all three distractor settings.
- Repeat with **3 random seeds** per setting.
- Compare: do within-dataset distractors cause a larger recall drop than cross-dataset distractors?

## Deliverable 2 — RQ identifier prefix-overlap analysis

Measure prefix overlap among COCO candidates at different RQ depths (levels 1–8,
see `codebook_config.codebook_level: 8` / `codebook_vocab: 4096` in the train/eval YAMLs).
Determine whether stronger within-dataset prefix overlap correlates with worse
retrieval performance — i.e. does this explain the Deliverable 1c result?

---

## Inputs available from Phase 1 (no extra work needed)

- Frozen RQ quantizer: `checkpoint/rq_clip_large.pth` — used to compute RQ codes per
  candidate for the prefix-overlap analysis.
- Stage 0 candidate embeddings for COCO and other M-BEIR datasets:
  `extracted_embed/CLIP_SF/cand/` (per-dataset, local pools already extracted for eval).
- Full M-BEIR union candidate pool JSONL: `cand_pool/global/mbeir_union_train_cand_pool.jsonl`
  (train-side; for eval-side per-dataset pools see `cand_pool/local/`).
- Full-union Stage 2 checkpoint (Phase 1 deliverable, see above).

## What's still missing / to build in this phase

- Scripts to assemble mixed candidate pools (COCO + N other datasets) at controlled sizes.
- Sampling harness for the 10K/50K/100K × {within, cross, mixed} × 3-seeds grid (27 runs total).
- RQ prefix-overlap measurement script (group candidates by dataset, compare code prefixes
  at each of the 8 levels).

None of this exists yet — Phase 2 has not been started.
