# GENIUS Phase B paradox — handoff brief for a fresh session

You are a fresh Claude Code session on the ILPS HPC cluster. This file hands you a **specific,
unsolved puzzle** that needs investigation. For full project/pipeline background read
`CLAUDE.md`, `TLDR_new_session.md`, and `GENIUS_reproduction_brief.md` first — this brief assumes
you've absorbed those and goes straight to the open problem.

---

## 0. The puzzle in one sentence

**A GENIUS decoder trained EXACTLY per the paper's recipe (COCO-only, 30 epochs, lr=1e-4,
batch=256, frozen `rq_clip_large.pth`) scores WORSE on COCO retrieval than two earlier models
that were trained the "wrong" way — and nobody yet knows why.**

This is backwards. The "correct" reproduction should have closed the gap to the paper. Instead
it opened a new, bigger one — while its training loss curve looked perfectly healthy. The bug
(if it is one) is invisible in training metrics and only manifests at eval/generalization time.

---

## 1. The numbers (T→I = text→image, COCO test set, task-specific local pool ~123K)

| Model | R@1 (no-rerank) | R@5 | R@10 | R@1 (rerank/GENIUS_R) | R@5 | R@10 |
|---|---|---|---|---|---|---|
| **Phase B: COCO-only 30-epoch (epoch_25), our "correct" repro** | **5.76%** | 18.49% | 26.14% | **11.47%** | 22.67% | 28.48% |
| Pretrained `GENIUS_t5small.pth` (HF, general 16-dataset model) | 14.2% | 25.9% | 31.2% | — | — | — |
| Locally trained 200-epoch COCO+FashionIQ union (epoch_195, overfit) | 11.6% | 23.0% | 28.8% | — | — | — |
| **Paper Table 3 target** | **40.1%** | 66.2% | 75.8% | 46.1% | 74.0% | 82.7% |

I→T (image→text): our new model gets no-rerank R@1=4.33/R@5=15.18/R@10=21.92, rerank
R@1=8.87/R@5=18.25/R@10=23.26 — paper R@5 = 83.2 (no rerank) / 91.1 (rerank).

Ranking from best to worst at T→I R@1: **paper (40.1) > pretrained-general (14.2) >
overfit-200ep (11.6) > our-"correct"-30ep (5.76)**. The supposedly-best-matched reproduction is
dead last.

---

## 2. How we got here — Phase A → Phase B timeline

### Phase A (completed 2026-06-05): why does the pretrained checkpoint underperform?
Diagnosed thoroughly (see `project_stage2_status.md` memory for full write-up):
- No pipeline bugs: RQ checkpoint MD5-verified consistent between training & eval, candidate
  encoding correctly uses CLIP-SF (not vanilla CLIP), Trie works (100% queries have ≥1 valid
  beam), recall computed correctly against qrels, re-ranking enabled.
- One harmless code typo found (`trie_type: trie_cpp` vs `'triecpp'` string match — falls back
  to slower Python trie, functionally fine).
- **Root cause concluded:** `GENIUS_t5small.pth` from HuggingFace is a **global** model trained
  on all 16 M-BEIR datasets/union pool — Table 3's 40.1 is from a **COCO-specific** model. Also,
  our own local 200-epoch retrain (on a COCO+FashionIQ union, 93%/7% mix) overfit badly: loss
  bottomed at ~1.90 by epoch ~50 then plateaued for 150 more epochs, batch R@1 stuck ~5%, and
  modality-code accuracy degraded to near-random (49.5% vs pretrained's 67.2% for code-0/image).
  **Conclusion: training duration + dataset mix mismatch (200 epochs / union vs paper's 30
  epochs / COCO-only) explained the gap. Fix = retrain COCO-only for 30 epochs.**

### What the supervisor (Yubao Tang, GENIUS's original author — "mentorica") advised
Quoted from `GENIUS_reproduction_brief.md` §5 (Phase B instructions she gave directly):
> "Skip Stage 1 RQ training; reuse the provided `rq_clip_large.pth`; just train the GR decoder."
> She confirmed this setting worked [on her end].
> Also: download the candidate pool from `TIGER-Lab/M-BEIR` `cand_pool/` (largest file ~2GB,
> batch-download is fine). Per-dataset training (COCO only), NOT full 16-dataset union training —
> this is the correct reproducibility comparison point for Table 3.

So Phase B = do **exactly** what she said: frozen official RQ + CLIP-SF, train GR decoder
per-dataset, COCO-only, matching the paper's hyperparameters (30 epochs, lr 1e-4, batch 256,
prefix length 30, μ~Beta(2,2) query augmentation, cosine LR schedule).

### Phase B execution (2026-06-08)
1. Filtered a clean COCO-only training query file: `query/union_train/mbeir_coco_only_train.jsonl`
   (213,287 queries, `qid` prefix `9:`).
2. New training config `inbatch_coco_only.yaml`: 30 epochs, lr=1e-4, batch=256, frozen
   `quantizer_path: checkpoint/rq_clip_large.pth`, `pretrained_name: clip_sf_large.pth`.
3. **Reused existing Stage-0 embeddings** rather than re-extracting — rationale logged in memory:
   "`id_to_index` dict in the `.pt` files maps hashed_qid→row; COCO-only JSONL only references
   `9:*` qids; FashionIQ rows just sit unused." ⚠️ **This assumption is a prime suspect — see §3.**
4. Slurm job 327739 (attempt 1) died at epoch 10/30 from a **node-level OOM-killer** on the
   overloaded `ilps-cn120` (cluster-load issue, not our bug — loss was healthily falling 25→4.7).
   Fixed by `--exclude=ilps-cn120` + `num_workers: 8→4`.
5. Slurm job **328613** (attempt 2) ran on idle `ilps-cn117`, **COMPLETED in 21 minutes**
   (30 epochs × ~35s/epoch + ~4 min setup — verified the throughput math checks out exactly,
   and matches the failed attempt's per-iteration speed, so cn120's failure really was OOM not
   slowness). Loss curve: monotonic 12.10 → 4.12, `Level123_acc` 0.00 → 0.146 still climbing —
   **textbook-healthy convergence, no overfitting signature** (contrast with the 200-epoch run
   that bottomed at loss 1.90 and stagnated). Checkpoints saved: epoch_{0,5,10,15,20,25}.
6. Eval job 328625 (attempt 1) crashed in 26s on a config typo (`model.name: T5GenerativeRetrieval`
   vs the code's expected `T5GenerativeRetriever` — fixed, one-word edit in
   `config_eval_coco_only.yaml`).
7. Eval job **328627** (attempt 2) completed cleanly in 46 min, no errors — produced the
   numbers in §1 above.

---

## 3. Why this is suspicious — leading hypotheses to investigate

The training loss/accuracy curves looked perfect. The eval pipeline ran clean (no crashes, no
NaNs, Trie loaded, beam search converged). Yet the retrieval numbers are the worst of all four
models compared. Something is silently wrong. Ranked by suspicion:

### Hypothesis A (top suspect): train-time vs eval-time candidate ID mismatch
Step 3 above reused the **union COCO+FashionIQ** pool/embedding files for training:
- `train_cand_pool_path: cand_pool/global/mbeir_coco_fashioniq_train_cand_pool.jsonl`
- `pool_path: extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt`

...while eval generates candidate IDs **fresh** from the COCO-only **test** pool/embeddings:
- `cand_pool_dir_name: cand_pool/local` → `mbeir_mscoco_task0_test_cand_pool.jsonl`
- `extracted_dir: extracted_embed/CLIP_SF/cand` → `cand_pool_mscoco_task0_test_IT_dict.pt`

In principle RQ encodes each document independently of what else is in the pool, so the same
COCO image should get the same code regardless of which pool file it's looked up from — **but
this assumption was never directly verified for this specific setup**. Phase A verified
RQ/CLIP-SF consistency for the *old* 200-epoch/pretrained setup, not for whether train-time
target-ID generation and eval-time candidate-ID generation produce **identical codes for the
same underlying document** here.
**How to test:** pick ~20 COCO doc IDs, regenerate their RQ codes via both paths (the training
data loader's pre-computed targets vs. the eval pipeline's `gen_code` step), and diff them byte
for byte. If they differ even in the low-order codes, that's the smoking gun — the model would
be trained against one ID for a concept and the Trie at eval time expects a different one for
the same concept.

### Hypothesis B: checkpoint selection
We evaluated `epoch_25` (last saved, due to `eval_freq: 5`), not `epoch_29` (true final epoch
of the cosine schedule, never checkpointed). The LR is already ~0 by epoch 25 so this is
unlikely to cause a 7× gap, but it's cheap to verify — could resume/save epoch 29 and re-eval.

### Hypothesis C: query augmentation / prefix construction differs from the paper subtly
Re-check that `inbatch_coco_only.yaml` actually wires up Beta(2,2) query↔target interpolation,
30 prefix embeddings, and the embed_projector exactly as the paper describes — a silent
misconfiguration here wouldn't show up as a loss explosion (cross-entropy would still decrease)
but could prevent the model from learning a *generalizable* query→ID mapping (it could still
fit the augmented training distribution while learning a degenerate mapping for raw test
queries). Compare side-by-side against how the *pretrained* checkpoint's training config looked
(if recoverable) or against the paper's description in `GENIUS_reproduction_brief.md` §3.

### Hypothesis D: is "healthy-looking" loss actually high enough to indicate undertraining?
Final loss ≈ 4.12 (perplexity ≈ e^4.12 ≈ 61.6 over a 4096-vocab codebook) vs the 200-epoch
run's loss ≈ 1.90 (perplexity ≈ 6.7) at the point it started overfitting. It's possible 30
epochs on 213K COCO queries is simply **not enough raw exposure** for T5-small to learn
precise ID generation — i.e., the paper's "30 epochs" might pair with a different
effective-batch-size / GPU count / data definition than ours, making our "30 epochs" much less
total gradient signal than theirs. Cross-check: how many *optimizer steps* did the paper's run
likely take vs ours (213,287 / 256 ≈ 833 steps/epoch if single-GPU; we did ≈208 steps/epoch ×
4 GPUs = same total samples/epoch, but DDP gradient averaging ≠ single-GPU — could matter for a
small-LR/short-schedule regime). Also worth asking Yubao directly: *what loss value did her run
converge to, and on how many GPUs?*

### Hypothesis E: something about the COCO-only query filter
`mbeir_coco_only_train.jsonl` was derived by filtering the union file on `qid` prefix `9:`.
Verify this filter captured the *correct and complete* COCO training query set (right modality
mix of text/image queries, right count vs official M-BEIR COCO train split, no leakage from
FashionIQ, no truncation) — a subtly wrong filter could train on a skewed or incomplete query
distribution while still producing a smooth loss curve.

---

## 4. Concrete next steps (suggested order)

1. **Test Hypothesis A first** — it's the most structural and the cheapest definitive test
   (just regenerate codes for a handful of docs both ways and diff). If confirmed, the fix is
   to either re-extract train-time pool embeddings from the COCO-only pool, or confirm/patch
   the `id_to_index` mapping so train and eval reference the *same* embedding rows.
2. If A is clean, do the **quick qualitative probe** from the original Phase A checklist
   (`GENIUS_reproduction_brief.md` step 6): print generated ID vs gold ID for ~10 COCO test
   queries from the new model, check whether the modality code (first token) is even right,
   and whether generated prefixes are valid Trie paths at all. This will immediately tell you
   whether the model is "close but imprecise" (suggests undertraining/Hypothesis D) or
   "fundamentally generating the wrong ID space" (suggests Hypothesis A/C).
3. Compare `inbatch_coco_only.yaml` line-by-line against the paper recipe description and
   (if you can find it) any config used for the original `GENIUS_t5small.pth` training run.
4. If still stuck, draft a focused question for Yubao: *"We trained COCO-only/30-epoch exactly
   per your instructions and get R@1=5.8 (11.5 w/ rerank) vs your reported 40.1 — loss curve
   looks healthy (12.1→4.1). Did your run converge to a similar loss? Anything about
   augmentation/embeddings/batch construction we might be missing?"*

---

## 5. Key files, jobs, logs, checkpoints

| What | Where |
|---|---|
| Phase B training config | `src/models/generative_retriever/configs_scripts/large/train/inbatch/inbatch_coco_only.yaml` |
| Phase B eval config | `src/models/generative_retriever/configs_scripts/large/eval/inbatch/config_eval_coco_only.yaml` |
| Training Slurm script | `scripts/slurm_train_stage2_coco_only.sh` |
| Eval Slurm script | `scripts/slurm_eval_coco_only.sh` |
| Training log (success, job 328613) | `~/logs/genius_stage2_coco_328613.log` |
| Eval log (success, job 328627) | `~/logs/genius_eval_coco_only_328627.log` |
| New checkpoints | `checkpoint/GENIUS_t5small/Large/Instruct/CocoOnly/genius_t5small_epoch_{0,5,10,15,20,25}.pth` |
| Results TSV | `retrieval_results/GENIUS_t5small/Large/Instruct/CocoOnly/final_tsv/eval_results_06-08-15.tsv` |
| COCO-only train query file | `query/union_train/mbeir_coco_only_train.jsonl` (213,287 queries) |
| Train cand pool (union, ⚠️ see Hyp. A) | `cand_pool/global/mbeir_coco_fashioniq_train_cand_pool.jsonl` |
| Train pool embeddings (union, ⚠️ see Hyp. A) | `extracted_embed/CLIP_SF/train/pool_SFpretrained_IT_dict.pt` |
| Eval cand pool (COCO test-only) | `cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl` |
| Eval pool embeddings (COCO test-only) | `extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_test_IT_dict.pt` |
| Memory with full diagnostic history | `project_stage2_status.md` (auto-memory; ask to recall it) |

---

## 6. Guardrails (same as always)
Shared cluster — confirm with the user before launching new Slurm training jobs (eval/diagnostic
runs are fine to launch freely). Never `--partition=gpu` for CPU-only work. Don't delete data or
modify shared files (`/fnwi_fs/ivi/irlab/datasets/`). `--exclude=ilps-cn120` if memory-heavy
(it's prone to oversubscription by other users' jobs).
