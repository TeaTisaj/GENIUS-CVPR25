# GENIUS reproduction — task brief

You are a fresh Claude Code session running on the ILPS HPC cluster over SSH. You have no
prior context. This file is the complete handoff. Read it fully before touching anything.

---

## 0. What this project is

We are **reproducing the GENIUS paper** (Kim et al., "GENIUS: A Generative Framework for
Universal Multimodal Search", CVPR 2025, arXiv:2503.19868) on a **subset** of its datasets.
We do NOT need to reproduce all datasets/tasks/metrics.

Target scope (in order):
1. **COCO** — standard multimodal retrieval sanity check (text→image and image→text).
2. **FashionIQ or Fashion200K** — one fine-grained / high-similarity dataset (main extension study).
3. (optional, if time) CIRR or NIGHTS.

Use the paper's own metrics: Recall@5 for COCO, Recall@10 for the Fashion datasets.

### How GENIUS works (1-paragraph mental model)
GENIUS is a **generative retrieval** model. Instead of embedding + nearest-neighbour search,
it generates a discrete **target ID** (a short sequence of codebook indices) directly from the
query. Pipeline has 3 training stages plus inference:
- **Stage 0 — Encoder:** frozen UniIR **CLIP-SF** (CLIP ViT-L/14, 768-dim) encodes image/text/instruction.
- **Stage 1 — RQ quantizer:** a fusion module + **residual quantizer** turns a fused embedding into
  a modality-decoupled semantic ID. First code = modality (0=image, 1=text, 2=image-text pair);
  later codes = coarse→fine semantics.
- **Stage 2 — Decoder (GR):** a **T5-small** decoder is trained to generate the target ID from the
  query embedding (fed in as prefix embeddings via cross-attention). Trained with query augmentation.
- **Inference:** Trie-constrained beam search over valid candidate IDs, then optional
  **embedding-based re-ranking** of the beam candidates (this re-ranked variant is called GENIUS_R).

**Critical dependency:** the decoder is trained to emit IDs produced by one specific RQ + one
specific encoder. At eval time the **candidate pool IDs and the Trie must be generated with the
SAME RQ checkpoint and the SAME CLIP-SF encoder**. Any mismatch here destroys recall. Remember this.

---

## 1. Environment & paths (cluster)

- Cluster: ILPS HPC, **Slurm**, **conda** (Miniconda in home), PyTorch + CUDA 12.4, DDP multi-GPU (2× A6000 available).
- Repo: `/home/tcetoje/GENIUS-CVPR25/`
- Personal storage: `/fnwi_fs/ivi/irlab/personal/tcetoje/`
- Shared datasets: `/fnwi_fs/ivi/irlab/datasets/`
- Logs: `/home/tcetoje/logs/`
- Checkpoints we have / expect: `GENIUS_t5small.pth` (pretrained decoder), `rq_clip_large.pth`
  (pretrained RQ quantizer, "large" = ViT-L/14 = 768-dim), and UniIR CLIP-SF encoder weights.
- COCO 2014 in M-BEIR format is already preprocessed (123,287 images).
- M-BEIR data + candidate pools live on HuggingFace: `TIGER-Lab/M-BEIR` (the `cand_pool/` dir).

**Guardrails:** This is shared infrastructure. Do NOT submit long training Slurm jobs without
checking with the user first and checking resource usage. Diagnostic/eval runs (Phase A) are fine.
Do not delete data or modify shared files. Linux/cluster commands are fine here.

---

## 2. The problem we are debugging

The user ran the **pretrained `GENIUS_t5small.pth`** on the COCO test split and got:

- text→image: **R@1 = 0.142**, R@5 = 0.259, R@10 = 0.312
- image→text: **R@1 = 0.127**, R@5 = 0.237, R@10 = 0.287

These looked catastrophic vs the paper, but part of that was a **wrong comparison**, and part is
likely a **real pipeline bug**. Both are explained below.

### 2a. The comparison was wrong (metric + pool mismatch)
The user compared her R@1 against the paper's "0.551 / 0.827". Those paper numbers are
**R@5 on the global 5.6M pool (Table 2)** — i.e. a different k AND a different pool. Not comparable.

The **correct reference for COCO-only train/eval is Table 3** (text→image, trained & evaluated on
COCO alone):

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| GENIUS (T→I) | **40.1** | 66.2 | 75.8 |
| GENIUS_R (T→I, re-ranked) | 46.1 | 74.0 | 82.7 |

So the matched R@1 target is ~40, and we got 14. **Even after fixing the comparison, the gap is too
large to be normal.** A smaller (COCO-only ~123K) pool should give HIGHER recall than the 5.6M pool,
not lower — so something in the pipeline is wrong.

### 2b. The likely real bug (investigate this first)
Most probable cause: the **candidate IDs / Trie were not generated with the same RQ checkpoint and/or
the same CLIP-SF encoder that the pretrained decoder expects.** Concretely, suspects are:
- candidate embeddings computed with **vanilla CLIP instead of UniIR CLIP-SF**;
- candidate IDs quantized with a **different / re-initialised RQ** rather than the exact `rq_clip_large.pth`;
- a **Trie built from a mismatched candidate set**, or first-code (modality) handling wrong;
- eval **run without re-ranking** while comparing to a re-ranked number (smaller effect, but report both).

---

## 3. Ground-truth paper recipe (so you don't need the PDF)

**Encoders:** UniIR CLIP-SF = CLIP **ViT-L/14**, 768-dim. Frozen after Stage 0. (Use UniIR's released
weights directly; do not retrain.)

**RQ quantizer (Stage 1):** codebook **4096 × 9 levels**; first codebook fixed size **3** (modality).
k-means init on first batch. AdamW, lr **1e-4**, **20 epochs**, batch **256**. Codebook updated via EMA.
Loss = `L_cl + 100·L_rq + 100·L_mse`, temperature τ = **0.01**.

**Decoder / GR (Stage 2):** **T5-small**, hidden d' = **512**, random init. Query embedding → MLP →
**30 prefix embeddings** via cross-attention. Query augmentation: interpolate query↔target,
`z' = μ·z_q + (1-μ)·z_c`, μ ~ **Beta(2,2)**. AdamW, lr **1e-4**, **30 epochs**, cosine schedule,
batch **256**, cross-entropy over the ID sequence.

**Inference:** Trie-constrained beam search, beam size **50** (ablations use 30). Optional embedding
re-ranking of beam candidates = GENIUS_R. Faiss only for the embedding baselines.

**Metrics:** R@5 for COCO; **R@10 for Fashion200K and FashionIQ**.

### Other paper targets we'll use
- COCO image→text (Table 1, task-specific pool): R@5 = 83.2 (GENIUS) / 91.1 (GENIUS_R). Global pool
  (Table 2): 82.7 / 90.6.
- Fashion200K (Table 1, R@10): qt→ci 13.7/16.2 ; qi→ct 12.8/16.3.
- FashionIQ — composed (qi,qt)→ci (Table 1/2, R@10): 13.1/19.2.

---

## 4. PHASE A — diagnose the pretrained-checkpoint eval (do this first)

Goal: explain the 0.142 and get COCO T→I close to **R@1 ≈ 40 / R@5 ≈ 66** (GENIUS) without retraining.

1. **Map the repo.** Explore `/home/tcetoje/GENIUS-CVPR25/`: find the eval entrypoint, config files,
   the code that (a) loads checkpoints, (b) encodes the candidate pool, (c) quantizes candidates into
   IDs, (d) builds the Trie, (e) runs beam search, (f) does re-ranking, (g) computes Recall. Summarise
   what each does and which files/paths/configs they use. **Do not assume file names — read the code.**

2. **Confirm checkpoints & dims.** Locate `GENIUS_t5small.pth`, `rq_clip_large.pth`, and the CLIP-SF
   encoder weights. Verify the encoder/RQ are **768-dim ViT-L/14**, RQ codebook is **4096×9 with first
   level size 3**. Print shapes to confirm.

3. **Verify the candidate-encoding path = CLIP-SF.** Trace how candidate embeddings are produced.
   Confirm they come from **UniIR CLIP-SF**, not plain CLIP. This is the #1 suspect.

4. **Verify RQ consistency.** Confirm the RQ used to quantize the candidate pool is **exactly the
   `rq_clip_large.pth` that the decoder was trained against** — not a freshly initialised or different RQ.

5. **Check the Trie / pool.** Confirm the Trie is built from the **COCO-only candidate pool** and that
   the first (modality) code is handled correctly for the task (target modality = image ⇒ first code 0
   for T→I; = text ⇒ first code 1 for I→T).

6. **Quick probe.** For ~10 queries, print the **generated ID** vs the **gold target ID** and whether
   the generated prefix is even valid in the Trie. If generated IDs never match valid prefixes or the
   modality code is wrong, that pinpoints the mismatch.

7. **Check the metric code + re-ranking flag.** Confirm R@1/5/10 are computed over the ranked beam
   list. Run eval **both with and without re-ranking** and report GENIUS and GENIUS_R separately.

8. **Re-run COCO eval** and compare to Table 3 (T→I: R@1 40.1 / R@5 66.2 ; re-ranked 46.1 / 74.0) and
   to the image→text targets in §3. Report the gap and your root-cause conclusion.

Deliver: a short root-cause write-up + corrected numbers. Don't start training until Phase A is understood.

---

## 5. PHASE B — clean COCO reproduction (Yubao's path)

The original author (Yubao Tang) advised: **skip Stage 1 RQ training; reuse the provided
`rq_clip_large.pth`; just train the GR decoder.** She confirmed this setting worked. Also: download the
candidate pool from `TIGER-Lab/M-BEIR` `cand_pool/` (largest file ~2GB; batch-download is fine).

Steps:
1. UniIR **CLIP-SF** weights (frozen) + **`rq_clip_large.pth`** (frozen RQ).
2. Download the **COCO candidate pool** from M-BEIR `cand_pool/` on HuggingFace.
3. Generate candidate IDs: CLIP-SF encode → fusion → RQ → discrete IDs → build Trie.
4. Train the **GR decoder** (T5-small) on COCO query→target-ID pairs with query augmentation
   (30 epochs, lr 1e-4, batch 256, prefix length 30, μ~Beta(2,2)). **Confirm with the user before
   launching the Slurm training job.**
5. Eval: Trie beam search (beam 50) + re-ranking. Targets: GENIUS R@1≈40 / R@5≈66 ; GENIUS_R ≈46/74.

Per-dataset training (COCO only), NOT full 16-dataset M-BEIR union training. Cheaper and is the correct
reproducibility comparison point for Table 3 / Table 1 task-specific pools.

---

## 6. PHASE C — Fashion (one fine-grained dataset)

Same pipeline, same **`rq_clip_large.pth`**, train the GR decoder per-dataset. Metric **R@10**.
- **Fashion200K** (qt→ci, qi→ct) — simpler cross-modal; good first choice. Targets R@10 in §3.
- **FashionIQ** — composed (qi,qt)→ci; better fit for the high-similarity extension study.
- Download the relevant `cand_pool/` + query files from M-BEIR.

---

## 7. First action for you

Start with **Phase A, steps 1–2**: explore `/home/tcetoje/GENIUS-CVPR25/`, map the eval pipeline, and
locate + inspect the three checkpoints. Report back what you find before changing any code or running eval.
