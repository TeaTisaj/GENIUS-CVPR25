# RQ1–RQ3 Balance Regularization Study — Final Tables

*All numbers below are the corrected, final versions (post pool-bug fix 2026-07-01, post trie-cache fix 2026-07-02). Anything superseded by these fixes is NOT included — this file only has current, trustworthy numbers.*

---

## Table 1 — ID Structure

### MSCOCO (N=713,679 training candidates)

| Method | λ | Utilization % | Entropy (bits) | Collision | Recon MSE | Cosine Sim | Neighbor pres.@10 |
|---|---|---|---|---|---|---|---|
| vanilla | 0.0 | 57.69 | 9.4453 | 0.87% | 0.000124 | 95.22% | 0.373 |
| weak | 0.3 | 55.91 | 7.3242 | 6.38% | 0.000647 | 75.14% | 0.117 |
| medium | 1.0 | 63.39 | 5.8957 | 22.66% | 0.000974 | 62.59% | 0.146 |
| strong | 3.0 | 68.21 | 4.3202 | 69.65% | 0.000954 | 63.37% | 0.167 |

### FashionIQ (N=74,380 candidates)

| Method | λ | Utilization % | Entropy (bits) | Collision | Recon MSE | Cosine Sim | Neighbor pres.@10 |
|---|---|---|---|---|---|---|---|
| vanilla | 0.0 | 18.40 | 8.7611 | 4.18% | 0.000037 | 98.58% | 0.668 |
| weak | 0.3 | 10.65 | 7.5315 | 4.37% | 0.000255 | 90.23% | 0.226 |
| medium | 1.0 | 11.55 | 7.7186 | 4.45% | 0.000310 | 88.10% | 0.259 |
| strong | 3.0 | 11.58 | 7.7531 | 4.45% | 0.000386 | 85.19% | 0.252 |

---

## Table 2 — Retrieval Performance (corrected, 5K/24.8K COCO test splits)

### MSCOCO (T→I / I→T, best epoch = 25 for all variants)

| Method | λ | Best ep | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 |
|---|---|---|---|---|---|---|
| vanilla | 0.0 | 25 | 14.08% | 18.08% | 18.11% | **9.44%** |
| weak | 0.3 | 25 | 14.74% | 22.05% | 22.11% | 0.04% |
| medium | 1.0 | 25 | 11.15% | 18.81% | 18.92% | 0.02% |
| strong | 3.0 | 25 | **21.24%** | **29.70%** | **29.78%** | 0.04% |

**Key finding**: strong (+50.9% over vanilla, single seed) wins T→I, but vanilla is the ONLY variant that works at all for I→T. Balance regularization destroys I→T retrieval. **See the "COCO Seed-Variance Check" section below — the +50.9% figure is a single-seed artifact; the seed-supported effect size is ~+22%, not +50.9%.**

### FashionIQ (IT→I, best epoch)

| Method | λ | Best ep | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| vanilla | 0.0 | 15 | **0.55%** | 2.92% | 4.82% |
| weak | 0.3 | 10 | 0.27% | 1.47% | 2.13% |
| medium | 1.0 | 10 | 0.42% | 1.33% | 2.07% |
| strong | 3.0 | 10 | 0.33% | 0.97% | 1.57% |

---

## Table 3 — Balance Trade-off (ID metrics + corrected Recall)

### MSCOCO

| Method | λ | Utilization % | Collision | Recon MSE | Cosine Sim | Neighbor pres.@10 | T→I R@1 |
|---|---|---|---|---|---|---|---|
| vanilla | 0.0 | 57.69 | 0.87% | 0.000124 | 95.22% | 0.373 | 14.08% |
| weak | 0.3 | 55.91 | 6.38% | 0.000647 | 75.14% | 0.117 | 14.74% |
| medium | 1.0 | 63.39 | 22.66% | 0.000974 | 62.59% | 0.146 | 11.15% |
| strong | 3.0 | 68.21 | 69.65% | 0.000954 | 63.37% | 0.167 | **21.24%** |

Balance reg for COCO **increases** utilization (58%→68%) and **degrades** reconstruction (cosine 95%→63%), yet strong still wins on Recall. The naive "better reconstruction → better retrieval" story does NOT hold for COCO T→I.

### FashionIQ

| Method | λ | Utilization % | Collision | Recon MSE | Cosine Sim | Neighbor pres.@10 | R@1 |
|---|---|---|---|---|---|---|---|
| vanilla | 0.0 | **18.40** | 4.18% | 0.000037 | **98.58%** | **0.668** | **0.55%** |
| weak | 0.3 | 10.65 | 4.37% | 0.000255 | 90.23% | 0.226 | 0.27% |
| medium | 1.0 | 11.55 | 4.45% | 0.000310 | 88.10% | 0.259 | 0.42% |
| strong | 3.0 | 11.58 | 4.45% | 0.000386 | 85.19% | 0.252 | 0.33% |

Opposite pattern from COCO: balance reg **decreases** utilization for FashionIQ (18%→11%). Vanilla wins on every column.

---

## Table 4 — Cross-Dataset Comparison (RQ3 headline)

| Property | MSCOCO | FashionIQ |
|---|---|---|
| Mean utilization (avg 4 variants) | 61.3% | 12.9% |
| Mean entropy (bits) | 6.75 | 7.90 |
| Mean prefix-bucket size (depth 8) | 1.478 | 1.029 |
| Mean full-ID collision rate | 24.9% | 4.4% |
| Mean reconstruction MSE | 0.000675 | 0.000247 |
| **Optimal balance strength** | **strong (λ=3.0)** | **vanilla (λ=0.0)** |
| **Relative Recall gain over vanilla** | **+50.9%** (single seed; ~+22% by 3-seed mean, see COCO Seed-Variance Check) | **+0.0%** |

**The two datasets require opposite strategies.** COCO benefits from strong balance regularization; FashionIQ is hurt by any regularization at all.

---

## Correlation Tables (n=4, descriptive — no p-values)

### MSCOCO (T→I Recall@1, corrected)

| ID metric | Pearson r | Spearman ρ |
|---|---|---|
| Utilization | +0.557 | +0.20 |
| Collision rate | +0.789 | +0.40 |
| Reconstruction MSE | +0.218 | −0.20 |

*Note: these are the corrected values. Pre-pool-bug-fix correlations (utilization −0.44/−0.60, collision −0.49/−0.80, recon MSE −0.94/−1.00) are invalid and superseded — the old "reconstruction fidelity is the strongest predictor" story no longer holds once decoded against the correct 5K test pool. Strong now has both the highest MSE and the highest Recall, so no simple ID-metric predicts COCO Recall cleanly.*

### FashionIQ (R@1)

| ID metric | Pearson r | Spearman ρ |
|---|---|---|
| **Utilization** | **+0.904** | **+0.80** |
| Collision rate | −0.699 | −0.40 |
| Reconstruction MSE | −0.749 | −0.40 |

For FashionIQ, utilization is the strongest (positive) predictor — opposite sign from COCO.

---

## Pool-Size Confound Check (COCO subsampled to 74K, matching FashionIQ)

| Source | Variant | N | Utilization % | Entropy (bits) | Collision | Recon Cos Sim |
|---|---|---|---|---|---|---|
| COCO-full | vanilla | 713,679 | 57.69 | 9.45 | 0.87% | 0.952 |
| COCO-full | weak | 713,679 | 55.91 | 7.32 | 6.38% | 0.751 |
| COCO-full | medium | 713,679 | 63.39 | 5.90 | 22.66% | 0.626 |
| COCO-full | strong | 713,679 | 68.21 | 4.32 | 69.65% | 0.634 |
| COCO-subsampled-74K | vanilla | 74,380 | 38.82 | 9.43 | 0.13% | 0.952 |
| COCO-subsampled-74K | weak | 74,380 | 37.87 | 7.33 | 1.56% | 0.752 |
| COCO-subsampled-74K | medium | 74,380 | 45.33 | 5.90 | 8.59% | 0.626 |
| COCO-subsampled-74K | strong | 74,380 | 46.94 | 4.33 | 52.33% | 0.635 |
| FashionIQ-real | vanilla | 74,380 | 18.40 | 8.76 | 4.18% | 0.986 |
| FashionIQ-real | weak | 74,380 | 10.65 | 7.53 | 4.37% | 0.902 |
| FashionIQ-real | medium | 74,380 | 11.55 | 7.72 | 4.45% | 0.881 |
| FashionIQ-real | strong | 74,380 | 11.58 | 7.75 | 4.45% | 0.852 |

**Verdict**: absolute numbers shrink when COCO is downsampled to FashionIQ's size, so pool size IS a real confound. But even at matched pool size, COCO still runs ~2–4× FashionIQ's utilization and much higher collision growth vanilla→strong — the RQ3 domain-dependence finding survives the control.

---

## FashionIQ Seed-Variance Check (3 seeds: 2023 / 7 / 13, Recall@1)

| Variant | seed 2023 | seed 7 | seed 13 | mean | std |
|---|---|---|---|---|---|
| vanilla | 0.55 | 0.73 | 0.55 | **0.61** | 0.104 |
| weak | 0.22 | 0.33 | 0.25 | 0.267 | 0.057 |
| medium | 0.30 | 0.13 | 0.20 | 0.210 | 0.085 |
| strong | 0.33 | 0.23 | 0.18 | 0.247 | 0.076 |

**Robust**: vanilla beats all regularized variants across all 3 seeds.
**Not robust**: the fine-grained ranking (medium > strong > weak, from seed=2023 alone) flips to weak > medium > strong by mean — medium's own seed range (0.13–0.42) exceeds the gap between any pair of regularized variants. Don't report a weak/medium/strong ordering as meaningful.
**Caveat**: weak/medium's seed=2023 point used the original Stage-1 quantizer; seed 7/13 used a repaired quantizer (after an accidental checkpoint overwrite, unrelated incident) that shows a ~50% shift in a fine-grained level-1 code-concentration statistic (not in the headline metrics, which matched <1%). So weak/medium's "3-seed spread" isn't perfectly clean seed-only noise — an extra reason to distrust the fine ranking.

## COCO Seed-Variance Check (3 seeds: 2023 / 7 / 13, T→I Recall@1) — found completed 2026-07-10, previously unincorporated

**This materially weakens the Table 2 headline claim ("strong beats vanilla by +50.9%") — read before citing that number anywhere.**

| Variant | seed 2023 | seed 7 | seed 13 | mean | std |
|---|---|---|---|---|---|
| vanilla | 14.08 | 20.78 | 20.70 | **18.52** | 3.85 |
| weak | 14.74 | 13.77 | 13.87 | 14.13 | 0.53 |
| medium | 11.15 | 10.02 | 9.90 | 10.36 | 0.69 |
| strong | 21.24 | 23.19 | 23.37 | **22.60** | 1.18 |

**Not robust**: vanilla's own seed-to-seed spread (14.08–20.78, std 3.85) is enormous — seed 2023 (the one used for every "vanilla" number quoted elsewhere in this document, including Table 2 and the direction-aware-λ follow-up below) happens to be vanilla's *lowest* of the three seeds by a wide margin. Seeds 7/13 land at 20.70–20.78%, nearly matching strong's *lowest* seed (21.24%).

**Corrected effect size**: the "strong wins T→I by +50.9%" headline (Table 2) is a single-seed artifact — it compares strong's seed-2023 (21.24%) against vanilla's seed-2023 (14.08%), which happens to be vanilla's worst run. Using 3-seed means instead: strong 22.60% vs. vanilla 18.52%, a **+22.0% gap** — a real, still-positive effect, but less than half the originally reported magnitude. Comparing vanilla's *best* seed (20.78%) against strong's *worst* seed (21.24%) — the most conservative framing — the gap shrinks to **+2.2%**, i.e. within noise for single-run comparisons.

**Robust**: weak (std 0.53) and medium (std 0.69) are both far more seed-stable than vanilla, and medium remains clearly the worst variant across all seeds. Strong is also comparatively stable (std 1.18) and its mean is above vanilla's mean in all 3 seed pairings, so "strong tends to beat vanilla on T→I" survives the check — just not by anywhere near the originally reported margin, and not certainly in every individual run.

**Practical implication for anything downstream that cites "strong +50.9%" or treats vanilla's 14.08% as a stable baseline** (including the direction-aware-λ follow-up below, whose comparison table uses vanilla's seed-2023 14.08% as the T→I anchor): treat that specific number as the low end of vanilla's real range, not its typical value. This does not change the direction-aware-λ verdict (img3txt0/img3txt0p3 at 7.97%/5.45% are still clearly below vanilla's full seed range of 14.08–20.78%, not just below its seed-2023 point), but it does mean "strong beats vanilla" should be reported as a real, seed-supported ~22% effect, not a dramatic 51% one, in any paper-facing text.

*(This job — `slurm_train_coco_seed_variance.sh`/`slurm_eval_coco_seed_variance.sh`, job chain 333701–333703 — actually completed 2026-07-03 13:20; the "still running" note that stood here for a week was stale, not a live status. Caught during a 2026-07-10 pass specifically checking for finished-but-unincorporated results.)*

---

## I→T Setup Double-Check — Modality Code Diagnostic

`code[0]` (modality indicator) is never directly regularized, yet varies wildly by variant (training pool, N=713,679):

| Variant | Image code[0] | Text code[0] | Pattern | I→T R@1 |
|---|---|---|---|---|
| vanilla | 0: 100% | 0: 100% | fully collapsed, shared constant | **9.44%** (works) |
| weak | 2: 100% | 2: 100% | fully collapsed, shared constant (same shape as vanilla) | 0.04% (fails) |
| medium | 1: 100% | 0: 99.99%, 1: 0.01% | clean separation, inverted vs. documented convention | 0.02% (fails) |
| strong | 0/1/2: 9.8/22.9/67.3% | 0/1/2: 40.9/47.7/11.4% | genuinely mixed, no clean assignment | 0.04% (fails) |

**Key result**: vanilla and weak have the *identical* collapse shape yet opposite I→T outcomes — rules out "modality-code collapse" as a sufficient standalone explanation. Confirmed (job 333652/333654) this holds even after the trie-cache fix, so it's not a decode-space artifact either. Root cause remains an open/partial finding — state as a writeup limitation.

---

## Story for the Paper (both datasets, corrected — use "results suggest" / "initial trend", not causal language)

> Results suggest the effect of balance regularization is strongly dataset-dependent (initial trend across 4 configurations per dataset; n=4, descriptive only). On COCO (broad-domain), strong regularization improves T→I retrieval — by ~51% in a single-seed comparison, or a more conservative ~22% once 3-seed variance is accounted for (see COCO Seed-Variance Check) — suggesting broader code usage helps discriminate diverse candidates — though it destroys I→T performance (~0% for all regularized variants vs. 9.4% vanilla, a finding that IS seed-independent: weak/medium/strong's I→T collapse and vanilla's I→T advantage do not depend on which seed produced the T→I number). On FashionIQ (fine-grained), vanilla achieves the best performance, and any regularization reduces recall. This divergence suggests a single RQ regularization strategy does not transfer across retrieval settings: broad-domain tasks may benefit from better code coverage, while fine-grained tasks require faithful reconstruction of subtle embedding differences. **Use "~22–51% depending on seed" or cite the 3-seed mean (+22%) as the primary number in any paper-facing text — do not quote 51% unqualified.**

---

## Direction-Aware λ Follow-up (2026-07-09) — Testing Yubao's Hypothesis

**Motivation.** Table 2 shows a sharp T→I/I→T tradeoff: strong (uniform λ=3.0) wins T→I (+50.9% single-seed / ~+22% by 3-seed mean, see COCO Seed-Variance Check above) but destroys I→T (~0% vs. vanilla's 9.44%, a seed-independent finding). Yubao's mentor note proposed that this might be an artifact of applying the *same* balance strength to both target modalities, and suggested trying separate λ_img/λ_txt — tuned per direction — as a simpler alternative to the retrieval-aware method (hard negatives + L_ret + gradient-conflict projection) explored earlier this week. Two new Stage-1/Stage-2/eval cells were run through the **real pipeline** (T5 + constrained beam search, not the cheap dense probe used in the earlier retrieval-aware-RQ study) to test this directly:

- **img3txt0**: λ_img=3.0, λ_txt=0.0 (all balance pressure on image codes, none on text — the most literal reading of "T→I and I→T need different treatment")
- **img3txt0p3**: λ_img=3.0, λ_txt=0.3 (asymmetric but nonzero on both)

Implementation: the balance/codebook-diversity loss (previously one scalar `codebook_diversity_loss_weight` applied identically to every RQ level regardless of the row's modality) was split into a per-row-weighted version via a forward hook on each level's `VectorQuantize._codebook`, recomputing the library's own entropy formula with `w = λ_img` for image rows / `λ_txt` for text rows. Verified equivalent to the existing single-λ code (λ_img==λ_txt reproduces the fused-loss path to 1e-6 tolerance on a CPU unit test, and a real 1-GPU pilot run) before any real training was trusted — see `extensions/rq2_balance_study/test_modality_split_balance_equivalence.py`.

### Full epoch curves (T5 training epochs 5/10/15/20/25, real beam-search eval, num_beams=50)

**img3txt0 (λ_img=3.0, λ_txt=0.0):**

| epoch | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 |
|---|---|---|---|---|---|---|
| 5 | 5.57% | 13.52% | 18.08% | 0.10% | 0.44% | 0.70% |
| 10 | 6.78% | 15.86% | 20.78% | 0.16% | 0.46% | 0.84% |
| 15 | 7.56% | 17.07% | 22.06% | 0.16% | 0.48% | 0.88% |
| 20 | 7.94% | 17.84% | 22.91% | 0.12% | 0.44% | 0.80% |
| 25 | 7.97% | 17.95% | 23.07% | 0.12% | 0.52% | 0.84% |

**img3txt0p3 (λ_img=3.0, λ_txt=0.3):**

| epoch | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 |
|---|---|---|---|---|---|---|
| 5 | 3.91% | 9.87% | 13.95% | 0.20% | 0.86% | 1.40% |
| 10 | 5.09% | 12.08% | 16.66% | 0.26% | 0.70% | 1.58% |
| 15 | 5.45% | 12.79% | 17.33% | 0.22% | 0.64% | 1.24% |
| 20 | 5.40% | 13.16% | 17.80% | 0.08% | 0.70% | 1.14% |
| 25 | 5.45% | 13.20% | 17.85% | 0.08% | 0.60% | 1.10% |

**Epoch-trend note**: img3txt0's T→I is still rising at epoch 25 (+43% from epoch 5 to 25, no plateau reached — epoch 25 may understate its ceiling if trained further, though this matches the same 30-epoch/save-at-25 convention used for vanilla/weak/medium/strong). img3txt0p3's T→I plateaus by epoch 15–20. **Both variants' I→T is flat and near-zero across every epoch** — not a peaked-then-declined pattern, just consistently collapsed from epoch 5 onward. More training would not have changed the I→T conclusion.

### Sanity-anchored comparison (best epoch = 25 for all, Recall@1)

| Variant | λ_img | λ_txt | T→I R@1 | I→T R@1 |
|---|---|---|---|---|
| vanilla | 0.0 | 0.0 | 14.08% | **9.44%** |
| weak | 0.3 | 0.3 | 14.74% | 0.04% |
| medium | 1.0 | 1.0 | 11.15% | 0.02% |
| **strong** | 3.0 | 3.0 | **21.24%** | 0.04% |
| img3txt0 | 3.0 | 0.0 | 7.97% | 0.12% |
| img3txt0p3 | 3.0 | 0.3 | 5.45% | 0.08% |
| *(reference) raw CLIP-SF dense, no RQ/T5* | — | — | *55.50%* | *74.00%* |

All six GENIUS-pipeline numbers sit well below the raw dense-embedding ceiling (55.50%/74.00%), so nothing here is implausible on its face — no near-0%/near-100% measurement-bug signature. Configs were re-checked directly (`quantizer_path` in each eval YAML correctly points at its own Stage-1 checkpoint, `lambda_img`/`lambda_txt` values confirmed in the Stage-1 YAMLs) to rule out a wiring mix-up before trusting this table.

### Verdict on Yubao's hypothesis: **not supported**

Her hypothesis was that decoupling λ per direction could recover I→T (which strong collapses to ~0%) while keeping T→I competitive with strong's gain. Neither new variant does this:

- **I→T stays collapsed in both** (0.08–0.16%), statistically in the same broken regime as strong's 0.04% — nowhere near vanilla's 9.44%. Even img3txt0's **zero** text-side balance pressure (λ_txt=0.0, literally the same as vanilla's text-side treatment) does not recover I→T. This is the most informative negative result here: if the *only* thing distinguishing img3txt0 from vanilla is λ_img (text side is identically unregularized in both), and I→T still collapses, then a single shared encoder's balance loss on the *image* side alone is sufficient to break I→T — the two directions are not as separable as the "just decouple the lambdas" framing assumed, likely because both modalities pass through the same encoder and the same 8 RQ levels rather than having any modality-specific capacity.
- **T→I is also worse than vanilla in both new variants** (7.97% and 5.45% vs. vanilla's 14.08%), let alone strong's 21.24%. This is the more surprising half: applying λ=3.0 to only the image-side rows performs *worse* on T→I than either applying nothing (vanilla) or applying the same λ=3.0 uniformly to both modalities (strong). Splitting the regularization is not merely "no better than uniform" — it is actively worse than both alternatives on the one metric it was supposed to preserve.

**Net result: direction-aware λ is dominated by the existing single-λ options on every metric.** Vanilla remains the only variant with working I→T; strong remains the best T→I. This is a real, useful negative result (a completed, principled test of the mentor's stated hypothesis, not left untested) but it does not offer a new best variant.

### Connecting to ID-structure diagnostics (why might this be happening?)

From the codebook-health gate (`extensions/rq2_balance_study/gate_check_img3txt_codebook_health.py`, job 334896) run on the training pool before Stage-2:

| Variant | Levels collapsed (entropy <5% of max or top1 >90%) | Level 1–8 mean relative entropy | Recon cosine sim |
|---|---|---|---|
| vanilla | none | ~79% | 0.9522 |
| strong | none by gate threshold, but levels 2/3/5 sit at 20–21% relative entropy (top1 code = 82.5% of mass) | ~40% | 0.6337 |
| img3txt0 | none | ~63% | 0.7233 |
| img3txt0p3 | none (level 1 closest: 24.2%, top1 52.7%) | ~54% | 0.7122 |

Both new variants have **more uniform per-level code usage than strong** (higher mean relative entropy, no single level as concentrated as strong's level-2/3/5), yet **worse T→I than strong and no better I→T**. This reproduces, in a new setting, the same paradox already documented in Table 2/3 above: reconstruction fidelity and code-usage uniformity do not straightforwardly predict COCO T→I retrieval — strong's *most* collapsed levels coincide with its *best* T→I score, and here the *least* collapsed of the balance-regularized variants (img3txt0) gets the *worst* T→I among all regularized variants tried. Splitting λ by modality changed the ID-structure statistics in the "healthier-looking" direction without improving — and while actively hurting — retrieval. This is additional evidence (not yet a mechanistic explanation) that this RQ setup's entropy/utilization diagnostics are not a reliable proxy for retrieval quality, on top of the existing Table 3 finding.

### Independent audit (2026-07-10, fable-model) — implementation confirmed correct, mechanism identified

Given how counterintuitive this result is, a full adversarial re-verification was run before trusting it (re-derived from code + live test execution + raw logs, not a re-read of this document's own narrative):

**No bug found.** Per-row weight formula verified correct (text-only rows get exactly `w=λ_txt=0`, confirmed by a shape-assert that never fired across 150 clean epochs). The equivalence unit test was re-run live and still passes exactly. Checkpoint wiring re-confirmed independently at every stage. A clean quantitative sanity check: img3txt0's Stage-1 `rq_loss` (≈−9.0) is almost exactly half of strong's (≈−16.5) — exactly what "half the rows carry λ=3, half carry λ=0" predicts.

**Mechanism — the "text side is unchanged from vanilla" premise is architecturally false**, and there is direct evidence of the coupling, not just a plausible story:
1. Image and text rows share a single `Combiner` encoder forward pass — the image-side diversity gradient updates encoder weights that also produce text embeddings.
2. `cl_loss`/`mse_loss` explicitly tie text-query geometry to image-pool geometry within the same batch.
3. RQ levels 1–8 are a **shared, EMA-updated codebook across all rows regardless of loss weight** — the EMA update has no dependence on which rows' loss included the diversity term at all.

**Direct evidence (not inferred)**: img3txt0's *text* candidate codes are massively restructured despite λ_txt=0 — 566 unique level-1 prefixes among text candidates vs. vanilla's 946 (while still 99.4% unique at full depth, unlike strong's 62.8%) — proving image-side regularization physically alters text-side code structure through the shared components above.

**What explains the T→I ordering specifically**: unique level-1 codes among the 5,000 image candidates track T→I R@1 almost monotonically — vanilla 1249 (R@1≈20%), strong 1230 (≈22%), img3txt0 603 (≈8.0%), img3txt0p3 391 (≈5.5%). Strong preserves level-1 diversity (the level constrained beam-search pruning is most sensitive to) while collapsing mid-levels to near-constants T5 can learn easily (Stage-2 training loss 1.79 vs. vanilla's 3.64); the asymmetric cells instead *concentrate* level-1 without that learnability payoff (Stage-2 loss 2.5–2.9, a normal/healthy range — Stage-2 training itself was not the problem).

**Still genuinely open** (correctly flagged, not resolved by the audit): why level-1 concentration specifically follows from asymmetric λ is correlational here, not causally proven; why I→T lands at exactly ~0.1% rather than some other collapsed value isn't directly demonstrated (most likely a T5 generation failure on the restructured text-code prefixes, but untested); and the n=1-per-cell limitation below stands as the single most valuable next check.

### Limitations (state explicitly, don't overstate confidence)

- **n=1 per cell** — no seed-variance run for img3txt0/img3txt0p3, unlike the existing 3-seed check for vanilla/weak/medium/strong on FashionIQ (and the in-progress COCO 2-seed check). Given how counterintuitive the T→I-worse-than-vanilla result is, a seed-variance check is the single most valuable next step before treating "direction-aware λ underperforms uniform λ" as a stable conclusion rather than one unlucky training trajectory — this project has already documented real seed-to-seed non-reproducibility in RQ training (DDP/cuDNN non-determinism interacting with VQ's multi-modal loss landscape, see FashionIQ weak/medium's ~50% level-1 concentration shift on repair-retrain).
- Only two points on the λ_img/λ_txt grid were tried (both with λ_img fixed at 3.0). The hypothesis isn't fully ruled out for other combinations (e.g. lower λ_img, or λ_txt < 0 conceptually meaningless but a finer λ_txt sweep near 0), only for these two specific settings.
- The "why is img3txt0 worse than both vanilla and strong on T→I" question is answered here only at the level of "the ID-structure diagnostics don't explain it either" — a genuine mechanistic account (e.g. does per-row-weighted loss create harder-to-optimize gradient conflicts within a batch that mixes image and text rows, compared to a single uniform target) was not investigated and would need dedicated follow-up, not assumed from this data alone.
