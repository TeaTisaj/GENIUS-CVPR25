# Senior-review fix plan — ECIR 2027 short paper (2026-09-14)

Reviewed: `paper-ecir/main.tex` + `sections/00`–`07` + `references.bib` (the short-paper
version made after Yubao's 2026-07-23 reply). Every item marked **[verified]** was checked
against raw files, logs, code, or the external source, not against an earlier draft.

Venue facts **[verified, ecir2027.co.uk/call-for-short-papers]**: 6 pages + unlimited
reference pages (refs do NOT count, so the current 6pp body + refs is compliant);
abstract due **5 Oct 2026**, paper **12 Oct 2026**; double-anonymous.

---

## Tier A — fix before sending to Yubao (a careful reviewer would catch these)

### A1. Finding 3 is contradicted by the paper's own Table 2
- Dense probe **[verified, `logs/genius_probe_collapse_localize_335034.log`]**: MSCOCO dense
  T→I is vanilla **26.65%** > strong **24.23%**. The real pipeline is the reverse
  (20.78 < 23.33).
- So the **MSCOCO T→I gain is NOT present at the tokenizer level**. It appears only after
  discretization and decoding. The tokenizer-level effects are the *degradations*: the I→T
  collapse (both datasets), the VisualNews T→I drop, and the weak/medium MSCOCO T→I drop.
- Wrong as written: abstract "(3) both effects are already present at the tokenizer level";
  intro item 3; §6 title "Originates at the Tokenizer Level"; §6 "T→I retains substantial,
  rank-correlated dense recall"; Conclusion "Both the direction-asymmetry and the
  dataset-dependence are already present at the RQ tokenizer level".
- Yubao's own wording was "the **degradation** can already be observed at the tokenizer
  level". That is exactly what the data supports.
- Optional, hedged observation that is worth one sentence: going from dense to real costs
  vanilla 22% relative (26.65→20.78) but strong only 4% (24.23→23.33). Strong's IDs may be
  easier for the decoder to generate, even though strong's embedding space is slightly worse.

### A2. Logic error in the per-modality argument (§5/§6)
- §6: "…per-modality decoupling cannot fix it: the damage is already encoded in the
  tokenizer, a place a **decoder-side loss weight** cannot reach." But λ_img/λ_txt is a
  **Stage-1 tokenizer** loss weight, not a decoder-side one. A reader who knows the setup
  will think the author does not understand their own method.
- §5: "This is not a tuning failure", "has no mechanism to produce a per-modality effect",
  "fails architecturally". Three settings cannot establish this. The paper's own observation
  (text codes change under image-only weight) shows the weight *does* have an effect; it just
  cannot *isolate* it.
- Suggested wording: "Image and text candidates share the RQ encoder and all codebooks, so a
  per-modality loss weight still changes parameters used by both modalities and cannot confine
  its effect to one of them. The observation that text-only regularization collapses I→T as
  completely as uniform regularization is consistent with this, but we do not isolate the
  mechanism."
- "566 vs. 946 unique level-1 prefixes": **no script or CSV produces this number**
  (it only appears in `rq1-rq3_final_tables.md` and is hardcoded in `figures/make_figures.py`).
  Cut it (this saves space) or recompute it with a persisted script.

### A3. Manipulation check: the regularizer does not make hard code usage more balanced
**[verified, `rq_analysis_{coco,fashioniq}_variants/tables/table1_id_structure_summary.csv`,
`table_collision_recon_summary.csv`]**

| | vanilla | weak | medium | strong |
|---|---|---|---|---|
| MSCOCO mean per-level entropy (bits, max 12) | 9.45 | 7.32 | 5.90 | 4.32 |
| MSCOCO utilization (%) | 57.7 | 55.9 | 63.4 | 68.2 |
| MSCOCO full-ID collision (%) | 0.9 | 6.4 | 22.7 | 69.7 |
| FashionIQ mean entropy (bits) | 8.77 | 7.45 | 7.65 | 7.75 |
| FashionIQ utilization (%) | 18.3 | 10.4 | 11.4 | 11.6 |

(MSCOCO medium, level 2: one code holds 575K of 713K candidates.)

- The loss **[verified, `vector_quantize_pytorch.py` L1243–1246]**:
  `prob = softmax(T·distances); avg = mean_batch(prob); loss = −H(avg)`. It maximizes the
  entropy of the *batch-averaged soft* assignment, with no per-sample sharpness term. It can
  be satisfied by flattening each sample's soft distribution rather than spreading hard
  assignments, which fits the table above.
- Consequence: the title and intro question ("does more balanced code usage improve
  retrieval?") is not what was manipulated. λ was manipulated, and higher λ gave *lower*
  hard-assignment entropy on both datasets. A reviewer who asks "did balance actually
  increase?" will find the answer is no.
- Fixes:
  1. Write the loss as one equation in Setup and cite the library. Stop attributing it to
     `zhang2023regularizedvq`, which is a different regularizer (a prior-distribution KL term).
  2. Add entropy and collision columns to Table 1, or one manipulation-check sentence.
  3. Decide the framing (**ask Yubao**): either rename it "diversity/balance *regularization*"
     and report that it does not produce balanced hard assignments (arguably a more
     interesting finding), or keep the title and state the manipulation result openly.
  4. VisualNews has no ID-structure diagnostics yet; `slurm_rq1_visualnews_variant_diagnostics.sh`
     exists (commit 21b8b2d). Check whether it ran.

### A4. The related-work novelty claim is factually wrong
**[verified, arxiv.org/html/2405.07314]** LETTER *does* evaluate the diversity loss on
retrieval: Table 2 ablation ("w/ d.r." improves TIGER's R@10/N@10), Fig. 6 (code-assignment
distribution with and without it), and Fig. 8 (a sweep of the diversity-loss weight).
- Wrong: "not evaluated for its effect on downstream retrieval" and "None of this prior work
  evaluates whether balance regularization helps or hurts retrieval".
- ECIR has many RecSys reviewers who know LETTER. Turn it into motivation instead:
  "LETTER reports that a diversity loss improves generative recommendation and spreads code
  assignments for single-modality item IDs. We ask whether this benefit holds across datasets
  and across retrieval directions when one codebook indexes both images and texts."
- Optional: LC-Rec (uniform semantic mapping to avoid ID collisions) and one multimodal GR
  paper (IRGen / GRACE). Verify the bib entries via web before adding.

### A5. The statistics are pseudo-replicated
- The p<10⁻⁶ test uses 5 **decoder** seeds on **one** tokenizer. The paper's own tokenizer
  replication shows tokenizer std (0.47/0.86) is about 8× decoder std (0.06/0.11).
- Welch's test over the 3 independent tokenizers **[computed today]**: t=5.60, df≈3.1,
  **p=0.010**; +14.9% relative (21.16±0.47 → 24.32±0.86).
- Make this the headline. Report the decoder-seed spread as secondary, without p<10⁻⁶.
  Remove "verified" from the abstract.
- VisualNews p<10⁻⁷ has the same issue. The −53% effect is large enough to report without a
  p-value, or label the test clearly as "across decoder seeds".
- Optional (IR reviewers like this): a paired per-query test (randomization or paired t-test
  over the 24,809 T→I queries), if the run files are kept.

### A6. Say "chance level"
**[verified]** The MSCOCO I→T pool is 24,809 captions and 5,000 queries with 5.0 positives
each, so random R@1 ≈ **0.02%**. Regularized I→T is 0.03–0.05%, which is chance. State it.
It is the clearest description of the collapse and answers the reader's first thought
("is this a bug?"), together with "also present in the decoder-free probe, so not a
decoding/trie artifact". Replace the "manufacture an impressive p-value… the effect size
speaks for itself" passage with this.

### A7. Absolute-performance paragraph (Yubao's explicit concern): currently the weakest part, and it contains a false sentence
- False: "Because the officially released GENIUS decoder is not public". **[verified]**
  `GENIUS/checkpoint/GENIUS_t5small.pth` is on disk (the HF release; the README says these are
  reimplemented checkpoints), and it was evaluated in this project (`GENIUS_reproduction_brief.md`).
- The paragraph discusses a union-trained model (8.1% COCO, 18.3% FashionIQ R@10) that no
  experiment in the paper uses. The FashionIQ "above published" argument is a non-sequitur:
  the study's FashionIQ vanilla R@10 is **4.76%**.
- The matched comparison already exists. GENIUS's COCO-only training result (T→I R@1 **40.1**,
  their Table 3, per `GENIUS_reproduction_brief.md`) is the direct analogue of the per-dataset
  pipeline, which gets **20.8%**. The dense CLIP-SF baseline on the same pool is 55.5%
  (memory; verify from `dense_clip_sf_raw_baseline.py` output, no log found).
- Replace the whole paragraph with about 3 sentences:
  "Our COCO-only vanilla pipeline reaches 20.8% T→I R@1, about half of the 40.1% GENIUS reports
  for COCO-only training [kim2025genius] and below dense CLIP-SF retrieval on the same pool
  (55.5%). We use no hard negatives and no validation-based checkpoint selection. We therefore
  present a controlled study: conclusions concern differences between λ settings under an
  identical pipeline, and we do not claim the effect sizes carry over to a better-tuned system."
  This saves about 150 words for A3/B1.

### A8. Pool sizes are wrong or inconsistent
- "MSCOCO 713,679" is the tokenizer-training candidate set. The **evaluation** pools are
  **5,000 images (T→I) and 24,809 captions (I→T)** **[verified, wc -l]**. Report eval pools
  (and check the FashionIQ 74,380 and VisualNews 199,903 figures the same way).
- "Pool size is also not what drives the dataset-dependence": the size-matched check only
  compares ID statistics, not retrieval. Soften to "is unlikely to be the only driver".

### A9. The MSCOCO λ curve is non-monotonic, and the text never says so
weak −32%, medium −50%, strong +12% (T→I). The abstract's "raises MSCOCO T→I significantly"
picks the single setting that helps. The honest version supports Finding 1 *better*:
"only the strongest setting improves MSCOCO T→I; weaker settings reduce it by up to 50%."

### A10. Citation misattributions (these read as AI-generated errors)
- `pradeep2023scale` studies corpus-size scaling on MS MARCO. It says nothing about "task
  diversity" (§2) or "capacity dilution across 16 joint tasks" (§3).
- `zhang2023regularizedvq` is not the loss used (A3).
- "RQ codebooks are EMA-fit … vanishing gradient": unused EMA codes get *no updates*, not
  vanishing gradients. EMA fitting is GENIUS's implementation, not a general property of RQ.
- "constrained beam search … over M-BEIR": beam search runs over a trie, not over a benchmark.
- "routine adoption … rarely evaluated directly" has no citation.

### A11. Limitations section
- Delete: "a limitation we regard as consistent with, rather than contradicting, this paper's
  central claim". This is unfalsifiable (every outcome supports the claim), and senior
  reviewers dislike it.
- Delete or support: "once code entropy partially recovers". That data is not in the paper.
- The tie-break check is not a limitation. Move it to Finding 1 as one clause, and say in
  Setup how ties within a shared ID are ranked.

---

## Tier B — important for ECIR reviewers

- **B1 Metric.** The M-BEIR standard is R@5 (MSCOCO, VisualNews) and R@10 (FashionIQ); the
  paper reports only R@1. FashionIQ R@1 differences (0.61 vs 0.25) are about 20 of roughly
  6K queries and look like noise. R@10 is decisive: **vanilla 4.76±0.05 vs 1.60–1.82**
  **[verified, `table_seed_variance_r1r5r10.csv`]**. Switch FashionIQ to R@10. Check whether
  MSCOCO/VisualNews R@5 exists for all seeds.
- **B2 Checkpoint selection.** MSCOCO and VisualNews: every variant is evaluated at epoch 25
  **[verified, eval configs]**. Say so. FashionIQ eval configs reference epochs 10, 15 and 20.
  Confirm which epoch each reported FashionIQ number uses. If it was best-per-variant on the
  test sweep, switch to a fixed epoch or disclose it.
- **B3 Probe definition.** State whether queries, candidates, or both go through the RQ in the
  dense probe.
- **B4 Code.** "Released at acceptance" is weak for an empirical paper. The anonymized repo
  exists locally (`~/genius_balance_reg_repro`); use an anonymous.4open.science link at
  submission (discuss with Yubao).
- **B5 Correlation paragraph.** It spends about 100 words explaining an MSCOCO r it then
  disowns, with Pearson r to 2 decimals at n=4. Compress to 2 sentences.
- **B6 Author block.** "Supervised by … April–July 2026" reads as an internship report. Ask
  Yubao about the author list (Tea / Yubao Tang / Maarten de Rijke) instead of deciding alone.
- **B7 Housekeeping.** Two overfull hboxes in `03_experimental_setup.tex` (lines 6, 8);
  add the abstract deadline (5 Oct) to the `main.tex` comment.

---

## Tier C — remove the "AI voice"

- **43 " -- " dashes** in about 3,000 words (12 in §4 alone). Dash-spliced sentences are the
  most recognizable LLM tell. Target fewer than 10 and split into shorter sentences.
- Rhetorical lines to delete or make plain:
  - "The effect size speaks for itself."
  - "would manufacture an impressive p-value rather than earn one"
  - "We claim the weaker but more damaging thing"
  - "It does not work."
  - "This is not a tuning failure"
  - "the identical checkpoints tell opposite stories"
  - "fails architecturally"
  - "shattering an already-narrow text cluster"
  - "One caveat on rigor:"
- The "neither X nor Y: it is A, B, C and D" pattern appears twice (end of §6, first sentence
  of the Conclusion).
- The same three findings are restated four times in nearly the same words (abstract, intro
  list, section titles, conclusion). Also: §4's last paragraph repeats the VisualNews reversal
  from the same section, and §5's last sentence previews §6. Once in the intro and once in the
  conclusion is enough.
- Very long sentences: the first abstract sentence (~60 words) and the §3 scope paragraph.
- "State-of-the-art" appears twice for GENIUS; heavy \emph{} for emphasis (already, halves,
  only). Cut most of it.
- Advice: write the abstract, contribution list, and conclusion yourself in plain, short
  English, then use AI only to correct grammar. Plain non-native English is not penalized
  at ECIR. Polished prose with claims the tables do not support is.

---

## Order of work (about 3 days, no new GPU compute needed)

1. **Day 1 (claim consistency):** A1, A2, A5, A6, A9, A11, A10, A4. Data already exists.
2. **Day 2 (content):** A3 (equation + columns), A7 rewrite, A8, B1–B3, B5; recompile to 6pp.
3. **Day 3 (style + check):** Tier C pass, then a fresh cold fact-check that **re-derives every
   number from raw CSVs/logs**, not from the previous draft.
4. **Send to Yubao** with 3 questions: framing after A3, authorship, anonymous repo at submission.

## Keep as is (strengths)
Controlled single-variable design; the tokenizer-reseeding replication; the decoder-free probe
idea; honest disclosure of the VisualNews validation split; the tie-break check; the 3-finding
structure matching Yubao's suggestion; page budget already compliant.
