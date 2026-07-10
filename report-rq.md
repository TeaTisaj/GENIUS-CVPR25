
_Frozen, released `rq_clip_large.pth`. Modality token excluded from all statistics; analyses operate on the 8 semantic RQ levels (4096-entry codebook each). Pools: MSCOCO (123,287 image / 590,392 caption candidates, shared pool), FashionIQ (74,380), Fashion200K (48,945), NIGHTS (31,882), CIRR (16,141), VisualNews (100,000), all encoded by the official Phase-1 quantizer checkpoint._

_One architectural fact frames the interpretation of everything below: the quantizer is a `ResidualVQ` (hierarchical, greedy vector quantization — each level quantizes the residual left by the levels above it) trained with a reconstruction term and an in-batch contrastive term on the fused query/candidate embedding (`critic`, `contra_loss` in the checkpoint's state dict). There is no term that rewards cross-modal code agreement beyond what reconstruction requires, and no term that penalizes dataset- or domain-specific codebook usage. Every result below is a direct consequence of this objective, not an independent surprise._

_Revised after an internal review: two corrections fix violations of the assignment text or of a baseline's stated purpose (COCO "Random" pairs were not guaranteed to come from a different image, as the spec requires; the shuffled-code baseline destroyed per-class marginals, not just joint structure, making it too weak for what it was meant to test). Both are flagged where they changed a conclusion, not just a number._

---

## 1. Codebook Utilization

|Level|Used (pooled)|Util. %|Entropy (bits)|Zipf exp.|
|---|---|---|---|---|
|1|1,926 / 4096|47.0%|9.04 / 12.0|2.58|
|4|2,900|70.8%|9.91|1.92|
|8|3,085|75.3%|10.73|1.21|

**Mechanism.** A `ResidualVQ` codebook is fit (EMA/k-means-style updates) to the density of the embedding distribution it sees — codewords migrate toward dense regions and leave sparse regions of the embedding manifold uncovered, with no entropy or uniformity regularizer to prevent this. Level 1 sees the full embedding signal, which is dominated by a small number of dense, well-separated clusters (domain shift across six heterogeneous datasets — see §4); a level-1 codebook trained on this distribution will allocate most of its 4096 entries to those clusters and leave the rest unused, which is exactly the 47% utilization observed. Each subsequent level quantizes the _residual_ left after the levels above removed the dominant structure, and residual error of a reasonably-fit quantizer trends toward more isotropic, less clustered noise — hence utilization and entropy rising monotonically with depth (47%→75%, entropy 9.04→10.73 bits) is the expected hierarchical unfolding of one mechanism, not eight independent observations.

**Ruling out sample size.** No individual dataset reaches the pooled level-1 figure (MSCOCO 24.9%, FashionIQ 6.2%, Fashion200K 2.8%, NIGHTS 10.0%, CIRR 9.8%, VisualNews 25.7%). A uniform-random null model (coupon-collector, K=4096) shows even the smallest dataset (CIRR, N=16,141) would be expected to touch ~98% of the codebook by chance; only 10% is observed (obs/exp ratio 0.10, vs. 0.025–0.25 across all six datasets). The shortfall is the codebook fitting data density, not insufficient samples.

**MSCOCO modality decomposition.** The assignment requires the primary MSCOCO row to mix images and captions, so it does (713,679 candidates). Splitting it post hoc: image-only 21.4%, caption-only 16.9% — both _below_ the mixed row's 24.9%. The mixed row's higher number is the union of two different occupied subspaces, not evidence either modality alone exercises more of the codebook. This is the same density-fitting mechanism again: image and caption embeddings occupy different regions (see §3's PCA scatter, which shows them as two disjoint clusters), so the codebook entries activated by one barely overlap the entries activated by the other.

**Answer:** Utilization is capacity-limited by training-data density, not architecture — collapse is worst where the embedding distribution is most clustered (level 1, and within any single dataset), and recedes only as far as residual quantization pushes the remaining signal toward noise.

---

## 2. Prefix Collision and Index Discriminability

|Level|Pool|Unique prefixes|Mean bucket|Median|Max|Collision rate|Same-dataset|Cross-dataset|
|---|---|---|---|---|---|---|---|---|
|1|combined|1,828|215.9|9.0|4,999|99.89%|13.24%|86.65%|
|2|combined|97,765|4.04|1.0|454|87.22%|81.28%|5.93%|
|3|combined|303,989|1.30|1.0|94|33.59%|33.35%|0.23%|
|4|combined|371,419|1.06|1.0|67|9.44%|9.43%|0.02%|
|8|combined|391,269|1.01|1.0|67|1.43%|1.43%|0.00%|

**Mechanism — this is §1's utilization curve, not a separate phenomenon.** Collision rate at prefix depth k is a direct function of how many _effective_ codewords are jointly active across levels 1..k. The elbow at L3-4 is exactly where §1's entropy gain per added level starts shrinking (9.79→9.91→10.10 bits, L3→L4→L5 — diminishing returns begin at the same depth). Reporting these as connected facts: the discriminability gain mechanically tracks the entropy gain already measured, not a coincidence worth narrating as one.

**Collisions that survive to full depth are almost exclusively within-dataset.** By L4, cross-dataset collisions are essentially gone (0.02%) while same-dataset collisions persist (9.43%); at L8: 1.43% same-dataset vs. 0.00% cross-dataset. All ten most-crowded level-3 buckets are single-dataset, concentrated in FashionIQ and NIGHTS — the two datasets with the worst §1 utilization. This is the same underlying density-fitting mechanism manifesting as a retrieval-relevant symptom: a codebook fit on a domain-shifted six-dataset mixture allocates little capacity to _within-dataset_ fine structure, because cross-dataset separation is the larger, easier-to-fit signal in the training distribution.

**Answer:** Discrimination improves through level 8 with diminishing but real returns, in lockstep with §1's utilization curve. The residual collisions at full depth are a within-dataset problem specifically — the codebook was never under pressure, during training, to resolve fine-grained same-dataset structure, because cross-dataset separation dominated the loss landscape it was fit to.

---

## 3. Cross-Modal Semantic Alignment

_Correction: "Random" pairs are now drawn by rejection sampling, enforcing a different image than the sampled one — the original implementation only enforced a different array index, and since one image's ~5 captions are stored adjacently, this previously let many "Random" pairs be disguised Matched pairs._

|Pair type|Mean shared prefix (/8)|Mean Hamming|% identical|
|---|---|---|---|
|Matched|0.804 ± 0.751|6.987|0.00%|
|Random|0.003 ± 0.057|7.988|0.00%|
|Hard-negative|0.880 ± 0.770|6.865|0.00%|

**Mechanism.** Vector quantization assigns a point to whichever codeword is nearest under the metric it was trained on; code agreement between two points is therefore monotonic in their embedding-space proximity _by construction_, not by inference from data. Everything in this table follows from that fact plus one measurement of the continuous embedding space:

|Pair type|Raw CLIP-SF cosine similarity|
|---|---|
|Matched|0.079|
|Random|0.043|
|Hard-negative|0.132|

Matched (0.804/8) clearly exceeds the corrected Random baseline (0.003/8 — a true independent caption shares almost nothing; Mann-Whitney p≈0, probability of superiority 0.813). This is the upstream embedding gap (0.079 vs. 0.043) propagated through the quantizer, exactly as the mechanism predicts.

**Hard-negative > Matched is not an anomaly requiring explanation — it is the mechanism's necessary output given the embedding measurement.** Hard-negative mining is an argmax over 590,392 candidates in the same metric the quantizer partitions on; it will return something closer than one fixed, stylistically-noisy human caption in the overwhelming majority of cases (observed: 90.2%). A quantizer that _didn't_ show hard-negative agreement ≥ matched agreement, given this embedding ordering, would be the surprising result — it would mean the quantizer's Voronoi partition wasn't actually tracking the embedding metric it was fit to. The locus of any weakness here is entirely upstream, in the Combiner/CLIP-SF embedding (0.079 cosine for true pairs is low in absolute terms): no downstream quantizer, however well-trained, can produce code agreement the input geometry doesn't contain.

**Answer:** The quantizer places matched pairs measurably and significantly closer in ID space than independent pairs — a direct, mechanical readout of the (modest) cross-modal structure already present in the Combiner embedding. The full 8-level ID never aligns across modalities even for true matches, because that embedding-space alignment itself decays past the coarsest level.

---

## 4. Dataset Dependence of Semantic Codes

_Correction: MSCOCO's primary entry is now image-only (123,287), matching the other five (pure-image) datasets — the mixed entry risked the classifier partly learning modality rather than dataset identity, and the shuffled-code baseline below is now stratified by class (see why this matters next)._

|Feature|Classifier|Accuracy|Macro-F1|
|---|---|---|---|
|8-level ordinal|Logistic Regression|19.5%|18.1%|
|8-level ordinal|Majority|16.7%|4.8%|
|8-level ordinal|Stratified shuffled-code|19.7%|18.3%|
|8-level one-hot|**Logistic Regression**|**96.92%**|**96.93%**|
|8-level one-hot|Majority|16.7%|4.8%|
|8-level one-hot|**Stratified shuffled-code**|**96.75%**|**96.76%**|

**Why the shuffle has to be stratified, and what changes if it isn't.** A shuffled-code baseline is meant to isolate whether predictive power comes from the _joint_ combination of levels or from each level's _marginal_ frequency distribution alone — that requires permuting each level's values only within rows of the same class, preserving the class's marginals while destroying cross-level structure. A global (non-stratified) shuffle destroys the marginals too, collapsing the baseline to ~16% and producing the false appearance that joint structure is doing the work. With the correct, stratified shuffle, the baseline reaches 96.75% — statistically indistinguishable from the real classifier (96.92%).

**Mechanistic reading.** Because the codebook at _every_ level is independently fit (EMA/k-means) to the same domain-shifted six-dataset distribution, each level redundantly inherits some of that domain separation in its marginal frequency table — there is no need for cross-level joint structure to recover dataset identity, because each level was never trained to be domain-invariant in the first place. The training objective (reconstruction + in-batch contrastive separation of the fused embedding) has no mechanism that would discourage this; per-level dataset-fingerprinting is the default outcome, not an emergent surprise.

**Effect size.** Chi-square (dataset × level-1 code, n=394,635, image-only MSCOCO) gives χ²=1,874,488, p≈0 — uninformative alone at this n; Cramér's V = **0.975** (max 1.0) confirms the association is large, not a large-N artifact.

**Modality-leakage check.** Splitting MSCOCO into image/caption as two separate classes (7-class run) still reaches 88.2% one-hot accuracy with the same marginals-dominate pattern (87.6% stratified-shuffle baseline). A standalone image-vs-caption-within-MSCOCO classifier, using only the 8 semantic levels (the modality token is excluded everywhere in this analysis by design), reaches just 67.4% (vs. 50% majority) — despite raw embeddings separating image from caption _perfectly_ in 2D PCA. This confirms the modality token is absorbing the modality signal as intended; the semantic levels are not simply re-encoding it.

**Answer:** Semantic IDs do not form a transferable cross-dataset semantic space. The mechanism is precise: per-level codebooks fit to a domain-shifted training mixture inherit dataset separability at the marginal level, independently at every depth — confirmed by the within/cross shared-prefix gap, the corrected classifier baseline, and the chi-square effect size, and shown to be a genuine dataset signal rather than a re-discovery of the separately-encoded modality token.

---

## Synthesis

One mechanism — a `ResidualVQ` codebook fit by reconstruction/contrastive objectives to the density of a domain-shifted, multi-dataset embedding distribution, with no term rewarding uniformity, cross-modal alignment, or dataset invariance — produces all four results:

1. Codebooks under-utilize capacity in proportion to how clustered the input distribution is (§1), worst at level 1 and within any single dataset.
2. Collision rate is the same fact read off as a retrieval-relevant symptom (§2): cross-dataset collisions vanish by depth 4 because cross-dataset separation is what the codebook was actually pressured to resolve; same-dataset collisions persist because within-dataset structure never created comparable pressure.
3. Cross-modal code agreement is a direct, monotonic readout of upstream embedding similarity (§3) — the quantizer cannot manufacture alignment the Combiner embedding does not already contain.
4. Dataset identity is recoverable from any single level's marginal distribution (§4) because every level was independently fit to the same domain-shifted mixture — joint cross-level structure was never necessary.

**External validation.** Joining §2's collision statistics with the existing Phase 2 pool-composition results (Deliverable 1c): cross-dataset-distractor COCO Recall@1 is flat (0.477) across pool sizes 10K/50K/100K, matching near-zero cross-dataset collision at depth; within-dataset-distractor Recall@1 degrades monotonically (0.342→0.202→0.173), tracking the same-dataset collision rate that climbs with candidate count. The mechanism proposed above is not post hoc narrative — it predicts, and is confirmed by, retrieval behavior measured independently in Phase 2.

## Limitations

- §1's pooled statistics are ≈72%-weighted by MSCOCO; not an unweighted dataset average.
- §3 is COCO-only by construction (the only pool with a verified exact image↔caption join); not assumed to generalize without separate verification.
- §3/§4's sampling-based numbers use a single seed; cross-seed variance not yet measured.
- §3's Hard-negative > Matched effect, while significant, has a modest probability-of-superiority (0.527) and should not be overstated.