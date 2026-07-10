# RQ Code Analysis: Semantic ID Diagnostic

**Date assigned:** June 22, 2026  
**Assigned by:** Yubao Tang  
**Context:** Extension 2 of the GENIUS reproducibility project. The goal is to diagnose whether the released residual quantizer (`rq_clip_large.pth`) produces semantic IDs that are discriminative, well-distributed, and semantically meaningful — especially in high-similarity retrieval settings.

---

## Setup and Data Requirements

**Quantizer checkpoint:** `rq_clip_large.pth` (frozen, do not retrain)  
**Location on cluster:** `/home/tcetoje/GENIUS-CVPR25/` (check `configs/` for exact path)  
**Embeddings needed:** CLIP-SF embeddings for COCO candidates (images + captions)  
**Embedding path:** `/fnwi_fs/ivi/irlab/personal/tcetoje/` (Stage 0 `.pt` files)  
**Dataset:** COCO (primary); M-BEIR multi-dataset subset for Analysis 4

**Key constraint:** The **first token** of every semantic ID encodes modality (0=image, 1=text, 2=image-text pair). **Exclude this token from all analyses below.** Work only with codes at positions 2 through M (i.e., `id[1:]`).

**RQ configuration (from paper):** Codebook size K=4096, levels M=9 (first level reserved for modality), so 8 semantic levels remain.

---

## Analysis 1: Codebook Utilization

**Goal:** Determine whether each codebook level uses its available capacity effectively, or whether the code space is dominated by a small number of codes (collapse/imbalance).

### Steps

1. Load `rq_clip_large.pth` and encode all COCO candidates (images + captions) to get their full RQ code sequences.
2. For each RQ level `l` in `{1, 2, ..., 8}` (0-indexed after excluding modality token):
   - Collect the code assigned at level `l` for every candidate.
   - Count frequency of each code value in `[0, 4095]`.
   - Compute:
     - **Used codes:** number of distinct codes that appear at least once
     - **Utilization rate:** `used / 4096 * 100` (%)
     - **Unused code ratio:** `(4096 - used) / 4096 * 100` (%)
     - **Code-frequency entropy:** `H = -sum(p * log2(p))` over all non-zero frequencies; max entropy = log2(4096) = 12 bits
     - **Top-5 most frequent codes** and their frequencies

### Outputs

- **Table 1:** Per-level statistics

  | RQ Level | Used Codes | Utilization (%) | Unused (%) | Entropy (bits) | Max Entropy (bits) | Top-1 Code Freq |
  |----------|-----------|-----------------|------------|----------------|-------------------|-----------------|
  | 1        | ...       | ...             | ...        | ...            | 12.0              | ...             |
  | ...      | ...       | ...             | ...        | ...            | 12.0              | ...             |

- **Figure 1:** Rank-frequency plot for each level (log-log scale); one subplot per level. X-axis = code rank, Y-axis = frequency.

### Main question
Does each RQ level use its available capacity effectively, or is there evidence of code collapse or heavy imbalance?

---

## Analysis 2: Prefix Collision and Index Discriminability

**Goal:** Measure how effectively longer ID prefixes narrow down the candidate pool. A perfect ID scheme would reduce the candidate set to ~1 item by the full prefix length.

### Definitions

- **Prefix of length k:** the tuple `(code_1, code_2, ..., code_k)` for a candidate (after excluding the modality token).
- **Collision:** two distinct candidates share the same prefix.
- **Prefix bucket:** the set of all candidates sharing a given prefix.

### Steps

For prefix lengths `k` in `{1, 2, 3, 4, 8}` (1=first semantic code, 8=full sequence):

1. Compute the prefix of length `k` for every candidate.
2. Group candidates by their prefix (build a dict: prefix → list of candidate IDs).
3. For each `k`, compute:
   - **Unique prefixes:** number of distinct prefix values observed
   - **Mean bucket size:** `total_candidates / unique_prefixes`
   - **Median bucket size**
   - **Max bucket size:** largest prefix group
   - **Collision rate:** `% of candidates sharing their prefix with at least one other candidate`
   - **Within-dataset collision rate:** collision rate computed only among COCO-COCO pairs
   - **Cross-dataset collision rate:** collision rate between COCO and non-COCO candidates (requires candidates from at least one other M-BEIR dataset)

### Outputs

- **Table 2:** Collision statistics per prefix length

  | Prefix Length | Unique Prefixes | Mean Bucket | Median Bucket | Max Bucket | Collision Rate (%) | Within-DS (%) | Cross-DS (%) |
  |---------------|----------------|-------------|---------------|------------|-------------------|---------------|--------------|
  | 1             | ...            | ...         | ...           | ...        | ...               | ...           | ...          |
  | ...           | ...            | ...         | ...           | ...        | ...               | ...           | ...          |

- **Figure 2a:** Box plot of bucket size distribution for each prefix length (log-scale Y-axis).
- **Figure 2b (optional):** Table of top-10 most crowded prefixes at length 3, showing prefix tuple and bucket size.

### Main question
Do later codes provide sufficient additional discrimination, or do many candidates remain grouped under the same prefixes even at full sequence length?

---

## Analysis 3: Cross-Modal Semantic Alignment

**Goal:** Test whether semantically matched image-caption pairs receive more similar semantic codes than random or hard-negative pairs. This verifies whether the quantizer actually encodes cross-modal semantics.

### Pair types

1. **Matched pairs:** COCO image + its ground-truth caption (from COCO annotations)
2. **Random pairs:** COCO image + randomly sampled caption from a different image
3. **Hard-negative pairs:** COCO image + the caption most similar to it by CLIP-SF cosine similarity, but *not* a ground-truth match (top-1 nearest neighbor excluding GT)

Sample at least **5,000 pairs** of each type for statistical stability.

### Metrics (computed per pair)

- **Shared prefix length:** longest common prefix between image ID and caption ID (both with modality token excluded); value in `{0, 1, ..., 8}`
- **Hamming distance:** number of positions where the two code sequences differ; value in `{0, 1, ..., 8}`
- **Code agreement at level l:** binary indicator `code_image[l] == code_caption[l]`; averaged over pairs gives agreement probability at that level

### Steps

1. For each pair type, encode both items to get code sequences (exclude modality token).
2. Compute the three metrics above for each pair.
3. Compute mean ± std per metric per pair type.
4. For code agreement, compute the agreement probability at each of the 8 levels.

### Outputs

- **Table 3:** Summary statistics per pair type

  | Pair Type     | Mean Shared Prefix | Mean Hamming Dist | % Identical IDs |
  |---------------|-------------------|--------------------|-----------------|
  | Matched       | ...               | ...                | ...             |
  | Random        | ...               | ...                | ...             |
  | Hard-negative | ...               | ...                | ...             |

- **Figure 3a:** Bar chart of code-agreement probability at each RQ level, with three grouped bars (matched / random / hard-negative) per level.
- **Figure 3b:** Violin plot or histogram of shared-prefix-length distribution for each pair type.

### Main question
Does the released quantizer place semantically matched image-text pairs closer in discrete ID space than unmatched or hard-negative pairs?

---

## Analysis 4: Dataset Dependence of Semantic Codes

**Goal:** Determine whether the semantic codes capture transferable semantics across datasets, or whether they strongly reflect dataset identity (i.e., candidates from the same dataset are more similar to each other than to candidates from different datasets).

### Data requirement

Need candidates from at least 3–4 M-BEIR datasets with known CLIP-SF embeddings. Use balanced samples (~1,000–2,000 candidates per dataset). Suggested datasets: COCO, FashionIQ, and any 2 others available from tnguyen5's M-BEIR data at `/fnwi_fs/ivi/irlab/personal/tnguyen5/M-BEIR/`.

### Sub-analysis A: Within-dataset vs. cross-dataset code similarity

1. Sample 500 candidate pairs from the *same* dataset (repeated for each dataset).
2. Sample 500 candidate pairs from *different* datasets (balanced across dataset pairs).
3. For each pair, compute shared-prefix length and Hamming distance.
4. Report mean ± std per setting.

### Sub-analysis B: Dataset predictability from semantic codes

Train a logistic regression classifier to predict which dataset a candidate comes from, using only its semantic code sequence as input.

- **Feature representation:** One-hot encode each level's code value and concatenate → sparse feature vector of dimension `8 × 4096 = 32768`. Alternatively, use the raw integer code sequence of length 8 as ordinal features (try both).
- **Train/test split:** 80/20, stratified by dataset.
- **Baselines:**
  - Majority-class baseline (always predict the most frequent dataset)
  - Shuffled-code baseline (randomly permute code values within each candidate, retrain)
- **Metrics:** Accuracy, macro-F1, per-class F1

### Outputs

- **Table 4:** Within- vs. cross-dataset code similarity

  | Setting         | Mean Shared Prefix | Mean Hamming Dist |
  |-----------------|-------------------|--------------------|
  | Within-dataset  | ...               | ...                |
  | Cross-dataset   | ...               | ...                |

- **Figure 4a:** Dataset-by-dataset heatmap of average shared prefix length (N_datasets × N_datasets matrix, symmetric).
- **Figure 4b:** Confusion matrix for dataset prediction.
- **Table 5:** Classification results

  | Classifier              | Accuracy | Macro-F1 |
  |-------------------------|----------|----------|
  | Logistic Regression     | ...      | ...      |
  | Majority baseline       | ...      | ...      |
  | Shuffled-code baseline  | ...      | ...      |

### Main question
Do the semantic IDs form a shared cross-dataset semantic space, or are they strongly partitioned by dataset/domain?

---

## Implementation Notes

### Loading the quantizer

```python
import torch
# Load the quantizer checkpoint
rq_state = torch.load("rq_clip_large.pth", map_location="cpu")
# The quantizer encodes a CLIP-SF embedding → sequence of M integer codes
# First code = modality token → always exclude: use codes[1:] in all analyses
```

Check `src/` or `models/` in the GENIUS repo for the `ResidualQuantizer` class and its `.encode()` method. The input is a normalized CLIP-SF embedding (dim=768).

### Embedding the candidates

Stage 0 `.pt` files already contain CLIP-SF embeddings. Load them directly rather than re-running the encoder:

```python
data = torch.load("/fnwi_fs/ivi/irlab/personal/tcetoje/<dataset>_candidates.pt")
embeddings = data["embeddings"]  # shape: [N, 768]
```

Then pass through the quantizer's encode step to get code sequences.

### Environment

Run on the ILPS cluster. Use an interactive A6000 job for development, then submit via Slurm for full-dataset runs. Logs → `/home/tcetoje/logs/rq_analysis/`.

---

## Output Files

Save all outputs under:
```
/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis/
  figures/
    analysis1_rank_freq.png
    analysis2_bucket_dist.png
    analysis3_agreement_bars.png
    analysis3_prefix_violin.png
    analysis4_dataset_heatmap.png
    analysis4_confusion_matrix.png
  tables/
    table1_utilization.csv
    table2_collisions.csv
    table3_alignment.csv
    table4_within_cross.csv
    table5_classification.csv
  logs/
    rq_analysis.log
```

---

## Connection to Research Questions

| Analysis | Research Question Addressed |
|----------|-----------------------------|
| 1 (Utilization) | Is the code space well-used, or is there collapse that limits discriminability? |
| 2 (Collisions) | How many candidates share the same prefix? Does the ID degrade to near-random guessing in large pools? |
| 3 (Cross-modal) | Does the quantizer actually encode cross-modal semantics, or are image and text IDs decorrelated? |
| 4 (Dataset dep.) | Are the IDs transferable, or is dataset identity the dominant signal? |

These analyses directly support **Extension 2** of the thesis: *"Semantic ID Diagnostic Analysis"* — understanding why GENIUS degrades within same-dataset pools (confirmed in prior collision experiments: 87% collision rate at prefix length 2 within COCO).
