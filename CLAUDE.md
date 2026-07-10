# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Conda env**: `genius2` (Python 3.10, PyTorch) — activate with `conda activate genius2`
- Install/recreate: `conda env create -f genius_env.yml`
- All training/eval scripts require `PYTHONPATH` to point to `src/`. The shell scripts in `configs_scripts/` set this automatically; if running Python directly, set `export PYTHONPATH=/path/to/GENIUS-CVPR25/src`

## Cluster (ILPS)

- Headnode: `ilps-h1.science.uva.nl` — never run GPU jobs here, always `sbatch`
- Slurm partition: `gpu`; logs in `/home/tcetoje/logs/`
- Preferred GPU nodes: `ilps-cn[116-120]` — A6000 (48 GB) nodes. GRES name: `nvidia_rtx_a6000`
  - cn116: 4× A6000; cn117, cn118, cn120: 8× A6000; cn119: 8× L40
- Job scripts live in `scripts/`; conda init at `/home/tcetoje/miniconda3/etc/profile.d/conda.sh`

## Data Paths (COCO / M-BEIR)

- MBEIR data root: `/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/`
- Raw COCO M-BEIR JSONLs: `.../mbeir_data/src_data/mscoco/`
- COCO images (read-only, symlinked from shared): `.../mbeir_data/mbeir_images/mscoco_images/`
- Shared datasets (never modify): `/fnwi_fs/ivi/irlab/datasets/`
- After running `scripts/setup_coco_mbeir_dirs.sh`, JSONL files are in the expected layout under `.../mbeir_data/cand_pool/local/` and `.../mbeir_data/query/`

## Three-Stage Pipeline

### Stage 0 — CLIP-SF Feature Extraction (no training)

Download the pretrained encoder if missing:
```bash
mkdir -p checkpoint/CLIP_SF
wget https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth \
     -O checkpoint/CLIP_SF/clip_sf_large.pth
```

Run extraction (edit `genir_dir` and `MBEIR_DATA_DIR` in the shell scripts first):
```bash
cd src/feature_extraction
bash run_feature_extraction_train.sh   # → extracted_embed/CLIP_SF/train/
bash run_feature_extraction_cand.sh    # → extracted_embed/CLIP_SF/cand/
```

### Stage 1 — Residual Quantization Training

```bash
cd src/models/residual_quantization
# Edit genir_dir + MBEIR_DATA_DIR in the script, and data paths in inbatch.yaml
vim configs_scripts/large/train/inbatch/inbatch.yaml
bash configs_scripts/large/train/inbatch/run_inbatch.sh
```

### Stage 2 — Generator Training

```bash
cd src/models/generative_retriever
vim configs_scripts/large/train/inbatch/inbatch.yaml   # set codebook_config.quantizer_path, data paths
bash configs_scripts/large/train/inbatch/run_inbatch.sh
```

### Inference / Evaluation

```bash
# Compile C++ trie (do once; required for trie_cpp mode)
cd src/models/generative_retriever
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    trie_cpp.cpp -o trie_cpp$(python3-config --extension-suffix)

# Edit genir_dir + MBEIR_DATA_DIR in the eval script, then run
bash configs_scripts/large/eval/inbatch/run_eval.sh
```

Trie options (set `model.trie_type` in eval config): `trie_cpp` (fastest, C++), `trie` (Python), `marisa`.

## Architecture

GENIUS performs **generative retrieval** — it decodes discrete document IDs rather than doing nearest-neighbour search over dense embeddings. Retrieval is O(1) with respect to corpus size.

| Component | File | Role |
|-----------|------|------|
| **CLIP-SF** (Score Fusion) | `src/models/uniir_clip/clip_scorefusion/clip_sf.py` | Shared multimodal encoder; produces 768-dim joint image/text embeddings |
| **Combiner** | `src/models/residual_quantization/residual_quantization.py` | Fuses image+text CLIP features via dynamic scalar weighting (from CLIP4CIR) |
| **RQ** (Residual Quantization) | `src/models/residual_quantization/residual_quantization.py` | Quantizes embeddings into layered discrete IDs: code[0] = modality indicator (0=image, 1=text, 2=image-text); codes[1..8] = semantic hierarchy (vocab size 4096 each) |
| **T5ForGenerativeRetrieval** | `src/models/generative_retriever/retriever.py` | Frozen RQ + T5-small decoder; an `embed_projector` maps 768-dim embeddings into 30 prefix tokens for T5 |

**Training flow for Stage 2**: query and pool embeddings are mixed via Beta-distribution augmentation (`aug_emb = sqrt(s)*q + sqrt(1-s)*p`) before being projected into T5 prefix tokens. T5 then generates the target document's ID token sequence.

**Inference flow**: CLIP-SF → RQ `inference()` → `embed_projector` → T5 `constrained_beam_search()` over a prebuilt trie of valid ID sequences.

**GENIUSᴿ** (reranking variant): same generative retrieval, followed by a reranking step using the original CLIP embeddings (`rerank: true` in eval config).

## Config System

All configs use [OmegaConf](https://omegaconf.readthedocs.io/) YAML. The `common/config_updater.py` script is called from shell scripts to patch YAML files before training (e.g., `--update_mbeir_yaml_instruct_status`).

Key fields to configure in YAML before running:
- `genir_dir` / `--genir_dir`: root of this repo (passed as CLI arg to training scripts)
- `mbeir_data_dir` / `--mbeir_data_dir`: path to downloaded M-BEIR dataset
- `codebook_config.quantizer_path`: path to the Stage 1 checkpoint (relative to `genir_dir`)
- `model.ckpt_config.ckpt_name`: checkpoint filename to load for eval

Training uses Weights & Biases (`wandb_config.wandb_key` must be set, or disable with `wandb_config.enabled: false`).

## Data Format

Training and evaluation use the **M-BEIR** format (JSONL). Each entry has fields for image path, text, modality masks, and task type. Dataset preprocessing scripts are in `src/data/preprocessing/` (one file per source dataset: CIRR, EDIS, Fashion200K, FashionIQ, etc.).

Supported M-BEIR task types (from `src/data/preprocessing/utils.py`):
- `text→image` (0), `text→text` (1), `text→(image,text)` (2)
- `image→text` (3), `image→image` (4)
- `(image,text)→text` (6), `(image,text)→image` (7), `(image,text)→(image,text)` (8)

## Key Files

- `src/models/generative_retriever/retriever.py` — main model class `T5ForGenerativeRetrieval`; also contains `ContrastiveLoss` and trie-based retrieval logic
- `src/models/residual_quantization/residual_quantization.py` — `RQ` model and `Combiner`
- `src/models/generative_retriever/beam.py` — Python `Trie` and `MarisaTrie` implementations
- `src/models/generative_retriever/trie_cpp.cpp` — C++ trie via pybind11 (must be compiled)
- `src/common/mbeir_generative_retriever.py` — retrieval pipeline (code generation → trie lookup → optional reranking)
- `src/data/mbeir_dataset.py` — dataset classes (`MBEIRListInstructioneDataset`, `MBEIRDictInstructioneDataset`)

## Model Checkpoints

All three checkpoints are required for end-to-end inference:
- **Stage 0** `checkpoint/CLIP_SF/clip_sf_large.pth` — from UniIR (TIGER-Lab)
- **Stage 1** `GENIUS/checkpoint/rq_clip_large.pth` — residual quantization model
- **Stage 2** `GENIUS/checkpoint/GENIUS_t5small.pth` — T5-small generator

The `GENIUS/` subdirectory is a clone of the HuggingFace model repo (checkpoints only); it has its own `CLAUDE.md` describing that repo's layout.


---

## Project Plan — Reproducibility Track (updated 2026-06-02)

This section reflects the current project direction as agreed with the supervisor.
See `new_guidelines.md` for the full original document.

### Project Goal

**Reproducibility-track submission.** Reproduce the core GENIUS pipeline, then study one key challenge:
> Can generative multimodal retrieval still work well when many candidates are highly similar?

This challenge appears in fashion datasets (FashionIQ, Fashion200K, CIRR, NIGHTS) **and** in medical imaging (chest X-rays).
The project does **not** depend on MIMIC-CXR — fashion datasets are the primary vehicle.
MIMIC-CXR is an optional domain generalization case if the download succeeds.

### Dataset Priority

1. **COCO** — standard retrieval baseline; verify reproduction works (preprocessing done)
2. **FashionIQ or Fashion200K** — fine-grained / high-similarity; main extension dataset
3. **CIRR or NIGHTS** — optional, if time allows
4. **MIMIC-CXR** — optional medical generalization; only if PhysioNet access succeeds

### Step 1: Reproduce GENIUS

Reproduce on COCO (standard) + one fine-grained dataset (FashionIQ recommended).
Use original evaluation metrics (Recall@5, Recall@10).

### Step 2: Compare with Original Paper

- Are results close to the paper?
- Which datasets/tasks are harder to reproduce?
- Does reranking matter? Does beam size matter?
- If results differ: briefly analyze (preprocessing, candidate pool size, codebook, training details).

### Step 3: Core Extensions — superseded by Phase 2 mentor directive (2026-06-17)

> **This step is now specified in detail by the mentor and tracked separately in
> [`GENIUS_phase2_pool_composition.md`](GENIUS_phase2_pool_composition.md). Read that
> file for the actual experiment plan. The sketch below is kept only for historical
> context and should not be followed instead of the Phase 2 doc.**
>
> **Phase 2 has NOT started yet.** It begins only after the Phase 1 reproduction
> pipeline (full-union Stage 2 model, currently job 329846 + eval jobs 329954/329955)
> is finished and documented. Do not interleave Phase 2 work with still-running
> Phase 1 training/eval jobs.

<details>
<summary>Original (superseded) sketch — Extension 1 & 2</summary>

**Extension 1 — High-Similarity Stress Test**

Construct harder candidate pools where many candidates are similar to the positive target.

Construction strategies:
- **Metadata-based**: use dataset labels/categories to select same-category candidates (e.g., all dresses, but different colors/styles)
- **Embedding-based**: use frozen encoder to retrieve nearest neighbors as hard candidates

Per query: high-similarity pool = positive target + top-K nearest non-positive neighbors (e.g., top-10 or top-50).

Compare GENIUS under:
- Standard candidate pool (original)
- High-similarity candidate pool (harder)

Goal: does GENIUS distinguish fine-grained differences, or does it rely on coarse semantic differences?

**Extension 2 — Semantic ID Diagnostic Analysis**

Analyze whether generated semantic IDs are discriminative in high-similarity settings:
- Do many candidates share the same full ID?
- Do similar candidates share the same ID prefix?
- Are wrong retrieval results close to the correct target in ID space?

</details>

### Optional Extensions

- **Hard-negative quantization**: use hard negatives during Stage 1 quantizer training so IDs better distinguish similar candidates. Only after Steps 1–3 are stable.
- **MIMIC-CXR medical setting**: if data becomes available — construct 10K study-level dataset, test the same two core extensions.

### Evaluation Metrics

- Recall@K — primary
- MRR@K
- NDCG@K — graded relevance where applicable

### Baselines (already partially implemented)

- BM25 over text (done)
- Dense embedding retrieval with CLIP-SF / BiomedCLIP (done)

### CXR-Specific Design (if MIMIC-CXR becomes available)

Encoder: **BiomedCLIP** (ViT-B/16, 512-dim) — frozen.
- Do NOT use CLIP-SF checkpoint (768-dim, incompatible)
- Change quantizer input dim to 512 directly; no adapter
- Relevance = CheXpert label overlap (≥1 shared positive label, excluding "No Finding")
- Data paths: TBD once PhysioNet access confirmed