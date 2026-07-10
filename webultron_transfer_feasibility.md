# WebUltron Transfer Feasibility — Reconnaissance Only (2026-07-09)

**Scope of this document**: read-only reconnaissance of https://github.com/smallporridge/WebUltron (cloned to scratchpad, NOT into this repo). No code was run, no dependencies installed, no data downloaded. This is scoping to inform a decision, not a start on implementation — per the plan, a full port was explicitly deprioritized for this internship's remaining week (see `project_yubao_conversation` memory / the fable-model decision earlier today); this document exists so that decision can be revisited later with real information instead of a guess.

## What WebUltron actually is

Official repo for "Ultron: An Ultimate Retriever on Corpus with a Model-based Indexer" (Zhou et al., arXiv:2208.09257). A **unimodal, text-only** generative retrieval system: T5 encoder-decoder, sequence-to-sequence, query in → docid out — architecturally the same generative-retrieval paradigm as GENIUS (query → discrete ID → constrained decode), but text-only (no CLIP-SF, no image modality, no cross-modal task at all).

- **Datasets**: MS MARCO Document Ranking, Natural Questions — not M-BEIR/COCO. Requires downloading these separately plus DocTTTTTQuery-generated pseudo-queries (10 per document) and precomputed document embeddings (DPR or GTR-based).
- **Training curriculum**: three stages — `general_pretrain` → `search_pretrain` → `finetune` — different from GENIUS's two-stage (Stage-1 RQ, Stage-2 T5) design.
- **Docid schemes supported**: `atomic`, `url`, `pq` — PQ is the one relevant here.

## The critical finding: PQ construction is NOT trained jointly with the encoder

This is the single most important fact for deciding whether "transfer the balance-regularization study" is even a coherent idea on this backbone. From `gen_instance/gen_t5_encoded_docid.py::product_quantization_docid`:

```python
pq = nanopq.PQ(M=args.sub_space, Ks=args.cluster_num)
pq.fit(doc_embeddings)              # offline KMeans per sub-space, run ONCE
X_code = pq.encode(doc_embeddings)  # encode to PQ codes, done, never revisited
```

This uses the `nanopq` library — classical product quantization via independent KMeans clustering per sub-space, fit **once, offline, on precomputed frozen document embeddings**, before any T5 training starts. There is no gradient flow through the codebook, no EMA update, no per-step training loss touching the ID assignment, and no joint optimization with an encoder at all (the embeddings themselves come from a separately pretrained DPR/GTR model, not from anything WebUltron trains).

**Consequence**: this project's entire research thread this month — `balance_loss_config`'s `codebook_diversity_loss` (an entropy term added to the *joint* encoder+RQ training loss every step, which the whole RQ2/RQ3/direction-aware-λ study manipulated) — has **no direct analog** in WebUltron as shipped. There is no training loop where a "balance regularization" term could be added, because the ID assignment isn't optimized during training at all; it's a fixed preprocessing step. Transferring the actual research question (does balance/diversity regularization on the ID space help or hurt retrieval, and does it differ by direction) would require **building a new, WebUltron-specific differentiable/trainable PQ layer from scratch** — not adapting an existing loss term, a materially bigger scope than "port the analysis pipeline to a simpler backbone" implies.

## Other scope-relevant facts

- **No modality split to test "direction-aware" anything.** GENIUS's T→I/I→T split comes from M-BEIR's multimodal task structure; WebUltron has no image modality at all, so the specific research question this week's experiment tested (separate λ per target modality) doesn't transpose — a WebUltron-based "direction-aware" study would need a different notion of "direction" (e.g. query-length buckets, or MS MARCO vs. NQ as two "domains" analogous to COCO vs. FashionIQ's broad/fine-grained split) invented from scratch, not reused.
- **Dependencies are old and likely incompatible with this project's environment**: `requirements.txt` pins `pytorch==1.7.1+cu110`, `transformers==4.18.0`, `datasets==1.18.4` — the `genius2` conda env here runs torch 2.6.0+cu124. A WebUltron environment would need to be built separately (new conda env or container), not reuse `genius2`.
- **Data engineering is nontrivial and separate from what's already set up.** MS MARCO Document Ranking + NQ downloads, DocTTTTTQuery pseudo-query generation (or downloading their pre-generated ones), and DPR/GTR document embeddings are all prerequisites before any docid generation or training can start — none of this overlaps with the M-BEIR/COCO pipeline already built in this repo.
- **What WOULD be reusable, conceptually, from this project's existing analysis tooling** (`extensions/rq2_balance_study/`, `extensions/retrieval_aware_id_refinement/`): the *methodology* (per-level entropy/utilization/collision diagnostics via `level_stats`-style functions, the epoch-sweep eval harness pattern, the seed-variance-check discipline, the "sanity-anchor against a dense baseline" habit) — none of the actual code, since WebUltron's docid structure (comma-separated integer codes per sub-space, `<int>+i*256` offset scheme, vocab-size-shifted) and model class (`T5ForPretrain`, HF T5ForConditionalGeneration subclass) are entirely different from GENIUS's `RQ`/`ResidualVQ`/level-token (`<a2><b1177>...`) representation. Every diagnostic script would need to be rewritten against WebUltron's data structures, not just repointed.

## Realistic time estimate (not verified by running anything — a scope estimate only)

Rough, and almost certainly optimistic given no code was executed to validate any of it:

1. Environment setup + dependency resolution (old pinned versions may not install cleanly on current CUDA/driver stack): **0.5–1 day**, real risk of dependency hell given how old the pins are.
2. Data acquisition + preprocessing (MS MARCO/NQ download, pseudo-query generation or download, doc embeddings): **1–2 days**, mostly waiting on downloads/preprocessing scripts, some risk of broken/stale Google Drive links (the repo relies on GDrive for the pre-generated docid/query files).
3. Get baseline PQ pipeline running end-to-end (their existing `train_t5_pipeline.py` + `test_t5.py`) on at least a small scale, to confirm the un-modified repo works at all: **1 day**, assuming no major breakage.
4. Design + implement a trainable/differentiable PQ layer with a balance-regularization hook (the actual research contribution, not present in the base repo): **2–4 days**, genuinely uncertain — this is new method design, not a port.
5. Rerun any version of the RQ1-style diagnostics + epoch-sweep eval on the new backbone: **1 day**, once (4) exists.

**Total: 5.5–9 days of uninterrupted, everything-goes-right work**, before accounting for the kind of debugging/cluster-contention friction this project has hit all week (a node crash and a scheduling saga cost real hours today alone, on a codebase this session already knows intimately — WebUltron's unknowns are all still ahead). This matches and reinforces fable's earlier verdict: not viable within the days remaining in this internship. Worth pursuing as a genuine post-internship or future-collaboration direction, with the important caveat now on record that it requires building a new trainable-PQ mechanism, not just re-plumbing the existing balance loss onto a different codebase.
