# Seed-2023 COCO re-eval correction (2026-07-11)

**What happened**: ECIR-paper Round 3 item 4 was meant only to fix a documented R@5/R@10
truncation artifact in the original `table2_retrieval_performance.csv` seed=2023 numbers
(see `project_ecir_paper_strict_audit.md` memory). Job 335179 re-ran eval for all 4
base MSCOCO variants (vanilla/weak/medium/strong) using the **same, unmodified**
checkpoints (`.../genius_checkpoints_stage2/GENIUS_t5small/Large/Instruct/CocoOnly{V}/
genius_t5small_epoch_25.pth`, confirmed untouched since Jun 30) and the **same,
unmodified** eval configs.

**Unexpected result**: R@1 also changed meaningfully, not just R@5/R@10 -- most
dramatically for vanilla (14.08% -> 20.86%). Vanilla's new R@1 now sits almost exactly
inside the tight cluster of its other 4 seeds (20.70/20.73/20.78/20.81%), while the old
14.08% was a clear outlier from that cluster.

**Diagnosis performed before trusting this**: checked whether this was the documented
stale-trie-cache bug (`feedback_trie_cache_staleness.md`) -- it was NOT: the eval log
shows the T->I trie was "Loaded" (reused) rather than rebuilt, but a `.pkl.hash` file
confirms the current candidate codes were hash-verified against the trie before reuse
(i.e. the trie-reuse was legitimate, not stale). Checkpoint file mtime confirmed
unchanged since Jun 30 -- not an accidental overwrite (the FashionIQ-style incident).
Root cause remains **not fully diagnosed** -- most likely genuine run-to-run
non-determinism in constrained beam-search decoding (DDP/cuDNN precision effects,
consistent with this project's already-documented non-bit-reproducibility of training;
apparently this also affects eval-time decoding to a non-trivial degree). This should
be disclosed as an open point in the paper, not hidden.

**Decision (user-confirmed 2026-07-11)**: treat today's re-eval as the more trustworthy
run for vanilla and strong specifically (same checkpoint, freshly-verified candidate
codes+trie, and results now consistent with a tight 5-seed cluster rather than an
outlier) and recompute the significance test with the corrected data.

## Corrected 5-seed values

| Variant | Old seed2023 | New (re-eval) | seed7 | seed13 | seed21 | seed42 |
|---|---|---|---|---|---|---|
| vanilla T->I R@1 | 14.08 | **20.86** | 20.78 | 20.70 | 20.73 | 20.81 |
| strong T->I R@1 | 21.24 | **23.33** | 23.19 | 23.37 | 23.48 | 23.27 |

Welch t-test (5 vs 5, corrected): **t=45.33, p=0.0000004** (effectively zero) --
decisively significant, non-overlapping +-1 std bands (strong low=23.22 vs vanilla
high=20.84). Relative gain: **+12.28%** (2.55 points), smaller in magnitude than the
original single-seed +50.9% or the (now superseded) 3-seed +22.0%, but far more
statistically solid than either.

weak/medium's R@1 also shifted modestly (weak 14.74->13.62, medium 11.15->9.85) but
neither was a cluster outlier before or after -- not flagged as needing the same
scrutiny; their existing 3-seed (2023/7/13) R@1 means are left as originally reported,
only R@5/R@10 are corrected per the original item-4 scope.

Full corrected table: `table2_retrieval_performance_seed2023_corrected.csv` (this
directory). Original (uncorrected) file is untouched at
`/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants/tables/
table2_retrieval_performance.csv` for the historical record.

**This reverses the paper's "no confirmed retrieval upside anywhere" framing for
MSCOCO T->I specifically** -- see paper-ecir/sections/*.tex for the resulting rewrite
and `project_ecir_paper_strict_audit.md` memory for the full narrative history.
