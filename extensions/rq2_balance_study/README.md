# RQ1 + RQ2 balance-regularization study (COCO)

Front door to the code behind the RQ1/RQ2 study in `rq-new-plan.txt` (repo root):
does RQ-ID structure relate to retrieval performance (RQ1), and does balancing
codebook usage help or hurt (RQ2)? Four Stage-1 quantizers were trained on
COCO, identical except for one knob — balance-loss strength
(`balance_loss_config.lambda`): **vanilla** (λ=0, disabled), **weak** (λ=0.3),
**medium** (λ=1.0), **strong** (λ=3.0).

Full narrative, decisions, and verification trail: see the plan doc
`/home/tcetoje/.claude/plans/radili-smo-onu-genius-snuggly-whistle.md`
("Finishing RQ1 + RQ2 on COCO"). Scope is **COCO only** — RQ3 (the FashionIQ
repeat) is a separate, later plan.

## What lives here vs. what stays where the rest of the pipeline keeps it

This folder holds the **study-specific analysis/orchestration code**. The
**config YAMLs stay in the main `configs_scripts/` trees**, alongside every
other sibling config for their stage — moving them would have broken the
generic launchers (`scripts/slurm_train_stage1_param.sh` /
`scripts/slurm_train_stage2_param.sh`, both reusable for *any* future config,
not just this study) and risked the Stage-2 jobs that were still queued on
Slurm when this was written.

| Config | Path |
|---|---|
| Stage-1 train (vanilla/weak/medium/strong) | `src/models/residual_quantization/configs_scripts/large/train/inbatch/inbatch_coco_{vanilla,weak,medium,strong}.yaml` |
| Stage-2 train (vanilla/weak/medium/strong) | `src/models/generative_retriever/configs_scripts/large/train/inbatch/inbatch_coco_only_{vanilla,weak,medium,strong}.yaml` |
| Stage-2 eval (vanilla/weak/medium/strong) | `src/models/generative_retriever/configs_scripts/large/eval/inbatch/config_eval_coco_only_{vanilla,weak,medium,strong}.yaml` |

## This folder's contents

```
analysis/
  rq1_coco_variant_diagnostics.py   # Step A: ID-property + reconstruction diagnostic
                                     # (no Stage-2/GPU needed; runs directly against
                                     #  each variant's frozen Stage-1 RQ checkpoint).
                                     # Also computes neighbor_preservation_at_10 (RQ2
                                     # "semantic fidelity" metric).
  plot_balance_strength_curves.py   # RQ2 "Outputs": balance-strength vs. utilization /
                                     # collision-rate / reconstruction-error curves.
                                     # Pure plotting from Step A's CSVs, no rerun needed.
scripts/
  slurm_rq1_coco_variant_diagnostics.sh   # Slurm wrapper for Step A (--partition=cpu)
  slurm_eval_coco_variant_sweep.sh        # Step C: real trie-beam-search Recall@1/5/10
                                            # eval, looped over (variant, epoch)
  summarize_coco_variant_study.py          # Step D: joins Step A + Step C into
                                            # table1_id_structure_summary.csv (literal
                                            # section-5 Table 1, one row/method) /
                                            # table2_retrieval_performance.csv / table3_
                                            # balance_tradeoff.csv (now also carries
                                            # neighbor_preservation_at_10) + the balance-
                                            # vs-Recall plot (4th RQ2 "Outputs" curve --
                                            # needs Recall, can't live in
                                            # plot_balance_strength_curves.py) + the 3 RQ1
                                            # scatter plots + correlation_table.csv (RQ1
                                            # "Outputs": Pearson r / Spearman rho, N=4,
                                            # descriptive only -- no p-value)
```

Step B (the 4 Stage-2 training runs themselves) uses the generic
`scripts/slurm_train_stage2_param.sh` launcher directly against the configs
table above — no study-specific script needed for training itself.

## Outputs (shared filesystem, not in this repo)

- Stage-1 checkpoints: `checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/` (local) and
  `/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/Coco{Weak,Medium,Strong}/`
- Stage-2 checkpoints: `/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage2/GENIUS_t5small/Large/Instruct/CocoOnly{Vanilla,Weak,Medium,Strong}/`
- Step A tables: `/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis_coco_variants/{tables,figures}/`
  (figures here also include the 3 standalone balance-strength curves from
  `plot_balance_strength_curves.py`: `balance_vs_{utilization,collision,reconstruction}.png`)
- Step C per-variant Recall sweeps: `retrieval_results/coco_epoch_sweep_{vanilla,weak,medium,strong}/`
- Step D final tables/plots: same root as Step A's output (`rq_analysis_coco_variants/`).
  `table1_id_structure_summary.csv` does NOT depend on Step C and already exists; everything
  else there (`table2_retrieval_performance.csv`, `table3_balance_tradeoff.csv`,
  `correlation_table.csv`, `balance_vs_recall.png`, the 3 RQ1 scatter plots) is blocked on
  Step C's Recall numbers and fails fast with a clear `FileNotFoundError` if run early
  (verified deliberately, not just assumed).

## Checkpoint selection rule

All 4 variants use Stage-1 `epoch_140` (the last saved checkpoint) — verified
by re-reading each variant's training log directly; no variant showed
divergence (NaN/inf or >2x regression) in its final 5 saved epochs. Decided
from Stage-1-only signals, before any downstream Recall number existed.
