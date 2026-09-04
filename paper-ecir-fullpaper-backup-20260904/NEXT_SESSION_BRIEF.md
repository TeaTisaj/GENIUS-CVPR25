# Brief for next session — ECIR paper, short-paper pivot

Read this first if you're picking up the ECIR balance-regularization paper cold.

## Where things are

- `paper-ecir/main.pdf` — current compiled draft, **16 pages**, named title page
  (Tea Cetojevic Tisaj / IRLab Internship, University of Amsterdam / supervised by
  Maarten de Rijke & Yubao Tang / April–July 2026). Already sent to Yubao.
- `paper-ecir/main.tex` + `paper-ecir/sections/*.tex` (9 files, `00_abstract.tex`
  through `08_discussion_limitations.tex`) — LaTeX source for the above.
- `paper-ecir/genius_balance_reg_paper.zip` — Overleaf-ready bundle (clean
  `main.tex` with the internal review-only comment stripped out, plus
  `references.bib`, `llncs.cls`, `splncs04.bst`, `sections/`, figures). Already
  uploaded to an Overleaf project shared with Yubao, so she can edit directly.
- `yubao_maarten_fullpaper_pivot_draft_20260710.md` (repo root) — an email drafted
  to ask Yubao to choose main track vs. Reproducibility track. **Superseded** — she
  didn't pick either; see her actual reply below. Keep this file only for its
  verified-numbers appendix (tokenizer-reseeding results etc.), not as a live plan.

## Where the paper stands

Full-paper draft, 16 pages, compiles clean (0 errors, 0 undefined refs). Current
headline numbers: MSCOCO T→I gain under strong balance regularization, verified
across 3 independent RQ tokenizers (not just decoder seeds); I→T collapses under
any regularization strength; VisualNews reverses the T→I sign (regularization
*hurts* there); FashionIQ is hurt by any regularization; a direction-aware λ
(separate λ per modality) doesn't rescue either direction. Full multi-round audit
history (statistical fixes, decode-crash root-cause, page-budget attempts) is in
the auto-memory system — see `project-ecir-paper-strict-audit` — not repeated here.

## Yubao's reply (received 2026-07-23), verbatim

> Hi Tea,
>
> I went through the paper and I think you have done a very solid job.
>
> If you would like to consider submitting part of this work, one possible option
> is the ECIR short paper track. Rather than framing it as a reproduction study of
> GENIUS, I think you could turn it into a focused empirical study of codebook
> balance in RQ-based generative multimodal retrieval. The introduction could start
> directly from the general question of whether more balanced semantic IDs
> actually lead to better retrieval, while GENIUS could be introduced later in the
> experimental setup as the architecture/testbed you build on.
>
> For a short paper, you probably would not need to include everything in the
> current report. The clearest story could focus on:
> (1) GenMR --> codebook balance does not consistently improve retrieval;
> (2) its effect differs substantially across retrieval directions; and
> (3) the degradation can already be observed at the tokenizer level and does not
> consistently transfer across datasets.
>
> One concern I have is that the absolute retrieval performance in your current
> setup is lower than that reported in some existing work. A reviewer might
> therefore question whether the observed effects generalize beyond your current
> setup. Since you have already spent substantial effort trying to close this gap,
> I would not suggest spending more time on reproducing the original performance.
> If you pursue the short-paper direction, I think the more practical option would
> be to acknowledge this limitation, frame the work as a controlled empirical
> study, and keep the claims appropriately scoped.

## What she's asking for — action list

1. **Reframe the intro.** Drop "we reproduced GENIUS" as the framing. Open with
   the general question (does more balanced semantic-ID usage improve retrieval?),
   introduce GENIUS later, in the experimental-setup section, as the testbed.
2. **Cut scope to her 3 findings.** Everything currently in the draft (RQ1
   correlation/hourglass analysis, tie-break sensitivity, the tokenizer-reseeding
   verification saga) is supporting evidence for these 3, not separate headline
   claims — decide what's cut entirely vs. compressed vs. pushed to a footnote:
   - (1) codebook balance does not consistently improve retrieval
   - (2) the effect differs substantially by retrieval direction (T→I vs I→T)
   - (3) the degradation is visible at the tokenizer level and doesn't
     consistently transfer across datasets
3. **Target length: short paper, 6 pages LNCS** (per the venue note already in
   `main.tex`'s own history — this was the paper's original scope before the
   full-paper pivot). This is a real cut from 16 pages, not a trim pass.
4. **Add one clear limitation statement** on absolute Recall vs. published
   numbers, framed as a controlled-empirical-study scoping choice. Do **not**
   spend more time trying to close that gap — she explicitly said not to.

## Explicitly not yet decided

- Which specific sections/paragraphs get cut vs. compressed vs. moved to a
  footnote or brief mention.
- Whether the retract/reinstate tokenizer-seed saga (§8 in the current draft)
  stays in as a short methodological-rigor note, or is dropped entirely for space.
- Whether to reply to Yubao confirming the short-paper direction before or after
  a restructured draft exists.

Don't assume answers to the above — ask Tea when picking this up.
