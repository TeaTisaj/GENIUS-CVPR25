"""
Generates the paper's single combined figure for the direction-aware-lambda
section from the numbers already reported in rq1-rq3_final_tables.md (unique
level-1 RQ prefix counts + T->I Recall@1). Static, print-safe (single hue
family + hatch texture for grayscale/CVD safety, one axis per panel, direct
labels), matching this repo's dataviz-skill conventions for an academic PDF
figure. Combined into one 2-panel figure (rather than two separate floats)
to fit the ECIR short-paper page budget.

Run with: python make_figures.py  (writes fig_prefixes.{pdf,png})
"""
import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
GRAY_FILL = "#d8d7d1"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK_PRIMARY,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.15), gridspec_kw={"width_ratios": [1.4, 1]})

# --- Panel (a): image-side unique level-1 prefixes vs T->I Recall@1 ---
variants = ["vanilla", "strong", "img3txt0", "img3txt0p3"]
prefixes = [1249, 1230, 603, 391]
recall = [20.78, 23.33, 7.87, 5.56]  # seed-mean: vanilla/strong 5 seeds, img3txt0/img3txt0p3 3 seeds (Table 4)
colors = [GRAY_FILL, BLUE, AQUA, AQUA]
hatches = [None, None, None, "////"]

bars1 = ax1.bar(variants, prefixes, color=colors, edgecolor=INK_PRIMARY, linewidth=0.6, width=0.62)
for bar, hatch in zip(bars1, hatches):
    if hatch:
        bar.set_hatch(hatch)
for bar, p, r in zip(bars1, prefixes, recall):
    ax1.annotate(f"{p}\nR@1={r:.1f}%", (bar.get_x() + bar.get_width() / 2, p), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=7,
                 color=INK_PRIMARY, linespacing=1.6)
ax1.set_ylabel("Unique level-1 prefixes\n(image cands., N=5,000)", fontsize=8.5)
ax1.set_ylim(0, 1650)
ax1.set_title("(a) Image-side codes track T→I Recall", fontsize=8.5)
ax1.tick_params(axis="x", labelsize=8)
ax1.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
ax1.set_axisbelow(True)
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)

# --- Panel (b): text-side unique level-1 prefixes, vanilla vs img3txt0 (both lambda_txt=0.0) ---
tvariants = ["vanilla", "img3txt0"]
tprefixes = [946, 566]
tcolors = [GRAY_FILL, AQUA]

bars2 = ax2.bar(tvariants, tprefixes, color=tcolors, edgecolor=INK_PRIMARY, linewidth=0.6, width=0.55)
for bar, p in zip(bars2, tprefixes):
    ax2.annotate(f"{p}", (bar.get_x() + bar.get_width() / 2, p), xytext=(0, 2),
                 textcoords="offset points", ha="center", fontsize=8, color=INK_PRIMARY)
ax2.set_ylabel("Unique level-1 prefixes\n(text cands., N=24,809)", fontsize=8.5)
ax2.set_ylim(0, 1050)
ax2.set_title("(b) Text codes restructured,\n" + r"$\lambda_{\mathrm{txt}}=0.0$ for both", fontsize=8.5)
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

fig.tight_layout()
fig.subplots_adjust(bottom=0.18, wspace=0.45)
fig.savefig("fig_prefixes.pdf")
fig.savefig("fig_prefixes.png", dpi=200)
plt.close(fig)

print("Wrote fig_prefixes.{pdf,png}")

# =====================================================================
# Second figure: Recall@1 vs balance strength lambda, MSCOCO (T->I and
# I->T) and FashionIQ (IT->I). Numbers transcribed from
# table_seed_variance.csv (MSCOCO: rq_analysis_coco_variants/tables/,
# FashionIQ: rq_analysis_fashioniq_variants/tables/) and the per-seed
# I->T TSVs under retrieval_results/coco_seed_variance/ -- same
# hand-verified-against-CSV convention as the panel above, not live
# CSV I/O, to match this script's existing style. MSCOCO vanilla/strong
# were updated 2026-07-11 twice: first to the 5-seed (2023/7/13/21/42)
# values from the extra-seed round, then again the same day once a
# re-eval of the seed-2023 point (originally chased only to fix an R@5/
# R@10 truncation artifact) revealed seed-2023's R@1 was ALSO anomalous
# for vanilla (14.08 -> 20.86, now consistent with its other 4 seeds).
# Corrected 5-seed comparison: p<1e-6, non-overlapping bands (see paper
# Section 5, Table 2, and SEED2023_CORRECTION_NOTE.md).
# =====================================================================

lambdas = [0.0, 0.3, 1.0, 3.0]

mscoco_t2i_mean = [20.78, 14.13, 10.36, 23.33]
mscoco_t2i_std = [0.06, 0.53, 0.69, 0.11]
mscoco_i2t_mean = [10.30, 0.05, 0.03, 0.03]
mscoco_i2t_std = [0.67, 0.02, 0.01, 0.02]

fashioniq_mean = [0.61, 0.27, 0.21, 0.25]
fashioniq_std = [0.10, 0.06, 0.09, 0.08]

fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(6.3, 1.75), gridspec_kw={"width_ratios": [1.3, 1]})

bx1.errorbar(lambdas, mscoco_t2i_mean, yerr=mscoco_t2i_std, color=BLUE, marker="o",
             markersize=5, linewidth=1.6, capsize=2.5, label=r"T$\to$I")
bx1.errorbar(lambdas, mscoco_i2t_mean, yerr=mscoco_i2t_std, color=AQUA, marker="s",
             markersize=5, linewidth=1.6, linestyle="--", capsize=2.5, label=r"I$\to$T")
bx1.set_xlabel(r"Balance strength $\lambda$", fontsize=8.5)
bx1.set_ylabel("Recall@1 (%)", fontsize=8.5)
bx1.set_title("(a) MSCOCO: T→I non-monotonic,\nI→T collapses", fontsize=8.5)
bx1.set_xticks(lambdas)
bx1.tick_params(axis="both", labelsize=8)
bx1.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
bx1.set_axisbelow(True)
bx1.legend(loc="center right", fontsize=7.5, frameon=False, handlelength=1.8)
for spine in ("top", "right"):
    bx1.spines[spine].set_visible(False)

bx2.errorbar(lambdas, fashioniq_mean, yerr=fashioniq_std, color=INK_SECONDARY, marker="D",
             markersize=5, linewidth=1.6, capsize=2.5)
bx2.set_xlabel(r"Balance strength $\lambda$", fontsize=8.5)
bx2.set_ylabel("Recall@1 (%)", fontsize=8.5)
bx2.set_title("(b) FashionIQ: vanilla wins,\nany regularization hurts", fontsize=8.5)
bx2.set_xticks(lambdas)
bx2.tick_params(axis="both", labelsize=8)
bx2.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
bx2.set_axisbelow(True)
for spine in ("top", "right"):
    bx2.spines[spine].set_visible(False)

fig2.tight_layout()
fig2.subplots_adjust(bottom=0.26, wspace=0.4)
fig2.savefig("fig_lambda_recall.pdf")
fig2.savefig("fig_lambda_recall.png", dpi=200)
plt.close(fig2)

print("Wrote fig_lambda_recall.{pdf,png}")
