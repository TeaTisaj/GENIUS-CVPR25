"""
Generates the short paper's single headline figure: vanilla vs. strong balance
regularization, grouped by dataset, faceted by retrieval direction. Numbers are
transcribed from the paper's main results table (sections/04_finding1_balance_tradeoff.tex),
already verified against source CSVs elsewhere in this project -- no new
computation happens here, only plotting.

Palette: blue (#2a78d6, categorical slot 1) = vanilla, orange (#eb6834, slot 2)
= strong -- validated via the dataviz skill's validate_palette.js (all checks
pass; the previously-used light-gray "vanilla" fill in this repo's older figures
fails the lightness/chroma-floor checks, so it is not reused here). Hatch marks
bars where strong regularization *decreased* Recall relative to vanilla (a
secondary, texture-based encoding of the sign of the effect, redundant with the
direct value labels, per the skill's CVD-safety guidance).

Run with: python make_summary_figure.py  (writes fig_summary_grouped.{pdf,png})
"""
import matplotlib.pyplot as plt
import numpy as np

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"     # vanilla
ORANGE = "#eb6834"   # strong

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK_PRIMARY,
})

# Dir A = text->image (MSCOCO, VisualNews) / (image,text)->image (FashionIQ)
dir_a_datasets = ["MSCOCO", "VisualNews", "FashionIQ"]
dir_a_vanilla = [20.78, 20.07, 0.61]
dir_a_vanilla_std = [0.06, 0.11, 0.10]
dir_a_strong = [23.33, 9.43, 0.25]
dir_a_strong_std = [0.11, 0.08, 0.08]
dir_a_hurts = [False, True, True]  # strong < vanilla?

# Dir B = image->text (MSCOCO, VisualNews only; FashionIQ has no second direction)
dir_b_datasets = ["MSCOCO", "VisualNews"]
dir_b_vanilla = [10.30, 2.88]
dir_b_vanilla_std = [0.67, 0.05]
dir_b_strong = [0.03, 0.01]
dir_b_strong_std = [0.02, 0.00]
dir_b_hurts = [True, True]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [1.4, 1]})

width = 0.32


def plot_panel(ax, datasets, vanilla, vanilla_std, strong, strong_std, hurts, title):
    x = np.arange(len(datasets))
    bars_v = ax.bar(x - width / 2, vanilla, width, yerr=vanilla_std, color=BLUE,
                     edgecolor=INK_PRIMARY, linewidth=0.6, capsize=2.5, label="vanilla")
    bars_s = ax.bar(x + width / 2, strong, width, yerr=strong_std, color=ORANGE,
                     edgecolor=INK_PRIMARY, linewidth=0.6, capsize=2.5, label="strong")
    for bar, hurt in zip(bars_s, hurts):
        if hurt:
            bar.set_hatch("////")
    for bars, vals, stds in ((bars_v, vanilla, vanilla_std), (bars_s, strong, strong_std)):
        for bar, v, s in zip(bars, vals, stds):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v + s),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=6.5, color=INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel("Recall@1 (%)", fontsize=8.5)
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return bars_v, bars_s


plot_panel(ax1, dir_a_datasets, dir_a_vanilla, dir_a_vanilla_std, dir_a_strong,
           dir_a_strong_std, dir_a_hurts, "(a) Dir. A: helps MSCOCO, hurts VisualNews\nand FashionIQ (hatched = strong < vanilla)")
ax1.set_ylim(0, 27)

bars_v2, bars_s2 = plot_panel(ax2, dir_b_datasets, dir_b_vanilla, dir_b_vanilla_std,
                               dir_b_strong, dir_b_strong_std, dir_b_hurts,
                               "(b) Dir. B: collapses under any\nregularization on both datasets")
ax2.set_ylim(0, 12.5)
ax2.legend(handles=[bars_v2, bars_s2], loc="upper right", fontsize=7.5, frameon=False, handlelength=1.5)

fig.tight_layout()
fig.subplots_adjust(bottom=0.2, wspace=0.4)
fig.savefig("fig_summary_grouped.pdf")
fig.savefig("fig_summary_grouped.png", dpi=200)
plt.close(fig)

print("Wrote fig_summary_grouped.{pdf,png}")
