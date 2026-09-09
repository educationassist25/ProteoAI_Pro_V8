"""
boxplot_module.py - "Boxplot of Proteins" module.

Lets users compare one or more proteins across any chosen subset of sample
groups, with statistics (p-value/FDR) computed on log2-transformed data (never
raw protein intensities, per this app's statistical policy) and displayed on each panel.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from modules import stats_analysis

matplotlib.use("Agg")

GROUP_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def compute_stats_for_proteins(log2_df: pd.DataFrame, meta: pd.DataFrame,
                                   proteins: list, groups: list, group_col: str = "Group") -> pd.DataFrame:
    """
    Compute p-value/FDR for the given proteins across the given groups (from
    `group_col` -- any categorical metadata column), using log2-transformed data.
    If exactly 2 groups are given, runs a Welch's t-test; if 3+, runs one-way
    ANOVA. FDR is corrected across only the selected protein subset shown here
    (not the full feature panel) — noted in the UI to avoid confusion with a
    panel-wide FDR from the Statistics tab.
    """
    sub = log2_df.loc[log2_df.index.intersection(proteins)]
    sample_cols = [c for c in sub.columns if meta.loc[c, group_col] in groups]
    sub = sub[sample_cols]

    if len(groups) == 2:
        a_samples = [c for c in sample_cols if meta.loc[c, group_col] == groups[0]]
        b_samples = [c for c in sample_cols if meta.loc[c, group_col] == groups[1]]
        result = stats_analysis.two_group_test(sub, a_samples, b_samples, method="ttest")
        return result
    else:
        group_map = meta.loc[sample_cols, group_col]
        anova_table, _ = stats_analysis.anova_test(sub, group_map, posthoc="tukey")
        return anova_table.rename(columns={"ANOVA p-value": "p-value"})


def boxplot_proteins(
    log2_df: pd.DataFrame, meta: pd.DataFrame, proteins: list, groups: list,
    stats_table: pd.DataFrame = None,
    fig_width_in: float = None, fig_height_in: float = None,
    font_size: float = 10, font_family: str = "sans-serif",
    group_colors: dict = None, show_points: bool = True,
    show_mean: bool = False, show_median: bool = True,
    ncols: int = None, group_col: str = "Group",
):
    """
    One panel per protein, boxes grouped by the selected sample groups (from
    `group_col` -- any categorical metadata column: Group/Diagnosis, Gender,
    Treatment, Ethnicity, etc.). Returns the matplotlib figure.
    """
    n = len(proteins)
    ncols = ncols or min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig_w = fig_width_in or max(4.0, 3.4 * ncols)
    fig_h = fig_height_in or max(3.5, 3.2 * nrows)

    group_colors = group_colors or {g: GROUP_PALETTE[i % len(GROUP_PALETTE)] for i, g in enumerate(groups)}

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    rng = np.random.default_rng(0)

    for idx, met in enumerate(proteins):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        data_per_group = []
        for g in groups:
            cols = [s for s in log2_df.columns if meta.loc[s, group_col] == g]
            vals = log2_df.loc[met, cols].dropna().values if met in log2_df.index else np.array([])
            data_per_group.append(vals)

        bp = ax.boxplot(data_per_group, tick_labels=groups, patch_artist=True,
                         showmeans=show_mean, meanline=show_mean, showfliers=not show_points)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(group_colors.get(g, "#4C72B0"))
            patch.set_alpha(0.7)
        if show_median:
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(1.2)
        else:
            for line in bp["medians"]:
                line.set_visible(False)

        if show_points:
            for i, gdata in enumerate(data_per_group):
                if len(gdata):
                    x = rng.normal(i + 1, 0.045, size=len(gdata))
                    ax.scatter(x, gdata, color="black", alpha=0.55, s=14, zorder=3, edgecolors="none")

        ax.set_title(met, fontsize=font_size + 1, fontfamily=font_family)
        ax.set_ylabel("Log2 Abundance", fontsize=font_size, fontfamily=font_family)
        ax.tick_params(labelsize=max(6, font_size - 1))
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(font_family)

        # Long/many group names crowd and overlap as straight horizontal labels --
        # angle them (and right-align, so the angled label still points at its tick)
        # once they'd plausibly collide, scaled to how bad the crowding actually is.
        max_label_len = max((len(str(g)) for g in groups), default=0)
        avg_px_per_label = fig_w * 72 / max(len(groups), 1)  # rough inches->points budget per tick
        if max_label_len * (font_size - 1) * 0.6 > avg_px_per_label or max_label_len > 12:
            rotation = 30 if max_label_len <= 18 else 45
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(rotation)
                lbl.set_ha("right")
                lbl.set_rotation_mode("anchor")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if stats_table is not None and met in stats_table.index:
            row = stats_table.loc[met]
            p = row.get("p-value", np.nan)
            fdr = row.get("FDR", np.nan)
            txt = f"p = {p:.3g}" + (f"\nFDR = {fdr:.3g}" if not pd.isna(fdr) else "")

            # Reserve headroom ABOVE the tallest data point (box, whiskers, fliers, and
            # scatter points all included) so the annotation never overlaps plot content,
            # regardless of where the data happens to sit in the axes.
            all_vals = np.concatenate([g for g in data_per_group if len(g)]) if any(len(g) for g in data_per_group) else np.array([0.0, 1.0])
            data_min, data_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
            data_range = (data_max - data_min) or 1.0
            ax.set_ylim(data_min - 0.05 * data_range, data_max + 0.28 * data_range)

            # Blended transform: x in axes-fraction (horizontally centered), y in data
            # coordinates (placed just above the actual tallest point) -- this keeps the
            # label centered regardless of box width while guaranteeing it sits in the
            # headroom just reserved, never over a box or point.
            blended = ax.get_yaxis_transform()
            ax.text(0.5, data_max + 0.06 * data_range, txt, transform=blended, ha="center", va="bottom",
                    fontsize=max(6, font_size - 2), fontfamily=font_family,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))

    # Hide any unused subplot cells
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    return fig


def export_figure(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """Export a matplotlib figure to bytes. fmt: 'png', 'pdf', 'svg', 'jpeg'/'jpg', or 'tiff'."""
    fmt = fmt.lower()
    buf = io.BytesIO()
    save_kwargs = {"format": "png" if fmt == "png" else fmt, "bbox_inches": "tight"}
    if fmt in ("jpeg", "jpg", "tiff", "png"):
        save_kwargs["dpi"] = dpi
    if fmt in ("jpeg", "jpg"):
        save_kwargs["facecolor"] = "white"
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    return buf.read()
