"""
volcano.py - Publication-quality volcano plot module.

Supports customizable thresholds, colors, point styling, label modes (auto/manual/
significant-only) with optional repelling, legend placement/format, axis and title
customization, gridlines, threshold-line styling, protein highlighting, marker
shapes, background, figure-size presets, journal-style themes, multi-format/multi-DPI
export, and data/settings export for reproducibility.

Not implemented here (would require a different architecture, noted for transparency):
  - True interactive hover/click (would need Plotly/Bokeh instead of static matplotlib)
  - HMDB/KEGG/pathway-based labeling or filtering (no compound-database integration
    in this app's data model — labels/filters work by protein identity only)
  - VIP-based filtering (would require a PLS-DA model, not implemented)
  - Freehand in-app annotation (arrows/rectangles/regions) — a fixed set of
    programmatic highlight/annotate options is provided instead
"""

import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
COLORBLIND_PALETTES = {
    "Default (Red/Teal/Gray)": {"up": "#B2182B", "down": "#00A9A5", "ns": "#B0B0B0"},
    "Classic (Red/Blue/Gray)": {"up": "#D62728", "down": "#1F77B4", "ns": "#C7C7C7"},
    "Colorblind-safe (Okabe-Ito)": {"up": "#D55E00", "down": "#0072B2", "ns": "#999999"},
    "Colorblind-safe (Viridis ends)": {"up": "#FDE725", "down": "#440154", "ns": "#AFAFAF"},
    "Monochrome": {"up": "#000000", "down": "#666666", "ns": "#D9D9D9"},
}

MARKER_SHAPES = {"Circle": "o", "Triangle": "^", "Square": "s", "Diamond": "D", "Cross": "x"}
LINE_STYLES = {"Dashed": "--", "Solid": "-", "Dotted": ":", "None": "None"}

FIGURE_SIZE_PRESETS = {
    "Single column (4x4 in)": (4.0, 4.0),
    "Double column (8x6 in)": (8.0, 6.0),
    "Standard (6x5 in)": (6.0, 5.0),
    "Presentation (10x7.5 in)": (10.0, 7.5),
    "Poster (14x10 in)": (14.0, 10.0),
    "Custom": None,
}

# Approximate journal-style presets: font family, grid, spine, and palette choices.
# These are stylistic approximations, not exact reproductions of any journal's
# official figure specification.
THEMES = {
    "Default": {"font": "sans-serif", "grid": "none", "spines_top_right": False, "palette": "Default (Red/Teal/Gray)"},
    "Nature": {"font": "sans-serif", "grid": "none", "spines_top_right": False, "palette": "Classic (Red/Blue/Gray)"},
    "Cell": {"font": "sans-serif", "grid": "none", "spines_top_right": False, "palette": "Default (Red/Teal/Gray)"},
    "Cancer Research": {"font": "serif", "grid": "none", "spines_top_right": False, "palette": "Classic (Red/Blue/Gray)"},
    "Clinical Cancer Research": {"font": "serif", "grid": "none", "spines_top_right": False, "palette": "Classic (Red/Blue/Gray)"},
    "PNAS": {"font": "sans-serif", "grid": "major", "spines_top_right": True, "palette": "Colorblind-safe (Okabe-Ito)"},
}

LEGEND_POSITIONS = {
    "Right": "right", "Left": "left", "Top": "top", "Bottom": "bottom", "Hidden": "none",
}


def _text_width_in(text: str, fontsize: float) -> float:
    tmp_fig = plt.figure()
    renderer = tmp_fig.canvas.get_renderer()
    t = tmp_fig.text(0, 0, text, fontsize=fontsize)
    width_in = t.get_window_extent(renderer=renderer).width / tmp_fig.dpi
    plt.close(tmp_fig)
    return width_in


def _measure_legend_size_in(legend_handles, title, fontsize=9.5, title_fontsize=11.5, ncol=1):
    tmp_fig = plt.figure()
    leg = tmp_fig.legend(handles=legend_handles, loc="center left", fontsize=fontsize,
                         frameon=False, title=title, title_fontsize=title_fontsize,
                         alignment="left", ncol=ncol)
    tmp_fig.canvas.draw()
    renderer = tmp_fig.canvas.get_renderer()
    bbox = leg.get_window_extent(renderer=renderer)
    size_in = (bbox.width / tmp_fig.dpi, bbox.height / tmp_fig.dpi)
    plt.close(tmp_fig)
    return size_in


def classify_direction(df: pd.DataFrame, fc_col: str, sig_col: str, sig_cutoff: float, fc_threshold: float):
    def _classify(row):
        if pd.isna(row[fc_col]) or pd.isna(row[sig_col]):
            return "NS"
        if row[sig_col] < sig_cutoff and row[fc_col] > fc_threshold:
            return "Up"
        if row[sig_col] < sig_cutoff and row[fc_col] < -fc_threshold:
            return "Down"
        return "NS"
    return df.apply(_classify, axis=1)


def volcano_plot(
    stats_table: pd.DataFrame,
    # --- 1. Thresholds ---
    fc_col: str = "Log2FC",
    use_fc_not_log2: bool = False,          # if True, x-axis shows linear FC (requires Linear_FC column)
    fc_threshold: float = 1.0,              # in log2FC units regardless of display axis
    y_metric: str = "pvalue",               # 'pvalue' or 'fdr'
    sig_cutoff: float = 0.05,
    fdr_cutoff: float = None,               # optional secondary FDR cutoff (combined significance)
    threshold_line_style: str = "Dashed",
    # --- 2. Colors ---
    palette: str = "Default (Red/Teal/Gray)",
    up_color: str = None, down_color: str = None, ns_color: str = None,
    alpha: float = 0.75, edge_color: str = "none", edge_width: float = 0.0,
    # --- 3. Point size ---
    point_size: float = 14, point_shape: str = "Circle",
    # --- 4. Labels ---
    label_mode: str = "top_n",              # 'top_n' | 'significant_only' | 'manual' | 'none'
    top_label_n: int = 10,
    manual_labels: list = None,
    label_font_size: float = 7.5, label_color: str = None, label_bold: bool = False,
    label_italic: bool = False, repel_labels: bool = True,
    # --- 5. Legend ---
    legend_position: str = "Right", legend_format: str = "full",  # 'full' -> "Upregulated (n=5)", 'short' -> "Up (5)"
    # --- 6. Axes ---
    x_title: str = None, y_title: str = None, axis_font_size: float = 12, axis_bold: bool = False,
    x_limits: tuple = None, y_limits: tuple = None, x_tick_spacing: float = None, y_decimals: int = None,
    # --- 7. Title ---
    title: str = None, subtitle: str = None, title_font_size: float = 14,
    title_bold: bool = True, title_italic: bool = False, title_align: str = "center", hide_title: bool = False,
    # --- 8. Grid ---
    grid_mode: str = "none",                # 'none' | 'major' | 'both'
    grid_style: str = "Dashed", grid_color: str = "#D9D9D9",
    # --- 9. Threshold line style ---
    threshold_line_color: str = "gray", threshold_line_width: float = 0.7,
    # --- 10. Highlight specific proteins ---
    highlight_names: list = None, highlight_color: str = "#FFD700",
    highlight_size_mult: float = 2.0, highlight_shape: str = "Star",
    # --- 12. Background ---
    background: str = "white",              # 'white' | 'transparent' | 'gray' | hex color
    # --- 13. Figure size ---
    fig_width_in: float = 7.2, fig_height_in: float = 6.4,
    # --- 15. Stats box ---
    show_stats_box: bool = False,
    # --- theme override ---
    theme: str = None,
):
    """
    Build a fully customizable, publication-oriented volcano plot.
    Returns (fig, annotated_df) where annotated_df has 'Direction' and the y-metric
    -log10 column added.
    """
    df = stats_table.copy()
    manual_labels = manual_labels or []
    highlight_names = highlight_names or []

    # Theme overrides (only fill in values the caller left at their defaults)
    if theme and theme in THEMES:
        t = THEMES[theme]
        if palette == "Default (Red/Teal/Gray)":
            palette = t["palette"]
        if grid_mode == "none":
            grid_mode = t["grid"]
        plt.rcParams["font.family"] = t["font"]

    colors = COLORBLIND_PALETTES.get(palette, COLORBLIND_PALETTES["Default (Red/Teal/Gray)"])
    up_c = up_color or colors["up"]
    down_c = down_color or colors["down"]
    ns_c = ns_color or colors["ns"]
    label_c_default = {"Up": up_c, "Down": down_c, "NS": "#666666"}

    sig_col = "p-value" if y_metric == "pvalue" else "FDR"
    y_label_default = f"-log$_{{10}}$({'p-value' if y_metric == 'pvalue' else 'FDR'})"
    x_col = "Linear_FC" if use_fc_not_log2 else fc_col
    x_label_default = "FC" if use_fc_not_log2 else "log$_2$FC"

    df["neglog10"] = -np.log10(df[sig_col].replace(0, np.nextafter(0, 1)))
    df["Direction"] = classify_direction(df, fc_col, sig_col, sig_cutoff, fc_threshold)
    if fdr_cutoff is not None and "FDR" in df.columns:
        df.loc[df["FDR"] >= fdr_cutoff, "Direction"] = "NS"

    counts = {d: int((df["Direction"] == d).sum()) for d in ("Down", "NS", "Up")}
    total = len(df)

    def _legend_label(direction_key, full_name):
        if legend_format == "short":
            short = {"Up": "Up", "Down": "Down", "NS": "NS"}[direction_key]
            return f"{short} ({counts[direction_key]})"
        return f"{full_name} (n={counts[direction_key]})"

    legend_handles = [
        Line2D([0], [0], marker=MARKER_SHAPES.get(point_shape, "o"), linestyle="",
               markerfacecolor=up_c, markeredgecolor=edge_color if edge_width else "none",
               markersize=7, label=_legend_label("Up", "Upregulated")),
        Line2D([0], [0], marker=MARKER_SHAPES.get(point_shape, "o"), linestyle="",
               markerfacecolor=ns_c, markeredgecolor=edge_color if edge_width else "none",
               markersize=7, label=_legend_label("NS", "Not significant")),
        Line2D([0], [0], marker=MARKER_SHAPES.get(point_shape, "o"), linestyle="",
               markerfacecolor=down_c, markeredgecolor=edge_color if edge_width else "none",
               markersize=7, label=_legend_label("Down", "Downregulated")),
    ]
    show_legend = legend_position != "Hidden"
    legend_pos_key = LEGEND_POSITIONS.get(legend_position, "right")

    # -------------------------------------------------------------------
    # Fixed-inch layout, adapting reserved space to whichever side the
    # legend is on (or omitting it entirely when hidden).
    # -------------------------------------------------------------------
    left_margin_in, right_margin_in = 0.75, 0.3
    bottom_margin_in, top_margin_in = 0.6, (0.75 if subtitle else 0.55)
    gap_in = 0.35

    legend_w_in = legend_h_in = 0.0
    if show_legend:
        ncol = 3 if legend_pos_key in ("top", "bottom") else 1
        legend_w_in, legend_h_in = _measure_legend_size_in(legend_handles, "Direction", ncol=ncol)
        legend_w_in += 0.25
        legend_h_in += 0.2

    if legend_pos_key == "right":
        W = left_margin_in + fig_width_in + gap_in + legend_w_in + right_margin_in
        H = top_margin_in + fig_height_in + bottom_margin_in
    elif legend_pos_key == "left":
        W = left_margin_in + legend_w_in + gap_in + fig_width_in + right_margin_in
        H = top_margin_in + fig_height_in + bottom_margin_in
    elif legend_pos_key == "top":
        W = left_margin_in + fig_width_in + right_margin_in
        H = top_margin_in + legend_h_in + gap_in + fig_height_in + bottom_margin_in
    elif legend_pos_key == "bottom":
        W = left_margin_in + fig_width_in + right_margin_in
        H = top_margin_in + fig_height_in + gap_in + legend_h_in + bottom_margin_in
    else:  # none
        W = left_margin_in + fig_width_in + right_margin_in
        H = top_margin_in + fig_height_in + bottom_margin_in

    fig = plt.figure(figsize=(W, H))
    if background == "transparent":
        fig.patch.set_alpha(0.0)
    elif background == "gray":
        fig.patch.set_facecolor("#EBEBEB")
    elif background not in ("white", None):
        fig.patch.set_facecolor(background)

    if legend_pos_key == "left":
        ax_left = (left_margin_in + legend_w_in + gap_in) / W
    else:
        ax_left = left_margin_in / W
    if legend_pos_key == "bottom":
        ax_bottom = (bottom_margin_in + legend_h_in + gap_in) / H
    else:
        ax_bottom = bottom_margin_in / H
    ax = fig.add_axes([ax_left, ax_bottom, fig_width_in / W, fig_height_in / H])
    if background == "gray":
        ax.set_facecolor("#F5F5F5")
    elif background not in ("white", "transparent", None):
        ax.set_facecolor(background)

    marker = MARKER_SHAPES.get(point_shape, "o")
    for direction, color in [("NS", ns_c), ("Down", down_c), ("Up", up_c)]:
        sub = df[df["Direction"] == direction]
        ax.scatter(sub[x_col], sub["neglog10"], c=color, s=point_size, alpha=alpha,
                   marker=marker, edgecolors=edge_color if edge_width else "none",
                   linewidths=edge_width)

    # Highlight specific proteins on top of everything else
    if highlight_names:
        hl = df[df.index.isin(highlight_names)]
        hl_marker = MARKER_SHAPES.get(highlight_shape, "*") if highlight_shape in MARKER_SHAPES else "*"
        ax.scatter(hl[x_col], hl["neglog10"], c=highlight_color, s=point_size * highlight_size_mult,
                   marker=hl_marker, edgecolors="black", linewidths=0.8, zorder=5)

    # Threshold lines
    if threshold_line_style != "None":
        ls = LINE_STYLES.get(threshold_line_style, "--")
        if not use_fc_not_log2:
            ax.axvline(fc_threshold, color=threshold_line_color, linestyle=ls, lw=threshold_line_width)
            ax.axvline(-fc_threshold, color=threshold_line_color, linestyle=ls, lw=threshold_line_width)
        ax.axhline(-np.log10(sig_cutoff), color=threshold_line_color, linestyle=ls, lw=threshold_line_width)

    # Grid
    if grid_mode != "none":
        gs = LINE_STYLES.get(grid_style, "--")
        ax.grid(True, which="major", linestyle=gs, color=grid_color, linewidth=0.6)
        if grid_mode == "both":
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=gs, color=grid_color, linewidth=0.3, alpha=0.5)
    else:
        ax.grid(False)

    # Labels
    texts = []
    if label_mode != "none":
        if label_mode == "manual":
            to_label = df[df.index.isin(manual_labels)]
        elif label_mode == "significant_only":
            to_label = df[df["Direction"] != "NS"]
        else:  # top_n
            to_label = df[df["Direction"] != "NS"].sort_values(sig_col).head(top_label_n)

        fontweight = "bold" if label_bold else "normal"
        fontstyle = "italic" if label_italic else "normal"
        for name, row in to_label.iterrows():
            color = label_color or label_c_default.get(row["Direction"], "#333333")
            texts.append(ax.text(row[x_col], row["neglog10"], str(name), fontsize=label_font_size,
                                  color=color, fontweight=fontweight, fontstyle=fontstyle))

        if repel_labels and texts and HAS_ADJUST_TEXT:
            try:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
            except Exception:
                pass  # fall back to unadjusted labels rather than failing the whole plot

    # Axes
    ax.set_xlabel(x_title or x_label_default, fontsize=axis_font_size,
                  fontweight="bold" if axis_bold else "normal")
    ax.set_ylabel(y_title or y_label_default, fontsize=axis_font_size,
                  fontweight="bold" if axis_bold else "normal")
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)
    if x_tick_spacing:
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.arange(np.floor(xmin / x_tick_spacing) * x_tick_spacing,
                                 np.ceil(xmax / x_tick_spacing) * x_tick_spacing + 1e-9, x_tick_spacing))
    if y_decimals is not None:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(f"%.{y_decimals}f"))

    theme_spines = THEMES.get(theme, {}).get("spines_top_right", False) if theme else False
    if not theme_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Title / subtitle
    if not hide_title:
        final_title = title or f"Volcano Plot ({'p-value' if y_metric == 'pvalue' else 'FDR'} based)"
        ax.set_title(final_title, fontsize=title_font_size, loc=title_align,
                     fontweight="bold" if title_bold else "normal",
                     fontstyle="italic" if title_italic else "normal")
        if subtitle:
            ax.text(0.5 if title_align == "center" else (0.0 if title_align == "left" else 1.0),
                    1.03, subtitle, transform=ax.transAxes, fontsize=title_font_size * 0.7,
                    ha=title_align, color="#555555")

    # Stats box
    if show_stats_box:
        stats_text = (f"Total: {total}\nUp: {counts['Up']}\nDown: {counts['Down']}\n"
                      f"FC cutoff: ±{fc_threshold}\n{'FDR' if fdr_cutoff else sig_col} cutoff: "
                      f"{fdr_cutoff if fdr_cutoff else sig_cutoff}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCCCCC", alpha=0.85))

    # Legend placement
    if show_legend:
        if legend_pos_key == "right":
            legend_x = (left_margin_in + fig_width_in + gap_in) / W
            fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(legend_x, 0.55),
                       bbox_transform=fig.transFigure, fontsize=9.5, frameon=False,
                       title="Direction", title_fontsize=11.5, alignment="left")
        elif legend_pos_key == "left":
            legend_x = left_margin_in / W
            fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(legend_x, 0.55),
                       bbox_transform=fig.transFigure, fontsize=9.5, frameon=False,
                       title="Direction", title_fontsize=11.5, alignment="left")
        elif legend_pos_key == "top":
            legend_y = (bottom_margin_in + fig_height_in + gap_in) / H
            fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, legend_y),
                       bbox_transform=fig.transFigure, fontsize=9.5, frameon=False, ncol=3,
                       title="Direction", title_fontsize=11.5)
        elif legend_pos_key == "bottom":
            legend_y = (bottom_margin_in + legend_h_in) / H
            fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, legend_y),
                       bbox_transform=fig.transFigure, fontsize=9.5, frameon=False, ncol=3,
                       title="Direction", title_fontsize=11.5)

    return fig, df


def get_direction_table(annotated_df: pd.DataFrame, direction: str = None) -> pd.DataFrame:
    """direction: 'Up', 'Down', 'NS', or None for all significant (Up+Down)."""
    if direction is None:
        return annotated_df[annotated_df["Direction"] != "NS"]
    return annotated_df[annotated_df["Direction"] == direction]


def export_settings_json(settings: dict) -> str:
    """Serialize the plot settings dict to a pretty JSON string for reproducibility."""
    return json.dumps(settings, indent=2, default=str)


def top_biomarker_labels(stats_table: pd.DataFrame, sig_col: str = "FDR", n: int = 20):
    return stats_table.sort_values(sig_col).head(n)


def export_figure(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """
    Export a matplotlib figure to bytes. fmt: 'png', 'pdf', 'svg', 'eps', 'jpeg'/'jpg', or 'tiff'.
    dpi applies to raster formats (300/600/1200 supported; ignored for vector pdf/svg/eps).
    """
    fmt = fmt.lower()
    buf = io.BytesIO()
    save_kwargs = {"format": "png" if fmt == "png" else fmt, "bbox_inches": "tight"}
    if fmt in ("jpeg", "jpg", "tiff", "png"):
        save_kwargs["dpi"] = dpi
    if fmt in ("jpeg", "jpg"):
        save_kwargs["facecolor"] = fig.get_facecolor() if fig.get_alpha() != 0 else "white"
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    return buf.read()
