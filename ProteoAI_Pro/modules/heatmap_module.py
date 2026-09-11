"""
heatmap_module.py - Publication-quality clustered heatmap of significant proteins.

Features:
  - Independent row/column clustering toggle (rows only / columns only / both / none)
  - Group annotation bar with legend
  - Predefined and custom color palettes, reversible, with adjustable Z-score range
  - Optional custom color breakpoints (discrete color bands)
  - Horizontal colorbar below the heatmap, labeled with Z-score / normalized intensity
  - Multi-format, high-resolution export (PNG, PDF, SVG, JPEG, TIFF)
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

matplotlib.use("Agg")

PREDEFINED_PALETTES = [
    "RdBu_r", "coolwarm", "seismic", "viridis", "plasma", "magma",
    "cividis", "PiYG", "PRGn", "Spectral", "bwr",
]

GROUP_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]

# Safety limits: without these, a large feature/sample count produces a figure so
# tall/wide that on-screen rendering becomes unusable and high-DPI export can crash
# (matplotlib has a hard ~65536px-per-dimension limit; a 500-row heatmap at the old
# unbounded 0.22in/row scaling reaches 110+ inches, which is 66,000+ px at 600 DPI —
# confirmed this crashes/OOMs during export).
MAX_HEATMAP_HEIGHT_IN = 30.0
MAX_HEATMAP_WIDTH_IN = 24.0
HIDE_ROW_LABELS_ABOVE = 150   # beyond this many features, per-row labels are unreadable anyway
HIDE_COL_LABELS_ABOVE = 120
MIN_ROW_LABEL_FONTSIZE = 4.0


def _zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def _text_width_in(text: str, fontsize: float) -> float:
    """Measure the actual rendered width (inches) of a text string at a given fontsize,
    using a throwaway Agg canvas. Used to size margins dynamically so long labels never
    collide with the legend or get clipped."""
    tmp_fig = plt.figure()
    renderer = tmp_fig.canvas.get_renderer()
    t = tmp_fig.text(0, 0, text, fontsize=fontsize)
    bbox = t.get_window_extent(renderer=renderer)
    width_in = bbox.width / tmp_fig.dpi
    plt.close(tmp_fig)
    return width_in


from modules.utils import build_colormap  # re-exported for backward compatibility


ANNOTATION_MISSING_LABEL = "(missing)"
ROW_ANNOTATION_UNMATCHED_LABEL = "(unannotated)"
CONTINUOUS_ANNOT_CMAPS = ["Purples", "Blues", "Greens", "Oranges", "Greys", "RdPu", "YlGnBu"]


def classify_annotation_series(series: pd.Series, max_unique: int = 15) -> str:
    """
    'continuous' if numeric with more distinct values than plausibly categorical
    (e.g. Age, BMI, Body Weight), else 'categorical' (text labels, or numeric codes
    with few distinct values). Mirrors utils.get_categorical_metadata_columns's
    classification logic, but returns a per-column verdict rather than a filtered list.
    """
    s = series.dropna()
    if len(s) == 0:
        return "categorical"
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > max_unique:
        return "continuous"
    return "categorical"


def count_row_annotation_matches(row_meta: pd.DataFrame, feature_index) -> int:
    """
    How many of `feature_index` (e.g. a log2 matrix's row index) can be found in
    `row_meta.index` -- exact match, or (for multi-dataset combined features named
    'Dataset::Feature') matching just the part after '::'. Mirrors the matching
    logic _annotation_track_values actually uses, so the UI's "X/Y matched" count
    reflects reality rather than only counting exact matches.
    """
    n = 0
    for item in feature_index:
        if item in row_meta.index:
            n += 1
        elif "::" in str(item) and str(item).split("::", 1)[1] in row_meta.index:
            n += 1
    return n


def _annotation_track_values(lookup_df: pd.DataFrame, col: str, index_order, unmatched_label: str):
    """Ordered values of `col` for each item in `index_order`, pulling from
    `lookup_df` (sample metadata or feature/row annotation table). Tries an exact
    match first; if that fails and the item looks like a multi-dataset combined
    feature name ('Dataset::Feature', from Tab 6's combine step), retries matching
    just the part after '::' -- so a plain row-annotation file (feature names with
    no dataset prefix) still matches correctly against combined multi-dataset data,
    without requiring the annotation file to be re-authored per dataset. Items
    missing from `lookup_df` entirely (even after that fallback) get
    `unmatched_label`; items present but with a NaN value get `unmatched_label`
    too -- both render as a neutral 'no data' color rather than crashing or
    silently mis-plotting."""
    if col not in lookup_df.columns:
        return [unmatched_label] * len(index_order)
    values = []
    for item in index_order:
        if item in lookup_df.index:
            v = lookup_df.loc[item, col]
        elif "::" in str(item) and str(item).split("::", 1)[1] in lookup_df.index:
            v = lookup_df.loc[str(item).split("::", 1)[1], col]
        else:
            v = None
        if v is None or pd.isna(v):
            values.append(unmatched_label)
        else:
            values.append(v)
    return values


def _build_annotation_track(values, kind: str, cmap_name: str = None, palette_offset: int = 0,
                             missing_label: str = ANNOTATION_MISSING_LABEL, custom_colors: dict = None):
    """
    Turn one annotation track's raw values into (rgba_array, legend_info).
    kind='continuous': legend_info = {'kind': 'continuous', 'cmap': cmap, 'vmin': .., 'vmax': ..}
    kind='categorical': legend_info = {'kind': 'categorical', 'colors': {value: hexcolor}}
    Missing/unmatched values always render as a fixed neutral gray, in both kinds.
    custom_colors: user overrides for this track -- for categorical, {value: hexcolor}
        (only the given values are overridden, everything else still gets an
        auto-assigned palette color); for continuous, a single colormap name string
        used in place of the auto-cycled default.
    """
    if kind == "continuous":
        arr = np.array([np.nan if (v is None or v == missing_label or
                                    (isinstance(v, float) and np.isnan(v))) else float(v) for v in values])
        finite = arr[~np.isnan(arr)]
        vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        if vmin == vmax:
            vmax = vmin + 1.0
        effective_cmap_name = custom_colors if isinstance(custom_colors, str) else (
            cmap_name or CONTINUOUS_ANNOT_CMAPS[palette_offset % len(CONTINUOUS_ANNOT_CMAPS)]
        )
        cmap = matplotlib.colormaps.get_cmap(effective_cmap_name)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        rgba = np.array([cmap(norm(v)) if not np.isnan(v) else (0.82, 0.82, 0.82, 1.0) for v in arr])
        return rgba, {"kind": "continuous", "cmap": cmap, "vmin": vmin, "vmax": vmax}
    else:
        str_vals = [str(v) if v != missing_label and pd.notna(v) else missing_label for v in values]
        uniq = [v for v in dict.fromkeys(str_vals) if v != missing_label]
        overrides = custom_colors if isinstance(custom_colors, dict) else {}
        colors = {v: overrides.get(v, GROUP_PALETTE[(i + palette_offset) % len(GROUP_PALETTE)])
                  for i, v in enumerate(uniq)}
        if missing_label in str_vals:
            colors[missing_label] = overrides.get(missing_label, "#CCCCCC")
        rgba = np.array([matplotlib.colors.to_rgba(colors[v]) for v in str_vals])
        return rgba, {"kind": "categorical", "colors": colors}


def _estimate_legend_block_height_in(legend_info: dict, font_size: float = 10.0) -> float:
    """Rough height budget (inches) for one legend block, for stacking without overlap."""
    scale = font_size / 10.0
    if legend_info["kind"] == "continuous":
        return 0.62 * scale
    return (0.20 + len(legend_info["colors"]) * 0.155) * scale


def _render_legend_block(fig, legend_info: dict, title: str, x: float, y_top: float, W: float, H: float,
                          font_size: float = 10.0, font_family: str = "sans-serif"):
    """Draw one legend block (categorical swatches, or a small continuous colorbar)
    with its top-left anchored at figure-fraction (x, y_top). Returns the height
    consumed (inches), so the caller can stack the next block below it."""
    if legend_info["kind"] == "categorical":
        handles = [Patch(facecolor=c, label=v) for v, c in legend_info["colors"].items()]
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y_top),
                          bbox_transform=fig.transFigure, fontsize=font_size * 0.65, frameon=False,
                          title=title, title_fontsize=font_size * 0.7)
        for text in leg.get_texts():
            text.set_fontfamily(font_family)
        leg.get_title().set_fontfamily(font_family)
        return _estimate_legend_block_height_in(legend_info, font_size)
    else:
        h_in = _estimate_legend_block_height_in(legend_info, font_size)
        cbar_w_in, cbar_h_in = 0.9, 0.13
        cbar_left = x
        cbar_bottom = y_top - (0.20 + cbar_h_in) / H
        ax = fig.add_axes([cbar_left, cbar_bottom, cbar_w_in / W, cbar_h_in / H])
        sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=legend_info["vmin"], vmax=legend_info["vmax"]),
            cmap=legend_info["cmap"]
        )
        cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
        cb.ax.tick_params(labelsize=font_size * 0.55)
        for lbl in cb.ax.get_xticklabels():
            lbl.set_fontfamily(font_family)
        fig.text(x, y_top, title, fontsize=font_size * 0.7, va="top", ha="left",
                 fontfamily=font_family, transform=fig.transFigure)
        return h_in


def clustered_heatmap(data_log2: pd.DataFrame, significant_features, meta: pd.DataFrame = None,
                       col_annot_cols=None,
                       cluster_rows: bool = True, cluster_cols: bool = True,
                       distance: str = "euclidean", linkage_method: str = "ward",
                       cmap_name: str = "RdBu_r", custom_colors=None, reverse_cmap: bool = False,
                       vmin: float = -2.5, vmax: float = 2.5, breakpoints=None,
                       heatmap_width_in: float = None, heatmap_height_in: float = None,
                       row_meta: pd.DataFrame = None, row_annot_cols=None,
                       annotation_colors: dict = None, font_family: str = "sans-serif",
                       font_size: float = 10.0, title_suffix: str = None):
    """
    data_log2: features x samples (log2 scale), biological samples only.
    significant_features: list/index of features to plot (rows).
    cluster_rows / cluster_cols: independently toggle hierarchical clustering per axis.
        Both False -> no clustering at all (original row/column order preserved).
    distance: 'euclidean' or 'correlation' (Pearson).
    linkage_method: 'ward', 'average', 'complete', etc. ('ward' requires euclidean distance).
    cmap_name: one of PREDEFINED_PALETTES, ignored if custom_colors is given.
    custom_colors: optional list of hex colors to build a custom gradient (overrides cmap_name).
    reverse_cmap: reverse whichever colormap is selected.
    vmin/vmax: Z-score color range (color scale limits).
    breakpoints: optional sorted list of numeric boundaries for discrete color banding
                 (uses BoundaryNorm instead of continuous Normalize).
    meta: sample metadata (indexed by sample). Used with `col_annot_cols` for the
          COLUMN annotation bars + legends. If None or col_annot_cols empty, no
          column annotation is drawn.
    col_annot_cols: list of metadata column names to render as stacked column
          annotation tracks (one per name, top to bottom, directly above the
          heatmap) -- e.g. ['Diagnosis', 'Gender', 'Age']. Each track is
          auto-classified continuous (numeric, many distinct values -- rendered as
          a color gradient with its own colorbar legend) or categorical (rendered
          as discrete color blocks with a swatch legend), same logic as
          utils.get_categorical_metadata_columns uses to decide what's offered as
          a *groupable* variable elsewhere in the app (continuous columns are
          plottable here as an annotation gradient even though they're not offered
          as a grouping/statistics variable there).
    row_meta: optional protein/feature annotation table (indexed by feature
          name, matching data_log2's row index).
    row_annot_cols: list of `row_meta` column names to render as stacked row
          annotation tracks (one per name, left to right, immediately left of the
          heatmap) -- e.g. ['Pathway', 'Method']. Same continuous/categorical
          auto-classification as column tracks. Features missing from row_meta (or
          row_meta not given) render as "(unannotated)" rather than raising --
          e.g. a row-annotation file that doesn't cover every protein, or a
          multi-dataset combined matrix whose 'Dataset::Feature' prefix doesn't
          literally match the annotation file's plain names.
    heatmap_width_in / heatmap_height_in: optional explicit size (inches) for the heatmap
          panel itself, overriding the automatic size-from-row/column-count default. All
          other elements (dendrograms, annotation bars, legends, colorbar, tick labels)
          are positioned relative to this and rescale automatically — nothing needs
          separate adjustment when you resize.
    annotation_colors: optional per-track color overrides, keyed by column name (the
          same strings passed in col_annot_cols/row_annot_cols). For a categorical
          track, the value is {category_value: hexcolor} (only the given values are
          overridden; anything else in that track still gets an auto-assigned
          palette color). For a continuous track, the value is a single colormap
          name string (e.g. 'Purples') used instead of the auto-cycled default.
    font_family / font_size: base font applied to the title, tick labels, legend
          text, colorbar label, and annotation track labels -- everything scales
          proportionally from this one size (e.g. the title is font_size+1, legend
          entries are font_size*0.65), so changing font_size doesn't require
          separately adjusting each element.
    title_suffix: optional string appended to the title in parentheses, e.g.
          '41 features, FDR ≤ 0.25' -> "Heatmap of Proteins (41 features, FDR ≤ 0.25)".
          If not given, the title is just "Heatmap of Proteins".

    Returns (fig, z_ordered, notes).
    """
    col_annot_cols = [c for c in (col_annot_cols or []) if meta is not None and c in meta.columns]
    row_annot_cols = [c for c in (row_annot_cols or []) if row_meta is not None and c in row_meta.columns]

    sub = data_log2.loc[data_log2.index.intersection(significant_features)]

    if sub.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 significant proteins to build a heatmap "
            f"(found {sub.shape[0]}). Try relaxing your significance cutoff."
        )
    if sub.shape[1] < 2:
        raise ValueError(f"Need at least 2 samples to build a heatmap (found {sub.shape[1]}).")

    z = _zscore(sub).fillna(0)

    lm = linkage_method
    if distance == "correlation":
        row_dist_fn = lambda M: pdist(M, metric="correlation")
        col_dist_fn = lambda M: pdist(M.T, metric="correlation")
        if lm == "ward":
            lm = "average"
    else:
        row_dist_fn = lambda M: pdist(M, metric="euclidean")
        col_dist_fn = lambda M: pdist(M.T, metric="euclidean")

    row_link = linkage(row_dist_fn(z.values), method=lm) if cluster_rows else None
    col_link = linkage(col_dist_fn(z.values), method=lm) if cluster_cols else None

    row_order = dendrogram(row_link, no_plot=True)["leaves"] if cluster_rows else list(range(z.shape[0]))
    col_order = dendrogram(col_link, no_plot=True)["leaves"] if cluster_cols else list(range(z.shape[1]))

    z_ordered = z.iloc[row_order, col_order]
    n_rows, n_cols = z_ordered.shape
    n_col_tracks = len(col_annot_cols)
    n_row_tracks = len(row_annot_cols)
    has_annotation = n_col_tracks > 0
    has_row_annotation = n_row_tracks > 0
    annotation_colors = annotation_colors or {}

    # Pre-build every track's colors + legend info up front, so layout sizing
    # (legend column width/height) and rendering both read from the same source.
    col_tracks = []
    for i, col in enumerate(col_annot_cols):
        kind = classify_annotation_series(meta[col])
        raw_vals = _annotation_track_values(meta, col, z_ordered.columns, ANNOTATION_MISSING_LABEL)
        rgba, legend_info = _build_annotation_track(raw_vals, kind, palette_offset=i,
                                                      custom_colors=annotation_colors.get(col))
        col_tracks.append((col, rgba, legend_info))

    row_tracks = []
    for i, col in enumerate(row_annot_cols):
        kind = classify_annotation_series(row_meta[col])
        raw_vals = _annotation_track_values(row_meta, col, z_ordered.index, ROW_ANNOTATION_UNMATCHED_LABEL)
        rgba, legend_info = _build_annotation_track(raw_vals, kind, palette_offset=i + n_col_tracks,
                                                      missing_label=ROW_ANNOTATION_UNMATCHED_LABEL,
                                                      custom_colors=annotation_colors.get(col))
        row_tracks.append((col, rgba, legend_info))

    # -------------------------------------------------------------------
    # Fixed-inch layout: every element gets an absolute size in inches,
    # then positions are converted to figure-fraction coordinates. This
    # avoids the distortion that fractional GridSpec ratios + hspace/wspace
    # produce for thin elements (annotation bar, colorbar) when the number
    # of rows/columns varies. heatmap_width_in/height_in let the caller
    # override the auto-computed panel size directly (e.g. for a specific
    # publication figure size); everything else below derives from these
    # two numbers, so it all rescales together automatically.
    # -------------------------------------------------------------------
    heat_w_in = heatmap_width_in if heatmap_width_in else max(4.5, 0.30 * n_cols)
    heat_h_in = heatmap_height_in if heatmap_height_in else max(3.5, 0.22 * n_rows)

    notes = []
    if heat_h_in > MAX_HEATMAP_HEIGHT_IN:
        notes.append(
            f"Height capped at {MAX_HEATMAP_HEIGHT_IN:.0f}in (uncapped size for {n_rows} features "
            f"would have been {heat_h_in:.0f}in, which risks unreadable output or export failure)."
        )
        heat_h_in = MAX_HEATMAP_HEIGHT_IN
    if heat_w_in > MAX_HEATMAP_WIDTH_IN:
        notes.append(
            f"Width capped at {MAX_HEATMAP_WIDTH_IN:.0f}in (uncapped size for {n_cols} samples "
            f"would have been {heat_w_in:.0f}in)."
        )
        heat_w_in = MAX_HEATMAP_WIDTH_IN

    show_row_labels = n_rows <= HIDE_ROW_LABELS_ABOVE
    show_col_labels = n_cols <= HIDE_COL_LABELS_ABOVE
    if not show_row_labels:
        notes.append(
            f"Row (protein) labels hidden — {n_rows} features exceeds the {HIDE_ROW_LABELS_ABOVE} "
            f"readable-label limit. Tighten your significance cutoff to see labels, or export as PDF/SVG "
            f"and zoom in on the vector file."
        )
    if not show_col_labels:
        notes.append(
            f"Column (sample) labels hidden — {n_cols} samples exceeds the {HIDE_COL_LABELS_ABOVE} "
            f"readable-label limit."
        )
    if n_col_tracks > 6:
        notes.append(f"{n_col_tracks} column annotation tracks selected — consider trimming for readability.")
    if n_row_tracks > 6:
        notes.append(f"{n_row_tracks} row annotation tracks selected — consider trimming for readability.")
    # Row label font shrinks gracefully as feature count grows, with a hard floor
    row_label_fontsize = max(MIN_ROW_LABEL_FONTSIZE, min(0.6 * font_size, 900.0 / max(n_rows, 1)))

    row_dend_w_in = 1.0 if cluster_rows else 0.0
    col_dend_h_in = 0.9 if cluster_cols else 0.0
    track_h_in = 0.20          # each column annotation track's height
    track_gap_in = 0.025       # gap between stacked column tracks
    annot_h_in = (n_col_tracks * track_h_in + max(0, n_col_tracks - 1) * track_gap_in) if has_annotation else 0.0
    annot_label_w_in = (max(_text_width_in(c, font_size * 0.65) for c in col_annot_cols) + 0.1) if has_annotation else 0.0
    annot_gap_in = 0.03 if has_annotation else 0.0   # tight gap: annotation block sits close to heatmap
    top_gap_in = 0.08 if cluster_cols else 0.0        # small gap between col dendrogram and annotation/heatmap
    row_track_w_in = 0.20      # each row annotation track's width
    row_track_gap_in = 0.025
    row_annot_w_in = (n_row_tracks * row_track_w_in + max(0, n_row_tracks - 1) * row_track_gap_in) if has_row_annotation else 0.0
    row_annot_gap_in = 0.05 if has_row_annotation else 0.0   # between row dendrogram and row annotation block
    row_annot_gap2_in = 0.03 if has_row_annotation else 0.0  # tight gap: block sits close to heatmap
    # Row track labels (e.g. "Method", "Pathway") are drawn rotated 90 degrees directly
    # above each strip, in the blank space above the heatmap that the column
    # dendrogram/annotation block doesn't reach (those only span the heatmap's own
    # x-range, not the row-track columns further left) -- so their reserved height is
    # the longest label's rendered text WIDTH (before rotation).
    row_track_label_h_in = (max(_text_width_in(c, font_size * 0.65) for c in row_annot_cols) + 0.12) if has_row_annotation else 0.0

    if show_row_labels:
        ytick_label_w_in = max(0.8, max(_text_width_in(str(s), row_label_fontsize) for s in z_ordered.index) + 0.3)
    else:
        ytick_label_w_in = 0.15
    if show_col_labels:
        xtick_label_h_in = max(0.6, max(_text_width_in(str(s), 0.6 * font_size) for s in z_ordered.columns) + 0.3)
    else:
        xtick_label_h_in = 0.15

    # Legend column width: widest of any categorical value text (across all tracks),
    # or a fixed width if only continuous tracks are present.
    legend_w_in = 1.1
    for _, _, info in col_tracks + row_tracks:
        if info["kind"] == "categorical":
            max_text_w = max(_text_width_in(str(v), font_size * 0.65) for v in info["colors"])
            legend_w_in = max(legend_w_in, max_text_w + 0.55)
    left_margin_in = 0.35
    right_margin_in = 0.35

    cbar_gap_in = 0.18
    cbar_h_in = 0.16          # slim colorbar, not the oversized default
    bottom_margin_in = 0.35
    title_h_in = 0.45 * (font_size / 10.0)

    W = (left_margin_in + row_dend_w_in + row_annot_gap_in + row_annot_w_in + row_annot_gap2_in
         + heat_w_in + ytick_label_w_in + legend_w_in + right_margin_in)
    H = (title_h_in + row_track_label_h_in + col_dend_h_in + top_gap_in + annot_h_in + annot_gap_in + heat_h_in
         + xtick_label_h_in + cbar_gap_in + cbar_h_in + bottom_margin_in)

    # If the stacked legend column (one block per annotation track) needs more
    # vertical room than the heatmap content naturally provides, grow the figure
    # (via extra bottom margin, so the heatmap's own position is unaffected) --
    # otherwise later legend blocks would silently run past the figure edge and
    # get clipped, which is much worse than a bit of extra whitespace at the
    # bottom for a heatmap with many annotation tracks.
    all_legend_blocks = col_tracks + row_tracks
    total_legend_h_in = sum(_estimate_legend_block_height_in(info, font_size) for _, _, info in all_legend_blocks)
    total_legend_h_in += 0.10 * max(0, len(all_legend_blocks) - 1)
    available_legend_span_in = H - title_h_in - bottom_margin_in
    if total_legend_h_in > available_legend_span_in:
        shortfall = total_legend_h_in - available_legend_span_in
        H += shortfall
        bottom_margin_in += shortfall

    fig = plt.figure(figsize=(W, H))

    # --- vertical positions (top-down) ---
    y = 1.0
    y -= title_h_in / H
    y -= row_track_label_h_in / H
    row_track_label_bottom = y
    if cluster_cols:
        y -= col_dend_h_in / H
        col_dend_bottom = y
    if cluster_cols or has_annotation:
        y -= top_gap_in / H if cluster_cols else 0
    if has_annotation:
        y -= annot_h_in / H
        annot_block_bottom = y
        y -= annot_gap_in / H
    y -= heat_h_in / H
    heat_bottom = y
    heat_top = heat_bottom + heat_h_in / H
    y -= xtick_label_h_in / H
    y -= cbar_gap_in / H
    y -= cbar_h_in / H
    cbar_bottom = y

    # --- horizontal positions (left-right) ---
    x = left_margin_in / W
    row_dend_left = x
    x += row_dend_w_in / W
    if has_row_annotation:
        x += row_annot_gap_in / W
        row_annot_left = x
        x += row_annot_w_in / W
        x += row_annot_gap2_in / W
    heat_left = x
    x += heat_w_in / W
    legend_x = x + ytick_label_w_in / W
    annot_label_x = heat_left + heat_w_in / W + 0.03 / W

    # Colorbar: slim, positioned at the bottom-RIGHT under the heatmap (not full width)
    cbar_w_in = min(2.4, heat_w_in * 0.55)
    cbar_left = heat_left + (heat_w_in - cbar_w_in) / W
    cbar_w = cbar_w_in / W

    cmap = build_colormap(cmap_name, custom_colors, reverse_cmap)

    if cluster_cols:
        ax_col_dend = fig.add_axes([heat_left, col_dend_bottom, heat_w_in / W, col_dend_h_in / H])
        dendrogram(col_link, ax=ax_col_dend, no_labels=True, color_threshold=0)
        ax_col_dend.axis("off")

    if has_annotation:
        track_y = annot_block_bottom + annot_h_in / H  # top of the stacked annotation block
        for col, rgba, info in col_tracks:
            track_y -= track_h_in / H
            ax_t = fig.add_axes([heat_left, track_y, heat_w_in / W, track_h_in / H])
            ax_t.imshow(rgba.reshape(1, -1, 4), aspect="auto", interpolation="nearest")
            ax_t.set_xticks([])
            ax_t.set_yticks([])
            for spine in ax_t.spines.values():
                spine.set_visible(False)
            fig.text(annot_label_x, track_y + (track_h_in / 2) / H, col,
                     fontsize=font_size * 0.65, va="center", ha="left",
                     fontfamily=font_family, transform=fig.transFigure)
            track_y -= track_gap_in / H

    if has_row_annotation:
        track_x = row_annot_left
        for col, rgba, info in row_tracks:
            ax_t = fig.add_axes([track_x, heat_bottom, row_track_w_in / W, heat_h_in / H])
            ax_t.imshow(rgba.reshape(-1, 1, 4), aspect="auto", interpolation="nearest")
            ax_t.set_xticks([])
            ax_t.set_yticks([])
            for spine in ax_t.spines.values():
                spine.set_visible(False)
            track_center_x = track_x + (row_track_w_in / 2) / W
            fig.text(track_center_x, row_track_label_bottom + 0.03 / H, col,
                     fontsize=font_size * 0.65, va="bottom", ha="center", rotation=90,
                     rotation_mode="anchor", fontfamily=font_family, transform=fig.transFigure)
            track_x += (row_track_w_in + row_track_gap_in) / W

    # Legends: one stacked column, all tracks (column annotations first, then row
    # annotations), top to bottom -- sizing above already guarantees this fits.
    legend_y = heat_top
    for col, rgba, info in col_tracks + row_tracks:
        consumed_in = _render_legend_block(fig, info, col, legend_x, legend_y, W, H,
                                            font_size=font_size, font_family=font_family)
        legend_y -= (consumed_in + 0.10) / H

    if cluster_rows:
        ax_row_dend = fig.add_axes([row_dend_left, heat_bottom, row_dend_w_in / W, heat_h_in / H])
        dendrogram(row_link, ax=ax_row_dend, orientation="left", no_labels=True, color_threshold=0)
        ax_row_dend.axis("off")

    ax_heat = fig.add_axes([heat_left, heat_bottom, heat_w_in / W, heat_h_in / H])

    if breakpoints:
        bp = sorted(breakpoints)
        norm = BoundaryNorm(bp, ncolors=cmap.N, clip=True)
        im = ax_heat.imshow(z_ordered.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    else:
        im = ax_heat.imshow(z_ordered.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    if show_col_labels:
        ax_heat.set_xticks(range(n_cols))
        ax_heat.set_xticklabels(z_ordered.columns, rotation=90, fontsize=0.6 * font_size,
                                 fontfamily=font_family)
    else:
        ax_heat.set_xticks([])
    if show_row_labels:
        ax_heat.set_yticks(range(n_rows))
        ax_heat.set_yticklabels(z_ordered.index, fontsize=row_label_fontsize, fontfamily=font_family)
    else:
        ax_heat.set_yticks([])
    ax_heat.yaxis.tick_right()

    ax_cbar = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h_in / H])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Z-Score", fontsize=0.7 * font_size, fontfamily=font_family)
    cbar.ax.tick_params(labelsize=0.6 * font_size)
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_fontfamily(font_family)

    title = "Heatmap of Proteins"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, y=1.0 - (title_h_in * 0.4) / H, fontsize=font_size + 1, fontfamily=font_family)

    return fig, z_ordered, notes


def export_figure(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """
    Export a matplotlib figure to bytes in the requested format.
    fmt: 'png', 'pdf', 'svg', 'jpeg'/'jpg', or 'tiff'.

    dpi applies here even for the "vector" formats (PDF/SVG): this heatmap draws its
    cells with imshow(), which embeds a RASTER bitmap inside the PDF/SVG regardless of
    the container format. Without an explicit dpi, matplotlib falls back to its default
    (~100), which for a heatmap with many rows produces too few embedded pixels per
    row — rows blend into each other and the figure looks blurry/smeared when zoomed in
    or printed, exactly as if it were a genuinely low-resolution export. Passing dpi
    here ensures the embedded bitmap has enough resolution to render cells crisply.

    Safety net: rasterization memory scales with TOTAL pixel count (width x height x 4
    bytes for RGBA), not just the per-dimension size. A tall heatmap at high DPI can
    demand a gigabyte-plus raw buffer before any file encoding even happens — confirmed
    this crashes the process (OOM) for a ~500-feature heatmap exported as 600 DPI TIFF.
    If the requested dpi would exceed a safe total-megapixel budget, dpi is
    automatically reduced. TIFF output also uses LZW (lossless) compression, which
    cuts file size by ~100x with no quality loss (uncompressed TIFF is the default
    otherwise and is needlessly enormous).
    """
    fmt = fmt.lower()
    w_in, h_in = fig.get_size_inches()
    safe_total_megapixels = 40.0  # ~40MP keeps the raw RGBA buffer under ~450MB
    requested_megapixels = (w_in * dpi) * (h_in * dpi) / 1_000_000
    if requested_megapixels > safe_total_megapixels:
        scale = (safe_total_megapixels / requested_megapixels) ** 0.5
        dpi = max(72, int(dpi * scale))
    buf = io.BytesIO()
    save_kwargs = {"format": "png" if fmt == "png" else fmt, "bbox_inches": "tight", "dpi": dpi}
    if fmt == "tiff":
        save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    if fmt in ("jpeg", "jpg"):
        # JPEG has no alpha channel; force a white background.
        save_kwargs["facecolor"] = "white"
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    return buf.read()
