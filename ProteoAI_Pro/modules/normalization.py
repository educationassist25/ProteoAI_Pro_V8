"""
normalization.py - Reference-channel (TMT) normalization, Median-IQR (label-free)
normalization, and Log2 transformation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# 1. Reference-channel normalization (TMT)
# ---------------------------------------------------------------------------
def reference_channel_normalize(intensity_df: pd.DataFrame, reference_channel: str) -> pd.DataFrame:
    """
    Normalized Protein Intensity = Channel Protein Intensity / Reference Channel Protein Intensity.

    TMT (Tandem Mass Tag) plexes commonly include one channel that is a pooled/bridge
    reference sample (the same pooled material run in every plex, or a designated
    reference channel within the plex). Every protein's intensity in every other
    channel is expressed as a ratio to that same protein's intensity in the reference
    channel, which corrects for plex-to-plex and channel-to-channel loading differences
    the same way ISTD normalization does in targeted small-molecule assays -- just
    computed per-protein (row) against a reference SAMPLE (column) instead of against a
    single internal-standard feature.

    reference_channel must be a column (sample) present in intensity_df. The reference
    channel itself is dropped from the output (its own ratio is trivially 1 for every
    protein and carries no information).
    """
    if reference_channel not in intensity_df.columns:
        raise ValueError(f"Reference channel '{reference_channel}' not found among samples.")
    ref_col = intensity_df[reference_channel]
    normalized = intensity_df.drop(columns=reference_channel).div(ref_col.replace(0, np.nan), axis=0)
    return normalized


# Backward-compatible alias retained for any external callers.
istd_normalize = reference_channel_normalize


# ---------------------------------------------------------------------------
# 2. Median centering normalization
# ---------------------------------------------------------------------------
def median_center_normalize(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    Median centering normalization (per-sample):
        X_norm_ij = X_ij - Median_j(X) + Grand_Median

    For each sample (column), subtract that sample's own median (computed across
    proteins) so every sample is recentered to the same level — the standard fix for
    systematic sample-to-sample loading/injection offsets. Grand_Median (the median of
    all per-sample medians) is added back so the overall scale of the data is
    preserved rather than collapsed to zero.

    exclude_cols: sample column(s) to exclude entirely — both from the per-sample
    median/grand-median calculation AND from the output. Use this for a TMT
    reference/pooled channel, which isn't a directly-comparable study sample and
    would distort the grand median if included.
    """
    exclude_cols = list(exclude_cols) if exclude_cols else []
    calc_cols = [c for c in df.columns if c not in exclude_cols]
    sub = df[calc_cols]
    sample_medians = sub.median(axis=0)
    grand_median = sample_medians.median()
    return sub.sub(sample_medians, axis=1).add(grand_median)


# ---------------------------------------------------------------------------
# 3. Global MAD-based variance scaling
# ---------------------------------------------------------------------------
def mad_scale_normalize(df: pd.DataFrame, exclude_cols=None):
    """
    Global normalization by Median Absolute Deviation (MAD)-based variance scaling:
        X_norm = X / (MAD_global x 1.4826)

    A single global MAD is computed across every value in the dataset at once (not
    per-protein or per-sample), then used as one scaling factor applied uniformly —
    standardizing the overall variance/spread of the whole dataset in one step. The
    1.4826 constant rescales MAD to be comparable to a standard deviation for
    normally-distributed data (the standard MAD-to-SD consistency correction). Pure
    scaling (no centering), so — unlike Median Centering or Median-IQR — this never
    introduces negative values from originally-positive protein intensities.

    exclude_cols: sample column(s) to exclude entirely — both from the global MAD
    calculation AND from the output. Use this for a TMT reference/pooled channel,
    which isn't a directly-comparable study sample and would distort the global MAD
    if included.

    Returns (normalized_df, mad_scale_used).
    """
    exclude_cols = list(exclude_cols) if exclude_cols else []
    calc_cols = [c for c in df.columns if c not in exclude_cols]
    sub = df[calc_cols]
    vals = sub.values.astype(float).flatten()
    vals = vals[~np.isnan(vals)]
    global_median = np.median(vals) if vals.size else 0.0
    mad = np.median(np.abs(vals - global_median)) if vals.size else 0.0
    mad_scale = mad * 1.4826 if mad > 0 else 1.0
    return sub / mad_scale, mad_scale


# ---------------------------------------------------------------------------
# 4. IQR normalization
# ---------------------------------------------------------------------------
def iqr_normalize(df: pd.DataFrame, axis: str = "feature", batch_map: pd.Series = None) -> pd.DataFrame:
    """
    Median-IQR normalization (a.k.a. robust scaling):
        X_norm = (X - Median(X)) / IQR(X)

    Recommended usage: apply this AFTER log2 transformation, not before. Robust-scaled
    values are frequently negative (anything below the median), so taking log2 of the
    OUTPUT of this function will produce NaNs for roughly half the data. If you need
    both steps, always log2 first, then robust-scale the log2 values.

    axis:
      - 'feature' : normalize each protein (row) across samples — matches
                    X_ij = (X_ij - Median(X_j)) / IQR(X_j) computed per protein j
                    across all samples i.
      - 'sample'  : normalize each sample/channel (column) across proteins — the
                    common choice for label-free proteomics, correcting for
                    sample-to-sample loading/injection differences.
      - 'batch'   : normalize each feature within each batch separately (batch_map required:
                    Series indexed by sample name -> batch label)
    """
    if axis == "feature":
        median = df.median(axis=1)
        q1 = df.quantile(0.25, axis=1)
        q3 = df.quantile(0.75, axis=1)
        iqr = (q3 - q1).replace(0, np.nan)
        return df.sub(median, axis=0).div(iqr, axis=0)

    elif axis == "sample":
        median = df.median(axis=0)
        q1 = df.quantile(0.25, axis=0)
        q3 = df.quantile(0.75, axis=0)
        iqr = (q3 - q1).replace(0, np.nan)
        return df.sub(median, axis=1).div(iqr, axis=1)

    elif axis == "batch":
        if batch_map is None:
            raise ValueError("batch_map (sample -> batch label) is required for batch-specific normalization.")
        out = df.copy()
        for batch in batch_map.unique():
            cols = batch_map.index[batch_map == batch].tolist()
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            sub = df[cols]
            median = sub.median(axis=1)
            q1 = sub.quantile(0.25, axis=1)
            q3 = sub.quantile(0.75, axis=1)
            iqr = (q3 - q1).replace(0, np.nan)
            out[cols] = sub.sub(median, axis=0).div(iqr, axis=0)
        return out

    else:
        raise ValueError("axis must be one of 'feature', 'sample', 'batch'")


# ---------------------------------------------------------------------------
# 5. Log2 transformation
# ---------------------------------------------------------------------------
def log2_transform(df: pd.DataFrame, constant: float = None, auto_zero_replace: bool = True):
    """
    Apply log2(x + constant). If constant is None, it is automatically set to a small
    fraction of the smallest positive value in the dataset (minimum value adjustment).
    This handles exact zeros (the common "not detected"/"not quantified" convention in
    MS-based proteomics) automatically: log2(0 + constant) is well-defined and represents
    "at or below the minimum detected level" — no separate zero-replacement step is needed
    once a sensible constant is chosen. Genuine NaN (true missing data) still propagates
    as NaN; handle real missing values via the Data Cleaning & Imputation step before this.
    Returns (transformed_df, constant_used).
    """
    work = df.copy()
    if auto_zero_replace:
        positive_vals = work.values[(work.values > 0) & (~np.isnan(work.values))]
        min_pos = positive_vals.min() if positive_vals.size else 1.0
        if constant is None:
            constant = min_pos * 0.5 if min_pos > 0 else 1.0
    else:
        if constant is None:
            constant = 1.0

    transformed = np.log2(work + constant)
    return transformed, constant


def shift_and_log2_transform(df: pd.DataFrame, shift: float = None):
    """
    Shift data to be strictly positive (if needed), then log2 transform.

    Use this — instead of log2_transform() — for data that has already passed through
    a CENTERING normalization such as the classic Median-IQR formula,
    (X - Median) / IQR, which produces negative values for any point below the median.
    log2 is undefined for those without an offset; log2_transform()'s zero-replacement
    logic would incorrectly treat all negative values as missing data and clobber them
    to a single small constant. This function instead shifts the entire dataset by a
    constant just large enough to make the global minimum slightly positive, preserving
    every value's relative position, then logs the shifted data.

    If the data is already all-positive, shift defaults to 0 (equivalent to a plain
    log2 with no zero-handling — use log2_transform() instead if you need automatic
    zero replacement for genuinely-zero raw values).

    Returns (transformed_df, shift_used).
    """
    finite_vals = df.values[np.isfinite(df.values)]
    min_val = finite_vals.min() if finite_vals.size else 0.0
    if shift is None:
        if min_val <= 0:
            shift = abs(min_val) + max(abs(min_val) * 0.01, 1e-3)
        else:
            shift = 0.0
    shifted = df + shift
    transformed = np.log2(shifted)
    return transformed, shift


def distribution_plots(before: pd.DataFrame, after: pd.DataFrame, sample_id: str = None):
    """
    Generate before/after density and box plots, each on ITS OWN appropriately-scaled
    panel. Raw protein intensities (before) typically span several orders of magnitude
    while log2-transformed values (after) span a much smaller range (~0-30); overlaying
    them on a single shared axis makes one distribution collapse to an invisible sliver.
    Using separate panels (with a log x-axis for the raw-scale density, since raw
    intensities are all positive and right-skewed) keeps both distributions legible.
    """
    b = before.values.flatten()
    b = b[~np.isnan(b) & (b > 0)]
    a = after.values.flatten()
    a = a[~np.isnan(a)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Before: density (log-x, since raw intensities are positive and span orders of magnitude)
    if b.size:
        axes[0, 0].hist(b, bins=60, color="#C44E52", alpha=0.8)
        axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Before: Density (raw protein intensity, log scale)")
    axes[0, 0].set_ylabel("Count")

    # Before: box plot (own y-axis, raw scale)
    if b.size:
        try:
            axes[0, 1].boxplot([b], tick_labels=["Before"])
        except TypeError:
            axes[0, 1].boxplot([b], labels=["Before"])
        axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Before: Box Plot (raw protein intensity, log scale)")

    # After: density (linear x, already log2 scale)
    if a.size:
        axes[1, 0].hist(a, bins=60, color="#55A868", alpha=0.8)
    axes[1, 0].set_title("After: Density (log2-transformed)")
    axes[1, 0].set_xlabel("log2(protein intensity)")
    axes[1, 0].set_ylabel("Count")

    # After: box plot (own y-axis, log2 scale)
    if a.size:
        try:
            axes[1, 1].boxplot([a], tick_labels=["After"])
        except TypeError:
            axes[1, 1].boxplot([a], labels=["After"])
    axes[1, 1].set_title("After: Box Plot (log2-transformed)")

    fig.tight_layout()
    return fig
