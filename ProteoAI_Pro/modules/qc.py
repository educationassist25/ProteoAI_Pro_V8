"""
qc.py - Quality Control validation module.

Implements:
  - Coefficient of Variation (CV) calculation across QC replicates
  - CV distribution / histogram plots
  - CV-based filtering table (<=20% acceptable, >20% variable)
  - QC sample correlation matrix (the only QC visualization retained per request —
    QC PCA, hierarchical clustering, and sample distance heatmap were removed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def calculate_cv(qc_df: pd.DataFrame) -> pd.DataFrame:
    """
    qc_df: features x QC-samples matrix (raw protein intensities).
    Returns a DataFrame with Mean, SD, CV(%) and Quality flag per feature.
    """
    mean = qc_df.mean(axis=1, skipna=True)
    sd = qc_df.std(axis=1, skipna=True, ddof=1)
    cv = (sd / mean.replace(0, np.nan)) * 100
    quality = np.where(cv <= 20, "Acceptable", "Variable")
    out = pd.DataFrame({"Mean": mean, "SD": sd, "CV(%)": cv, "Quality": quality}, index=qc_df.index)
    return out.sort_values("CV(%)")


def cv_distribution_plot(cv_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cv_table["CV(%)"].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(20, color="red", linestyle="--", label="20% threshold")
    ax.set_xlabel("Coefficient of Variation (%)")
    ax.set_ylabel("Number of features")
    ax.set_title("QC CV Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def cv_histogram(cv_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = cv_table["Quality"].value_counts()
    ax.bar(counts.index, counts.values, color=["#55A868", "#C44E52"])
    ax.set_ylabel("Number of features")
    ax.set_title("Feature Count by CV Quality")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    return fig


def sample_correlation_matrix(qc_log_data: pd.DataFrame):
    """QC-only Pearson correlation matrix. qc_log_data must contain ONLY QC replicates."""
    corr = qc_log_data.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="viridis", vmin=corr.values.min(), vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=6)
    ax.set_title("QC Sample Correlation Matrix (QC replicates only)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    return fig, corr


def protein_intensity_histogram(peak_df: pd.DataFrame, title: str = "Protein Intensity Distribution"):
    """
    Histogram of all raw protein-intensity values in a features x samples matrix (every
    feature x every sample cell, flattened), on a log10 scale -- protein intensities span
    several orders of magnitude, so a linear-scale histogram would just show one
    tall bar near zero. Values <= 0 (already-imputed zeros/negatives shouldn't
    normally occur, but guard anyway) are dropped before logging.
    """
    values = peak_df.values.flatten()
    values = values[np.isfinite(values) & (values > 0)]
    log_values = np.log10(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(log_values, bins=50, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("log10(Protein Intensity)")
    ax.set_ylabel("Count (feature × sample values)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
