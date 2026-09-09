"""
pca_module.py - PCA score plot (with confidence ellipses), loading plot, variance plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

matplotlib.use("Agg")



def _confidence_ellipse(ax, x, y, color, n_std=1.96, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse((np.mean(x), np.mean(y)), width, height, angle=angle,
                       facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5, **kwargs)
    ax.add_patch(ellipse)


PALETTES = {
    "Default (tab10)": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"],
    "Set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "Dark2": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"],
    "Paired": ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00"],
    "Colorblind-safe": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"],
}
MARKER_STYLES = ["o", "s", "^", "D", "v", "P", "X", "*"]


def run_pca(data_log2: pd.DataFrame, n_components: int = 5):
    """
    data_log2: features x samples — must contain ONLY biological samples (QC already
    excluded upstream during normalization). Returns fitted PCA object and sample scores DataFrame.
    """
    X = data_log2.T.fillna(data_log2.T.mean())
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    scores_df = pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(n_components)])
    return pca, scores_df, X.columns


def pca_score_plot(pca, scores_df: pd.DataFrame, meta: pd.DataFrame,
                    palette: str = "Default (tab10)", marker_map: dict = None,
                    show_ellipse: bool = True, group_col: str = "Group"):
    """
    Biological-sample PCA score plot, color-coded by `group_col` (any categorical
    metadata column -- Group/Diagnosis, Gender, Treatment, Ethnicity, etc.).
    palette: key into PALETTES, a list of hex colors, or a {group_name: hex_color}
        dict (e.g. built from individual 2D color pickers in the UI) for exact
        per-group control regardless of iteration order.
    marker_map: optional {group_name: matplotlib marker symbol}. Defaults to 'o' for all groups.
    show_ellipse: draw 95% confidence ellipses per group (optional, per spec).
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    var = pca.explained_variance_ratio_ * 100
    groups = meta.loc[scores_df.index, group_col]
    colors = PALETTES.get(palette, palette) if isinstance(palette, str) else palette
    marker_map = marker_map or {}

    for i, g in enumerate(groups.unique()):
        mask = groups == g
        color = colors.get(g, "#333333") if isinstance(colors, dict) else colors[i % len(colors)]
        marker = marker_map.get(g, "o")
        ax.scatter(scores_df.loc[mask, "PC1"], scores_df.loc[mask, "PC2"], label=g,
                   color=color, marker=marker, alpha=0.85, s=70, edgecolor="k", linewidth=0.4)
        if show_ellipse:
            _confidence_ellipse(ax, scores_df.loc[mask, "PC1"].values, scores_df.loc[mask, "PC2"].values, color)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_title(f"PCA Score Plot (Biological Samples Only, colored by {group_col})")
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def pca_loading_plot(pca, feature_names, top_n: int = 20):
    loadings = pd.DataFrame(pca.components_[:2].T, index=feature_names, columns=["PC1", "PC2"])
    loadings["magnitude"] = np.sqrt(loadings["PC1"]**2 + loadings["PC2"]**2)
    top = loadings.sort_values("magnitude", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(loadings["PC1"], loadings["PC2"], color="lightgray", s=15, alpha=0.6)
    ax.scatter(top["PC1"], top["PC2"], color="#C44E52", s=35)
    for name, row in top.iterrows():
        ax.annotate(name, (row["PC1"], row["PC2"]), fontsize=6, alpha=0.85)
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title(f"PCA Loading Plot (top {top_n} contributing proteins)")
    fig.tight_layout()
    return fig, top


def pca_variance_plot(pca):
    var = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(1, len(var) + 1), var, color="#4C72B0")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title(f"Explained Variance (Total top-2: {var[:2].sum():.1f}%)")
    for i, v in enumerate(var):
        ax.text(i + 1, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig
