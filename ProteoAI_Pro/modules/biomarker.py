"""
biomarker.py - Biomarker discovery via statistical filtering criteria.
"""

import numpy as np
import pandas as pd


def discover_biomarkers(stats_table: pd.DataFrame, criterion: str = "combined",
                         p_cutoff: float = 0.05, fdr_cutoff: float = 0.25,
                         fc_col: str = "Log2FC"):
    """
    criterion: 'pvalue' | 'fdr' | 'combined'
    Returns a table: Protein, p-value, FDR, Linear_FC, Log2FC, Direction
    """
    df = stats_table.copy()

    if criterion == "pvalue":
        mask = df["p-value"] < p_cutoff
    elif criterion == "fdr":
        mask = df["FDR"] < fdr_cutoff
    else:  # combined
        mask = (df["p-value"] < p_cutoff) & (df["FDR"] < fdr_cutoff)

    result = df[mask].copy()
    result["Direction"] = np.where(result[fc_col] > 0, "Up", "Down")

    cols = []
    for c in ["p-value", "FDR", "Linear_FC", fc_col, "Direction"]:
        if c in result.columns:
            cols.append(c)
    result = result[cols].rename(columns={fc_col: "Log2FC"} if fc_col != "Log2FC" else {})
    return result.sort_values("p-value")
