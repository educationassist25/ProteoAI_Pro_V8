"""
stats_analysis.py - Two-group and multi-group (ANOVA) statistical comparison module.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_TUKEY = True
except ImportError:
    HAS_TUKEY = False


def _fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(pvals)
    fdr = np.full(pvals.shape, np.nan)
    if mask.sum():
        fdr[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return fdr


def _welch_ci(x, y, confidence: float = 0.95):
    """95% CI (default) for the mean difference (x - y) using Welch's t formula."""
    nx, ny = len(x), len(y)
    mean_diff = x.mean() - y.mean()
    varx, vary = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(varx / nx + vary / ny)
    denom = (varx / nx) ** 2 / (nx - 1) + (vary / ny) ** 2 / (ny - 1)
    df = (varx / nx + vary / ny) ** 2 / denom if denom > 0 else nx + ny - 2
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df)
    return mean_diff - t_crit * se, mean_diff + t_crit * se


def two_group_test(data_log2: pd.DataFrame, group_a_samples, group_b_samples, method: str = "ttest"):
    """
    All statistics (mean abundance, fold change, p-value, FDR, 95% CI) are computed from the
    log2-transformed, normalized data — never from raw protein intensities. Raw protein intensities are for
    storage/traceability only and are not used here.

    data_log2   : log2-transformed, normalized features x samples matrix (biological samples only —
                  QC samples must already be excluded upstream)
    method: 'ttest' (Student's t-test) or 'wilcoxon' (Wilcoxon rank-sum / Mann-Whitney U)
    Returns a results DataFrame: Mean_Log2_GroupA, Mean_Log2_GroupB, Linear_FC, Log2FC,
    CI_Lower_Log2FC, CI_Upper_Log2FC, p-value, FDR, Significant
    """
    a2 = data_log2[group_a_samples]
    b2 = data_log2[group_b_samples]

    pvals, ci_lower, ci_upper = [], [], []
    for feat in data_log2.index:
        x = a2.loc[feat].dropna().values
        y = b2.loc[feat].dropna().values
        if len(x) < 2 or len(y) < 2:
            pvals.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            continue
        if method == "ttest":
            _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        else:
            _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        pvals.append(p)
        lo, hi = _welch_ci(x, y)
        ci_lower.append(lo)
        ci_upper.append(hi)

    mean_a = a2.mean(axis=1)
    mean_b = b2.mean(axis=1)
    log2fc = mean_a - mean_b
    linear_fc = 2 ** log2fc
    fdr = _fdr(pvals)

    result = pd.DataFrame({
        "Mean_Log2_GroupA": mean_a,
        "Mean_Log2_GroupB": mean_b,
        "Linear_FC": linear_fc,
        "Log2FC": log2fc,
        "CI_Lower_Log2FC": ci_lower,
        "CI_Upper_Log2FC": ci_upper,
        "p-value": pvals,
        "FDR": fdr,
    }, index=data_log2.index)
    result["Significant"] = (result["p-value"] < 0.05) & (result["FDR"] < 0.25)
    return result.sort_values("p-value")


def anova_test(data_log2: pd.DataFrame, group_map: pd.Series, posthoc: str = "tukey"):
    """
    One-way ANOVA across >=3 groups.
    group_map: Series indexed by sample name -> group label (subset of data_log2.columns)
    posthoc: 'tukey', 'dunnett', or 'pairwise' (pairwise t-tests with BH correction)

    Returns (anova_table, posthoc_results_dict)
      anova_table: Protein, F-statistic, ANOVA p-value, FDR
      posthoc_results_dict: {feature: DataFrame of pairwise comparisons} for significant features
    """
    groups = group_map.unique().tolist()
    if len(groups) < 3:
        raise ValueError("ANOVA module requires 3 or more groups.")

    f_stats, pvals = [], []
    for feat in data_log2.index:
        samples_by_group = [
            data_log2.loc[feat, group_map.index[group_map == g]].dropna().values
            for g in groups
        ]
        samples_by_group = [s for s in samples_by_group if len(s) >= 2]
        if len(samples_by_group) < 2:
            f_stats.append(np.nan)
            pvals.append(np.nan)
            continue
        f, p = stats.f_oneway(*samples_by_group)
        f_stats.append(f)
        pvals.append(p)

    fdr = _fdr(pvals)
    anova_table = pd.DataFrame({
        "F-statistic": f_stats,
        "ANOVA p-value": pvals,
        "FDR": fdr,
    }, index=data_log2.index).sort_values("ANOVA p-value")

    # Post-hoc for significant features (FDR < 0.25), capped to avoid huge runtime
    sig_features = anova_table[anova_table["FDR"] < 0.25].index.tolist()
    posthoc_results = {}

    for feat in sig_features:
        vals = data_log2.loc[feat]
        sub_df = pd.DataFrame({"value": vals, "group": group_map})
        sub_df = sub_df.dropna()

        if posthoc == "tukey" and HAS_TUKEY:
            try:
                res = pairwise_tukeyhsd(sub_df["value"], sub_df["group"])
                ph = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
            except Exception:
                ph = pd.DataFrame()
        elif posthoc == "dunnett":
            # Dunnett vs first group as control; approximate with pairwise t-tests + BH
            control = groups[0]
            rows = []
            ctrl_vals = sub_df.loc[sub_df["group"] == control, "value"].values
            for g in groups:
                if g == control:
                    continue
                gv = sub_df.loc[sub_df["group"] == g, "value"].values
                if len(gv) >= 2 and len(ctrl_vals) >= 2:
                    _, p = stats.ttest_ind(gv, ctrl_vals, equal_var=False)
                    rows.append({"control": control, "group": g, "p-value": p})
            ph = pd.DataFrame(rows)
            if len(ph):
                ph["FDR"] = _fdr(ph["p-value"].values)
        else:  # pairwise
            rows = []
            for g1, g2 in combinations(groups, 2):
                v1 = sub_df.loc[sub_df["group"] == g1, "value"].values
                v2 = sub_df.loc[sub_df["group"] == g2, "value"].values
                if len(v1) >= 2 and len(v2) >= 2:
                    _, p = stats.ttest_ind(v1, v2, equal_var=False)
                    rows.append({"group1": g1, "group2": g2, "p-value": p})
            ph = pd.DataFrame(rows)
            if len(ph):
                ph["FDR"] = _fdr(ph["p-value"].values)

        posthoc_results[feat] = ph

    return anova_table, posthoc_results


def complete_statistical_table(data_log2: pd.DataFrame, group_a_samples, group_b_samples):
    """
    Full table computed entirely from log2-transformed, normalized data: mean log2 abundance
    (both groups), Linear FC, Log2 FC, 95% CI on Log2FC, p-value, FDR, Significance.
    """
    return two_group_test(data_log2, group_a_samples, group_b_samples, method="ttest")
