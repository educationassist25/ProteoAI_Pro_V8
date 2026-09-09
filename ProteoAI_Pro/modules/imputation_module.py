"""
imputation_module.py - Data Cleaning & Missing Value Imputation, applied before
normalization.

Step 1: assess missingness per protein and filter out poor-quality features.
Step 2: impute remaining missing values using one of several standard methods.

Missing values are defined as NaN or exact zero (both commonly represent
"not detected" / below the limit of detection in LC-MS/MS protein intensity tables).

Implemented methods:
  - Half-Minimum (LOD/2): the most common LC-MS/MS convention — replace with half the
    smallest observed (non-missing) value for that protein.
  - Mean / Median: replace with the feature's mean or median of observed values.
  - K-Nearest Neighbors (KNN): estimate from the most similar samples
    (scikit-learn's KNNImputer).
  - Random Forest (MissForest-style): iteratively predicts each missing value from
    all other proteins using a random forest regressor
    (scikit-learn's IterativeImputer with a RandomForestRegressor estimator — the
    standard Python approach to MissForest, since no official `missForest` package
    exists outside R).
  - BPCA (approximated): Bayesian-PCA-style imputation approximated via
    scikit-learn's IterativeImputer with a BayesianRidge estimator (regression-based,
    using a Bayesian linear model rather than true probabilistic PCA — a reasonable
    approximation, not an exact reproduction of the original BPCA algorithm).
  - QRILC (approximated): the original QRILC algorithm (quantile regression on
    left-censored data, R's `imputeLCMD` package) is not available in Python. This
    implements the same underlying ASSUMPTION QRILC is built on — missing values in
    LC-MS/MS data are usually left-censored (below the detection limit), not missing at
    random — by drawing imputed values from a normal distribution truncated to the
    left tail of each feature's observed distribution (mean and SD shifted/shrunk
    downward, a common simplification of this method family). This is NOT the exact
    quantile-regression algorithm from the original QRILC paper.
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

MISSINGNESS_GUIDANCE = [
    ("< 20%", "Keep", 0, 20),
    ("20-50%", "Keep with careful imputation", 20, 50),
    ("50-70%", "Usually remove", 50, 70),
    ("70-80%+", "Remove unless biologically essential", 70, 100.0001),
]

METHOD_INFO = {
    "Half-Minimum (LOD/2)": "Replace missing values with half the minimum observed value for that protein. "
                            "The most common convention for LC-MS/MS data.",
    "Mean": "Replace missing values with the feature's mean of observed values.",
    "Median": "Replace missing values with the feature's median of observed values.",
    "K-Nearest Neighbors (KNN)": "Estimate each missing value from the most similar samples "
                                 "(based on other proteins' profiles).",
    "Random Forest (MissForest-style)": "Iteratively predicts each missing value from all other "
                                        "proteins using a random forest regressor.",
    "BPCA (approximated)": "Bayesian-PCA-style imputation, approximated via a Bayesian ridge "
                           "regression model applied iteratively (not the exact original BPCA algorithm).",
    "QRILC (approximated)": "Left-censored imputation: draws missing values from the lower tail of each "
                            "protein's observed distribution, matching the assumption that missing "
                            "LC-MS/MS values are usually below the detection limit rather than random "
                            "(approximates the assumption behind QRILC — not the exact quantile-regression "
                            "algorithm from the original R package, which isn't available in Python).",
}


def compute_missingness(df: pd.DataFrame, treat_zero_as_missing: bool = True) -> pd.DataFrame:
    """
    df: features x samples raw protein intensity matrix.
    Returns a DataFrame indexed by feature with: N_Missing, Pct_Missing, Quality category.
    """
    is_missing = df.isna()
    if treat_zero_as_missing:
        is_missing = is_missing | (df == 0)
    n_missing = is_missing.sum(axis=1)
    pct_missing = 100 * n_missing / df.shape[1]

    def _quality(pct):
        if pct < 20:
            return "Keep"
        elif pct < 50:
            return "Keep (careful imputation)"
        elif pct < 70:
            return "Usually remove"
        else:
            return "Remove unless essential"

    quality = pct_missing.apply(_quality)
    return pd.DataFrame({
        "N_Missing": n_missing, "Pct_Missing": pct_missing, "Quality": quality
    }, index=df.index).sort_values("Pct_Missing", ascending=False)


def filter_by_missingness(df: pd.DataFrame, missingness_table: pd.DataFrame,
                           max_pct_missing: float = 50.0) -> pd.DataFrame:
    """Keep only features with missingness <= max_pct_missing."""
    keep = missingness_table.index[missingness_table["Pct_Missing"] <= max_pct_missing]
    return df.loc[df.index.intersection(keep)]


def _as_missing_mask(df: pd.DataFrame, treat_zero_as_missing: bool = True) -> pd.DataFrame:
    mask = df.isna()
    if treat_zero_as_missing:
        mask = mask | (df == 0)
    return mask


def count_missing(df: pd.DataFrame, treat_zero_as_missing: bool = True) -> int:
    """Total count of missing values (NaN, and optionally exact zero) across the whole matrix."""
    return int(_as_missing_mask(df, treat_zero_as_missing).values.sum())


def impute_half_minimum(df: pd.DataFrame, treat_zero_as_missing: bool = True) -> pd.DataFrame:
    mask = _as_missing_mask(df, treat_zero_as_missing)
    work = df.mask(mask)
    out = work.copy()
    for idx in out.index:
        row = work.loc[idx]
        nonmissing = row.dropna()
        fill_val = (nonmissing.min() / 2) if len(nonmissing) else 0.0
        out.loc[idx] = row.fillna(fill_val)
    return out


def impute_mean_median(df: pd.DataFrame, method: str = "mean", treat_zero_as_missing: bool = True) -> pd.DataFrame:
    mask = _as_missing_mask(df, treat_zero_as_missing)
    work = df.mask(mask)
    if method == "median":
        fill_vals = work.median(axis=1)
    else:
        fill_vals = work.mean(axis=1)
    return work.apply(lambda row: row.fillna(fill_vals[row.name]), axis=1)


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5, treat_zero_as_missing: bool = True) -> pd.DataFrame:
    """KNN imputation. Samples are treated as observations (rows) for neighbor-finding,
    so the matrix is transposed internally to (samples x features) before imputing."""
    from sklearn.impute import KNNImputer
    mask = _as_missing_mask(df, treat_zero_as_missing)
    work = df.mask(mask)
    X = work.T.values  # samples x features
    n_neighbors = min(n_neighbors, max(1, work.shape[1] - 1))
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed.T, index=df.index, columns=df.columns)


def impute_random_forest(df: pd.DataFrame, n_estimators: int = 10, max_iter: int = 3,
                          treat_zero_as_missing: bool = True, random_state: int = 0) -> pd.DataFrame:
    """MissForest-style imputation via IterativeImputer + RandomForestRegressor."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor
    mask = _as_missing_mask(df, treat_zero_as_missing)
    work = df.mask(mask)
    X = work.T.values
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, max_depth=5, n_jobs=-1, random_state=random_state),
        max_iter=max_iter, random_state=random_state, tol=1e-2,
    )
    X_imputed = imputer.fit_transform(X)
    X_imputed = np.clip(X_imputed, a_min=0, a_max=None)
    return pd.DataFrame(X_imputed.T, index=df.index, columns=df.columns)


def impute_bpca(df: pd.DataFrame, max_iter: int = 5, treat_zero_as_missing: bool = True,
                 random_state: int = 0) -> pd.DataFrame:
    """BPCA-approximated imputation via IterativeImputer + BayesianRidge."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    mask = _as_missing_mask(df, treat_zero_as_missing)
    work = df.mask(mask)
    X = work.T.values
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter, random_state=random_state)
    X_imputed = imputer.fit_transform(X)
    X_imputed = np.clip(X_imputed, a_min=0, a_max=None)  # regression has no positivity constraint
    return pd.DataFrame(X_imputed.T, index=df.index, columns=df.columns)


def impute_qrilc_approx(df: pd.DataFrame, q_min: float = 0.25, shift_factor: float = 1.6,
                         scale_factor: float = 0.4, treat_zero_as_missing: bool = True,
                         random_state: int = 0) -> pd.DataFrame:
    """
    Approximates the left-censored-data assumption behind QRILC: for each feature,
    draws missing values from a normal distribution truncated to values below the
    minimum observed value, with mean shifted down and SD shrunk relative to the
    observed distribution (reflecting that censored/low-abundance values cluster near
    the detection limit with less spread than the full observed range).
    """
    rng = np.random.default_rng(random_state)
    mask = _as_missing_mask(df, treat_zero_as_missing)
    work = df.mask(mask)
    out = work.copy()
    for idx in out.index:
        row = work.loc[idx]
        observed = row.dropna().values
        n_missing = row.isna().sum()
        if n_missing == 0:
            continue
        if len(observed) < 2:
            fill_val = observed.min() / 2 if len(observed) else 0.0
            out.loc[idx] = row.fillna(fill_val)
            continue
        obs_mean, obs_sd = observed.mean(), observed.std(ddof=1) or 1e-6
        censor_point = np.quantile(observed, q_min)
        imputed_mean = obs_mean - shift_factor * obs_sd
        imputed_sd = max(obs_sd * scale_factor, 1e-6)
        a = (-np.inf - imputed_mean) / imputed_sd  # no lower bound
        b = (censor_point - imputed_mean) / imputed_sd  # upper bound = censor point
        draws = truncnorm.rvs(a, b, loc=imputed_mean, scale=imputed_sd, size=n_missing,
                               random_state=rng)
        draws = np.clip(draws, a_min=0, a_max=None)  # protein intensities can't be negative
        filled = row.copy()
        filled[row.isna()] = draws
        out.loc[idx] = filled
    return out


def impute(df: pd.DataFrame, method: str, treat_zero_as_missing: bool = True, **kwargs) -> pd.DataFrame:
    """Dispatcher: method is one of the keys in METHOD_INFO."""
    if method == "Half-Minimum (LOD/2)":
        return impute_half_minimum(df, treat_zero_as_missing)
    elif method == "Mean":
        return impute_mean_median(df, "mean", treat_zero_as_missing)
    elif method == "Median":
        return impute_mean_median(df, "median", treat_zero_as_missing)
    elif method == "K-Nearest Neighbors (KNN)":
        return impute_knn(df, kwargs.get("n_neighbors", 5), treat_zero_as_missing)
    elif method == "Random Forest (MissForest-style)":
        return impute_random_forest(df, kwargs.get("n_estimators", 10), kwargs.get("max_iter", 3),
                                     treat_zero_as_missing)
    elif method == "BPCA (approximated)":
        return impute_bpca(df, kwargs.get("max_iter", 5), treat_zero_as_missing)
    elif method == "QRILC (approximated)":
        return impute_qrilc_approx(df, treat_zero_as_missing=treat_zero_as_missing)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
