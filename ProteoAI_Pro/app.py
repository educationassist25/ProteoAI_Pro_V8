"""
ProteoAI Pro — Automated LC-MS/MS Proteomics Statistical Analysis and Reporting Platform
Streamlit application entry point.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as _plt
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from modules import utils, qc, normalization, imputation_module, stats_analysis, pca_module, volcano, biomarker, heatmap_module, boxplot_module, enrichment, ppi

st.set_page_config(page_title="ProteoAI Pro", layout="wide", page_icon="🧪")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
for key, default in [
    ("raw_peak_df", None), ("meta", None), ("data_type", "Label-Free Proteomics"),
    ("row_annotations", None), ("gene_map", None),
    ("qc_cols", []), ("sample_cols", []), ("istd_normalized", None),
    ("iqr_normalized", None), ("log2_data", None), ("log2_raw", None), ("log2_constant", None),
    ("stats_result", None), ("stats_result_groups", None),
    ("anova_result", None), ("posthoc_result", None), ("anova_groups_used", None),
    ("stats_result_group_col", None),
    ("processing_notes", []), ("qc_figs", {}), ("viz_figs", {}), ("heatmap_fig", None), ("heatmap_zscore", None),
    ("volcano_fig", None), ("volcano_annotated", None), ("volcano_settings", None),
    ("boxplot_fig", None), ("boxplot_stats", None),
    ("missingness_table", None), ("cleaned_data", None), ("imputed_data", None), ("imputation_method_used", None),
    ("raw_peak_df_qc", None), ("cv_table", None), ("qc_log_data", None),
    ("gsea_result", None), ("gsea_gene_sets", None), ("gsea_ranked_scores", None), ("gsea_method_used", None),
    ("ppi_edge_df", None), ("ppi_graph", None), ("ppi_fig", None), ("ppi_summary", None), ("ppi_enrichment_stats", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


class _SingleDatasetState(dict):
    """Adapter so the shared render_cleaning_ui/render_qc_ui/render_normalization_ui
    functions can read/write the flat session_state keys through a dict-like
    interface (kept as a thin indirection layer rather than inlined throughout)."""
    _MAP = {
        "cleaned_data": "cleaned_data", "imputed_data": "imputed_data",
        "missingness_table": "missingness_table", "imputation_method_used": "imputation_method_used",
        "raw_peak_df_qc": "raw_peak_df_qc", "cv_table": "cv_table", "qc_figs": "qc_figs",
        "qc_log_data": "qc_log_data",
        "istd_normalized": "istd_normalized", "iqr_normalized": "iqr_normalized",
        "log2_data": "log2_data", "log2_constant": "log2_constant",
    }

    def __getitem__(self, key):
        if key == "processing_notes":
            return st.session_state.processing_notes
        return st.session_state[self._MAP[key]]

    def __setitem__(self, key, value):
        if key == "processing_notes":
            st.session_state.processing_notes = value
        else:
            st.session_state[self._MAP[key]] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def reset_downstream_analysis_state():
    """
    Clear every downstream-analysis result (Statistics, ANOVA, post-hoc, Biomarkers)
    whenever new data is loaded. Without this, a stale stats_result from a previous
    dataset/mode can persist across a reload, and tabs that read it (Heatmap, Volcano,
    Biomarker Discovery) can then reference a log2 matrix that no longer matches --
    e.g. crashing on mismatched columns -- since those tabs assume stats_result is
    None until freshly (re)computed against the currently-loaded data.
    """
    st.session_state.stats_result = None
    st.session_state.stats_result_groups = None
    st.session_state.stats_result_group_col = None
    st.session_state.anova_result = None
    st.session_state.posthoc_result = None
    st.session_state.anova_groups_used = None
    st.session_state.biomarkers = None


st.title("ProteoAI Pro")
st.caption("LC-MS/MS based Proteomics Statistical Analysis (Label-Free & TMT)")

TABS = st.tabs([
    "**Data Upload**", "**Data Cleaning & Imputation**", "**QC Validation**",
    "**Normalization**", "**PCA**", "**Statistics**",
    "**Volcano Plot**", "**Biomarker Discovery**", "**Heatmap**", "**Boxplot**",
    "**GSEA**", "**P-P Interaction**"
])

# ===========================================================================
# TAB 1 — DATA UPLOAD
# ===========================================================================
with TABS[0]:
    st.header("Data Upload & Study Configuration")
    st.caption(
        "A proteomics experiment here is a single dataset — one protein intensity matrix "
        "plus one metadata table — cleaned, QC'd, normalized, and analyzed straight through."
    )

    st.session_state.data_type = st.selectbox(
        "Quantification method",
        ["Label-Free Proteomics", "TMT Proteomics"],
        help="Label-Free (LFQ): each sample is acquired separately; intensities are "
             "normalized across samples (Median-IQR robust scaling), since there's no "
             "shared internal reference spiked into every run. TMT (Tandem Mass Tag): "
             "samples are chemically labeled and multiplexed into shared MS runs, "
             "typically including a pooled/bridge reference channel — every other "
             "channel is normalized as a ratio to that reference channel."
    )
    is_tmt_demo = st.session_state.data_type == "TMT Proteomics"
    demo_prefix = "tmt" if is_tmt_demo else "labelfree"
    demo_desc = (
        "60 proteins incl. a pooled reference channel, 24 biological samples across 4 groups "
        "(Control/Mild/Moderate/Severe), 6 QC replicates"
        if is_tmt_demo else
        "243 proteins (no reference channel — label-free has no shared spiked standard), "
        "24 biological samples across 4 groups (Control/Mild/Moderate/Severe), 6 QC replicates"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Protein Intensity Matrix")
        st.caption("First column = Protein ID, optional second column = Gene (gene symbol, "
                   "recommended for real-data GSEA/P-P Interaction results), remaining columns "
                   "= sample protein intensities (incl. QC).")
        peak_file = st.file_uploader("Upload protein intensity matrix (CSV/XLSX)", type=["csv", "xlsx"], key="peak_upload")
        use_demo = st.checkbox("Use built-in demo dataset instead", value=(peak_file is None))
        demo_choice = "Standard"
        if use_demo or peak_file is None:
            demo_choice = st.radio(
                "Demo dataset", ["Standard", "Rich Clinical Demo"], horizontal=True,
                key="single_demo_choice"
            )
            if demo_choice == "Standard":
                st.caption(
                    f"Suggested demo for **{st.session_state.data_type}**: {demo_desc}. "
                    f"The matching dataset is selected automatically."
                )
            else:
                st.caption(
                    "200 proteins, 36 biological samples across 4 diagnosis groups "
                    "(Healthy Control, Prediabetic, Type 2 Diabetes, Metabolic Syndrome, "
                    "9 each) + 6 QC replicates + 1 pooled reference channel = 43 total. Metadata includes Diagnosis, "
                    "Age, Gender, Treatment, Ethnicity, and Body Weight — any of these can "
                    "be picked as the grouping/coloring variable throughout the app, not "
                    "just Diagnosis. Also loads protein Method + Pathway row annotations "
                    "for the Heatmap tab's row annotation tracks."
                )
    with col2:
        st.subheader("Sample Metadata")
        st.caption("Columns: Sample, Group, IsQC (True/False), Batch (optional) — any additional "
                   "columns (Diagnosis, Age, Gender, Treatment, Ethnicity, ...) are kept and become "
                   "selectable grouping/coloring variables in PCA, Statistics, Heatmap, and Boxplot.")
        meta_file = st.file_uploader("Upload metadata table (CSV/XLSX)", type=["csv", "xlsx"], key="meta_upload")

    st.subheader("Protein Row Annotations (optional)")
    st.caption("Columns: Protein, then any annotation columns (e.g. Method, Pathway) — enables "
               "the Heatmap tab's row annotation tracks. Not required for PCA/Statistics/Volcano/"
               "Biomarker/Boxplot.")
    row_annot_file = st.file_uploader("Upload row annotation table (CSV/XLSX)",
                                       type=["csv", "xlsx"], key="row_annot_upload")

    if st.button("Load & Validate Data", type="primary"):
        try:
            row_annotations = None
            if use_demo or peak_file is None:
                if demo_choice == "Rich Clinical Demo":
                    peak_path = os.path.join(os.path.dirname(__file__), "sample_protein_intensity_matrix_richdemo.csv")
                    meta_path = os.path.join(os.path.dirname(__file__), "sample_metadata_richdemo.csv")
                    annot_path = os.path.join(os.path.dirname(__file__), "protein_row_annotations_richdemo.csv")
                    peak_raw = pd.read_csv(peak_path)
                    meta_raw = pd.read_csv(meta_path)
                    row_annotations = pd.read_csv(annot_path).set_index("Protein")
                    st.info("Using the built-in **Rich Clinical Demo** dataset (200 proteins, "
                            "43 samples incl. a pooled reference channel, 6 metadata variables, "
                            "Method + Pathway row annotations).")
                else:
                    peak_path = os.path.join(os.path.dirname(__file__), f"sample_protein_intensity_matrix_{demo_prefix}.csv")
                    meta_path = os.path.join(os.path.dirname(__file__), f"sample_metadata_{demo_prefix}.csv")
                    peak_raw = pd.read_csv(peak_path)
                    meta_raw = pd.read_csv(meta_path)
                    st.info(f"Using the built-in **{demo_prefix}** demo dataset, matching your "
                            f"'{st.session_state.data_type}' selection: {demo_desc}.")
            else:
                peak_raw = utils.load_table(peak_file)
                meta_raw = utils.load_table(meta_file) if meta_file is not None else None
                if meta_raw is None:
                    st.error("Please upload a metadata table (or check 'use demo dataset').")
                    st.stop()

            peak_df = utils.validate_peak_matrix(peak_raw)
            meta = utils.validate_metadata(meta_raw, peak_df.columns)
            qc_cols, sample_cols = utils.split_qc_and_samples(peak_df, meta)

            if row_annot_file is not None and row_annotations is None:
                row_annot_raw = utils.load_table(row_annot_file)
                if "Protein" not in row_annot_raw.columns:
                    st.warning("Row annotation file has no 'Protein' column — ignoring it.")
                else:
                    row_annotations = row_annot_raw.set_index("Protein")

            st.session_state.raw_peak_df = peak_df
            st.session_state.meta = meta
            st.session_state.qc_cols = qc_cols
            st.session_state.sample_cols = sample_cols
            st.session_state.row_annotations = row_annotations
            st.session_state.gene_map = peak_df.attrs.get("gene_map")
            st.session_state.processing_notes = [f"Quantification method: {st.session_state.data_type}"]
            st.session_state.log2_data = None
            reset_downstream_analysis_state()

            st.success(f"Loaded {peak_df.shape[0]} proteins × {peak_df.shape[1]} samples "
                       f"({len(qc_cols)} QC, {len(sample_cols)} study samples)."
                       + (f" Row annotations: {', '.join(row_annotations.columns)}."
                          if row_annotations is not None else "")
                       + (f" Gene column detected: {len(st.session_state.gene_map)} protein→gene "
                          "mappings available for the GSEA/P-P Interaction tabs."
                          if st.session_state.gene_map else
                          " No Gene column detected — GSEA/P-P Interaction will use the Protein "
                          "column directly (works only if it already contains gene symbols)."))
        except utils.DataValidationError as e:
            st.error(f"Validation error: {e}")
        except Exception as e:
            st.error(f"Unexpected error loading data: {e}")

    if st.session_state.raw_peak_df is not None:
        st.subheader("Preview")
        st.dataframe(st.session_state.raw_peak_df.head(10), width='stretch')
        st.write("**Metadata:**")
        st.dataframe(st.session_state.meta, width='stretch')
        if st.session_state.get("row_annotations") is not None:
            st.write("**Row Annotations:**")
            st.dataframe(st.session_state.row_annotations.head(10), width='stretch')


# ---------------------------------------------------------------------------
# QC Validation UI for the single-dataset workflow.
# ---------------------------------------------------------------------------
def render_qc_ui(peak_df, qc_cols, sample_cols, key_prefix, state):
    if len(qc_cols) < 2:
        st.warning("At least 2 QC samples are required for CV calculation. Skipping QC module.")
        return

    cv_table = qc.calculate_cv(peak_df[qc_cols])
    state["cv_table"] = cv_table
    n_accept = (cv_table["Quality"] == "Acceptable").sum()
    n_var = (cv_table["Quality"] == "Variable").sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Features", len(cv_table))
    m2.metric("Acceptable (CV≤20%)", n_accept)
    m3.metric("Variable (CV>20%)", n_var)

    st.markdown("**QC CV Distribution**")
    fig_cv_dist = qc.cv_distribution_plot(cv_table)
    st.pyplot(fig_cv_dist)
    utils.render_figure_download(st, fig_cv_dist, f"{key_prefix}_QC_CV_Distribution",
                                            key_prefix=f"{key_prefix}_cvdist")

    st.markdown("**Feature Counts by CV Quality**")
    fig_cv_hist = qc.cv_histogram(cv_table)
    st.pyplot(fig_cv_hist)
    utils.render_figure_download(st, fig_cv_hist, f"{key_prefix}_Feature_Counts_by_Quality",
                                            key_prefix=f"{key_prefix}_cvhist")

    st.markdown("**Protein Intensity Distribution**")
    st.caption("Raw protein-intensity values across all features and samples (QC + biological), log10 scale.")
    fig_peak_hist = qc.protein_intensity_histogram(peak_df, title=f"{key_prefix} — Protein Intensity Distribution")
    st.pyplot(fig_peak_hist)
    utils.render_figure_download(st, fig_peak_hist, f"{key_prefix}_Peak_Area_Histogram",
                                            key_prefix=f"{key_prefix}_peakhist")

    st.subheader("CV Filtering Table")
    st.dataframe(cv_table, width='stretch', height=250)
    st.download_button(
        f"Download {key_prefix.replace(' ', '_')}_CV_Table.csv",
        utils.to_download_bytes_csv(cv_table),
        file_name=f"{key_prefix.replace(' ', '_')}_CV_Table.csv", mime="text/csv",
        key=f"{key_prefix}_dl_cv_table"
    )

    filter_cv = st.checkbox(
        "Filter out 'Variable' features (CV>20%) from downstream analysis",
        value=False, key=f"{key_prefix}_filter_cv"
    )
    st.caption(
        "Unchecked by default — **all features are kept** unless you explicitly opt in to "
        "removing high-CV features before normalization/statistics."
    )

    st.subheader("QC Visualization")
    st.caption(
        "Shown below using **QC replicates only** — biological samples are excluded, since this "
        "module evaluates analytical reproducibility and instrument stability, not biology."
    )
    qc_log, _ = normalization.log2_transform(peak_df[qc_cols])
    state["qc_log_data"] = qc_log
    fig_corr, corr_df = qc.sample_correlation_matrix(qc_log)
    st.pyplot(fig_corr)
    utils.render_figure_download(st, fig_corr, f"{key_prefix}_QC_Sample_Correlation",
                                            key_prefix=f"{key_prefix}_corr")

    if st.button("Confirm QC & Proceed", key=f"{key_prefix}_confirm_qc"):
        if state.get("imputed_data") is not None:
            working_df = state["imputed_data"].copy()
            source_desc = "cleaned & imputed data from Tab 2"
        else:
            working_df = peak_df[sample_cols].copy()
            source_desc = "raw biological samples (no cleaning/imputation was applied in Tab 2)"
        if filter_cv:
            keep = cv_table.index[cv_table["Quality"] == "Acceptable"]
            working_df = working_df.loc[working_df.index.intersection(keep)]
        state["raw_peak_df_qc"] = working_df
        state["qc_figs"] = {"QC CV Distribution": fig_cv_dist, "QC Sample Correlation Matrix": fig_corr}
        state["processing_notes"].append(
            f"QC validation: {len(cv_table)} features assessed via {len(qc_cols)} QC replicates; "
            f"{'variable features (CV>20%) removed' if filter_cv else 'no CV-based filtering applied — all features kept'}, "
            f"applied on top of {source_desc}. "
            f"{len(qc_cols)} QC samples excluded from all downstream normalization/statistics/visualization "
            f"({working_df.shape[0]} features, {len(sample_cols)} biological samples proceed)."
        )
        st.success(
            f"QC step confirmed — {len(qc_cols)} QC samples excluded going forward. "
            f"Proceed to Normalization with {working_df.shape[0]} features across "
            f"{len(sample_cols)} biological samples."
        )


# ===========================================================================
# TAB 3 — QC VALIDATION
# ===========================================================================
with TABS[2]:
    st.header("Quality Control (QC) Validation")
    if st.session_state.raw_peak_df is None:
        st.warning("Load data in Tab 1 first.")
    else:
        render_qc_ui(st.session_state.raw_peak_df, st.session_state.qc_cols,
                     st.session_state.sample_cols, "single", _SingleDatasetState())

# ===========================================================================
# ---------------------------------------------------------------------------
# Reusable Data Cleaning & Imputation UI.
# ---------------------------------------------------------------------------
def render_cleaning_ui(source_df, key_prefix, state):
    treat_zero_as_missing = st.checkbox(
        "Treat exact-zero values as missing (common LC-MS/MS convention for 'not detected')",
        value=True, key=f"{key_prefix}_treat_zero"
    )

    st.subheader("Step 1: Assess & Filter by Missingness")
    st.caption(
        "Guidance: **<20%** missing → keep as-is · **20-50%** → keep, impute carefully · "
        "**50-70%** → usually remove · **>70-80%** → remove unless biologically essential."
    )
    missingness_table = imputation_module.compute_missingness(source_df, treat_zero_as_missing)
    state["missingness_table"] = missingness_table

    m1, m2, m3, m4 = st.columns(4)
    counts = missingness_table["Quality"].value_counts()
    m1.metric("Keep (<20%)", int(counts.get("Keep", 0)))
    m2.metric("Careful imputation (20-50%)", int(counts.get("Keep (careful imputation)", 0)))
    m3.metric("Usually remove (50-70%)", int(counts.get("Usually remove", 0)))
    m4.metric("Remove unless essential (>70%)", int(counts.get("Remove unless essential", 0)))

    fig_miss, ax_miss = _plt.subplots(figsize=(7, 3.5))
    ax_miss.hist(missingness_table["Pct_Missing"], bins=30, color="#4C72B0", edgecolor="white")
    for x, lbl in [(20, "20%"), (50, "50%"), (70, "70%")]:
        ax_miss.axvline(x, color="red", linestyle="--", lw=0.8)
    ax_miss.set_xlabel("% Missing")
    ax_miss.set_ylabel("Number of features")
    ax_miss.set_title("Missingness Distribution Across Features")
    st.pyplot(fig_miss)

    with st.expander("View missingness table"):
        st.dataframe(missingness_table, width='stretch', height=250)

    max_pct_missing = st.slider(
        "Remove features with missingness above this threshold (%)",
        min_value=0, max_value=100, value=50, step=5, key=f"{key_prefix}_max_pct"
    )
    cleaned_preview = imputation_module.filter_by_missingness(source_df, missingness_table, max_pct_missing)
    st.caption(f"At this threshold: **{cleaned_preview.shape[0]} of {source_df.shape[0]}** "
               f"features retained ({source_df.shape[0] - cleaned_preview.shape[0]} removed).")

    if st.button("Apply Missingness Filter", type="primary", key=f"{key_prefix}_apply_filter"):
        state["cleaned_data"] = cleaned_preview
        state["processing_notes"].append(
            f"Missingness filtering: removed {source_df.shape[0] - cleaned_preview.shape[0]} of "
            f"{source_df.shape[0]} features exceeding {max_pct_missing}% missingness; "
            f"{cleaned_preview.shape[0]} features retained."
        )
        st.success(f"Filter applied — {cleaned_preview.shape[0]} features retained.")

    if state.get("cleaned_data") is not None:
        remaining_missing = imputation_module.count_missing(state["cleaned_data"], treat_zero_as_missing)

        st.subheader("Step 2: Choose an Imputation Method")
        if remaining_missing == 0:
            st.success("No missing values remain in the filtered data — imputation isn't needed. "
                       "You can skip ahead.")
            state["imputed_data"] = state["cleaned_data"]
        else:
            st.caption(f"{remaining_missing} missing values remain across the filtered features.")
            method = st.selectbox("Imputation method", list(imputation_module.METHOD_INFO.keys()),
                                   key=f"{key_prefix}_method")
            st.caption(imputation_module.METHOD_INFO[method])

            kwargs = {}
            if method == "K-Nearest Neighbors (KNN)":
                kwargs["n_neighbors"] = st.slider("Number of neighbors (k)", 2, 15, 5, key=f"{key_prefix}_knn_k")
            elif method == "Random Forest (MissForest-style)":
                c1, c2 = st.columns(2)
                kwargs["n_estimators"] = c1.slider("Number of trees", 5, 50, 10, key=f"{key_prefix}_rf_n")
                kwargs["max_iter"] = c2.slider("Iterations", 1, 10, 3, key=f"{key_prefix}_rf_iter")
            elif method == "BPCA (approximated)":
                kwargs["max_iter"] = st.slider("Iterations", 1, 15, 5, key=f"{key_prefix}_bpca_iter")

            if st.button("Apply Imputation", type="primary", key=f"{key_prefix}_apply_impute"):
                with st.spinner(f"Running {method}... this may take a moment for Random Forest or BPCA."):
                    imputed = imputation_module.impute(
                        state["cleaned_data"], method, treat_zero_as_missing=treat_zero_as_missing, **kwargs
                    )
                state["imputed_data"] = imputed
                state["imputation_method_used"] = method
                state["processing_notes"].append(
                    f"Missing value imputation: {method} applied to "
                    f"{state['cleaned_data'].shape[0]} features ({remaining_missing} missing values filled)."
                )
                st.success(f"Imputation complete using {method}.")

        if state.get("imputed_data") is not None:
            st.subheader("Result: Complete Post-Imputation Data")
            st.caption(f"This cleaned & imputed dataset ({state['imputed_data'].shape[0]} features × "
                       f"{state['imputed_data'].shape[1]} samples) now feeds QC validation and "
                       "normalization. Full table (scroll to see all rows):")
            st.dataframe(state["imputed_data"], width='stretch', height=400)
            imputed_filename = (f"{key_prefix.replace(' ', '_')}_Imputed_Data.csv"
                                 if key_prefix != "single" else "Imputed_Data.csv")
            st.download_button(
                f"Download {imputed_filename}",
                utils.to_download_bytes_csv(state["imputed_data"]),
                file_name=imputed_filename, mime="text/csv", key=f"{key_prefix}_dl_imputed"
            )


# ===========================================================================
# TAB 2 — DATA CLEANING & MISSING VALUE IMPUTATION
# ===========================================================================
with TABS[1]:
    st.header("Data Cleaning & Missing Value Imputation")
    st.caption(
        "Optional, but recommended if your data has missing values (encoded as blank/NaN or "
        "zero, both common for 'not detected' in LC-MS/MS protein intensities). Runs on the biological "
        "samples (QC replicates are never included here), before QC validation and normalization."
    )
    if st.session_state.raw_peak_df is None:
        st.warning("Load data in Tab 1 first.")
    else:
        source_df = st.session_state.raw_peak_df[st.session_state.sample_cols]
        render_cleaning_ui(source_df, "single", _SingleDatasetState())

# ===========================================================================
# ---------------------------------------------------------------------------
# Normalization UI for the single-dataset workflow.
# ---------------------------------------------------------------------------
def render_normalization_ui(base_df, data_type, key_prefix, state, meta):
    is_tmt = data_type == "TMT Proteomics"

    if is_tmt:
        method_options = ["Reference-Channel Normalization", "Median Centering Normalization",
                           "Global MAD-based Variance Scaling"]
        state_key_prelog2 = "istd_normalized"
    else:
        method_options = ["IQR Normalization", "Median Centering Normalization",
                           "Global MAD-based Variance Scaling"]
        state_key_prelog2 = "iqr_normalized"

    st.subheader(f"{data_type}: Normalization → Log2 Pipeline")
    method = st.selectbox(
        "Normalization method", method_options, index=0, key=f"{key_prefix}_norm_method",
        help="The first option is the recommended default for this quantification method "
             "(reference-channel ratio for TMT, IQR robust scaling for Label-Free); the other "
             "two are alternative, dataset-wide normalization strategies available for either "
             "quantification method."
    )

    # Clear any stale normalized/log2 data left over from a previously-selected method,
    # so Step 2 never silently shows results from a different normalization approach.
    last_method_key = f"{key_prefix}_last_norm_method"
    if st.session_state.get(last_method_key) != method:
        state[state_key_prelog2] = None
        state["log2_data"] = None
        st.session_state[last_method_key] = method

    ref_choice = None
    if is_tmt and method in ("Median Centering Normalization", "Global MAD-based Variance Scaling"):
        ref_choice = st.selectbox(
            "Select reference/pooled channel (sample column) to exclude from this normalization",
            base_df.columns.tolist(), key=f"{key_prefix}_ref_exclude_choice",
            help="The pooled/bridge reference channel isn't a directly-comparable study sample, "
                 "so it's excluded from the normalization calculation and dropped from the output "
                 "for this method — the same way it's consumed and dropped by Reference-Channel "
                 "Normalization."
        )

    st.markdown("**Step 1: Normalization**")

    if method == "Reference-Channel Normalization":
        st.caption(
            "Formula: **Normalized Protein Intensity = Channel Protein Intensity ÷ Reference "
            "Channel Protein Intensity**. Pick the sample column that is the pooled/bridge "
            "reference channel included in the TMT plex (present in every biological sample's "
            "protein intensity matrix)."
        )
        istd_choice = st.selectbox("Select reference/pooled channel (sample column)",
                                    base_df.columns.tolist(), key=f"{key_prefix}_istd_choice")
        if st.button("Apply Reference-Channel Normalization", key=f"{key_prefix}_apply_istd"):
            try:
                working = normalization.reference_channel_normalize(base_df, istd_choice)
                state[state_key_prelog2] = working
                state["processing_notes"].append(f"Reference-channel normalization applied using '{istd_choice}'.")
                st.success(f"Reference-channel normalization complete using {istd_choice}.")
            except Exception as e:
                st.error(str(e))

    elif method == "IQR Normalization":
        st.caption(
            "Formula: **X_norm = (X − Median(X)) ÷ IQR(X)**, computed per protein across "
            "samples (feature-based) or per sample across proteins (sample-based), applied to "
            "the **raw (QC-validated) protein intensities**. Robust scaling is less sensitive "
            "to outliers than mean/SD scaling — a standard choice for label-free proteomics, "
            "where there's no shared spiked reference across every channel."
        )
        iqr_axis = st.radio("Normalization axis", ["feature", "sample", "batch"], horizontal=True,
                             index=0, key=f"{key_prefix}_iqr_axis")
        if st.button("Apply IQR Normalization", key=f"{key_prefix}_iqr_untargeted_btn"):
            batch_map = meta["Batch"] if iqr_axis == "batch" else None
            scaled = normalization.iqr_normalize(base_df, axis=iqr_axis, batch_map=batch_map)
            state[state_key_prelog2] = scaled
            state["processing_notes"].append(
                f"IQR robust scaling applied to raw abundance ({iqr_axis}-based)."
            )
            st.success(f"IQR normalization complete ({iqr_axis}-based).")

    elif method == "Median Centering Normalization":
        st.caption(
            "Formula: **X_norm = X − Median(Sample) + Grand Median**. For each sample, "
            "subtracts that sample's own median (computed across proteins) so every sample is "
            "recentered to the same level, correcting systematic sample-to-sample loading "
            "offsets; the grand median (median of all per-sample medians) is added back so the "
            "overall scale is preserved rather than collapsed to zero."
            + (" The reference/pooled channel selected above is excluded from the calculation "
               "and dropped from the output." if is_tmt else "")
        )
        exclude_cols = [ref_choice] if (is_tmt and ref_choice) else None
        if st.button("Apply Median Centering Normalization", key=f"{key_prefix}_apply_mediancenter"):
            working = normalization.median_center_normalize(base_df, exclude_cols=exclude_cols)
            state[state_key_prelog2] = working
            state["processing_notes"].append(
                "Median centering normalization applied (per-sample)."
                + (f" Reference channel '{ref_choice}' excluded and dropped." if exclude_cols else "")
            )
            st.success("Median centering normalization complete.")

    elif method == "Global MAD-based Variance Scaling":
        st.caption(
            "Formula: **X_norm = X ÷ (MAD_global × 1.4826)**. A single MAD is computed across "
            "the entire dataset at once (not per-protein or per-sample) and used as one global "
            "scaling factor, standardizing overall variance/spread in one step (the 1.4826 "
            "constant rescales MAD to be comparable to a standard deviation)."
            + (" The reference/pooled channel selected above is excluded from the calculation "
               "and dropped from the output." if is_tmt else "")
        )
        exclude_cols = [ref_choice] if (is_tmt and ref_choice) else None
        if st.button("Apply Global MAD-based Variance Scaling", key=f"{key_prefix}_apply_madscale"):
            working, mad_scale_used = normalization.mad_scale_normalize(base_df, exclude_cols=exclude_cols)
            state[state_key_prelog2] = working
            state["processing_notes"].append(
                f"Global MAD-based variance scaling applied (scale factor = {mad_scale_used:.4g})."
                + (f" Reference channel '{ref_choice}' excluded and dropped." if exclude_cols else "")
            )
            st.success(f"Global MAD-based variance scaling complete (scale factor = {mad_scale_used:.4g}).")

    working = state.get(state_key_prelog2)

    st.markdown("**Step 2: Log2 Transformation**")
    if working is None:
        st.info("Run Step 1 (Normalization) above first.")
    else:
        # Pure-scaling methods (Reference-Channel, Global MAD) never introduce negative
        # values from originally-positive protein intensities, so they use the
        # zero-replacement log2. Centering methods (IQR, Median Centering) frequently
        # produce negative values, which need a positivity shift before logging instead.
        needs_shift = method in ("IQR Normalization", "Median Centering Normalization")
        st.caption(
            ("Formula: **Log2(Normalized Value)**, shifted by a small constant just large "
             "enough to make the global minimum slightly positive first (this method's "
             "output is frequently negative, and log2 is undefined for negative numbers)."
             if needs_shift else
             "Formula: **Log2(Normalized Protein Intensity)**, with automatic zero replacement.")
        )
        if st.button("Apply Log2 Transformation", key=f"{key_prefix}_log2_btn"):
            if needs_shift:
                log2_df, const = normalization.shift_and_log2_transform(working)
                const_label = "positivity shift"
            else:
                log2_df, const = normalization.log2_transform(working)
                const_label = "constant"
            state["log2_data"] = log2_df
            state["log2_constant"] = const
            state["processing_notes"].append(
                f"Log2 transformation applied to {method.lower()} output ({const_label} = {const:.4g})."
            )
            st.success(f"Log2 transformation complete ({const_label} = {const:.4g}).")
            fig_dist = normalization.distribution_plots(working, log2_df)
            st.pyplot(fig_dist)

    prelog2_data = state.get("istd_normalized") if state.get("istd_normalized") is not None else state.get("iqr_normalized")
    if prelog2_data is not None or state.get("log2_data") is not None:
        st.markdown("**Current Working Matrix**")
        file_prefix = "" if key_prefix == "single" else f"{key_prefix.replace(' ', '_')}_"

        if prelog2_data is not None:
            st.markdown("*Normalized — Without Log2 Transformation*")
            st.dataframe(prelog2_data, width='stretch', height=300)
            st.download_button(
                f"Download {file_prefix}Normalized_NoLog2.csv",
                utils.to_download_bytes_csv(prelog2_data),
                file_name=f"{file_prefix}Normalized_NoLog2.csv", mime="text/csv",
                key=f"{key_prefix}_dl_norm_nolog2"
            )
        else:
            st.info("Run Step 1 (normalization) above to enable the pre-Log2 download.")

        if state.get("log2_data") is not None:
            st.markdown("*Normalized — With Log2 Transformation*")
            st.dataframe(state["log2_data"], width='stretch', height=300)
            st.download_button(
                f"Download {file_prefix}Normalized_Log2.csv",
                utils.to_download_bytes_csv(state["log2_data"]),
                file_name=f"{file_prefix}Normalized_Log2.csv", mime="text/csv",
                key=f"{key_prefix}_dl_norm_log2"
            )
        else:
            st.info("Run Step 2 (Log2 transformation) above to enable the Log2 download.")


# ===========================================================================
# TAB 4 — NORMALIZATION
# ===========================================================================
with TABS[3]:
    if st.session_state.raw_peak_df is None:
        st.warning("Load data in Tab 1 first.")
    else:
        if st.session_state.get("raw_peak_df_qc") is not None:
            base_df = st.session_state.raw_peak_df_qc
        else:
            base_df = st.session_state.raw_peak_df[st.session_state.sample_cols]
            st.info("QC tab wasn't confirmed yet — proceeding with biological samples only "
                     "(QC columns excluded automatically). Visit Tab 3 to review QC first.")
        render_normalization_ui(base_df, st.session_state.data_type, "single",
                                 _SingleDatasetState(), st.session_state.meta)

# ===========================================================================
# TAB 6 — STATISTICS
# ===========================================================================
with TABS[5]:
    st.header("Statistical Comparison")
    if st.session_state.log2_data is None:
        st.warning("Complete the Normalization tab (Log2 transform) first.")
    else:
        log2_df = st.session_state.log2_data
        meta = st.session_state.meta
        sample_cols = log2_df.columns.tolist()
        stats_cat_cols = utils.get_categorical_metadata_columns(meta, sample_cols)
        grouping_var = st.selectbox(
            "Grouping variable:", stats_cat_cols,
            index=stats_cat_cols.index("Group") if "Group" in stats_cat_cols else 0,
            key="stats_grouping_var"
        )
        groups_available = meta.loc[meta.index.intersection(sample_cols), grouping_var].unique().tolist()

        st.caption(
            "All statistics below (mean abundance, fold change, p-value, FDR, 95% CI) are computed "
            "from the log2-transformed, normalized data — raw protein intensities are not used for inference."
        )

        mode = st.radio("Comparison type", ["Two-group comparison", "ANOVA (≥3 groups)"], horizontal=True)

        if mode == "Two-group comparison":
            c1, c2, c3 = st.columns(3)
            group_a = c1.selectbox("Group A", groups_available, index=0)
            group_b = c2.selectbox("Group B", groups_available, index=min(1, len(groups_available) - 1))
            method = c3.selectbox("Method", ["Student's t-test", "Wilcoxon rank-sum"])
            method_key = "ttest" if method.startswith("Student") else "wilcoxon"

            if st.button("Run Two-Group Test"):
                a_samples = meta.index[meta[grouping_var] == group_a].tolist()
                b_samples = meta.index[meta[grouping_var] == group_b].tolist()
                a_samples = [s for s in a_samples if s in log2_df.columns]
                b_samples = [s for s in b_samples if s in log2_df.columns]

                result = stats_analysis.complete_statistical_table(log2_df, a_samples, b_samples)
                st.session_state.stats_result = result
                st.session_state.stats_result_groups = (group_a, group_b)
                st.session_state.stats_result_group_col = grouping_var
                st.session_state.processing_notes.append(
                    f"Statistical comparison ({grouping_var}): {group_a} vs {group_b} using {method} "
                    f"(BH-FDR correction); {result['Significant'].sum()} significant proteins "
                    f"(p<0.05 & FDR<0.25)."
                )
                st.success(f"Test complete: {result['Significant'].sum()} significant proteins found.")

            if st.session_state.stats_result is not None and st.session_state.stats_result_groups is not None:
                result = st.session_state.stats_result
                g_a, g_b = st.session_state.stats_result_groups
                comparison_name = f"{g_a}_vs_{g_b}"
                st.dataframe(result, width='stretch', height=400)
                st.download_button(
                    f"Download {comparison_name}_Statistics.csv",
                    utils.to_download_bytes_csv(result),
                    file_name=f"{comparison_name}_Statistics.csv", mime="text/csv",
                    key="dl_stats_twogroup"
                )
                st.caption(f"Filename includes the comparison ({comparison_name}) so results from "
                           "different comparisons stay distinguishable.")

        else:
            if len(groups_available) < 3:
                st.warning(f"Need at least 3 values of '{grouping_var}' (excluding QC) for ANOVA. "
                           "Pick a different grouping variable, or add more groups in metadata.")
            else:
                anova_groups = st.multiselect(
                    f"{grouping_var} values to include in ANOVA (select 3 or more)",
                    groups_available, default=groups_available, key="anova_group_select"
                )
                posthoc_method = st.selectbox("Post-hoc test", ["tukey", "dunnett", "pairwise"])

                if len(anova_groups) < 3:
                    st.warning(f"Select at least 3 groups to run ANOVA (currently {len(anova_groups)} selected).")
                elif st.button("Run ANOVA"):
                    group_map = meta.loc[sample_cols, grouping_var]
                    group_map = group_map[group_map.isin(anova_groups)]
                    group_map = group_map[group_map.index.isin(log2_df.columns)]
                    anova_table, posthoc_results = stats_analysis.anova_test(
                        log2_df[group_map.index], group_map, posthoc=posthoc_method
                    )
                    st.session_state.anova_result = anova_table
                    st.session_state.posthoc_result = posthoc_results
                    st.session_state.anova_groups_used = list(anova_groups)
                    st.session_state.stats_result_group_col = grouping_var
                    st.session_state.processing_notes.append(
                        f"One-way ANOVA across {len(anova_groups)} selected {grouping_var} values "
                        f"({', '.join(anova_groups)}) with {posthoc_method} post-hoc; "
                        f"{(anova_table['FDR'] < 0.25).sum()} significant proteins (FDR<0.25)."
                    )
                    st.success(f"ANOVA complete: {(anova_table['FDR'] < 0.25).sum()} significant proteins (FDR<0.25).")

                if st.session_state.anova_result is not None:
                    anova_table = st.session_state.anova_result
                    anova_groups_used = st.session_state.get("anova_groups_used") or anova_groups
                    anova_comparison_name = "ANOVA_" + "_vs_".join(anova_groups_used)
                    st.dataframe(anova_table, width='stretch', height=350)
                    st.download_button(
                        f"Download {anova_comparison_name}.csv",
                        utils.to_download_bytes_csv(anova_table),
                        file_name=f"{anova_comparison_name}.csv", mime="text/csv",
                        key="dl_stats_anova"
                    )
                    st.caption(f"Filename includes the groups compared ({', '.join(anova_groups_used)}) "
                               "so results from different ANOVA runs stay distinguishable.")

                    posthoc_results = st.session_state.get("posthoc_result")
                    if posthoc_results:
                        feat_choice = st.selectbox("View post-hoc results for feature:", list(posthoc_results.keys()))
                        st.dataframe(posthoc_results[feat_choice], width='stretch')
                        st.download_button(
                            f"Download {anova_comparison_name}_Posthoc_{feat_choice}.csv",
                            utils.to_download_bytes_csv(posthoc_results[feat_choice]),
                            file_name=f"{anova_comparison_name}_Posthoc_{feat_choice}.csv", mime="text/csv",
                            key="dl_stats_posthoc_one"
                        )
                        all_posthoc = pd.concat(
                            [df.assign(Feature=feat) for feat, df in posthoc_results.items()],
                            ignore_index=True
                        )
                        st.download_button(
                            f"Download {anova_comparison_name}_Posthoc_All_Features.csv",
                            utils.to_download_bytes_csv(all_posthoc),
                            file_name=f"{anova_comparison_name}_Posthoc_All_Features.csv", mime="text/csv",
                            key="dl_stats_posthoc_all"
                        )

# ===========================================================================
# TAB 7 — PCA
# ===========================================================================
with TABS[4]:
    st.header("PCA Visualization (Biological Samples Only)")
    if st.session_state.log2_data is None:
        st.warning("Complete the Normalization tab (Log2 transform) first.")
    else:
        log2_df_full = st.session_state.log2_data
        meta = st.session_state.meta
        pca_cat_cols = utils.get_categorical_metadata_columns(meta, log2_df_full.columns)
        pca_group_col = st.selectbox(
            "Color/group samples by:", pca_cat_cols,
            index=pca_cat_cols.index("Group") if "Group" in pca_cat_cols else 0,
            key="pca_group_col"
        )
        groups_all_pca = meta.loc[meta.index.intersection(log2_df_full.columns), pca_group_col].unique().tolist()

        st.subheader("Select Groups")
        selected_pca_groups = st.multiselect(
            f"{pca_group_col} values to include in PCA (choose any subset — 2, 3, 4, or more)",
            groups_all_pca, default=groups_all_pca, key="pca_group_select"
        )
        pca_cols = [c for c in log2_df_full.columns if meta.loc[c, pca_group_col] in selected_pca_groups]
        log2_df = log2_df_full[pca_cols]

        if len(selected_pca_groups) < 1 or len(pca_cols) < 3:
            st.warning("Select at least one group with enough samples (≥3 total) to run PCA.")
        else:
            n_comp = st.slider("Number of components", 2, min(10, log2_df.shape[1] - 1 if log2_df.shape[1] > 2 else 2), 5)

            st.subheader("Customize Appearance")
            c1, c2 = st.columns(2)
            color_mode = c1.radio("Color mode", ["Preset palette", "Custom colors (pick each group)"],
                                   key="pca_color_mode")
            show_ellipse = c2.checkbox("Show 95% confidence ellipses", value=True)

            if color_mode == "Preset palette":
                palette_choice = st.selectbox("Color palette", list(pca_module.PALETTES.keys()))
            else:
                st.caption("Pick an exact color for each group with the color picker below.")
                default_swatches = pca_module.PALETTES["Default (tab10)"]
                palette_choice = {}
                pick_cols = st.columns(min(4, len(selected_pca_groups)) or 1)
                for i, g in enumerate(selected_pca_groups):
                    with pick_cols[i % len(pick_cols)]:
                        palette_choice[g] = st.color_picker(
                            f"Color: {g}", default_swatches[i % len(default_swatches)], key=f"pca_color_{g}"
                        )

            st.caption("Optional: assign a marker style per group (defaults to circles for all).")
            marker_map = {}
            marker_cols = st.columns(min(4, len(selected_pca_groups)) or 1)
            for i, g in enumerate(selected_pca_groups):
                with marker_cols[i % len(marker_cols)]:
                    marker_map[g] = st.selectbox(f"Marker: {g}", pca_module.MARKER_STYLES, key=f"marker_{g}")

            pca, scores_df, cols = pca_module.run_pca(log2_df, n_components=n_comp)
            fig_score = pca_module.pca_score_plot(pca, scores_df, meta, palette=palette_choice,
                                                    marker_map=marker_map, show_ellipse=show_ellipse,
                                                    group_col=pca_group_col)
            fig_loading, top_loadings = pca_module.pca_loading_plot(pca, log2_df.index, top_n=20)
            fig_var = pca_module.pca_variance_plot(pca)

            st.session_state.viz_figs["PCA Score Plot"] = fig_score
            st.session_state.viz_figs["PCA Loading Plot"] = fig_loading
            st.session_state.viz_figs["PCA Variance Plot"] = fig_var

            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_score)
                utils.render_figure_download(st, fig_score, "PCA_Score_Plot", key_prefix="pca_score")
            with c2:
                st.pyplot(fig_loading)
                utils.render_figure_download(st, fig_loading, "PCA_Loading_Plot", key_prefix="pca_loading")
            st.pyplot(fig_var)
            utils.render_figure_download(st, fig_var, "PCA_Variance_Plot", key_prefix="pca_var")

            st.subheader("Top Contributing Proteins (Loadings)")
            st.dataframe(top_loadings, width='stretch')

            st.divider()
            st.subheader("Download PCA Data")
            st.caption(
                "The underlying data behind the plots above: sample scores per principal "
                "component (with group assignment), % variance explained per component, "
                "the protein loadings, and the exact normalized data matrix PCA was run on."
            )
            explained_var_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                "Variance Explained (%)": (pca.explained_variance_ratio_ * 100).round(3),
                "Cumulative Variance Explained (%)": (np.cumsum(pca.explained_variance_ratio_) * 100).round(3),
            })
            scores_with_group = scores_df.copy()
            scores_with_group.insert(0, pca_group_col, meta.loc[scores_with_group.index, pca_group_col].values)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Download PCA_Scores.csv", utils.to_download_bytes_csv(scores_with_group),
                file_name="PCA_Scores.csv", mime="text/csv", key="dl_pca_scores"
            )
            dl2.download_button(
                "Download PCA_Explained_Variance.csv", utils.to_download_bytes_csv(explained_var_df),
                file_name="PCA_Explained_Variance.csv", mime="text/csv", key="dl_pca_variance"
            )
            dl3, dl4 = st.columns(2)
            dl3.download_button(
                "Download PCA_Loadings.csv", utils.to_download_bytes_csv(top_loadings),
                file_name="PCA_Loadings.csv", mime="text/csv", key="dl_pca_loadings"
            )
            dl4.download_button(
                "Download PCA_Input_Data.csv", utils.to_download_bytes_csv(log2_df),
                file_name="PCA_Input_Data.csv", mime="text/csv", key="dl_pca_input"
            )

# ===========================================================================
# TAB 9 — VOLCANO PLOT
# ===========================================================================
with TABS[6]:
    st.header("Volcano Plot")
    if st.session_state.stats_result is None:
        st.warning("Run a two-group statistical comparison in the Statistics tab first.")
    else:
        result = st.session_state.stats_result
        groups_used = st.session_state.get("stats_result_groups")
        group_col_used = st.session_state.get("stats_result_group_col") or "Group"
        if groups_used:
            st.caption(
                f"Reflects the two {group_col_used} values compared in the Statistics tab: "
                f"**{groups_used[0]}** vs **{groups_used[1]}**."
                + " To compare a different pair or variable, go back to Statistics and "
                "re-run the two-group test."
            )

        # ---- 1. Thresholds ----
        with st.expander("1 — Statistical Thresholds", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            y_metric_label = c1.radio("Y-axis metric", ["p-value", "FDR"], horizontal=False)
            y_metric = "pvalue" if y_metric_label == "p-value" else "fdr"
            sig_cutoff = c2.number_input("Significance cutoff", value=0.05 if y_metric == "pvalue" else 0.25,
                                          min_value=0.0001, max_value=1.0, step=0.01)
            fc_threshold = c3.number_input("Fold change threshold (log2 units)", value=1.0,
                                            min_value=0.0, max_value=5.0, step=0.1)
            use_fdr_secondary = c4.checkbox("Also require FDR <", value=False)
            fdr_cutoff = c4.number_input("Secondary FDR cutoff", value=0.25, min_value=0.0001, max_value=1.0,
                                          step=0.01, disabled=not use_fdr_secondary) if use_fdr_secondary else None
            c5, c6 = st.columns(2)
            use_fc_axis = c5.checkbox("Show linear FC on x-axis instead of log2FC", value=False)
            threshold_line_style = c6.selectbox("Threshold line style", list(volcano.LINE_STYLES.keys()))

        # ---- 2 & 3. Colors and point style ----
        with st.expander("2-3 — Colors & Point Style"):
            c1, c2 = st.columns(2)
            palette = c1.selectbox("Color palette (colorblind-friendly options included)",
                                    list(volcano.COLORBLIND_PALETTES.keys()))
            override_colors = c2.checkbox("Override individual colors", value=False)
            up_color = down_color = ns_color = None
            if override_colors:
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    up_color = st.color_picker("Upregulated color", volcano.COLORBLIND_PALETTES[palette]["up"], key="volcano_up_color")
                with cc2:
                    down_color = st.color_picker("Downregulated color", volcano.COLORBLIND_PALETTES[palette]["down"], key="volcano_down_color")
                with cc3:
                    ns_color = st.color_picker("Not significant color", volcano.COLORBLIND_PALETTES[palette]["ns"], key="volcano_ns_color")
            c3, c4, c5 = st.columns(3)
            alpha = c3.slider("Point transparency (alpha)", 0.2, 1.0, 0.75, 0.05)
            point_size = c4.slider("Point size", 1, 100, 14, 1)
            point_shape = c5.selectbox("Point shape", list(volcano.MARKER_SHAPES.keys()))
            c6, c7 = st.columns(2)
            edge_width = c6.slider("Point border width", 0.0, 2.0, 0.0, 0.1)
            edge_color = "none"
            if edge_width > 0:
                with c7:
                    edge_color = st.color_picker("Point border color", "#000000", key="volcano_edge_color")

        # ---- 4. Labels ----
        with st.expander("4 — Protein Labels"):
            label_mode_label = st.radio(
                "Label mode",
                ["Top N significant", "All significant", "Manually selected", "None"],
                horizontal=True
            )
            label_mode = {"Top N significant": "top_n", "All significant": "significant_only",
                          "Manually selected": "manual", "None": "none"}[label_mode_label]
            top_label_n = 10
            manual_labels = []
            if label_mode == "top_n":
                top_label_n = st.number_input("Label top N proteins", value=10, min_value=0, max_value=100, step=1)
            elif label_mode == "manual":
                manual_labels = st.multiselect("Search and select proteins to label by name",
                                                result.index.tolist())
                st.caption("Note: labeling by HMDB ID isn't available — this dataset identifies "
                           "proteins by name only (no compound-database annotation).")
            c1, c2, c3 = st.columns(3)
            label_font_size = c1.slider("Label font size", 5.0, 16.0, 7.5, 0.5)
            label_bold = c2.checkbox("Bold labels", value=False)
            label_italic = c3.checkbox("Italic labels", value=False)
            c4, c5 = st.columns(2)
            override_label_color = c4.checkbox("Override label color (default: match point color)", value=False)
            label_color = None
            if override_label_color:
                with c5:
                    label_color = st.color_picker("Label color", "#333333", key="volcano_label_color")
            repel_labels = st.checkbox("Repel overlapping labels automatically", value=True)

        # ---- 5. Legend ----
        with st.expander("5 — Legend"):
            c1, c2 = st.columns(2)
            legend_position = c1.selectbox("Legend position", ["Right", "Left", "Top", "Bottom", "Hidden"])
            legend_format = c2.radio("Legend label format",
                                      ["Full (\"Upregulated (n=5)\")", "Short (\"Up (5)\")"], horizontal=False)
            legend_format_key = "full" if legend_format.startswith("Full") else "short"

        # ---- 6 & 7. Axes and title ----
        with st.expander("6-7 — Axes & Title"):
            st.markdown("**Axes**")
            c1, c2, c3 = st.columns(3)
            axis_font_size = c1.slider("Axis font size", 8.0, 20.0, 12.0, 0.5)
            axis_bold = c2.checkbox("Bold axis titles", value=False)
            y_decimals = c3.selectbox("Y-axis decimals", [None, 0, 1, 2, 3], index=0,
                                      format_func=lambda x: "Auto" if x is None else str(x))
            c4, c5 = st.columns(2)
            custom_x_limits = c4.checkbox("Set custom X-axis limits", value=False)
            x_limits = None
            if custom_x_limits:
                xlo, xhi = st.columns(2)
                x_limits = (xlo.number_input("X min", value=-6.0), xhi.number_input("X max", value=6.0))
            custom_y_limits = c5.checkbox("Set custom Y-axis limits", value=False)
            y_limits = None
            if custom_y_limits:
                ylo, yhi = st.columns(2)
                y_limits = (ylo.number_input("Y min", value=0.0), yhi.number_input("Y max", value=10.0))
            x_tick_spacing = st.number_input("X-axis tick spacing (0 = auto)", value=0.0, min_value=0.0, step=0.5)
            x_tick_spacing = x_tick_spacing or None

            st.markdown("**Title**")
            c6, c7 = st.columns(2)
            default_title = f"{groups_used[0]} vs {groups_used[1]}" if groups_used else "Volcano Plot"
            custom_title = c6.text_input("Title", value=default_title)
            subtitle = c7.text_input("Subtitle (optional)", value="")
            c8, c9, c10, c11 = st.columns(4)
            title_bold = c8.checkbox("Bold title", value=True)
            title_italic = c9.checkbox("Italic title", value=False)
            title_align = c10.selectbox("Title alignment", ["center", "left", "right"])
            hide_title = c11.checkbox("Hide title", value=False)

        # ---- 8 & 9. Gridlines and threshold line style ----
        with st.expander("8-9 — Gridlines & Threshold Line Style"):
            c1, c2, c3 = st.columns(3)
            grid_mode = c1.selectbox("Gridlines", ["none", "major", "both"],
                                     format_func=lambda x: {"none": "No grid", "major": "Major only",
                                                              "both": "Major + minor"}[x])
            with c2:
                grid_color = st.color_picker("Grid color", "#D9D9D9", key="volcano_grid_color")
            grid_style = c3.selectbox("Grid line style", list(volcano.LINE_STYLES.keys()), index=0)
            c4, c5 = st.columns(2)
            with c4:
                threshold_line_color = st.color_picker("Threshold line color", "#808080", key="volcano_threshold_color")
            threshold_line_width = c5.slider("Threshold line width", 0.2, 3.0, 0.7, 0.1)

        # ---- 10. Highlight specific proteins ----
        with st.expander("10 — Highlight Specific Proteins"):
            highlight_names = st.multiselect(
                "Search and select proteins to highlight (e.g. known markers)",
                result.index.tolist()
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                highlight_color = st.color_picker("Highlight color", "#FFD700", key="volcano_highlight_color")
            highlight_size_mult = c2.slider("Highlight size multiplier", 1.0, 5.0, 2.0, 0.25)
            highlight_shape = c3.selectbox("Highlight shape", ["Star", "Circle", "Triangle", "Square", "Diamond"])

        # ---- 12 & 13. Background and figure size ----
        with st.expander("12-13 — Background & Figure Size"):
            c1, c2 = st.columns(2)
            background_choice = c1.selectbox("Background", ["White", "Transparent", "Gray", "Custom"])
            background = {"White": "white", "Transparent": "transparent", "Gray": "gray"}.get(background_choice)
            if background_choice == "Custom":
                with c2:
                    background = st.color_picker("Custom background color", "#FFFFFF", key="volcano_bg_color")
            c3, c4 = st.columns(2)
            size_preset = c3.selectbox("Figure size preset", list(volcano.FIGURE_SIZE_PRESETS.keys()), index=2)
            if size_preset == "Custom":
                cw, ch = st.columns(2)
                fig_width_in = cw.number_input("Width (in)", value=7.2, min_value=2.0, max_value=20.0)
                fig_height_in = ch.number_input("Height (in)", value=6.4, min_value=2.0, max_value=20.0)
            else:
                fig_width_in, fig_height_in = volcano.FIGURE_SIZE_PRESETS[size_preset]

        # ---- 15. Statistics box, 18. Theme ----
        with st.expander("Statistics Display & Publication Theme"):
            c1, c2 = st.columns(2)
            show_stats_box = c1.checkbox("Show statistics box on figure (total/up/down/cutoffs)", value=False)
            theme_choice = c2.selectbox("Publication theme (stylistic approximation, not an official spec)",
                                        list(volcano.THEMES.keys()))

        fig_volc, annotated = volcano.volcano_plot(
            result,
            use_fc_not_log2=use_fc_axis, fc_threshold=fc_threshold, y_metric=y_metric,
            sig_cutoff=sig_cutoff, fdr_cutoff=fdr_cutoff, threshold_line_style=threshold_line_style,
            palette=palette, up_color=up_color, down_color=down_color, ns_color=ns_color,
            alpha=alpha, edge_color=edge_color, edge_width=edge_width,
            point_size=point_size, point_shape=point_shape,
            label_mode=label_mode, top_label_n=top_label_n, manual_labels=manual_labels,
            label_font_size=label_font_size, label_color=label_color, label_bold=label_bold,
            label_italic=label_italic, repel_labels=repel_labels,
            legend_position=legend_position, legend_format=legend_format_key,
            axis_font_size=axis_font_size, axis_bold=axis_bold,
            x_limits=x_limits, y_limits=y_limits, x_tick_spacing=x_tick_spacing, y_decimals=y_decimals,
            title=custom_title, subtitle=subtitle or None, title_bold=title_bold, title_italic=title_italic,
            title_align=title_align, hide_title=hide_title,
            grid_mode=grid_mode, grid_style=grid_style, grid_color=grid_color,
            threshold_line_color=threshold_line_color, threshold_line_width=threshold_line_width,
            highlight_names=highlight_names, highlight_color=highlight_color,
            highlight_size_mult=highlight_size_mult, highlight_shape=highlight_shape,
            background=background, fig_width_in=fig_width_in, fig_height_in=fig_height_in,
            show_stats_box=show_stats_box, theme=theme_choice,
        )
        st.session_state.viz_figs[f"Volcano Plot ({y_metric_label})"] = fig_volc
        st.session_state.volcano_fig = fig_volc
        st.session_state.volcano_annotated = annotated
        st.session_state.volcano_settings = {
            "y_metric": y_metric, "sig_cutoff": sig_cutoff, "fc_threshold": fc_threshold,
            "fdr_cutoff": fdr_cutoff, "palette": palette, "point_size": point_size,
            "point_shape": point_shape, "label_mode": label_mode, "legend_position": legend_position,
            "theme": theme_choice, "groups": groups_used,
        }
        st.pyplot(fig_volc)

        sig_col = "p-value" if y_metric == "pvalue" else "FDR"

        st.subheader("Top 20 Biomarkers")
        st.dataframe(volcano.top_biomarker_labels(annotated, sig_col=sig_col, n=20), width='stretch')

        st.subheader("Export")
        exp_tab1, exp_tab2 = st.tabs(["Figure", "Data & Settings"])

        with exp_tab1:
            ec1, ec2, ec3 = st.columns(3)
            export_fmt = ec1.selectbox("Format", ["PNG", "PDF", "SVG", "EPS", "JPEG", "TIFF"],
                                        key="volcano_export_fmt")
            export_dpi = ec2.selectbox("Resolution (DPI)", [300, 600, 1200], index=0,
                                        disabled=export_fmt in ("PDF", "SVG", "EPS"), key="volcano_export_dpi")
            mime_map = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml",
                        "EPS": "application/postscript", "JPEG": "image/jpeg", "TIFF": "image/tiff"}
            ext_map = {"PNG": "png", "PDF": "pdf", "SVG": "svg", "EPS": "eps", "JPEG": "jpg", "TIFF": "tiff"}
            fmt_key = "jpeg" if export_fmt == "JPEG" else export_fmt.lower()
            file_bytes = volcano.export_figure(fig_volc, fmt=fmt_key, dpi=export_dpi)
            with ec3:
                st.write("")
                st.write("")
                st.download_button(
                    f"Download VolcanoPlot.{ext_map[export_fmt]}", file_bytes,
                    file_name=f"VolcanoPlot.{ext_map[export_fmt]}", mime=mime_map[export_fmt]
                )

        with exp_tab2:
            dc1, dc2, dc3 = st.columns(3)
            up_tbl = volcano.get_direction_table(annotated, "Up")
            down_tbl = volcano.get_direction_table(annotated, "Down")
            sig_tbl = volcano.get_direction_table(annotated, None)
            with dc1:
                st.download_button("Download Upregulated.csv", utils.to_download_bytes_csv(up_tbl),
                                    "Upregulated_Proteins.csv", "text/csv")
            with dc2:
                st.download_button("Download Downregulated.csv", utils.to_download_bytes_csv(down_tbl),
                                    "Downregulated_Proteins.csv", "text/csv")
            with dc3:
                st.download_button("Download Complete_Volcano_Data.csv", utils.to_download_bytes_csv(annotated),
                                    "Complete_Volcano_Data.csv", "text/csv")
            st.caption("Figure settings (for reproducibility):")
            settings_json = volcano.export_settings_json(st.session_state.volcano_settings)
            st.download_button("Download Figure_Settings.json", settings_json.encode(),
                                "Volcano_Figure_Settings.json", "application/json")

# ===========================================================================
# TAB 10 — BIOMARKER DISCOVERY
# ===========================================================================
with TABS[7]:
    st.header("Biomarker Discovery")
    if st.session_state.stats_result is None:
        st.warning("Run a two-group statistical comparison in the Statistics tab first.")
    else:
        result = st.session_state.stats_result
        groups_used = st.session_state.get("stats_result_groups")
        group_col_used = st.session_state.get("stats_result_group_col") or "Group"
        if groups_used:
            st.caption(
                f"Reflects the two {group_col_used} values compared in the Statistics tab: "
                f"**{groups_used[0]}** vs **{groups_used[1]}**."
                + " To compare a different pair or variable, go back to Statistics and "
                "re-run the two-group test."
            )
        c1, c2, c3 = st.columns(3)
        criterion = c1.selectbox("Filtering criterion", ["combined", "pvalue", "fdr"],
                                  format_func=lambda x: {"combined": "p<0.05 AND FDR<0.25",
                                                          "pvalue": "p<0.05 only", "fdr": "FDR<0.25 only"}[x])
        p_cutoff = c2.number_input("p-value cutoff", value=0.05, min_value=0.0001, max_value=1.0, step=0.01)
        fdr_cutoff = c3.number_input("FDR cutoff", value=0.25, min_value=0.0001, max_value=1.0, step=0.01)

        if st.button("Discover Biomarkers"):
            biomarkers = biomarker.discover_biomarkers(result, criterion=criterion,
                                                        p_cutoff=p_cutoff, fdr_cutoff=fdr_cutoff)
            st.session_state.biomarkers = biomarkers
            st.success(f"{len(biomarkers)} biomarker candidates identified.")

        if st.session_state.get("biomarkers") is not None:
            biomarkers = st.session_state.biomarkers
            st.dataframe(biomarkers, width='stretch', height=450)
            comparison_name = (f"{groups_used[0]}_vs_{groups_used[1]}" if groups_used else "Comparison")
            biomarker_filename = f"{comparison_name}_Biomarkers.csv"
            st.download_button(
                f"Download {biomarker_filename}", utils.to_download_bytes_csv(biomarkers),
                file_name=biomarker_filename, mime="text/csv", key="dl_biomarkers"
            )
            st.caption(f"Filename includes the comparison ({comparison_name}) so results from "
                       "different comparisons stay distinguishable.")

# ===========================================================================
# TAB 11 — HEATMAP
# ===========================================================================
with TABS[8]:
    st.header("Clustered Heatmap")
    if st.session_state.stats_result is None:
        st.warning(
            "Run a two-group comparison in the Statistics tab first — the heatmap needs "
            "that result to know which proteins are significant."
        )
    else:
        log2_df_full = st.session_state.log2_data
        result = st.session_state.stats_result
        meta = st.session_state.meta

        heat_cat_cols = utils.get_categorical_metadata_columns(meta, log2_df_full.columns)
        default_group_col = st.session_state.get("stats_result_group_col") or "Group"
        heat_group_col = st.selectbox(
            "Filter samples by:", heat_cat_cols,
            index=heat_cat_cols.index(default_group_col) if default_group_col in heat_cat_cols else 0,
            key="heatmap_group_col"
        )
        groups_all_heat = meta.loc[meta.index.intersection(log2_df_full.columns), heat_group_col].unique().tolist()
        st.subheader("Select Groups")
        selected_heat_groups = st.multiselect(
            f"{heat_group_col} values to include in the heatmap (choose any subset — 2, 3, 4, or more)",
            groups_all_heat, default=groups_all_heat, key="heatmap_group_select"
        )
        heat_cols = [c for c in log2_df_full.columns if meta.loc[c, heat_group_col] in selected_heat_groups]
        log2_df = log2_df_full[heat_cols]

        st.subheader("Column (Sample) Annotation")
        st.caption(
            "Pick any number of metadata columns to show as stacked annotation tracks above the "
            "heatmap — categorical columns (Diagnosis, Gender, Treatment, ...) render as discrete "
            "color blocks; numeric columns with many distinct values (Age, Body Weight, ...) render "
            "as a color gradient with its own scale."
        )
        col_annot_all_options = [c for c in meta.columns if c != "IsQC"]
        default_col_annot = [heat_group_col] if heat_group_col in col_annot_all_options else []
        col_annot_cols = st.multiselect(
            "Column annotation tracks:", col_annot_all_options, default=default_col_annot,
            key="heatmap_col_annot_cols"
        )

        row_annotations = st.session_state.get("row_annotations")
        row_annot_cols = []
        if row_annotations is not None:
            st.subheader("Row (Protein) Annotation")
            row_annot_cols = st.multiselect(
                "Row annotation tracks:", list(row_annotations.columns),
                default=list(row_annotations.columns)[:1], key="heatmap_row_annot_cols"
            )
            if row_annot_cols:
                n_matched = heatmap_module.count_row_annotation_matches(row_annotations, log2_df_full.index)
                st.caption(f"{n_matched}/{len(log2_df_full.index)} features matched to row annotations "
                           f"by name; unmatched features show as '(unannotated)' in the row bar(s).")

        annotation_colors = {}
        if col_annot_cols or row_annot_cols:
            with st.expander("🎨 Customize Annotation Track Colors", expanded=False):
                st.caption(
                    "Assign specific colors per category (or a colormap for numeric tracks), per "
                    "annotation track — updates the column/row annotation bars on the heatmap below "
                    "dynamically as you change them."
                )
                MAX_VALUES_FOR_COLOR_UI = 12

                def _annotation_color_controls(col, series, axis_label):
                    kind = heatmap_module.classify_annotation_series(series)
                    if kind == "continuous":
                        st.markdown(f"**{col}** ({axis_label} track, continuous)")
                        return st.selectbox(
                            f"Colormap for {col}", heatmap_module.CONTINUOUS_ANNOT_CMAPS,
                            key=f"heatmap_annotcmap_{axis_label}_{col}"
                        )
                    values = list(series.dropna().unique())
                    if len(values) > MAX_VALUES_FOR_COLOR_UI:
                        st.caption(f"**{col}** ({axis_label} track) has {len(values)} distinct values — "
                                   "too many to customize individually; using auto-assigned colors.")
                        return None
                    st.markdown(f"**{col}** ({axis_label} track)")
                    value_cols = st.columns(min(4, len(values)) or 1)
                    track_colors = {}
                    for i, v in enumerate(values):
                        default_c = heatmap_module.GROUP_PALETTE[i % len(heatmap_module.GROUP_PALETTE)]
                        with value_cols[i % len(value_cols)]:
                            track_colors[str(v)] = st.color_picker(
                                str(v), default_c, key=f"heatmap_annotcolor_{axis_label}_{col}_{v}"
                            )
                    return track_colors

                for col in col_annot_cols:
                    result_colors = _annotation_color_controls(col, meta[col], "col")
                    if result_colors:
                        annotation_colors[col] = result_colors
                for col in row_annot_cols:
                    result_colors = _annotation_color_controls(col, row_annotations[col], "row")
                    if result_colors:
                        annotation_colors[col] = result_colors

        st.subheader("Feature Selection")
        st.caption("Dynamically adjust which proteins appear in the heatmap by FDR or p-value — "
                   "the figure below updates immediately as you change the metric or threshold.")
        c1, c2, c3 = st.columns([1.2, 1.5, 1])
        sig_metric = c1.selectbox("Filter by", ["FDR", "p-value"], key="heatmap_sig_metric")
        sig_threshold = c2.slider(
            f"{sig_metric} ≤", min_value=0.0, max_value=1.0,
            value=0.25 if sig_metric == "FDR" else 0.05, step=0.01, key="heatmap_sig_threshold"
        )
        n_preview = (result[sig_metric] <= sig_threshold).sum()
        with c3:
            st.metric("Proteins at this cutoff", n_preview)
        if n_preview > heatmap_module.HIDE_ROW_LABELS_ABOVE:
            st.warning(
                f"{n_preview} proteins selected — beyond {heatmap_module.HIDE_ROW_LABELS_ABOVE}, "
                f"row labels become unreadable and will be hidden automatically. Tighten the cutoff "
                f"above for a labeled figure, or proceed for an unlabeled overview heatmap."
            )

        st.subheader("Clustering Options")
        c3, c4, c5 = st.columns(3)
        cluster_mode = c3.selectbox("Clustering", ["Both rows and columns", "Rows only", "Columns only", "No clustering"])
        distance = c4.selectbox("Distance metric", ["euclidean", "correlation"])
        linkage_m = c5.selectbox("Linkage method", ["ward", "average", "complete"])
        cluster_rows = cluster_mode in ("Both rows and columns", "Rows only")
        cluster_cols = cluster_mode in ("Both rows and columns", "Columns only")

        st.subheader("Color Scale Customization")
        c6, c7 = st.columns(2)
        use_custom_gradient = c6.checkbox("Use a custom color gradient instead of a preset palette", value=False)
        reverse_cmap = c7.checkbox("Reverse colormap", value=False)

        if use_custom_gradient:
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                color_low = st.color_picker("Low color", "#2166AC", key="heatmap_grad_low")
            with gc2:
                color_mid = st.color_picker("Mid color", "#FFFFFF", key="heatmap_grad_mid")
            with gc3:
                color_high = st.color_picker("High color", "#B2182B", key="heatmap_grad_high")
            custom_colors = [color_low, color_mid, color_high]
            cmap_name = None
        else:
            cmap_name = st.selectbox("Predefined color palette", heatmap_module.PREDEFINED_PALETTES)
            custom_colors = None

        c8, c9 = st.columns(2)
        vmin = c8.number_input("Z-score minimum", value=-2.5, step=0.1)
        vmax = c9.number_input("Z-score maximum", value=2.5, step=0.1)

        use_breakpoints = st.checkbox("Use custom discrete color breakpoints (instead of a continuous scale)", value=False)
        breakpoints = None
        if use_breakpoints:
            bp_text = st.text_input("Comma-separated breakpoints (e.g. -3,-1,0,1,3)",
                                     value=f"{vmin},{vmin/2:.2g},0,{vmax/2:.2g},{vmax}")
            try:
                breakpoints = [float(x.strip()) for x in bp_text.split(",") if x.strip()]
            except ValueError:
                st.warning("Couldn't parse breakpoints — using a continuous scale instead.")
                breakpoints = None

        st.subheader("Figure Size")
        c_size1, c_size2, c_size3 = st.columns(3)
        use_custom_size = c_size1.checkbox("Customize heatmap panel size", value=False)
        heatmap_width_in = heatmap_height_in = None
        if use_custom_size:
            heatmap_width_in = c_size2.number_input("Width (inches)", value=8.0, min_value=2.0, max_value=30.0, step=0.5)
            heatmap_height_in = c_size3.number_input("Height (inches)", value=6.0, min_value=2.0, max_value=30.0, step=0.5)
            st.caption("All other elements (dendrograms, annotation bar, legend, colorbar, labels) "
                       "scale automatically with the size you set here.")

        st.subheader("Title & Font Size")
        ft1, ft2 = st.columns(2)
        font_family_ui = ft1.selectbox("Font family", ["sans-serif", "serif", "monospace"], key="heatmap_font_family")
        font_size_ui = ft2.slider("Base font size", 6, 18, 10, key="heatmap_font_size",
                                   help="Applied proportionally to the title, tick labels, legend text, "
                                        "colorbar label, and annotation track labels.")
        st.caption("Row labels still shrink automatically for very large protein counts, capped at the "
                   "size chosen above — the cap doesn't force oversized text when hundreds of rows are shown.")

        if len(heat_cols) < 2:
            st.warning("Select at least one group with 2+ samples to build a heatmap.")
        else:
            sig_feats = result[result[sig_metric] <= sig_threshold].index

            if len(sig_feats) < 2:
                st.warning(
                    f"Only {len(sig_feats)} significant feature(s) at this cutoff — "
                    "clustering needs at least 2. Relax the threshold above."
                )
            else:
                try:
                    title_suffix = f"{len(sig_feats)} features, {sig_metric} ≤ {sig_threshold}"
                    fig_heat, z_ordered, notes = heatmap_module.clustered_heatmap(
                        log2_df, sig_feats, meta=meta, col_annot_cols=col_annot_cols,
                        cluster_rows=cluster_rows, cluster_cols=cluster_cols,
                        distance=distance, linkage_method=linkage_m,
                        cmap_name=cmap_name, custom_colors=custom_colors, reverse_cmap=reverse_cmap,
                        vmin=vmin, vmax=vmax, breakpoints=breakpoints,
                        heatmap_width_in=heatmap_width_in, heatmap_height_in=heatmap_height_in,
                        row_meta=row_annotations, row_annot_cols=row_annot_cols,
                        annotation_colors=annotation_colors,
                        font_family=font_family_ui, font_size=font_size_ui, title_suffix=title_suffix,
                    )
                    st.session_state.viz_figs["Clustered Heatmap"] = fig_heat
                    st.session_state.heatmap_fig = fig_heat
                    st.session_state.heatmap_zscore = z_ordered
                    for note in notes:
                        st.info(f"ℹ️ {note}")
                    st.pyplot(fig_heat)
                    st.caption(f"{len(sig_feats)} proteins shown ({sig_metric} ≤ {sig_threshold}), "
                               f"{len(heat_cols)} samples across {len(selected_heat_groups)} selected group(s), "
                               f"row-scaled (z-score).")

                    st.subheader("Export Heatmap")
                    ec1, ec2, ec3 = st.columns(3)
                    export_fmt = ec1.selectbox("Format", ["PNG", "PDF", "SVG", "JPEG", "TIFF"])
                    export_dpi = ec2.selectbox("Resolution (DPI)", [150, 300, 600], index=1)
                    st.caption(
                        "Note: unlike typical vector plots, this heatmap's cells are drawn as an image "
                        "internally, so DPI affects sharpness even for PDF/SVG — use 300+ to avoid a "
                        "blurry/smeared appearance when zoomed in or printed."
                    )
                    mime_map = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml",
                                "JPEG": "image/jpeg", "TIFF": "image/tiff"}
                    ext_map = {"PNG": "png", "PDF": "pdf", "SVG": "svg", "JPEG": "jpg", "TIFF": "tiff"}
                    fmt_key = "jpeg" if export_fmt == "JPEG" else export_fmt.lower()
                    file_bytes = heatmap_module.export_figure(fig_heat, fmt=fmt_key, dpi=export_dpi)
                    with ec3:
                        st.write("")
                        st.write("")
                        st.download_button(
                            f"Download Heatmap.{ext_map[export_fmt]}", file_bytes,
                            file_name=f"Heatmap.{ext_map[export_fmt]}", mime=mime_map[export_fmt]
                        )

                    st.subheader("Z-score Data Table")
                    st.caption(
                        "The exact row-scaled Z-score values used to render the heatmap above "
                        "(rows/columns in the same clustered order shown)."
                    )
                    st.dataframe(z_ordered, width='stretch', height=300)
                    st.download_button(
                        "Download Zscore_Table.csv", utils.to_download_bytes_csv(z_ordered),
                        file_name="Zscore_Table.csv", mime="text/csv"
                    )
                except ValueError as e:
                    st.warning(str(e))

# ===========================================================================
# TAB 12 — BOXPLOT OF METABOLITES
# ===========================================================================
with TABS[9]:
    st.header("Boxplot of Proteins")
    if st.session_state.log2_data is None:
        st.warning("Complete the Normalization tab (Log2 transform) first.")
    else:
        log2_df_full = st.session_state.log2_data
        meta = st.session_state.meta

        st.subheader("Select Proteins & Groups")
        selected_proteins = st.multiselect(
            "Search and select one or more proteins",
            log2_df_full.index.tolist()
        )
        box_cat_cols = utils.get_categorical_metadata_columns(meta, log2_df_full.columns)
        box_group_col = st.selectbox(
            "Grouping variable:", box_cat_cols,
            index=box_cat_cols.index("Group") if "Group" in box_cat_cols else 0,
            key="boxplot_group_col"
        )
        groups_all_box = meta.loc[meta.index.intersection(log2_df_full.columns), box_group_col].unique().tolist()
        selected_box_groups = st.multiselect(
            f"{box_group_col} values to include in the comparison (choose any subset — 2, 3, 4, or more)",
            groups_all_box, default=groups_all_box, key="boxplot_group_select"
        )

        if selected_proteins and len(selected_box_groups) >= 2:
            test_name = "Welch's t-test" if len(selected_box_groups) == 2 else "one-way ANOVA"
            st.caption(
                f"Statistics computed on log2-transformed data: {test_name} "
                f"across the {len(selected_box_groups)} selected group(s). FDR here is corrected across "
                f"only the {len(selected_proteins)} protein(s) shown, not the full feature panel — "
                f"for a panel-wide FDR, use the Statistics tab."
            )

        st.subheader("Customize Appearance")
        c1, c2, c3 = st.columns(3)
        font_family = c1.selectbox("Font family", ["sans-serif", "serif", "monospace"])
        font_size = c2.slider("Font size", 6, 20, 10, 1)
        ncols = c3.number_input("Panels per row", min_value=1, max_value=6, value=3, step=1)

        c4, c5 = st.columns(2)
        use_custom_box_size = c4.checkbox("Customize figure size", value=False)
        fig_width_in = fig_height_in = None
        if use_custom_box_size:
            fig_width_in = c5.number_input("Width (inches)", value=10.0, min_value=3.0, max_value=30.0, step=0.5)
            fig_height_in = st.number_input("Height (inches)", value=6.0, min_value=3.0, max_value=30.0, step=0.5)

        c6, c7, c8 = st.columns(3)
        show_points = c6.checkbox("Show individual data points", value=True)
        show_mean = c7.checkbox("Show mean (dashed line)", value=False)
        show_median = c8.checkbox("Show median (solid line)", value=True)

        st.caption("Optional: assign a color per group (defaults to a standard palette).")
        group_colors = {}
        if selected_box_groups:
            color_cols = st.columns(min(4, len(selected_box_groups)) or 1)
            for i, g in enumerate(selected_box_groups):
                with color_cols[i % len(color_cols)]:
                    default_c = boxplot_module.GROUP_PALETTE[i % len(boxplot_module.GROUP_PALETTE)]
                    group_colors[g] = st.color_picker(f"Color: {g}", default_c, key=f"boxcolor_{g}")

        if not selected_proteins:
            st.info("Select at least one protein above to generate a boxplot.")
        elif len(selected_box_groups) < 2:
            st.warning("Select at least 2 groups to compare.")
        else:
            box_cols = [c for c in log2_df_full.columns if meta.loc[c, box_group_col] in selected_box_groups]
            log2_df_box = log2_df_full[box_cols]

            stats_table = boxplot_module.compute_stats_for_proteins(
                log2_df_box, meta, selected_proteins, selected_box_groups, group_col=box_group_col
            )
            fig_box = boxplot_module.boxplot_proteins(
                log2_df_box, meta, selected_proteins, selected_box_groups,
                stats_table=stats_table, fig_width_in=fig_width_in, fig_height_in=fig_height_in,
                font_size=font_size, font_family=font_family, group_colors=group_colors,
                show_points=show_points, show_mean=show_mean, show_median=show_median, ncols=ncols,
                group_col=box_group_col,
            )
            st.session_state.viz_figs["Boxplot of Proteins"] = fig_box
            st.session_state.boxplot_fig = fig_box
            st.session_state.boxplot_stats = stats_table
            st.pyplot(fig_box)

            st.subheader("Statistical Results")
            st.dataframe(stats_table, width='stretch')

            st.subheader("Export")
            ec1, ec2, ec3 = st.columns(3)
            export_fmt = ec1.selectbox("Format", ["PNG", "PDF", "SVG", "JPEG", "TIFF"], key="boxplot_export_fmt")
            export_dpi = ec2.selectbox("Resolution (DPI)", [150, 300, 600], index=1,
                                        disabled=export_fmt in ("PDF", "SVG"), key="boxplot_export_dpi")
            mime_map = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml",
                        "JPEG": "image/jpeg", "TIFF": "image/tiff"}
            ext_map = {"PNG": "png", "PDF": "pdf", "SVG": "svg", "JPEG": "jpg", "TIFF": "tiff"}
            fmt_key = "jpeg" if export_fmt == "JPEG" else export_fmt.lower()
            file_bytes = boxplot_module.export_figure(fig_box, fmt=fmt_key, dpi=export_dpi)
            with ec3:
                st.write("")
                st.write("")
                st.download_button(
                    f"Download Boxplot.{ext_map[export_fmt]}", file_bytes,
                    file_name=f"Boxplot.{ext_map[export_fmt]}", mime=mime_map[export_fmt]
                )

def render_significant_protein_selector(key_prefix: str):
    """
    Shared control for GSEA/PPI: build a significant-GENE list from the Statistics
    tab's two-group result table (Log2FC / p-value / FDR), with adjustable cutoffs.
    Translates Protein identifiers to gene symbols via st.session_state.gene_map
    (from an optional 'Gene' column in the uploaded/demo protein intensity matrix)
    when available -- STRING/Enrichr/MSigDB gene sets all key on gene symbol, not
    raw protein accessions. Falls back to using the Protein column directly if no
    Gene column was provided (works only if Protein IS already a gene symbol).

    Returns (sig_genes: list[str], stats_df: DataFrame indexed by Protein,
    stats_by_gene: DataFrame re-indexed by Gene symbol, deduplicated by keeping
    the most significant protein per gene) or (None, None, None) if no
    stats_result is available yet.
    """
    if st.session_state.stats_result is None:
        st.warning("Run a two-group statistical comparison in the Statistics tab first — "
                   "GSEA and P-P Interaction analyze the resulting Log2FC/p-value/FDR list.")
        return None, None, None
    stats_df = st.session_state.stats_result
    groups_used = st.session_state.get("stats_result_groups")
    group_col_used = st.session_state.get("stats_result_group_col") or "Group"
    if groups_used:
        st.caption(f"Reflects the two {group_col_used} values compared in the Statistics tab: "
                   f"**{groups_used[0]}** vs **{groups_used[1]}**.")

    gene_map = st.session_state.get("gene_map")
    stats_by_gene = stats_df.copy()
    if gene_map:
        stats_by_gene["Gene"] = stats_by_gene.index.map(lambda p: gene_map.get(p, p))
        sort_col = "FDR" if "FDR" in stats_by_gene.columns else "p-value"
        stats_by_gene = (stats_by_gene.sort_values(sort_col)
                          .drop_duplicates(subset="Gene", keep="first")
                          .set_index("Gene"))
        st.caption(
            f"{len(gene_map)} protein→gene mappings available (from the uploaded/demo Gene "
            "column) — using gene symbols for STRING/Enrichr lookups."
        )
    else:
        st.caption(
            "No Gene column was provided with this dataset — using the Protein column "
            "directly for STRING/Enrichr lookups (works only if it already contains gene "
            "symbols; the built-in demo datasets include a Gene column, so this note only "
            "applies to your own uploaded data)."
        )

    c1, c2 = st.columns(2)
    metric = c1.selectbox("Filter by", ["FDR", "p-value"], key=f"{key_prefix}_sig_metric")
    threshold = c2.number_input(f"{metric} <", min_value=0.0, max_value=1.0,
                                 value=0.25 if metric == "FDR" else 0.05, step=0.01,
                                 key=f"{key_prefix}_sig_threshold")
    sig_df = stats_df[stats_df[metric] < threshold]
    sig_proteins = sig_df.index.tolist()
    if gene_map:
        sig_genes = sorted(set(gene_map.get(p, p) for p in sig_proteins))
    else:
        sig_genes = sig_proteins
    st.caption(f"{len(sig_proteins)} of {len(stats_df)} proteins pass {metric} < {threshold}"
               + (f" → {len(sig_genes)} unique gene symbols." if gene_map else "."))
    return sig_genes, stats_df, stats_by_gene


# ===========================================================================
# TAB 11 — GSEA (Gene Set Enrichment Analysis)
# ===========================================================================
with TABS[10]:
    st.header("Gene Set Enrichment Analysis")
    st.caption(
        "Three literature-standard methods, in order of typical setup effort: "
        "STRING functional enrichment (Szklarczyk et al. 2023) needs only a protein list; "
        "Over-Representation Analysis via Enrichr (Chen et al. 2013; Kuleshov et al. 2016) "
        "tests a significant list against one chosen curated library; Preranked GSEA "
        "(Subramanian et al. 2005; permutation approach as in GSEAPY's prerank mode) ranks "
        "every detected protein rather than requiring a hard cutoff first. "
        "**STRING and Enrichr require internet access** (they call public APIs); Preranked "
        "GSEA can run fully offline if you upload your own .gmt gene set file."
    )
    sig_proteins, stats_df, stats_by_gene = render_significant_protein_selector("gsea")

    if stats_df is not None:
        method = st.selectbox(
            "Method", ["STRING Functional Enrichment", "Over-Representation Analysis (Enrichr)",
                       "Preranked GSEA (Subramanian et al. 2005)"],
            key="gsea_method"
        )

        if method == "STRING Functional Enrichment":
            species_label = st.selectbox("Species", list(enrichment.STRING_SPECIES.keys()), key="gsea_string_species")
            if st.button("Run STRING Functional Enrichment", type="primary", key="gsea_run_string"):
                if len(sig_proteins) < 1:
                    st.warning("No proteins pass the current cutoff — relax it above.")
                else:
                    try:
                        with st.spinner("Querying STRING (string-db.org)..."):
                            result = enrichment.run_string_enrichment(
                                sig_proteins, species=enrichment.STRING_SPECIES[species_label]
                            )
                        st.session_state.gsea_result = result
                        st.session_state.gsea_method_used = method
                        st.success(f"{len(result)} enriched terms returned.")
                    except enrichment.EnrichmentError as e:
                        st.error(str(e))

        elif method == "Over-Representation Analysis (Enrichr)":
            library = st.selectbox("Gene set library", enrichment.ENRICHR_LIBRARIES, key="gsea_enrichr_library")
            if st.button("Run Over-Representation Analysis", type="primary", key="gsea_run_enrichr"):
                if len(sig_proteins) < 3:
                    st.warning("Need at least 3 proteins passing the current cutoff — relax it above.")
                else:
                    try:
                        with st.spinner("Querying Enrichr (maayanlab.cloud)..."):
                            result = enrichment.run_enrichr_ora(sig_proteins, library=library)
                        st.session_state.gsea_result = result
                        st.session_state.gsea_method_used = f"{method} ({library})"
                        st.success(f"{len(result)} enriched terms returned.")
                    except enrichment.EnrichmentError as e:
                        st.error(str(e))

        else:  # Preranked GSEA
            st.markdown("**Gene set source**")
            gs_source = st.radio("Gene sets from:", ["Fetch library from Enrichr", "Upload custom .gmt file"],
                                  horizontal=True, key="gsea_prerank_source")
            gene_sets = None
            if gs_source == "Fetch library from Enrichr":
                library = st.selectbox("Gene set library", enrichment.ENRICHR_LIBRARIES, key="gsea_prerank_library")
                if st.button("Fetch gene set library", key="gsea_fetch_library"):
                    try:
                        with st.spinner("Fetching gene set library from Enrichr..."):
                            st.session_state.gsea_gene_sets = enrichment.get_enrichr_gene_set_library(library)
                        st.success(f"Fetched {len(st.session_state.gsea_gene_sets)} gene sets from '{library}'.")
                    except enrichment.EnrichmentError as e:
                        st.error(str(e))
            else:
                gmt_file = st.file_uploader("Upload a .gmt gene set file", type=["gmt", "txt"], key="gsea_gmt_upload")
                if gmt_file is not None:
                    st.session_state.gsea_gene_sets = enrichment.parse_gmt_text(
                        gmt_file.getvalue().decode("utf-8", errors="ignore")
                    )
                    st.success(f"Parsed {len(st.session_state.gsea_gene_sets)} gene sets from the uploaded file.")

            c1, c2, c3 = st.columns(3)
            rank_method = c1.selectbox("Ranking score", ["signed_neglogp", "log2fc"], key="gsea_rank_method",
                                        help="signed_neglogp = sign(Log2FC) × -log10(p-value), the standard "
                                             "choice combining direction and significance.")
            min_size = c2.number_input("Min gene set size", min_value=2, value=15, step=1, key="gsea_min_size")
            max_size = c3.number_input("Max gene set size", min_value=5, value=500, step=5, key="gsea_max_size")
            n_perm = st.slider("Permutations (per gene set)", 100, 2000, 500, 100, key="gsea_n_perm",
                                help="More permutations give finer-grained p-values but take longer — "
                                     "500-1000 is typical for exploratory use.")

            if st.session_state.gsea_gene_sets is None:
                st.info("Fetch or upload a gene set library above first.")
            elif st.button("Run Preranked GSEA", type="primary", key="gsea_run_prerank"):
                try:
                    ranked_scores = enrichment.compute_ranking_score(stats_by_gene, method=rank_method)
                    with st.spinner(f"Running Preranked GSEA against {len(st.session_state.gsea_gene_sets)} "
                                     f"gene sets ({n_perm} permutations each — this can take a while)..."):
                        result = enrichment.run_prerank_gsea(
                            ranked_scores, st.session_state.gsea_gene_sets,
                            n_perm=n_perm, min_size=min_size, max_size=max_size
                        )
                    st.session_state.gsea_result = result
                    st.session_state.gsea_ranked_scores = ranked_scores
                    st.session_state.gsea_method_used = f"{method} ({rank_method})"
                    st.success(f"{len(result)} gene sets tested (after size filtering).")
                except enrichment.EnrichmentError as e:
                    st.error(str(e))

        if st.session_state.gsea_result is not None:
            st.subheader(f"Results — {st.session_state.gsea_method_used}")
            st.dataframe(st.session_state.gsea_result, width='stretch', height=350)
            st.download_button(
                "Download GSEA_Results.csv", utils.to_download_bytes_csv(st.session_state.gsea_result),
                file_name="GSEA_Results.csv", mime="text/csv", key="dl_gsea_results"
            )

            if (st.session_state.gsea_ranked_scores is not None and st.session_state.gsea_gene_sets is not None
                    and "Term" in st.session_state.gsea_result.columns):
                st.subheader("Enrichment Plot")
                term_choice = st.selectbox("Gene set to plot", st.session_state.gsea_result["Term"].tolist(),
                                            key="gsea_plot_term")
                if st.button("Generate Enrichment Plot", key="gsea_plot_btn"):
                    try:
                        fig_gsea = enrichment.running_score_plot(
                            st.session_state.gsea_ranked_scores, st.session_state.gsea_gene_sets, term_choice
                        )
                        st.pyplot(fig_gsea)
                        utils.render_figure_download(st, fig_gsea, f"GSEA_{term_choice[:40]}",
                                                      key_prefix="gsea_plot")
                    except enrichment.EnrichmentError as e:
                        st.error(str(e))

# ===========================================================================
# TAB 12 — P-P INTERACTION (Protein-Protein Interaction network)
# ===========================================================================
with TABS[11]:
    st.header("Protein-Protein Interaction Network")
    st.caption(
        "Powered by the STRING database (Szklarczyk et al. 2023, *Nucleic Acids Research*), "
        "the most widely used protein-protein interaction resource in proteomics — combining "
        "physical interactions and functional associations from experiments, curated "
        "databases, co-expression, and text-mining into one confidence score per pair. "
        "**Requires internet access** (calls the public STRING API)."
    )
    sig_proteins_ppi, stats_df_ppi, stats_by_gene_ppi = render_significant_protein_selector("ppi")

    if stats_df_ppi is not None:
        c1, c2, c3 = st.columns(3)
        species_label_ppi = c1.selectbox("Species", list(ppi.STRING_SPECIES.keys()), key="ppi_species")
        confidence_label = c2.selectbox("Confidence threshold", list(ppi.CONFIDENCE_PRESETS.keys()),
                                         index=1, key="ppi_confidence")
        layout_choice = c3.selectbox("Layout", ["spring", "circular", "kamada_kawai"], key="ppi_layout")

        if st.button("Fetch STRING Network", type="primary", key="ppi_fetch_btn"):
            if len(sig_proteins_ppi) < 2:
                st.warning("Need at least 2 proteins passing the current cutoff — relax it above.")
            else:
                try:
                    with st.spinner("Querying STRING (string-db.org)..."):
                        species_id = ppi.STRING_SPECIES[species_label_ppi]
                        required_score = ppi.CONFIDENCE_PRESETS[confidence_label]
                        edge_df = ppi.fetch_string_network(sig_proteins_ppi, species=species_id,
                                                            required_score=required_score)
                        enrichment_stats = ppi.fetch_string_ppi_enrichment(sig_proteins_ppi, species=species_id)
                    st.session_state.ppi_edge_df = edge_df
                    st.session_state.ppi_enrichment_stats = enrichment_stats
                    st.session_state.ppi_graph = ppi.build_graph(edge_df, node_stats=stats_by_gene_ppi)
                    st.success(f"Network fetched: {edge_df.shape[0]} interactions among "
                               f"{st.session_state.ppi_graph.number_of_nodes()} proteins.")
                except ppi.PPIError as e:
                    st.error(str(e))

        if st.session_state.ppi_enrichment_stats:
            st.subheader("Network Enrichment Statistics")
            es = st.session_state.ppi_enrichment_stats
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Nodes", es.get("number_of_nodes", "—"))
            ec2.metric("Edges", es.get("number_of_edges", "—"))
            ec3.metric("Avg. Degree", f"{float(es.get('average_node_degree', 0)):.2f}"
                       if es.get("average_node_degree") is not None else "—")
            ec4.metric("PPI Enrichment p-value", f"{float(es.get('p_value', 1)):.3g}"
                       if es.get("p_value") is not None else "—")
            st.caption(
                "PPI enrichment p-value tests whether this protein list is significantly MORE "
                "interconnected than a random protein set of the same size — the standard "
                "network-level statistic reported in STRING-based proteomics papers."
            )

        if st.session_state.get("ppi_graph") is not None and st.session_state.ppi_graph.number_of_nodes() > 0:
            st.subheader("Network Visualization")
            G = st.session_state.ppi_graph
            c1, c2 = st.columns(2)
            node_size_by = c1.selectbox("Node size by", ["pvalue", "degree"], key="ppi_size_by",
                                         help="'pvalue' sizes nodes by -log10(p-value) (bigger = more "
                                              "significant); 'degree' sizes nodes by number of connections "
                                              "(bigger = more connected hub).")
            show_labels = c2.checkbox("Show protein labels", value=True, key="ppi_show_labels")

            color_mode_ppi = st.radio("Node color scale", ["Preset palette", "Custom gradient (pick your own colors)"],
                                       key="ppi_color_mode", horizontal=True)
            custom_colors_ppi = None
            reverse_cmap_ppi = False
            if color_mode_ppi == "Preset palette":
                cc1, cc2 = st.columns(2)
                cmap_choice = cc1.selectbox("Color palette", ["RdBu_r", "coolwarm", "seismic", "PiYG"], key="ppi_cmap")
                reverse_cmap_ppi = cc2.checkbox("Reverse", value=False, key="ppi_cmap_reverse")
            else:
                st.caption("Pick low/mid/high colors to build a custom diverging gradient for Log2FC.")
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    low_c = st.color_picker("Low (downregulated)", "#2166AC", key="ppi_grad_low")
                with gc2:
                    mid_c = st.color_picker("Mid (unchanged)", "#FFFFFF", key="ppi_grad_mid")
                with gc3:
                    high_c = st.color_picker("High (upregulated)", "#B2182B", key="ppi_grad_high")
                custom_colors_ppi = [low_c, mid_c, high_c]
                cmap_choice = None
            vmax_ppi = st.slider("Log2FC color scale (±)", 0.5, 5.0, 2.0, 0.5, key="ppi_vmax")

            try:
                fig_ppi, _ = ppi.draw_network(
                    G, layout=layout_choice, cmap_name=cmap_choice, custom_colors=custom_colors_ppi,
                    reverse_cmap=reverse_cmap_ppi, vmin=-vmax_ppi, vmax=vmax_ppi,
                    node_size_by=node_size_by, show_labels=show_labels
                )
                st.session_state.ppi_fig = fig_ppi
                st.pyplot(fig_ppi)
                utils.render_figure_download(st, fig_ppi, "PPI_Network", key_prefix="ppi_network")
            except ppi.PPIError as e:
                st.warning(str(e))

            st.subheader("Interaction Table")
            st.dataframe(st.session_state.ppi_edge_df, width='stretch', height=250)
            st.download_button(
                "Download PPI_Interactions.csv", utils.to_download_bytes_csv(st.session_state.ppi_edge_df),
                file_name="PPI_Interactions.csv", mime="text/csv", key="dl_ppi_edges"
            )

            st.subheader("Hub Proteins (Degree, Betweenness, Clustering)")
            st.caption("Standard PPI hub-identification statistics — proteins with high degree/betweenness "
                       "are more likely to be functionally central to the network.")
            st.session_state.ppi_summary = ppi.network_summary_table(G)
            st.dataframe(st.session_state.ppi_summary, width='stretch', height=250)
            st.download_button(
                "Download PPI_Hub_Table.csv", utils.to_download_bytes_csv(st.session_state.ppi_summary),
                file_name="PPI_Hub_Table.csv", mime="text/csv", key="dl_ppi_hubs"
            )

st.sidebar.title("ProteoAI Pro")
st.sidebar.markdown(
    "**LC-MS/MS based Proteomics Analysis**\n\n"
    "Workflow: QC → Reference-Channel/Median-IQR/Log2 Normalization → Statistics "
    "(t-test/Wilcoxon/ANOVA) → PCA → Volcano → Biomarker Discovery → Heatmap → "
    "GSEA → P-P Interaction (every step has its own downloads)"
)
if st.session_state.processing_notes:
    st.sidebar.subheader("Processing Log")
    for note in st.session_state.processing_notes:
        st.sidebar.markdown(f"- {note}")
