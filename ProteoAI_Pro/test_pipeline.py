import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from modules import utils, qc, normalization, imputation_module, stats_analysis, pca_module, volcano, biomarker, heatmap_module

# =====================================================================
# PART A: TMT dataset — exercises the Reference-Channel -> Log2 branch only
# =====================================================================
peak_raw_t = pd.read_csv(f"{BASE}/sample_protein_intensity_matrix_tmt.csv")
meta_raw_t = pd.read_csv(f"{BASE}/sample_metadata_tmt.csv")
peak_df_t = utils.validate_peak_matrix(peak_raw_t)
meta_t = utils.validate_metadata(meta_raw_t, peak_df_t.columns)
qc_cols_t, sample_cols_t = utils.split_qc_and_samples(peak_df_t, meta_t)
assert "Reference_Pool" in peak_df_t.columns, "TMT demo must include a Reference_Pool channel"
assert set(meta_t["Group"].unique()) >= {"Control", "Mild", "Moderate", "Severe"}
print("A1. TMT demo loaded OK:", peak_df_t.shape, "groups:", meta_t["Group"].unique().tolist())

# Reference_Pool is already included in sample_cols_t (IsQC=False, like any other
# channel) and rides along through cleaning, then is consumed (and dropped) by
# reference-channel normalization.
working_t = peak_df_t[sample_cols_t].copy()
missingness_t = imputation_module.compute_missingness(working_t)
n_before_t = working_t.shape[0]
cleaned_t = imputation_module.filter_by_missingness(working_t, missingness_t, max_pct_missing=50)
n_missing_t = imputation_module.count_missing(cleaned_t)
imputed_t = imputation_module.impute_half_minimum(cleaned_t) if n_missing_t > 0 else cleaned_t
assert imputation_module.count_missing(imputed_t, treat_zero_as_missing=False) == 0
print(f"A1b. TMT cleaning+imputation OK: {imputed_t.shape[0]}/{n_before_t} proteins retained, "
      f"{n_missing_t} values imputed")

ref_norm = normalization.reference_channel_normalize(imputed_t, "Reference_Pool")
assert "Reference_Pool" not in ref_norm.columns, "Reference channel should be dropped after normalization"
log2_t, const_t = normalization.log2_transform(ref_norm)
assert set(log2_t.columns) == set(sample_cols_t) - {"Reference_Pool"}
assert log2_t.isna().sum().sum() == 0, "log2_transform left NaN values (zero-handling bug regressed)"
print("A2. Reference-Channel -> Log2 pipeline OK:", log2_t.shape, "NaNs:", log2_t.isna().sum().sum())

# A3/A4: TMT's two other normalization methods must exclude AND drop Reference_Pool
mc_t = normalization.median_center_normalize(imputed_t, exclude_cols=["Reference_Pool"])
assert "Reference_Pool" not in mc_t.columns, "Median centering must drop the excluded reference channel"
assert set(mc_t.columns) == set(sample_cols_t) - {"Reference_Pool"}
log2_mc_t, shift_mc_t = normalization.shift_and_log2_transform(mc_t)
assert log2_mc_t.isna().sum().sum() == 0
print("A3. TMT Median Centering (Reference_Pool excluded+dropped) -> Log2 OK:", log2_mc_t.shape)

mad_t, mad_scale_t = normalization.mad_scale_normalize(imputed_t, exclude_cols=["Reference_Pool"])
assert "Reference_Pool" not in mad_t.columns, "MAD scaling must drop the excluded reference channel"
assert set(mad_t.columns) == set(sample_cols_t) - {"Reference_Pool"}
assert (mad_t.values > 0).all(), "MAD scaling is pure division of positive intensities -- must stay positive"
log2_mad_t, const_mad_t = normalization.log2_transform(mad_t)
assert log2_mad_t.isna().sum().sum() == 0
print(f"A4. TMT Global MAD Scaling (Reference_Pool excluded+dropped, scale={mad_scale_t:.4g}) -> Log2 OK:",
      log2_mad_t.shape)

# Sanity: excluding Reference_Pool from the MAD/median-centering calculation must
# actually change the result vs. naively including it (i.e. the exclusion is load-bearing)
mad_t_incl, mad_scale_incl = normalization.mad_scale_normalize(imputed_t, exclude_cols=None)
assert mad_scale_incl != mad_scale_t, "Excluding Reference_Pool should change the global MAD scale factor"
print("A5. Confirmed excluding Reference_Pool changes the MAD scale factor "
      f"({mad_scale_t:.4g} excluded vs {mad_scale_incl:.4g} included) -- exclusion is load-bearing")

# =====================================================================
# PART B: LABEL-FREE dataset — exercises the Median-IQR -> Log2 branch,
# and is used for the rest of the pipeline (stats/PCA/volcano/heatmap/report)
# =====================================================================
peak_raw = pd.read_csv(f"{BASE}/sample_protein_intensity_matrix_labelfree.csv")
meta_raw = pd.read_csv(f"{BASE}/sample_metadata_labelfree.csv")
peak_df = utils.validate_peak_matrix(peak_raw)
meta = utils.validate_metadata(meta_raw, peak_df.columns)
qc_cols, sample_cols = utils.split_qc_and_samples(peak_df, meta)
assert "Reference_Pool" not in peak_df.columns, "Label-free demo must NOT include a reference channel"
assert set(meta["Group"].unique()) >= {"Control", "Mild", "Moderate", "Severe"}
print("B1. Label-free demo loaded OK:", peak_df.shape, "groups:", meta["Group"].unique().tolist())

# 2. QC — visualization uses QC-ONLY data (no biological samples mixed in).
# Only the correlation matrix is retained (PCA/dendrogram/distance-heatmap removed per request).
cv_table = qc.calculate_cv(peak_df[qc_cols])
fig_cv_dist = qc.cv_distribution_plot(cv_table)
fig_cv_hist = qc.cv_histogram(cv_table)

qc_log, _ = normalization.log2_transform(peak_df[qc_cols])
assert qc_log.shape[1] == len(qc_cols)
fig_corr, corr_df = qc.sample_correlation_matrix(qc_log)
assert set(corr_df.columns) == set(qc_cols)
print("B2. QC-only plots OK (correlation matrix only):", cv_table.shape, cv_table["Quality"].value_counts().to_dict())

keep = cv_table.index[cv_table["Quality"] == "Acceptable"]
working_df = peak_df[sample_cols].copy()
working_df = working_df.loc[working_df.index.intersection(keep)]
assert set(working_df.columns) == set(sample_cols)
print("B3. After CV filter + QC exclusion:", working_df.shape)

# 3b. Data Cleaning & Missing Value Imputation (new step, runs before normalization)
from modules import imputation_module
missingness_table = imputation_module.compute_missingness(working_df)
n_before = working_df.shape[0]
cleaned_df = imputation_module.filter_by_missingness(working_df, missingness_table, max_pct_missing=50)
print(f"B3a. Missingness filter OK: {cleaned_df.shape[0]}/{n_before} features retained "
      f"(categories: {missingness_table['Quality'].value_counts().to_dict()})")

n_missing_before = imputation_module.count_missing(cleaned_df)
assert n_missing_before > 0, "Expected some missing values in the filtered demo data"

for method in imputation_module.METHOD_INFO:
    imputed = imputation_module.impute(cleaned_df, method)
    assert imputation_module.count_missing(imputed, treat_zero_as_missing=False) == 0, f"{method} left NaN"
    assert (imputed.values < 0).sum() == 0, f"{method} produced negative protein intensities"
print(f"B3b. All 6 imputation methods OK, no NaN/negative values ({n_missing_before} values imputed)")

# Use Half-Minimum (fast, deterministic) for the rest of the pipeline
working_df = imputation_module.impute_half_minimum(cleaned_df)
assert imputation_module.count_missing(working_df, treat_zero_as_missing=False) == 0
print("B3c. Working dataset fully imputed, proceeding to normalization:", working_df.shape)

# Untargeted path: Median-IQR normalization (on raw abundance) -> shift + Log2
# (Median-IQR produces negative values for points below the median; log2 needs a
# positivity shift first -- shift_and_log2_transform handles this.)
iqr_norm_feat = normalization.iqr_normalize(working_df, axis="feature")
iqr_norm_sample = normalization.iqr_normalize(working_df, axis="sample")
iqr_norm_batch = normalization.iqr_normalize(working_df, axis="batch", batch_map=meta["Batch"])
print("B4. Median-IQR robust scaling OK (feature/sample/batch), pre-log2:",
      iqr_norm_feat.shape, iqr_norm_sample.shape, iqr_norm_batch.shape)
assert (iqr_norm_feat.values < 0).any(), "Expected negative values from centering normalization"

log2_df, shift_used = normalization.shift_and_log2_transform(iqr_norm_feat)
assert np.isfinite(log2_df.values).all(), "shift_and_log2_transform produced NaN/Inf"
print(f"B5. Shift+Log2 (untargeted path) OK, shift={shift_used:.4g}", log2_df.shape,
      "NaNs:", log2_df.isna().sum().sum())
assert set(log2_df.columns) == set(sample_cols)

fig_dist = normalization.distribution_plots(iqr_norm_feat, log2_df)
print("B6. Distribution plots OK")

# Label-Free's two other normalization methods (no reference channel to exclude here)
mc_lf = normalization.median_center_normalize(working_df)
assert set(mc_lf.columns) == set(sample_cols)
log2_mc_lf, shift_mc_lf = normalization.shift_and_log2_transform(mc_lf)
assert np.isfinite(log2_mc_lf.values).all()
print("B6a. Label-Free Median Centering -> Log2 OK:", log2_mc_lf.shape)

mad_lf, mad_scale_lf = normalization.mad_scale_normalize(working_df)
assert set(mad_lf.columns) == set(sample_cols)
assert (mad_lf.values > 0).all(), "MAD scaling is pure division of positive intensities -- must stay positive"
log2_mad_lf, const_mad_lf = normalization.log2_transform(mad_lf)
assert log2_mad_lf.isna().sum().sum() == 0
print(f"B6b. Label-Free Global MAD Scaling (scale={mad_scale_lf:.4g}) -> Log2 OK:", log2_mad_lf.shape)

# 4. Statistics — two-group (Control vs Severe), computed purely from log2 data
a_samples = meta.index[meta["Group"] == "Control"].tolist()
b_samples = meta.index[meta["Group"] == "Severe"].tolist()
a_samples = [s for s in a_samples if s in log2_df.columns]
b_samples = [s for s in b_samples if s in log2_df.columns]
result = stats_analysis.complete_statistical_table(log2_df, a_samples, b_samples)
assert "Mean_Log2_GroupA" in result.columns and "CI_Lower_Log2FC" in result.columns
print("B7. Two-group stats (Control vs Severe) OK:", result.shape, "Significant:", result["Significant"].sum())
print(result.head(3))

# 4b. ANOVA across the real 4 groups (Control/Mild/Moderate/Severe) — no fake relabeling needed
group_map = meta.loc[sample_cols, "Group"]
group_map = group_map[group_map.index.isin(log2_df.columns)]
anova_table, posthoc_results = stats_analysis.anova_test(log2_df[group_map.index], group_map, posthoc="tukey")
print("B8. ANOVA (4 real groups) OK:", anova_table.shape, "Sig(FDR<0.25):", (anova_table["FDR"] < 0.25).sum())
if posthoc_results:
    k = list(posthoc_results.keys())[0]
    print("    posthoc example:\n", posthoc_results[k])

anova_table2, posthoc2 = stats_analysis.anova_test(log2_df[group_map.index], group_map, posthoc="dunnett")
print("B8b. ANOVA dunnett OK")
anova_table3, posthoc3 = stats_analysis.anova_test(log2_df[group_map.index], group_map, posthoc="pairwise")
print("B8c. ANOVA pairwise OK")

# 5. PCA — biological samples only, with customization
pca, scores_df, cols = pca_module.run_pca(log2_df, n_components=5)
assert set(scores_df.index) == set(sample_cols)
fig_score = pca_module.pca_score_plot(pca, scores_df, meta, palette="Set2",
                                        marker_map={"Control": "o", "Mild": "s", "Moderate": "^", "Severe": "D"},
                                        show_ellipse=True)
fig_loading, top_loadings = pca_module.pca_loading_plot(pca, log2_df.index, top_n=20)
fig_var = pca_module.pca_variance_plot(pca)
print("B9. PCA OK (4 groups, QC-free, customizable):", scores_df.shape, top_loadings.shape)

# 6. Volcano
fig_volc_p, annotated_p = volcano.volcano_plot(result, y_metric="pvalue", sig_cutoff=0.05, top_label_n=10)
fig_volc_f, annotated_f = volcano.volcano_plot(result, y_metric="fdr", sig_cutoff=0.25, top_label_n=10)
print("B10. Volcano OK:", annotated_p["Direction"].value_counts().to_dict())

# New publication-grade features: custom thresholds, highlight, theme, legend positions
fig_volc_custom, ann_custom = volcano.volcano_plot(
    result, fc_threshold=0.5, sig_cutoff=0.01, palette="Colorblind-safe (Okabe-Ito)",
    point_shape="Triangle", legend_position="Bottom", theme="Nature",
    highlight_names=list(result.index[:2]), show_stats_box=True,
)
print("B10a. Volcano custom thresholds/palette/shape/legend-position/theme/highlight OK")

up_tbl = volcano.get_direction_table(annotated_p, "Up")
down_tbl = volcano.get_direction_table(annotated_p, "Down")
settings_json = volcano.export_settings_json({"fc_threshold": 1.0, "sig_cutoff": 0.05})
assert len(settings_json) > 0
print("B10a2. Direction tables + settings JSON export OK:", len(up_tbl), len(down_tbl))

for fmt in ["png", "pdf", "svg", "eps", "jpeg", "tiff"]:
    b = volcano.export_figure(fig_volc_p, fmt=fmt, dpi=300)
    assert len(b) > 0
print("B10b. Volcano export (png/pdf/svg/eps/jpeg/tiff) OK")

# 7. Biomarker discovery
bio_combined = biomarker.discover_biomarkers(result, criterion="combined")
bio_p = biomarker.discover_biomarkers(result, criterion="pvalue")
bio_fdr = biomarker.discover_biomarkers(result, criterion="fdr")
print("B11. Biomarker discovery OK:", len(bio_combined), len(bio_p), len(bio_fdr))

# 8. Heatmap — all 4 clustering modes + color customization + export
sig_feats = result[result["p-value"] < 0.05].index
for cr, cc, label in [(True, True, "both"), (True, False, "rows-only"),
                      (False, True, "cols-only"), (False, False, "none")]:
    fig_heat, z_ordered, notes = heatmap_module.clustered_heatmap(
        log2_df, sig_feats, meta=meta, cluster_rows=cr, cluster_cols=cc,
        distance="euclidean", linkage_method="ward"
    )
    print(f"B12. Heatmap clustering={label} OK:", z_ordered.shape)

for fmt in ["png", "pdf", "svg", "jpeg", "tiff"]:
    b = heatmap_module.export_figure(fig_heat, fmt=fmt, dpi=300)
    assert len(b) > 0
print("B13. Heatmap export (png/pdf/svg/jpeg/tiff) OK")

# 8a. Regression test: PDF export must actually embed a high-resolution raster
# (previously ignored dpi for "vector" formats, producing blurry/smeared cells)
pdf_bytes_lowdpi = heatmap_module.export_figure(fig_heat, fmt="pdf", dpi=72)
pdf_bytes_hidpi = heatmap_module.export_figure(fig_heat, fmt="pdf", dpi=300)
assert len(pdf_bytes_hidpi) > len(pdf_bytes_lowdpi), (
    "300 DPI PDF should embed more raster data than 72 DPI -- if sizes are equal, "
    "dpi is being ignored for PDF export again (the blur bug has regressed)."
)
print(f"B13a. PDF actually respects DPI (72dpi={len(pdf_bytes_lowdpi)}B vs "
      f"300dpi={len(pdf_bytes_hidpi)}B) OK")

# 8b. Regression test: large feature-count heatmap must not crash or produce an
# unbounded figure size (previously OOM-crashed at ~500 rows exported as 600 DPI TIFF)
np.random.seed(0)
n_rows_stress, n_cols_stress = 800, log2_df.shape[1]
stress_data = pd.DataFrame(np.random.randn(n_rows_stress, n_cols_stress),
                            index=[f"StressFeature_{i:04d}" for i in range(n_rows_stress)],
                            columns=log2_df.columns)
fig_stress, z_stress, notes_stress = heatmap_module.clustered_heatmap(
    stress_data, stress_data.index, meta=meta, cluster_rows=True, cluster_cols=True
)
w_in, h_in = fig_stress.get_size_inches()
assert h_in <= heatmap_module.MAX_HEATMAP_HEIGHT_IN + 5, f"Height not capped: {h_in}in"
assert len(notes_stress) > 0, "Expected a size-cap/label-hiding note for an 800-row heatmap"
for fmt in ["png", "tiff", "jpeg"]:
    b = heatmap_module.export_figure(fig_stress, fmt=fmt, dpi=600)
    assert len(b) > 0
print(f"B13b. Large-panel heatmap (800 features) OK: figsize={w_in:.1f}x{h_in:.1f}in, "
      f"notes={len(notes_stress)}, 600dpi export succeeded (would have crashed before this fix)")

# 9. Excel export
xlsx_bytes = utils.to_download_bytes_xlsx({
    "Log2_Normalized": log2_df, "Statistics": result, "Differential_Proteins": bio_combined, "ANOVA": anova_table
})
with open(f"{BASE}/test_output_Statistics.xlsx", "wb") as f:
    f.write(xlsx_bytes)
print("B14. Excel export OK:", len(xlsx_bytes), "bytes")

# 16. Boxplot of Proteins (new module)
from modules import boxplot_module
box_mets = log2_df.index[:4].tolist()
box_stats_2g = boxplot_module.compute_stats_for_proteins(log2_df, meta, box_mets, ["Control", "Severe"])
box_stats_4g = boxplot_module.compute_stats_for_proteins(
    log2_df, meta, box_mets, meta["Group"].unique().tolist()[:4]
)
fig_box = boxplot_module.boxplot_proteins(log2_df, meta, box_mets, ["Control", "Severe"],
                                              stats_table=box_stats_2g, show_points=True, show_mean=True)
for fmt in ["png", "pdf", "svg", "jpeg", "tiff"]:
    b = boxplot_module.export_figure(fig_box, fmt=fmt, dpi=300)
    assert len(b) > 0
print("B16. Boxplot module OK:", box_stats_2g.shape, box_stats_4g.shape, "all export formats OK")

# =====================================================================
# PART C: GSEA (Gene Set Enrichment Analysis) -- offline-testable pieces
# =====================================================================
from modules import enrichment

rng_gsea = np.random.default_rng(0)
lfc_syn = rng_gsea.normal(0, 1, log2_df.shape[0])
pval_syn = rng_gsea.uniform(0.0001, 1, log2_df.shape[0])
stats_syn = pd.DataFrame({"Log2FC": lfc_syn, "p-value": pval_syn}, index=log2_df.index)

ranked = enrichment.compute_ranking_score(stats_syn, method="signed_neglogp")
assert len(ranked) == log2_df.shape[0] and np.isfinite(ranked.values).all()
print("C1. Ranking score computation OK:", ranked.shape)

ranked_sorted = ranked.sort_values(ascending=False)
enriched_set = ranked_sorted.index[:20].tolist()
random_set = list(rng_gsea.choice(log2_df.index, size=20, replace=False))
gene_sets_syn = {"ENRICHED": enriched_set, "RANDOM": random_set, "TOO_SMALL": log2_df.index[:3].tolist()}
gsea_result = enrichment.run_prerank_gsea(ranked, gene_sets_syn, n_perm=200, min_size=10, seed=1)
assert "TOO_SMALL" not in gsea_result["Term"].values, "min_size filter should exclude tiny gene sets"
assert gsea_result.loc[gsea_result["Term"] == "ENRICHED", "NES"].iloc[0] > 0
print("C2. Preranked GSEA (offline math) OK:", gsea_result.shape,
      "ENRICHED NES:", round(gsea_result.loc[gsea_result['Term'] == 'ENRICHED', 'NES'].iloc[0], 3))

fig_gsea = enrichment.running_score_plot(ranked, gene_sets_syn, "ENRICHED")
b = fig_gsea.savefig(f"{BASE}/test_output_gsea_mountain.png", dpi=100)
print("C3. GSEA running-score (mountain) plot rendered OK")

gmt_text = "SET_A\tdesc\tGENEA\tGENEB\tGENEC\nSET_B\tdesc2\tGENED\tGENEE\n"
parsed_gmt = enrichment.parse_gmt_text(gmt_text)
assert parsed_gmt == {"SET_A": ["GENEA", "GENEB", "GENEC"], "SET_B": ["GENED", "GENEE"]}
print("C4. GMT parsing OK:", parsed_gmt)

# Network-dependent calls (STRING/Enrichr) -- must fail gracefully (not crash) when
# unreachable, and return a well-formed DataFrame when reachable. This sandbox has
# no network access, so we expect (and accept) the graceful-failure path here.
try:
    r = enrichment.run_string_enrichment(["TP53", "BRCA1", "EGFR"])
    print("C5. STRING enrichment reachable, returned:", r.shape)
except enrichment.EnrichmentError as e:
    print("C5. STRING enrichment unreachable (expected in this sandbox), handled gracefully:", str(e)[:80])
try:
    r = enrichment.run_enrichr_ora(["TP53", "BRCA1", "EGFR"], library="KEGG_2021_Human")
    print("C6. Enrichr ORA reachable, returned:", r.shape)
except enrichment.EnrichmentError as e:
    print("C6. Enrichr ORA unreachable (expected in this sandbox), handled gracefully:", str(e)[:80])

# =====================================================================
# PART D: P-P Interaction -- offline-testable pieces
# =====================================================================
from modules import ppi

rng_ppi = np.random.default_rng(2)
ppi_genes = log2_df.index[:15].tolist()
edges = []
for i in range(len(ppi_genes)):
    for j in range(i + 1, len(ppi_genes)):
        if rng_ppi.random() < 0.25:
            edges.append({"preferredName_A": ppi_genes[i], "preferredName_B": ppi_genes[j],
                          "score": round(rng_ppi.uniform(0.4, 0.99), 3)})
edge_df_syn = pd.DataFrame(edges)
G = ppi.build_graph(edge_df_syn, node_stats=stats_syn)
assert G.number_of_nodes() > 0 and G.number_of_edges() == len(edge_df_syn)
print("D1. PPI graph construction OK:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

fig_ppi, _ = ppi.draw_network(G, layout="spring", node_size_by="pvalue")
png_bytes = ppi.export_figure(fig_ppi, fmt="png", dpi=150)
assert len(png_bytes) > 0
print("D2. PPI network figure rendered + exported OK:", len(png_bytes), "bytes")

summary_ppi = ppi.network_summary_table(G)
assert set(summary_ppi.columns) == {"Protein", "Degree", "Betweenness_Centrality", "Clustering_Coefficient"}
assert len(summary_ppi) == G.number_of_nodes()
print("D3. PPI hub summary table OK:", summary_ppi.shape)

try:
    ppi.fetch_string_network(["TP53"])
    print("UNEXPECTED: should have raised for <2 proteins")
except ppi.PPIError as e:
    print("D4. Correctly rejected <2 proteins:", str(e))

try:
    r = ppi.fetch_string_network(["TP53", "BRCA1", "EGFR"])
    print("D5. STRING network reachable, returned:", r.shape)
except ppi.PPIError as e:
    print("D5. STRING network unreachable (expected in this sandbox), handled gracefully:", str(e)[:80])

print("\nALL PIPELINE STEPS PASSED (TMT + label-free, 4-group demo datasets; "
      "GSEA + PPI offline logic; PDF report removed).")
