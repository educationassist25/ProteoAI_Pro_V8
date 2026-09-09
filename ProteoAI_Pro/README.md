# ProteoAI Pro

**Automated LC-MS/MS Proteomics Statistical Analysis and Reporting Platform**

Converts raw LC-MS/MS protein intensity data into statistically validated, biologically
interpretable, publication-ready results. Supports **Label-Free (LFQ)** and **TMT**
quantification workflows. A proteomics study is a single dataset — one protein
intensity matrix and one metadata table — so there's no multi-dataset/combine step.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` deliberately pins **nothing** — just package names, no versions.
Earlier rounds tried exact pins matched to a specific Python version, but each
package's version was chosen independently by searching "does X support Python
3.14/3.12" — with no way to verify the *combination* resolves without conflict
(streamlit, scikit-learn, etc. each carry their own transitive version
requirements). An unpinned file hands that resolution problem to pip, which is
built for exactly this — it'll pick a mutually-compatible set for whatever Python
Streamlit Cloud actually runs, rather than being locked to a snapshot no one has
actually installed together. `runtime.txt` requests Python 3.12 (Cloud's own
default) but isn't load-bearing if you'd rather manage the version from Cloud's
"Advanced settings" instead.

If you need a fully reproducible environment later (e.g. for CI), run
`pip freeze > requirements.lock.txt` after a successful install and use that
file instead — but start unpinned.

The app opens with a 10-tab workflow. Check "Use built-in demo dataset" on the first
tab to explore the full pipeline immediately — two demo options are offered:
- **Standard**: matched automatically to whichever quantification method you pick —
  60 proteins incl. a pooled `Reference_Pool` channel for TMT, or 243 proteins (no
  reference channel) for Label-Free — 24 biological samples across Control/Mild/
  Moderate/Severe, 6 QC replicates.
- **Rich Clinical Demo**: 200 proteins, 36 biological samples across 4 diagnosis
  groups (Healthy Control, Prediabetic, Type 2 Diabetes, Metabolic Syndrome, 9 each)
  + 6 QC replicates + 1 pooled `Reference_Pool` channel = 43 total. Metadata includes
  Diagnosis, Age, Gender, Treatment, Ethnicity, and Body Weight, plus a protein-level
  Method + Pathway row-annotation file — built to exercise the **dynamic grouping
  variable** support described below, and works for TMT's reference-channel
  normalization out of the box too (select `Reference_Pool` in the Normalization tab).

**Any categorical metadata column can drive grouping/coloring**, not just a column
literally named "Group". PCA ("Color/group samples by:"), Statistics ("Grouping
variable:"), Heatmap ("Filter samples by:"), and Boxplot ("Grouping variable:") each
expose a selector populated from whichever categorical columns your metadata
actually has (auto-detected — non-numeric columns, or numeric columns with ≤15
distinct values among biological samples, so continuous covariates like Age or Body
Weight are correctly excluded from the grouping-variable list even though they're
still visible in the metadata table, and still usable as a Heatmap annotation track
— see below). Volcano Plot and Biomarker Discovery inherit whichever variable
Statistics last used and say so in a caption. Heatmap's default sample-filter
variable follows Statistics' choice too, but can be overridden independently. This
works with only `Group`/`IsQC`/`Batch` present (the Standard demo, and any dataset
using the older simpler schema) exactly as before — `Group` is always offered first
when present — richer metadata just adds more options, it isn't required.

**Heatmap: multiple stacked row & column annotation tracks**, matching how
published figures typically layer several sample/feature attributes at once rather
than showing just one. "Column annotation tracks:" is a multiselect over every
metadata column (any number, in any combination) — each renders as its own stacked
strip directly above the heatmap, auto-classified continuous (numeric, many
distinct values — e.g. Age, Body Weight — rendered as a color gradient with its own
small labeled colorbar) or categorical (e.g. Diagnosis, Gender, Treatment,
Ethnicity — rendered as discrete color blocks with a swatch legend). Upload an
optional protein row-annotation table in Tab 1 (columns: `Protein`, then any
annotation columns — e.g. `Method`, `Pathway`) to unlock "Row annotation tracks:",
the same multi-track multiselect for the row (protein) side — useful for seeing
whether significant proteins cluster by acquisition method, biological pathway, or
any other protein-level grouping you provide. Each row track also gets its own
rotated label directly above the strip (matching how column tracks are labeled
beside theirs), so which strip is which is clear at a glance without cross-checking
the legend. Every track gets its own legend
block, stacked in one column to the right without overlap (the figure grows taller
automatically if you pick enough tracks that they need more room than the heatmap
provides). Proteins not covered by the row-annotation file, or samples missing a
value for a selected column, show as "(unannotated)"/"(missing)" in neutral gray
rather than causing an error. A soft on-figure note appears past 6 tracks on either
axis suggesting you trim for readability, but there's no hard cap.

PCA, Volcano Plot, Heatmap, and Boxplot update **in real time** — figures redraw
automatically as you change any customization control, with no "Generate" button to
click. Every plot (QC, PCA, Heatmap, ...) has a **format (PNG/TIFF/SVG/PDF) and DPI
(150/300/600) selector** next to its download button, for publication-ready exports.

## Workflow (tabs, left to right)

1. **Data Upload** — Choose your quantification method (**Label-Free** or **TMT**),
   then upload a protein intensity matrix (CSV/XLSX, rows = proteins, columns =
   samples) and a metadata table (`Sample, Group, IsQC, Batch`). A matching built-in
   demo is selected automatically:
   - **TMT** → 60 proteins incl. a pooled/bridge `Reference_Pool` channel (a sample
     column present in every TMT plex, used for reference-channel normalization), 24
     biological samples across 4 groups (Control/Mild/Moderate/Severe), 6 QC
     replicates.
   - **Label-Free** → 243 proteins, no reference channel (label-free runs have no
     shared spiked-in standard across acquisitions), same 4-group/QC design.
   Both demos include enough groups to exercise ANOVA (≥3 groups) directly, not just
   the two-group t-test.
2. **Data Cleaning & Missing Value Imputation** (optional, runs on biological samples
   before QC) — Missing values (NaN or exact zero, both common MS-based proteomics
   conventions for "not detected"/"not quantified") are handled in two steps:
   - **Filter by missingness**: a per-protein missingness table with an adjustable
     threshold slider, following standard guidance (<20% keep · 20-50% keep with
     careful imputation · 50-70% usually remove · >70-80% remove unless essential).
   - **Impute remaining missing values** using one of six methods: Half-Minimum
     (LOD/2, the most common convention), Mean, Median, K-Nearest Neighbors,
     Random Forest (MissForest-style, via scikit-learn's IterativeImputer +
     RandomForestRegressor — no official `missForest` package exists outside R), BPCA
     (approximated via IterativeImputer + BayesianRidge — a reasonable regression-based
     approximation, not the exact original algorithm), and QRILC (approximated —
     draws left-censored values from a truncated distribution reflecting the
     below-detection-limit assumption; the exact quantile-regression algorithm from
     R's `imputeLCMD` package isn't available in Python). All methods clip to
     non-negative output, since protein intensities can't be negative.
   If skipped, QC validation and normalization fall back to the raw biological data.
   The complete post-imputation table is shown in full (not just a preview) and is
   downloadable.
3. **QC Validation** — Coefficient of Variation (CV) across QC replicates, CV
   distribution, feature counts by CV quality, a **QC sample correlation matrix**, and
   a **protein-intensity distribution histogram** (log10 scale) — **CV/correlation are
   computed on QC replicates only** (biological samples are never mixed into those
   diagnostics), while the protein-intensity histogram covers all values. Every plot has a
   format (PNG/TIFF/SVG/PDF) and DPI (150/300/600) selector next to its download
   button. Optionally filter out proteins with CV > 20% — **unchecked by default,
   so all proteins are kept unless you explicitly opt in**; this filtering applies
   on top of whatever came out of Tab 2 (cleaned/imputed if you ran it, raw
   otherwise). Confirming this step **excludes all QC samples** from every tab
   downstream.
4. **Normalization** — pick one of 3 normalization methods, then Log2 transform. The
   method list is tailored by quantification method (selected in Tab 1), with the
   standard/recommended method listed first:
   - **TMT Proteomics**: (1) **Reference-Channel Normalization** — channel protein
     intensity ÷ Reference_Pool intensity, for the same protein; the reference
     channel is dropped from the output once used (its own ratio is trivially 1 for
     every protein). (2) **Median Centering Normalization** — each sample is
     recentered by subtracting its own median (computed across proteins), with a
     grand median added back to preserve overall scale. (3) **Global MAD-based
     Variance Scaling** — a single Median Absolute Deviation is computed across the
     entire dataset and used as one scaling factor (× 1.4826, the standard MAD→SD
     consistency constant) applied uniformly. For (2) and (3), you pick which sample
     column is the pooled/bridge reference channel and it's **excluded from the
     calculation and dropped from the output**, same as (1) — it isn't a
     directly-comparable study sample.
   - **Label-Free Proteomics**: (1) **IQR Normalization** — `(x − median)/IQR`,
     computed per protein across samples, per sample across proteins, or per batch
     (selectable). (2) **Median Centering Normalization** and (3) **Global MAD-based
     Variance Scaling** — same formulas as the TMT versions above, just with no
     reference channel to exclude.
   Methods (1)/(2) (IQR, Median Centering) frequently produce negative values, so
   Log2 uses an automatic positivity shift; Reference-Channel and Global MAD Scaling
   are pure-scaling methods that stay positive, so Log2 uses standard zero
   replacement instead. Only biological samples (and, for TMT, the reference
   channel) ever reach this tab — QC is already excluded in Tab 3.
5. **PCA** — Choose any subset of groups to include (2, 3, 4, or more), score plot
   (biological samples only — QC has its own correlation matrix in Tab 3), customizable
   color mode (5 preset palettes, or a 2D color picker to set an exact color per
   group) and per-group marker style, optional 95% confidence ellipses,
   loading plot (top contributing proteins), and variance-explained plot. Every
   plot has a format/DPI download control; the sample scores (with group labels),
   % variance explained per component, the loadings table, and the exact input data
   matrix are all downloadable as CSV.
6. **Statistics** — Two-group comparison (Student's t-test or Wilcoxon rank-sum,
   choosing any 2 of the available groups) or one-way ANOVA (choose any 3+ groups)
   with Tukey HSD / Dunnett / pairwise post-hoc tests.
   **Every statistic — mean abundance, linear/log2 fold change, p-value, FDR, and 95%
   confidence interval — is computed from the log2-transformed, normalized data.** Raw
   protein intensities are stored for traceability only and never feed into inference. Every
   downloadable result — the two-group table, the ANOVA table, and post-hoc results
   — has the comparison (e.g. `Control_vs_Severe`) or groups compared baked into the
   filename, so results from different comparisons never get confused with each other.
7. **Volcano Plot** — publication-grade, journal-oriented customization:
   - **Thresholds**: user-defined fold-change and significance cutoffs, optional
     secondary FDR requirement, p-value or FDR on the y-axis, log2FC or linear FC on
     the x-axis, selectable threshold line style
   - **Colors**: 5 palettes including colorblind-safe options, individual
     up/down/NS color overrides, adjustable transparency, point border color/width
   - **Point style**: size, shape (circle/triangle/square/diamond/cross)
   - **Labels**: top-N / all-significant / manually-selected-by-name modes, font
     size, bold/italic, color override, automatic overlap-avoidance (label repelling)
   - **Legend**: right/left/top/bottom/hidden placement, full or short label format
   - **Axes & title**: custom titles/subtitles, font size, bold, custom axis limits
     and tick spacing, decimal precision, alignment, hide option
   - **Gridlines & threshold lines**: on/off/major/minor, custom color/style/width
   - **Highlight specific proteins** (e.g. known markers) with a distinct
     color/size/shape
   - **Background**: white/transparent/gray/custom (transparent useful for
     Illustrator workflows)
   - **Figure size presets**: single column, double column, presentation, poster,
     or custom inches
   - **Publication theme presets**: stylistic approximations of Nature/Cell/Cancer
     Research/Clinical Cancer Research/PNAS (not exact journal specifications)
   - **On-figure statistics box**: total/up/down counts and cutoffs
   - **Export**: figure as PNG/PDF/SVG/EPS/JPEG/TIFF at 300/600/1200 DPI; data as
     Upregulated/Downregulated/Complete CSVs; figure settings as JSON for
     reproducibility
   - Not included (would need a different architecture): true interactive
     hover/click (would require Plotly/Bokeh instead of static matplotlib),
     pathway-database-based labeling or filtering (no external database
     integration in this app), VIP-based filtering (needs a PLS-DA model), and
     freehand in-app annotation (arrows/shapes) — labeling and highlighting work
     by protein identity instead.
8. **Biomarker Discovery** — Filter significant proteins (from the same two
   selected groups in Statistics) by p-value, FDR, or combined (p<0.05 AND
   FDR<0.25) criteria. Downloadable with the comparison name baked into the
   filename (e.g. `Control_vs_Severe_Biomarkers.csv`).
9. **Heatmap** — Choose any subset of groups to include (2, 3, 4, or more) before
   rendering. Title is **"Heatmap of Proteins"**, with an automatic suffix showing
   the feature count and cutoff used (e.g. "Heatmap of Proteins (25 features, FDR
   ≤ 0.25)"). Feature selection is **dynamic**: pick FDR or p-value and drag a
   slider (0-1, continuous) — the protein count and the figure both update
   immediately as you move it, rather than choosing from a handful of fixed
   preset cutoffs. **Font family and size controls** apply proportionally to the
   title, tick labels, legend text, colorbar label, and annotation track labels
   from one base size — row labels still shrink automatically for very large
   protein counts, capped at whatever size you choose rather than forcing
   oversized text. Optionally set an exact width/height (inches) for the heatmap
   panel — dendrograms, annotation bar, legend, colorbar, and labels all rescale
   automatically to match. Fully customizable clustered heatmap of significant
   proteins:
   - Independent row/column clustering (both, rows only, columns only, or none)
   - Euclidean/Ward or correlation/average distance metrics
   - Group annotation bar with legend
   - Predefined palettes or a custom 3-color gradient builder, with reverse toggle
   - Per-track color overrides — a **"🎨 Customize Annotation Track Colors"**
     expander lets you assign an exact color to each category value in every
     column/row annotation track (or a colormap for continuous tracks), updating
     the annotation bars dynamically
   - Adjustable Z-score color range and optional discrete color breakpoints
   - Horizontal colorbar below the plot, labeled with Z-score
   - Export to PNG, PDF, SVG, JPEG, or TIFF (150/300/600 DPI)
   - **Downloadable Z-score table**: the exact row-scaled Z-score values used to
     render the heatmap (same clustered row/column order) as CSV
   - **Large-panel safety limits**: with hundreds of significant proteins, the panel
     height/width auto-caps at 30in/24in (uncapped scaling previously produced
     100+ inch figures that could crash on export), and row/column labels
     auto-hide above 150/120 items since they'd be unreadable anyway — a preview
     metric and warning appear before you generate, and any auto-adjustments are
     reported after. TIFF export uses lossless LZW compression (~100x smaller
     files) and DPI auto-reduces if a request would demand an excessive raw pixel
     buffer, preventing out-of-memory crashes during export.
10. **Boxplot** — Select one or more proteins and any subset of groups (2, 3, 4, or
    more) to compare. Statistics (Welch's t-test for 2 groups, one-way ANOVA for 3+)
    are computed on log2-transformed data and shown directly on each panel — note the
    FDR here is corrected only across the proteins you've selected for display, not
    the full protein panel (use the Statistics tab for a panel-wide FDR).
    Customizable: figure width/height, font size/family, per-group box colors,
    individual data points, mean/median overlay, panels-per-row, and export to
    PNG/PDF/SVG/JPEG/TIFF.
11. **GSEA** — Gene Set Enrichment Analysis on the Statistics tab's two-group
    Log2FC/p-value/FDR table, offering three literature-standard methods (in order of
    typical setup effort):
    - **STRING Functional Enrichment** (Szklarczyk et al. 2023, *STRING v12*, Nucleic
      Acids Research) — hypergeometric enrichment computed server-side by STRING
      against its own curated GO/KEGG/Reactome/Pfam/InterPro annotation sets. Needs
      only a significant protein list (filterable by FDR or p-value cutoff above) —
      no library selection required.
    - **Over-Representation Analysis via Enrichr** (Chen et al. 2013, *BMC
      Bioinformatics*; Kuleshov et al. 2016, *Nucleic Acids Research*) — hypergeometric
      test of the significant list against one chosen curated library
      (GO Biological Process/Molecular Function/Cellular Component, KEGG, Reactome,
      WikiPathways, or MSigDB Hallmark), whole-genome background.
    - **Preranked GSEA** (Subramanian et al. 2005, *PNAS*; gene-set permutation
      approach as implemented by GSEAPY's prerank mode) — ranks *every* detected
      protein by a signed score (`sign(Log2FC) × -log10(p-value)` by default) rather
      than requiring a hard cutoff first, then computes a weighted running-sum
      enrichment statistic (ES/NES) per gene set with permutation-based p-values and
      BH-FDR across tested sets. Gene sets come from an Enrichr library fetch or an
      uploaded `.gmt` file (MSigDB format) — the only method of the three that can run
      **fully offline**. Includes the classic 3-panel GSEA "mountain plot" (running
      score, hit barcode, ranking metric) for any tested gene set.
    STRING and Enrichr require internet access (they call public APIs); results and
    the enrichment plot are downloadable as CSV/PNG etc. Protein identifiers must be
    (or map to) **gene symbols** — the convention Enrichr, STRING, and MSigDB gene
    sets all share.
12. **P-P Interaction** — Protein-protein interaction network for the same
    significant protein list, via the STRING database (Szklarczyk et al. 2023) — by
    far the most widely used PPI resource in proteomics, combining physical
    interactions and functional associations from experiments, curated databases,
    co-expression, and text-mining into one confidence score per pair. Adjustable
    species and confidence threshold (STRING's own Low/Medium/High/Highest presets).
    Reports the **PPI enrichment p-value** (is this list more interconnected than a
    random protein set of the same size? — the standard network-level statistic in
    STRING-based papers), renders the network styled consistently with the rest of
    the app (nodes colored by Log2FC on a diverging scale — a preset palette or a
    custom low/mid/high gradient via a 2D color picker, matching the Heatmap tab's
    approach — sized by significance or degree, edges weighted by STRING's
    confidence score; spring/circular/Kamada-Kawai layouts), and computes a hub
    table (Degree, Betweenness Centrality, Clustering Coefficient) to flag likely
    functionally-central proteins. Interaction table, hub table, and network figure
    (PNG/PDF/SVG/JPEG/TIFF) are all downloadable. Requires internet access.

Every step in this app (Cleaning, QC, Normalization, PCA, Statistics, Volcano,
Biomarker Discovery, Heatmap, Boxplot, GSEA, P-P Interaction) has its own downloads —
there's no separate all-in-one export tab.

## Project structure

```
ProteoAI_Pro/
├── app.py                     # Streamlit UI (10-tab, single-dataset workflow)
├── modules/
│   ├── utils.py                # Data loading, validation, zero replacement, figure export
│   ├── qc.py                   # CV calculation, QC distribution/histogram/correlation matrix
│   ├── imputation_module.py    # Data cleaning & missing value imputation (6 methods)
│   ├── normalization.py        # Reference-channel (TMT), IQR (feature/sample/batch), Log2
│   ├── stats_analysis.py       # t-test/Wilcoxon, ANOVA + post-hoc, BH-FDR
│   ├── pca_module.py           # PCA score/loading/variance plots with ellipses
│   ├── volcano.py              # p-value and FDR based volcano plots
│   ├── biomarker.py            # Statistical filtering for biomarker discovery
│   ├── heatmap_module.py       # Clustered heatmap (Euclidean/Ward, correlation)
│   ├── boxplot_module.py       # Per-protein boxplot comparisons across groups
│   ├── enrichment.py            # GSEA: STRING enrichment, Enrichr ORA, Preranked GSEA
│   └── ppi.py                   # Protein-protein interaction network via STRING
├── make_sample_data.py        # Synthetic demo dataset generator (TMT, Label-Free)
├── generate_rich_demo_data.py # Richer 4-diagnosis-group clinical demo generator
├── demo_gene_symbols.py       # Curated real gene symbol panel shared by both generators
├── sample_protein_intensity_matrix_tmt.csv
├── sample_metadata_tmt.csv
├── sample_protein_intensity_matrix_labelfree.csv
├── sample_metadata_labelfree.csv
├── sample_protein_intensity_matrix_richdemo.csv
├── sample_metadata_richdemo.csv
├── protein_row_annotations_richdemo.csv
├── test_pipeline.py           # Headless smoke test of the full pipeline
└── requirements.txt
```

## Input format notes

- **Protein intensity matrix**: first column is the protein identifier; an
  **optional second column named `Gene`** (or `Gene Name`/`Gene_Name`/`GeneSymbol`/
  `Gene Symbol`/`Genes`) provides a Protein → Gene SYMBOL mapping; every remaining
  column is a sample (including QC samples, and for TMT, the pooled reference
  channel). Values must be numeric protein intensities. The `Gene` column feeds
  the **GSEA** and **P-P Interaction** tabs, since STRING/Enrichr/MSigDB gene sets
  all key on gene symbol rather than raw protein accessions — without it, those
  tabs fall back to using the Protein column directly (works only if it already
  contains gene symbols). All three built-in demo datasets (TMT, Label-Free, Rich
  Clinical Demo) include a real `Gene` column drawn from a curated, biologically
  coherent panel (see below), so GSEA/PPI return genuine results out of the box.
- **Metadata table**: one row per sample. `Sample` must match a column name in the
  protein intensity matrix. `IsQC` marks QC replicates (True/False, 1/0, "QC", etc.).
  `Batch` is optional and only used for batch-specific IQR normalization.
- For TMT, include the pooled/bridge reference channel as an ordinary sample column
  in the protein intensity matrix (with a corresponding metadata row, `IsQC=False`)
  and select it by name in the Normalization tab.

## Demo data gene panel

`demo_gene_symbols.py` holds a curated panel of 243 **real, verifiable human gene
symbols** (HGNC nomenclature), themed around metabolic syndrome / type 2 diabetes /
lipid metabolism / inflammation / cardiovascular comorbidity — drawn from standard,
widely-published pathway gene sets (KEGG insulin signaling, type II diabetes
mellitus, PPAR signaling, adipocytokine signaling, AMPK signaling, NF-κB signaling,
complement and coagulation cascades) plus well-known GWAS type-2-diabetes
susceptibility loci and canonical lipid/cholesterol/bile-acid metabolism genes. All
three demo datasets draw their `Protein`/`Gene` columns from this panel (Protein
uses the UniProt "entry name" convention, e.g. `INS_HUMAN`; Gene is the plain
symbol, e.g. `INS`) — so running GSEA or P-P Interaction on demo data returns real
enriched pathways (e.g. "Type II diabetes mellitus", "Insulin signaling", "PPAR
signaling") and a genuinely interconnected STRING network, not placeholder/no-match
output.

## Notes on statistical conventions

- Significance in the two-group table and biomarker module defaults to
  **p < 0.05 AND FDR < 0.25**, matching common exploratory proteomics practice —
  adjust cutoffs in the UI as needed for your study.
- Two-group results include a 95% confidence interval (`CI_Lower_Log2FC`,
  `CI_Upper_Log2FC`) on the log2 fold change, computed via Welch's t formula.
- ANOVA post-hoc tests are only run for proteins with FDR < 0.25 to keep runtime
  reasonable on large protein panels.
- Log2 transformation replaces zeros/missing values with half the minimum positive
  value observed (per protein) before transforming, avoiding -Inf.
- QC samples are used exclusively for the QC Validation tab's reproducibility
  diagnostics and are excluded from every other tab (normalization, statistics, PCA,
  volcano, biomarker discovery, heatmap) once QC is confirmed.

## Color selection

Every color choice throughout the app (Volcano point/grid/threshold/highlight/
background colors, Heatmap's custom gradient, P-P Interaction's custom gradient,
PCA and Boxplot per-group colors) uses the standard continuous-gradient color
picker (drag within the hue/saturation square, or type a hex code directly).

**Heatmap annotation tracks are the one place with dedicated multi-color
controls**: expand **"🎨 Customize Annotation Track Colors"** (shown once you've
picked at least one column or row annotation track) to assign an exact color to
each category value in a categorical track (e.g. give "Type 2 Diabetes" a specific
red and "Healthy Control" a specific green within the Diagnosis track), or pick a
colormap for a continuous track (e.g. Age, Body Weight) — independently, per
track, updating the heatmap's annotation bars dynamically as you change them.
Categorical tracks with more than 12 distinct values skip the per-value picker
(too many swatches to be practical) and keep their auto-assigned colors.

## External services (GSEA / P-P Interaction tabs)

The GSEA tab's STRING Functional Enrichment and Over-Representation Analysis (Enrichr)
methods, and the entire P-P Interaction tab, call public third-party APIs
(`string-db.org`, `maayanlab.cloud`) and therefore **require internet access** from
wherever the app is running. Preranked GSEA is the only GSEA method that can run
fully offline, provided you upload your own `.gmt` gene set file instead of fetching
one from Enrichr. If a request fails (no internet, or the service is temporarily
down), the app reports a clear error rather than crashing — retry, check your
connection, or switch to Preranked GSEA with an uploaded `.gmt` file.

## Testing

Run `python test_pipeline.py` for a headless smoke test that exercises every module
against both the TMT and Label-Free synthetic demo datasets (useful in environments
without a display/Streamlit server). The GSEA and P-P Interaction modules' offline
logic (ranking scores, the preranked enrichment statistic, `.gmt` parsing, network
construction/visualization/hub statistics) is fully covered; the STRING/Enrichr API
calls themselves are exercised too, but tolerate an unreachable network (expected in
sandboxed/offline CI environments) by asserting the failure is handled gracefully
rather than requiring a live connection.
