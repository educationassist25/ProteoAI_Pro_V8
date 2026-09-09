"""
generate_rich_demo_data.py - A richer, standalone example dataset for ProteoAI Pro
(and for general exploration outside the app), distinct from make_sample_data.py's
simpler 4-group demos.

Produces three files:
  1. sample_protein_intensity_matrix_richdemo.csv  - 200 proteins x 43 samples
  2. sample_metadata_richdemo.csv          - 43 samples x clinical/demographic variables
  3. protein_row_annotations_richdemo.csv - 200 proteins x Gene/Method/Pathway

Design:
  - 4 Diagnosis groups (Healthy Control, Prediabetic, Type 2 Diabetes, Metabolic
    Syndrome) x 9 biological samples each = 36, + 6 QC replicates + 1 pooled
    Reference_Pool channel = 43 total. "Diagnosis" doubles as the "Group" column
    the app's stats/PCA/ANOVA logic reads, so this drops straight into ProteoAI
    Pro's single-dataset upload with no changes needed -- Age/Gender/Treatment/
    Ethnicity/Body Weight ride along as extra columns the app simply passes
    through.
  - Reference_Pool is a pooled/bridge channel (Group="Reference", IsQC=False) so
    the dataset works out of the box for TMT's reference-channel normalization,
    not just Label-Free's Median-IQR path -- select it in the Normalization tab
    regardless of which quantification method you pick in Tab 1.
  - Proteins use REAL gene symbols (via demo_gene_symbols.GENE_SYMBOLS, the same
    curated metabolic-syndrome/diabetes/lipid/inflammation-themed panel used by
    make_sample_data.py) so GSEA and P-P Interaction return genuine results against
    the real STRING/Enrichr APIs. Protein uses the UniProt "entry name" convention
    (GENE_HUMAN); Gene is the plain symbol, included both in the main matrix and in
    the row-annotation file. Proteins also carry a Method (one of 4 analysis types)
    and Pathway annotation in the row-annotation file, for external pathway-level
    interpretation, grouping, or filtering -- the app doesn't currently ingest a
    row-annotation file for that purpose, so Method/Pathway are supplementary
    reference data, not something app.py reads (Gene IS read, from the main matrix).
  - Abundance values are realistic-ish: a lognormal per-protein baseline, small
    per-sample technical noise, a shared per-pathway "activity" factor per sample
    (so proteins in the same pathway co-vary -- meaningful for correlation
    analysis), diagnosis-driven fold changes on a differential subset (meaningful
    for ANOVA/heatmap/PCA separation), and a continuous Age/Body-Weight-linked
    effect on a small subset (meaningful for correlation against continuous
    covariates). Reference_Pool is computed as the average of the true biological
    samples (with small technical noise on top) after all of that, since a pooled
    reference reflects the whole cohort rather than any one patient's biology.
    Missingness is injected with the same tiered scheme as the other demos (QC and
    Reference_Pool excluded, matching real-world high reproducibility for both),
    so Data Cleaning & Imputation has something to do.
"""
import os
import numpy as np
import pandas as pd
from demo_gene_symbols import GENE_SYMBOLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 777

DIAGNOSES = ["Healthy Control", "Prediabetic", "Type 2 Diabetes", "Metabolic Syndrome"]
N_PER_DIAGNOSIS = 9
N_QC = 6
N_METABOLITES = 200

METHODS = ["Label-Free Proteomics", "TMT Proteomics", "Label-Free Proteomics", "TMT Proteomics"]
METHOD_WEIGHTS = [0.40, 0.20, 0.25, 0.15]  # sums to 1.0, mirrors the multi-method demo's proportions

PATHWAYS = [
    "Amino Acid Metabolism", "TCA Cycle", "Glycolysis", "Lipid Metabolism",
    "Fatty Acid Oxidation", "Bile Acid Metabolism", "Sphingolipid Metabolism",
    "Purine Metabolism", "Nucleotide Metabolism", "Steroid Hormone Biosynthesis",
]

GENDERS = ["Female", "Male"]
ETHNICITIES = ["Caucasian", "Hispanic/Latino", "African American", "Asian", "Other"]
TREATMENTS_BY_DIAGNOSIS = {
    "Healthy Control": ["Untreated"],
    "Prediabetic": ["Untreated", "Diet & Exercise"],
    "Type 2 Diabetes": ["Metformin", "Insulin", "Diet & Exercise"],
    "Metabolic Syndrome": ["Diet & Exercise", "Metformin"],
}


def _build_sample_metadata(rng):
    sample_names, diagnosis, is_qc, batch = [], [], [], []
    age, gender, treatment, ethnicity, body_weight = [], [], [], [], []

    for d_i, dx in enumerate(DIAGNOSES):
        # Disease groups skew a bit older and heavier than Healthy Control -- realistic,
        # and gives Age/Body Weight a genuine (if modest) relationship to Diagnosis for
        # anyone cross-tabulating rather than just eyeballing group means.
        age_center = 38 if dx == "Healthy Control" else rng.uniform(48, 58)
        weight_center = 68 if dx == "Healthy Control" else rng.uniform(82, 95)
        for i in range(N_PER_DIAGNOSIS):
            sample_names.append(f"{dx.replace(' ', '')}_{i+1}")
            diagnosis.append(dx)
            is_qc.append(False)
            batch.append("1" if i < N_PER_DIAGNOSIS // 2 else "2")
            age.append(int(np.clip(rng.normal(age_center, 8), 20, 85)))
            gender.append(rng.choice(GENDERS))
            treatment.append(rng.choice(TREATMENTS_BY_DIAGNOSIS[dx]))
            ethnicity.append(rng.choice(ETHNICITIES, p=[0.45, 0.20, 0.15, 0.15, 0.05]))
            body_weight.append(round(float(np.clip(rng.normal(weight_center, 12), 45, 140)), 1))

    for i in range(N_QC):
        sample_names.append(f"QC_{i+1}")
        diagnosis.append("QC")
        is_qc.append(True)
        batch.append("1" if i < N_QC // 2 else "2")
        age.append(pd.NA)
        gender.append("QC")
        treatment.append("QC")
        ethnicity.append("QC")
        body_weight.append(pd.NA)

    # Pooled/bridge reference channel -- present in every TMT plex, used by the
    # Normalization tab's reference-channel normalization. Not a QC replicate (it's
    # not a repeat injection of the same material to check reproducibility) and not
    # a real patient (it has no diagnosis/age/weight of its own), so it gets its own
    # Group label and NA demographic fields, same convention as the other demos.
    sample_names.append("Reference_Pool")
    diagnosis.append("Reference")
    is_qc.append(False)
    batch.append("1")
    age.append(pd.NA)
    gender.append("Reference")
    treatment.append("Reference")
    ethnicity.append("Reference")
    body_weight.append(pd.NA)

    meta_df = pd.DataFrame({
        "Sample": sample_names,
        "Group": diagnosis,          # app-compatible alias of Diagnosis (drives PCA/ANOVA/stats grouping)
        "IsQC": is_qc,
        "Batch": batch,
        "Diagnosis": diagnosis,
        "Age": age,
        "Gender": gender,
        "Treatment": treatment,
        "Ethnicity": ethnicity,
        "Body Weight": body_weight,
    })
    return meta_df


def _build_row_annotations(rng):
    gene_symbols = sorted(rng.choice(GENE_SYMBOLS, size=min(N_METABOLITES, len(GENE_SYMBOLS)),
                                      replace=False).tolist())
    protein_names = [f"{g}_HUMAN" for g in gene_symbols]
    methods = rng.choice(METHODS, size=len(protein_names), p=METHOD_WEIGHTS)
    # Roughly even pathway spread with natural random variation, not a hard-forced quota
    pathways = rng.choice(PATHWAYS, size=len(protein_names))
    return pd.DataFrame({"Protein": protein_names, "Gene": gene_symbols, "Method": methods, "Pathway": pathways})


def generate_rich_demo():
    rng = np.random.default_rng(SEED)
    meta_df = _build_sample_metadata(rng)
    row_annot = _build_row_annotations(rng)
    sample_names = meta_df["Sample"].tolist()
    diagnosis = meta_df["Diagnosis"].tolist()
    is_qc = meta_df["IsQC"].tolist()
    age_lookup = dict(zip(meta_df["Sample"], meta_df["Age"]))
    weight_lookup = dict(zip(meta_df["Sample"], meta_df["Body Weight"]))
    protein_names = row_annot["Protein"].tolist()
    pathway_of = dict(zip(row_annot["Protein"], row_annot["Pathway"]))
    n_samples = len(sample_names)

    ref_idx = sample_names.index("Reference_Pool")
    # True biological patient samples only -- excludes QC replicates (technical
    # repeats, no individual biology of their own) AND the Reference_Pool channel
    # (a pooled average, not an individual patient) from every step that models
    # real inter-patient variability.
    study_idx = [i for i, (dx, q) in enumerate(zip(diagnosis, is_qc)) if not q and dx != "Reference"]

    # 1. Per-protein baseline abundance + per-sample technical noise
    base = rng.lognormal(mean=10, sigma=1.2, size=(N_METABOLITES, 1))
    data = base * rng.lognormal(mean=0, sigma=0.15, size=(N_METABOLITES, n_samples))

    # 2. Shared per-pathway "activity" factor per BIOLOGICAL sample -> co-regulated
    #    proteins within a pathway (meaningful signal for correlation analysis).
    #    QC replicates and Reference_Pool are not individual biological subjects,
    #    so they must NOT carry this variation -- applying it to QC would inflate
    #    QC CV into unrealistic territory, and Reference_Pool is set from the
    #    biological samples' own values in step 5, after this runs.
    for pw in PATHWAYS:
        pw_feats = [i for i, m in enumerate(protein_names) if pathway_of[m] == pw]
        if not pw_feats:
            continue
        pathway_activity = np.ones(n_samples)
        pathway_activity[study_idx] = rng.lognormal(mean=0, sigma=0.22, size=len(study_idx))
        data[np.ix_(pw_feats, range(n_samples))] *= pathway_activity

    # 3. Diagnosis-driven fold changes on a differential subset (~25% of features),
    #    each direction/magnitude independent per non-control diagnosis group --
    #    Diagnosis is nominal (no inherent severity order), unlike the other demos'
    #    Control->Severe staging.
    diag_idx = {dx: [i for i, d in enumerate(diagnosis) if d == dx] for dx in DIAGNOSES}
    n_diff = max(5, int(0.25 * N_METABOLITES))
    diff_feats = rng.choice(N_METABOLITES, size=n_diff, replace=False)
    for f in diff_feats:
        for dx in DIAGNOSES:
            if dx == "Healthy Control":
                continue
            fold = rng.choice([rng.uniform(1.5, 2.8), rng.uniform(0.3, 0.65)])
            data[f, diag_idx[dx]] *= fold

    # 4. Continuous covariate effects on a small subset (~5% of features), linked to
    #    Age and/or Body Weight -- gives correlation analysis against continuous
    #    metadata something genuine to find, not just categorical group differences.
    ages = np.array([age_lookup[s] if not pd.isna(age_lookup[s]) else 45 for s in sample_names], dtype=float)
    weights = np.array([weight_lookup[s] if not pd.isna(weight_lookup[s]) else 70 for s in sample_names], dtype=float)
    age_z = (ages - ages[study_idx].mean()) / ages[study_idx].std()
    weight_z = (weights - weights[study_idx].mean()) / weights[study_idx].std()
    n_covariate = max(3, int(0.05 * N_METABOLITES))
    covariate_feats = rng.choice([i for i in range(N_METABOLITES) if i not in diff_feats],
                                  size=n_covariate, replace=False)
    for f in covariate_feats:
        driver = age_z if rng.random() < 0.5 else weight_z
        coef = rng.uniform(0.15, 0.35) * rng.choice([1, -1])
        multiplier = np.ones(n_samples)
        multiplier[study_idx] = np.exp(coef * driver[study_idx])  # QC/Reference unaffected -- no real age/weight
        data[f, :] *= multiplier

    # 5. Reference_Pool = the average of the true biological samples' (already
    #    pathway- and diagnosis-adjusted) values for each protein, with a small
    #    extra layer of channel-to-channel technical noise on top -- the same
    #    logic make_sample_data.py's TMT demo uses, just applied after this
    #    dataset's richer per-sample effects instead of before them.
    data[:, ref_idx] = data[:, study_idx].mean(axis=1) * rng.lognormal(mean=0, sigma=0.05, size=N_METABOLITES)

    # 6. QC replicate stability is already covered by the same low-sigma technical
    #    noise applied to everyone in step 1 -- no ISTD row in this combined panel
    #    (it spans multiple methods, so a single spiked standard wouldn't apply to
    #    all of it; use Group/Diagnosis + Median-IQR normalization, or
    #    Reference_Pool + reference-channel normalization for TMT, when analyzing).

    # 7. Realistic missingness (encoded as exact zero), true biological samples
    #    only -- QC and Reference_Pool stay complete, same tiered scheme as the
    #    other demos so Cleaning & Imputation has real work:
    #    ~70% features 0-15% missing, ~20% features 20-50%, ~10% features 55-85%.
    for f in range(N_METABOLITES):
        tier = rng.random()
        if tier < 0.70:
            miss_rate = rng.uniform(0.0, 0.15)
        elif tier < 0.90:
            miss_rate = rng.uniform(0.20, 0.50)
        else:
            miss_rate = rng.uniform(0.55, 0.85)
        n_miss = int(round(miss_rate * len(study_idx)))
        if n_miss > 0:
            miss_cols = rng.choice(study_idx, size=n_miss, replace=False)
            data[f, miss_cols] = 0.0

    peak_df = pd.DataFrame(data, index=protein_names, columns=sample_names)
    gene_of = dict(zip(row_annot["Protein"], row_annot["Gene"]))
    peak_df.insert(0, "Gene", [gene_of[p] for p in protein_names])
    peak_df.insert(0, "Protein", protein_names)
    peak_path = f"{BASE_DIR}/sample_protein_intensity_matrix_richdemo.csv"
    peak_df.to_csv(peak_path, index=False)

    meta_path = f"{BASE_DIR}/sample_metadata_richdemo.csv"
    meta_df.to_csv(meta_path, index=False)

    annot_path = f"{BASE_DIR}/protein_row_annotations_richdemo.csv"
    row_annot.to_csv(annot_path, index=False)

    print(f"[richdemo] {N_METABOLITES} proteins x {n_samples} samples "
          f"({N_QC} QC, {len(study_idx)} biological across {len(DIAGNOSES)} diagnosis groups, "
          f"1 reference channel) -> {peak_path}")
    print(f"[richdemo] metadata (Diagnosis, Age, Gender, Treatment, Ethnicity, Body Weight) -> {meta_path}")
    print(f"[richdemo] row annotations (Method, Pathway) -> {annot_path}")
    return peak_df, meta_df, row_annot


if __name__ == "__main__":
    generate_rich_demo()
    print("Rich demo dataset written.")
