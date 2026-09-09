"""
Generate two synthetic demo datasets for ProteoAI Pro:
  1. TMT         - includes a pooled/bridge Reference_Pool channel (sample column),
                   smaller panel (typical of a single TMT plex)
  2. LABEL-FREE  - no reference channel (no shared spiked-in standard across runs),
                   larger feature set (typical of label-free acquisitions)

Both use the same 4-group design (Control / Mild / Moderate / Severe) with QC
replicates, so the demo also exercises ANOVA (>=3 groups) out of the box, not
just the two-group t-test.

Protein identifiers use REAL gene symbols (via demo_gene_symbols.GENE_SYMBOLS, a
curated metabolic-syndrome/diabetes/lipid/inflammation-themed panel) so the GSEA
and P-P Interaction tabs return genuine, biologically meaningful results against
the real STRING/Enrichr APIs, not placeholder/no-match output. The matrix's first
column ("Protein") uses the UniProt "entry name" convention (GENE_HUMAN); the
second column ("Gene") carries the plain gene symbol the GSEA/PPI tabs actually
use for lookups.
"""
import os
import numpy as np
import pandas as pd
from demo_gene_symbols import GENE_SYMBOLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUPS = ["Control", "Mild", "Moderate", "Severe"]
N_PER_GROUP = 6
N_QC = 6


def _build_sample_metadata(rng, include_reference: bool):
    sample_names, group_labels, is_qc, batch = [], [], [], []
    for g in GROUPS:
        for i in range(N_PER_GROUP):
            sample_names.append(f"{g}_{i+1}")
            group_labels.append(g)
            is_qc.append(False)
            batch.append("1" if i < N_PER_GROUP // 2 else "2")
    for i in range(N_QC):
        sample_names.append(f"QC_{i+1}")
        group_labels.append("QC")
        is_qc.append(True)
        batch.append("1" if i < N_QC // 2 else "2")
    if include_reference:
        sample_names.append("Reference_Pool")
        group_labels.append("Reference")
        is_qc.append(False)
        batch.append("1")
    return sample_names, group_labels, is_qc, batch


def generate_dataset(n_proteins: int, include_reference: bool, seed: int, out_prefix: str, write_meta: bool = True):
    rng = np.random.default_rng(seed)
    sample_names, group_labels, is_qc, batch = _build_sample_metadata(rng, include_reference)
    n_samples = len(sample_names)

    n_proteins = min(n_proteins, len(GENE_SYMBOLS))
    gene_symbols = sorted(rng.choice(GENE_SYMBOLS, size=n_proteins, replace=False).tolist())
    protein_names = [f"{g}_HUMAN" for g in gene_symbols]

    base = rng.lognormal(mean=10, sigma=1.0, size=(n_proteins, 1))
    data = base * rng.lognormal(mean=0, sigma=0.15, size=(n_proteins, n_samples))

    group_idx = {g: [i for i, lab in enumerate(group_labels) if lab == g] for g in GROUPS}

    # ~15% of features: progressive trend across Control -> Mild -> Moderate -> Severe
    n_trend = max(3, int(0.15 * n_proteins))
    trend_feats = rng.choice(range(n_proteins), size=n_trend, replace=False)
    for f in trend_feats:
        direction = rng.choice([1, -1])  # increasing or decreasing severity trend
        step = rng.uniform(0.35, 0.7)
        for stage_i, g in enumerate(GROUPS):
            if g == "Control":
                continue
            fold = (1 + direction * step * stage_i)
            fold = max(fold, 0.15)  # keep positive
            data[f, group_idx[g]] *= fold

    # ~8% of features: only "Severe" differs sharply (acute marker pattern)
    remaining = [i for i in range(n_proteins) if i not in trend_feats]
    n_acute = max(2, int(0.08 * n_proteins))
    acute_feats = rng.choice(remaining, size=min(n_acute, len(remaining)), replace=False)
    for f in acute_feats:
        fold = rng.choice([3.0, 3.5, 0.25, 0.3])
        data[f, group_idx["Severe"]] *= fold

    if include_reference:
        # Reference/pooled bridge channel: the same pooled material in every plex,
        # so it should sit near the average of the study samples for each protein,
        # with only small technical (channel-to-channel) variation on top.
        ref_idx = sample_names.index("Reference_Pool")
        study_idx = [i for i, q in enumerate(is_qc) if not q and sample_names[i] != "Reference_Pool"]
        data[:, ref_idx] = data[:, study_idx].mean(axis=1) * rng.lognormal(mean=0, sigma=0.05, size=n_proteins)

    # Inject realistic missingness (encoded as exact zero, the common MS-based
    # proteomics convention for "not detected"/"not quantified") into biological
    # samples only -- QC replicates and the reference channel stay complete,
    # matching real-world expectations of high QC reproducibility and a reliably-
    # quantified pooled reference. Mixture of missingness rates so the demo
    # exercises every category in the Data Cleaning & Imputation tab:
    #   ~70% of features: 0-15% missing (typical low-level "not detected" noise)
    #   ~20% of features: 20-50% missing (moderate -- "keep with careful imputation")
    #   ~10% of features: 55-85% missing (heavy -- demonstrates the removal guidance)
    bio_sample_idx = [i for i, is_q in enumerate(is_qc) if not is_q and sample_names[i] != "Reference_Pool"]
    for f in range(n_proteins):
        tier = rng.random()
        if tier < 0.70:
            miss_rate = rng.uniform(0.0, 0.15)
        elif tier < 0.90:
            miss_rate = rng.uniform(0.20, 0.50)
        else:
            miss_rate = rng.uniform(0.55, 0.85)
        n_miss = int(round(miss_rate * len(bio_sample_idx)))
        if n_miss > 0:
            miss_cols = rng.choice(bio_sample_idx, size=n_miss, replace=False)
            data[f, miss_cols] = 0.0

    peak_df = pd.DataFrame(data, index=protein_names, columns=sample_names)
    peak_df.insert(0, "Gene", gene_symbols)
    peak_df.insert(0, "Protein", protein_names)
    peak_path = f"{BASE_DIR}/sample_protein_intensity_matrix_{out_prefix}.csv"
    peak_df.to_csv(peak_path, index=False)

    meta_df = pd.DataFrame({
        "Sample": sample_names, "Group": group_labels, "IsQC": is_qc, "Batch": batch,
    })
    if write_meta:
        meta_path = f"{BASE_DIR}/sample_metadata_{out_prefix}.csv"
        meta_df.to_csv(meta_path, index=False)

    print(f"[{out_prefix}] {peak_df.shape[0]} proteins (real gene symbols) x {len(sample_names)} samples "
          f"({N_QC} QC, {len(sample_names) - N_QC - int(include_reference)} biological across "
          f"{len(GROUPS)} groups"
          + (", 1 reference channel" if include_reference else "")
          + f") -> {peak_path}")
    return peak_df, meta_df


# 1. TMT demo: includes a pooled Reference_Pool channel, smaller panel (typical of one TMT plex)
generate_dataset(n_proteins=60, include_reference=True, seed=42, out_prefix="tmt")

# 2. Label-Free demo: no reference channel, larger feature set (typical of LFQ) --
#    uses the full curated gene panel
generate_dataset(n_proteins=len(GENE_SYMBOLS), include_reference=False, seed=43, out_prefix="labelfree")

print("Demo datasets written.")
