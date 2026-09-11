"""
enrichment.py - Gene Set Enrichment Analysis for proteomics, offering the three
methods most commonly used in the literature:

1. STRING functional enrichment (Szklarczyk D et al. 2023, "The STRING database in
   2023", Nucleic Acids Research) -- hypergeometric enrichment computed server-side
   by STRING against its own curated GO/KEGG/Reactome/Pfam/InterPro annotation sets.
   Zero setup beyond a protein list; extremely common in proteomics papers since it
   reuses the same STRING lookup as the P-P Interaction tab.
2. Over-Representation Analysis via Enrichr (Chen EY et al. 2013, BMC Bioinformatics;
   Kuleshov MV et al. 2016, Nucleic Acids Research) -- hypergeometric test of a
   significant gene list against a chosen curated gene set library, whole-genome
   background. The most widely used general-purpose web enrichment tool in biology.
3. Preranked GSEA (Subramanian A et al. 2005, PNAS "Gene set enrichment analysis: a
   knowledge-based approach..."; gene-set permutation as implemented by GSEAPY's
   prerank mode, Fang Z et al. 2023, Bioinformatics) -- ranks ALL detected proteins
   by a signed score and computes a weighted running-sum enrichment statistic per
   gene set, rather than requiring a hard significance cutoff first.

All three expect gene SYMBOLS (not raw UniProt accessions) as protein identifiers --
the convention Enrichr, STRING, and MSigDB/.gmt gene sets all share. If your protein
intensity matrix uses a different identifier, convert to gene symbol (e.g. via
UniProt's ID mapping service, or the "Gene names" column many search engines like
MaxQuant already report) before using this tab.
"""

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
STRING_BASE = "https://string-db.org/api"

ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2026", "GO_Molecular_Function_2026", "GO_Cellular_Component_2026",
    "KEGG_2026", "Reactome_Pathways_2024", "WikiPathways_2024_Human", "WikiPathways_2024_Mouse", "HMDB_Metabolites", "MSigDB_Hallmark_2020",
]

STRING_SPECIES = {
    "Human (Homo sapiens)": 9606, "Mouse (Mus musculus)": 10090, "Rat (Rattus norvegicus)": 10116,
    "Yeast (S. cerevisiae)": 4932, "Zebrafish (Danio rerio)": 7955,
}


class EnrichmentError(Exception):
    pass


# ---------------------------------------------------------------------------
# Method 1: STRING functional enrichment
# ---------------------------------------------------------------------------
def run_string_enrichment(gene_list, species=9606):
    """
    Functional enrichment via the STRING database API. Returns a DataFrame with
    columns: category (e.g. 'Process', 'KEGG', 'RCTM'), term, description,
    number_of_genes, number_of_genes_in_background, p_value, fdr, inputGenes.
    """
    genes = [str(g).strip() for g in gene_list if str(g).strip()]
    if len(genes) < 1:
        raise EnrichmentError("Need at least 1 gene for STRING enrichment.")
    try:
        resp = requests.post(
            f"{STRING_BASE}/tsv/enrichment",
            data={"identifiers": "%0d".join(genes), "species": species,
                  "caller_identity": "ProteoAI_Pro"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EnrichmentError(
            f"Could not reach STRING (string-db.org) — check your internet connection. "
            f"Details: {e}"
        )
    text = resp.text.strip()
    if not text or text.startswith("Error"):
        return pd.DataFrame(columns=["category", "term", "description", "number_of_genes",
                                      "p_value", "fdr", "inputGenes"])
    return pd.read_csv(io.StringIO(text), sep="\t")


# ---------------------------------------------------------------------------
# Method 2: Over-Representation Analysis via Enrichr
# ---------------------------------------------------------------------------
def run_enrichr_ora(gene_list, library="GO_Biological_Process_2023", description="ProteoAI_Pro"):
    """
    Over-representation analysis via the Enrichr REST API (addList -> enrich).
    Returns a DataFrame: Rank, Term, P-value, Adjusted P-value, Z-score,
    Combined Score, Overlap (n genes), Genes (overlapping gene symbols).
    """
    genes = [str(g).strip() for g in gene_list if str(g).strip()]
    if len(genes) < 3:
        raise EnrichmentError("Need at least 3 genes for over-representation analysis.")
    try:
        payload = {"list": (None, "\n".join(genes)), "description": (None, description)}
        resp = requests.post(f"{ENRICHR_BASE}/addList", files=payload, timeout=30)
        resp.raise_for_status()
        user_list_id = resp.json()["userListId"]

        resp2 = requests.get(
            f"{ENRICHR_BASE}/enrich",
            params={"userListId": user_list_id, "backgroundType": library},
            timeout=60,
        )
        resp2.raise_for_status()
        data = resp2.json().get(library, [])
    except requests.exceptions.RequestException as e:
        raise EnrichmentError(
            f"Could not reach Enrichr (maayanlab.cloud) — check your internet connection. "
            f"Details: {e}"
        )
    except (KeyError, ValueError) as e:
        raise EnrichmentError(f"Unexpected response from Enrichr: {e}")

    if not data:
        return pd.DataFrame(columns=["Rank", "Term", "P-value", "Adjusted P-value",
                                      "Z-score", "Combined Score", "Overlap", "Genes"])
    rows = []
    for row in data:
        # Enrichr row format: [rank, term, p-value, z-score, combined score, genes, adj p-value, ...]
        rank, term, pval, zscore, combined_score, overlap_genes = row[0], row[1], row[2], row[3], row[4], row[5]
        adj_pval = row[6] if len(row) > 6 else np.nan
        rows.append({
            "Rank": rank, "Term": term, "P-value": pval, "Adjusted P-value": adj_pval,
            "Z-score": zscore, "Combined Score": combined_score,
            "Overlap": len(overlap_genes), "Genes": ";".join(overlap_genes),
        })
    return pd.DataFrame(rows)


def get_enrichr_gene_set_library(library="GO_Biological_Process_2023"):
    """
    Fetch a full gene set library from Enrichr in GMT format for use as the gene-set
    source for Preranked GSEA (Method 3) when no custom .gmt file is uploaded.
    """
    try:
        resp = requests.get(
            f"{ENRICHR_BASE}/geneSetLibrary",
            params={"mode": "text", "libraryName": library}, timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EnrichmentError(
            f"Could not reach Enrichr (maayanlab.cloud) — check your internet connection. "
            f"Details: {e}"
        )
    return parse_gmt_text(resp.text)


def parse_gmt_text(text: str) -> dict:
    """
    Parse GMT-format text (the standard MSigDB/Enrichr convention:
    term<TAB>description<TAB>gene1<TAB>gene2<TAB>...) into {term: [gene_symbols]}.
    """
    gene_sets = {}
    for line in text.strip().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        term = parts[0].strip()
        genes = [g.strip().upper() for g in parts[2:] if g.strip()]
        if genes:
            gene_sets[term] = genes
    return gene_sets


# ---------------------------------------------------------------------------
# Method 3: Preranked GSEA (classic weighted running-sum statistic)
# ---------------------------------------------------------------------------
def compute_ranking_score(stats_df: pd.DataFrame, method: str = "signed_neglogp") -> pd.Series:
    """
    Build the per-protein ranking score for Preranked GSEA from a Statistics-tab-style
    table (must contain a Log2FC-like column and a PValue-like column).

    method:
      'signed_neglogp' : sign(log2FC) * -log10(p-value) -- the standard choice,
                          combining direction and significance (as used throughout
                          the GSEAPY prerank documentation/tutorials).
      'log2fc'         : log2 fold change alone.
    """
    lfc_col = next((c for c in stats_df.columns if c.lower() in ("log2fc", "log2_fold_change")), None)
    p_col = next((c for c in stats_df.columns if c.lower() in ("pvalue", "p_value", "p-value")), None)
    if lfc_col is None or p_col is None:
        raise EnrichmentError("Statistics table must contain a Log2FC and a PValue column.")
    lfc = stats_df[lfc_col].astype(float)
    pval = stats_df[p_col].astype(float).clip(lower=1e-300)
    if method == "log2fc":
        score = lfc
    else:
        score = np.sign(lfc) * -np.log10(pval)
    return score.replace([np.inf, -np.inf], 0).fillna(0)


def _weighted_running_sum(ranked_scores: np.ndarray, hit_mask: np.ndarray, weight: float = 1.0):
    """
    Core Subramanian et al. 2005 weighted Kolmogorov-Smirnov running-sum statistic.
    Returns (ES, running_sum_array).
    """
    n = len(ranked_scores)
    n_hits = int(hit_mask.sum())
    if n_hits == 0:
        return 0.0, np.zeros(n)
    abs_scores = np.abs(ranked_scores) ** weight
    hit_sum = abs_scores[hit_mask].sum()
    if hit_sum == 0:
        hit_step = np.where(hit_mask, 1.0 / n_hits, 0.0)
    else:
        hit_step = np.where(hit_mask, abs_scores / hit_sum, 0.0)
    n_miss = n - n_hits
    miss_step = np.where(~hit_mask, 1.0 / max(n_miss, 1), 0.0)
    running_sum = np.cumsum(hit_step - miss_step)
    peak_idx = int(np.argmax(np.abs(running_sum)))
    es = running_sum[peak_idx]
    return es, running_sum


def run_prerank_gsea(ranked_scores: pd.Series, gene_sets: dict, weight: float = 1.0,
                      n_perm: int = 1000, min_size: int = 15, max_size: int = 500,
                      seed: int = 42) -> pd.DataFrame:
    """
    Preranked GSEA. ranked_scores: Series indexed by gene symbol (from
    compute_ranking_score()). gene_sets: {term: [genes]}.

    Significance is estimated via GENE-SET permutation (shuffle which ranked
    positions count as "hits", same set size, recompute ES, repeat n_perm times) --
    the same approach GSEAPY's prerank mode uses, since the original Subramanian et
    al. phenotype-permutation procedure needs the per-sample expression matrix and
    group labels, which aren't available from a ranked list alone.

    Returns a DataFrame: Term, Size, ES, NES, P-value, FDR (BH across tested gene
    sets), Leading_Edge_Size, Leading_Edge (top 20 leading-edge genes).
    """
    rng = np.random.default_rng(seed)
    ranked_scores = ranked_scores.sort_values(ascending=False)
    genes_ranked = ranked_scores.index.to_numpy()
    scores_arr = ranked_scores.to_numpy(dtype=float)
    n = len(genes_ranked)
    gene_pos = {g: i for i, g in enumerate(genes_ranked)}

    results = []
    for term, term_genes in gene_sets.items():
        hit_idx = [gene_pos[g] for g in term_genes if g in gene_pos]
        size = len(hit_idx)
        if size < min_size or size > max_size:
            continue
        hit_mask = np.zeros(n, dtype=bool)
        hit_mask[hit_idx] = True
        es, running_sum = _weighted_running_sum(scores_arr, hit_mask, weight)

        null_es = np.empty(n_perm)
        for p in range(n_perm):
            perm_idx = rng.choice(n, size=size, replace=False)
            perm_mask = np.zeros(n, dtype=bool)
            perm_mask[perm_idx] = True
            null_es[p], _ = _weighted_running_sum(scores_arr, perm_mask, weight)

        if es >= 0:
            pos_null = null_es[null_es >= 0]
            denom = pos_null.mean() if pos_null.size and pos_null.mean() != 0 else 1.0
            nes = es / denom
            pval = ((pos_null >= es).sum() + 1) / (pos_null.size + 1) if pos_null.size else 1.0
        else:
            neg_null = null_es[null_es < 0]
            denom = abs(neg_null.mean()) if neg_null.size and neg_null.mean() != 0 else 1.0
            nes = es / denom
            pval = ((neg_null <= es).sum() + 1) / (neg_null.size + 1) if neg_null.size else 1.0

        peak_idx = int(np.argmax(np.abs(running_sum)))
        if es >= 0:
            le_genes = [genes_ranked[i] for i in hit_idx if i <= peak_idx]
        else:
            le_genes = [genes_ranked[i] for i in hit_idx if i >= peak_idx]

        results.append({
            "Term": term, "Size": size, "ES": es, "NES": nes, "P-value": pval,
            "Leading_Edge_Size": len(le_genes), "Leading_Edge": ";".join(le_genes[:20]),
        })

    if not results:
        return pd.DataFrame(columns=["Term", "Size", "ES", "NES", "P-value", "FDR",
                                      "Leading_Edge_Size", "Leading_Edge"])
    out = pd.DataFrame(results)
    out["FDR"] = multipletests(out["P-value"], method="fdr_bh")[1]
    return out.sort_values("NES", ascending=False).reset_index(drop=True)


def running_score_plot(ranked_scores: pd.Series, gene_sets: dict, term: str, weight: float = 1.0):
    """
    The classic GSEA 'mountain plot' for one gene set: running enrichment score
    across the ranked list (top), a barcode of hit positions (middle), and the
    ranking metric itself (bottom) -- the same 3-panel layout used by the Broad
    Institute's GSEA desktop tool and reproduced by GSEAPY's plotting functions.
    """
    if term not in gene_sets:
        raise EnrichmentError(f"Gene set '{term}' not found.")
    ranked_scores = ranked_scores.sort_values(ascending=False)
    genes_ranked = ranked_scores.index.to_numpy()
    scores_arr = ranked_scores.to_numpy(dtype=float)
    n = len(genes_ranked)
    gene_pos = {g: i for i, g in enumerate(genes_ranked)}
    hit_idx = sorted(gene_pos[g] for g in gene_sets[term] if g in gene_pos)
    if not hit_idx:
        raise EnrichmentError(f"None of '{term}'s genes were found in the ranked list.")
    hit_mask = np.zeros(n, dtype=bool)
    hit_mask[hit_idx] = True
    es, running_sum = _weighted_running_sum(scores_arr, hit_mask, weight)
    peak_idx = int(np.argmax(np.abs(running_sum)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 0.6, 1.5], "hspace": 0.08})
    axes[0].plot(range(n), running_sum, color="#2E7D32", linewidth=1.6)
    axes[0].axhline(0, color="gray", linewidth=0.6)
    axes[0].axvline(peak_idx, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Running Enrichment Score")
    axes[0].set_title(f"{term}\nES = {es:.3f}", fontsize=10)

    axes[1].vlines(hit_idx, 0, 1, color="black", linewidth=0.6)
    axes[1].set_yticks([])
    axes[1].set_ylabel("Hits", fontsize=8)

    axes[2].fill_between(range(n), scores_arr, color="#4C72B0", alpha=0.7, linewidth=0)
    axes[2].axhline(0, color="gray", linewidth=0.6)
    axes[2].set_ylabel("Ranking Score")
    axes[2].set_xlabel("Rank in ordered protein list")

    return fig
