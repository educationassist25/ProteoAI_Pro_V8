"""
ppi.py - Protein-Protein Interaction network analysis via the STRING database
(Szklarczyk D et al. 2023, "The STRING database in 2023", Nucleic Acids Research)
-- by far the most widely used PPI resource in proteomics, integrating physical
interactions and functional associations from experiments, curated databases,
co-expression, and text-mining into one "combined_score" confidence per pair.
"""

import io
import numpy as np
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from modules.utils import build_colormap

matplotlib.use("Agg")

STRING_BASE = "https://string-db.org/api"

STRING_SPECIES = {
    "Human (Homo sapiens)": 9606, "Mouse (Mus musculus)": 10090, "Rat (Rattus norvegicus)": 10116,
    "Yeast (S. cerevisiae)": 4932, "Zebrafish (Danio rerio)": 7955,
}

# STRING's own confidence-tier convention (required_score is 0-1000, i.e. the
# combined_score x 1000 threshold).
CONFIDENCE_PRESETS = {
    "Low confidence (0.15)": 150, "Medium confidence (0.4)": 400,
    "High confidence (0.7)": 700, "Highest confidence (0.9)": 900,
}


class PPIError(Exception):
    pass


def fetch_string_network(gene_list, species=9606, required_score=400):
    """
    Query STRING's network endpoint for interactions among the given identifiers.
    Returns a DataFrame with one row per edge: preferredName_A, preferredName_B,
    score (combined_score, 0-1), plus individual evidence-channel scores
    (nscore=neighborhood, fscore=fusion, pscore=phylogenetic co-occurrence,
    ascore=co-expression, escore=experimental, dscore=curated databases,
    tscore=text-mining) -- STRING's standard evidence breakdown.
    """
    genes = [str(g).strip() for g in gene_list if str(g).strip()]
    if len(genes) < 2:
        raise PPIError("Need at least 2 proteins to build an interaction network.")
    try:
        resp = requests.post(
            f"{STRING_BASE}/tsv/network",
            data={"identifiers": "%0d".join(genes), "species": species,
                  "required_score": required_score, "caller_identity": "ProteoAI_Pro"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise PPIError(
            f"Could not reach STRING (string-db.org) — check your internet connection. "
            f"Details: {e}"
        )
    text = resp.text.strip()
    if not text or text.startswith("Error"):
        return pd.DataFrame(columns=["preferredName_A", "preferredName_B", "score"])
    return pd.read_csv(io.StringIO(text), sep="\t")


def fetch_string_ppi_enrichment(gene_list, species=9606):
    """
    STRING's network-level statistic: is this protein list significantly MORE
    interconnected than a random protein set of the same size? Reported in
    virtually every proteomics paper that uses STRING (the "PPI enrichment
    p-value"). Returns a dict: number_of_nodes, number_of_edges,
    average_node_degree, local_clustering_coefficient, expected_number_of_edges,
    p_value. Empty dict if STRING returns nothing (e.g. too few proteins).
    """
    genes = [str(g).strip() for g in gene_list if str(g).strip()]
    if len(genes) < 2:
        raise PPIError("Need at least 2 proteins for network enrichment statistics.")
    try:
        resp = requests.post(
            f"{STRING_BASE}/tsv/ppi_enrichment",
            data={"identifiers": "%0d".join(genes), "species": species,
                  "caller_identity": "ProteoAI_Pro"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise PPIError(
            f"Could not reach STRING (string-db.org) — check your internet connection. "
            f"Details: {e}"
        )
    text = resp.text.strip()
    if not text or text.startswith("Error"):
        return {}
    df = pd.read_csv(io.StringIO(text), sep="\t")
    return df.iloc[0].to_dict() if len(df) else {}


def build_graph(edge_df: pd.DataFrame, node_stats: pd.DataFrame = None) -> nx.Graph:
    """
    Build a networkx Graph from a STRING edge DataFrame (as returned by
    fetch_string_network()). node_stats (optional): DataFrame indexed by gene
    symbol with Log2FC/PValue-like columns, attached as node attributes for
    coloring/sizing in draw_network().
    """
    G = nx.Graph()
    if edge_df is None or edge_df.empty:
        return G
    a_col = "preferredName_A" if "preferredName_A" in edge_df.columns else edge_df.columns[2]
    b_col = "preferredName_B" if "preferredName_B" in edge_df.columns else edge_df.columns[3]
    score_col = "score" if "score" in edge_df.columns else edge_df.columns[-1]
    for _, row in edge_df.iterrows():
        G.add_edge(str(row[a_col]), str(row[b_col]), weight=float(row[score_col]))

    lfc_col = pval_col = None
    if node_stats is not None:
        lfc_col = next((c for c in node_stats.columns if c.lower() in ("log2fc", "log2_fold_change")), None)
        pval_col = next((c for c in node_stats.columns if c.lower() in ("pvalue", "p_value", "p-value")), None)
    for node in G.nodes():
        log2fc, pval = 0.0, 1.0
        if node_stats is not None and node in node_stats.index:
            if lfc_col:
                log2fc = float(node_stats.loc[node, lfc_col])
            if pval_col:
                pval = float(node_stats.loc[node, pval_col])
        G.nodes[node]["log2fc"] = log2fc
        G.nodes[node]["pvalue"] = pval
    return G


def draw_network(G: nx.Graph, layout: str = "spring", seed: int = 42, cmap_name: str = "RdBu_r",
                  custom_colors=None, reverse_cmap: bool = False,
                  vmin: float = -2.0, vmax: float = 2.0, node_size_by: str = "pvalue",
                  min_node_size: float = 150, max_node_size: float = 900,
                  figsize=None, show_labels: bool = True, label_fontsize: float = 8,
                  edge_alpha: float = 0.5):
    """
    Render the PPI network styled consistently with the rest of ProteoAI Pro: nodes
    colored by Log2FC (diverging colormap, same red=up/blue=down convention as the
    Volcano and Heatmap tabs), node size scaled by -log10(p-value) or by degree
    (selectable), and edge width/opacity scaled by STRING's combined confidence
    score. Returns (fig, pos).

    cmap_name: a predefined matplotlib colormap name, used unless custom_colors is given.
    custom_colors: optional list of hex colors (e.g. from a 2D color picker) to build
    a custom gradient through — overrides cmap_name, same convention as the Heatmap
    tab's custom color gradient.
    """
    if G.number_of_nodes() == 0:
        raise PPIError("No nodes to draw (empty network).")
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=1.6 / max(np.sqrt(G.number_of_nodes()), 1))
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    if figsize is None:
        n = G.number_of_nodes()
        side = max(6.0, min(14.0, 4.0 + 0.12 * n))
        figsize = (side, side)
    fig, ax = plt.subplots(figsize=figsize)

    nodes = list(G.nodes())
    log2fcs = np.array([G.nodes[n_].get("log2fc", 0.0) for n_ in nodes])
    pvals = np.array([G.nodes[n_].get("pvalue", 1.0) for n_ in nodes])
    cmap = build_colormap(cmap_name, custom_colors, reverse_cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    node_colors = [cmap(norm(v)) for v in log2fcs]

    if node_size_by == "pvalue":
        neglogp = -np.log10(np.clip(pvals, 1e-300, 1))
        if neglogp.max() > neglogp.min():
            sizes = min_node_size + (neglogp - neglogp.min()) / (neglogp.max() - neglogp.min()) * (max_node_size - min_node_size)
        else:
            sizes = np.full(len(pvals), (min_node_size + max_node_size) / 2)
    else:
        degrees = np.array([G.degree(n_) for n_ in nodes])
        if degrees.max() > degrees.min():
            sizes = min_node_size + (degrees - degrees.min()) / (degrees.max() - degrees.min()) * (max_node_size - min_node_size)
        else:
            sizes = np.full(len(degrees), (min_node_size + max_node_size) / 2)

    edge_weights = np.array([G[u][v].get("weight", 0.5) for u, v in G.edges()]) if G.number_of_edges() else np.array([])
    edge_widths = 0.5 + edge_weights * 3.5

    if G.number_of_edges():
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="#999999", alpha=edge_alpha)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_color=node_colors, node_size=sizes,
                            edgecolors="black", linewidths=0.6)
    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=label_fontsize)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Log2 Fold Change", fontsize=8)
    ax.set_title(f"Protein-Protein Interaction Network ({G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig, pos


def network_summary_table(G: nx.Graph) -> pd.DataFrame:
    """
    Per-node Degree, Betweenness Centrality, and Clustering Coefficient -- the
    standard trio of PPI hub-identification statistics reported in proteomics
    papers to flag likely "hub" proteins (high degree/betweenness).
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["Protein", "Degree", "Betweenness_Centrality", "Clustering_Coefficient"])
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    rows = [{"Protein": n_, "Degree": degree[n_], "Betweenness_Centrality": betweenness[n_],
             "Clustering_Coefficient": clustering[n_]} for n_ in G.nodes()]
    return pd.DataFrame(rows).sort_values("Degree", ascending=False).reset_index(drop=True)


def export_figure(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """Shared figure export, consistent with the other visualization modules."""
    buf = io.BytesIO()
    save_fmt = "jpeg" if fmt == "jpg" else fmt
    fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
