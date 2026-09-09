"""
demo_gene_symbols.py - A curated panel of REAL, verifiable human gene symbols
(HGNC nomenclature), themed around metabolic syndrome / type 2 diabetes / lipid
metabolism / inflammation / cardiovascular comorbidity -- matching the demo
datasets' Control/Mild/Moderate/Severe and Healthy/Prediabetic/T2D/Metabolic
Syndrome group designs.

Used (instead of placeholder "Protein_001"-style names) so the built-in demo
datasets return genuine, biologically meaningful results when run through the
GSEA and P-P Interaction tabs, which call the real STRING and Enrichr APIs --
those services only recognize real gene symbols, and a thematically coherent
panel like this one will show real enriched pathways (e.g. "Type II diabetes
mellitus", "Insulin signaling", "PPAR signaling", "Adipocytokine signaling") and
a real, densely-interconnected STRING network, rather than sparse/no results.

Curated from standard, widely-published pathway gene sets: KEGG hsa04910
(Insulin signaling), hsa04930 (Type II diabetes mellitus), hsa03320 (PPAR
signaling), hsa04920 (Adipocytokine signaling), hsa04152 (AMPK signaling),
hsa04064 (NF-kB signaling), hsa04610 (Complement and coagulation cascades),
plus well-known GWAS type-2-diabetes susceptibility loci and canonical
lipid/cholesterol/bile-acid metabolism genes.
"""

GENE_SYMBOLS = list(dict.fromkeys([
    # Insulin signaling / glucose handling
    "INS", "INSR", "IRS1", "IRS2", "IRS4", "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2",
    "AKT1", "AKT2", "AKT3", "PDPK1", "GSK3B", "GSK3A", "FOXO1", "FOXO3", "PRKCZ",
    "SLC2A1", "SLC2A2", "SLC2A3", "SLC2A4", "GCK", "G6PC", "G6PC2", "PCK1", "PCK2",
    "PYGL", "PYGM", "GYS1", "GYS2", "PPP1R3A", "PRKAA1", "PRKAA2", "PRKAB1",
    "PRKAB2", "PRKAG1", "STK11", "TBC1D1", "TBC1D4", "SORBS1", "CBL", "CAP1",
    "SOCS1", "SOCS3", "PTPN1", "PTEN", "RHEB",

    # Adipokines, cytokines, inflammation
    "ADIPOQ", "ADIPOR1", "ADIPOR2", "LEP", "LEPR", "RETN", "NAMPT", "TNF",
    "TNFRSF1A", "TNFRSF1B", "IL6", "IL6R", "IL1B", "IL1R1", "IL10", "IL18",
    "CCL2", "CCR2", "NFKB1", "NFKB2", "RELA", "IKBKB", "CHUK", "NLRP3", "CASP1",
    "TLR4", "TLR2", "MYD88", "CRP", "SAA1", "SERPINE1", "ICAM1", "VCAM1", "SELE",
    "IFNG", "IL4", "IL13", "IL17A", "TGFB1",

    # Lipid / cholesterol / bile acid metabolism
    "APOA1", "APOA2", "APOA5", "APOB", "APOC1", "APOC2", "APOC3", "APOE", "LDLR",
    "LDLRAP1", "PCSK9", "LPL", "LIPC", "LIPG", "CETP", "ABCA1", "ABCG1", "ABCG5",
    "ABCG8", "SCARB1", "HMGCR", "HMGCS1", "SREBF1", "SREBF2", "SCAP", "INSIG1",
    "INSIG2", "FASN", "ACACA", "ACACB", "SCD", "ELOVL6", "DGAT1", "DGAT2",
    "PLIN1", "PLIN2", "PNPLA2", "LIPE", "MGLL", "CPT1A", "CPT1B", "CPT2", "ACOX1",
    "ACADVL", "ACADM", "PPARA", "PPARG", "PPARD", "PPARGC1A", "PPARGC1B", "RXRA",
    "NR1H3", "NR1H4", "FGF21", "FGF19", "FGFR1", "KLB", "CYP7A1", "CYP27A1",
    "NR0B2", "ABCB11", "SLC10A1",

    # mTOR / energy sensing / sirtuins
    "MTOR", "RPTOR", "RICTOR", "AKT1S1", "TSC1", "TSC2", "EIF4EBP1", "RPS6KB1",
    "SIRT1", "SIRT3",

    # Type-2-diabetes GWAS susceptibility genes
    "TCF7L2", "KCNJ11", "ABCC8", "KCNQ1", "HNF1A", "HNF4A", "HNF1B", "GCKR",
    "SLC30A8", "CDKAL1", "CDKN2A", "CDKN2B", "IGF2BP2", "FTO", "MC4R", "WFS1",
    "GLIS3", "ADCY5",

    # Coagulation / complement (metabolic-syndrome-associated vascular risk)
    "C3", "C4A", "CFB", "F2", "F7", "F10", "PLAT", "PLAU", "SERPINA1",
    "SERPINC1", "FGA", "FGB", "FGG", "VWF",

    # Myokines / muscle
    "MSTN", "IL15", "FNDC5", "CKM", "MYOD1",

    # Liver / kidney function markers commonly co-panelled in metabolic studies
    "ALB", "CST3", "LCN2", "GPT", "GOT1", "GGT1",

    # Cardiovascular / renin-angiotensin (metabolic syndrome comorbidity)
    "AGT", "REN", "ACE", "AGTR1", "NPPA", "NPPB", "EDN1", "NOS3", "NOS2",
    "VEGFA", "HIF1A", "EPO", "EPOR",

    # Oxidative stress / ER stress (implicated in beta-cell dysfunction)
    "SOD1", "SOD2", "CAT", "GPX1", "NFE2L2", "HMOX1", "HSPA5", "XBP1", "ATF4",
    "ATF6", "ERN1", "DDIT3",

    # Cell cycle / apoptosis / growth signaling (general panel breadth)
    "TP53", "MDM2", "BAX", "BCL2", "CASP3", "CASP9", "CDKN1A", "MYC", "JUN",
    "FOS", "EGFR", "ERBB2",

    # Housekeeping / general reference proteins
    "GAPDH", "ACTB", "TUBB", "HSP90AA1",
]))
