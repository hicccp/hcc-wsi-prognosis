"""
Step 3 — Differential expression and GSEA between High- and Low-risk groups.

Inputs:
  RISK_COHORT_CSV : 106-patient cohort with group labels (from step2)
  TPM_MATRIX      : genes × samples TPM matrix (gzipped TSV)

Outputs:
  DEG_RESULTS_CSV  : per-gene statistics (gene, log2FC, t_stat, pval, qval)
  GSEA_RESULTS_CSV : fgsea enrichment results (MSigDB Hallmark gene sets)

Method:
  - DEG: Welch two-sample t-test (unequal variance); BH-FDR correction
  - GSEA: gseapy prerank on t-statistic; gene set = MSigDB_Hallmark_2020
"""

import warnings

import gseapy as gp
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from configs.paths import DEG_RESULTS_CSV, GSEA_RESULTS_CSV, RISK_COHORT_CSV, TPM_MATRIX

warnings.filterwarnings("ignore")

MIN_TPM = 0.5        # exclude genes with median TPM below this threshold
MIN_SAMPLES_EXPR = 10  # minimum samples with TPM > 0 in each group


def load_tpm(tpm_path, sample_ids_12: list) -> pd.DataFrame:
    """Load TPM matrix and subset to given 12-character TCGA IDs."""
    print("Loading TPM matrix (this may take a moment)...")
    tpm = pd.read_csv(tpm_path, sep="\t", index_col=0)
    # Columns are full barcodes or 12-char IDs
    tpm.columns = [c[:12] for c in tpm.columns]
    # Keep only samples in cohort
    avail = [s for s in sample_ids_12 if s in tpm.columns]
    print(f"  Samples matched: {len(avail)} / {len(sample_ids_12)}")
    return tpm[avail]


def filter_lowly_expressed(tpm: pd.DataFrame) -> pd.DataFrame:
    """Remove genes with very low expression across all samples."""
    median_expr = tpm.median(axis=1)
    keep = median_expr >= MIN_TPM
    tpm = tpm.loc[keep]
    print(f"  Genes after expression filter: {len(tpm)}")
    return tpm


def run_deg(tpm_log: pd.DataFrame, hi_ids: list, lo_ids: list) -> pd.DataFrame:
    """
    Welch t-test comparing High vs Low groups.
    Returns DataFrame sorted by t-statistic (descending).
    """
    hi_cols = [c for c in hi_ids if c in tpm_log.columns]
    lo_cols = [c for c in lo_ids if c in tpm_log.columns]
    print(f"  High samples: {len(hi_cols)}, Low samples: {len(lo_cols)}")

    hi_mat = tpm_log[hi_cols].values
    lo_mat = tpm_log[lo_cols].values

    t_stats, pvals = stats.ttest_ind(hi_mat, lo_mat, axis=1, equal_var=False)
    log2fc = hi_mat.mean(axis=1) - lo_mat.mean(axis=1)

    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")

    results = pd.DataFrame(
        {
            "gene": tpm_log.index,
            "log2FC": log2fc,
            "t_stat": t_stats,
            "pval": pvals,
            "qval": qvals,
        }
    ).sort_values("t_stat", ascending=False)

    n_sig = ((results["pval"] < 0.05) & (results["log2FC"].abs() > 0.5)).sum()
    print(f"  DEGs (p<0.05, |FC|>0.5): {n_sig}")
    return results


def run_gsea(deg_results: pd.DataFrame, gene_set: str = "MSigDB_Hallmark_2020") -> pd.DataFrame:
    """
    Prerank GSEA using t-statistic ranking.
    Returns enrichment results DataFrame.
    """
    ranking = (
        deg_results[["gene", "t_stat"]]
        .dropna()
        .drop_duplicates("gene")
        .set_index("gene")["t_stat"]
        .sort_values(ascending=False)
    )

    pre_res = gp.prerank(
        rnk=ranking,
        gene_sets=gene_set,
        permutation_num=1000,
        min_size=15,
        max_size=500,
        seed=42,
        verbose=False,
    )
    res = pre_res.res2d.sort_values("NES", ascending=False)
    print(f"  Enriched terms (FDR<0.25): {(res['FDR q-val'] < 0.25).sum()}")
    return res


def main():
    # ── Load cohort ──────────────────────────────────────────────────────────
    cohort = pd.read_csv(RISK_COHORT_CSV)
    cohort.columns = cohort.columns.str.strip()
    hi_ids = cohort[cohort["group"] == "High"]["ID12"].tolist()
    lo_ids = cohort[cohort["group"] == "Low"]["ID12"].tolist()
    print(f"Cohort: High={len(hi_ids)}, Low={len(lo_ids)}")

    # ── Load and preprocess TPM ──────────────────────────────────────────────
    all_ids = hi_ids + lo_ids
    tpm = load_tpm(TPM_MATRIX, all_ids)
    tpm = filter_lowly_expressed(tpm)
    tpm_log = np.log2(tpm + 1)

    # ── DEG ──────────────────────────────────────────────────────────────────
    print("\nRunning DEG analysis...")
    deg = run_deg(tpm_log, hi_ids, lo_ids)
    deg.to_csv(DEG_RESULTS_CSV, index=False)
    print(f"Saved: {DEG_RESULTS_CSV}")

    # ── GSEA ─────────────────────────────────────────────────────────────────
    print("\nRunning GSEA (prerank, MSigDB Hallmark)...")
    gsea = run_gsea(deg)
    gsea.to_csv(GSEA_RESULTS_CSV)
    print(f"Saved: {GSEA_RESULTS_CSV}")

    # Print top enriched pathways
    print("\nTop enriched pathways (NES sorted):")
    top = gsea[gsea["FDR q-val"] < 0.25].head(10)
    for _, row in top.iterrows():
        print(f"  {row['Term']:<50s}  NES={row['NES']:.2f}  FDR={row['FDR q-val']:.3f}")


if __name__ == "__main__":
    main()
