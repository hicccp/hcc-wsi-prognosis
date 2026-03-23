"""
Figure: Kaplan-Meier survival curve + Gene expression heatmap.

Inputs:
  RISK_COHORT_CSV : 106-patient cohort with group labels (from step2)
  TPM_MATRIX      : genes × samples TPM matrix
  DEG_RESULTS_CSV : differential expression results (from step3)

Outputs:
  OUT_DIR/Fig_KM.pdf / .png
  OUT_DIR/Fig_Heatmap.pdf / .png
"""

import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import zscore

from configs.paths import DEG_RESULTS_CSV, OUT_DIR, RISK_COHORT_CSV, TPM_MATRIX

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

HIGH_COLOR = "#D73027"
LOW_COLOR  = "#4575B4"
DPI        = 200
TOP_GENES  = 25      # top N up and down DEGs for heatmap


def load_cohort():
    df = pd.read_csv(RISK_COHORT_CSV)
    df.columns = df.columns.str.strip()
    return df


def load_tpm_subset(cohort: pd.DataFrame) -> pd.DataFrame:
    tpm = pd.read_csv(TPM_MATRIX, sep="\t", index_col=0)
    tpm.columns = [c[:12] for c in tpm.columns]
    return tpm


# ════════════════════════════════════════════════════════════════════════════
# Figure A: Kaplan-Meier curve
# ════════════════════════════════════════════════════════════════════════════
def plot_km(cohort: pd.DataFrame):
    g_hi = cohort[cohort["group"] == "High"]
    g_lo = cohort[cohort["group"] == "Low"]

    # Convert days to months
    hi_t = g_hi["time_day"] / 30.44
    lo_t = g_lo["time_day"] / 30.44

    lrt = logrank_test(hi_t, lo_t, g_hi["STATUS"], g_lo["STATUS"])

    fig, ax = plt.subplots(figsize=(6, 5))

    kmf_hi = KaplanMeierFitter()
    kmf_lo = KaplanMeierFitter()
    kmf_hi.fit(hi_t, g_hi["STATUS"], label=f"High-risk (n={len(g_hi)})")
    kmf_lo.fit(lo_t, g_lo["STATUS"], label=f"Low-risk (n={len(g_lo)})")

    kmf_hi.plot_survival_function(ax=ax, color=HIGH_COLOR, ci_show=True, ci_alpha=0.12, linewidth=2)
    kmf_lo.plot_survival_function(ax=ax, color=LOW_COLOR,  ci_show=True, ci_alpha=0.12, linewidth=2)

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Overall Survival Probability", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.85)

    p_fmt = f"p = {lrt.p_value:.4f}" if lrt.p_value >= 0.0001 else "p < 0.0001"
    ax.text(0.03, 0.05, f"Log-rank {p_fmt}", transform=ax.transAxes,
            fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"Fig_KM.{ext}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()

    print(f"Log-rank p = {lrt.p_value:.6f}")
    print(f"High: {len(g_hi)} patients, Low: {len(g_lo)} patients, Events: {cohort['STATUS'].sum()}")


# ════════════════════════════════════════════════════════════════════════════
# Figure B: Gene expression heatmap
# ════════════════════════════════════════════════════════════════════════════
def plot_heatmap(cohort: pd.DataFrame, tpm: pd.DataFrame, deg: pd.DataFrame):
    tpm_log = np.log2(tpm + 1)

    # Select top DEGs
    sig_up = deg[(deg["pval"] < 0.05) & (deg["log2FC"] > 0)].nsmallest(TOP_GENES, "pval")
    sig_dn = deg[(deg["pval"] < 0.05) & (deg["log2FC"] < 0)].nsmallest(TOP_GENES, "pval")
    top_genes = pd.concat([sig_up, sig_dn])["gene"].tolist()

    if len(sig_up) < 10:
        top_genes = deg.nsmallest(TOP_GENES * 2, "pval")["gene"].tolist()

    hi_samples = [s for s in cohort[cohort["group"] == "High"]["ID12"] if s in tpm_log.columns]
    lo_samples = [s for s in cohort[cohort["group"] == "Low"]["ID12"]  if s in tpm_log.columns]
    avail_genes = [g for g in top_genes if g in tpm_log.index]

    print(f"Heatmap: High={len(hi_samples)}, Low={len(lo_samples)}, Genes={len(avail_genes)}")

    mat = tpm_log.loc[avail_genes, hi_samples + lo_samples]
    mat_z = mat.apply(lambda row: zscore(row, ddof=1), axis=1).clip(-3, 3)

    # Gene annotation colors
    gene_color = []
    for g in avail_genes:
        fc = deg[deg["gene"] == g]["log2FC"].values[0]
        gene_color.append(HIGH_COLOR if fc > 0 else LOW_COLOR)

    n_hi = len(hi_samples)
    n_lo = len(lo_samples)
    n_genes = len(avail_genes)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, max(6, n_genes * 0.22 + 1)),
        gridspec_kw={"width_ratios": [n_hi, n_lo], "wspace": 0.02},
    )

    cmap = plt.cm.RdBu_r
    vmin, vmax = -3, 3

    for j, (ax, samples, label, color) in enumerate([
        (axes[0], hi_samples, f"High-risk\n(n={n_hi})", HIGH_COLOR),
        (axes[1], lo_samples, f"Low-risk\n(n={n_lo})",  LOW_COLOR),
    ]):
        ax.imshow(mat_z[samples].values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        if j == 0:
            ax.set_yticks(range(n_genes))
            ax.set_yticklabels(avail_genes, fontsize=7)
            for tick, gc in zip(ax.get_yticklabels(), gene_color):
                tick.set_color(gc)
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel(label, fontsize=10, color=color, fontweight="bold")

    axes[0].text(0.5, 1.01, "High-risk", transform=axes[0].transAxes,
                 ha="center", va="bottom", fontsize=10, color=HIGH_COLOR, fontweight="bold")
    axes[1].text(0.5, 1.01, "Low-risk",  transform=axes[1].transAxes,
                 ha="center", va="bottom", fontsize=10, color=LOW_COLOR,  fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-3, vmax=3))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Z-score (log\u2082TPM+1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"Fig_Heatmap.{ext}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    print(f"Cohort: {len(cohort)} patients  (High={( cohort['group']=='High').sum()}, "
          f"Low={(cohort['group']=='Low').sum()})")

    print("\n── KM curve ──")
    plot_km(cohort)

    print("\n── Heatmap ──")
    tpm = load_tpm_subset(cohort)
    deg = pd.read_csv(DEG_RESULTS_CSV)
    plot_heatmap(cohort, tpm, deg)

    print("\nAll figures done.")


if __name__ == "__main__":
    main()
