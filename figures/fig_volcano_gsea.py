"""
Figure: Volcano plot (DEG) + GSEA enrichment bar chart.

Inputs:
  DEG_RESULTS_CSV  : per-gene statistics (from step3)
  GSEA_RESULTS_CSV : fgsea enrichment results (from step3)

Outputs:
  OUT_DIR/Fig_Volcano.pdf / .png
  OUT_DIR/Fig_GSEA.pdf / .png
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.paths import DEG_RESULTS_CSV, GSEA_RESULTS_CSV, OUT_DIR

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

HIGH_COLOR = "#D73027"
LOW_COLOR  = "#4575B4"
GREY_COLOR = "#AAAAAA"
DPI        = 200

LABEL_FC   = 1.0    # |log2FC| threshold for labelling genes
LABEL_PVAL = 0.01   # p-value threshold for labelling genes
MAX_LABELS = 15     # maximum gene labels per direction


# ════════════════════════════════════════════════════════════════════════════
# Figure A: Volcano plot
# ════════════════════════════════════════════════════════════════════════════
def plot_volcano(deg: pd.DataFrame):
    deg = deg.copy()
    deg["-log10p"] = -np.log10(deg["pval"].clip(lower=1e-300))

    # Significance categories
    sig_up = (deg["pval"] < 0.05) & (deg["log2FC"] > 0.5)
    sig_dn = (deg["pval"] < 0.05) & (deg["log2FC"] < -0.5)
    ns      = ~(sig_up | sig_dn)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(deg.loc[ns,     "log2FC"], deg.loc[ns,     "-log10p"],
               c=GREY_COLOR, s=6, alpha=0.5, linewidths=0, label="NS")
    ax.scatter(deg.loc[sig_up, "log2FC"], deg.loc[sig_up, "-log10p"],
               c=HIGH_COLOR, s=8, alpha=0.7, linewidths=0, label=f"Up in High ({sig_up.sum()})")
    ax.scatter(deg.loc[sig_dn, "log2FC"], deg.loc[sig_dn, "-log10p"],
               c=LOW_COLOR,  s=8, alpha=0.7, linewidths=0, label=f"Up in Low ({sig_dn.sum()})")

    # Threshold lines
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline( 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(-0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    # Gene labels for top significant genes
    label_up = deg[sig_up & (deg["log2FC"].abs() > LABEL_FC) & (deg["pval"] < LABEL_PVAL)]
    label_dn = deg[sig_dn & (deg["log2FC"].abs() > LABEL_FC) & (deg["pval"] < LABEL_PVAL)]
    label_up = label_up.nsmallest(MAX_LABELS, "pval")
    label_dn = label_dn.nsmallest(MAX_LABELS, "pval")

    for _, row in pd.concat([label_up, label_dn]).iterrows():
        ax.text(row["log2FC"], row["-log10p"], row["gene"],
                fontsize=6, ha="center", va="bottom", alpha=0.85)

    ax.set_xlabel("log\u2082 Fold Change (High / Low)", fontsize=12)
    ax.set_ylabel("-log\u2081\u2080 p-value", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"Fig_Volcano.{ext}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure B: GSEA horizontal bar chart
# ════════════════════════════════════════════════════════════════════════════
def plot_gsea(gsea: pd.DataFrame, fdr_cutoff: float = 0.25, top_n: int = 20):
    # Filter significant pathways
    sig = gsea[gsea["FDR q-val"] < fdr_cutoff].copy()
    if len(sig) == 0:
        print("No GSEA terms pass FDR cutoff — plotting top results regardless.")
        sig = gsea.head(top_n).copy()

    # Clean term names (remove leading database prefix)
    sig["Term"] = sig["Term"].str.replace(r"^HALLMARK_", "", regex=True).str.replace("_", " ")

    # Sort by NES
    sig = sig.sort_values("NES")
    if len(sig) > top_n:
        # Keep top_n/2 most positive and top_n/2 most negative
        half = top_n // 2
        sig = pd.concat([sig.head(half), sig.tail(half)])
        sig = sig.sort_values("NES")

    colors = [HIGH_COLOR if nes > 0 else LOW_COLOR for nes in sig["NES"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sig) * 0.35 + 1)))
    bars = ax.barh(range(len(sig)), sig["NES"], color=colors, edgecolor="none")

    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["Term"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=11)

    # FDR annotations
    for i, (_, row) in enumerate(sig.iterrows()):
        fdr = row["FDR q-val"]
        label = f"FDR={fdr:.3f}" if fdr >= 0.001 else "FDR<0.001"
        x_pos = row["NES"] + (0.05 if row["NES"] > 0 else -0.05)
        ha = "left" if row["NES"] > 0 else "right"
        ax.text(x_pos, i, label, va="center", ha=ha, fontsize=6, color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=HIGH_COLOR, label="Enriched in High-risk"),
        mpatches.Patch(color=LOW_COLOR,  label="Enriched in Low-risk"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right", framealpha=0.85)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"Fig_GSEA.{ext}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    deg  = pd.read_csv(DEG_RESULTS_CSV)
    gsea = pd.read_csv(GSEA_RESULTS_CSV, index_col=0)

    print(f"DEG entries: {len(deg)}")
    print(f"GSEA entries: {len(gsea)}")

    print("\n── Volcano plot ──")
    plot_volcano(deg)

    print("\n── GSEA bar chart ──")
    plot_gsea(gsea)

    print("\nAll figures done.")


if __name__ == "__main__":
    main()
