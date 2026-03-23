"""
Figure: Cox-LASSO regularization path for MRI arterial-phase radiomics.

Inputs:
  MRI_FEATURES_CSV : radiomics feature matrix (from step4)
  MRI_OS_CSV       : clinical survival data

Outputs:
  OUT_DIR/Fig_LASSO_Path.pdf / .png

The figure shows coefficient trajectories vs. log(alpha).
Selected features (non-zero at chosen alpha) are highlighted in distinct
tab20 colors; all others are drawn in light grey.
A vertical dashed line marks the cross-validated best alpha.
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from configs.paths import MRI_FEATURES_CSV, MRI_OS_CSV, OUT_DIR

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

CV_FOLDS = 5
ALPHA_N  = 100
DPI      = 200


def build_survival_array(os_df: pd.DataFrame, ids: list):
    sub = os_df.set_index("ID").loc[ids]
    return np.array(
        [(bool(row.STATUS), float(row.OS_day)) for _, row in sub.iterrows()],
        dtype=[("event", bool), ("time", float)],
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    feat_df = pd.read_csv(MRI_FEATURES_CSV, index_col="ID")
    os_df   = pd.read_csv(MRI_OS_CSV)
    os_df.columns = os_df.columns.str.strip()

    common = [i for i in feat_df.index if i in os_df["ID"].values]
    X = feat_df.loc[common].values.astype(float)
    y = build_survival_array(os_df, common)
    feature_names = feat_df.columns.tolist()

    # Remove constant features
    std = X.std(axis=0)
    keep = std > 1e-8
    X = X[:, keep]
    feature_names = [f for f, k in zip(feature_names, keep) if k]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # ── Fit LASSO path ───────────────────────────────────────────────────────
    print("Fitting LASSO path...")
    cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=ALPHA_N, fit_baseline_model=False)
    cox.fit(X_s, y)
    alphas      = cox.alphas_            # shape (n_alphas,)
    coef_matrix = cox.coef_             # shape (n_features, n_alphas)

    # ── Cross-validate to pick best alpha ────────────────────────────────────
    print(f"Cross-validating ({CV_FOLDS}-fold)...")
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    ci_matrix = np.zeros((CV_FOLDS, len(alphas)))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_s)):
        Xtr, Xva = X_s[tr_idx], X_s[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=alphas, fit_baseline_model=False)
        m.fit(Xtr, ytr)
        for j, a in enumerate(alphas):
            try:
                risk = m.predict(Xva, alpha=a)
                ci = concordance_index_censored(yva["event"], yva["time"], risk)[0]
            except Exception:
                ci = 0.5
            ci_matrix[fold, j] = ci

    mean_ci   = ci_matrix.mean(axis=0)
    best_idx  = int(np.argmax(mean_ci))
    best_alpha = alphas[best_idx]
    print(f"Best alpha: {best_alpha:.6f}  (CI = {mean_ci[best_idx]:.4f})")

    # Selected features at best alpha
    coefs_best = coef_matrix[:, best_idx]
    selected   = np.where(coefs_best != 0)[0]
    print(f"Selected features: {len(selected)}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    log_alphas = np.log10(alphas)
    colors     = plt.cm.tab20(np.linspace(0, 1, max(len(selected), 1)))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Grey lines for unselected features
    for i in range(coef_matrix.shape[0]):
        if i not in selected:
            ax.plot(log_alphas, coef_matrix[i], color="#CCCCCC", linewidth=0.6, alpha=0.5)

    # Coloured lines for selected features
    for k, feat_idx in enumerate(selected):
        short_name = feature_names[feat_idx]
        # Shorten long pyradiomics feature names for readability
        parts = short_name.split("_")
        if len(parts) >= 3:
            short_name = f"{parts[1]}: {parts[-1]}"
        ax.plot(log_alphas, coef_matrix[feat_idx], color=colors[k], linewidth=1.8,
                label=short_name)

    # Vertical line at chosen alpha
    ax.axvline(np.log10(best_alpha), color="black", linestyle="--", linewidth=1.2,
               label=f"Chosen \u03bb (CV CI={mean_ci[best_idx]:.3f})")

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("log\u2081\u2080(\u03b1)", fontsize=12)
    ax.set_ylabel("Coefficient", fontsize=12)

    # Legend — only for selected features (+ the chosen-alpha line)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85, ncol=1,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"Fig_LASSO_Path.{ext}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
