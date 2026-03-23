"""
Step 4 — MRI arterial-phase radiomics extraction and Cox-LASSO survival model.

Inputs:
  MRI_DIR    : Root directory containing per-patient subdirs.
               Each subdir must have:
                 A.nrrd      — arterial-phase MRI volume
                 A-mask.nrrd — tumour segmentation mask
  MRI_OS_CSV : Clinical data with columns: ID, OS_day, STATUS

Outputs:
  MRI_FEATURES_CSV : radiomics feature matrix (patients × features)
  Prints selected features and Cox model coefficients.

Radiomics:
  PyRadiomics with firstorder, glcm, glrlm, glszm, gldm, shape feature classes.

Model:
  CoxnetSurvivalAnalysis (scikit-survival) with l1_ratio=1.0 (LASSO);
  alpha chosen by 5-fold cross-validation (highest concordance index).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from configs.paths import MRI_DIR, MRI_FEATURES_CSV, MRI_OS_CSV

warnings.filterwarnings("ignore")

# ── PyRadiomics settings ──────────────────────────────────────────────────────
RADIOMICS_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": [],
        "glcm": [],
        "glrlm": [],
        "glszm": [],
        "gldm": [],
        "shape": [],
    },
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "label": 1,
    },
}

CV_FOLDS = 5
ALPHA_N = 50     # number of alpha values to search


def extract_features_for_patient(patient_dir: Path) -> dict | None:
    """Extract radiomics features from A.nrrd + A-mask.nrrd for one patient."""
    img_path = patient_dir / "A.nrrd"
    mask_path = patient_dir / "A-mask.nrrd"
    if not img_path.exists() or not mask_path.exists():
        return None

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    for fc in RADIOMICS_PARAMS["featureClass"]:
        extractor.enableFeatureClassByName(fc)
    extractor.settings.update(RADIOMICS_PARAMS["setting"])

    try:
        result = extractor.execute(str(img_path), str(mask_path))
        features = {k: float(v) for k, v in result.items()
                    if k.startswith("original_") and not isinstance(v, sitk.Image)}
        return features
    except Exception as e:
        print(f"  Warning: extraction failed for {patient_dir.name}: {e}")
        return None


def extract_all_features(mri_dir: Path) -> pd.DataFrame:
    """Iterate all patient directories and aggregate feature matrix."""
    patient_dirs = sorted([d for d in mri_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories")

    rows = {}
    for pdir in patient_dirs:
        feats = extract_features_for_patient(pdir)
        if feats is not None:
            rows[pdir.name] = feats
        else:
            print(f"  Skipped: {pdir.name}")

    df = pd.DataFrame(rows).T
    df.index.name = "ID"
    print(f"Feature matrix: {df.shape[0]} patients × {df.shape[1]} features")
    return df


def build_survival_array(os_df: pd.DataFrame, ids: list):
    """Build structured array for scikit-survival."""
    sub = os_df.set_index("ID").loc[ids]
    return np.array(
        [(bool(row.STATUS), float(row.OS_day)) for _, row in sub.iterrows()],
        dtype=[("event", bool), ("time", float)],
    )


def cross_validate_cox(X: np.ndarray, y, alphas: np.ndarray) -> np.ndarray:
    """5-fold CV to choose regularization strength; returns mean CI per alpha."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    ci_matrix = np.zeros((CV_FOLDS, len(alphas)))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=alphas, fit_baseline_model=False)
        model.fit(X_tr_s, y_tr)

        for j, alpha in enumerate(alphas):
            try:
                risk = model.predict(X_va_s, alpha=alpha)
                ci = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
            except Exception:
                ci = 0.5
            ci_matrix[fold, j] = ci

    return ci_matrix.mean(axis=0)


def main():
    # ── Feature extraction ───────────────────────────────────────────────────
    if MRI_FEATURES_CSV.exists():
        print(f"Loading cached features from {MRI_FEATURES_CSV}")
        feat_df = pd.read_csv(MRI_FEATURES_CSV, index_col="ID")
    else:
        print("Extracting MRI radiomics features...")
        feat_df = extract_all_features(MRI_DIR)
        feat_df.to_csv(MRI_FEATURES_CSV)
        print(f"Saved: {MRI_FEATURES_CSV}")

    # ── Load survival labels ──────────────────────────────────────────────────
    os_df = pd.read_csv(MRI_OS_CSV)
    os_df.columns = os_df.columns.str.strip()

    # Align patients
    common_ids = [i for i in feat_df.index if i in os_df["ID"].values]
    print(f"\nPatients with both features and OS: {len(common_ids)}")

    X = feat_df.loc[common_ids].values.astype(float)
    y = build_survival_array(os_df, common_ids)
    feature_names = feat_df.columns.tolist()

    # Remove constant features
    std = X.std(axis=0)
    keep_mask = std > 1e-8
    X = X[:, keep_mask]
    feature_names = [f for f, k in zip(feature_names, keep_mask) if k]
    print(f"Features after variance filter: {X.shape[1]}")

    # ── Cox-LASSO ────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Generate alpha grid
    cox_init = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=ALPHA_N, fit_baseline_model=False)
    cox_init.fit(X_s, y)
    alphas = cox_init.alphas_

    print(f"\nCross-validating over {len(alphas)} alphas ({CV_FOLDS}-fold)...")
    mean_ci = cross_validate_cox(X_s, y, alphas)
    best_idx = int(np.argmax(mean_ci))
    chosen_alpha = cox_init.alphas_[best_idx]
    print(f"Best alpha: {chosen_alpha:.6f}  (CV CI = {mean_ci[best_idx]:.4f})")

    # Refit on all data
    cox_final = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[chosen_alpha], fit_baseline_model=False)
    cox_final.fit(X_s, y)
    coefs = cox_final.coef_[:, 0]

    # Selected features
    selected_mask = coefs != 0
    selected = [(feature_names[i], coefs[i]) for i in range(len(coefs)) if selected_mask[i]]
    selected.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nSelected features (LASSO, n={len(selected)}):")
    for fname, coef in selected:
        print(f"  {fname:<60s}  coef={coef:+.4f}")

    # Full concordance index on training data
    risk_all = cox_final.predict(X_s)
    ci_train = concordance_index_censored(y["event"], y["time"], risk_all)[0]
    print(f"\nTraining C-index: {ci_train:.4f}")


if __name__ == "__main__":
    main()
