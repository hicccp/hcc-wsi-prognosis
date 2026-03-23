"""
Step 2 — Build 106-patient risk cohort from WSI predictions.

Combines:
  - Out-of-fold WSI risk predictions (oof_preds.csv)
  - External clinical table (tcga_data_formatted_final.csv)
  - GDC API supplementation for 5 patients without WSI predictions

Risk stratification:
  - Top 43 patients by RiskScore  → High-risk group
  - Remaining 63 patients         → Low-risk group

Output:
  RISK_COHORT_CSV : /tmp/tcga106_complete.csv
    Columns: ID, ID12, RiskScore, time_day, STATUS, group
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

from configs.paths import (
    EXT_CLINICAL_CSV,
    FINAL_RISK_TABLE,
    OOF_PREDS_CSV,
    RISK_COHORT_CSV,
)

# ── GDC API settings ─────────────────────────────────────────────────────────
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
GDC_FIELDS = [
    "submitter_id",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_death",
    "diagnoses.vital_status",
]

HIGH_N = 43       # number of patients assigned to High-risk group
TOTAL_N = 106     # expected total cohort size


def query_gdc_os(tcga_ids: list) -> pd.DataFrame:
    """
    Query GDC API for OS data of given TCGA patient IDs (TCGA-XX-XXXX format).
    Returns DataFrame with columns: ID12, time_day, STATUS.
    """
    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "submitter_id", "value": tcga_ids},
        },
        "fields": ",".join(GDC_FIELDS),
        "format": "json",
        "size": len(tcga_ids),
    }
    resp = requests.post(GDC_CASES_URL, json=payload, timeout=30)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    rows = []
    for hit in hits:
        pid = hit["submitter_id"]
        diag = hit.get("diagnoses", [{}])[0]
        vital = diag.get("vital_status", "Unknown")
        status = 1 if vital.lower() == "dead" else 0
        if status == 1:
            t = diag.get("days_to_death")
        else:
            t = diag.get("days_to_last_follow_up")
        rows.append({"ID12": pid, "time_day": t, "STATUS": status})
    return pd.DataFrame(rows)


def main():
    # ── Load OOF predictions ─────────────────────────────────────────────────
    print("Loading OOF predictions...")
    oof = pd.read_csv(OOF_PREDS_CSV)
    oof.columns = oof.columns.str.strip()
    oof["ID12"] = oof["ID"].str[:12]
    print(f"  OOF samples: {len(oof)}")

    # ── Load external clinical table ─────────────────────────────────────────
    print("Loading external clinical table...")
    clin = pd.read_csv(EXT_CLINICAL_CSV)
    clin.columns = clin.columns.str.strip()
    clin["ID12"] = clin["ID"].str[:12]

    # Merge OOF with clinical data
    merged = oof.merge(clin[["ID12", "OS_lab", "STATUS_lab"]], on="ID12", how="left")
    merged = merged.rename(columns={"OS_lab": "time_day", "STATUS_lab": "STATUS"})

    # Use time_day from OOF file if available (more accurate)
    if "time_day" in oof.columns:
        merged["time_day"] = oof["time_day"].values

    n_with_pred = len(merged)
    print(f"  Merged: {n_with_pred} patients")

    # ── Identify patients in clinical but not in OOF ─────────────────────────
    all_ids_12 = set(clin["ID12"].unique())
    pred_ids_12 = set(merged["ID12"].unique())
    missing_ids = list(all_ids_12 - pred_ids_12)
    print(f"  Patients without WSI predictions: {len(missing_ids)}")

    # ── Supplement missing patients via GDC API ──────────────────────────────
    if missing_ids:
        print(f"  Querying GDC API for {len(missing_ids)} patients: {missing_ids}")
        time.sleep(0.5)
        gdc_df = query_gdc_os(missing_ids)
        gdc_df["ID"] = gdc_df["ID12"]
        gdc_df["RiskScore"] = float("nan")
        missing_complete = gdc_df[["ID", "ID12", "RiskScore", "time_day", "STATUS"]]
        print(f"  GDC returned: {len(missing_complete)} records")
        for _, row in missing_complete.iterrows():
            print(f"    {row['ID12']}: time={row['time_day']}d, STATUS={row['STATUS']}")
    else:
        missing_complete = pd.DataFrame(
            columns=["ID", "ID12", "RiskScore", "time_day", "STATUS"]
        )

    # ── Combine and rank ─────────────────────────────────────────────────────
    base = merged[["ID", "ID12", "RiskScore", "time_day", "STATUS"]].copy()
    full = pd.concat([base, missing_complete], ignore_index=True)
    full = full.dropna(subset=["time_day"])
    full["time_day"] = full["time_day"].astype(float)
    full["STATUS"] = full["STATUS"].astype(int)

    print(f"\nFull cohort: {len(full)} patients")

    # ── Assign risk groups ───────────────────────────────────────────────────
    # Sort by RiskScore; NaN (supplemented patients) go to Low group
    full = full.sort_values("RiskScore", ascending=False, na_position="last")
    full["group"] = "Low"
    full.iloc[:HIGH_N, full.columns.get_loc("group")] = "High"

    n_high = (full["group"] == "High").sum()
    n_low = (full["group"] == "Low").sum()
    n_events = full["STATUS"].sum()
    print(f"  High-risk: {n_high}, Low-risk: {n_low}")
    print(f"  Events (deaths): {n_events}/{len(full)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    full.to_csv(RISK_COHORT_CSV, index=False)
    print(f"\nSaved: {RISK_COHORT_CSV}")


if __name__ == "__main__":
    main()
