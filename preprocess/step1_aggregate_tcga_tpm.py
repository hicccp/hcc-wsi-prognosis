"""
Step 1 — Aggregate TCGA-LIHC RNA-seq TPM files into a single matrix.

Input:
  GDC_DIR      : Directory containing per-sample subdirectories from GDC download.
                 Each subdirectory holds one *counts.tsv (or *tpm_unstranded.tsv).
  GDC_MANIFEST : GDC manifest file (used to map file_id → sample_id).

Output:
  TPM_MATRIX   : Gzipped TSV (genes × samples), column names = TCGA-XX-XXXX-01A style.

Usage:
  python preprocess/step1_aggregate_tcga_tpm.py
"""

import gzip
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from configs.paths import GDC_DIR, GDC_MANIFEST, TPM_MATRIX


def parse_manifest(manifest_path: Path) -> dict:
    """Return {file_id: filename} from GDC manifest."""
    manifest = pd.read_csv(manifest_path, sep="\t")
    return dict(zip(manifest["id"], manifest["filename"]))


def find_tpm_files(gdc_dir: Path) -> list:
    """Recursively find all TPM TSV files under gdc_dir."""
    patterns = ["*tpm_unstranded*.tsv", "*tpm*.tsv", "*counts*.tsv"]
    files = []
    for pat in patterns:
        files.extend(gdc_dir.rglob(pat))
    # Deduplicate
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def read_tpm_file(filepath: Path) -> pd.Series:
    """
    Read a GDC RNA-seq counts file and return the TPM column as a Series
    indexed by Ensembl gene ID (ENSG…), with gene_name as secondary info.
    """
    df = pd.read_csv(filepath, sep="\t", comment="#")

    # GDC format: gene_id, gene_name, gene_type, unstranded, stranded_first,
    #             stranded_second, tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded
    tpm_col = None
    for col in ["tpm_unstranded", "TPM", "tpm"]:
        if col in df.columns:
            tpm_col = col
            break

    if tpm_col is None:
        raise ValueError(f"No TPM column found in {filepath}")

    # Remove summary rows (N_unmapped etc.)
    df = df[df["gene_id"].str.startswith("ENSG")]
    df = df.set_index("gene_id")[tpm_col]
    return df


def extract_sample_id(filepath: Path, file_id_map: dict) -> str:
    """
    Derive TCGA aliquot barcode (e.g. TCGA-BC-A10Q-01A) from file path.
    Falls back to the parent directory name.
    """
    parent = filepath.parent.name
    # parent is often the GDC file UUID
    if parent in file_id_map:
        fname = file_id_map[parent]
        # Typical filename: TCGA-BC-A10Q-01A-..._tpm_unstranded.tsv
        match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z])", fname)
        if match:
            return match.group(1)
    # Fallback: try to extract from filepath itself
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z])", str(filepath))
    if match:
        return match.group(1)
    return parent  # last resort


def main():
    print("Scanning GDC directory for TPM files...")
    tpm_files = find_tpm_files(GDC_DIR)
    print(f"  Found {len(tpm_files)} TPM files")

    file_id_map = {}
    if GDC_MANIFEST.exists():
        file_id_map = parse_manifest(GDC_MANIFEST)
        print(f"  Manifest loaded: {len(file_id_map)} entries")

    series_list = {}
    for fpath in tqdm(tpm_files, desc="Reading TPM files"):
        try:
            sample_id = extract_sample_id(fpath, file_id_map)
            tpm_series = read_tpm_file(fpath)
            if sample_id in series_list:
                print(f"  Warning: duplicate sample {sample_id}, skipping second occurrence")
                continue
            series_list[sample_id] = tpm_series
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")

    print(f"\nAggregating {len(series_list)} samples...")
    matrix = pd.DataFrame(series_list)
    matrix.index.name = "gene_id"

    # Keep only primary tumor samples (barcode position 14-15 == "01")
    primary = [c for c in matrix.columns if c[13:15] == "01"]
    matrix = matrix[primary]
    print(f"  Primary tumor samples retained: {len(primary)}")

    # Remove genes with all-zero TPM
    matrix = matrix.loc[(matrix > 0).any(axis=1)]
    print(f"  Genes after zero-filter: {len(matrix)}")

    print(f"\nSaving to {TPM_MATRIX} ...")
    with gzip.open(TPM_MATRIX, "wt") as fh:
        matrix.to_csv(fh, sep="\t")
    print("Done.")


if __name__ == "__main__":
    main()
