"""
Figure: ABMIL attention heatmap for representative High- and Low-risk patients.

Layout: 2 rows (High-risk / Low-risk) × 3 columns
  Col 1 — Full H&E thumbnail
  Col 2 — Full attention heatmap (jet colormap, white background for non-tissue)
           with a blue ROI box indicating the zoomed region
  Col 3 — Zoomed H&E of the selected ROI (no overlay)

Inputs:
  SVS_INDEX_CSV : table with columns ID, svs_path — maps patient ID to SVS file
  MODEL_DIR     : directory with fold0/ … fold4/ subdirs, each containing ckpt.pt
  EMBEDDINGS_DIR: directory with per-patient .pt files containing
                    tile features (shape [N, 768]) and
                    tile coordinates encoded in filenames as _x{x}_y{y}_

Usage:
  python figures/fig_attention_heatmap.py

Outputs:
  OUT_DIR/Fig_AttnHeatmap.pdf / .png
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

from configs.paths import EMBEDDINGS_DIR, MODEL_DIR, OUT_DIR, SVS_INDEX_CSV
from models.abmil import ensemble_attention, load_ensemble

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Patient selection ─────────────────────────────────────────────────────────
# Replace with representative patients from your cohort
HIGH_PATIENT = "TCGA-BC-A10Q"   # high-risk patient ID (12-char)
LOW_PATIENT  = "TCGA-DD-AADG"   # low-risk patient ID (12-char)

# Zoom region: relative position (0–1) in image space [x_start, y_start, width, height]
# Adjust per patient to highlight histologically interesting areas
HIGH_ROI = (0.30, 0.20, 0.25, 0.35)
LOW_ROI  = (0.25, 0.25, 0.30, 0.35)

# Visual settings
BOX_COLOR  = "#1565C0"
BOX_LW     = 2.5
THUMB_LEVEL = 2      # OpenSlide level for thumbnail (lower = higher resolution)
DPI         = 200
TILE_SIZE   = 512    # pixels per tile used during feature extraction


def load_svs_paths(svs_index_csv: Path) -> dict:
    """Return {ID12: svs_path} from index CSV."""
    import pandas as pd
    df = pd.read_csv(svs_index_csv)
    df.columns = df.columns.str.strip()
    return dict(zip(df["ID"].str[:12], df["svs_path"]))


def read_thumbnail(svs_path: str, level: int = THUMB_LEVEL) -> np.ndarray:
    """Read WSI thumbnail as RGB numpy array at the given pyramid level."""
    import openslide
    slide = openslide.OpenSlide(svs_path)
    level = min(level, slide.level_count - 1)
    dims  = slide.level_dimensions[level]
    thumb = slide.read_region((0, 0), level, dims).convert("RGB")
    slide.close()
    return np.array(thumb)


def get_full_dims(svs_path: str):
    """Return (width, height) of the WSI at level 0."""
    import openslide
    slide = openslide.OpenSlide(svs_path)
    dims  = slide.dimensions
    slide.close()
    return dims


def parse_tile_coords(embed_file: Path) -> tuple:
    """
    Extract (x, y) tile origin (level-0 pixels) from filename.
    Expected pattern: ..._x{x}_y{y}_...pt or ..._x{x}_y{y}.pt
    """
    import re
    m = re.search(r"_x(\d+)_y(\d+)", embed_file.stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def build_attention_map(
    patient_id: str,
    models: list,
    embeddings_dir: Path,
    full_w: int,
    full_h: int,
) -> np.ndarray:
    """
    Compute full-resolution attention heatmap as a float32 array [H, W].
    Non-tissue pixels are left at -1 (rendered white).
    """
    # Find embedding files for this patient
    pt_files = sorted(embeddings_dir.glob(f"{patient_id}*.pt"))
    if not pt_files:
        # Try subdirectory
        pt_files = sorted((embeddings_dir / patient_id).glob("*.pt"))

    if not pt_files:
        raise FileNotFoundError(f"No embedding files found for {patient_id} in {embeddings_dir}")

    # Load all tile features
    feats_list, coords_list = [], []
    for pf in pt_files:
        x, y = parse_tile_coords(pf)
        if x is None:
            continue
        feat = torch.load(str(pf), map_location="cpu")
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        feats_list.append(feat)
        coords_list.append((x, y))

    if not feats_list:
        raise ValueError(f"No parseable tile coordinates found for {patient_id}")

    # Stack and compute ensemble attention
    all_feats = torch.cat(feats_list, dim=0).unsqueeze(0)   # [1, N, D]
    attn = ensemble_attention(all_feats, models)             # [N]

    # Normalize to [0, 1]
    attn_min, attn_max = attn.min(), attn.max()
    if attn_max > attn_min:
        attn_norm = (attn - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attn)

    # Render into full-resolution float map
    heat = np.full((full_h, full_w), -1.0, dtype=np.float32)
    for (x, y), a in zip(coords_list, attn_norm):
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + TILE_SIZE, full_w), min(y1 + TILE_SIZE, full_h)
        heat[y1:y2, x1:x2] = a

    return heat


def render_heatmap_rgb(heat: np.ndarray) -> np.ndarray:
    """
    Convert float attention map to RGB image.
    Non-tissue pixels (value == -1) → white.
    Tissue pixels → jet colormap.
    """
    cmap = plt.cm.jet
    norm = Normalize(vmin=0.0, vmax=1.0)
    rgba = cmap(norm(np.clip(heat, 0, 1)))
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)

    # White background for non-tissue
    non_tissue = (heat < 0)
    rgb[non_tissue] = 255

    return rgb


def plot_patient_row(
    axes,
    svs_path: str,
    patient_id: str,
    models: list,
    roi: tuple,
    row_label: str,
):
    """Fill one row (3 axes) for a given patient."""
    ax_he, ax_ht, ax_zoom = axes

    # ── Col 1: full H&E thumbnail ────────────────────────────────────────────
    thumb = read_thumbnail(svs_path)
    H_he, W_he = thumb.shape[:2]

    ax_he.imshow(thumb, aspect="auto")
    ax_he.set_xlim(0, W_he)
    ax_he.set_ylim(H_he, 0)
    ax_he.autoscale(False)
    ax_he.axis("off")

    # ── Compute attention heatmap ────────────────────────────────────────────
    full_w, full_h = get_full_dims(svs_path)
    heat = build_attention_map(patient_id, models, EMBEDDINGS_DIR, full_w, full_h)

    # Downscale heat map to match thumbnail resolution
    from PIL import Image
    heat_img = Image.fromarray(np.uint8(np.clip(heat * 255, 0, 255)))
    heat_img = heat_img.resize((W_he, H_he), Image.BILINEAR)
    heat_thumb = np.array(heat_img).astype(np.float32) / 255.0

    # Mark non-tissue (original -1) regions
    scale_x = W_he / full_w
    scale_y = H_he / full_h
    non_tissue_mask = heat < 0
    # Downscale mask
    mask_img = Image.fromarray(non_tissue_mask.astype(np.uint8) * 255)
    mask_img = mask_img.resize((W_he, H_he), Image.NEAREST)
    mask_thumb = np.array(mask_img) > 128

    heat_rgb = render_heatmap_rgb(heat_thumb)
    heat_rgb[mask_thumb] = 255   # white for non-tissue

    # ── Col 2: heatmap + ROI box ─────────────────────────────────────────────
    rx, ry, rw, rh = roi
    bx = int(rx * W_he)
    by = int(ry * H_he)
    bw = int(rw * W_he)
    bh = int(rh * H_he)

    ax_ht.imshow(heat_rgb, aspect="auto", interpolation="nearest")
    ax_ht.set_xlim(0, W_he)
    ax_ht.set_ylim(H_he, 0)
    ax_ht.autoscale(False)
    ax_ht.add_patch(Rectangle((bx, by), bw, bh,
                               linewidth=BOX_LW, edgecolor=BOX_COLOR, facecolor="none"))
    ax_ht.axis("off")

    # ── Col 3: zoomed H&E ────────────────────────────────────────────────────
    import openslide
    slide = openslide.OpenSlide(svs_path)

    zoom_x = int(rx * full_w)
    zoom_y = int(ry * full_h)
    zoom_w = int(rw * full_w)
    zoom_h = int(rh * full_h)
    # Read at level 1 for a reasonable resolution
    read_level = min(1, slide.level_count - 1)
    ds = slide.level_downsamples[read_level]
    region = slide.read_region(
        (zoom_x, zoom_y), read_level,
        (int(zoom_w / ds), int(zoom_h / ds))
    ).convert("RGB")
    slide.close()

    zoom_arr = np.array(region)
    Hzm, Wzm = zoom_arr.shape[:2]
    ax_zoom.imshow(zoom_arr, aspect="auto")
    ax_zoom.set_xlim(0, Wzm)
    ax_zoom.set_ylim(Hzm, 0)
    ax_zoom.autoscale(False)
    ax_zoom.axis("off")

    # Row label on the left
    ax_he.set_ylabel(row_label, fontsize=11, fontweight="bold", rotation=90,
                     va="center", labelpad=8)
    ax_he.yaxis.set_label_position("left")
    ax_he.axis("off")   # must re-call after set_ylabel for axis off


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading ABMIL ensemble...")
    models = load_ensemble(MODEL_DIR, n_folds=5, device="cpu")
    print(f"  Loaded {len(models)} fold models")

    # ── Load SVS paths ────────────────────────────────────────────────────────
    svs_map = load_svs_paths(SVS_INDEX_CSV)
    hi_svs  = svs_map.get(HIGH_PATIENT)
    lo_svs  = svs_map.get(LOW_PATIENT)

    if not hi_svs or not Path(hi_svs).exists():
        raise FileNotFoundError(f"SVS not found for High patient: {HIGH_PATIENT}")
    if not lo_svs or not Path(lo_svs).exists():
        raise FileNotFoundError(f"SVS not found for Low patient: {LOW_PATIENT}")

    # ── Layout ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 9),
        gridspec_kw={"wspace": 0.03, "hspace": 0.06},
    )

    # Column headers
    col_labels = ["H&E", "Attention Heatmap", "ROI (H&E)"]
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=11, pad=6)

    # ── Row 0: High-risk ─────────────────────────────────────────────────────
    print(f"Processing High-risk patient: {HIGH_PATIENT}")
    plot_patient_row(axes[0], hi_svs, HIGH_PATIENT, models, HIGH_ROI,
                     row_label="High-risk")

    # ── Row 1: Low-risk ──────────────────────────────────────────────────────
    print(f"Processing Low-risk patient: {LOW_PATIENT}")
    plot_patient_row(axes[1], lo_svs, LOW_PATIENT, models, LOW_ROI,
                     row_label="Low-risk")

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention Weight", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    cbar.ax.tick_params(labelsize=8)

    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"Fig_AttnHeatmap.{ext}"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
