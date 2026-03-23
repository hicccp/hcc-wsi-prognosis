"""
Attention-Based Multiple Instance Learning (ABMIL) model for survival prediction.

Architecture:
  - Attention network: Linear(D→H) → Tanh → Dropout → Linear(H→1) → Softmax
  - Risk head:         Linear(D→H) → ReLU  → Dropout → Linear(H→1)
  - Aggregation:       z = sum(softmax(A) * X)  [attention-weighted pooling]

Reference:
  Ilse et al. "Attention-based deep multiple instance learning." ICML 2018.
"""

import torch
import torch.nn as nn


class ABMIL(nn.Module):
    """
    Attention-Based MIL for Cox proportional-hazards survival prediction.

    Args:
        input_dim  : Feature dimension per tile (default 768 for CTransPath).
        hidden_dim : Hidden dimension of attention and risk networks (default 256).
        dropout    : Dropout probability (default 0.1).

    Inputs:
        x    : Tensor of shape [B, N, D]  (batch, tiles, features)
        mask : Optional BoolTensor [B, N] — True for valid tiles, False for padding.

    Outputs:
        risk : Tensor [B, 1]   — log partial-hazard (higher = higher risk)
        A    : Tensor [B, N, 1] — normalized attention weights per tile
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: [B, N, D]
        A = self.attention_net(x)                  # [B, N, 1]

        if mask is not None:
            mask_exp = mask.unsqueeze(-1)          # [B, N, 1]
            A = A.masked_fill(mask_exp == 0, -1e4)

        A = torch.softmax(A, dim=1)                # [B, N, 1]

        if mask is not None:
            A = A * mask.unsqueeze(-1).float()

        z = torch.sum(A * x, dim=1)                # [B, D]
        risk = self.risk_head(z)                   # [B, 1]
        return risk, A


def load_ensemble(model_dir, n_folds: int = 5, device: str = "cpu") -> list:
    """
    Load all fold checkpoints and return a list of eval-mode ABMIL models.

    Args:
        model_dir : Path to directory containing fold0/, fold1/, ... subdirs.
        n_folds   : Number of folds (default 5).
        device    : Torch device string.

    Returns:
        List of loaded ABMIL models in eval mode.
    """
    import torch
    from pathlib import Path

    models = []
    for fold in range(n_folds):
        ckpt = Path(model_dir) / f"fold{fold}" / "ckpt.pt"
        if not ckpt.exists():
            continue
        m = ABMIL()
        m.load_state_dict(torch.load(str(ckpt), map_location=device))
        m.eval()
        models.append(m)
    return models


def ensemble_attention(feat: torch.Tensor, models: list) -> "np.ndarray":
    """
    Compute per-tile attention weights averaged over all ensemble models.

    Args:
        feat   : Float tensor [1, N, D].
        models : List of ABMIL models (from load_ensemble).

    Returns:
        Numpy array [N] of averaged attention weights.
    """
    import numpy as np
    import torch

    attn_list = []
    with torch.no_grad():
        for m in models:
            _, A = m(feat)
            attn_list.append(A.squeeze().numpy())
    return np.mean(np.stack(attn_list), axis=0)
