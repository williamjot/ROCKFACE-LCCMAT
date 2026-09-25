"""
metricas.py
===========
Métricas por pixel (IoU, Dice, precisão, revocação), F1 de borda e porosidade.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def seg_metrics(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    if valid is not None:
        pred, ref = pred[valid], ref[valid]
    tp = np.sum(pred & ref)
    fp = np.sum(pred & ~ref)
    fn = np.sum(~pred & ref)
    tn = np.sum(~pred & ~ref)
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def boundary_f1(pred: np.ndarray, ref: np.ndarray, tol: int = 2) -> float:
    """F1 de borda: fração de pixels de contorno a até ``tol`` px do contorno da outra máscara."""
    def edges(m):
        m8 = m.astype(np.uint8)
        return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0
    ep, er = edges(pred), edges(ref)
    if not ep.any() or not er.any():
        return float(ep.any() == er.any())
    dp = ndi.distance_transform_edt(~ep)
    dr = ndi.distance_transform_edt(~er)
    prec = np.mean(dr[ep] <= tol)
    rec = np.mean(dp[er] <= tol)
    return 2 * prec * rec / (prec + rec + 1e-9)


def porosity(mask: np.ndarray, rock: np.ndarray | None = None) -> float:
    """Porosidade (%) = poros ∩ rocha / rocha. Sem ``rock``, usa a área inteira."""
    if rock is None:
        return 100.0 * mask.mean()
    return 100.0 * np.sum(mask & rock) / max(np.sum(rock), 1)
