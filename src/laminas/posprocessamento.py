"""
posprocessamento.py
===================
Probabilidade de poro → máscara (Task 1.1, Seção 6.2, passo 3).
"""

from __future__ import annotations

import cv2
import numpy as np


def postprocess(prob: np.ndarray, tau: float = 0.5, open_px: int = 1, min_area: int = 0) -> np.ndarray:
    """Limiar τ + abertura morfológica + remoção de objetos menores que ``min_area`` (em px)."""
    m = prob >= tau
    if open_px > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, se) > 0
    if min_area > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        m = keep[lab]
    return m


def disagreement_overlay(rgb_norm: np.ndarray, pred: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Imagem de comparação: amarelo = ambos poro; magenta = só o modelo; ciano = só o baseline."""
    img = (np.clip(rgb_norm, 0, 1) ** 0.6 * 255).astype(np.uint8).copy()
    for m, color in ((pred & base, (255, 220, 0)), (pred & ~base, (255, 0, 255)), (~pred & base, (0, 255, 255))):
        img[m] = (0.45 * img[m] + 0.55 * np.array(color)).astype(np.uint8)
    return img
