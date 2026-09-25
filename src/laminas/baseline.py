"""
baseline.py
===========
Baseline obrigatório (prioridade 0 da Task 1.1): limiar HSV do pacote ``rockface``
(LCCMat-UnB/rockface, ``czi_process/rockface/masks.py``), reproduzido fielmente.
"""

from __future__ import annotations

import cv2
import numpy as np


def rockface_mask(rgb: np.ndarray) -> np.ndarray:
    """Reprodução de ``generate_pore_mask_array`` do rockface. Retorna bool (True = poro).

    O rockface recebe BGR; aqui convertemos de RGB antes de aplicar os mesmos passos:
    blur 5x5 → HSV → matiz 75–125 (escala OpenCV) → S·V ≥ 0,1 → abertura com elipse 3x3.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    hue, sat, val = cv2.split(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV))
    coarse = cv2.inRange(hue, 75, 125)
    sv = (sat.astype(np.float32) / 255.0) * (val.astype(np.float32) / 255.0)
    refined = np.where((coarse > 0) & (sv >= 0.1), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel) > 0


def rockface_rock_area(rgb: np.ndarray) -> np.ndarray:
    """Máscara de 'rocha' do rockface (``legacy/petrophysical_properties.py``): qualquer canal > 5.

    Atenção: nestes patches escuros, 7–27 % dos pixels (grãos escuros) ficam com todos
    os canais ≤ 5 e saem da área de rocha, o que infla a porosidade.
    """
    return np.any(rgb > 5, axis=-1)
