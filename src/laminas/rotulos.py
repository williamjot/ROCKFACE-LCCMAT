"""
rotulos.py
==========
Rótulos: leitura de rabiscos/máscaras de especialistas, pseudo-rabiscos do piloto e
gravação de PNG indexado.

Formato: PNG indexado (modo "P") com 0 = sem rótulo, 1 = poro, 2 = sólido; ou PNG
RGB(A) pintado em vermelho (poro) e verde (sólido), com o resto preto/transparente.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import PORE, SOLID
from .io import resize_mask

# 0 = transparente, 1 = vermelho (poro), 2 = verde (sólido)
PALETTE = [0, 0, 0, 255, 0, 0, 0, 200, 0] + [0] * (256 - 3) * 3


def load_labels(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Lê rótulos (indexado 0/1/2 ou RGB vermelho/verde) → uint8 0/1/2."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(int)
        lab = np.zeros(arr.shape[:2], np.uint8)
        lab[(rgb[..., 0] > 128) & (rgb[..., 1] < 100)] = PORE
        lab[(rgb[..., 1] > 128) & (rgb[..., 0] < 100)] = SOLID
        if arr.shape[-1] == 4:
            lab[arr[..., 3] == 0] = 0
        arr = lab
    return resize_mask(arr, shape) if shape is not None else arr


def save_labels(lab: np.ndarray, path: Path) -> None:
    """Grava rótulos 0/1/2 como PNG indexado com paleta vermelho/verde e fundo transparente."""
    im = Image.fromarray(lab.astype(np.uint8), mode="P")
    im.putpalette(PALETTE)
    im.save(path, transparency=0)


def bootstrap_scribbles(base_mask: np.ndarray, rng: np.random.Generator,
                        erode_px: int = 6, n_per_class: int = 20000) -> np.ndarray:
    """PSEUDO-rótulos provisórios para o piloto, enquanto não há rabiscos de especialistas.

    Amostra pixels só do *miolo* das regiões do baseline (erodidas de ``erode_px``),
    evitando bordas e objetos pequenos, onde o limiar é menos confiável. O modelo ainda
    herda o viés do limiar — por isso o piloto mede concordância com o rockface, não acerto.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    pore_core = cv2.erode(base_mask.astype(np.uint8), se) > 0
    solid_core = cv2.erode((~base_mask).astype(np.uint8), se) > 0
    lab = np.zeros(base_mask.shape, np.uint8)
    for cls, region in ((PORE, pore_core), (SOLID, solid_core)):
        idx = np.flatnonzero(region)
        if idx.size:
            pick = rng.choice(idx, size=min(n_per_class, idx.size), replace=False)
            lab.flat[pick] = cls
    return lab
