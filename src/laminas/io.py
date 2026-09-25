"""
io.py
=====
Leitura dos patches e reamostragem de imagens e máscaras.

Convenções: imagens RGB ``uint8`` (os PNGs ``patch_y*_x*_c0.png`` já estão em RGB);
``scale`` é o fator de reamostragem da imagem de trabalho (1.0 = resolução nativa).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import DATA_DIR

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class Patch:
    path: Path
    y: int
    x: int

    @property
    def name(self) -> str:
        return f"y{self.y}_x{self.x}"


def list_patches(data_dir: Path = DATA_DIR) -> list[Patch]:
    patches = []
    for p in sorted(Path(data_dir).glob("patch_y*_x*_c0.png")):
        m = re.match(r"patch_y(\d+)_x(\d+)_c0", p.stem)
        patches.append(Patch(p, int(m.group(1)), int(m.group(2))))
    if not patches:
        raise FileNotFoundError(f"Nenhum patch_y*_x*_c0.png em {data_dir} (ver data/README.md).")
    return patches


def load_rgb(patch: Patch, scale: float = 1.0) -> np.ndarray:
    img = np.asarray(Image.open(patch.path).convert("RGB"))
    return resize(img, scale, cv2.INTER_AREA)


def resize(arr: np.ndarray, scale: float, interp: int) -> np.ndarray:
    if scale == 1.0:
        return arr
    h, w = arr.shape[:2]
    return cv2.resize(arr, (round(w * scale), round(h * scale)), interpolation=interp)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reamostra máscara/rótulo por vizinho mais próximo para ``shape`` (h, w)."""
    if mask.shape[:2] == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
