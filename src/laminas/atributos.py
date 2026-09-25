"""
atributos.py
============
Banco de atributos por pixel (Tabela 3 da Task 1.1), no padrão do ilastik:

- 9 atributos de cor: RGB, HSV, CIE L*a*b* (valor do pixel);
- 8 filtros × 6 escalas σ × 3 canais (L*, b*, S): gaussiana, magnitude do gradiente,
  DoG, LoG, 2 autovalores da Hessiana, 2 autovalores do tensor de estrutura.

Total: 9 + 3 × 6 × 8 = 153 atributos. As escalas σ são em pixels da imagem de trabalho.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hsv, rgb2lab
from skimage.feature import (
    hessian_matrix,
    hessian_matrix_eigvals,
    structure_tensor,
    structure_tensor_eigenvalues,
)

from .config import COLOR_NAMES, FILTER_NAMES, FILTERED_CHANNELS, SIGMAS


def feature_names() -> list[str]:
    names = [f"color_{c}" for c in COLOR_NAMES]
    for ch in FILTERED_CHANNELS:
        for s in SIGMAS:
            names += [f"{ch}_{f}_s{s}" for f in FILTER_NAMES]
    return names


def color_indices() -> list[int]:
    """Índices dos 9 atributos de cor (para a ablação 'só cor')."""
    return list(range(len(COLOR_NAMES)))


def color_channels(norm: np.ndarray) -> dict[str, np.ndarray]:
    hsv = rgb2hsv(norm)
    lab = rgb2lab(norm)
    # Lab reescalado para ordem de grandeza ~[0, 1], como os demais canais.
    return {
        "R": norm[..., 0], "G": norm[..., 1], "B": norm[..., 2],
        "H": hsv[..., 0], "S": hsv[..., 1], "V": hsv[..., 2],
        "L": lab[..., 0] / 100.0, "a": lab[..., 1] / 128.0, "b": lab[..., 2] / 128.0,
    }


def _filters_one_scale(ch: np.ndarray, s: float) -> list[np.ndarray]:
    g = ndi.gaussian_filter(ch, s)
    grad = ndi.gaussian_gradient_magnitude(ch, s)
    dog = g - ndi.gaussian_filter(ch, 1.6 * s)
    log = ndi.gaussian_laplace(ch, s)
    h1, h2 = hessian_matrix_eigvals(hessian_matrix(ch, sigma=s, order="rc", use_gaussian_derivatives=False))
    st1, st2 = structure_tensor_eigenvalues(structure_tensor(ch, sigma=s, order="rc"))
    return [g, grad, dog, log, h1, h2, st1, st2]


def compute_features(norm: np.ndarray) -> np.ndarray:
    """Calcula os 153 atributos de uma imagem normalizada em [0, 1]. Retorna float32 (H, W, 153)."""
    h, w = norm.shape[:2]
    names = feature_names()
    out = np.empty((h, w, len(names)), dtype=np.float32)
    chans = {k: v.astype(np.float32) for k, v in color_channels(norm).items()}
    k = 0
    for c in COLOR_NAMES:
        out[..., k] = chans[c]
        k += 1
    for c in FILTERED_CHANNELS:
        for s in SIGMAS:
            for f in _filters_one_scale(chans[c], s):
                out[..., k] = f
                k += 1
    assert k == len(names)
    return out


def compute_features_tiled(norm: np.ndarray, tile: int = 1024, margin: int = 48) -> np.ndarray:
    """Igual a ``compute_features``, em blocos com sobreposição (para resolução nativa).

    ``margin`` deve cobrir o suporte do maior filtro (≈ 4·σmax).
    """
    h, w = norm.shape[:2]
    out = np.empty((h, w, len(feature_names())), dtype=np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, xa = max(0, y0 - margin), max(0, x0 - margin)
            yb, xb = min(h, y0 + tile + margin), min(w, x0 + tile + margin)
            f = compute_features(norm[ya:yb, xa:xb])
            th, tw = min(tile, h - y0), min(tile, w - x0)
            out[y0:y0 + th, x0:x0 + tw] = f[y0 - ya:y0 - ya + th, x0 - xa:x0 - xa + tw]
    return out
