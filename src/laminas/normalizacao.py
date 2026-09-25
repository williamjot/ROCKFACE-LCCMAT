"""
normalizacao.py
===============
Normalização de cor por lâmina (Task 1.1, Seção 6.3): os mesmos percentis por canal
para todos os patches da lâmina, preservando a variação de cor entre patches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import Patch, load_rgb


def slide_color_stats(patches: list[Patch], lo: float = 0.5, hi: float = 99.5, step: int = 16) -> np.ndarray:
    """Percentis por canal sobre todos os patches. Retorna (2, 3): linha 0 = ``lo``, linha 1 = ``hi``."""
    samples = [load_rgb(p)[::step, ::step].reshape(-1, 3) for p in patches]
    return np.percentile(np.concatenate(samples), [lo, hi], axis=0).astype(np.float32)


def load_or_compute_stats(patches: list[Patch], cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    stats = slide_color_stats(patches)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, stats)
    return stats


def normalize(img: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """Estica cada canal linearmente entre os percentis da lâmina → float32 em [0, 1]."""
    lo, hi = stats
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def enhance(img: np.ndarray, stats: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Versão realçada para visualização/anotação (normalização + gama) → uint8."""
    return (normalize(img, stats) ** gamma * 255).astype(np.uint8)
