"""
Métricas de avaliação: comparação por pixel, qualidade das bordas e porosidade.
"""

import cv2
import numpy as np
from scipy import ndimage


def metricas_de_segmentacao(previsto, referencia):
    """Compara duas máscaras (True = poro) pixel a pixel.

    IoU e Dice medem a sobreposição; precisão = quanto do que foi previsto como
    poro é poro; revocação = quanto dos poros de referência foi encontrado.
    """
    vp = np.sum(previsto & referencia)      # verdadeiro positivo
    fp = np.sum(previsto & ~referencia)     # falso positivo
    fn = np.sum(~previsto & referencia)     # falso negativo
    vn = np.sum(~previsto & ~referencia)    # verdadeiro negativo
    eps = 1e-9                              # evita divisão por zero
    return {
        "iou": vp / (vp + fp + fn + eps),
        "dice": 2 * vp / (2 * vp + fp + fn + eps),
        "precision": vp / (vp + fp + eps),
        "recall": vp / (vp + fn + eps),
        "accuracy": (vp + vn) / (vp + vn + fp + fn + eps),
    }


def contorno(mascara):
    """Pixels da borda da máscara (a máscara menos ela mesma encolhida 1 px)."""
    m = mascara.astype(np.uint8)
    encolhida = cv2.erode(m, np.ones((3, 3), np.uint8))
    return (m - encolhida) > 0


def f1_de_borda(previsto, referencia, tolerancia=2):
    """F1 dos contornos: um pixel de borda conta como certo se houver borda da
    outra máscara a até `tolerancia` pixels dele."""
    borda_prev = contorno(previsto)
    borda_ref = contorno(referencia)
    if not borda_prev.any() or not borda_ref.any():
        return float(borda_prev.any() == borda_ref.any())

    # distância de cada pixel até a borda mais próxima
    dist_ate_ref = ndimage.distance_transform_edt(~borda_ref)
    dist_ate_prev = ndimage.distance_transform_edt(~borda_prev)

    precisao = np.mean(dist_ate_ref[borda_prev] <= tolerancia)
    revocacao = np.mean(dist_ate_prev[borda_ref] <= tolerancia)
    return 2 * precisao * revocacao / (precisao + revocacao + 1e-9)


def porosidade(mascara_poros, mascara_rocha=None):
    """Porosidade em %: pixels de poro / pixels de rocha.
    Sem mascara_rocha, usa a área inteira da imagem."""
    if mascara_rocha is None:
        return 100.0 * np.mean(mascara_poros)
    return 100.0 * np.sum(mascara_poros & mascara_rocha) / max(np.sum(mascara_rocha), 1)
