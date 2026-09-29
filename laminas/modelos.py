"""
Modelos comparados na Task 1.2 e funções para prever e pós-processar.
"""

import cv2
import lightgbm
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .atributos import N_ATRIBUTOS_DE_COR


def criar_modelos(semente=42):
    """Os três classificadores da comparação.

    Devolve um dicionário: nome -> (modelo, colunas de atributos usadas).
    Colunas = None significa "todos os 153 atributos".

    Obs.: usamos class_weight="balanced". A opção "balanced_subsample" falhou de
    forma intermitente no scikit-learn 1.9 / Python 3.14.
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=semente)

    lgbm = lightgbm.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        class_weight="balanced", n_jobs=-1, random_state=semente, verbose=-1)

    # Mesmo RF, mas só com os 9 atributos de cor: serve para ver se os filtros ajudam
    rf_cor = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=semente)
    colunas_de_cor = list(range(N_ATRIBUTOS_DE_COR))

    return {
        "RF": (rf, None),
        "LGBM": (lgbm, None),
        "RF-cor": (rf_cor, colunas_de_cor),
    }


def prever_probabilidade(modelo, atributos, colunas=None):
    """Probabilidade de poro para cada pixel da imagem de atributos (altura, largura, 153).

    Processa 1 milhão de pixels por vez para não estourar a memória.
    """
    altura, largura, n = atributos.shape
    pixels = atributos.reshape(-1, n)          # uma linha por pixel
    if colunas is not None:
        pixels = pixels[:, colunas]

    prob = np.zeros(len(pixels), np.float32)
    bloco = 1_000_000
    for inicio in range(0, len(pixels), bloco):
        fim = inicio + bloco
        prob[inicio:fim] = modelo.predict_proba(pixels[inicio:fim])[:, 1]
    return prob.reshape(altura, largura)


def mascara_final(prob, limiar=0.5, raio_abertura=1):
    """Probabilidade -> máscara de poros: limiar e abertura morfológica (tira pontinhos)."""
    mascara = (prob >= limiar).astype(np.uint8)
    if raio_abertura > 0:
        lado = 2 * raio_abertura + 1
        elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento)
    return mascara > 0
