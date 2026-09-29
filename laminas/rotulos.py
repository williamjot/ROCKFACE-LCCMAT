"""
Rótulos: ler e salvar rabiscos/máscaras, e gerar os pseudo-rabiscos do piloto.

Valores dos rótulos: 0 = sem rótulo, 1 = poro, 2 = sólido.
"""

import cv2
import numpy as np
from PIL import Image

from .config import PORO, SOLIDO

# Paleta para salvar os rótulos como PNG colorido: 0 = preto, 1 = vermelho, 2 = verde
PALETA = [0, 0, 0, 255, 0, 0, 0, 200, 0]


def ler_rotulos(arquivo):
    """Lê um PNG de rótulos e devolve um array com 0, 1 e 2.

    Aceita dois formatos:
      - PNG com os números 0/1/2 (como o napari e o 01_gerar_kit_anotacao.py salvam);
      - PNG colorido (GIMP): vermelho = poro, verde = sólido, resto = sem rótulo.
    """
    img = np.array(Image.open(arquivo))
    if img.ndim == 2:                  # já está em números 0/1/2
        return img.astype(np.uint8)

    vermelho = img[:, :, 0].astype(int)
    verde = img[:, :, 1].astype(int)
    rotulos = np.zeros(img.shape[:2], np.uint8)
    rotulos[(vermelho > 128) & (verde < 100)] = PORO
    rotulos[(verde > 128) & (vermelho < 100)] = SOLIDO
    if img.shape[2] == 4:              # com canal de transparência: transparente = sem rótulo
        rotulos[img[:, :, 3] == 0] = 0
    return rotulos


def salvar_rotulos(rotulos, arquivo):
    """Salva os rótulos 0/1/2 como PNG com paleta (poro aparece vermelho, sólido verde)."""
    img = Image.fromarray(rotulos.astype(np.uint8), mode="P")
    img.putpalette(PALETA)
    img.save(arquivo, transparency=0)


def pseudo_rabiscos(mascara_base, gerador, erosao=6, pixels_por_classe=20000):
    """Rótulos PROVISÓRIOS para o piloto, enquanto não há rabiscos de especialistas.

    Pega pixels só do "miolo" dos poros e dos sólidos da máscara do rockface
    (encolhendo as regiões em `erosao` pixels), para fugir das bordas, onde o
    limiar erra mais. Sorteia `pixels_por_classe` pixels de cada classe.

    Cuidado: o modelo treinado assim aprende a imitar o rockface.
    """
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosao + 1, 2 * erosao + 1))
    miolo_poro = cv2.erode(mascara_base.astype(np.uint8), elemento) > 0
    miolo_solido = cv2.erode((~mascara_base).astype(np.uint8), elemento) > 0

    rotulos = np.zeros(mascara_base.shape, np.uint8)
    for classe, regiao in [(PORO, miolo_poro), (SOLIDO, miolo_solido)]:
        posicoes = np.flatnonzero(regiao)          # índices dos pixels da região
        if len(posicoes) > 0:
            n = min(pixels_por_classe, len(posicoes))
            sorteados = gerador.choice(posicoes, size=n, replace=False)
            rotulos.flat[sorteados] = classe
    return rotulos
