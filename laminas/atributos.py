"""
Banco de atributos por pixel (Tabela 3 da Task 1.1), no estilo do ilastik.

Cada pixel é descrito por 153 números:
  - 9 de cor: R, G, B, H, S, V, L*, a*, b*;
  - 144 de filtros: 8 filtros x 6 escalas x 3 canais (L*, b*, S).

Os 8 filtros, para cada escala sigma:
  gauss    média da vizinhança (suavização gaussiana)
  gradmag  intensidade da borda (magnitude do gradiente)
  dog      diferença de gaussianas (realça manchas do tamanho de sigma)
  log      laplaciano da gaussiana (realça pontos e poros pequenos)
  hess1/2  autovalores da Hessiana (formas alongadas, gargantas)
  st1/2    autovalores do tensor de estrutura (textura)
"""

import numpy as np
from scipy import ndimage
from skimage.color import rgb2hsv, rgb2lab
from skimage.feature import (hessian_matrix, hessian_matrix_eigvals,
                             structure_tensor, structure_tensor_eigenvalues)

from .config import CANAIS_DE_COR, CANAIS_FILTRADOS, ESCALAS, FILTROS

N_ATRIBUTOS_DE_COR = len(CANAIS_DE_COR)   # os 9 primeiros atributos são os de cor


def nomes_dos_atributos():
    """Nomes dos 153 atributos, na mesma ordem das colunas: 'color_R', ..., 'b_gauss_s3.5', ..."""
    nomes = []
    for canal in CANAIS_DE_COR:
        nomes.append("color_" + canal)
    for canal in CANAIS_FILTRADOS:
        for sigma in ESCALAS:
            for filtro in FILTROS:
                nomes.append(f"{canal}_{filtro}_s{sigma}")
    return nomes


def canais_de_cor(img_normalizada):
    """Separa a imagem (valores entre 0 e 1) nos 9 canais: RGB, HSV e L*a*b*."""
    hsv = rgb2hsv(img_normalizada)
    lab = rgb2lab(img_normalizada)
    canais = {
        "R": img_normalizada[:, :, 0],
        "G": img_normalizada[:, :, 1],
        "B": img_normalizada[:, :, 2],
        "H": hsv[:, :, 0],
        "S": hsv[:, :, 1],
        "V": hsv[:, :, 2],
        "L": lab[:, :, 0] / 100.0,   # L* vai de 0 a 100; dividimos para ficar perto de [0, 1]
        "a": lab[:, :, 1] / 128.0,
        "b": lab[:, :, 2] / 128.0,   # b* alto = amarelo, b* baixo (negativo) = azul
    }
    for nome in canais:
        canais[nome] = canais[nome].astype(np.float32)
    return canais


def filtros_em_uma_escala(canal, sigma):
    """Aplica os 8 filtros a um canal, numa escala sigma. Retorna uma lista de 8 imagens."""
    gauss = ndimage.gaussian_filter(canal, sigma)
    gradmag = ndimage.gaussian_gradient_magnitude(canal, sigma)
    dog = gauss - ndimage.gaussian_filter(canal, 1.6 * sigma)
    log = ndimage.gaussian_laplace(canal, sigma)

    hessiana = hessian_matrix(canal, sigma=sigma, order="rc", use_gaussian_derivatives=False)
    hess1, hess2 = hessian_matrix_eigvals(hessiana)

    tensor = structure_tensor(canal, sigma=sigma, order="rc")
    st1, st2 = structure_tensor_eigenvalues(tensor)

    return [gauss, gradmag, dog, log, hess1, hess2, st1, st2]


def calcular_atributos(img_normalizada):
    """Calcula os 153 atributos de cada pixel. Retorna um array (altura, largura, 153).

    Com escala 0,5 (imagem de 2048 x 2048) o resultado ocupa ~2,6 GB de memória.
    """
    canais = canais_de_cor(img_normalizada)
    lista = []
    for nome in CANAIS_DE_COR:
        lista.append(canais[nome])
    for nome in CANAIS_FILTRADOS:
        for sigma in ESCALAS:
            lista.extend(filtros_em_uma_escala(canais[nome], sigma))
    return np.stack(lista, axis=2).astype(np.float32)
