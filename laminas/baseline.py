"""
Baseline: o limiar de cor do pacote rockface (LCCMat-UnB/rockface, arquivo masks.py),
reproduzido passo a passo.
"""

import cv2
import numpy as np


def mascara_rockface(img_rgb):
    """Máscara de poros do rockface. Retorna True onde é poro.

    Passos (iguais aos do rockface):
      1. desfoque gaussiano 5x5;
      2. converte para HSV;
      3. poro = matiz entre 75 e 125 (azul/ciano na escala 0-180 do OpenCV)
         E saturação x brilho >= 0,1 (descarta pixels escuros ou acinzentados);
      4. abertura morfológica 3x3 para tirar pontinhos isolados.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   # o rockface trabalha em BGR
    desfocada = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(desfocada, cv2.COLOR_BGR2HSV)
    matiz = hsv[:, :, 0]
    saturacao = hsv[:, :, 1] / 255.0
    brilho = hsv[:, :, 2] / 255.0

    poro = (matiz >= 75) & (matiz <= 125) & (saturacao * brilho >= 0.1)

    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    poro = cv2.morphologyEx(poro.astype(np.uint8), cv2.MORPH_OPEN, elemento)
    return poro > 0


def area_de_rocha_rockface(img_rgb):
    """'Rocha' segundo o rockface (petrophysical_properties.py): algum canal > 5.

    Atenção: nestes patches escuros, 7-27 % dos pixels são grãos muito escuros
    (todos os canais <= 5) e ficam de fora, o que aumenta a porosidade calculada.
    """
    return np.any(img_rgb > 5, axis=2)
