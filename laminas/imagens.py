"""
Leitura dos patches, mudança de tamanho e normalização de cor.
"""

import cv2
import numpy as np
from PIL import Image

from .config import PASTA_DADOS

Image.MAX_IMAGE_PIXELS = None  # os patches são grandes (4096 x 4096); tira o limite do Pillow


def listar_patches(pasta=PASTA_DADOS):
    """Lista os arquivos patch_y*_x*_c0.png da pasta, em ordem alfabética."""
    arquivos = sorted(pasta.glob("patch_y*_x*_c0.png"))
    if len(arquivos) == 0:
        raise FileNotFoundError(f"Nenhum patch encontrado em {pasta} (ver data/README.md)")
    return arquivos


def nome_do_patch(arquivo):
    """'patch_y22800_x34200_c0.png' -> 'y22800_x34200'."""
    return arquivo.stem.replace("patch_", "").replace("_c0", "")


def carregar_rgb(arquivo, escala=1.0):
    """Lê o patch como RGB (uint8). Com escala < 1, a imagem é reduzida."""
    img = np.array(Image.open(arquivo).convert("RGB"))
    if escala != 1.0:
        lado = round(img.shape[0] * escala)
        img = cv2.resize(img, (lado, lado), interpolation=cv2.INTER_AREA)
    return img


def redimensionar_mascara(mascara, tamanho):
    """Muda o tamanho de uma máscara/rótulo sem misturar valores (vizinho mais próximo)."""
    if mascara.shape[0] == tamanho:
        return mascara
    return cv2.resize(mascara.astype(np.uint8), (tamanho, tamanho), interpolation=cv2.INTER_NEAREST)


# ----------------------------------------------------------------------------
# Normalização de cor
# ----------------------------------------------------------------------------
# As imagens são muito escuras (média ~20 de 255). Esticamos cada canal de cor
# entre o percentil 0,5 % e o 99,5 %, usando os MESMOS limites para todos os
# patches da lâmina (assim a diferença de cor entre patches é preservada).

def calcular_limites_de_cor(arquivos):
    """Percentis 0,5 % e 99,5 % de cada canal, sobre todos os patches (amostrando 1 a cada 16 px)."""
    amostras = []
    for arquivo in arquivos:
        img = carregar_rgb(arquivo)
        amostras.append(img[::16, ::16].reshape(-1, 3))
    todos = np.concatenate(amostras)
    return np.percentile(todos, [0.5, 99.5], axis=0).astype(np.float32)


def limites_de_cor(arquivos, arquivo_cache):
    """Lê os limites do cache; se não existir, calcula e salva."""
    if arquivo_cache.exists():
        return np.load(arquivo_cache)
    limites = calcular_limites_de_cor(arquivos)
    arquivo_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(arquivo_cache, limites)
    return limites


def normalizar(img, limites):
    """Leva cada canal para o intervalo [0, 1] usando os limites da lâmina."""
    minimo = limites[0]
    maximo = limites[1]
    return np.clip((img.astype(np.float32) - minimo) / (maximo - minimo), 0, 1)


def realcar(img, limites):
    """Versão clara da imagem para olhar/anotar: normaliza e clareia (gama 0,6)."""
    return (normalizar(img, limites) ** 0.6 * 255).astype(np.uint8)
