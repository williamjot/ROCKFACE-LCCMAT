"""
05_figuras.py — imagens de comparação entre o RF e o rockface (do piloto).

Cores:
    amarelo  = poro para os dois
    magenta  = poro só para o RF
    ciano    = poro só para o rockface

Precisa das máscaras salvas pelo 03_validacao_cruzada_piloto.py (mesma --escala).
Saída: cache/task_1_2/piloto_escala<E>/figuras/<nome>_RF_vs_rockface.jpg

Uso:
    python 05_figuras.py --escala 0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # para o Python achar a pasta laminas/

import numpy as np
from PIL import Image

import laminas as L

AMARELO = np.array([255, 220, 0])
MAGENTA = np.array([255, 0, 255])
CIANO = np.array([0, 255, 255])

parser = argparse.ArgumentParser()
parser.add_argument("--escala", type=float, default=0.5)
args = parser.parse_args()

pasta = L.PASTA_CACHE / "task_1_2" / f"piloto_escala{args.escala}"
pasta_figuras = pasta / "figuras"
pasta_figuras.mkdir(parents=True, exist_ok=True)

arquivos = L.listar_patches()
limites = L.limites_de_cor(arquivos, L.PASTA_CACHE / "limites_de_cor.npy")

for arquivo in arquivos:
    nome = L.nome_do_patch(arquivo)
    arquivo_mascaras = pasta / "mascaras" / f"{nome}.npz"
    if not arquivo_mascaras.exists():
        continue
    mascaras = np.load(arquivo_mascaras)
    rf = mascaras["RF"]
    rockface = mascaras["rockface"]

    lado = rf.shape[0]
    img = L.realcar(L.carregar_rgb(arquivo, args.escala), limites)[:lado, :lado].astype(float)

    # Pinta por cima da imagem, misturando 55 % da cor
    for regiao, cor in [(rf & rockface, AMARELO), (rf & ~rockface, MAGENTA), (~rf & rockface, CIANO)]:
        img[regiao] = 0.45 * img[regiao] + 0.55 * cor

    Image.fromarray(img.astype(np.uint8)).save(pasta_figuras / f"{nome}_RF_vs_rockface.jpg", quality=90)
    print("ok", nome)

print("Figuras em", pasta_figuras)
