"""
00_explorar_dados.py — primeira olhada nos patches.

Para cada patch calcula: brilho médio de cada canal, porosidade segundo o
rockface e a fração de pixels que o rockface NÃO considera rocha.

Saídas:
    results/task_1_2/exploracao_patches.csv
    cache/task_1_2/mosaico.jpg   (todos os patches lado a lado, realçados)

Uso:
    python 00_explorar_dados.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # para o Python achar a pasta laminas/

import numpy as np
import pandas as pd
from PIL import Image

import laminas as L

pasta_resultados = L.PASTA_RESULTADOS / "task_1_2"
pasta_cache = L.PASTA_CACHE / "task_1_2"
pasta_resultados.mkdir(parents=True, exist_ok=True)
pasta_cache.mkdir(parents=True, exist_ok=True)

arquivos = L.listar_patches()
limites = L.limites_de_cor(arquivos, L.PASTA_CACHE / "limites_de_cor.npy")

linhas = []
miniaturas = []
for arquivo in arquivos:
    img = L.carregar_rgb(arquivo)
    janela = img[:L.PASSO_PATCH, :L.PASSO_PATCH]      # área sem sobreposição
    poros = L.mascara_rockface(janela)
    rocha = L.area_de_rocha_rockface(janela)

    linhas.append({
        "patch": L.nome_do_patch(arquivo),
        "mean_R": janela[:, :, 0].mean(),
        "mean_G": janela[:, :, 1].mean(),
        "mean_B": janela[:, :, 2].mean(),
        "frac_nao_rocha_rockface": 1 - rocha.mean(),
        "porosidade_rockface_area_total": L.porosidade(poros),
        "porosidade_rockface_area_rocha": L.porosidade(poros, rocha),
    })
    miniaturas.append(Image.fromarray(L.realcar(img, limites)).resize((512, 512)))
    print("ok", L.nome_do_patch(arquivo))

tabela = pd.DataFrame(linhas).round(4)
tabela.to_csv(pasta_resultados / "exploracao_patches.csv", index=False)

# Mosaico: 6 miniaturas por linha
colunas = 6
n_linhas = (len(miniaturas) + colunas - 1) // colunas
mosaico = Image.new("RGB", (colunas * 512, n_linhas * 512))
for i, miniatura in enumerate(miniaturas):
    mosaico.paste(miniatura, ((i % colunas) * 512, (i // colunas) * 512))
mosaico.save(pasta_cache / "mosaico.jpg", quality=85)

print(tabela.to_string(index=False))
print(f"\nMédia de pixels fora da 'rocha' do rockface: {100 * tabela.frac_nao_rocha_rockface.mean():.1f} %")
