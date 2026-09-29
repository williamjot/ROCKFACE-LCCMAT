"""
01_gerar_kit_anotacao.py — prepara o material para os especialistas anotarem.

Divide os patches em:
  - TESTE (6): em cada um, uma janela de 1024 x 1024 px será corrigida à mão
    pelos especialistas (máscara completa). Esses patches não recebem rabiscos.
  - TREINO (12): recebem rabiscos de poro e sólido.

A janela de teste é a que tem mais borda de poro (os casos mais difíceis).

Saídas em annotations/:
    images/        patches realçados (mais claros) para anotar
    scribbles/     vazia — aqui entram os rabiscos
    test_masks/    pré-rótulo do rockface na janela de teste — CORRIGIR
    test_windows/  recorte de cada janela de teste + pré-rótulo, para revisar
    split.json     quais patches são treino/teste e onde estão as janelas

Uso:
    python 01_gerar_kit_anotacao.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # para o Python achar a pasta laminas/

import cv2
import numpy as np
from PIL import Image

import laminas as L

N_TESTE = 6          # quantos patches ficam para teste
LADO_JANELA = 1024   # tamanho da janela de teste (px)
SEMENTE = 42


def janela_com_mais_borda(mascara_poros):
    """Procura, de 256 em 256 px, a janela com mais pixels de borda de poro."""
    borda = cv2.morphologyEx(mascara_poros.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    melhor_total = -1
    melhor_y, melhor_x = 0, 0
    for y in range(0, L.PASSO_PATCH - LADO_JANELA + 1, 256):
        for x in range(0, L.PASSO_PATCH - LADO_JANELA + 1, 256):
            total = borda[y:y + LADO_JANELA, x:x + LADO_JANELA].sum()
            if total > melhor_total:
                melhor_total = total
                melhor_y, melhor_x = y, x
    return melhor_y, melhor_x


pasta = L.PASTA_ANOTACOES
arquivo_split = pasta / "split.json"
if arquivo_split.exists():
    split_antigo = json.load(open(arquivo_split, encoding="utf-8"))
    if split_antigo.get("test_masks_corrected"):
        sys.exit("split.json já tem máscaras corrigidas pelos especialistas; não vou sobrescrever.")

for sub in ["images", "scribbles", "test_masks", "test_windows"]:
    (pasta / sub).mkdir(parents=True, exist_ok=True)

arquivos = L.listar_patches()
nomes = [L.nome_do_patch(a) for a in arquivos]
limites = L.limites_de_cor(arquivos, L.PASTA_CACHE / "limites_de_cor.npy")

# Sorteia os patches de teste
gerador = np.random.default_rng(SEMENTE)
teste = sorted(gerador.choice(nomes, N_TESTE, replace=False).tolist())
treino = [n for n in nomes if n not in teste]
split = {"train": treino, "test": teste, "windows": {}}

for arquivo, nome in zip(arquivos, nomes):
    img = L.carregar_rgb(arquivo)
    realcada = L.realcar(img, limites)
    Image.fromarray(realcada).save(pasta / "images" / f"patch_{nome}_enhanced.png")

    if nome in teste:
        poros = L.mascara_rockface(img)
        y, x = janela_com_mais_borda(poros)
        fim_y, fim_x = y + LADO_JANELA, x + LADO_JANELA

        # Pré-rótulo: dentro da janela, 1 onde o rockface diz poro e 2 no resto; fora, 0
        rotulos = np.zeros(poros.shape, np.uint8)
        rotulos[y:fim_y, x:fim_x] = np.where(poros[y:fim_y, x:fim_x], L.PORO, L.SOLIDO)
        L.salvar_rotulos(rotulos, pasta / "test_masks" / f"patch_{nome}_mask.png")

        Image.fromarray(realcada[y:fim_y, x:fim_x]).save(pasta / "test_windows" / f"patch_{nome}_y{y}_x{x}_window.png")
        L.salvar_rotulos(rotulos[y:fim_y, x:fim_x], pasta / "test_windows" / f"patch_{nome}_y{y}_x{x}_prelabel.png")
        split["windows"][nome] = {"y": y, "x": x, "size": LADO_JANELA}
        print("ok", nome, "(teste)")
    else:
        print("ok", nome, "(treino)")

# Mude para true SÓ depois que os especialistas corrigirem todas as janelas de test_masks/
split["test_masks_corrected"] = False
json.dump(split, open(arquivo_split, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Kit gerado em", pasta)
