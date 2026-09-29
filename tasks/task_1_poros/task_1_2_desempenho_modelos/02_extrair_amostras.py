"""
02_extrair_amostras.py — monta os dados de treino (pixels rotulados e seus atributos).

Para cada patch:
  1. calcula os 153 atributos de cada pixel;
  2. pega os rótulos:
       - se existir annotations/scribbles/patch_<nome>_scribbles.png, usa os
         rabiscos do especialista;
       - senão, usa pseudo-rabiscos tirados do miolo das regiões do rockface
         (provisório, só para o piloto);
  3. guarda só os pixels rotulados: X (atributos) e y (1 = poro, 0 = sólido).

Saída: cache/task_1_2/amostras_escala<E>/<nome>.npz  (X, y, source)

É a etapa mais demorada (~30 s por patch na escala 0,5). As próximas etapas
reaproveitam esses arquivos.

Uso:
    python 02_extrair_amostras.py --escala 0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # para o Python achar a pasta laminas/

import numpy as np

import laminas as L

parser = argparse.ArgumentParser()
parser.add_argument("--escala", type=float, default=0.5, help="1.0 = resolução original; 0.5 = metade")
parser.add_argument("--semente", type=int, default=42)
args = parser.parse_args()

pasta_saida = L.PASTA_CACHE / "task_1_2" / f"amostras_escala{args.escala}"
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivos = L.listar_patches()
limites = L.limites_de_cor(arquivos, L.PASTA_CACHE / "limites_de_cor.npy")
gerador = np.random.default_rng(args.semente)

for arquivo in arquivos:
    nome = L.nome_do_patch(arquivo)

    # Imagem na escala de trabalho e seus atributos
    img = L.carregar_rgb(arquivo, args.escala)
    lado = img.shape[0]
    atributos = L.calcular_atributos(L.normalizar(img, limites))

    # Rótulos: rabiscos do especialista, se houver; senão, pseudo-rabiscos do rockface
    arquivo_rabiscos = L.PASTA_ANOTACOES / "scribbles" / f"patch_{nome}_scribbles.png"
    if arquivo_rabiscos.exists():
        rotulos = L.redimensionar_mascara(L.ler_rotulos(arquivo_rabiscos), lado)
        origem = "especialista"
    else:
        # o rockface é calculado na resolução original e depois reduzido
        mascara_base = L.mascara_rockface(L.carregar_rgb(arquivo))
        mascara_base = L.redimensionar_mascara(mascara_base, lado) > 0
        rotulos = L.pseudo_rabiscos(mascara_base, gerador)
        origem = "pseudo (rockface)"

    # Guarda só os pixels rotulados
    rotulados = rotulos > 0
    X = atributos[rotulados]                         # (n_pixels, 153)
    y = (rotulos[rotulados] == L.PORO).astype(np.uint8)
    np.savez(pasta_saida / f"{nome}.npz", X=X, y=y, source=origem)
    print(f"{nome}: {len(y)} pixels ({origem}), {y.sum()} poro / {len(y) - y.sum()} sólido")

print("Amostras salvas em", pasta_saida)
