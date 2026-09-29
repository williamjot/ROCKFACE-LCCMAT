"""
04_avaliar_especialista.py — compara os modelos com os rótulos dos ESPECIALISTAS.

Só roda depois que:
  1. os 12 patches de treino tiverem rabiscos em annotations/scribbles/;
  2. as 6 janelas de teste em annotations/test_masks/ tiverem sido corrigidas;
  3. annotations/split.json tiver "test_masks_corrected": true;
  4. o 02_extrair_amostras.py tiver sido rodado de novo (para ler os rabiscos).

Treina nos patches de treino e mede o acerto de cada método nas janelas corrigidas
(só nos pixels rotulados). Aqui as métricas medem ACERTO.

Saídas:
    results/task_1_2/especialista_escala<E>/metrics_per_patch.csv
    results/task_1_2/especialista_escala<E>/metrics_summary.csv
    results/task_1_2/especialista_escala<E>/rf_feature_importance.csv

Uso:
    python 04_avaliar_especialista.py --escala 0.5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # para o Python achar a pasta laminas/

import numpy as np
import pandas as pd

import laminas as L

parser = argparse.ArgumentParser()
parser.add_argument("--escala", type=float, default=0.5)
parser.add_argument("--semente", type=int, default=42)
args = parser.parse_args()

# 1. Confere se os rótulos dos especialistas estão prontos
split = json.load(open(L.PASTA_ANOTACOES / "split.json", encoding="utf-8"))
if not split.get("test_masks_corrected"):
    sys.exit('As janelas de teste ainda não foram corrigidas ("test_masks_corrected": false em split.json).')

pasta_amostras = L.PASTA_CACHE / "task_1_2" / f"amostras_escala{args.escala}"
pasta_resultados = L.PASTA_RESULTADOS / "task_1_2" / f"especialista_escala{args.escala}"
pasta_resultados.mkdir(parents=True, exist_ok=True)

arquivos = L.listar_patches()
limites = L.limites_de_cor(arquivos, L.PASTA_CACHE / "limites_de_cor.npy")

# 2. Junta as amostras dos patches de treino (só as que vieram de especialista)
lista_X, lista_y = [], []
for nome in split["train"]:
    dados = np.load(pasta_amostras / f"{nome}.npz")
    if str(dados["source"]) != "especialista":
        sys.exit(f"{nome}: amostras sem rabiscos de especialista. Rode o 02_extrair_amostras.py de novo.")
    lista_X.append(dados["X"])
    lista_y.append(dados["y"])
X = np.concatenate(lista_X)
y = np.concatenate(lista_y)
print(f"Treino: {len(y)} pixels rotulados ({y.sum()} poro)")

# 3. Treina os modelos
modelos = L.criar_modelos(args.semente)
for nome_modelo, (modelo, colunas) in modelos.items():
    if colunas is None:
        modelo.fit(X, y)
    else:
        modelo.fit(X[:, colunas], y)
    print(nome_modelo, "treinado")

# 4. Avalia na janela corrigida de cada patch de teste
linhas = []
for nome in split["test"]:
    arquivo = L.PASTA_DADOS / f"patch_{nome}_c0.png"
    img_original = L.carregar_rgb(arquivo)
    img = L.carregar_rgb(arquivo, args.escala)
    lado = img.shape[0]
    atributos = L.calcular_atributos(L.normalizar(img, limites))

    # Janela de teste na escala de trabalho
    janela = split["windows"][nome]
    y0 = round(janela["y"] * args.escala)
    x0 = round(janela["x"] * args.escala)
    y1 = round((janela["y"] + janela["size"]) * args.escala)
    x1 = round((janela["x"] + janela["size"]) * args.escala)

    referencia = L.ler_rotulos(L.PASTA_ANOTACOES / "test_masks" / f"patch_{nome}_mask.png")
    referencia = L.redimensionar_mascara(referencia, lado)[y0:y1, x0:x1]
    rotulado = referencia > 0                     # pixels que o especialista rotulou
    ref_poro = referencia == L.PORO

    base = L.redimensionar_mascara(L.mascara_rockface(img_original), lado) > 0
    previsoes = {"rockface": base[y0:y1, x0:x1]}
    for nome_modelo, (modelo, colunas) in modelos.items():
        prob = L.prever_probabilidade(modelo, atributos, colunas)
        previsoes[nome_modelo] = L.mascara_final(prob)[y0:y1, x0:x1]

    for nome_modelo, mascara in previsoes.items():
        linha = {"patch": nome, "method": nome_modelo, "reference": "especialista"}
        linha.update(L.metricas_de_segmentacao(mascara[rotulado], ref_poro[rotulado]))
        linha["boundary_f1"] = L.f1_de_borda(mascara, ref_poro)
        linha["porosity_pred"] = L.porosidade(mascara[rotulado])
        linha["porosity_ref"] = L.porosidade(ref_poro[rotulado])
        linhas.append(linha)
    print(nome, "ok")

# 5. Salva as tabelas
tabela = pd.DataFrame(linhas)
tabela.to_csv(pasta_resultados / "metrics_per_patch.csv", index=False)
colunas_resumo = ["iou", "dice", "precision", "recall", "boundary_f1", "porosity_pred", "porosity_ref"]
resumo = tabela.groupby("method")[colunas_resumo].agg(["mean", "std"])
resumo.to_csv(pasta_resultados / "metrics_summary.csv")
importancia = pd.Series(modelos["RF"][0].feature_importances_, index=L.nomes_dos_atributos())
importancia.sort_values(ascending=False).to_csv(pasta_resultados / "rf_feature_importance.csv",
                                                header=["gini_importance"])

print("\n=== Acerto nas janelas corrigidas (média entre patches de teste) ===")
print(resumo.round(3).to_string())
print("\nResultados em", pasta_resultados)
