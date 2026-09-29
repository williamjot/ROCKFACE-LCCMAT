"""
03_validacao_cruzada_piloto.py — compara os modelos no PILOTO (sem rótulos de especialista).

Modelos: rockface (baseline), RF, LightGBM e RF só com cor.

Validação cruzada agrupada por patch: os 18 patches são divididos em 6 grupos
(folds). Em cada rodada, os modelos treinam nos patches de 5 grupos e preveem os
patches do grupo que ficou de fora. Assim nenhum patch é previsto por um modelo
que treinou nele.

ATENÇÃO: no piloto o treino usa pseudo-rótulos do rockface e a comparação é feita
CONTRA o rockface. As métricas medem concordância com o rockface, não acerto.

Precisa das amostras do 02_extrair_amostras.py (mesma --escala).

Saídas:
    results/task_1_2/piloto_escala<E>/metrics_per_patch.csv    uma linha por patch x método
    results/task_1_2/piloto_escala<E>/metrics_summary.csv      média e desvio padrão
    results/task_1_2/piloto_escala<E>/rf_feature_importance.csv
    cache/task_1_2/piloto_escala<E>/mascaras/<nome>.npz        máscaras (para o 05_figuras.py)

Uso:
    python 03_validacao_cruzada_piloto.py --escala 0.5 --folds 6
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # para o Python achar a pasta laminas/

import numpy as np
import pandas as pd

import laminas as L

parser = argparse.ArgumentParser()
parser.add_argument("--escala", type=float, default=0.5)
parser.add_argument("--folds", type=int, default=6)
parser.add_argument("--semente", type=int, default=42)
parser.add_argument("--limite", type=int, default=0, help="usar só os N primeiros patches (teste rápido)")
args = parser.parse_args()

inicio = time.time()
pasta_amostras = L.PASTA_CACHE / "task_1_2" / f"amostras_escala{args.escala}"
pasta_resultados = L.PASTA_RESULTADOS / "task_1_2" / f"piloto_escala{args.escala}"
pasta_mascaras = L.PASTA_CACHE / "task_1_2" / f"piloto_escala{args.escala}" / "mascaras"
if args.limite:
    pasta_resultados = pasta_resultados.with_name(pasta_resultados.name + f"_limite{args.limite}")
    pasta_mascaras = L.PASTA_CACHE / "task_1_2" / f"piloto_escala{args.escala}_limite{args.limite}" / "mascaras"
pasta_resultados.mkdir(parents=True, exist_ok=True)
pasta_mascaras.mkdir(parents=True, exist_ok=True)

arquivos = L.listar_patches()
limites = L.limites_de_cor(arquivos, L.PASTA_CACHE / "limites_de_cor.npy")
if args.limite:
    arquivos = arquivos[:args.limite]
nomes = [L.nome_do_patch(a) for a in arquivos]

# 1. Carrega as amostras de treino de cada patch (feitas pelo 02_extrair_amostras.py)
amostras = {}
for nome in nomes:
    dados = np.load(pasta_amostras / f"{nome}.npz")
    amostras[nome] = (dados["X"], dados["y"])

# 2. Sorteia o fold de cada patch
gerador = np.random.default_rng(args.semente)
ordem = gerador.permutation(len(nomes))
fold_do_patch = {}
for posicao, indice in enumerate(ordem):
    fold_do_patch[nomes[indice]] = posicao % args.folds

# Janela sem sobreposição, na escala de trabalho
lado_janela = round(L.PASSO_PATCH * args.escala)

linhas = []
importancias = []
for fold in range(args.folds):
    nomes_teste = [n for n in nomes if fold_do_patch[n] == fold]
    nomes_treino = [n for n in nomes if fold_do_patch[n] != fold]

    # 3. Treina os modelos com os patches de treino
    X = np.concatenate([amostras[n][0] for n in nomes_treino])
    y = np.concatenate([amostras[n][1] for n in nomes_treino])
    print(f"[fold {fold}] treino: {len(y)} pixels ({y.sum()} poro); teste: {nomes_teste}")

    modelos = L.criar_modelos(args.semente)
    for nome_modelo, (modelo, colunas) in modelos.items():
        if colunas is None:
            modelo.fit(X, y)
        else:
            modelo.fit(X[:, colunas], y)
    importancias.append(modelos["RF"][0].feature_importances_)

    # 4. Prevê cada patch de teste e compara com o rockface
    for nome in nomes_teste:
        arquivo = arquivos[nomes.index(nome)]
        img_original = L.carregar_rgb(arquivo)
        img = L.carregar_rgb(arquivo, args.escala)
        lado = img.shape[0]
        atributos = L.calcular_atributos(L.normalizar(img, limites))

        # rockface e "rocha" calculados na resolução original, depois reduzidos
        base = L.redimensionar_mascara(L.mascara_rockface(img_original), lado) > 0
        rocha = L.redimensionar_mascara(L.area_de_rocha_rockface(img_original), lado) > 0
        base = base[:lado_janela, :lado_janela]
        rocha = rocha[:lado_janela, :lado_janela]

        previsoes = {"rockface": base}
        for nome_modelo, (modelo, colunas) in modelos.items():
            prob = L.prever_probabilidade(modelo, atributos, colunas)
            previsoes[nome_modelo] = L.mascara_final(prob[:lado_janela, :lado_janela])

        for nome_modelo, mascara in previsoes.items():
            linha = {"patch": nome, "fold": fold, "method": nome_modelo, "reference": "rockface"}
            linha.update(L.metricas_de_segmentacao(mascara, base))
            linha["boundary_f1"] = L.f1_de_borda(mascara, base)
            linha["porosity_total"] = L.porosidade(mascara)
            linha["porosity_rockface_rock"] = L.porosidade(mascara, rocha)
            linhas.append(linha)

        np.savez_compressed(pasta_mascaras / f"{nome}.npz", rockface=base, RF=previsoes["RF"])
        print(f"   {nome} ok ({(time.time() - inicio) / 60:.1f} min)")

# 5. Salva as tabelas
tabela = pd.DataFrame(linhas)
tabela.to_csv(pasta_resultados / "metrics_per_patch.csv", index=False)

colunas_resumo = ["iou", "dice", "precision", "recall", "boundary_f1", "porosity_total", "porosity_rockface_rock"]
resumo = tabela.groupby("method")[colunas_resumo].agg(["mean", "std"])
resumo.to_csv(pasta_resultados / "metrics_summary.csv")

importancia = pd.Series(np.mean(importancias, axis=0), index=L.nomes_dos_atributos())
importancia = importancia.sort_values(ascending=False)
importancia.to_csv(pasta_resultados / "rf_feature_importance.csv", header=["gini_importance"])

info = {"modo": "piloto", "escala": args.escala, "folds": args.folds, "semente": args.semente,
        "fold_do_patch": fold_do_patch, "minutos": round((time.time() - inicio) / 60, 1)}
json.dump(info, open(pasta_resultados / "run_info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("\n=== Concordância com o rockface (média entre patches) ===")
print(resumo.round(3).to_string())
print("\n=== 10 atributos mais importantes do RF ===")
print(importancia.head(10).round(4).to_string())
print("\nResultados em", pasta_resultados)
