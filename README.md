# ROCKFACE-LCCMAT — Segmentação de Lâminas Petrográficas (Equipe A)

Código, documentação e resultados da **Equipe A** do projeto *Segmentação de Lâminas
Petrográficas* do LCCMat-UnB (fluxo 1: segmentação de poros por IA; o método escolhido
será reaproveitado na segmentação de grãos, fluxo 2).

- Repositório do projeto: <https://github.com/LCCMat-UnB/segmentacao-laminas>
- Pacote `rockface` (patching e máscaras por limiar): <https://github.com/LCCMat-UnB/rockface>
- Equipe A: Larissa Marques Quirino, William de Souza Jota Filho

> 🗺️ **Roteiro do projeto** — etapas, anotação passo a passo (napari/GIMP) e teoria de ML:
> [`docs/ROTEIRO_DO_PROJETO.md`](docs/ROTEIRO_DO_PROJETO.md)
>
> 📘 **Guia de uso dos scripts:** [`docs/GUIA_DE_USO.md`](docs/GUIA_DE_USO.md)

## Andamento

**Task 1 — Prospecção de modelos (poros)**

| Task | Descrição | Estado | Pasta |
|---|---|---|---|
| 1.1 | Revisão de literatura de modelos de segmentação | ✅ concluída | [`tasks/task_1_poros/task_1_1_revisao_literatura`](tasks/task_1_poros/task_1_1_revisao_literatura) |
| 1.2 | Desempenho dos modelos candidatos | 🔄 piloto concluído; aguardando anotações | [`tasks/task_1_poros/task_1_2_desempenho_modelos`](tasks/task_1_poros/task_1_2_desempenho_modelos) |
| 1.3 | Escolha do(s) modelo(s) para aplicação em escala | ⏳ | [`tasks/task_1_poros/task_1_3_selecao_modelo`](tasks/task_1_poros/task_1_3_selecao_modelo) |
| 1.4 | Documentação das decisões e paper | ⏳ | [`tasks/task_1_poros/task_1_4_documentacao_paper`](tasks/task_1_poros/task_1_4_documentacao_paper) |

**Task 2 — Segmentação de grãos** ⏳ — [`tasks/task_2_graos`](tasks/task_2_graos)

**Atividades de formação** — [`atividades/`](atividades)

## Estrutura

```
ROCKFACE-LCCMAT/
├── README.md                     este arquivo
├── requirements.txt              bibliotecas necessárias
├── docs/
│   ├── ROTEIRO_DO_PROJETO.md     etapas, anotação, teoria de ML — leia primeiro
│   ├── GUIA_DE_USO.md            instalação, scripts, formatos
│   └── Task_1.1_Revisao_Literatura_Equipe_A.pdf
├── laminas/                      funções usadas por vários scripts
│   ├── config.py                 pastas e constantes
│   ├── imagens.py                ler patches, mudar tamanho, normalizar a cor
│   ├── baseline.py               limiar de cor do rockface (baseline)
│   ├── atributos.py              os 153 atributos por pixel
│   ├── rotulos.py                rabiscos, máscaras, pseudo-rótulos
│   ├── modelos.py                RF, LightGBM, RF-cor; previsão
│   └── metricas.py               IoU, Dice, F1 de borda, porosidade
├── tasks/
│   ├── task_1_poros/
│   │   ├── task_1_1_revisao_literatura/
│   │   ├── task_1_2_desempenho_modelos/   scripts 00_… a 05_…
│   │   ├── task_1_3_selecao_modelo/
│   │   └── task_1_4_documentacao_paper/
│   └── task_2_graos/
├── atividades/
│   └── atividade_1_pytorch_cat_not_cat/
├── data/            (git-ignored) patches das lâminas — ver data/README.md
├── annotations/     kit de anotação dos especialistas — ver annotations/README.md
├── results/         tabelas de resultados versionadas (CSV/JSON)
└── cache/           (git-ignored) intermediários: amostras, probabilidades, imagens
```

## Início rápido

```bash
git clone https://github.com/williamjot/ROCKFACE-LCCMAT.git
cd ROCKFACE-LCCMAT
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# copie os patch_y*_x*_c0.png para data/patches/  (não vêm no git)
cd tasks/task_1_poros/task_1_2_desempenho_modelos
python 00_explorar_dados.py
python 01_gerar_kit_anotacao.py
python 02_extrair_amostras.py --escala 0.5
python 03_validacao_cruzada_piloto.py --escala 0.5 --folds 6
python 05_figuras.py --escala 0.5
```

Detalhes, opções e formatos em [`docs/GUIA_DE_USO.md`](docs/GUIA_DE_USO.md).

## Dados

As imagens das lâminas **não são versionadas** (repositório público; dados do projeto).
Veja [`data/README.md`](data/README.md).
