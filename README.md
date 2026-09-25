# ROCKFACE-LCCMAT — Segmentação de Lâminas Petrográficas (Equipe A)

Código, documentação e resultados da **Equipe A** do projeto *Segmentação de Lâminas
Petrográficas* do LCCMat-UnB (fluxo 1: segmentação de poros por IA; o método escolhido
será reaproveitado na segmentação de grãos, fluxo 2).

- Repositório do projeto: <https://github.com/LCCMat-UnB/segmentacao-laminas>
- Pacote `rockface` (patching e máscaras por limiar): <https://github.com/LCCMat-UnB/rockface>
- Equipe A: Larissa Marques Quirino, William de Souza Jota Filho

> 📘 **Guia completo de uso:** [`docs/GUIA_DE_USO.md`](docs/GUIA_DE_USO.md)

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
├── pyproject.toml                pacote `laminas` + dependências
├── docs/
│   ├── GUIA_DE_USO.md            instalação, scripts, formatos, fluxos — leia primeiro
│   └── Task_1.1_Revisao_Literatura_Equipe_A.pdf
├── src/laminas/                  biblioteca compartilhada por todas as tasks
│   ├── config.py                 caminhos (variáveis de ambiente) e constantes
│   ├── io.py                     leitura de patches e reamostragem
│   ├── normalizacao.py           normalização de cor por lâmina
│   ├── baseline.py               limiar HSV do rockface (baseline)
│   ├── atributos.py              banco de 153 atributos por pixel
│   ├── rotulos.py                rabiscos, máscaras, pseudo-rótulos
│   ├── metricas.py               IoU, Dice, F1 de borda, porosidade
│   └── posprocessamento.py       probabilidade → máscara; sobreposições
├── tasks/
│   ├── task_1_poros/
│   │   ├── task_1_1_revisao_literatura/
│   │   ├── task_1_2_desempenho_modelos/   scripts 00_…, 01_…, 02_…
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
pip install -e .
# copie os patch_y*_x*_c0.png para data/patches/  (não vêm no git)
cd tasks/task_1_poros/task_1_2_desempenho_modelos
python 00_explorar_dados.py
python 01_gerar_kit_anotacao.py
python 02_comparar_modelos.py --scale 0.5 --folds 6
```

Detalhes, opções e formatos em [`docs/GUIA_DE_USO.md`](docs/GUIA_DE_USO.md).

## Dados

As imagens das lâminas **não são versionadas** (repositório público; dados do projeto).
Veja [`data/README.md`](data/README.md).
