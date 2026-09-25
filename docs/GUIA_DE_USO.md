# Guia de uso — ROCKFACE-LCCMAT

Guia completo para instalar, rodar e estender o código da Equipe A. Tudo aqui foi
testado em Windows 11 + Python 3.14 (Git Bash); em Linux/macOS só muda a ativação
do ambiente.

**Sumário**

1. [Instalação](#1-instalação)
2. [Dados de entrada](#2-dados-de-entrada)
3. [Configuração de caminhos](#3-configuração-de-caminhos)
4. [Biblioteca `laminas`](#4-biblioteca-laminas)
5. [Task 1.2 — scripts](#5-task-12--scripts)
6. [Anotação pelos especialistas](#6-anotação-pelos-especialistas)
7. [Modos piloto × especialista](#7-modos-piloto--especialista)
8. [Resultados e como ler as métricas](#8-resultados-e-como-ler-as-métricas)
9. [Desempenho e memória](#9-desempenho-e-memória)
10. [Problemas conhecidos](#10-problemas-conhecidos)
11. [Como contribuir](#11-como-contribuir)

---

## 1. Instalação

```bash
git clone https://github.com/williamjot/ROCKFACE-LCCMAT.git
cd ROCKFACE-LCCMAT
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -e .
```

`pip install -e .` instala a biblioteca `laminas` (pasta `src/laminas`) em modo
editável — alterações no código valem na hora, e qualquer script em qualquer pasta
pode fazer `import laminas`. Dependências (em `pyproject.toml`): numpy, scipy, pandas,
pillow, matplotlib, opencv-python-headless, scikit-image, scikit-learn, lightgbm.

Verificação:

```bash
python -c "import laminas as L; print(L.__version__, len(L.feature_names()), 'atributos')"
```

Sem ativar o ambiente, use o Python dele diretamente: `.venv/Scripts/python script.py`.

## 2. Dados de entrada

Patches exportados pelo `rockface` (`Patching.run`), canal 0 (luz normal/PPL):

- nome `patch_y{Y}_x{X}_c0.png`, onde `Y`, `X` são as coordenadas do canto do patch
  na lâmina (px);
- RGB 8 bits, 4096 × 4096 px, passo de patching 3800 px (296 px de sobreposição);
- resina epóxi azul nos poros.

Coloque-os em `data/patches/` (git-ignored). Os 18 patches atuais são de uma única
lâmina. As imagens são muito escuras (média ≈ 20/255 por canal), por isso todo o
pipeline normaliza a cor antes de calcular atributos.

## 3. Configuração de caminhos

Padrões relativos à raiz do repositório, trocáveis por variáveis de ambiente
(`src/laminas/config.py`):

| Variável | Padrão | Conteúdo |
|---|---|---|
| `LAMINAS_DATA` | `data/patches` | patches de entrada |
| `LAMINAS_ANNOTATIONS` | `annotations` | kit e rótulos dos especialistas |
| `LAMINAS_RESULTS` | `results` | tabelas de resultados (versionadas) |
| `LAMINAS_CACHE` | `cache` | intermediários pesados (git-ignored) |

Exemplo (outra lâmina em outro disco):

```bash
LAMINAS_DATA=D:/laminas/lamina_02 python 02_comparar_modelos.py
```

## 4. Biblioteca `laminas`

| Módulo | Principais funções | Para quê |
|---|---|---|
| `config` | `DATA_DIR`, `STRIDE`, `SIGMAS`, `PORE`, `SOLID` | caminhos e constantes |
| `io` | `list_patches`, `load_rgb(patch, scale)`, `resize_mask` | leitura e reamostragem |
| `normalizacao` | `load_or_compute_stats`, `normalize`, `enhance` | cor por lâmina (percentis 0,5–99,5 % de todos os patches) |
| `baseline` | `rockface_mask`, `rockface_rock_area` | reprodução fiel do limiar HSV do rockface |
| `atributos` | `compute_features`, `compute_features_tiled`, `feature_names` | 153 atributos por pixel |
| `rotulos` | `load_labels`, `save_labels`, `bootstrap_scribbles` | rótulos 0/1/2 e pseudo-rabiscos |
| `metricas` | `seg_metrics`, `boundary_f1`, `porosity` | avaliação |
| `posprocessamento` | `postprocess`, `disagreement_overlay` | máscara final e figuras |

Exemplo mínimo — segmentar um patch com o baseline e com atributos:

```python
import laminas as L

patches = L.list_patches()
stats = L.load_or_compute_stats(patches, L.CACHE_DIR / "slide_color_stats.npy")
rgb = L.load_rgb(patches[0], scale=0.5)
poros_rockface = L.rockface_mask(rgb)
feats = L.compute_features(L.normalize(rgb, stats))   # (H, W, 153) float32
print(L.porosity(poros_rockface), feats.shape)
```

### Banco de atributos (Tabela 3 da Task 1.1)

- **Cor (9):** R, G, B, H, S, V, L\*, a\*, b\* do pixel normalizado.
- **Filtros (144):** para cada canal L\*, b\*, S e cada σ ∈ {0,7; 1; 1,6; 3,5; 5; 10} px:
  gaussiana, magnitude do gradiente, DoG (σ vs 1,6σ), LoG, 2 autovalores da Hessiana,
  2 autovalores do tensor de estrutura.
- Nomes no formato `b_gauss_s3.5`, `S_st2_s5.0`, `color_b`, … (`L.feature_names()`).
- σ é em pixels **da imagem de trabalho**: com `--scale 0.5`, σ = 10 cobre 20 px nativos.

### Baseline `rockface`

`baseline.rockface_mask` reproduz `rockface/masks.py::generate_pore_mask_array`:
blur gaussiano 5×5 → HSV → matiz 75–125 (escala OpenCV 0–180) → S·V ≥ 0,1 → abertura
com elipse 3×3. A "área de rocha" do rockface (`legacy/petrophysical_properties.py`)
é "algum canal > 5" (`rockface_rock_area`) — ver [problemas conhecidos](#10-problemas-conhecidos).

## 5. Task 1.2 — scripts

Pasta `tasks/task_1_poros/task_1_2_desempenho_modelos/`. Rode de dentro dela, na ordem:

### `00_explorar_dados.py`

```bash
python 00_explorar_dados.py
```

Por patch: brilho médio por canal, porosidade do rockface (sobre a área toda e sobre
a "rocha" do rockface) e fração de pixels que o rockface não conta como rocha.
Saídas: `results/task_1_2/exploracao_patches.csv`, `cache/task_1_2/mosaico.jpg`.

### `01_gerar_kit_anotacao.py`

```bash
python 01_gerar_kit_anotacao.py [--n-test 6] [--win 1024] [--seed 42]
```

Sorteia `--n-test` patches de **teste** e escolhe, em cada um, a janela `--win`² com
mais borda de poro (casos difíceis); os demais são de **treino**. Gera o kit em
`annotations/` (seção 6). Recusa-se a sobrescrever se `split.json` já estiver marcado
como corrigido.

### `02_comparar_modelos.py`

```bash
python 02_comparar_modelos.py --scale 0.5 --folds 6          # piloto completo (~25 min)
python 02_comparar_modelos.py --scale 0.25 --folds 3 --limit 3   # teste rápido (~2 min)
```

| Opção | Padrão | Efeito |
|---|---|---|
| `--scale` | 0.5 | reamostragem (1.0 = nativo, 4× mais lento que 0.5) |
| `--folds` | 6 | folds da validação cruzada agrupada por patch (modo piloto) |
| `--seed` | 42 | sorteio dos folds e dos pseudo-rabiscos |
| `--limit` | 0 | usa só os N primeiros patches |

Modelos comparados:

| Nome | Descrição |
|---|---|
| `rockface` | limiar HSV (baseline, prioridade 0) |
| `RF` | Random Forest, 200 árvores, `max_features="sqrt"`, `min_samples_leaf=5`, 153 atributos |
| `LGBM` | LightGBM, 400 árvores, `num_leaves=63`, lr 0,05, mesmos atributos |
| `RF-cor` | RF só com os 9 atributos de cor (ablação do banco de filtros) |

Pós-processamento dos modelos: τ = 0,5 e abertura com raio 1 px. Avaliação só na
janela sem sobreposição `[0:3800, 0:3800]` (escalada), como no rockface.

## 6. Anotação pelos especialistas

| Pasta em `annotations/` | Conteúdo | Ação |
|---|---|---|
| `images/` | patch realçado (normalização + gama 0,6), resolução nativa | base para anotar |
| `scribbles/` | vazia | salvar `patch_<nome>_scribbles.png` dos 12 patches de **treino** |
| `test_masks/` | pré-rótulo do rockface numa janela 1024² de cada patch de **teste** | **corrigir a janela inteira** |
| `test_windows/` | recorte realçado + pré-rótulo de cada janela | conveniência para revisar |
| `split.json` | treino/teste, janelas, `test_masks_corrected` | marcar `true` ao terminar |

**Formato dos rótulos:** PNG do mesmo tamanho do patch (4096²), indexado com
0 = sem rótulo, 1 = poro, 2 = sólido; ou RGB(A) pintado em **vermelho** (poro) e
**verde** (sólido), com o resto preto/transparente.

**Ferramentas:** napari (camada *Labels*, salvar como PNG), GIMP (camada nova sobre
a imagem realçada, pincel sem suavização), ou ilastik (exportar *Labels*).

**Regras:**
- Rabiscos cobrem de propósito bordas, poros pequenos, bolhas e variação de tingimento;
  não só o "miolo" fácil. Pixels ambíguos ficam sem rótulo.
- Nenhum patch de teste recebe rabisco.
- A janela de teste é densa: todo pixel dela tem de estar revisado (1 ou 2).
- Decisões pendentes (definir antes de anotar): microporosidade/franja ciano-clara
  nas bordas dos poros é poro, sólido ou terceira classe? Área mínima de poro `A_min`?

Os rótulos dos especialistas **não** sobem para o git por padrão (ver
`.gitignore`); combinem com o grupo onde guardá-los.

## 7. Modos piloto × especialista

O `02_comparar_modelos.py` escolhe o modo sozinho:

| | Piloto | Especialista |
|---|---|---|
| Quando | `split.json` ausente ou `test_masks_corrected: false` | `test_masks_corrected: true` |
| Treino | pseudo-rabiscos: 20 mil px/classe/patch do miolo (erosão 6 px) das regiões do rockface | só `annotations/scribbles/` dos patches de treino |
| Teste | validação cruzada agrupada por patch | janelas densas corrigidas dos patches de teste |
| Referência | o próprio rockface | máscara corrigida (só pixels rotulados) |
| As métricas medem | **concordância com o rockface** | acerto |

No modo piloto o RF herda o viés do limiar; ele serve para validar o pipeline, medir
custo, ver a importância dos atributos e localizar onde os métodos discordam — não
para escolher o modelo.

## 8. Resultados e como ler as métricas

Pasta `results/task_1_2/<modo>_scale<s>/`:

| Arquivo | Conteúdo |
|---|---|
| `metrics_per_patch.csv` | uma linha por patch de teste × método |
| `metrics_summary.csv` | média e desvio padrão entre patches |
| `rf_feature_importance.csv` | importância de Gini dos 153 atributos (média dos folds) |
| `run_info.json` | parâmetros, folds, origem dos rótulos, fração de "rocha" do rockface |

Em `cache/task_1_2/<modo>_scale<s>/`: `overlays/*.jpg` (amarelo = ambos poro,
magenta = só o RF, ciano = só o rockface), `prob/*.npy` (probabilidade do RF),
`samples/*.npz` (amostras de treino reaproveitadas entre execuções).

| Métrica | Significado |
|---|---|
| `iou`, `dice` | sobreposição por pixel com a referência |
| `precision`, `recall` | poro previsto que é poro / poro de referência encontrado |
| `boundary_f1` | F1 dos contornos com tolerância de 2 px |
| `porosity_total` | % de poro na janela inteira |
| `porosity_rockface_rock` | % de poro na "rocha" do rockface (definição do rockface) |

## 9. Desempenho e memória

Medido numa máquina com 16 threads e 64 GB, sem GPU no pipeline:

| Etapa (por patch) | `--scale 0.5` (2048²) |
|---|---|
| 153 atributos | ~30 s, ~2,6 GB de RAM |
| Treino RF (600 mil px) | ~2 min |
| Treino LGBM | ~10 s |
| Previsão RF / LGBM | ~12 s / ~10 s |

Em resolução nativa (`--scale 1.0`) o array de atributos teria ~10 GB por patch; use
`laminas.compute_features_tiled` (blocos de 1024 px com margem de 48 px).

## 10. Problemas conhecidos

- **"Rocha" do rockface subestimada.** Nestes patches escuros, 7–27 % dos pixels
  (grãos escuros/peloides) têm todos os canais ≤ 5 e saem da área de rocha, o que
  **infla a porosidade** calculada por `legacy/petrophysical_properties.py`.
  Reportar a @oi-silva (task de métricas petrofísicas).
- **Franja ciano-clara nas bordas dos poros.** O rockface deixa de fora uma faixa de
  ~20–40 px; é preciso decisão dos especialistas (seção 6).
- **`class_weight="balanced_subsample"`** falhou de forma intermitente no
  scikit-learn 1.9 / Python 3.14 (`classes should have valid labels that are in y`);
  usamos `"balanced"`.
- **Uma só lâmina.** A validação é agrupada por patch; a Task 1.1 pede divisão por
  lâmina/poço — refazer quando houver mais lâminas.

## 11. Como contribuir

- Uma task nova = uma pasta em `tasks/`, com `README.md` e scripts numerados
  (`00_…`, `01_…`) na ordem de execução.
- Código reutilizável vai para `src/laminas/`; scripts só orquestram.
- Não versionar imagens das lâminas nem nada derivado delas (máscaras em PNG, figuras
  com a lâmina); versionar tabelas (CSV/JSON) em `results/`.
- Commits pequenos, mensagem dizendo o quê e por quê.
