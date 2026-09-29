# Guia de uso — ROCKFACE-LCCMAT

Como instalar, rodar e entender os scripts da Equipe A. Testado em Windows 11 com
Python 3.14 (Git Bash); em Linux/macOS só muda o comando de ativar o ambiente.

**Sumário**

1. [Instalação](#1-instalação)
2. [Dados de entrada](#2-dados-de-entrada)
3. [Como o código está organizado](#3-como-o-código-está-organizado)
4. [Task 1.2 — os scripts, na ordem](#4-task-12--os-scripts-na-ordem)
5. [Anotação pelos especialistas](#5-anotação-pelos-especialistas)
6. [Piloto × especialista](#6-piloto--especialista)
7. [Resultados e métricas](#7-resultados-e-métricas)
8. [Tempo e memória](#8-tempo-e-memória)
9. [Problemas conhecidos](#9-problemas-conhecidos)
10. [Regras para novos scripts](#10-regras-para-novos-scripts)

---

## 1. Instalação

```bash
git clone https://github.com/williamjot/ROCKFACE-LCCMAT.git
cd ROCKFACE-LCCMAT
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Bibliotecas usadas (todas comuns em ciência de dados): numpy, scipy, pandas, pillow,
matplotlib, OpenCV, scikit-image, scikit-learn e LightGBM.

Não é preciso instalar a pasta `laminas/`: cada script tem, no topo, uma linha que
diz ao Python onde ela está:

```python
sys.path.append(str(Path(__file__).resolve().parents[3]))  # para o Python achar a pasta laminas/
```

## 2. Dados de entrada

Patches exportados pelo `rockface`, canal 0 (luz normal/PPL):

- nome `patch_y{Y}_x{X}_c0.png` (Y, X = posição do patch na lâmina, em pixels);
- RGB 8 bits, 4096 × 4096 px; o rockface anda de 3800 em 3800 px, então patches
  vizinhos se sobrepõem 296 px;
- poros preenchidos com resina epóxi azul.

Coloque os arquivos em `data/patches/` (essa pasta não vai para o git). Para usar
outra pasta, mude `PASTA_DADOS` em `laminas/config.py`.

As imagens são muito escuras (média ≈ 20 de 255), por isso os scripts normalizam a
cor antes de calcular qualquer atributo.

## 3. Como o código está organizado

```
laminas/                     funções usadas por vários scripts
├── config.py                pastas e constantes (edite aqui)
├── imagens.py               ler patches, mudar tamanho, normalizar a cor
├── baseline.py              limiar de cor do rockface
├── atributos.py             os 153 atributos por pixel
├── rotulos.py               ler/salvar rótulos; pseudo-rabiscos do piloto
├── modelos.py               RF, LightGBM, RF-cor; prever; máscara final
└── metricas.py              IoU, Dice, F1 de borda, porosidade

tasks/task_1_poros/task_1_2_desempenho_modelos/
├── 00_explorar_dados.py
├── 01_gerar_kit_anotacao.py
├── 02_extrair_amostras.py
├── 03_validacao_cruzada_piloto.py
├── 04_avaliar_especialista.py
└── 05_figuras.py
```

Nos scripts, a biblioteca é usada assim:

```python
import laminas as L

arquivos = L.listar_patches()
limites = L.limites_de_cor(arquivos, L.PASTA_CACHE / "limites_de_cor.npy")
img = L.carregar_rgb(arquivos[0], escala=0.5)
poros = L.mascara_rockface(img)                         # True = poro
atributos = L.calcular_atributos(L.normalizar(img, limites))   # (2048, 2048, 153)
print(L.porosidade(poros), atributos.shape)
```

### O baseline (`baseline.py`)

Reproduz o limiar do rockface (`rockface/masks.py`): desfoque 5×5 → HSV → poro onde
a matiz está entre 75 e 125 (azul/ciano) e saturação × brilho ≥ 0,1 → abertura 3×3.
A "área de rocha" do rockface é "algum canal > 5" (ver [problemas conhecidos](#9-problemas-conhecidos)).

### Os 153 atributos (`atributos.py`, Tabela 3 da Task 1.1)

- **9 de cor:** R, G, B, H, S, V, L\*, a\*, b\*.
- **144 de filtros:** 8 filtros × 6 escalas × 3 canais (L\*, b\*, S).
  - Filtros: gaussiana (média da vizinhança), gradiente (borda), DoG e LoG (manchas
    e poros pequenos), Hessiana (formas alongadas), tensor de estrutura (textura).
  - Escalas σ = 0,7; 1; 1,6; 3,5; 5; 10 pixels. Cada filtro "enxerga" cerca de 3σ em
    volta do pixel.
- Nomes como `b_gauss_s3.5` (canal b\*, gaussiana, σ = 3,5) e `color_b`.

## 4. Task 1.2 — os scripts, na ordem

Rode de dentro de `tasks/task_1_poros/task_1_2_desempenho_modelos/`.

| # | Script | O que faz | Tempo* |
|---|---|---|---|
| 00 | `00_explorar_dados.py` | tabela por patch: brilho, porosidade do rockface, pixels fora da "rocha" | 2 min |
| 01 | `01_gerar_kit_anotacao.py` | separa treino/teste e gera o kit para os especialistas | 5 min |
| 02 | `02_extrair_amostras.py` | calcula os atributos e guarda os pixels rotulados de cada patch | 9 min |
| 03 | `03_validacao_cruzada_piloto.py` | compara os modelos no piloto | 30 min |
| 04 | `04_avaliar_especialista.py` | compara os modelos com os rótulos dos especialistas | ~10 min |
| 05 | `05_figuras.py` | imagens de comparação RF × rockface | 1 min |

\* escala 0,5, 16 threads, sem GPU.

```bash
python 00_explorar_dados.py
python 01_gerar_kit_anotacao.py
python 02_extrair_amostras.py --escala 0.5
python 03_validacao_cruzada_piloto.py --escala 0.5 --folds 6
python 05_figuras.py --escala 0.5
# quando houver rótulos dos especialistas:
python 02_extrair_amostras.py --escala 0.5
python 04_avaliar_especialista.py --escala 0.5
```

**Opções:**

| Opção | Scripts | Padrão | Significado |
|---|---|---|---|
| `--escala` | 02, 03, 04, 05 | 0.5 | tamanho de trabalho: 1.0 = original (4096 px), 0.5 = metade (2048 px); use a mesma em todos |
| `--folds` | 03 | 6 | em quantos grupos os patches são divididos na validação cruzada |
| `--semente` | 02, 03, 04 | 42 | semente dos sorteios (mesma semente = mesmo resultado) |
| `--limite` | 03 | 0 | usar só os N primeiros patches (teste rápido) |

Teste rápido do pipeline inteiro (~5 min):

```bash
python 02_extrair_amostras.py --escala 0.25
python 03_validacao_cruzada_piloto.py --escala 0.25 --folds 3 --limite 3
```

**Os modelos comparados** (`laminas/modelos.py`):

| Nome | O que é |
|---|---|
| `rockface` | limiar de cor (baseline) |
| `RF` | Random Forest: 200 árvores, `max_features="sqrt"`, `min_samples_leaf=5`, 153 atributos |
| `LGBM` | LightGBM: 400 árvores, `num_leaves=63`, taxa de aprendizado 0,05, 153 atributos |
| `RF-cor` | o mesmo RF, só com os 9 atributos de cor (para ver se os filtros ajudam) |

A probabilidade de poro vira máscara com limiar 0,5 e uma abertura morfológica de
raio 1 px. Tudo é avaliado na janela sem sobreposição `[0:3800, 0:3800]` de cada patch.

## 5. Anotação pelos especialistas

O `01_gerar_kit_anotacao.py` cria em `annotations/`:

| Pasta | Conteúdo | O que fazer |
|---|---|---|
| `images/` | patches realçados (mais claros) | base para anotar |
| `scribbles/` | vazia | salvar `patch_<nome>_scribbles.png` dos 12 patches de **treino** |
| `test_masks/` | pré-rótulo do rockface numa janela 1024² de cada patch de **teste** | **corrigir a janela inteira** |
| `test_windows/` | recorte de cada janela + pré-rótulo | para revisar com calma |
| `split.json` | patches de treino/teste e posição das janelas | pôr `"test_masks_corrected": true` ao terminar |

**Formato:** PNG com o mesmo tamanho do patch (4096 × 4096), com os números
0 = sem rótulo, 1 = poro, 2 = sólido (é o que o napari salva). Também vale PNG
colorido: **vermelho = poro**, **verde = sólido**, resto preto ou transparente.

**Programas:**

- **napari** (recomendado): `pip install "napari[all]"`; abra a imagem, crie uma
  camada *Labels*, pinte com o rótulo 1 (poro) e 2 (sólido) e salve a camada como PNG.
- **GIMP:** camada nova transparente por cima da imagem; ferramenta **Lápis** (não
  Pincel, que mistura as cores) em vermelho `#FF0000` e verde `#00C800`; exporte só
  essa camada.

**Regras:**

- Rabisque bordas, poros pequenos, bolhas e variações de cor, não só o meio dos poros.
- Na dúvida, não rabisque (pixel sem rótulo não entra no treino).
- Patches de teste não recebem rabiscos; a janela de teste tem de ser revisada inteira.
- Antes de começar, o grupo decide: a franja ciano-clara nas bordas dos poros é
  poro, sólido ou terceira classe? Qual a área mínima de poro?

Os rótulos não vão para o git por padrão (são derivados das lâminas).

## 6. Piloto × especialista

| | Piloto (`03_…`) | Especialista (`04_…`) |
|---|---|---|
| Quando | agora, sem rótulos | depois da anotação |
| Treino | pseudo-rabiscos: 20 mil pixels de cada classe por patch, tirados do miolo das regiões do rockface | rabiscos dos 12 patches de treino |
| Teste | validação cruzada agrupada por patch | janelas corrigidas dos 6 patches de teste |
| Comparado com | o próprio rockface | a correção dos especialistas |
| As métricas medem | **concordância com o rockface** | **acerto** |

No piloto o modelo aprende a imitar o rockface. Ele serve para testar o pipeline,
medir o tempo, ver quais atributos importam e achar onde os métodos discordam — não
para escolher o modelo.

## 7. Resultados e métricas

Em `results/task_1_2/piloto_escala<E>/` (e `especialista_escala<E>/`):

| Arquivo | Conteúdo |
|---|---|
| `metrics_per_patch.csv` | uma linha por patch × método |
| `metrics_summary.csv` | média e desvio padrão entre patches |
| `rf_feature_importance.csv` | importância de cada atributo no RF |
| `run_info.json` | parâmetros usados |

Em `cache/task_1_2/` (não vai para o git): `amostras_escala<E>/` (pixels de treino),
`piloto_escala<E>/mascaras/` e `piloto_escala<E>/figuras/` (amarelo = poro para os
dois, magenta = só RF, ciano = só rockface).

| Métrica | Significado |
|---|---|
| `iou`, `dice` | quanto as duas máscaras de poro se sobrepõem (1 = iguais) |
| `precision` | do que o modelo chamou de poro, quanto é poro |
| `recall` | dos poros de referência, quanto o modelo encontrou |
| `boundary_f1` | quanto os contornos coincidem (tolerância de 2 px) |
| `porosity_total` | % de poro na janela inteira |
| `porosity_rockface_rock` | % de poro dentro da "rocha" do rockface |

## 8. Tempo e memória

Numa máquina com 16 threads e 64 GB de RAM, escala 0,5 (2048 × 2048):

- atributos de um patch: ~30 s e ~5 GB de RAM;
- treino: RF ~2 min, LightGBM ~10 s (600 mil pixels);
- previsão de um patch: RF ~12 s, LightGBM ~10 s.

Na escala 1,0 os atributos de um patch ocupariam ~10 GB (o dobro no pico); para isso
será preciso processar a imagem em blocos.

## 9. Problemas conhecidos

- **"Rocha" do rockface subestimada:** nestes patches escuros, 7–27 % dos pixels
  (grãos escuros) têm todos os canais ≤ 5 e saem da área de rocha, o que **aumenta a
  porosidade** calculada pelo rockface (até 3,5 pontos percentuais por patch).
- **Franja ciano-clara** nas bordas dos poros: nenhum método a pega; precisa de
  decisão dos especialistas.
- **`class_weight="balanced_subsample"`** falhou de forma intermitente no
  scikit-learn 1.9 / Python 3.14; usamos `"balanced"`.
- **Falha aleatória em execuções longas (em investigação).** Na máquina de testes, o
  `03_validacao_cruzada_piloto.py` às vezes para depois de 10–25 min com
  `operands could not be broadcast together with shapes (1000000,2) (1000000,3)`,
  `classes should have valid labels that are in y` ou fecha sem mensagem. Aconteceu com
  Python 3.14 e 3.10, com o código antigo e o novo, dentro e fora do Claude, em pontos
  diferentes; trechos isolados sempre passam. Suspeitas: condição de corrida na previsão
  paralela do scikit-learn ou instabilidade da máquina sob carga. Se acontecer, rode de
  novo; a execução completa do piloto publicada em `results/` terminou sem erro.
- **Só uma lâmina:** a validação separa por patch; a Task 1.1 pede separar por
  lâmina/poço, o que deve ser feito quando houver mais lâminas.

## 10. Regras para novos scripts

O código deve ser simples o bastante para um aluno de graduação ler e escrever:

- Um script = uma etapa, numerado na ordem de execução (`00_…`, `01_…`), lido de
  cima para baixo, com uma docstring no topo dizendo o que faz, o que precisa e o que salva.
- Função só quando ela é usada em mais de um lugar ou deixa o código mais claro;
  funções usadas por vários scripts vão para `laminas/`.
- Nomes em português, laços `for` explícitos, sem truques de sintaxe, sem dicas de
  tipo nem classes quando uma função resolve.
- Parâmetros no topo do script ou com `argparse` (poucos).
- Não versionar imagens das lâminas nem nada derivado delas; versionar tabelas
  (CSV/JSON) em `results/`.
