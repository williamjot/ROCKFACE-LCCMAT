# Roteiro do projeto — Segmentação de Lâminas Petrográficas (Equipe A)

Documento-guia da Equipe A (LCCMat-UnB): **o que** o projeto quer, **como** vamos
chegar lá, **quem faz o quê** em cada etapa, **como anotar** as imagens e a **teoria de
aprendizado de máquina** por trás dos métodos.

- Repositório do projeto: <https://github.com/LCCMat-UnB/segmentacao-laminas>
- Pacote `rockface`: <https://github.com/LCCMat-UnB/rockface>
- Revisão de literatura (Task 1.1): [`Task_1.1_Revisao_Literatura_Equipe_A.pdf`](Task_1.1_Revisao_Literatura_Equipe_A.pdf)
- Como rodar os scripts: [`GUIA_DE_USO.md`](GUIA_DE_USO.md)

**Sumário**

- [Parte I — O projeto](#parte-i--o-projeto)
  - [1. Contexto e objetivo](#1-contexto-e-objetivo)
  - [2. Etapas (tasks)](#2-etapas-tasks)
  - [3. Os dados](#3-os-dados)
  - [4. Pipeline de ponta a ponta](#4-pipeline-de-ponta-a-ponta)
  - [5. Onde estamos e próximos passos](#5-onde-estamos-e-próximos-passos)
- [Parte II — Roteiro de anotação](#parte-ii--roteiro-de-anotação)
  - [6. Por que anotar e o que anotar](#6-por-que-anotar-e-o-que-anotar)
  - [7. Decisões antes de começar](#7-decisões-antes-de-começar)
  - [8. Ferramentas: napari e GIMP, passo a passo](#8-ferramentas-napari-e-gimp-passo-a-passo)
  - [9. Rabiscos de treino](#9-rabiscos-de-treino)
  - [10. Correção das janelas de teste](#10-correção-das-janelas-de-teste)
  - [11. Controle de qualidade da anotação](#11-controle-de-qualidade-da-anotação)
- [Parte III — Teoria de aprendizado de máquina](#parte-iii--teoria-de-aprendizado-de-máquina)
  - [12. Segmentação como classificação de pixels](#12-segmentação-como-classificação-de-pixels)
  - [13. Atributos: como um pixel vira um vetor de números](#13-atributos-como-um-pixel-vira-um-vetor-de-números)
  - [14. Árvores de decisão](#14-árvores-de-decisão)
  - [15. Random Forest](#15-random-forest)
  - [16. Gradient boosting e LightGBM](#16-gradient-boosting-e-lightgbm)
  - [17. Redes neurais convolucionais e U-Net](#17-redes-neurais-convolucionais-e-u-net)
  - [18. Modelos de fundação: SAM e DINO](#18-modelos-de-fundação-sam-e-dino)
  - [19. Treino, validação e teste](#19-treino-validação-e-teste)
  - [20. Métricas](#20-métricas)
  - [21. Armadilhas: pseudo-rótulos, atalhos e vazamento](#21-armadilhas-pseudo-rótulos-atalhos-e-vazamento)
  - [22. Anotação ativa](#22-anotação-ativa)
  - [23. Pós-processamento e segmentação de grãos](#23-pós-processamento-e-segmentação-de-grãos)
- [Glossário](#glossário)
- [Leituras recomendadas](#leituras-recomendadas)

---

# Parte I — O projeto

## 1. Contexto e objetivo

Uma **lâmina delgada** é uma fatia de rocha de ~30 µm colada em vidro e fotografada no
microscópio. Antes de cortar, a rocha é impregnada com **resina epóxi azul**: tudo que
era espaço vazio (poro) fica azul na imagem. Medir a área azul dá a **porosidade**, uma
das propriedades mais importantes de uma rocha-reservatório.

Hoje o pacote `rockface` do grupo separa o azul por um **limiar de cor** (HSV). Isso
funciona bem no "miolo" dos poros grandes, mas erra em bordas, poros pequenos,
microporosidade e variações de tingimento/iluminação.

**Objetivo da Equipe A (fluxo 1):** encontrar e validar um método de aprendizado de
máquina que segmente poros melhor que o limiar, de forma **mensurável** (métricas por
pixel e petrofísicas) e **aplicável em escala** (lâminas inteiras). O mesmo método será
depois retreinado para segmentar **grãos** (fluxo 2).

## 2. Etapas (tasks)

| Etapa | Pergunta | Entrega | Estado |
|---|---|---|---|
| **1.1** Revisão de literatura | Que métodos já foram usados em lâminas/rochas? | PDF com 40 referências, shortlist e modelo escolhido | ✅ |
| **1.2** Desempenho dos modelos | Qual candidato segmenta melhor os nossos poros? | Comparação com métricas por pixel e petrofísicas | 🔄 piloto feito; falta anotação |
| **1.3** Seleção do modelo | Qual modelo (e parâmetros) usar em escala? | Decisão justificada; pipeline para lâmina inteira | ⏳ |
| **1.4** Documentação e paper | Como registrar e publicar? | Relatório e rascunho de artigo | ⏳ |
| **2** Grãos | Como separar grãos individuais? | Pipeline PPL + XPL + watershed | ⏳ |
| **3** Classificação de poros | Que tipo de poro é cada um? | Métodos (classes de Lucia etc.) | ⏳ (fora da Equipe A) |
| **4** Classificação de grãos | Granulometria, seleção, arredondamento | Métodos | ⏳ (fora da Equipe A) |

**Candidatos da Task 1.2** (shortlist da Task 1.1, Seção 5):

| Prioridade | Candidato | Papel |
|---|---|---|
| 0 | Limiar HSV do `rockface` | baseline obrigatório |
| 1 | **Random Forest pixel a pixel + banco de filtros** | candidato principal |
| 1 | LightGBM com os mesmos atributos | alternativa direta |
| 2 | RF por superpixel (SLIC) | menos pixels, atributos de região |
| 2 | U-Net (encoder ResNet34) | referência de deep learning |
| — | DINOv3, SAM 2 | trabalho futuro |

**Critério de troca (Task 1.1, Seção 6):** o LightGBM substitui o RF se ganhar ≥ 2
pontos de IoU (ou tiver erro de porosidade menor) no teste corrigido. Se a U-Net superar
ambos com margem relevante — sobretudo em bordas e microporosidade — a escolha é
reavaliada na Task 1.3, pesando o ganho contra o custo de rótulos densos e GPU.

## 3. Os dados

- **Patches** exportados pelo `rockface`: `patch_y{Y}_x{X}_c0.png`, RGB, 4096 × 4096 px.
  O rockface anda de 3800 em 3800 px, então patches vizinhos se sobrepõem 296 px; por
  isso só a janela `[0:3800, 0:3800]` de cada patch entra nas contas.
- **Canal 0** = luz normal (PPL, plano-polarizada), onde a resina azul aparece. Os outros
  canais (XPL, polarizadores cruzados, vários ângulos) serão usados para grãos.
- **Conjunto atual:** 18 patches de **uma** lâmina de carbonato (grãos arredondados,
  tipo oóides/peloides), muito escuros (brilho médio ≈ 20 de 255).
- **Não vão para o GitHub** (repositório público): imagens das lâminas e tudo que é
  derivado delas (máscaras, figuras). Ver [`data/README.md`](../data/README.md).

## 4. Pipeline de ponta a ponta

```
           ┌────────────────────────────── Task 1 (poros) ──────────────────────────────┐
lâmina ──► patches PPL ──► normalização ──► 153 atributos ──► classificador ──► prob. de ──► máscara ──► porosidade,
(.czi)     (rockface)      de cor           por pixel         (RF/LightGBM)     poro         de poros    tamanho, forma
                                                   ▲                  ▲
                                   rabiscos dos especialistas ────────┘
                                   (treino)          janelas corrigidas (teste) ──► métricas

           ┌────────────────────────────── Task 2 (grãos) ──────────────────────────────┐
PPL + XPL (n ângulos) ──► atributos PPL + atributos de extinção XPL ──► RF (grão/borda/poro/matriz)
                      ──► watershed marcado ──► grãos individuais ──► granulometria, arredondamento
```

Scripts da Task 1.2 (detalhes em [`GUIA_DE_USO.md`](GUIA_DE_USO.md)):

| # | Script | Faz |
|---|---|---|
| 00 | `00_explorar_dados.py` | estatísticas dos patches |
| 01 | `01_gerar_kit_anotacao.py` | separa treino/teste e prepara o kit de anotação |
| 02 | `02_extrair_amostras.py` | atributos dos pixels rotulados |
| 03 | `03_validacao_cruzada_piloto.py` | comparação no piloto (sem especialistas) |
| 04 | `04_avaliar_especialista.py` | comparação com os rótulos dos especialistas |
| 05 | `05_figuras.py` | imagens de comparação |

## 5. Onde estamos e próximos passos

**Feito**

- Revisão de literatura e escolha do candidato principal (Task 1.1).
- Pipeline completo da Task 1.2 e **piloto** com pseudo-rótulos: IoU com o `rockface`
  de 0,93–0,97; porosidade quase igual entre métodos; as diferenças ficam nas bordas e
  em poros pequenos/difusos ([`RESULTADOS_PILOTO.md`](../results/task_1_2/RESULTADOS_PILOTO.md)).
- Achado: a "área de rocha" do `rockface` exclui 7–27 % dos pixels (grãos escuros), o que
  aumenta a porosidade em até 3,5 pontos percentuais por patch.
- Kit de anotação pronto em `annotations/`.

**Próximos passos, em ordem**

| # | Passo | Quem | Depende de |
|---|---|---|---|
| 1 | Decidir regras de rotulagem (franja ciano, microporosidade, área mínima) | grupo + orientador | — |
| 2 | Rabiscos nos 12 patches de treino | especialistas/equipe | 1 |
| 3 | Correção das 6 janelas de teste (2 anotadores numa parte delas) | especialistas | 1 |
| 4 | Rodar 02 e 04 (modo especialista) | Equipe A | 2, 3 |
| 5 | Anotação ativa: mais rabiscos onde o modelo tem dúvida; rodar de novo | especialistas + Equipe A | 4 |
| 6 | RF por superpixel e U-Net | Equipe A | 4 (U-Net precisa de rótulos densos) |
| 7 | Métricas petrofísicas (com a task do `rockface`) | Equipe A + @oi-silva | 4 |
| 8 | Mais lâminas → validação por lâmina/poço | grupo | dados |
| 9 | Escolha do modelo (Task 1.3) e escrita (Task 1.4) | Equipe A | 4–8 |

---

# Parte II — Roteiro de anotação

## 6. Por que anotar e o que anotar

Um modelo supervisionado aprende com **exemplos rotulados** (seção 12). Se os exemplos
vierem do próprio `rockface`, o modelo só aprende a imitá-lo, com os mesmos erros — e as
métricas contra o `rockface` medem concordância, não acerto (seção 21). Por isso
precisamos de dois tipos de rótulo feitos por pessoas que conhecem a rocha:

| Tipo | Onde | Quanto | Para quê |
|---|---|---|---|
| **Rabiscos** (esparsos) | 12 patches de **treino** | alguns traços por patch | ensinar o modelo |
| **Máscara densa corrigida** | 1 janela de 1024 × 1024 px em cada um dos 6 patches de **teste** | todos os pixels da janela | medir o acerto |

Os dois conjuntos **nunca** se misturam: um patch de teste não recebe rabiscos, senão o
teste deixa de ser independente (seção 19).

O kit (`annotations/`, gerado pelo `01_gerar_kit_anotacao.py`):

| Pasta | Conteúdo |
|---|---|
| `images/` | patches **realçados** (mais claros) para anotar |
| `scribbles/` | onde salvar os rabiscos (`patch_<nome>_scribbles.png`) |
| `test_masks/` | pré-rótulo do `rockface` nas janelas de teste, para **corrigir** |
| `test_windows/` | recorte de cada janela + pré-rótulo, para revisar com calma |
| `split.json` | lista de patches de treino/teste e posição das janelas |

## 7. Decisões antes de começar

Registrar as decisões (em `annotations/REGRAS.md` ou na issue do projeto) **antes** de
anotar, e todos seguirem as mesmas:

1. **Franja ciano-clara nas bordas dos poros** (faixa de ~20–40 px, azul mais claro que o
   poro): é poro, sólido ou fica sem rótulo? Pode ser cimento sobre o grão, resina em
   lâmina mais fina, ou microporosidade.
2. **Microporosidade** (azul difuso dentro de grãos): poro, sólido ou terceira classe?
   Uma terceira classe exige mudar o código (hoje: 2 classes).
3. **Bolhas, arrancamentos, riscos, poeira:** como rotular.
4. **Área mínima de poro** (`A_min`): abaixo de quantos pixels um objeto azul é ruído?
   Precisa do **tamanho do pixel em µm** (metadados do `.czi`).

## 8. Ferramentas: napari e GIMP, passo a passo

**Formato que os scripts aceitam:** PNG do mesmo tamanho do patch (4096 × 4096), com
- números **0 = sem rótulo, 1 = poro, 2 = sólido** (o que o napari salva), **ou**
- cores: **vermelho = poro**, **verde = sólido**, resto preto/transparente (GIMP).

### napari (recomendado)

Gratuito, em Python, feito para anotar imagens de microscopia.

**Instalar (uma vez)** — num ambiente separado do projeto:

```bash
py -3.10 -m venv C:\Users\<usuario>\napari-env
C:\Users\<usuario>\napari-env\Scripts\python -m pip install "napari[all]"
```

(Linux/macOS: `python3 -m venv ~/napari-env && ~/napari-env/bin/pip install "napari[all]"`.)

**Abrir:** `C:\Users\<usuario>\napari-env\Scripts\napari`

**Anotar um patch:**

1. Arraste `annotations/images/patch_<nome>_enhanced.png` para a janela.
2. Painel esquerdo → ícone **"New labels layer"** (etiqueta). Aparece a camada *Labels*.
3. Ferramenta **pincel** (tecla `P`); ajuste o *brush size* (5–15 px para rabiscos).
4. Campo **label = 1** → pinte poros. Campo **label = 2** → pinte sólidos.
5. Borracha: tecla `E` (ou label 0). Zoom: roda do mouse. Mover: arrastar com a mão (`Space`).
6. Salvar: selecione a camada *Labels* → **File → Save Selected Layer(s)…** →
   `annotations/scribbles/patch_<nome>_scribbles.png`.

Dica: diminua a opacidade da camada *Labels* (0,5) para ver a rocha por baixo.

### GIMP (alternativa)

1. Abra `annotations/images/patch_<nome>_enhanced.png`.
2. **Camada → Nova camada…** → preenchimento **Transparência**.
3. Ferramenta **Lápis** (`N`) — **não** use o Pincel, que suaviza as bordas e cria cores
   misturadas. Cor **vermelho `#FF0000`** para poro, **verde `#00C800`** para sólido.
4. Pinte na camada nova.
5. Esconda a camada da imagem (olho) e **Arquivo → Exportar como…** →
   `annotations/scribbles/patch_<nome>_scribbles.png`.

### Outras opções

- **ilastik** — mesma ideia (rabisco + Random Forest) com interface pronta; útil para
  testar rapidamente, mas os rótulos precisam ser exportados como imagem 0/1/2.
- **QuPath** — bom para lâminas inteiras e anotações por polígono.

## 9. Rabiscos de treino

**Onde:** os 12 patches de `"train"` em `annotations/split.json`.

**Quanto:** algo como 10–20 traços por patch, cobrindo:

| Poro (1) | Sólido (2) |
|---|---|
| miolo de poros grandes (poucos traços bastam) | grãos escuros (peloides) |
| **bordas** de poros (traço fino a 2–3 px da borda) | grãos claros e cristalinos |
| **poros pequenos** | cimento entre grãos |
| poros com tingimento mais fraco/escuro | regiões muito escuras (que o rockface nem conta como rocha) |
| gargantas finas entre poros | bolhas/riscos (se a regra for "sólido") |

**Regras de ouro**

- **Na dúvida, não rabisque.** Pixel sem rótulo simplesmente não entra no treino; pixel
  com rótulo errado ensina errado.
- Prefira **variedade** a quantidade: 20 traços em lugares diferentes valem mais que um
  traço enorme num só poro.
- Cubra de propósito os **casos difíceis** (bordas, poros pequenos, cores atípicas) — é
  neles que o modelo precisa aprender.
- Não pinte o patch inteiro: rabisco é esparso.

## 10. Correção das janelas de teste

**Onde:** os 6 patches de `"test"`; a janela está em `split.json` (`"windows"`), e o
pré-rótulo em `annotations/test_masks/patch_<nome>_mask.png`.

1. Abra no napari a imagem realçada **e** o pré-rótulo (arraste os dois; o pré-rótulo
   deve ser aberto como *Labels*: botão direito na camada → *Convert to Labels*).
2. Percorra a janela inteira com zoom (sugestão: quadrantes de 256 × 256 px).
3. Corrija: pinte com 1 o que é poro e ficou como sólido; com 2 o que é sólido e ficou
   como poro. Atenção especial a bordas, poros pequenos e a franja ciano.
4. Pixels realmente ambíguos: apague (0) — ficam fora da avaliação.
5. Salve por cima de `annotations/test_masks/patch_<nome>_mask.png`.
6. Quando as 6 janelas estiverem revisadas, mude em `split.json`:
   `"test_masks_corrected": true`. O script `04_avaliar_especialista.py` só roda assim.

**Importante:** o pré-rótulo vem do limiar. Se a janela não for revisada **inteira**, o
viés do limiar volta para a referência e a avaliação fica a favor do `rockface`.

## 11. Controle de qualidade da anotação

- **Dois anotadores** corrigem as mesmas 2 janelas, sem ver o trabalho um do outro.
  O IoU entre eles é a **variabilidade humana** — o teto realista para qualquer modelo.
- **Registro:** quem anotou o quê e quando (planilha simples).
- **Revisão cruzada:** um segundo membro olha rapidamente os rabiscos de cada patch.
- **Versão:** rótulos corrigidos são dados valiosos — guardar cópia em local combinado
  pelo grupo (não vão para o GitHub público).

---

# Parte III — Teoria de aprendizado de máquina

## 12. Segmentação como classificação de pixels

**Segmentar** é dizer, para cada pixel, a que classe ele pertence (aqui: poro ou sólido).
A forma mais simples de fazer isso com aprendizado de máquina é tratar **cada pixel como
um exemplo** de um problema de **classificação supervisionada**:

- **entrada** $\mathbf{x} \in \mathbb{R}^d$: um vetor de $d$ números que descreve o pixel
  e sua vizinhança (os *atributos*, seção 13);
- **saída** $y \in \{0, 1\}$: sólido ou poro;
- **treino:** dado um conjunto de pares $(\mathbf{x}_i, y_i)$ rotulados, o algoritmo
  aprende uma função $f$ tal que $f(\mathbf{x}) \approx y$;
- **previsão:** aplica $f$ a todos os pixels da imagem e obtém um **mapa de
  probabilidade** $\hat p(\text{poro} \mid \mathbf{x})$, que vira máscara com um limiar.

*Supervisionado* quer dizer que aprendemos a partir de exemplos com a resposta certa.
Por isso a qualidade dos rótulos (Parte II) limita a qualidade do modelo.

Por que não usar só um limiar de cor? Porque a cor de um pixel isolado é ambígua:
um pixel na borda do poro mistura azul e grão; um poro mal tingido é azul-escuro; um
grão pode ter reflexos azulados. Olhar a **vizinhança** em várias escalas resolve boa
parte dessas ambiguidades — é o que os atributos fazem.

## 13. Atributos: como um pixel vira um vetor de números

Usamos 153 atributos por pixel (Tabela 3 da Task 1.1, padrão do ilastik).

**Cor (9):** R, G, B; H, S, V (matiz, saturação, brilho); L\*, a\*, b\* (espaço CIELAB,
em que L\* é a luminosidade e b\* vai de azul, negativo, a amarelo, positivo). O canal
b\* separa bem a resina azul.

**Filtros (144 = 8 filtros × 6 escalas × 3 canais L\*, b\*, S).** Cada filtro é uma
convolução da imagem com um núcleo; a **escala** $\sigma$ controla o tamanho da
vizinhança considerada (≈ $3\sigma$ pixels para cada lado). Usamos
$\sigma \in \{0{,}7;\ 1;\ 1{,}6;\ 3{,}5;\ 5;\ 10\}$.

| Filtro | Definição | O que detecta |
|---|---|---|
| Gaussiana $G_\sigma * I$ | média ponderada da vizinhança | cor típica da região; reduz ruído |
| Magnitude do gradiente $\lVert \nabla (G_\sigma * I) \rVert$ | quanto a intensidade muda | bordas poro/grão |
| DoG $G_\sigma * I - G_{1{,}6\sigma} * I$ | diferença de duas suavizações | manchas do tamanho de $\sigma$ |
| LoG $\nabla^2 (G_\sigma * I)$ | laplaciano da gaussiana | pontos e poros pequenos ("blobs") |
| Autovalores da Hessiana $\lambda_{1,2}$ | curvatura local (2ª derivada) | estruturas alongadas: gargantas, fraturas |
| Autovalores do tensor de estrutura | orientação dominante dos gradientes | textura (microporosidade, cimento) |

Na Hessiana, $H = \begin{pmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{pmatrix}$ (derivadas
da imagem suavizada); se um autovalor é grande e o outro pequeno, há uma estrutura
linear; se os dois são grandes, uma mancha. O tensor de estrutura é
$S = G_\sigma * \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{pmatrix}$: autovalores
parecidos indicam textura sem direção; muito diferentes, bordas orientadas.

Combinando escalas, o classificador "vê" desde 2 px (detalhe fino) até ~30 px (contexto).
No piloto, os atributos mais usados pelo RF foram o **b\* suavizado** em várias escalas
("quão azul é a vizinhança") e o **tensor de estrutura da saturação** (textura).

## 14. Árvores de decisão

Uma **árvore de decisão** classifica fazendo perguntas do tipo "o atributo $j$ é menor
que $t$?". Cada pergunta divide os dados em dois grupos; cada grupo é dividido de novo,
até as **folhas**, que dão a resposta (a fração de poros entre os exemplos de treino que
caíram ali).

```
               b_gauss_s3.5 < -0,12 ?
                 /               \
               sim               não
         S_st2_s5 < 0,01 ?     → sólido (98 %)
            /        \
     poro (97 %)   poro (61 %)
```

**Como a árvore escolhe a pergunta:** em cada nó, testa atributos e limiares e fica com o
que deixa os dois grupos mais "puros". A impureza mais usada é o **índice de Gini**:

$$G = 1 - \sum_{k} p_k^2$$

onde $p_k$ é a fração da classe $k$ no nó. $G = 0$ se o nó só tem uma classe; $G = 0{,}5$
se metade é poro e metade sólido. A árvore escolhe o corte que mais **diminui** a
impureza média ponderada dos filhos.

**Problema:** uma árvore sozinha, crescida até o fim, **decora** os dados de treino
(sobreajuste): acerta tudo no treino e erra muito em dados novos. Pequenas mudanças nos
dados geram árvores muito diferentes (alta **variância**).

## 15. Random Forest

O **Random Forest** (Breiman, 2001) resolve a variância fazendo a **média de muitas
árvores diferentes**:

1. **Bootstrap:** cada árvore é treinada com uma amostra sorteada **com reposição** dos
   dados de treino (≈ 63 % de exemplos distintos).
2. **Atributos aleatórios:** em cada nó, a árvore só pode escolher entre um subconjunto
   aleatório de atributos — usamos $\sqrt{d} \approx 12$ dos 153 (`max_features="sqrt"`).
3. **Votação:** a probabilidade de poro é a média das árvores:

$$\hat p(\text{poro} \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} p_t(\text{poro} \mid \mathbf{x})$$

Os dois sorteios fazem as árvores errarem de formas diferentes; na média, os erros se
cancelam em parte. Se cada árvore tem variância $\sigma^2$ e correlação $\rho$ entre si, a
variância da média é $\rho\sigma^2 + \frac{1-\rho}{T}\sigma^2$: mais árvores ($T$) e árvores
menos correlacionadas ($\rho$ menor) → previsão mais estável.

**Parâmetros usados** (`laminas/modelos.py`):

| Parâmetro | Valor | Efeito |
|---|---|---|
| `n_estimators` | 200 | número de árvores (estável acima de ~100) |
| `max_features` | `"sqrt"` | atributos sorteados por nó |
| `min_samples_leaf` | 5 | folha precisa de ≥ 5 exemplos (limita o sobreajuste) |
| `class_weight` | `"balanced"` | compensa se uma classe tiver bem menos exemplos |

**Por que RF é o candidato principal** (Task 1.1, Seção 6.1): aprende com **poucos
rótulos** (rabiscos), treina em segundos/minutos **sem GPU**, tem poucos parâmetros
sensíveis, dá **probabilidades** (úteis para achar onde anotar) e é **interpretável**.

**Importância de atributos:** somando, em todas as árvores, quanto cada atributo reduziu
a impureza de Gini, obtemos a *importância de Gini*. Ela mostra se o modelo decide por
cor, borda ou textura — e ajuda a detectar atalhos (seção 21). Alternativa mais confiável
(e mais cara): *importância por permutação* — embaralhar um atributo e medir quanto o
desempenho cai.

## 16. Gradient boosting e LightGBM

O **boosting** também combina muitas árvores, mas **em sequência**: cada árvore nova é
treinada para corrigir os erros da soma das anteriores.

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \, h_m(\mathbf{x})$$

- $h_m$ é uma árvore pequena ajustada ao **gradiente da perda** (para classificação, a
  entropia cruzada) — ou seja, à direção em que o erro mais diminui;
- $\eta$ é a **taxa de aprendizado** (usamos 0,05): passos pequenos exigem mais árvores
  mas generalizam melhor;
- a probabilidade sai de $\hat p = 1 / (1 + e^{-F(\mathbf{x})})$ (função sigmoide).

RF × boosting: o RF reduz **variância** (média de árvores grandes e independentes); o
boosting reduz **viés** (soma de árvores pequenas que se corrigem). O boosting costuma
ganhar alguns pontos, mas tem mais parâmetros e sobreajusta mais facilmente com poucos
rótulos.

O **LightGBM** é uma implementação rápida: agrupa os valores de cada atributo em
histogramas e cresce as árvores "pela folha de maior ganho". Parâmetros usados: 400
árvores, `num_leaves=63`, `learning_rate=0.05`, 80 % dos exemplos e 50 % dos atributos
por árvore.

## 17. Redes neurais convolucionais e U-Net

Uma **CNN** aprende os próprios filtros em vez de usarmos filtros fixos. Cada camada
aplica muitos filtros $3 \times 3$ cujos pesos são ajustados no treino, seguidos de uma
não-linearidade (ReLU, $\max(0, z)$); empilhando camadas, a rede combina bordas em
texturas e texturas em objetos.

A **U-Net** (Ronneberger et al., 2015) é a arquitetura padrão de segmentação:

```
imagem ─► [codificador: convoluções + redução] ─► contexto ─► [decodificador: ampliação] ─► máscara
                     │                                                  ▲
                     └──────────── conexões de atalho (detalhe fino) ───┘
```

O codificador reduz a imagem e captura contexto amplo; o decodificador volta à resolução
original; as **conexões de atalho** trazem os detalhes finos para as bordas saírem
precisas. Treina-se minimizando uma perda por pixel (entropia cruzada, Dice) com
descida de gradiente.

**Custo:** precisa de **máscaras densas** (todo pixel rotulado) e GPU; com rótulos do
limiar, aprende a imitá-lo. Por isso na Task 1.2 ela é a **referência de deep learning**,
treinada depois que houver rótulos corrigidos. Usar um codificador **pré-treinado**
(ResNet34 no ImageNet) reduz muito a quantidade de rótulos necessária (*transfer learning*).

## 18. Modelos de fundação: SAM e DINO

Modelos treinados em milhões/bilhões de imagens, reaproveitáveis em outras tarefas:

- **SAM / SAM 2** (Segment Anything): segmenta objetos a partir de "dicas" (um ponto, uma
  caixa). Sem ajuste, vai mal em imagens de rocha; com *fine-tuning*, melhora muito.
- **DINOv2 / DINOv3**: aprende representações densas sem rótulos; um classificador
  simples sobre essas representações (ou ajuste leve com LoRA) funcionou bem em
  micro-CT de rocha com poucos e ruidosos rótulos.

Ficaram para trabalho futuro (custo, GPU, licença). A Task 2 pode usá-los para
pré-segmentar grãos.

## 19. Treino, validação e teste

- **Treino:** dados usados para ajustar o modelo.
- **Validação:** dados usados para escolher configurações (limiar $\tau$, número de
  árvores, escalas $\sigma$…).
- **Teste:** dados usados **uma vez**, no fim, para medir o desempenho. Nunca usados
  para escolher nada.

**Sobreajuste:** o modelo vai muito bem no treino e mal em dados novos. Só dados que o
modelo nunca viu revelam isso.

**Vazamento de dados:** pixels vizinhos são quase idênticos. Se treino e teste tiverem
pixels do **mesmo patch** (ou da mesma lâmina), o teste fica fácil demais e superestima
o desempenho. Por isso:

- separamos **por patch** (hoje, com uma só lâmina);
- o objetivo é separar **por lâmina** e, idealmente, **por poço** — no pré-sal, o F1 de um
  classificador caiu de 0,77 para 0,51 quando o teste passou a ser um poço fora do treino
  (Basso et al., 2025).

**Validação cruzada agrupada (GroupKFold):** divide os grupos (patches) em $K$ partes;
treina em $K-1$ e testa na restante; repete $K$ vezes. Todo patch é testado uma vez por
um modelo que não o viu. No piloto: $K = 6$, 3 patches por parte.

## 20. Métricas

Contagem de pixels comparando previsão e referência:

| | Referência: poro | Referência: sólido |
|---|---|---|
| **Previsto: poro** | VP (verdadeiro positivo) | FP (falso positivo) |
| **Previsto: sólido** | FN (falso negativo) | VN (verdadeiro negativo) |

| Métrica | Fórmula | Leitura |
|---|---|---|
| **IoU** (Jaccard) | $\frac{VP}{VP + FP + FN}$ | sobreposição das áreas de poro (0 a 1) |
| **Dice** (F1) | $\frac{2VP}{2VP + FP + FN}$ | parecido com IoU, um pouco mais "generoso" |
| Precisão | $\frac{VP}{VP + FP}$ | do que chamei de poro, quanto era poro |
| Revocação | $\frac{VP}{VP + FN}$ | dos poros reais, quanto encontrei |
| Acurácia | $\frac{VP + VN}{\text{total}}$ | **enganosa** aqui: com 8 % de poros, dizer "tudo sólido" dá 92 % |
| **F1 de borda** | F1 entre os pixels de contorno, com tolerância de 2 px | qualidade das bordas |
| **Porosidade** | $\frac{\text{pixels de poro}}{\text{pixels de rocha}}$ | a grandeza física de interesse |

IoU e porosidade se complementam: dois métodos podem ter a mesma porosidade (erros que se
cancelam: FP ≈ FN) com IoU diferente. Por isso reportamos os dois, e também métricas
**petrofísicas** (porosidade contra análises de laboratório; permeabilidade, na Task 1.3).

**Referência humana:** a concordância entre dois especialistas (seção 11) é o teto
prático — um modelo com IoU igual ao IoU entre humanos já está no limite do que dá para medir.

## 21. Armadilhas: pseudo-rótulos, atalhos e vazamento

- **Pseudo-rótulos:** rótulos gerados por outro método (aqui, o `rockface`). O modelo
  herda os erros dele, e avaliar contra ele mede **concordância**, não acerto. No piloto,
  o RF só com cor concordou mais com o `rockface` que o RF completo — porque o `rockface`
  também é um limiar de cor, não porque seja melhor.
- **Atalho por cor** (*shortcut learning*): o modelo pode aprender a tonalidade de um lote
  de resina ou de iluminação em vez da estrutura do poro, e falhar em outra lâmina.
  Mitigação: normalização de cor por lâmina, anotar lâminas diferentes, olhar a
  importância dos atributos, testar em lâminas/poços não vistos.
- **Desbalanceamento:** poros são ~8 % dos pixels. Métricas como acurácia enganam;
  `class_weight="balanced"` evita que o modelo ignore a classe rara.
- **Vazamento:** ver seção 19.

## 22. Anotação ativa

Em vez de anotar ao acaso, **anotar onde o modelo mais tem dúvida**:

1. treinar com os rabiscos iniciais;
2. calcular o mapa de probabilidade; as regiões com $\hat p \approx 0{,}5$ (ou onde o modelo
   discorda do `rockface`) são as mais informativas;
3. o especialista rabisca **só ali**;
4. retreinar e repetir até o desempenho na validação estabilizar.

Com isso, poucas horas de anotação rendem mais do que anotar o dobro ao acaso. É o ciclo
"anotar → treinar → ver o erro → anotar onde errou" do ilastik.

## 23. Pós-processamento e segmentação de grãos

**Da probabilidade à máscara** (`mascara_final`):

1. **limiar** $\tau$ (partida: 0,5; ajustar na validação para o melhor IoU ou menor erro de
   porosidade);
2. **abertura morfológica** (erosão seguida de dilatação): remove pontinhos isolados;
3. **área mínima** $A_{\min}$: descarta objetos menores (a definir com os especialistas).

**Grãos (Task 2):** classificar pixels não separa grãos que se tocam. O plano:

- classes **grão / borda / poro / matriz**, com a borda anotada como linhas finas;
- atributos do **XPL em vários ângulos**: grãos vizinhos se extinguem (ficam escuros) em
  ângulos diferentes; máximo, mínimo, amplitude e desvio ao longo dos ângulos resumem
  isso;
- **watershed marcado:** a imagem de "probabilidade de borda" vira um relevo; sementes
  nos centros dos grãos (máximos da transformada de distância) "inundam" o relevo até se
  encontrarem nas bordas → um rótulo por grão;
- depois, por grão: área, diâmetro, arredondamento → granulometria e seleção.

---

## Glossário

| Termo | Significado |
|---|---|
| Lâmina delgada | fatia de rocha de ~30 µm em vidro, vista ao microscópio |
| PPL / XPL | luz plano-polarizada / polarizadores cruzados |
| Resina epóxi azul | preenche os poros para ficarem visíveis |
| Patch | recorte quadrado da imagem da lâmina (aqui, 4096 px) |
| Atributo (*feature*) | número que descreve um pixel (cor, filtro…) |
| Rabisco (*scribble*) | anotação esparsa: alguns traços por classe |
| Máscara densa | todo pixel rotulado |
| Pseudo-rótulo | rótulo gerado por um método, não por pessoa |
| Baseline | método de referência a ser superado (o `rockface`) |
| Sobreajuste | modelo decora o treino e generaliza mal |
| Fold | uma das partes da validação cruzada |
| IoU / Dice | medidas de sobreposição entre máscaras |
| Watershed | algoritmo que separa objetos que se tocam |

## Leituras recomendadas

Para começar (nível graduação):

- scikit-learn — *User Guide*: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html),
  [Ensembles (Random Forest, Gradient Boosting)](https://scikit-learn.org/stable/modules/ensemble.html),
  [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- scikit-image — [Trainable segmentation using local features and random forests](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html)
  (é praticamente o nosso método em 30 linhas).
- ilastik — [Pixel Classification workflow](https://www.ilastik.org/documentation/pixelclassification/pixelclassification).
- napari — [tutorial de anotação com Labels](https://napari.org/stable/howtos/layers/labels.html).

Artigos-chave (referências completas na Task 1.1):

- Breiman (2001) — Random Forests.
- Berg et al. (2019) — ilastik.
- Ronneberger et al. (2015) — U-Net.
- Rubo et al. (2019) — ML em lâminas petrográficas (porosidade e mineralogia).
- Saxena et al. (2021) — deep learning em lâminas de arenito.
- Basso et al. (2025) — CNNs em carbonatos do pré-sal (generalização entre poços).
