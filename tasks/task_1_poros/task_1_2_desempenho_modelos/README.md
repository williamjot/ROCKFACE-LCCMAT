# Task 1.2 — Desempenho dos modelos de segmentação de poros 🔄

Implementa a shortlist da Task 1.1 (Seção 5) sobre os 18 patches `patch_y*_x*_c0.png`
(4096 × 4096, canal 0 = luz normal, uma lâmina, passo de patching 3800 px).

| Prioridade | Candidato | Estado |
|---|---|---|
| 0 | Limiar HSV do `rockface` (baseline) | ✅ reprodução fiel de `rockface/masks.py` |
| 1 | Random Forest pixel a pixel + 153 atributos | ✅ |
| 1 | LightGBM (mesmos atributos) | ✅ |
| — | RF só com cor (9 atributos) — ablação | ✅ |
| 2 | RF por superpixel (SLIC) | ⏳ |
| 2 | U-Net (`segmentation_models_pytorch`) | ⏳ precisa de rótulos densos |

## Scripts (rodar nesta ordem, de dentro desta pasta)

| Script | O que faz | Saídas |
|---|---|---|
| `00_explorar_dados.py` | brilho, porosidade do rockface, pixels fora da "rocha" | `results/task_1_2/exploracao_patches.csv` |
| `01_gerar_kit_anotacao.py` | divisão treino/teste e kit para os especialistas | `annotations/` |
| `02_comparar_modelos.py` | treina e compara rockface × RF × LGBM × RF-cor | `results/task_1_2/<modo>_scale<s>/` |

```bash
python 00_explorar_dados.py
python 01_gerar_kit_anotacao.py
python 02_comparar_modelos.py --scale 0.5 --folds 6
```

Opções, formatos e interpretação: [`docs/GUIA_DE_USO.md`](../../../docs/GUIA_DE_USO.md).

## Estado atual

- **Piloto concluído** (pseudo-rótulos do rockface, 18 patches, 6 folds, 40 min):
  pipeline validado de ponta a ponta. ⚠️ As métricas medem **concordância com o
  rockface**, não acerto. Análise: [`results/task_1_2/RESULTADOS_PILOTO.md`](../../../results/task_1_2/RESULTADOS_PILOTO.md).
  - IoU com o rockface: RF-cor 0,970 · RF 0,937 · LightGBM 0,930.
  - Porosidade quase igual entre métodos (≤ 0,3 p.p.); as diferenças ficam nas bordas
    e em poros pequenos/difusos, onde o RF é mais conservador que o rockface.
  - Atributos mais usados: b\* suavizado (quão azul é a vizinhança) e textura da saturação.
- **Aguardando especialistas:** rabiscos nos 12 patches de treino e correção das 6
  janelas de teste (`annotations/`), e as decisões abaixo.

## Decisões pendentes para os especialistas

1. **Microporosidade / franja ciano-clara.** Há uma franja ciano-clara de ~20–40 px ao
   longo das bordas dos poros e manchas azuis difusas dentro de grãos; o rockface
   exclui a maior parte. É poro, sólido ou terceira classe?
2. **Área mínima `A_min`** de poro (ruído × microporo) e tamanho do pixel em µm.

## Achado sobre o `rockface` (para @oi-silva)

`legacy/petrophysical_properties.py` define rocha como "qualquer canal > 5". Nestes
patches muito escuros, **7–27 % dos pixels de cada patch têm todos os canais ≤ 5**
(grãos escuros, não fundo; média de 12,3 %). Eles saem da área de rocha e **inflam a
porosidade**: a porosidade do rockface sobe de 0,3 a 3,5 pontos percentuais por patch
(ex.: `y38000_x22800`, 16,7 % → 20,2 %). Tabela: `results/task_1_2/exploracao_patches.csv`.

## Próximos passos

1. Rodar o modo especialista assim que houver rótulos.
2. Ajustar τ e `A_min` na validação; seleção de atributos por permutação.
3. RF por superpixel (SLIC) e U-Net (ResNet34) como referência de deep learning.
4. Métricas petrofísicas (com a task do rockface) e avaliação em resolução nativa.
