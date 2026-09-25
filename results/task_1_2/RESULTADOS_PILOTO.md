# Task 1.2 — Resultados do piloto (25/09/2026)

> ⚠️ **Leia primeiro.** Sem anotações de especialistas, os modelos foram treinados com
> pseudo-rótulos tirados do próprio `rockface` e avaliados **contra o `rockface`**.
> Os números abaixo medem **concordância com o `rockface`, não acerto**. Eles não
> servem para escolher o modelo (isso é o modo especialista).

## Configuração

- 18 patches de uma lâmina, `--scale 0.5` (2048 × 2048), janela sem sobreposição.
- Validação cruzada em 6 folds agrupados por patch (3 patches de teste por fold).
- Treino: 20 mil px/classe/patch do miolo (erosão 6 px) das regiões do `rockface`;
  600 mil px por fold, balanceados.
- Pós-processamento: τ = 0,5, abertura de raio 1 px.
- Tempo total: 40 min (16 threads, sem GPU).

Arquivos: [`piloto_scale0.5/`](piloto_scale0.5) — `metrics_per_patch.csv`,
`metrics_summary.csv`, `rf_feature_importance.csv`, `run_info.json`.

## Concordância com o `rockface` (média ± dp, 18 patches)

| Método | IoU | Dice | Precisão | Revocação | F1 de borda | Porosidade (%) |
|---|---|---|---|---|---|---|
| rockface | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 8,47 ± 3,03 |
| RF-cor (9 atributos) | 0,970 ± 0,009 | 0,985 | 0,982 | 0,987 | 0,973 ± 0,018 | 8,52 ± 3,06 |
| RF (153 atributos) | 0,937 ± 0,016 | 0,967 | 0,966 | 0,969 | 0,821 ± 0,052 | 8,52 ± 3,11 |
| LightGBM (153 atributos) | 0,930 ± 0,018 | 0,963 | 0,959 | 0,968 | 0,801 ± 0,043 | 8,56 ± 3,08 |

## O que o piloto mostra

1. **O pipeline funciona de ponta a ponta** — baseline, 153 atributos, três
   classificadores, validação cruzada, métricas e figuras — e está pronto para o modo
   especialista.
2. **A porosidade praticamente não muda entre métodos** (diferença ≤ 0,3 p.p. por
   patch). Nestes patches, o grosso da porosidade é o "miolo fácil" dos poros grandes,
   em que todos concordam — como previsto na Task 1.1 (Seção 4, item 1).
3. **As diferenças estão nas bordas e nos poros pequenos/difusos.** O F1 de borda cai
   para ~0,8 com o banco de filtros. Nas sobreposições (`cache/…/overlays`), quase toda
   a discordância é "só `rockface`" (ciano): bordas finas e manchas azuis difusas dentro
   de grãos (possível microporosidade), que o RF marca como sólido. O RF é **mais
   conservador**, o que é esperado: ele só viu exemplos do miolo das regiões.
4. **RF-cor concorda mais com o `rockface` que o RF completo.** Também esperado: o
   `rockface` é um limiar de cor, e o modelo só de cor o imita melhor. Isso *não*
   indica que o banco de filtros é pior — só o teste corrigido dirá se o contexto
   espacial acerta mais nas bordas.
5. **Atributos mais usados pelo RF:** b\* suavizado em várias escalas (`b_gauss_s10.0`,
   `s3.5`, `s1.6`, `s5.0`) e o 2º autovalor do tensor de estrutura da saturação
   (`S_st2_s5.0`, `s10.0`). Ou seja: "quão azul é a vizinhança" + textura da saturação.
6. **Nenhum método captura a franja ciano-clara** nas bordas dos poros. Precisa de
   decisão dos especialistas (poro, sólido ou terceira classe).

## Próximo passo

Rótulos dos especialistas (rabiscos nos 12 patches de treino + correção das 6 janelas
de teste) → `02_comparar_modelos.py` em modo especialista. Ver
[`docs/GUIA_DE_USO.md`](../../docs/GUIA_DE_USO.md), seção 6.
