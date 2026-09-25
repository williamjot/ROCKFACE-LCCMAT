# annotations/ — kit de anotação dos especialistas

Gerado por `tasks/task_1_poros/task_1_2_desempenho_modelos/01_gerar_kit_anotacao.py`.
Instruções completas: [`docs/GUIA_DE_USO.md`](../docs/GUIA_DE_USO.md), seção 6.

| Pasta/arquivo | No git? | Conteúdo |
|---|---|---|
| `split.json` | ✅ | patches de treino (12) e teste (6), janelas de teste, `test_masks_corrected` |
| `images/` | ❌ | patches realçados para anotar |
| `test_windows/` | ❌ | recortes realçados das janelas de teste + pré-rótulo |
| `test_masks/` | ❌ | pré-rótulo do rockface nas janelas de teste — **corrigir** |
| `scribbles/` | ❌ | rabiscos dos especialistas (`patch_<nome>_scribbles.png`) |

As pastas marcadas ❌ são derivadas das lâminas; regenere-as com o script 01 ou
peça os rótulos à equipe. Formato: PNG 4096², 0 = sem rótulo, 1 = poro, 2 = sólido
(ou vermelho = poro, verde = sólido).

Quando todas as janelas de `test_masks/` estiverem revisadas, mude
`"test_masks_corrected"` para `true` em `split.json`; o script 02 passa então para o
modo especialista.
