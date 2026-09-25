# Atividades de formação

| Atividade | Descrição | Estado |
|---|---|---|
| [Atividade 1](atividade_1_pytorch_cat_not_cat) | Primeiro contato com PyTorch — Cat vs. Not Cat | 🔄 etapa 1 feita |

## Atividade 1 — PyTorch Cat vs. Not Cat

Notebook: [`atividade_1_pytorch_cat_not_cat/atividade_1_cat_not_cat.ipynb`](atividade_1_pytorch_cat_not_cat/atividade_1_cat_not_cat.ipynb)

Rede simples (12288 → 16 → ReLU → 1 logit), `BCEWithLogitsLoss`, Adam (lr 0,001),
10 épocas, batch 32; depois uma única alteração e comparação com o baseline.

Dataset `cat_not_cat.csv` (258 imagens 64×64×3 achatadas + label; 104 gatos, 154
não-gatos): colocar na pasta da atividade. **Não versionado**, conforme a instrução
da atividade.

Dependências extras: `pip install torch matplotlib pandas`.
