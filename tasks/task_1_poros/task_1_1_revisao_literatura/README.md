# Task 1.1 — Revisão de literatura ✅

Documento completo: [`docs/Task_1.1_Revisao_Literatura_Equipe_A.pdf`](../../../docs/Task_1.1_Revisao_Literatura_Equipe_A.pdf)
(23/09/2026, 40 referências).

## Resumo

**Famílias revisadas:** ML clássico (Random Forest, gradient boosting, ilastik, jPOR),
CNNs encoder-decoder (U-Net, DeepLabv3+, nnU-Net), transformers (SegFormer,
Swin-Unet) e modelos de fundação (SAM, DINOv2/v3).

**Implicações para o projeto:**

1. Poro × sólido com resina azul é "fácil" no pixel médio; a dificuldade está em
   bordas, microporosidade e artefatos → teste pequeno corrigido por especialistas.
2. Máscaras do `rockface` são pseudo-rótulos: IoU contra elas mede concordância, não acerto.
3. Avaliar também com métricas petrofísicas (porosidade, permeabilidade).
4. Dividir treino/teste por lâmina (idealmente por poço), nunca por recortes da mesma lâmina.
5. Entrada PPL + XPL é natural num classificador pixel a pixel (Task 2).
6. Risco de atalho por cor (tonalidade da resina/iluminação).
7. Lacuna: não há benchmark público de poros com resina azul com máscaras por pixel.

## Decisão (hipótese de trabalho para a Task 1.2)

**Random Forest pixel a pixel sobre banco de filtros multiescala**, treinado com
rabiscos de especialistas, comparado com o limiar do `rockface`, LightGBM (mesmos
atributos) e U-Net (ResNet34).

Critério de troca: LightGBM vira o classificador se ganhar ≥ 2 pontos de IoU (ou menor
erro de porosidade) no teste corrigido; se a U-Net superar ambos com margem relevante,
sobretudo em bordas e microporosidade, a escolha é reavaliada na Task 1.3.

## Shortlist

| Prioridade | Candidato | Implementado em |
|---|---|---|
| 0 | Limiar HSV/Otsu (`rockface`) | `laminas/baseline.py` |
| 1 | RF pixel a pixel + banco de filtros | Task 1.2 |
| 1 | LightGBM/XGBoost | Task 1.2 |
| 2 | RF por superpixel (SLIC) | ⏳ |
| 2 | U-Net (`segmentation_models_pytorch`) | ⏳ |
| — | DINOv3, SAM 2 | fora do escopo desta etapa |
