# Task 1.3 — Seleção do(s) modelo(s) para aplicação em escala ⏳

Determinar o(s) melhor(es) modelo(s) candidato(s), incluindo parâmetros gerais da
arquitetura e do dataset, para aplicação em escala no projeto.

Entrada: comparação da Task 1.2 no modo especialista (métricas por pixel no teste
corrigido + métricas petrofísicas).

Critério definido na Task 1.1 (Seção 6):

- LightGBM substitui o RF se ganhar ≥ 2 pontos de IoU ou tiver erro de porosidade menor;
- se a U-Net superar ambos com margem relevante nas lâminas/poços de teste, sobretudo
  em bordas e microporosidade, pesar o ganho contra o custo de rótulos densos e GPU.

Planejado: aplicação em lâminas inteiras (processamento em blocos com sobreposição),
validação cruzada por lâmina/poço, custo por lâmina.
