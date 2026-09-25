# Task 2 — Segmentação de grãos ⏳

Reaproveitar o método escolhido na Task 1 para segmentar grãos (Task 1.1, Seção 6.4).

Fluxo proposto:

1. Banco de atributos da Task 1 aplicado à PPL e a cada ângulo de XPL, mais atributos
   entre ângulos (máximo, mínimo, amplitude, desvio) que resumem a extinção.
2. Concatenação PPL + XPL no vetor de cada pixel.
3. Random Forest (mesma configuração, retreinado) com as classes grão / borda / poro / matriz.
4. Watershed marcado: sementes nos máximos da transformada de distância do interior,
   relevo = probabilidade de borda → instâncias de grão.
5. Saídas: granulometria, seleção, arredondamento; poros da Task 1 no mesmo mapa.

Anotação: rabiscos de grão, borda (linhas de 1–3 px nos contatos), poro e matriz sobre
a pilha PPL + XPL; teste com poucas janelas delineadas grão a grão.

Dependências: patches XPL (canais `c1…cN` do rockface) — ainda não disponíveis aqui.
