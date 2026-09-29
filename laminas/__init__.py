"""
laminas: funções compartilhadas pelos scripts das tasks.

Uso nos scripts:
    import laminas as L
    arquivos = L.listar_patches()
"""

from .atributos import N_ATRIBUTOS_DE_COR, calcular_atributos, nomes_dos_atributos
from .baseline import area_de_rocha_rockface, mascara_rockface
from .config import (PASSO_PATCH, PASTA_ANOTACOES, PASTA_CACHE, PASTA_DADOS,
                     PASTA_PROJETO, PASTA_RESULTADOS, PORO, SOLIDO)
from .imagens import (carregar_rgb, limites_de_cor, listar_patches, nome_do_patch,
                      normalizar, realcar, redimensionar_mascara)
from .metricas import f1_de_borda, metricas_de_segmentacao, porosidade
from .modelos import criar_modelos, mascara_final, prever_probabilidade
from .rotulos import ler_rotulos, pseudo_rabiscos, salvar_rotulos
