"""
Caminhos das pastas e constantes do projeto.

Para usar outra pasta de dados, basta trocar PASTA_DADOS aqui.
"""

from pathlib import Path

# Pasta raiz do repositório (a pasta acima de laminas/)
PASTA_PROJETO = Path(__file__).resolve().parent.parent

PASTA_DADOS = PASTA_PROJETO / "data" / "patches"       # imagens patch_y*_x*_c0.png
PASTA_ANOTACOES = PASTA_PROJETO / "annotations"        # kit e rótulos dos especialistas
PASTA_RESULTADOS = PASTA_PROJETO / "results"           # tabelas (vão para o git)
PASTA_CACHE = PASTA_PROJETO / "cache"                  # arquivos intermediários (não vão para o git)

# O rockface corta a lâmina em patches de 4096 px, andando de 3800 em 3800 px.
# Por isso patches vizinhos se sobrepõem 296 px. Para não contar a mesma área
# duas vezes, só usamos a janela [0:3800, 0:3800] de cada patch.
PASSO_PATCH = 3800

# Rótulos: 0 = sem rótulo, 1 = poro, 2 = sólido
PORO = 1
SOLIDO = 2

# Banco de atributos (Tabela 3 da Task 1.1)
ESCALAS = [0.7, 1.0, 1.6, 3.5, 5.0, 10.0]          # sigma dos filtros, em pixels
CANAIS_DE_COR = ["R", "G", "B", "H", "S", "V", "L", "a", "b"]
CANAIS_FILTRADOS = ["L", "b", "S"]                  # canais que passam pelos filtros
FILTROS = ["gauss", "gradmag", "dog", "log", "hess1", "hess2", "st1", "st2"]
