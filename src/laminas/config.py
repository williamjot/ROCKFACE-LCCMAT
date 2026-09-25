"""
config.py
=========
Caminhos e constantes do projeto.

Os caminhos podem ser trocados por variáveis de ambiente, sem editar código:

- ``LAMINAS_DATA``        pasta com os patches ``patch_y*_x*_c0.png`` (padrão: ``data/patches``)
- ``LAMINAS_ANNOTATIONS`` pasta do kit de anotação (padrão: ``annotations``)
- ``LAMINAS_RESULTS``     pasta de resultados (padrão: ``results``)
- ``LAMINAS_CACHE``       arquivos intermediários (padrão: ``cache``)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("LAMINAS_DATA", REPO_ROOT / "data" / "patches"))
ANN_DIR = Path(os.environ.get("LAMINAS_ANNOTATIONS", REPO_ROOT / "annotations"))
RESULTS_DIR = Path(os.environ.get("LAMINAS_RESULTS", REPO_ROOT / "results"))
CACHE_DIR = Path(os.environ.get("LAMINAS_CACHE", REPO_ROOT / "cache"))  # arquivos intermediários (git-ignored)

# Passo do patching do rockface: patches de 4096 px com passo de 3800 px.
# Para porosidade sem dupla contagem, só a janela [0:STRIDE, 0:STRIDE] conta.
STRIDE = 3800

# Banco de atributos (Tabela 3 da Task 1.1).
SIGMAS = (0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
FILTER_NAMES = ("gauss", "gradmag", "dog", "log", "hess1", "hess2", "st1", "st2")
COLOR_NAMES = ("R", "G", "B", "H", "S", "V", "L", "a", "b")
FILTERED_CHANNELS = ("L", "b", "S")

# Classes dos rótulos (0 = sem rótulo).
PORE, SOLID = 1, 2
