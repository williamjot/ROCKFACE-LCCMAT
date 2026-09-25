"""laminas — segmentação de poros e grãos em lâminas petrográficas (LCCMat-UnB, Equipe A)."""

from .atributos import color_indices, compute_features, compute_features_tiled, feature_names
from .baseline import rockface_mask, rockface_rock_area
from .config import ANN_DIR, CACHE_DIR, DATA_DIR, PORE, RESULTS_DIR, SOLID, STRIDE
from .io import Patch, list_patches, load_rgb, resize, resize_mask
from .metricas import boundary_f1, porosity, seg_metrics
from .normalizacao import enhance, load_or_compute_stats, normalize, slide_color_stats
from .posprocessamento import disagreement_overlay, postprocess
from .rotulos import bootstrap_scribbles, load_labels, save_labels

__version__ = "0.1.0"
