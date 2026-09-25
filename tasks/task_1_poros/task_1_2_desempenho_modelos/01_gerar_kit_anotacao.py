"""
01_gerar_kit_anotacao.py
========================
Gera o material de anotação para os especialistas (Task 1.1, Seção 6.5).

Divisão fixa (uma só lâmina → grupo = patch):
  - patches de TESTE (6): recebem só uma janela densa corrigida, nunca rabiscos;
  - patches de TREINO (12): recebem rabiscos esparsos.

Saídas em ``annotations/``:
  images/patch_<nome>_enhanced.png   imagem realçada (normalização da lâmina + gama)
  scribbles/                         (vazio) rabiscos dos patches de treino
  test_masks/patch_<nome>_mask.png   PRÉ-RÓTULO do rockface só na janela de teste, para CORRIGIR
  test_windows/…_window.png          recorte realçado da janela + pré-rótulo (conveniência)
  split.json                         patches de treino/teste, janelas e ``test_masks_corrected``

Uso:
    python 01_gerar_kit_anotacao.py [--n-test 6] [--win 1024] [--seed 42]
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from PIL import Image

import laminas as L


def best_window(base: np.ndarray, win: int, step: int = 256) -> tuple[int, int]:
    """Janela (dentro da área sem sobreposição) com mais borda de poro — casos difíceis."""
    edges = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    integ = cv2.integral(edges)
    best, pos = -1.0, (0, 0)
    for y in range(0, L.STRIDE - win + 1, step):
        for x in range(0, L.STRIDE - win + 1, step):
            s = integ[y + win, x + win] - integ[y, x + win] - integ[y + win, x] + integ[y, x]
            if s > best:
                best, pos = s, (y, x)
    return pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-test", type=int, default=6, help="número de patches reservados para teste")
    ap.add_argument("--win", type=int, default=1024, help="lado da janela densa de teste (px, resolução nativa)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann = L.ANN_DIR
    split_path = ann / "split.json"
    if split_path.exists() and json.load(open(split_path, encoding="utf-8")).get("test_masks_corrected"):
        raise SystemExit(f"{split_path} já tem máscaras corrigidas por especialistas; não vou sobrescrever.")
    for sub in ("images", "scribbles", "test_masks", "test_windows"):
        (ann / sub).mkdir(parents=True, exist_ok=True)

    patches = L.list_patches()
    stats = L.load_or_compute_stats(patches, L.CACHE_DIR / "slide_color_stats.npy")

    rng = np.random.default_rng(args.seed)
    test = sorted(rng.choice([p.name for p in patches], args.n_test, replace=False).tolist())
    split = {"train": [p.name for p in patches if p.name not in test], "test": test, "windows": {}}

    for p in patches:
        rgb = L.load_rgb(p)
        enh = L.enhance(rgb, stats)
        Image.fromarray(enh).save(ann / "images" / f"patch_{p.name}_enhanced.png")
        if p.name in test:
            base = L.rockface_mask(rgb)
            y, x = best_window(base, args.win)
            w = args.win
            lab = np.zeros(base.shape, np.uint8)
            lab[y:y + w, x:x + w] = np.where(base[y:y + w, x:x + w], L.PORE, L.SOLID)
            L.save_labels(lab, ann / "test_masks" / f"patch_{p.name}_mask.png")
            Image.fromarray(enh[y:y + w, x:x + w]).save(ann / "test_windows" / f"patch_{p.name}_y{y}_x{x}_window.png")
            L.save_labels(lab[y:y + w, x:x + w], ann / "test_windows" / f"patch_{p.name}_y{y}_x{x}_prelabel.png")
            split["windows"][p.name] = {"y": y, "x": x, "size": w}
        print("ok", p.name, "(teste)" if p.name in test else "(treino)", flush=True)

    # Mude para true SÓ depois que os especialistas corrigirem todas as janelas de test_masks/.
    split["test_masks_corrected"] = False
    json.dump(split, open(split_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Kit gerado em {ann}")


if __name__ == "__main__":
    main()
