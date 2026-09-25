"""
00_explorar_dados.py
====================
Inspeção inicial dos patches: brilho por canal, fração de poro segundo o rockface e
fração de pixels que o rockface NÃO conta como rocha (todos os canais ≤ 5).

Saídas:
  results/task_1_2/exploracao_patches.csv   estatísticas por patch
  cache/task_1_2/mosaico.jpg                mosaico realçado de todos os patches (git-ignored)

Uso:
    python 00_explorar_dados.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

import laminas as L


def main():
    patches = L.list_patches()
    stats = L.load_or_compute_stats(patches, L.CACHE_DIR / "slide_color_stats.npy")
    out = L.RESULTS_DIR / "task_1_2"
    cache = L.CACHE_DIR / "task_1_2"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    rows, thumbs = [], []
    for p in patches:
        rgb = L.load_rgb(p)
        win = rgb[:L.STRIDE, :L.STRIDE]
        base = L.rockface_mask(win)
        rock = L.rockface_rock_area(win)
        mean = win[::4, ::4].reshape(-1, 3).mean(0)
        rows.append(dict(
            patch=p.name, mean_R=mean[0], mean_G=mean[1], mean_B=mean[2],
            frac_nao_rocha_rockface=1 - rock.mean(),
            porosidade_rockface_area_total=L.porosity(base),
            porosidade_rockface_area_rocha=L.porosity(base, rock),
        ))
        thumbs.append(Image.fromarray(L.enhance(rgb, stats)).resize((512, 512)))
        print("ok", p.name, flush=True)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "exploracao_patches.csv", index=False)

    cols = 6
    mosaic = Image.new("RGB", (cols * 512, -(-len(thumbs) // cols) * 512))
    for i, t in enumerate(thumbs):
        mosaic.paste(t, ((i % cols) * 512, (i // cols) * 512))
    mosaic.save(cache / "mosaico.jpg", quality=85)

    print(df.to_string(index=False))
    print(f"\nMédia de pixels fora da 'rocha' do rockface: {100 * df.frac_nao_rocha_rockface.mean():.1f} %")


if __name__ == "__main__":
    main()
