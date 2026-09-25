"""
02_comparar_modelos.py
======================
Compara os candidatos da shortlist da Task 1.1 (Seção 5):

    0. rockface  — limiar HSV (baseline obrigatório)
    1. RF        — Random Forest pixel a pixel sobre o banco de 153 atributos
    1. LGBM      — LightGBM com os mesmos atributos
    -  RF-cor    — RF só com os 9 atributos de cor (ablação: o banco de filtros ajuda?)

Dois modos, escolhidos automaticamente:

- PILOTO (sem anotações): treino com pseudo-rabiscos do miolo das regiões do rockface;
  validação cruzada agrupada por patch contra o próprio rockface. As métricas medem
  CONCORDÂNCIA com o rockface, não acerto.
- ESPECIALISTA (``annotations/split.json`` com ``"test_masks_corrected": true``): treino
  só com rabiscos em ``annotations/scribbles/`` nos patches de treino; avaliação nas
  janelas densas corrigidas dos patches de teste.

Saídas em ``results/task_1_2/<modo>_scale<s>/``: métricas por patch e resumo (CSV),
importância de atributos, ``run_info.json``; em ``cache/`` ficam probabilidades e
sobreposições (git-ignored, derivadas das imagens).

Uso:
    python 02_comparar_modelos.py --scale 0.5 --folds 6
    python 02_comparar_modelos.py --scale 0.25 --folds 3 --limit 3   # teste rápido
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

import laminas as L


def make_models(seed: int) -> dict:
    """Modelos da comparação: nome → (estimador, colunas de atributos ou None = todas)."""
    return {
        # class_weight="balanced" (e não "balanced_subsample"): o piloto já amostra as
        # classes balanceadas, e "balanced_subsample" falhou de forma intermitente no
        # scikit-learn 1.9 / Python 3.14 ("classes should have valid labels that are in y").
        "RF": (RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=5,
                                      class_weight="balanced", n_jobs=-1, random_state=seed), None),
        "LGBM": (lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                    class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1), None),
        "RF-cor": (RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=5,
                                          class_weight="balanced", n_jobs=-1, random_state=seed),
                   L.color_indices()),
    }


def expert_split() -> dict | None:
    """Divisão fixa treino/teste do kit de anotação, se as máscaras de teste já foram corrigidas."""
    path = L.ANN_DIR / "split.json"
    if not path.exists():
        return None
    split = json.load(open(path, encoding="utf-8"))
    return split if split.get("test_masks_corrected") else None


def prepare_patch(patch: L.Patch, stats: np.ndarray, scale: float, rng: np.random.Generator, use_expert_ref: bool):
    """Carrega o patch e calcula baseline, atributos, rótulos e referência na resolução de trabalho."""
    rgb_native = L.load_rgb(patch)
    base_native = L.rockface_mask(rgb_native)
    rock_native = L.rockface_rock_area(rgb_native)

    rgb = L.resize(rgb_native, scale, cv2.INTER_AREA)
    shape = rgb.shape[:2]
    base = L.resize_mask(base_native, shape) > 0
    rock = L.resize_mask(rock_native, shape) > 0
    feats = L.compute_features(L.normalize(rgb, stats))

    scrib_path = L.ANN_DIR / "scribbles" / f"patch_{patch.name}_scribbles.png"
    if scrib_path.exists():
        labels, source = L.load_labels(scrib_path, shape), "especialista"
    else:
        labels, source = L.bootstrap_scribbles(base, rng), "pseudo (rockface)"

    ref_path = L.ANN_DIR / "test_masks" / f"patch_{patch.name}_mask.png"
    ref = L.load_labels(ref_path, shape) if (use_expert_ref and ref_path.exists()) else None
    return dict(rgb=rgb, base=base, rock=rock, feats=feats, labels=labels, label_source=source, ref=ref)


def sample_training(feats: np.ndarray, labels: np.ndarray):
    idx = np.flatnonzero(labels)
    X = feats.reshape(-1, feats.shape[-1])[idx]
    y = (labels.flat[idx] == L.PORE).astype(np.uint8)
    return X, y


def predict_map(model, feats: np.ndarray, cols, chunk: int = 1_000_000) -> np.ndarray:
    flat = feats.reshape(-1, feats.shape[-1])
    if cols is not None:
        flat = flat[:, cols]
    out = np.empty(flat.shape[0], np.float32)
    for i in range(0, flat.shape[0], chunk):
        out[i:i + chunk] = model.predict_proba(flat[i:i + chunk])[:, 1]
    return out.reshape(feats.shape[:2])


def crop_to(valid: np.ndarray, *arrays):
    """Recorta os arrays para a caixa envolvente da região rotulada (janela de teste)."""
    ys, xs = np.nonzero(valid)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return [a[box] for a in arrays]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=0.5, help="fator de reamostragem (1.0 = nativo)")
    ap.add_argument("--folds", type=int, default=6, help="folds da validação cruzada do piloto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="usar só os N primeiros patches (teste rápido)")
    args = ap.parse_args()

    split = expert_split()
    mode = "especialista" if split else "piloto"
    tag = f"{mode}_scale{args.scale}" + (f"_limit{args.limit}" if args.limit else "")
    out = L.RESULTS_DIR / "task_1_2" / tag
    cache = L.CACHE_DIR / "task_1_2" / tag
    for d in (out, cache / "overlays", cache / "samples", cache / "prob"):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    patches = L.list_patches()
    stats = L.load_or_compute_stats(patches, L.CACHE_DIR / "slide_color_stats.npy")
    if args.limit:
        patches = patches[:args.limit]

    if split:
        # Um único "fold": treina nos patches de treino, testa nas janelas corrigidas.
        fold_of = {p.name: (0 if p.name in split["test"] else -1) for p in patches}
        args.folds = 1
        print("Modo ESPECIALISTA: divisão fixa de annotations/split.json", flush=True)
    else:
        # Folds agrupados por patch (numa só lâmina, o patch é o melhor proxy de grupo).
        order = rng.permutation(len(patches))
        fold_of = {patches[i].name: k % args.folds for k, i in enumerate(order)}
        print("Modo PILOTO: pseudo-rótulos do rockface — métricas = concordância com o rockface", flush=True)

    # Passo 1: amostras de treino de cada patch (atributos só nos pixels rotulados).
    t0 = time.time()
    train, label_sources = {}, {}
    for p in patches:
        cfile = cache / "samples" / f"{p.name}.npz"
        has_scribbles = (L.ANN_DIR / "scribbles" / f"patch_{p.name}_scribbles.png").exists()
        if cfile.exists() and not has_scribbles:
            z = np.load(cfile)
            train[p.name], label_sources[p.name] = (z["X"], z["y"]), str(z["source"])
            continue
        d = prepare_patch(p, stats, args.scale, rng, use_expert_ref=False)
        train[p.name] = sample_training(d["feats"], d["labels"])
        label_sources[p.name] = d["label_source"]
        # Só guarda cache de pseudo-rótulos; rabiscos de especialista sempre são relidos.
        if d["label_source"] != "especialista":
            np.savez(cfile, X=train[p.name][0], y=train[p.name][1], source=d["label_source"])
        print(f"[amostras] {p.name}: {len(train[p.name][1])} px rotulados "
              f"({d['label_source']}) — {time.time() - t0:.0f}s", flush=True)

    names = L.feature_names()
    rows, importances = [], []
    edge_crop = int(round(L.STRIDE * args.scale))

    # Passo 2: para cada fold, treina nos outros patches e prevê os patches do fold.
    for k in range(args.folds):
        test_names = [p.name for p in patches if fold_of[p.name] == k]
        if not test_names:
            continue
        train_names = [n for n in train if n not in test_names]
        if split:
            # Não misturar pseudo-rótulos com rabiscos de especialistas.
            train_names = [n for n in train_names if label_sources[n] == "especialista"]
            if not train_names:
                raise SystemExit("Modo especialista sem rabiscos em annotations/scribbles/.")
        X = np.concatenate([train[n][0] for n in train_names])
        y = np.concatenate([train[n][1] for n in train_names])
        print(f"[fold {k}] treino: {len(y)} px, poro={int(y.sum())}, sólido={int((y == 0).sum())}", flush=True)

        models = make_models(args.seed)
        for mname, (model, cols) in models.items():
            t = time.time()
            model.fit(X if cols is None else X[:, cols], y)
            print(f"[fold {k}] {mname} treinado ({time.time() - t:.0f}s)", flush=True)
            if mname == "RF":
                importances.append(model.feature_importances_)

        for p in (p for p in patches if p.name in test_names):
            d = prepare_patch(p, stats, args.scale, rng, use_expert_ref=bool(split))
            win = (slice(0, edge_crop), slice(0, edge_crop))   # janela sem sobreposição
            base, rock = d["base"][win], d["rock"][win]
            if d["ref"] is not None:
                ref, valid, ref_kind = d["ref"][win] == L.PORE, d["ref"][win] > 0, "especialista"
            else:
                ref, valid, ref_kind = base, None, "rockface"

            preds = {"rockface": base}
            for mname, (model, cols) in models.items():
                t = time.time()
                prob = predict_map(model, d["feats"], cols)[win]
                preds[mname] = L.postprocess(prob, tau=0.5, open_px=1)
                print(f"[fold {k}] {p.name} {mname} previsto ({time.time() - t:.0f}s)", flush=True)
                if mname == "RF":
                    np.save(cache / "prob" / f"prob_RF_{p.name}.npy", prob.astype(np.float16))

            for mname, pred in preds.items():
                m = L.seg_metrics(pred, ref, valid)
                bf1 = (L.boundary_f1(pred, ref, tol=2) if valid is None
                       else L.boundary_f1(*crop_to(valid, pred, ref), tol=2))
                rows.append(dict(
                    patch=p.name, fold=k, method=mname, reference=ref_kind, **m, boundary_f1=bf1,
                    porosity_total=L.porosity(pred),
                    porosity_rockface_rock=L.porosity(pred, rock),
                ))

            norm = L.normalize(d["rgb"], stats)[win]
            Image.fromarray(L.disagreement_overlay(norm, preds["RF"], base)).save(
                cache / "overlays" / f"{p.name}_RF_vs_rockface.jpg", quality=90)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_per_patch.csv", index=False)
    cols = ["iou", "dice", "precision", "recall", "boundary_f1", "porosity_total", "porosity_rockface_rock"]
    summary = df.groupby("method")[cols].agg(["mean", "std"])
    summary.to_csv(out / "metrics_summary.csv")
    imp = pd.Series(np.mean(importances, axis=0), index=names).sort_values(ascending=False)
    imp.to_csv(out / "rf_feature_importance.csv", header=["gini_importance"])

    rock_frac = {p.name: float(L.rockface_rock_area(L.load_rgb(p)[:L.STRIDE, :L.STRIDE]).mean()) for p in patches}
    json.dump(dict(mode=mode, scale=args.scale, folds=args.folds, seed=args.seed,
                   fold_of_patch=fold_of, label_sources=label_sources,
                   rockface_rock_fraction=rock_frac, slide_color_stats=stats.tolist(),
                   minutes=round((time.time() - t0) / 60, 1)),
              open(out / "run_info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    pd.set_option("display.width", 200)
    print("\n=== Resumo (média ± dp entre patches de teste) ===")
    print(summary.round(3).to_string())
    print("\n=== Top 15 atributos (RF, Gini) ===")
    print(imp.head(15).round(4).to_string())
    print(f"\nTempo total: {(time.time() - t0) / 60:.1f} min. Resultados em {out}")


if __name__ == "__main__":
    main()
