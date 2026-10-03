"""FASE 38 (P2, P3, P4 y P5) — Bloques léxicos, control lexicon-only, redundancia y ablación.

Ejecuta las cuatro fases de construcción de señal, en el orden que fija el brief (§34):

* **P2/P3/P4**: construye el bloque de agregados de cada familia y lo evaluates **aislado**
  (control lexicon-only, §14) con LogReg y LinearSVC bajo corrected CV. El objetivo del control no
  es superar a V12, sino determinar si la señal existe y cuánto aporta por sí sola.
* **P5**: cribado de redundancia (§12) contra las variables que ya existen y ablación por
  subcategoría (§16), de modo que solo lo que sobreviva llegue a la fusión de P6.

Las features son **independientes de las etiquetas** (§13): la matriz se calcula una vez para todo
el corpus y no se recalcula in-fold, a diferencia del canal supervisado de entidades.

Salida: ``features/lexicon_features_<familia>.npy``, ``experiments/phase38_bloques_lexicos.json``,
``reports/age_lexicon_ablation.csv``, ``reports/age_lexicon_redundancy.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_ponderacion import build_features  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src import lexicons as lx  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "phase38_bloques_lexicos.json"
ABLACION = cfg.REPORTS / "age_lexicon_ablation.csv"
REDUNDANCIA = cfg.REPORTS / "age_lexicon_redundancy.csv"
ETIQUETAS = np.arange(len(cfg.CLASSES))
#: Variables existentes contra las que se mide la redundancia (§12).
VARIABLES_EXISTENTES = ["n_chars", "n_words", "n_unique_words", "ttr", "stopword_ratio",
                        "content_ratio", "n_mention", "n_hashtag", "n_url"]
UMBRAL_CORRELACION = 0.80
#: Filas usadas para el contraste contra la matriz de V12 (Spearman sobre 169k x 279 por columna
#: sería demasiado lento; con 20.000 filas la estimación de rho es estable para |rho| > 0,2).
FILAS_CONTRASTE = 20000


def main() -> int:
    """Construye los tres bloques léxicos y ejecuta control, redundancia y ablación."""
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    textos = train["text"]
    y = train[cfg.TARGET].astype(str).to_numpy()
    indices = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    lexicon = lx.carga_lexicones()
    base, _, _, _, _ = build_features()
    base = np.concatenate([base, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    matriz_contraste = base
    hashes = {clave: lx.sha256_de(ruta) for clave, ruta in sorted(lexicon["rutas"].items())}

    resumen_familias = {}
    filas_redundancia, filas_ablacion = [], []
    for familia in lx.FAMILIAS:
        matriz, nombres = lx.matriz_lexica(textos, lexicon, familias=(familia,))
        np.save(cfg.FEATURES / f"lexicon_features_{familia}.npy", matriz)
        (cfg.FEATURES / f"lexicon_features_{familia}_nombres.json").write_text(
            json.dumps(nombres, ensure_ascii=False), encoding="utf-8")
        no_vacias = int(sum(matriz[:, i].std() > 0 for i in range(matriz.shape[1])))
        print(f"[{familia}] bloque {matriz.shape} ({no_vacias} columnas con varianza) | "
              f"media ratio = {matriz[:, 1::3].mean():.4f}", flush=True)

        # --- control lexicon-only (§14)
        controles = {}
        for nombre, constructor in (
                ("logreg", lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)),
                ("linearsvc", lambda: LinearSVC(C=0.2, random_state=cfg.SEED))):
            por_fold = []
            for fold in folds:
                entrenamiento = np.concatenate([f["val_idx"] for f in folds
                                               if f["fold"] != fold["fold"]])
                estimador = constructor()
                estimador.fit(matriz[entrenamiento], indices[entrenamiento])
                por_fold.append(f1_score(indices[fold["val_idx"]],
                                         estimador.predict(matriz[fold["val_idx"]]),
                                         average="macro", labels=ETIQUETAS))
            controles[nombre] = {"macro_f1": round(float(np.mean(por_fold)), 4),
                                 "std": round(float(np.std(por_fold)), 4),
                                 "per_fold": [round(float(v), 4) for v in por_fold]}
            print(f"  control lexicon-only {nombre}: {controles[nombre]['macro_f1']} "
                  f"± {controles[nombre]['std']}", flush=True)

        # --- redundancia (§12)
        # Dos contrastes: (a) contra variables densas ya presentes, (b) contra las 279 features de
        # V12, que es donde de verdad puede estar la redundancia (el bloque base ya codifica
        # identidad de palabra con char_wb 3-5, word TF-IDF, margenes NB-SVM y columnas
        # supervisadas). Sin (b) el cribado no certificaria nada sobre novedad frente a V12.
        denso = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
        # Spearman vectorizado: se rankean una sola vez las columnas de la matriz base y se
        # correlacionan con Pearson sobre los rangos (equivalente exacto a Spearman, pero evita
        # 20.000 llamadas a scipy). Ojo: pasar una matriz 2D a spearmanr devuelve la matriz de
        # correlaciones entre variables, no una correlación por columna.
        base_contraste = matriz_contraste
        generador = np.random.default_rng(cfg.SEED)
        filas_contraste = generador.choice(len(matriz), size=FILAS_CONTRASTE, replace=False)
        rangos_base = rankdata(base_contraste[filas_contraste].astype(float), axis=0)
        rangos_base = (rangos_base - rangos_base.mean(axis=0)) / np.where(
            rangos_base.std(axis=0) < 1e-12, 1.0, rangos_base.std(axis=0))
        for indice_columna, nombre in enumerate(nombres):
            correlaciones = {}
            for variable in VARIABLES_EXISTENTES:
                if variable in denso.columns:
                    coeficiente = spearmanr(matriz[:, indice_columna],
                                            denso[variable].to_numpy()).statistic
                    if not np.isnan(coeficiente):
                        correlaciones[variable] = round(float(coeficiente), 3)
            columna = matriz[filas_contraste, indice_columna].astype(float)
            if np.std(columna) > 0:
                rangos_columna = rankdata(columna)
                rangos_columna = (rangos_columna - rangos_columna.mean()) / (
                    rangos_columna.std() if rangos_columna.std() > 1e-12 else 1.0)
                correlaciones_v12 = (rangos_base.T @ rangos_columna) / (len(rangos_columna) - 1)
                maximo_base = float(np.nanmax(np.abs(correlaciones_v12)))
                columna_base = int(np.nanargmax(np.abs(correlaciones_v12)))
            else:
                maximo_base, columna_base = 0.0, 0
            maximo = max((abs(v) for v in correlaciones.values()), default=0.0)
            filas_redundancia.append({
                "familia": familia, "feature": nombre,
                "varianza": round(float(matriz[:, indice_columna].std()), 5),
                "correlacion_maxima_denso": round(maximo, 3),
                "variable_mas_correlacionada": max(correlaciones, key=correlaciones.get)
                if correlaciones else "",
                "correlacion_maxima_v12": round(maximo_base, 3),
                "feature_v12_mas_correlacionada": int(columna_base),
                "redundante": bool(max(maximo, maximo_base) >= UMBRAL_CORRELACION),
            })

        # --- ablación por subcategoría (§16): una subcategoría cada vez, control LogReg
        subcategorias = sorted(lexicon["familias"][familia])
        for subcategoria in subcategorias:
            columnas = [i for i, nombre in enumerate(nombres)
                        if f"lex_{familia}_{subcategoria}_" in nombre]
            if not columnas:
                continue
            submatriz = matriz[:, columnas]
            por_fold = []
            for fold in folds:
                entrenamiento = np.concatenate([f["val_idx"] for f in folds
                                               if f["fold"] != fold["fold"]])
                estimador = LogisticRegression(C=1.0, max_iter=3000, random_state=cfg.SEED)
                estimador.fit(submatriz[entrenamiento], indices[entrenamiento])
                por_fold.append(f1_score(indices[fold["val_idx"]],
                                         estimador.predict(submatriz[fold["val_idx"]]),
                                         average="macro", labels=ETIQUETAS))
            filas_ablacion.append({
                "familia": familia, "subcategoria": subcategoria,
                "n_features": len(columnas),
                "entradas_lexicon": len(lexicon["familias"][familia][subcategoria]),
                "filas_con_senal_pct": round(100.0 * float((submatriz > 0).any(axis=1).mean()), 3),
                "media_ratio": round(float(submatriz[:, 1::3].mean()), 5),
                "macro_f1": round(float(np.mean(por_fold)), 4),
                "std": round(float(np.std(por_fold)), 4),
            })
        resumen_familias[familia] = {
            "n_features": int(matriz.shape[1]),
            "columnas_con_varianza": no_vacias,
            "media_ratio": round(float(matriz[:, 1::3].mean()), 5),
            "filas_con_alguna_senal_pct": round(100.0 * float((matriz > 0).any(axis=1).mean()), 3),
            "control_lexicon_only": controles,
            "n_subcategorias": len(subcategorias),
        }

    pd.DataFrame(filas_redundancia).to_csv(REDUNDANCIA, index=False)
    pd.DataFrame(filas_ablacion).to_csv(ABLACION, index=False)
    redundancia = pd.DataFrame(filas_redundancia)
    resumen = {
        "fase": "P2-P5 — bloques lexicos, control lexicon-only, redundancia y ablacion",
        "fecha": "2026-09-25",
        "umbral_correlacion_redundancia": UMBRAL_CORRELACION,
        "variables_de_contraste": VARIABLES_EXISTENTES,
        "familias": resumen_familias,
        "features_redundantes": int(redundancia["redundante"].sum()),
        "features_totales": int(len(redundancia)),
        "redundantes_por_denso": int((redundancia["correlacion_maxima_denso"]
                                      >= UMBRAL_CORRELACION).sum()),
        "redundantes_por_v12": int((redundancia["correlacion_maxima_v12"]
                                    >= UMBRAL_CORRELACION).sum()),
        "max_correlacion_v12": round(float(redundancia["correlacion_maxima_v12"].max()), 3),
        "sha256_lexicons": hashes,
        "subcategorias_retiradas_por_minimo": lexicon["descartadas_por_minimo"],
        "nota_cribado": "el cribado de frecuencia minima usa el vocabulario del train completo: es "
                        "seleccion no supervisada y transductiva, sin usar etiquetas en ningun punto",
        "ablation_mejores": pd.DataFrame(filas_ablacion).nlargest(5, "macro_f1")[
            ["familia", "subcategoria", "n_features", "macro_f1"]].to_dict("records"),
        "segundos": round(time.time() - inicio, 1),
    }
    RESULTADO.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nredundantes (|rho| >= {UMBRAL_CORRELACION}): "
          f"{resumen['features_redundantes']}/{resumen['features_totales']}")
    print("mejores subcategorias (control lexicon-only):", resumen["ablation_mejores"])
    print(f"artefactos: {ABLACION.name}, {REDUNDANCIA.name}, features/lexicon_features_*.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
