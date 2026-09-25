"""FASE 12 — Auditoría del estado real (punto de partida de la búsqueda de señal residual).

Observación
    La búsqueda de señal residual solo es interpretable si el sustrato experimental es el que dice
    ser: modelo final v12 (14 fuentes + promedio de combinadores sobre 279 features), matrices
    cacheadas del meta, OOF de los miembros, clústeres de near-duplicados, holdout congelado y
    última submission. El script del modelo final valida el lado test **solo por forma** y existe el
    precedente D-021 (desalineación semántica de columnas), así que esta auditoría comprueba
    invariantes de bloque, reconstruye las matrices y reproduce los resultados registrados.

Protocolo
    1. Alineación de filas: etiquetas, OOF, clústeres y textos (train/test).
    2. Reconstrucción exacta de la matriz del meta del lado train (273) contra la caché.
    3. Firmas de bloque en train y test: tamaños, invariantes de probabilidad, NaN/Inf, varianza
       por columna y estadísticos comparados.
    4. Reproducción del CV LOFO del v12 (referencia 0,6005 ± 0,0005).
    5. Verificación end-to-end: el modelo v11 reconstruido debe reproducir
       ``age_submission_v11.csv`` a partir de la matriz de test cacheada.
    6. Holdout congelado: reconstrucción determinista y tamaño (semilla 2026, 15%).
    7. Validación formal y de distribución de ``age_submission_v12.csv``.
    8. Conteo de envíos usados y disponibles en Kaggle (solo lectura; no se envía nada).

Salida: ``experiments/phase12_auditoria_estado.json`` y ``reports/auditoria_estado_fase12.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv, data, submission, textometry
from src.config import set_seed

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import MEMBER_FILES, EncodedMLP, build_features

RESULTS_PATH = cfg.EXPERIMENTS / "phase12_auditoria_estado.json"
REPORT_PATH = cfg.REPORTS / "auditoria_estado_fase12.md"
HOLDOUT_SEED, HOLDOUT_SHARE = 2026, 0.15
TOLERANCIA_LOFO = 0.0005


def registrar(payload: dict, seccion: str, valor: dict) -> None:
    """Guarda una sección del informe en el payload, la imprime y persiste el avance."""
    payload[seccion] = valor
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"[{seccion}] {json.dumps(valor, ensure_ascii=False, default=str)[:400]}", flush=True)


def masas_de_bloque(matriz: np.ndarray, bloques: list[tuple[str, int, str]]) -> list[dict]:
    """Comprueba invariantes por bloque (exp-suma para bloques logarítmicos)."""
    filas: list[dict] = []
    inicio = 0
    for nombre, ancho, tipo in bloques:
        bloque = matriz[:, inicio:inicio + ancho]
        fila = {"bloque": nombre, "columnas": ancho, "tipo": tipo,
                "varianza_cero": int((bloque.var(axis=0) == 0).sum()),
                "nan": int(np.isnan(bloque).sum())}
        if tipo == "log":
            masa = np.exp(bloque).sum(axis=1)
            fila["exp_suma_media"] = round(float(masa.mean()), 8)
            fila["exp_suma_max_desviacion"] = round(float(np.abs(masa - 1).max()), 8)
        filas.append(fila)
        inicio += ancho
    return filas


def comparar_bloques(entrenamiento: np.ndarray, prueba: np.ndarray,
                     bloques: list[tuple[str, int, str]]) -> list[dict]:
    """Compara estadísticos por columna entre train y test (detección de desalineación)."""
    filas: list[dict] = []
    inicio = 0
    for nombre, ancho, _ in bloques:
        izquierda = entrenamiento[:, inicio:inicio + ancho]
        derecha = prueba[:, inicio:inicio + ancho]
        escala = izquierda.std(axis=0)
        escala = np.where(escala < 1e-9, 1e-9, escala)
        z = np.abs(izquierda.mean(axis=0) - derecha.mean(axis=0)) / (escala / np.sqrt(len(prueba)))
        filas.append({"bloque": nombre, "z_max": round(float(z.max()), 2),
                      "z_medio": round(float(z.mean()), 2),
                      "columnas_z_mayor_10": int((z > 10).sum())})
        inicio += ancho
    return filas


def evaluar_combinadores(features: np.ndarray, y: np.ndarray, folds: list[dict]) -> dict:
    """Reproduce el CV LOFO del modelo final v12 (promedio MLP + LogReg)."""
    puntajes = []
    for fold in folds:
        train_idx = np.concatenate([f["val_idx"] for f in folds if f["fold"] != fold["fold"]])
        valid_idx = fold["val_idx"]
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                                                      random_state=cfg.SEED))])
        logreg.fit(features[train_idx], y[train_idx])
        mlp = EncodedMLP()
        mlp.fit(features[train_idx], y[train_idx])
        logreg_probabilities = logreg.predict_proba(features[valid_idx])
        mlp_probabilities = mlp.model.predict_proba(mlp.scaler.transform(features[valid_idx]))
        predictions = np.array(cfg.CLASSES)[np.argmax(
            0.5 * (mlp_probabilities + logreg_probabilities), axis=1)]
        puntajes.append(float(f1_score(y[valid_idx], predictions, average="macro",
                                       labels=cfg.CLASSES)))
        print(f"    fold {fold['fold']}: {puntajes[-1]:.4f}", flush=True)
    return {"macro_f1": round(float(np.mean(puntajes)), 4),
            "std": round(float(np.std(puntajes)), 4),
            "per_fold": [round(valor, 4) for valor in puntajes]}


def envíos_kaggle(payload: dict) -> None:
    """Lista los envíos registrados en Kaggle (solo lectura, tolerante a fallos de red)."""
    import os

    try:
        credenciales = Path.home() / ".kaggle" / "kaggle.json"
        if credenciales.exists() and "KAGGLE_API_TOKEN" not in os.environ:
            os.environ["KAGGLE_API_TOKEN"] = json.loads(credenciales.read_text())["token"]
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        for intento in range(3):
            try:
                lista = api.competition_submissions(cfg.COMPETITION) or []
                break
            except Exception:
                if intento == 2:
                    raise
        registros = []
        for item in lista:
            registro = {}
            for clave in ("ref", "date", "status", "description", "error_description", "file_name",
                          "public_score", "private_score", "url"):
                try:
                    registro[clave] = str(getattr(item, clave))
                except BaseException:  # noqa: BLE001 - atributo ausente según el endpoint
                    registro[clave] = ""
            registros.append(registro)
        completos = [r for r in registros if "COMPLETE" in r.get("status", "")]
        con_puntaje = [r for r in registros if r.get("public_score") not in ("", "None")]
        registrar(payload, "envios_kaggle", {
            "n_registrados": len(registros), "n_completos": len(completos),
            "n_con_puntaje_publico": len(con_puntaje),
            "mejores": sorted(con_puntaje, key=lambda r: float(r["public_score"]),
                              reverse=True)[:4],
            "registros": registros[:20],
        })
    except BaseException as error:  # noqa: BLE001 - SystemExit del cliente de Kaggle incluido
        registrar(payload, "envios_kaggle", {"error": f"{type(error).__name__}: {error}"})


def main() -> int:
    """Ejecuta la auditoría del estado real del pipeline."""
    set_seed()
    started = time.time()
    payload: dict = {"fecha": "2026-09-25", "fase": "A — auditoría del estado real"}

    # 1 ------------------------------------------------------------------ Alineación de filas
    train = data.load_train()
    test = data.load_test()
    sample = data.load_sample_submission()
    y = train[cfg.TARGET].astype(str).to_numpy()
    oof_labels = np.load(cfg.FEATURES / "oof_labels.npy", allow_pickle=True)
    oof_labels = np.asarray([str(valor) for valor in oof_labels])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    normalizado = cv.normalize_for_dedup(pd.concat([train[cfg.TEXT_COL], test[cfg.TEXT_COL]],
                                                  ignore_index=True))
    muestra = np.arange(len(normalizado)) % 9973 == 0
    coincide_norm = (clusters["norm"].to_numpy() == normalizado.to_numpy())[muestra].mean()
    registrar(payload, "alineacion", {
        "train": len(train), "test": len(test),
        "etiquetas_iguales_oof": bool(np.array_equal(y, oof_labels)),
        "clusters_filas": len(clusters), "clusters_esperados": len(train) + len(test),
        "clusters_coincide_normalizado_pct": round(float(100 * coincide_norm), 4),
        "clases": {clase: int((y == clase).sum()) for clase in cfg.CLASSES},
    })

    # 2 --------------------------------------------- Reconstrucción de la matriz del meta (train)
    features, y_reconstruido, _, _, _ = build_features()
    cache_train = np.load(cfg.FEATURES / "meta_features_train_v11.npy")
    diferencia = float(np.abs(features - cache_train).max()) if features.shape == cache_train.shape else -1.0
    registrar(payload, "matriz_train", {
        "forma_reconstruida": list(features.shape), "forma_cache": list(cache_train.shape),
        "etiquetas_iguales": bool(np.array_equal(y, y_reconstruido)),
        "max_diferencia_absoluta": diferencia,
        "reconstruccion_exacta": bool(features.shape == cache_train.shape and diferencia == 0.0),
    })

    # 3 ------------------------------------------------------- Firmas de bloque en train y test
    anchos = []
    tipos = []
    for nombre, transform in MEMBER_FILES:
        matriz = np.load(cfg.FEATURES / nombre, mmap_mode="r")
        ancho = 1 if matriz.ndim == 1 else matriz.shape[1]
        anchos.append((Path(nombre).stem, ancho, transform))
    densas = [nombre for nombre in textometry.TEXTOMETRY_FEATURES + textometry.STYLE_FEATURES
              if nombre not in {"n_emoji", "emoji_diversity", "has_emoji"}]
    estructuradas = pd.read_parquet(cfg.FEATURES / "age_structured_features.parquet")
    supervisadas = np.load(cfg.FEATURES / "age_supervised_features.npy", mmap_mode="r")
    bloques = (anchos + [("densas", len(densas), "denso"),
                         ("estructuradas", estructuradas.shape[1], "denso"),
                         ("supervisadas", supervisadas.shape[1], "denso")])
    cache_test = np.load(cfg.FEATURES / "meta_features_test_v11.npy")
    suma_anchos = sum(ancho for _, ancho, _ in bloques)
    registrar(payload, "bloques_train", {
        "suma_anchos": suma_anchos, "matriz": list(cache_train.shape),
        "firmas": masas_de_bloque(cache_train, bloques),
    })
    registrar(payload, "bloques_test", {
        "matriz": list(cache_test.shape), "firmas": masas_de_bloque(cache_test, bloques),
        "comparacion_train_test": comparar_bloques(cache_train, cache_test, bloques),
    })

    # 3b ------------------------------------------------- Envíos en Kaggle (solo lectura, barato)
    envíos_kaggle(payload)

    # 4 ------------------------------------------------------------ Reproducción del CV LOFO v12
    v12 = np.concatenate([features, np.load(cfg.FEATURES / "oof_ADV07_nbsvm_word.npy")], axis=1)
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[:len(y)], n_splits=5,
                          n_repeats=1, scheme="group")
    referencia = 0.6005
    reproducido = evaluar_combinadores(v12, y, folds)
    registrar(payload, "reproduccion_lofo_v12", {
        "referencia": referencia, "tolerancia": TOLERANCIA_LOFO,
        "resultado": reproducido,
        "dentro_de_tolerancia": abs(reproducido["macro_f1"] - referencia) <= TOLERANCIA_LOFO,
    })

    # 5 ------------------------------------------- End-to-end: la caché de test reproduce la v11
    meta_v11 = EncodedMLP()
    meta_v11.fit(cache_train, y)
    predicciones_v11 = meta_v11.predict(cache_test)
    enviada_v11 = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v11.csv")
    enviada_v12 = pd.read_csv(cfg.SUBMISSIONS / "age_submission_v12.csv")
    acuerdo_v11 = float((np.asarray(enviada_v11[cfg.TARGET].astype(str)) == predicciones_v11).mean())
    acuerdo_v11_v12 = float((enviada_v11[cfg.TARGET].astype(str)
                             == enviada_v12[cfg.TARGET].astype(str)).mean())
    registrar(payload, "end_to_end_v11", {
        "acuerdo_con_submission_v11": round(acuerdo_v11, 6),
        "acuerdo_submission_v11_vs_v12": round(acuerdo_v11_v12, 6),
        "ids_iguales_orden": bool(np.array_equal(enviada_v11[cfg.ID_COL].to_numpy(),
                                                 test[cfg.ID_COL].to_numpy())),
    })

    # 6 ------------------------------------------------------------- Holdout congelado (semilla)
    ids_cluster = clusters["cluster"].to_numpy()[:len(y)]
    rng_uno = np.random.default_rng(HOLDOUT_SEED)
    elegidos_uno = set(rng_uno.choice(np.unique(ids_cluster),
                                      size=int(len(np.unique(ids_cluster)) * HOLDOUT_SHARE),
                                      replace=False).tolist())
    rng_dos = np.random.default_rng(HOLDOUT_SEED)
    elegidos_dos = set(rng_dos.choice(np.unique(ids_cluster),
                                      size=int(len(np.unique(ids_cluster)) * HOLDOUT_SHARE),
                                      replace=False).tolist())
    en_holdout = np.array([cluster in elegidos_uno for cluster in ids_cluster])
    registrar(payload, "holdout", {
        "semilla": HOLDOUT_SEED, "fraccion_clusteres": HOLDOUT_SHARE,
        "determinista": bool(elegidos_uno == elegidos_dos),
        "n_clusteres": len(elegidos_uno),
        "filas_holdout": int(en_holdout.sum()),
        "share_filas_pct": round(float(100 * en_holdout.mean()), 3),
    })

    # 7 ------------------------------------------------------------ Validación de la submission v12
    checks = submission.validate_submission(enviada_v12, test[cfg.ID_COL], sample)
    registrar(payload, "submission_v12", {
        "checks": checks, "filas": len(enviada_v12),
        "distribucion": enviada_v12[cfg.TARGET].value_counts().reindex(cfg.CLASSES)
                                  .fillna(0).astype(int).to_dict(),
    })

    # 8 ------------------------------------------------------------------- Cierre del informe
    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")

    # Informe en Markdown
    lineas = [
        "# Auditoría del estado real — Fase A (2026-09-25)", "",
        "Generado por `scripts/phase12_auditoria_estado.py`. Sin escritura sobre artefactos previos.", "",
        "## 1. Alineación", "",
        f"- Train {payload['alineacion']['train']:,} · test {payload['alineacion']['test']:,}",
        f"- Etiquetas vs OOF: {'idénticas' if payload['alineacion']['etiquetas_iguales_oof'] else 'DIVERGEN'}",
        f"- Clústeres: {payload['alineacion']['clusters_filas']:,} filas "
        f"(esperadas {payload['alineacion']['clusters_esperados']:,}) · "
        f"coincidencia de normalizado {payload['alineacion']['clusters_coincide_normalizado_pct']}%",
        "", "## 2. Matriz del meta (train)", "",
        f"- Reconstruida {payload['matriz_train']['forma_reconstruida']} vs caché "
        f"{payload['matriz_train']['forma_cache']}",
        f"- Diferencia máxima absoluta: {payload['matriz_train']['max_diferencia_absoluta']}"
        f" · exacta: {payload['matriz_train']['reconstruccion_exacta']}",
        "", "## 3. Firmas de bloque", "",
        "| Bloque | Cols | exp-suma = 1 | Varianza cero | z máx train↔test |",
        "|---|---|---|---|---|",
    ]
    comparacion = {fila["bloque"]: fila for fila in payload["bloques_test"]["comparacion_train_test"]}
    for firma in payload["bloques_train"]["firmas"]:
        comp = comparacion.get(firma["bloque"], {})
        masa = firma.get("exp_suma_max_desviacion", "—")
        lineas.append(f"| {firma['bloque']} | {firma['columnas']} | {masa} | "
                      f"{firma['varianza_cero']} | {comp.get('z_max', '—')} |")
    lineas += [
        "", "## 4. Reproducción del CV LOFO v12", "",
        f"- Referencia {referencia:.4f} · reproducido {reproducido['macro_f1']:.4f} "
        f"± {reproducido['std']:.4f} · por fold {reproducido['per_fold']}",
        f"- Dentro de tolerancia: {payload['reproduccion_lofo_v12']['dentro_de_tolerancia']}",
        "", "## 5. End-to-end (sustrato de test)", "",
        f"- Acuerdo con `age_submission_v11.csv`: **{100 * acuerdo_v11:.4f}%**",
        f"- Acuerdo v11 vs v12: {100 * acuerdo_v11_v12:.4f}%",
        "", "## 6. Holdout congelado", "",
        f"- {payload['holdout']['filas_holdout']:,} filas "
        f"({payload['holdout']['share_filas_pct']}%) · determinista: "
        f"{payload['holdout']['determinista']}",
        "", "## 7. Submission v12", "",
        f"- Checks: {payload['submission_v12']['checks']}",
        f"- Distribución: {payload['submission_v12']['distribucion']}",
        "", "## 8. Envíos en Kaggle", "",
        f"- {json.dumps(payload['envios_kaggle'], ensure_ascii=False)[:600]}",
        "", f"_Duración: {payload['segundos']} s._",
    ]
    REPORT_PATH.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\ninforme: {REPORT_PATH}")
    print(f"resultado: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
