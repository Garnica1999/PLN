"""P8 v4: refit del candidato elegido, evaluacion final en el holdout y submission.

Contexto
--------
La comparativa de candidatos (`experiments/36_candidatos_v4.py`) dejo cuatro
candidatos. El elegido es **C** (``meta36_downweight``) porque es el unico que
supera los tres criterios de D-016 a la vez (nivel >= +0.005, >= 3/5 pliegues e
IC95 del delta pareado > 0) y el que corrige mas errores de V03 sin introducir
mas (neto +34). El candidato D tiene el corrected_cv mas alto (0.7904) pero su
IC95 pareado toca cero, asi que su evidencia es mas fragile.

Este script ejecuta la cadena de entrega de la spec (§100):

1. **Refit** sobre todo el desarrollo (n=1.483), sin tocar el holdout ni el test.
2. **Transform + predict** sobre el test (1.154) y sobre el holdout (248).
3. **Thresholds** congelados de V03 (0.44/0.51/0.46), los mismos que se usaron
   para medir el corrected_cv.
4. **Submission** CSV validada (columnas, ids, binario, sin NaN) con SHA256.
5. **Holdout**: se consume la quinta y ultima evaluacion autorizada (D-025) con
   este unico finalista y se registra el resultado.

Parametros del refit (pre-registrados)
--------------------------------------
- Pesos de ejemplo: downweight de la violacion ``1 - 0.5 * z(error_score)``
  acotado a [0.5, 1.5], con ``error_score`` calculado **solo con OOF** del
  desarrollo.
- Pesos de clase: **mediana** de los multiplicadores que el barrido anidado
  eligio en cada uno de los 5 pliegues (evita elegir con el holdout y reduce la
  varianza del ajuste).
- Meta: Ridge alpha=3 sobre 36 features (4 bases x 3 etiquetas x [p, margen,
  entropia]).

Uso
---
    /home/carlos/venv/bin/python experiments/38_refit_final_v4.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import (  # noqa: E402
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
)

OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
HARDNESS_PATH = PROJECT_ROOT / "features" / "hardness_v4.npz"
CANDIDATOS_PATH = PROJECT_ROOT / "reports" / "v4_candidates.csv"
PESOS_PATH = PROJECT_ROOT / "reports" / "supervised_weighting_v4.csv"
HOLDOUT_EVAL_PATH = PROJECT_ROOT / "reports" / "holdout_evaluaciones_v2.csv"
SUBMISSION_PATH = (
    PROJECT_ROOT / "submissions" / "toxic_submission_v12_meta36_downweight.csv"
)
MANIFEST_PATH = PROJECT_ROOT / "submissions" / "submission_manifest.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "refit_final_v4.json"
HASHES_PATH = PROJECT_ROOT / "reports" / "submission_hashes_v4.json"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
ALPHA_META = 3.0
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
CANDIDATO = "C_meta36_downweight"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-REFIT] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def features_meta(probas: list[np.ndarray]) -> np.ndarray:
    """Construye las 36 features del meta.

    Args:
        probas: lista de cuatro vectores de probabilidades.

    Returns:
        Matriz de 36 columnas.
    """
    columnas = []
    for proba in probas:
        recorte = np.clip(proba, 1e-9, 1.0 - 1e-9)
        columnas.extend([recorte[:, j] for j in range(len(ETIQUETAS))])
        columnas.extend([recorte[:, j] - 0.5 for j in range(len(ETIQUETAS))])
        columnas.extend(
            [
                -(recorte[:, j] * np.log(recorte[:, j]) + (1 - recorte[:, j]) * np.log(1 - recorte[:, j]))
                for j in range(len(ETIQUETAS))
            ]
        )
    return np.column_stack(columnas)


def _pesos_de_clase(y: np.ndarray, multiplicador: float) -> dict[int, float]:
    """Construye los pesos de clase a partir de un multiplicador del balanceo.

    Args:
        y: vector de etiquetas binarias.
        multiplicador: factor aplicado al peso de la clase positiva.

    Returns:
        Diccionario ``{clase: peso}``.
    """
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    return {
        0: (n / (2.0 * max(n_neg, 1))),
        1: (n / (2.0 * max(n_pos, 1))) * multiplicador,
    }


def multiplicadores_medianos() -> tuple[np.ndarray, list[list[float]]]:
    """Lee los multiplicadores por pliegue del barrido anidado y toma la mediana.

    Returns:
        Tupla con el vector de multiplicadores por etiqueta y la lista de
        configuraciones por pliegue.
    """
    with PESOS_PATH.open(encoding="utf-8") as handle:
        for fila in csv_filas(handle):
            if fila["variante"] == "clase:barrio_anidado_por_etiqueta":
                configuraciones = [
                    [float(valor) for valor in combo.split("/")]
                    for combo in fila["multiplicadores_por_fold"].split(";")
                ]
                break
        else:
            raise RuntimeError("no se encontro el barrido anidado en el registro del TRACK C")
    arreglo = np.array(configuraciones)
    return np.median(arreglo, axis=0), configuraciones


def csv_filas(handle) -> list[dict[str, str]]:
    """Lee un CSV completo como lista de diccionarios.

    Args:
        handle: archivo abierto.

    Returns:
        Lista de filas.
    """
    import csv

    return list(csv.DictReader(handle))


def ajustar_cuatro_bases(
    textos_entrenar: pd.Series,
    textos_aplicar: pd.Series,
    y_entrenar: np.ndarray,
    sample_weight: np.ndarray,
    multiplicadores: np.ndarray,
) -> list[np.ndarray]:
    """Ajusta las cuatro bases de la referencia con pesos.

    Args:
        textos_entrenar: textos de entrenamiento.
        textos_aplicar: textos a predecir.
        y_entrenar: etiquetas del entrenamiento.
        sample_weight: peso por ejemplo.
        multiplicadores: multiplicadores de peso de clase por etiqueta.

    Returns:
        Lista de cuatro vectores de probabilidades.
    """
    palabra = TfidfVectorizer(**PARAMETROS_WORD)
    caracter = TfidfVectorizer(**PARAMETROS_CHAR)
    palabra_entrenar = palabra.fit_transform(textos_entrenar)
    palabra_aplicar = palabra.transform(textos_aplicar)
    caracter_entrenar = caracter.fit_transform(textos_entrenar)
    caracter_aplicar = caracter.transform(textos_aplicar)
    union_entrenar = sparse.hstack([palabra_entrenar, caracter_entrenar], format="csr")
    union_aplicar = sparse.hstack([palabra_aplicar, caracter_aplicar], format="csr")

    def ajustar(matriz_entrenar, matriz_aplicar, modelo: str) -> np.ndarray:
        """Ajusta tres modelos binarios con pesos y devuelve probabilidades.

        Args:
            matriz_entrenar: matriz de entrenamiento.
            matriz_aplicar: matriz a predecir.
            modelo: ``lr`` o ``svc``.

        Returns:
            Matriz de probabilidades.
        """
        salida = np.zeros((matriz_aplicar.shape[0], len(ETIQUETAS)))
        for posicion in range(len(ETIQUETAS)):
            if modelo == "lr":
                estimador = LogisticRegression(
                    C=2.0, max_iter=1000, solver="liblinear", random_state=SEMILLA
                )
            else:
                estimador = LinearSVC(C=1.0, max_iter=5000, random_state=SEMILLA)
            estimador.set_params(
                class_weight=_pesos_de_clase(y_entrenar[:, posicion], multiplicadores[posicion])
            )
            estimador.fit(
                matriz_entrenar, y_entrenar[:, posicion], sample_weight=sample_weight
            )
            if hasattr(estimador, "predict_proba"):
                salida[:, posicion] = estimador.predict_proba(matriz_aplicar)[:, 1]
            else:
                salida[:, posicion] = 1.0 / (
                    1.0 + np.exp(-estimador.decision_function(matriz_aplicar))
                )
        return salida

    return [
        ajustar(palabra_entrenar, palabra_aplicar, "lr"),
        ajustar(union_entrenar, union_aplicar, "lr"),
        ajustar(union_entrenar, union_aplicar, "svc"),
        ajustar(caracter_entrenar, caracter_aplicar, "lr"),
    ]


def main() -> None:
    """Ejecuta el refit final, la evaluacion en holdout y la submission."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log(f"Refit final del candidato {CANDIDATO} (P8)")

    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, indices_holdout = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    holdout = train.iloc[indices_holdout].reset_index(drop=True)

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    dureza = np.load(HARDNESS_PATH, allow_pickle=True)
    assert np.array_equal(dureza["ids"].astype(np.int64), ids_dev)
    violacion = dureza["error_score"].astype(float)
    escala = np.percentile(violacion, 75) - np.percentile(violacion, 25) or 1.0
    z = (violacion - np.median(violacion)) / escala
    pesos_ejemplo = np.clip(1.0 - 0.5 * z, 0.5, 1.5)
    multiplicadores, configuraciones = multiplicadores_medianos()
    log(f"   multiplicadores (mediana de 5 pliegues): {list(multiplicadores)}")
    log(f"   pesos de ejemplo: media {pesos_ejemplo.mean():.3f}, rango [{pesos_ejemplo.min():.2f}, {pesos_ejemplo.max():.2f}]")

    log("   ajustando las cuatro bases sobre todo el desarrollo")
    y_dev = desarrollo[["HS", "TR", "AG"]].to_numpy(dtype=int)
    probas_holdout = ajustar_cuatro_bases(
        desarrollo["text"], holdout["text"], y_dev, pesos_ejemplo, multiplicadores
    )
    features_holdout = features_meta(probas_holdout)

    log("   ajustando el meta (Ridge alpha=3) con features OOF del desarrollo")
    # El meta se entrena con las features OOF ya calculadas en el corrected_cv,
    # que son las unicas que no dependen de un modelo que vio el ejemplo.
    candidatas = np.load(
        PROJECT_ROOT / "features" / "oof_candidates_v4.npz", allow_pickle=True
    )
    assert np.array_equal(candidatas["ids"].astype(np.int64), ids_dev)
    probas_meta_oof = candidatas[CANDIDATO].astype(float)
    # El meta aprende de las probabilidades de la combinacion; para predecir se
    # necesitan las features del meta, no sus probabilidades.
    log("   reconstruyendo las features del meta en el desarrollo (OOF anidado)")

    # Se reconstruyen las features del meta sobre el desarrollo con un CV interno
    # de 4 pliegues, igual que en la construccion del corrected_cv.
    from sklearn.model_selection import StratifiedGroupKFold

    from src.splits import combinaciones, pliegues_corregidos

    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    features_oof = np.zeros((len(desarrollo), 36))
    for indice_fold, (entrenar, validar) in enumerate(pliegues):
        interno = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=SEMILLA + indice_fold
        )
        pares = list(
            interno.split(
                desarrollo.iloc[entrenar],
                combinaciones(desarrollo.iloc[entrenar]).reset_index(drop=True),
                groups=grupos[indices_dev][entrenar],
            )
        )
        for filas, columnas in pares:
            probas = ajustar_cuatro_bases(
                desarrollo["text"].iloc[entrenar[filas]],
                desarrollo["text"].iloc[entrenar[columnas]],
                y_dev[entrenar[filas]],
                pesos_ejemplo[entrenar[filas]],
                multiplicadores,
            )
            features_oof[entrenar[columnas]] = features_meta(probas)
    meta = Ridge(alpha=ALPHA_META)
    meta.fit(features_oof, y_dev)

    proba_holdout = np.clip(meta.predict(features_holdout), 0.0, 1.0)
    pred_holdout = (proba_holdout >= UMBRALES_V03[None, :]).astype(int)
    metricas_holdout = metricas_multietiqueta(
        holdout[["HS", "TR", "AG"]].to_numpy(dtype=int), pred_holdout
    )
    log(
        f"   HOLDOUT (248 tweets, 5/5 evaluaciones): F1 media "
        f"{metricas_holdout['f1_media_columnas']:.4f} "
        f"(V03 en el mismo holdout: 0.8285)"
    )

    log("   prediciendo el test y escribiendo la submission")
    probas_test = ajustar_cuatro_bases(
        desarrollo["text"], test["text"], y_dev, pesos_ejemplo, multiplicadores
    )
    proba_test = np.clip(meta.predict(features_meta(probas_test)), 0.0, 1.0)
    pred_test = (proba_test >= UMBRALES_V03[None, :]).astype(int)
    submission = pd.DataFrame(
        {
            "id": test["id"].astype(int).to_numpy(),
            "HS": pred_test[:, 0],
            "TR": pred_test[:, 1],
            "AG": pred_test[:, 2],
        }
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    hash_submission = sha256_de(SUBMISSION_PATH)

    plantilla = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "submission.csv")
    validacion = {
        "columnas": list(submission.columns) == ["id", "HS", "TR", "AG"],
        "n_filas": int(len(submission)),
        "n_filas_esperado": int(len(plantilla)),
        "ids_coinciden": bool(
            submission["id"].tolist() == plantilla["id"].astype(int).tolist()
        ),
        "valores_binarios": bool(
            submission[["HS", "TR", "AG"]].isin([0, 1]).all().all()
        ),
        "sin_nulos": bool(not submission.isna().any().any()),
        "ids_unicos": bool(submission["id"].is_unique),
        "positivos": {
            etiqueta: int(submission[etiqueta].sum()) for etiqueta in ETIQUETAS
        },
    }
    validacion["ok"] = bool(
        validacion["columnas"]
        and validacion["n_filas"] == validacion["n_filas_esperado"]
        and validacion["ids_coinciden"]
        and validacion["valores_binarios"]
        and validacion["sin_nulos"]
        and validacion["ids_unicos"]
    )
    log(f"   submission v12 escrita: {SUBMISSION_PATH.name} (sha256 {hash_submission[:16]}...)")
    log(f"   validacion: {'OK' if validacion['ok'] else 'FALLA'} | positivos {validacion['positivos']}")

    # Registro de la quinta evaluacion de holdout.
    with HOLDOUT_EVAL_PATH.open(encoding="utf-8", newline="") as handle:
        import csv

        lector = csv.DictReader(handle)
        campos = list(lector.fieldnames)
        filas = list(lector)
    ETIQUETA_HOLDOUT = "P8-C-META36-DOWNWEIGHT (5/5)"
    if not any(fila.get("experimento") == ETIQUETA_HOLDOUT for fila in filas):
        filas.append(
            {
            "fecha": datetime.now(timezone.utc).date().isoformat(),
            "experimento": ETIQUETA_HOLDOUT,
            "motivo": (
                "Candidato final de v4: unico que supera los tres criterios de "
                "D-016 (nivel, pliegues e IC95 del delta pareado)"
            ),
            "resultado": json.dumps(
                {
                    "holdout_f1_media_columnas": round(
                        metricas_holdout["f1_media_columnas"], 6
                    ),
                    "holdout_f1_por_etiqueta": {
                        k: round(v, 6) for k, v in metricas_holdout["f1_por_etiqueta"].items()
                    },
                    "referencia_v03_holdout": 0.8285,
                    "delta_pareado_vs_v03": round(
                        metricas_holdout["f1_media_columnas"] - 0.8285, 6
                    ),
                },
                ensure_ascii=False,
            ),
            }
        )
    with HOLDOUT_EVAL_PATH.open("w", encoding="utf-8", newline="") as handle:
        import csv

        escritor = csv.DictWriter(handle, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)
    log("   holdout: 5/5 evaluaciones usadas")

    # Manifest de submissions.
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        import csv

        lector = csv.DictReader(handle)
        campos_manifest = list(lector.fieldnames)
        filas_manifest = list(lector)
    if not any(
        fila.get("submission_id") == "v12_meta36_downweight" for fila in filas_manifest
    ):
        if "sha256" not in campos_manifest:
            campos_manifest.append("sha256")
        filas_manifest.append(
            {
                "submission_id": "v12_meta36_downweight",
                "version": "v12",
                "model": (
                    "Ridge(alpha=3) sobre 36 features de 4 bases ajustadas con "
                    "downweight por violacion del umbral y class_weight por etiqueta"
                ),
                "features": "union TF-IDF word(1,2)+char_wb(3,5) -> meta de 36 features",
                "thresholds": "HS=0.44;TR=0.51;AG=0.46 (congelados de V03)",
                "corrected_cv": 0.788173,
                "historical_cv": "no medido (protocolo v2 corregido)",
                "holdout": round(metricas_holdout["f1_media_columnas"], 6),
                "hypothesis": (
                    "Bajar el peso de los ejemplos que mas violan el umbral y "
                    "ajustar el peso de clase por etiqueta, y usar esa base dentro "
                    "de un meta Ridge sobre las cuatro bases"
                ),
                "expected_behavior": (
                    "Cumple los tres criterios de D-016 (nivel, pliegues e IC95 "
                    "pareado); unico candidato final de v4"
                ),
                "status": "READY_FOR_KAGGLE_PROBE (el envio lo hace Carlos a mano)",
                "sha256": hash_submission,
            }
        )
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        import csv

        escritor = csv.DictWriter(handle, fieldnames=campos_manifest)
        escritor.writeheader()
        escritor.writerows(filas_manifest)

    HASHES_PATH.write_text(
        json.dumps(
            {
                "toxic_submission_v12_meta36_downweight.csv": {
                    "sha256": hash_submission,
                    "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "candidato": CANDIDATO,
                    "umbrales": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    informe = {
        "experiment_id": "P8-REFIT-FINAL",
        "candidato": CANDIDATO,
        "fecha": inicio.isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "multiplicadores_medianos": [float(v) for v in multiplicadores],
        "multiplicadores_por_fold": configuraciones,
        "pesos_ejemplo": {
            "definicion": "clip(1 - 0.5 * z(error_score), 0.5, 1.5) con error_score OOF",
            "media": round(float(pesos_ejemplo.mean()), 4),
            "minimo": round(float(pesos_ejemplo.min()), 4),
            "maximo": round(float(pesos_ejemplo.max()), 4),
        },
        "holdout": {
            "n": int(len(holdout)),
            "f1_media": round(metricas_holdout["f1_media_columnas"], 6),
            "f1_por_etiqueta": {
                k: round(v, 6) for k, v in metricas_holdout["f1_por_etiqueta"].items()
            },
            "referencia_v03": 0.8285,
            "delta": round(metricas_holdout["f1_media_columnas"] - 0.8285, 6),
            "evaluaciones_usadas": "5/5",
        },
        "submission": {
            "archivo": SUBMISSION_PATH.name,
            "sha256": hash_submission,
            "validacion": validacion,
        },
        "corrected_cv_candidato": 0.788173,
        "referencia_v03_corrected_cv": 0.777211,
    }
    JSON_PATH.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"   informe escrito en {JSON_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
