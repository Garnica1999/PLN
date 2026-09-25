"""TRACK E v4: modelo conjunto de etiquetas sobre el espacio de combinaciones.

Contexto
--------
Las cadenas de etiquetas fallaron (E2-01) porque inyectan scores continuos de
otras etiquetas en la representacion textual, y el meta de 44 features mejoro
localmente pero no transfirio (D-017, D-020). Ninguna de las dos cosas es lo que
propone la spec v4 (§35-§39): un **modelo conjunto** ``P(y_HS, y_TR, y_AG | X)``
entrenado directamente sobre las combinaciones observadas y marginalizado a las
tres etiquetas.

La revision bibliografica (`reports/estado_arte_v4.md`, tema 5) recoge dos
comparativas recientes (Bogatinovski et al., ESWA 2022; Garcia-Pedrajas et al.,
Pattern Recognition 2024) que coinciden en que **Label Powerset** funciona bien
cuando hay pocas combinaciones activas, y que las combinaciones raras requieren
tratamiento jerarquico en lugar de exclusion. Este script implementa las tres
variantes de tratamiento y las compara contra los tres binarios independientes en
la misma tabla, con una sola decision.

Variantes
---------
1. ``joint_lr6``: multinomial sobre las 6 combinaciones observadas
   (000, 100, 101, 110, 111, 001), marginalizado a P(HS), P(TR), P(AG).
2. ``joint_lr4_mapeo``: solo las 4 combinaciones frecuentes (000, 100, 101, 111);
   las dos raras se mapean a su combinacion mas cercana por distancia de
   Hamming, con desempate a la clase mas positiva (asimetría de la metrica F1).
3. ``joint_lr4_excluir``: solo las 4 frecuentes, las raras quedan fuera del
   entrenamiento (diagnostico del techo de exclusion).
4. ``joint_lr6_prior``: multinomial sobre las 6 con peso de clase invertido
   (smoothing del prior para 110 y 001, que tienen 28 y 26 ejemplos).

Uso
---
    /home/carlos/venv/bin/python experiments/31_joint_label_v4.py
"""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import (  # noqa: E402
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

OOF_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CSV_PATH = PROJECT_ROOT / "reports" / "joint_label_v4.csv"
JSON_PATH = PROJECT_ROOT / "reports" / "joint_label_v4.json"
NPZ_PATH = PROJECT_ROOT / "features" / "oof_joint_label_v4.npz"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_35_joint_label_v4.png"

PARAMETROS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMETROS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
N_BOOTSTRAP = 2000
SEMILLA = 42
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
REJILLA_C = (0.5, 1.0, 2.0)
COMBINACIONES_FRECUENTES = ("000", "100", "101", "111")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-E] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Utilidades de combinaciones
# ---------------------------------------------------------------------------
def a_combinacion(y: np.ndarray) -> np.ndarray:
    """Convierte una matriz de etiquetas en cadenas de tres digitos.

    Args:
        y: matriz binaria de etiquetas.

    Returns:
        Vector de cadenas.
    """
    return np.array(["".join(str(int(v)) for v in fila) for fila in y])


def marginalizar(proba_combinaciones: np.ndarray, clases: list[str]) -> np.ndarray:
    """Marginaliza P(combinacion) a P(HS), P(TR) y P(AG).

    Args:
        proba_combinaciones: matriz de probabilidades por combinacion.
        clases: lista de nombres de clase en el orden de las columnas.

    Returns:
        Matriz de probabilidades por etiqueta.
    """
    posicion_etiqueta = {"HS": 0, "TR": 1, "AG": 2}
    salida = np.zeros((proba_combinaciones.shape[0], len(ETIQUETAS)))
    for posicion, etiqueta in enumerate(ETIQUETAS):
        columna = posicion_etiqueta[etiqueta]
        for indice, clase in enumerate(clases):
            if clase[columna] == "1":
                salida[:, posicion] += proba_combinaciones[:, indice]
    return salida


def mapear_raras(combinaciones: np.ndarray) -> np.ndarray:
    """Mapea las combinaciones rares a la combinacion frecuente mas cercana.

    La distancia es la de Hamming y el desempate va a la combinacion con mas
    positivos, coherente con una metrica que solo cuenta la clase positiva.

    Args:
        combinaciones: vector de combinaciones.

    Returns:
        Vector de combinaciones mapeadas.
    """
    frecuentes = list(COMBINACIONES_FRECUENTES)
    salida = []
    for combinacion in combinaciones:
        if combinacion in COMBINACIONES_FRECUENTES:
            salida.append(combinacion)
            continue
        candidata = min(
            frecuentes,
            key=lambda clase: (
                sum(1 for a, b in zip(combinacion, clase) if a != b),
                -sum(int(digito) for digito in clase),
            ),
        )
        salida.append(candidata)
    return np.array(salida)


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def f1_por_ejemplo(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calcula la fraccion de celdas correctas por ejemplo.

    Args:
        y_true: matriz verdadera.
        y_pred: matriz predicha.

    Returns:
        Vector de F1 por ejemplo.
    """
    return (y_true == y_pred).mean(axis=1).astype(float)


def bootstrap(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_bootstrap: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    """Bootstrap pareado del delta de F1 media por ejemplo.

    Args:
        y_true: matriz verdadera.
        pred_a: predicciones de referencia.
        pred_b: predicciones del candidato.
        n_bootstrap: numero de remuestreos.

    Returns:
        Tupla `(delta, ic95_bajo, ic95_alto)`.
    """
    f1_a = f1_por_ejemplo(y_true, pred_a)
    f1_b = f1_por_ejemplo(y_true, pred_b)
    generador = np.random.default_rng(SEMILLA)
    n = len(f1_a)
    deltas = np.empty(n_bootstrap)
    for indice in range(n_bootstrap):
        indices = generador.integers(0, n, n)
        deltas[indice] = f1_b[indices].mean() - f1_a[indices].mean()
    percentiles = np.percentile(deltas, [2.5, 97.5])
    return float(f1_b.mean() - f1_a.mean()), float(percentiles[0]), float(percentiles[1])


def evaluar(
    nombre: str,
    proba: np.ndarray,
    y_true: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    pred_ref: np.ndarray,
) -> dict[str, Any]:
    """Evalua un modelo conjunto con los umbrales de V03 y con 0.5.

    Args:
        nombre: identificador de la variante.
        proba: probabilidades marginales por etiqueta.
        y_true: matriz verdadera.
        pliegues: lista de pliegues corregidos.
        pred_ref: predicciones de referencia.

    Returns:
        Fila de resultados.
    """
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    metricas = metricas_multietiqueta(y_true, pred)
    metricas_cinco = metricas_multietiqueta(y_true, (proba >= 0.5).astype(int))
    f1_pliegues = [
        metricas_multietiqueta(y_true[validar], pred[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]
    f1_ref = [
        metricas_multietiqueta(y_true[validar], pred_ref[validar])["f1_media_columnas"]
        for _, validar in pliegues
    ]
    delta, ic_bajo, ic_alto = bootstrap(y_true, pred_ref, pred)
    return {
        "variante": nombre,
        "corrected_cv_umbrales_v03": round(metricas["f1_media_columnas"], 6),
        "corrected_cv_umbral_0_5": round(metricas_cinco["f1_media_columnas"], 6),
        "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 6),
        "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 6),
        "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 6),
        "exact_match": round(metricas["exact_match"], 6),
        "std_pliegues": round(float(np.std(f1_pliegues, ddof=1)), 6),
        "pliegues_mejorados": int(np.sum(np.array(f1_pliegues) > np.array(f1_ref) + 1e-12)),
        "delta_f1_ejemplo": round(delta, 6),
        "ic95_bajo": round(ic_bajo, 6),
        "ic95_alto": round(ic_alto, 6),
        "f1_pliegues": ";".join(f"{valor:.4f}" for valor in f1_pliegues),
    }


def matriz_correccion(
    y_true: np.ndarray, pred_ref: np.ndarray, pred_nuevo: np.ndarray
) -> dict[str, Any]:
    """Construye la matriz de correccion del candidato frente a V03.

    Args:
        y_true: matriz verdadera.
        pred_ref: predicciones de referencia.
        pred_nuevo: predicciones del candidato.

    Returns:
        Diccionario con los conteos de acierto/fallo por modelo.
    """
    acierto_ref = (pred_ref == y_true).all(axis=1)
    acierto_nuevo = (pred_nuevo == y_true).all(axis=1)
    return {
        "n": int(len(y_true)),
        "ambos_correctos": int((acierto_ref & acierto_nuevo).sum()),
        "solo_v03": int((acierto_ref & ~acierto_nuevo).sum()),
        "solo_candidato": int((~acierto_ref & acierto_nuevo).sum()),
        "ambos_incorrectos": int((~acierto_ref & ~acierto_nuevo).sum()),
        "tasa_correccion": round(
            float((~acierto_ref & acierto_nuevo).sum() / max(int(acierto_ref.sum()), 1)), 4
        ),
        "tasa_dano": round(
            float((acierto_ref & ~acierto_nuevo).sum() / max(int(acierto_ref.sum()), 1)), 4
        ),
        "neto": int((~acierto_ref & acierto_nuevo).sum()) - int((acierto_ref & ~acierto_nuevo).sum()),
    }


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta la comparacion de modelos conjuntos de etiquetas."""
    inicio_tiempo = time.time()
    inicio = datetime.now(timezone.utc)
    log("Modelo conjunto de etiquetas (TRACK E): Label Powerset y marginalizacion")

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids_dev = oof["ids"].astype(np.int64)
    y_dev = oof["y_true"].astype(int)
    pred_ref = oof["pred"].astype(int)

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    textos = desarrollo["text"]

    distribucion = pd.Series(a_combinacion(y_dev)).value_counts().to_dict()
    log(f"   combinaciones en el desarrollo: {distribucion}")

    cache: list[tuple[sparse.csr_matrix, sparse.csr_matrix]] = []
    for entrenar, validar in pliegues:
        palabra = TfidfVectorizer(**PARAMETROS_WORD)
        caracter = TfidfVectorizer(**PARAMETROS_CHAR)
        izquierda = sparse.hstack(
            [palabra.fit_transform(textos.iloc[entrenar]), caracter.fit_transform(textos.iloc[entrenar])],
            format="csr",
        )
        derecha = sparse.hstack(
            [palabra.transform(textos.iloc[validar]), caracter.transform(textos.iloc[validar])],
            format="csr",
        )
        cache.append((izquierda, derecha))
    log("   matrices por pliegue cacheadas")

    resultados: list[dict[str, Any]] = []
    oof_guardado: dict[str, np.ndarray] = {}

    def ajustar_conjunto(
        indice_fold: int,
        entrenar: np.ndarray,
        indices: np.ndarray,
        mapa: str,
        c: float,
        peso_clase: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Entrena un multinomial sobre combinaciones y devuelve probabilidades.

        Args:
            indice_fold: posicion del pliegue en la cache de matrices.
            entrenar: indices absolutos del pliegue de entrenamiento.
            indices: subconjunto de `entrenar` que se usa en esta variante.
            mapa: tratamiento de las combinaciones raras.
            c: fuerza de regularizacion.
            peso_clase: ``balanced`` o ``none``.

        Returns:
            Tupla con las probabilidades de las filas de validacion y las clases.
        """
        combinaciones = a_combinacion(y_dev[indices])
        if mapa == "lr6":
            etiquetas = combinaciones
        elif mapa == "lr4_mapeo":
            etiquetas = mapear_raras(combinaciones)
        elif mapa == "lr4_excluir":
            mascara = np.isin(combinaciones, COMBINACIONES_FRECUENTES)
            etiquetas = combinaciones[mascara]
            indices = indices[mascara]
        else:
            raise ValueError(f"Tratamiento desconocido: {mapa}")
        modelo = LogisticRegression(
            C=c,
            class_weight="balanced" if peso_clase == "balanced" else None,
            max_iter=2000,
            solver="lbfgs",
            random_state=SEMILLA,
        )
        # En `lr4_excluir` se descartan filas: la matriz se recorta con las
        # posiciones *relativas al pliegue* de las mismas filas, porque la cache
        # esta indexada en el orden de `entrenar`.
        posiciones = np.searchsorted(entrenar, indices)
        modelo.fit(cache[indice_fold][0][posiciones], etiquetas)
        proba = modelo.predict_proba(cache[indice_fold][1])
        return proba, [str(clase) for clase in modelo.classes_]

    for mapa, peso_clase in (
        ("lr6", "balanced"),
        ("lr4_mapeo", "balanced"),
        ("lr4_excluir", "balanced"),
        ("lr6", "none"),
    ):
        for C in REJILLA_C:
            proba_oof = np.zeros((len(desarrollo), len(ETIQUETAS)))
            for indice_fold, (entrenar, validar) in enumerate(pliegues):
                proba_combinaciones, clases = ajustar_conjunto(
                    indice_fold, entrenar, entrenar, mapa, C, peso_clase
                )
                proba_oof[validar] = marginalizar(proba_combinaciones, clases)
            nombre = f"joint:{mapa}_{'bal' if peso_clase == 'balanced' else 'unb'}_C{C}"
            oof_guardado[nombre] = proba_oof
            fila = evaluar(nombre, proba_oof, y_dev, pliegues, pred_ref)
            fila["correccion"] = matriz_correccion(
                y_dev, pred_ref, (proba_oof >= UMBRALES_V03[None, :]).astype(int)
            )
            resultados.append(fila)
            log(
                f"   {nombre:34s} corrected_cv={fila['corrected_cv_umbrales_v03']:.4f} "
                f"pliegues={fila['pliegues_mejorados']}/5 neto={fila['correccion']['neto']:+d}"
            )

    tabla = pd.DataFrame(resultados).sort_values("corrected_cv_umbrales_v03", ascending=False)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.drop(columns=["correccion"]).to_csv(CSV_PATH, index=False)

    referencia = 0.777211
    mejor = tabla.iloc[0]
    cumple = bool(
        float(mejor["corrected_cv_umbrales_v03"]) - referencia >= 0.005
        and int(mejor["pliegues_mejorados"]) >= 3
    )
    hay_neto_positivo = any(fila["correccion"]["neto"] > 0 for fila in resultados)
    resumen = {
        "experiment_id": "V4-E01-E03",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.time() - inicio_tiempo, 2),
        "referencia_v03": referencia,
        "distribucion_combinaciones": distribucion,
        "n_variantes": int(len(tabla)),
        "mejor_variante": str(mejor["variante"]),
        "mejor_corrected_cv": float(mejor["corrected_cv_umbrales_v03"]),
        "mejor_ic95": [float(mejor["ic95_bajo"]), float(mejor["ic95_alto"])],
        "cumple_criterio_D016": cumple,
        "alguna_variante_corrige_mas_de_lo_que_dana": hay_neto_positivo,
        "decision_familia": (
            "ADOPTAR" if cumple else "NO_SIGNAL: el modelo conjunto no supera a los tres binarios independientes"
        ),
        "detalle": resultados,
    }
    JSON_PATH.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        NPZ_PATH,
        ids=ids_dev,
        y_true=y_dev,
        **{nombre.replace(":", "__"): valor for nombre, valor in oof_guardado.items()},
    )
    _escribir_figura(tabla)
    log(f"   mejor variante: {resumen['mejor_variante']} ({resumen['mejor_corrected_cv']:.4f})")
    log(f"   cumple D-016: {cumple} -> {resumen['decision_familia']}")


def _escribir_figura(tabla: pd.DataFrame) -> None:
    """Dibuja el corrected_cv de las variantes del modelo conjunto.

    Args:
        tabla: tabla de resultados.
    """
    figura, eje = plt.subplots(figsize=(11, 5))
    colores = [
        "#C44E52" if valor <= 0.777211 else "#4C72B0"
        for valor in tabla["corrected_cv_umbrales_v03"]
    ]
    eje.barh(tabla["variante"], tabla["corrected_cv_umbrales_v03"], color=colores)
    eje.axvline(0.777211, color="#55A868", linestyle="--", label="V03 = 0.7772")
    eje.set_xlabel("corrected_cv con umbrales de V03")
    eje.set_title("Modelo conjunto de etiquetas (v4, TRACK E)")
    eje.legend()
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


if __name__ == "__main__":
    main()
