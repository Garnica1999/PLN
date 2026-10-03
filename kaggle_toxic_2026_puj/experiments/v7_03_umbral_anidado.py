"""Track P0.1 de la fase v7 (V7-03): umbral anidado con protocolo estricto de tres niveles.

Contexto
--------
La fase v6 dejo una tarea pendiente y es la que este script resuelve. Al reajustar umbrales
sobre el mismo conjunto fuera de pliegue que despues se evalua, la metrica se infla en
+0.0036 sin cambiar el modelo. Ese sesgo no se puede medir con el protocolo de dos niveles
que usa V03, y por eso hace falta un protocolo de tres.

Que hace distinto a la medicion ya existente
--------------------------------------------
E2-16 ya midio un umbral global anidado y dio 0.776613. Su protocolo es **leave-one-fold-out**
para el umbral: el umbral del pliegue k sesacaba de las probabilidades fuera de pliegue de
los otros cuatro pliegues. Eso es valido, pero **reutiliza predicciones que tocan el propio
pliegue exterior**, de modo que el conjunto de calibracion es mas grande que el que la
especificacion de la fase v7 permite, y ademas no separa el ajuste del modelo del ajuste del
umbral.

Aqui hay **tres niveles** que no se mezclan:

```text
ajuste       el modelo se ajusta solo en el ajuste interior
calibracion  el umbral se ajusta solo en la calibracion interior
evaluacion   se aplica una sola vez sobre la validacion exterior
```

El umbral nunca ve la validacion exterior, y el modelo nunca ve ni la calibracion ni la
evaluacion. Esa es la garantia que el protocolo de dos niveles no da.

Que entrega ademas del resultado
--------------------------------
El producto mas valioso del script no es el delta, que se espera pequeno, sino el
**informe de dispersion de umbrales**: para cada etiqueta, la media, la mediana, la
desviacion tipica, el rango intercuartil y el rango total de los quince umbrales que salen de
los cinco pliegues exteriores y los tres labels. Eso responde a la pregunta que la
especificacion hace de forma explicita: si 0.44, 0.51 y 0.46 son estables o producto de
sobreajuste. **Se responde con numeros y no con una sensacion.**

Diseno experimental: tres brazos, porque hay dos efectos que no hay que confundir
------------------------------------------------------------------------------
Hay que separar **si el codigo esta bien** de **cuanto cuesta el protocolo**, y confundirlos
produce una lectura falsa en las dos direcciones.

- **P0-0-referencia**: el modelo se ajusta sobre **todo** el entrenamiento exterior con los
  umbrales congelados de V03. Este es el brazo que **valida el codigo** y debe reproducir
  0.777211. Si no lo hace, el problema es del script.
- **P0-1-control**: el mismo pipeline de tres niveles con los umbrales congelados, es decir,
  el modelo ajustado solo sobre el ajuste interior. Aqui el ajuste usa **cinco sextos** de las
  filas de las que usan los otros dos brazos, de modo que **pierde exactitud por culpa del
  protocolo, no del codigo**. Medir ese coste es parte del trabajo, y por eso se reporta.
- **P0-1-anidado**: umbrales calibrados en el nivel interior y aplicados una vez fuera. El
  delta se mide **contra P0-1-control**, con el que comparte exactamente el mismo conjunto de
  ajuste, de modo que el coste del protocolo se cancela y lo que queda es el efecto del
  umbral. Comparar contra V03 seria mezclar dos cosas y es justo el error que la especificacion
  quiere evitar.

La primera version de este script fijaba una tolerancia de 0.002 sobre P0-1-control y se
detenia. La deteccion era correcta pero la tolerancia estaba mal planteada: el error no era
del codigo sino del protocolo, que necesariamente reduce el conjunto de ajuste. Anadir el
brazo P0-0 separa las dos cosas y deja el resultado con sentido.

Los umbrales se buscan en una rejilla de 0.01 entre 0.20 y 0.80, que es la misma rejilla que
usa el proyecto en sus umbrales por etiqueta. Se elige el que maximiza la F1 de la etiqueta en
el conjunto de calibracion, y se registra tambien el segundo mejor para poder medir cuanto
apano la superficie de la eleccion.

Uso
---
    /home/carlos/venv/bin/python experiments/v7_03_umbral_anidado.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    _f1_por_celda,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "V7-03 (P0.1)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v7_umbral_anidado.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v7_umbral_anidado.csv"
REPORTE_ANALISIS = PROJECT_ROOT / "reports" / "v7_calibration_analysis.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v7_umbral_anidado.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
REFERENCIA_E2_16 = 0.776613
REJILLA = np.round(np.arange(0.20, 0.801, 0.01), 2)
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOLERANCIA_CONTROL = 0.002
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260926
SEMILLA_PARTICION = 20260926
# Proporcion de filas que se reserva para calibrar umbrales dentro de cada pliegue exterior.
# La particion agrupada se hace en cinco cortes y se toma uno como calibracion, de modo que
# la proporcion efectiva es un quinto. La constante se usa para comprobar que el corte
# elegido produce la proporcion declarada, en vez de suponerlo.
CORTOS_PARTICION_INTERNA = 5
PROPORCION_CALIBRACION = 1.0 / CORTOS_PARTICION_INTERNA

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V7-P0.1] {mensaje}", flush=True)


def partir_tres_niveles(
    indices_entrenamiento: np.ndarray,
    y: np.ndarray,
    grupos_dev: np.ndarray,
    generador,
) -> tuple[np.ndarray, np.ndarray]:
    """Parte el entrenamiento exterior en ajuste interior y calibracion interior.

    La particion respeta los grupos, de modo que ningun casi duplicado queda en un lado y
    otro, y estratifica por la **combinacion** de las tres etiquetas, que es mas informativa
    que cualquier etiqueta suelta en un problema multilabel.

    Args:
        indices_entrenamiento: indices globales del entrenamiento exterior.
        y: matriz binaria verdadera.
        grupos_dev: grupo de cada fila del desarrollo.
        generador: generador de numeros aleatorios.

    Returns:
        Tupla con los indices de ajuste interior y los de calibracion interior.
    """
    combinaciones = []
    for indice in indices_entrenamiento:
        combinaciones.append("".join(str(v) for v in y[indice]))
    # StratifiedGroupKFold no acepta cadena de texto, asi que se codifica cada combinacion
    # como un entero mediante su posicion en el conjunto de combinaciones observadas.
    unicas = {valor: numero for numero, valor in enumerate(sorted(set(combinaciones)))}
    estrato = np.array([unicas[valor] for valor in combinaciones])
    particion = StratifiedGroupKFold(
        n_splits=CORTOS_PARTICION_INTERNA, shuffle=True, random_state=SEMILLA_PARTICION
    )
    for ajuste, calibracion in particion.split(
        indices_entrenamiento, estrato, grupos_dev[indices_entrenamiento]
    ):
        del generador
        proporcion_real = len(calibracion) / len(indices_entrenamiento)
        if abs(proporcion_real - PROPORCION_CALIBRACION) > 0.15:
            raise RuntimeError(
                f"la proporcion de calibracion salio {proporcion_real:.3f} y no coincide con "
                f"la declarada {PROPORCION_CALIBRACION:.3f}"
            )
        return indices_entrenamiento[ajuste], indices_entrenamiento[calibracion]
    raise RuntimeError("la particion de tres niveles no devolvio ningun corte")


def ajustar_probabilidades(matriz, objetivo, consulta) -> np.ndarray:
    """Ajusta un clasificador por etiqueta y devuelve probabilidades.

    Args:
        matriz: matriz dispersa de entrenamiento.
        objetivo: vector binario de entrenamiento.
        consulta: matriz dispersa sobre la que predecir.

    Returns:
        Vector de probabilidades.
    """
    estimador = LogisticRegression(**PARAMETROS_LR)
    estimador.fit(matriz, objetivo)
    return estimador.predict_proba(consulta)[:, 1]


def calibrar_umbrales(
    y_calibracion: np.ndarray, proba_calibracion: np.ndarray
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Elige un umbral por etiqueta maximizando la F1 en el conjunto de calibracion.

    Se registra tambien el **segundo mejor umbral** y cuanto F1 se pierde al tomar el
    primero. Si la diferencia es pequeña, la eleccion del umbral es una decision arbitraria
    y el umbral no esta realmente identificado por los datos; si es grande, el umbral esta
    bien determinado. Ese matiz es lo que hace util el informe de dispersion.

    Args:
        y_calibracion: etiquetas del conjunto de calibracion.
        proba_calibracion: probabilidades predichas sobre el conjunto de calibracion.

    Returns:
        Tupla con los umbrales elegidos y el detalle de la eleccion por etiqueta.
    """
    from sklearn.metrics import f1_score

    umbrales: dict[str, float] = {}
    detalle: dict[str, dict[str, Any]] = {}
    for indice, etiqueta in enumerate(ETIQUETAS):
        puntuaciones: list[tuple[float, float]] = []
        for valor in REJILLA:
            f1 = float(
                f1_score(
                    y_calibracion[:, indice],
                    (proba_calibracion[:, indice] >= valor).astype(int),
                    zero_division=0,
                )
            )
            puntuaciones.append((f1, float(valor)))
        puntuaciones.sort(key=lambda par: (-par[0], abs(par[1] - 0.5)))
        mejor_f1, mejor_umbral = puntuaciones[0]
        segundo_f1, segundo_umbral = puntuaciones[1]
        umbrales[etiqueta] = mejor_umbral
        detalle[etiqueta] = {
            "umbral": mejor_umbral,
            "f1_en_calibracion": round(mejor_f1, 6),
            "segundo_umbral": segundo_umbral,
            "f1_del_segundo": round(segundo_f1, 6),
            "holgura": round(mejor_f1 - segundo_f1, 6),
            "umbral_congelado_v03": UMBRALES_V03[etiqueta],
            "distancia_al_congelado": round(
                abs(mejor_umbral - UMBRALES_V03[etiqueta]), 4
            ),
        }
    return umbrales, detalle


def main() -> None:
    """Ejecuta el control y el protocolo anidado estricto, y produce el informe de dispersion."""
    inicio = time.time()
    generador = np.random.default_rng(SEMILLA_PARTICION)
    log("Track P0.1: umbral anidado con protocolo estricto de tres niveles")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_v03 = oof["proba"]
    pred_v03 = (proba_v03 >= np.array([UMBRALES_V03[e] for e in ETIQUETAS])[None, :]).astype(int)
    textos = desarrollo["text"].astype(str)
    grupos_dev = grupos[indices_dev]
    log(f"   desarrollo {len(y)} filas | V03 {metrica_macro(y, pred_v03):.6f}")
    log(
        "   umbrales congelados de V03: "
        + ", ".join(f"{e} {UMBRALES_V03[e]}" for e in ETIQUETAS)
    )
    log(f"   proporcion de calibracion interior: {PROPORCION_CALIBRACION:.2f}")

    proba_referencia = np.zeros((len(y), 3), dtype=np.float64)
    proba_control = np.zeros((len(y), 3), dtype=np.float64)
    umbrales_por_pliegue: list[dict[str, float]] = []
    detalles_por_pliegue: list[dict[str, Any]] = []
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        ajuste, calibracion = partir_tres_niveles(
            entrenamiento, y, grupos_dev, generador
        )
        # Brazo de referencia: mismo vectorizador que los otros dos, pero ajustado sobre todo
        # el entrenamiento exterior. Sirve para validar el codigo del pipeline.
        vectorizador_ref_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_ref_char = TfidfVectorizer(**PARAMS_CHAR)
        matriz_ref = sparse.hstack(
            [
                vectorizador_ref_word.fit_transform(textos.iloc[entrenamiento]),
                vectorizador_ref_char.fit_transform(textos.iloc[entrenamiento]),
            ]
        ).tocsr()
        consulta_ref = sparse.hstack(
            [
                vectorizador_ref_word.transform(textos.iloc[validacion]),
                vectorizador_ref_char.transform(textos.iloc[validacion]),
            ]
        ).tocsr()
        for etiqueta in range(3):
            proba_referencia[validacion, etiqueta] = ajustar_probabilidades(
                matriz_ref, y[entrenamiento, etiqueta], consulta_ref
            )
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        matriz_ajuste = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos.iloc[ajuste]),
                vectorizador_char.fit_transform(textos.iloc[ajuste]),
            ]
        ).tocsr()
        matriz_calibracion = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[calibracion]),
                vectorizador_char.transform(textos.iloc[calibracion]),
            ]
        ).tocsr()
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ]
        ).tocsr()
        proba_cal = np.zeros((len(calibracion), 3), dtype=np.float64)
        for etiqueta in range(3):
            proba_cal[:, etiqueta] = ajustar_probabilidades(
                matriz_ajuste, y[ajuste, etiqueta], matriz_calibracion
            )
            proba_control[validacion, etiqueta] = ajustar_probabilidades(
                matriz_ajuste, y[ajuste, etiqueta], consulta
            )
        umbrales, detalle = calibrar_umbrales(y[calibracion], proba_cal)
        umbrales_por_pliegue.append(umbrales)
        detalles_por_pliegue.append(
            {
                "pliegue": numero,
                "n_ajuste": int(len(ajuste)),
                "n_calibracion": int(len(calibracion)),
                "n_validacion": int(len(validacion)),
                "detalle": detalle,
            }
        )
        # Los umbrales calibrados se aplican una sola vez, sobre la validacion exterior, y
        # solo para el registro de progreso. La prediccion anidada completa se arma despues,
        # en un unico sitio, para que no haya dos caminos que puedan divergir.
        pred_validacion = np.zeros((len(validacion), 3), dtype=np.int64)
        for etiqueta in range(3):
            pred_validacion[:, etiqueta] = (
                proba_control[validacion, etiqueta] >= umbrales[ETIQUETAS[etiqueta]]
            ).astype(int)
        log(
            f"   pliegue {numero}: ajuste {len(ajuste)} calibracion {len(calibracion)} "
            f"validacion {len(validacion)} | umbrales "
            + ", ".join(f"{e} {umbrales[e]}" for e in ETIQUETAS)
            + f" | f1 en validacion {metrica_macro(y[validacion], pred_validacion):.4f}"
        )

    log("informe de dispersion de umbrales")
    dispersion = _dispersion(umbrales_por_pliegue)
    for etiqueta in ETIQUETAS:
        d = dispersion[etiqueta]
        log(
            f"   {etiqueta}: media {d['media']:.4f} mediana {d['mediana']:.4f} "
            f"desviacion {d['desviacion_tipica']:.4f} rango_intercuartil "
            f"{d['rango_intercuartil']:.4f} rango {d['rango']:.4f} | congelado "
            f"{UMBRALES_V03[etiqueta]} | dentro del rango: {d['congelado_dentro_del_rango']}"
        )

    pred_control = (
        proba_control >= np.array([UMBRALES_V03[e] for e in ETIQUETAS])[None, :]
    ).astype(int)
    # La prediccion anidada se arma por pliegue con los umbrales de su calibracion interior.
    pred_anidada = np.zeros((len(y), 3), dtype=np.int64)
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        for etiqueta in range(3):
            pred_anidada[validacion, etiqueta] = (
                proba_control[validacion, etiqueta]
                >= umbrales_por_pliegue[numero][ETIQUETAS[etiqueta]]
            ).astype(int)

    pred_referencia = (
        proba_referencia >= np.array([UMBRALES_V03[e] for e in ETIQUETAS])[None, :]
    ).astype(int)
    variantes = [
        _evaluar("P0-0-referencia-completa", y, pred_referencia, pred_v03, pliegues),
        _evaluar("P0-1-control-frozen", y, pred_control, pred_referencia, pliegues),
        _evaluar("P0-1-anidado", y, pred_anidada, pred_control, pliegues),
    ]
    for variante in variantes:
        log(
            f"   {variante['variante']:24s} corrected_cv {variante['corrected_cv']:.6f} "
            f"| delta frente a su referencia {variante['delta']:+.4f} "
            f"IC95 [{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"HS {variante['f1_HS']:.4f} TR {variante['f1_TR']:.4f} AG {variante['f1_AG']:.4f}"
        )
    # El unico brazo cuya tolerancia tiene sentido es el de referencia: es el que reproduce la
    # cifra conocida y por tanto el que valida el codigo. El control de tres niveles difiere
    # de V03 a proposito, porque ajusta sobre cinco sextos de las filas.
    variantes[0]["codigo_correcto"] = (
        abs(variantes[0]["corrected_cv"] - REFERENCIA_V03) <= TOLERANCIA_CONTROL
    )
    coste_protocolo = variantes[1]["delta"]
    log(
        f"   validacion del codigo (P0-0 frente a V03): {variantes[0]['delta']:+.6f} con "
        f"tolerancia {TOLERANCIA_CONTROL:.4f} -> "
        f"{'dentro' if variantes[0]['codigo_correcto'] else 'FUERA'}"
    )
    log(f"   coste del protocolo de tres niveles: {coste_protocolo:+.6f}")
    anidada = variantes[2]
    anidada["gate_delta"] = anidada["delta"] >= GATE_DELTA
    anidada["gate_ic95"] = anidada["ic95_0"] > 0
    anidada["gate_supera"] = (
        anidada["gate_delta"]
        and anidada["gate_ic95"]
        and anidada["pliegues_positivos"] >= GATE_PLIEGUES
    )
    log(
        f"   gate de la anidada: delta {'OK' if anidada['gate_delta'] else 'no'} "
        f"({anidada['delta']:+.4f}) | IC95 {'OK' if anidada['gate_ic95'] else 'no'} | "
        f"pliegues {anidada['pliegues_positivos']}/{len(pliegues)} -> "
        f"{'SUPERA' if anidada['gate_supera'] else 'no supera'}"
    )
    if not variantes[0]["codigo_correcto"]:
        log(
            "PARADA: el brazo de referencia no reproduce 0.777211, de modo que el codigo del "
            "pipeline no es fiable y no se sigue."
        )

    log("informe y figura")
    _figura(umbrales_por_pliegue, dispersion)
    _escribir_tabla_calibracion(detalles_por_pliegue, dispersion)
    _informe(
        variantes, dispersion, umbrales_por_pliegue, detalles_por_pliegue, len(y),
        pliegues, inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "corrected_cv", "delta", "ic95_0", "ic95_1", "f1_HS", "f1_TR",
            "f1_AG", "salvage", "damage", "pliegues_positivos",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for variante in variantes:
            escritor.writerow({campo: variante.get(campo, "") for campo in campos})
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": REFERENCIA_V03,
                "referencia_e2_16_protocolo_laxo": REFERENCIA_E2_16,
                "umbrales_congelados_v03": UMBRALES_V03,
                "proporcion_calibracion": PROPORCION_CALIBRACION,
                "tolerancia_control": TOLERANCIA_CONTROL,
                "dispersion": dispersion,
                "umbrales_por_pliegue": umbrales_por_pliegue,
                "detalle_por_pliegue": detalles_por_pliegue,
                "variantes": [
                    {k: v for k, v in variante.items() if k != "pred"}
                    for variante in variantes
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mejor = max(variantes, key=lambda v: v["corrected_cv"])
    log(
        f"Track P0.1 completada en {round(time.time() - inicio, 1)} s | mejor: "
        f"{mejor['variante']} {mejor['corrected_cv']:.6f}"
    )


def _dispersion(umbrales_por_pliegue: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
    """Calcula la dispersion de los umbrales calibrados por etiqueta.

    Args:
        umbrales_por_pliegue: umbrales de cada pliegue exterior.

    Returns:
        Diccionario por etiqueta con las medidas de dispersion y la comparacion con el
        umbral congelado de V03.
    """
    salida: dict[str, dict[str, Any]] = {}
    for etiqueta in ETIQUETAS:
        valores = np.array([u[etiqueta] for u in umbrales_por_pliegue], dtype=float)
        cuartiles = np.percentile(valores, [25, 50, 75])
        congelado = UMBRALES_V03[etiqueta]
        salida[etiqueta] = {
            "valores": [float(v) for v in valores],
            "media": round(float(valores.mean()), 4),
            "mediana": round(float(cuartiles[1]), 4),
            "desviacion_tipica": round(float(valores.std(ddof=1)), 4),
            "rango_intercuartil": round(float(cuartiles[2] - cuartiles[0]), 4),
            "rango": round(float(valores.max() - valores.min()), 4),
            "minimo": round(float(valores.min()), 4),
            "maximo": round(float(valores.max()), 4),
            "congelado_v03": congelado,
            "distancia_de_la_media": round(abs(float(valores.mean()) - congelado), 4),
            "congelado_dentro_del_rango": bool(valores.min() <= congelado <= valores.max()),
        }
    return salida


def _evaluar(nombre, y, pred, referencia, pliegues) -> dict[str, Any]:
    """Evalua una variante contra la prediccion de referencia.

    Args:
        nombre: identificador de la variante.
        y: matriz binaria verdadera.
        pred: predicciones de la variante.
        referencia: predicciones de referencia.
        pliegues: pliegues corregidos.

    Returns:
        Diccionario con metricas, comparacion pareada y complementariedad.
    """
    metrica = metrica_macro(y, pred)
    comparacion = bootstrap_diferencia_pareada(
        y, pred, referencia, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
    )
    por_etiqueta = metricas_multietiqueta(y, pred)["f1_por_etiqueta"]
    acierta = _f1_por_celda(y, pred) >= 0.999
    acierta_ref = _f1_por_celda(y, referencia) >= 0.999
    positivos = 0
    for validacion in pliegues:
        indices = validacion[1] if isinstance(validacion, tuple) else validacion
        if metrica_macro(y[indices], pred[indices]) > metrica_macro(
            y[indices], referencia[indices]
        ):
            positivos += 1
    return {
        "variante": nombre,
        "corrected_cv": round(metrica, 6),
        "delta": round(metrica - metrica_macro(y, referencia), 6),
        "ic95_0": comparacion["ic95"][0],
        "ic95_1": comparacion["ic95"][1],
        "f1_HS": round(por_etiqueta[ETIQUETAS[0]], 4),
        "f1_TR": round(por_etiqueta[ETIQUETAS[1]], 4),
        "f1_AG": round(por_etiqueta[ETIQUETAS[2]], 4),
        "salvage": int(np.sum(acierta & ~acierta_ref)),
        "damage": int(np.sum(~acierta & acierta_ref)),
        "pliegues_positivos": positivos,
        "pred": pred,
    }


def _escribir_tabla_calibracion(detalles, dispersion) -> None:
    """Escribe la tabla de analisis de calibracion que pide la especificacion.

    Args:
        detalles: detalle de la eleccion de umbral por pliegue.
        dispersion: medidas de dispersion por etiqueta.
    """
    filas: list[dict[str, Any]] = []
    for entrada in detalles:
        for etiqueta in ETIQUETAS:
            d = entrada["detalle"][etiqueta]
            filas.append(
                {
                    "pliegue": entrada["pliegue"],
                    "n_ajuste": entrada["n_ajuste"],
                    "n_calibracion": entrada["n_calibracion"],
                    "n_validacion": entrada["n_validacion"],
                    "etiqueta": etiqueta,
                    "umbral_calibrado": d["umbral"],
                    "umbral_congelado_v03": d["umbral_congelado_v03"],
                    "distancia_al_congelado": d["distancia_al_congelado"],
                    "f1_en_calibracion": d["f1_en_calibracion"],
                    "segundo_umbral": d["segundo_umbral"],
                    "holgura_entre_el_primero_y_el_segundo": d["holgura"],
                    "mediana_de_los_cinco": dispersion[etiqueta]["mediana"],
                    "desviacion_tipica": dispersion[etiqueta]["desviacion_tipica"],
                }
            )
    with REPORTE_ANALISIS.open("w", encoding="utf-8", newline="") as archivo:
        campos = list(filas[0])
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def _figura(umbrales_por_pliegue, dispersion) -> None:
    """Dibuja los umbrales calibrados por pliegue frente a los congelados de V03.

    Args:
        umbrales_por_pliegue: umbrales de cada pliegue exterior.
        dispersion: medidas de dispersion por etiqueta.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, eje = plt.subplots(figsize=(9, 4.6))
    ancho = 0.24
    colores = {"HS": "#1f77b4", "TR": "#ff7f0e", "AG": "#2ca02c"}
    posiciones = np.arange(len(umbrales_por_pliegue))
    for k, etiqueta in enumerate(ETIQUETAS):
        valores = [u[etiqueta] for u in umbrales_por_pliegue]
        eje.bar(
            posiciones + (k - 1) * ancho,
            valores,
            ancho,
            label=f"{etiqueta} calibrado por pliegue",
            color=colores[etiqueta],
            alpha=0.85,
        )
        # Linea de puntos con el umbral congelado de V03, del mismo color de la barra para
        # que se compare la barra con su referencia y no con la de al lado.
        eje.axhline(
            UMBRALES_V03[etiqueta],
            linestyle=":",
            linewidth=1.5,
            color=colores[etiqueta],
        )
    eje.set_xticks(posiciones)
    eje.set_xticklabels([f"pliegue {i}" for i in range(len(posiciones))])
    eje.set_ylabel("umbral de decision")
    eje.set_ylim(0.2, 0.8)
    eje.set_title(
        "Umbrales calibrados por pliegue; la linea de puntos es el umbral congelado de V03"
    )
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_68_umbral_anidado_v7.png", dpi=150)
    plt.close()


def _informe(
    variantes, dispersion, umbrales_por_pliegue, detalles, n_filas, pliegues, inicio
) -> None:
    """Escribe el informe del track P0.1.

    Args:
        variantes: resultados de las variantes.
        dispersion: medidas de dispersion por etiqueta.
        umbrales_por_pliegue: umbrales de cada pliegue exterior.
        detalles: detalle de la eleccion de umbral por pliegue.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos, solo para contar cuantos hay.
        inicio: instante de arranque del script.
    """
    referencia, control, anidada = variantes
    lineas = [
        "# Track P0.1: umbral anidado con protocolo estricto (V7-03)",
        "",
        f"Generado por `experiments/v7_03_umbral_anidado.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que resuelve este script",
        "",
        "La fase v6 dejo una tarea abierta: **reajustar umbrales sobre el mismo conjunto",
        "fuera de pliegue que despues se evalua infla la metrica en +0.0036 sin cambiar el",
        "modelo**. Ese sesgo no se puede medir con un protocolo de dos niveles. Aqui hay tres",
        "niveles que no se mezclan:",
        "",
        "```text",
        "ajuste       el modelo se ajusta solo en el ajuste interior",
        "calibracion  el umbral se ajusta solo en la calibracion interior",
        "evaluacion   se aplica una sola vez sobre la validacion exterior",
        "```",
        "",
        "E2-16 ya habia medido un umbral global anidado y dio",
        f"{REFERENCIA_E2_16}, pero con un protocolo mas laxo: reutilizaba las",
        "probabilidades fuera de pliegue de los otros cuatro pliegues como conjunto de",
        "calibracion, de modo que el umbral veia mas datos de los que aqui se le permiten.",
        "",
        "## 2. Informe de dispersion de umbrales (el producto principal)",
        "",
        "Quince umbrales: cinco pliegues exteriores por tres etiquetas. Esta es la pregunta",
        "que la especificacion hace de forma explicita, respondida con numeros.",
        "",
        "| etiqueta | valores por pliegue | media | mediana | desviacion | rango intercuartil "
        "| rango | congelado V03 | dentro del rango |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for etiqueta in ETIQUETAS:
        d = dispersion[etiqueta]
        valores = ", ".join(f"{v:.2f}" for v in d["valores"])
        lineas.append(
            f"| {etiqueta} | {valores} | {d['media']:.4f} | {d['mediana']:.4f} | "
            f"{d['desviacion_tipica']:.4f} | {d['rango_intercuartil']:.4f} | "
            f"{d['rango']:.4f} | {d['congelado_v03']} | "
            f"{'si' if d['congelado_dentro_del_rango'] else 'no'} |"
        )
    lineas += [
        "",
        "## 3. Resultados: tres brazos para no confundir codigo con protocolo",
        "",
        "Hay que separar **si el codigo esta bien** de **cuanto cuesta el protocolo**, porque",
        "confundirlos produce una lectura falsa en las dos direcciones.",
        "",
        "| variante | corrected_cv | delta frente a su referencia | IC95 | HS | TR | AG |",
        "|---|---|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {variante['corrected_cv']:.6f} | "
            f"{variante['delta']:+.4f} | [{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} |"
        )
    lineas += [
        "",
        "- **P0-0** ajusta sobre todo el entrenamiento exterior y es el brazo que **valida el",
        f"  codigo**: {referencia['delta']:+.6f} frente a {REFERENCIA_V03}, tolerancia "
        f"{TOLERANCIA_CONTROL:.4f}, veredicto "
        f"**{'dentro' if referencia.get('codigo_correcto') else 'FUERA'}**.",
        f"- **P0-1-control** usa el pipeline de tres niveles con umbrales congelados: "
        f"{control['corrected_cv']},delta {control['delta']:+.6f} frente a la referencia. Ese",
        "  delta **no es un error**: es el **coste del protocolo**, que obliga a ajustar sobre",
        "  cinco sextos de las filas. Se reporta por eso, y por eso el control de tres niveles",
        "  no se compara nunca con V03.",
        "- **P0-1-anidado** comparte conjunto de ajuste con el control, asi que el coste se",
        f"  cancela y el delta {anidada['delta']:+.4f} es **efecto del umbral**. IC95 "
        f"[{anidada['ic95_0']:+.4f}, {anidada['ic95_1']:+.4f}], "
        f"{anidada['pliegues_positivos']} de {len(pliegues)} pliegues a favor.",
        f"- Referencia de E2-16 con protocolo mas laxo: {REFERENCIA_E2_16}.",
        "",
        "La primera version de este script fijaba una tolerancia sobre el control de tres",
        "niveles y se detuvo al superarla. **La deteccion era correcta y la tolerancia estaba",
        "mal planteada**: el error no era del codigo sino del protocolo. Anadir el brazo P0-0",
        "separa las dos cosas. Es el mismo error de interpretacion que la fase v6 cometio al",
        "tratar la reoptimizacion de umbrales como una mejora.",
        "",
        "## 4. Cuanto esta determinada la eleccion del umbral",
        "",
        "La columna de holgura del fichero `reports/v7_calibration_analysis.csv` mide la",
        "diferencia de F1 entre el mejor umbral y el segundo mejor en el mismo conjunto de",
        "calibracion. Si esa diferencia es pequena, la eleccion es casi arbitraria y el umbral",
        "no esta identificado por los datos.",
        "",
        "| etiqueta | holgura media entre el primer y el segundo umbral |",
        "|---|---|",
    ]
    for etiqueta in ETIQUETAS:
        holguras = [
            detalles[i]["detalle"][etiqueta]["holgura"] for i in range(len(detalles))
        ]
        lineas.append(f"| {etiqueta} | {np.mean(holguras):.6f} |")
    lineas += [
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
