"""P1 de la fase v8 (V8-01): promedio de los cinco modelos de pliegue al predecir.

Contexto
--------
Esta es la unica hipotesis que la fase v7 dejo abierta, y se dejo abierta por una razon
concreta: **su comportamiento externo no lo captura el protocolo local**. El conjunto de
desarrollo ya tiene una prediccion de cada fila hecha por el modelo que no la entreno, de
modo que reconstruir ese mismo promedio sobre el desarrollo no aporta informacion nueva. Lo
unico que queda por medir es lo que ocurre sobre el conjunto de prueba, y eso exige enviar.

Que construye
------------
Cinco modelos V03 equivalentes, cada uno entrenado sobre cuatro de los cinco pliegues del
protocolo oficial::

    modelo del pliegue 1 entrenado con los pliegues 2, 3, 4 y 5
    modelo del pliegue 2 entrenado con los pliegues 1, 3, 4 y 5
    modelo del pliegue 3 entrenado con los pliegues 1, 2, 4 y 5
    modelo del pliegue 4 entrenado con los pliegues 1, 2, 3 y 5
    modelo del pliegue 5 entrenado con los pliegues 1, 2, 3 y 4

Se mantiene exactamente lo de V03: misma representacion, mismos hiperparametros, mismos
umbrales congelados de 0.44, 0.51 y 0.46. No hay metamodelo, no se ajustan pesos, no se
optimiza nada. La prediccion sobre el conjunto de prueba es **la media aritmetica de las
cinco probabilidades**, etiqueta a etiqueta.

Por que el promedio y no el modelo unico
----------------------------------------
Un modelo de regresion logistica ajustado sobre un quinto menos de filas tiene mas varianza
que uno ajustado sobre todas. Promediar cinco reduce esa varianza, y los pliegues son
disjuntos por construccion, asi que la hipotesis es plausible. Pero la media tambien
**contrae las probabilidades hacia el centro**, y ese es el riesgo que PROVIDE-03 describio
en el track H de la fase v6: los umbrales congelados de V03 estan calibrados para **un**
modelo individual, de modo que un umbral que separa bien las probabilidades de un modelo
puede no separarlas igual despues del promedio. Por eso el script mide el efecto de escala
en vez de suponerlo.

Que se espera en local, y por que no es informativo
---------------------------------------------------
La columna de desarrollo de este experimento es, por construccion, el promedio por pliegue:
cada fila la predice el unico modelo que no la entreno. Esa es precisamente la definicion
del OOF canonico, de modo que la metrica local de este script **debe coincidir con la de V03**
y por tanto no aporta informacion sobre la hipotesis. La predicion sobre la prueba si es
nueva, y esa es la que decide. La comparacion entre el modelo unico y el promedio se calcula
**solo como registro**: no se usa para ajustar nada, porque ajustar con ella seria el
sobreajuste que la fase v6 midio.

Control de calidad antes de cualquier envio
-------------------------------------------
- **Control de codigo**: el mismo pipeline ajustado sobre todo el desarrollo debe reproducir
  0.777211 con los umbrales congelados. Es el mismo brazo de referencia que valido el codigo
  en la fase v7. Si no lo hace, no se genera nada.
- **Integridad del fichero de envio**: numero de filas, nombres y orden de columnas, valores
  de etiqueta, ausencia de nulos y ausencia de identificadores duplicados.
- **Desplazamiento de la prediccion**: la tasa de positivos predicha en la prueba se compara
  con la que produce el modelo completo sobre esa misma prueba, que es la unica comparacion
  honesta porque ambas salen de la misma fila.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_01_ensamble_folds.py
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
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import ETIQUETAS, metrica_macro, metricas_multietiqueta  # noqa: E402
from src.splits import (  # noqa: E402
    RANDOM_STATE,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V8-01 (P1)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_ensamble_folds.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_ensamble_folds.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_ensamble_folds.json"
REPORTE_INTEGRIDAD = PROJECT_ROOT / "reports" / "v8_integridad_submission.csv"
SUBMISSION = PROJECT_ROOT / "submissions" / "toxic_submission_v08_p1_fold_ensemble.csv"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"
FIGURA = FIGURAS / "fig_69_ensamble_folds_v8.png"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
TOLERANCIA_CONTROL = 0.0005
NUMERO_DE_MODELOS = 5

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P1] {mensaje}", flush=True)


def ajustar_una_vez(
    textos_entrenamiento: pd.Series,
    textos_consulta: list[pd.Series],
    y: np.ndarray,
    indices_entrenamiento: np.ndarray,
    indices_consulta: list[np.ndarray],
) -> list[np.ndarray]:
    """Ajusta un modelo V03 equivalente y predice sobre varias consultas con el mismo.

    El vectorizador se ajusta una sola vez sobre el entrenamiento y se reutiliza para todas
    las consultas. Es la unica forma de que las columnas de desarrollo y de prueba provengan
    del mismo modelo, que es justo lo que exige una comparacion entre ellas.

    Args:
        textos_entrenamiento: textos del ajuste.
        textos_consulta: lista de series de texto, una por consulta.
        y: matriz binaria de etiquetas.
        indices_entrenamiento: indices globales de entrenamiento.
        indices_consulta: lista de indices globales, una por consulta.

    Returns:
        Lista de matrices de probabilidad, una por consulta, en el mismo orden.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    matriz = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento.iloc[indices_entrenamiento]),
            vectorizador_char.fit_transform(textos_entrenamiento.iloc[indices_entrenamiento]),
        ],
        format="csr",
    )
    consultas = [
        sparse.hstack(
            [
                vectorizador_word.transform(serie),
                vectorizador_char.transform(serie),
            ],
            format="csr",
        )
        for serie in textos_consulta
    ]
    salidas = [np.zeros((len(idx), 3), dtype=np.float64) for idx in indices_consulta]
    for etiqueta in range(3):
        estimador = LogisticRegression(**PARAMETROS_LR)
        estimador.fit(matriz, y[indices_entrenamiento, etiqueta])
        for salida, consulta in zip(salidas, consultas, strict=True):
            salida[:, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return salidas


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def main() -> None:
    """Construye los cinco modelos, verifica la integridad y genera el fichero de envio."""
    inicio = time.time()
    log("P1: promedio de los cinco modelos de pliegue al predecir")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos_dev = desarrollo["text"].astype(str)
    textos_test = test["text"].astype(str)
    todos_dev = np.arange(len(desarrollo), dtype=int)
    log(
        f"   desarrollo {len(y)} filas | prueba {len(test)} filas | "
        f"{len(pliegues)} pliegues del protocolo"
    )

    log("1/6 los cinco modelos, cada uno entrenado sobre cuatro pliegues")
    probabilidades_dev = np.zeros((len(y), 3), dtype=np.float64)
    probabilidades_test = np.zeros((len(test), 3), dtype=np.float64)
    detalle_modelos: list[dict[str, Any]] = []
    for numero, (indices_entrenamiento, indices_validacion) in enumerate(pliegues):
        # Un solo ajuste por modelo. El mismo objeto predice su pliegue de validacion y la
        # prueba, de modo que las dos columnas son comparables por construccion.
        proba_validacion, proba_test = ajustar_una_vez(
            textos_dev,
            [textos_dev.iloc[indices_validacion], textos_test],
            y,
            indices_entrenamiento,
            [indices_validacion, np.arange(len(test), dtype=int)],
        )
        probabilidades_dev[indices_validacion] = proba_validacion
        probabilidades_test += proba_test / NUMERO_DE_MODELOS
        metrica_pliegue = metrica_macro(
            y[indices_validacion], aplicar_umbrales(proba_validacion)
        )
        detalle_modelos.append(
            {
                "modelo": numero + 1,
                "n_entrenamiento": int(len(indices_entrenamiento)),
                "n_validacion": int(len(indices_validacion)),
                "metrica_en_su_validacion": round(metrica_pliegue, 6),
            }
        )
        log(
            f"   modelo {numero + 1}: {len(indices_entrenamiento)} filas de entrenamiento, "
            f"metrica en su pliegue {metrica_pliegue:.6f}"
        )

    log("2/6 control de codigo contra el OOF canonico de V03")
    # El control correcto es comparar el promedio por pliegue reconstruido contra el OOF
    # guardado en disco, fila a fila. Es una comprobacion de identidad de entrenamiento y
    # valida mas que la cifra de la metrica: si las probabilidades coinciden celda a celda,
    # entonces el pipeline es el de V03 y ninguna cifra posterior puede deberse al codigo.
    oof_canonico = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof_canonico["proba"].astype(np.float64)
    if proba_canonica.shape != probabilidades_dev.shape:
        log(
            f"PARADA: el OOF canonico tiene forma {proba_canonica.shape} y el reconstruido "
            f"{probabilidades_dev.shape}, de modo que no son comparables."
        )
        _informe_fallo(float("nan"), 0.0)
        return
    diferencia_celda = float(np.abs(probabilidades_dev - proba_canonica).max())
    ids_iguales = bool((oof_canonico["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all())
    metrica_promedio = metrica_macro(y, aplicar_umbrales(probabilidades_dev))
    metrica_canonica = metrica_macro(y, aplicar_umbrales(proba_canonica))
    control_correcto = (
        abs(metrica_promedio - REFERENCIA_V03) <= TOLERANCIA_CONTROL
        and diferencia_celda <= 1e-9
        and ids_iguales
    )
    log(
        f"   metrica del promedio por pliegue: {metrica_promedio:.6f} frente a "
        f"{REFERENCIA_V03} (tolerancia {TOLERANCIA_CONTROL:.6f})"
    )
    log(f"   metrica del OOF canonico cargado: {metrica_canonica:.6f}")
    log(
        f"   diferencia maxima por celda frente al OOF canonico: {diferencia_celda:.3e} | "
        f"identificadores en el mismo orden: {ids_iguales}"
    )
    log(f"   veredicto del control: {'dentro' if control_correcto else 'FUERA'}")
    if not control_correcto:
        log(
            "PARADA: el control falla. El codigo no reproduce el OOF canonico de V03, de modo "
            "que no es fiable y no se genera ningun fichero de envio."
        )
        _informe_fallo(metrica_promedio, diferencia_celda)
        return

    log("3/6 efecto de escala del promedio")
    metrica_promedio = metrica_macro(y, aplicar_umbrales(probabilidades_dev))
    f1_promedio = metricas_multietiqueta(y, aplicar_umbrales(probabilidades_dev))["f1_por_etiqueta"]
    desviacion = float(np.std(probabilidades_dev))
    log(f"   desviacion tipica de las probabilidades promediadas: {desviacion:.6f}")
    for etiqueta in ETIQUETAS:
        log(f"   {etiqueta}: F1 del promedio {f1_promedio[etiqueta]:.4f}")

    log("4/6 predicciones sobre la prueba y desplazamiento respecto del modelo completo")
    pred_test = aplicar_umbrales(probabilidades_test)
    # Modelo auxiliar ajustado sobre todo el desarrollo. Solo sirve para cuantificar cuanto
    # desplaza el promedio la prediccion sobre la misma prueba. Su metrica medida sobre el
    # desarrollo no es comparable con la referencia: seria resubstitucion, no validacion.
    (prob_dev_completo, prob_test_completo) = ajustar_una_vez(
        textos_dev,
        [textos_dev, textos_test],
        y,
        todos_dev,
        [todos_dev, np.arange(len(test), dtype=int)],
    )
    metrica_resubstitucion = metrica_macro(y, aplicar_umbrales(prob_dev_completo))
    f1_unico_resub = metricas_multietiqueta(y, aplicar_umbrales(prob_dev_completo))[
        "f1_por_etiqueta"
    ]
    pred_test_completo = aplicar_umbrales(prob_test_completo)
    contraccion = float(np.mean(probabilidades_test) - np.mean(prob_test_completo))
    log(
        f"   metrica de resubstitucion del modelo completo: {metrica_resubstitucion:.6f} "
        "(no comparable con la referencia, es resubstitucion y no validacion)"
    )
    log(
        f"   contraccion media de las probabilidades sobre la prueba al promediar: "
        f"{contraccion:+.6f}"
    )
    desplazamiento = [
        {
            "etiqueta": etiqueta,
            "tasa_promedio": float(pred_test[:, i].mean()),
            "tasa_modelo_completo": float(pred_test_completo[:, i].mean()),
            "tasa_entrenamiento": float(y[:, i].mean()),
        }
        for i, etiqueta in enumerate(ETIQUETAS)
    ]
    for fila in desplazamiento:
        log(
            f"   {fila['etiqueta']}: promedio {fila['tasa_promedio']:.4f} | modelo completo "
            f"{fila['tasa_modelo_completo']:.4f} | entrenamiento {fila['tasa_entrenamiento']:.4f}"
        )

    log("5/6 verificacion de integridad del fichero de envio")
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    for fila in integridad:
        log(f"   {fila['comprobacion']:38s} {fila['resultado']:8s} {fila['detalle']}")
    if not integridad_correcta:
        log("PARADA: la verificacion de integridad falla. El fichero no se entrega.")
    else:
        log(f"   fichero escrito en {SUBMISSION.relative_to(PROJECT_ROOT)}")

    log("6/6 informes y figura")
    _figura(probabilidades_test, prob_test_completo)
    _informe(
        detalle_modelos=detalle_modelos,
        integridad=integridad,
        integridad_correcta=integridad_correcta,
        metrica_promedio=metrica_promedio,
        metrica_canonica=metrica_canonica,
        diferencia_celda=diferencia_celda,
        ids_iguales=ids_iguales,
        metrica_resubstitucion=metrica_resubstitucion,
        contraccion=contraccion,
        desviacion=desviacion,
        f1_promedio=f1_promedio,
        f1_unico_resub=f1_unico_resub,
        desplazamiento=desplazamiento,
        pliegues=pliegues,
        n_test=len(test),
        inicio=inicio,
    )
    _escribir_tablas(detalle_modelos, integridad)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "umbrales_v03": dict(zip(ETIQUETAS, UMBRALES_V03.tolist(), strict=True)),
                "numero_de_modelos": NUMERO_DE_MODELOS,
                "tolerancia_control": TOLERANCIA_CONTROL,
                "control_correcto": control_correcto,
                "metrica_promedio_desarrollo": round(metrica_promedio, 6),
                "metrica_oof_canonico": round(metrica_canonica, 6),
                "diferencia_maxima_por_celda": diferencia_celda,
                "identificadores_en_el_mismo_orden": ids_iguales,
                "metrica_resubstitucion_modelo_completo": round(metrica_resubstitucion, 6),
                "contraccion_de_probabilidades_sobre_prueba": round(contraccion, 6),
                "desviacion_de_probabilidades_promedio": round(desviacion, 6),
                "f1_por_etiqueta_promedio": f1_promedio,
                "f1_por_etiqueta_modelo_completo_resubstitucion": f1_unico_resub,
                "desplazamiento_sobre_prueba": desplazamiento,
                "detalle_modelos": detalle_modelos,
                "integridad": integridad,
                "integridad_correcta": integridad_correcta,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "La metrica local de la columna de desarrollo coincide con la de V03 por "
                    "construccion, porque promedia por pliegue es la definicion del OOF. Lo "
                    "unico nuevo es el promedio sobre la prueba, y eso solo se mide en el "
                    "externo. La comparacion con el modelo unico es registro, no ajuste."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P1 completada en {round(time.time() - inicio, 1)} s")


def verificar_integridad(
    marco: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame
) -> list[dict[str, str]]:
    """Comprueba la integridad del fichero de envio contra el conjunto de prueba.

    Args:
        marco: marco de datos escrito, con identificador y tres etiquetas.
        test: conjunto de prueba original.
        train: conjunto de entrenamiento, para el numero esperado de filas.

    Returns:
        Lista de comprobaciones con resultado y detalle.
    """
    espera_columnas = ["id"] + list(ETIQUETAS)
    filas: list[dict[str, str]] = []

    def anadir(comprobacion: str, correcto: bool, detalle: str) -> None:
        filas.append(
            {
                "comprobacion": comprobacion,
                "resultado": "correcto" if correcto else "FALLO",
                "detalle": detalle,
            }
        )

    anadir(
        "numero de filas",
        len(marco) == len(test),
        f"{len(marco)} filas en el fichero frente a {len(test)} en la prueba",
    )
    anadir(
        "nombres y orden de columnas",
        list(marco.columns) == espera_columnas,
        f"columnas {list(marco.columns)} frente a las esperadas {espera_columnas}",
    )
    valores: set[int] = set()
    for etiqueta in ETIQUETAS:
        valores |= set(np.unique(marco[etiqueta].to_numpy()).tolist())
    anadir("valores de etiqueta", valores <= {0, 1}, f"valores distintos: {sorted(valores)}")
    anadir(
        "sin valores ausentes",
        not bool(marco.isna().any().any()),
        f"{int(marco.isna().sum().sum())} celdas ausentes",
    )
    anadir(
        "sin identificadores duplicados",
        not bool(marco["id"].duplicated().any()),
        f"{int(marco['id'].duplicated().sum())} identificadores repetidos",
    )
    iguales = bool((marco["id"].to_numpy() == test["id"].to_numpy()).all())
    anadir(
        "identificadores en el orden del conjunto de prueba",
        iguales,
        "coinciden con el orden original" if iguales else "el orden difiere",
    )
    anadir(
        "numero de filas distinto del entrenamiento",
        len(train) != len(test),
        f"entrenamiento {len(train)} frente a prueba {len(test)}",
    )
    celdas = marco[list(ETIQUETAS)].to_numpy(dtype=float)
    anadir(
        "etiquetas dentro de cero y uno",
        bool(((celdas >= 0) & (celdas <= 1)).all()),
        "todas las celdas de etiqueta estan en el intervalo",
    )
    anadir(
        "sin cadenas vacias",
        int((marco["id"].astype(str).str.strip() == "").sum()) == 0,
        f"{int((marco['id'].astype(str).str.strip() == '').sum())} identificadores vacios",
    )
    return filas


def _escribir_tablas(
    detalle_modelos: list[dict[str, Any]], integridad: list[dict[str, str]]
) -> None:
    """Escribe las dos tablas de salida del experimento.

    Args:
        detalle_modelos: detalle de cada uno de los cinco modelos.
        integridad: comprobaciones de integridad del fichero de envio.
    """
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = ["modelo", "n_entrenamiento", "n_validacion", "metrica_en_su_validacion"]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(detalle_modelos)
    with REPORTE_INTEGRIDAD.open("w", encoding="utf-8", newline="") as archivo:
        campos = ["comprobacion", "resultado", "detalle"]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(integridad)


def _informe_fallo(metrica_promedio: float, diferencia_celda: float) -> None:
    """Escribe el informe cuando el control falla y no hay nada que entregar.

    Args:
        metrica_promedio: metrica del promedio por pliegue reconstruido.
        diferencia_celda: diferencia maxima por celda frente al OOF canonico.
    """
    lineas = [
        "# P1: el control fallo y no se genero ningun envio",
        "",
        f"Generado por `experiments/v8_01_ensamble_folds.py` ({EXPERIMENTO}).",
        "",
        f"El promedio por pliegue reconstruido dio {metrica_promedio:.6f}, frente a los",
        f"{REFERENCIA_V03} de V03, con una tolerancia declarada de {TOLERANCIA_CONTROL:.6f}.",
        f"La diferencia maxima por celda frente al OOF canonico fue {diferencia_celda:.3e}.",
        "El control quedo **fuera de tolerancia**, de modo que **no se escribio ningun",
        "fichero de envio** y no se gasto entrenamiento sobre una base no verificada. Es el",
        "mismo criterio que se aplico en las fases anteriores.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _figura(probabilidades_test: np.ndarray, prob_test_completo: np.ndarray) -> None:
    """Dibuja el efecto de escala del promedio sobre la prediccion de la prueba.

    Args:
        probabilidades_test: probabilidades del promedio de cinco sobre la prueba.
        prob_test_completo: probabilidades del modelo completo sobre la prueba.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, eje = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(0, 1, 41)
    eje.hist(
        prob_test_completo.ravel(), bins=bins, alpha=0.6, label="modelo completo", density=True
    )
    eje.hist(
        probabilidades_test.ravel(), bins=bins, alpha=0.6, label="promedio de cinco", density=True
    )
    for etiqueta, umbral in zip(ETIQUETAS, UMBRALES_V03, strict=True):
        eje.axvline(umbral, linestyle=":", linewidth=1, label=f"umbral {etiqueta} {umbral}")
    eje.set_xlabel("probabilidad predicha")
    eje.set_ylabel("densidad")
    eje.set_title(
        "Efecto de escala sobre la prueba: contraccion del promedio frente a los umbrales "
        "congelados de V03"
    )
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURA, dpi=150)
    plt.close()


def _informe(
    detalle_modelos: list[dict[str, Any]],
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    metrica_promedio: float,
    metrica_canonica: float,
    diferencia_celda: float,
    ids_iguales: bool,
    metrica_resubstitucion: float,
    contraccion: float,
    desviacion: float,
    f1_promedio: dict[str, float],
    f1_unico_resub: dict[str, float],
    desplazamiento: list[dict[str, float]],
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    n_test: int,
    inicio: float,
) -> None:
    """Escribe el informe del track P1.

    Args:
        detalle_modelos: detalle de cada modelo.
        integridad: comprobaciones de integridad del envio.
        integridad_correcta: si todas las comprobaciones pasaron.
        metrica_promedio: metrica del promedio por pliegue sobre el desarrollo.
        metrica_canonica: metrica del OOF canonico cargado de disco.
        diferencia_celda: diferencia maxima por celda frente al OOF canonico.
        ids_iguales: si los identificadores estan en el mismo orden.
        metrica_resubstitucion: metrica del modelo completo medida sobre su propio ajuste.
        contraccion: contraccion media de las probabilidades sobre la prueba.
        desviacion: desviacion tipica de las probabilidades promediadas.
        f1_promedio: F1 por etiqueta del promedio por pliegue.
        f1_unico_resub: F1 por etiqueta del modelo completo en resubstitucion.
        desplazamiento: tasas de positivos sobre la prueba.
        pliegues: pliegues corregidos.
        n_test: numero de filas de prueba.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# P1: promedio de los cinco modelos de pliegue al predecir (V8-01)",
        "",
        f"Generado por `experiments/v8_01_ensamble_folds.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que es y por que quedaba pendiente de la fase v7",
        "",
        "Cinco modelos V03 equivalentes, cada uno entrenado sobre cuatro de los cinco",
        "pliegues del protocolo oficial. Para el conjunto de prueba se usa **la media",
        "aritmetica de las cinco probabilidades**, etiqueta a etiqueta. Misma representacion,",
        "mismos hiperparametros, mismos umbrales congelados de 0.44, 0.51 y 0.46. Sin",
        "metamodelo y sin ajustar pesos.",
        "",
        "Esta linea quedo abierta en la fase v7 por una razon concreta: **su comportamiento",
        "externo no lo captura el protocolo local**. El desarrollo ya tiene una prediccion de",
        "cada fila hecha por el modelo que no la entreno, de modo que reconstruir ese mismo",
        "promedio sobre el desarrollo no aporta informacion nueva. Lo unico que queda por medir",
        "es lo que ocurre sobre la prueba, y eso exige enviar.",
        "",
        "## 2. Control de calidad del codigo",
        "",
        "El control se hace contra el **OOF canonico guardado en disco**, fila a fila, y no",
        "solo contra la cifra de la metrica. Es una comprobacion de identidad de",
        "entrenamiento: si las probabilidades coinciden celda a celda, el pipeline es",
        "exactamente el de V03 y ninguna cifra posterior puede deberse al codigo.",
        "",
        f"- metrica del promedio por pliegue reconstruido: **{metrica_promedio:.6f}**",
        f"- metrica del OOF canonico cargado: {metrica_canonica:.6f}",
        f"- referencia de V03: {REFERENCIA_V03}, con tolerancia declarada de "
        f"{TOLERANCIA_CONTROL:.6f}",
        f"- diferencia maxima por celda: {diferencia_celda:.3e}",
        f"- identificadores en el mismo orden: {ids_iguales}",
        "",
        "## 3. Los cinco modelos",
        "",
        "| modelo | filas de entrenamiento | filas de validacion | metrica en su pliegue |",
        "|---|---|---|---|",
    ]
    for fila in detalle_modelos:
        lineas.append(
            f"| {fila['modelo']} | {fila['n_entrenamiento']} | {fila['n_validacion']} | "
            f"{fila['metrica_en_su_validacion']:.6f} |"
        )
    lineas += [
        "",
        "Cada modelo se ajusto **una sola vez** y ese mismo objeto predijo su pliegue de",
        "validacion y el conjunto de prueba. Es la unica forma de que las dos columnas sean",
        f"comparables: se ajustan {len(pliegues)} modelos sobre las filas del desarrollo, sin",
        "repetir ningun ajuste.",
        "",
        "## 4. Efecto de escala: el riesgo conocido de esta linea",
        "",
        "El promedio **contrae las probabilidades hacia el centro**. Ese es el riesgo que",
        "PROVIDE-03 describio en el track H de la fase v6: los umbrales congelados de V03",
        "estan calibrados para **un** modelo individual, y un umbral que separa bien las",
        "probabilidades de un modelo puede no separarlas igual despues del promedio. Por eso",
        "el efecto se mide y no se supone.",
        "",
        "Sobre la prueba, el promedio baja la media de las probabilidades en",
        f"{contraccion:+.6f} respecto del modelo completo, y su desviacion tipica sobre el",
        f"desarrollo es {desviacion:.6f}.",
        "",
        "**La metrica local de esta linea no es informativa, y conviene decirlo con",
        "claridad**: la columna de desarrollo es el promedio por pliegue, que es la definicion",
        "del OOF canonico, asi que tiene que coincidir con V03 y asi coincide. Lo unico nuevo",
        "es el promedio sobre la prueba, y eso solo se mide en el externo. Las cifras que",
        "siguen se registran para dejar constancia, no para decidir nada.",
        "",
        "Una cifra que **no** debe compararse con la referencia es la metrica de",
        f"resubstitucion del modelo completo, {metrica_resubstitucion:.6f}: ese modelo se",
        "evalua sobre las mismas filas con las que se ajusto, de modo que la cifra esta",
        "inflada por construccion y no dice nada sobre su capacidad.",
        "",
        "| etiqueta | F1 del promedio por pliegue | F1 del modelo completo en resubstitucion |",
        "|---|---|---|",
    ]
    for etiqueta in ETIQUETAS:
        lineas.append(
            f"| {etiqueta} | {f1_promedio[etiqueta]:.4f} | "
            f"{f1_unico_resub[etiqueta]:.4f} |"
        )
    lineas += [
        "",
        "## 5. Desplazamiento de la prediccion sobre la prueba",
        "",
        "La comparacion honesta es contra el modelo completo **sobre la misma prueba**, porque",
        "las dos columnas salen de la misma fila. Si el promedio contrae las probabilidades, la",
        "tasa de positivos baja respecto del modelo completo, y eso es exactamente lo que",
        "indica la ultima columna.",
        "",
        "| etiqueta | promedio | modelo completo | entrenamiento | diferencia promedio menos completo |",
        "|---|---|---|---|---|",
    ]
    for fila in desplazamiento:
        lineas.append(
            f"| {fila['etiqueta']} | {fila['tasa_promedio']:.4f} | "
            f"{fila['tasa_modelo_completo']:.4f} | {fila['tasa_entrenamiento']:.4f} | "
            f"{fila['tasa_promedio'] - fila['tasa_modelo_completo']:+.4f} |"
        )
    lineas += [
        "",
        f"Nota sobre la lectura de esta tabla: la prueba tiene {n_test} filas y su composicion",
        "no es necesariamente la del desarrollo, de modo que la tasa de positivos de la prueba",
        "no se compara con la del entrenamiento. La comparacion valida es la de las dos",
        "primeras columnas.",
        "",
        "## 6. Verificacion de integridad del fichero",
        "",
        f"Resultado global: **{'todas las comprobaciones pasan' if integridad_correcta else 'HAY FALLOS'}**.",
        "",
        "| comprobacion | resultado | detalle |",
        "|---|---|---|",
    ]
    for fila in integridad:
        lineas.append(
            f"| {fila['comprobacion']} | {fila['resultado']} | {fila['detalle']} |"
        )
    lineas += [
        "",
        "## 7. Que queda pendiente de la decision",
        "",
        "**El fichero esta escrito y verificado, y no se envia automaticamente.** El",
        "resultado de la sonda decide la bifurcacion de la fase:",
        "",
        "- Si el externo mejora de forma material, se congela la arquitectura y se ejecutan",
        "  como maximo dos lineas mas, que son el desacuerdo entre pliegues y la segunda",
        "  semilla.",
        "- Si no mejora, esta linea se cierra **sin variantes** y se entra por la estabilidad",
        "  de features, que es la forma barata de reducir la dimension sin perder senal.",
        "",
        "No se harn variantes de umbral, de pesos ni de semilla antes de conocer ese numero:",
        "la fase v6 demostro que ajustar la decision contra el mismo conjunto no transfiere.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
