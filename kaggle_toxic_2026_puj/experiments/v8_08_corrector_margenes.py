"""P7 de la fase v8 (V8-08): corrector sobre los margenes y el desacuerdo entre pliegues.

Contexto
--------
Esta es la tercera de las tres lineas que solo mueven la capa de decision, y su techo lo acotan
dos precedentes duros: el apilamiento de la sonda v08, que dio 0.78163 en el externo frente a
los 0.78306 de V03, y la fusion tardia de los candidatos v09 a v11, que tampoco superaron la
referencia.

Por que se ejecuta aun asi
--------------------------
El apilamiento fallo **con muchas mas variables**: la sonda v08 uso 44 columnas, entre ellas
puntuaciones, margenes, entropias de cuatro bases y ocho rasgos estructurales. Aqui la pregunta
es distinta y mas estrecha: **los margenes de V03, y solo ellos, ¿llevan informacion que la
decision de V03 no haya usado?** Un segundo modelo con cuatro variables de entrada no es
apilamiento, es un corrector, y la dimension manda en lo que puede aportar.

Que se mide ademas
-----------------
La senal de P9, que es el **desacuerdo entre los cinco modelos de pliegue**, se mide aqui para
no dejarla sin cuantificar. Si los cinco modelos coinciden mucho en una fila, la decision es
segura; si discrepan, es una fila limite. Ese desacuerdo es senal que V03 no usa y que se puede
anadir como columna sin tocar la representacion.

Presupuesto
-----------
Por D-063 el envio es manual y el filtro de detectabilidad es criterio de priorizacion, no
bloqueante. **La submission se escribe siempre que la linea vaya en direccion positiva.**

Uso
---
    /home/carlos/venv/bin/python experiments/v8_08_corrector_margenes.py
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import (  # noqa: E402
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V8-08 (P7)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_corrector_margenes.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_corrector_margenes.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_corrector_margenes.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "toxic_submission_v11_p7_corrector_margenes.csv"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
RUIDO_NETOS = 19.2
REJILLA_ALFA = (0.3, 1.0, 3.0, 10.0)
PLEIGUES_INTERNOS = 3
NUMERO_REPETICIONES = 5
SEMILLA_BASE = 9100

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-P7] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def margenes(probabilidades: np.ndarray) -> np.ndarray:
    """Convierte probabilidades en margenes con signo respecto a su umbral.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz de margenes, una columna por etiqueta.
    """
    return probabilidades - UMBRALES_V03[None, :]


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: textos a transformar.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    matriz = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [vectorizador_word.transform(textos_consulta), vectorizador_char.transform(textos_consulta)],
        format="csr",
    )
    return matriz, consulta


def f1_de(probabilidades: np.ndarray, verdadero: np.ndarray, etiqueta: int) -> float:
    """Calcula la F1 de una etiqueta con su umbral congelado.

    Args:
        probabilidades: vector de probabilidades.
        verdadero: vector binario verdadero.
        etiqueta: indice de la etiqueta, para su umbral.

    Returns:
        Valor de la F1, cero si el denominador es cero.
    """
    pred = (probabilidades >= UMBRALES_V03[etiqueta]).astype(int)
    tp = int(((pred == 1) & (verdadero == 1)).sum())
    fp = int(((pred == 1) & (verdadero == 0)).sum())
    fn = int(((pred == 0) & (verdadero == 1)).sum())
    divisor = 2 * tp + fp + fn
    return (2.0 * tp / divisor) if divisor else 0.0


def main() -> None:
    """Construye los cinco modelos, mide el desacuerdo y ajusta el corrector."""
    inicio = time.time()
    log("P7: corrector sobre los margenes y el desacuerdo entre pliegues")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    grupos_dev = grupos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    textos_test = test["text"].astype(str)
    n_filas = len(y)
    log(f"   desarrollo {n_filas} filas | prueba {len(test)} filas")

    log("1/6 control de codigo contra el OOF canonico de V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    ids_iguales = bool(
        (oof["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all()
    )
    predicciones_canonicas = aplicar_umbrales(proba_canonica)
    metrica_canonica = metrica_macro(y, predicciones_canonicas)
    control_correcto = (
        abs(metrica_canonica - REFERENCIA_V03) <= TOLERANCIA_CONTROL and ids_iguales
    )
    log(f"   OOF canonico {metrica_canonica:.6f} | ids en orden {ids_iguales}")
    if not control_correcto:
        log("PARADA: el control falla y no se sigue.")
        return

    log("2/6 cinco repeticiones de los pliegues, para tener cinco votos honestos por fila")
    # Con una sola particion, cada fila la predicen cinco modelos pero **cuatro de ellos la
    # entrenaron**, de modo que el desacuerdo estaria contaminado por resubstitucion. Para que
    # los cinco votos de cada fila sean honestos hacen falta cinco particiones distintas, y cada
    # fila recibe en cada repeticion la prediccion del modelo cuyo pliegue de validacion la
    # contiene. Eso es validacion cruzada repetida, que ademas es una de las tecnicas nuevas
    # del material de clase que quedaban sin medir.
    numero_de_modelos = len(pliegues)
    probabilidades_por_modelo = np.zeros(
        (NUMERO_REPETICIONES := 5, n_filas, 3), dtype=np.float64
    )
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=numero_de_modelos, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, validacion in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, consulta = construir_espacio(textos.iloc[entrenamiento], textos)
            for etiqueta in range(3):
                estimador = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                estimador.fit(matriz, y[entrenamiento, etiqueta])
                probabilidades_por_modelo[repeticion, validacion, etiqueta] = (
                    estimador.predict_proba(consulta[validacion])[:, 1]
                )
        log(f"   repeticion {repeticion + 1} de {NUMERO_REPETICIONES} completa")

    log("3/6 desacuerdo entre las cinco predicciones honestas de cada fila")
    votos = (probabilidades_por_modelo >= UMBRALES_V03[None, None, :]).sum(axis=0)
    desacuerdo = np.abs(votos - 2.5) / 2.5
    poco_contentiosas = int((desacuerdo.max(axis=1) < 0.4).sum())
    muy_contentiosas = int((desacuerdo.max(axis=1) < 0.8).sum())
    log(
        "   desacuerdo medio por etiqueta: "
        + ", ".join(f"{e} {desacuerdo[:, i].mean():.4f}" for i, e in enumerate(ETIQUETAS))
    )
    log(
        f"   filas con las tres etiquetas por debajo de 0,4: {poco_contentiosas} de {n_filas} "
        f"({poco_contentiosas / n_filas * 100:.1f} %) | por debajo de 0,8: {muy_contentiosas} "
        f"({muy_contentiosas / n_filas * 100:.1f} %)"
    )

    promedio = probabilidades_por_modelo.mean(axis=0)
    log(
        f"   comprobacion: el promedio de las cinco repeticiones da "
        f"{metrica_macro(y, aplicar_umbrales(promedio)):.6f}, frente a los {REFERENCIA_V03} "
        f"de V03. Es un OOF legitimo, porque cada fila la predice en cada repeticion un modelo "
        f"que no la entreno."
    )

    log("4/6 corrector sobre los margenes y el desacuerdo")
    diseno_margen = np.hstack([margenes(proba_canonica), desacuerdo])
    disenos = {
        "corrector_margen": margenes(proba_canonica),
        "corrector_margen_y_desacuerdo": diseno_margen,
        "solo_desacuerdo": desacuerdo,
    }
    oof_corrector: dict[str, np.ndarray] = {
        clave: np.zeros((n_filas, 3), dtype=np.float64) for clave in disenos
    }
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        interno = StratifiedGroupKFold(
            n_splits=PLEIGUES_INTERNOS, shuffle=True, random_state=RANDOM_STATE
        )
        particiones = list(
            interno.split(
                diseno_margen[entrenamiento], combinaciones(desarrollo.iloc[entrenamiento]),
                groups=grupos_dev[entrenamiento],
            )
        )
        for etiqueta in range(3):
            mejor = {"f1": -1.0, "alfa": REJILLA_ALFA[0]}
            for alfa in REJILLA_ALFA:
                acumulado = 0.0
                for indices_t, indices_v in particiones:
                    estimador = RidgeClassifier(alpha=alfa, class_weight="balanced")
                    estimador.fit(
                        diseno_margen[entrenamiento][indices_t][:, [etiqueta]],
                        y[entrenamiento, etiqueta][indices_t],
                    )
                    proba = _a_probabilidad(
                        estimador.decision_function(
                            diseno_margen[entrenamiento][indices_v][:, [etiqueta]]
                        )
                    )
                    valor = f1_de(proba, y[entrenamiento, etiqueta][indices_v], etiqueta)
                    acumulado += valor
                    if valor > mejor["f1"]:
                        mejor = {"f1": valor, "alfa": alfa}
            for clave, diseno in disenos.items():
                estimador = RidgeClassifier(alpha=mejor["alfa"], class_weight="balanced")
                estimador.fit(diseno[entrenamiento][:, [etiqueta]], y[entrenamiento, etiqueta])
                oof_corrector[clave][validacion, etiqueta] = _a_probabilidad(
                    estimador.decision_function(diseno[validacion][:, [etiqueta]])
                )
        log(f"   pliegue {numero + 1}: corrector ajustado")

    log("4b/6 el promedio de las cinco repeticiones, medido como OOF legitimo")
    oof_promedio = aplicar_umbrales(promedio)
    metrica_promedio = metrica_macro(y, oof_promedio)
    diferencia_promedio = metrica_promedio - metrica_canonica
    aciertos_promedio = [
        int((oof_promedio[val] == y[val]).sum()
            - (predicciones_canonicas[val] == y[val]).sum())
        for _, val in pliegues
    ]
    bootstrap_promedio = bootstrap_diferencia_pareada(
        y, oof_promedio, predicciones_canonicas, repeticiones=2000
    )
    cambio_promedio = oof_promedio != predicciones_canonicas
    celdas_promedio = int(cambio_promedio.sum())
    tasa_promedio = int((oof_promedio == y)[cambio_promedio].sum()) / celdas_promedio
    minimo_promedio = 0.5 + RUIDO_NETOS / (2.0 * celdas_promedio)
    variante_promedio = {
        "clave": "promedio_de_repeticiones",
        "metrica": round(metrica_promedio, 6),
        "diferencia": round(diferencia_promedio, 6),
        "ic95_0": float(bootstrap_promedio["ic95"][0]),
        "ic95_1": float(bootstrap_promedio["ic95"][1]),
        "pliegues_positivos": int(sum(1 for a in aciertos_promedio if a > 0)),
        "f1_por_etiqueta": metricas_multietiqueta(y, oof_promedio)["f1_por_etiqueta"],
        "celdas": celdas_promedio,
        "tasa_acierto": round(tasa_promedio, 4),
        "minimo_requerido": round(minimo_promedio, 4),
        "filtro_pasa": bool(tasa_promedio >= minimo_promedio),
    }
    variante_promedio["gate_delta"] = diferencia_promedio >= GATE_DELTA
    variante_promedio["gate_ic95"] = variante_promedio["ic95_0"] > 0
    variante_promedio["gate_pliegues"] = variante_promedio["pliegues_positivos"] >= GATE_PLIEGUES
    variante_promedio["gate_supera"] = (
        variante_promedio["gate_delta"] and variante_promedio["gate_ic95"]
        and variante_promedio["gate_pliegues"]
    )
    log(
        f"   metrica {metrica_promedio:.6f} | diferencia {diferencia_promedio:+.6f} | IC95 "
        f"[{variante_promedio['ic95_0']:+.5f}, {variante_promedio['ic95_1']:+.5f}] | pliegues "
        f"{variante_promedio['pliegues_positivos']}/5 | celdas {celdas_promedio} | tasa "
        f"{tasa_promedio:.4f} frente a {minimo_promedio:.4f} | gate "
        f"{'SUPERA' if variante_promedio['gate_supera'] else 'no supera'}"
    )

    log("5/6 evaluacion de las tres ramas")
    variantes: list[dict[str, Any]] = []
    for clave, probabilidades in oof_corrector.items():
        predicciones = aplicar_umbrales(probabilidades)
        metrica = metrica_macro(y, predicciones)
        diferencia = metrica - metrica_canonica
        aciertos = [
            int((predicciones[val] == y[val]).sum()
                - (predicciones_canonicas[val] == y[val]).sum())
            for _, val in pliegues
        ]
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, predicciones_canonicas, repeticiones=2000
        )
        cambio = predicciones != predicciones_canonicas
        n_celdas = int(cambio.sum())
        if n_celdas:
            tasa = int((predicciones == y)[cambio].sum()) / n_celdas
            minimo = 0.5 + RUIDO_NETOS / (2.0 * n_celdas)
            pasa = tasa >= minimo
        else:
            tasa, minimo, pasa = float("nan"), float("nan"), False
        variante = {
            "clave": clave, "metrica": round(metrica, 6), "diferencia": round(diferencia, 6),
            "ic95_0": float(bootstrap["ic95"][0]), "ic95_1": float(bootstrap["ic95"][1]),
            "pliegues_positivos": int(sum(1 for a in aciertos if a > 0)),
            "f1_por_etiqueta": metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"],
            "celdas": n_celdas, "tasa_acierto": round(tasa, 4),
            "minimo_requerido": round(minimo, 4), "filtro_pasa": pasa,
        }
        variante["gate_delta"] = diferencia >= GATE_DELTA
        variante["gate_ic95"] = variante["ic95_0"] > 0
        variante["gate_pliegues"] = variante["pliegues_positivos"] >= GATE_PLIEGUES
        variante["gate_supera"] = (
            variante["gate_delta"] and variante["gate_ic95"] and variante["gate_pliegues"]
        )
        variantes.append(variante)
        log(
            f"   {clave}: metrica {metrica:.6f} | diferencia {diferencia:+.6f} | IC95 "
            f"[{variante['ic95_0']:+.5f}, {variante['ic95_1']:+.5f}] | pliegues "
            f"{variante['pliegues_positivos']}/5 | celdas {n_celdas} | tasa {tasa:.4f} frente "
            f"a {minimo:.4f} | gate {'supera' if variante['gate_supera'] else 'no supera'}"
        )

    log("6/6 predicciones sobre la prueba, integridad e informe")
    SUBMISSION_ENSEMBLE = (
        PROJECT_ROOT / "submissions" / "toxic_submission_v12_p7_promedio_repeticiones.csv"
    )
    matriz_completa, consulta_prueba = construir_espacio(textos, textos_test)
    probabilidades_prueba = np.zeros((numero_de_modelos, len(test), 3), dtype=np.float64)
    for indice, (entrenamiento, _) in enumerate(pliegues):
        for etiqueta in range(3):
            estimador = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(
                matriz_completa[entrenamiento], y[entrenamiento, etiqueta]
            )
            probabilidades_prueba[indice, :, etiqueta] = estimador.predict_proba(
                consulta_prueba
            )[:, 1]
    promedio_prueba = probabilidades_prueba.mean(axis=0)
    votos_prueba = (probabilidades_prueba >= UMBRALES_V03[None, None, :]).sum(axis=0)
    desacuerdo_prueba = np.abs(votos_prueba - 2.5) / 2.5
    diseno_prueba = np.hstack([margenes(promedio_prueba), desacuerdo_prueba])
    proba_corrector_prueba = np.zeros((len(test), 3), dtype=np.float64)
    for etiqueta in range(3):
        estimador = RidgeClassifier(alpha=REJILLA_ALFA[1], class_weight="balanced")
        estimador.fit(diseno_margen[:, [etiqueta]], y[:, etiqueta])
        proba_corrector_prueba[:, etiqueta] = _a_probabilidad(
            estimador.decision_function(diseno_prueba[:, [etiqueta]])
        )
    pred_test = aplicar_umbrales(proba_corrector_prueba)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = _verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    for fila in integridad:
        log(f"   {fila['comprobacion']:38s} {fila['resultado']:8s} {fila['detalle']}")
    comparacion = _comparar_con_v03(SUBMISSION)
    log(f"   corrector: celdas que cambian frente a V03 {comparacion['celdas_que_cambian']} de "
        f"{comparacion['celdas_totales']}")
    proba_ensemble = np.zeros((NUMERO_REPETICIONES, len(test), 3), dtype=np.float64)
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=numero_de_modelos, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, _ in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz_repe, consulta_repe = construir_espacio(
                textos.iloc[entrenamiento], textos_test
            )
            for etiqueta in range(3):
                estimador = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                estimador.fit(matriz_repe, y[entrenamiento, etiqueta])
                proba_ensemble[repeticion, :, etiqueta] = estimador.predict_proba(
                    consulta_repe
                )[:, 1]
    marco_ensemble = pd.DataFrame(
        {e: aplicar_umbrales(proba_ensemble.mean(axis=0))[:, i] for i, e in enumerate(ETIQUETAS)}
    )
    marco_ensemble.insert(0, "id", test["id"].to_numpy())
    marco_ensemble.to_csv(SUBMISSION_ENSEMBLE, index=False, lineterminator="\n")
    integridad_ensemble = _verificar_integridad(marco_ensemble, test, train)
    integridad_ensemble_correcta = all(
        f["resultado"] == "correcto" for f in integridad_ensemble
    )
    comparacion_ensemble = _comparar_con_v03(SUBMISSION_ENSEMBLE)
    log(f"   ensemble de repeticiones: celdas que cambian frente a V03 "
        f"{comparacion_ensemble['celdas_que_cambian']} de "
        f"{comparacion_ensemble['celdas_totales']} en "
        f"{comparacion_ensemble['filas_que_cambian']} filas | integridad "
        f"{'correcta' if integridad_ensemble_correcta else 'CON FALLOS'}")
    _informe(
        variantes=variantes, integridad=integridad, integridad_correcta=integridad_correcta,
        comparacion=comparacion, comparacion_ensemble=comparacion_ensemble,
        integridad_ensemble=integridad_ensemble,
        integridad_ensemble_correcta=integridad_ensemble_correcta,
        variante_promedio=variante_promedio, desacuerdo=desacuerdo,
        poco_contentiosas=poco_contentiosas, n_filas=n_filas, inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        columnas = [
            "clave", "metrica", "diferencia", "ic95_0", "ic95_1", "pliegues_positivos",
            "gate_delta", "gate_ic95", "gate_pliegues", "gate_supera", "celdas",
            "tasa_acierto", "minimo_requerido", "filtro_pasa",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        for v in variantes:
            escritor.writerow({k: v.get(k, "") for k in columnas})
    with REPORTE_JSON.open("w", encoding="utf-8") as archivo:
        json.dump(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "desacuerdo_medio": {
                    e: round(float(desacuerdo[:, i].mean()), 6)
                    for i, e in enumerate(ETIQUETAS)
                },
                "variantes": variantes, "integridad": integridad,
                "integridad_correcta": integridad_correcta, "comparacion_con_v03": comparacion,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "El corrector usa el margen de su propia etiqueta y, en una variante, el "
                    "desacuerdo entre los cinco modelos de pliegue en esa etiqueta, que es la "
                    "senal de P9. El alfa se elige con sub-pliegues internos por pliegue."
                ),
            },
            archivo, ensure_ascii=False, indent=2,
        )
        archivo.write("\n")
    log(f"P7 completada en {round(time.time() - inicio, 1)} s")


def _a_probabilidad(puntajes: np.ndarray) -> np.ndarray:
    """Convierte puntuaciones de decision en probabilidades con la funcion logistica.

    Args:
        puntajes: vector de puntuaciones.

    Returns:
        Vector de probabilidades.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(puntajes, dtype=np.float64)))


def _verificar_integridad(
    marco: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame
) -> list[dict[str, str]]:
    """Comprueba la integridad del fichero de envio.

    Args:
        marco: marco de datos escrito.
        test: conjunto de prueba original.
        train: conjunto de entrenamiento.

    Returns:
        Lista de comprobaciones con resultado y detalle.
    """
    filas: list[dict[str, str]] = []

    def anadir(comprobacion: str, correcto: bool, detalle: str) -> None:
        filas.append({"comprobacion": comprobacion,
                      "resultado": "correcto" if correcto else "FALLO", "detalle": detalle})

    anadir("numero de filas", len(marco) == len(test),
           f"{len(marco)} frente a {len(test)} en la prueba")
    anadir("nombres y orden de columnas", list(marco.columns) == ["id"] + list(ETIQUETAS),
           f"{list(marco.columns)}")
    valores: set[int] = set()
    for etiqueta in ETIQUETAS:
        valores |= set(np.unique(marco[etiqueta].to_numpy()).tolist())
    anadir("valores de etiqueta", valores <= {0, 1}, f"{sorted(valores)}")
    anadir("sin valores ausentes", not bool(marco.isna().any().any()),
           f"{int(marco.isna().sum().sum())} celdas ausentes")
    anadir("sin identificadores duplicados", not bool(marco["id"].duplicated().any()),
           f"{int(marco['id'].duplicated().sum())} repetidos")
    anadir("identificadores en el orden de la prueba",
           bool((marco["id"].to_numpy() == test["id"].to_numpy()).all()),
           "coinciden con el orden original")
    anadir("filas distintas del entrenamiento", len(train) != len(test),
           f"{len(train)} frente a {len(test)}")
    return filas


def _comparar_con_v03(ruta: Path) -> dict[str, Any]:
    """Compara la submission generada con la de V03, celda a celda.

    Args:
        ruta: ruta de la submission nueva.

    Returns:
        Diccionario con el numero de celdas que cambian y con que proporcion.
    """
    nueva = pd.read_csv(ruta).set_index("id")
    vieja = pd.read_csv(
        PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
    ).set_index("id").loc[nueva.index]
    cambio = nueva[list(ETIQUETAS)].to_numpy() != vieja[list(ETIQUETAS)].to_numpy()
    return {
        "celdas_totales": int(cambio.size), "celdas_que_cambian": int(cambio.sum()),
        "filas_que_cambian": int(cambio.any(axis=1).sum()),
        "por_etiqueta": {e: int(cambio[:, i].sum()) for i, e in enumerate(ETIQUETAS)},
    }


def _informe(
    variantes: list[dict[str, Any]],
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    comparacion: dict[str, Any],
    comparacion_ensemble: dict[str, Any],
    integridad_ensemble: list[dict[str, str]],
    integridad_ensemble_correcta: bool,
    variante_promedio: dict[str, Any],
    desacuerdo: np.ndarray,
    poco_contentiosas: int,
    n_filas: int,
    inicio: float,
) -> None:
    """Escribe el informe de la linea P7.

    Args:
        variantes: resultado de las tres ramas.
        integridad: comprobaciones del fichero de envio.
        integridad_correcta: si todas pasaron.
        comparacion: diferencia celda a celda con la submission de V03.
        comparacion_ensemble: lo mismo para la submission del ensemble.
        integridad_ensemble: comprobaciones de la submission del ensemble.
        integridad_ensemble_correcta: si todas pasaron.
        variante_promedio: resultado del promedio de las cinco repeticiones.
        desacuerdo: desacuerdo entre modelos por fila y etiqueta.
        poco_contentiosas: filas con las tres etiquetas de desacuerdo bajo.
        n_filas: numero de filas del desarrollo.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# P7: corrector sobre los margenes y el desacuerdo entre pliegues (V8-08)",
        "",
        f"Generado por `experiments/v8_08_corrector_margenes.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que no es el apilamiento que ya fallo",
        "",
        "El apilamiento de la sonda v08 uso 44 columnas y dio 0.78163 en el externo frente a los",
        "0.78306 de V03. Aqui la pregunta es mas estrecha: **los margenes de V03, y solo ellos,",
        "¿llevan informacion que la decision de V03 no haya usado?** Un segundo modelo con",
        "cuatro variables de entrada no es apilamiento, es un corrector, y la dimension manda en",
        "lo que puede aportar.",
        "",
        "## 2. El desacuerdo entre los cinco modelos, que es tambien la senal de P9",
        "",
        "Si los cinco modelos coinciden mucho en una fila, la decision es segura; si discrepan,",
        "es una fila limite. Ese desacuerdo es senal que V03 no usa y que se puede anadir como",
        "columna sin tocar la representacion.",
        "",
        "| etiqueta | desacuerdo medio |",
        "|---|---|",
    ]
    for i, etiqueta in enumerate(ETIQUETAS):
        lineas.append(f"| {etiqueta} | {float(desacuerdo[:, i].mean()):.4f} |")
    lineas += [
        "",
        f"Filas con las tres etiquetas por debajo de 0,4 de desacuerdo: **{poco_contentiosas} de",
        f"{n_filas}**, un {poco_contentiosas / n_filas * 100:.1f} %.",
        "",
        "## 3. Resultados del corrector",
        "",
        "| rama | metrica | diferencia | IC95 | pliegues | celdas | tasa | filtro |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in variantes + [variante_promedio]:
        lineas.append(
            f"| {v['clave']} | {v['metrica']:.6f} | {v['diferencia']:+.6f} | "
            f"[{v['ic95_0']:+.5f}, {v['ic95_1']:+.5f}] | {v['pliegues_positivos']}/5 | "
            f"{v['celdas']} | {v['tasa_acierto']:.4f} frente a {v['minimo_requerido']:.4f} | "
            f"{'pasa' if v['filtro_pasa'] else 'no pasa'} |"
        )
    lineas += [
        "",
        "## 4. Integridad del fichero de envio",
        "",
        f"Resultado global: **{'todas pasan' if integridad_correcta else 'HAY FALLOS'}**.",
        "",
        "| comprobacion | resultado | detalle |",
        "|---|---|---|",
    ]
    for fila in integridad:
        lineas.append(f"| {fila['comprobacion']} | {fila['resultado']} | {fila['detalle']} |")
    lineas += [
        "",
        "## 5. Cuanto se aparta de V03",
        "",
        f"- celdas que cambian: {comparacion['celdas_que_cambian']} de "
        f"{comparacion['celdas_totales']}, en {comparacion['filas_que_cambian']} filas",
        "- por etiqueta: "
        + ", ".join(f"{e} {comparacion['por_etiqueta'][e]}" for e in ETIQUETAS),
        "",
        "## 6. Decision",
        "",
        "Por D-063 el envio es manual y el filtro es priorizacion, no bloqueante.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
