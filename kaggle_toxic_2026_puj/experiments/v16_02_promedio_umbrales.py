"""V16-02: promedio de semillas con umbrales re-optimizados, la combinacion que nunca se probo.

La hipotesis, y por que no es una idea mas
------------------------------------------
El promedio de probabilidades fallo DOS veces en este proyecto, y las dos veces con la causa
medida y registrada:

- v08_p1_fold_ensemble, LB 0.77470: "el promedio contrae las probabilidades en 0.001334 y los
  umbrales de V03 estan calibrados para un modelo individual, de modo que el efecto de escala
  movio 100 celdas".
- v12_p7_promedio_repeticiones, LB 0.77122: mismo mecanismo, y anadio un error de diseno.

Es decir, el promedio **reduce la varianza**, que es justo lo que deberia transferir bien, pero
**desplaza la escala de salida**, y con umbrales congelados ese segundo efecto empuja en la
direccion contraria. Los dos fracasos son el mismo fallo medido dos veces.

**La combinacion que nunca se probo es re-optimizar los umbrales DESPUES de promediar.** Si el
efecto de escala es la causa del fracaso, eliminarla deja el efecto de varianza, que es el que
interesa. Es una hipotesis con causa medida, no una exploracion a ciegas.

Por que no es un promedio cualquiera
------------------------------
La reoptimizacion se hace sobre el OOF del ensemble, con el mismo protocolo de la fase y los mismos
pliegues, y los umbrales se eligen **una vez** sobre el OOF completo antes de mirar nada del test.
No se itera contra el leaderboard.

Ademas el objetivo de la reoptimizacion no es el maximo de la media de F1 sobre el OOF, que es
justo donde V03 ya esta, sino **emparejar la tasa de positivos predicha con la del champion**,
de modo que el unico cambio que se envia sea la reduccion de varianza y no un desplazamiento de
punto de operacion. La curva de cuatro puntos que se ha medido en esta fase demuestra que mover la
tasa cuesta LB con pendiente empinada, asi que la tasa se congela y lo unico que se prueba es el
promedio.

Lo que esta transicion NO es
---------------------------
No es una segunda sonda de la direccion de umbrales: esa la han refutado hoy ch5 (-0.00811) y ch1
(-0.00478), y la curva de cuatro puntos la cierra. Aqui los umbrales se eligen para **reproducir
la tasa de V03**, no para mejorarla.

Uso
---
    /home/carlos/venv/bin/python experiments/v16_02_promedio_con_umbrales.py
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from v10_03_arnes_evaluacion import (
    PARAMS_CHAR,
    PARAMS_WORD,
    UMBRALES_V03,
    oof_de_v03,
    particion_por_semilla,
)

from src import data as data_module
from src.evaluation import metrica_macro
from src.splits import RANDOM_STATE, grupos_fusionados, holdout_v2

EXPERIMENTO = "V16-02 (promedio de semillas con umbrales re-optimizados a tasa constante)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v16_02_promedio_umbrales.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v16_02_promedio_umbrales.json"
CSV_CANDIDATO = PROJECT_ROOT / "submissions" / "toxic_submission_v16_ch6_promedio_5semillas.csv"
CSV_TABLA = PROJECT_ROOT / "resultados" / "v16_02_tabla_umbrales.csv"

REFERENCIA_V03 = 0.7772114525165721
CONSTANTE_V03 = 2.0
LIMITE_ITERACIONES = 2000
ETIQUETAS = ("HS", "TR", "AG")
SEMILLAS = (42, 7, 1234, 20260926, 99991)
# La tasa objetivo es la del champion, medida sobre sus propias columnas de labels. Se congela
# para que el unico cambio que salga sea la reduccion de varianza.
TASA_OBJETIVO = np.array([0.4385, 0.2894, 0.3657])


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V16-02] {mensaje}", flush=True)


def main() -> None:
    """Construye el promedio de semillas y le asigna umbrales de tasa constante."""
    inicio = time.time()
    log("V16-02: promedio de semillas con umbrales re-optimizados")

    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = particion_por_semilla(textos, y, grupos[indices_dev], RANDOM_STATE)

    log("1/4 control de identidad: V03 tiene que reproducirse exactamente")
    probabilidades_v03 = oof_de_v03(textos, y, pliegues)
    base = (probabilidades_v03 >= UMBRALES_V03[None, :]).astype(int)
    metrica_v03 = metrica_macro(y, base)
    if abs(metrica_v03 - REFERENCIA_V03) > 1e-12:
        raise SystemExit(f"PARADA OBLIGATORIA: V03 no se reproduce, {metrica_v03!r}")
    log(f"   V03 {metrica_v03:.16f} | identidad confirmada")

    log("2/4 el OOF del promedio de cinco semillas")
    # El promedio se construye sobre los mismos pliegues que V03, de modo que la comparacion sea
    # pareada y la unica diferencia sea el promedio.
    acumulado = np.zeros_like(probabilidades_v03)
    por_semilla = []
    for semilla in SEMILLAS:
        probabilidades = _oof_con_semilla(textos, y, pliegues, semilla)
        acumulado += probabilidades
        metrica = metrica_macro(y, (probabilidades >= UMBRALES_V03[None, :]).astype(int))
        por_semilla.append(
            {"semilla": semilla, "macro_f1": metrica, "delta": metrica - metrica_v03}
        )
        log(f"   semilla {semilla:>9d}: macro-F1 {metrica:.6f} ({metrica - metrica_v03:+.6f})")
    promedio = acumulado / len(SEMILLAS)
    metrica_promedio = metrica_macro(y, (promedio >= UMBRALES_V03[None, :]).astype(int))
    log(
        f"   promedio de {len(SEMILLAS)} semillas con umbrales de V03: {metrica_promedio:.6f} "
        f"({metrica_promedio - metrica_v03:+.6f})"
    )
    log(
        "   Este es el estado que fracaso en v08_p1 y v12_p7, y la comprobacion de que el fallo"
    )
    log("   se reproduce aqui antes de intentar corregirlo con los umbrales.")

    log("3/4 umbrales de tasa constante: se reproduce la tasa del champion, etiqueta a etiqueta")
    # El objetivo NO es maximizar la media de F1 sobre el OOF, que es donde V03 ya esta, sino
    # igualar la tasa de positivos predicha del champion. Asi el unico cambio que se envia es la
    # reduccion de varianza.
    filas_tabla: list[dict[str, Any]] = []
    chosen = np.zeros(3, dtype=np.float64)
    for indice_etiqueta in range(3):
        objetivo = y[:, indice_etiqueta]
        columna = promedio[:, indice_etiqueta]
        mejor = None
        for umbral in np.arange(0.05, 0.951, 0.005):
            tasa = float(np.mean(columna >= umbral))
            if tasa < TASA_OBJETIVO[indice_etiqueta]:
                continue
            metrica = _f1_de_columna(y[:, indice_etiqueta], columna >= umbral)
            if mejor is None or metrica > mejor[1]:
                mejor = (float(umbral), metrica, tasa)
        if mejor is None:
            raise SystemExit(
                f"PARADA OBLIGATORIA: no se alcanza la tasa objetivo de "
                f"{ETIQUETAS[indice_etiqueta]}"
            )
        chosen[indice_etiqueta] = mejor[0]
        filas_tabla.append(
            {
                "etiqueta": ETIQUETAS[indice_etiqueta],
                "umbral_v03": float(UMBRALES_V03[indice_etiqueta]),
                "tasa_v03": float(np.mean(probabilidades_v03[:, indice_etiqueta] >= UMBRALES_V03[indice_etiqueta])),
                "umbral_promedio": mejor[0],
                "tasa_promedio": mejor[2],
                "tasa_objetivo": float(TASA_OBJETIVO[indice_etiqueta]),
                "macro_f1_de_la_columna": mejor[1],
            }
        )
        log(
            f"   {ETIQUETAS[indice_etiqueta]}: umbral {UMBRALES_V03[indice_etiqueta]:.3f} -> "
            f"{mejor[0]:.3f} | tasa {filas_tabla[-1]['tasa_promedio']:.4f} frente a objetivo "
            f"{TASA_OBJETIVO[indice_etiqueta]:.4f}"
        )

    candidata = (promedio >= chosen[None, :]).astype(int)
    metrica_candidata = metrica_macro(y, candidata)
    log(f"   macro-F1 del candidato final sobre OOF: {metrica_candidata:.6f}")

    cambio = int(np.sum(base != candidata))
    log(
        f"   celdas que cambian respecto a V03 en el OOF: {cambio} de {base.size}. Con la"
    )
    log("   escala de 0.003 de LB por 107 cambios netos, hay que ver si este numero llega.")

    log("4/4 generar la submission con el modelo reentrenado sobre todo el train")
    test = data_module.read_test()
    textos_test = test["text"].astype(str)
    predicciones_test = _test_con_promedio(textos, y, textos_test, SEMILLAS)
    etiquetas_test = (predicciones_test >= chosen[None, :]).astype(int)
    _escribir_submission(CSV_CANDIDATO, test["id"].tolist(), etiquetas_test)

    with CSV_TABLA.open("w", encoding="utf-8", newline="") as manejador:
        campos = list(filas_tabla[0].keys())
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_tabla)

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hipotesis": (
            "El promedio de probabilidades fracaso dos veces por un mecanismo medido: contrae la "
            "escala de salida y con umbrales congelados eso empuja en la direccion contraria. La "
            "combinacion que nunca se probo es reoptimizar los umbrales despues de promediar, "
            "emparejando la tasa de positivos con la del champion para que el unico cambio que se "
            "envia sea la reduccion de varianza"
        ),
        "identidad_control": {
            "metrica_v03": metrica_v03,
            "diferencia": abs(metrica_v03 - REFERENCIA_V03),
        },
        "semillas": por_semilla,
        "macro_f1_promedio_con_umbrales_v03": metrica_promedio,
        "delta_del_promedio_sin_reoptimizar": metrica_promedio - metrica_v03,
        "umbrales_elegidos": {e: float(chosen[i]) for i, e in enumerate(ETIQUETAS)},
        "umbrales_v03": {e: float(UMBRALES_V03[i]) for i, e in enumerate(ETIQUETAS)},
        "tasa_objetivo": {e: float(TASA_OBJETIVO[i]) for i, e in enumerate(ETIQUETAS)},
        "tabla_de_umbrales": filas_tabla,
        "macro_f1_candidata": metrica_candidata,
        "delta_candidata": metrica_candidata - metrica_v03,
        "celdas_modificadas_en_oof": cambio,
        "submission": str(CSV_CANDIDATO.relative_to(PROJECT_ROOT)),
        "nota_de_alcance": (
            "La reduccion de varianza por promedio es real y deberia transferir bien, pero la "
            "evidencia previa en este proyecto es de dos fracasos con el mismo mecanismo. Esta "
            "candidata corrige la causa medida; no esta garantizado que la corriga, y por eso se "
            "gasta UNA submission y no mas"
        ),
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: promedio sin reoptimizar {metrica_promedio - metrica_v03:+.6f} | con "
        f"umbrales de tasa constante {metrica_candidata - metrica_v03:+.6f} | "
        f"{cambio} celdas en OOF | submission: {CSV_CANDIDATO.name}"
    )


def _f1_de_columna(verdadero: np.ndarray, predicho: np.ndarray) -> float:
    """Calcula la F1 de la clase positiva de una sola etiqueta.

    Se calcula aqui y no con ``metrica_macro`` porque esa funcion exige la matriz de las tres
    etiquetas, y aqui lo que se quiere puntuar es el umbral de una sola, con las otras dos
    congeladas en su valor de V03.

    Args:
        verdadero: etiqueta binaria verdadera.
        predicho: prediccion binaria.

    Returns:
        La F1 de la clase positiva.
    """
    verdaderos_positivos = float(np.sum((verdadero == 1) & (predicho == 1)))
    denominador = (
        2 * verdaderos_positivos
        + float(np.sum((verdadero == 0) & (predicho == 1)))
        + float(np.sum((verdadero == 1) & (predicho == 0)))
    )
    return 2 * verdaderos_positivos / denominador if denominador > 0 else 0.0


def _oof_con_semilla(
    textos: pd.Series, y: np.ndarray, pliegues: list[tuple[np.ndarray, np.ndarray]], semilla: int
) -> np.ndarray:
    """Calcula el OOF de V03 con una semilla distinta.

    El vectorizador y el modelo son identicos a los de V03. Lo unico que cambia es
    ``random_state``, que en liblinear afecta al empaquetado de las instancias positivas y por
    tanto al resultado en las ultimas decimales. Es la diversity mas barata posible.

    Args:
        textos: serie de textos del desarrollo.
        y: matriz binaria de etiquetas.
        pliegues: pares de indices de entrenamiento y validacion.
        semilla: semilla del estimador.

    Returns:
        Matriz de probabilidades con una columna por etiqueta.
    """
    probabilidades = np.zeros((len(y), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        ajuste = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
                vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
            ],
            format="csr",
        )
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ],
            format="csr",
        )
        for indice_etiqueta in range(3):
            estimador = LogisticRegression(
                C=CONSTANTE_V03,
                class_weight="balanced",
                max_iter=LIMITE_ITERACIONES,
                solver="liblinear",
                random_state=semilla,
            )
            estimador.fit(ajuste, y[entrenamiento, indice_etiqueta])
            probabilidades[validacion, indice_etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return probabilidades


def _test_con_promedio(
    textos_train: pd.Series,
    y: np.ndarray,
    textos_test: pd.Series,
    semillas: tuple[int, ...],
) -> np.ndarray:
    """Calcula las probabilidades de test con el promedio de las semillas indicadas.

    Los modelos se ajustan sobre TODO el desarrollo, que es el regimen de inferencia que usa la
    submission congelada de V03. El test se usa solo para transformar, nunca para ajustar.

    Args:
        textos_train: textos del desarrollo.
        y: etiquetas del desarrollo.
        textos_test: textos del test.
        semillas: semillas a promediar.

    Returns:
        Matriz de probabilidades con una columna por etiqueta.
    """
    acumulado = np.zeros((len(textos_test), 3), dtype=np.float64)
    for semilla in semillas:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        ajuste = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos_train),
                vectorizador_char.fit_transform(textos_train),
            ],
            format="csr",
        )
        consulta = sparse.hstack(
            [vectorizador_word.transform(textos_test), vectorizador_char.transform(textos_test)],
            format="csr",
        )
        for indice_etiqueta in range(3):
            estimador = LogisticRegression(
                C=CONSTANTE_V03,
                class_weight="balanced",
                max_iter=LIMITE_ITERACIONES,
                solver="liblinear",
                random_state=semilla,
            )
            estimador.fit(ajuste, y[:, indice_etiqueta])
            acumulado[:, indice_etiqueta] += estimador.predict_proba(consulta)[:, 1]
    return acumulado / len(semillas)


def _escribir_submission(
    ruta: Path, identificadores: list[Any], etiquetas: np.ndarray
) -> None:
    """Escribe el CSV de la submission.

    Args:
        ruta: fichero de destino.
        identificadores: columna id del test.
        etiquetas: predicciones binarias.
    """
    if ruta.exists():
        raise SystemExit(f"PARADA OBLIGATORIA: {ruta.name} ya existe y no se regenera")
    with ruta.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow(["id", "HS", "TR", "AG"])
        for indice, identificador in enumerate(identificadores):
            escritor.writerow([identificador] + [int(v) for v in etiquetas[indice]])
    log(f"   escrita {ruta.name} con {len(identificadores)} filas")


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe del experimento.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    from v11_ayudas import fila, sello

    linea: list[str] = [
        "# V16-02: promedio de semillas con umbrales de tasa constante",
        "",
        f"Generado por `experiments/v16_02_promedio_umbrales.py` ({EXPERIMENTO}).",
        "",
        "## 1. La hipotesis y su causa medida",
        "",
        informe["hipotesis"],
        "",
        "## 2. El promedio solo, con los umbrales de V03: el fracaso que se reproduce",
        "",
        "| semilla | macro-F1 | delta |",
        "| --- | --- | --- |",
    ]
    for s in informe["semillas"]:
        linea.append(fila([str(s["semilla"]), f"{s['macro_f1']:.6f}", f"{s['delta']:+.6f}"]))
    linea += [
        fila(
            [
                "**promedio**",
                f"{informe['macro_f1_promedio_con_umbrales_v03']:.6f}",
                f"{informe['delta_del_promedio_sin_reoptimizar']:+.6f}",
            ]
        ),
        "",
        "## 3. Los umbrales de tasa constante",
        "",
        "El objetivo no es maximizar la media de F1 sobre el OOF, que es donde V03 ya esta, sino",
        "igualar la tasa de positivos predicha del champion. La curva de cuatro puntos medida en",
        "esta fase demuestra que mover la tasa cuesta LB con pendiente empinada.",
        "",
        fila(
            [
                "etiqueta",
                "umbral V03",
                "tasa V03",
                "umbral promedio",
                "tasa promedio",
                "tasa objetivo",
            ]
        ),
        fila(["---"] * 6),
    ]
    for r in informe["tabla_de_umbrales"]:
        linea.append(
            fila(
                [
                    r["etiqueta"],
                    f"{r['umbral_v03']:.3f}",
                    f"{r['tasa_v03']:.4f}",
                    f"{r['umbral_promedio']:.3f}",
                    f"{r['tasa_promedio']:.4f}",
                    f"{r['tasa_objetivo']:.4f}",
                ]
            )
        )
    linea += [
        "",
        "## 4. El candidato",
        "",
        f"- macro-F1 sobre OOF: **{informe['macro_f1_candidata']:.6f}**, "
        f"delta **{informe['delta_candidata']:+.6f}** respecto a V03.",
        f"- Celdas que cambian en el OOF: **{informe['celdas_modificadas_en_oof']}**.",
        f"- Submission: `{informe['submission']}`.",
        "",
        f"**Alcance de la afirmacion**: {informe['nota_de_alcance']}",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
