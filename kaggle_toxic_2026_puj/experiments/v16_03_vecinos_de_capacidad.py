"""V16-03: las tres sondas competitivas de capacidad y regularizacion alrededor de V03.

Que decide este script
----------------------
La fase V16 tiene TRES submissions restantes y el encargo es gastarlas de forma controlada en el
unico espacio competitivo razonable: los vecinos inmediatos de V03. Cada sonda cambia **un solo**
hiperparametro y nada mas, y se registra CV, delta CV, submission ID, LB y delta LB.

Correccion al encargo, y por que es necesaria
----------------------------------------------
El encargo tomaba V03 con ``min_df=1`` y proponia ``min_df=2`` como tercera sonda. **V03 tiene
``min_df=2``**, y esta verificado en tres fuentes independientes: las constantes ``PARAMS_WORD`` y
``PARAMS_CHAR`` del arnes validado, el manifiesto de la submission campeona, y el pipeline
congelado. Ademas el proyecto ya envio V03 hoy, ref 56571480, yObtuvo 0.78306.

Tomado al pie de la letra, la tercera sonda habria sido **V03 misma**, y habria gastado una de las
tres submissions restantes para confirmar un resultado que ya se conoce. La correccion que preserva
la intencion del encargo es sonda el vecino inmediato **por debajo** del valor real, es decir
``min_df=1``, que es el mismo salto de una muesca que el encargo pretendia dar.

Las tres sondas, y el unico parametro que cambia en cada una
-----------------------------------------------------------
1. ``C=1.5`` con ``min_df=2``. Un solo parametro sobre V03: la constante de regularizacion.
2. ``C=2.5`` con ``min_df=2``. El mismo parametro, en la direccion contraria.
3. ``min_df=1`` con ``C=2.0``. Un solo parametro sobre V03: el corte de frecuencias, una muesca
   por debajo del valor real de V03.

Invariantes que se comprueban, no se suponen
---------------------------------------------
- Los thresholds son los de V03, 0.44, 0.51 y 0.46, **sin tocar en ninguna sonda**. Cambiarlos
  seria cambiar dos parametros a la vez.
- No se anade ninguna feature, ni se cambia el modelo, ni se altera el protocolo de generacion.
- El test se usa **solo para transformar**. Ningun modelo se ajusta con el test.
- Cada sonda se compara contra la submission campeona celda a celda, y se reporta cuantas celdas
  cambian, porque la escala del proyecto es de 0.003 de LB por 107 cambios netos.

Uso
---
    /home/carlos/venv/bin/python experiments/v16_03_vecinos_de_capacidad.py
"""

from __future__ import annotations

import csv
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
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from v10_03_arnes_evaluacion import (
    PARAMS_CHAR,
    PARAMS_WORD,
    UMBRALES_V03,
    particion_por_semilla,
)

from src import data as data_module
from src.evaluation import metrica_macro
from src.splits import RANDOM_STATE, grupos_fusionados, holdout_v2

EXPERIMENTO = "V16-03 (tres sondas de capacidad y regularizacion alrededor de V03)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v16_03_vecinos_de_capacidad.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v16_03_vecinos_de_capacidad.json"
CSV_TABLA = PROJECT_ROOT / "resultados" / "v16_03_tabla_de_sondas.csv"

REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
LIMITE_ITERACIONES = 2000
ETIQUETAS = ("HS", "TR", "AG")
SUBMISSION_CAMPEONA = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "submission_v03.csv"

# Las tres sondas. Cada una cambia UN solo parametro respecto de V03, que tiene C=2.0 y min_df=2.
SONDAS: tuple[dict[str, Any], ...] = (
    {
        "id": "v16_ch7_C1p5",
        "descripcion": "V03 exacta salvo C=1.5",
        "C": 1.5,
        "min_df": 2,
        "unico_parametro": "C, de 2.0 a 1.5",
        "pregunta": (
            "V03 esta en el pico de la curva LB contra capacidad, con v01 en 4k features y C=1 "
            "dando 0.75677 y v05 en 32k y C=3.26 dando 0.77602. El pico esta en C=2.0. Esta "
            "sonda pregunta si el pico es ancho o agudo, reduciendo la regularizacion un paso"
        ),
    },
    {
        "id": "v16_ch8_C2p5",
        "descripcion": "V03 exacta salvo C=2.5",
        "C": 2.5,
        "min_df": 2,
        "unico_parametro": "C, de 2.0 a 2.5",
        "pregunta": (
            "La direccion contraria de la misma sonda. Si el pico fuese agudo, subir C "
            "empeoraria de forma medida y la curva quedaria cerrada por los dos lados"
        ),
    },
    {
        "id": "v16_ch9_mindf1",
        "descripcion": "V03 exacta salvo min_df=1",
        "C": 2.0,
        "min_df": 1,
        "unico_parametro": "min_df, de 2 a 1",
        "pregunta": (
            "El corte de frecuencias controla el tamaño del vocabulario y por tanto la "
            "capacidad de forma distinta de C. Con min_df=2 el espacio tiene 24646 dimensiones; "
            "con min_df=1 crecera. Esta sonda separa 'mas capacidad' de 'mas regularizacion', "
            "que hasta ahora solo se han explorado juntas"
        ),
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V16-03] {mensaje}", flush=True)


def main() -> None:
    """Genera las tres sondas, con su CV, su delta y su hash."""
    inicio = time.time()
    log("V16-03: sondas de capacidad y regularizacion alrededor de V03")

    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = particion_por_semilla(textos, y, grupos[indices_dev], RANDOM_STATE)

    log("1/4 control de identidad: V03 tiene que reproducirse antes de generar nada")
    parametros_v03 = _parametros(2.0, 2)
    probabilidades_v03 = _oof(textos, y, pliegues, parametros_v03)
    metrica_v03 = metrica_macro(y, (probabilidades_v03 >= UMBRALES_V03[None, :]).astype(int))
    if abs(metrica_v03 - REFERENCIA_V03) > 1e-12:
        raise SystemExit(
            f"PARADA OBLIGATORIA: V03 no se reproduce, {metrica_v03!r} frente a "
            f"{REFERENCIA_V03!r}. No se genera ninguna sonda sobre una referencia rota."
        )
    log(f"   V03 {metrica_v03:.16f} | identidad confirmada")

    campeona = np.array(
        [[int(f["HS"]), int(f["TR"]), int(f["AG"])]
         for f in csv.DictReader(SUBMISSION_CAMPEONA.open(encoding="utf-8"))]
    )
    log(f"   submission campeona: {campeona.shape[0]} filas, sha256 {_hash(SUBMISSION_CAMPEONA)[:12]}")

    test = data_module.read_test()
    textos_test = test["text"].astype(str)

    log("2/4 las tres sondas: CV local, delta CV, celdas que cambian y hash")
    filas: list[dict[str, Any]] = []
    for sonda in SONDAS:
        parametros = _parametros(sonda["C"], sonda["min_df"])
        probabilidades = _oof(textos, y, pliegues, parametros)
        metrica = metrica_macro(y, (probabilidades >= UMBRALES_V03[None, :]).astype(int))
        etiquetas_test = _test(textos, y, textos_test, parametros)
        ruta = PROJECT_ROOT / "submissions" / f"toxic_submission_{sonda['id']}.csv"
        if ruta.exists():
            raise SystemExit(f"PARADA OBLIGATORIA: {ruta.name} ya existe y no se regenera")
        _escribir(ruta, test["id"].tolist(), etiquetas_test)
        cambio = int(np.sum(campeona != etiquetas_test))
        a01 = int(np.sum((etiquetas_test == 1) & (campeona == 0)))
        a10 = int(np.sum((etiquetas_test == 0) & (campeona == 1)))
        filas.append(
            {
                "id": sonda["id"],
                "descripcion": sonda["descripcion"],
                "unico_parametro": sonda["unico_parametro"],
                "C": sonda["C"],
                "min_df": sonda["min_df"],
                "umbrales": "0.44/0.51/0.46, los de V03 sin cambio",
                "corrected_cv": metrica,
                "delta_cv": metrica - metrica_v03,
                "celdas_modificadas": cambio,
                "neto_0a1_menos_1a0": a01 - a10,
                "fichero": ruta.name,
                "sha256": _hash(ruta),
                "pregunta": sonda["pregunta"],
                "submission_id": "",
                "kaggle_f1": "",
                "delta_lb": "",
            }
        )
        log(
            f"   {sonda['id']:16s} CV {metrica:.6f} ({metrica - metrica_v03:+.6f}) | "
            f"{cambio:4d} celdas ({a01 - a10:+d} neto) | sha {filas[-1]['sha256'][:12]}"
        )

    with CSV_TABLA.open("w", encoding="utf-8", newline="") as manejador:
        campos = [k for k in filas[0] if k != "pregunta"]
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in filas:
            escritor.writerow({k: f[k] for k in campos})

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "correccion_al_encargo": (
            "El encargo tomaba V03 con min_df=1 y proponia min_df=2 como tercera sonda. V03 tiene "
            "min_df=2, verificado en las constantes PARAMS_WORD y PARAMS_CHAR del arnes validado, en "
            "el manifiesto de la submission campeona y en el pipeline congelado. Ademas el proyecto "
            "ya envio V03 hoy, ref 56571480, y obtuvo 0.78306. La tercera sonda habria sido V03 "
            "misma. Se corrige a min_df=1, que es el vecino inmediato por debajo del valor real y "
            "el mismo salto de una muesca que el encargo pretendia"
        ),
        "identidad_control": {
            "metrica_v03": metrica_v03,
            "diferencia": abs(metrica_v03 - REFERENCIA_V03),
            "parametros_v03": parametros_v03,
        },
        "lb_v03": LB_V03,
        "sondas": filas,
        "invariantes": [
            "Los thresholds son los de V03 en las tres sondas, sin cambio",
            "Un solo hiperparametro cambia por sonda",
            "Ninguna feature nueva, ningun modelo nuevo",
            "El test se usa solo para transformar; ningun modelo se ajusta con el",
            "Mismo protocolo de generacion: ajuste sobre todo el desarrollo y prediccion del test",
        ],
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: 3 sondas generadas, {' | '.join(f['id'] + ' CV ' + format(f['delta_cv'], '+.6f') for f in filas)}"
    )


def _parametros(constante: float, minimo: int) -> dict[str, Any]:
    """Construye los parametros de los dos vectorizadores para una sonda.

    Args:
        constante: valor de C.
        minimo: valor de min_df.

    Returns:
        Diccionario con los parametros de word y char.
    """
    return {
        "word": {**PARAMS_WORD, "min_df": minimo},
        "char": {**PARAMS_CHAR, "min_df": minimo},
        "C": constante,
    }


def _espacio(
    textos: pd.Series, entrenamiento: np.ndarray, consulta_idx: np.ndarray, parametros: dict[str, Any]
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los dos vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos: serie de textos.
        entrenamiento: indices de ajuste.
        consulta_idx: indices a transformar.
        parametros: parametros de la sonda.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    vectorizador_word = TfidfVectorizer(**parametros["word"])
    vectorizador_char = TfidfVectorizer(**parametros["char"])
    ajuste = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
            vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos.iloc[consulta_idx]),
            vectorizador_char.transform(textos.iloc[consulta_idx]),
        ],
        format="csr",
    )
    return ajuste, consulta


def _oof(
    textos: pd.Series,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    parametros: dict[str, Any],
) -> np.ndarray:
    """Calcula el OOF con los parametros de una sonda, sobre los pliegues canonicos.

    Args:
        textos: serie de textos del desarrollo.
        y: matriz binaria de etiquetas.
        pliegues: pares de indices de entrenamiento y validacion.
        parametros: parametros de la sonda.

    Returns:
        Matriz de probabilidades con una columna por etiqueta.
    """
    probabilidades = np.zeros((len(y), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        ajuste, consulta = _espacio(textos, entrenamiento, validacion, parametros)
        for indice_etiqueta in range(3):
            estimador = LogisticRegression(
                C=parametros["C"],
                class_weight="balanced",
                max_iter=LIMITE_ITERACIONES,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(ajuste, y[entrenamiento, indice_etiqueta])
            probabilidades[validacion, indice_etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return probabilidades


def _test(
    textos_train: pd.Series,
    y: np.ndarray,
    textos_test: pd.Series,
    parametros: dict[str, Any],
) -> np.ndarray:
    """Ajusta sobre todo el desarrollo y predice el test.

    El test se usa **solo para transformar**. Ningun modelo se ajusta con el.

    Args:
        textos_train: textos del desarrollo.
        y: etiquetas del desarrollo.
        textos_test: textos del test.
        parametros: parametros de la sonda.

    Returns:
        Matriz de predicciones binarias del test.
    """
    # Se ajusta sobre todas las filas del desarrollo y se transforma el test.
    vectorizador_word = TfidfVectorizer(**parametros["word"])
    vectorizador_char = TfidfVectorizer(**parametros["char"])
    ajuste_completo = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_train),
            vectorizador_char.fit_transform(textos_train),
        ],
        format="csr",
    )
    consulta_test = sparse.hstack(
        [vectorizador_word.transform(textos_test), vectorizador_char.transform(textos_test)],
        format="csr",
    )
    predicciones = np.zeros((len(textos_test), 3), dtype=int)
    for indice_etiqueta in range(3):
        estimador = LogisticRegression(
            C=parametros["C"],
            class_weight="balanced",
            max_iter=LIMITE_ITERACIONES,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        estimador.fit(ajuste_completo, y[:, indice_etiqueta])
        predicciones[:, indice_etiqueta] = (
            estimador.predict_proba(consulta_test)[:, 1] >= UMBRALES_V03[indice_etiqueta]
        ).astype(int)
    return predicciones


def _escribir(ruta: Path, identificadores: list[Any], etiquetas: np.ndarray) -> None:
    """Escribe el CSV de la submission.

    Args:
        ruta: fichero de destino.
        identificadores: columna id del test.
        etiquetas: predicciones binarias.
    """
    with ruta.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow(["id", "HS", "TR", "AG"])
        for indice, identificador in enumerate(identificadores):
            escritor.writerow([identificador] + [int(v) for v in etiquetas[indice]])


def _hash(ruta: Path) -> str:
    """Calcula el sha256 de un fichero.

    Args:
        ruta: fichero a resumir.

    Returns:
        El hash en hexadecimal.
    """
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de las tres sondas.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    from v11_ayudas import fila, sello

    linea: list[str] = [
        "# V16-03: tres sondas de capacidad y regularizacion alrededor de V03",
        "",
        f"Generado por `experiments/v16_03_vecinos_de_capacidad.py` ({EXPERIMENTO}).",
        "",
        "## 1. Correccion al encargo, y por que era necesaria",
        "",
        informe["correccion_al_encargo"],
        "",
        "## 2. Las tres sondas",
        "",
        fila(
            [
                "sonda",
                "unico parametro que cambia",
                "corrected_cv",
                "delta CV",
                "celdas que cambian",
                "neto",
            ]
        ),
        fila(["---"] * 6),
    ]
    for f in informe["sondas"]:
        linea.append(
            fila(
                [
                    f["id"],
                    f["unico_parametro"],
                    f"{f['corrected_cv']:.6f}",
                    f"{f['delta_cv']:+.6f}",
                    str(f["celdas_modificadas"]),
                    f"{f['neto_0a1_menos_1a0']:+d}",
                ]
            )
        )
    linea += [
        "",
        "## 3. La pregunta que responde cada sonda",
        "",
    ]
    for f in informe["sondas"]:
        linea += [f"**{f['id']}**: {f['pregunta']}", ""]
    linea += [
        "## 4. Invariantes comprobados",
        "",
    ]
    for v in informe["invariantes"]:
        linea.append(f"- {v}")
    linea += [
        "",
        "## 5. Como se leen los resultados",
        "",
        "El criterio de adopcion es el leaderboard, no el CV local. El historial de esta fase ha",
        "demostrado que el CV local **anti-correlaciona** con el LB: las cuatro sondas con CV",
        "superior a V03 que se han enviado transferieron a la baja. Por eso las tres sondas de",
        "aqui se envian aunque su CV local sea peor, y el veredicto se toma del score externo.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
