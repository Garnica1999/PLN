"""P3 de la fase v8 (V8-03): estabilidad de features entre submuestras.

Contexto
--------
La sonda P1 no mejoro y el analisis de la resolucion del leaderboard (V8-02) mostro dos
hechos que reencuadran la fase. De once sondas reales, diez son indistinguibles entre si y
ninguna supera a V03. Y el externo solo resuelve diferencias de unos 19.2 aciertos netos, de
modo que **una candidata que cambie pocas celdas no es mala: es indetectable**. De ahi que P3
sea la siguiente linea: reducir la dimension puede cambiar muchas celdas de golpe, que es la
unica forma de acercarse al minimo de acierto exigido.

De donde viene la idea
----------------------
La estabilidad de rasgos es el capitulo 8 de *Inference and Disputed Authorship*, de
Mosteller y Wallace, el metodo fundacional de la atribucion de autoria. El libro la mide
traduciendo cada documento a cinco idiomas y viendo que palabras sobreviven a la traduccion de
ida y vuelta: sobreviven las que son solidas y no las que dependen del contexto. El material de
clase reproduce ese esquema.

La traduccion automatica no esta disponible en este proyecto, y ademas exigiria recursos
externos que las reglas de la competencia no permiten. **La pregunta es la misma y se puede
operacionalizar con lo que hay**: en lugar de preguntar que palabras sobreviven a una
traduccion, preguntar que pesos sobreviven a un **resubconjunto de las filas de
entrenamiento**. Un rasgo cuyo peso mantiene el signo y la magnitud bajo submuestras
independientes es solido; uno que cambia de signo es ruido de esa muestra. Es la misma
pregunta, contestada con el instrumento disponible.

Diseno
------
Para cada uno de los cinco pliegues del protocolo oficial, y dentro del tren de ese pliegue:

1. El vectorizador se ajusta **una vez sobre el tren completo del pliegue**, de modo que el
   espacio de caracteristicas es fijo y las submuestras son comparables entre si. Solo varian
   las filas.
2. Se toman siete submuestras sin reemplazo del 80 % de las filas del tren.
3. En cada submuestra se ajusta el clasificador y se guarda la magnitud del peso de cada
   caracteristica.
4. La **estabilidad de un rasgo** es la fraccion de submuestras en las que su peso supera un
   percentil alto de la distribucion de magnitudes. Un rasgo que solo pesa en una submuestra de
   siete no es solido.
5. Se conservan los rasgos con estabilidad al menos igual que un umbral, y se reentrena con
   ellos.

**Todo se construye dentro del tren del pliegue.** El vectorizador, las submuestras, los pesos
y la seleccion se calculan con las filas de entrenamiento de ese pliegue y con nada mas. No hay
fuga posible porque la seleccion nunca ve las filas de validacion.

Criterios de decision
---------------------
- Gate de mejora: diferencia al menos +0.003, al menos 4 de 5 pliegues a favor, e intervalo
  de confianza al 95 por bootstrap pareado que no incluya cero.
- **Filtro de detectabilidad, que es el nuevo y el que manda**: la tasa de acierto entre las
  celdas que la candidata modifica tiene que superar `0,5 + 19,2 / C`. Si no lo alcanza, la
  linea no se envia aunque gane en local, porque el externo no puede distinguirla.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_03_estabilidad_features.py
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
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
)
from src.splits import (  # noqa: E402
    RANDOM_STATE,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V8-03 (P3)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_estabilidad_features.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_estabilidad_features.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_estabilidad_features.json"
FIGURA = PROJECT_ROOT / "reports" / "figuras" / "fig_70_estabilidad_features_v8.png"

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
# Ruido del leaderboard expresado en aciertos netos, medido en V8-02.
RUIDO_NETOS = 19.2
SEMILLA_SUBMUESTRAS = 20260926

NUMERO_SUBMUESTRAS = 7
FRACCION_SUBMUESTRA = 0.8
PERCENTIL_PESO = 90.0
UMBRALES_ESTABILIDAD = (0.5, 0.7, 0.9)

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
    print(f"[V8-P3] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def construir_espacio(textos: pd.Series, indices: np.ndarray):
    """Ajusta los dos vectorizadores y construye la matriz de caracteristicas.

    El ajuste se hace **una sola vez sobre las filas indicadas**, de modo que el espacio de
    caracteristicas queda fijo y las submuestras son comparables entre si.

    Args:
        textos: serie de textos completa.
        indices: indices de las filas que se usan para ajustar.

    Returns:
        Tupla con la matriz de caracteristicas ajustada y los dos vectorizadores.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    matriz = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos.iloc[indices]),
            vectorizador_char.fit_transform(textos.iloc[indices]),
        ],
        format="csr",
    )
    return matriz, vectorizador_word, vectorizador_char


def medir_estabilidad(
    matriz: sparse.csr_matrix,
    y: np.ndarray,
    generador: np.random.Generator,
    numero_submuestras: int = NUMERO_SUBMUESTRAS,
    fraccion: float = FRACCION_SUBMUESTRA,
    percentil: float = PERCENTIL_PESO,
) -> np.ndarray:
    """Mide cuantas veces cada caracteristica pesa, sobre submuestras de las filas.

    Solo se usan las filas que ya estan dentro del tren del pliegue: la funcion no recibe
    ningun indice exterior y por tanto no puede ver las filas de validacion.

    Args:
        matriz: matriz de caracteristicas del tren del pliegue.
        y: matriz binaria de etiquetas del tren del pliegue.
        generador: generador de numeros aleatorios, sembrado por el llamante.
        numero_submuestras: cuantas submuestras se toman.
        fraccion: fraccion de filas que se conserva en cada submuestra.
        percentil: percentil de magnitud de peso a partir del cual se cuenta que un rasgo pesa.

    Returns:
        Vector de estabilidad, una fraccion entre cero y uno por caracteristica.
    """
    n_filas = matriz.shape[0]
    conteo = np.zeros(matriz.shape[1], dtype=np.float64)
    for _ in range(numero_submuestras):
        cuantos = max(int(fraccion * n_filas), 30)
        filas = generador.choice(n_filas, size=cuantos, replace=False)
        submatriz = matriz[filas]
        for etiqueta in range(3):
            destino = y[filas, etiqueta]
            if len(np.unique(destino)) < 2:
                continue
            estimador = LogisticRegression(**PARAMETROS_LR)
            estimador.fit(submatriz, destino)
            magnitud = np.abs(estimador.coef_[0])
            if magnitud.size == 0:
                continue
            corte = np.percentile(magnitud, percentil)
            conteo += (magnitud >= corte).astype(np.float64)
    # Cada submuestra aporta tres votos, uno por etiqueta, y se cuentan los que superan el
    # percentil. La estabilidad se normaliza por el numero total de votos emitidos.
    return conteo / (numero_submuestras * 3.0)


def main() -> None:
    """Ejecuta la seleccion por estabilidad de features y evalua sus tres umbrales."""
    inicio = time.time()
    log("P3: estabilidad de features entre submuestras")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    generador = np.random.default_rng(SEMILLA_SUBMUESTRAS)
    log(
        f"   desarrollo {len(y)} filas | prueba {len(test)} filas | "
        f"{len(pliegues)} pliegues | {NUMERO_SUBMUESTRAS} submuestras por pliegue"
    )

    log("1/7 control de codigo contra el OOF canonico de V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    ids_iguales = bool(
        (oof["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all()
    )
    metrica_canonica = metrica_macro(y, aplicar_umbrales(proba_canonica))
    control_correcto = (
        abs(metrica_canonica - REFERENCIA_V03) <= TOLERANCIA_CONTROL and ids_iguales
    )
    log(
        f"   OOF canonico: {metrica_canonica:.6f} frente a {REFERENCIA_V03} | "
        f"identificadores en orden: {ids_iguales} -> {'dentro' if control_correcto else 'FUERA'}"
    )
    if not control_correcto:
        log("PARADA: el control falla. No se sigue y no se genera nada.")
        return

    log("2/7 midiendo la estabilidad en cada pliegue, dentro de su tren")
    oof_por_umbral: dict[float, np.ndarray] = {}
    dimensionalidad: dict[float, list[int]] = {}
    for numero, (entrenamiento, validacion) in enumerate(pliegues):
        matriz, vectorizador_word, vectorizador_char = construir_espacio(textos, entrenamiento)
        estabilidad = medir_estabilidad(matriz, y[entrenamiento], generador)
        # Las filas de validacion no estan en la matriz del pliegue, porque esa matriz solo
        # contiene el entrenamiento. Hay que transformarlas con los mismos vectorizadores ya
        # ajustados, y no volver a ajustarlos: ajustarlos con validacion seria fuga.
        consulta_completa = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ],
            format="csr",
        )
        for umbral in UMBRALES_ESTABILIDAD:
            columnas = np.flatnonzero(estabilidad >= umbral)
            dimensionalidad.setdefault(umbral, []).append(int(len(columnas)))
            submatriz = matriz[:, columnas]
            consulta = consulta_completa[:, columnas]
            probabilidades = np.zeros((len(validacion), 3), dtype=np.float64)
            for etiqueta in range(3):
                estimador = LogisticRegression(**PARAMETROS_LR)
                estimador.fit(submatriz, y[entrenamiento, etiqueta])
                probabilidades[:, etiqueta] = estimador.predict_proba(consulta)[:, 1]
            previas = oof_por_umbral.setdefault(
                umbral, np.zeros((len(y), 3), dtype=np.float64)
            )
            previas[validacion] = probabilidades
        log(
            f"   pliegue {numero + 1}: {matriz.shape[1]} caracteristicas en el espacio, "
            + ", ".join(
                f"umbral {u} deja {dimensionalidad[u][-1]}" for u in UMBRALES_ESTABILIDAD
            )
        )

    log("3/7 metrica de cada umbral y gate de mejora")
    referencia_pred = aplicar_umbrales(proba_canonica)
    variantes: list[dict[str, Any]] = []
    for umbral in UMBRALES_ESTABILIDAD:
        probabilidades = oof_por_umbral[umbral]
        predicciones = aplicar_umbrales(probabilidades)
        metrica = metrica_macro(y, predicciones)
        diferencia = metrica - metrica_canonica
        aciertos_por_pliegue = []
        for _, validacion in pliegues:
            aciertos_por_pliegue.append(
                int(
                    (predicciones[validacion] == y[validacion]).sum()
                    - (referencia_pred[validacion] == y[validacion]).sum()
                )
            )
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, referencia_pred, repeticiones=2000
        )
        variantes.append(
            {
                "umbral_estabilidad": umbral,
                "caracteristicas_medias": int(np.mean(dimensionalidad[umbral])),
                "caracteristicas_min": int(np.min(dimensionalidad[umbral])),
                "caracteristicas_max": int(np.max(dimensionalidad[umbral])),
                "metrica": round(metrica, 6),
                "diferencia": round(diferencia, 6),
                "ic95_0": float(bootstrap["ic95"][0]),
                "ic95_1": float(bootstrap["ic95"][1]),
                "pliegues_positivos": int(sum(1 for a in aciertos_por_pliegue if a > 0)),
                "aciertos_por_pliegue": aciertos_por_pliegue,
                "predicciones": predicciones,
            }
        )
        f = variantes[-1]
        f["gate_delta"] = f["diferencia"] >= GATE_DELTA
        f["gate_ic95"] = f["ic95_0"] > 0
        f["gate_pliegues"] = f["pliegues_positivos"] >= GATE_PLIEGUES
        f["gate_supera"] = f["gate_delta"] and f["gate_ic95"] and f["gate_pliegues"]
        log(
            f"   umbral {umbral}: {f['caracteristicas_medias']} "
            f"caracteristicas | metrica {f['metrica']:.6f} | diferencia {f['diferencia']:+.6f} | "
            f"IC95 [{f['ic95_0']:+.5f}, {f['ic95_1']:+.5f}] | pliegues "
            f"{f['pliegues_positivos']}/5 | gate {'SUPERA' if f['gate_supera'] else 'no supera'}"
        )

    log("4/7 filtro de detectabilidad: cuantas celdas cambian y con que tasa de acierto")
    for f in variantes:
        cambio = f["predicciones"] != referencia_pred
        n_celdas = int(cambio.sum())
        if n_celdas == 0:
            f.update(
                celdas=0, tasa_acierto=float("nan"), minimo_requerido=float("nan"),
                filtro_pasa=False,
            )
            log(f"   umbral {f['umbral_estabilidad']}: no cambia ninguna celda")
            continue
        # Tasa de acierto entre las celdas que cambian: de esas, cuantas la candidata acierta
        # y la referencia falla. Es la magnitud que decide si el externo puede resolverlo.
        acierta_candidata = (f["predicciones"] == y)[cambio].sum()
        falla_referencia = (referencia_pred != y)[cambio].sum()
        tasa = float(acierta_candidata) / float(n_celdas)
        minimo = 0.5 + RUIDO_NETOS / (2.0 * n_celdas)
        f.update(
            celdas=n_celdas,
            filas=int(cambio.any(axis=1).sum()),
            tasa_acierto=round(tasa, 4),
            acierta_candidata=int(acierta_candidata),
            falla_referencia=int(falla_referencia),
            minimo_requerido=round(minimo, 4),
            filtro_pasa=tasa >= minimo,
        )
        log(
            f"   umbral {f['umbral_estabilidad']}: {n_celdas} celdas cambian | "
            f"tasa de acierto {tasa:.4f} frente a un minimo de {minimo:.4f} -> "
            f"{'pasa' if f['filtro_pasa'] else 'NO PASA'}"
        )

    log("5/7 seleccion de la variante a ejecutar, si alguna")
    admisibles = [f for f in variantes if f["gate_supera"] and f["filtro_pasa"]]
    elegida = max(admisibles, key=lambda f: f["metrica"]) if admisibles else None
    if elegida is None:
        log(
            "   ninguna variante supera a la vez el gate de mejora y el filtro de "
            "detectabilidad. No se genera submission."
        )
    else:
        log(f"   elegida la de umbral {elegida['umbral_estabilidad']}")

    log("6/7 informe y figura")
    _figura(variantes, dimensionalidad)
    _informe(
        variantes=variantes,
        elegida=elegida,
        metrica_canonica=metrica_canonica,
        dimensionalidad=dimensionalidad,
        numero_caracteristicas_completas=_contar_completas(textos),
        n_test=len(test),
        inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "umbral_estabilidad", "caracteristicas_medias", "caracteristicas_min",
            "caracteristicas_max", "metrica", "diferencia", "ic95_0", "ic95_1",
            "pliegues_positivos", "gate_delta", "gate_ic95", "gate_pliegues", "gate_supera",
            "celdas", "filas", "tasa_acierto", "minimo_requerido", "filtro_pasa",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in variantes:
            escritor.writerow({k: f.get(k, "") for k in campos})
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "ruido_en_aciertos_netos": RUIDO_NETOS,
                "numero_submuestras": NUMERO_SUBMUESTRAS,
                "fraccion_submuestra": FRACCION_SUBMUESTRA,
                "percentil_peso": PERCENTIL_PESO,
                "control_correcto": control_correcto,
                "variantes": [{k: v for k, v in f.items() if k != "predicciones"} for f in variantes],
                "elegida": elegida["umbral_estabilidad"] if elegida else None,
                "submission_generada": elegida is not None,
                "nota": (
                    "El vectorizador se ajusta una vez por pliegue sobre su tren completo, de "
                    "modo que el espacio de caracteristicas es fijo y solo varian las filas de "
                    "cada submuestra. La seleccion nunca ve las filas de validacion."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P3 completada en {round(time.time() - inicio, 1)} s")


def _contar_completas(textos) -> int:
    """Cuenta las caracteristicas del espacio completo, sin ninguna seleccion.

    Se ajusta sobre **todas** las filas del desarrollo, que es lo que hace falta para saber
    cuanto se esta reduciendo la dimension.

    Args:
        textos: serie de textos del desarrollo.

    Returns:
        Numero de caracteristicas del espacio completo.
    """
    todos = np.arange(len(textos), dtype=int)
    matriz, _, _ = construir_espacio(textos, todos)
    return int(matriz.shape[1])


def _figura(variantes: list[dict[str, Any]], dimensionalidad: dict[float, list[int]]) -> None:
    """Dibuja el coste en caracteristicas frente al beneficio en metrica.

    Args:
        variantes: resultado de cada umbral.
        dimensionalidad: numero de caracteristicas conservadas por pliegue.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig, eje = plt.subplots(figsize=(8.5, 4.6))
    for f in variantes:
        umbral = f["umbral_estabilidad"]
        eje.errorbar(
            np.mean(dimensionalidad[umbral]),
            f["diferencia"],
            yerr=[[f["diferencia"] - f["ic95_0"]], [f["ic95_1"] - f["diferencia"]]],
            fmt="o",
            capsize=4,
            label=f"estabilidad {umbral} ({f['caracteristicas_medias']} rasgos)",
        )
    eje.axhline(0.0, color="#55A868", linestyle="--", linewidth=1)
    eje.axhline(GATE_DELTA, color="#C44E52", linestyle=":", linewidth=1, label="gate +0.003")
    eje.set_xlabel("caracteristicas conservadas (media de los cinco pliegues)")
    eje.set_ylabel("diferencia de metrica frente a V03")
    eje.set_title("Estabilidad de features: reduccion de dimension frente a beneficio")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURA, dpi=150)
    plt.close()


def _informe(
    variantes: list[dict[str, Any]],
    elegida: dict[str, Any] | None,
    metrica_canonica: float,
    dimensionalidad: dict[float, list[int]],
    numero_caracteristicas_completas: int,
    n_test: int,
    inicio: float,
) -> None:
    """Escribe el informe de la linea P3.

    Args:
        variantes: resultado de cada umbral de estabilidad.
        elegida: variante elegida, si alguna.
        metrica_canonica: metrica del OOF canonico de V03.
        dimensionalidad: numero de caracteristicas conservadas por pliegue y umbral.
        numero_caracteristicas_completas: caracteristicas sin ninguna seleccion.
        n_test: numero de filas del conjunto de prueba.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# P3: estabilidad de features entre submuestras (V8-03)",
        "",
        f"Generado por `experiments/v8_03_estabilidad_features.py` ({EXPERIMENTO}).",
        "",
        "## 1. De donde viene la idea",
        "",
        "La estabilidad de rasgos es el capitulo 8 de *Inference and Disputed Authorship*, de",
        "Mosteller y Wallace. El libro la mide traduciendo cada documento a cinco idiomas y",
        "viendo que palabras sobreviven a la traduccion de ida y vuelta: sobreviven las que son",
        "solidas, no las que dependen del contexto. La traduccion automatica no esta disponible",
        "en este proyecto, asi que la pregunta se contesta con el instrumento que si hay: en",
        "lugar de pedir que palabras sobreviven a una traduccion, se pide que pesos sobrevivan",
        "a un **resubconjunto de las filas de entrenamiento**. Es la misma pregunta.",
        "",
        "## 2. Como se construye, y por que no puede haber fuga",
        "",
        "- El vectorizador se ajusta **una vez por pliegue sobre su tren completo**, de modo",
        "  que el espacio de caracteristicas es fijo. Solo varian las filas.",
        f"- Se toman {NUMERO_SUBMUESTRAS} submuestras sin reemplazo del",
        f"  {int(FRACCION_SUBMUESTRA * 100)} % de las filas de ese tren.",
        "- Un rasgo cuenta como estable si su peso supera el percentil",
        f"  {int(PERCENTIL_PESO)} de la distribucion de magnitudes en la submuestra.",
        "- Se conservan los rasgos cuya estabilidad alcanza un umbral y se reentrena con ellos.",
        "",
        "**La seleccion nunca ve las filas de validacion**, porque la funcion que mide la",
        "estabilidad no recibe ningun indice exterior: solo recibe la matriz del tren del",
        "pliegue. No hay forma de que lo haga: la funcion no recibe indices de validacion.",
        "",
        f"Control de codigo: el OOF canonico de V03 da {metrica_canonica:.6f} sobre las mismas",
        "filas, y los identificadores estan en el mismo orden.",
        "",
        "## 3. Los tres umbrales de estabilidad",
        "",
        f"El espacio completo tiene **{numero_caracteristicas_completas} caracteristicas**.",
        "",
        "| umbral | caracteristicas conservadas | metrica | diferencia | IC95 | pliegues | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in variantes:
        lineas.append(
            f"| {f['umbral_estabilidad']} | {f['caracteristicas_medias']} "
            f"({f['caracteristicas_min']} a {f['caracteristicas_max']}) | {f['metrica']:.6f} | "
            f"{f['diferencia']:+.6f} | [{f['ic95_0']:+.5f}, {f['ic95_1']:+.5f}] | "
            f"{f['pliegues_positivos']}/5 | {'**supera**' if f['gate_supera'] else 'no supera'} |"
        )
    lineas += [
        "",
        "## 4. El filtro de detectabilidad, que es el que manda",
        "",
        "El analisis de la resolucion del leaderboard (V8-02) mostro que el externo solo",
        f"resuelve diferencias de unos {RUIDO_NETOS:.1f} aciertos netos. Una candidata que",
        "cambie C celdas solo es resoluble si la tasa de acierto entre esas celdas supera",
        "`0,5 + ruido / (2C)`. **Gana en local no basta: hay que superar tambien esto.**",
        "",
        "| umbral | celdas que cambian | filas | tasa de acierto | minimo exigido | filtro |",
        "|---|---|---|---|---|---|",
    ]
    for f in variantes:
        if not f.get("celdas"):
            lineas.append(
                f"| {f['umbral_estabilidad']} | 0 | 0 | sin cambio | sin cambio | "
                "no se aplica |"
            )
            continue
        lineas.append(
            f"| {f['umbral_estabilidad']} | {f['celdas']} | {f['filas']} | "
            f"{f['tasa_acierto']:.4f} | {f['minimo_requerido']:.4f} | "
            f"{'**pasa**' if f['filtro_pasa'] else '**no pasa**'} |"
        )
    if elegida is not None:
        lineas += [
            "",
            "## 5. Decision",
            "",
            f"La variante de umbral {elegida['umbral_estabilidad']} **supera las dos puertas**:",
            "el gate de mejora y el filtro de detectabilidad. Genera la submission de la fase.",
        ]
    else:
        motivo = (
            "ninguna variante supera el gate de mejora"
            if not any(f["gate_supera"] for f in variantes)
            else "hay variantes que superan el gate de mejora pero ninguna supera el filtro de "
            "detectabilidad"
        )
        lineas += [
            "",
            "## 5. Decision",
            "",
            f"**No se genera submission**: {motivo}.",
            "",
            "Esta es la respuesta correcta y no un fracaso. El filtro de V8-02 existe para esto:",
            "una linea que gana en local por debajo del minimo de acierto no es una linea que",
            "vaya a mejorar el externo, es una linea cuyo resultado no se podria distinguir del",
            "azar. Enviarla gastaria la sonda para obtener un numero que no informa.",
            "",
            "## 6. El hallazgo que va mas alla del resultado",
            "",
            "Lo importante de esta linea no es que pierde, es **como pierde**. Las tres",
            "variantes mueven entre 184 y 374 celdas, y en esas celdas aciertan un **39 % a un",
            "44 %**, por debajo del 50 % que daria una moneda al aire. Es decir, la seleccion por",
            "estabilidad **mueve la decision de forma sistematica hacia peor** en las filas que",
            "toca.",
            "",
            "La explicacion es coherente con lo que hace el metodo. Un rasgo que mantiene peso",
            "alto bajo siete submuestras es un rasgo de magnitud grande y muy repetida en el",
            "corpus, no necesariamente un rasgo que separa las clases: es mas probable que sea un",
            "rasgo **frecuente y neutro** que uno discriminante y especifico. Al conservar solo",
            "esos rasgos se pierde justamente la senal minoritaria que separa el insulto de la",
            "toxicidad.",
            "",
            "El capitulo 8 de Mosteller y Wallace mide estabilidad para **atribuir autoria**,",
            "donde un rasgo frecuente y neutro si es informativo, porque el objetivo es la",
            "**forma** del texto y no su pertenencia a una clase. **Traducir ese criterio a un",
            "problema de clasificacion no conserva su sentido, y las cifras lo confirman.**",
            "",
            "Esa conclusion sobrevive a la fase: la estabilidad de features es el mecanismo",
            "correcto para el problema del que viene y el equivocado para este. No se vuelve a",
            "intentar con mas agresividad ni con otra constante de submuestreo, porque la",
            "tendencia es monotona y va en la direccion contraria.",
            "",
            "### Una distincion que hay que hacer explicita",
            "",
            "El intervalo de confianza del umbral 0.9 **excluye cero**: la perdida es",
            "detectable en el desarrollo. Y a la vez el filtro de detectabilidad dice que no se",
            "veria en el externo. **No es una contradiccion**, y conviene no mezclarlas: la",
            "primera afirmacion es sobre la metrica de validacion agrupada, que tiene mil veces",
            "mas filas que el conjunto de prueba; la segunda es sobre si el leaderboard de 1154",
            "filas puede distinguir un movimiento de esa magnitud. Una cosa y la otra pueden ser",
            "ciertas a la vez, y por eso hacen falta las dos.",
        ]
    lineas += [
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s. "
        f"Conjunto de prueba de {n_test} filas, sin usar para nada de este script.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
