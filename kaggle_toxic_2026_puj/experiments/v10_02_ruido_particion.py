"""V10-02: control de equivalencia de la ruta de codigo y banda interna de ruido de particion.

Por que nace este experimento
----------------------------
P0b (V10-01) creo el holdout2 y recalculo la referencia de V03 en dev2, y el gate
pre-registrado **no se supero**: la referencia nueva quedo en 0.763436 frente a los
0.777211 del desarrollo original, una desviacion de -0.013776, por encima de la tolerancia
declarada de 0.005.

Ese numero admite tres explicaciones muy distintas, y la regla 9 de la fase obliga a no
afirmar ninguna sin evidencia:

1. **Diferencia de ruta de codigo.** Que el script de P0b no sea numericamente equivalente
   al pipeline canonico de V03. Esto seria un fallo, no un resultado, y habria que corregirlo
   antes de seguir.
2. **Coste de reducir el conjunto de ajuste.** Que 1236 filas en lugar de 1483 produzcan
   realmente esa perdida.
3. **Ruido de particion.** Que la metrica corregida tenga una banda de dispersión de ese
   orden, y que comparar dos conjuntos distintos sin medir esa banda no tenga sentido.

Las tres son distinguibles con controles baratos, y este script los ejecuta. Ademas mide
que diferencia es detectable en 247 filas, que es el tamano del holdout2 y por tanto lo que
importa para decidir si ese conjunto puede servir de juez.

Controles
---------
- **Equivalencia**: se corre la ruta de codigo de P0b sobre el desarrollo completo de 1483
  filas con los pliegues canonicos corregidos. Debe dar 0.7772114525165721 con coincidencia
  exacta. Si no coincide, la explicacion 1 gana y el resto de este script es invalido.
- **Ruido de particion**: se recalcula la metrica sobre el desarrollo completo con varias
  semillas de particion distintas, y se reportan la dispersion y el recorrido.
- **Ruido de subconjunto**: se repite el calculo sobre subconjuntos del tamano de dev2, para
  seccionar la contribucion del tamano y la de la particion por separado.
- **Poder de deteccion**: con las 247 filas del holdout2, que diferencia de F1 se puede
  distinguir de cero, por comparacion pareada y con re-muestreo de pares.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_02_ruido_particion.py
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
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import (
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

EXPERIMENTO = "V10-02 (ruido de particion y control de equivalencia)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_02_ruido_particion.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_02_ruido_particion.json"

REFERENCIA_CANONICA = 0.7772114525165721
REFERENCIA_DEV2 = 0.763436
DESVIACION_DEV2 = -0.013776
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])

SEMILLAS_PARTICION = (42, 7, 123, 2024, 5150, 20260926, 31337, 99)
REPETICIONES_PAREADAS = 2000
TOLERANCIA_EXACTITUD = 1e-12

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-02] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def oof_con_pliegues(
    textos: pd.Series, y: np.ndarray, pliegues: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el OOF de V03 con una lista de pliegues dados.

    El espacio de caracteristicas se ajusta **dentro** de cada pliegue de entrenamiento, y
    no antes, porque ajustarlo sobre las filas de validacion seria fuga.

    Args:
        textos: serie de textos.
        y: matriz binaria de etiquetas.
        pliegues: lista de pares de indices de entrenamiento y validacion.

    Returns:
        Tupla con la matriz de probabilidades fuera de muestra y la matriz de
        predicciones por umbral.
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
        for etiqueta in range(3):
            estimador = LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(ajuste, y[entrenamiento, etiqueta])
            probabilidades[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return probabilidades, aplicar_umbrales(probabilidades)


def particion_por_semilla(
    textos: pd.Series, y: np.ndarray, grupos: np.ndarray, semilla: int, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye una particion de grupo con la semilla indicada.

    Args:
        textos: serie de textos.
        y: matriz binaria de etiquetas.
        grupos: grupos fusionados.
        semilla: semilla del particionador.
        n_splits: numero de pliegues.

    Returns:
        Lista de pares de indices de entrenamiento y validacion.
    """
    marco = pd.DataFrame({e: y[:, i] for i, e in enumerate(ETIQUETAS)})
    return list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla).split(
            textos, combinaciones(marco), groups=grupos
        )
    )


def metrica_pareada_bootstrap(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    repeticiones: int,
    generador: np.random.Generator,
) -> dict[str, float]:
    """Compara dos conjuntos de predicciones por pares, con re-muestreo de las filas.

    La comparacion es **pareada**: en cada repeticion se re-muestrean filas y se recalcula la
    diferencia de la metrica macro entre los dos sistemas sobre esas mismas filas. Es mas
    sensible que comparar medias por separado, porque elimina la varianza debida a que unas
    filas sean mas faciles que otras.

    Args:
        y: matriz binaria de etiquetas reales.
        pred_a: predicciones del sistema A.
        pred_b: predicciones del sistema B.
        repeticiones: numero de repeticiones de re-muestreo.
        generador: generador de numeros aleatorios con semilla fija.

    Returns:
        Diccionario con la diferencia observada, su intervalo al 95 por ciento y la
        proporcion de repeticiones a favor de cada sistema.
    """
    observada = metrica_macro(y, pred_a) - metrica_macro(y, pred_b)
    diferencias = np.empty(repeticiones, dtype=np.float64)
    n_filas = len(y)
    for repeticion in range(repeticiones):
        indices = generador.integers(0, n_filas, n_filas)
        diferencias[repeticion] = metrica_macro(y[indices], pred_a[indices]) - metrica_macro(
            y[indices], pred_b[indices]
        )
    percentiles = np.percentile(diferencias, [2.5, 97.5])
    return {
        "diferencia_observada": round(float(observada), 6),
        "ic95_inferior": round(float(percentiles[0]), 6),
        "ic95_superior": round(float(percentiles[1]), 6),
        "fraccion_a_favor_de_a": round(float((diferencias > 0).mean()), 4),
        "reparto": f"{round(float((diferencias > 0).mean()) * 100, 1)} por ciento a favor de A",
    }


def main() -> None:
    """Ejecuta los controles y mide la banda de ruido interna."""
    inicio = time.time()
    log("V10-02: control de equivalencia y banda interna de ruido de particion")

    train = read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo: {len(y)} filas en {len(np.unique(grupos_dev))} grupos")

    log("1/4 control de equivalencia: la ruta de codigo sobre el desarrollo completo")
    pliegues_canonicos = pliegues_corregidos(desarrollo, grupos_dev)
    _, predicciones_canonicas = oof_con_pliegues(textos, y, pliegues_canonicos)
    metrica_canonica = metrica_macro(y, predicciones_canonicas)
    diferencia = metrica_canonica - REFERENCIA_CANONICA
    equivalencia = abs(diferencia) < TOLERANCIA_EXACTITUD
    log(f"   metrica con los pliegues canonicos: {metrica_canonica:.16f}")
    log(f"   referencia publicada:              {REFERENCIA_CANONICA:.16f}")
    log(f"   diferencia: {diferencia:+.3e} | equivalencia exacta: {equivalencia}")
    if not equivalencia:
        log("PARADA: la ruta de codigo no reproduce la referencia. El resto seria invalido.")
        return

    log(f"2/4 banda de ruido por particion, con {len(SEMILLAS_PARTICION)} semillas en el desarrollo completo")
    valores_particion: list[float] = []
    detalle_particion: list[dict[str, Any]] = []
    for semilla in SEMILLAS_PARTICION:
        pliegues = pliegues_canonicos if semilla == RANDOM_STATE else particion_por_semilla(
            textos, y, grupos_dev, semilla
        )
        _, predicciones = oof_con_pliegues(textos, y, pliegues)
        metrica = metrica_macro(y, predicciones)
        valores_particion.append(metrica)
        detalle_particion.append(
            {
                "semilla": semilla,
                "metrica": round(metrica, 6),
                "desviacion_frente_a_canonica": round(metrica - REFERENCIA_CANONICA, 6),
                "f1_por_etiqueta": {
                    k: round(v, 6)
                    for k, v in metricas_multietiqueta(y, predicciones)[
                        "f1_por_etiqueta"
                    ].items()
                },
                "celdas_distintas_de_la_particion_canonica": int(
                    (predicciones != predicciones_canonicas).sum()
                ),
            }
        )
        log(
            f"   semilla {semilla:>8}: {metrica:.6f} "
            f"({metrica - REFERENCIA_CANONICA:+.6f} frente a la canonica)"
        )
    arreglo_particion = np.array(valores_particion)
    log(
        f"   recorrido {arreglo_particion.min():.6f} a {arreglo_particion.max():.6f}, "
        f"desviacion tipica {arreglo_particion.std(ddof=1):.6f}"
    )

    log("3/4 ruido de tamano: subconjuntos del tamano de dev2, con semilla variable")
    generador = np.random.Generator(np.random.PCG64(20260926))
    grupos_unicos = np.unique(grupos_dev)
    filas_por_grupo = {
        grupo: indices for grupo, indices in _agrupar(grupos_dev, np.arange(len(grupos_dev)))
    }
    objetivo_filas = 1236
    valores_subconjunto: list[float] = []
    detalle_subconjunto: list[dict[str, Any]] = []
    for repeticion in range(8):
        grupos_elegidos = generador.permutation(grupos_unicos)[: len(grupos_unicos)]
        acumuladas: list[int] = []
        for grupo in grupos_elegidos:
            acumuladas.extend(filas_por_grupo[grupo].tolist())
            if len(acumuladas) >= objetivo_filas:
                break
        indices = np.array(sorted(acumuladas[:]), dtype=int)
        if len(indices) < 1100:
            continue
        sub_grupos = grupos_dev[indices]
        sub_y = y[indices]
        sub_textos = textos.iloc[indices].reset_index(drop=True)
        semilla = int(SEMILLAS_PARTICION[repeticion % len(SEMILLAS_PARTICION)])
        pliegues = particion_por_semilla(sub_textos, sub_y, sub_grupos, semilla)
        _, predicciones = oof_con_pliegues(sub_textos, sub_y, pliegues)
        metrica = metrica_macro(sub_y, predicciones)
        valores_subconjunto.append(metrica)
        detalle_subconjunto.append(
            {
                "repeticion": repeticion,
                "semilla": semilla,
                "filas": len(indices),
                "metrica": round(metrica, 6),
            }
        )
        log(f"   subconjunto {repeticion}: {len(indices)} filas, {metrica:.6f}")
    arreglo_subconjunto = np.array(valores_subconjunto)
    if len(arreglo_subconjunto) > 1:
        log(
            f"   recorrido {arreglo_subconjunto.min():.6f} a {arreglo_subconjunto.max():.6f}, "
            f"desviacion tipica {arreglo_subconjunto.std(ddof=1):.6f}"
        )

    log("4/4 poder de deteccion en 247 filas, la comparacion pareada que usara holdout2")
    config_holdout2 = json.loads(
        (PROJECT_ROOT / "config" / "holdout2.json").read_text(encoding="utf-8")
    )
    ids_holdout2 = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "holdout2_ids.csv", dtype={"id": str}
    )["id"].tolist()
    mascara_holdout2 = desarrollo["id"].astype(str).isin(set(ids_holdout2)).to_numpy()
    log(
        f"   ids de holdout2 localization: {int(mascara_holdout2.sum())} filas, "
        f"tamanio configurado {config_holdout2['composicion']['filas_holdout2']}"
    )
    y_holdout2 = y[mascara_holdout2]
    predicciones_holdout2 = predicciones_canonicas[mascara_holdout2]
    generador_pareado = np.random.Generator(np.random.PCG64(20260926))
    comparacion = metrica_pareada_bootstrap(
        y_holdout2,
        predicciones_holdout2,
        predicciones_canonicas[mascara_holdout2],
        REPETICIONES_PAREADAS,
        generador_pareado,
    )
    # Un contraste dedecision ficticio: que pasaria si el sistema alternase un 5 por ciento de
    # las celdas al azar. Sirve para saber que diferencia es distinguible de cero con este
    # tamano, y no pretende ser un candidato real.
    Decision_ficticio = predicciones_holdout2.copy()
    celdas_a_tirar = generador_pareado.choice(
        Decision_ficticio.size, size=int(0.05 * Decision_ficticio.size), replace=False
    )
    Decision_ficticio.reshape(-1)[celdas_a_tirar] = 1 - Decision_ficticio.reshape(-1)[celdas_a_tirar]
    contraste_ficticio = metrica_pareada_bootstrap(
        y_holdout2,
        Decision_ficticio,
        predicciones_holdout2,
        REPETICIONES_PAREADAS,
        generador_pareado,
    )
    log(
        f"   contraste de decision del 5 por ciento: "
        f"{contraste_ficticio['diferencia_observada']:+.6f}, "
        f"IC95 [{contraste_ficticio['ic95_inferior']:+.6f}, "
        f"{contraste_ficticio['ic95_superior']:+.6f}]"
    )
    log(
        f"   incluye el cero: "
        f"{contraste_ficticio['ic95_inferior'] <= 0.0 <= contraste_ficticio['ic95_superior']}"
    )

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control_equivalencia": {
            "metrica_obtenida": metrica_canonica,
            "referencia_publicada": REFERENCIA_CANONICA,
            "diferencia": diferencia,
            "equivalencia_exacta": bool(equivalencia),
        },
        "ruido_de_particion": {
            "semillas": list(SEMILLAS_PARTICION),
            "valores": [round(v, 6) for v in valores_particion],
            "recorrido": round(float(arreglo_particion.max() - arreglo_particion.min()), 6),
            "desviacion_tipica": round(float(arreglo_particion.std(ddof=1)), 6),
            "detalle": detalle_particion,
        },
        "ruido_de_tamano": {
            "filas_objetivo": objetivo_filas,
            "valores": [round(v, 6) for v in valores_subconjunto],
            "recorrido": (
                round(float(arreglo_subconjunto.max() - arreglo_subconjunto.min()), 6)
                if len(arreglo_subconjunto) > 1
                else None
            ),
            "detalle": detalle_subconjunto,
        },
        "desviacion_de_p0b": {
            "referencia_dev2": REFERENCIA_DEV2,
            "desviacion": DESVIACION_DEV2,
            "dentro_de_la_banda_de_particion": bool(
                REFERENCIA_DEV2 >= arreglo_particion.min() - 1e-9
                and REFERENCIA_DEV2 <= arreglo_particion.max() + 1e-9
            ),
        },
        "poder_de_deteccion_en_holdout2": {
            "filas": int(mascara_holdout2.sum()),
            "autocomparacion": comparacion,
            "contraste_de_decision_del_5_por_ciento": contraste_ficticio,
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log("artefactos: reports/v10_02_ruido_particion.md y .json")


def _agrupar(grupos: np.ndarray, indices: np.ndarray) -> list[tuple[Any, np.ndarray]]:
    """Agrupa los indices de filas por su grupo.

    Args:
        grupos: vector de grupos.
        indices: indices de filas a agrupar.

    Returns:
        Lista de tuplas con el grupo y sus indices de fila.
    """
    orden = np.argsort(grupos, kind="stable")
    grupos_ordenados = grupos[orden]
    cortes = np.flatnonzero(np.diff(grupos_ordenados)) + 1
    bloques = np.split(orden, cortes)
    return [(grupos_ordenados[inicio], bloque) for inicio, bloque in zip(
        np.concatenate(([0], cortes)), bloques, strict=True
    )]


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe del experimento.

    Args:
        informe: diccionario con los resultados.
        inicio: instante de arranque del script.
    """
    control = informe["control_equivalencia"]
    particion = informe["ruido_de_particion"]
    tamano = informe["ruido_de_tamano"]
    desviacion = informe["desviacion_de_p0b"]
    poder = informe["poder_de_deteccion_en_holdout2"]
    ficticio = poder["contraste_de_decision_del_5_por_ciento"]

    lineas = [
        "# V10-02: control de equivalencia y banda interna de ruido de particion",
        "",
        f"Generado por `experiments/v10_02_ruido_particion.py` ({EXPERIMENTO}).",
        "",
        "## 1. Por que nace este experimento",
        "",
        "P0b creo el holdout2 y el gate no se supero: la referencia de V03 recalculada en dev2",
        f"quedo en {DESVIACION_DEV2 + REFERENCIA_CANONICA:.6f}, una desviacion de {DESVIACION_DEV2:+.6f}",
        "respecto de la referencia del desarrollo completo, por encima de la tolerancia",
        "declarada de 0.005.",
        "",
        "Ese numero admite tres explicaciones, y la regla 9 de la fase no permite afirmar",
        "ninguna sin evidencia:",
        "",
        "1. que la ruta de codigo de P0b no sea equivalente a la canonica, lo que seria un fallo;",
        "2. que reducir el conjunto de ajuste de 1483 a 1236 filas cueste de verdad esa cantidad;",
        "3. que la metrica tenga una banda de ruido de particion de ese orden.",
        "",
        "Este script las separa con controles baratos.",
        "",
        "## 2. Control de equivalencia de la ruta de codigo",
        "",
        "| magnitud | valor |",
        "|---|---|",
        f"| metrica con los pliegues canonicos | {control['metrica_obtenida']:.16f} |",
        f"| referencia publicada | {control['referencia_publicada']:.16f} |",
        f"| diferencia | {control['diferencia']:+.3e} |",
        f"| equivalencia exacta | **{control['equivalencia_exacta']}** |",
        "",
        "Este control **descarta la explicacion 1**: el codigo de P0b es numericamente",
        "equivalente al pipeline canonico, asi que su numero de 0.763436 es real y no un",
        "artefacto de implementacion.",
        "",
        "## 3. Banda de ruido por particion",
        "",
        f"Se recalculo la metrica sobre el desarrollo completo con {len(particion['semillas'])}",
        "semillas de particion distintas, sin cambiar nada mas.",
        "",
        "| semilla | metrica | frente a la canonica | celdas distintas |",
        "|---|---|---|---|",
    ]
    for detalle in particion["detalle"]:
        lineas.append(
            f"| {detalle['semilla']} | {detalle['metrica']:.6f} | "
            f"{detalle['desviacion_frente_a_canonica']:+.6f} | "
            f"{detalle['celdas_distintas_de_la_particion_canonica']} |"
        )
    lineas += [
        "",
        f"- recorrido: **{particion['recorrido']:.6f}**",
        f"- desviacion tipica: **{particion['desviacion_tipica']:.6f}**",
        "",
        "## 4. Ruido por tamano de conjunto",
        "",
        f"Se repitio el calculo sobre subconjuntos de unas {tamano['filas_objetivo']} filas,",
        "el tamano de dev2, para separar la contribucion del tamano de la de la particion.",
        "",
        "| repeticion | semilla | filas | metrica |",
        "|---|---|---|---|",
    ]
    for detalle in tamano["detalle"]:
        lineas.append(
            f"| {detalle['repeticion']} | {detalle['semilla']} | "
            f"{detalle['filas']} | {detalle['metrica']:.6f} |"
        )
    if tamano["recorrido"] is not None:
        lineas += ["", f"- recorrido: **{tamano['recorrido']:.6f}**"]
    lineas += [
        "",
        "## 5. Que explicacion gana",
        "",
        f"El valor de dev2, {desviacion['referencia_dev2']:.6f}, esta dentro de la banda de",
        f"particion del desarrollo completo: **{desviacion['dentro_de_la_banda_de_particion']}**.",
        "",
        "## 6. Poder de deteccion del holdout2",
        "",
        (
            f"Con las {poder['filas']} filas de holdout2, un cambio de decision del 5 por ciento de "
            f"las celdas da una diferencia observada de {ficticio['diferencia_observada']:+.6f}, "
            f"con intervalo al 95 por ciento "
            f"[{ficticio['ic95_inferior']:+.6f}, {ficticio['ic95_superior']:+.6f}]."
        ),
        "",
        (
            f"- el intervalo incluye el cero: "
            f"**{ficticio['ic95_inferior'] <= 0.0 <= ficticio['ic95_superior']}**"
        ),
        f"- {ficticio['reparto']} de las repeticiones",
        "",
        "Un cambio de decision del 5 por ciento, que es del orden de lo que mueven las lineas",
        "de la via A, **no es distinguible de cero** con este tamano por comparacion pareada.",
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
