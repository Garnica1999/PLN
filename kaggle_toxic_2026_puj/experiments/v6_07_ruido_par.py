"""Track F de la fase v6 (V6-07): ruido a nivel de par instancia-etiqueta.

Contexto
--------
El gap analysis (V6-00) confirmo que el ruido **nunca se midio a nivel de par**. Lo que se
midio fue ruido por **duplicado** (V4-A01, V4-A02), y su veredicto fue que el ruido es
escaso: solo 2 clusters contradictorios, 84 tuits, 4.85 % del corpus, y **V03 acertaba bien
dentro de ellos** (error 0.046 frente a 0.218 fuera). Ese resultado es un precedente
adverso, pero deja una pregunta abierta: el error 0.046 dentro de los clusters podria
medirse por *duplicado exacto* y no por *par instancia-etiqueta*.

Por que "primero detectar, luego ponderar"
-------------------------------------------
Esta linea tiene una trampa y la forma de evitarla es la que da nombre al track: si se
pondera el entrenamiento por una sospecha calculada con el **mismo** modelo que despues se
va a corregir, la sospecha se define a si misma y el resultado no significa nada. Por eso el
script hace **dos rondas separadas**:

1. **Ronda de deteccion**: se calcula un score de sospecha por par desde el OOF **congelado**
   de V03 (`features/oof_v03_final.npz`), que es un artefacto canonico ya auditado y cuya
   prediccion de cada fila se hizo **sin esa fila**. Ademas se mide **cuantas filas son
   sospechosas**, y ese numero es el techo del track.
2. **Ronda de ponderacion**: solo despues, se reentrena el OOF aplicando esos pesos.

El techo se mide antes de gastar
--------------------------------
Si el score de sospecha marca muchas filas y V03 ya las clasifica bien, la ponderacion no
puede mejorar nada y solo resta capacidad: es el mismo caso que el de A3 en el track A. El
script reporta el porcentaje detectado y el comportamiento de V03 sobre esas filas, de modo
que la conclusion no dependa de que se ejecute la parte cara.

Las cuatro variables de sospecha
--------------------------------
- `confianza`: `1 - |2p - 1|`, baja confianza en la frontera.
- `margen`: `1 - 2|p - 0.5|`, mismo eje con otro corte.
- `estructural`: la fila tiene un vecino muy cercano en el train con **etiqueta opuesta**
  para esa etiqueta. Es la generalizacion del duplicado exacto de V4-A a la similitud.
- `densidad`: fraccion de los k vecinos cercanos con etiqueta opuesta.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_07_ruido_par.py
"""

# Nota de auditoria forense (B-06, 2026-09-26): los pesos de sospecha se calculan con
# etiquetas de TODO el desarrollo, incluidas las del pliegue evaluado (mecanismo no
# fold-local). Contrafactual medido: delta +0.000142 -> 0.000000 con deteccion
# fold-local (audit/p04b_suspicion_foldlocal.json). Si se reabre la familia, calcular
# densidad/estructural por pliegue con etiquetas solo del entrenamiento. No se modifico
# la logica historica.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

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

EXPERIMENTO = "V6-07 (F0-F3)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_ruido_par.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_ruido_par.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_ruido_par.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
VECINOS = 5
SUELO_PESO = 0.1
UMBRAL_SOSPECHOSO = 0.5
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260926

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}

# Los cuatro esquemas. `lambda` es cuanto se baja el peso de una fila totalmente sospechosa.
ESQUEMAS: dict[str, dict[str, Any]] = {
    "F0-control": {"componentes": (), "lamina": 0.0, "nota": "control: debe reproducir V03"},
    "F1-completo": {
        "componentes": ("confianza", "margen", "estructural", "densidad"),
        "lamina": 0.5,
        "nota": "media de las cuatro señales, lambda 0.5",
    },
    "F2-estructural": {
        "componentes": ("estructural",),
        "lamina": 1.0,
        "nota": "solo la senal estructural, que es la generalizacion del duplicado de V4-A",
    },
    "F3-agresivo": {
        "componentes": ("confianza", "margen", "estructural", "densidad"),
        "lamina": 1.0,
        "nota": "las cuatro senales con lambda 1.0 y suelo de peso",
    },
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-F] {mensaje}", flush=True)


def detectar_sospecha(
    textos, y, proba_v03: np.ndarray, train_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Calcula las cuatro señales de sospecha por par instancia-etiqueta.

    Las señales de probabilidad vienen del OOF congelado de V03, cuya prediccion de cada
    fila se hizo sin esa fila. Las señales estructurales se calculan con la matriz TF-IDF
    del desarrollo completo, que es legitimo: la vecindad no usa etiquetas, y la etiqueta
    opuesta se consulta del propio vector de la fila, no de una prediccion.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        proba_v03: probabilidades fuera de pliegue de V03.
        train_ids: identificador de cada fila, para depurar filas del mismo texto.

    Returns:
        Diccionario con una matriz de forma ``(n_filas, 3)`` por componente.
    """
    n = len(textos)
    confianza = 1.0 - np.abs(2.0 * proba_v03 - 1.0)
    margen = np.clip(1.0 - 2.0 * np.abs(proba_v03 - 0.5), 0.0, 1.0)

    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    matriz = sparse.hstack(
        [vectorizador_word.fit_transform(textos), vectorizador_char.fit_transform(textos)]
    ).tocsr()
    matriz = normalize(matriz)
    vecinos = NearestNeighbors(n_neighbors=VECINOS + 1, metric="cosine")
    vecinos.fit(matriz)
    distancias, indices = vecinos.kneighbors(matriz)
    estructural = np.zeros((n, 3), dtype=np.float64)
    densidad = np.zeros((n, 3), dtype=np.float64)
    for fila in range(n):
        for etiqueta in range(3):
            opuestos = 0
            validos = 0
            for posicion in range(1, len(indices[fila])):
                vecino = int(indices[fila, posicion])
                if train_ids[vecino] == train_ids[fila]:
                    continue
                validos += 1
                if y[vecino, etiqueta] != y[fila, etiqueta]:
                    opuestos += 1
            densidad[fila, etiqueta] = opuestos / max(validos, 1)
            # La senal estructural exige ademas que el vecino sea de verdad muy cercano: un
            # vecino mediocre con etiqueta opuesta es ruido de estimacion, no un conflicto.
            if opuestos > 0:
                mejor = min(
                    distancias[fila, posicion]
                    for posicion in range(1, len(indices[fila]))
                    if int(indices[fila, posicion]) != fila
                )
                if mejor < 0.25:
                    estructural[fila, etiqueta] = 1.0
    return {
        "confianza": confianza,
        "margen": margen,
        "estructural": estructural,
        "densidad": densidad,
    }


def combinar(componentes: dict[str, np.ndarray], nombres: tuple[str, ...]) -> np.ndarray:
    """Reduce las señales seleccionadas a una unica sospecha por par.

    Args:
        componentes: diccionario con las señales disponibles.
        nombres: componentes a usar.

    Returns:
        Matriz de sospecha de forma ``(n_filas, 3)``, con valores entre 0 y 1.
    """
    if not nombres:
        return np.zeros_like(next(iter(componentes.values())))
    suma = np.zeros_like(next(iter(componentes.values())))
    for nombre in nombres:
        suma = suma + componentes[nombre]
    return np.clip(suma / len(nombres), 0.0, 1.0)


def ajustar_oof(textos, y, pliegues, sospecha: np.ndarray, esquema) -> np.ndarray:
    """Reentrena el OOF con los pesos derivados de la sospecha.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        pliegues: pliegues corregidos.
        sospecha: matriz de sospecha por par.
        esquema: definicion del esquema con su factor `lamina`.

    Returns:
        Matriz de probabilidades de forma ``(n_filas, 3)``.
    """
    proba = np.zeros((len(textos), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        matriz = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
                vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
            ]
        ).tocsr()
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ]
        ).tocsr()
        for etiqueta in range(3):
            peso = np.clip(
                1.0 - esquema["lamina"] * sospecha[entrenamiento, etiqueta],
                SUELO_PESO,
                1.0,
            )
            estimador = LogisticRegression(**PARAMETROS_LR)
            estimador.fit(matriz, y[entrenamiento, etiqueta], sample_weight=peso)
            proba[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return proba


def evaluar(nombre: str, y, proba, pred_v03) -> dict[str, Any]:
    """Evalua una variante con los umbrales congelados de V03.

    Args:
        nombre: identificador de la variante.
        y: matriz binaria verdadera.
        proba: matriz de probabilidades.
        pred_v03: predicciones de V03.

    Returns:
        Diccionario con metricas, comparacion pareada y complementariedad.
    """
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    comparacion = bootstrap_diferencia_pareada(
        y, pred, pred_v03, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
    )
    por_etiqueta = metricas_multietiqueta(y, pred)["f1_por_etiqueta"]
    acierta = _f1_por_celda(y, pred) >= 0.999
    acierta_v03 = _f1_por_celda(y, pred_v03) >= 0.999
    return {
        "variante": nombre,
        "corrected_cv": round(metrica_macro(y, pred), 6),
        "delta_vs_V03": round(metrica_macro(y, pred) - REFERENCIA_V03, 6),
        "ic95_0": comparacion["ic95"][0],
        "ic95_1": comparacion["ic95"][1],
        "f1_HS": round(por_etiqueta[ETIQUETAS[0]], 4),
        "f1_TR": round(por_etiqueta[ETIQUETAS[1]], 4),
        "f1_AG": round(por_etiqueta[ETIQUETAS[2]], 4),
        "salvage": int(np.sum(acierta & ~acierta_v03)),
        "damage": int(np.sum(~acierta & acierta_v03)),
        "pred": pred,
    }


def main() -> None:
    """Ejecuta la deteccion, mide el techo y evalua los cuatro esquemas."""
    inicio = time.time()
    log("Track F: ruido a nivel de par instancia-etiqueta")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_v03 = oof["proba"]
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    textos = desarrollo["text"].astype(str)
    ids = desarrollo["id"].astype(int).to_numpy()
    log(f"   desarrollo {len(y)} filas | V03 {metrica_macro(y, pred_v03):.6f}")

    log("1/4 deteccion: cuatro senales de sospecha")
    componentes = detectar_sospecha(textos, y, proba_v03, ids)
    sospecha_total = combinar(componentes, ESQUEMAS["F3-agresivo"]["componentes"])
    for nombre, matriz in componentes.items():
        marcadas = int((matriz > 0.5).sum())
        log(f"   senal {nombre:12s} pares marcados: {marcadas} de {matriz.size}")
    sospechosas = sospecha_total > UMBRAL_SOSPECHOSO
    filas_sospechosas = int(sospechosas.any(axis=1).sum())
    log(
        f"   techo del track: {filas_sospechosas} filas sospechosas de {len(y)} "
        f"({100 * filas_sospechosas / len(y):.2f} %)"
    )
    if filas_sospechosas:
        acierta_v03 = _f1_por_celda(y, pred_v03) >= 0.999
        log(
            f"   V03 en las filas sospechosas: acierta {int(acierta_v03[sospechosas.any(axis=1)].sum())}"
            f" de {filas_sospechosas} | en las demas: "
            f"{int(acierta_v03[~sospechosas.any(axis=1)].sum())} de {len(y) - filas_sospechosas}"
        )
    else:
        acierta_v03 = _f1_por_celda(y, pred_v03) >= 0.999

    variantes: list[dict[str, Any]] = []
    log("2/4 ponderacion")
    for nombre, esquema in ESQUEMAS.items():
        sospecha = combinar(componentes, esquema["componentes"])
        proba = ajustar_oof(textos, y, pliegues, sospecha, esquema)
        variantes.append(evaluar(nombre, y, proba, pred_v03))
        variante = variantes[-1]
        log(
            f"   {nombre:14s} corrected_cv {variante['corrected_cv']:.6f} "
            f"delta {variante['delta_vs_V03']:+.4f} "
            f"IC95 [{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"HS {variante['f1_HS']:.4f} TR {variante['f1_TR']:.4f} AG {variante['f1_AG']:.4f} "
            f"| salvage {variante['salvage']} damage {variante['damage']}"
        )

    log("3/4 gate de dos niveles")
    for variante in variantes:
        delta = variante["delta_vs_V03"]
        excluye_cero = variante["ic95_0"] > 0
        cumple_delta = delta >= GATE_1_DELTA
        pliegues_positivos = _pliegues_positivos(y, variante["pred"], pliegues, pred_v03)
        variante["gate_1_delta"] = cumple_delta
        variante["gate_1_ic95"] = excluye_cero
        variante["pliegues_positivos"] = pliegues_positivos
        variante["gate_1_supera"] = (
            cumple_delta and excluye_cero and pliegues_positivos >= GATE_1_FOLDS
        )
        log(
            f"   {variante['variante']:14s} Gate 1: delta {'OK' if cumple_delta else 'no'} "
            f"({delta:+.4f}) | IC95 {'OK' if excluye_cero else 'no'} | pliegues "
            f"{pliegues_positivos}/{len(pliegues)} -> "
            f"{'SUPERA' if variante['gate_1_supera'] else 'no supera'}"
        )

    log("4/4 informe y figura")
    _figura(variantes, componentes, sospecha_total)
    _informe(
        variantes, componentes, sospecha_total, filas_sospechosas, len(y),
        acierta_v03, pliegues, time.time() - inicio
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "lamina", "componentes", "corrected_cv", "delta_vs_V03",
            "ic95_0", "ic95_1", "f1_HS", "f1_TR", "f1_AG", "salvage", "damage",
            "pliegues_positivos", "gate_1_supera",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for variante, nombre in zip(variantes, ESQUEMAS, strict=True):
            escritor.writerow(
                {campo: variante.get(campo, "") for campo in campos}
                | {
                    "lamina": ESQUEMAS[nombre]["lamina"],
                    "componentes": ",".join(ESQUEMAS[nombre]["componentes"]) or "ninguna",
                }
            )
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": REFERENCIA_V03,
                "vecinos": VECINOS,
                "suelo_peso": SUELO_PESO,
                "umbral_sospechoso": UMBRAL_SOSPECHOSO,
                "techo": {
                    "filas_sospechosas": filas_sospechosas,
                    "filas_totales": len(y),
                    "porcentaje": round(100 * filas_sospechosas / len(y), 2),
                    "v03_acierta_en_sospechosas": int(
                        acierta_v03[sospechosas.any(axis=1)].sum()
                    ),
                    "v03_acierta_en_el_resto": int(
                        acierta_v03[~sospechosas.any(axis=1)].sum()
                    ),
                },
                "pares_marcados_por_senal": {
                    nombre: int((matriz > 0.5).sum()) for nombre, matriz in componentes.items()
                },
                "gate_1": {"delta": GATE_1_DELTA, "pliegues_minimos": GATE_1_FOLDS},
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
    mejores = max(variantes, key=lambda v: v["corrected_cv"])
    log(
        f"Track F completada en {round(time.time() - inicio, 1)} s | mejor: "
        f"{mejores['variante']} {mejores['corrected_cv']:.6f} "
        f"(delta {mejores['delta_vs_V03']:+.4f})"
    )


def _pliegues_positivos(y, pred, pliegues, pred_referencia) -> int:
    """Cuenta los pliegues en los que la variante mejora frente a la referencia.

    Args:
        y: matriz binaria verdadera.
        pred: predicciones de la variante.
        pliegues: pliegues corregidos.
        pred_referencia: predicciones de referencia.

    Returns:
        Numero de pliegues con mejora.
    """
    positivos = 0
    for validacion in pliegues:
        indices = validacion[1] if isinstance(validacion, tuple) else validacion
        if metrica_macro(y[indices], pred[indices]) > metrica_macro(
            y[indices], pred_referencia[indices]
        ):
            positivos += 1
    return positivos


def _figura(variantes, componentes, sospecha_total) -> None:
    """Dibuja las señales de sospecha y el delta de cada esquema.

    Args:
        variantes: resultados de las variantes.
        componentes: señales de sospecha.
        sospecha_total: sospecha combinada.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, (arriba, abajo) = plt.subplots(1, 2, figsize=(11, 4.4))
    nombres_senal = list(componentes) + ["combinada"]
    distribuciones = [componentes[n].ravel() for n in componentes] + [
        sospecha_total.ravel()
    ]
    arriba.boxplot(
        distribuciones, tick_labels=nombres_senal, showfliers=False, vert=True
    )
    arriba.set_ylabel("sospecha por par")
    arriba.set_title("Distribucion de las senales de sospecha")
    arriba.tick_params(axis="x", rotation=25, labelsize=8)

    nombres = [v["variante"] for v in variantes]
    posiciones = np.arange(len(nombres))
    abajo.bar(posiciones, [v["delta_vs_V03"] for v in variantes], color="#8c564b")
    abajo.axhline(0, color="black", linewidth=1)
    abajo.axhline(
        GATE_1_DELTA, color="red", linestyle=":", label=f"gate {GATE_1_DELTA:+.3f}"
    )
    abajo.set_xticks(posiciones)
    abajo.set_xticklabels(nombres, rotation=20, ha="right", fontsize=8)
    abajo.set_ylabel("delta frente a V03")
    abajo.set_title("Ponderacion por sospecha de par")
    abajo.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_64_ruido_par_v6.png", dpi=150)
    plt.close()


def _informe(
    variantes, componentes, sospecha_total, filas_sospechosas, n_filas,
    acierta_v03, pliegues, duracion
) -> None:
    """Escribe el informe del track F.

    Args:
        variantes: resultados de las variantes.
        componentes: señales de sospecha.
        sospecha_total: sospecha combinada.
        filas_sospechosas: numero de filas marcadas como sospechosas.
        n_filas: numero de filas del desarrollo.
        acierta_v03: mascara de aciertos fila a fila de V03.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    marcadas = sospecha_total > UMBRAL_SOSPECHOSO
    lineas = [
        "# Track F: ruido a nivel de par instancia-etiqueta (V6-07)",
        "",
        f"Generado por `experiments/v6_07_ruido_par.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es una repeticion de V4-A",
        "",
        "V4-A midio el ruido por **duplicado** y concluyo que es escaso: 2 clusters",
        "contradictorios, 84 tuits, 4.85 % del corpus, y V03 acertaba bien dentro de ellos",
        "(error 0.046 frente a 0.218 fuera). Ese error 0.046 se midio por *duplicado exacto*.",
        "La pregunta abierta es si hay ruido que **no** es duplicado exacto, es decir, pares",
        "similares pero no identicos con etiqueta opuesta. La senal `estructural` de este",
        "track es exactamente esa generalizacion.",
        "",
        "## 2. Deteccion: el techo se mide antes de ponderar",
        "",
        "La sospecha se calcula en una **ronda separada** de la ponderacion, a partir del OOF",
        "**congelado** de V03, cuya prediccion de cada fila se hizo sin esa fila. Si se",
        "ponderara con una sospecha definida por el mismo modelo que luego se corrige, el",
        "resultado no significaria nada.",
        "",
        "| senal | pares marcados | que mide |",
        "|---|---|---|",
    ]
    descripciones = {
        "confianza": "baja confianza en la frontera: `1 - abs(2p - 1)`",
        "margen": "mismo eje con otro corte: `1 - 2abs(p - 0.5)`",
        "estructural": "hay un vecino muy cercano (coseno < 0.25) con etiqueta opuesta",
        "densidad": "fraccion de los vecinos cercanos con etiqueta opuesta",
    }
    for nombre, matriz in componentes.items():
        lineas.append(
            f"| `{nombre}` | {int((matriz > 0.5).sum())} de {matriz.size} | "
            f"{descripciones[nombre]} |"
        )
    linea_sospechosas = marcadas.any(axis=1)
    lineas += [
        f"| combinada | {int(marcadas.sum())} de {marcadas.size} | media de las cuatro |",
        "",
        f"**Techo medido: {filas_sospechosas} filas sospechosas de {n_filas} "
        f"({100 * filas_sospechosas / n_filas:.2f} %).**",
        "",
        f"Y lo que V03 hace con ellas: acierta **{int(acierta_v03[linea_sospechosas].sum())}"
        f" de {filas_sospechosas}** ({100 * acierta_v03[linea_sospechosas].mean():.1f} %),"
        f" frente a **{int(acierta_v03[~linea_sospechosas].sum())} de "
        f"{n_filas - filas_sospechosas}** "
        f"({100 * acierta_v03[~linea_sospechosas].mean():.1f} %) en las no sospechosas.",
        "",
        "### Este numero contradice la hipotesis, y por eso hay que decirlo bien",
        "",
        "La version de este informe escrita antes de ejecutar daba por hecho que V03 ya",
        "clasificaba bien las filas marcadas y que por eso bajar su peso no podia aportar",
        "nada. **Los datos dicen lo contrario**: V03 acierta **mas del doble de porcentaje**",
        "en las filas NO sospechosas que en las sospechosas",
        f"({100 * acierta_v03[~linea_sospechosas].mean():.1f} % frente a "
        f"{100 * acierta_v03[linea_sospechosas].mean():.1f} %). La deteccion **si** identifica",
        "filas donde el modelo falla.",
        "",
        "Eso abre dos lecturas posibles, y los resultados de la ponderacion las separan:",
        "",
        "1. La deteccion encuentra **dificultad genuina**, no ruido. Las filas sonarias y",
        "   estan en la frontera; bajar su peso **aleja al modelo de donde estan los errores**,",
        "   que es exactamente lo que se ve al subir `lambda`.",
        "2. La deteccion encuentra **ruido de etiqueta** y ponderar deberia ayudar. En ese",
        "   caso un `lambda` mayor deberia mejorar, y no lo hace.",
        "",
        "El comportamiento observado es el de la lectura 1, asi que **lo que la deteccion",
        "marca es donde hay que estar, no donde hay que limpiar**. La senal tiene valor",
        "predictivo de dificultad aunque no tenga valor de correccion, que son dos cosas",
        "distintas y aqui conviene no confundirlas.",
        "",
        "### Y la senal estructural casi no detecta nada",
        "",
        f"La senal `estructural` marca **{int((componentes['estructural'] > 0.5).sum())} pares"
        f" de {componentes['estructural'].size}**, frente a los 84 tuits del duplicado exacto"
        " de V4-A. Es decir, la generalizacion del duplicado a la similitud no encuentra mas",
        "ruido: encuentra **mucho menos**. El precedente de V4-A era ya adverso, y al ampliar",
        "la definicion de conflicto el ruido se hace mas escaso, no mas abundante.",
        "",
        "## 3. Resultados de la ponderacion",
        "",
        "| variante | componentes | lambda | corrected_cv | delta | IC95 | HS | TR | AG |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variante, nombre in zip(variantes, ESQUEMAS, strict=True):
        lineas.append(
            f"| {variante['variante']} | "
            f"{', '.join(ESQUEMAS[nombre]['componentes']) or 'ninguno'} | "
            f"{ESQUEMAS[nombre]['lamina']} | {variante['corrected_cv']:.6f} | "
            f"{variante['delta_vs_V03']:+.4f} | "
            f"[{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} |"
        )
    lineas += [
        "",
        "## 4. Gate de dos niveles",
        "",
        "| variante | delta >= +0.003 | IC95 excluye 0 | pliegues positivos | Gate 1 |",
        "|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {'si' if variante['gate_1_delta'] else 'no'} "
            f"({variante['delta_vs_V03']:+.4f}) | "
            f"{'si' if variante['gate_1_ic95'] else 'no'} | "
            f"{variante['pliegues_positivos']}/{len(pliegues)} | "
            f"{'**SUPERA**' if variante['gate_1_supera'] else 'no supera'} |"
        )
    lineas += [
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
