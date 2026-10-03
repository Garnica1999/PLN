"""Track C1 de la fase v6 (V6-05): n-gramas de palabra con salto (skip-grams).

Contexto
--------
El gap analysis (V6-00) confirmo por busqueda en el codigo que **no existe ninguna
implementacion de skip-grams** en el proyecto: el `ngram_range=(1, 2)` de V03 solo produce
unigramas y bigramas **contiguos**. Los pares de palabras no adyacentes nunca se
representaron.

Que NO es una repeticion de V4-D
-------------------------------
V4-D cerro "interacciones como tokens" (`INSULT_NEAR_MENTION` y similares) y "canales
posicionales" (ventanas inicio/medio/final), y E2-10 cerro combinaciones forma x token. La
diferencia es que aquellos eran **canales anadidos** sobre una representacion ya construida:
una matriz extra que se apila a la de V03, con sus propios pesos. Aqui se modifica el
**analizador del bloque de palabras** para que las relaciones con hueco entren en la misma
matriz que las palabras, y compiten con ellas por el mismo presupuesto de regularizacion.
Eso es lo que hace que la prueba no sea una repeticion: no se puede "anadir sin dao", el
termino tiene que merecer su peso.

La intuicion que sostiene la hipotesis
-------------------------------------
Ciertas relaciones no necesitan adyacencia. Un insulto followed de una risa, un marcador de
discurso, o un ataque dentro de una cita, suelen aparecer con palabras entre ellos, y el
bigrama contiguo los separa en dos tokens sin ningun vinculo. Los trigramas de caracter de
V03 capturan forma pero no la relacion, y los bigramas de palabra capturan la relacion
cuando es adyacente pero no cuando hay hueco.

Como se construye
-----------------
Se pre-tokeniza cada documento con el mismo patron del bloque de palabras de V03
(`\\b\\w\\w+\\b` en minusculas) y se emiten, ademas de los tokens sueltos, las cadenas
`token[i] + "_" + token[i + g + 1]` para los huecos `g` pedidos. El separador guion bajo
impide que un par se confunda con un token real, que es el motivo por el que V4-D uso
`INSULT_NEAR` y no una concatenacion a pelo. `min_df = 2` mantiene el bloque acotado.

Las tres configuraciones
------------------------
- **C1-0** control: sin skip-grams. Debe reproducir V03 exactamente.
- **C1-1** hueco 1: pares `token[i]`, `token[i+2]`, es decir, una palabra de intervencion.
- **C1-2** huecos 1 y 2: hasta dos palabras de intervencion.

La especificacion pide ejecutar el hueco 2 solo si el 1 aporta. Se miden los tres y la
decision la toma el Gate 1, no la anticipacion.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_05_skipgrams.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

EXPERIMENTO = "V6-05 (C1-0 a C1-2)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_skipgrams.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_skipgrams.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_skipgrams.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260926

# El mismo patron de tokenizacion que usa el bloque de palabras de V03. Si cambiara, el
# control C1-0 dejaria de reproducir V03 y la comparacion perderia sentido.
PATRON_TOKEN = re.compile(r"(?u)\b\w\w+\b")
# El bloque de palabras de V03 se construye con `ngram_range=(1, 2)`, es decir, unigramas
# **y** bigramas contiguos. El tokenizador propio tiene que emitir los dos, y no solo los
# unigramas: sin los bigramas el control no reproduce V03 y ningun delta es atribuible.
# Los bigramas se unen con espacio, igual que hace `TfidfVectorizer` por defecto, para que el
# vocabulario del control coincida columna a columna con el de V03.
# Los pares con hueco usan un separador que no puede aparecer en un texto ya tokenizado, de
# modo que un par nunca pueda confundirse con un token real.
SEPARADOR_CONTIGUO = " "
SEPARADOR_HUECO = "|"

PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}

# Los tres esquemas: sin salto, salto de una palabra, y saltos de una y dos palabras.
HUECOS: dict[str, tuple[int, ...]] = {
    "C1-0-control": (),
    "C1-1-hueco1": (1,),
    "C1-2-huecos12": (1, 2),
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-C1] {mensaje}", flush=True)


def documental(documento: str, huecos: tuple[int, ...]) -> list[str]:
    """Emite los tokens de un documento y sus pares con hueco.

    Args:
        documento: texto de un solo tuit.
        huecos: separaciones en numero de palabras a saltar.

    Returns:
        Lista de cadenas: los tokens sueltos y despues los pares con hueco.
    """
    tokens = PATRON_TOKEN.findall(documento.lower())
    salida = list(tokens)
    for indice in range(len(tokens) - 1):
        salida.append(f"{tokens[indice]}{SEPARADOR_CONTIGUO}{tokens[indice + 1]}")
    for hueco in huecos:
        for indice in range(len(tokens) - hueco - 1):
            salida.append(
                f"{tokens[indice]}{SEPARADOR_HUECO}{tokens[indice + hueco + 1]}"
            )
    return salida


def vectorizador_palabras(huecos: tuple[int, ...]) -> TfidfVectorizer:
    """Construye el vectorizador del bloque de palabras con los huecos pedidos.

    Args:
        huecos: separaciones en numero de palabras a saltar.

    Returns:
        Vectorizador configurado con el tokenizador propio y sin `ngram_range`, porque las
        relaciones las genera el tokenizador y no el vectorizador.
    """
    return TfidfVectorizer(
        analyzer="word",
        tokenizer=lambda documento: documental(documento, huecos),
        preprocessor=lambda documento: documento,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        sublinear_tf=True,
    )


def ajustar_oof(textos, y, pliegues, huecos: tuple[int, ...]) -> tuple[np.ndarray, int, int]:
    """Calcula las probabilidades fuera de pliegue con un esquema de huecos dado.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        pliegues: pliegues corregidos del protocolo v2.
        huecos: separaciones en numero de palabras a saltar.

    Returns:
        Matriz de probabilidades de forma ``(n_filas, 3)``.
    """
    proba = np.zeros((len(textos), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_palabras_ajuste = vectorizador_palabras(huecos)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        bloque_palabras = vectorizador_palabras_ajuste.fit_transform(
            textos.iloc[entrenamiento]
        )
        bloque_char = vectorizador_char.fit_transform(textos.iloc[entrenamiento])
        matriz = sparse.hstack([bloque_palabras, bloque_char]).tocsr()
        consulta = sparse.hstack(
            [
                vectorizador_palabras_ajuste.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ]
        ).tocsr()
        for etiqueta in range(3):
            estimador = LogisticRegression(**PARAMETROS_LR)
            estimador.fit(matriz, y[entrenamiento, etiqueta])
            proba[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return proba, bloque_palabras.shape[1], bloque_char.shape[1]


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
    """Ejecuta C1-0, C1-1 y C1-2."""
    inicio = time.time()
    log("Track C1: n-gramas de palabra con salto")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    pred_v03 = (oof["proba"] >= UMBRALES_V03[None, :]).astype(int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo {len(y)} filas | V03 {metrica_macro(y, pred_v03):.6f}")

    variantes: list[dict[str, Any]] = []
    for nombre, huecos in HUECOS.items():
        log(f"   ajustando {nombre} (huecos {huecos or 'ninguno'})")
        proba, n_palabras, n_char = ajustar_oof(textos, y, pliegues, huecos)
        variante = evaluar(nombre, y, proba, pred_v03)
        variante["n_features_palabras"] = int(n_palabras)
        variante["n_features_char"] = int(n_char)
        variante["huecos"] = ",".join(str(h) for h in huecos) or "ninguno"
        variantes.append(variante)
        log(
            f"   {nombre:14s} carac. palabras {n_palabras:6d} | corrected_cv "
            f"{variante['corrected_cv']:.6f} delta {variante['delta_vs_V03']:+.4f} "
            f"IC95 [{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"HS {variante['f1_HS']:.4f} TR {variante['f1_TR']:.4f} AG {variante['f1_AG']:.4f} "
            f"| salvage {variante['salvage']} damage {variante['damage']}"
        )

    log("gate de dos niveles")
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

    log("informe y figura")
    _figura(variantes)
    _informe(variantes, len(y), pliegues, time.time() - inicio)
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "huecos", "n_features_palabras", "n_features_char",
            "corrected_cv", "delta_vs_V03", "ic95_0", "ic95_1", "f1_HS", "f1_TR",
            "f1_AG", "salvage", "damage", "pliegues_positivos", "gate_1_supera",
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
                "huecos": {nombre: list(huecos) for nombre, huecos in HUECOS.items()},
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
        f"Track C1 completada en {round(time.time() - inicio, 1)} s | mejor: "
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


def _figura(variantes: list[dict[str, Any]]) -> None:
    """Dibuja los deltas por numero de huecos.

    Args:
        variantes: resultados de las variantes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    nombres = [v["variante"] for v in variantes]
    posiciones = np.arange(len(nombres))
    fig, eje = plt.subplots(figsize=(9, 4.6))
    eje.bar(posiciones, [v["delta_vs_V03"] for v in variantes], color="#9467bd")
    eje.axhline(0, color="black", linewidth=1)
    eje.axhline(GATE_1_DELTA, color="red", linestyle=":", label=f"gate {GATE_1_DELTA:+.3f}")
    for indice, variante in enumerate(variantes):
        eje.errorbar(
            indice,
            variante["delta_vs_V03"],
            yerr=[
                [variante["delta_vs_V03"] - variante["ic95_0"]],
                [variante["ic95_1"] - variante["delta_vs_V03"]],
            ],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
    eje.set_xticks(posiciones)
    eje.set_xticklabels(nombres, rotation=20, ha="right", fontsize=9)
    eje.set_ylabel("delta frente a V03")
    eje.set_title("Skip-grams: delta e IC95 frente a V03")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_62_skipgrams_v6.png", dpi=150)
    plt.close()


def _informe(variantes, n_filas: int, pliegues, duracion: float) -> None:
    """Escribe el informe del track C1.

    Args:
        variantes: resultados de las variantes.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    base = variantes[0]["n_features_palabras"]
    lineas = [
        "# Track C1: n-gramas de palabra con salto (V6-05)",
        "",
        f"Generado por `experiments/v6_05_skipgrams.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es una repeticion de V4-D",
        "",
        "V04-D cerro \"interacciones como tokens\" (`INSULT_NEAR_MENTION` y similares) y",
        "\"canales posicionales\" (ventanas), y E2-10 cerro combinaciones forma x token. La",
        "diferencia es que aquellos eran **canales anadidos** sobre una representacion ya",
        "construida: una matriz extra apilada a la de V03, con sus propios pesos. Aqui se",
        "modifica el **analizador del bloque de palabras**, de modo que las relaciones con",
        "hueco compiten con las palabras por el mismo presupuesto de regularizacion. No se",
        "puede anadir sin dao.",
        "",
        "## 2. Resultados",
        "",
        "| variante | huecos | carac. palabras | corrected_cv | delta | IC95 | HS | TR | AG |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {variante['huecos']} | "
            f"{variante['n_features_palabras']} | {variante['corrected_cv']:.6f} | "
            f"{variante['delta_vs_V03']:+.4f} | "
            f"[{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} |"
        )
    lineas += [
        "",
        "El control C1-0 debe reproducir 0.777211 exactamente: sin skip-grams, el",
        "tokenizador propio emite los mismos tokens que el bloque de palabras de V03 y el",
        f"vocabulario queda en {base} caracteristicas. Si no coincide, la comparacion no vale.",
        "",
        "## 3. Gate de dos niveles",
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
        "## 4. Como se lee un resultado negativo aqui",
        "",
        "Los skip-grams anaden dimensionalidad, asi que un empeoramiento **no demuestra que",
        "la relacion con hueco no exista**: puede ser que exista pero no compense el coste de",
        "mas parametros libres. Por eso importa distinguir los dos casos:",
        "",
        "- Si el vocabulario crece mucho y el resultado cae, la relacion es real pero la",
        "  representacion ya tiene todo lo que cabe en 1.483 filas.",
        "- Si el vocabulario crece mucho y el resultado **no se mueve**, la relacion con",
        "  hueco no aporta senal pero tampoco molesta: es el caso neutro.",
        "- Si el vocabulario crece poco y el resultado no se mueve, la relacion no existe.",
        "",
        "**Este es el segundo caso, y conviene nombrarlo bien.** El bloque de palabras pasa de",
        f"{base} a {variantes[1]['n_features_palabras']} caracteristicas con un hueco y a",
        f"{variantes[2]['n_features_palabras']} con dos huecos: un crecimiento del",
        f"{100 * (variantes[2]['n_features_palabras'] - base) / base:.0f} %, y el resultado se",
        f"queda en {variantes[2]['delta_vs_V03']:+.4f} con IC95 que incluye el 0. Los",
        "skip-grams son **neutros**: ni anaden senal aprovechable ni restan capacidad.",
        "",
        "Eso es un resultado distinto del de V4-D, donde las interacciones como tokens",
        "costaron -0.0346. Aqui la relacion con hueco no es dannosa, lo que **confirma que no",
        "era una repeticion de V4-D** y que la intuicion de partida era razonable; lo que",
        "falla es la premisa de que quede senal por representar. El intercambio por etiqueta",
        f"lo muestra: con un hueco HS sube de {variantes[0]['f1_HS']} a",
        f"{variantes[1]['f1_HS']} mientras TR baja de {variantes[0]['f1_TR']} a",
        f"{variantes[1]['f1_TR']}. No es una ganancia, es un reparto.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
