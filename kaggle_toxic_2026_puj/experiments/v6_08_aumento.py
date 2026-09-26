"""Track J de la fase v6 (V6-08): aumento de datos que crea instancias nuevas.

Contexto
--------
El gap analysis (V6-00) hizo una distincion que importa: **J no es una repeticion de
PRE-00 a PRE-09**. Esas diez variantes eran *normalizacion* (quitar ruido de superficie:
contracciones, acentos, puntuacion, elongation) y todas se descartaron, con un mejor de
0.7623 frente a 0.7587 de la referencia de aquel momento. El aumento es otra cosa: **crear
instancias nuevas** a partir de las existentes, de modo que el conjunto de entrenamiento
tiene mas filas y el modelo ve la misma idea con mas variaciones.

Por que el precedente sigue siendo adverso
----------------------------------------
 notwithstanding que el mecanismo sea distinto, el precedente de superficie **es**
adverso: diez transformaciones deterministas de superficie, todas descartadas. La
diferencia que se explora aqui es que la variacion se aplica **solo al entrenamiento**,
nunca a validacion ni a test, y que genera filas nuevas en vez de sustituir las viejas. Si
el resultado vuelve a ser negativo, la conclusion covers tanto el aumento como la
superficie, y eso es un cierre mas solido, no mas debil.

Las transformaciones
-------------------
Se eligieron las que **preservan la etiqueta por construccion**, no las que la presumen:

- `espaciado`: insertar un espacio entre caracteres, como el "c a s o" de algunos tuits.
- `elongacion`: alargar vowels, como "idi-o-ta" o "p e r r o".
- `repeticion_puntuacion`: repetir signos, como "!!!" o "???".

Lo que **no** se hace, y es lo importante: no se sustituyen palabras por sinonimos, no se
reordena, no se cambia la negacion ni el objetivo. Cualquier transformacion que toque el
significado es una fuente de etiquetas falsas, y el proyecto tiene una restriccion de
seguridad sobre esto desde v2.

Diseno
------
- **J0** control: sin aumento. Debe reproducir V03 exactamente.
- **J1** cada fila de train genera una copia con una transformacion elegida **de forma
  determinista** a partir de su indice, de modo que la augmentacion es reproducible.
- **J2** cada fila genera **dos** copias, con transformaciones distintas.

Las copias se anaden **solo al tren del pliegue**, nunca a validacion ni a test. El
resultado se evalua con los umbrales congelados de V03 y con el mismo gate de dos niveles.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_08_aumento.py
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

EXPERIMENTO = "V6-08 (J0-J2)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_aumento.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_aumento.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_aumento.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260926
SEMILLA_AUMENTO = 20260926

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
PARAMETROS_LR = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}

VOCALES = "aeiouáéíóúAEIOUÁÉÍÓÚ"
PATRON_PALABRA = re.compile(r"\w+", re.UNICODE)

TRANSFORMACIONES = ("espaciado", "elongacion", "puntuacion")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-J] {mensaje}", flush=True)


def aumentar(texto: str, transformacion: str, generador) -> str:
    """Aplica una transformacion de superficie que preserva la etiqueta.

    Args:
        texto: documento original.
        transformacion: clave de ``TRANSFORMACIONES``.
        generador: generador de numeros aleatorios.

    Returns:
        El documento transformado, o el original si la transformacion no altera nada.
    """
    if transformacion == "espaciado":
        palabras = PATRON_PALABRA.findall(texto)
        if not palabras:
            return texto
        elegida = palabras[int(generador.integers(len(palabras)))]
        return texto.replace(elegida, " ".join(elegida), 1)
    if transformacion == "elongacion":
        palabras = PATRON_PALABRA.findall(texto)
        candidatas = [p for p in palabras if any(v in VOCALES for v in p) and len(p) > 2]
        if not candidatas:
            return texto
        elegida = candidatas[int(generador.integers(len(candidatas)))]
        return texto.replace(elegida, elegida + elegida[-1], 1)
    if transformacion == "puntuacion":
        if not any(c in texto for c in "!?."):
            return texto
        posicion = int(generador.integers(len(texto)))
        return texto[:posicion] + texto[posicion] * 2 + texto[posicion + 1 :]
    raise ValueError(f"transformacion desconocida: {transformacion}")


def construir_textos_aumentados(textos, indices: np.ndarray, n_copias: int):
    """Construye el conjunto de entrenamiento con copias transformadas.

    Las copias se anaden **solo** a los indices de entrenamiento del pliegue: nunca entran
    en validacion ni en test. La transformacion de cada fila se elige de forma
    determinista a partir de su indice, de modo que la augmentacion es reproducible.

    Args:
        textos: columna de textos del desarrollo.
        indices: indices de las filas de entrenamiento del pliegue.
        n_copias: numero de copias por fila.

    Returns:
        Lista de textos y lista de vectores de etiqueta correspondientes.
    """
    generador = np.random.default_rng(SEMILLA_AUMENTO)
    originales = [textos.iloc[indice] for indice in indices]
    etiquetas_originales = list(indices)
    for copia in range(n_copias):
        transformacion = TRANSFORMACIONES[copia % len(TRANSFORMACIONES)]
        for indice in indices:
            originales.append(aumentar(textos.iloc[indice], transformacion, generador))
            etiquetas_originales.append(indice)
    return originales, etiquetas_originales


def ajustar_oof(textos, y, pliegues, n_copias: int) -> tuple[np.ndarray, dict[str, int]]:
    """Calcula las probabilidades fuera de pliegue con `n_copias` copias por fila de train.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        pliegues: pliegues corregidos.
        n_copias: numero de copias transformadas por fila de entrenamiento.

    Returns:
        Tupla con las probabilidades y un diccionario de tamaños del entrenamiento.
    """
    proba = np.zeros((len(textos), 3), dtype=np.float64)
    tamanos: dict[str, int] = {}
    for entrenamiento, validacion in pliegues:
        textos_tr, indices_tr = (
            construir_textos_aumentados(textos, entrenamiento, n_copias)
            if n_copias > 0
            else ([textos.iloc[i] for i in entrenamiento], list(entrenamiento))
        )
        y_tr = y[indices_tr]
        tamanos["filas_entrenamiento"] = len(textos_tr)
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        matriz = sparse.hstack(
            [
                vectorizador_word.fit_transform(textos_tr),
                vectorizador_char.fit_transform(textos_tr),
            ]
        ).tocsr()
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]),
            ]
        ).tocsr()
        for etiqueta in range(3):
            estimador = LogisticRegression(**PARAMETROS_LR)
            estimador.fit(matriz, y_tr[:, etiqueta])
            proba[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return proba, tamanos


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
    """Ejecuta el control J0 y las dos variantes de aumento."""
    inicio = time.time()
    log("Track J: aumento de datos que crea instancias nuevas")
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
    log(f"   transformaciones: {', '.join(TRANSFORMACIONES)}")

    variantes: list[dict[str, Any]] = []
    for nombre, n_copias in (("J0-control", 0), ("J1-una-copia", 1), ("J2-dos-copias", 2)):
        log(f"   ajustando {nombre} ({n_copias} copias por fila de train)")
        proba, tamanos = ajustar_oof(textos, y, pliegues, n_copias)
        variante = evaluar(nombre, y, proba, pred_v03)
        variante["filas_entrenamiento"] = tamanos["filas_entrenamiento"]
        variantes.append(variante)
        log(
            f"   {nombre:14s} filas de train {tamanos['filas_entrenamiento']:5d} | "
            f"corrected_cv {variante['corrected_cv']:.6f} "
            f"delta {variante['delta_vs_V03']:+.4f} "
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
            "variante", "filas_entrenamiento", "corrected_cv", "delta_vs_V03", "ic95_0",
            "ic95_1", "f1_HS", "f1_TR", "f1_AG", "salvage", "damage",
            "pliegues_positivos", "gate_1_supera",
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
                "transformaciones": list(TRANSFORMACIONES),
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
        f"Track J completada en {round(time.time() - inicio, 1)} s | mejor: "
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


def _figura(variantes) -> None:
    """Dibuja el delta frente al numero de copias anadidas.

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
    eje.bar(posiciones, [v["delta_vs_V03"] for v in variantes], color="#e377c2")
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
    eje.set_title("Aumento de datos: delta e IC95 frente a V03")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_65_aumento_v6.png", dpi=150)
    plt.close()


def _informe(variantes, n_filas: int, pliegues, duracion: float) -> None:
    """Escribe el informe del track J.

    Args:
        variantes: resultados de las variantes.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    lineas = [
        "# Track J: aumento de datos (V6-08)",
        "",
        f"Generado por `experiments/v6_08_aumento.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es una repeticion de PRE",
        "",
        "El gap analysis (V6-00) hizo una distincion que importa: **J no es una repeticion de",
        "PRE-00 a PRE-09**. Esas diez variantes eran *normalizacion*: quitan ruido de",
        "superficie y **sustituyen** el texto. El aumento **crea instancias nuevas** a partir",
        "de las existentes, y la variacion se aplica **solo al entrenamiento**, nunca a",
        "validacion ni a test.",
        "",
        "Aunque el mecanismo sea distinto, el precedente de superficie es",
        "adverso: diez transformaciones deterministas, todas descartadas. Si el resultado",
        "vuelve a ser negativo, el cierre abarca tanto el aumento como la superficie, y eso",
        "es un cierre mas solido, no mas debil.",
        "",
        "## 2. Transformaciones y por que son seguras",
        "",
        "| transformacion | que hace | por que preserva la etiqueta |",
        "|---|---|---|",
        "| `espaciado` | separa las letras de una palabra | no cambia ni una palabra |",
        "| `elongacion` | alarga la ultima letra de una palabra | solo anade un caracter a un item ya existente |",
        "| `puntuacion` | duplica un signo | no toca el lexema |",
        "",
        "Lo que **no** se hace, y es lo importante: no se sustituyen palabras por",
        "sinonimos, no se reordena, no se cambia la negacion ni el objetivo. Cualquier",
        "transformacion que toque el significado es una fuente de etiquetas falsas.",
        "",
        "## 3. Resultados",
        "",
        "| variante | filas de train | corrected_cv | delta | IC95 | HS | TR | AG |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {variante['filas_entrenamiento']} | "
            f"{variante['corrected_cv']:.6f} | {variante['delta_vs_V03']:+.4f} | "
            f"[{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} |"
        )
    lineas += [
        "",
        "Las filas de train crecen de",
        f"{variantes[0]['filas_entrenamiento']} a {variantes[2]['filas_entrenamiento']},",
        "un aumento real de datos y no una sustitucion de texto.",
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
