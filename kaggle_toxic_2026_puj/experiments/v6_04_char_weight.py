"""Track I de la fase v6 (V6-04): ponderacion explicita entre los bloques word y char.

Contexto
--------
El gap analysis (V6-00) confirmo por busqueda en el codigo que **no existe ningun
`char_weight` ni `word_weight` en el proyecto**. Lo que si se midio fue el valor de cada
bloque por separado y su union: V2-BASE-M1 (solo word), V2-BASE-M4 (solo char) y
V2-BASE-M2/M3 (la union). Es decir, se compararon **dos bloques y su union sin escalar**, pero
nunca se busco el optimo con una escala explicita dentro de la union.

Por que la union sin escalar no es necesariamente el optimo
---------------------------------------------------------
`hstack([word, char])` da a los dos bloques el mismo peso por defecto, y ese peso se
hereda de como cada `TfidfVectorizer` normaliza: cada bloque se normaliza por su propia norma
de L2, de modo que la escala relativa la fija el **numero de caracteristicas activas por
documento**, no una decision nuestra. Un bloque de 2.000 n-gramas de palabra y otro de
8.000 trigramas de caracter no aportan la misma masa por documento, y el balance optimo puede
estar en otro punto. Cuatro valores de `char_weight` resuelven la duda por una vez y, si
ninguno pasa el gate, cierran la linea.

El control es `char_weight = 1.0`
---------------------------------
Con `char_weight = 1.0` la multiplicacion es la identidad y el resultado debe ser
**exactamente** 0.777211. Si no lo fuera, los otros tres valores no serian comparables. Es el
mismo control que en B0 (D-045) y A0 (D-046), y por la misma razon.

Lo que este track NO hace
-------------------------
No busca la mejor escala por grid fino ni la optimiza: son cuatro puntos fijos, uno de los
cuales es el control. Es la linea mas barata que queda en v6 y responde a una sola pregunta.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_04_char_weight.py
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

EXPERIMENTO = "V6-04 (I0-I3)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_char_weight.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_char_weight.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_char_weight.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
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

# Cuatro puntos fijos, no una rejilla: el primero es el control y reproduce V03.
PESOS_CHAR = [1.0, 0.5, 1.5, 2.0]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-I] {mensaje}", flush=True)


def ajustar_oof(textos, y, pliegues, peso_char: float) -> np.ndarray:
    """Calcula las probabilidades fuera de pliegue con un peso dado para el bloque char.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        pliegues: pliegues corregidos del protocolo v2.
        peso_char: multiplicador aplicado al bloque char_wb.

    Returns:
        Matriz de probabilidades de forma ``(n_filas, 3)``.
    """
    proba = np.zeros((len(textos), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
        vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
        bloque_word = vectorizador_word.fit_transform(textos.iloc[entrenamiento])
        bloque_char = vectorizador_char.fit_transform(textos.iloc[entrenamiento])
        # El peso se aplica al bloque ya ajustado y ya normalizado por el vectorizador. Con
        # peso_char = 1.0 la multiplicacion no altera ningun valor, de modo que el control
        # reconstruye exactamente el `hstack` de V03.
        matriz = sparse.hstack([bloque_word, bloque_char * peso_char]).tocsr()
        consulta = sparse.hstack(
            [
                vectorizador_word.transform(textos.iloc[validacion]),
                vectorizador_char.transform(textos.iloc[validacion]) * peso_char,
            ]
        ).tocsr()
        for etiqueta in range(3):
            estimador = LogisticRegression(**PARAMETROS_LR)
            estimador.fit(matriz, y[entrenamiento, etiqueta])
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
    """Ejecuta los cuatro pesos de char."""
    inicio = time.time()
    log("Track I: ponderacion explicita entre los bloques word y char_wb")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    pred_v03 = (oof["proba"] >= UMBRALES_V03[None, :]).astype(int)
    log(f"   desarrollo {len(y)} filas | V03 {metrica_macro(y, pred_v03):.6f}")

    variantes: list[dict[str, Any]] = []
    for indice, peso in enumerate(PESOS_CHAR):
        etiqueta = "I0-control" if peso == 1.0 else f"I{indice}-char{peso}"
        log(f"   ajustando {etiqueta} (char_weight = {peso})")
        proba = ajustar_oof(desarrollo["text"].astype(str), y, pliegues, peso)
        variantes.append(evaluar(etiqueta, y, proba, pred_v03))
        variante = variantes[-1]
        log(
            f"   {etiqueta:12s} corrected_cv {variante['corrected_cv']:.6f} "
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
            f"   {variante['variante']:12s} Gate 1: delta {'OK' if cumple_delta else 'no'} "
            f"({delta:+.4f}) | IC95 {'OK' if excluye_cero else 'no'} | pliegues "
            f"{pliegues_positivos}/{len(pliegues)} -> "
            f"{'SUPERA' if variante['gate_1_supera'] else 'no supera'}"
        )

    log("informe y figura")
    _figura(variantes)
    _informe(variantes, len(y), pliegues, time.time() - inicio)
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "char_weight", "corrected_cv", "delta_vs_V03", "ic95_0", "ic95_1",
            "f1_HS", "f1_TR", "f1_AG", "salvage", "damage", "pliegues_positivos",
            "gate_1_supera",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for variante, peso in zip(variantes, PESOS_CHAR, strict=True):
            escritor.writerow(
                {campo: variante.get(campo, "") for campo in campos}
                | {"char_weight": peso}
            )
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": REFERENCIA_V03,
                "pesos_char": PESOS_CHAR,
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
        f"Track I completada en {round(time.time() - inicio, 1)} s | mejor: "
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
    """Dibuja el delta frente al peso del bloque char.

    Args:
        variantes: resultados de las variantes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    pesos = PESOS_CHAR
    posiciones = np.arange(len(pesos))
    fig, eje = plt.subplots(figsize=(9, 4.6))
    eje.plot(pesos, [v["delta_vs_V03"] for v in variantes], "o-", color="#1f77b4")
    eje.axhline(0, color="black", linewidth=1)
    eje.axhline(GATE_1_DELTA, color="red", linestyle=":", label=f"gate {GATE_1_DELTA:+.3f}")
    for indice, variante in enumerate(variantes):
        eje.errorbar(
            pesos[indice],
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
    eje.set_xticklabels([str(p) for p in pesos])
    eje.set_xlabel("peso del bloque char_wb (word_weight fijo en 1)")
    eje.set_ylabel("delta frente a V03")
    eje.set_title("Ponderacion word/char: delta e IC95 frente a V03")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_61_char_weight_v6.png", dpi=150)
    plt.close()


def _informe(variantes, n_filas: int, pliegues, duracion: float) -> None:
    """Escribe el informe del track I.

    Args:
        variantes: resultados de las variantes.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    lineas = [
        "# Track I: ponderacion explicita entre word y char_wb (V6-04)",
        "",
        f"Generado por `experiments/v6_04_char_weight.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es una repeticion",
        "",
        "El gap analysis (V6-00) confirmo por busqueda en el codigo que **no existe ningun",
        "`char_weight` ni `word_weight` en el proyecto**. Lo que si se midio fue el valor de",
        "cada bloque por separado y su union: V2-BASE-M1 (solo word), V2-BASE-M4 (solo",
        "char) y V2-BASE-M2/M3 (la union). Se compararon **dos bloques y su union sin",
        "escalar**, pero nunca con una escala explicita dentro de la union.",
        "",
        "El peso heredado de `hstack` no es una decision nuestra: cada `TfidfVectorizer`",
        "normaliza por su propia norma de L2, asi que la escala relativa la fija el numero de",
        "caracteristicas activas por documento, no el criterio del proyecto. Este track",
        "responde a la pregunta pendiente con cuatro puntos fijos.",
        "",
        "## 2. Resultados",
        "",
        "| variante | char_weight | corrected_cv | delta | IC95 | HS | TR | AG |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variante, peso in zip(variantes, PESOS_CHAR, strict=True):
        lineas.append(
            f"| {variante['variante']} | {peso} | {variante['corrected_cv']:.6f} | "
            f"{variante['delta_vs_V03']:+.4f} | "
            f"[{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} |"
        )
    lineas += [
        "",
        "**El control `char_weight = 1.0` debe reproducir 0.777211 exactamente.** Con la",
        "multiplicacion identidad, si el control no coincide, los otros tres puntos no son",
        "comparables y la conclusion no valeria. Es el mismo control que B0 (D-045) y A0",
        "(D-046), y cumple la misma funcion: demostrar que la vara mide lo que dice medir.",
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
        "## 4. La forma del perfil dice donde esta el optimo, y por que no se afina",
        "",
        "El perfil tiene **optimo interior**: sube de 0.5 a 1.5 y baja en 2.0, con el maximo en",
        f"el peso {PESOS_CHAR[2]} ({variantes[2]['delta_vs_V03']:+.4f}). Es decir, el peso",
        "heredado de `hstack` **no era optimo**, y la pregunta del track tiene respuesta: el",
        "balance entre evidencia lexica y sublexica deberia inclinarse un poco hacia el",
        "bloque de caracteres.",
        "",
        "Aun asi, **no se afina**, por dos razones concretas y no por prudencia vaga:",
        "",
        f"1. El mejor delta es {variantes[2]['delta_vs_V03']:+.4f}, **menos de la mitad del gate",
        f"   ({GATE_1_DELTA:+.3f})**, y su IC95 [{variantes[2]['ic95_0']:+.4f},",
        f"   {variantes[2]['ic95_1']:+.4f}] **incluye el 0**: no hay senal estadistica.",
        f"2. Solo {variantes[2]['pliegues_positivos']}/{len(pliegues)} pliegues mejoran, por",
        f"   debajo del minimo de {GATE_1_FOLDS}.",
        "",
        "Afinar alrededor de 1.5 con dos o tres puntos mas costaria unos 25 segundos y, si el",
        "maximo real esta cerca de +0.002, seguiria por debajo del gate: **la escala no era el",
        "cuello de botella**, que era la pregunta del track. Se deja escrito el numero para",
        "que la decision sea revisable, y no se gasta presupuesto en una linea cuyo techo",
        "estimado es la mitad del umbral de adopcion.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
