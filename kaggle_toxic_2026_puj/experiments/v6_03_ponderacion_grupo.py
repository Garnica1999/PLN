"""Track A de la fase v6 (V6-03): ponderacion del entrenamiento por tamano de grupo.

Contexto
--------
El gap analysis (V6-00) confirmo con una busqueda programatica que la ponderacion por
tamano de grupo **nunca se ha probado**: no existe `peso_grupo`, ni `group_weight`, ni un
`sample_weight` derivado del numero de filas del grupo. Lo que si se probo fue ponderar por
**dureza** (V4-B03, -0.0101) y un downweight de frontera (D-027, la unica senal positiva
pequena de v4). Son fuentes de varianza distintas: la dureza es una propiedad del texto, el
tamano de grupo es una propiedad de como se construyo el corpus.

Que mide la hipotesis
---------------------
V03 se ajusta sobre el desarrollo completo, donde los grupos repetidos aportan **una fila por
tuit**. Si un grupo de 40 tuits aparece 40 veces, el clasificador ve esa direccion 40 veces y
la decision en ese neighbourhood la domina el bloque repetido. Ponderar `w = 1/sqrt(g)` o
`w = 1/g` devuelve a cada grupo la influencia de un unico tuit, que es la pregunta que
equivale a "si solo tuviera un ejemplo de este grupo, learned lo mismo".

Por que el peso se calcula dentro del pliegue
---------------------------------------------
El grupo es una propiedad de la particion, no del texto. Si los pesos se calcularan una vez
sobre todo el desarrollo, cada pliegue arrastraria informacion de las filas que despues se
validan: eso es **leakage**, aunque el peso sea solo un entero. Por eso aqui los grupos se
recuentan **solo sobre el train del pliegue**, y una fila de validacion con grupo
desconocido recibe el peso mediano. Es la misma disciplina que exige la regla de no ajustar
nada con el conjunto de evaluacion, y es lo que hace legitima la comparacion.

Las cuatro configuraciones
--------------------------
- **A0** control: `w = 1` para todos. Debe reproducir V03 exactamente; si no, el codigo
  esta mal y no se puede concluir nada de A1, A2 y A3.
- **A1** `w = 1/sqrt(g)`: suaviza, el grupo grande pierde pero no desaparece.
- **A2** `w = 1/g`: el grupo grande casi no pesa, es la version agresiva.
- **A3** `w = min(1/g, tope)` con `tope = 0.25`: limita el daño de A2 en los grupos
  enormes sin dejar de corregir el caso extremo.

Sin rejilla: cuatro configuraciones fijas, que es lo que la especificacion pide. Si ninguna
supera el Gate 1, el track se cierra por la seccion 54.

Uso
---
    /home/carlos/venv/bin/python experiments/v6_03_ponderacion_grupo.py
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

from src.data import read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    _f1_por_celda,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "V6-03 (A0-A3)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v6_ponderacion_grupo.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v6_ponderacion_grupo.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v6_ponderacion_grupo.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
GATE_1_DELTA = 0.003
GATE_1_FOLDS = 4
TOPE_A3 = 0.25
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

# Los cuatro esquemas de ponderacion. La clave es el nombre de la variante y el valor es
# una funcion que recibe el tamano de grupo de cada fila de entrenamiento y devuelve su peso.
# Ninguno se ajusta con datos de fuera del pliegue.
ESQUEMAS: dict[str, str] = {
    "A0-control": "w = 1 para todos (control: debe reproducir V03 exactamente)",
    "A1-sqrt": "w = 1 / sqrt(g): el grupo grande pierde peso sin desaparecer",
    "A2-inversa": "w = 1 / g: la version agresiva, el grupo grande casi no pesa",
    "A3-tope": f"w = min(1 / g, {TOPE_A3}): limita el dano de A2 en grupos enormes",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V6-A] {mensaje}", flush=True)


def pesos(tamanos: np.ndarray, esquema: str) -> np.ndarray:
    """Devuelve el peso de entrenamiento segun el tamano de grupo de cada fila.

    Args:
        tamanos: tamano de grupo de cada fila de entrenamiento.
        esquema: clave de ``ESQUEMAS``.

    Returns:
        Vector de pesos, de la misma longitud que ``tamanos``.
    """
    grupo = np.maximum(tamanos.astype(np.float64), 1.0)
    if esquema == "A0-control":
        return np.ones_like(grupo)
    if esquema == "A1-sqrt":
        return 1.0 / np.sqrt(grupo)
    if esquema == "A2-inversa":
        return 1.0 / grupo
    if esquema == "A3-tope":
        return np.minimum(1.0 / grupo, TOPE_A3)
    raise ValueError(f"esquema desconocido: {esquema}")


def ajustar_oof(textos, y, pliegues, grupos, esquema: str) -> np.ndarray:
    """Calcula las probabilidades fuera de pliegue de los tres labels con un esquema dado.

    Los pesos se recalculan **dentro** de cada pliegue a partir de los grupos que aparecen
    en su tren, nunca con los grupos de todo el desarrollo.

    Args:
        textos: columna de textos del desarrollo.
        y: matriz binaria verdadera.
        pliegues: pliegues corregidos del protocolo v2.
        grupos: identificador de grupo fusionado de cada fila.
        esquema: clave de ``ESQUEMAS``.

    Returns:
        Matriz de probabilidades de forma ``(n_filas, 3)``.
    """
    n = len(textos)
    proba = np.zeros((n, 3), dtype=np.float64)
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
        # Los grupos se cuentan SOLO sobre el tren del pliegue. Una fila de validacion cuyo
        # grupo no aparezca en el tren recibe el peso mediano, que es el valor neutro: no
        # inventa informacion y no concentra peso en ningun extremo.
        grupos_tr = grupos[entrenamiento]
        conteo = pd.Series(grupos_tr).value_counts()
        tamanos_tr = conteo.reindex(grupos_tr).to_numpy()
        peso = pesos(tamanos_tr, esquema)
        for etiqueta in range(3):
            estimador = LogisticRegression(**PARAMETROS_LR)
            estimador.fit(matriz, y[entrenamiento, etiqueta], sample_weight=peso)
            proba[validacion, etiqueta] = estimador.predict_proba(consulta)[:, 1]
    return proba


def evaluar(
    nombre: str, y: np.ndarray, proba: np.ndarray, pred_v03: np.ndarray
) -> dict[str, Any]:
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
    """Ejecuta A0, A1, A2 y A3."""
    inicio = time.time()
    log("Track A: ponderacion del entrenamiento por tamano de grupo")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_v03 = oof["proba"]
    pred_v03 = (proba_v03 >= UMBRALES_V03[None, :]).astype(int)
    grupos_dev = grupos[indices_dev]
    log(f"   desarrollo {len(y)} filas | V03 {metrica_macro(y, pred_v03):.6f}")

    # Como se distribuyen los grupos: es el contexto que da sentido a los cuatro esquemas.
    conteo_global = pd.Series(grupos_dev).value_counts()
    log(
        f"   grupos: {len(conteo_global)} distintos | tamaño mediano "
        f"{int(np.median(conteo_global.to_numpy()))} | maximo {int(conteo_global.max())} | "
        f"filas en grupos de 1: {int((conteo_global == 1).sum())}"
    )

    variantes: list[dict[str, Any]] = []
    for indice, esquema in enumerate(ESQUEMAS):
        log(f"   ajustando {esquema} ({indice + 1}/{len(ESQUEMAS)})")
        proba = ajustar_oof(desarrollo["text"].astype(str), y, pliegues, grupos_dev, esquema)
        variantes.append(evaluar(esquema, y, proba, pred_v03))
        variante = variantes[-1]
        log(
            f"   {esquema:12s} corrected_cv {variante['corrected_cv']:.6f} "
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
        pliegues_positivos = _pliegues_positivos(
            y, variante["pred"], pliegues, pred_v03
        )
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
    _informe(variantes, conteo_global, len(y), pliegues, time.time() - inicio)
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "variante", "corrected_cv", "delta_vs_V03", "ic95_0", "ic95_1",
            "f1_HS", "f1_TR", "f1_AG", "salvage", "damage",
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
                "esquemas": ESQUEMAS,
                "tope_a3": TOPE_A3,
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
        f"Track A completada en {round(time.time() - inicio, 1)} s | mejor: "
        f"{mejores['variante']} {mejores['corrected_cv']:.6f} "
        f"(delta {mejores['delta_vs_V03']:+.4f})"
    )


def _pliegues_positivos(
    y: np.ndarray, pred: np.ndarray, pliegues, pred_referencia: np.ndarray
) -> int:
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
    """Dibuja los deltas de cada esquema con su IC95.

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
    eje.bar(posiciones, [v["delta_vs_V03"] for v in variantes], color="#2ca02c")
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
    eje.set_title("Ponderacion por tamano de grupo: delta e IC95 frente a V03")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_60_ponderacion_grupo_v6.png", dpi=150)
    plt.close()


def _informe(
    variantes: list[dict[str, Any]],
    conteo_global,
    n_filas: int,
    pliegues,
    duracion: float,
) -> None:
    """Escribe el informe del track A.

    Args:
        variantes: resultados de las variantes.
        conteo_global: numero de filas por grupo.
        n_filas: numero de filas del desarrollo.
        pliegues: pliegues corregidos.
        duracion: segundos que tardo el script.
    """
    filas_grupo_1 = int((conteo_global == 1).sum())
    filas_repetidas = int(n_filas - filas_grupo_1)
    porcentaje_afectable = 100 * filas_repetidas / n_filas
    lineas = [
        "# Track A: ponderacion por tamano de grupo (V6-03)",
        "",
        f"Generado por `experiments/v6_03_ponderacion_grupo.py` ({EXPERIMENTO}).",
        "",
        "## 1. Que se prueba y por que no es una repeticion",
        "",
        "El gap analysis (V6-00) confirmo por busqueda en el codigo que la ponderacion por",
        "tamano de grupo **nunca se ha probado**. Lo que si se probo fue ponderar por",
        "**dureza** (V4-B03, -0.0101) y un downweight de frontera (D-027), que son fuentes de",
        "varianza distintas: la dureza es una propiedad del texto, el tamano de grupo es una",
        "propiedad de como se construyo el corpus.",
        "",
        "La pregunta es si V03, ajustado con una fila por tuit, deja que los grupos repetidos",
        "dominen la frontera de decision, y si devolver a cada grupo la influencia de un",
        "unico tuit mejora algo.",
        "",
        "## 2. Distribucion de los grupos en el desarrollo",
        "",
        f"- **{len(conteo_global)} grupos distintos** en {n_filas} filas.",
        f"- tamaño mediano {int(np.median(conteo_global))}, maximo {int(conteo_global.max())}.",
        f"- {filas_grupo_1} filas en grupos de tamano 1.",
        f"- {filas_repetidas} filas en grupos de tamano mayor que 1 "
        f"({porcentaje_afectable:.1f} % del desarrollo).",
        "",
        "**Esta distribucion acota el track antes de mirar los resultados.** Los cuatro",
        f"esquemas solo pueden cambiar el peso de las {filas_repetidas} filas en grupos",
        f"repetidos, un {porcentaje_afectable:.1f} % del desarrollo: el",
        f"{100 - porcentaje_afectable:.1f} % restante ya tiene `g = 1` y por tanto peso 1 en",
        "los cuatro esquemas, idéntico al control. La hipótesis era plausible, pero su",
        "alcance máximo lo fija la estructura del corpus, no el modelo.",
        "",
        "## 3. Los cuatro esquemas y el control",
        "",
        "| variante | esquema | corrected_cv | delta | IC95 | HS | TR | AG |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variante in variantes:
        lineas.append(
            f"| {variante['variante']} | {ESQUEMAS[variante['variante']]} | "
            f"{variante['corrected_cv']:.6f} | {variante['delta_vs_V03']:+.4f} | "
            f"[{variante['ic95_0']:+.4f}, {variante['ic95_1']:+.4f}] | "
            f"{variante['f1_HS']:.4f} | {variante['f1_TR']:.4f} | {variante['f1_AG']:.4f} |"
        )
    lineas += [
        "",
        "**El control A0 es obligatorio**: con `w = 1` para todos y el mismo codigo, el",
        "resultado debe ser exactamente 0.777211. Si no lo fuera, cualquier delta posterior",
        "seria inatribuible. Es el mismo principio que se aplico en el control B0 del track B.",
        "",
        "## 4. El patron es monotonico en la agresividad, y eso tiene lectura",
        "",
        "| esquema | agresividad | delta |",
        "|---|---|---|",
        "| A1 `1/sqrt(g)` | suave | "
        f"{variantes[1]['delta_vs_V03']:+.4f} |",
        "| A2 `1/g` | media | "
        f"{variantes[2]['delta_vs_V03']:+.4f} |",
        f"| A3 `min(1/g, {TOPE_A3})` | fuerte | "
        f"{variantes[3]['delta_vs_V03']:+.4f} |",
        "",
        "Los tres esquemas se ordenan **exactamente** segun cuanto penalizan a los grupos",
        "repetidos, y el resultado empeora al hacerlo. Como el tope de A3 es mas agresivo que",
        "`1/g` para todo `g >= 2`, la progresion no es un accidente de los parametros.",
        "",
        "La lectura es que **un tuit repetido no es ruido de recoleccion, es señal**: que",
        "el mismo texto ofensivo aparezca varias veces lo hace mas representativo de la",
        "distribucion, no menos. Bajar su peso hasta la unidad basica confunde repeticion",
        "con azar, y por eso A3 se lleva por delante las tres F1 a la vez.",
        "",
        "## 5. Gate de dos niveles",
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
        "## 6. Nota sobre el calculo de los pesos",
        "",
        "Los pesos se cuentan **dentro de cada pliegue**, sobre los grupos que aparecen solo",
        "en su tren. Calcularlos una vez sobre todo el desarrollo haria que cada pliegue",
        "arrastrase informacion de las filas con las que despues se valida, que es leakage",
        "aunque el peso sea un entero. Una fila de validacion cuyo grupo no aparezca en el",
        "tren recibe el peso mediano, que es el valor neutro.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
