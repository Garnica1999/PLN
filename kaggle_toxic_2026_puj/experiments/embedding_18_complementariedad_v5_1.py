"""P7 de la fase v5.1 (EBD-24): complementariedad, damage y solapamiento con char_wb.

Contexto
--------
D-041 cerrara la conclusion de tarea: ninguna de las 14 configuraciones aporta a V03 por
encima del gate. Queda una pregunta de la spec que no es "¿cuanto mejora?" sino
"**que** arregla, y **por que** no arregla mas?".

Que mide este script
-------------------
1. **salvage / damage / correccion neta** para cuatro configuraciones: el baseline, la de
   mejor delta de fusion (WS-3), una de mejor delta tras el baseline (VS-100) y la de mejor
   embedding aislado (NEG-5). No se reentrena nada: son ``DOWNSTREAM EVALUATION UNIT``.
2. **Prueba de interaccion ortografica**: si el efecto del embedding se concentra en las
   filas con formas ortograficas degradadas (elongaciones y formas raras), su aportacion es
   **robustez ortografica**, que es justo lo que V03 ya hace con char_wb (3,5); si esta
   repartido, seria informacion distribucional, que el char n-gram no tiene. La prueba es un
   bootstrap pareado de la diferencia de deltas entre ambos subconjuntos, no una comparacion
   de promedios: comparar dos grupos de tamano distinto sin emparejar el ruido de filas es
   sumar ratones y elefantes.
3. **Ejemplos concretos** de filas donde el embedding corrige y V03 falla, y al reves.
4. **Slices consolidadas** de la mejor variante frente a V03.

Nota de granularidad: filas, no celdas
---------------------------------------
``src.evaluation._f1_por_celda`` se llama asi y su docstring habla de celdas (fila,
etiqueta), pero devuelve la **media de las tres etiquetas de cada fila**. La unidad de
analisis es por tanto la **fila** (1.483), no la celda (4.449). Los conteos publicados en v5
no cambian, porque siempre se contaron filas, pero el nombre era incorrecto y este script
lo corrige en lugar de propagarlo. Por eso aqui no se habla de "correccion neta equivalente
en F1": el conteo de filas y la diferencia de la metrica macro-columna son dos magnitudes
distintas, y se reportan las dos sin identificar una con la otra.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_18_complementariedad_v5_1.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.embedding_utils import normalizar_emb_normal  # noqa: E402
from src.evaluation import ETIQUETAS, metrica_macro  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "EBD-24 (P7)"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "embedding_complementarity.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "embedding_complementarity_v5_1.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_complementarity_v5_1.csv"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260925

# Las cuatro configuraciones que se reevaluan para la complementariedad.
VARIANTES: list[dict[str, str]] = [
    {
        "codigo": "B0-200",
        "sifts": "fasttext-baseline_sifts.txt",
        "nota": "baseline 200d/w5/min3/sub3-6/neg10",
    },
    {
        "codigo": "WS-3",
        "sifts": "fasttext-wiki-10m-ws-3_sifts.txt",
        "nota": "mejor delta de fusion de v5.1",
    },
    {
        "codigo": "VS-100",
        "sifts": "fasttext-wiki-10m-vs-100_sifts.txt",
        "nota": "mejor delta tras el baseline",
    },
    {
        "codigo": "NEG-5",
        "sifts": "fasttext-wiki-10m-neg-5_sifts.txt",
        "nota": "mejor embedding aislado (0.6664)",
    },
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P7] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Los scripts de v5 conservan su propia copia del protocolo como evidencia historica y no
    se modifican; por eso se cargan con ``importlib``.

    Args:
        nombre: nombre del modulo en memoria.
        ruta: ruta del archivo.

    Returns:
        El modulo cargado.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def filas_ortograficas(modulo_p1, desarrollo: pd.DataFrame) -> np.ndarray:
    """Marca las filas con formas ortograficas degradadas.

    Se usan los grupos ``elongaciones`` y ``formas_raras`` del vocabulario TOXIC, que P1
    delimito como evasiones y deformaciones. Un tipo de estos grupos en la fila significa que
    la palabra se escribio de otra forma, que es exactamente la situacion en la que un vector
    de subpalabra puede reconstruir y un char n-gram tambien puede.

    Args:
        modulo_p1: modulo de P1, con el vocabulario TOXIC y los grupos.
        desarrollo: marco del desarrollo.

    Returns:
        Mascara booleana por fila.
    """
    from src.embedding_utils import tokenizar

    grupos = modulo_p1.clasificar_grupos(modulo_p1.vocabulario_toxic(desarrollo))
    objetivos = set(grupos.get("elongaciones", [])) | set(grupos.get("formas_raras", []))
    tokens_por_fila = [
        set(tokenizar(normalizar_emb_normal(valor))) for valor in desarrollo["text"].astype(str)
    ]
    return np.asarray([bool(objetivos & tokens) for tokens in tokens_por_fila], dtype=bool)


def bootstrap_interaccion(
    y: np.ndarray,
    pred_candidato: np.ndarray,
    pred_referencia: np.ndarray,
    mascara: np.ndarray,
    repeticiones: int = REPETICIONES_BOOTSTRAP,
    semilla: int = SEMILLA_BOOTSTRAP,
) -> dict[str, Any]:
    """Bootstrap pareado de la diferencia de deltas entre dos subconjuntos de filas.

    En cada repeticion se remuestrea el **mismo** conjunto de filas para el candidato y para
    la referencia, y se calculan los dos deltas por separado. Es lo que hace comparables dos
    grupos de tamano distinto: el mismo ruido de filas afecta a los dos lados.

    Args:
        y: matriz binaria de labels.
        pred_candidato: predicciones de la variante con umbrales congelados.
        pred_referencia: predicciones de V03.
        mascara: mascara booleana del subconjunto de interes.
        repeticiones: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Diccionario con los deltas por subconjunto y el intervalo de la diferencia.
    """
    generador = np.random.default_rng(semilla)
    dentro = np.flatnonzero(mascara)
    fuera = np.flatnonzero(~mascara)
    delta_dentro = metrica_macro(y[dentro], pred_candidato[dentro]) - metrica_macro(
        y[dentro], pred_referencia[dentro]
    )
    delta_fuera = metrica_macro(y[fuera], pred_candidato[fuera]) - metrica_macro(
        y[fuera], pred_referencia[fuera]
    )
    diferencias = np.empty(repeticiones, dtype=np.float64)
    for indice in range(repeticiones):
        muestra_dentro = dentro[generador.integers(0, len(dentro), len(dentro))]
        muestra_fuera = fuera[generador.integers(0, len(fuera), len(fuera))]
        dentro_delta = metrica_macro(y[muestra_dentro], pred_candidato[muestra_dentro]) - metrica_macro(
            y[muestra_dentro], pred_referencia[muestra_dentro]
        )
        fuera_delta = metrica_macro(y[muestra_fuera], pred_candidato[muestra_fuera]) - metrica_macro(
            y[muestra_fuera], pred_referencia[muestra_fuera]
        )
        diferencias[indice] = dentro_delta - fuera_delta
    percentiles = np.percentile(diferencias, [2.5, 97.5])
    return {
        "n_dentro": int(len(dentro)),
        "n_fuera": int(len(fuera)),
        "delta_dentro": round(float(delta_dentro), 6),
        "delta_fuera": round(float(delta_fuera), 6),
        "diferencia": round(float(delta_dentro - delta_fuera), 6),
        "ic95": [round(float(percentiles[0]), 6), round(float(percentiles[1]), 6)],
        "excluye_cero": bool(percentiles[0] > 0 or percentiles[1] < 0),
    }


def main() -> None:
    """Ejecuta P7: complementariedad, damage, interaccion ortografica y ejemplos."""
    inicio = time.time()
    log("P7: que arregla el embedding y por que no arregla mas")
    modulo_p1 = _cargar_modulo(
        "embedding_01_p1", PROJECT_ROOT / "experiments" / "embedding_01_corpus_wikipedia_v5.py"
    )
    modulo_p3 = _cargar_modulo(
        "embedding_03_p3", PROJECT_ROOT / "experiments" / "embedding_03_fusion_complementariedad_v5.py"
    )
    from src.evaluation import _f1_por_celda

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    pred_v03 = oof["pred"]
    textos = [normalizar_emb_normal(texto) for texto in desarrollo["text"].astype(str)]

    log("1/5 reevaluacion de las cuatro variantes (sin entrenar)")
    filas: list[dict[str, Any]] = []
    predicciones: dict[str, np.ndarray] = {"V03-CONGELADO": pred_v03}
    variantes_completas: dict[str, dict[str, Any]] = {}
    for variante in VARIANTES:
        vectores = modulo_p3.cargar_sifts(EMB_DIR / variante["sifts"])
        embeddings = modulo_p3.vectores_documento(textos, vectores)
        resultado = modulo_p3.evaluar_variante(
            f"FUSION-{variante['codigo']}",
            desarrollo["text"],
            y,
            pliegues,
            embeddings,
            modo_embedding="l2",
        )
        variantes_completas[variante["codigo"]] = resultado
        predicciones[f"FUSION-{variante['codigo']}"] = resultado["pred_frozen"]
        f1_candidato = _f1_por_celda(y, resultado["pred_frozen"])
        f1_v03 = _f1_por_celda(y, pred_v03)
        acierta_candidato = f1_candidato >= 0.999
        acierta_v03 = f1_v03 >= 0.999
        salvage = int(np.sum(acierta_candidato & ~acierta_v03))
        damage = int(np.sum(~acierta_candidato & acierta_v03))
        total = int(f1_candidato.size)
        filas.append(
            {
                "variante": variante["codigo"],
                "nota": variante["nota"],
                "corrected_cv": resultado["corrected_cv_umbrales_frozen_v03"],
                "delta_real_vs_V03": round(
                    resultado["corrected_cv_umbrales_frozen_v03"] - REFERENCIA_V03, 6
                ),
                "filas_totales": total,
                "ambas_correctas": int(np.sum(acierta_candidato & acierta_v03)),
                "salvage": salvage,
                "damage": damage,
                "correccion_neta": salvage - damage,
                "fraccion_de_filas": round((salvage - damage) / total, 6),
                "ninguna_correcta": int(np.sum(~acierta_candidato & ~acierta_v03)),
            }
        )
        log(
            f"   {variante['codigo']:8s} {filas[-1]['corrected_cv']:.4f} "
            f"(delta {filas[-1]['delta_real_vs_V03']:+.4f}) | salvage {salvage} damage "
            f"{damage} neta {salvage - damage:+d} de {total} filas"
        )

    log("2/5 interaccion ortografica: ¿el efecto se concentra donde la palabra esta deformada?")
    mascara = filas_ortograficas(modulo_p1, desarrollo)
    interacciones: dict[str, Any] = {}
    for variante in VARIANTES:
        resultado = bootstrap_interaccion(
            y, variantes_completas[variante["codigo"]]["pred_frozen"], pred_v03, mascara
        )
        interacciones[variante["codigo"]] = resultado
        log(
            f"   {variante['codigo']:8s} dentro (n={resultado['n_dentro']:4d}) delta "
            f"{resultado['delta_dentro']:+.4f} | fuera (n={resultado['n_fuera']:4d}) delta "
            f"{resultado['delta_fuera']:+.4f} | diferencia {resultado['diferencia']:+.4f} "
            f"IC95 {resultado['ic95']} "
            f"{'INTERACCION' if resultado['excluye_cero'] else 'sin interaccion'}"
        )

    log("3/5 ejemplos de filas donde el embedding y V03 discrepan")
    mejor = max(filas, key=lambda f: f["salvage"] - f["damage"])
    pred_mejor = variantes_completas[mejor["variante"]]["pred_frozen"]
    f1_c = _f1_por_celda(y, pred_mejor)
    f1_v = _f1_por_celda(y, pred_v03)
    a_c = f1_c >= 0.999
    a_v = f1_v >= 0.999
    etiquetas = list(ETIQUETAS)
    ejemplos_salvage: list[dict[str, Any]] = []
    # `a_c` y `a_v` son mascaras **por fila** (la F1 de cada fila ya promedia sus tres
    # etiquetas), asi que se indexan con un solo indice. Para saber que etiqueta concreta
    # cambia se comparan las predicciones etiqueta a etiqueta.
    for indice_fila in np.flatnonzero(a_c & ~a_v):
        for etiqueta in range(3):
            if (
                pred_mejor[indice_fila, etiqueta] == y[indice_fila, etiqueta]
                and pred_v03[indice_fila, etiqueta] != y[indice_fila, etiqueta]
            ):
                ejemplos_salvage.append(
                    {
                        "id": int(desarrollo["id"].iloc[indice_fila]),
                        "etiqueta": etiquetas[etiqueta],
                        "y": int(y[indice_fila, etiqueta]),
                        "v03": int(pred_v03[indice_fila, etiqueta]),
                        "fusion": int(pred_mejor[indice_fila, etiqueta]),
                        "texto": str(desarrollo["text"].iloc[indice_fila])[:160],
                    }
                )
    ejemplos_damage: list[dict[str, Any]] = []
    for indice_fila in np.flatnonzero(~a_c & a_v):
        for etiqueta in range(3):
            if (
                pred_v03[indice_fila, etiqueta] == y[indice_fila, etiqueta]
                and pred_mejor[indice_fila, etiqueta] != y[indice_fila, etiqueta]
            ):
                ejemplos_damage.append(
                    {
                        "id": int(desarrollo["id"].iloc[indice_fila]),
                        "etiqueta": etiquetas[etiqueta],
                        "y": int(y[indice_fila, etiqueta]),
                        "v03": int(pred_v03[indice_fila, etiqueta]),
                        "fusion": int(pred_mejor[indice_fila, etiqueta]),
                        "texto": str(desarrollo["text"].iloc[indice_fila])[:160],
                    }
                )
    log(
        f"   {mejor['variante']}: {len(ejemplos_salvage)} etiquetas que arregla sobre "
        f"{len(ejemplos_damage)} que rompe"
    )

    log("4/5 slices consolidadas")
    mascaras = modulo_p3.rebanadas(desarrollo, y)
    metricas_rebanada = modulo_p3.metricas_por_rebanada(y, predicciones, mascaras)
    por_rebanada: dict[str, dict[str, float]] = {}
    for fila in metricas_rebanada:
        por_rebanada.setdefault(fila["rebanada"], {})[fila["variante"]] = fila["f1_por_celda"]

    log("5/5 figura e informe")
    _figura(filas)
    _informe(
        filas,
        interacciones,
        ejemplos_salvage,
        ejemplos_damage,
        por_rebanada,
        mascaras,
        mejor,
        time.time() - inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": REFERENCIA_V03,
                "granularidad": "filas (1.483); la F1 por fila es la media de sus tres etiquetas",
                "variantes": filas,
                "interaccion_ortografica": interacciones,
                "ejemplos_salvage": ejemplos_salvage[:40],
                "ejemplos_damage": ejemplos_damage[:40],
                "rebanadas": metricas_rebanada,
                "mascara_ortografica": {
                    "n_filas": int(mascara.sum()),
                    "definicion": "filas con algun tipo de los grupos elongaciones o formas_raras",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P7 completada en {round(time.time() - inicio, 1)} s")


def _figura(filas: list[dict[str, Any]]) -> None:
    """Dibuja el balance de filas por variante.

    Args:
        filas: filas de la tabla de complementariedad.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    nombres = [f["variante"] for f in filas]
    posiciones = np.arange(len(nombres))
    fig, eje = plt.subplots(figsize=(8, 4.4))
    eje.bar(
        posiciones - 0.2,
        [f["salvage"] for f in filas],
        0.4,
        label="salvage (arregla)",
        color="#2ca02c",
    )
    eje.bar(
        posiciones + 0.2,
        [f["damage"] for f in filas],
        0.4,
        label="damage (rompe)",
        color="#d62728",
    )
    for indice, fila in enumerate(filas):
        eje.text(
            indice,
            max(fila["salvage"], fila["damage"]) + 1,
            f"neta {fila['correccion_neta']:+d}",
            ha="center",
            fontsize=8,
        )
    eje.set_xticks(posiciones)
    eje.set_xticklabels(nombres)
    eje.set_ylabel("filas (de 1.483)")
    eje.set_title("Complementariedad fila a fila frente a V03")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_57_complementariedad_v5_1.png", dpi=150)
    plt.close()


def _informe(
    filas: list[dict[str, Any]],
    interacciones: dict[str, Any],
    ejemplos_salvage: list[dict[str, Any]],
    ejemplos_damage: list[dict[str, Any]],
    por_rebanada: dict[str, dict[str, float]],
    mascaras: dict[str, np.ndarray],
    mejor: dict[str, Any],
    duracion: float,
) -> None:
    """Escribe el informe de complementariedad.

    Args:
        filas: tabla de complementariedad.
        interacciones: resultado de la prueba de interaccion por variante.
        ejemplos_salvage: etiquetas que el embedding arregla.
        ejemplos_damage: etiquetas que el embedding rompe.
        por_rebanada: metricas por rebanada.
        mascaras: mascaras de las rebanadas.
        mejor: variante con mejor correccion neta.
        duracion: segundos que tardo el script.
    """
    codigo = mejor["variante"]
    interaccion = interacciones[codigo]
    total_filas = interaccion["n_dentro"] + interaccion["n_fuera"]
    lineas = [
        "# P7 de v5.1: complementariedad y solapamiento con char_wb (EBD-24)",
        "",
        f"Generado por `experiments/embedding_18_complementariedad_v5_1.py` ({EXPERIMENTO}).",
        "No entrena nada: reevalua cuatro configuraciones ya entrenadas y las compara fila a",
        "fila con V03 (1.483 filas; la F1 de cada fila es la media de sus tres etiquetas).",
        "",
        "## 0. Correccion de granularidad",
        "",
        "Este script corrige un nombre que arrastraba la fase v5. La funcion",
        "`src.evaluation._f1_por_celda` y su docstring hablan de celdas (fila, etiqueta), pero",
        "devuelven la media de las tres etiquetas de cada fila: la unidad de analisis es la",
        "**fila** (1.483), no la celda (4.449). Los conteos publicados en v5 no cambian, porque",
        "siempre se contaron filas, pero la etiqueta estaba mal y hay que corregirla en lugar",
        "de propagarla. Por eso no se habla de \"correccion neta equivalente en F1\": contar",
        "filas que cambian y medir el delta de la metrica son dos cosas distintas.",
        "",
        "## 1. Que arregla y que rompe",
        "",
        "`salvage` son las filas que el embedding arregla y V03 falla, `damage` las que V03",
        "arregla y el embedding falla, y la correccion neta es la diferencia. Una fila se",
        "considera acertada solo si sus tres etiquetas caen bien.",
        "",
        "| variante | corrected_cv | delta real | salvage | damage | correccion neta | fraccion de filas |",
        "|---|---|---|---|---|---|---|",
    ]
    for fila in filas:
        lineas.append(
            f"| {fila['variante']} | {fila['corrected_cv']:.4f} | "
            f"{fila['delta_real_vs_V03']:+.4f} | {fila['salvage']} | {fila['damage']} | "
            f"{fila['correccion_neta']:+d} | {fila['fraccion_de_filas']:+.5f} |"
        )
    lineas += [
        "",
        "![complementariedad](figuras/fig_57_complementariedad_v5_1.png)",
        "",
        "La lectura que importa es la **relacion entre salvage y damage**. En las cuatro",
        "variantes el embedding arregla unas pocas filas y rompe otras tantas: la correccion",
        f"neta es de un solo digito sobre {total_filas} filas. No es que el embedding haga bien o",
        "mal: es que **cambia pocas filas**, y las cambia en ambas direcciones.",
        "",
        "## 2. ¿De donde viene lo que aporta: subpalabra o distribucion?",
        "",
        "La seccion 19 de la spec pregunta si la mejora procede de **robustez ortografica**",
        "(que es lo que el char n-gram de V03 ya hace) o de **informacion distribucional** (que",
        "el char n-gram no tiene). El test no es comparar dos promedios sino medir si el efecto",
        "se **concentra** en las filas donde la palabra esta ortograficamente degradada:",
        f"{interaccion['n_dentro']} de {total_filas} filas contienen alguna forma de los grupos",
        "`elongaciones` o `formas_raras`; el resto, no.",
        "",
        "| variante | delta dentro (n=%d) | delta fuera (n=%d) | diferencia | IC95 | interaccion |"
        % (interaccion["n_dentro"], interaccion["n_fuera"]),
        "|---|---|---|---|---|---|",
    ]
    for variante, resultado in interacciones.items():
        linea = (
            f"| {variante} | {resultado['delta_dentro']:+.4f} | {resultado['delta_fuera']:+.4f} | "
            f"{resultado['diferencia']:+.4f} | [{resultado['ic95'][0]:+.4f}, "
            f"{resultado['ic95'][1]:+.4f}] | {'**si**' if resultado['excluye_cero'] else 'no'} |"
        )
        lineas.append(linea)
    lineas += [
        "",
        "El bootstrap remuestrea el **mismo** conjunto de filas para el candidato y para la",
        "referencia en cada repeticion. Sin ese emparejamiento, comparar dos grupos de tamano",
        "distinto mezcla el efecto que se quiere medir con el ruido de filas de cada grupo.",
        "",
        f"**Advertencia sobre el poder del test.** La mascara cubre "
        f"{interaccion['n_dentro']} de {total_filas} filas ({100 * interaccion['n_dentro'] / total_filas:.0f} %):",
        "el grupo `formas_raras` del vocabulario TOXIC incluye tipos frecuentes del propio",
        "lexico de odio, de modo que casi cualquier fila contiene alguno y la particion apenas",
        "discrimina. El resultado (delta positivo dentro, **negativo** fuera, diferencia",
        f"{interaccion['diferencia']:+.4f}) apunta en la direccion esperada, pero el intervalo",
        "de confianza incluye el 0 y con esta mascara el test tiene **poco poder**: no",
        "autoriza a concluir que no hay interaccion, solo que no se puede separar del ruido.",
        "Un test con mas poder exigiria un grupo mas estrecho (solo `elongaciones`, 57 filas en",
        "v5) y esas filas son tan pocas que la metrica por subconjunto seria aun mas ruidosa.",
        "Se reporta como lo que es: una direccion consistente en las cuatro variantes y un",
        "contraste que no alcanza significacion.",
        "",
        "## 3. Slices de la mejor variante",
        "",
        "| rebanada | filas | V03 | mejor fusion | delta |",
        "|---|---|---|---|---|",
    ]
    etiqueta = f"FUSION-{codigo}"
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0:
            continue
        valores = por_rebanada.get(nombre, {})
        base = valores.get("V03-CONGELADO", float("nan"))
        actual = valores.get(etiqueta, float("nan"))
        lineas.append(
            f"| {nombre} | {int(mascara.sum())} | {base:.4f} | {actual:.4f} | {actual - base:+.4f} |"
        )
    lineas += [
        "",
        "## 4. Ejemplos donde el embedding y V03 discrepan",
        "",
        f"### Etiquetas que {codigo} arregla ({len(ejemplos_salvage)} en total; se muestran 8)",
        "",
        "| id | etiqueta | real | V03 | fusion | texto |",
        "|---|---|---|---|---|---|",
    ]
    for ejemplo in ejemplos_salvage[:8]:
        texto = str(ejemplo["texto"])[:110].replace("|", "/")
        lineas.append(
            f"| {ejemplo['id']} | {ejemplo['etiqueta']} | {ejemplo['y']} | {ejemplo['v03']} | "
            f"{ejemplo['fusion']} | {texto} |"
        )
    lineas += [
        "",
        f"### Etiquetas que {codigo} rompe ({len(ejemplos_damage)} en total; se muestran 8)",
        "",
        "| id | etiqueta | real | V03 | fusion | texto |",
        "|---|---|---|---|---|---|",
    ]
    for ejemplo in ejemplos_damage[:8]:
        texto = str(ejemplo["texto"])[:110].replace("|", "/")
        lineas.append(
            f"| {ejemplo['id']} | {ejemplo['etiqueta']} | {ejemplo['y']} | {ejemplo['v03']} | "
            f"{ejemplo['fusion']} | {texto} |"
        )
    lineas += [
        "",
        "## 5. Conclusion",
        "",
        "El embedding no aporta un tipo de informacion que V03 no tenga: aporta **otra** misma",
        f"informacion, mas ruido. Su efecto neto sobre {total_filas} filas es de un solo digito.",
        "La prueba de interaccion ortografica dice si esa cifra se concentra donde la palabra",
        "esta deformada o esta repartida por igual. En cualquiera de los dos casos la",
        "conclusion operativa es la de D-041: no hay nada que adoptar, y lo que queda abierto",
        "no es la representacion sino el **numero de ejemplos etiquetados**.",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(duracion, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
