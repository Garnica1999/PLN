"""P7 de la fase v5: embedding sobre 50M tokens y curva de aprendizaje 10M -> 50M.

Contexto
--------
Ultima dimension que quedaba por medir: el **tamano** del corpus con la
arquitectura fija (seccion 10 de la spec). P4-P5 cerro la dimension de registro
(la cobertura subio y la aportacion no se movio); P6 construire el corpus de 50M
y verifico que contiene identicamente los 10M anteriores, de modo que la
comparacion sea exacta.

Este script entrena FastText y su control Word2Vec sobre los 50M tokens y vuelve a
medir lo unico que decide la fase: la aportacion complementaria sobre V03, en el
mismo punto de operacion (umbrales congelados de V03) y con el mismo bootstrap
pareado.

Las dos hipotesis en competencia (declaradas en el RATIONALE de P6):
- **H-50M**: si el limite era la calidad de la representacion, 5x mas tokens de la
  misma fuente no cambian la aportacion y la fusion se queda en ~+0.004.
- **H-contra**: si al menos el embedding **aislado** mejora de forma clara, el
  limite es el techo de V03 y no la representacion.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_08_fasttext_50M_v5.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.embedding_utils import (  # noqa: E402
    entrenar_fasttext,
    normalizar_emb_normal,
    sha256_de,
    vocabulario_textos,
    vectores_de_tipos,
)
from src.evaluation import ETIQUETAS, bootstrap_diferencia_pareada  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "EBD-13..14 (P7)"
FASTTEXT_BIN = PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
CORPUS_50M = (
    PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es_50M" / "emb_normal" / "tokens_50M.txt"
)
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "embedding_curve_50M_v5.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "embedding_curve_50M_v5.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_curve_50M_v5.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_curve_50M_slices_v5.csv"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

REFERENCIA_V03 = 0.777211
MEJORA_MINIMA = 0.005
REPETICIONES_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 20260925
ARQUITECTURA = {
    "dim": 200,
    "ventana": 5,
    "min_count": 3,
    "negativos": 10,
    "epocas": 5,
    "hilos": 16,
    "bucket": 200_000,
    "subpalabras": (3, 6),
    "semilla": 42,
}

MODELOS = {
    "FASTTEXT-WIKI-50M": {"corpus": CORPUS_50M, "minn": 3, "maxn": 6},
    "WORD2VEC-WIKI-50M": {"corpus": CORPUS_50M, "minn": 0, "maxn": 0},
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5-P7] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

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


def main() -> None:
    """Ejecuta P7: entrenamiento a 50M y curva de aprendizaje."""
    inicio = time.time()
    log("P7: FastText y control sobre 50M tokens, y curva 10M -> 50M")
    modulo_p3 = _cargar_modulo(
        "embedding_03_p3", PROJECT_ROOT / "experiments" / "embedding_03_fusion_complementariedad_v5.py"
    )

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos[indices_dev])
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    oof = np.load(OOF_V03, allow_pickle=True)
    if not np.array_equal(
        np.sort(oof["ids"].astype(int)), np.sort(desarrollo["id"].astype(int).to_numpy())
    ):
        raise RuntimeError("Los ids del OOF canonico no coinciden con el desarrollo")
    pred_v03 = oof["pred"]
    tipos_consulta = sorted(
        vocabulario_textos([normalizar_emb_normal(texto) for texto in train["text"].astype(str)]).keys()
    )

    log(f"1/4 entrenamiento (5 epocas x 50M tokens, {ARQUITECTURA['hilos']} hilos)")
    registro: dict[str, Any] = {}
    for nombre, configuracion in MODELOS.items():
        prefijo = EMB_DIR / nombre.lower().replace("-", "_")
        segundos = 0.0
        if not Path(f"{prefijo}.vec").exists():
            t0 = time.time()
            entrenar_fasttext(
                FASTTEXT_BIN,
                configuracion["corpus"],
                prefijo,
                dim=ARQUITECTURA["dim"],
                ventana=ARQUITECTURA["ventana"],
                min_count=ARQUITECTURA["min_count"],
                negativos=ARQUITECTURA["negativos"],
                epocas=ARQUITECTURA["epocas"],
                hilos=ARQUITECTURA["hilos"],
                semilla=ARQUITECTURA["semilla"],
                minn=configuracion["minn"],
                maxn=configuracion["maxn"],
                bucket=ARQUITECTURA["bucket"],
            )
            segundos = time.time() - t0
        vectores = vectores_de_tipos(
            FASTTEXT_BIN, Path(f"{prefijo}.bin"), tipos_consulta, EMB_DIR / f"{nombre.lower()}_sifts.txt"
        )
        with Path(f"{prefijo}.vec").open(encoding="utf-8") as archivo:
            vocabulario, dimension = archivo.readline().split()
        registro[nombre] = {
            "corpus": str(configuracion["corpus"].relative_to(PROJECT_ROOT)),
            "vocab_size": int(vocabulario),
            "dim": int(dimension),
            "sha256": sha256_de(Path(f"{prefijo}.vec")),
            "segundos_entrenamiento": round(segundos, 1),
            "tipos_con_vector": len(vectores),
            "minn": configuracion["minn"],
        }
        log(
            f"   {nombre}: {segundos:.0f} s | vocab {vocabulario} | tipos con vector "
            f"{len(vectores)} de {len(tipos_consulta)}"
        )

    log("2/4 evaluacion (umbrales congelados de V03)")
    variantes = []
    for nombre in MODELOS:
        prefijo = EMB_DIR / nombre.lower().replace("-", "_")
        vectores = modulo_p3.cargar_sifts(EMB_DIR / f"{nombre.lower()}_sifts.txt")
        embeddings = modulo_p3.vectores_documento(
            [normalizar_emb_normal(texto) for texto in desarrollo["text"]], vectores
        )
        variantes.append(
            modulo_p3.evaluar_variante(
                f"FUSION-{nombre}", desarrollo["text"], y, pliegues, embeddings, modo_embedding="l2"
            )
        )
        variantes.append(
            modulo_p3.evaluar_variante(
                f"EMB-{nombre}", desarrollo["text"], y, pliegues, embeddings,
                usar_tfidf=False, modo_embedding="estandar",
            )
        )

    log("3/4 comparaciones y rebanadas")
    filas = []
    for variante in variantes:
        comparacion = bootstrap_diferencia_pareada(
            y, variante["pred_frozen"], pred_v03, REPETICIONES_BOOTSTRAP, SEMILLA_BOOTSTRAP
        )
        matriz = modulo_p3.matriz_complementariedad(y, variante["pred_frozen"], pred_v03)
        filas.append(
            {
                "variante": variante["variante"],
                "corrected_cv_frozen_v03": variante["corrected_cv_umbrales_frozen_v03"],
                "delta_vs_v03": round(variante["corrected_cv_umbrales_frozen_v03"] - REFERENCIA_V03, 6),
                "ic95_0": comparacion["ic95"][0],
                "ic95_1": comparacion["ic95"][1],
                "celdas_mejoradas": comparacion["celdas_mejoradas"],
                "celdas_empeoradas": comparacion["celdas_empeoradas"],
                "solo_candidato": matriz["solo_candidato"],
                "solo_v03": matriz["solo_referencia"],
                "f1_HS": round(variante["f1_por_etiqueta_ajustados"]["HS"], 4),
                "f1_TR": round(variante["f1_por_etiqueta_ajustados"]["TR"], 4),
                "f1_AG": round(variante["f1_por_etiqueta_ajustados"]["AG"], 4),
            }
        )
        log(
            f"   {variante['variante']:28s} {filas[-1]['corrected_cv_frozen_v03']:.4f} "
            f"delta {filas[-1]['delta_vs_v03']:+.4f} "
            f"IC95 [{filas[-1]['ic95_0']:+.4f}, {filas[-1]['ic95_1']:+.4f}]"
        )

    # Curva completa: reutiliza los resultados ya medidos de 10M y web/mixto.
    curva = [
        ("wikipedia 10M", "FUSION-FASTTEXT-WIKI-10M", 0.7818, 0.6530, 0.6839, 0.8833),
        ("web 10M", "FUSION-FASTTEXT-WEB-10M", 0.7812, 0.6638, 0.8256, 0.9512),
        ("mixto 20M", "FUSION-FASTTEXT-MIXTO-20M", 0.7776, 0.6658, 0.8413, 0.9553),
    ]
    emb_50m = next(f for f in filas if f["variante"] == "EMB-FASTTEXT-WIKI-50M")
    fus_50m = next(f for f in filas if f["variante"] == "FUSION-FASTTEXT-WIKI-50M")
    cobertura_50m = json.loads(
        (PROJECT_ROOT / "reports" / "embedding_corpus_50M_v5.json").read_text(encoding="utf-8")
    )["cobertura_50M"]
    curva.append(
        ("wikipedia 50M", "FUSION-FASTTEXT-WIKI-50M", fus_50m["corrected_cv_frozen_v03"],
         emb_50m["corrected_cv_frozen_v03"], cobertura_50m["type_coverage"], cobertura_50m["token_coverage"])
    )

    mascaras = modulo_p3.rebanadas(desarrollo, y)
    metricas_rebanada = modulo_p3.metricas_por_rebanada(
        y,
        {v["variante"]: v["pred_frozen"] for v in variantes} | {"V03-CONGELADO": pred_v03},
        mascaras,
    )

    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    with REPORTE_REBANADAS.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["rebanada", "variante", "n_filas", "f1_por_celda"], lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(metricas_rebanada)

    log("4/4 curva, informe y figura")
    _figura(curva, filas)
    _informe(registro, filas, curva, metricas_rebanada, mascaras)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "arquitectura": {**ARQUITECTURA, "subpalabras": list(ARQUITECTURA["subpalabras"])},
                "modelos": registro,
                "comparaciones": filas,
                "curva": [
                    {"corpus": c, "variante": v, "f1_fusion": f, "f1_embedding": e,
                     "cobertura_tipo": ct, "cobertura_token": ctk}
                    for c, v, f, e, ct, ctk in curva
                ],
                "rebanadas": metricas_rebanada,
                "referencia_v03": REFERENCIA_V03,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P7 completada en {round(time.time() - inicio, 1)} s")


def _figura(curva: list[tuple], filas: list[dict[str, Any]]) -> None:
    """Genera la figura de la curva de aprendizaje.

    Args:
        curva: filas de la curva (corpus, variante, fusion, embedding, cov_tipo, cov_token).
        filas: comparaciones de esta corrida.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    etiquetas = [fila[0] for fila in curva]
    fusiones = [fila[2] for fila in curva]
    embeddings = [fila[3] for fila in curva]
    posiciones = np.arange(len(etiquetas))
    plt.figure(figsize=(8.5, 4.5))
    plt.plot(posiciones, fusiones, "o-", label="fusion con V03", color="#1f77b4")
    plt.plot(posiciones, embeddings, "s-", label="embedding solo", color="#7f7f7f")
    plt.axhline(REFERENCIA_V03, color="black", linestyle="--", label="V03 0.7772")
    plt.xticks(posiciones, etiquetas, fontsize=9)
    plt.ylabel("corrected_cv (umbrales congelados de V03)")
    plt.title("Curva 10M -> 50M tokens y efecto del registro (arquitectura fija)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_48_curva_aprendizaje_v5.png", dpi=150)
    plt.close()


def _informe(
    registro: dict[str, Any],
    filas: list[dict[str, Any]],
    curva: list[tuple],
    metricas_rebanada: list[dict[str, Any]],
    mascaras: dict[str, np.ndarray],
) -> None:
    """Escribe el informe de la curva 10M -> 50M.

    Args:
        registro: metadatos de los modelos.
        filas: comparaciones de la corrida.
        curva: filas de la curva.
        metricas_rebanada: metricas por rebanada.
        mascaras: tamanos de las rebanadas.
    """
    por_rebanada: dict[str, dict[str, float]] = {}
    for fila in metricas_rebanada:
        por_rebanada.setdefault(fila["rebanada"], {})[fila["variante"]] = fila["f1_por_celda"]
    mejor_fusion = max(
        (f for f in filas if f["variante"].startswith("FUSION")),
        key=lambda f: f["corrected_cv_frozen_v03"],
    )
    lineas = [
        "# Curva de aprendizaje 10M -> 50M y efecto del registro (fase v5, P7)",
        "",
        f"Generado por `experiments/embedding_08_fasttext_50M_v5.py` ({EXPERIMENTO}).",
        "",
        "## 1. Modelos (arquitectura fija de la §4 de la spec)",
        "",
        "| modelo | corpus | subpalabras | vocabulario | s | tipos con vector | sha256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for nombre, datos in registro.items():
        lineas.append(
            f"| {nombre} | 50M | {datos['minn']}-6 | {datos['vocab_size']} | "
            f"{datos['segundos_entrenamiento']} | {datos['tipos_con_vector']} | "
            f"`{datos['sha256'][:16]}` |"
        )
    lineas += [
        "",
        "## 2. Resultados de esta corrida",
        "",
        "| variante | corrected_cv | delta vs V03 | IC95 pareado | celdas + | celdas - | solo esta | solo V03 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fila in filas:
        lineas.append(
            f"| {fila['variante']} | {fila['corrected_cv_frozen_v03']:.4f} | "
            f"{fila['delta_vs_v03']:+.4f} | [{fila['ic95_0']:+.4f}, {fila['ic95_1']:+.4f}] | "
            f"{fila['celdas_mejoradas']} | {fila['celdas_empeoradas']} | {fila['solo_candidato']} | "
            f"{fila['solo_v03']} |"
        )
    lineas += [
        "",
        "## 3. La curva completa",
        "",
        "Las cuatro filas comparten arquitectura, semilla, protocolo, umbrales y bootstrap. La",
        "diferencia es unicamente el corpus.",
        "",
        "| corpus | cobertura por tipo | cobertura por token | embedding solo | fusion con V03 | delta fusion |",
        "|---|---|---|---|---|---|",
    ]
    for nombre, _variante, fusion, embedding, cobertura_tipo, cobertura_token in curva:
        lineas.append(
            f"| {nombre} | {cobertura_tipo:.4f} | {cobertura_token:.4f} | {embedding:.4f} | "
            f"{fusion:.4f} | {fusion - REFERENCIA_V03:+.4f} |"
        )
    embeddings_10m = curva[0][3]
    embeddings_50m = curva[-1][3]
    fusiones_10m = curva[0][2]
    fusiones_50m = curva[-1][2]
    lineas += [
        "",
        "## 4. Lectura",
        "",
        f"- **Embedding aislado**: 10M {embeddings_10m:.4f} -> 50M {embeddings_50m:.4f} "
        f"({embeddings_50m - embeddings_10m:+.4f}).",
        f"- **Fusion con V03**: 10M {fusiones_10m:.4f} -> 50M {fusiones_50m:.4f} "
        f"({fusiones_50m - fusiones_10m:+.4f}), con delta sobre V03 de "
        f"{fusiones_50m - REFERENCIA_V03:+.4f}.",
        "",
        "## 5. Complementariedad por rebanada",
        "",
        "| rebanada | filas | V03 | fusion wiki 50M |",
        "|---|---|---|---|",
    ]
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0:
            continue
        valores = por_rebanada.get(nombre, {})
        lineas.append(
            f"| {nombre} | {int(mascara.sum())} | {valores.get('V03-CONGELADO', float('nan')):.4f} | "
            f"{valores.get('FUSION-FASTTEXT-WIKI-50M', float('nan')):.4f} |"
        )
    cumple = mejor_fusion["delta_vs_v03"] >= MEJORA_MINIMA and mejor_fusion["ic95_0"] > 0
    lineas += [
        "",
        "## 6. Decision",
        "",
        f"Criterio pre-registrado de la §29 sin cambios: adopted exige delta >= +{MEJORA_MINIMA:.3f} y",
        f"IC95 que excluya el 0. Mejor fusion de la corrida: **{mejor_fusion['variante']}** con",
        f"{mejor_fusion['corrected_cv_frozen_v03']:.4f} (delta {mejor_fusion['delta_vs_v03']:+.4f}, IC95",
        f"[{mejor_fusion['ic95_0']:+.4f}, {mejor_fusion['ic95_1']:+.4f}]). Adoptado: **{'si' if cumple else 'no'}**.",
        "",
        "Si la aportacion complementaria no crece al quintuplicar los tokens, no se escala a 100M:",
        "la conclusion de D-032 queda confirmada tambien en la dimension de tamano. Ninguna",
        "variante genera submission ni sonda de leaderboard.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
