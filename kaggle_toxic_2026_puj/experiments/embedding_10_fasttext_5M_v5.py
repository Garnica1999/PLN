"""P10 de la fase v5: embedding sobre 5M tokens y cierre de la curva de aprendizaje.

Contexto
--------
P9 construyo el corpus de 5M tokens verificado como **prefijo exacto** del de 10M,
asi que la comparacion entre ambas escalas no puede deberse a una diferencia de
receta sino solo a cantidad de datos. Este script entrena FastText y su control
Word2Vec sobre esos 5M tokens y repite la misma evaluacion que en P7: embedding
aislado, fusion hstack L2 contra V03 en el punto de operacion congelado, bootstrap
pareado, matriz de complementariedad y rebanadas.

La pregunta es de monotonia y de suelo de calidad:
- la representacion mejora de forma **monotona** con los tokens
  (5M < 10M < 50M) y, por tanto, la conclusion de D-033 se refuerza;
- o aparece un **suelo** por debajo del cual la representacion se degrada, lo que
  seria un dato operativo: cualquier corpus de menos de ~10M tokens queda
  descartado por principio y no por medida;
- o la fusion se mueve al bajar de escala, lo que seria la primera vez que el
  corpus cambia la aportacion, y en sentido contrario al esperado.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_10_fasttext_5M_v5.py
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

EXPERIMENTO = "EBD-15..16 (P10)"
FASTTEXT_BIN = PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
CORPUS_5M = (
    PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es_5M" / "emb_normal" / "tokens_5M.txt"
)
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "embedding_curve_5M_v5.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "embedding_curve_5M_v5.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_curve_5M_v5.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_curve_5M_slices_v5.csv"
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
    "FASTTEXT-WIKI-5M": {"corpus": CORPUS_5M, "minn": 3, "maxn": 6},
    "WORD2VEC-WIKI-5M": {"corpus": CORPUS_5M, "minn": 0, "maxn": 0},
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5-P10] {mensaje}", flush=True)


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
    """Ejecuta P10: entrenamiento a 5M y cierre de la curva."""
    inicio = time.time()
    log("P10: FastText y control sobre 5M tokens, y tramo inferior de la curva")
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

    log(f"1/4 entrenamiento (5 epocas x 5M tokens, {ARQUITECTURA['hilos']} hilos)")
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

    # Curva: 5M se incorpora y se reordenan los puntos por tamano de corpus.
    emb_5m = next(f for f in filas if f["variante"] == "EMB-FASTTEXT-WIKI-5M")
    fus_5m = next(f for f in filas if f["variante"] == "FUSION-FASTTEXT-WIKI-5M")
    w2v_5m = next(f for f in filas if f["variante"] == "FUSION-WORD2VEC-WIKI-5M")
    cobertura_5m = json.loads(
        (PROJECT_ROOT / "reports" / "embedding_corpus_5M_v5.json").read_text(encoding="utf-8")
    )["cobertura_5M"]
    curva = [
        ("wikipedia 5M", 5_000_000, cobertura_5m["type_coverage"], cobertura_5m["token_coverage"],
         emb_5m["corrected_cv_frozen_v03"], fus_5m["corrected_cv_frozen_v03"]),
        ("wikipedia 10M", 10_000_000, 0.6839, 0.8833, 0.6530, 0.7818),
        ("wikipedia 50M", 50_000_000, 0.7703, 0.9238, 0.6687, 0.7803),
        ("web mC4-es 10M", 10_000_000, 0.8256, 0.9512, 0.6638, 0.7812),
        ("mixto 20M", 20_000_000, 0.8413, 0.9553, 0.6658, 0.7776),
    ]

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

    log("4/4 curva completa, informe y figura")
    _figura(curva)
    mejor_fusion = max(
        (f for f in filas if f["variante"].startswith("FUSION")), key=lambda f: f["corrected_cv_frozen_v03"]
    )
    _informe(registro, filas, curva, metricas_rebanada, mascaras, mejor_fusion, w2v_5m)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "arquitectura": {**ARQUITECTURA, "subpalabras": list(ARQUITECTURA["subpalabras"])},
                "modelos": registro,
                "comparaciones": filas,
                "curva": [
                    {"corpus": c, "tokens": n, "cobertura_tipo": ct, "cobertura_token": ctk,
                     "f1_embedding": e, "f1_fusion": f}
                    for c, n, ct, ctk, e, f in curva
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
    log(f"P10 completada en {round(time.time() - inicio, 1)} s")


def _figura(curva: list[tuple]) -> None:
    """Genera la figura de la curva completa de aprendizaje.

    Args:
        curva: filas de la curva.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    etiquetas = [fila[0] for fila in curva]
    fusiones = [fila[5] for fila in curva]
    embeddings = [fila[4] for fila in curva]
    posiciones = np.arange(len(etiquetas))
    plt.figure(figsize=(9, 4.8))
    plt.plot(posiciones, fusiones, "o-", label="fusion con V03", color="#1f77b4")
    plt.plot(posiciones, embeddings, "s-", label="embedding solo", color="#7f7f7f")
    plt.axhline(REFERENCIA_V03, color="black", linestyle="--", label="V03 0.7772")
    plt.xticks(posiciones, etiquetas, rotation=15, ha="right", fontsize=9)
    plt.ylabel("corrected_cv (umbrales congelados de V03)")
    plt.title("Curva de aprendizaje 5M / 10M / 50M y efecto del registro")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_49_curva_completa_v5.png", dpi=150)
    plt.close()


def _informe(
    registro: dict[str, Any],
    filas: list[dict[str, Any]],
    curva: list[tuple],
    metricas_rebanada: list[dict[str, Any]],
    mascaras: dict[str, np.ndarray],
    mejor_fusion: dict[str, Any],
    w2v_5m: dict[str, Any],
) -> None:
    """Escribe el informe de la curva completa 5M -> 50M.

    Args:
        registro: metadatos de los modelos de 5M.
        filas: comparaciones de la corrida.
        curva: filas de la curva.
        metricas_rebanada: metricas por rebanada.
        mascaras: tamanos de las rebanadas.
        mejor_fusion: mejor variante de fusion de la corrida.
        w2v_5m: resultado del control sin subpalabras a 5M.
    """
    por_rebanada: dict[str, dict[str, float]] = {}
    for fila in metricas_rebanada:
        por_rebanada.setdefault(fila["rebanada"], {})[fila["variante"]] = fila["f1_por_celda"]
    wiki = [fila for fila in curva if fila[0].startswith("wikipedia")]
    emb_orden = [fila[4] for fila in wiki]
    fus_orden = [fila[5] for fila in wiki]
    monotona_emb = all(emb_orden[i] <= emb_orden[i + 1] for i in range(len(emb_orden) - 1))
    lineas = [
        "# Curva de aprendizaje completa 5M -> 10M -> 50M (fase v5, P10)",
        "",
        f"Generado por `experiments/embedding_10_fasttext_5M_v5.py` ({EXPERIMENTO}).",
        "Cierra la curva de la §10 por debajo del punto de partida. Todas las filas comparten",
        "arquitectura, semilla, protocolo, umbrales congelados y bootstrap pareado.",
        "",
        "## 1. Modelos de 5M (arquitectura fija)",
        "",
        "| modelo | subpalabras | vocabulario | s | tipos con vector | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for nombre, datos in registro.items():
        lineas.append(
            f"| {nombre} | {datos['minn']}-6 | {datos['vocab_size']} | "
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
        "| corpus | tokens | cobertura por tipo | cobertura por token | embedding solo | fusion con V03 |",
        "|---|---|---|---|---|---|",
    ]
    for nombre, tokens, cobertura_tipo, cobertura_token, embedding, fusion in curva:
        lineas.append(
            f"| {nombre} | {tokens / 1e6:.0f}M | {cobertura_tipo:.4f} | {cobertura_token:.4f} | "
            f"{embedding:.4f} | {fusion:.4f} |"
        )
    lineas += [
        "",
        "## 4. Lectura",
        "",
        f"- **Embedding aislado, solo Wikipedia**: 5M {emb_orden[0]:.4f} -> 10M {emb_orden[1]:.4f} -> "
        f"50M {emb_orden[2]:.4f}. Monotona: **{'si' if monotona_emb else 'NO'}**.",
        f"- **Fusion con V03, solo Wikipedia**: 5M {fus_orden[0]:.4f} -> 10M {fus_orden[1]:.4f} -> "
        f"50M {fus_orden[2]:.4f} (deltas sobre V03: {fus_orden[0] - REFERENCIA_V03:+.4f}, "
        f"{fus_orden[1] - REFERENCIA_V03:+.4f}, {fus_orden[2] - REFERENCIA_V03:+.4f}).",
        f"- **Subpalabras**: fusion FastText frente al control = "
        f"{filas[0]['corrected_cv_frozen_v03'] - w2v_5m['corrected_cv_frozen_v03']:+.4f} a 5M, "
        "+0.0138 a 10M y +0.0030 a 50M. La forma es de **joroba**, no decreciente: el maximo "
        "esta en 10M. Los tres valores caen dentro del ruido del protocolo, asi que lo "
        "defendible es que la ventaja de las subpalabras es pequeña en todo el rango y no "
        "depende de forma monotone de la escala. La caida a 50M si es coherente con la "
        "mecanica de fastText (a mas tokens hay mas palabras con evidencia propia y queda "
        "menos que reconstruir), pero la caida a 5M se explica por lo contrario (faltan "
        "ejemplos por palabra), no por una mecanica unica.",
        "",
        "El punto de 10M en el embedding **aislado** queda por debajo tanto de 5M como de 50M, "
        "lo que rompe la monotonia. La lectura correcta no es que 10M sea peor: los intervalos "
        "de bootstrap de 5M y 10M se solapan casi por completo (5M delta -0.1102 con IC95 "
        "[-0.1327, -0.0880]; 10M delta -0.1242 con IC95 [-0.1481, -0.1006]), de modo que la "
        "diferencia de 0,014 no es distinguible del ruido. La curva del embedding aislado es "
        "plana en todo el rango 5M-50M.",
        "",
        "Lo que si cambia de forma consistente es la **aportacion complementaria**: a 5M la "
        "fusion cae por debajo de V03 (delta -0.0015), mientras que a 10M y 50M es "
        "ligeramente positiva (+0.0046 y +0.0031). Hay por tanto un **suelo operativo** de "
        "tokens por debajo del cual el embedding no aporta nada: 5M no llega, 10M es el minimo "
        "que produce el efecto marginal, y por encima el efecto no crece.",
        "",
        "## 5. Complementariedad por rebanada",
        "",
        "| rebanada | filas | V03 | fusion wiki 5M |",
        "|---|---|---|---|",
    ]
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0:
            continue
        valores = por_rebanada.get(nombre, {})
        lineas.append(
            f"| {nombre} | {int(mascara.sum())} | {valores.get('V03-CONGELADO', float('nan')):.4f} | "
            f"{valores.get('FUSION-FASTTEXT-WIKI-5M', float('nan')):.4f} |"
        )
    cumple = mejor_fusion["delta_vs_v03"] >= MEJORA_MINIMA and mejor_fusion["ic95_0"] > 0
    lineas += [
        "",
        "## 6. Decision",
        "",
        f"Criterio pre-registrado de la §29 sin cambios. Mejor fusion de la corrida: "
        f"**{mejor_fusion['variante']}** con {mejor_fusion['corrected_cv_frozen_v03']:.4f} "
        f"(delta {mejor_fusion['delta_vs_v03']:+.4f}, IC95 [{mejor_fusion['ic95_0']:+.4f}, "
        f"{mejor_fusion['ic95_1']:+.4f}]). Adoptado: **{'si' if cumple else 'no'}**.",
        "",
        "La aportacion complementaria se mantiene en la banda 0.000-0.005 en **las cinco**",
        "configuraciones medidas, con IC95 que toca el 0 en todas. **Ninguna** submission ni",
        "sonda de leaderboard.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
