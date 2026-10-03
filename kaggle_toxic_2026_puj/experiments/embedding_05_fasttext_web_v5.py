"""P5 de la fase v5: embedding sobre corpus web y mixto, y revision de la fusion.

Contexto
--------
P4 confirmo la palanca de D-031 en el nivel de la cobertura: el registro web sube
la cobertura del vocabulario TOXIC de 0.6839 a 0.8256 por tipo (y la union
Wikipedia+web llega a 0.8413), con las mayores ganancias justo en los grupos que
P1 marco como debiles (`elongaciones` de 0.039 a 0.221, `coloquial` de 0.692 a
0.941, `insultos` de 0.778 a 0.983). La cobertura era la **condicion previa**; este
script mide si se traduce en laaportacion complementaria de la fusion.

Si la mecanica de P3 es correcta (la representacion ayuda en las rebanadas que la
cobertura mejora), entonces:
- la fusion con el embedding del corpus **web** debe superar a la fusion con el de
  Wikipedia, y
- la fusion con el corpus **mixto** debe ser la mejor de las tres, porque es la que
  conserva el texto enciclopedico y anade el registro web.

Y si la diferencia es plana, la conclusion honesta sera que la cobertura no es el
mecanismo y que la aportacion del embedding es marginal por otras razones.

Modelos entrenados desde cero (200d, ws5, subpalabras 3-6, neg10, minCount3,
5 epocas, semilla 42):
1. ``FASTTEXT-WEB-10M`` sobre el corpus web.
2. ``FASTTEXT-MIXTO-20M`` sobre la union Wikipedia + web.
3. ``WORD2VEC-WEB-10M`` (control sin subpalabras) sobre el corpus web, para
   comprobar si el efecto de las subpalabras se mantiene en registro web.

Todos se comparan contra ``FASTTEXT-WIKI-10M`` (el de P2/P3) y contra el control
``V03-TFIDF``, en el mismo punto de operacion: umbrales congelados de V03.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_05_fasttext_web_v5.py
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
from src.embedding_utils import (  # noqa: E402
    cargar_matriz_vec,
    entrenar_fasttext,
    normalizar_emb_normal,
    sha256_de,
    tokenizar,
    vectores_de_tipos,
    vocabulario_textos,
)
from src.evaluation import ETIQUETAS, bootstrap_diferencia_pareada  # noqa: E402
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

EXPERIMENTO = "EBD-10..12 (P5)"
FASTTEXT_BIN = PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
CORPUS_WIKI = PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es" / "emb_normal" / "tokens_10M.txt"
CORPUS_WEB = PROJECT_ROOT / "data" / "external" / "corpora" / "mc4_es" / "emb_normal" / "tokens_10M.txt"
CORPUS_MIXTO = PROJECT_ROOT / "data" / "external" / "corpora" / "mixto" / "emb_normal" / "tokens_20M.txt"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "embedding_web_results_v5.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "embedding_web_v5.json"
REPORTE_CSV = PROJECT_ROOT / "reports" / "embedding_web_results_v5.csv"
REPORTE_REBANADAS = PROJECT_ROOT / "reports" / "embedding_web_slices_v5.csv"
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
    "FASTTEXT-WEB-10M": {"corpus": CORPUS_WEB, "minn": 3, "maxn": 6, "etiqueta": "FASTTEXT-SG-SUBWORDS-3-6"},
    "FASTTEXT-MIXTO-20M": {"corpus": CORPUS_MIXTO, "minn": 3, "maxn": 6, "etiqueta": "FASTTEXT-SG-SUBWORDS-3-6"},
    "WORD2VEC-WEB-10M": {"corpus": CORPUS_WEB, "minn": 0, "maxn": 0, "etiqueta": "WORD2VEC-CONTROL / FASTTEXT-SG-SUBWORDS-OFF"},
}
# Modelo de referencia de P2/P3, que se reevalua en este script para que toda la
# tabla se calcule con el mismo codigo y las mismas cifras.
MODELO_REFERENCIA = "FASTTEXT-WIKI-10M"
# El modelo de P2 se guardo con el prefijo `fasttext_baseline`; el resto de
# modelos usan el nombre en minusculas con guiones bajos.
PREFIJOS = {MODELO_REFERENCIA: "fasttext_baseline"}


def prefijo_de(nombre: str) -> Path:
    """Devuelve el prefijo de archivos de un modelo.

    Args:
        nombre: nombre logico del modelo.

    Returns:
        Ruta del prefijo (sin extension).
    """
    return EMB_DIR / PREFIJOS.get(nombre, nombre.lower().replace("-", "_"))


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5-P5] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Los modulos de las fases anteriores se conservan intactos como evidencia
    historica, asi que se importan por ruta en vez de refactorizarlos.

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
    """Ejecuta P5: entrenamiento web/mixto y revision de la fusion."""
    inicio = time.time()
    log("P5: embedding sobre corpus web y mixto, y revision de la fusion")
    modulo_p3 = _cargar_modulo(
        "embedding_03_p3", PROJECT_ROOT / "experiments" / "embedding_03_fusion_complementariedad_v5.py"
    )

    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)
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
    log(f"   desarrollo {len(desarrollo)} filas | {len(tipos_consulta)} tipos a consultar")

    # --- 1. entrenamiento ---
    log("1/5 entrenamiento desde cero")
    registro: dict[str, Any] = {}
    EMB_DIR.mkdir(parents=True, exist_ok=True)
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
            log(f"   {nombre}: {segundos:.0f} s")
        else:
            log(f"   {nombre}: ya existe, se reutiliza")
        prefijo = EMB_DIR / nombre.lower().replace("-", "_")
        vectores = vectores_de_tipos(
            FASTTEXT_BIN, Path(f"{prefijo}.bin"), tipos_consulta, EMB_DIR / f"{nombre.lower()}_sifts.txt"
        )
        with Path(f"{prefijo}.vec").open(encoding="utf-8") as archivo:
            vocabulario, dimension = archivo.readline().split()
        registro[nombre] = {
            "etiqueta": configuracion["etiqueta"],
            "corpus": str(configuracion["corpus"].relative_to(PROJECT_ROOT)),
            "vocab_size": int(vocabulario),
            "dim": int(dimension),
            "embedding_hash_sha256": sha256_de(Path(f"{prefijo}.vec")),
            "segundos": round(segundos, 1),
            "tipos_con_vector": len(vectores),
            "tipos_consultados": len(tipos_consulta),
            "prefijo": str(prefijo.relative_to(PROJECT_ROOT)),
        }
        log(
            f"   {nombre}: vocab {vocabulario} | tipos con vector {len(vectores)} de {len(tipos_consulta)}"
        )

    # El modelo de referencia de P2 tiene sus vectores de tipo ya calculados.
    if MODELO_REFERENCIA not in registro:
        prefijo = prefijo_de(MODELO_REFERENCIA)
        vectores = vectores_de_tipos(
            FASTTEXT_BIN,
            Path(f"{prefijo}.bin"),
            tipos_consulta,
            EMB_DIR / "fasttext-baseline_sifts.txt",
        )
        with Path(f"{prefijo}.vec").open(encoding="utf-8") as archivo:
            vocabulario, dimension = archivo.readline().split()
        registro[MODELO_REFERENCIA] = {
            "etiqueta": "FASTTEXT-SG-SUBWORDS-3-6 (modelo de P2, reevaluado aqui)",
            "corpus": str(CORPUS_WIKI.relative_to(PROJECT_ROOT)),
            "vocab_size": int(vocabulario),
            "dim": int(dimension),
            "embedding_hash_sha256": sha256_de(Path(f"{prefijo}.vec")),
            "segundos": 0.0,
            "tipos_con_vector": len(vectores),
            "tipos_consultados": len(tipos_consulta),
            "prefijo": str(prefijo.relative_to(PROJECT_ROOT)),
        }

    # --- 2. vectores de documento y evaluacion ---
    log("2/5 vectores de documento y evaluacion (umbrales congelados de V03)")
    variantes = []
    for nombre in (MODELO_REFERENCIA, "FASTTEXT-WEB-10M", "FASTTEXT-MIXTO-20M", "WORD2VEC-WEB-10M"):
        prefijo = EMB_DIR / nombre.lower().replace("-", "_")
        sifts = EMB_DIR / f"{nombre.lower()}_sifts.txt"
        if not sifts.exists():
            sifts = EMB_DIR / "fasttext-baseline_sifts.txt"
        vectores = modulo_p3.cargar_sifts(sifts)
        embeddings = modulo_p3.vectores_documento(
            [normalizar_emb_normal(texto) for texto in desarrollo["text"]], vectores
        )
        fusion = modulo_p3.evaluar_variante(
            f"FUSION-{nombre}", desarrollo["text"], y, pliegues, embeddings, modo_embedding="l2"
        )
        solo = modulo_p3.evaluar_variante(
            f"EMB-{nombre}", desarrollo["text"], y, pliegues, embeddings,
            usar_tfidf=False, modo_embedding="estandar",
        )
        variantes.append(fusion)
        variantes.append(solo)

    # --- 3. comparaciones ---
    log("3/5 comparaciones pareadas, complementariedad y rebanadas")
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
                "corrected_cv_ajustados": variante["corrected_cv_umbrales_ajustados"],
                "f1_HS": round(variante["f1_por_etiqueta_ajustados"]["HS"], 4),
                "f1_TR": round(variante["f1_por_etiqueta_ajustados"]["TR"], 4),
                "f1_AG": round(variante["f1_por_etiqueta_ajustados"]["AG"], 4),
            }
        )
        log(
            f"   {variante['variante']:32s} {filas[-1]['corrected_cv_frozen_v03']:.4f} "
            f"delta {filas[-1]['delta_vs_v03']:+.4f} IC95 [{filas[-1]['ic95_0']:+.4f}, {filas[-1]['ic95_1']:+.4f}]"
        )

    mascaras = modulo_p3.rebanadas(desarrollo, y)
    metricas_rebanada = modulo_p3.metricas_por_rebanada(
        y,
        {variante["variante"]: variante["pred_frozen"] for variante in variantes}
        | {"V03-CONGELADO": pred_v03},
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

    # --- 4. mecanismo: vecinos de las variantes ortograficas ---
    log("4/5 vecinos de variantes ortograficas por modelo")
    vocabulario_toxic = vocabulario_textos(
        [normalizar_emb_normal(texto) for texto in train["text"].astype(str)]
    )
    variantes_ortograficas = [
        tipo for tipo in ["callate", "cllate", "violacin", "espaa", "imbcil", "arabe", "estn", "mereces"]
        if tipo in vocabulario_toxic
    ]
    matrices = {}
    for nombre in (MODELO_REFERENCIA, "FASTTEXT-WEB-10M", "FASTTEXT-MIXTO-20M"):
        matrices[nombre] = cargar_matriz_vec(Path(f"{prefijo_de(nombre)}.vec"))
    vecinos_tabla: list[dict[str, Any]] = []
    for tipo in variantes_ortograficas:
        fila = {"tipo": tipo, "freq_toxic": vocabulario_toxic[tipo]}
        for nombre in matrices:
            indice, matriz = matrices[nombre]
            if tipo not in indice:
                fila[nombre] = "(sin vector)"
                continue
            centro, centrada = _centrar(matriz)
            consulta = matriz[indice[tipo]] - centro
            consulta = consulta / max(float(np.linalg.norm(consulta)), 1e-9)
            similitudes = centrada @ consulta
            posicion = int(np.argmax(similitudes))
            lista = [None] * len(indice)
            for palabra, indice_palabra in indice.items():
                lista[indice_palabra] = palabra
            fila[nombre] = f"`{lista[posicion]}` ({float(similitudes[posicion]):.3f})"
        vecinos_tabla.append(fila)

    # --- 5. veredicto e informes ---
    log("5/5 veredicto e informes")
    mejor = max(filas, key=lambda fila: fila["corrected_cv_frozen_v03"])
    cumple_nivel = mejor["delta_vs_v03"] >= MEJORA_MINIMA
    cumple_ic = mejor["ic95_0"] > 0
    mejor_fusion = max(
        (fila for fila in filas if fila["variante"].startswith("FUSION")),
        key=lambda fila: fila["corrected_cv_frozen_v03"],
    )
    log(f"   mejor global: {mejor['variante']} = {mejor['corrected_cv_frozen_v03']:.4f} "
        f"(delta {mejor['delta_vs_v03']:+.4f})")
    log(f"   mejor fusion: {mejor_fusion['variante']} = {mejor_fusion['corrected_cv_frozen_v03']:.4f} "
        f"(delta {mejor_fusion['delta_vs_v03']:+.4f})")

    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "arquitectura": {**ARQUITECTURA, "subpalabras": list(ARQUITECTURA["subpalabras"])},
                "modelos": registro,
                "comparaciones": filas,
                "rebanadas": metricas_rebanada,
                "vecinos_variantes": vecinos_tabla,
                "referencia_v03": REFERENCIA_V03,
                "cumple_nivel": bool(cumple_nivel),
                "cumple_ic95": bool(cumple_ic),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _escribir_informe(registro, filas, metricas_rebanada, vecinos_tabla, mascaras, mejor, mejor_fusion)
    _figura(filas)
    log(f"P5 completada en {round(time.time() - inicio, 1)} s")


def _centrar(matriz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resta la media de la matriz y renormaliza.

    Args:
        matriz: matriz normalizada de vectores.

    Returns:
        Tupla ``(centro, matriz_centrada)``.
    """
    centro = matriz.mean(axis=0)
    centrada = matriz - centro
    normas = np.linalg.norm(centrada, axis=1, keepdims=True)
    return centro, centrada / np.maximum(normas, 1e-9)


def _figura(filas: list[dict[str, Any]]) -> None:
    """Genera la figura de comparacion de variantes.

    Args:
        filas: comparaciones por variante.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    nombres = [fila["variante"].replace("FUSION-", "F:").replace("EMB-", "S:") for fila in filas]
    valores = [fila["corrected_cv_frozen_v03"] for fila in filas]
    colores = ["#1f77b4" if v >= REFERENCIA_V03 else "#d62728" for v in valores]
    plt.figure(figsize=(9, 4.5))
    plt.bar(range(len(nombres)), valores, color=colores)
    plt.axhline(REFERENCIA_V03, color="black", linestyle="--", label="V03 0.7772")
    plt.xticks(range(len(nombres)), nombres, rotation=25, ha="right", fontsize=8)
    plt.ylabel("corrected_cv (umbrales congelados de V03)")
    plt.title("Fusion con embedding de corpus web, mixto y enciclopedico")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_46_fusion_web_v5.png", dpi=150)
    plt.close()


def _escribir_informe(
    registro: dict[str, Any],
    filas: list[dict[str, Any]],
    metricas_rebanada: list[dict[str, Any]],
    vecinos_tabla: list[dict[str, Any]],
    mascaras: dict[str, np.ndarray],
    mejor: dict[str, Any],
    mejor_fusion: dict[str, Any],
) -> None:
    """Escribe el informe de P5.

    Args:
        registro: metadatos de los modelos.
        filas: comparaciones por variante.
        metricas_rebanada: metricas por rebanada.
        vecinos_tabla: vecinos de las variantes ortograficas.
        mascaras: tamanos de las rebanadas.
        mejor: mejor variante global.
        mejor_fusion: mejor variante de fusion.
    """
    por_rebanada: dict[str, dict[str, float]] = {}
    for fila in metricas_rebanada:
        por_rebanada.setdefault(fila["rebanada"], {})[fila["variante"]] = fila["f1_por_celda"]
    lineas = [
        "# Embedding sobre corpus web y mixto, y revision de la fusion (fase v5, P5)",
        "",
        f"Generado por `experiments/embedding_05_fasttext_web_v5.py` ({EXPERIMENTO}).",
        "",
        "## 1. Pregunta",
        "",
        "P4 subio la cobertura del vocabulario TOXIC de 0.6839 a 0.8413 por tipo con la union",
        "Wikipedia+web. Esta es la prueba de si **esa cobertura se traduce en aportacion**",
        "complementaria: si la mecanica de P3 es correcta, la fusion con el embedding del corpus",
        "web o del mixto debe superar a la fusion con el de Wikipedia.",
        "",
        "## 2. Modelos (entrenados desde cero, misma arquitectura)",
        "",
        "| modelo | corpus | vocabulario | tipos con vector | sha256 del .vec | s |",
        "|---|---|---|---|---|---|",
    ]
    for nombre, datos in registro.items():
        lineas.append(
            f"| {nombre} | `{datos['corpus']}` | {datos['vocab_size']} | "
            f"{datos['tipos_con_vector']} de {datos['tipos_consultados']} | "
            f"`{datos['embedding_hash_sha256'][:16]}` | {datos['segundos']} |"
        )
    lineas += [
        "",
        "## 3. Resultados (umbrales congelados de V03, la metrica primaria)",
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
        f"Mejor variante: **{mejor['variante']}** ({mejor['corrected_cv_frozen_v03']:.4f}, delta "
        f"{mejor['delta_vs_v03']:+.4f}). Mejor fusion: **{mejor_fusion['variante']}** "
        f"({mejor_fusion['corrected_cv_frozen_v03']:.4f}, delta {mejor_fusion['delta_vs_v03']:+.4f}).",
        "",
        "## 4. Complementariedad por rebanada",
        "",
        "| rebanada | filas | V03 | F wiki | F web | F mixto |",
        "|---|---|---|---|---|---|",
    ]
    claves = ["V03-CONGELADO", f"FUSION-{MODELO_REFERENCIA}", "FUSION-FASTTEXT-WEB-10M", "FUSION-FASTTEXT-MIXTO-20M"]
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0:
            continue
        valores = por_rebanada.get(nombre, {})
        celdas = " | ".join(f"{valores.get(clave, float('nan')):.4f}" for clave in claves)
        lineas.append(f"| {nombre} | {int(mascara.sum())} | {celdas} |")
    lineas += [
        "",
        "## 5. Vecinos de las variantes ortograficas",
        "",
        "Sin labels externos no se afirma que un vecino sea el lexema correcto; la tabla se",
        "muestra para comparar el comportamiento entre corpora.",
        "",
        "| variante (freq. TOXIC) | wiki 10M | web 10M | mixto 20M |",
        "|---|---|---|---|",
    ]
    for fila in vecinos_tabla:
        lineas.append(
            f"| `{fila['tipo']}` ({fila['freq_toxic']}) | {fila.get(MODELO_REFERENCIA, '-')} | "
            f"{fila.get('FASTTEXT-WEB-10M', '-')} | {fila.get('FASTTEXT-MIXTO-20M', '-')} |"
        )
    lineas += [
        "",
        "## 6. Decision",
        "",
        "Se aplica el criterio pre-registrado de la seccion 29 sin modificarlo: `adopted` exige",
        f"delta >= +{MEJORA_MINIMA:.3f} **y** IC95 pareado que excluya el 0.",
        "",
        f"- Mejor fusion: delta {mejor_fusion['delta_vs_v03']:+.4f}, IC95 "
        f"[{mejor_fusion['ic95_0']:+.4f}, {mejor_fusion['ic95_1']:+.4f}].",
        "- Ninguna variante genera submission ni se envia al leaderboard.",
        "",
        "## 7. Limites",
        "",
        "1. El corpus web es un tramo del 13,9 % del shard 0 de C4, sin reordenacion interna.",
        "2. La metrica primaria es el `corrected_cv`, que la leccion operativa de v4 demuestra que",
        "   no es un selector fiable: cuatro mejoras locales consecutivas no transfirieron.",
        "3. No hay holdout disponible (agotado 5/5), de modo que la unica evidencia externa posible",
        "   seria el leaderboard, y enviar una sonda se decide de forma expresa.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
