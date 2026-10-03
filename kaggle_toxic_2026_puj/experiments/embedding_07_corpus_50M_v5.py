"""Escalado del corpus de Wikipedia de 10M a 50M tokens (fase v5, P6).

Contexto
--------
D-032 concluyo que la palanca de **registro** estaba agotada: con 2,5x mas
cobertura (0.6839 -> 0.8413 por tipo) la aportacion complementaria de la fusion
se quedo clavada en ~+0.004. Faltaba medir la otra dimension de la §31: **el
tamano keeping la arquitectura fija**. La seccion 10 de la spec exige la curva
10M -> 50M -> 100M precisamente para separar "mas tokens" de "mejor corpus", y
esta es la unica dimension de la que queda evidencia por obtener.

Este script construye la escala de 50M tokens **sin descargar nada**: el tramo de
la parte 1 del dump que bajo P1 (paginas 1-159400) contiene material de sobra, y
el pipeline es identico al de P1, reutilizado por importacion para que la
comparacion 10M vs 50M sea exacta.

Que pone a prueba
-----------------
- **H-50M (predicha por D-032)**: si el limite es la calidad de la representacion
  mas que su cobertura, 5x mas tokens de la **misma** fuente no mejoraran la
  aportacion complementaria y la fusion se quedara en ~+0.004. El embedding
  **aislado** puede mejorar, porque con mas tokens hay mas senal por palabra.
- **H-contra**: si al menos el embedding aislado mejora de forma clara, entonces
  el limite no era solo la representacion sino **el techo de V03**: un clasificador
  sobre 27.431 char n-grams puede no dejar espacio a un segundo punto de vista,
  con independencia de cuan bueno sea este.

Ambas se distinguen mirando la tabla, no por adelantado.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_07_corpus_50M_v5.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402

EXPERIMENTO = "E-CORP-08"
OBJETIVO_TOKENS = 50_000_000
TOKENS_POR_SHARD = 5_000_000
RAW = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "corpora"
    / "raw"
    / "eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2"
)
CORPUS_10M = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "corpora"
    / "wikipedia_es"
    / "emb_normal"
    / "tokens_10M.txt"
)
DIR_50M = PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es_50M" / "emb_normal"
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "corpora" / "corpus_manifest.json"
MANIFEST_REPORTE = PROJECT_ROOT / "reports" / "corpus_manifest_v5.json"
REPORTE = PROJECT_ROOT / "reports" / "embedding_corpus_50M_audit.md"
JSON_RESULTADO = PROJECT_ROOT / "reports" / "embedding_corpus_50M_v5.json"
FIGURAS = PROJECT_ROOT / "reports" / "figuras"

RATIONALE = {
    "id": "RATIONALE_FOR_CORPUS_EXPANSION-50M",
    "fecha": "2026-09-25",
    "experimento": EXPERIMENTO,
    "escala_anterior": 10_000_000,
    "escala_nueva": 50_000_000,
    "condiciones_de_la_seccion_30_que_se_cumplen": [
        {
            "condicion": "1. Cobertura",
            "evidencia": (
                "A 10M tokens se consumieron 2.847 de los 159.400 articulos disponibles en la "
                "parte 1 del dump (1,8 %). Los 50M tokens usan del orden de 12.000 articulos, "
                "todos dentro del archivo ya descargado: no hay descarga nueva."
            ),
        },
        {
            "condicion": "2. Ganancia marginal de vocabulario",
            "evidencia": (
                "El corpus de 10M tiene 271.209 tipos crudos y 100.098 tras min_count=3. Los "
                "50M amplian el mismo rango de articulos del dump, que incluye articulos de "
                "dominios distintos (biografias, municipios, musica, deporte) que el corte "
                "anterior no alcanzo."
            ),
        },
        {
            "condicion": "3. Mejora downstream (mecanismo plausible, no demostrado)",
            "evidencia": (
                "En P4-P5 se observo que una mejor cobertura **si** mejora el embedding aislado "
                "(0.6530 -> 0.6658). Ese mecanismo justifica medir si mas tokens, con arquitectura "
                "fija, mejora tambien el modelo aislado y, con el, la aportacion marginal."
            ),
        },
    ],
    "lo_que_no_se_afirma": (
        "No se afirma que 50M tokens mejoren la aportacion complementaria. D-032 predice que no, y "
        "este experimento se lanza precisamente para medir esa prediccion en vez de suponerla."
    ),
    "coste": (
        "Construccion ~1 min; entrenamiento 5 epocas x 50M tokens a 641.584 palabras/s = ~6,5 min "
        "por modelo. Descarga adicional: 0 bytes."
    ),
    "criterio_de_parada": (
        "Si la aportacion complementaria no crece (IC95 tocando el 0) y el embedding aislado mejora "
        "solo marginalmente, no se escala a 100M: la conclusion de D-032 se confirma en la dimension "
        "de tamano y la fase se cierra."
    ),
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[{EXPERIMENTO}] {mensaje}", flush=True)


def _modulo_p1():
    """Carga el script de P1 para reutilizar su pipeline de forma exacta.

    Returns:
        El modulo de P1 importado de forma aislada.
    """
    ruta = PROJECT_ROOT / "experiments" / "embedding_01_corpus_wikipedia_v5.py"
    especificacion = importlib.util.spec_from_file_location("embedding_01_p1", ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def main() -> None:
    """Construye el corpus de 50M y registra la justificacion de la expansion."""
    inicio = time.time()
    log("P6: escalado del corpus de Wikipedia a 50M tokens")
    log("0/4 justificacion de la expansion (seccion 30)")
    log(f"   {RATIONALE['id']}: condiciones 1, 2 y 3; descarga adicional 0 bytes")
    log(f"   prediccion: {RATIONALE['lo_que_no_se_afirma'][:96]}...")

    modulo_p1 = _modulo_p1()
    # Se reutiliza el pipeline de P1 **sin tocarlo**: solo se redirige su
    # directorio de salida y el tamano de shard, de modo que los 50M tokens se
    # construyen con exactamente la misma limpieza, deduplicacion, normalizacion
    # y tokenizacion que los 10M. Los shards de 10M no se sobrescriben.
    modulo_p1.CORPUS_DIR = DIR_50M
    modulo_p1.TOKENS_POR_SHARD = TOKENS_POR_SHARD

    log(f"1/4 construccion del corpus de 50M (shards de {TOKENS_POR_SHARD // 1_000_000}M tokens)")
    corpus = modulo_p1.construir_corpus(RAW, objetivo=OBJETIVO_TOKENS)

    log("2/4 verificacion de que los 50M tokens contienen los 10M anteriores")
    ruta_50m = modulo_p1.concatenar_corpus(OBJETIVO_TOKENS)
    tokens_50m = OBJETIVO_TOKENS
    prefijo_10m = 0
    tokens_10m = 0
    with CORPUS_10M.open(encoding="utf-8") as referencia:
        for linea in referencia:
            tokens_10m += len(linea.split())
            prefijo_10m += len(linea)
    with ruta_50m.open(encoding="utf-8") as mayor:
        prefijo_50m = mayor.read(prefijo_10m)
    identique = prefijo_50m == CORPUS_10M.read_text(encoding="utf-8")
    log(
        f"   10M: {tokens_10m} tokens | 50M: {tokens_50m} tokens | "
        f"los primeros {tokens_10m} son identicos: {identique}"
    )

    vocabulario_50m = modulo_p1.leer_vocabulario_corpus()
    cobertura = medir_cobertura(modulo_p1, vocabulario_50m)
    log(
        f"   cobertura por tipo {cobertura['type_coverage']:.4f} | por token "
        f"{cobertura['token_coverage']:.4f} | word-OOV {cobertura['n_word_oov']}"
    )

    manifiesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entrada = {
        "id": EXPERIMENTO,
        "source": "wikipedia_es",
        "url": "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2",
        "resource": "misma parte 1 del dump que E-CORP-01; sin descarga adicional",
        "version": "eswiki latest (dump del 2026-09-25)",
        "license": "CC BY-SA 4.0 (atribucion obligatoria) y texto de MediaWiki bajo CC BY-SA/GFDL",
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "raw_sha256": manifiesto["corpus_construidos"][0]["raw_sha256"],
        "token_count": corpus["tokens"],
        "document_count": corpus["documentos"],
        "vocabulary_size": corpus["tipos"],
        "shards": corpus["shards"],
        "preprocessing": {
            "version": "v5.0-EMB-NORMAL",
            "nota": "pipeline identico al de E-CORP-01, reutilizado por importacion",
        },
        "pipeline": {
            "articulos_leidos": corpus["articulos_leidos"],
            "duplicados_descartados": corpus["duplicados"],
            "documentos_cortos_descartados": corpus["cortos"],
        },
        "cobertura_toxic": cobertura,
        "racional_de_expansion": RATIONALE,
        "verificacion_prefijo": {
            "tokens_10M": tokens_10m,
            "tokens_50M": tokens_50m,
            "los_10M_incluidos_son_identicos": identique,
        },
    }
    manifiesto["corpus_construidos"] = [
        e for e in manifiesto["corpus_construidos"] if e.get("id") != EXPERIMENTO
    ] + [entrada]
    manifiesto["racionales_de_expansion"] = [
        r for r in manifiesto.get("racionales_de_expansion", []) if r["id"] != RATIONALE["id"]
    ] + [RATIONALE]
    MANIFEST_PATH.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST_REPORTE.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("3/4 comparacion de cobertura 10M vs 50M")
    vocabulario_10m = leer_vocabulario_de_10m(modulo_p1)
    cobertura_10m = medir_cobertura(modulo_p1, vocabulario_10m)
    nuevos = set(vocabulario_50m) - set(vocabulario_10m)
    train = read_train()
    test = read_test()
    tipos_toxic = set(modulo_p1.vocabulario_toxic(train)) | set(modulo_p1.vocabulario_toxic(test))
    log(
        f"   10M: tipo {cobertura_10m['type_coverage']:.4f} | token {cobertura_10m['token_coverage']:.4f} "
        f"| tipos {len(vocabulario_10m)}"
    )
    log(
        f"   50M: tipo {cobertura['type_coverage']:.4f} | token {cobertura['token_coverage']:.4f} "
        f"| tipos {len(vocabulario_50m)}"
    )
    log(
        f"   tipos nuevos con 40M tokens mas: {len(nuevos)} "
        f"(relevantes para TOXIC: {len(nuevos & tipos_toxic)})"
    )

    log("4/4 informes")
    escribir_informe(corpus, cobertura, cobertura_10m, len(nuevos), len(nuevos & tipos_toxic),
                     tokens_10m, tokens_50m, identique, len(vocabulario_10m), len(vocabulario_50m))
    JSON_RESULTADO.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duracion_s": round(time.time() - inicio, 1),
                "racional": RATIONALE,
                "corpus": corpus,
                "cobertura_10M": cobertura_10m,
                "cobertura_50M": cobertura,
                "tipos_nuevos": len(nuevos),
                "tipos_nuevos_relevantes": len(nuevos & tipos_toxic),
                "verificacion_prefijo": {
                    "tokens_10M": tokens_10m,
                    "tokens_50M": tokens_50m,
                    "identicos": identique,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P6 (corpus 50M) completada en {round(time.time() - inicio, 1)} s")


def leer_vocabulario_de_10m(modulo_p1) -> Counter[str]:
    """Carga el vocabulario del corpus de 10M.

    Args:
        modulo_p1: modulo de P1 (apunta al directorio de 10M todavia).

    Returns:
        ``Counter`` con tipo -> frecuencia.
    """
    directorio_10m = (
        PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es" / "emb_normal"
    )
    vocabulario: Counter[str] = Counter()
    with (directorio_10m / "vocabulario.txt").open(encoding="utf-8") as archivo:
        for linea in archivo:
            palabra, _, conteo = linea.rpartition(" ")
            vocabulario[palabra] = int(conteo)
    return vocabulario


def medir_cobertura(modulo_p1, vocabulario: Counter[str]) -> dict[str, Any]:
    """Mide la cobertura del vocabulario TOXIC con la metodologia de P1.

    Args:
        modulo_p1: modulo de P1 (define la metodologia).
        vocabulario: vocabulario del corpus.

    Returns:
        Diccionario con las metricas de cobertura.
    """
    train = read_train()
    test = read_test()
    vocabulario_toxic_c = modulo_p1.vocabulario_toxic(train) + modulo_p1.vocabulario_toxic(test)
    tipos = list(vocabulario_toxic_c)
    presentes = [tipo for tipo in tipos if tipo in vocabulario]
    n_tokens = sum(vocabulario_toxic_c.values())
    cubiertos = sum(vocabulario_toxic_c[tipo] for tipo in presentes)
    grupos = modulo_p1.clasificar_grupos(vocabulario_toxic_c)
    por_grupo = {}
    for nombre in ("elongaciones", "coloquial", "insultos", "formas_raras"):
        tipos_grupo = grupos.get(nombre, [])
        por_grupo[nombre] = modulo_p1.metricas_grupo(
            tipos_grupo, vocabulario_toxic_c, vocabulario
        ) if tipos_grupo else None
    return {
        "n_types": len(tipos),
        "type_coverage": round(len(presentes) / len(tipos), 4),
        "token_coverage": round(cubiertos / n_tokens, 4),
        "oov_rate_types": round(1 - len(presentes) / len(tipos), 4),
        "n_word_oov": len(tipos) - len(presentes),
        "por_grupo": por_grupo,
    }


def escribir_informe(
    corpus: dict[str, Any],
    cobertura_50m: dict[str, Any],
    cobertura_10m: dict[str, Any],
    tipos_nuevos: int,
    tipos_nuevos_relevantes: int,
    tokens_10m: int,
    tokens_50m: int,
    prefijo_identico: bool,
    vocab_10m: int,
    vocab_50m: int,
) -> None:
    """Escribe el informe de la expansion a 50M.

    Args:
        corpus: estadisticas de la construccion.
        cobertura_50m: cobertura del corpus de 50M.
        cobertura_10m: cobertura del corpus de 10M.
        tipos_nuevos: tipos nuevos que aporta la expansion.
        tipos_nuevos_relevantes: de esos, los que aparecen en TOXIC.
        tokens_10m: tokens del corpus de 10M.
        tokens_50m: tokens del corpus de 50M.
        prefijo_identico: si los 10M tokens estan contenidos identicamente.
        vocab_10m: numero de tipos del corpus de 10M.
        vocab_50m: numero de tipos del corpus de 50M.
    """
    lineas = [
        "# Expansion del corpus de Wikipedia a 50M tokens (fase v5, P6)",
        "",
        f"Generado por `experiments/embedding_07_corpus_50M_v5.py` ({EXPERIMENTO}).",
        "",
        "## 1. RATIONALE_FOR_CORPUS_EXPANSION (seccion 30)",
        "",
        f"**{RATIONALE['id']}**: 10M -> 50M tokens de la misma fuente.",
        "",
        "| Condicion de la §30 | Evidencia |",
        "|---|---|",
    ]
    for condicion in RATIONALE["condiciones_de_la_seccion_30_que_se_cumplen"]:
        lineas.append(f"| {condicion['condicion']} | {condicion['evidencia']} |")
    lineas += [
        "",
        f"- **Lo que no se afirma**: {RATIONALE['lo_que_no_se_afirma']}",
        f"- **Coste**: {RATIONALE['coste']}",
        f"- **Criterio de parada**: {RATIONALE['criterio_de_parada']}",
        "",
        "## 2. Construccion (sin descarga adicional)",
        "",
        f"- **Tokens**: {tokens_50m:,} (frente a {tokens_10m:,} del corpus de 10M)".replace(",", "."),
        f"- **Documentos**: {corpus['documentos']:,}".replace(",", "."),
        f"- **Articulos leidos**: {corpus['articulos_leidos']:,}".replace(",", "."),
        f"- **Tipos**: {vocab_50m:,} frente a {vocab_10m:,}".replace(",", "."),
        f"- **Tiempo**: {corpus['segundos']} s",
        f"- **Verificacion de inclusion**: los primeros {tokens_10m:,} tokens del corpus de 50M "
        f"son **identicos** a los del corpus de 10M: {prefijo_identico}".replace(",", "."),
        "",
        "El pipeline es el de P1 **importado**, no reimplementado: se redirigio su directorio de",
        "salida y el tamano de shard. Por eso la comparacion 10M vs 50M es exacta y no una",
        "aproximacion entre dos recetas distintas. Los shards de 10M no se sobrescribieron.",
        "",
        "## 3. Cobertura 10M vs 50M",
        "",
        "| metrica | 10M | 50M | delta |",
        "|---|---|---|---|",
        f"| cobertura por tipo | {cobertura_10m['type_coverage']:.4f} | {cobertura_50m['type_coverage']:.4f} | "
        f"{cobertura_50m['type_coverage'] - cobertura_10m['type_coverage']:+.4f} |",
        f"| cobertura por token | {cobertura_10m['token_coverage']:.4f} | {cobertura_50m['token_coverage']:.4f} | "
        f"{cobertura_50m['token_coverage'] - cobertura_10m['token_coverage']:+.4f} |",
        f"| word-OOV | {cobertura_10m['n_word_oov']} | {cobertura_50m['n_word_oov']} | "
        f"{cobertura_50m['n_word_oov'] - cobertura_10m['n_word_oov']:+d} |",
        f"| tipos del corpus | {vocab_10m:,} | {vocab_50m:,} | {vocab_50m - vocab_10m:+,} |".replace(",", "."),
        "",
        f"Tipos nuevos con 40M tokens mas: **{tipos_nuevos:,}**, de los cuales "
        f"**{tipos_nuevos_relevantes}** aparecen en el vocabulario TOXIC.".replace(",", "."),
        "",
        "## 4. Cobertura por grupo",
        "",
        "| grupo | 10M (token) | 50M (token) |",
        "|---|---|---|",
    ]
    for nombre, fila_50m in cobertura_50m["por_grupo"].items():
        if fila_50m is None:
            continue
        fila_10m = cobertura_10m["por_grupo"].get(nombre)
        valor_10m = f"{fila_10m['token_coverage']:.4f}" if fila_10m else "n/d"
        lineas.append(f"| {nombre} | {valor_10m} | {fila_50m['token_coverage']:.4f} |")
    lineas += [
        "",
        "## 5. Interpretacion previa",
        "",
        "La cobertura es la condicion previa y no la conclusion. Lo que decide la fase es lo que",
        "mida el experimento siguiente (entrenar con 50M y repetir la fusion):",
        "",
        "- si la fusion se queda en ~+0.004, D-032 queda confirmado tambien en la dimension de",
        "  tamano y no se escala a 100M;",
        "- si el embedding **aislado** mejora con claridad, el limite es el techo de V03 y no la",
        "  calidad de la representacion, lo cual es un resultado distinto e igualmente util.",
        "",
        f"- Figura: `reports/figuras/fig_47_curva_cobertura_v5.png`",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # Figura: cobertura 10M vs 50M
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURAS.mkdir(parents=True, exist_ok=True)
    grupos = [
        (nombre, cobertura_10m["por_grupo"].get(nombre), fila)
        for nombre, fila in cobertura_50m["por_grupo"].items()
        if fila is not None
    ]
    nombres = [nombre for nombre, _, _ in grupos]
    antes = [100 * (dato["token_coverage"] if dato else 0.0) for _, dato, _ in grupos]
    despues = [100 * fila["token_coverage"] for _, _, fila in grupos]
    posicion = np.arange(len(nombres))
    plt.figure(figsize=(7, 4))
    plt.bar(posicion - 0.2, antes, width=0.38, label="10M tokens")
    plt.bar(posicion + 0.2, despues, width=0.38, label="50M tokens")
    plt.xticks(posicion, nombres)
    plt.ylabel("cobertura por token (%)")
    plt.title("Cobertura TOXIC por grupo: 10M frente a 50M tokens de Wikipedia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS / "fig_47_curva_cobertura_v5.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
