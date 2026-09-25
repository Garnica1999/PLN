"""Escalado **hacia abajo**: corpus de 5M tokens (fase v5, P9).

Contexto
--------
La curva de aprendizaje de la §10 se midio hasta ahora hacia arriba (10M -> 50M,
D-033) y resulto plana en aportacion complementaria. Falta el tramo inferior: con
5M tokens se puede comprobar si la curva es **monotona** y donde esta el punto en
que la representacion empieza a degradarse.

Que mide, mas alla de "un numero mas":
- Si el embedding **aislado** mejora de forma monotona con los tokens
  (5M < 10M < 50M), la calidad de la representacion depende del tamaño de forma
  suave y la conclusion de D-033 se refuerza.
- Si el embedding aislado **se degrada** ya en 5M, existe un suelo de calidad por
  debajo del cual la representacion pierde information util, y ese suelo es un
  dato operativo: cualquier corpus de menos de ~10M tokens es descarta por
  principio y no por medida.
- Si la fusion se mueve al bajar a 5M, seria la primera vez que el corpus cambia la
  aportacion, en sentido contrario al esperado.

Nota de gobernanza: la §30 exige `RATIONALE_FOR_CORPUS_EXPANSION` para **expandir**.
Bajar de 10M a 5M no es una expansion, asi que no le aplica; lo que se registra aqui
es el motivo de la reduccion (completar la curva de aprendizaje de la §10 por debajo
del punto de partida) y el hecho de que **no** hay descarga adicional ni coste
irreversible: los 5M tokens son un subconjunto exacto de los 10M.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_09_corpus_5M_v5.py
"""

from __future__ import annotations

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

EXPERIMENTO = "E-CORP-09"
OBJETIVO_TOKENS = 5_000_000
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
DIR_5M = PROJECT_ROOT / "data" / "external" / "corpora" / "wikipedia_es_5M" / "emb_normal"
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "corpora" / "corpus_manifest.json"
MANIFEST_REPORTE = PROJECT_ROOT / "reports" / "corpus_manifest_v5.json"
REPORTE = PROJECT_ROOT / "reports" / "embedding_corpus_5M_audit.md"
JSON_RESULTADO = PROJECT_ROOT / "reports" / "embedding_corpus_5M_v5.json"

MOTIVO = {
    "id": "REDUCCION_A_5M",
    "fecha": "2026-09-25",
    "motivo": (
        "Completar la curva de aprendizaje de la seccion 10 por debajo del punto de "
        "partida de 10M, a peticion del usuario, para comprobar si la curva es monotona y "
        "si existe un suelo de calidad de la representacion por debajo de 10M tokens."
    ),
    "gobernanza": (
        "La seccion 30 exige RATIONALE_FOR_CORPUS_EXPANSION para expandir. Una reduccion no "
        "es una expansion, de modo que no le aplica; se registra el motivo. Ademas la "
        "reduccion es un subconjunto exacto de un corpus ya construido: no hay descarga "
        "adicional, ni coste irreversible, ni riesgo de perder un corpus existente."
    ),
    "coste": "Construccion ~10 s; entrenamiento ~1 min por modelo con 5 epocas.",
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


def leer_vocabulario(directorio: Path) -> Counter[str]:
    """Carga un ``vocabulario.txt`` de cualquiera de las escalas.

    Args:
        directorio: directorio con ``vocabulario.txt``.

    Returns:
        ``Counter`` con tipo -> frecuencia.
    """
    vocabulario: Counter[str] = Counter()
    with (directorio / "vocabulario.txt").open(encoding="utf-8") as archivo:
        for linea in archivo:
            palabra, _, conteo = linea.rpartition(" ")
            if palabra:
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
        por_grupo[nombre] = (
            modulo_p1.metricas_grupo(tipos_grupo, vocabulario_toxic_c, vocabulario)
            if tipos_grupo
            else None
        )
    return {
        "n_types": len(tipos),
        "type_coverage": round(len(presentes) / len(tipos), 4),
        "token_coverage": round(cubiertos / n_tokens, 4),
        "n_word_oov": len(tipos) - len(presentes),
        "por_grupo": por_grupo,
    }


def main() -> None:
    """Construye el corpus de 5M y mide su cobertura."""
    inicio = time.time()
    log("P9: corpus de 5M tokens (reduccion de la curva de aprendizaje)")
    log(f"   motivo: {MOTIVO['id']} - sin descarga adicional, subconjunto exacto de 10M")

    modulo_p1 = _modulo_p1()
    modulo_p1.CORPUS_DIR = DIR_5M
    modulo_p1.TOKENS_POR_SHARD = TOKENS_POR_SHARD

    log(f"1/3 construccion del corpus de 5M")
    corpus = modulo_p1.construir_corpus(RAW, objetivo=OBJETIVO_TOKENS)
    ruta_5m = modulo_p1.concatenar_corpus(OBJETIVO_TOKENS)

    log("2/3 verificacion de que los 5M tokens son prefijo exacto de los 10M")
    tokens_5m = 0
    longitud_5m = 0
    with ruta_5m.open(encoding="utf-8") as archivo:
        texto_5m = archivo.read()
    for linea in texto_5m.splitlines(keepends=True):
        tokens_5m += len(linea.split())
        longitud_5m += len(linea)
    texto_10m = CORPUS_10M.read_text(encoding="utf-8")
    prefijo_identico = texto_10m[:longitud_5m] == texto_5m
    log(
        f"   5M: {tokens_5m} tokens | es prefijo exacto de los 10M: {prefijo_identico}"
    )

    vocabulario_5m = leer_vocabulario(DIR_5M)
    directorio_10m = CORPUS_10M.parent
    vocabulario_10m = leer_vocabulario(directorio_10m)
    cobertura_5m = medir_cobertura(modulo_p1, vocabulario_5m)
    cobertura_10m = medir_cobertura(modulo_p1, vocabulario_10m)
    # Tipos que aporta pasar de 5M a 10M: conjunto de 10M menos el de 5M.
    nuevos = set(vocabulario_10m) - set(vocabulario_5m)
    train = read_train()
    test = read_test()
    tipos_toxic = set(modulo_p1.vocabulario_toxic(train)) | set(modulo_p1.vocabulario_toxic(test))
    log(
        f"   5M: cobertura tipo {cobertura_5m['type_coverage']:.4f} | token "
        f"{cobertura_5m['token_coverage']:.4f} | tipos {len(vocabulario_5m)}"
    )
    log(
        f"   10M: cobertura tipo {cobertura_10m['type_coverage']:.4f} | token "
        f"{cobertura_10m['token_coverage']:.4f} | tipos {len(vocabulario_10m)}"
    )
    log(
        f"   tipos nuevos que aporta el salto de 5M a 10M: {len(nuevos)} "
        f"(relevantes para TOXIC: {len(nuevos & tipos_toxic)})"
    )

    manifiesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entrada = {
        "id": EXPERIMENTO,
        "source": "wikipedia_es",
        "url": "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2",
        "resource": "mismo tramo del dump que E-CORP-01; sin descarga adicional",
        "version": "eswiki latest (dump del 2026-09-25)",
        "license": "CC BY-SA 4.0 y texto de MediaWiki bajo CC BY-SA/GFDL",
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "raw_sha256": manifiesto["corpus_construidos"][0]["raw_sha256"],
        "token_count": tokens_5m,
        "document_count": corpus["documentos"],
        "vocabulary_size": len(vocabulario_5m),
        "shards": corpus["shards"],
        "preprocessing": {
            "version": "v5.0-EMB-NORMAL",
            "nota": "pipeline identico al de E-CORP-01 y E-CORP-08, importado sin reimplementar",
        },
        "pipeline": {
            "articulos_leidos": corpus["articulos_leidos"],
            "duplicados_descartados": corpus["duplicados"],
            "documentos_cortos_descartados": corpus["cortos"],
        },
        "cobertura_toxic": cobertura_5m,
        "reduccion": MOTIVO,
        "verificacion_prefijo": {
            "tokens_5M": tokens_5m,
            "es_prefijo_exacto_de_10M": prefijo_identico,
        },
    }
    manifiesto["corpus_construidos"] = [
        e for e in manifiesto["corpus_construidos"] if e.get("id") != EXPERIMENTO
    ] + [entrada]
    manifiesto["reducciones_de_corpus"] = [
        r for r in manifiesto.get("reducciones_de_corpus", []) if r["id"] != MOTIVO["id"]
    ] + [MOTIVO]
    MANIFEST_PATH.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST_REPORTE.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("3/3 informe")
    REPORTE.write_text(
        "\n".join(
            [
                "# Corpus de 5M tokens: tramo inferior de la curva de aprendizaje (fase v5, P9)",
                "",
                f"Generado por `experiments/embedding_09_corpus_5M_v5.py` ({EXPERIMENTO}).",
                "",
                "## 1. Motivo de la reduccion",
                "",
                f"- {MOTIVO['motivo']}",
                f"- **Gobernanza**: {MOTIVO['gobernanza']}",
                f"- **Coste**: {MOTIVO['coste']}",
                "",
                "## 2. Construccion",
                "",
                f"- Tokens: {tokens_5m:,}".replace(",", ".") + f" | documentos: {corpus['documentos']:,}".replace(",", ".")
                + f" | tipos: {len(vocabulario_5m):,}".replace(",", "."),
                f"- **Verificacion**: los 5M tokens son un **prefijo exacto** de los 10M: "
                f"{prefijo_identico}. Por eso la comparacion 5M vs 10M no puede atribuirse a "
                "una diferencia de receta, solo de cantidad de datos.",
                f"- Tiempo de construccion: {corpus['segundos']} s",
                "",
                "## 3. Cobertura 5M vs 10M",
                "",
                "| metrica | 5M | 10M | delta |",
                "|---|---|---|---|",
                f"| cobertura por tipo | {cobertura_5m['type_coverage']:.4f} | {cobertura_10m['type_coverage']:.4f} | "
                f"{cobertura_10m['type_coverage'] - cobertura_5m['type_coverage']:+.4f} |",
                f"| cobertura por token | {cobertura_5m['token_coverage']:.4f} | {cobertura_10m['token_coverage']:.4f} | "
                f"{cobertura_10m['token_coverage'] - cobertura_5m['token_coverage']:+.4f} |",
                f"| word-OOV | {cobertura_5m['n_word_oov']} | {cobertura_10m['n_word_oov']} | "
                f"{cobertura_10m['n_word_oov'] - cobertura_5m['n_word_oov']:+d} |",
                f"| tipos del corpus | {len(vocabulario_5m):,} | {len(vocabulario_10m):,} | "
                f"{len(vocabulario_10m) - len(vocabulario_5m):+,}".replace(",", "."),
                "",
                f"Tipos **nuevos** que aporta el salto de 5M a 10M: **{len(nuevos):,}**, de los cuales "
                f"**{len(nuevos & tipos_toxic)}** aparecen en el vocabulario TOXIC.".replace(",", "."),
                "",
                "## 4. Cobertura por grupo",
                "",
                "| grupo | 5M (token) | 10M (token) |",
                "|---|---|---|",
            ]
            + [
                f"| {nombre} | {cobertura_5m['por_grupo'][nombre]['token_coverage']:.4f} | "
                f"{cobertura_10m['por_grupo'][nombre]['token_coverage']:.4f} |"
                for nombre in cobertura_5m["por_grupo"]
                if cobertura_5m["por_grupo"].get(nombre) and cobertura_10m["por_grupo"].get(nombre)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    JSON_RESULTADO.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duracion_s": round(time.time() - inicio, 1),
                "motivo": MOTIVO,
                "corpus": corpus,
                "tokens": tokens_5m,
                "cobertura_5M": cobertura_5m,
                "cobertura_10M": cobertura_10m,
                "tipos_nuevos_que_aporta_el_salto_10M": len(nuevos),
                "tipos_nuevos_relevantes": len(nuevos & tipos_toxic),
                "verificacion_prefijo": {
                    "es_prefijo_exacto": prefijo_identico,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"P9 completada en {round(time.time() - inicio, 1)} s")


if __name__ == "__main__":
    main()
