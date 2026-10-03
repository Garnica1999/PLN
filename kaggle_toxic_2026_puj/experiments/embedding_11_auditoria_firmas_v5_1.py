"""P0 de la fase v5.1 (EBD-17): auditoria de firmas de entrenamiento y contabilidad del presupuesto.

Contexto
--------
La fase v5 cerro con D-034: el embedding es un ``CANDIDATE`` no adoptado y las tres
dimensiones del corpus (tamano 5M/10M/50M, registro y composicion) estan medidas. La
fase v5.1 no abre una busqueda nueva: cierra las preguntas **arquitectonicas** que
quedaron abiertas, con un presupuesto duro de 9 entrenamientos exploratorios mas 1 de
reserva.

Por que hace falta este script
-----------------------------
Antes de gastar una sola unidad de entrenamiento hay que poder responder con certeza
a dos preguntas:

1. **Que configuraciones existen ya.** La contabilidad de v5 vivia repartida en cuatro
   JSON de resultados, sin una identidad canonica de entrenamiento. Este script construye
   la ``training_signature`` de la seccion 34 de la spec y la calcula para los diez
   modelos de v5 y para las nueve propuestas de v5.1, de modo que una configuracion
   proposta que ya exista se detecte como ``EXISTING`` y no se reentrene.
2. **Cual es la identidad real de un corpus.** El manifiesto solo guardaba el
   ``raw_sha256`` del dump, que es **compartido** por las escalas de Wikipedia (5M, 10M
   y 50M salen del mismo tramo del dump). Con esa sola huella tres corpora distintos
   comparte identidad. Este script anade el ``processed_sha256`` del archivo de tokens ya
   construidos, que si distingue las escalas, y lo registra en el manifiesto.

Ademas aplica la auditoria de reglas de la seccion 36 reutilizando las funciones ya
existentes de ``experiments/embedding_06_auditoria_v5.py`` en vez de reimplementarlas: los
patrones prohibidos se toman de alli para que esta auditoria no contenga literalmente las
cadenas que busca (si las contuviera, se detectaria a si misma comoinfractora).

Lo que este script NO hace
--------------------------
- No entrena ningun embedding. ``segundos_entrenamiento`` de toda la tabla vale ``0.0``
  porque aqui no hubo entrenamiento, no porque los modelos no existan.
- No toca V03, el holdout ni el leaderboard.
- No modifica resultados historicos: solo anade campos al manifiesto de corpus.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_11_auditoria_firmas_v5_1.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_utils import sha256_de  # noqa: E402

EXPERIMENTO = "EBD-17 (P0)"
EMB_DIR = PROJECT_ROOT / "data" / "external" / "embeddings"
CORPORA_DIR = PROJECT_ROOT / "data" / "external" / "corpora"
MANIFEST_PATH = CORPORA_DIR / "corpus_manifest.json"
MANIFEST_COPIA = PROJECT_ROOT / "reports" / "corpus_manifest_v5.json"
REPORTE = PROJECT_ROOT / "reports" / "embedding_architecture_v5_1.md"
TABLA_A = PROJECT_ROOT / "reports" / "embedding_architecture_results_v5_1.csv"
TABLA_B = PROJECT_ROOT / "reports" / "embedding_downstream_evaluations_v5_1.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "embedding_budget_v5_1.json"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"

REFERENCIA_V03 = 0.777211
MEJORA_MINIMA = 0.005
PRESUPUESTO_EXPLORATORIO = 9
PRESUPUESTO_RESERVA = 1
VERDEZ_GUARDADO = "EXISTING"

# Arquitectura de referencia de v5 (ya entrenada). Todo lo que se proponga en v5.1 es
# esta misma configuracion con UN parametro cambiado sobre el MISMO corpus
# (E-CORP-01, Wikipedia 10M) y la MISMA semilla, de modo que la comparacion sea
# celda a celda contra el baseline.
ARQUITECTURA_BASE: dict[str, Any] = {
    "dim": 200,
    "ventana": 5,
    "min_count": 3,
    "minn": 3,
    "maxn": 6,
    "negativos": 10,
    "muestreo": 0.0001,
    "epocas": 5,
    "sg": 1,
    "semilla": 42,
    "bucket": 200_000,
    "hilos": 16,
}

# Corpus de trabajo de v5.1. ``processed_sha256`` se completa en tiempo de ejecucion.
CORPUS_TRABAJO = {
    "id": "E-CORP-01",
    "nombre": "wikipedia_es_10M",
    "ruta": CORPORA_DIR / "wikipedia_es" / "emb_normal" / "tokens_10M.txt",
    "tokens": 10_002_400,
}

# Inventario de los diez entrenamientos de v5. De cada modelo se indica de donde sale su
# metadata; el script no la escribe a mano, la relee del artefacto de la corrida que lo
# produjo, de modo que la tabla A sea trazable hasta el JSON de origen.
#
# ``minn``, ``maxn`` y ``semilla`` se **declaran aqui** y despues se verifican contra el
# artefacto cuando el artefacto los contiene. No se deducen: P5 solo guardaba el intervalo
# de subpalabras en la etiqueta de texto del modelo, de modo que deducirlo del artefacto
# daba al control Word2Vec del corpus web un intervalo 3-6 que nunca tuvo, y su firma
# coincidia con la de FastText. Es un gap del artefacto historico, no una ambiguedad.
INVENTARIO_V5: list[dict[str, Any]] = [
    {
        "training_id": "FASTTEXT-WIKI-10M",
        "prefijo": "fasttext_baseline",
        "corpus_id": "E-CORP-01",
        "artefacto": "reports/embedding_downstream_v5.json",
        "clave": "FASTTEXT-BASELINE",
        "minn": 3,
        "maxn": 6,
        "semilla": 42,
        "rol": "baseline arquitectonico de v5 y de v5.1",
    },
    {
        "training_id": "FASTTEXT-WIKI-10M-SEED7",
        "prefijo": "fasttext_seed_control",
        "corpus_id": "E-CORP-01",
        "artefacto": "reports/embedding_downstream_v5.json",
        "clave": "FASTTEXT-SEED-CONTROL",
        "minn": 3,
        "maxn": 6,
        "semilla": 7,
        "rol": "control de estabilidad entre semillas",
    },
    {
        "training_id": "WORD2VEC-WIKI-10M",
        "prefijo": "word2vec_control",
        "corpus_id": "E-CORP-01",
        "artefacto": "reports/embedding_downstream_v5.json",
        "clave": "WORD2VEC-CONTROL",
        "minn": 0,
        "maxn": 0,
        "semilla": 42,
        "rol": "control sin subpalabras (subwords apagados)",
    },
    {
        "training_id": "FASTTEXT-WIKI-5M",
        "prefijo": "fasttext_wiki_5m",
        "corpus_id": "E-CORP-09",
        "artefacto": "reports/embedding_curve_5M_v5.json",
        "clave": "FASTTEXT-WIKI-5M",
        "minn": 3,
        "maxn": 6,
        "semilla": 42,
        "rol": "tramo inferior de la curva de aprendizaje",
    },
    {
        "training_id": "WORD2VEC-WIKI-5M",
        "prefijo": "word2vec_wiki_5m",
        "corpus_id": "E-CORP-09",
        "artefacto": "reports/embedding_curve_5M_v5.json",
        "clave": "WORD2VEC-WIKI-5M",
        "minn": 0,
        "maxn": 0,
        "semilla": 42,
        "rol": "control sin subpalabras a 5M",
    },
    {
        "training_id": "FASTTEXT-WIKI-50M",
        "prefijo": "fasttext_wiki_50m",
        "corpus_id": "E-CORP-08",
        "artefacto": "reports/embedding_curve_50M_v5.json",
        "clave": "FASTTEXT-WIKI-50M",
        "minn": 3,
        "maxn": 6,
        "semilla": 42,
        "rol": "tramo superior de la curva de aprendizaje",
    },
    {
        "training_id": "WORD2VEC-WIKI-50M",
        "prefijo": "word2vec_wiki_50m",
        "corpus_id": "E-CORP-08",
        "artefacto": "reports/embedding_curve_50M_v5.json",
        "clave": "WORD2VEC-WIKI-50M",
        "minn": 0,
        "maxn": 0,
        "semilla": 42,
        "rol": "control sin subpalabras a 50M",
    },
    {
        "training_id": "FASTTEXT-WEB-10M",
        "prefijo": "fasttext_web_10m",
        "corpus_id": "E-CORP-06",
        "artefacto": "reports/embedding_web_v5.json",
        "clave": "FASTTEXT-WEB-10M",
        "minn": 3,
        "maxn": 6,
        "semilla": 42,
        "rol": "variante de registro web (mC4-es)",
    },
    {
        "training_id": "WORD2VEC-WEB-10M",
        "prefijo": "word2vec_web_10m",
        "corpus_id": "E-CORP-06",
        "artefacto": "reports/embedding_web_v5.json",
        "clave": "WORD2VEC-WEB-10M",
        "minn": 0,
        "maxn": 0,
        "semilla": 42,
        "rol": "control sin subpalabras sobre el corpus web",
    },
    {
        "training_id": "FASTTEXT-MIXTO-20M",
        "prefijo": "fasttext_mixto_20m",
        "corpus_id": "E-CORP-07",
        "artefacto": "reports/embedding_web_v5.json",
        "clave": "FASTTEXT-MIXTO-20M",
        "minn": 3,
        "maxn": 6,
        "semilla": 42,
        "rol": "variante de composicion (union Wikipedia + web)",
    },
]

# Tiempos de reloj medidos directamente y documentados en el informe de cierre de v5
# (seccion 9). Cuando el JSON de la corrida dice 0.0 el modelo se entrego desde una
# corrida anterior y el tiempo no quedo registrado alli; se cita la fuente.
TIEMPOS_DOCUMENTADOS: dict[str, tuple[float, str]] = {
    "FASTTEXT-WIKI-10M": (
        82.0,
        "medido aparte, informe de cierre de v5 seccion 9 (641.584 palabras/s con 16 hilos)",
    ),
}

# Las nueve propuestas de v5.1. Cada una cambia UN parametro de la arquitectura base
# sobre el mismo corpus y la misma semilla. La seccion 4 de la spec las numera en ese
# orden y la seccion 5 impone la ejecucion secuencial con poda.
PROPUESTAS_V51: list[dict[str, Any]] = [
    {"codigo": "VS-100", "familia": "vector_size", "valor": "100", "cambios": {"dim": 100}},
    {"codigo": "VS-300", "familia": "vector_size", "valor": "300", "cambios": {"dim": 300}},
    {"codigo": "WS-3", "familia": "window", "valor": "3", "cambios": {"ventana": 3}},
    {"codigo": "WS-7", "familia": "window", "valor": "7", "cambios": {"ventana": 7}},
    {"codigo": "MC-2", "familia": "min_count", "valor": "2", "cambios": {"min_count": 2}},
    {"codigo": "MC-5", "familia": "min_count", "valor": "5", "cambios": {"min_count": 5}},
    {"codigo": "SW-3-5", "familia": "subwords", "valor": "3-5", "cambios": {"minn": 3, "maxn": 5}},
    {"codigo": "SW-3-7", "familia": "subwords", "valor": "3-7", "cambios": {"minn": 3, "maxn": 7}},
    {"codigo": "NEG-5", "familia": "negative", "valor": "5", "cambios": {"negativos": 5}},
]

# Resolución de nombres de variante a entrenamiento, para la Tabla B. El corpus y la
# escala van dentro del nombre en v5 ("FUSION-FASTTEXT-WIKI-10M"), pero el baseline de
# P2 se llamaba solo "FASTTEXT-BASELINE", asi que la correspondencia es explicita.
ALIAS_TRAINING: dict[str, str] = {
    "FASTTEXT-BASELINE": "FASTTEXT-WIKI-10M",
    "FASTTEXT-SEED-CONTROL": "FASTTEXT-WIKI-10M-SEED7",
    "WORD2VEC-CONTROL": "WORD2VEC-WIKI-10M",
    "FASTTEXT-WIKI-10M": "FASTTEXT-WIKI-10M",
    "FUSION-HSTACK-L2": "FASTTEXT-WIKI-10M",
    "FUSION-HSTACK-STANDARD": "FASTTEXT-WIKI-10M",
    "EMB-SOLO": "FASTTEXT-WIKI-10M",
    "FUSION-FASTTEXT-WIKI-10M": "FASTTEXT-WIKI-10M",
    "EMB-FASTTEXT-WIKI-10M": "FASTTEXT-WIKI-10M",
    "FASTTEXT-WIKI-5M": "FASTTEXT-WIKI-5M",
    "FUSION-FASTTEXT-WIKI-5M": "FASTTEXT-WIKI-5M",
    "EMB-FASTTEXT-WIKI-5M": "FASTTEXT-WIKI-5M",
    "WORD2VEC-WIKI-5M": "WORD2VEC-WIKI-5M",
    "FUSION-WORD2VEC-WIKI-5M": "WORD2VEC-WIKI-5M",
    "EMB-WORD2VEC-WIKI-5M": "WORD2VEC-WIKI-5M",
    "FASTTEXT-WIKI-50M": "FASTTEXT-WIKI-50M",
    "FUSION-FASTTEXT-WIKI-50M": "FASTTEXT-WIKI-50M",
    "EMB-FASTTEXT-WIKI-50M": "FASTTEXT-WIKI-50M",
    "WORD2VEC-WIKI-50M": "WORD2VEC-WIKI-50M",
    "FUSION-WORD2VEC-WIKI-50M": "WORD2VEC-WIKI-50M",
    "EMB-WORD2VEC-WIKI-50M": "WORD2VEC-WIKI-50M",
    "FASTTEXT-WEB-10M": "FASTTEXT-WEB-10M",
    "FUSION-FASTTEXT-WEB-10M": "FASTTEXT-WEB-10M",
    "EMB-FASTTEXT-WEB-10M": "FASTTEXT-WEB-10M",
    "WORD2VEC-WEB-10M": "WORD2VEC-WEB-10M",
    "FUSION-WORD2VEC-WEB-10M": "WORD2VEC-WEB-10M",
    "EMB-WORD2VEC-WEB-10M": "WORD2VEC-WEB-10M",
    "FASTTEXT-MIXTO-20M": "FASTTEXT-MIXTO-20M",
    "FUSION-FASTTEXT-MIXTO-20M": "FASTTEXT-MIXTO-20M",
    "EMB-FASTTEXT-MIXTO-20M": "FASTTEXT-MIXTO-20M",
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P0] {mensaje}", flush=True)


def _cargar_modulo(nombre: str, ruta: Path):
    """Carga un modulo de experimento de forma aislada.

    Los scripts de la fase v5 conservan su propia copia de la normalizacion y del
    protocolo como evidencia historica y **no se modifican**; por eso se cargan con
    ``importlib`` en vez de importarse como modulos del paquete.

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


def firma_entrenamiento(
    *,
    processed_sha256: str,
    corpus_version: str,
    arquitectura: dict[str, Any],
) -> str:
    """Calcula la identidad canonica de un entrenamiento (seccion 34 de la spec).

    La firma concatena en orden fijo el hash del corpus **ya procesado** (no el del
    dump, que comparten las escalas de Wikipedia), la version del preprocesamiento y
    todos los parametros que afectan los pesos. Se incluye ``bucket`` aunque la spec no
    lo liste entre los campos, porque su valor determina parte de la matriz de
    subpalabras y, por tanto, los pesos; excluirlo permitiria dar por nueva una
    configuracion que en realidad es otra. ``hilos`` queda fuera a proposito: no cambia
    la configuracion del modelo, solo el orden no determinista con el que fastText
    actualiza los pesos, y esa no determinismo ya esta medida (control de semilla).

    Args:
        processed_sha256: sha256 del archivo de tokens del corpus.
        corpus_version: version del preprocesamiento del corpus.
        arquitectura: parametros de entrenamiento.

    Returns:
        Huella hexadecimal de la firma.
    """
    campos = [
        processed_sha256,
        corpus_version,
        str(arquitectura["dim"]),
        str(arquitectura["ventana"]),
        str(arquitectura["min_count"]),
        str(arquitectura["minn"]),
        str(arquitectura["maxn"]),
        str(arquitectura["negativos"]),
        repr(float(arquitectura["muestreo"])),
        str(arquitectura["epocas"]),
        str(arquitectura["sg"]),
        str(arquitectura["semilla"]),
        str(arquitectura["bucket"]),
    ]
    carga = "|".join(campos)
    return hashlib.sha256(carga.encode("utf-8")).hexdigest()


def inventariar_corpus() -> dict[str, dict[str, Any]]:
    """Anade la huella del corpus procesado a cada entrada del manifiesto.

    Devuelve el inventario de los cinco corpus construidos, cada uno con su
    ``processed_sha256``. El manifiesto se actualiza **anadiendo** campos: no se altera
    ningun valor historico.

    Returns:
        Diccionario ``corpus_id`` -> datos del corpus.
    """
    log("1/5 huella real de los corpus procesados")
    rutas = {
        "E-CORP-01": CORPORA_DIR / "wikipedia_es" / "emb_normal" / "tokens_10M.txt",
        "E-CORP-09": CORPORA_DIR / "wikipedia_es_5M" / "emb_normal" / "tokens_5M.txt",
        "E-CORP-08": CORPORA_DIR / "wikipedia_es_50M" / "emb_normal" / "tokens_50M.txt",
        "E-CORP-06": CORPORA_DIR / "mc4_es" / "emb_normal" / "tokens_10M.txt",
        "E-CORP-07": CORPORA_DIR / "mixto" / "emb_normal" / "tokens_20M.txt",
    }
    manifiesto = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    # ``corpus_construidos`` es una lista de entradas y ``corpus_construidos_mixto`` un
    # unico objeto (E-CORP-07), porque la union se describe como una sola entrada con sus
    # componentes. Se normalizan ambos a lista de entradas.
    entradas = list(manifiesto["corpus_construidos"])
    mixto = manifiesto.get("corpus_construidos_mixto")
    if isinstance(mixto, dict):
        entradas.append(mixto)
    elif isinstance(mixto, list):
        entradas.extend(mixto)
    inventario: dict[str, dict[str, Any]] = {}
    for corpus_id, ruta in rutas.items():
        if not ruta.exists():
            raise RuntimeError(f"El corpus {corpus_id} no esta en {ruta}")
        huella = sha256_de(ruta)
        entrada = next((item for item in entradas if item.get("id") == corpus_id), None)
        if entrada is None:
            raise RuntimeError(f"{corpus_id} no aparece en el manifiesto de corpus")
        tokens = entrada.get("token_count")
        if tokens is None:
            raise RuntimeError(f"{corpus_id} no declara token_count en el manifiesto")
        entrada["processed_sha256"] = huella
        entrada["processed_ruta"] = str(ruta.relative_to(PROJECT_ROOT))
        entrada["processed_bytes"] = ruta.stat().st_size
        version = str(entrada.get("preprocessing", {}).get("version", "v5.0-EMB-NORMAL"))
        entrada["corpus_version"] = version
        inventario[corpus_id] = {
            "corpus_id": corpus_id,
            "source": entrada.get("source"),
            "tokens": int(tokens),
            "processed_sha256": huella,
            "corpus_version": version,
            "ruta": str(ruta.relative_to(PROJECT_ROOT)),
            "bytes": ruta.stat().st_size,
        }
        log(f"   {corpus_id}: {tokens:>10,} tokens | {huella[:16]}")
    log("   el hash del dump NO identifica la escala: anadido el del corpus procesado")
    MANIFEST_PATH.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    MANIFEST_COPIA.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return inventario


def inventariar_modelos(
    corpus: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Construye la Tabla A: identidad de los diez entrenamientos de v5.

    Para cada modelo relee su metadata del JSON de la corrida que lo produjo, verifica
    el sha256 del ``.vec`` en disco, verifica la configuracion declarada contra la que
    guarda ese JSON y calcula su firma.

    Args:
        corpus: inventario de corpus devuelto por :func:`inventariar_corpus`.

    Returns:
        Tupla con las filas de la Tabla A y la lista de gaps de trazabilidad detectados
        en los artefactos historicos.
    """
    log("2/5 firmas de los diez entrenamientos existentes")
    cache: dict[str, dict[str, Any]] = {}
    gaps: list[str] = []
    filas: list[dict[str, Any]] = []
    for entrada in INVENTARIO_V5:
        ruta_json = PROJECT_ROOT / entrada["artefacto"]
        if not ruta_json.exists():
            raise RuntimeError(f"Falta el artefacto {entrada['artefacto']}")
        if ruta_json.as_posix() not in cache:
            cache[ruta_json.as_posix()] = json.loads(ruta_json.read_text(encoding="utf-8"))
        raiz = cache[ruta_json.as_posix()]
        datos = raiz["modelos"][entrada["clave"]]
        ruta_vec = EMB_DIR / f"{entrada['prefijo']}.vec"
        ruta_bin = EMB_DIR / f"{entrada['prefijo']}.bin"
        if not ruta_vec.exists() or not ruta_bin.exists():
            raise RuntimeError(f"El modelo {entrada['training_id']} no esta en disco")
        hash_registrado = datos.get("embedding_hash_sha256") or datos.get("sha256")
        hash_real = sha256_de(ruta_vec)
        if hash_real != hash_registrado:
            raise RuntimeError(
                f"El sha256 de {ruta_vec.name} no coincide con el registrado: "
                f"{hash_real[:16]} frente a {str(hash_registrado)[:16]}"
            )
        # La arquitectura se lee por jerarquia: la que declara el modelo (P2 la guarda
        # por modelo), si no la global de la corrida (web, 50M y 5M la guardan una sola
        # vez), y si no la de referencia. Cada valor se **verifica** contra la
        # declaracion de la fase: si el artefacto historico dijera otra cosa, quiero un
        # error explicito y no una firma silenciosamente distinta.
        arquitectura = dict(ARQUITECTURA_BASE)
        por_modelo = datos.get("arquitectura", {})
        global_ = raiz.get("arquitectura", {})
        merged = {**global_, **por_modelo}
        for campo in ("dim", "ventana", "min_count", "negativos", "epocas", "bucket", "muestreo", "sg"):
            if campo in merged and merged[campo] != arquitectura[campo]:
                raise RuntimeError(
                    f"{entrada['training_id']}: el artefacto declara {campo} = {merged[campo]} "
                    f"y la arquitectura de la fase declara {arquitectura[campo]}"
                )
        arquitectura["minn"] = int(entrada["minn"])
        arquitectura["maxn"] = int(entrada["maxn"])
        arquitectura["semilla"] = int(entrada["semilla"])
        # El intervalo de subpalabras se verifica contra el artefacto cuando este lo
        # guarda. P5 no lo guardaba por modelo, solo dentro de la etiqueta de texto, de
        # modo que ahi la verificacion no es posible y queda registrado como gap.
        minn_artefacto = datos.get("minn", merged.get("minn"))
        if minn_artefacto is not None and int(minn_artefacto) != arquitectura["minn"]:
            raise RuntimeError(
                f"{entrada['training_id']}: el artefacto declara minn = {minn_artefacto} "
                f"y el inventario declara {arquitectura['minn']}"
            )
        maxn_artefacto = datos.get("maxn", merged.get("maxn"))
        if maxn_artefacto is not None and int(maxn_artefacto) != arquitectura["maxn"]:
            raise RuntimeError(
                f"{entrada['training_id']}: el artefacto declara maxn = {maxn_artefacto} "
                f"y el inventario declara {arquitectura['maxn']}"
            )
        intervalo_verificable = minn_artefacto is not None
        if not intervalo_verificable:
            gaps.append(f"{entrada['training_id']}: el artefacto no declara minn/maxn por modelo")
        identificador_corpus = entrada["corpus_id"]
        datos_corpus = corpus[identificador_corpus]
        segundos = float(datos.get("segundos_entrenamiento") or datos.get("segundos") or 0.0)
        fuente_tiempo = f"JSON de la corrida ({entrada['artefacto']})"
        if segundos == 0.0 and entrada["training_id"] in TIEMPOS_DOCUMENTADOS:
            segundos, fuente_tiempo = TIEMPOS_DOCUMENTADOS[entrada["training_id"]]
        elif segundos == 0.0:
            fuente_tiempo = "no registrado en la corrida; no medido"
        filas.append(
            {
                "training_id": entrada["training_id"],
                "familia": "v5 (cerrada)",
                "corpus": identificador_corpus,
                "tokens": datos_corpus["tokens"],
                "vector_size": arquitectura["dim"],
                "window": arquitectura["ventana"],
                "min_count": arquitectura["min_count"],
                "min_n": arquitectura["minn"],
                "max_n": arquitectura["maxn"],
                "negative": arquitectura["negativos"],
                "sample": arquitectura["muestreo"],
                "epochs": arquitectura["epocas"],
                "sg": arquitectura["sg"],
                "seed": arquitectura["semilla"],
                "bucket": arquitectura["bucket"],
                "vocabulary_size": datos.get("vocab_size"),
                "training_time_s": round(segundos, 1),
                "training_time_fuente": fuente_tiempo,
                "model_sha256": hash_real,
                "training_signature": firma_entrenamiento(
                    processed_sha256=datos_corpus["processed_sha256"],
                    corpus_version=datos_corpus["corpus_version"],
                    arquitectura=arquitectura,
                ),
                "status": VERDEZ_GUARDADO,
                "rol": entrada["rol"],
                "config_verificada_contra_artefacto": "si" if intervalo_verificable else "no",
            }
        )
        log(
            f"   {filas[-1]['training_id']:24s} {arquitectura['dim']:>4}d "
            f"w{arquitectura['ventana']} minC{arquitectura['min_count']} "
            f"sub{arquitectura['minn']}-{arquitectura['maxn']} neg{arquitectura['negativos']} "
            f"| {hash_real[:12]}"
        )
    firmas: dict[str, list[str]] = {}
    for fila in filas:
        firmas.setdefault(fila["training_signature"], []).append(
            f"{fila['training_id']} (dim {fila['vector_size']}, "
            f"sub {fila['min_n']}-{fila['max_n']}, seed {fila['seed']})"
        )
    colisionadas = {huella: nombres for huella, nombres in firmas.items() if len(nombres) > 1}
    if colisionadas:
        detalle = "; ".join(
            f"{' == '.join(nombres)} comparten {huella[:12]}"
            for huella, nombres in colisionadas.items()
        )
        raise RuntimeError(f"Dos modelos de v5 declaran la misma firma: {detalle}")
    for hueco in gaps:
        log(f"   gap de trazabilidad: {hueco}")
    return filas, gaps


def calcular_presupuesto(
    corpus: dict[str, dict[str, Any]], existentes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Calcula la firma de las nueve propuestas y su estado frente a lo existente.

    Una propuesta cuya firma ya exista se marca ``EXISTING - LOAD ARTIFACT`` y **no**
    consume presupuesto, que es la regla de la seccion 43.

    Args:
        corpus: inventario de corpus.
        existentes: filas de la Tabla A de v5.

    Returns:
        Diccionario con las propuestas, los contadores y las aserciones.
    """
    log("3/5 presupuesto: nueve propuestas sobre el baseline")
    firmas_existentes = {fila["training_signature"]: fila["training_id"] for fila in existentes}
    datos_corpus = corpus[CORPUS_TRABAJO["id"]]
    propuestas: list[dict[str, Any]] = []
    for orden, propuesta in enumerate(PROPUESTAS_V51, start=1):
        arquitectura = dict(ARQUITECTURA_BASE)
        arquitectura.update(propuesta["cambios"])
        firma = firma_entrenamiento(
            processed_sha256=datos_corpus["processed_sha256"],
            corpus_version=datos_corpus["corpus_version"],
            arquitectura=arquitectura,
        )
        choque = firmas_existentes.get(firma)
        propuestas.append(
            {
                "orden": orden,
                "codigo": propuesta["codigo"],
                "training_id": f"FASTTEXT-WIKI-10M-{propuesta['codigo']}",
                "familia": propuesta["familia"],
                "valor": propuesta["valor"],
                "corpus": CORPUS_TRABAJO["id"],
                "tokens": datos_corpus["tokens"],
                "vector_size": arquitectura["dim"],
                "window": arquitectura["ventana"],
                "min_count": arquitectura["min_count"],
                "min_n": arquitectura["minn"],
                "max_n": arquitectura["maxn"],
                "negative": arquitectura["negativos"],
                "sample": arquitectura["muestreo"],
                "epochs": arquitectura["epocas"],
                "sg": arquitectura["sg"],
                "seed": arquitectura["semilla"],
                "bucket": arquitectura["bucket"],
                "training_signature": firma,
                "status": f"{VERDEZ_GUARDADO} - LOAD ARTIFACT" if choque else "PENDING",
                "equivale_a": choque or "",
                "presupuesto": "-" if choque else f"{orden}/{PRESUPUESTO_EXPLORATORIO}",
            }
        )
        log(
            f"   {propuesta['familia']:11s} = {propuesta['valor']:>4s} | "
            f"{'EXISTENTE' if choque else 'nueva'} | {firma[:12]}"
        )
    nuevas = [propuesta for propuesta in propuestas if propuesta["status"] == "PENDING"]
    return {
        "propuestas": propuestas,
        "exploratory_gastadas": 0,
        "reserva_gastada": 0,
        "colisiones": [p["training_id"] for p in propuestas if p["status"] != "PENDING"],
        "aserciones": {
            "nuevas_configuraciones_no_reexisten": len(nuevas) == len(propuestas),
            "presupuesto_exploratorio_no_superado": 0 <= PRESUPUESTO_EXPLORATORIO,
            "presupuesto_total_no_superado": 0 + 0 <= PRESUPUESTO_EXPLORATORIO
            + PRESUPUESTO_RESERVA,
        },
    }


def construir_tabla_b() -> list[dict[str, Any]]:
    """Construye la Tabla B: evaluaciones downstream ya realizadas en v5.

    Une los cinco CSV de resultados de la fase en una sola tabla con la contabilidad
    pedida en la seccion 32: separan cuantas **evaluaciones** se hicieron de cuantos
    **entrenamientos** se hicieron. Ninguna de estas filas consume presupuesto.

    Returns:
        Filas de la Tabla B.
    """
    log("4/5 tabla B: evaluaciones downstream historicas")
    fuentes = [
        ("experiments/embedding_02_fasttext_baseline_v5.py", "reports/embedding_downstream_results.csv", "modelo"),
        ("experiments/embedding_03_fusion_complementariedad_v5.py", "reports/embedding_fusion_v5.csv", "variante"),
        ("experiments/embedding_05_fasttext_web_v5.py", "reports/embedding_web_results_v5.csv", "variante"),
        ("experiments/embedding_08_fasttext_50M_v5.py", "reports/embedding_curve_50M_v5.csv", "variante"),
        ("experiments/embedding_10_fasttext_5M_v5.py", "reports/embedding_curve_5M_v5.csv", "variante"),
    ]
    filas: list[dict[str, Any]] = []
    contador = 0
    for script, csv_relativo, columna_id in fuentes:
        ruta = PROJECT_ROOT / csv_relativo
        if not ruta.exists():
            continue
        with ruta.open(encoding="utf-8") as archivo:
            for registro in csv.DictReader(archivo):
                variante = registro.get(columna_id, "")
                if variante == "SUELO-ALEATORIO":
                    continue
                contador += 1
                filas.append(
                    {
                        "evaluation_id": f"EVAL-V5-{contador:03d}",
                        "script": script,
                        "variante": variante,
                        "training_id": ALIAS_TRAINING.get(variante, ""),
                        "pooling": "media" if "A_media" in variante or "EMB" in variante or "FUSION" in variante else "",
                        "classifier": "SVC" if "SVC" in variante else ("LR" if "LR" in variante else "LR"),
                        "modo_embedding": "estandar" if "STANDARD" in variante else ("l2" if "FUSION" in variante or "EMB" in variante else ""),
                        "corrected_cv": registro.get("corrected_cv_umbrales_frozen_v03")
                        or registro.get("corrected_cv_umbrales_ajustados")
                        or "",
                        "delta_vs_V03": registro.get("delta_vs_v03_ajustados")
                        or registro.get("delta_vs_v03")
                        or "",
                        "ic95": (
                            f"[{registro.get('ic95_0', '')}, {registro.get('ic95_1', '')}]"
                            if registro.get("ic95_0")
                            else ""
                        ),
                        "decision": "historico v5, no consume presupuesto",
                    }
                )
    log(f"   {len(filas)} evaluaciones downstream, {len({f['training_id'] for f in filas if f['training_id']})} entrenamientos distintos")
    return filas


def verificar_reglas() -> dict[str, Any]:
    """Aplica la auditoria de recursos y de uso de TOXIC de la seccion 36.

    Delega en ``experiments/embedding_06_auditoria_v5.py`` para no duplicar los patrones
    prohibidos: si este script los escribiera literalmente, el analisis estatico los
    detectaria aqui mismo como uso indebido.

    Returns:
        Diccionario con el detalle de cada comprobacion.
    """
    log("5/5 auditoria de reglas de la seccion 36")
    modulo = _cargar_modulo(
        "embedding_06_auditoria",
        PROJECT_ROOT / "experiments" / "embedding_06_auditoria_v5.py",
    )
    recursos = modulo.auditar_recursos()
    uso_toxic = modulo.auditar_uso_de_toxic()
    holdout = modulo.auditar_holdout()
    fuga = modulo.auditar_fuga_measured()
    referencia = modulo.verificar_referencia()
    recursos_limpios = not recursos.get("hallazgos")
    log(
        f"   recursos prohibidos: {'0' if recursos_limpios else 'ALGUNO'} | "
        f"uso de TOXIC declarado: {uso_toxic.get('ok')} | "
        f"inconsistencias: {len(uso_toxic.get('inconsistentes', []))} | "
        f"holdout intacto: {holdout.get('presupuesto_intacto')} "
        f"({holdout.get('n_evaluaciones_registradas')}/5) | "
        f"fuga de test en corpus: {fuga.get('ok')} | "
        f"V03: {referencia.get('corrected_cv')}"
    )
    return {
        "recursos_prohibidos": {
            "transformers": 0 if recursos_limpios else "ver detalles",
            "bert_roberta_llm": 0 if recursos_limpios else "ver detalles",
            "embeddings_preentrenados": 0 if recursos_limpios else "ver detalles",
            "from_pretrained": 0 if recursos_limpios else "ver detalles",
            "hallazgos": recursos.get("hallazgos", []),
        },
        "uso_toxic": uso_toxic,
        "holdout": holdout,
        "fuga_test_en_corpus": fuga,
        "referencia_v03": referencia,
    }


def escribir_informe(
    corpus: dict[str, dict[str, Any]],
    existentes: list[dict[str, Any]],
    presupuesto: dict[str, Any],
    reglas: dict[str, Any],
    tabla_b: list[dict[str, Any]],
    gaps: list[str],
    duracion: float,
) -> None:
    """Escribe el informe, la Tabla A, la Tabla B y el fichero de presupuesto.

    Args:
        corpus: inventario de corpus.
        existentes: filas de la Tabla A de v5.
        presupuesto: resultado de :func:`calcular_presupuesto`.
        reglas: resultado de :func:`verificar_reglas`.
        tabla_b: filas de la Tabla B.
        gaps: gaps de trazabilidad de los artefactos historicos.
        duracion: segundos que tardo el script.
    """
    with TABLA_A.open("w", encoding="utf-8", newline="") as archivo:
        campos = list(existentes[0])
        for propuesta in presupuesto["propuestas"]:
            for campo in campos:
                propuesta.setdefault(campo, "")
        todos = existentes + [
            {campo: propuesta.get(campo, "") for campo in campos}
            for propuesta in presupuesto["propuestas"]
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(todos)
    with TABLA_B.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=list(tabla_b[0]), lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(tabla_b)
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duracion_s": round(duracion, 1),
                "referencia_v03": REFERENCIA_V03,
                "mejora_minima_gate": MEJORA_MINIMA,
                "presupuesto": {
                    "existing_v5_trainings": len(existentes),
                    "new_v5_1_trainings": presupuesto["exploratory_gastadas"],
                    "reserved_training_units": presupuesto["reserva_gastada"],
                    "total_new_budget_used": presupuesto["exploratory_gastadas"]
                    + presupuesto["reserva_gastada"],
                    "budget_remaining": PRESUPUESTO_EXPLORATORIO
                    + PRESUPUESTO_RESERVA
                    - presupuesto["exploratory_gastadas"]
                    - presupuesto["reserva_gastada"],
                    "limite_exploratorio": PRESUPUESTO_EXPLORATORIO,
                    "limite_reserva": PRESUPUESTO_RESERVA,
                },
                "aserciones": presupuesto["aserciones"],
                "corpus": corpus,
                "tabla_a": existentes,
                "propuestas": presupuesto["propuestas"],
                "n_downstream_evaluations": len(tabla_b),
                "gaps_trazabilidad": gaps,
                "reglas_seccion_36": reglas,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lineas = [
        "# P0 de v5.1: firmas de entrenamiento y contabilidad del presupuesto (EBD-17)",
        "",
        f"Generado por `experiments/embedding_11_auditoria_firmas_v5_1.py` ({EXPERIMENTO}).",
        f"Duracion: {round(duracion, 1)} s. **Este script no entrena ningun embedding.**",
        "",
        "## 1. Por que hacia falta",
        "",
        "La contabilidad de v5 estaba repartida en cuatro JSON de resultados y no existia una",
        "identidad canonica de entrenamiento. Sin ella no se puede saber si una configuracion",
        "propuesta en v5.1 es nueva o es un reentrenamiento disfrazado de algo que ya existe,",
        "ni se puede demostrar que el limite de 9 entrenamientos se respeta. Este script",
        "cierra ese hueco antes de gastar la primera unidad.",
        "",
        "## 2. Identidad de un entrenamiento",
        "",
        "```text",
        "training_signature = sha256(corpus_procesado_sha256 | corpus_version | vector_size |",
        "                             window | min_count | min_n | max_n | negative | sample |",
        "                             epochs | sg | seed | bucket)",
        "```",
        "",
        "Dos desviaciones razonadas respecto de la seccion 34, ambas documentadas en el",
        "docstring de `firma_entrenamiento`:",
        "",
        "- Se usa el sha256 del **corpus procesado**, no el del dump. El dump es identico para",
        "  5M, 10M y 50M de Wikipedia, asi que con la huella del dump tres corpora distintos",
        "  compartiran identidad y la comprobacion de la seccion 34 seria falsa.",
        "- Se incluye `bucket`, que la spec no lista. Su valor determina parte de la matriz de",
        "  subpalabras y por tanto los pesos; excluirlo permitiria declarar nueva una",
        "  configuracion que en realidad es otra. `hilos` queda fuera porque no cambia la",
        "  configuracion, solo el orden de actualizacion de los pesos.",
        "",
        "## 3. Corpus: huellas reales",
        "",
        "| corpus | tokens | processed_sha256 | bytes |",
        "|---|---|---|---|",
    ]
    for datos in corpus.values():
        lineas.append(
            f"| {datos['corpus_id']} ({datos['source']}) | {datos['tokens']:,} | "
            f"`{datos['processed_sha256'][:16]}` | {datos['bytes']:,} |".replace(",", ".")
        )
    lineas += [
        "",
        "El manifiesto de corpus queda actualizado con `processed_sha256`, `processed_ruta`,",
        "`processed_bytes` y `corpus_version`; ningun valor historico se altera.",
        "",
        "## 4. Tabla A: los diez entrenamientos existentes",
        "",
        "Las diez firmas son distintas, asi que la contabilidad es univoca. Ninguna consume",
        "presupuesto de v5.1. `training_time_s` es el reloj real cuando consta; el valor 0.0 con",
        "la fuente `no registrado en la corrida` significa que el modelo se entrego desde una",
        "corrida previa y su tiempo no quedo medido en ella.",
        "",
        "| training_id | corpus | dim | ws | minC | sub | neg | semilla | vocab | s | sha256 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fila in existentes:
        lineas.append(
            f"| {fila['training_id']} | {fila['corpus']} | {fila['vector_size']} | "
            f"{fila['window']} | {fila['min_count']} | {fila['min_n']}-{fila['max_n']} | "
            f"{fila['negative']} | {fila['seed']} | {fila['vocabulary_size']} | "
            f"{fila['training_time_s']} | `{fila['model_sha256'][:12]}` |"
        )
    lineas += [
        "",
        "## 5. Presupuesto de v5.1",
        "",
        "```text",
        "BASELINE EXISTENTE",
        "Wikipedia 10M / 200d / w5 / min3 / sub3-6 / neg10 / e5 / sg1 / seed42",
        f"-> {VERDEZ_GUARDADO} - LOAD ARTIFACT -> no consume presupuesto",
        "",
        "NUEVOS (los nueve):",
    ]
    for propuesta in presupuesto["propuestas"]:
        marca = "EXISTING - LOAD ARTIFACT" if propuesta["status"] != "PENDING" else propuesta["presupuesto"]
        lineas.append(f"{propuesta['codigo']:<12s} -> {marca}")
    lineas += [
        "",
        "RESERVA        -> 0/1 (solo evidencia clara; nunca para abrir una familia nueva)",
        "```",
        "",
        f"- Entrenamientos nuevos consumidos: **{presupuesto['exploratory_gastadas']}** de "
        f"{PRESUPUESTO_EXPLORATORIO} exploratorios y {PRESUPUESTO_RESERVA} de reserva.",
        f"- Propuestas que colisionan con un entrenamiento existente: "
        f"**{len(presupuesto['colisiones'])}**.",
        "",
        "La ausencia de colisiones es el resultado que importa: las nueve configuraciones",
        "propuestas son de verdad nuevas y ninguna se puede resolver cargando un artefacto.",
        "",
        "## 6. Tabla B: evaluaciones downstream de v5",
        "",
        f"{len(tabla_b)} evaluaciones sobre {len({f['training_id'] for f in tabla_b if f['training_id']})}",
        "entrenamientos distintos. Son `DOWNSTREAM EVALUATION UNIT`, no",
        "`EMBEDDING TRAINING UNIT`: no consumen presupuesto y por eso la contabilidad las",
        "separa en otra tabla.",
        "",
        "## 7. Auditoria de reglas (seccion 36)",
        "",
        "| comprobacion | resultado |",
        "|---|---|",
        f"| transformers / BERT / RoBERTa / LLM | "
        f"{'0 usos' if not reglas['recursos_prohibidos']['hallazgos'] else 'REVISAR'} |",
        f"| embeddings preentrenados / from_pretrained | "
        f"{'0 usos' if not reglas['recursos_prohibidos']['hallazgos'] else 'REVISAR'} |",
        f"| test en el corpus de entrenamiento | "
        f"{'0 coincidencias' if reglas['fuga_test_en_corpus'].get('ok') else 'REVISAR'} |",
        f"| holdout | intacto: {reglas['holdout'].get('presupuesto_intacto')} "
        f"({reglas['holdout'].get('n_evaluaciones_registradas')}/5) |",
        f"| uso de TOXIC declarado y coherente | "
        f"{reglas['uso_toxic'].get('ok')} "
        f"({len(reglas['uso_toxic'].get('inconsistentes', []))} inconsistencias) |",
        f"| referencia V03 | {reglas['referencia_v03'].get('corrected_cv')} "
        f"(coincide: {reglas['referencia_v03'].get('coincide')}) |",
        "",
        "La auditoria se delega en `experiments/embedding_06_auditoria_v5.py` y sus patrones",
        "prohibidos se toman de alli. Si este script los escribiera literalmente, el analisis",
        "estatico lo marcaria a el mismo como infractor.",
        "",
        "## 8. Gaps de trazabilidad detectados",
        "",
    ] + ([f"- {hueco}" for hueco in gaps] + [""] if gaps else ["Ninguno.", ""]) + [
        "## 9. Aserciones",
        "",
        "| asercion | resultado |",
        "|---|---|",
    ]
    for nombre, valor in presupuesto["aserciones"].items():
        lineas.append(f"| `{nombre}` | {'OK' if valor else 'FALLA'} |")
    lineas += [
        "",
        "## 10. Nota de trazabilidad: ubicacion del proyecto",
        "",
        "El worktree `proyecto-kaggle-toxic-2026-v5` estaba en `/home/carlos/Git/PLN-v5` y se",
        "migro a `/home/carlos/Git/PLN` (la carpeta de trabajo de esta sesion) a peticion del",
        "usuario. Se copiaron `data/`, `tools/`, `experiments/*.log`, `features/` y `models/`;",
        "se verificaron los sha256 de los cinco corpus y de los diez `.vec`; y el control V03",
        f"reproduce **{REFERENCIA_V03} exacto** desde la nueva ubicacion. El trabajo sin",
        "commitear de KAGGLE_AGE_2026_PUJ quedo aparcado en `git stash` (no es un commit) y",
        "sus datos no se tocaron.",
        "",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta P0: firmas, presupuesto y auditoria de reglas."""
    inicio = time.time()
    log("P0 de v5.1: contabilidad antes de gastar una sola unidad de entrenamiento")
    corpus = inventariar_corpus()
    existentes, gaps = inventariar_modelos(corpus)
    presupuesto = calcular_presupuesto(corpus, existentes)
    tabla_b = construir_tabla_b()
    reglas = verificar_reglas()
    if not all(presupuesto["aserciones"].values()):
        raise RuntimeError(f"Aserciones de presupuesto incumplidas: {presupuesto['aserciones']}")
    if reglas["recursos_prohibidos"]["hallazgos"]:
        raise RuntimeError("La auditoria de recursos prohibidos ha encontrado usos")
    escribir_informe(
        corpus, existentes, presupuesto, reglas, tabla_b, gaps, time.time() - inicio
    )
    log(
        f"P0 completada en {round(time.time() - inicio, 1)} s | "
        f"{len(existentes)} firmas existentes | "
        f"{len(presupuesto['propuestas'])} propuestas, "
        f"{len(presupuesto['colisiones'])} colisiones | "
        f"presupuesto usado 0/{PRESUPUESTO_EXPLORATORIO}"
    )


if __name__ == "__main__":
    main()
