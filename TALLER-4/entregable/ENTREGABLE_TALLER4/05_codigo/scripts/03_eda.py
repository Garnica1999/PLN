"""EDA del corpus CEV (Fase 5 del TALLER-4).

Fecha: 2026-09-04
Descripcion: analisis exploratorio sobre los archivos preprocesados de la Fase 4
    (data/corpus/preprocesado/{libros,entrevistas}_{lemas,tokens}.txt), segun la
    seccion (e) de plan/arquitectura.md y el punto 2 del taller:
    1) estadisticas descriptivas por corpus (docs, tokens, tipos, longitudes con
       histogramas; chars y paginas en entrevistas);
    2) diversidad lexica: TTR con muestra fija de 1000 tokens (Bestgen 2023) y
       Yule's K por documento, comparacion libros vs entrevistas;
    3) frecuencias top-30 de uni/bi/trigramas por corpus (CSV);
    4) nubes de palabras uni/bi/trigramas por corpus (wordcloud sobre las
       frecuencias calculadas; 3 figuras minimo por corpus);
    5) near-duplicates entre entrevistas: Jaccard de shingles de 5 lemas
       (Broder 1997; minhash + LSH y verificacion exacta de los candidatos);
       los pares > 0.85 se reportan SIN eliminar (corpus oficial integro);
    6) comparacion lexica libros <-> entrevistas: solapamiento de vocabulario y
       terminos distintivos top-25 por log-odds con previo dirichlet de 0.5;
    7) lectura del tesauro CEV (ADR-004, uso cualitativo, NO se integra al
       modelo): cache de etiquetas SKOS y cruce con los terminos distintivos.

Decisiones de consumo (documentadas; los archivos congelados NO se modifican):
    - Correcciones de lemas (config/lemas_correcciones.json): mapa lema -> lema
      (p. ej. 'atreveir' -> 'atrever') aplicado EN MEMORIA a cada token.
    - Parada adicional en memoria: spacy.lang.es.STOP_WORDS un config/
      stopwords_habla.txt. Verificacion de la Fase 5: las stopwords SI se
      aplicaron en la Fase 4 por superficie (18.8M tokens eliminados, 73.25 %),
      pero el archivo de lemas conserva residuales funcionales (formas
      capitalizadas que escapan a is_stop y formas verbales/cliticos no
      marcados: p. ej. el lema 'el' 122K, 'haber' 35.5K, 'estar' 34.5K, 'ser'
      12.9K en entrevistas); el EDA aplica la parada sobre los lemas en
      consumo y congela la lista final de muletillas.
    - Corpus integro: los 2 docs de entrevista vacios legitimos y los
      near-duplicates se documentan, NO se eliminan.

Salidas:
    - imagenes/eda_*.png                       (figuras, dpi=150, tight)
    - resultados/eda/*.csv                      (estadisticas, frecuencias,
       diversidad, near-dups, solapamiento, distintivos)
    - resultados/eda/resumen_eda.md             (hallazgos, <= 60 lineas)
    - config/stopwords_habla.txt                (congelacion: lista final)
    - config/tfidf_config.json                  (congelacion: propuesta Fase 6)
    - config/lemas_correcciones.json            (validado; esqueleto si falta)
    - resultados/metricas.json                  (entrada fase5_eda anadida)

Uso:
    python scripts/03_eda.py                          # todas las etapas
    python scripts/03_eda.py --solo estadisticas
    python scripts/03_eda.py --solo frecuencias nubes
    python scripts/03_eda.py --solo comparacion tesauro congelar

Notas de rendimiento: sin multiprocessing (numpy vectorizado); el corpus de
entrevistas (10.9M lemas) se lee por pasadas; presupuesto total < 15 min en la
maquina de referencia (plan/00_maquina.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spacy.lang.es import STOP_WORDS
from wordcloud import WordCloud

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_PREPROCESADO = RUTA_PROYECTO / "data" / "corpus" / "preprocesado"
RUTA_CORPUS = RUTA_PROYECTO / "data" / "corpus"
RUTA_IMAGENES = RUTA_PROYECTO / "imagenes"
RUTA_EDA = RUTA_PROYECTO / "resultados" / "eda"
RUTA_CONFIG = RUTA_PROYECTO / "config"

N_MUESTRA_TTR = 1000  # muestra fija para TTR muestreado (Bestgen 2023)
TAMANO_SHINGLE = 5  # shingles de 5 lemas consecutivos (Broder 1997)
N_HASHES_MINHASH = 60  # k de minhash = bandas * r
N_BANDAS = 20
N_R_BANDA = 3
UMBRAL_NEAR_DUP = 0.85  # pares casi identicos: se reportan, NO se eliminan
ALFA_LOGODDS = 0.5  # previo dirichlet para log-odds (Monroe et al. 2008)
SEMILLA = 42
N_TOP = 30  # top de frecuencias por n-grama
N_DISTINTIVOS = 25  # top de distintivos por direccion (log-odds)
MIN_APOYO_LOGODDS = 40  # frecuencia conjunta minima para el log-odds
MIN_LEMAS_NEAR_DUP = 100  # docs con >= N lemas entran al analisis de pares
TOKEN_SEP = -1  # separador de documento en la codificacion de n-gramas

CORPUS = ("libros", "entrevistas")

# Candidatas evaluadas como parada/muletilla. La evidencia se vuelca en
# resultados/eda/muletillas_evidencia.csv y la decision (reglas explicitas en
# MULETILLAS_ACTUALES / MULETILLAS_NUEVAS) en muletillas_decision.csv.
CANDIDATAS_EVIDENCIA: tuple[str, ...] = (
    "eh", "ehh", "digamos", "listo", "aja", "ajá", "ah", "ahh", "mmm", "mm",
    "umm", "um", "vea", "ve", "jum", "uy", "ujum", "uf", "mire", "mirá",
    "mira", "osea", "pa", "allá", "acá", "ahí", "entonces", "pues", "bueno",
    "cierto", "este", "esto", "esta", "cómo", "cosa",
)

# Reglas de congelacion de config/stopwords_habla.txt (etapa congelar):
#   (token, motivo, umbral_frec_lema_archivo, umbral_frec_superficie_cruda)
# con umbral None cuando no aplica. La evidencia es estable (punto fijo entre
# ejecuciones): la frecuencia del token como LEMA en entrevistas_lemas.txt SIN
# parada (archivo congelado de la Fase 4) o como superficie en
# entrevistas_tokens.txt. Nota: hay muletillas cuya superficie difiere del
# lema (p. ej. 'o sea' escrito 'osea'/'osea' -> lema 'oseo'); por eso se usan
# ambas frecuencias.
MULETILLAS_ACTUALES: tuple[tuple[str, str, float | None, float | None], ...] = (
    ("eh", "interjeccion de relleno", None, 300),
    ("ehh", "interjeccion de relleno", None, 300),
    ("digamos", "muletilla de aproximacion", None, 300),
    ("listo", "muletilla de cierre/aceptacion", None, 300),
    ("aja", "interjeccion de asentimiento (sin tilde)", None, 300),
    ("ajá", "interjeccion de asentimiento", None, 300),
    ("ah", "interjeccion", None, 300),
    ("ahh", "interjeccion alargada", None, 300),
    ("mmm", "interjeccion de duda", None, 300),
    ("mm", "interjeccion de duda", None, 300),
    ("umm", "interjeccion de duda", None, 300),
    ("um", "interjeccion de duda", None, 300),
    ("vea", "imperativo de atencion (registro oral)", None, 300),
    ("ve", "imperativo de atencion", None, 300),
    ("jum", "interjeccion de duda", None, 300),
    ("uy", "interjeccion", None, 300),
    ("ujum", "interjeccion de asentimiento", None, 300),
    ("uf", "interjeccion", None, 300),
    ("mire", "imperativo de atencion ('mire usted')", None, 300),
    ("mira", "imperativo de atencion ('mira que...')", None, 300),
)
# Muletillas/interjecciones detectadas como distintivas del habla en el EDA y
# lemas funcionales residuales (verbos auxiliares cuyo lema no cubre spaCy:
# la Fase 4 filtro sus formas conjugadas por superficie, pero 'estando' ->
# 'estar' escapo; se completan para coherencia con ADR-003).
MULETILLAS_NUEVAS: tuple[tuple[str, str, float | None, float | None], ...] = (
    ("mirá", "imperativo voseante de atencion", 100, 300),
    ("osea", "muletilla contraida de 'o sea'", 1000, 300),
    ("óseo", "muletilla 'o sea' con grafia erronea ('osea'/'osea')", 300, 300),
    ("pa", "forma oral de la preposicion 'para'", 1000, 300),
    ("allá", "deictico espacial del habla (sin valor lexico)", 1000, 300),
    ("acá", "deictico espacial del habla (sin valor lexico)", 1000, 300),
    ("ok", "afirmacion del habla (prestamo del ingles)", 300, 300),
    ("okey", "afirmacion del habla (prestamo del ingles)", 300, 300),
    ("okay", "afirmacion del habla (prestamo del ingles)", 300, 300),
    ("uhm", "interjeccion de duda", 300, 300),
    ("mmjm", "interjeccion de asentimiento", 300, 300),
    ("mjmm", "interjeccion de asentimiento", 200, 300),
    ("eee", "interjeccion de duda alargada", 300, 300),
    ("hmm", "interjeccion de duda", 300, 300),
    ("ehhh", "interjeccion alargada (variante de 'eh')", 300, 300),
    ("ehm", "interjeccion de duda", 300, 300),
    ("bla", "onomatopeya de relleno ('bla bla')", 300, 300),
    ("aló", "interjeccion de llamada telefonica", 300, 300),
    ("mjm", "interjeccion de asentimiento", 300, 300),
    ("ujumm", "interjeccion de asentimiento (variante de 'ujum')", 200, 300),
    ("etc", "abreviatura 'etc.' sin punto (ruido de transcripcion)", 200, 300),
    ("inf", "etiqueta de hablante residual 'Inf:' (ruido de transcripcion)", 200, 300),
    ("interrup", "marca de interrupcion residual '[INTERRUP]' sin corchete", 200, 300),
    ("jumm", "interjeccion de duda (variante de 'jum')", 200, 300),
    ("hum", "interjeccion de duda", 200, 300),
    ("mmmm", "interjeccion de duda (variante de 'mmm')", 200, 300),
    ("ósea", "muletilla 'o sea' con grafia erronea ('osea'/'osea')", 200, 300),
    ("seño", "truncamiento oral de 'senor/senora' (trato)", 200, 300),
    ("unuma", "interjeccion de duda", 200, 300),
    ("em", "interjeccion de duda (variante de 'eh')", 200, 300),
    ("ser", "lema funcional residual (verbo copulativo)", 1000, 50000),
    ("estar", "lema funcional residual (verbo copulativo)", 1000, 50000),
    ("haber", "lema funcional residual (verbo auxiliar)", 1000, 50000),
)

# Ruido tipografico/bibliografico del corpus de LIBROS (palabras partidas por
# el guion de fin de linea del PDF y abreviatura bibliografica 'ibid.') que
# escapo al filtro alfanumerico de la Fase 4. Se excluye de los analisis
# lexicos comparativos (solapamiento y distintivos); documentado en
# config/tfidf_config.json para que la Fase 6 decida su tratamiento.
RUIDO_LIBROS: frozenset[str] = frozenset(
    {
        # Fragmentos de palabras partidas por el guion de linea del PDF
        # (segunda mitad tras el guion; verificados por contexto).
        "ción", "nidad", "nes", "nización", "zación", "tivo", "tado",
        "lidad", "mento", "nas", "nal", "flicto", "litar", "tación",
        "mación", "lencia", "tamiento", "cio", "mos", "nsa", "sona",
        "sión", "dad", "ban", "torio", "zado", "tes", "nar", "nado",
        "tro", "exintegrant",
        # Ruido bibliografico: abreviatura 'ibid.', numeracion romana de
        # tomos, latin/ingles de citas.
        "ibíd", "project", "documentation", "contínuum", "párr", "iii",
        "iv", "of", "et", "vs", "map",
    }
)


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def leer_jsonl(ruta: Path) -> list[dict[str, Any]]:
    """Lee un archivo JSONL y devuelve la lista de registros."""
    with ruta.open(encoding="utf-8") as fh:
        return [json.loads(linea) for linea in fh]


def escribir_json(objeto: Any, ruta: Path) -> None:
    """Escribe un objeto JSON con indentacion y encoding utf-8."""
    with ruta.open("w", encoding="utf-8") as fh:
        json.dump(objeto, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def preparar_directorios() -> None:
    """Crea los directorios de salida si no existen."""
    for ruta in (RUTA_IMAGENES, RUTA_EDA):
        ruta.mkdir(parents=True, exist_ok=True)


def guardar_figura(figura: plt.Figure, nombre: str) -> None:
    """Guarda una figura en imagenes/ con dpi=150 y bbox ajustado."""
    ruta = RUTA_IMAGENES / nombre
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)
    print(f"  [figura] imagenes/{nombre}")


def metadatos_corpus(nombre: str) -> list[dict[str, Any]]:
    """Devuelve los metadatos alineados por linea de *_docs.json."""
    datos = json.loads(
        (RUTA_PREPROCESADO / f"{nombre}_docs.json").read_text(encoding="utf-8")
    )
    return datos["docs"]


def cargar_correcciones() -> dict[str, str]:
    """Carga config/lemas_correcciones.json; crea el esqueleto si no existe.

    Returns:
        Mapa lema anomalo -> lema correcto (claves con la forma del corpus).
    """
    ruta = RUTA_CONFIG / "lemas_correcciones.json"
    if not ruta.exists():
        escribir_json(
            {
                "_comentario": "Correcciones de lemas anomalos detectadas en la "
                "Fase 5 (EDA). Aplicar EN CONSUMO sobre *_lemas.txt; no "
                "reescribir los archivos congelados.",
                "correcciones": {},
            },
            ruta,
        )
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    correcciones = datos.get("correcciones", {})
    if not isinstance(correcciones, dict):
        raise TypeError(f"{ruta.name}: 'correcciones' debe ser un objeto")
    return {str(k): str(v) for k, v in correcciones.items()}


def cargar_muletillas_config() -> set[str]:
    """Carga los tokens de config/stopwords_habla.txt (ignora '#' y vacias)."""
    ruta = RUTA_CONFIG / "stopwords_habla.txt"
    if not ruta.exists():
        return set()
    return {
        linea.strip()
        for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.startswith("#")
    }


def parada_en_consumo() -> set[str]:
    """Parada en memoria: STOP_WORDS de spaCy mas config/stopwords_habla.txt.

    La Fase 4 filtro por superficie (is_stop); aqui se filtra sobre el lema
    para eliminar los residuales funcionales conservados en *_lemas.txt.
    """
    return set(STOP_WORDS) | cargar_muletillas_config()


def corregir_y_filtrar(
    linea: str, correcciones: dict[str, str], parada: set[str]
) -> list[str]:
    """Aplica correcciones de lemas y parada en consumo a una linea.

    Orden por token: correccion (lema -> lema) y, sobre el resultado, parada.
    Anos (1900-2025) y categorias canonicas no se ven afectados.

    Returns:
        Lista de lemas de contenido del documento.
    """
    salida: list[str] = []
    for token in linea.split():
        if correcciones:
            token = correcciones.get(token, token)
        if token not in parada:
            salida.append(token)
    return salida


def resumen_vector(valores: np.ndarray) -> dict[str, float]:
    """Resumen estadistico (min, percentiles, media, max) de un vector."""
    percentiles = np.percentile(valores, [1, 10, 25, 50, 75, 90, 99])
    return {
        "n": len(valores),
        "min": float(valores.min()),
        "media": float(valores.mean()),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "mediana": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "max": float(valores.max()),
    }


# ---------------------------------------------------------------------------
# Etapa 1: estadisticas descriptivas por corpus (arquitectura e.1)
# ---------------------------------------------------------------------------


def contar_tokens_archivo(ruta: Path) -> tuple[int, list[int]]:
    """Cuenta tokens por linea de un archivo preprocesado (sin split)."""
    total = 0
    por_doc: list[int] = []
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            n = linea.count(" ") + 1 if linea.strip() else 0
            por_doc.append(n)
            total += n
    return total, por_doc


def etapa_estadisticas(
    correcciones: dict[str, str], parada: set[str]
) -> dict[str, Any]:
    """Estadisticas por corpus: docs, tokens, tipos, longitudes y figuras.

    Returns:
        Contexto con estadisticas agregadas y longitudes por documento.
    """
    print("\n[1/7] Estadisticas descriptivas por corpus")
    ctx: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    por_doc: dict[str, dict[str, list[int]]] = {}

    for nombre in CORPUS:
        ruta_lemas = RUTA_PREPROCESADO / f"{nombre}_lemas.txt"
        ruta_tokens = RUTA_PREPROCESADO / f"{nombre}_tokens.txt"
        n_crudos, crudos_por_doc = contar_tokens_archivo(ruta_tokens)
        n_lemas_por_doc: list[int] = []
        n_contenido_por_doc: list[int] = []
        tipos_fase4: set[str] = set()
        tipos_contenido: set[str] = set()
        vacios: list[int] = []
        with ruta_lemas.open(encoding="utf-8") as fh:
            for idx, linea in enumerate(fh):
                n_lemas_por_doc.append(
                    linea.count(" ") + 1 if linea.strip() else 0
                )
                tipos_fase4.update(linea.split())
                contenido = corregir_y_filtrar(linea, correcciones, parada)
                tipos_contenido.update(contenido)
                n_contenido_por_doc.append(len(contenido))
                if not contenido:
                    vacios.append(idx)
        stats = resumen_vector(np.asarray(n_contenido_por_doc, dtype=float))
        filas.append(
            {
                "corpus": nombre,
                "n_docs": len(n_lemas_por_doc),
                "tokens_crudos_spacy": n_crudos,
                "lemas_emitidos_fase4": int(sum(n_lemas_por_doc)),
                "tipos_lemas_fase4": len(tipos_fase4),
                "lemas_contenido_eda": int(sum(n_contenido_por_doc)),
                "tipos_contenido_eda": len(tipos_contenido),
                "lemas_doc_media": round(stats["media"], 1),
                "lemas_doc_mediana": round(stats["mediana"], 1),
                "lemas_doc_p90": round(stats["p90"], 1),
                "lemas_doc_p99": round(stats["p99"], 1),
                "lemas_doc_min": stats["min"],
                "lemas_doc_max": stats["max"],
                "docs_vacios_contenido": len(vacios),
            }
        )
        por_doc[nombre] = {
            "crudos": crudos_por_doc,
            "lemas": n_lemas_por_doc,
            "contenido": n_contenido_por_doc,
        }
        f = filas[-1]
        print(
            f"  {nombre}: {f['n_docs']} docs | crudos {f['tokens_crudos_spacy']:,} "
            f"| lemas contenido {f['lemas_contenido_eda']:,} "
            f"| tipos contenido {f['tipos_contenido_eda']:,} | vacios {len(vacios)}"
        )

    # Chars y paginas de entrevistas (rango del crudo conocido: 3-850K chars).
    chars_ent: list[int] = []
    with (RUTA_CORPUS / "entrevistas_normalizadas.jsonl").open(
        encoding="utf-8"
    ) as fh:
        for linea in fh:
            chars_ent.append(len(json.loads(linea)["text_normalizado"]))
    metas_ent = metadatos_corpus("entrevistas")
    pages_ent = [int(m["pages"]) for m in metas_ent]

    metas_lib = metadatos_corpus("libros")
    segmentos = leer_jsonl(RUTA_CORPUS / "segmentos_libros.jsonl")
    palabras_raw = {s["id_segmento"]: int(s["n_palabras"]) for s in segmentos}
    palabras_libros = [palabras_raw[m["id"]] for m in metas_lib]
    tomos_libros = [m["tomo"] for m in metas_lib]

    # Figuras: histogramas de longitudes.
    figuras: list[tuple[str, str, list[int], bool]] = [
        ("libros", "lemas de contenido por segmento", por_doc["libros"]["contenido"], False),
        ("entrevistas", "lemas de contenido por entrevista", por_doc["entrevistas"]["contenido"], True),
        ("entrevistas", "chars del texto normalizado", chars_ent, True),
        ("entrevistas", "paginas por entrevista", pages_ent, True),
    ]
    for nombre, eje, valores, recortar_p99 in figuras:
        arr = np.asarray(valores, dtype=float)
        if recortar_p99:
            corte = float(np.percentile(arr, 99))
            datos_hist = arr[arr <= corte]
            extra = f"p99={corte:,.0f} "
        else:
            datos_hist = arr
            extra = ""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(datos_hist, bins=60, color="#4C72B0", edgecolor="white")
        ax.set_title(
            f"{nombre}: {eje}\nn={len(arr)} media={arr.mean():,.0f} "
            f"mediana={np.median(arr):,.0f} {extra}max={arr.max():,.0f}",
            fontsize=10,
        )
        ax.set_xlabel(eje)
        ax.set_ylabel("documentos")
        ax.axvline(np.median(arr), color="darkred", ls="--", lw=1, label="mediana")
        ax.legend(fontsize=8)
        slug = eje.split(" ")[0]
        guardar_figura(fig, f"eda_{nombre}_hist_{slug}.png")

    # Boxplot por tomo (libros).
    orden_tomos = sorted(set(tomos_libros))
    datos_tomo = [
        np.asarray(
            [
                v
                for v, t in zip(
                    por_doc["libros"]["contenido"], tomos_libros, strict=True
                )
                if t == tomo
            ],
            dtype=float,
        )
        for tomo in orden_tomos
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(datos_tomo, labels=orden_tomos, showfliers=False)
    ax.set_title("libros: lemas de contenido por segmento y tomo")
    ax.set_ylabel("lemas por segmento")
    ax.set_xlabel("tomo")
    guardar_figura(fig, "eda_libros_lemas_por_tomo.png")

    # CSVs de estadisticas.
    pd.DataFrame(filas).to_csv(RUTA_EDA / "estadisticas_corpus.csv", index=False)
    filas_largo: list[dict[str, Any]] = []
    for nombre in CORPUS:
        metas = metadatos_corpus(nombre)
        for idx, meta in enumerate(metas):
            fila: dict[str, Any] = {
                "corpus": nombre,
                "idx": idx,
                "id": meta["id"],
                "n_tokens_crudos_spacy": por_doc[nombre]["crudos"][idx],
                "n_lemas_fase4": por_doc[nombre]["lemas"][idx],
                "n_lemas_contenido_eda": por_doc[nombre]["contenido"][idx],
            }
            if nombre == "libros":
                fila.update(
                    {
                        "tomo": meta["tomo"],
                        "titulo": meta.get("titulo", ""),
                        "n_palabras_raw": palabras_libros[idx],
                    }
                )
            else:
                fila.update(
                    {
                        "pages": int(meta["pages"]),
                        "chars_normalizados": chars_ent[idx],
                    }
                )
            filas_largo.append(fila)
    pd.DataFrame(filas_largo).to_csv(
        RUTA_EDA / "longitudes_por_doc.csv", index=False
    )

    filas_tomo: list[dict[str, Any]] = []
    for tomo in orden_tomos:
        idx_tomo = [i for i, t in enumerate(tomos_libros) if t == tomo]
        arr_tomo = np.asarray(
            [por_doc["libros"]["contenido"][i] for i in idx_tomo], dtype=float
        )
        filas_tomo.append(
            {
                "tomo": tomo,
                "n_segmentos": len(idx_tomo),
                "lemas_contenido_total": int(arr_tomo.sum()),
                "media": round(float(arr_tomo.mean()), 1),
                "mediana": round(float(np.median(arr_tomo)), 1),
                "min": float(arr_tomo.min()),
                "max": float(arr_tomo.max()),
            }
        )
    pd.DataFrame(filas_tomo).to_csv(
        RUTA_EDA / "estadisticas_por_tomo.csv", index=False
    )

    ctx.update(
        {
            "estadisticas": filas,
            "por_doc": por_doc,
            "chars_entrevistas": chars_ent,
            "pages_entrevistas": pages_ent,
            "tomos_libros": orden_tomos,
        }
    )
    return ctx


# ---------------------------------------------------------------------------
# Etapa 2: frecuencias top-30 uni/bi/trigramas (arquitectura e.3)
# ---------------------------------------------------------------------------


def construir_vocabulario_y_arrays(
    nombre: str,
    correcciones: dict[str, str],
    parada: set[str],
    contar_anomalos: dict[str, str],
) -> tuple[list[str], list[np.ndarray], np.ndarray, Counter[str]]:
    """Construye el vocabulario ordenado y codifica los documentos.

    El vocabulario se ordena por frecuencia descendente (empates alfabeticos),
    por lo que el id 0 es el tipo mas frecuente del corpus. Los lemas
    anomalos del mapa de correcciones se cuentan ANTES de corregir (evidencia
    de la correccion, sin pasada extra).

    Returns:
        Tupla (vocabulario id->token, arrays por documento, df por tipo,
        contador de lemas anomalos sin corregir). El df (docs con el tipo)
        solo se computa para 'libros' (la coleccion del IR).
    """
    contador: Counter[str] = Counter()
    anomalos: Counter[str] = Counter()
    ruta = RUTA_PREPROCESADO / f"{nombre}_lemas.txt"
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if contar_anomalos:
                anomalos.update(t for t in linea.split() if t in contar_anomalos)
            contador.update(corregir_y_filtrar(linea, correcciones, parada))
    vocabulario = [
        token for token, _ in sorted(contador.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    token_a_id = {token: idx for idx, token in enumerate(vocabulario)}
    documentos: list[np.ndarray] = []
    df_por_tipo = np.zeros(len(vocabulario), dtype=np.int32)
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            contenido = corregir_y_filtrar(linea, correcciones, parada)
            doc = np.asarray([token_a_id[t] for t in contenido], dtype=np.int32)
            documentos.append(doc)
            if nombre == "libros" and len(doc):
                df_por_tipo[np.unique(doc)] += 1
    return vocabulario, documentos, df_por_tipo, anomalos


def decodificar_top_ngramas(
    ids: np.ndarray,
    conteos: np.ndarray,
    n: int,
    base: int,
    vocabulario: list[str],
    top: int,
) -> list[tuple[str, int]]:
    """Decodifica los 'top' n-gramas mas frecuentes a 'a_b_c'."""
    if len(ids) == 0:
        return []
    k = min(top, len(ids))
    indices = np.argpartition(-conteos, k - 1)[:k]
    posiciones = indices[np.argsort(-conteos[indices])]
    resultado: list[tuple[str, int]] = []
    for pos in posiciones:
        resto = int(ids[pos])
        partes: list[str] = []
        for _ in range(n):
            resto, idx = divmod(resto, base)
            partes.append(vocabulario[idx])
        resultado.append(("_".join(reversed(partes)), int(conteos[pos])))
    return resultado


def ngramas_top(
    documentos: list[np.ndarray],
    n: int,
    vocabulario: list[str],
    top: int,
) -> list[tuple[str, int]]:
    """Top n-gramas consecutivos sin cruzar limites de documento."""
    if n == 1:
        total = np.concatenate(documentos).astype(np.int64)
        ids, conteos = np.unique(total, return_counts=True)
        return decodificar_top_ngramas(
            ids, conteos, 1, len(vocabulario) + 1, vocabulario, top
        )
    base = len(vocabulario) + 1
    partes = [
        np.concatenate([doc.astype(np.int64), [TOKEN_SEP]])
        for doc in documentos
        if len(doc) >= n
    ]
    if not partes:
        return []
    arr = np.concatenate(partes)
    if n == 2:
        ids_cod = arr[:-1] * base + arr[1:]
        valido = (arr[:-1] >= 0) & (arr[1:] >= 0)
    else:
        ids_cod = (arr[:-2] * base + arr[1:-1]) * base + arr[2:]
        valido = (arr[:-2] >= 0) & (arr[1:-1] >= 0) & (arr[2:] >= 0)
    ids, conteos = np.unique(ids_cod[valido], return_counts=True)
    return decodificar_top_ngramas(ids, conteos, n, base, vocabulario, top)


def ngramas_counter_top(
    documentos: list[np.ndarray],
    n: int,
    vocabulario: list[str],
    limite: int,
) -> Counter[str]:
    """Counter de n-gramas 'a_b' con los 'limite' mas frecuentes (para nubes)."""
    if n == 1:
        raise ValueError("los unigramas salen del contador global")
    base = len(vocabulario) + 1
    partes = [
        np.concatenate([doc.astype(np.int64), [TOKEN_SEP]])
        for doc in documentos
        if len(doc) >= n
    ]
    if not partes:
        return Counter()
    arr = np.concatenate(partes)
    if n == 2:
        ids_cod = arr[:-1] * base + arr[1:]
        valido = (arr[:-1] >= 0) & (arr[1:] >= 0)
    else:
        ids_cod = (arr[:-2] * base + arr[1:-1]) * base + arr[2:]
        valido = (arr[:-2] >= 0) & (arr[1:-1] >= 0) & (arr[2:] >= 0)
    ids, conteos = np.unique(ids_cod[valido], return_counts=True)
    k = min(limite, len(ids))
    indices = np.argpartition(-conteos, k - 1)[:k]
    salida: Counter[str] = Counter()
    for pos in indices:
        resto = int(ids[pos])
        partes_dec: list[str] = []
        for _ in range(n):
            resto, idx = divmod(resto, base)
            partes_dec.append(vocabulario[idx])
        salida["_".join(reversed(partes_dec))] = int(conteos[pos])
    return salida


def conteo_lemas_candidatas(ruta: Path, candidatas: set[str]) -> dict[str, int]:
    """Cuenta los candidatas como lemas en el archivo SIN aplicar parada.

    Usa el archivo congelado de la Fase 4 (inmutable), por lo que la evidencia
    es estable entre ejecuciones (punto fijo de la congelacion).
    """
    conteo: dict[str, int] = {c: 0 for c in candidatas}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            for token in linea.split():
                if token in conteo:
                    conteo[token] += 1
    return conteo


def conteo_crudo_candidatas(ruta: Path, candidatas: set[str]) -> dict[str, int]:
    """Cuenta en tokens crudos (superficie) las candidatas de un archivo."""
    conteo: dict[str, int] = {c: 0 for c in candidatas}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            for token in linea.split():
                if token in conteo:
                    conteo[token] += 1
    return conteo


def etapa_frecuencias(
    correcciones: dict[str, str], parada: set[str]
) -> dict[str, Any]:
    """Top-30 uni/bi/trigramas por corpus (CSV) y evidencia de muletillas.

    Returns:
        Contexto con vocabularios, arrays codificados, contadores globales,
        df de la coleccion y tablas de evidencia.
    """
    print("\n[2/7] Frecuencias top-30 (uni/bi/trigramas) por corpus")
    ctx: dict[str, Any] = {}
    contadores: dict[str, Counter[str]] = {}
    vocabularios: dict[str, list[str]] = {}
    documentos: dict[str, list[np.ndarray]] = {}
    df_libros: np.ndarray = np.asarray([], dtype=np.int32)
    anomalos_totales: Counter[str] = Counter()
    anomalos_map = cargar_correcciones()

    for nombre in CORPUS:
        vocabulario, arrays, df, anomalos = construir_vocabulario_y_arrays(
            nombre, correcciones, parada, anomalos_map
        )
        vocabularios[nombre] = vocabulario
        documentos[nombre] = arrays
        anomalos_totales.update(anomalos)
        if nombre == "libros":
            df_libros = df
        total = np.concatenate(arrays).astype(np.int64)
        ids, conteos = np.unique(total, return_counts=True)
        contadores[nombre] = Counter(
            {
                vocabulario[int(i)]: int(c)
                for i, c in zip(ids, conteos, strict=True)
            }
        )
        print(
            f"  {nombre}: vocabulario {len(vocabulario):,} tipos | "
            f"{sum(len(a) for a in arrays):,} tokens de contenido"
        )

    for nombre in CORPUS:
        vocabulario = vocabularios[nombre]
        filas: list[dict[str, Any]] = []
        for n, prefijo in ((1, "uni"), (2, "bi"), (3, "tri")):
            top = ngramas_top(documentos[nombre], n, vocabulario, N_TOP)
            filas.extend(
                {
                    "corpus": nombre,
                    "ngrama": termino,
                    "n_tokens": n,
                    "frecuencia": frecuencia,
                }
                for termino, frecuencia in top
            )
            pd.DataFrame(
                [{"ngrama": t, "frecuencia": f} for t, f in top]
            ).to_csv(
                RUTA_EDA / f"frecuencias_{nombre}_{prefijo}gramas_top30.csv",
                index=False,
            )
        pd.DataFrame(filas).to_csv(
            RUTA_EDA / f"frecuencias_{nombre}_top30_todas.csv", index=False
        )
        print(f"  {nombre}: tablas top-30 uni/bi/tri listas")

    # Evidencia de muletillas (lemas y superficie cruda en entrevistas).
    candidatas = (
        set(CANDIDATAS_EVIDENCIA)
        | {t for t, *_ in MULETILLAS_ACTUALES}
        | {t for t, *_ in MULETILLAS_NUEVAS}
    )
    crudo_ent = conteo_crudo_candidatas(
        RUTA_PREPROCESADO / "entrevistas_tokens.txt", candidatas
    )
    lemas_archivo = conteo_lemas_candidatas(
        RUTA_PREPROCESADO / "entrevistas_lemas.txt", candidatas
    )
    en_spacy = set(STOP_WORDS)
    filas_muletilla = [
        {
            "token": token,
            "en_stopwords_spacy": token in en_spacy,
            "en_config_fase4": token in cargar_muletillas_config(),
            "frec_lema_archivo_sin_parada": lemas_archivo.get(token, 0),
            "frec_lema_libros": contadores["libros"].get(token, 0),
            "frec_crudo_entrevistas": crudo_ent.get(token, 0),
        }
        for token in sorted(candidatas)
    ]
    pd.DataFrame(filas_muletilla).to_csv(
        RUTA_EDA / "muletillas_evidencia.csv", index=False
    )
    print(
        "  evidencia de muletillas: muletillas_evidencia.csv "
        f"({len(filas_muletilla)} filas)"
    )

    filas_corr = [
        {
            "lema_anomalo": malo,
            "lema_correcto": bueno,
            "frec_entrevistas_sin_corregir": anomalos_totales[malo],
            "en_stopwords_spacy": bueno in en_spacy,
        }
        for malo, bueno in sorted(anomalos_map.items())
    ]
    pd.DataFrame(filas_corr).to_csv(
        RUTA_EDA / "lemas_anomalos_candidatos.csv", index=False
    )
    print("  evidencia de correcciones: lemas_anomalos_candidatos.csv")

    ctx["lemas_archivo_candidatas"] = lemas_archivo
    ctx["crudo_candidatas"] = crudo_ent
    ctx.update(
        {
            "vocabularios": vocabularios,
            "documentos": documentos,
            "contadores": contadores,
            "df_libros": df_libros,
            "evidencia_muletillas": filas_muletilla,
            "evidencia_correcciones": filas_corr,
        }
    )
    return ctx


# ---------------------------------------------------------------------------
# Etapa 3: diversidad lexica TTR muestreado + Yule's K (arquitectura e.2)
# ---------------------------------------------------------------------------


def ttr_muestreado(
    doc_ids: np.ndarray, rng: np.random.Generator
) -> tuple[float, bool]:
    """TTR con muestra fija de N_MUESTRA_TTR tokens (Bestgen 2023).

    Returns:
        Tupla (TTR de la muestra, True si el doc fue mas corto que la muestra).
    """
    if len(doc_ids) >= N_MUESTRA_TTR:
        muestra = rng.choice(doc_ids, size=N_MUESTRA_TTR, replace=False)
        return len(np.unique(muestra)) / N_MUESTRA_TTR, False
    if len(doc_ids) == 0:
        return float("nan"), True
    return len(np.unique(doc_ids)) / len(doc_ids), True


def yule_k(doc_ids: np.ndarray) -> float:
    """Yule's K (x10^4): 10^4 (sum_v v^2 f_v - N) / N^2 de un documento.

    Valores altos indican menor diversidad lexica (Bestgen 2023).
    """
    n = len(doc_ids)
    if n == 0:
        return float("nan")
    frecuencias = np.bincount(doc_ids)
    frecuencias = frecuencias[frecuencias > 0].astype(np.float64)
    s1 = float(np.sum(frecuencias**2))
    return 1e4 * (s1 - n) / (n * n)


def etapa_diversidad(ctx: dict[str, Any]) -> dict[str, Any]:
    """TTR muestreado y Yule's K por documento; violin por corpus y CSV."""
    print("\n[3/7] Diversidad lexica (TTR muestreado n=1000 y Yule's K)")
    filas: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEMILLA)
    resumen: dict[str, dict[str, float]] = {}

    for nombre in CORPUS:
        metas = metadatos_corpus(nombre)
        documentos = ctx["documentos"][nombre]
        ttrs: list[float] = []
        yules: list[float] = []
        parciales = 0
        for idx, doc in enumerate(documentos):
            ttr, parcial = ttr_muestreado(doc, rng)
            ky = yule_k(doc)
            ttrs.append(ttr)
            yules.append(ky)
            parciales += int(parcial)
            filas.append(
                {
                    "corpus": nombre,
                    "idx": idx,
                    "id": metas[idx]["id"],
                    "tomo": metas[idx].get("tomo", ""),
                    "n_tokens_contenido": len(doc),
                    "ttr_m1000": None if np.isnan(ttr) else round(ttr, 4),
                    "ttr_muestra_parcial": parcial or None,
                    "yule_k": None if np.isnan(ky) else round(ky, 2),
                }
            )
        arr_yule = np.asarray(yules, dtype=float)
        resumen[nombre] = {
            "ttr_m1000_media": float(np.nanmean(ttrs)),
            "ttr_m1000_mediana": float(np.nanmedian(ttrs)),
            "yule_k_media": float(np.nanmean(arr_yule)),
            "yule_k_mediana": float(np.nanmedian(arr_yule)),
            "n_docs_ttr_parcial": parciales,
        }
        print(
            f"  {nombre}: TTR_m1000 media={resumen[nombre]['ttr_m1000_media']:.3f} "
            f"| Yule K media={resumen[nombre]['yule_k_media']:.1f} | "
            f"TTR parciales={parciales}"
        )
    pd.DataFrame(filas).to_csv(RUTA_EDA / "diversidad_lexica.csv", index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, clave, titulo, formato in [
        (ejes[0], "ttr_m1000", "TTR muestreado (n=1000 tokens)", ".3f"),
        (ejes[1], "yule_k", "Yule's K (x10^4; menor = mas diverso)", ".0f"),
    ]:
        datos = [
            np.asarray(
                [
                    f[clave]
                    for f in filas
                    if f["corpus"] == c and f[clave] is not None
                ],
                dtype=float,
            )
            for c in CORPUS
        ]
        partes = ax.violinplot(datos, showmedians=True, showextrema=False)
        for cuerpo, color in zip(
            partes["bodies"], ["#4C72B0", "#DD8452"], strict=True
        ):
            cuerpo.set_facecolor(color)
            cuerpo.set_alpha(0.7)
        ax.set_xticks([1, 2], CORPUS)
        ax.set_title(titulo)
        for pos, arr in enumerate(datos, start=1):
            ax.text(
                pos + 0.08,
                float(np.median(arr)),
                f"{np.median(arr):{formato}}",
                va="center",
                fontsize=8,
            )
    fig.suptitle("Diversidad lexica por corpus (lemas de contenido)", y=1.02)
    fig.tight_layout()
    guardar_figura(fig, "eda_diversidad_lexica.png")

    ctx["diversidad"] = {"filas": filas, "resumen": resumen}
    return ctx


# ---------------------------------------------------------------------------
# Etapa 4: nubes de palabras uni/bi/trigramas (arquitectura e.4)
# ---------------------------------------------------------------------------


def etapa_nubes(ctx: dict[str, Any]) -> dict[str, Any]:
    """Genera 3 nubes por corpus (uni/bi/trigramas) desde las frecuencias."""
    print("\n[4/7] Nubes de palabras uni/bi/trigramas")
    max_palabras = 150
    for nombre in CORPUS:
        for n, tipo in ((1, "unigramas"), (2, "bigramas"), (3, "trigramas")):
            if n == 1:
                frecuencias: Counter[str] = ctx["contadores"][nombre]
            else:
                frecuencias = ngramas_counter_top(
                    ctx["documentos"][nombre],
                    n,
                    ctx["vocabularios"][nombre],
                    max_palabras * 3,
                )
            fig, ax = plt.subplots(figsize=(10, 6))
            if frecuencias:
                nube = WordCloud(
                    width=1200,
                    height=800,
                    background_color="white",
                    max_words=max_palabras,
                    collocations=False,
                    random_state=SEMILLA,
                )
                nube.generate_from_frequencies(dict(frecuencias))
                ax.imshow(nube, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(
                f"{nombre}: {tipo} mas frecuentes (lemas de contenido, "
                f"top {max_palabras})",
                fontsize=11,
            )
            guardar_figura(fig, f"eda_{nombre}_nube_{tipo}.png")
        print(f"  {nombre}: 3 nubes generadas")
    return ctx


# ---------------------------------------------------------------------------
# Etapa 5: near-duplicates entre entrevistas (arquitectura e.5)
# ---------------------------------------------------------------------------


def hash_shingles(ventanas: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Hash de 64 bits por ventana: w @ pesos mod 2^64 (wrap de uint64).

    Mezcla tipo hashing universal; suficiente para LSH aproximado. Los pares
    candidatos se verifican despues con Jaccard exacto sobre los shingles.
    """
    return ventanas.astype(np.uint64) @ pesos.astype(np.uint64)


def firma_minhash(doc: np.ndarray, pesos: np.ndarray) -> tuple[int, ...]:
    """Firma minhash (N_HASHES_MINHASH) de un documento codificado."""
    if len(doc) < TAMANO_SHINGLE:
        return (0,) * N_HASHES_MINHASH
    ventanas = np.lib.stride_tricks.sliding_window_view(doc, TAMANO_SHINGLE)
    minimos = hash_shingles(ventanas, pesos).min(axis=0)
    return tuple(int(v) for v in minimos)


def clave_shingles(
    doc: np.ndarray, pesos_a: np.ndarray, pesos_b: np.ndarray
) -> frozenset[int]:
    """Conjunto de shingles del doc con doble hash de 64 bits (128 total)."""
    if len(doc) < TAMANO_SHINGLE:
        return frozenset()
    ventanas = np.lib.stride_tricks.sliding_window_view(doc, TAMANO_SHINGLE)
    h1 = hash_shingles(ventanas, pesos_a).ravel()
    h2 = hash_shingles(ventanas, pesos_b).ravel()
    return frozenset(
        (int(a) << 64) | int(b) for a, b in zip(h1, h2, strict=True)
    )


def etapa_near_duplicates(ctx: dict[str, Any]) -> dict[str, Any]:
    """Detecta pares casi-duplicados de entrevistas (Jaccard de shingles).

    Pipeline (Broder 1997): minhash k=60 + LSH de bandas (20 x 3) para
    generar pares candidatos y verificacion EXACTA con Jaccard sobre los
    conjuntos de shingles de 5 lemas. Los pares con Jaccard > 0.85 se reportan
    en CSV y figura SIN eliminar del corpus (conservado integro).
    """
    print("\n[5/7] Near-duplicates entre entrevistas (Jaccard de shingles=5)")
    nombre = "entrevistas"
    documentos = ctx["documentos"][nombre]
    metas = metadatos_corpus(nombre)
    rng = np.random.default_rng(SEMILLA)
    pesos = rng.integers(
        0, 2**63, size=(TAMANO_SHINGLE, N_HASHES_MINHASH), dtype=np.uint64
    )
    pesos_b = rng.integers(0, 2**63, size=(TAMANO_SHINGLE, 1), dtype=np.uint64)

    indices_largos = [
        i for i, doc in enumerate(documentos) if len(doc) >= MIN_LEMAS_NEAR_DUP
    ]
    firmas = {i: firma_minhash(documentos[i], pesos) for i in indices_largos}
    print(f"  docs considerados (>= {MIN_LEMAS_NEAR_DUP} lemas): {len(indices_largos)}")

    candidatos: set[tuple[int, int]] = set()
    ancho = N_HASHES_MINHASH // N_BANDAS
    for bando in range(N_BANDAS):
        buckets: dict[tuple[int, ...], list[int]] = {}
        for i in indices_largos:
            clave = firmas[i][bando * ancho : (bando + 1) * ancho]
            buckets.setdefault(clave, []).append(i)
        for grupo in buckets.values():
            for a in range(len(grupo)):
                for c in range(a + 1, len(grupo)):
                    i1, i2 = grupo[a], grupo[c]
                    candidatos.add((min(i1, i2), max(i1, i2)))
    print(f"  pares candidatos (LSH bandas 20x3): {len(candidatos)}")

    @lru_cache(maxsize=8192)
    def _shingles(idx: int) -> frozenset[int]:
        return clave_shingles(documentos[idx], pesos[:, :1], pesos_b)

    filas: list[dict[str, Any]] = []
    for i1, i2 in sorted(candidatos):
        s1, s2 = _shingles(i1), _shingles(i2)
        if not s1 or not s2:
            continue
        inter = len(s1 & s2)
        jaccard = inter / (len(s1) + len(s2) - inter)
        filas.append(
            {
                "id_a": metas[i1]["id"],
                "id_b": metas[i2]["id"],
                "jaccard_shingles5": round(jaccard, 4),
                "n_lemas_contenido_a": len(documentos[i1]),
                "n_lemas_contenido_b": len(documentos[i2]),
                "near_dup_mayor_0.85": bool(jaccard > UMBRAL_NEAR_DUP),
            }
        )
    filas.sort(key=lambda f: -f["jaccard_shingles5"])
    df_pares = pd.DataFrame(filas)
    df_pares.to_csv(RUTA_EDA / "near_duplicates_entrevistas.csv", index=False)
    n_near = (
        int(df_pares["near_dup_mayor_0.85"].sum()) if len(df_pares) else 0
    )
    print(f"  pares verificados: {len(filas)} | Jaccard>0.85: {n_near}")

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(filas):
        valores = np.asarray([f["jaccard_shingles5"] for f in filas])
        ejes[0].hist(valores, bins=40, color="#4C72B0", edgecolor="white")
        ejes[0].axvline(UMBRAL_NEAR_DUP, color="darkred", ls="--", lw=1.2)
        ejes[0].set_title(
            "Jaccard de shingles (5 lemas) entre pares candidatos\n"
            f"pares > {UMBRAL_NEAR_DUP}: {n_near} (no eliminados)"
        )
        ejes[0].set_xlabel("Jaccard")
        ejes[0].set_ylabel("pares")
        top = df_pares.head(10)
        ejes[1].barh(
            np.arange(len(top)),
            top["jaccard_shingles5"],
            color="#DD8452",
        )
        ejes[1].set_yticks(
            np.arange(len(top)),
            [
                f"{r['id_a'][:12]}.. / {r['id_b'][:12]}.."
                for _, r in top.iterrows()
            ],
            fontsize=7,
        )
        ejes[1].set_xlabel("Jaccard")
        ejes[1].set_title("Top-10 pares casi identicos (se conservan)")
        ejes[1].invert_yaxis()
    else:
        ejes[0].text(0.5, 0.5, "sin pares candidatos", ha="center")
        ejes[1].axis("off")
    fig.tight_layout()
    guardar_figura(fig, "eda_near_duplicates_entrevistas.png")

    ctx["near_duplicates"] = {
        "filas": filas,
        "n_pares": len(filas),
        "n_pares_mayor_0_85": n_near,
    }
    return ctx


# ---------------------------------------------------------------------------
# Etapa 6: comparacion lexica libros <-> entrevistas (arquitectura e.6)
# ---------------------------------------------------------------------------


def log_odds_distintivos(
    contador_a: Counter[str],
    contador_b: Counter[str],
    top: int,
    alfa: float,
    min_apoyo: int,
) -> tuple[list[tuple[str, float, int, int]], list[tuple[str, float, int, int]]]:
    """Terminos distintivos por log-odds con previo dirichlet (Monroe 2008).

    score(w) = log((c_a + alfa) / (N_a + alfa V)) -
               log((c_b + alfa) / (N_b + alfa V)),  con V tipos de la union.
    score > 0: distintivo de A; score < 0: distintivo de B. Se exige apoyo
    conjunto c_a + c_b >= min_apoyo (evita rarezas sin respaldo).

    Returns:
        Tupla (distintivos de A, distintivos de B) ordenadas por |log-odds|,
        cada una como (token, |log-odds|, c_a, c_b).
    """
    n_a = sum(contador_a.values())
    n_b = sum(contador_b.values())
    tipos_union = set(contador_a) | set(contador_b)
    v = len(tipos_union)
    puntuados: list[tuple[str, float, int, int]] = []
    for token in tipos_union:
        ca = contador_a.get(token, 0)
        cb = contador_b.get(token, 0)
        if ca + cb < min_apoyo:
            continue
        score = np.log((ca + alfa) / (n_a + alfa * v)) - np.log(
            (cb + alfa) / (n_b + alfa * v)
        )
        puntuados.append((token, float(score), ca, cb))
    puntuados.sort(key=lambda t: -t[1])
    distintos_a = [(t, s, ca, cb) for t, s, ca, cb in puntuados if s > 0][:top]
    distintos_b = [
        (t, -s, ca, cb)
        for t, s, ca, cb in reversed(puntuados)
        if s < 0
    ][:top]
    return distintos_a, distintos_b


def etapa_comparacion(ctx: dict[str, Any]) -> dict[str, Any]:
    """Solapamiento de vocabulario y distintivos por log-odds (CSV + figura).

    Se excluye RUIDO_LIBROS (fragmentos tipograficos de palabras partidas por
    el guion del PDF y ruido bibliografico tipo 'ibid.') de ambos analisis.
    """
    print("\n[6/7] Comparacion lexica libros <-> entrevistas")
    ca_raw = ctx["contadores"]["libros"]
    cb = ctx["contadores"]["entrevistas"]
    ca = Counter({t: f for t, f in ca_raw.items() if t not in RUIDO_LIBROS})
    n_ruido = len(ca_raw) - len(ca)
    tipos_libros = set(ca)
    tipos_entrevistas = set(cb)
    compartidos = tipos_libros & tipos_entrevistas
    union = tipos_libros | tipos_entrevistas

    solapamiento = {
        "tipos_libros_sin_ruido": len(tipos_libros),
        "tipos_ruido_excluidos": n_ruido,
        "tipos_entrevistas": len(tipos_entrevistas),
        "tipos_compartidos": len(compartidos),
        "tipos_union": len(union),
        "jaccard_vocabulario": round(len(compartidos) / len(union), 4),
        "cobertura_tokens_entrevistas_pct": round(
            100 * sum(cb[t] for t in compartidos) / sum(cb.values()), 2
        ),
        "cobertura_tokens_libros_pct": round(
            100 * sum(ca[t] for t in compartidos) / sum(ca.values()), 2
        ),
    }
    pd.DataFrame([solapamiento]).to_csv(
        RUTA_EDA / "solapamiento_vocabulario.csv", index=False
    )
    print(
        f"  solapamiento: {len(compartidos):,}/{len(union):,} tipos "
        f"(Jaccard {solapamiento['jaccard_vocabulario']:.3f})"
    )

    distintos_a, distintos_b = log_odds_distintivos(
        ca, cb, N_DISTINTIVOS, ALFA_LOGODDS, MIN_APOYO_LOGODDS
    )
    filas = [
        {
            "direccion": "libros",
            "token": t,
            "log_odds": round(s, 3),
            "frec_libros": fl,
            "frec_entrevistas": fe,
        }
        for t, s, fl, fe in distintos_a
    ] + [
        {
            "direccion": "entrevistas",
            "token": t,
            "log_odds": round(-s, 3),
            "frec_libros": fl,
            "frec_entrevistas": fe,
        }
        for t, s, fl, fe in distintos_b
    ]
    df_dist = pd.DataFrame(filas)
    df_dist.to_csv(RUTA_EDA / "distintivos_logodds.csv", index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, lista, titulo, color in [
        (
            ejes[0],
            list(reversed(distintos_a)),
            "Distintivos de LIBROS (log-odds > 0)",
            "#4C72B0",
        ),
        (
            ejes[1],
            list(reversed(distintos_b)),
            "Distintivos de ENTREVISTAS (log-odds < 0)",
            "#DD8452",
        ),
    ]:
        if lista:
            etiquetas = [t for t, *_ in lista]
            valores = [s for _, s, *_ in lista]
            ax.barh(np.arange(len(valores)), valores, color=color)
            ax.set_yticks(np.arange(len(valores)), etiquetas, fontsize=8)
            ax.axvline(0, color="black", lw=0.6)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("log-odds (previo dirichlet 0.5)")
    fig.suptitle(
        "Terminos distintivos libros vs entrevistas (lemas de contenido)",
        y=1.01,
    )
    fig.tight_layout()
    guardar_figura(fig, "eda_comparacion_distintivos_logodds.png")

    ctx["comparacion"] = {
        "solapamiento": solapamiento,
        "distintivos_a": distintos_a,
        "distintivos_b": distintos_b,
        "distintivos_csv": df_dist,
    }
    return ctx


# ---------------------------------------------------------------------------
# Etapa 7: tesauro CEV (ADR-004, lectura cualitativa opcional)
# ---------------------------------------------------------------------------


def cargar_etiquetas_tesauro() -> dict[str, list[str]]:
    """Cache de etiquetas SKOS (prefLabel/altLabel es) del tesauro CEV.

    Returns:
        Mapa etiqueta normalizada (minusculas, sin parentesis) -> URIs.
    """
    cache = RUTA_EDA / "tesauro_cache_etiquetas.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    ruta_rdf = RUTA_PROYECTO / "data" / "raw" / "Resource-tesauro.rdf"
    if not ruta_rdf.exists():
        print("  aviso: no existe data/raw/Resource-tesauro.rdf; se omite")
        return {}
    import rdflib

    print("  parseando Resource-tesauro.rdf (5 MB, rdflib)...")
    grafo = rdflib.Graph()
    grafo.parse(ruta_rdf, format="xml")
    pref = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
    alt = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#altLabel")
    etiquetas: dict[str, list[str]] = {}
    for predicado in (pref, alt):
        for sujeto, _, objeto in grafo.triples((None, predicado, None)):
            if getattr(objeto, "language", None) != "es":
                continue
            normalizada = re.sub(r"\(.*?\)", "", str(objeto)).strip().lower()
            if normalizada:
                etiquetas.setdefault(normalizada, []).append(str(sujeto))
    escribir_json(etiquetas, cache)
    print(f"  etiquetas SKOS es cacheadas: {len(etiquetas)}")
    return etiquetas


def _match_tesauro(token: str, etiquetas: dict[str, list[str]]) -> list[str]:
    """Busca el token entre etiquetas (exacta o prefijo de palabra >= 4)."""
    coincide: list[str] = []
    if token in etiquetas:
        coincide.append(token)
    for etiqueta in etiquetas:
        if etiqueta == token:
            continue
        if any(
            palabra.startswith(token) and len(token) >= 4
            for palabra in etiqueta.split()
        ):
            coincide.append(etiqueta)
        if len(coincide) >= 3:
            break
    return coincide


def etapa_tesauro(ctx: dict[str, Any]) -> dict[str, Any]:
    """Cruza los distintivos con el tesauro CEV (interpretacion cualitativa)."""
    print("\n[7/7] Tesauro CEV (ADR-004): cruce cualitativo con distintivos")
    if "comparacion" not in ctx:
        raise RuntimeError("ejecutar la etapa comparacion antes que tesauro")
    etiquetas = cargar_etiquetas_tesauro()
    if not etiquetas:
        ctx["tesauro"] = {"etiquetas": 0, "n_distintivos_encontrados": 0}
        return ctx
    filas = ctx["comparacion"]["distintivos_csv"].copy()
    coincide_con: list[str] = []
    for _, fila in filas.iterrows():
        matches = _match_tesauro(fila["token"], etiquetas)
        coincide_con.append(" | ".join(matches[:3]) if matches else "")
    filas["tesauro_etiquetas"] = coincide_con
    filas.to_csv(RUTA_EDA / "distintivos_logodds.csv", index=False)
    # El DataFrame enriquecido se devuelve al contexto para el resumen.
    ctx["comparacion"]["distintivos_csv"] = filas
    n_con_tesauro = int((filas["tesauro_etiquetas"] != "").sum())
    print(f"  distintivos con etiqueta en el tesauro: {n_con_tesauro}/50")
    ctx["tesauro"] = {
        "etiquetas": len(etiquetas),
        "n_distintivos_encontrados": n_con_tesauro,
    }
    return ctx


# ---------------------------------------------------------------------------
# Etapa 8: congelacion de config (salida formal del EDA)
# ---------------------------------------------------------------------------


def aplicar_reglas_muletillas(
    ctx: dict[str, Any],
    lemas_archivo: dict[str, int],
    crudo_ent: dict[str, int],
) -> list[str]:
    """Aplica las reglas de MULETILLAS_ACTUALES/NUEVAS sobre la evidencia.

    Args:
        ctx: contexto del EDA (no usado directamente; se mantiene por firma).
        lemas_archivo: frecuencia de cada candidata como lema en el archivo
            sin parada (evidencia estable).
        crudo_ent: frecuencia de superficie en tokens crudos.

    Returns:
        Lista final de tokens para config/stopwords_habla.txt.
    """
    finales: list[str] = []
    decisiones: list[dict[str, Any]] = []
    for token, motivo, umbral_lema, umbral_crudo in (
        MULETILLAS_ACTUALES + MULETILLAS_NUEVAS
    ):
        f_lema = lemas_archivo.get(token, 0)
        f_crudo = crudo_ent.get(token, 0)
        pasa_lema = umbral_lema is not None and f_lema >= umbral_lema
        pasa_crudo = umbral_crudo is not None and f_crudo >= umbral_crudo
        incluir = bool(pasa_lema or pasa_crudo)
        decisiones.append(
            {
                "token": token,
                "motivo": motivo,
                "frec_lema_archivo_sin_parada": f_lema,
                "frec_superficie_cruda": f_crudo,
                "incluir": incluir,
            }
        )
        if incluir:
            finales.append(token)
    pd.DataFrame(decisiones).to_csv(
        RUTA_EDA / "muletillas_decision.csv", index=False
    )
    return finales


def escribir_stopwords_final(finales: list[str]) -> None:
    """Escribe config/stopwords_habla.txt con la lista congelada y evidencia."""
    lineas = [
        "# stopwords_habla.txt - lista CONGELADA en el EDA (Fase 5, 2026-09-04).",
        "# Uso: parada en consumo sobre lemas = spacy.lang.es.STOP_WORDS + este",
        "# archivo (ver scripts/03_eda.py y config/tfidf_config.json para F6).",
        "# Regla: cada token se incluye si supera su umbral de evidencia en",
        "# entrevistas (frecuencia como lema o como token crudo); la evidencia",
        "# esta en resultados/eda/muletillas_evidencia.csv y",
        "# resultados/eda/muletillas_decision.csv.",
        "# Nota Fase 5: las stopwords SI se aplicaron en la Fase 4 por",
        "# superficie, pero *_lemas.txt conserva residuales funcionales",
        "# (formas capitalizadas y verbos/cliticos no marcados); la parada en",
        "# consumo sobre lemas los elimina (p. ej. 'el' 122K en entrevistas).",
    ]
    (RUTA_CONFIG / "stopwords_habla.txt").write_text(
        "\n".join(lineas + finales) + "\n", encoding="utf-8"
    )
    print(f"  config/stopwords_habla.txt: {len(finales)} tokens congelados")


def escribir_tfidf_config(ctx: dict[str, Any]) -> None:
    """Escribe config/tfidf_config.json con la propuesta justificada para F6."""
    df = ctx.get("df_libros")
    if df is None or len(df) == 0:
        raise RuntimeError("no hay df_libros; ejecutar la etapa frecuencias antes")
    n_docs = len(ctx["por_doc"]["libros"]["contenido"])
    hapax = int((df == 1).sum())
    sobre_max = int((df / n_docs > 0.85).sum())
    estadisticas = {
        "n_docs_coleccion": n_docs,
        "n_tipos_unigramas": len(df),
        "n_tipos_df1_hapax": hapax,
        "hapax_pct": round(100 * hapax / len(df), 2),
        "media_docs_por_tipo": round(float(df.mean()), 2),
        "mediana_docs_por_tipo": int(np.median(df)),
        "n_tipos_en_mas_del_85pct_docs": sobre_max,
        "tipos_en_mas_del_85pct_pct": round(100 * sobre_max / len(df), 4),
    }
    config = {
        "_comentario": "Configuracion TF-IDF propuesta para la Fase 6, "
        "congelada en el EDA (Fase 5) con evidencia de resultados/eda/. Se "
        "aplica sobre los textos pre-lematizados (1 doc por linea) con las "
        "correcciones de config/lemas_correcciones.json y la parada "
        "spacy.lang.es.STOP_WORDS + config/stopwords_habla.txt (pasar esos "
        "tokens a 'stop_words' del vectorizador).",
        "vectorizador": {
            "analyzer": "word",
            "token_pattern": r"(?u)\b\w+\b",
            "lowercase": True,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
            "min_df": 2,
            "max_df": 0.85,
        },
        "justificacion_eda": {
            "min_df_2": (
                f"el {estadisticas['hapax_pct']} % de los tipos unicos de la "
                "coleccion son hapax (df=1, "
                f"{estadisticas['n_tipos_df1_hapax']:,} de "
                f"{estadisticas['n_tipos_unigramas']:,}); min_df=2 descarta "
                "tipos de una sola aparicion. Los bigramas son mas dispersos "
                "aun (ADR-005): min_df=2 tambien los poda."
            ),
            "max_df_0_85": (
                f"solo {estadisticas['n_tipos_en_mas_del_85pct_docs']} tipos "
                f"aparecen en >85 % de los {n_docs} segmentos "
                "(terminos ubiquos sin poder discriminativo); max_df=0.85 los "
                "elimina."
            ),
            "ngram_range_1_2": (
                "colocaciones de dominio (desaparicion forzada, violencia "
                "sexual, falsos positivos) con coste controlado por min_df=2; "
                "los trigramas del EDA no muestran colocaciones estables "
                "(ver frecuencias_*_trigramas_top30.csv)."
            ),
            "sublinear_tf_y_norm": (
                "saturacion de tf (Salton y Buckley 1988) y norma l2 "
                "(Manning et al. 2008), ADR-005."
            ),
        },
        "estadisticas_coleccion": estadisticas,
        "stop_words": "tokens de config/stopwords_habla.txt",
        "ruido_tipografico_libros_excluido_en_eda": sorted(RUIDO_LIBROS),
        "nota_ruido": "los tokens anteriores son fragmentos de palabras "
        "partidas por el guion de fin de linea del PDF (p. ej. 'cion' de "
        "'viola-cion') y ruido bibliografico ('ibid.', titulos en ingles) que "
        "escapo al filtro alfanumerico de la Fase 4; se excluyeron de los "
        "analisis lexicos del EDA y se recomienda pasarlos a stop_words en la "
        "Fase 6 si superan min_df=2 en la coleccion.",
    }
    escribir_json(config, RUTA_CONFIG / "tfidf_config.json")
    print("  config/tfidf_config.json escrito (min_df=2, max_df=0.85)")


def escribir_resumen(ctx: dict[str, Any]) -> None:
    """Escribe resultados/eda/resumen_eda.md (objetivo <= 60 lineas)."""
    est = {f["corpus"]: f for f in ctx["estadisticas"]}
    div = ctx["diversidad"]["resumen"]
    comp = ctx["comparacion"]
    nd = ctx["near_duplicates"]
    sol = comp["solapamiento"]
    top_a = comp["distintivos_a"][:5]
    top_b = comp["distintivos_b"][:5]
    tesauro = ctx.get("tesauro", {})

    lineas = [
        "# Resumen EDA (Fase 5) - corpus CEV",
        "",
        (
            "- Fecha: 2026-09-04 | Script: scripts/03_eda.py | Corpus: 917 "
            "segmentos de libro (10 tomos) y 2.486 entrevistas."
        ),
        "",
        "## 1. Tamano y longitudes",
        (
            f"1. Libros: {est['libros']['n_docs']} docs, "
            f"{est['libros']['lemas_contenido_eda']:,} lemas de contenido "
            f"({est['libros']['tokens_crudos_spacy']:,} tokens crudos spaCy), "
            f"{est['libros']['tipos_contenido_eda']:,} tipos; media "
            f"{est['libros']['lemas_doc_media']:.0f} lemas/segmento "
            f"(min {est['libros']['lemas_doc_min']:.0f}, max "
            f"{est['libros']['lemas_doc_max']:.0f})."
        ),
        (
            f"2. Entrevistas: {est['entrevistas']['n_docs']} docs, "
            f"{est['entrevistas']['lemas_contenido_eda']:,} lemas de contenido "
            f"({est['entrevistas']['tokens_crudos_spacy']:,} tokens crudos), "
            f"{est['entrevistas']['tipos_contenido_eda']:,} tipos; media "
            f"{est['entrevistas']['lemas_doc_media']:.0f} lemas/entrevista "
            f"(min {est['entrevistas']['lemas_doc_min']:.0f}, max "
            f"{est['entrevistas']['lemas_doc_max']:.0f}); chars normalizados "
            f"min-max {min(ctx['chars_entrevistas']):,}-"
            f"{max(ctx['chars_entrevistas']):,} (el texto crudo iba de 3 a 850K)."
        ),
        (
            "3. Asimetria extrema en entrevistas (media muy superior a la "
            "mediana; p99 recorta la cola hasta ~50K lemas). 2 docs vacios "
            "legitimos documentados en longitudes_por_doc.csv (no se eliminan)."
        ),
        "",
        "## 2. Diversidad lexica (lemas de contenido)",
        (
            f"4. TTR muestreado n=1000: libros media "
            f"{div['libros']['ttr_m1000_media']:.3f} vs entrevistas "
            f"{div['entrevistas']['ttr_m1000_media']:.3f}."
        ),
        (
            f"5. Yule's K: libros {div['libros']['yule_k_media']:.0f} vs "
            f"entrevistas {div['entrevistas']['yule_k_media']:.0f} (menor = "
            "mas diverso): la oralidad transcrita es lexicamente mas "
            "repetitiva (ruido sistematico, Moreno-Sandoval 2024)."
        ),
        "",
        "## 3. Frecuencias y nubes",
        (
            "6. Top-10 unigramas libros: "
            + ", ".join(t for t, _ in ctx["contadores"]["libros"].most_common(10))
            + ". Entrevistas: "
            + ", ".join(
                t for t, _ in ctx["contadores"]["entrevistas"].most_common(10)
            )
            + " (verbo-narracion oral)."
        ),
        (
            "7. Nubes uni/bi/trigramas: 6 figuras (3 por corpus) en "
            "imagenes/; tablas top-30 en resultados/eda/frecuencias_*.csv."
        ),
        "",
        "## 4. Near-duplicates entre entrevistas",
        (
            f"8. {nd['n_pares_mayor_0_85']} pares con Jaccard de shingles(5) "
            f"> 0.85 (de {nd['n_pares']} pares candidatos LSH verificados con "
            "Jaccard exacto). Son casi-duplicados reales del corpus: se "
            "REPORTAN sin eliminar (near_duplicates_entrevistas.csv)."
        ),
        "",
        "## 5. Comparacion lexica libros <-> entrevistas",
        (
            f"9. Vocabulario: libros {sol['tipos_libros_sin_ruido']:,} vs "
            f"entrevistas "
            f"{sol['tipos_entrevistas']:,} tipos; compartidos "
            f"{sol['tipos_compartidos']:,} "
            f"(Jaccard {sol['jaccard_vocabulario']:.3f}). El vocabulario de "
            f"libros cubre el {sol['cobertura_tokens_entrevistas_pct']:.1f} % "
            f"de los tokens de entrevistas y el de entrevistas el "
            f"{sol['cobertura_tokens_libros_pct']:.1f} % de los de libros."
        ),
        "10. Distintivos de libros (top-5): "
        + ", ".join(f"{t} ({s:.1f})" for t, s, _, _ in top_a)
        + ".",
        "11. Distintivos de entrevistas (top-5): "
        + ", ".join(f"{t} ({s:.1f})" for t, s, _, _ in top_b)
        + ".",
        (
            "12. Hallazgo de calidad en libros: fragmentos de palabras "
            "partidas por el guion de linea del PDF (p. ej. 'cion' 542 oc, "
            "'nes' 189) y ruido bibliografico ('ibid.' 852 oc); se excluyen "
            "del analisis comparativo y se listan en config/tfidf_config.json "
            "para que la Fase 6 decida su tratamiento."
        ),
        (
            "13. Siglas asimetricas: 'far' (519 oc) es distintivo de "
            "entrevistas (testimonios: 'las FAR'); en libros predomina "
            "'farc': desajuste lexico del registro oral a tener en cuenta en "
            "el matching."
        ),
    ]

    n_tesauro = tesauro.get("n_distintivos_encontrados", 0)
    lineas.append("")
    lineas.append("## 6. Tesauro CEV (ADR-004, lectura cualitativa)")
    if n_tesauro:
        con_etiqueta = [
            (row["token"], row["tesauro_etiquetas"])
            for _, row in comp["distintivos_csv"].iterrows()
            if row.get("tesauro_etiquetas", "")
        ]
        lineas.append(
            f"14. {n_tesauro} de 50 distintivos tienen etiqueta SKOS "
            "(prefLabel/altLabel) en el tesauro; ejemplos:"
        )
        for token, etiqueta in con_etiqueta[:4]:
            lineas.append(f"    - '{token}': {etiqueta.split(' | ')[0]} ...")
    else:
        lineas.append(
            "14. Sin coincidencias de distintivos con etiquetas SKOS del "
            "tesauro (o tesauro no disponible); se omite la interpretacion."
        )

    lineas.append("")
    lineas.append("## 7. Congelaciones para la Fase 6")
    lineas.append(
        "- config/stopwords_habla.txt: lista final con evidencia "
        "(muletillas_evidencia.csv y muletillas_decision.csv); parada en "
        "consumo = spacy STOP_WORDS + esta lista."
    )
    lineas.append(
        "- config/lemas_correcciones.json: "
        f"{len(ctx.get('evidencia_correcciones', []))} correcciones de lemas "
        "(corruptos del lematizador, formas de 2a persona sin lematizar y "
        "truncamientos orales; ver el archivo y lemas_anomalos_candidatos.csv)."
    )
    lineas.append(
        "- config/tfidf_config.json: min_df=2, max_df=0.85, "
        "ngram_range=(1,2), sublinear_tf=True, norm=l2 (ADR-005 confirmado "
        "con cifras del EDA)."
    )
    (RUTA_EDA / "resumen_eda.md").write_text(
        "\n".join(lineas) + "\n", encoding="utf-8"
    )
    print(f"  resultados/eda/resumen_eda.md escrito ({len(lineas)} lineas)")
    if len(lineas) > 60:
        print("  aviso: resumen_eda.md supera 60 lineas; revisar")


def actualizar_metricas(ctx: dict[str, Any]) -> None:
    """Anade la entrada fase5_eda a resultados/metricas.json (sin borrar)."""
    ruta = RUTA_PROYECTO / "resultados" / "metricas.json"
    metricas = json.loads(ruta.read_text(encoding="utf-8"))
    est = {f["corpus"]: f for f in ctx["estadisticas"]}
    div = ctx["diversidad"]["resumen"]
    comp = ctx["comparacion"]
    nd = ctx["near_duplicates"]
    tesauro = ctx.get("tesauro", {})
    # Idempotencia: se reemplaza la entrada previa de la misma fase.
    metricas["metricas"] = [
        m for m in metricas.get("metricas", []) if m.get("experimento") != "fase5_eda"
    ]
    entrada = {
        "experimento": "fase5_eda",
        "fase": "5",
        "script": "scripts/03_eda.py",
        "fecha": "2026-09-04",
        "descripcion": "EDA del corpus preprocesado (estadisticas, diversidad "
        "lexica TTR/Yule, frecuencias y nubes uni/bi/tri, near-duplicates, "
        "comparacion libros vs entrevistas por log-odds) y congelacion de "
        "config para la Fase 6",
        "corpus": {
            "libros": {
                "n_docs": est["libros"]["n_docs"],
                "lemas_contenido": est["libros"]["lemas_contenido_eda"],
                "tipos_contenido": est["libros"]["tipos_contenido_eda"],
                "lemas_doc_media": est["libros"]["lemas_doc_media"],
            },
            "entrevistas": {
                "n_docs": est["entrevistas"]["n_docs"],
                "lemas_contenido": est["entrevistas"]["lemas_contenido_eda"],
                "tipos_contenido": est["entrevistas"]["tipos_contenido_eda"],
                "lemas_doc_media": est["entrevistas"]["lemas_doc_media"],
                "docs_vacios_legitimos": est["entrevistas"][
                    "docs_vacios_contenido"
                ],
            },
        },
        "diversidad_lexica": {
            "ttr_m1000_libros": round(div["libros"]["ttr_m1000_media"], 4),
            "ttr_m1000_entrevistas": round(
                div["entrevistas"]["ttr_m1000_media"], 4
            ),
            "yule_k_libros": round(div["libros"]["yule_k_media"], 1),
            "yule_k_entrevistas": round(
                div["entrevistas"]["yule_k_media"], 1
            ),
            "metodo": "TTR con muestra fija n=1000 (Bestgen 2023); Yule K "
            "sobre el documento completo",
        },
        "solapamiento_vocabulario": {
            "jaccard": comp["solapamiento"]["jaccard_vocabulario"],
            "tipos_compartidos": comp["solapamiento"]["tipos_compartidos"],
            "cobertura_tokens_entrevistas_pct": comp["solapamiento"][
                "cobertura_tokens_entrevistas_pct"
            ],
            "cobertura_tokens_libros_pct": comp["solapamiento"][
                "cobertura_tokens_libros_pct"
            ],
        },
        "near_duplicates_entrevistas": {
            "pares_verificados": nd["n_pares"],
            "pares_jaccard_mayor_0_85": nd["n_pares_mayor_0_85"],
            "accion": "reportados sin eliminar (corpus oficial integro)",
            "metodo": "Jaccard de shingles de 5 lemas; minhash k=60 + LSH "
            "20x3 y verificacion exacta de candidatos",
        },
        "tesauro_cev": {
            "etiquetas_skos_es": tesauro.get("etiquetas", 0),
            "distintivos_con_etiqueta": tesauro.get(
                "n_distintivos_encontrados", 0
            ),
            "uso": "cualitativo (ADR-004); no se integra al modelo",
        },
        "config_congelada": {
            "stopwords_habla_tokens": len(cargar_muletillas_config()),
            "tfidf_min_df": 2,
            "tfidf_max_df": 0.85,
            "tfidf_ngram_range": "(1,2)",
            "lemas_corregidos": len(cargar_correcciones()),
        },
        "anomalias_documentadas": [
            "2 docs de entrevista vacios legitimos (no eliminados)",
            "near-duplicates >0.85 reportados, no eliminados",
            (
                "lemas funcionales residuales en *_lemas.txt (p. ej. 'el' "
                "122K, 'haber' 35.5K en entrevistas) por filtrado de "
                "superficie en la Fase 4; parada en consumo sobre lemas en "
                "el EDA y F6"
            ),
            (
                "lemas corruptos del lematizador corregidos en consumo "
                "(config/lemas_correcciones.json)"
            ),
        ],
        "estado": "completado",
    }
    metricas.setdefault("metricas", []).append(entrada)
    ruta.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("  resultados/metricas.json: entrada fase5_eda anadida")


def etapa_congelar(ctx: dict[str, Any]) -> dict[str, Any]:
    """Congela configs finales, escribe resumen_eda.md y actualiza metricas."""
    print(
        "\n[8/8] Congelacion de config (stopwords final, tfidf_config, resumen)"
    )
    requeridas = (
        "contadores",
        "diversidad",
        "comparacion",
        "near_duplicates",
        "por_doc",
    )
    faltantes = [k for k in requeridas if k not in ctx]
    if faltantes:
        raise RuntimeError(
            "la etapa congelar requiere el flujo completo; faltan: "
            + ", ".join(faltantes)
        )
    candidatas = (
        set(CANDIDATAS_EVIDENCIA)
        | {t for t, *_ in MULETILLAS_ACTUALES}
        | {t for t, *_ in MULETILLAS_NUEVAS}
    )
    crudo_ent = ctx.get("crudo_candidatas") or conteo_crudo_candidatas(
        RUTA_PREPROCESADO / "entrevistas_tokens.txt", candidatas
    )
    lemas_archivo = ctx.get("lemas_archivo_candidatas")
    if lemas_archivo is None:
        lemas_archivo = conteo_lemas_candidatas(
            RUTA_PREPROCESADO / "entrevistas_lemas.txt", candidatas
        )
    finales = aplicar_reglas_muletillas(ctx, lemas_archivo, crudo_ent)
    escribir_stopwords_final(finales)
    escribir_tfidf_config(ctx)
    escribir_resumen(ctx)
    actualizar_metricas(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ETAPAS_DISPONIBLES = (
    "estadisticas",
    "frecuencias",
    "diversidad",
    "nubes",
    "near_dups",
    "comparacion",
    "tesauro",
    "congelar",
)


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    """Construye el parser de argumentos de la linea de comandos."""
    parser = argparse.ArgumentParser(
        description="EDA del corpus CEV (Fase 5 del TALLER-4)."
    )
    parser.add_argument(
        "--solo",
        nargs="+",
        choices=ETAPAS_DISPONIBLES,
        metavar="ETAPA",
        help="ejecuta solo las etapas indicadas (default: todas)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: ejecuta las etapas del EDA en orden."""
    args = parsear_argumentos(argv)
    preparar_directorios()
    correcciones = cargar_correcciones()
    parada = parada_en_consumo()
    print(
        f"Correcciones de lemas: {len(correcciones)} | "
        f"parada en consumo: {len(parada)} tokens (spaCy + config)"
    )
    pendientes = args.solo if args.solo else ETAPAS_DISPONIBLES
    ctx: dict[str, Any] = {}

    inicio = time.time()
    if "estadisticas" in pendientes:
        ctx.update(etapa_estadisticas(correcciones, parada))
    if "frecuencias" in pendientes:
        ctx.update(etapa_frecuencias(correcciones, parada))
    if "diversidad" in pendientes:
        if "documentos" not in ctx:
            raise RuntimeError(
                "la etapa diversidad requiere la etapa frecuencias"
            )
        ctx = etapa_diversidad(ctx)
    if "nubes" in pendientes:
        if "contadores" not in ctx:
            raise RuntimeError("la etapa nubes requiere la etapa frecuencias")
        ctx = etapa_nubes(ctx)
    if "near_dups" in pendientes:
        if "documentos" not in ctx:
            raise RuntimeError(
                "la etapa near_dups requiere la etapa frecuencias"
            )
        ctx = etapa_near_duplicates(ctx)
    if "comparacion" in pendientes:
        if "contadores" not in ctx:
            raise RuntimeError(
                "la etapa comparacion requiere la etapa frecuencias"
            )
        ctx = etapa_comparacion(ctx)
    if "tesauro" in pendientes:
        ctx = etapa_tesauro(ctx)
    if "congelar" in pendientes:
        ctx = etapa_congelar(ctx)
    print(f"\nFin del EDA en {time.time() - inicio:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
