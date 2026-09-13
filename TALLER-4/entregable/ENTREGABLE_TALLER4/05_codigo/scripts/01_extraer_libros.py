"""01_extraer_libros.py

Fecha: 2026-09-04
Descripcion: Fase 3 del TALLER-4 (IR sobre corpus CEV). Extraccion de texto por
    paginas (pdftotext -layout) y segmentacion de los 10 tomos del Informe Final
    de la Comision de la Verdad siguiendo ADR-001 y la seccion (b) de
    plan/arquitectura.md: parser de TOC por tomo, calibracion de offset
    pagina impresa <-> pagina PDF, matching de titulos TOC contra encabezados
    del cuerpo (con blacklist automatica de running heads/pies repetidos),
    construccion de segmentos (niveles 1-2, fusion <300 palabras, techo 4.000)
    y sanity checks (mapeo >=85 %, cobertura >=90 %, solapamiento 0,
    % fuera de banda <15 %, segmentos vacios 0).

Salidas:
    - data/corpus/paginas/<TOMO>/paginas.jsonl   (texto por pagina)
    - data/corpus/segmentos/<TOMO>.jsonl          (segmentos por tomo)
    - data/corpus/segmentos_libros.jsonl          (segmentos globales)
    - resultados/extraccion_libros_report.json    (reporte con sanity checks)
    - config/toc_tomos.json                       (config por tomo, editable)

Uso:
    python scripts/01_extraer_libros.py                       # tomos que faltan
    python scripts/01_extraer_libros.py --libros HALLAZGOS TESTIMONIAL --force

Estado del arte aplicado (herencia del plan):
    - ADR-001: documento = seccion TOC niveles 1-2 por tomo; fusion <300
      palabras con la siguiente (o anterior); techo 4.000 con fragmentos
      <=3.500 cortando en limite de pagina; id TOMO-p<pagina>-<titulo_normalizado>.
    - arquitectura (b): pdftotext -layout pagina a pagina (paralelo 32 hilos);
      parser de TOC por tomo con config en config/toc_tomos.json.
    Nota de implementacion verificada sobre HALLAZGOS y TESTIMONIAL: la
    sangria del TOC medida en caracteres de pdftotext -layout NO es lineal
    (el grid de celdas depende de la fuente) y el margen izquierdo cambia
    14.2 pt entre paginas pares e impares (recto/verso); por eso el nivel de
    cada entrada se mide con coordenadas x0 reales (PyMuPDF) relativas a la
    linea base de su pagina segun paridad.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

# ============================================================================
# SECCION 0: CONFIGURACION GLOBAL
# ============================================================================

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_ZIP = RUTA_PROYECTO / "data/raw/books-20260904T025322Z-1-001.zip"
DIR_PDFS = RUTA_PROYECTO / "data/raw/books"
DIR_CORPUS = RUTA_PROYECTO / "data/corpus"
DIR_PAGINAS = DIR_CORPUS / "paginas"
DIR_SEGMENTOS = DIR_CORPUS / "segmentos"
RUTA_SEGMENTOS_GLOBAL = DIR_CORPUS / "segmentos_libros.jsonl"
RUTA_REPORTE = RUTA_PROYECTO / "resultados" / "extraccion_libros_report.json"
RUTA_CONFIG_TOC = RUTA_PROYECTO / "config" / "toc_tomos.json"

PDFTOTEXT = "pdftotext"

# Mapeo fijo (miembro del zip -> clave corta del tomo). Los nombres reales
# contienen espacios y prefijos irregulares ("FINAL CEV_..."), por lo que el
# mapeo es explicito, no derivado del nombre.
LIBROS: dict[str, str] = {
    "books/CEV_NNA_DIGITAL_2022.pdf": "NNA",
    "books/CEV_CONVOCATORIAALAPAZGRANDE_DIGITAL_2022.pdf": "CONVOCATORIA",
    "books/CEV_NARRATIVA HISTORICA_DIGITAL_2022.pdf": "NARRATIVA",
    "books/CEV_SUFRIR LA GUERRA Y REHACER LA VIDA_DIGITAL_2022.pdf": "SUFRIR",
    "books/CEV_MI CUERPO ES LA VERDAD_DIGITAL_2022.pdf": "CUERPO",
    "books/CEV_LA COLOMBIA FUERA DE COLOMBIA_DIGITAL_2022.pdf": "FUERA",
    "books/FINAL CEV_HALLAZGOS_DIGITAL_2022.pdf": "HALLAZGOS",
    "books/CEV_VIOLACIONES_DIGITAL_2022.pdf": "VIOLACIONES",
    "books/CEV_ETNICO_DIGITAL_2022.pdf": "ETNICO",
    "books/CEV_TESTIMONIAL_DIGITAL_2022.pdf": "TESTIMONIAL",
}
TOMO_A_MIEMBRO = {v: k for k, v in LIBROS.items()}
ORDEN_TOMOS = list(LIBROS.values())

# Umbrales de sanity (arquitectura (b.5) y ADR-001).
UMBRALES = {
    "mapeo_titulos_min": 0.85,      # (i)  % entradas TOC mapeadas a encabezado
    "cobertura_paginas_min": 0.90,  # (ii) % paginas de contenido cubiertas
    "solapamiento_max": 0,          # (iii) paginas duplicadas entre segmentos
    "fuera_banda_max": 0.15,        # (iv) % segmentos fuera de 300-4000 palabras
    "segmentos_vacios_max": 0,      # (v)  segmentos sin texto util
}

# Parametros por defecto del pipeline (fusion/techo segun ADR-001).
DEFAULTS_TOMO: dict[str, Any] = {
    "niveles_segmento": [1, 2],        # niveles TOC que son limites de segmento
    "fusion_min_palabras": 300,        # secciones mas cortas se fusionan
    "techo_palabras": 4000,            # banda superior objetivo de un segmento
    "fragmento_max_palabras": 3500,    # tamano de fragmento al partir por techo
    "ventana_mapeo_paginas": 1,        # busqueda del encabezado en impresa +- 1
    "blacklist_min_paginas": 3,        # linea repetida en >=N paginas -> running head
    "titulos_descartar": [],           # titulos TOC (norm.) = front/back matter
    "patrones_back_matter": [          # encabezados de cola no listados en TOC
        "referencias", "bibliograf", "fuentes", "anexo", "glosario",
        "siglas", "abreviaturas", "creditos", "colofon", "contraportada",
        "indice", "nota editorial", "acerca de este",
    ],
}

CONFIG_DOCUMENTACION = (
    "Config por tomo de scripts/01_extraer_libros.py. El pipeline escribe aqui "
    "los parametros efectivos usados en la ultima ejecucion (auto-detectados). "
    "Editar este archivo cambia el comportamiento de la siguiente ejecucion: "
    "'niveles_segmento' fija que niveles TOC (1-based, por sangria x0) son "
    "limites de segmento; 'titulos_descartar' recibe titulos TOC normalizados "
    "(minusculas, sin acentos) a descartar como front/back matter; el campo "
    "'_comentario' documenta el motivo de cada ajuste manual."
)

# Patrones de titulos TOC que se descartan como front/back matter aunque
# aparezcan listados (ADR-001: portada, citar, bibliografia, glosario, etc.).
PATRONES_TITULO_DESCARTE = [
    r"^portada", r"^como citar", r"^comisionados", r"^creditos",
    r"^catalogaci[oó]n", r"^tabla de contenido", r"^contenido$",
    r"^referencias", r"^bibliograf", r"^glosario", r"^siglas",
    r"^abreviaturas", r"^anexo", r"^colof[oó]n", r"^contraportada",
    r"^listado", r"^recomendaci[oó]n \d", r"^[íi]ndice", r"^nota editorial",
]

MAX_LINEA_CORTA = 75  # chars: una linea mas larga no es running head/pie

# ----------------------------------------------------------------------------
# Normalizacion y utilidades de texto
# ----------------------------------------------------------------------------


def norm_es(texto: str) -> str:
    """Normaliza texto para comparaciones: minusculas, sin acentos/diacriticos,
    no alfanumericos convertidos a espacio y espacios colapsados.

    Args:
        texto: cadena original (pdftotext o PyMuPDF).

    Returns:
        Cadena normalizada.
    """
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9]+", " ", texto.lower())
    return re.sub(r"\s+", " ", texto).strip()


def sin_numero_esquinas(texto: str) -> str:
    """Quita un numero de 1-4 digitos suelto al inicio o al final de una linea
    (patron de running heads 'NN  titulo' y 'titulo... NN').

    Args:
        texto: linea cruda.

    Returns:
        Linea sin el numero de esquina (strip incluido).
    """
    t = texto.strip()
    t = re.sub(r"^\d{1,4}\s+", "", t)
    t = re.sub(r"\s+\d{1,4}$", "", t)
    return t


def tiene_numero_esquina(texto: str) -> bool:
    """True si la linea empieza o termina con un numero de 1-4 digitos suelto
    (patron de los pies/encabezados con numero de pagina impreso).

    Args:
        texto: linea cruda.

    Returns:
        True si la linea porta numero de esquina.
    """
    t = texto.strip()
    return bool(re.match(r"^\d{1,4}\s+\S", t)) or bool(re.search(r"\S\s+\d{1,4}$", t))


def slug_titulo(titulo: str, maxlen: int = 80) -> str:
    """Convierte un titulo en slug (id de segmento ADR-001).

    Args:
        titulo: titulo original del TOC.
        maxlen: longitud maxima del slug.

    Returns:
        Slug en minusculas con guiones.
    """
    t = unicodedata.normalize("NFKD", titulo)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9]+", "-", t.lower())
    t = re.sub(r"-+", "-", t).strip("-")
    return t[:maxlen].rstrip("-")


def contar_palabras(texto: str) -> int:
    """Cuenta palabras de un texto (cualquier whitespace separa).

    Args:
        texto: texto a contar.

    Returns:
        Numero de palabras.
    """
    return len(texto.split()) if texto else 0


# ----------------------------------------------------------------------------
# SECCION 1: descompresion y extraccion de paginas (pdftotext -layout)
# ----------------------------------------------------------------------------


def descomprimir_pdfs(tomos: list[str]) -> dict[str, Path]:
    """Descomprime del zip solo los PDFs pedidos a data/raw/books/ (idempotente:
    si el PDF ya existe no hace nada; nunca sobrescribe).

    Args:
        tomos: claves cortas de tomo.

    Returns:
        Dict {tomo: Path al pdf}.
    """
    if not RUTA_ZIP.exists():
        raise FileNotFoundError(f"No existe el zip: {RUTA_ZIP}")
    DIR_PDFS.mkdir(parents=True, exist_ok=True)
    resultado: dict[str, Path] = {}
    with zipfile.ZipFile(RUTA_ZIP) as z:
        for tomo in tomos:
            miembro = TOMO_A_MIEMBRO[tomo]
            destino = DIR_PDFS / Path(miembro).name
            if not destino.exists():
                with z.open(miembro) as src, open(destino, "wb") as dst:
                    dst.write(src.read())
            resultado[tomo] = destino
    return resultado


def _pdftotext_pagina(ruta_pdf: Path, pagina: int) -> str:
    """Extrae el texto -layout de una pagina PDF con pdftotext (poppler).

    Args:
        ruta_pdf: ruta del PDF.
        pagina: numero de pagina PDF (1-based).

    Returns:
        Texto de la pagina (UTF-8) sin form feeds.
    """
    proc = subprocess.run(
        [PDFTOTEXT, "-layout", "-f", str(pagina), "-l", str(pagina), str(ruta_pdf), "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return proc.stdout.replace("\x0c", "").replace("\r", "")


def extraer_paginas_tomo(ruta_pdf: Path, tomo: str, n_workers: int = 32, force: bool = False) -> Path:
    """Extrae el texto de todas las paginas de un tomo en paralelo (32 hilos) y
    lo guarda como JSONL en data/corpus/paginas/<TOMO>/paginas.jsonl con
    registros {tomo, pagina_pdf, texto}. Reanudable: si el JSONL existe y
    force=False se omite.

    Args:
        ruta_pdf: PDF del tomo.
        tomo: clave corta del tomo.
        n_workers: hilos para lanzar pdftotext.
        force: re-extrae aunque el JSONL exista.

    Returns:
        Ruta del JSONL de paginas.
    """
    salida_dir = DIR_PAGINAS / tomo
    salida_dir.mkdir(parents=True, exist_ok=True)
    salida = salida_dir / "paginas.jsonl"
    if salida.exists() and not force:
        return salida
    with fitz.open(ruta_pdf) as doc:
        n_paginas = doc.page_count
    tmp = salida.with_suffix(".jsonl.tmp")
    con_orden: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futuros = {pool.submit(_pdftotext_pagina, ruta_pdf, p): p for p in range(1, n_paginas + 1)}
        for futuro, p in futuros.items():
            try:
                con_orden.append((p, futuro.result()))
            except Exception as exc:  # noqa: BLE001 - una pagina no es fatal
                con_orden.append((p, ""))
                print(f"[aviso] pdftotext fallo en pagina {p} de {tomo}: {exc}")
    con_orden.sort()
    with open(tmp, "w", encoding="utf-8") as fh:
        for p, texto in con_orden:
            fh.write(json.dumps({"tomo": tomo, "pagina_pdf": p, "texto": texto}, ensure_ascii=False) + "\n")
    os.replace(tmp, salida)
    return salida


def leer_paginas(tomo: str) -> list[str]:
    """Carga el JSONL de paginas de un tomo.

    Args:
        tomo: clave corta del tomo.

    Returns:
        Lista indexada 0 -> pagina PDF 1 con el texto de cada pagina.
    """
    ruta = DIR_PAGINAS / tomo / "paginas.jsonl"
    if not ruta.exists():
        raise FileNotFoundError(f"Faltan paginas de {tomo}: {ruta}")
    with open(ruta, encoding="utf-8") as fh:
        return [json.loads(linea)["texto"] for linea in fh]


# ----------------------------------------------------------------------------
# SECCION 2: parser del TOC (estructura con coordenadas PyMuPDF)
# ----------------------------------------------------------------------------


def lineas_visuales_pymupdf(ruta_pdf: Path, paginas: list[int]) -> dict[int, dict[str, Any]]:
    """Agrupa las palabras de paginas dadas en lineas visuales (misma banda y)
    con PyMuPDF.

    Args:
        ruta_pdf: PDF a leer.
        paginas: numeros de pagina PDF (1-based).

    Returns:
        {pagina: {"ancho": float, "filas": [{x0, x1, xult, y, texto}]}} con las
        filas ordenadas de arriba a abajo. x0 = borde izquierdo de la fila,
        x1 = borde derecho, xult = x0 de la ultima palabra.
    """
    salida: dict[int, dict[str, Any]] = {}
    with fitz.open(ruta_pdf) as doc:
        for pno in paginas:
            page = doc[pno - 1]
            ancho = page.rect.width
            palabras = [w for w in page.get_text("words") if w[4].strip()]
            lineas: list[dict[str, Any]] = []
            for w in sorted(palabras, key=lambda w: (w[1] + w[3]) / 2):
                yc = (w[1] + w[3]) / 2
                if lineas and abs(yc - lineas[-1]["y"]) <= 4.0:
                    lineas[-1]["palabras"].append(w)
                    centros = [p[1] + p[3] for p in lineas[-1]["palabras"]]
                    lineas[-1]["y"] = sum(centros) / (2 * len(centros))
                else:
                    lineas.append({"y": yc, "palabras": [w]})
            filas: list[dict[str, Any]] = []
            for ln in lineas:
                ln["palabras"].sort(key=lambda w: w[0])
                x0 = min(w[0] for w in ln["palabras"])
                x1 = max(w[2] for w in ln["palabras"])
                xult = ln["palabras"][-1][0]
                texto = " ".join(w[4] for w in ln["palabras"])
                filas.append({"x0": x0, "x1": x1, "xult": xult, "y": ln["y"], "texto": texto})
            salida[pno] = {"ancho": ancho, "filas": filas}
    return salida


def _es_fila_con_numero(fila: dict[str, Any], ancho: float) -> bool:
    """True si la fila visual termina en un numero de la columna derecha
    (x0 de la ultima palabra >= 0.78 del ancho), patron de las entradas del TOC.

    Args:
        fila: fila visual (x0, x1, xult, y, texto).
        ancho: ancho de la pagina en puntos.

    Returns:
        True si la fila es una entrada cerrada con numero de pagina.
    """
    ultima = fila["texto"].strip().split()[-1]
    return bool(re.fullmatch(r"\d{1,4}", ultima)) and fila["xult"] >= 0.78 * ancho


def detectar_paginas_toc(
    ruta_pdf: Path, textos_paginas: list[str], max_inicio: int = 60
) -> list[int]:
    """Detecta las paginas del TOC de un tomo. La primera pagina cuyo primer
    renglon es 'contenido' inicia el TOC; se extiende pagina a pagina sobre las
    filas visuales (PyMuPDF) mientras la pagina siga pareciendo TOC: su primera
    fila termina en numero de la columna derecha, o la pagina anterior dejo una
    entrada abierta (titulo que continua), o la primera fila es un titulo largo
    (>4 tokens) con >=3 filas numeradas en la pagina (entrada que empieza y se
    parte dentro de la pagina). Una pagina con encabezado propio (p.ej.
    'Indice de recomendaciones') detiene la extension.

    Args:
        ruta_pdf: PDF del tomo.
        textos_paginas: textos de las paginas (indice 0 = pdf 1).
        max_inicio: limite de paginas a explorar en busca de 'Contenido'.

    Returns:
        Paginas PDF del TOC ([] si no se detecta).
    """
    inicio = None
    for i, t in enumerate(textos_paginas[:max_inicio]):
        renglones = [r for r in t.splitlines() if r.strip()]
        if renglones and norm_es(renglones[0]) == "contenido":
            inicio = i + 1
            break
    if inicio is None:
        return []
    paginas_toc = [inicio]
    # paginas candidatas tras 'Contenido': leer filas visuales una a una
    prev_filas: list[dict[str, Any]] = []
    vacias = 0
    for pno in range(inicio + 1, len(textos_paginas) + 1):
        filas_pag = lineas_visuales_pymupdf(ruta_pdf, [pno])[pno]
        filas = [f for f in filas_pag["filas"] if norm_es(f["texto"]) != "contenido"]
        if not filas:
            vacias += 1
            if vacias >= 3:
                break
            prev_filas = []
            continue
        vacias = 0
        ancho = filas_pag["ancho"]
        primera = filas[0]
        prim_num = _es_fila_con_numero(primera, ancho)
        n_num = sum(1 for f in filas if _es_fila_con_numero(f, ancho))
        prev_abierta = bool(prev_filas) and not _es_fila_con_numero(prev_filas[-1], ancho)
        primera_larga = len(primera["texto"].split()) > 4
        if prim_num or prev_abierta or (n_num >= 3 and primera_larga):
            paginas_toc.append(pno)
            prev_filas = filas
        else:
            break
    return paginas_toc


def parsear_entradas_toc(ruta_pdf: Path, paginas_toc: list[int]) -> list[dict[str, Any]]:
    """Reconstruye las entradas del TOC a partir de sus filas visuales. Toda
    entrada termina con un numero en la columna derecha (x0 >= 0.78 del ancho);
    los renglones previos sin numero (continuaciones de titulo largo o titulos
    multilinea) pertenecen a esa entrada.

    Args:
        ruta_pdf: PDF del tomo.
        paginas_toc: paginas PDF del TOC.

    Returns:
        Entradas [{titulo, pagina_impresa, pagina_toc, x0, y}] en orden de
        lectura del TOC.
    """
    paginas_data = lineas_visuales_pymupdf(ruta_pdf, paginas_toc)
    entradas: list[dict[str, Any]] = []
    pendiente_texto: list[str] = []   # renglones del titulo en curso
    pendiente_x0: float | None = None
    pendiente_y: float | None = None

    for pno in paginas_toc:
        ancho = paginas_data[pno]["ancho"]
        for fila in paginas_data[pno]["filas"]:
            texto = fila["texto"].strip()
            if not texto:
                continue
            if norm_es(texto) == "contenido":
                continue
            palabras = texto.split()
            # numero de columna derecha cierra la entrada
            if _es_fila_con_numero(fila, ancho):
                ultima = palabras[-1]
                cuerpo = " ".join(palabras[:-1])
                if cuerpo:
                    filas_titulo = [*pendiente_texto, cuerpo]
                    entradas.append(
                        {
                            "titulo": " ".join(filas_titulo).strip(),
                            "lineas": filas_titulo,
                            "pagina_impresa": int(ultima),
                            "pagina_toc": pno,
                            "x0": pendiente_x0 if pendiente_x0 is not None else fila["x0"],
                            "y": pendiente_y if pendiente_y is not None else fila["y"],
                        }
                    )
                elif pendiente_texto:
                    # renglon de solo numero: cierra el titulo pendiente
                    entradas.append(
                        {
                            "titulo": " ".join(pendiente_texto).strip(),
                            "lineas": list(pendiente_texto),
                            "pagina_impresa": int(ultima),
                            "pagina_toc": pno,
                            "x0": pendiente_x0 if pendiente_x0 is not None else fila["x0"],
                            "y": pendiente_y if pendiente_y is not None else fila["y"],
                        }
                    )
                pendiente_texto = []
                pendiente_x0 = None
                pendiente_y = None
            else:
                if not pendiente_texto:
                    pendiente_x0 = fila["x0"]
                pendiente_texto.append(texto)
                if pendiente_y is None:
                    pendiente_y = fila["y"]
    return entradas


def asignar_niveles(entradas: list[dict[str, Any]]) -> None:
    """Asigna 'nivel' (1-based) a cada entrada segun la sangria x0 medida sobre
    la linea base de su pagina. El margen izquierdo del TOC cambia ~14.2 pt
    entre paginas pares e impares (recto/verso), por lo que la base se estima
    por paridad: mediana de los x0 minimos de las paginas TOC de esa paridad.

    Args:
        entradas: entradas del TOC (mutadas in-place con 'nivel').
    """
    if not entradas:
        return
    minimos: dict[int, list[float]] = {0: [], 1: []}
    por_pagina: dict[int, list[float]] = {}
    for e in entradas:
        por_pagina.setdefault(e["pagina_toc"], []).append(e["x0"])
    for pno, xs in por_pagina.items():
        minimos[pno % 2].append(min(xs))

    def mediana(vals: list[float]) -> float:
        vals = sorted(vals)
        return vals[len(vals) // 2]

    base_par = mediana(minimos[0]) if minimos[0] else 56.7
    base_impar = mediana(minimos[1]) if minimos[1] else base_par + 14.2

    rels = sorted({e["x0"] - (base_par if e["pagina_toc"] % 2 == 0 else base_impar) for e in entradas})
    # agrupa sangrias relativas con tolerancia de 8 pt y asigna nivel por orden
    grupos: list[list[float]] = []
    for r in rels:
        if grupos and abs(r - grupos[-1][-1]) <= 8.0:
            grupos[-1].append(r)
        else:
            grupos.append([r])
    nivel_por_rel = {}
    for nivel, grupo in enumerate(grupos, start=1):
        for r in grupo:
            nivel_por_rel[r] = nivel
    for e in entradas:
        rel = e["x0"] - (base_par if e["pagina_toc"] % 2 == 0 else base_impar)
        e["nivel"] = nivel_por_rel[rel]


# ----------------------------------------------------------------------------
# SECCION 3: calibracion de offset y blacklist de running heads
# ----------------------------------------------------------------------------


def calibrar_offset(
    textos_paginas: list[str],
    desde: int = 1,
    muestrear: int = 2,
) -> tuple[int, int, str]:
    """Estima el offset (pagina_impresa - pagina PDF) a partir de los running
    heads/pies que incluyen el numero impreso ('la colombia herida 31' o
    '628  cuando los pajaros no cantaban'). Muestrea paginas del cuerpo.

    Args:
        textos_paginas: textos por pagina.
        desde: primera pagina a muestrear (1-based).
        muestrear: espaciado del muestreo.

    Returns:
        (offset, n_muestras, evidencia).
    """
    muestras: list[int] = []
    inicio = max(1, desde)
    for i in range(inicio - 1, len(textos_paginas), muestrear):
        renglones = [r for r in textos_paginas[i].splitlines() if r.strip()]
        for r in renglones[:4] + renglones[-4:]:
            if len(r) > MAX_LINEA_CORTA + 30:
                continue
            r = r.strip()
            n_palabras = len(r.split())
            m_izq = re.match(r"^(\d{1,4})\s+[A-Za-z\u00c0-\u024f]", r)
            m_der = re.search(r"[A-Za-z\u00c0-\u024f]\s+(\d{1,4})$", r)
            if n_palabras > 10:
                continue
            if m_izq:
                muestras.append(int(m_izq.group(1)) - (i + 1))
            elif m_der:
                muestras.append(int(m_der.group(1)) - (i + 1))
    if not muestras:
        return 0, 0, "sin evidencia de numero impreso; offset asumido 0"
    offset, n = Counter(muestras).most_common(1)[0]
    if n < 10:
        return 0, n, "evidencia insuficiente (<10 muestras); offset asumido 0"
    return offset, n, f"moda offset={offset} en {n} muestras"


def construir_blacklist(textos_paginas: list[str], min_paginas: int = 3) -> set[str]:
    """Detecta running heads/pies repetidos (arquitectura (b.4)): lineas cortas
    que contienen un numero de pagina impreso al inicio o al final ('la
    colombia herida 31', '628  cuando los pajaros no cantaban') y que, sin ese
    numero, aparecen identicas en >= min_paginas paginas distintas. La
    exigencia del numero evita marcar como running head cualquier linea corta
    repetida (titulos de seccion cortos, etc.).

    Args:
        textos_paginas: textos por pagina del tomo.
        min_paginas: minimo de paginas para considerar la linea repetida.

    Returns:
        Set con las claves normalizadas de lineas a excluir.
    """
    por_pagina: dict[str, set[int]] = {}
    for i, t in enumerate(textos_paginas):
        for r in t.splitlines():
            r = r.strip()
            if not r:
                continue
            tiene_num = tiene_numero_esquina(r)
            if not tiene_num:
                continue
            contenido = sin_numero_esquinas(r)
            # el relleno del layout puede alargar la linea: se mide el contenido
            if len(contenido) > MAX_LINEA_CORTA:
                continue
            clave = norm_es(contenido)
            if not clave or len(clave.split()) < 1:
                continue
            por_pagina.setdefault(clave, set()).add(i + 1)
    return {k for k, paginas in por_pagina.items() if len(paginas) >= min_paginas}


def _lineas_utiles_pagina(texto_pagina: str, blacklist: set[str]) -> list[str]:
    """Filtra las lineas de una pagina: vacias, running heads/pies (linea con
    numero de esquina cuya clave sin numero esta en la blacklist) y lineas
    compuestas solo por digitos (numeros de pagina sueltos). Una linea sin
    numero de esquina nunca se elimina, aunque su texto coincida con un running
    head: los encabezados genuinos de seccion comparten texto con los pies.

    Args:
        texto_pagina: texto crudo de la pagina.
        blacklist: claves de running heads detectados.

    Returns:
        Lista de lineas utiles.
    """
    limpias: list[str] = []
    for r in texto_pagina.splitlines():
        r = r.rstrip()
        if not r.strip():
            continue
        if tiene_numero_esquina(r) and norm_es(sin_numero_esquinas(r)) in blacklist:
            continue
        if re.fullmatch(r"\s*\d{1,4}\s*", r):
            continue
        limpias.append(r)
    return limpias


# ----------------------------------------------------------------------------
# SECCION 4: matching de titulos TOC contra encabezados del cuerpo
# ----------------------------------------------------------------------------


def _lineas_con_huecos(texto_pagina: str, blacklist: set[str]) -> list[str]:
    """Devuelve las lineas de la pagina con los running heads/pies (linea con
    numero de esquina repetida en la blacklist) y numeros sueltos convertidos
    en huecos (""), de modo que no falseen la condicion de 'blanco previo' del
    detector de encabezados.

    Args:
        texto_pagina: texto crudo de la pagina.
        blacklist: claves de running heads.

    Returns:
        Lista de lineas con huecos.
    """
    lineas: list[str] = []
    for r in texto_pagina.splitlines():
        r = r.rstrip()
        if not r.strip():
            lineas.append("")
            continue
        if tiene_numero_esquina(r) and norm_es(sin_numero_esquinas(r)) in blacklist:
            lineas.append("")
            continue
        if re.fullmatch(r"\s*\d{1,4}\s*", r):
            lineas.append("")
            continue
        lineas.append(r)
    return lineas


def _bloque_encabezado(lineas: list[str], ini: int, max_lineas: int) -> list[str]:
    """Devuelve los tokens normalizados de hasta max_lineas lineas consecutivas
    (sin huecos) empezando en ini.

    Args:
        lineas: lineas de la pagina (con huecos "").
        ini: indice inicial.
        max_lineas: maximo de lineas del bloque (dinamico: los titulos largos
            de los tomos CEV se componen centrados en lineas cortas de 4-7
            palabras, por lo que un titulo de ~28 tokens ocupa ~8-10 lineas).

    Returns:
        Tokens concatenados ([] si la linea inicial esta vacia).
    """
    stream: list[str] = []
    for j in range(ini, min(ini + max_lineas, len(lineas))):
        if not lineas[j].strip():
            break
        stream.extend(norm_es(lineas[j]).split())
    return stream


def _inicios_bloque(lineas: list[str]) -> list[int]:
    """Indices de las lineas que pueden iniciar un encabezado: linea no vacia
    precedida de hueco/linea vacia o inicio de pagina.

    Args:
        lineas: lineas de la pagina (con huecos "").

    Returns:
        Indices candidatos a inicio de bloque.
    """
    return [i for i in range(len(lineas)) if lineas[i].strip() and (i == 0 or not lineas[i - 1].strip())]


def _pagina_contiene_titulo(
    texto_pagina: str,
    titulo_tokens: list[str],
    filas_tokens: list[list[str]],
    blacklist: set[str],
) -> bool:
    """Determina si el titulo TOC aparece como encabezado(s) en la pagina.

    Estrategia en dos pasos:
      (1) contiguo: los tokens del titulo completo aparecen como prefijo de un
          bloque de lineas consecutivas (limite dinamico min(12, 2 + n_tokens),
          porque los titulos largos se componen en ~8-10 lineas cortas) que
          empieza tras un blanco (o al inicio de la pagina). Cubre titulos de
          una o varias lineas pegadas.
      (2) por filas: cuando el TOC registro varias filas visuales para el
          titulo (p.ej. 'La espera' / 'Introduccion a El libro del porvenir'),
          cada fila debe aparecer como bloque de encabezado y en orden dentro
          de la pagina (las filas pueden estar separadas por blancos).

    Args:
        texto_pagina: texto crudo de la pagina (pdftotext).
        titulo_tokens: tokens normalizados del titulo TOC completo.
        filas_tokens: tokens normalizados de cada fila visual del titulo en el TOC.
        blacklist: running heads a excluir (se tratan como huecos).

    Returns:
        True si el titulo aparece como encabezado en la pagina.
    """
    lineas = _lineas_con_huecos(texto_pagina, blacklist)
    n = len(titulo_tokens)
    # paso 1: coincidencia contigua del titulo completo
    for ini in _inicios_bloque(lineas):
        stream = _bloque_encabezado(lineas, ini, min(12, 2 + n))
        if len(stream) >= n and stream[:n] == titulo_tokens:
            return True
    # paso 2: coincidencia por filas visuales del TOC, en orden
    if len(filas_tokens) > 1:
        pos = -1
        for fila in filas_tokens:
            encontrada = False
            for ini in _inicios_bloque(lineas):
                if ini <= pos:
                    continue
                stream = _bloque_encabezado(lineas, ini, min(12, 2 + len(fila)))
                if len(stream) >= len(fila) and stream[: len(fila)] == fila:
                    pos = ini
                    encontrada = True
                    break
            if not encontrada:
                return False
        return True
    return False


def matchear_titulos(
    entradas: list[dict[str, Any]],
    textos_paginas: list[str],
    blacklist: set[str],
    offset: int,
    ventana: int = 1,
) -> None:
    """Matching de cada entrada TOC contra los encabezados del cuerpo en la
    pagina impresa +- ventana (convertida a PDF con el offset calibrado).
    Anade a cada entrada: 'mapeado' (bool) y 'pagina_pdf' (pagina PDF donde se
    encontro el encabezado o, si no, la estimada por offset).

    Args:
        entradas: entradas TOC (mutadas in-place).
        textos_paginas: textos de las paginas del tomo.
        blacklist: running heads detectados.
        offset: pagina impresa - pagina PDF.
        ventana: amplitud de la ventana de busqueda.
    """
    n_paginas = len(textos_paginas)
    for e in entradas:
        tokens = norm_es(e["titulo"]).split()
        filas = [norm_es(linea).split() for linea in e.get("lineas", [])]
        if not filas:
            filas = [tokens]
        objetivo = e["pagina_impresa"] + offset
        encontrada = None
        for p in range(max(1, objetivo - ventana), min(n_paginas, objetivo + ventana) + 1):
            if _pagina_contiene_titulo(textos_paginas[p - 1], tokens, filas, blacklist):
                encontrada = p
                break
        e["mapeado"] = encontrada is not None
        e["pagina_pdf"] = encontrada if encontrada is not None else objetivo


# ----------------------------------------------------------------------------
# SECCION 5: construccion de segmentos (ADR-001)
# ----------------------------------------------------------------------------


def _titulo_descartable(titulo: str, descartes_manuales: list[str]) -> bool:
    """True si el titulo TOC corresponde a front/back matter (patrones
    genericos o lista manual de la config del tomo).

    Args:
        titulo: titulo de la entrada TOC.
        descartes_manuales: titulos normalizados a descartar (por config).

    Returns:
        True si debe descartarse.
    """
    clave = norm_es(titulo)
    if clave in descartes_manuales:
        return True
    return any(re.search(p, clave) for p in PATRONES_TITULO_DESCARTE)


def detectar_back_matter_cola(
    textos_paginas: list[str],
    pagina_desde: int,
    patrones: list[str],
) -> int | None:
    """Detecta el inicio del back matter no listado en el TOC: primera pagina
    (>= pagina_desde) cuyo primer renglon significativo coincide con un patron
    de back matter (ej. 'Referencias' al inicio de la pagina 729 en HALLAZGOS).

    Args:
        textos_paginas: textos por pagina.
        pagina_desde: solo se inspeccionan paginas desde esta (1-based).
        patrones: patrones normalizados de back matter.

    Returns:
        Numero de pagina donde empieza el back matter o None.
    """
    for i in range(max(1, pagina_desde) - 1, len(textos_paginas)):
        renglones = [r for r in textos_paginas[i].splitlines() if r.strip()]
        if not renglones:
            continue
        primero = renglones[0].strip()
        if len(primero) > 70:
            continue
        clave = norm_es(primero)
        if any(clave.startswith(p) for p in patrones):
            return i + 1
    return None


def construir_segmentos(
    entradas: list[dict[str, Any]],
    textos_paginas: list[str],
    tomo: str,
    params: dict[str, Any],
    blacklist: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construye los segmentos de un tomo (ADR-001) a partir de las entradas
    TOC ya mapeadas: filtra niveles_segmento, marca front/back matter por
    patron de titulo, ordena por pagina, crea rangos entre limites
    consecutivos, recorta el back matter de cola (limite descartado o
    encabezado detectado en el cuerpo), aplica el techo (>4.000 palabras ->
    fragmentos <=3.500 en limite de pagina) y fusiona segmentos cortos (<300
    palabras con el siguiente; el ultimo con el anterior).

    Args:
        entradas: entradas TOC con nivel, mapeado y pagina_pdf.
        textos_paginas: textos de las paginas del tomo.
        tomo: clave corta del tomo.
        params: parametros efectivos del tomo.
        blacklist: running heads detectados.

    Returns:
        (segmentos, estadisticas).
    """
    niveles = set(params["niveles_segmento"])
    n_paginas = len(textos_paginas)

    # limites = entradas de nivel en niveles_segmento, con marca de descarte
    limites = []
    for e in entradas:
        if e["nivel"] not in niveles:
            continue
        limites.append(
            {
                "titulo": e["titulo"],
                "nivel": e["nivel"],
                "pagina_impresa": e["pagina_impresa"],
                "pagina_pdf": e["pagina_pdf"],
                "mapeado": e.get("mapeado", False),
                "pagina_toc": e["pagina_toc"],
                "y": e["y"],
                "descartado": _titulo_descartable(e["titulo"], params["titulos_descartar"]),
            }
        )
    if not limites:
        return [], {"descartes": [], "sin_limites": True}
    limites.sort(key=lambda x: (x["pagina_pdf"], x["pagina_toc"], x["y"]))

    # dedupe: si dos limites caen en la misma pagina se conserva el de nivel
    # mas profundo (el padre sin paginas propias no aporta rango); si mismo
    # nivel, se conserva el primero (irregularidad reportada).
    unicos: list[dict[str, Any]] = []
    n_duplicados = 0
    for lim in limites:
        if unicos and unicos[-1]["pagina_pdf"] == lim["pagina_pdf"]:
            if lim["nivel"] > unicos[-1]["nivel"]:
                unicos[-1] = lim  # el hijo reemplaza al padre sin paginas
            elif lim["nivel"] == unicos[-1]["nivel"]:
                n_duplicados += 1
            continue
        unicos.append(lim)

    # inicio del back matter de cola: primer limite descartado sin limites
    # retenidos despues, o encabezado de back matter detectado en el cuerpo
    ultimo_retenido = max(
        (i for i, u in enumerate(unicos) if not u["descartado"]), default=None
    )
    candidatos_back: list[int] = []
    if ultimo_retenido is not None:
        for u in unicos[ultimo_retenido + 1:]:
            if u["descartado"]:
                candidatos_back.append(u["pagina_pdf"])
    desde_escaneo = (unicos[-1]["pagina_pdf"] + 1) if unicos else 1
    encabezado_back = detectar_back_matter_cola(
        textos_paginas, desde_escaneo, params["patrones_back_matter"]
    )
    if encabezado_back is not None:
        candidatos_back.append(encabezado_back)
    back_inicio = min(candidatos_back) if candidatos_back else None

    def texto_rango(pag_ini: int, pag_fin: int) -> str:
        partes: list[str] = []
        for p in range(pag_ini, pag_fin + 1):
            partes.extend(_lineas_utiles_pagina(textos_paginas[p - 1], blacklist))
        return "\n".join(partes)

    # rangos: [inicio_i, inicio_{i+1}) recortados al back matter de cola
    segmentos: list[dict[str, Any]] = []
    for idx, lim in enumerate(unicos):
        if lim["descartado"]:
            continue
        fin = unicos[idx + 1]["pagina_pdf"] - 1 if idx + 1 < len(unicos) else n_paginas
        if back_inicio is not None and fin >= back_inicio:
            fin = back_inicio - 1
        if fin < lim["pagina_pdf"]:
            continue
        texto = texto_rango(lim["pagina_pdf"], fin)
        segmentos.append(
            {
                "titulo": lim["titulo"],
                "nivel_toc": lim["nivel"],
                "pagina_inicio": lim["pagina_pdf"],
                "pagina_fin": fin,
                "n_palabras": contar_palabras(texto),
                "texto": texto,
                "no_mapeado": not lim["mapeado"],
            }
        )

    techo = params["techo_palabras"]
    frag_max = params["fragmento_max_palabras"]

    def partir_por_paginas(seg: dict[str, Any]) -> list[dict[str, Any]]:
        """Parte un segmento en fragmentos <= frag_max palabras cortando en
        limites de pagina."""
        paginas_txt: list[tuple[int, str]] = []
        for p in range(seg["pagina_inicio"], seg["pagina_fin"] + 1):
            txt_p = "\n".join(_lineas_utiles_pagina(textos_paginas[p - 1], blacklist))
            paginas_txt.append((p, txt_p))
        fragmentos: list[dict[str, Any]] = []
        acum: list[str] = []
        acum_pags: list[int] = []
        n_words = 0
        for p, txt_p in paginas_txt:
            w_p = contar_palabras(txt_p)
            if acum and n_words + w_p > frag_max:
                fragmentos.append(
                    {
                        **seg,
                        "pagina_inicio": acum_pags[0],
                        "pagina_fin": acum_pags[-1],
                        "n_palabras": n_words,
                        "texto": "\n".join(acum),
                    }
                )
                acum, acum_pags, n_words = [], [], 0
            acum.append(txt_p)
            acum_pags.append(p)
            n_words += w_p
        if acum:
            fragmentos.append(
                {
                    **seg,
                    "pagina_inicio": acum_pags[0],
                    "pagina_fin": acum_pags[-1],
                    "n_palabras": n_words,
                    "texto": "\n".join(acum),
                }
            )
        return fragmentos

    # (a) techo: partir segmentos > techo_palabras
    expandidos: list[dict[str, Any]] = []
    n_fragmentos = 0
    for s in segmentos:
        if s["n_palabras"] <= techo:
            expandidos.append(s)
        else:
            frags = partir_por_paginas(s)
            for f in frags:
                f["fragmento_de"] = s["titulo"]
            n_fragmentos += len(frags)
            expandidos.extend(frags)

    # (b) fusion de segmentos cortos (< fusion_min) con el siguiente (o con el
    # anterior si el corto es el ultimo de la lista).
    fusion_min = params["fusion_min_palabras"]
    n_fusiones = 0
    finales: list[dict[str, Any]] = []
    i = 0
    while i < len(expandidos):
        s = expandidos[i]
        if s["n_palabras"] < fusion_min:
            if i + 1 < len(expandidos):
                sig = expandidos[i + 1]
                texto_unido = (s["texto"].strip() + "\n\n" + sig["texto"]).strip()
                expandidos[i + 1] = {
                    **sig,
                    "pagina_inicio": s["pagina_inicio"],
                    "texto": texto_unido,
                    "n_palabras": contar_palabras(texto_unido),
                    "fusion_con": s["titulo"],
                }
                n_fusiones += 1
                i += 1
                continue
            if finales:
                prev = finales.pop()
                texto_unido = (prev["texto"].strip() + "\n\n" + s["texto"]).strip()
                finales.append(
                    {
                        **prev,
                        "pagina_fin": s["pagina_fin"],
                        "texto": texto_unido,
                        "n_palabras": contar_palabras(texto_unido),
                        "fusion_con": s["titulo"],
                    }
                )
                n_fusiones += 1
                i += 1
                continue
        finales.append(s)
        i += 1

    # (c) si la fusion supero el techo, se vuelve a partir por paginas
    post_fusion: list[dict[str, Any]] = []
    for s in finales:
        if s["n_palabras"] > techo:
            frags = partir_por_paginas(s)
            n_fragmentos += len(frags)
            post_fusion.extend(frags)
        else:
            post_fusion.append(s)

    for s in post_fusion:
        s["id_segmento"] = f"{tomo}-p{s['pagina_inicio']}-{slug_titulo(s['titulo'])}"
        s["tomo"] = tomo
    return post_fusion, {
        "descartes": [
            {"titulo": u["titulo"], "pagina": u["pagina_pdf"], "nivel": u["nivel"]}
            for u in unicos
            if u["descartado"]
        ],
        "fusiones": n_fusiones,
        "fragmentos_techo": n_fragmentos,
        "sin_limites": False,
        "back_matter_inicio": back_inicio,
        "limites_duplicados": n_duplicados,
    }


# ----------------------------------------------------------------------------
# SECCION 6: sanity checks y reporte
# ----------------------------------------------------------------------------


def sanity_checks(
    tomo: str,
    entradas: list[dict[str, Any]],
    segmentos: list[dict[str, Any]],
    n_paginas_pdf: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta los sanity checks (i)-(v) de arquitectura (b.5) para un tomo.

    Args:
        tomo: clave del tomo.
        entradas: entradas TOC parseadas (todas).
        segmentos: segmentos construidos.
        n_paginas_pdf: total de paginas del PDF.
        stats: estadisticas de construccion.

    Returns:
        Dict del reporte del tomo.
    """
    entradas_b = [e for e in entradas if e.get("nivel", 99) in {1, 2}]
    n_bound = len(entradas_b)
    n_mapeadas = sum(1 for e in entradas_b if e.get("mapeado"))
    pct_mapeo = n_mapeadas / n_bound if n_bound else 0.0
    no_mapeados = [e["titulo"] for e in entradas_b if not e.get("mapeado")]

    primera = min((s["pagina_inicio"] for s in segmentos), default=None)
    ultima = max((s["pagina_fin"] for s in segmentos), default=None)
    paginas_segmentos: set[int] = set()
    for s in segmentos:
        paginas_segmentos.update(range(s["pagina_inicio"], s["pagina_fin"] + 1))

    back_inicio = stats.get("back_matter_inicio")
    fin_contenido = ultima if ultima is not None else n_paginas_pdf
    if back_inicio is not None:
        fin_contenido = min(fin_contenido, back_inicio - 1)
    paginas_contenido = set(range(primera, fin_contenido + 1)) if primera else set()
    cubiertas = paginas_contenido & paginas_segmentos
    pct_cobertura = len(cubiertas) / len(paginas_contenido) if paginas_contenido else 0.0

    n_pag_seg = sum(s["pagina_fin"] - s["pagina_inicio"] + 1 for s in segmentos)
    solapamientos = n_pag_seg - len(paginas_segmentos)

    palabras = [s["n_palabras"] for s in segmentos]
    n_seg = len(palabras)
    fuera_banda = [w for w in palabras if w < 300 or w > 4000]
    pct_fuera = len(fuera_banda) / n_seg if n_seg else 0.0
    vacios = [s["titulo"] for s in segmentos if s["n_palabras"] == 0]

    front_paginas = (primera - 1) if primera else n_paginas_pdf
    if back_inicio is not None:
        back_paginas = n_paginas_pdf - back_inicio + 1
    elif ultima is not None:
        back_paginas = n_paginas_pdf - ultima
    else:
        back_paginas = 0

    return {
        "tomo": tomo,
        "pdf_paginas": n_paginas_pdf,
        "toc_paginas": sorted({e["pagina_toc"] for e in entradas}),
        "toc_entradas": len(entradas),
        "entradas_nivel_1_2": n_bound,
        "entradas_mapeadas": n_mapeadas,
        "pct_mapeo": round(pct_mapeo, 4),
        "mapeo_ok": pct_mapeo >= UMBRALES["mapeo_titulos_min"],
        "n_titulos_no_mapeados": len(no_mapeados),
        "titulos_no_mapeados": no_mapeados[:20],
        "n_segmentos": n_seg,
        "fusiones": stats.get("fusiones", 0),
        "fragmentos_techo": stats.get("fragmentos_techo", 0),
        "palabras_totales": sum(palabras),
        "palabras_media": round(sum(palabras) / n_seg, 1) if n_seg else 0.0,
        "palabras_min": min(palabras) if palabras else 0,
        "palabras_max": max(palabras) if palabras else 0,
        "n_fuera_banda": len(fuera_banda),
        "pct_fuera_banda": round(pct_fuera, 4),
        "fuera_banda_ok": pct_fuera < UMBRALES["fuera_banda_max"],
        "segmentos_vacios": len(vacios),
        "vacios_ok": len(vacios) == 0,
        "paginas_contenido": len(paginas_contenido),
        "paginas_cubiertas": len(cubiertas),
        "pct_cobertura": round(pct_cobertura, 4),
        "cobertura_ok": pct_cobertura >= UMBRALES["cobertura_paginas_min"],
        "solapamientos": max(0, solapamientos),
        "solapamiento_ok": solapamientos <= 0,
        "front_matter_paginas": front_paginas,
        "back_matter_paginas": back_paginas,
        "back_matter_inicio": back_inicio,
        "descartes_titulos": stats.get("descartes", []),
        "limites_duplicados": stats.get("limites_duplicados", 0),
        "cumple_umbrales": (
            pct_mapeo >= UMBRALES["mapeo_titulos_min"]
            and pct_cobertura >= UMBRALES["cobertura_paginas_min"]
            and solapamientos <= 0
            and pct_fuera < UMBRALES["fuera_banda_max"]
            and len(vacios) == 0
        ),
    }


# ----------------------------------------------------------------------------
# SECCION 7: orquestacion por tomo, reporte global y consola
# ----------------------------------------------------------------------------


def cargar_config_tomo() -> dict[str, Any]:
    """Carga config/toc_tomos.json si existe (ajustes manuales por tomo).

    Returns:
        Dict {tomo: params} con los ajustes manuales.
    """
    if not RUTA_CONFIG_TOC.exists():
        return {}
    try:
        datos = json.loads(RUTA_CONFIG_TOC.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return datos.get("tomo_params", {})


def guardar_config_tomo(usados: dict[str, dict[str, Any]]) -> None:
    """Persiste la config por tomo efectiva (merge con ajustes manuales previos
    para no perder comentarios/documentacion).

    Args:
        usados: parametros efectivos por tomo.
    """
    manuales = cargar_config_tomo()
    datos: dict[str, Any] = {"_comentario": CONFIG_DOCUMENTACION, "tomo_params": {}}
    for tomo in ORDEN_TOMOS:
        if tomo in usados:
            datos["tomo_params"][tomo] = usados[tomo]
    for tomo, params in manuales.items():
        datos["tomo_params"].setdefault(tomo, params)
    RUTA_CONFIG_TOC.parent.mkdir(parents=True, exist_ok=True)
    RUTA_CONFIG_TOC.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def procesar_tomo(tomo: str, ruta_pdf: Path, params: dict[str, Any], force: bool) -> dict[str, Any]:
    """Ejecuta el pipeline completo de extraccion y segmentacion de un tomo.

    Args:
        tomo: clave corta del tomo.
        ruta_pdf: PDF del tomo.
        params: parametros efectivos del tomo.
        force: re-extrae paginas aunque existan.

    Returns:
        Reporte del tomo.
    """
    t0 = datetime.now(timezone.utc)
    extraer_paginas_tomo(ruta_pdf, tomo, n_workers=32, force=force)
    textos = leer_paginas(tomo)
    print(f"[{tomo}] {len(textos)} paginas cargadas")

    paginas_toc = detectar_paginas_toc(ruta_pdf, textos)
    if not paginas_toc:
        return {
            "tomo": tomo,
            "pdf_paginas": len(textos),
            "toc_paginas": [],
            "error": "TOC no detectado: 'Contenido' no aparece como primer renglon",
            "cumple_umbrales": False,
        }

    entradas = parsear_entradas_toc(ruta_pdf, paginas_toc)
    asignar_niveles(entradas)
    print(
        f"[{tomo}] TOC: paginas pdf {paginas_toc[0]}-{paginas_toc[-1]} "
        f"({len(paginas_toc)}), {len(entradas)} entradas"
    )

    offset, _n_muestras, evidencia = calibrar_offset(textos)
    print(f"[{tomo}] offset impreso-PDF: {evidencia}")
    blacklist = construir_blacklist(textos, params["blacklist_min_paginas"])
    print(f"[{tomo}] blacklist running heads: {len(blacklist)}")
    matchear_titulos(entradas, textos, blacklist, offset, params["ventana_mapeo_paginas"])
    n_map = sum(1 for e in entradas if e.get("mapeado"))
    print(f"[{tomo}] matching: {n_map}/{len(entradas)} titulos mapeados")

    segmentos, stats = construir_segmentos(entradas, textos, tomo, params, blacklist)
    stats["offset"] = offset
    reporte = sanity_checks(tomo, entradas, segmentos, len(textos), stats)

    DIR_SEGMENTOS.mkdir(parents=True, exist_ok=True)
    with open(DIR_SEGMENTOS / f"{tomo}.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in segmentos)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    print(
        f"[{tomo}] {reporte['n_segmentos']} segmentos | mapeo {reporte['pct_mapeo']:.1%} "
        f"| cobertura {reporte['pct_cobertura']:.1%} | fuera banda "
        f"{reporte['pct_fuera_banda']:.1%} | media {reporte['palabras_media']:.0f} pal "
        f"| {dt:.0f} s"
    )
    return reporte


def reconstruir_global() -> None:
    """Reconstruye data/corpus/segmentos_libros.jsonl concatenando los JSONL
    por tomo en orden canonico (idempotente y reanudable)."""
    DIR_SEGMENTOS.mkdir(parents=True, exist_ok=True)
    tmp = RUTA_SEGMENTOS_GLOBAL.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as out:
        for tomo in ORDEN_TOMOS:
            ruta = DIR_SEGMENTOS / f"{tomo}.jsonl"
            if not ruta.exists():
                continue
            with open(ruta, encoding="utf-8") as fh:
                for linea in fh:
                    out.write(linea)
                    n += 1
    os.replace(tmp, RUTA_SEGMENTOS_GLOBAL)
    print(f"segmentos_libros.jsonl reconstruido: {n} segmentos totales")


def main() -> None:
    """Orquesta el pipeline: argumentos, descompresion, procesamiento por tomo
    y reporte global."""
    parser = argparse.ArgumentParser(
        description="Extraccion y segmentacion de los tomos CEV (Fase 3)"
    )
    parser.add_argument(
        "--libros",
        nargs="*",
        default=None,
        help="Tomos a procesar: " + ", ".join(ORDEN_TOMOS) + ". "
        "Por defecto: los que falten (reanudable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extrae las paginas aunque el JSONL exista.",
    )
    args = parser.parse_args()

    manuales = cargar_config_tomo()
    if args.libros:
        tomos = args.libros
    else:
        tomos = [t for t in ORDEN_TOMOS if not (DIR_SEGMENTOS / f"{t}.jsonl").exists()]
    invalidos = [t for t in tomos if t not in TOMO_A_MIEMBRO]
    if invalidos:
        print(f"Tomos invalidos: {invalidos}. Validos: {', '.join(ORDEN_TOMOS)}", file=sys.stderr)
        sys.exit(2)
    if not tomos:
        print("Nada que hacer: todos los tomos ya tienen segmentos (use --force).")
        return

    rutas = descomprimir_pdfs(tomos)
    reportes: list[dict[str, Any]] = []
    params_usados: dict[str, dict[str, Any]] = {}
    for tomo in tomos:
        params = {**DEFAULTS_TOMO, **manuales.get(tomo, {})}
        params_usados[tomo] = params
        reportes.append(procesar_tomo(tomo, rutas[tomo], params, force=args.force))
    guardar_config_tomo(params_usados)
    reconstruir_global()

    reporte_global: dict[str, Any] = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/01_extraer_libros.py",
        "umbrales": UMBRALES,
        "tomos": {r["tomo"]: r for r in reportes},
    }
    ok = [r["tomo"] for r in reportes if r.get("cumple_umbrales")]
    reporte_global["resumen"] = {
        "tomos_procesados": len(reportes),
        "tomos_ok": ok,
        "tomos_revisar": [r["tomo"] for r in reportes if not r.get("cumple_umbrales")],
        "segmentos_totales": sum(r.get("n_segmentos", 0) for r in reportes),
        "palabras_totales": sum(r.get("palabras_totales", 0) for r in reportes),
    }
    RUTA_REPORTE.parent.mkdir(parents=True, exist_ok=True)
    RUTA_REPORTE.write_text(
        json.dumps(reporte_global, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Reporte: {RUTA_REPORTE.relative_to(RUTA_PROYECTO)}")
    print("-" * 72)
    for r in reportes:
        if "error" in r:
            print(f"  {r['tomo']:<12} ERROR: {r['error']}")
            continue
        print(
            f"  {r['tomo']:<12} seg={r['n_segmentos']:>4} mapeo={r['pct_mapeo']:>6.1%} "
            f"cob={r['pct_cobertura']:>6.1%} fuera={r['pct_fuera_banda']:>6.1%} "
            f"media={r['palabras_media']:>6.0f}  "
            f"{'OK' if r['cumple_umbrales'] else 'REVISAR'}"
        )
    print("-" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
