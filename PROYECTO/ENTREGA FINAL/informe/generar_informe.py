"""Genera el informe final consolidado (E1 + E2 + E3) de Juris-PLN en DOCX.

El documento se compone de bloques declarativos (encabezados, párrafos, listas,
tablas y figuras) que se renderizan con python-docx. La tabla de contenido se
inserta como campo TOC con el resultado cacheado; como LibreOffice no
recalcula índices al convertir, el número de página de cada entrada se obtiene
en una primera pasada sobre el PDF y se reinyecta en una segunda pasada.

Uso:
    /home/carlos/venv/bin/python informe/generar_informe.py

Salidas:
    informe/informe_final_juris_pln.docx
    informe/informe_final_juris_pln.pdf
"""

# ruff: noqa: ISC004  (los textos largos del informe se concatenan a proposito;
# unirlos en una sola linea haria ilegible el contenido)

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from pypdf import PdfReader

BASE = Path(__file__).resolve().parent
FIGURAS = BASE.parent / "figuras"
SALIDA_DOCX = BASE / "informe_final_juris_pln.docx"
SALIDA_PDF = BASE / "informe_final_juris_pln.pdf"
BORRADOR = BASE / "_borrador_informe.docx"

FUENTE = "Times New Roman"
TINTA = RGBColor(0x16, 0x25, 0x36)
AZUL = RGBColor(0x1F, 0x3A, 0x5F)
GRIS = RGBColor(0x42, 0x54, 0x66)
ANCHO_TEXTO_CM = 16.0

TITULO = (
    "Juris-PLN: recuperación de fundamentos constitucionales y contexto "
    "jurisprudencial sobre 23.742 sentencias de la Corte Constitucional de Colombia"
)
SUBTITULO = "Informe final consolidado de las entregas 1, 2 y 3"
AUTOR = "Carlos Andrés Garnica Salazar"
CURSO = "Procesamiento de Lenguaje Natural"
PROGRAMA = "Maestría en Ingeniería de Sistemas y Computación"
UNIVERSIDAD = "Pontificia Universidad Javeriana"
CIUDAD_FECHA = "Bogotá D.C., noviembre de 2026"


# ---------------------------------------------------------------------------
# Utilidades de formato
# ---------------------------------------------------------------------------
def _rfonts(elemento, nombre: str) -> None:
    """Fija la fuente para todos los alfabetos del run o estilo."""
    rpr = elemento.get_or_add_rPr()
    fuentes = rpr.find(qn("w:rFonts"))
    if fuentes is None:
        fuentes = OxmlElement("w:rFonts")
        rpr.append(fuentes)
    for atributo in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fuentes.set(qn(atributo), nombre)
    for atributo in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if fuentes.get(qn(atributo)) is not None:
            del fuentes.attrib[qn(atributo)]


def _sombreado(celda, color_hex: str) -> None:
    """Aplica sombreado de fondo a una celda."""
    tc_pr = celda._tc.get_or_add_tcPr()
    sombra = OxmlElement("w:shd")
    sombra.set(qn("w:val"), "clear")
    sombra.set(qn("w:color"), "auto")
    sombra.set(qn("w:fill"), color_hex)
    tc_pr.append(sombra)


def _borde_inferior(parrafo, color: str = "1F3A5F", tamano: int = 8) -> None:
    """Dibuja una línea inferior en el párrafo."""
    p_pr = parrafo._p.get_or_add_pPr()
    bordes = OxmlElement("w:pBdr")
    borde = OxmlElement("w:bottom")
    borde.set(qn("w:val"), "single")
    borde.set(qn("w:sz"), str(tamano))
    borde.set(qn("w:space"), "4")
    borde.set(qn("w:color"), color)
    bordes.append(borde)
    p_pr.append(bordes)


def _campo(parrafo, instruccion: str, valor_inicial: str = "1") -> None:
    """Inserta un campo de Word (PAGE, NUMPAGES) con valor cacheado."""
    run = parrafo.add_run()
    inicio = OxmlElement("w:fldChar")
    inicio.set(qn("w:fldCharType"), "begin")
    texto = OxmlElement("w:instrText")
    texto.set(qn("xml:space"), "preserve")
    texto.text = f" {instruccion} "
    separador = OxmlElement("w:fldChar")
    separador.set(qn("w:fldCharType"), "separate")
    valor = OxmlElement("w:t")
    valor.text = valor_inicial
    fin = OxmlElement("w:fldChar")
    fin.set(qn("w:fldCharType"), "end")
    for pieza in (inicio, texto, separador, valor, fin):
        run._r.append(pieza)


def agregar_texto_enriquecido(parrafo, texto: str, tamano: float | None = None) -> None:
    """Agrega texto interpretando **negrita** como marcas de énfasis."""
    partes = texto.split("**")
    for indice, parte in enumerate(partes):
        if not parte:
            continue
        run = parrafo.add_run(parte)
        run.bold = indice % 2 == 1
        if tamano is not None:
            run.font.size = Pt(tamano)


# ---------------------------------------------------------------------------
# Configuración del documento
# ---------------------------------------------------------------------------
def configurar_estilos(doc: Document) -> None:
    """Define tipografía, interlineado y jerarquía de títulos."""
    estilos = doc.styles

    normal = estilos["Normal"]
    normal.font.name = FUENTE
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    _rfonts(normal.element, FUENTE)
    formato = normal.paragraph_format
    formato.line_spacing = 1.15
    formato.space_after = Pt(6)
    formato.space_before = Pt(0)
    formato.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    formato.widow_control = True

    especificaciones = {
        "Heading 1": {
            "tamano": 16,
            "antes": 18,
            "despues": 8,
            "color": AZUL,
            "negrita": True,
        },
        "Heading 2": {
            "tamano": 12.5,
            "antes": 12,
            "despues": 4,
            "color": TINTA,
            "negrita": True,
        },
        "Heading 3": {
            "tamano": 11,
            "antes": 10,
            "despues": 3,
            "color": TINTA,
            "negrita": False,
        },
    }
    for nombre, spec in especificaciones.items():
        estilo = estilos[nombre]
        estilo.font.name = FUENTE
        estilo.font.size = Pt(spec["tamano"])
        estilo.font.bold = spec["negrita"]
        estilo.font.italic = False
        estilo.font.color.rgb = spec["color"]
        _rfonts(estilo.element, FUENTE)
        pf = estilo.paragraph_format
        pf.space_before = Pt(spec["antes"])
        pf.space_after = Pt(spec["despues"])
        pf.line_spacing = 1.15
        pf.keep_with_next = True
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT

    for nombre in ("List Bullet", "List Number"):
        estilo = estilos[nombre]
        estilo.font.name = FUENTE
        estilo.font.size = Pt(11)
        _rfonts(estilo.element, FUENTE)
        estilo.paragraph_format.line_spacing = 1.15
        estilo.paragraph_format.space_after = Pt(3)


def configurar_seccion(doc: Document) -> None:
    """A4, márgenes de 2,5 cm, encabezado y pie con numeración."""
    seccion = doc.sections[0]
    seccion.page_width = Cm(21.0)
    seccion.page_height = Cm(29.7)
    seccion.top_margin = Cm(2.5)
    seccion.bottom_margin = Cm(2.5)
    seccion.left_margin = Cm(2.5)
    seccion.right_margin = Cm(2.5)
    seccion.header_distance = Cm(1.2)
    seccion.footer_distance = Cm(1.2)
    seccion.different_first_page_header_footer = True

    encabezado = seccion.header.paragraphs[0]
    encabezado.text = ""
    encabezado.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = encabezado.add_run("Juris-PLN  |  Informe final  |  Entrega 3")
    run.font.size = Pt(8.5)
    run.font.name = FUENTE
    run.font.color.rgb = GRIS
    _borde_inferior(encabezado, color="B7C3D0", tamano=4)

    pie = seccion.footer.paragraphs[0]
    pie.text = ""
    pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = pie.add_run("Página ")
    run.font.size = Pt(8.5)
    run.font.name = FUENTE
    run.font.color.rgb = GRIS
    _campo(pie, "PAGE")
    run = pie.add_run(" de ")
    run.font.size = Pt(8.5)
    run.font.name = FUENTE
    run.font.color.rgb = GRIS
    _campo(pie, "NUMPAGES")


# ---------------------------------------------------------------------------
# Portada y tabla de contenido
# ---------------------------------------------------------------------------
def construir_portada(doc: Document) -> None:
    """Dibuja la portada institucional."""
    logo = BASE / "logo_javeriana.png"
    if logo.exists():
        parrafo = doc.add_paragraph()
        parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        parrafo.paragraph_format.space_before = Pt(6)
        parrafo.add_run().add_picture(str(logo), width=Cm(3.4))

    for texto, tamano, espacio in (
        (UNIVERSIDAD.upper(), 10, 10),
        (PROGRAMA, 10, 2),
        (CURSO, 10, 26),
    ):
        parrafo = doc.add_paragraph()
        parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        parrafo.paragraph_format.space_after = Pt(espacio)
        run = parrafo.add_run(texto)
        run.font.size = Pt(tamano)
        run.font.name = FUENTE
        run.font.color.rgb = GRIS
        run.font.small_caps = True

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.space_before = Pt(18)
    parrafo.paragraph_format.space_after = Pt(4)
    run = parrafo.add_run("JURIS-PLN")
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.name = FUENTE
    run.font.color.rgb = AZUL

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.space_after = Pt(14)
    run = parrafo.add_run(
        "Recuperación de fundamentos constitucionales y contexto jurisprudencial "
        "sobre 23.742 sentencias de la Corte Constitucional de Colombia"
    )
    run.font.size = Pt(15)
    run.font.name = FUENTE
    run.font.color.rgb = TINTA

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _borde_inferior(parrafo, color="1F3A5F", tamano=8)
    parrafo.paragraph_format.space_after = Pt(12)

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.space_after = Pt(2)
    run = parrafo.add_run(SUBTITULO)
    run.font.size = Pt(12)
    run.font.name = FUENTE
    run.font.italic = True
    run.font.color.rgb = GRIS

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.space_before = Pt(60)
    parrafo.paragraph_format.space_after = Pt(2)
    run = parrafo.add_run(AUTOR)
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.name = FUENTE

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.space_after = Pt(90)
    run = parrafo.add_run(
        "Con base en las entregas 1 y 2 del grupo Rivera, Carrillo, Garnica y Prieto"
    )
    run.font.size = Pt(9.5)
    run.font.name = FUENTE
    run.font.italic = True
    run.font.color.rgb = GRIS

    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = parrafo.add_run(CIUDAD_FECHA)
    run.font.size = Pt(10.5)
    run.font.name = FUENTE
    run.font.color.rgb = GRIS


def construir_toc(doc: Document, entradas: list[tuple[int, str]], paginas: dict[str, int] | None) -> None:
    """Inserta el campo TOC con las entradas y sus páginas cacheadas."""
    doc.add_page_break()
    titulo = doc.add_paragraph()
    titulo.paragraph_format.space_after = Pt(10)
    run = titulo.add_run("Contenido")
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.name = FUENTE
    run.font.color.rgb = AZUL

    primero = doc.add_paragraph()
    run = primero.add_run()
    inicio = OxmlElement("w:fldChar")
    inicio.set(qn("w:fldCharType"), "begin")
    instruccion = OxmlElement("w:instrText")
    instruccion.set(qn("xml:space"), "preserve")
    instruccion.text = ' TOC \\o "1-2" \\h \\z \\u '
    separador = OxmlElement("w:fldChar")
    separador.set(qn("w:fldCharType"), "separate")
    run._r.append(inicio)
    run._r.append(instruccion)
    run._r.append(separador)

    for indice, (nivel, texto) in enumerate(entradas):
        parrafo = primero if indice == 0 else doc.add_paragraph()
        formato = parrafo.paragraph_format
        formato.left_indent = Cm(0.0 if nivel == 1 else 0.55)
        formato.space_before = Pt(1.8 if nivel == 1 else 0)
        formato.space_after = Pt(0)
        formato.line_spacing = 0.92
        formato.tab_stops.add_tab_stop(Cm(ANCHO_TEXTO_CM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        run = parrafo.add_run(texto)
        run.font.size = Pt(8.2 if nivel == 2 else 9.4)
        run.font.name = FUENTE
        run.bold = nivel == 1
        pagina = "" if paginas is None else str(paginas.get(texto, ""))
        run = parrafo.add_run("\t" + pagina)
        run.font.size = Pt(8.2 if nivel == 2 else 9.4)
        run.font.name = FUENTE
        if indice == len(entradas) - 1:
            fin = OxmlElement("w:fldChar")
            fin.set(qn("w:fldCharType"), "end")
            parrafo.add_run()._r.append(fin)
    doc.add_page_break()


# ---------------------------------------------------------------------------
# Renderizadores de bloques
# ---------------------------------------------------------------------------
def renderizar_titulo(doc: Document, nivel: int, texto: str) -> None:
    """Agrega un título con el estilo de encabezado correspondiente."""
    parrafo = doc.add_paragraph(style=f"Heading {nivel}")
    parrafo.add_run(texto)


def renderizar_parrafo(doc: Document, texto: str) -> None:
    """Agrega un párrafo justificado con soporte de negritas."""
    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    agregar_texto_enriquecido(parrafo, texto)


def renderizar_lista(doc: Document, elementos: list[str], numerada: bool = False) -> None:
    """Agrega una lista. Las numeradas se escriben con número manual."""
    for posicion, elemento in enumerate(elementos, start=1):
        if numerada:
            parrafo = doc.add_paragraph()
            parrafo.paragraph_format.left_indent = Cm(0.9)
            parrafo.paragraph_format.first_line_indent = Cm(-0.9)
            parrafo.paragraph_format.space_after = Pt(3)
            parrafo.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            run = parrafo.add_run(f"{posicion}.\t")
            run.bold = True
            agregar_texto_enriquecido(parrafo, elemento)
        else:
            parrafo = doc.add_paragraph(style="List Bullet")
            parrafo.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            agregar_texto_enriquecido(parrafo, elemento)


def renderizar_tabla(doc: Document, spec: dict, numero: int) -> None:
    """Agrega una tabla con encabezado sombreado, pie de fuente y numeración."""
    parrafo = doc.add_paragraph()
    parrafo.paragraph_format.space_before = Pt(10)
    parrafo.paragraph_format.space_after = Pt(4)
    parrafo.paragraph_format.keep_with_next = True
    run = parrafo.add_run(f"Tabla {numero}. ")
    run.bold = True
    run.font.size = Pt(9.5)
    run.font.name = FUENTE
    run = parrafo.add_run(spec["titulo"])
    run.font.size = Pt(9.5)
    run.font.name = FUENTE

    columnas = len(spec["encabezado"])
    tabla = doc.add_table(rows=1, cols=columnas)
    tabla.style = "Table Grid"
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    tabla.autofit = False
    anchos = spec.get("anchos") or [ANCHO_TEXTO_CM / columnas] * columnas
    tamano = spec.get("tamano", 8.8)
    alineaciones = spec.get("alineacion") or ["l"] * columnas

    def escribir(celda, texto: str, negrita: bool, indice: int) -> None:
        celda.text = ""
        parrafo = celda.paragraphs[0]
        parrafo.paragraph_format.space_before = Pt(2)
        parrafo.paragraph_format.space_after = Pt(2)
        parrafo.paragraph_format.line_spacing = 1.0
        parrafo.alignment = {
            "l": WD_ALIGN_PARAGRAPH.LEFT,
            "c": WD_ALIGN_PARAGRAPH.CENTER,
            "r": WD_ALIGN_PARAGRAPH.RIGHT,
        }[alineaciones[indice]]
        agregar_texto_enriquecido(parrafo, texto, tamano)
        for run in parrafo.runs:
            run.font.name = FUENTE
            if negrita:
                run.bold = True

    encabezado = tabla.rows[0]
    tr_pr = encabezado._tr.get_or_add_trPr()
    cabecera = OxmlElement("w:tblHeader")
    tr_pr.append(cabecera)
    for indice, texto in enumerate(spec["encabezado"]):
        escribir(encabezado.cells[indice], texto, True, indice)
        _sombreado(encabezado.cells[indice], "E7ECF2")

    for fila in spec["filas"]:
        celdas = tabla.add_row().cells
        for indice, texto in enumerate(fila):
            escribir(celdas[indice], str(texto), False, indice)
        tr_pr = tabla.rows[-1]._tr.get_or_add_trPr()
        no_partir = OxmlElement("w:cantSplit")
        tr_pr.append(no_partir)

    for fila in tabla.rows:
        for indice, celda in enumerate(fila.cells):
            celda.width = Cm(anchos[indice])

    if spec.get("nota"):
        parrafo = doc.add_paragraph()
        parrafo.paragraph_format.space_before = Pt(3)
        parrafo.paragraph_format.space_after = Pt(0)
        run = parrafo.add_run("Nota. " + spec["nota"])
        run.font.size = Pt(8)
        run.font.name = FUENTE
        run.font.italic = True
        run.font.color.rgb = GRIS

    parrafo = doc.add_paragraph()
    parrafo.paragraph_format.space_before = Pt(2)
    parrafo.paragraph_format.space_after = Pt(10)
    run = parrafo.add_run("Fuente: " + spec["fuente"])
    run.font.size = Pt(8)
    run.font.name = FUENTE
    run.font.color.rgb = GRIS


def renderizar_figura(doc: Document, spec: dict, numero: int) -> None:
    """Agrega una figura centrada con pie numerado y fuente."""
    ruta = FIGURAS / spec["archivo"]
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la figura: {ruta}")
    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.space_before = Pt(10)
    parrafo.paragraph_format.space_after = Pt(3)
    parrafo.paragraph_format.keep_with_next = True
    parrafo.add_run().add_picture(str(ruta), width=Cm(spec.get("ancho", 15.4)))

    pie = doc.add_paragraph()
    pie.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pie.paragraph_format.space_after = Pt(2)
    run = pie.add_run(f"Figura {numero}. ")
    run.bold = True
    run.font.size = Pt(9)
    run.font.name = FUENTE
    run = pie.add_run(spec["titulo"])
    run.font.size = Pt(9)
    run.font.name = FUENTE

    fuente = doc.add_paragraph()
    fuente.paragraph_format.space_after = Pt(10)
    run = fuente.add_run("Fuente: " + spec["fuente"])
    run.font.size = Pt(8)
    run.font.name = FUENTE
    run.font.color.rgb = GRIS


def renderizar_referencia(doc: Document, texto: str) -> None:
    """Agrega una referencia bibliográfica con sangría francesa (APA 7)."""
    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    parrafo.paragraph_format.left_indent = Cm(0.9)
    parrafo.paragraph_format.first_line_indent = Cm(-0.9)
    parrafo.paragraph_format.space_after = Pt(4)
    agregar_texto_enriquecido(parrafo, texto, tamano=10)


def renderizar_cita(doc: Document, texto: str) -> None:
    """Agrega un bloque de cita o traza con sangría."""
    for linea in texto.split("\n"):
        parrafo = doc.add_paragraph()
        parrafo.paragraph_format.left_indent = Cm(0.8)
        parrafo.paragraph_format.space_after = Pt(2)
        parrafo.paragraph_format.line_spacing = 1.0
        parrafo.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = parrafo.add_run(linea if linea else " ")
        run.font.size = Pt(9)
        run.font.name = "Consolas" if spec_es_traza(texto) else FUENTE
        run.font.color.rgb = TINTA


def spec_es_traza(texto: str) -> bool:
    """Decide si un bloque de cita debe usar tipografía monoespaciada."""
    return "->" in texto or "tool_calls" in texto


def renderizar(doc: Document, bloques: list[tuple]) -> tuple[int, int]:
    """Recorre los bloques y devuelve el número de tablas y figuras."""
    tablas = 0
    figuras = 0
    for bloque in bloques:
        tipo = bloque[0]
        if tipo == "h1":
            renderizar_titulo(doc, 1, bloque[1])
        elif tipo == "h2":
            renderizar_titulo(doc, 2, bloque[1])
        elif tipo == "h3":
            renderizar_titulo(doc, 3, bloque[1])
        elif tipo == "p":
            renderizar_parrafo(doc, bloque[1])
        elif tipo == "ul":
            renderizar_lista(doc, bloque[1])
        elif tipo == "ol":
            renderizar_lista(doc, bloque[1], numerada=True)
        elif tipo == "tabla":
            tablas += 1
            renderizar_tabla(doc, bloque[1], tablas)
        elif tipo == "figura":
            figuras += 1
            renderizar_figura(doc, bloque[1], figuras)
        elif tipo == "cita":
            renderizar_cita(doc, bloque[1])
        elif tipo == "referencia":
            renderizar_referencia(doc, bloque[1])
        elif tipo == "salto":
            doc.add_page_break()
        else:
            raise ValueError(f"Bloque desconocido: {tipo}")
    return tablas, figuras


# ---------------------------------------------------------------------------
# Constructores de bloques
# ---------------------------------------------------------------------------
def tabla(titulo, encabezado, filas, anchos, fuente, alineacion=None, nota=None, tamano=8.8):
    """Construye el bloque de una tabla."""
    return (
        "tabla",
        {
            "titulo": titulo,
            "encabezado": encabezado,
            "filas": filas,
            "anchos": anchos,
            "fuente": fuente,
            "alineacion": alineacion,
            "nota": nota,
            "tamano": tamano,
        },
    )


def figura(archivo, titulo, fuente, ancho=15.4):
    """Construye el bloque de una figura."""
    return ("figura", {"archivo": archivo, "titulo": titulo, "fuente": fuente, "ancho": ancho})


def bloques_1_4() -> list[tuple]:
    """Devuelve los bloques de las secciones 1 a 4."""
    b: list[tuple] = []

    # ------------------------------------------------------------------
    # 1. Resumen ejecutivo
    # ------------------------------------------------------------------
    b.append(("h1", "1. Resumen ejecutivo"))
    b.append(("p", (
        "Este informe consolida las tres entregas del proyecto Juris-PLN, un sistema de "
        "procesamiento de lenguaje natural sobre 23.742 sentencias de la Corte Constitucional "
        "de Colombia emitidas entre 1992 y 2021. El producto responde una pregunta acotada: "
        "dada una providencia, qué artículos de la Constitución Política (CP) la fundamentan, "
        "con evidencia textual verificable, y qué sentencias la rodean en la red de precedentes. "
        "La delimitación es deliberada: el sistema no emite conceptos jurídicos ni predice "
        "decisiones; ordena, ubica y hace auditable la evidencia que un jurista revisaría a mano."
    )))
    b.append(("p", (
        "El corpus quedó en 23.742 sentencias (17.099 de tutela, 6.246 de constitucionalidad y "
        "397 de unificación) y 260.988.316 palabras crudas, con un promedio de 10.992,68 palabras "
        "por documento. El análisis exploratorio estableció tres hechos que condicionaron el "
        "diseño: los documentos son largos (el 99,95 % supera los 512 tokens), el léxico está "
        "dominado por fórmulas procesales y la unidad de análisis útil es la sentencia completa, "
        "no el párrafo (ADR-8)."
    )))
    b.append(("p", (
        "En recuperación, la ruta adoptada (retriever denso B2 + reranker B1 + prior de hubs "
        "con alpha=1,5) obtiene un nDCG@10 de 0,6907 [0,654; 0,725] en el benchmark silver de 90 "
        "casos y de 0,7852 [0,745; 0,826] con el gold humano de 50 casos y 820 juicios, con "
        "F1@5 de 0,478 y MRR de 0,950. La entrega 2 cerraba con F1@5 de 0,425 y nDCG@10 de "
        "0,6198 en la misma familia de ruta. La mejora pareada frente a la variante previa fue de "
        "+0,0673 [+0,0418; +0,0892] sin hubs y +0,0617 [+0,0381; +0,0852] con hubs, y el gate con "
        "anotación humana confirmó el orden de las rutas (B2 > B1 > v8)."
    )))
    b.append(("p", (
        "La clasificación de área temática (M4) se mantiene en un macro-F1 de 0,2870 y quedó "
        "declarada exploratoria. La auditoría con cleanlab (Northcutt, Athalye y Mueller, "
        "2021) estimó 31,81 % de issues en out-of-fold y la validación humana de 200 "
        "documentos (O1) arrojó una exactitud de 23,5 % "
        "(kappa 0,153) frente a la etiqueta automática: el problema es de etiqueta, no solo de "
        "modelo (ADR-19). El área se muestra como indicio, nunca como conclusión."
    )))
    b.append(("p", (
        "El agente conversacional se evaluó con un gold set de 100 prompts etiquetados por ruta. "
        "En la versión final acierta el 90,0 % del ruteo estricto, rechaza el 100,0 % de las "
        "consultas fuera de alcance, no rechaza ningún saludo, contiene el 100,0 % de los "
        "adversarios, emite tool calls en el 100,0 % de las consultas en alcance y verifica el "
        "100,0 % de las citas contra la evidencia recuperada; la latencia de la ruta del agente "
        "es de 13,4 s en p50 y 60,8 s en p95. El producto se entrega como API FastAPI y frontend "
        "Streamlit separados, con 44 pruebas aprobadas y 1 omitida en el backend y 29 aprobadas y "
        "1 omitida en el frontend."
    )))
    b.append(("p", (
        "Las limitaciones se declaran en el capítulo 10: dependencia de un LLM local, corpus "
        "hasta 2021, ausencia de validación por un experto legal, M4 no fiable, latencia alta en "
        "p95 y riesgo de alucinación acotado por el grounding. Cada cifra de este informe "
        "proviene de un JSON de resultados o de un informe previo, y se identifica en la fuente "
        "de la tabla o en el pie de la figura correspondiente."
    )))

    # ------------------------------------------------------------------
    # 2. Introducción y motivación
    # ------------------------------------------------------------------
    b.append(("h1", "2. Introducción y motivación"))
    b.append(("h2", "2.1 El problema y su escala"))
    b.append(("p", (
        "La Corte Constitucional de Colombia publica providencias desde 1992. El corpus curado "
        "para este proyecto reúne 23.742 de ellas, con 260.988.316 palabras crudas. Un documento "
        "promedio tiene 10.992,68 palabras y una mediana de 7.158 palabras limpias: son textos de "
        "decenas de páginas en los que la información relevante (problema jurídico, norma "
        "aplicada, decisión) convive con fórmulas procesales repetidas y citas normativas."
    )))
    b.append(("p", (
        "Esa escala produce un problema práctico concreto. Para saber qué artículos de la "
        "Constitución fundamentan una sentencia, un jurista debe leer el documento completo, "
        "localizar las citas dispersas y distinguir las que sostienen la ratio de las que "
        "aparecen en antecedentes o en transcripciones de otras providencias. La tarea es "
        "repetitiva, costosa y difícil de auditar: dos analistas pueden llegar a conjuntos de "
        "artículos distintos sin que ninguno esté equivocado, porque la noción misma de "
        "fundamento admite grados."
    )))
    b.append(("p", (
        "El proyecto ataca esa dificultad con tres piezas. Primero, un modelo de recuperación que "
        "ordena los artículos de la CP por probabilidad de estar vinculados a la sentencia. "
        "Segundo, un módulo de evidencia que devuelve los fragmentos del propio documento que "
        "activan cada artículo, de modo que la sugerencia sea verificable. Tercero, una capa de "
        "contexto jurisprudencial que sitúa la sentencia en la red de citas de la Corte y muestra "
        "qué precedentes la rodean y cuáles son centrales. Ninguna de las tres piezas reemplaza "
        "el juicio del jurista; las tres reducen el tiempo de localización y hacen explícito el "
        "criterio con el que se sugiere un artículo."
    )))

    b.append(("h2", "2.2 Público objetivo y utilidad delimitada"))
    b.append(("p", (
        "El sistema está dirigido a tres perfiles. El primero es el litigante o el abogado de "
        "entidad pública que prepara un caso y necesita ubicar con rapidez la jurisprudencia "
        "constitucional que trata un artículo o un problema. El segundo es el investigador en "
        "derecho constitucional que trabaja con el corpus completo y requiere medir recurrencias, "
        "centralidad de precedentes o evolución temática. El tercero es el docente que prepara "
        "material de clase y necesita ejemplos verificables con su fuente."
    )))
    b.append(("p", (
        "La utilidad está delimitada en cuatro puntos. El sistema (i) sugiere artículos con un "
        "puntaje y una lista de fragmentos de respaldo; (ii) informa la incertidumbre de esa "
        "sugerencia mediante intervalos de confianza y advertencias de calibración; (iii) separa "
        "lo que es evidencia textual de lo que es inferencia del modelo; y (iv) declara sus "
        "límites en la propia interfaz. No sustituye la lectura del expediente, no produce "
        "dictámenes y no estima probabilidades de éxito de una acción."
    )))
    b.append(("p", (
        "La diferencia frente a un buscador genérico o a un asistente conversacional sin "
        "herramientas es el anclaje. Cada artículo que el sistema muestra proviene de una cita "
        "extraída del documento o de una recuperación sobre el texto de la CP, y cada cita que el "
        "agente escribe se contrasta contra la salida de las herramientas; si no existe, se "
        "elimina. Esa decisión de arquitectura (ADR-22 y ADR-23) es la que permite hablar de "
        "grounding medible y no solo de fluidez."
    )))
    b.append(("p", (
        "El proyecto se inscribe en una línea de trabajo consolidada. La clasificación y el "
        "análisis de textos legales con PLN tiene antecedentes en la predicción de decisiones "
        "(Aletras et al., 2016), en la clasificación por área jurídica (Soh et al., 2019) y en el "
        "uso de modelos de dominio legal (Chalkidis et al., 2020). La recuperación de estatutos y "
        "precedentes se beneficia de BM25 (Robertson y Zaragoza, 2009) y de los rerankers "
        "neuronales (Nogueira y Cho, 2019). Este trabajo retoma esas piezas y las combina con un "
        "protocolo de evaluación por intervalos de confianza y con anotación humana, que es "
        "justamente lo que suele faltar en los prototipos sobre jurisprudencia latinoamericana."
    )))

    b.append(("h2", "2.3 Objetivos y preguntas de investigación"))
    b.append(("p", (
        "El objetivo general es construir y evaluar un sistema que vincule sentencias de la Corte "
        "Constitucional con los artículos de la CP que las fundamentan, con evidencia textual y "
        "contexto de precedentes. Los objetivos específicos son: (i) caracterizar el corpus y su "
        "evolución; (ii) diseñar un protocolo de recuperación robusto a la deriva temporal y "
        "medirlo con intervalos de confianza; (iii) evaluar la fiabilidad de las etiquetas "
        "temáticas disponibles; y (iv) construir un producto usable con guardrails verificables."
    )))
    b.append(("p", (
        "Tres preguntas guiaron el trabajo. Primera: ¿puede un modelo de dominio, entrenado con "
        "las citas propias del corpus, superar a un reranker genérico en la recuperación de "
        "artículos constitucionales? Segunda: ¿es fiable la etiqueta de área temática derivada "
        "del bloque mayoritario de citas, o el techo del clasificador es un techo de etiqueta? "
        "Tercera: ¿puede un asistente conversacional mantenerse en el dominio jurídico sin "
        "rechazar interacciones conversacionales legítimas y sin inventar citas? Las respuestas "
        "se desarrollan en los capítulos 5, 6 y 9."
    )))
    b.append(("p", (
        "Queda fuera de alcance: la predicción del sentido de la decisión, la generación de "
        "escritos, el análisis de documentos anteriores a 1992 o posteriores a 2021 (el corpus "
        "C3 de 2022-2026 se usa solo como prueba de robustez), la asesoría legal y cualquier "
        "afirmación sobre la corrección jurídica de una providencia."
    )))

    # ------------------------------------------------------------------
    # 3. Datos y EDA
    # ------------------------------------------------------------------
    b.append(("h1", "3. Datos y análisis exploratorio"))
    b.append(("h2", "3.1 Fuente y composición del corpus"))
    b.append(("p", (
        "El corpus proviene del repositorio abierto sentencias-corte-cons-colombia-1992-2021, "
        "publicado en Hugging Face por el usuario Manuel bajo licencia CC-BY-4.0 (Manuel, 2022). "
        "El archivo original es un CSV de 1,561 GB con 23.750 filas y dos columnas: un "
        "identificador consecutivo y el texto completo de la providencia. El nombre del archivo "
        "sugiere cobertura hasta 2022, pero la verificación mostró que no hay sentencias de 2022 "
        "y que 2021 está incompleto, con 20 documentos."
    )))
    b.append(("p", (
        "La curaduría excluyó 8 filas que no son sentencias: 6 autos (identificados con el "
        "prefijo A) y 2 fragmentos HTML residuales del proceso de recolección. El corpus de "
        "análisis quedó en 23.742 documentos. La Tabla 1 resume la composición por tipo de "
        "providencia. La distribución es muy desigual: las tutelas son el 72,0 % del corpus y "
        "las sentencias de unificación menos del 2 %."
    )))
    b.append(tabla(
        "Composición del corpus por tipo de providencia.",
        ["Tipo", "Providencias", "% del corpus", "MTLD medio"],
        [
            ["Tutela (T)", "17.099", "72,02", "61,63"],
            ["Constitucionalidad (C)", "6.246", "26,31", "59,54"],
            ["Unificación (SU)", "397", "1,67", "58,88"],
            ["Total", "23.742", "100,00", "s. d."],
        ],
        [4.6, 3.4, 3.2, 4.0],
        "resultados/diacronia.json y informe_final_e2.md, sección 3; el MTLD por tipo proviene "
        "de data/lemas y se reporta en la sección 3.4.",
        ["l", "r", "r", "r"],
        nota="El MTLD total no se reporta porque la media por documento no es agregable sin "
             "ponderar; las medias por tipo se calcularon sobre los documentos de cada grupo.",
    ))
    b.append(("p", (
        "El corpus no trae metadatos estructurados: el identificador de la sentencia (por "
        "ejemplo, T-293/97), el tipo y el año se extraen del encabezado con expresiones "
        "regulares. La extracción tuvo éxito en 23.742 de 23.750 filas (99,97 %). Esa dependencia "
        "del encabezado es una fuente de error conocida y se documenta en la sección 4.2."
    )))

    b.append(("h2", "3.2 Limpieza y normalización"))
    b.append(("p", (
        "La limpieza se aplicó por etapas medibles, porque la elección del preprocesamiento "
        "altera los resultados y su efecto depende del corpus y de la tarea (HaCohen-Kerner et "
        "al., 2020). El pipeline convierte a minúsculas, protege la letra ñ, normaliza Unicode "
        "con descomposición NFD y elimina diacríticos, conserva solo letras de [a-zñ], colapsa "
        "espacios, tokeniza por separación y descarta tokens de un carácter y la abreviatura art. "
        "La lista de palabras vacías del español de NLTK se sometió a la misma transformación: "
        "quedan 306 formas sin tildes, porque 84 formas acentuadas se normalizaron igual que el "
        "corpus y, de otro modo, dejarían de filtrarse."
    )))
    b.append(("p", (
        "El resultado son 250.012.101 tokens alfabéticos y 128.806.664 tokens de contenido tras "
        "remover las palabras funcionales; es decir, cerca del 48 % del texto limpio son "
        "stopwords. La limpieza se portó a PySpark y se verificó documento a documento: los "
        "23.742 registros coinciden con la referencia de la primera entrega en el conteo de "
        "palabras limpias. La Tabla 2 resume las medidas textuales del corpus."
    )))
    b.append(tabla(
        "Medidas textuales del corpus de análisis (23.742 sentencias, 1992-2021).",
        ["Medida", "Valor"],
        [
            ["Documentos", "23.742"],
            ["Palabras crudas", "260.988.316"],
            ["Palabras limpias", "250.012.101"],
            ["Palabras de contenido (sin stopwords)", "128.806.664"],
            ["Media de palabras crudas por documento", "10.992,68"],
            ["Media de palabras limpias por documento", "10.530,37"],
            ["Mediana de palabras limpias por documento", "7.158"],
            ["Documentos con más de 512 tokens", "99,95 %"],
            ["Proporción de stopwords sobre el texto limpio", "≈ 48 %"],
        ],
        [10.5, 5.5],
        "informe_final_e2.md, sección 2 (cifras canónicas de la entrega 1) y figuras del EDA; "
        "la mediana y el porcentaje de documentos largos provienen de "
        "codigo/entrega_2_eda/scripts/31_figuras_informe_f8x.py, que verifica ambas cifras.",
        ["l", "r"],
    ))

    b.append(("h2", "3.3 Textometría del corpus"))
    b.append(("p", (
        "Con palabras vacías, el top-5 lo encabezan de (22.543.375), la (14.299.738), el "
        "(8.225.551), que (7.505.218) y en (7.208.184). Sin palabras vacías, el léxico dominante "
        "es jurídico y procesal: articulo (947.616), derecho (898.926), ley (809.313), corte "
        "(760.197), sentencia (736.326), derechos (718.194), constitucional (689.709), tutela "
        "(649.302), accion (524.583) y caso (473.150). Las grafías sin tilde son efecto de la "
        "normalización descrita."
    )))
    b.append(("p", (
        "La distribución de frecuencias sigue la forma característica de la ley de Zipf: unas "
        "pocas formas concentran la mayor parte de las ocurrencias y una cola muy larga reúne la "
        "mayoría de los tipos. La Figura 1 muestra, además, que la distribución de longitudes "
        "está sesgada a la derecha: la media (10.530 palabras limpias) supera con holgura la "
        "mediana (7.158), de modo que un puñado de sentencias muy extensas empuja el promedio."
    )))
    b.append(figura(
        "eda/fig_longitudes_documentos.png",
        "Longitud de las 23.742 sentencias. El panel izquierdo muestra la distribución de "
        "palabras limpias por documento, con mediana de 7.158 y media de 10.530; el derecho "
        "compara la distribución por tipo de providencia.",
        "figuras/eda/fig_longitudes_documentos.png, generada por "
        "codigo/entrega_2_eda/scripts/31_figuras_informe_f8x.py sobre data/.",
    ))
    b.append(("p", (
        "El espectro de frecuencias confirma que el vocabulario de contenido tiene una cola "
        "larga pronunciada y que los estratos de altísima frecuencia concentran una fracción "
        "desproporcionada de los tokens. Esa estructura es la que justifica dos decisiones "
        "posteriores: usar TF-IDF con mínimo de documento para el clasificador y añadir "
        "stopwords jurídicas y bigramas de colocación al índice de recuperación."
    )))
    b.append(figura(
        "eda/fig_espectro_frecuencias.png",
        "Espectro de frecuencias del vocabulario de contenido (sin palabras vacías). El panel "
        "izquierdo muestra los tipos por rango de frecuencia en escala logarítmica; el derecho, "
        "la concentración de tipos y de tokens por estrato.",
        "figuras/eda/fig_espectro_frecuencias.png, generada por "
        "codigo/entrega_2_eda/scripts/31_figuras_informe_f8x.py sobre data/.",
    ))
    b.append(("p", (
        "Los n-gramas están dominados por fórmulas institucionales. El bigrama más frecuente es "
        "corte constitucional (286.786) y el trigrama más frecuente es accion de tutela "
        "(344.893). Las colocaciones extraídas por verosimilitud (log-likelihood ratio) fueron "
        "más precisas que las de frecuencia pura: el 75 % de las fórmulas resultó válido frente "
        "al 55 % con frecuencia y al 0 % con información mutua puntual. Por eso el índice de "
        "recuperación incorpora bigramas de colocación."
    )))
    b.append(("p", (
        "La distribución morfosintáctica es la esperable en un corpus jurídico: sustantivo "
        "(22,58 %) y adposición (20,45 %) encabezan el conteo, seguidos de determinante y "
        "adjetivo. Las sentencias de constitucionalidad se distinguen por la presencia de verbo "
        "en el top-5, mientras que en tutelas y unificaciones aparece el nombre propio. La "
        "Figura 3 compara ambos niveles."
    )))
    b.append(figura(
        "eda/fig_pos_distribucion.png",
        "Distribución morfosintáctica del corpus. El panel izquierdo presenta las diez etiquetas "
        "POS más frecuentes; el derecho compara las seis primeras por tipo de providencia "
        "(T, C y SU).",
        "figuras/eda/fig_pos_distribucion.png, generada por "
        "codigo/entrega_2_eda/scripts/31_figuras_informe_f8x.py sobre data/lemas.",
    ))

    b.append(("h2", "3.4 Diversidad léxica y diacronía"))
    b.append(("p", (
        "La diversidad léxica se midió con MTLD, una medida que estima la longitud media de los "
        "segmentos textuales que mantienen una razón tipo-token dada (McCarthy y Jarvis, 2010). "
        "El MTLD medio es mayor en tutelas (61,63) que en constitucionalidad (59,54) y en "
        "unificaciones (58,88). La serie anual desciende de 64,88 en 1992 a 56,77 en 2021, con "
        "una tendencia de -0,319 por año. El descenso debe leerse con cautela: el MTLD se "
        "correlaciona con la longitud del documento (Spearman -0,2628), de modo que parte de la "
        "caída podría reflejar cambios en la extensión y no solo en la riqueza léxica."
    )))
    b.append(figura(
        "eda/fig_mtld_distribucion.png",
        "Diversidad léxica MTLD. El panel izquierdo compara la media por tipo de providencia "
        "(T 61,63; C 59,54; SU 58,88); el derecho muestra la media anual entre 1992 y 2021, con "
        "tendencia de -0,319 por año.",
        "figuras/eda/fig_mtld_distribucion.png y resultados/diacronia.json (mtld.por_ano, "
        "tendencia_mtld_por_ano).",
    ))
    b.append(("p", (
        "El análisis diacrónico de áreas (Hamilton, Leskovec y Jurafsky, 2016) muestra un "
        "desplazamiento de la agenda de la Corte. "
        "Entre el primer trienio (1992-1994) y el último (2019-2021), el área de derechos "
        "fundamentales cae 24,2 puntos porcentuales (de 59,6 % a 35,4 %) y la de principios "
        "fundamentales crece 20,2 puntos (de 21,1 % a 41,3 %). En 2021, con solo 20 documentos, "
        "las proporciones son 30,0 % y 50,0 % respectivamente, y por eso el trienio se reporta "
        "agregado y no como año suelto."
    )))
    b.append(("p", (
        "El léxico acompaña ese desplazamiento. En la transición 1992-2001 a 2002-2011 suben "
        "términos asociados a la reforma agraria y a la política territorial (incoder, 7,13; "
        "rupd, 7,04; fonvivienda, 6,74; bancadas, 6,69). En la transición 2002-2011 a 2012-2021 "
        "suben covid (9,51), uariv (9,21), jep (9,04) y coronavirus (8,97): el vocabulario pasa "
        "del conflicto armado y la tierra al posacuerdo y la pandemia. La Figura 4 resume los "
        "extremos de la segunda transición."
    )))
    b.append(figura(
        "eda/fig_logodds_periodos.png",
        "Cambio léxico entre 2002-2011 y 2012-2021. El panel izquierdo muestra los lemas con "
        "mayor log-odds de subida y el derecho los de mayor caída, con un mínimo de 50 "
        "apariciones por lema.",
        "figuras/eda/fig_logodds_periodos.png y resultados/diacronia_logodds.csv.",
    ))

    b.append(("h2", "3.5 Granularidad y consecuencias metodológicas"))
    b.append(("p", (
        "Una decisión temprana fue determinar la unidad de análisis. El corpus no trae doble "
        "salto de línea, de modo que el párrafo detectable coincide con la oración en 600 de 600 "
        "casos revisados. La alternativa era segmentar por subsentencia usando marcadores "
        "temáticos, pero esos marcadores aparecen solo en el 4,64 % de las unidades y la pureza "
        "temática no superó el criterio pre-registrado de +15 %. La conclusión (ADR-8) fue usar "
        "la sentencia completa como unidad, aun sabiendo que mezcla secciones heterogéneas."
    )))
    b.append(("p", (
        "La decisión tiene costos. Un documento de 11.000 palabras contiene antecedentes, "
        "consideraciones y resuelve; el modelo que lo representa como un solo vector diluye la "
        "señal de las secciones cortas. También tiene ventajas: evita errores de segmentación y "
        "hace que la evaluación sea comparable entre documentos. En recuperación, el costo se "
        "mitigó con chunking y max-pooling por pasaje, no cambiando la unidad de análisis."
    )))

    b.append(("h2", "3.6 Hallazgos del análisis exploratorio"))
    b.append(("ul", [
        "**Documentos largos y asimétricos.** La media de 10.530 palabras limpias duplica la "
        "mediana de 7.158 y el 99,95 % de los documentos supera los 512 tokens. Cualquier modelo "
        "con ventana fija pierde la mayor parte del texto.",
        "**Fórmulas por encima de contenido.** El 48 % de los tokens limpios son stopwords y los "
        "n-gramas más frecuentes son fórmulas institucionales (corte constitucional, accion de "
        "tutela), no conceptos jurídicos.",
        "**Cola larga.** El espectro de frecuencias muestra una concentración extrema: unos pocos "
        "términos explican la mayoría de las ocurrencias y la mayor parte del vocabulario aparece "
        "una o pocas veces.",
        "**Perfil morfosintáctico estable.** Sustantivo y adposición dominan el corpus; las "
        "sentencias C se separan por el peso del verbo y las T/SU por el nombre propio.",
        "**Diversidad léxica en descenso.** El MTLD cae de 64,88 a 56,77 entre 1992 y 2021 "
        "(-0,319 por año), con una correlación negativa moderada con la longitud del documento.",
        "**Desplazamiento temático.** Derechos fundamentales pierde 24,2 pp y principios "
        "fundamentales gana 20,2 pp entre el primer y el último trienio, según la etiqueta "
        "automática de área.",
        "**Vocabulario que sigue la historia del país.** El léxico pasa de incoder, rupd y "
        "bancadas a covid, uariv, jep y coronavirus en las dos transiciones de periodo.",
        "**La sentencia es la unidad.** El párrafo no es recuperable de forma fiable y la "
        "subsentencia no aporta pureza suficiente; la unidad de análisis es el documento "
        "completo (ADR-8).",
    ]))
    b.append(("p", (
        "Estos hallazgos no son decorativos: cada uno tiene una consecuencia de diseño. La "
        "longitud justificó el chunking y descartó el truncamiento a 512 tokens como estrategia "
        "principal. Las fórmulas justificaron las stopwords jurídicas. La cola larga justificó "
        "TF-IDF con mínimo de documento. El desplazamiento temático justificó estratificar el "
        "benchmark por periodo. La granularidad fijó la unidad de evaluación."
    )))

    # ------------------------------------------------------------------
    # 4. Metodología
    # ------------------------------------------------------------------
    b.append(("h1", "4. Metodología"))
    b.append(("h2", "4.1 Marco CRISP-DM y organización por módulos"))
    b.append(("p", (
        "El proyecto siguió CRISP-DM con ciclos incrementales. La primera entrega cubrió la "
        "comprensión del negocio y de los datos, y produjo el corpus curado y la textometría. La "
        "segunda entrega desarrolló la preparación y el modelado, y cerró con un producto de "
        "lenguaje de línea de comandos y un QA con veredicto GO. La tercera entrega corrigió el "
        "protocolo de evaluación, adoptó los modelos de dominio, auditó la etiqueta temática y "
        "convirtió el producto en una aplicación con API, agente e interfaz."
    )))
    b.append(("p", (
        "El trabajo se organizó en módulos con identificadores estables. Los más relevantes para "
        "este informe son M0 (validación de entrada), M2 (extracción de entidades), M4 "
        "(clasificación de área), M5 (red de citas), M6 (diacronía) y M7 (vínculo entre sentencia "
        "y artículos de la CP). La Tabla 3 resume su función, la técnica empleada y el resultado "
        "principal."
    )))
    b.append(tabla(
        "Módulos del sistema, técnica y resultado principal.",
        ["Módulo", "Función", "Técnica", "Resultado"],
        [
            ["M0", "Validar si el texto es una providencia de la Corte", "Tres reglas sobre el encabezado", "Exactitud 1,0 en 52 casos"],
            ["M1", "Lematizar el corpus", "spaCy es_core_news_md", "195.959 lemas"],
            ["M2", "Extraer artículos, normas, citas y ponente", "Regex con ventana de 60 caracteres", "Precisión 1,00 en artículos CP"],
            ["M3", "Asignar área temática por bloque de la CP", "Bloque mayoritario de citas", "99,63 % asignado"],
            ["M4", "Clasificar el área temática", "TF-IDF de lemas + regresión logística", "Macro-F1 0,2870"],
            ["M5", "Construir la red de citas y precedentes", "Grafo dirigido + PageRank", "456.321 aristas resueltas"],
            ["M6", "Medir la diacronía", "Log-odds, MTLD y áreas por periodo", "Tendencia MTLD -0,319/año"],
            ["M7", "Vincular la sentencia con artículos de la CP", "BM25 + B2 + B1", "nDCG@10 0,6907"],
        ],
        [1.5, 4.4, 5.3, 4.8],
        "plan/PLAN.md, informe_final_e2.md (sección 5) y resultados/metricas.json; los valores "
        "corresponden a la ruta adoptada en la entrega final.",
        ["c", "l", "l", "l"],
        tamano=8.5,
    ))

    b.append(("h2", "4.2 M0 y M2: validación y extracción"))
    b.append(("p", (
        "M0 decide si un texto es una providencia de la Corte. La heurística usa tres reglas "
        "sobre el encabezado y se evaluó con 30 positivos (uno por año, estratificados por tipo) "
        "y 22 negativos tomados de discursos de posesión presidencial del corpus abierto "
        "open_corpus_co_es, de Gaitán en 1942 a De la Espriella en 2026. El resultado fue 30 "
        "verdaderos positivos, 22 verdaderos negativos y ninguna confusión: exactitud de 1,0 en "
        "52 casos. La muestra es pequeña y se declara como limitación."
    )))
    b.append(("p", (
        "M2 extrae del texto los artículos de la CP citados, las normas, las sentencias citadas "
        "y el ponente, con expresiones regulares y una ventana de 60 caracteres para resolver "
        "contextos como artículo 86 o art. 86. La precisión muestral sobre 100 documentos (20 por "
        "tipo, semilla 42) fue de 1,00 para artículos CP, 1,00 para normas, 0,90 para sentencias "
        "resueltas en el corpus, 1,00 para el rango válido de sentencias y 1,00 para ponente. La "
        "cobertura global fue de 98,71 % en citas CP, 99,67 % en normas, 99,94 % en sentencias "
        "citadas y 85,38 % en ponente; esta última cae a 57,28 % en 2017-2021, porque el "
        "encabezado cambia de formato."
    )))
    b.append(("p", (
        "La extracción produjo 540.181 aristas documento-cita con self-loops y 516.529 sin "
        "ellos. Los artículos más citados son el 241 (20.289 documentos), el 86 (16.405) y el 13 "
        "(8.757). Esa distribución no es neutra: los artículos procesales aparecen en casi "
        "cualquier providencia y los derechos concretos solo en algunas, lo que más adelante "
        "motivó el prior de hubs."
    )))

    b.append(("h2", "4.3 M4 y M5: área temática y red de citas"))
    b.append(("p", (
        "M4 clasifica cada sentencia en una de 16 áreas derivadas de los bloques temáticos de la "
        "CP. La etiqueta de origen (M3) se construye asignando cada artículo citado a su bloque y "
        "eligiendo el bloque mayoritario, con desempate por el artículo de menor número y un "
        "respaldo léxico para los documentos sin citas. El etiquetado cubrió el 99,63 % del "
        "corpus: 98,71 % por cita CP, 0,92 % por respaldo y 0,37 % sin área. La distribución está "
        "muy desbalanceada: derechos fundamentales concentra el 42,46 % (10.081 documentos), "
        "principios fundamentales el 27,59 % (6.551) y derechos sociales el 12,05 % (2.861); "
        "ocho clases tienen menos del 1 %."
    )))
    b.append(("p", (
        "El clasificador es una regresión logística sobre TF-IDF de lemas (100.000 features, "
        "n-gramas 1-2, mínimo de documento 3), seleccionada con Optuna y comparada con un SVM "
        "lineal y con BETO (Cañete et al., 2020). La evaluación usa dos escenarios: texto "
        "completo y texto con las "
        "citas enmascaradas, para medir si el modelo aprende del contenido o de los atajos de "
        "cita. Los resultados se presentan en el capítulo 6."
    )))
    b.append(("p", (
        "M5 construye un grafo dirigido de citas resueltas al corpus. De las 540.181 citas "
        "únicas, 456.321 resuelven a un documento del corpus (84,48 %) y 60.208 quedan sin "
        "resolver, con su año extraído cuando es posible. El grafo tiene 23.742 nodos, 7.688 "
        "componentes fuertemente conexas y una componente gigante que agrupa al 67,6 % de los "
        "nodos (16.050). El PageRank global está dominado por los hitos fundacionales: C-224 de "
        "1992 (0,00403), SU-2716 de 1997 (0,00360) y T-985 de 1993 (0,00301). El análisis por "
        "periodo muestra que la Corte cita más precedentes recientes: en 1992-2001 el "
        "top-10 de PageRank vive en 1992-1997 y en 2012-2021 vive en 2012-2017. El hallazgo "
        "conecta con la literatura que mide la importancia de precedentes con centralidad de red "
        "(Fowler et al., 2007)."
    )))

    b.append(("h2", "4.4 M6 y M7: diacronía y vínculo con la Constitución"))
    b.append(("p", (
        "M6 compara el primer trienio con el último en tres frentes: proporción de áreas, "
        "log-odds del léxico con un mínimo de 50 apariciones y MTLD por año. El módulo es "
        "descriptivo y no se usa para tomar decisiones de producto; su valor es contextual y "
        "alimenta la narrativa del informe y de la presentación."
    )))
    b.append(("p", (
        "M7 es el núcleo del producto. Dado el texto de una sentencia, recupera los artículos de "
        "la CP que la fundamentan. El protocolo de evaluación usa como referencia las citas "
        "extraídas por M2 (etiquetas silver, no humanas) sobre un benchmark de 90 casos, tres por "
        "año entre 1992 y 2021, con cupos proporcionales al tipo de providencia. Las métricas son "
        "P@5, R@5, F1@5, MRR y nDCG@10; esta última es la métrica norte porque premia el orden y "
        "no solo la presencia (Järvelin y Kekäläinen, 2002)."
    )))
    b.append(("p", (
        "La construcción del índice atravesó varias correcciones. La primera versión indexaba los "
        "497 artículos completos; la segunda indexó 931 pasajes con max-pooling por artículo y "
        "consultó el texto completo en chunks de 512 tokens con solape de 64, lo que elevó el "
        "F1@5 de 0,119 a 0,218. La tercera limpió el índice de artículos transitorios (24,4 % de "
        "contaminación en el top-5) y la cuarta añadió un reranker y un prior de hubs. La quinta, "
        "objeto de esta entrega, sustituyó el reranker genérico por modelos entrenados con las "
        "citas propias del corpus."
    )))

    b.append(("h2", "4.5 Anti-leakage, particiones y gates de adopción"))
    b.append(("p", (
        "El proyecto trata el leakage como un riesgo estructural, no como una comprobación "
        "final. En M4 se adoptó un doble escenario (texto completo frente a texto con citas "
        "enmascaradas) y una prueba de oro: un modelo entrenado solo con features de cita alcanza "
        "un F1 de 0,0611, prácticamente azar, lo que descarta que el clasificador esté usando el "
        "atajo de las citas. En M7, todas las variantes se evalúan contra el mismo benchmark y "
        "los entrenamientos de dominio excluyen explícitamente las 90 sentencias de evaluación."
    )))
    b.append(("p", (
        "Las particiones de M4 se generaron una sola vez, con semilla 42 y estratificación por "
        "clase, sobre los 23.654 documentos etiquetados (Tabla 4). No se regeneraron en ningún "
        "experimento posterior. Para la fase de auditoría se añadieron una partición temporal y "
        "un holdout sellado de 2.481 documentos, que no se usó para entrenar ni evaluar en este "
        "ciclo."
    )))
    b.append(tabla(
        "Particiones canónicas del clasificador M4.",
        ["Partición", "Documentos", "Proporción", "Uso"],
        [
            ["Entrenamiento", "16.557", "70 %", "Ajuste de modelos"],
            ["Validación", "3.549", "15 %", "Selección de hiperparámetros"],
            ["Prueba", "3.548", "15 %", "Evaluación final, una sola lectura"],
            ["Total etiquetado", "23.654", "100 %", "Semilla 42, estratificado"],
        ],
        [4.2, 3.4, 3.0, 5.4],
        "informe_final_e2.md, sección 4.5 y resultados/auditoria_etiquetas_m4 (n_train, n_val, "
        "n_test).",
        ["l", "r", "r", "l"],
    ))
    b.append(("p", (
        "El ciclo v3 añadió un gate estadístico único (ADR-16): una variante se adopta solo si "
        "la métrica norte mejora con un intervalo de confianza del 95 % obtenido por bootstrap "
        "pareado (Efron y Tibshirani, 1993; 1.000 remuestreos, semilla 42), si la mejora "
        "es consistente en las mitades A y "
        "B del benchmark y si el costo es compatible con la latencia declarada. Los smokes de 10 "
        "casos no deciden gates; esa regla se adoptó después de comprobar que dos de tres smokes "
        "habían anticipado conclusiones falsas. El criterio se aplicó por igual a las variantes "
        "que se adoptaron y a las que se descartaron."
    )))

    return b


def bloques_5_7() -> list[tuple]:
    """Devuelve los bloques de las secciones 5 a 7."""
    b: list[tuple] = []

    # ------------------------------------------------------------------
    # 5. Experimentos y resultados de recuperación
    # ------------------------------------------------------------------
    b.append(("h1", "5. Experimentos y resultados de recuperación"))
    b.append(("h2", "5.1 Línea base y ciclo v2"))
    b.append(("p", (
        "Los criterios de comparación de recuperación y de embeddings siguen los benchmarks "
        "públicos de BEIR (Thakur et al., 2021) y MTEB (Muennighoff et al., 2023). Como "
        "ninguno cubre el vínculo sentencia-artículo en español jurídico, las decisiones se "
        "toman sobre el benchmark interno de 90 casos descrito en la sección 4.5."
    )))
    b.append(("p", (
        "La primera versión del vínculo sentencia-artículos usaba BM25 sobre pasajes de la CP y "
        "alcanzó un F1@5 de 0,218 con MRR de 0,717. El ciclo v2 identificó tres problemas y los "
        "atacó en orden. Primero, el índice contenía artículos transitorios que contaminaban el "
        "top-5 en el 24,4 % de los casos y generaba 45 identificadores malformados; la limpieza "
        "dejó solo los 384 artículos permanentes y subió el F1@5 7,1 % (0,218 a 0,233). Segundo, "
        "el truncamiento de candidatos limitaba el techo; ampliar a los 50 mejores del híbrido "
        "BM25 más E5 (Wang et al., 2022) llevó el F1@5 a 0,340 y el nDCG@10 a 0,513. "
        "Tercero, el orden dentro del "
        "top-10 no explotaba la frecuencia global de citación de cada artículo; un prior de hubs "
        "calibrado con validación cruzada (alpha=1,5) elevó el F1@5 a 0,425 y el nDCG@10 a 0,620."
    )))
    b.append(("p", (
        "El ciclo v2 cerró con F1@5 de 0,425, MRR de 0,950, nDCG@10 de 0,6198 y F1@10 de 0,379 "
        "en la ruta híbrida con reranker genérico y prior de hubs. La ruta por defecto del "
        "producto, BM25 con prior de hubs y alpha=0,75, quedó en F1@5 de 0,396, MRR de 0,901 y "
        "nDCG@10 de 0,601. El QA de regresión emitió GO con 10 de 10 casos end-to-end y sin "
        "deriva de métricas."
    )))
    b.append(("p", (
        "Sin embargo, el ciclo v2 dejó tres deudas que el ciclo v3 convirtió en su agenda. La "
        "primera era de protocolo: las adopciones se justificaban con deltas puntuales, sin un "
        "criterio estadístico único ni intervalos de confianza. La segunda era de referencia: "
        "todas las métricas se medían contra citas NER (silver), sin control humano. La tercera "
        "era de techo: la clasificación de área llevaba dos ciclos estancada en 0,2870 sin que se "
        "supiera cuánto de ese techo era ruido de etiqueta. La Tabla 5 resume las rutas "
        "evaluadas; las cifras de la entrega 3 se discuten en las secciones siguientes."
    )))
    b.append(tabla(
        "Rutas de recuperación evaluadas en el benchmark silver de 90 casos (1992-2021).",
        ["Ruta", "nDCG@10", "IC 95 %", "F1@5", "MRR"],
        [
            ["BM25 por pasajes (v1, entrega 2)", "s. d.", "s. d.", "0,218", "0,717"],
            ["Híbrido + bge + hubs (v2, v8)", "0,6198", "s. d.", "0,425", "0,950"],
            ["B1 + E5, sin hubs (v12)", "0,6253", "[0,590; 0,663]", "0,446", "0,899"],
            ["B1 + E5, con hubs (v12)", "0,6289", "[0,596; 0,662]", "s. d.", "s. d."],
            ["B2 + B1, sin hubs (v13)", "**0,6926**", "[0,657; 0,725]", "**0,478**", "**0,950**"],
            ["B2 + B1, con hubs (v13)", "**0,6907**", "[0,654; 0,725]", "s. d.", "s. d."],
            ["B2 época 2, sin/con hubs (v14)", "0,6939 / 0,6840", "s. d.", "s. d.", "s. d."],
            ["C2 consulta-LLM (ALIA zero-shot)", "0,493", "[0,453; 0,536]", "s. d.", "s. d."],
            ["C2 rerank listwise (ALIA zero-shot)", "0,473", "[0,429; 0,516]", "s. d.", "s. d."],
            ["C2-LoRA, top-50 recomputado", "0,6842", "s. d.", "s. d.", "s. d."],
        ],
        [6.2, 3.1, 3.3, 1.8, 1.6],
        "resultados/vinculo_cp_benchmark_v13.json, resultados/c3_b1_reranker.json, "
        "metricas.json (b2_retriever, b2_ep2_comparacion, c2_lora_eval), "
        "resultados/c2_llm/ y informe_final_ciclo_v2.md. s. d. = sin dato en la fuente.",
        ["l", "r", "c", "r", "r"],
        nota="La ruta v2 se midió con el mismo benchmark de 90 casos pero con otra referencia de "
             "candidatos; por eso su IC no es comparable con el de v12/v13. Las filas de C2 y "
             "LoRA son variantes negativas y no forman parte del producto.",
        tamano=8.4,
    ))

    b.append(("h2", "5.2 Ciclo v3, Sprint A: diagnóstico y modelos legales"))
    b.append(("p", (
        "El Sprint A ejecutó cinco experimentos sobre los 90 casos y no adoptó ninguno. El valor "
        "del sprint fue diagnóstico. A1 cuantificó el techo: el recall de candidatos en el top-50 "
        "del híbrido es 0,509 (0,413 en BM25), de modo que ningún reranker puede superar ese "
        "límite; el techo del F1@5 con referencia silver es 0,649; y el prior de hubs aporta "
        "0,108 de nDCG@10 al reranker genérico. Esa última cifra explica por qué el prior "
        "compensaba al reranker y no al revés, y anticipó el hallazgo del Sprint B."
    )))
    b.append(figura(
        "c3_recall_curvas.png",
        "Diagnóstico A1: recall@k de las estrategias de primera etapa. El recall del top-50 del "
        "híbrido (0,509) fija el techo alcanzable por cualquier reranker que solo reordene esos "
        "candidatos.",
        "figuras/c3_recall_curvas.png y resultados/c3_diagnostico_recall.json.",
    ))
    b.append(("p", (
        "A2 comparó cuatro embeddings legales en español en la primera etapa. ALIA-MrBERT logró "
        "el mejor recall de candidatos (0,723 en el top-100 frente a 0,619 de E5) y una mejora "
        "significativa de F1@10 con hubs (+0,017 [+0,001; +0,031]), pero no ganó en nDCG@10 con "
        "intervalo de confianza; por eso pasó al Sprint B como base del retriever denso y no como "
        "sustituto directo. A3 midió el reranker legal de SINAI y una fusión por rango "
        "recíproco (Cormack, Clarke y Büttcher, 2009): "
        "SINAI perdió con intervalo (-0,143 sin hubs y -0,055 con hubs) y la fusión no se "
        "distinguió de la referencia (-0,007 y -0,020). A4 mostró que ninguna fusión de primera "
        "etapa supera la mezcla 0,5/0,5 y que el top-50 es la profundidad recomendada (+0,027 "
        "[+0,014; +0,039] sin hubs y no inferior con hubs); el top-200 se rechazó. A5 no adoptó "
        "la calibración C ni el ensamble con BETO: C frente a la referencia dio +0,011 "
        "[-0,023; +0,030], con el intervalo cruzando cero, y el ensamble quedó en 0,2901 frente a "
        "0,2980."
    )))
    b.append(("p", (
        "El balance del Sprint A es metodológico antes que técnico. Cinco experimentos, tres de "
        "ellos con modelos legales específicos para español, no movieron el producto. Los smokes "
        "de 10 casos habían sugerido mejoras en dos de tres ocasiones y el run completo las "
        "desmintió. Esa experiencia es la que justificó elevar el gate a un criterio pareado con "
        "intervalos de confianza y prohibir que los smokes decidieran adopciones."
    )))

    b.append(("h2", "5.3 Sprint B: modelos de dominio B1 y B2"))
    b.append(("p", (
        "El Sprint B partió de una hipótesis distinta: si el problema es de dominio, los modelos "
        "deben entrenarse con el dominio. Se construyeron 299.999 pares de entrenamiento a partir "
        "de las citas propias del corpus (150.000 positivos y 149.999 negativos: 52,8 % "
        "aleatorios, 29,9 % co-citados y 17,3 % recuperados por BM25), excluyendo las 90 "
        "sentencias del benchmark. Con esos pares se ajustó un reranker cross-encoder sobre "
        "bge-reranker-v2-m3. El mejor checkpoint fue el paso 10.000, con AUC de validación de "
        "0,8513 y exactitud de 0,7671."
    )))
    b.append(("p", (
        "B1 se evaluó primero contra el reranker genérico en la ruta de 50 candidatos. Sin prior "
        "de hubs, el nDCG@10 subió de 0,512 [0,470; 0,554] a 0,625 [0,590; 0,663], con un delta "
        "pareado de +0,113 y F1@10 de +0,071. Con prior de hubs, en cambio, la diferencia se "
        "redujo a +0,009 [-0,007; +0,025] y dejó de ser concluyente: el prior estaba compensando "
        "al reranker genérico, no al de dominio. El gate del sprint exigía 0,66 y B1 solo "
        "alcanzaba 0,629, así que no se adoptó todavía."
    )))
    b.append(("p", (
        "B2 llevó la misma lógica a la primera etapa. Se ajustó un retriever denso contrastivo "
        "sobre ALIA-MrBERT-es-legal-administrative-embeddings con 143.004 tripletas y 96.328 "
        "pares, una época y 3.762 pasos, con un pico de 5,97 GB de VRAM. La validación dio una "
        "exactitud de tripletas de 0,6864 y un recall@10 de 0,3616. La Figura 7 muestra la curva "
        "de pérdida y la métrica de validación; el entrenamiento se estabilizó sin saturación."
    )))
    b.append(figura(
        "fig_b2_entrenamiento.png",
        "Entrenamiento del retriever denso B2: pérdida contrastiva (MNRL) promediada por ventana "
        "de pasos y métrica de validación por época. La época 1 cerró con exactitud de tripletas "
        "de 0,6864.",
        "figuras/fig_b2_entrenamiento.png y resultados/b2_entrenamiento_v1.json.",
    ))
    b.append(("p", (
        "La ruta completa, híbrido BM25+B2, rerank con B1 y prior de hubs con alpha=1,5, se "
        "evaluó en el benchmark de 90 casos (v13). Sin hubs obtuvo nDCG@10 de 0,6926 [0,657; "
        "0,725], F1@10 de 0,447, F1@5 de 0,478, MRR de 0,950, P@5 de 0,711 y R@5 de 0,424; con "
        "hubs, nDCG@10 de 0,6907 [0,654; 0,725] y F1@10 de 0,464. El delta pareado frente a v12 "
        "fue de +0,0673 [+0,0418; +0,0892] sin hubs y +0,0617 [+0,0381; +0,0852] con hubs, con "
        "probabilidad de mejora de 1,0 en ambos casos. El umbral del gate (0,66) se superó en los "
        "dos contextos y la mejora frente a ALIA en el top-100 también ganó con intervalo."
    )))

    b.append(("h2", "5.4 Adopción del producto v2.2"))
    b.append(("p", (
        "Con la evidencia anterior se adoptó la ruta v2.2 para el modo rerank del producto "
        "(ADR-18): B2 en la primera etapa, B1 en el reordenamiento y prior de hubs con alpha=1,5. "
        "La ruta por defecto, BM25 con prior de hubs y alpha=0,75, quedó intacta, y los fallbacks "
        "a E5 y a bge se mantuvieron documentados para cuando falten los modelos de dominio o la "
        "GPU. La adopción no cambió la latencia declarada del modo rerank, que sigue entre 13 y "
        "15 segundos en caliente."
    )))
    b.append(("p", (
        "El QA de la adopción emitió GO. Los diez casos end-to-end pasaron, la integridad md5 de "
        "los cinco artefactos canónicos de v2 y de los ocho componentes de B2/B1 se verificó, la "
        "equivalencia entre producto y benchmark fue exacta en 3 de 3 casos (ids y puntajes) y el "
        "modo por defecto resultó byte-comparable con el QA anterior salvo el campo de tiempo. "
        "Los controles negativos discriminaron: la suite nueva falla en 8 puntos contra v2 y en "
        "92 contra v1. Las latencias de los runs limpios fueron de 4,64 a 5,51 s en el modo por "
        "defecto y de 13,90 a 14,51 s en el modo rerank, dentro de los límites de 10 y 30 s."
    )))
    b.append(("p", (
        "La Figura 8 muestra el comportamiento por periodo de la ruta adoptada. El nDCG@10 es de "
        "0,644 en 1992-2001, 0,692 en 2002-2011 y 0,736 en 2012-2021: la ruta no se degrada en "
        "los periodos antiguos, que es donde el vocabulario y el formato difieren más del "
        "presente."
    )))
    b.append(figura(
        "deriva_retrieval_v13.png",
        "Comportamiento por periodo de la ruta v13 (B2 + B1 + hubs alpha=1,5). El nDCG@10 sube de "
        "0,644 en 1992-2001 a 0,736 en 2012-2021, sin caídas en los periodos antiguos.",
        "figuras/deriva_retrieval_v13.png y resultados/vinculo_cp_benchmark_v13.json.",
    ))

    b.append(("h2", "5.5 Validación con gold humano (B3)"))
    b.append(("p", (
        "El benchmark silver mide acuerdo con las citas extraídas, no relevancia jurídica. Para "
        "cerrar esa brecha se construyó un mini-gold de 50 casos y 820 candidatos, anotado a mano "
        "con una escala graduada de 0 a 2. La anotación tomó 41 minutos y produjo 189 grados 2, "
        "267 grados 1 y 364 grados 0. El acuerdo ítem a ítem con la pre-anotación del LLM fue "
        "moderado (kappa de Cohen de 0,3448; proporción observada de 0,5939), y el acuerdo con un "
        "segundo anotador LLM fue de 0,370 sobre 174 ítems. El Fleiss de tres evaluadores fue de "
        "0,4446, en el límite entre acuerdo moderado y sustancial."
    )))
    b.append(("p", (
        "Pese a la varianza por ítem, las métricas agregadas de ranking convergen. Con gold "
        "humano, B2 obtiene nDCG@10 de 0,7852 [0,745; 0,826], nDCG@5 de 0,7674, P@5 de 0,796, "
        "R@5 de 0,5196 y recall graduado@10 de 0,7719. B1 obtiene 0,6903 [0,645; 0,739], la "
        "referencia v8 0,5700 y la cita exacta 0,6812. El delta pareado B2 frente a B1 es de "
        "+0,0948 [+0,0668; +0,1239], con probabilidad de mejora de 1,0; B2 frente a v8 es de "
        "+0,2152 y B1 frente a v8 de +0,1204. El orden de rutas es el mismo que había estimado el "
        "proxy LLM (B2 > B1 > v8), y las diferencias absolutas son pequeñas: 0,7852 frente a "
        "0,7831 en B2, 0,6903 frente a 0,6850 en B1 y 0,5700 frente a 0,5658 en v8."
    )))
    b.append(tabla(
        "Evaluación graduada con gold humano (50 casos, 820 juicios).",
        ["Ruta", "nDCG@10", "IC 95 %", "nDCG@5", "P@5", "R@5", "Recall grad.@10"],
        [
            ["B2 (v13)", "**0,7852**", "[0,745; 0,826]", "**0,7674**", "**0,796**", "**0,5196**", "**0,7719**"],
            ["B1 (v12)", "0,6903", "[0,645; 0,739]", "0,7281", "0,756", "0,4849", "0,6395"],
            ["Cita exacta", "0,6812", "[0,629; 0,724]", "0,6250", "0,648", "0,4115", "0,6749"],
            ["v8 (híbrido + bge + hubs)", "0,5700", "[0,522; 0,620]", "0,5974", "0,568", "0,3504", "0,5379"],
        ],
        [4.4, 2.3, 2.9, 2.1, 1.6, 1.5, 2.2],
        "metricas.json:b3_gold_humano (evaluacion_gold_humano.rutas_completo) y "
        "resultados/gold_mini/evaluacion_gold_mini.json.",
        ["l", "r", "c", "r", "r", "r", "r"],
        nota="La cita exacta no es un competidor legítimo: usa el mismo regex que generó las "
             "etiquetas y por eso mide el techo de la extracción explícita, no el desempeño de un "
             "recuperador.",
        tamano=8.4,
    ))
    b.append(("p", (
        "El resultado cumple la meta que el plan v3 había fijado para el ciclo (nDCG@10 mayor o "
        "igual a 0,70 con gold humano) y convierte la adopción de v2.2 en una decisión respaldada "
        "por anotación humana y no solo por un proxy automático. Queda declarada una advertencia: "
        "la varianza del juicio humano por ítem es alta y el kappa moderado; por eso el informe "
        "reporta el intervalo y no solo la media, y evita presentar la diferencia de 0,0948 como "
        "una verdad exacta."
    )))

    b.append(("h2", "5.6 Robustez temporal: corpus 2022-2026 (C3)"))
    b.append(("p", (
        "Para probar que la ruta no depende del periodo de entrenamiento se construyó un corpus "
        "posterior. La descarga recuperó 2.382 providencias de 2022 a 2026 (474, 584, 530, 520 y "
        "274 por año), de las cuales 2.334 tenían HTML válido y 48 resultaron stubs aún no "
        "publicados. La preparación dejó 2.330 documentos utilizables (4 con texto demasiado "
        "corto) y replicó el pipeline: NER con cobertura de citas CP de 99,96 %, etiquetado por "
        "cita de 99,96 % y 55,2 millones de lemas."
    )))
    b.append(("p", (
        "La evaluación temporal usó 100 casos (20 por año) con referencia silver. El nDCG@10 fue "
        "de 0,7472 [0,719; 0,777], el F1@5 de 0,4422, el F1@10 de 0,4723, el MRR de 0,99 y la "
        "P@5 de 0,822. Los valores por año son estables: 0,748 en 2022, 0,755 en 2023, 0,745 en "
        "2024, 0,724 en 2025 y 0,764 en 2026. El histórico comparable es 0,6907, de modo que la "
        "diferencia descriptiva es de +0,0565. La comparación no es pareada, porque las muestras, "
        "el corpus y la fuente de etiquetas difieren; aun así, no hay evidencia de degradación "
        "temporal."
    )))
    b.append(figura(
        "c3_deriva_temporal.png",
        "nDCG@10 por año en el corpus 2022-2026 (100 casos, referencia silver) frente al "
        "histórico 1992-2021. La ruta se mantiene por encima de 0,72 en todos los años.",
        "figuras/c3_deriva_temporal.png y resultados/c3_eval_temporal.json "
        "(metricas_c3.ndcg_at_10_por_anio).",
    ))

    b.append(("h2", "5.7 Rutas descartadas: LLM zero-shot, LoRA y segunda época"))
    b.append(("p", (
        "El frente C2 evaluó si un LLM legal podía mejorar la recuperación sin entrenamiento. "
        "ALIA-7B, ajustado para dominio administrativo español, se usó de dos maneras. Como "
        "intérprete de la consulta, reescribió 86 de 90 casos y la ruta resultante obtuvo nDCG@10 "
        "de 0,493 [0,453; 0,536] frente a 0,691 de v13, un delta de -0,198 [-0,238; -0,158]. Como "
        "reranker listwise, reordenó 90 de 90 casos y obtuvo 0,473 [0,429; 0,516], un delta de "
        "-0,217 [-0,258; -0,175]. La latencia sí cumplió el gate (media de 7,1 s y p95 de 11,4 s, "
        "por debajo de 30 s), pero la calidad cayó. La interpretación documentada es que ALIA "
        "produce consultas cortas y genéricas y juicios fuera de dominio para el constitucional "
        "colombiano."
    )))
    b.append(figura(
        "c2_query_llm.png",
        "C2(a): efecto de la consulta reescrita por ALIA sobre la ruta v13. La reescritura reduce "
        "el nDCG@10 de 0,691 a 0,493 y el intervalo pareado no cruza cero.",
        "figuras/c2_query_llm.png y resultados/c2_llm/.",
    ))
    b.append(("p", (
        "La respuesta natural fue adaptar el modelo con LoRA. El adaptador se entrenó con 20.000 "
        "pares de B1 (r=16, alpha=32, qkvo, una época, VRAM pico de 15,0 GiB) y se evaluó en tres "
        "niveles. En el nivel 1, la exactitud pasó de 0,503 a 0,769 y el AUC de 0,647 a 0,847, "
        "con un delta de exactitud de +0,265 [0,24; 0,29]. En el nivel 2, sobre el gold humano "
        "B3, el AUC para grado mayor o igual a 1 fue de 0,654 y la correlación de Spearman de "
        "0,297, por encima de la base (0,597 y 0,200) pero por debajo de la referencia zero-shot "
        "(0,727). En el nivel 3, con el top-50 recomputado, el nDCG@10 fue de 0,6842 frente a "
        "0,6907 de v13, un delta de -0,0065 no significativo; las fusiones con B1 quedaron entre "
        "0,6909 y 0,6923, sin significancia, y con un p95 de 45,6 s que excede el límite de 30 s. "
        "La decisión (ADR-21) fue no adoptar y cerrar el frente generativo."
    )))
    b.append(figura(
        "fig_c2_lora_gold.png",
        "C2-LoRA sobre el gold humano B3: distribución de la probabilidad asignada a cada grado y "
        "curva ROC para grado humano mayor o igual a 1 (AUC 0,654 frente a 0,597 de la base).",
        "figuras/fig_c2_lora_gold.png y resultados/c2_lora/.",
    ))
    b.append(("p", (
        "La segunda época de B2 se evaluó por completitud. La validación mejoró de 0,6864 a "
        "0,6871 en exactitud de tripletas, pero el benchmark v14 dio nDCG@10 de 0,6939 sin hubs y "
        "0,6840 con hubs frente a 0,6926 y 0,6907 de v13. El delta pareado fue de +0,0013 "
        "[-0,0093; +0,0119] con probabilidad de 0,609 sin hubs y de -0,0067 [-0,0195; +0,0066] "
        "con hubs, es decir, indistinguible del ruido. Se mantuvo la época 1."
    )))
    b.append(("p", (
        "Queda preparado el pack C1 de gold estándar: 150 casos (40 del benchmark y 110 del "
        "corpus 2022-2026) y 2.740 candidatos, todos fuera del entrenamiento de B1 y B2. La "
        "herramienta de anotación ya existe y el pack no se usó en este informe; su anotación es "
        "el paso pendiente para llevar la validación humana de 50 a 150 casos."
    )))

    b.append(("h2", "5.8 Síntesis de la decisión"))
    b.append(("p", (
        "La evidencia sostiene tres conclusiones. Primera: los modelos de dominio funcionan "
        "cuando se entrenan con las citas del propio corpus; el reranker B1 mejora al genérico "
        "sobre todo cuando no hay prior de hubs, y el retriever B2 aporta la mejora que el "
        "reranker solo no alcanzaba. Segunda: la mejora es robusta, porque gana con intervalos de "
        "confianza pareados, se mantiene en los tres periodos del benchmark y no se degrada en el "
        "corpus 2022-2026. Tercera: el frente generativo no aporta; ni el LLM zero-shot ni el "
        "LoRA superan a los encoders de dominio, y la fusión que empata excede el presupuesto de "
        "latencia."
    )))
    b.append(("p", (
        "La diferencia entre 0,6926 y 0,6907 no es una contradicción: son la misma ruta medida "
        "sin y con prior de hubs. La primera cifra describe el aporte puro de B2 más B1; la "
        "segunda es la que usa el producto, porque el prior de hubs estabiliza los artículos "
        "procesales muy citados. Que la diferencia sea pequeña es informativo: con un reranker de "
        "dominio, el prior deja de ser un parche y se vuelve un componente menor."
    )))

    # ------------------------------------------------------------------
    # 6. M4
    # ------------------------------------------------------------------
    b.append(("h1", "6. M4: clasificación de área temática"))
    b.append(("h2", "6.1 Rendimiento del clasificador"))
    b.append(("p", (
        "La Tabla 7 resume el benchmark 2x2 de clasificación. La regresión logística sobre TF-IDF "
        "de lemas enmascarado alcanza el mejor macro-F1 de test (0,2870) con una exactitud de "
        "0,5936. El SVM lineal queda en 0,2657 y BETO fine-tuneado en 0,2042. La brecha entre "
        "escenarios (texto completo frente a texto con citas enmascaradas) es de -0,0018 para "
        "LogReg, -0,0003 para el SVM y +0,0039 para BETO: no hay evidencia de leakage por cita, y "
        "la prueba de oro con features de cita solas (F1 de 0,0611) lo confirma."
    )))
    b.append(tabla(
        "Benchmark 2x2 de clasificación de área (macro-F1 en test, 16 clases).",
        ["Modelo", "Texto completo", "Texto enmascarado", "Brecha"],
        [
            ["SVM lineal", "0,2654", "0,2657", "-0,0003"],
            ["Regresión logística", "0,2852", "**0,2870**", "-0,0018"],
            ["BETO fine-tuneado", "0,2080", "0,2042", "+0,0039"],
        ],
        [5.0, 3.6, 3.8, 2.6],
        "resultados/m4_baseline.json y resultados/m4_transformer.json; informe_final_e2.md, "
        "sección 5.2.",
        ["l", "r", "r", "r"],
        nota="La exactitud del mejor modelo es 0,5936. El espacio de evaluación es de 16 clases; "
             "en el espacio fijo que excluye las clases ausentes en test el macro-F1 es 0,2691.",
    ))
    b.append(("p", (
        "El hallazgo de que LogReg supere a BETO se documentó en ADR-10. La explicación es "
        "mecánica: las sentencias promedian 11.000 palabras y la ventana de 512 tokens descarta "
        "alrededor del 95 % del contenido, incluidas las secciones donde vive la señal temática, "
        "mientras que TF-IDF sobre lemas procesa el documento completo. El resultado es "
        "consistente con la literatura sobre documentos legales largos, donde los modelos de "
        "bolsa de palabras siguen siendo competitivos (Chalkidis et al., 2022)."
    )))
    b.append(figura(
        "fig_matriz_confusion_m4.png",
        "Matriz de confusión del clasificador M4 (regresión logística enmascarada, macro-F1 "
        "0,2870) sobre los 3.548 documentos de test y las 16 clases. Las confusiones se concentran "
        "entre áreas vecinas de la CP.",
        "figuras/fig_matriz_confusion_m4.png y resultados/m4_baseline.json.",
    ))
    b.append(("p", (
        "Las importancias del modelo lineal (equivalente SHAP) están dominadas por bigramas de "
        "fórmula jurídica (competente este, política en) y por rasgos temáticos (el niño, "
        "protección especial). Esa mezcla explica el techo: el modelo aprende tanto contenido "
        "como estilo, y el estilo no distingue áreas."
    )))

    b.append(("h2", "6.2 Auditoría de ruido de la etiqueta"))
    b.append(("p", (
        "La auditoría con cleanlab reprodujo exactamente la referencia de v1 (macro-F1 de test "
        "0,287008; espacio fijo 0,2691) y cuantificó el ruido. El 31,81 % de los documentos de "
        "entrenamiento presenta issues en la validación out-of-fold (5.266 de 16.557) y el "
        "31,09 % en test fuera de muestra (1.103 de 3.548). El macro-F1 out-of-fold es 0,2597 y "
        "el intervalo de confianza del 95 % del test en espacio fijo es [0,2461; 0,2912]; el "
        "baseline de la clase mayoritaria es 0,0374."
    )))
    b.append(("p", (
        "Con soporte de al menos 100 documentos, las clases más ruidosas son rama ejecutiva "
        "(0,767), derechos sociales (0,434) y principios fundamentales (0,321); la menos ruidosa "
        "es derechos fundamentales (0,193). El patrón no es aleatorio: las clases con definición "
        "más difusa son las que concentran el desacuerdo."
    )))
    b.append(("p", (
        "Sobre ese diagnóstico se probaron tres hipótesis de mejora. H1 (reentrenamiento con "
        "ruido filtrado) no superó a la referencia: en validación perdió 0,0267 y 0,0154, y en el "
        "suplemento no visto la referencia quedó arriba (0,2301 frente a 0,2049 y 0,1976). H2 "
        "(objetivo suave) dio una mejora de +0,0344 [-0,0051; +0,0749] con p=0,128, sin ganador. "
        "H2b encontró un ganador específico de validación (+0,1716, con intervalo positivo y "
        "consistencia A/B) que se degradó en la prueba temporal (-0,1642 y -0,1611 en macro-F1 "
        "fijo, y -0,178 en exactitud). Ninguna se adoptó."
    )))
    b.append(figura(
        "fig_h1_ruido_m4.png",
        "Hipótesis H1: macro-F1 de validación en el espacio fijo de 16 clases para las variantes "
        "evaluadas y F1 por clase. Ninguna variante supera a la referencia v1r, que gana en el "
        "suplemento no visto (0,2301 frente a 0,2049 y 0,1976).",
        "figuras/fig_h1_ruido_m4.png y resultados/m4_h1_ruido_v1.json.",
    ))

    b.append(("h2", "6.3 Validación humana del silver (O1)"))
    b.append(("p", (
        "Para saber si el problema era de modelo o de etiqueta se anotaron a ciegas 200 "
        "documentos estratificados por clase silver (semilla 42, modo lote ciego). El acuerdo "
        "humano-silver fue de 23,5 % de exactitud con kappa de 0,153, un nivel leve según "
        "las bandas de Landis y Koch (1977). Los dos proxies LLM ciegos obtuvieron 41,5 % "
        "(kappa 0,269) y 29,9 % (kappa 0,203), y el acuerdo entre anotadores fue bajo, "
        "medido como en Artstein y Poesio (2008): kappa de 0,249 y 0,268 entre el humano y "
        "cada LLM, 0,230 entre los dos LLM y Fleiss de 0,208. La tarea es ambigua incluso "
        "entre anotadores."
    )))
    b.append(("p", (
        "La divergencia tiene dirección. El silver derechos fundamentales se reasignó a derechos "
        "sociales en 29 de 57 casos (exactitud de 17,5 %), el silver principios fundamentales no "
        "logró ninguna coincidencia (0 de 39) y el silver de protección y aplicación de derechos "
        "pasó a derechos sociales en 6 de 11. El spot-check cualitativo lo confirma en tres "
        "tutelas de salud y pensión (T-853/02, T-231/20 y T-235/20), donde el silver asignó "
        "bloques genéricos citados y el humano asignó el bloque del tema."
    )))
    b.append(tabla(
        "Auditoría de ruido y validación humana del silver de M4.",
        ["Indicador", "Valor"],
        [
            ["Issues de cleanlab en out-of-fold (entrenamiento)", "31,81 % (5.266 / 16.557)"],
            ["Issues de cleanlab en test fuera de muestra", "31,09 % (1.103 / 3.548)"],
            ["Macro-F1 out-of-fold", "0,2597"],
            ["IC 95 % del macro-F1 en test (espacio fijo)", "[0,2461; 0,2912]"],
            ["Baseline de la clase mayoritaria", "0,0374"],
            ["Exactitud humano-silver (O1, n=200)", "23,5 %"],
            ["Kappa humano-silver", "0,153 (leve)"],
            ["Exactitud gemma4-silver / kappa", "41,5 % / 0,269"],
            ["Exactitud qwen-silver / kappa", "29,9 % / 0,203"],
            ["Kappa inter-anotador (LLM-LLM)", "0,230"],
            ["Fleiss de tres evaluadores", "0,208 (leve)"],
        ],
        [10.2, 5.8],
        "resultados/auditoria_etiquetas_m4.json, metricas.json:o1_validacion_humana y "
        "resultados/validacion_m4/acuerdo_v1.json.",
        ["l", "r"],
    ))

    b.append(("h2", "6.4 Decisión: M4 exploratorio"))
    b.append(("p", (
        "La conclusión (ADR-19) es que el silver de M4 no es un gold fiable. El macro-F1 de "
        "0,2870 no mide solo la calidad del modelo: mide, sobre todo, el desacuerdo entre la "
        "etiqueta automática y la lectura humana de qué área trata una sentencia. Por eso M4 se "
        "mantiene como característica diagnóstica con advertencias explícitas, no se usa para "
        "decidir y no se invierte más en mejorar el clasificador sobre este objetivo. Cualquier "
        "mejora futura exige una etiqueta mejor, del tipo C1, o una redefinición de la tarea "
        "(por ejemplo, el bloque del tema y no el bloque de las citas)."
    )))
    b.append(("p", (
        "En el producto, el área se rotula como exploratoria y se acompaña del aviso de que "
        "proviene de una referencia automática. La interfaz no la usa para ordenar resultados ni "
        "para filtrar; solo la muestra como contexto. Esa decisión es una forma de honestidad "
        "metodológica: es preferible un indicador con ruido declarado que una etiqueta precisa "
        "solo en apariencia."
    )))

    # ------------------------------------------------------------------
    # 7. Correcciones respecto a la Entrega 2
    # ------------------------------------------------------------------
    b.append(("h1", "7. Correcciones respecto a la Entrega 2"))
    b.append(("h2", "7.1 Qué falló"))
    b.append(("p", (
        "La entrega 2 cerró con veredicto GO y un producto funcional, pero su propia evidencia "
        "mostraba cuatro debilidades. La primera era el protocolo de adopción: las variantes se "
        "elegían con deltas puntuales sobre 90 casos, sin intervalos de confianza ni consistencia "
        "entre mitades, y esa práctica había producido al menos una adopción dudosa en ciclos "
        "anteriores. La segunda era la referencia: todas las métricas de recuperación se medían "
        "contra citas NER, es decir, contra una etiqueta automática que podía estar equivocada en "
        "la noción misma de fundamento. La tercera era el techo no diagnosticado de M4: dos ciclos "
        "en 0,2870 sin saber cuánto correspondía a ruido de etiqueta. La cuarta era la forma del "
        "producto: una interfaz de línea de comandos, sin API, sin interfaz gráfica y sin "
        "asistente conversacional."
    )))
    b.append(("p", (
        "Había además un problema de cobertura en la primera etapa. El diagnóstico A1 mostró que "
        "el recall de candidatos del top-50 era 0,509: la mitad de los artículos relevantes nunca "
        "llegaba al reranker. Ese cuello no se arreglaba reordenando mejor, sino cambiando la "
        "primera etapa. Y había un problema de dominio: el reranker genérico bge funcionaba, pero "
        "no distinguía el lenguaje constitucional colombiano."
    )))

    b.append(("h2", "7.2 Correcciones aplicadas y su evidencia"))
    b.append(("p", (
        "La Tabla 9 relaciona cada fallo con la corrección y con la evidencia que la respalda. "
        "Las correcciones no fueron retoques cosméticos: cambiaron el protocolo, la referencia, "
        "los modelos y la forma del producto."
    )))
    b.append(tabla(
        "Correcciones entre la entrega 2 y la entrega final.",
        ["Fallo identificado", "Corrección", "Evidencia"],
        [
            ["Adopciones sin criterio estadístico único",
             "Gate ADR-16: delta pareado con IC 95 % bootstrap y consistencia A/B",
             "Sprint A: 5 experimentos sin adopción; v2.2 adoptada con delta +0,0673 [+0,0418; +0,0892]"],
            ["Referencia solo silver, sin control humano",
             "Mini-gold B3 de 50 casos y 820 juicios humanos",
             "nDCG@10 0,7852 con humano frente a 0,7831 con proxy; mismo orden de rutas"],
            ["Techo de M4 sin diagnosticar",
             "Auditoría cleanlab y validación humana O1 de 200 documentos",
             "31,81 % de issues OOF; exactitud humano-silver 23,5 % (kappa 0,153); ADR-19"],
            ["Recall de primera etapa insuficiente (0,509 en top-50)",
             "Retriever denso de dominio B2 entrenado con citas propias",
             "nDCG@10 0,6926 sin hubs frente a 0,6253 de B1 con E5"],
            ["Reranker genérico sin dominio",
             "Reranker B1 ajustado con 299.999 pares del corpus",
             "nDCG@10 0,625 frente a 0,512 de bge sin hubs (delta +0,113)"],
            ["Producto solo de línea de comandos",
             "API FastAPI, frontend Streamlit y agente con herramientas",
             "8 endpoints, 44 pruebas de backend y 29 de frontend aprobadas"],
            ["Guardrails conversacionales frágiles (v1)",
             "Router de intención reescrito y plan determinista de respaldo",
             "Rechazo fuera de alcance de 56 % a 100 %; saludos de 90 % a 100 %"],
        ],
        [4.6, 5.6, 5.8],
        "plan/decisiones/ADR-16, ADR-18, ADR-19, ADR-22 y ADR-23; metricas.json:b2_retriever, "
        "b1_reranker, b3_gold_humano y agente_guardrails_eval.",
        ["l", "l", "l"],
        tamano=8.2,
    ))
    b.append(("p", (
        "Dos correcciones merecen detalle. La primera es la del protocolo. Antes del ciclo v3, "
        "una mejora de 0,01 en nDCG@10 podía justificar un cambio; después, la misma mejora se "
        "declara indistinguible del ruido si el intervalo cruza cero. Eso ocurrió con la segunda "
        "época de B2 (+0,0013 [-0,0093; +0,0119]) y con las fusiones de C2-LoRA (+0,0002 a "
        "+0,0016). La segunda es la de la referencia. El gold humano confirmó la adopción "
        "y mostró que el proxy LLM sobreestimaba ligeramente la cita exacta (0,6437 "
        "frente a 0,6812) y que las diferencias agregadas entre humano y LLM eran pequeñas en las "
        "rutas de encoders (0,0021 en B2). Esa convergencia es la que permite usar proxies en "
        "iteraciones futuras sin renunciar a los gates humanos."
    )))

    b.append(("h2", "7.3 Guardrails: de la versión 1 a la versión 3"))
    b.append(("p", (
        "El agente conversacional se construyó en la entrega final y su primera versión falló de "
        "forma medible. Con el mismo gold set de 100 prompts y el mismo modelo, la versión 1 "
        "acertó el 77,0 % del ruteo, rechazó solo el 56,0 % de las consultas fuera de alcance, no "
        "rechazó el 90,0 % de los saludos y emitió tool calls en el 82,5 % de las consultas en "
        "alcance. Trece prompts fallaron el guardrail. Los casos concretos permiten ver el patrón: "
        "¿Cuál es el precio del dólar?, ¿Cómo se hace una pizza margarita? y ¿Quién es el "
        "presidente de Brasil? caían en aclaración en lugar de rechazo; Muy buenas se trataba "
        "como aclaración; y ¿Cómo se dice hola en japonés? se respondía como saludo, porque el "
        "detector encontraba la palabra hola sin mirar el contexto."
    )))
    b.append(("p", (
        "La versión 2 corrigió el router y llevó el rechazo fuera de alcance al 100 %, los "
        "saludos al 100 % y los tool calls al 100 %, con la misma exactitud de ruteo de 90,0 %. La "
        "versión 3 añadió cuatro ajustes: el saludo se ancla al inicio del mensaje, el "
        "identificador de ayuda acepta sufijos, los ambiguos se detectan con un patrón "
        "referencial y el fuera de alcance se vuelve la ruta por defecto cuando ninguna regla de "
        "dominio se activa. Además, si el modelo no recupera evidencia, el backend aplica un plan "
        "determinista que responde con la evidencia disponible en lugar de improvisar."
    )))
    b.append(tabla(
        "Evolución de los guardrails del agente (gold set de 100 prompts).",
        ["Métrica", "v1", "v2", "v3"],
        [
            ["Ruteo exacto", "77,0 %", "90,0 %", "**90,0 %**"],
            ["Ruta jurídica alcanzada en consultas en alcance", "100,0 %", "100,0 %", "**100,0 %**"],
            ["Saludos no rechazados", "90,0 %", "100,0 %", "**100,0 %**"],
            ["Rechazo correcto de temas fuera de alcance", "56,0 %", "100,0 %", "**100,0 %**"],
            ["Rechazo seguro (incluye aclaración)", "96,0 %", "100,0 %", "**100,0 %**"],
            ["Aclaración correcta en consultas ambiguas", "100,0 %", "100,0 %", "**100,0 %**"],
            ["Adversarios contenidos", "100,0 %", "100,0 %", "**100,0 %**"],
            ["Tool call en consultas en alcance", "82,5 %", "100,0 %", "**100,0 %**"],
            ["Grounding de citas", "100,0 %", "100,0 %", "**100,0 %**"],
            ["Latencia p50 de la ruta del agente (ms)", "12.062", "13.992", "13.444"],
            ["Latencia p95 de la ruta del agente (ms)", "57.103", "57.581", "60.799"],
            ["Advertencias emitidas", "1", "8", "10"],
        ],
        [7.6, 2.8, 2.8, 2.8],
        "evaluacion/agente_eval_v1.json, agente_eval_v2.json y agente_eval_v3.json; "
        "metricas.json:agente_guardrails_eval.comparacion_corridas.",
        ["l", "r", "r", "r"],
        nota="El gold set y el modelo no cambiaron entre corridas; solo cambió el router y el "
             "plan determinista. Las advertencias de v3 no son fallos de guardrail: nueve "
             "corresponden al plan determinista y una a una respuesta sin evidencia.",
    ))
    b.append(("p", (
        "El resultado de v3 tiene una lectura matizada. El ruteo exacto se queda en 90,0 % porque "
        "la métrica estricta exige que el modo observado coincida con el esperado, y diez "
        "consultas en alcance se resolvieron con el plan determinista (nueve) o con una respuesta "
        "sin evidencia (una). Ninguna de esas diez es una fuga de guardrail: no hay contenido de "
        "memoria ni temas ajenos. La distinción importa: el sistema puede ser conservador en la "
        "forma de responder sin perder el control de lo que responde."
    )))

    return b


def bloques_8_11() -> list[tuple]:
    """Devuelve los bloques de las secciones 8 a 11."""
    b: list[tuple] = []

    # ------------------------------------------------------------------
    # 8. Producto final
    # ------------------------------------------------------------------
    b.append(("h1", "8. El producto final"))
    b.append(("h2", "8.1 Alcance y arquitectura"))
    b.append(("p", (
        "El producto es una aplicación de investigación jurídica con una promesa acotada: dada "
        "una sentencia, mostrar los artículos de la Constitución que la fundamentan con evidencia "
        "textual y el contexto de precedentes que la rodea. La arquitectura separa tres piezas "
        "(ADR-22): un núcleo de análisis reutilizado de la entrega 2, un backend FastAPI que lo "
        "expone por HTTP y un frontend Streamlit que consume la API. La separación no es "
        "decorativa: permite probar el núcleo sin interfaz, desplegar cada parte por separado y "
        "sustituir la interfaz sin tocar los modelos."
    )))
    b.append(("p", (
        "El núcleo conserva los módulos validados: M0 para la validación de entrada, M2 para la "
        "extracción de entidades, M7 para la recuperación de artículos, M5 para la red de citas y "
        "M4 como característica exploratoria. No se reescribió el producto de la entrega 2; se "
        "importó desde SEGUNDA ENTREGA/scripts/lib/componentes_producto.py. Esa decisión reduce "
        "el riesgo de divergencia entre lo evaluado y lo entregado, que es una fuente clásica de "
        "errores en proyectos de este tipo."
    )))
    b.append(("p", (
        "La decisión de diseño más discutida fue no usar un RAG estricto. Un RAG convencional "
        "recupera fragmentos y redacta sobre ellos; aplicado a un asistente jurídico, trata un "
        "saludo como una consulta al corpus, no recupera nada y responde con un mensaje de vacío "
        "o, peor, redacta sobre fragmentos irrelevantes. El sistema adoptó un router determinista "
        "previo al modelo y un agente con herramientas (ADR-23), en la línea de los enfoques de "
        "razonamiento con acciones (Yao et al., 2023) y de la generación aumentada por "
        "recuperación cuando hay evidencia disponible (Lewis et al., 2020)."
    )))
    b.append(("ul", [
        "**Capa 1, router de intención.** Clasifica el mensaje en saludo, small talk, fuera de "
        "alcance, ambiguo o en alcance, antes de cargar el modelo. Resuelve sin LLM los casos "
        "conversacionales y los rechazos.",
        "**Capa 2, agente con herramientas.** El modelo no responde de memoria: debe llamar "
        "validar_sentencia, vincular_cp, contexto_citas, explorar_articulo o "
        "sentencias_similares. Si no hay evidencia, responde que no la encontró.",
        "**Capa 3, verificación de grounding.** Cada cita de la respuesta se contrasta contra la "
        "salida de las herramientas; si no existe, se elimina. La traza de tool calls queda "
        "registrada.",
    ]))
    b.append(("p", (
        "El modelo por defecto es qwen3.8:diario-8k-q3_k_s, servido por ollama en local y "
        "seleccionable en la barra lateral. La elección no es una recomendación de calidad: es el "
        "modelo que cupo en la VRAM disponible y que soporta tool calling de forma estable. Si el "
        "modelo elegido no soporta herramientas, el backend usa el router determinista y el LLM "
        "solo redacta."
    )))

    b.append(("h2", "8.2 API y agente"))
    b.append(("p", (
        "La API expone diez rutas. Las de lectura devuelven JSON con la procedencia de cada "
        "puntaje; las de escritura reciben el texto o el identificador de la sentencia. La Tabla "
        "11 resume el contrato."
    )))
    b.append(tabla(
        "Endpoints de la API del producto.",
        ["Método", "Ruta", "Función"],
        [
            ["GET", "/health", "Estado, artefactos presentes y componentes cargados"],
            ["GET", "/v1/models", "Modelos de ollama y si soportan tool calling"],
            ["POST", "/v1/validate", "Validación de entrada (M0)"],
            ["POST", "/v1/link", "Sentencia a artículos de la CP, área exploratoria, citas y resumen"],
            ["POST", "/v1/evidencia", "Fragmentos de la sentencia que activan cada artículo"],
            ["GET", "/v1/article/{n}", "Texto del artículo constitucional y su bloque"],
            ["GET", "/v1/article/{n}/jurisprudencia", "Sentencias que citan el artículo, con PageRank"],
            ["GET", "/v1/similar/{id}", "Sentencias similares por similitud de texto"],
            ["POST", "/v1/ask", "Agente con tool calling y guardrails"],
            ["GET", "/v1/eval/summary", "Métricas con procedencia y comparativa de rutas"],
        ],
        [1.8, 5.6, 8.6],
        "app/backend/main.py, app/backend/README.md y README.md de la entrega final.",
        ["c", "l", "l"],
    ))
    b.append(("p", (
        "El agente usa cinco herramientas. validar_sentencia aplica M0; vincular_cp ejecuta M2, "
        "M7, M4 y M5 sobre un identificador o un texto; contexto_citas devuelve las sentencias "
        "citadas con su PageRank y su área; explorar_articulo recupera el texto del artículo y su "
        "jurisprudencia; sentencias_similares calcula los vecinos por TF-IDF. La selección de "
        "herramienta la hace el modelo, no el router: el router decide si la consulta entra al "
        "agente, y el agente decide qué herramienta usar y con qué argumentos."
    )))
    b.append(("p", (
        "La suite de pruebas cubre el contrato y los caminos de error. El backend tiene 44 pruebas "
        "aprobadas y 1 omitida; el frontend, 29 aprobadas y 1 omitida. La prueba de integración "
        "con el LLM se activa con la variable JURIS_LLM_TESTS=1 y exige 14 GB de VRAM libre, "
        "porque el modelo por defecto ocupa alrededor de 13,5 GB."
    )))

    b.append(("h2", "8.3 Interfaz y pantallas"))
    b.append(("p", (
        "La interfaz tiene cuatro pantallas y un panel. P1, Sentencia a Constitución, recibe un "
        "identificador o un texto, muestra la validación M0, el área temática rotulada como "
        "exploratoria, los artículos extraídos y los recuperados con su puntaje y su bloque, las "
        "sentencias citadas, los fragmentos de evidencia por artículo y un botón de descarga del "
        "JSON. P2, Artículo a jurisprudencia, muestra el total de sentencias que citan un "
        "artículo, la distribución por año y el top-10 por PageRank con filtro por tipo T/C/SU. "
        "P3, Evaluación, presenta las métricas congeladas con su procedencia y sus avisos. La "
        "cuarta pantalla es la consulta asistida, que conversa con el agente y permite elegir "
        "modelo. El panel P5, dentro de P1, lista las cinco sentencias más similares del corpus."
    )))
    b.append(("p", (
        "Cada pantalla declara sus límites. El área temática lleva la etiqueta de exploratoria y "
        "remite a ADR-19. Los conteos de jurisprudencia se presentan como derivados de etiquetas "
        "débiles de M2, no como gold. Lo que la API no expone se muestra como no expuesto, en "
        "lugar de completarse con una estimación. Esa disciplina de interfaz es parte del "
        "resultado: un producto académico que oculta su incertidumbre no es reproducible."
    )))

    b.append(("h2", "8.4 Requisitos y operación"))
    b.append(("p", (
        "El sistema corre con Python 3.14 en el entorno /home/carlos/venv, ollama en el puerto "
        "11434 y, para la ruta con reranker o para el chat, una GPU AMD con ROCm y al menos 14 GB "
        "de VRAM libre. Antes de usar la GPU se verifica la memoria con rocm-smi; el backend "
        "espera si no hay espacio y cae al plan determinista si el modelo no responde. El backend "
        "se levanta en el puerto 8123 y el frontend en el 8502, porque el 8501 está ocupado por "
        "otro proyecto de la máquina."
    )))
    b.append(("p", (
        "Las latencias separan dos regímenes. En frío, la primera petición paga la carga del "
        "corpus, del índice BM25 y de spaCy: entre 4,64 y 5,51 s en el modo por defecto y entre "
        "15,81 y 23,03 s en el modo rerank bajo contención. En caliente, el enlace por defecto "
        "responde en 0,19 s en CPU, la evidencia en 144 ms, la jurisprudencia por artículo en 9 ms "
        "si está cacheada y las sentencias similares en 95 ms. La Tabla 12 resume los valores."
    )))
    b.append(tabla(
        "Latencias del producto por operación.",
        ["Operación", "Latencia", "Condición"],
        [
            ["Enlace por defecto", "0,19 s", "Caliente, CPU"],
            ["Enlace por defecto", "4,64 a 5,51 s", "Frío, incluye inicialización"],
            ["Enlace con rerank", "13 a 15 s", "GPU, caliente"],
            ["Enlace con rerank", "15,81 a 23,03 s", "QA bajo contención"],
            ["Evidencia por artículo", "144 ms", "Caliente"],
            ["Jurisprudencia por artículo", "9 ms", "Cacheada"],
            ["Sentencias similares", "95 ms", "Caliente"],
            ["Agente, p50 / p95", "13,4 s / 60,8 s", "40 prompts en alcance"],
            ["Validación negativa", "0,89 a 0,90 s", "Texto fuera de alcance"],
        ],
        [6.4, 4.6, 5.0],
        "README.md (resultados principales), metricas.json:qa_v2_2.latencias_s y "
        "evaluacion/agente_eval_v3.json (latencia_ms_ruta_agente).",
        ["l", "r", "l"],
        nota="El modo por defecto no requiere GPU. La ruta rerank y el chat requieren GPU con "
             "ROCm; si no hay VRAM, el backend degrada a la ruta por defecto o al plan "
             "determinista.",
    ))

    # ------------------------------------------------------------------
    # 9. Evaluación del agente
    # ------------------------------------------------------------------
    b.append(("h1", "9. Evaluación del agente y guardrails"))
    b.append(("h2", "9.1 Diseño del gold set"))
    b.append(("p", (
        "El agente se evaluó con un conjunto de 100 prompts etiquetados por ruta esperada: 40 en "
        "alcance con cánones e identificadores reales del corpus, 10 saludos, 10 de small talk, "
        "25 fuera de alcance, 5 ambiguos y 10 adversarios. La etiqueta indica qué debería hacer el "
        "sistema, no qué hizo. Las rutas cubren los cuatro modos del router (saludo o small talk, "
        "fuera de alcance, aclaración y agente) y añaden los adversarios, que prueban si el "
        "sistema se deja sacar del dominio."
    )))
    b.append(("p", (
        "La procedencia de las etiquetas exige una precisión terminológica. En este proyecto se "
        "distinguen tres tipos de referencia. Silver es una referencia automática derivada del "
        "propio corpus, como las citas extraídas por M2; no tiene contrato de anotación y puede "
        "estar equivocada en la noción misma de fundamento. Gold humano es anotación hecha por una "
        "persona con una guía, como B3 y O1. Gold-LLM es una etiqueta producida por un modelo a "
        "partir de un contrato explícito y verificable, no inducida del comportamiento del "
        "sistema. El gold set del agente es de este tercer tipo: las rutas se derivaron del "
        "contrato de la API y de sus esquemas, y por eso se llama gold-LLM y no silver. Sigue "
        "siendo una referencia sin verificación humana, y así se reporta."
    )))
    b.append(("p", (
        "Las métricas se definen sobre esa referencia. El ruteo exacto exige coincidencia estricta "
        "entre la ruta esperada y el modo observado. La ruta jurídica alcanzada mide si una "
        "consulta en alcance termina en el agente o en un fallback con evidencia. Los saludos no "
        "rechazados miden si un saludo recibe respuesta conversacional. El rechazo correcto mide "
        "si un tema ajeno recibe el mensaje de alcance. El rechazo seguro admite también la "
        "aclaración. La aclaración correcta mide los ambiguos. Los adversarios contenidos miden si "
        "una inyección de instrucciones se queda sin efecto. El tool call mide si el agente usó al "
        "menos una herramienta. El grounding mide si todas las citas de la respuesta existen en "
        "la salida de las herramientas."
    )))

    b.append(("h2", "9.2 Resultados del gold set"))
    b.append(("p", (
        "La corrida final se hizo con qwen3.8:diario-8k-q3_k_s, sin rerank, con 14,39 GiB de VRAM "
        "libre al inicio y una duración de 15,6 minutos. El ruteo exacto fue de 90,0 %; la ruta "
        "jurídica se alcanzó en el 100,0 % de las consultas en alcance; los saludos no se "
        "rechazaron en el 100,0 %; el rechazo correcto de temas fuera de alcance fue del 100,0 %; "
        "los adversarios se contuvieron en el 100,0 %; el tool call ocurrió en el 100,0 %; y el "
        "grounding fue correcto en el 100,0 %. Se emitieron 10 advertencias y no hubo fallos de "
        "guardrail."
    )))
    b.append(tabla(
        "Exactitud por ruta esperada en el gold set de 100 prompts (v3).",
        ["Ruta esperada", "n", "Aciertos", "Exactitud"],
        [
            ["Adversario", "10", "10", "1,000"],
            ["Ambiguo", "5", "5", "1,000"],
            ["En alcance", "40", "30", "0,750"],
            ["Fuera de alcance", "25", "25", "1,000"],
            ["Saludo", "10", "10", "1,000"],
            ["Small talk", "10", "10", "1,000"],
        ],
        [6.0, 2.4, 2.8, 4.0],
        "evaluacion/agente_eval_v3.json (metricas_por_ruta) y agente_eval_v3.md.",
        ["l", "r", "r", "r"],
    ))
    b.append(("p", (
        "La matriz de confusión explica el 90,0 %. De las 40 consultas en alcance, 30 llegaron al "
        "modo agente, 9 se resolvieron con el plan determinista y 1 respondió sin evidencia. Las "
        "nueve del plan determinista son consultas cuya llamada al modelo falló o no produjo "
        "herramientas; el backend respondió con la evidencia disponible, que es el "
        "comportamiento deseado. La restante es T-500/90, una sentencia anterior a 1990 que no "
        "está en el corpus 1992-2021: el sistema dijo que no tenía evidencia en lugar de "
        "inventarla. Ninguno de los diez casos es una fuga de guardrail."
    )))
    b.append(("p", (
        "Los fallos individuales están documentados con su prompt y su causa. alc-02 pidió "
        "analizar la T-760/08 y el modelo devolvió argumentos de herramienta inválidos "
        "(unexpected end of JSON input en explorar_articulo); alc-07, alc-12, alc-17, alc-18, "
        "alc-20, alc-31 y alc-39 recibieron el mensaje de sin evidencia sin emitir herramienta; "
        "alc-11 y alc-40 también cayeron en el plan determinista. En los diez casos la respuesta "
        "final se apoyó en evidencia verificable o declaró su ausencia. La conclusión es que el "
        "sistema es conservador en la forma y correcto en el fondo."
    )))
    b.append(("p", (
        "La latencia es el costo visible. La ruta del agente tiene una mediana de 13.444 ms y un "
        "p95 de 60.799 ms, con un máximo de 105.291 ms. La mediana global aparece como 0 ms en el "
        "reporte porque 60 de los 100 prompts los resuelve el router determinista sin cargar el "
        "modelo; esa cifra no representa la experiencia conversacional y por eso el informe usa la "
        "latencia de la ruta del agente. El p95 alto se explica por la contención de GPU y por "
        "respuestas largas, no por el router."
    )))

    b.append(("h2", "9.3 Comparación de las tres versiones"))
    b.append(("p", (
        "La comparación v1, v2 y v3 usa el mismo gold set y el mismo modelo; solo cambia el "
        "router y el plan determinista. El salto grande ocurrió entre v1 y v2: el rechazo fuera de "
        "alcance pasó de 56,0 % a 100,0 %, los saludos de 90,0 % a 100,0 % y los tool calls de "
        "82,5 % a 100,0 %. La v3 no mejoró el ruteo exacto, que ya estaba en 90,0 %, pero "
        "estabilizó los casos límite y añadió el plan determinista como red de seguridad. El "
        "costo fue un aumento de advertencias (de 8 a 10) y del p95 (de 57,6 a 60,8 s), "
        "consistente con respuestas más largas y con la contención de la máquina."
    )))
    b.append(("p", (
        "La lección metodológica es que un guardrail no se mide con un caso feliz. La v1 fallaba "
        "en once de veinticinco consultas fuera de alcance y en un saludo, y ese 56,0 % no era "
        "visible sin un gold set por rutas. La evaluación por rutas, con prompts adversarios y "
        "ambiguos, es la que hizo observable el problema y la que permitió verificarlo después de "
        "cada cambio."
    )))

    b.append(("h2", "9.4 Traza de ejemplo"))
    b.append(("p", (
        "La traza siguiente corresponde a alc-01, una consulta en alcance resuelta por el agente. "
        "Se reproduce con recortes y sin cambios de fondo; el objetivo es mostrar qué información "
        "queda registrada y cómo se verifica el grounding."
    )))
    b.append(("cita", (
        "Prompt: ¿Qué artículos de la Constitución están vinculados a la sentencia T-293/97?\n"
        "Ruta: agente  |  tool_calls: 1  |  latencia: 20,9 s\n"
        "  -> vincular_cp(referencia=\"T-293/97\", k=6, rerank=false)  [ok, 5.569 ms]\n"
        "     resumen: id=2195 canon=T-293-1997 articulos=86,241,1,9,2,53\n"
        "Respuesta (recorte): Para la sentencia T-293/97 (id 2195), los artículos vinculados son:\n"
        "  citados en el texto: 11, 13, 25, 49, 86 y 241;\n"
        "  recuperados por similitud: 86, 241, 1, 9, 2 y 53.\n"
        "Grounding: 11 citas verificadas (1, 2, 9, 11, 13, 25, 49, 53, 86, 241 y T-293-1997),\n"
        "  0 citas no verificadas."
    )))
    b.append(("p", (
        "La traza muestra las tres capas en funcionamiento. El router envió la consulta al agente; "
        "el agente eligió vincular_cp con el canon y k=6; el verificador contrastó cada artículo "
        "de la respuesta contra la salida de la herramienta y contra el catálogo de la CP. La "
        "distinción entre artículos citados y artículos recuperados es deliberada: el primero es "
        "evidencia textual, el segundo es una sugerencia del modelo de recuperación. El usuario ve "
        "ambos y puede decidir."
    )))
    b.append(("p", (
        "El ejemplo también ilustra el límite. Que un artículo aparezca citado no significa que "
        "sea la ratio decidendi; el sistema no hace ese juicio. Lo que hace es reducir el espacio "
        "de búsqueda y dejar la evidencia a la vista. La evaluación del grounding, en la línea de "
        "las métricas automáticas para sistemas aumentados por recuperación (Es et al., 2024), "
        "mide existencia y consistencia de las citas, no corrección jurídica."
    )))

    b.append(("h2", "9.5 Límites del router y riesgo residual"))
    b.append(("p", (
        "El router es determinista y léxico. Su ventaja es la previsibilidad y su desventaja es la "
        "cobertura: un tema ajeno que no coincide con la lista de temas excluidos cae en "
        "aclaración en lugar de rechazo. El efecto práctico es que el sistema no responde el tema, "
        "pero tampoco da el mensaje de dominio; la tabla de confusión de la versión 3 muestra que "
        "ese caso no ocurrió en los 100 prompts, aunque el límite existe y se declara."
    )))
    b.append(("p", (
        "Se probaron dos sondas de jailbreak ancladas en términos jurídicos, fuera del gold set. "
        "Ambas pasaron el router por diseño, porque una consulta jurídica gana sobre el resto, y "
        "ambas se quedaron en la evidencia jurídica: una respondió sobre la T-293/97 y la otra "
        "sobre las sentencias citadas por la SU-480/97. El riesgo residual existe, pero depende "
        "del prompt del sistema y no del router. La conclusión es que el router no es un "
        "mecanismo de seguridad suficiente por sí solo; el grounding y el prompt del sistema son "
        "la barrera que contiene la alucinación."
    )))
    b.append(("p", (
        "El riesgo de alucinación se mitiga, no se elimina. La literatura sobre generación "
        "muestra que los modelos pueden producir contenido plausible y falso, y que la "
        "verificación posterior es una de las pocas defensas prácticas (Ji et al., 2023). En este "
        "sistema, la verificación es obligatoria: si una cita no existe en la evidencia, se "
        "elimina. Esa regla puede empobrecer una respuesta, pero nunca la deja pasar con una cita "
        "inventada."
    )))

    # ------------------------------------------------------------------
    # 10. Análisis, limitaciones y ética
    # ------------------------------------------------------------------
    b.append(("h1", "10. Análisis, limitaciones y consideraciones éticas"))
    b.append(("h2", "10.1 Limitaciones"))
    b.append(("p", (
        "Las limitaciones se agrupan en cinco frentes. El primero es la dependencia del LLM local. "
        "El agente necesita ollama y alrededor de 13,5 GB de VRAM; con la GPU ocupada la latencia "
        "crece hasta 60 s en p95 o la respuesta cae al plan determinista. Esa dependencia afecta "
        "la reproducibilidad en otra máquina y explica por qué el producto declara dos regímenes "
        "de latencia."
    )))
    b.append(("p", (
        "El segundo es la cobertura del corpus. El corpus principal termina en 2021 y no incluye "
        "2022 en adelante; el corpus C3 de 2022-2026 se construyó para probar robustez, no para "
        "reemplazar la base. Una sentencia de 2027 no se puede analizar. Además, 48 documentos de "
        "C3 quedaron como stubs no publicados y 4 tenían texto demasiado corto, de modo que la "
        "extensión tampoco es exhaustiva."
    )))
    b.append(("p", (
        "El tercero es la ausencia de validación por un experto legal. Las etiquetas humanas de B3 "
        "y O1 las produjo un integrante del equipo con una guía, no un abogado con experiencia en "
        "constitucional. La guía y el gold set están documentados, pero la validez externa de la "
        "anotación no está establecida. El acuerdo ítem a ítem con los proxies LLM fue moderado "
        "(kappa 0,3448 en B3 y 0,153 en O1), lo que refuerza la cautela."
    )))
    b.append(("p", (
        "El cuarto es M4. El macro-F1 de 0,2870 y la exactitud humano-silver de 23,5 % no permiten "
        "usar el área como conclusión. El sistema la muestra como indicio y la excluye de las "
        "decisiones de ordenamiento y filtrado. La etiqueta tiene un techo estructural: mientras "
        "se derive del bloque de las citas, confundirá tutelas de salud con áreas genéricas, como "
        "muestran los casos T-853/02, T-231/20 y T-235/20."
    )))
    b.append(("p", (
        "El quinto es la escala de la evidencia. M0 se validó con 52 casos, el benchmark de "
        "recuperación con 90 y el gold humano con 50 casos y 820 juicios. Son muestras "
        "suficientes para detectar efectos grandes, no para estimar con precisión efectos "
        "pequeños. Los intervalos de confianza reportados reflejan esa incertidumbre y deben "
        "leerse junto con la media, no en su lugar."
    )))

    b.append(("h2", "10.2 Consideraciones éticas"))
    b.append(("p", (
        "El corpus es público y su licencia es CC-BY-4.0. Las sentencias son documentos oficiales "
        "y no contienen datos personales sensibles en el sentido estricto, aunque incluyen nombres "
        "de las partes, de los ponentes y de terceros. El sistema no reidentifica personas, no "
        "cruza el corpus con otras fuentes y no genera perfiles. Aun así, la agregación de "
        "información sobre casos individuales puede tener efectos: un litigante podría usar el "
        "sistema para anticipar la posición de un despacho o de un ponente. Esa posibilidad no se "
        "explota en el producto y se declara como riesgo."
    )))
    b.append(("p", (
        "El riesgo principal es la confianza excesiva. Un asistente que responde con seguridad "
        "sobre materia legal puede inducir a error si el usuario olvida que la sugerencia es "
        "probabilística y que la evidencia debe leerse. Las mitigaciones son de diseño: el área se "
        "rotula exploratoria, las citas se verifican, las respuestas sin evidencia se declaran y "
        "la interfaz muestra la procedencia de cada cifra. La literatura sobre modelos grandes "
        "advierte que el volumen y la fluidez no garantizan veracidad (Bender et al., 2021); este "
        "producto intenta que esa advertencia sea visible en la pantalla y no solo en el informe."
    )))
    b.append(("p", (
        "La documentación del corpus y de los artefactos sigue el espíritu de las hojas de datos "
        "(Gebru et al., 2021): se declara la fuente, la licencia, el proceso de curaduría, las "
        "exclusiones y las limitaciones. Los modelos y los resultados quedan versionados con "
        "md5 y con scripts reproducibles. Esa trazabilidad es una condición para que el trabajo "
        "pueda auditarse y para que un tercero pueda refutar las conclusiones."
    )))
    b.append(("p", (
        "El sistema no constituye asesoría legal. Su salida es un insumo de investigación y debe "
        "acompañarse de la lectura del expediente y del criterio de un profesional. Esa "
        "advertencia aparece en la interfaz y se repite aquí porque el límite es ético, no solo "
        "técnico."
    )))

    # ------------------------------------------------------------------
    # 11. Conclusiones y trabajo futuro
    # ------------------------------------------------------------------
    b.append(("h1", "11. Conclusiones y trabajo futuro"))
    b.append(("h2", "11.1 Conclusiones"))
    b.append(("p", (
        "La primera conclusión es que los modelos de dominio funcionan cuando se entrenan con las "
        "citas del propio corpus. La ruta B2 más B1 supera a la mejor ruta previa con intervalos "
        "de confianza pareados, se mantiene en los tres periodos del benchmark y no se degrada en "
        "el corpus 2022-2026. El nDCG@10 de 0,6907 en silver y de 0,7852 en gold humano, con F1@5 "
        "de 0,478 y MRR de 0,950, es el mejor resultado no circular del proyecto."
    )))
    b.append(("p", (
        "La segunda es que el protocolo importa tanto como el modelo. El gate de intervalos de "
        "confianza y consistencia A/B evitó adoptar cinco variantes del Sprint A, una segunda "
        "época de B2 y tres fusiones de LoRA que parecían mejores en smokes o en deltas puntuales. "
        "La disciplina de no adoptar sin evidencia es, en sí misma, un resultado del trabajo."
    )))
    b.append(("p", (
        "La tercera es que la etiqueta de área temática es el cuello de botella de M4, no el "
        "clasificador. La auditoría de ruido y la validación humana muestran que el silver no es "
        "un gold fiable y que la tarea es ambigua incluso entre anotadores. Mejorar el modelo "
        "sobre esa etiqueta sería optimizar el ruido."
    )))
    b.append(("p", (
        "La cuarta es que un asistente jurídico puede ser conversacional sin perder el dominio. El "
        "router determinista, las herramientas y la verificación de grounding logran 100,0 % en "
        "rechazo, saludos, adversarios, tool calls y citas verificadas, con un 90,0 % de ruteo "
        "estricto que se explica por fallbacks conservadores, no por fugas."
    )))
    b.append(("p", (
        "La quinta es que el producto es usable y reproducible. La API y la interfaz están "
        "separadas, las pruebas pasan, los artefactos tienen md5 y los scripts permiten regenerar "
        "los resultados. El sistema corre en CPU para la ruta por defecto y en GPU para la ruta "
        "con reranker y el chat."
    )))

    b.append(("h2", "11.2 Trabajo futuro"))
    b.append(("p", (
        "El trabajo inmediato es cerrar el gold estándar C1: 150 casos y 2.740 candidatos ya "
        "preparados, fuera del entrenamiento de B1 y B2. Su anotación permitiría llevar la "
        "validación humana de 50 a 150 casos y reducir la varianza de las estimaciones. En "
        "paralelo, conviene repetir el spot-check humano de B3 con un segundo anotador "
        "independiente y, si es posible, con un experto legal."
    )))
    b.append(("p", (
        "En recuperación, la línea natural es la segunda época de B2 con un modo de continuación "
        "bien controlado y la exploración de un reranker que combine B1 con señales de cita. En "
        "clasificación, M4 solo tiene sentido si se redefine la tarea o se consigue una etiqueta "
        "humana del tema, no del bloque de citas. Las variantes de agregación de chunks y de "
        "features por sección quedan como hipótesis, no como prioridad."
    )))
    b.append(("p", (
        "En producto, el despliegue externo es el paso pendiente. La separación entre API y "
        "frontend ya lo permite, pero faltan autenticación, cuotas, empaquetado de los modelos y "
        "una política de privacidad para consultas de usuarios. También falta unificar el corpus "
        "C3 con el principal y reintentar los 48 stubs, para que la cobertura llegue hasta 2026 "
        "sin dos bases separadas."
    )))
    b.append(("p", (
        "Finalmente, hay dos frentes de calidad. El primero es la calibración de las "
        "probabilidades del producto, hoy no calibradas, o su rotulación explícita como no "
        "calibradas. El segundo es el barrido completo de md5 de los artefactos congelados, "
        "observación que dejó el QA de v2.2 y que sigue pendiente. Ninguno de los dos cambia las "
        "conclusiones, pero ambos mejoran la reproducibilidad."
    )))

    return b


REFERENCIAS = [
    "Aletras, N., Tsarapatsanis, D., Preoţiuc-Pietro, D., & Lampos, V. (2016). Predicting judicial "
    "decisions of the European Court of Human Rights: A natural language processing perspective. "
    "PeerJ Computer Science, 2, e93. https://doi.org/10.7717/peerj-cs.93",

    "Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. "
    "Computational Linguistics, 34(4), 555-596. https://doi.org/10.1162/coli.07-034-R2",

    "Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of "
    "stochastic parrots: Can language models be too big? En Proceedings of the 2021 ACM Conference "
    "on Fairness, Accountability, and Transparency (pp. 610-623). ACM. "
    "https://doi.org/10.1145/3442188.3445922",

    "Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). Spanish "
    "pre-trained BERT model and evaluation data. En PML4DC at ICLR 2020.",

    "Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). "
    "LEGAL-BERT: The muppets straight out of law school. En Findings of the Association for "
    "Computational Linguistics: EMNLP 2020 (pp. 2898-2904). "
    "https://doi.org/10.18653/v1/2020.findings-emnlp.261",

    "Chalkidis, I., Garneau, N., Goanta, C., Katz, D. M., & Søgaard, A. (2022). Processing long "
    "legal documents with pre-trained transformers: Modding LegalBERT and Longformer. "
    "arXiv:2211.00974. https://arxiv.org/abs/2211.00974",

    "Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms "
    "Condorcet and individual rank learning methods. En Proceedings of the 32nd International ACM "
    "SIGIR Conference (pp. 758-759). https://doi.org/10.1145/1571941.1572114",

    "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.",

    "Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of "
    "retrieval augmented generation. En Proceedings of the 18th Conference of the European Chapter "
    "of the ACL: System Demonstrations (pp. 150-158). https://doi.org/10.18653/v1/2024.eacl-demo.16",

    "Fowler, J. H., Johnson, T. R., Spriggs, J. F., Jeon, S., & Wahlbeck, P. J. (2007). Network "
    "analysis and the law: Measuring the legal importance of precedents at the U.S. Supreme Court. "
    "Political Analysis, 15(3), 324-346. https://doi.org/10.1093/pan/mpm011",

    "Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & "
    "Crawford, K. (2021). Datasheets for datasets. Communications of the ACM, 64(12), 86-92. "
    "https://doi.org/10.1145/3458723",

    "HaCohen-Kerner, Y., Miller, D., & Yigal, Y. (2020). The influence of preprocessing on text "
    "classification using a bag-of-words representation. PLOS ONE, 15(5), e0232525. "
    "https://doi.org/10.1371/journal.pone.0232525",

    "Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal "
    "statistical laws of semantic change. En Proceedings of ACL 2016 (pp. 1489-1501). "
    "https://doi.org/10.18653/v1/P16-1141",

    "Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM "
    "Transactions on Information Systems, 20(4), 422-446. https://doi.org/10.1145/582415.582418",

    "Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & "
    "Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing "
    "Surveys, 55(12), 1-38. https://doi.org/10.1145/3571730",

    "Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical "
    "data. Biometrics, 33(1), 159-174. https://doi.org/10.2307/2529310",

    "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, "
    "M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation "
    "for knowledge-intensive NLP tasks. En Advances in Neural Information Processing Systems 33 "
    "(pp. 9459-9474).",

    "Manuel. (2022). Sentencias Corte Constitucional de Colombia 1992-2021 [Conjunto de datos]. "
    "Hugging Face. https://huggingface.co/datasets/Manuel/sentencias-corte-cons-colombia-1992-2021",

    "McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of "
    "sophisticated approaches to lexical diversity assessment. Behavior Research Methods, 42(2), "
    "381-392. https://doi.org/10.3758/BRM.42.2.381",

    "Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive text embedding "
    "benchmark. En Proceedings of EACL 2023 (pp. 2014-2037). "
    "https://doi.org/10.18653/v1/2023.eacl-main.148",

    "Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. arXiv:1901.04085. "
    "https://arxiv.org/abs/1901.04085",

    "Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident learning: Estimating "
    "uncertainty in dataset labels. Journal of Machine Learning Research, 22(137), 1-41.",

    "Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. "
    "Foundations and Trends in Information Retrieval, 3(4), 333-389. "
    "https://doi.org/10.1561/1500000019",

    "Soh, J., Lim, H. K., & Chai, I. E. (2019). Legal area classification: A comparative study of "
    "text classifiers on Singapore Supreme Court judgments. En Proceedings of the Natural Legal "
    "Language Processing Workshop 2019 (pp. 67-77). https://doi.org/10.18653/v1/W19-2208",

    "Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A "
    "heterogeneous benchmark for zero-shot evaluation of information retrieval models. "
    "arXiv:2104.08663. https://arxiv.org/abs/2104.08663",

    "Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). "
    "Text embeddings by weakly-supervised contrastive pre-training. arXiv:2212.03533. "
    "https://arxiv.org/abs/2212.03533",

    "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: "
    "Synergizing reasoning and acting in language models. En International Conference on Learning "
    "Representations (ICLR 2023).",
]


def bloques_12_13() -> list[tuple]:
    """Devuelve las referencias y los anexos."""
    b: list[tuple] = []

    b.append(("h1", "12. Referencias"))
    b.append(("p", (
        "Las referencias siguen el formato APA 7. Se incluyen solo las obras citadas en el texto; "
        "los conjuntos de datos y los informes internos del proyecto se referencian en el anexo A."
    )))
    for referencia in REFERENCIAS:
        b.append(("referencia", referencia))

    b.append(("h1", "13. Anexos"))
    b.append(("h2", "13.1 Anexo A. Inventario de artefactos"))
    b.append(("p", (
        "Todos los artefactos viven en la carpeta ENTREGA FINAL, salvo los resultados pesados, que "
        "se enlazan desde SEGUNDA ENTREGA para no duplicar 47 GB. La Tabla 14 resume los "
        "principales."
    )))
    b.append(tabla(
        "Inventario de artefactos de la entrega.",
        ["Artefacto", "Ruta", "Contenido"],
        [
            ["Métricas consolidadas", "SEGUNDA ENTREGA/resultados/metricas.json", "Fuente de las tablas de resultados"],
            ["Gold humano B3", "resultados/gold_mini/", "50 casos, 820 grados, evaluación y acuerdo"],
            ["Validación O1", "resultados/validacion_m4/", "200 documentos anotados a ciegas"],
            ["Corpus C3", "data/corpus_2022_2026/", "2.330 documentos de 2022-2026"],
            ["Modelos B1 y B2", "modelos/b1_reranker, modelos/b2_retriever", "Reranker y retriever de dominio"],
            ["Adaptador LoRA", "modelos/c2_lora/adapter_v1", "Adaptador de ALIA-7B (no adoptado)"],
            ["Backend", "app/backend/", "FastAPI, agente, herramientas y pruebas"],
            ["Frontend", "app/frontend/", "Streamlit, pantallas y pruebas"],
            ["Evaluación del agente", "evaluacion/agente_eval_v3.json", "100 prompts, métricas y trazas"],
            ["Decisiones", "SEGUNDA ENTREGA/plan/decisiones/", "ADR-16 a ADR-23"],
            ["Figuras", "figuras/", "Figuras del EDA y de los experimentos"],
            ["Generador del informe", "informe/generar_informe.py", "Este documento, reproducible"],
        ],
        [3.6, 6.4, 6.0],
        "Estructura de la carpeta ENTREGA FINAL y README.md; los md5 de los artefactos congelados "
        "están en metricas.json y en los informes de QA.",
        ["l", "l", "l"],
        tamano=8.4,
    ))

    b.append(("h2", "13.2 Anexo B. Cómo reproducir"))
    b.append(("p", (
        "La reproducción completa exige el CSV original, el entorno /home/carlos/venv, Java 17 para "
        "Spark y, para los modelos de dominio, la GPU AMD con ROCm. Los pasos siguientes resumen "
        "el orden; los detalles están en SEGUNDA ENTREGA/documento_e2/README_ENTREGA.md."
    )))
    b.append(("ol", [
        "Preparar el entorno: exportar JAVA_HOME a Java 17, activar el venv e instalar las "
        "dependencias de app/backend/requirements.txt.",
        "Construir el corpus: ejecutar scripts/00_csv_a_parquet.py y verificar las 23.742 filas y "
        "los 8 registros excluidos.",
        "Descargar y parsear la Constitución: scripts/01_descarga_constitucion.py y "
        "scripts/02_parseo_constitucion.py, que producen 497 filas y 19 bloques.",
        "Extraer entidades, etiquetar áreas y lematizar: scripts/03, 04 y 05. El etiquetado debe "
        "cerrar en 99,63 % de cobertura.",
        "Entrenar y evaluar M4: scripts/06_baseline_clasico.py y scripts/07_transformer.py. La "
        "referencia es macro-F1 0,2870 en test.",
        "Construir la red de citas y la diacronía: scripts/08_red_citas.py y scripts/09_diacronia.py.",
        "Evaluar el vínculo CP: scripts/12_evaluar_vinculo_cp_v2.py con las variantes --b1, --b2 y "
        "--rerank; el benchmark de referencia es v13.",
        "Reproducir el gold: scripts/16_anotador_gold.py --evaluar --forzar para B3 y "
        "scripts/21_pack_validacion_m4.py --calcular-acuerdo para O1.",
        "Levantar el producto: uvicorn main:app en el puerto 8123 y streamlit run streamlit_app.py "
        "en el 8502 con API_URL apuntando al backend.",
        "Regenerar este informe: /home/carlos/venv/bin/python informe/generar_informe.py, que "
        "produce el DOCX y el PDF con la tabla de contenido actualizada.",
    ]))
    b.append(("p", (
        "La verificación final del documento comprueba tres cosas: que el PDF abra, que el número "
        "de páginas esté en el rango solicitado y que no queden marcadores de texto incompleto. "
        "El generador informa esos tres valores al terminar."
    )))

    return b


# ---------------------------------------------------------------------------
# Construcción, conversión y verificación
# ---------------------------------------------------------------------------
def contenido() -> list[tuple]:
    """Ensambla todos los bloques del informe en orden."""
    return bloques_1_4() + bloques_5_7() + bloques_8_11() + bloques_12_13()


def entradas_toc(bloques: list[tuple]) -> list[tuple[int, str]]:
    """Extrae los títulos de nivel 1 y 2 que entran en la tabla de contenido."""
    return [(1 if tipo == "h1" else 2, texto) for tipo, texto in bloques if tipo in ("h1", "h2")]


def construir(ruta: Path, paginas: dict[str, int] | None) -> tuple[list[tuple[int, str]], int, int]:
    """Genera el DOCX completo y devuelve las entradas del TOC y los conteos."""
    bloques = contenido()
    entradas = entradas_toc(bloques)

    doc = Document()
    configurar_estilos(doc)
    configurar_seccion(doc)
    construir_portada(doc)
    construir_toc(doc, entradas, paginas)
    tablas, figuras = renderizar(doc, bloques)

    propiedades = doc.core_properties
    propiedades.title = TITULO
    propiedades.author = AUTOR
    propiedades.subject = "Informe final consolidado de las entregas 1, 2 y 3"
    propiedades.keywords = "PLN, jurisprudencia, recuperación, grounding, CRISP-DM"
    propiedades.comments = "Generado por informe/generar_informe.py"
    doc.save(str(ruta))
    return entradas, tablas, figuras


def convertir(ruta_docx: Path) -> Path:
    """Convierte un DOCX a PDF con LibreOffice en modo headless."""
    perfil = Path("/tmp/opencode/lo_juris")
    perfil.mkdir(parents=True, exist_ok=True)
    comando = [
        "soffice",
        f"-env:UserInstallation=file://{perfil}",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(ruta_docx.parent),
        str(ruta_docx),
    ]
    resultado = subprocess.run(
        comando, capture_output=True, text=True, timeout=300, check=False
    )
    pdf = ruta_docx.with_suffix(".pdf")
    if resultado.returncode != 0 or not pdf.exists():
        raise RuntimeError(
            "LibreOffice no pudo convertir el documento.\n"
            f"stdout: {resultado.stdout}\nstderr: {resultado.stderr}"
        )
    return pdf


def compactar(texto: str) -> str:
    """Normaliza texto para comparar títulos extraídos del PDF."""
    return re.sub(r"[^0-9a-záéíóúüñ]", "", texto.lower())


def paginas_de_titulos(pdf: Path, titulos: list[str], inicio: int = 2) -> dict[str, int]:
    """Localiza la primera página de cada título a partir de la página indicada."""
    lector = PdfReader(str(pdf))
    textos = [compactar(pagina.extract_text() or "") for pagina in lector.pages]
    paginas: dict[str, int] = {}
    for titulo in titulos:
        clave = compactar(titulo)
        for indice in range(inicio, len(textos)):
            if clave and clave in textos[indice]:
                paginas[titulo] = indice + 1
                break
    return paginas


def contar_palabras(bloques: list[tuple]) -> int:
    """Cuenta las palabras de los párrafos y listas del informe."""
    total = 0
    for bloque in bloques:
        if bloque[0] in ("p", "cita"):
            total += len(str(bloque[1]).split())
        elif bloque[0] in ("ul", "ol"):
            total += sum(len(str(elemento).split()) for elemento in bloque[1])
    return total


def verificar(pdf: Path, esperadas: int) -> tuple[int, list[str]]:
    """Comprueba que el PDF abra, cuente páginas y no tenga marcadores pendientes."""
    lector = PdfReader(str(pdf))
    paginas = len(lector.pages)
    texto = "\n".join((pagina.extract_text() or "") for pagina in lector.pages)
    marcadores = [
        marca for marca in ("TODO", "XXX") if marca in texto
    ] + [
        marca
        for marca in ("(completar)", "RUTA_IMAGEN", "Lorem ipsum")
        if marca.lower() in texto.lower()
    ]
    return paginas, marcadores


def main() -> int:
    """Ejecuta las dos pasadas, convierte a PDF y reporta el resultado."""
    print("Construyendo el borrador para calcular la tabla de contenido...")
    entradas, tablas, figuras = construir(BORRADOR, None)
    convertir(BORRADOR)
    pdf_borrador = BORRADOR.with_suffix(".pdf")
    titulos = [texto for _, texto in entradas]
    paginas = paginas_de_titulos(pdf_borrador, titulos)

    if len(paginas) != len(titulos):
        faltantes = [t for t in titulos if t not in paginas]
        raise RuntimeError(f"No se localizaron {len(faltantes)} títulos en el PDF: {faltantes[:5]}")

    lector = PdfReader(str(pdf_borrador))
    if len(lector.pages) > 1:
        toc = compactar(lector.pages[1].extract_text() or "")
        if compactar(titulos[-1]) not in toc:
            raise RuntimeError(
                "La tabla de contenido no cabe en una página; ajuste el tamaño o reduzca entradas."
            )

    print("Construyendo el documento final con las páginas calculadas...")
    for intento in range(3):
        construir(SALIDA_DOCX, paginas)
        convertir(SALIDA_DOCX)
        nuevas = paginas_de_titulos(SALIDA_PDF, titulos)
        if nuevas == paginas:
            break
        print(f"Ajuste de paginación en el intento {intento + 1}.")
        paginas = nuevas

    total_paginas, marcadores = verificar(SALIDA_PDF, len(entradas))
    palabras = contar_palabras(contenido())

    for ruta in (BORRADOR, pdf_borrador):
        if ruta.exists():
            ruta.unlink()

    print()
    print(f"DOCX: {SALIDA_DOCX}")
    print(f"PDF:  {SALIDA_PDF}")
    print(f"Páginas: {total_paginas}")
    print(f"Tablas: {tablas} | Figuras: {figuras} | Entradas TOC: {len(entradas)}")
    print(f"Palabras de prosa: {palabras}")
    print(f"Marcadores pendientes: {marcadores if marcadores else 'ninguno'}")
    if not 28 <= total_paginas <= 36:
        print(f"ADVERTENCIA: el documento tiene {total_paginas} páginas, fuera del rango 28-36.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
