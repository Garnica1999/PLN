"""pdf_ficha.py
Fecha: 2026-10-31
Descripcion: Genera en el frontend la ficha PDF de una sentencia (P4). Toma
    el resultado de /v1/link y lo compone en una pagina A4 con reportlab,
    siguiendo la paleta y la tipografia de DESIGN.md: titulos serif, cuerpo
    sans, acento azul institucional y sin emojis. Las secciones sin datos se
    omiten; nunca se rellenan valores.
Estado del arte aplicado: PDF en memoria con BytesIO (sin archivos
    temporales), Platypus para el flujo, tablas compactas con pie de
    procedencia y nombres de archivo saneados.
"""

from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

AZUL = colors.HexColor("#2F75B5")
TINTA = colors.HexColor("#162536")
GRIS = colors.HexColor("#425466")
BORDE = colors.HexColor("#DCE3EB")
SUPERFICIE = colors.HexColor("#F5F7FA")

AVISO_PROCEDENCIA = (
    "Modelo v2.2 evaluado en 1992-2021; el producto integra el corpus hasta 2026. "
    "Las etiquetas de area son automaticas (exploratorias, ADR-19) y no "
    "constituyen una "
    "clasificacion verificada por humanos."
)
MAX_FILAS_ARTICULOS = 4
MAX_FILAS_CITAS = 6
MAX_CARACTERES_RESUMEN = 620

ESTILO_TITULO = ParagraphStyle(
    "titulo",
    fontName="Times-Bold",
    fontSize=16,
    leading=19,
    textColor=TINTA,
    spaceAfter=2,
)
ESTILO_SUBTITULO = ParagraphStyle(
    "subtitulo",
    fontName="Helvetica",
    fontSize=8.5,
    leading=11,
    textColor=GRIS,
)
ESTILO_SECCION = ParagraphStyle(
    "seccion",
    fontName="Times-Bold",
    fontSize=10.5,
    leading=13,
    textColor=TINTA,
    spaceBefore=8,
    spaceAfter=3,
)
ESTILO_CUERPO = ParagraphStyle(
    "cuerpo",
    fontName="Helvetica",
    fontSize=8.2,
    leading=10.6,
    textColor=TINTA,
)
ESTILO_CELDA = ParagraphStyle(
    "celda",
    fontName="Helvetica",
    fontSize=7.4,
    leading=9.2,
    textColor=TINTA,
)
ESTILO_DATO = ParagraphStyle(
    "dato",
    fontName="Courier",
    fontSize=7.4,
    leading=9.2,
    textColor=TINTA,
)
ESTILO_NOTA = ParagraphStyle(
    "nota",
    fontName="Helvetica",
    fontSize=7,
    leading=9.2,
    textColor=GRIS,
)


def _escapar(valor: Any) -> str:
    """Escapa un valor para un Paragraph de reportlab.

    Args:
        valor: Texto o escalar a mostrar.

    Returns:
        Cadena escapada y recortada a 300 caracteres.
    """
    texto = "" if valor is None else str(valor)
    texto = re.sub(r"\s+", " ", texto).strip()[:300]
    return (
        texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _recortar(valor: Any, limite: int) -> str:
    """Recorta un texto y agrega puntos suspensivos si excede el limite.

    Args:
        valor: Valor a recortar.
        limite: Maximo de caracteres.

    Returns:
        Texto recortado y escapado.
    """
    texto = "" if valor is None else re.sub(r"\s+", " ", str(valor)).strip()
    if len(texto) > limite:
        texto = texto[: limite - 1].rstrip() + "…"
    return _escapar(texto)


def _tabla(
    cabeceras: list[str],
    filas: list[list[str]],
    anchos: list[float],
    numericas: set[int],
) -> Table:
    """Construye una tabla compacta con la paleta del informe.

    Args:
        cabeceras: Nombres de columna.
        filas: Celdas ya escapadas.
        anchos: Ancho de cada columna en milimetros.
        numericas: Indices de columnas con tipografia monoespaciada.

    Returns:
        Tabla lista para el flujo de Platypus.
    """
    datos = [
        [Paragraph(_escapar(cabecera), ESTILO_CELDA) for cabecera in cabeceras]
    ]
    for fila in filas:
        datos.append(
            [
                Paragraph(
                    valor, ESTILO_DATO if indice in numericas else ESTILO_CELDA
                )
                for indice, valor in enumerate(fila)
            ]
        )
    tabla = Table(datos, colWidths=[ancho * mm for ancho in anchos], repeatRows=1)
    tabla.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SUPERFICIE),
                ("TEXTCOLOR", (0, 0), (-1, -1), TINTA),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return tabla


def _seccion(titulo: str) -> list:
    """Devuelve el flujo de una seccion con su titulo.

    Args:
        titulo: Titulo de la seccion.

    Returns:
        Lista de flowables.
    """
    return [Paragraph(_escapar(titulo), ESTILO_SECCION)]


def _bloque_area(link: dict[str, Any]) -> list:
    """Compone el area tematica top-3 si esta disponible.

    Args:
        link: Respuesta de /v1/link.

    Returns:
        Lista de flowables (vacia si no hay area).
    """
    area = link.get("area") or {}
    top3 = area.get("top3") if isinstance(area, dict) else None
    if not isinstance(top3, list) or not top3:
        return []
    etiquetas = []
    for item in top3[:3]:
        if not isinstance(item, dict):
            continue
        nombre = item.get("area") or item.get("bloque") or "sin etiqueta"
        prob = item.get("prob")
        if isinstance(prob, (int, float)) and not isinstance(prob, bool):
            etiquetas.append(f"{nombre} ({prob:.3f})")
        else:
            etiquetas.append(str(nombre))
    if not etiquetas:
        return []
    return [
        *_seccion("Area tematica (exploratoria, ADR-19)"),
        Paragraph(_escapar("; ".join(etiquetas)), ESTILO_CUERPO),
    ]


def _bloque_articulos(link: dict[str, Any], clave: str, titulo: str) -> list:
    """Compone la tabla de articulos extraidos o recuperados.

    Args:
        link: Respuesta de /v1/link.
        clave: articulos_extraidos o articulos_recuperados.
        titulo: Titulo de la seccion.

    Returns:
        Lista de flowables (vacia si no hay filas).
    """
    articulos = [
        item
        for item in (link.get(clave) or [])
        if isinstance(item, dict) and item.get("numero")
    ]
    if not articulos:
        return []
    if clave == "articulos_recuperados":
        cabeceras = ["Articulo", "Score", "Bloque", "Texto"]
        anchos = [16, 16, 28, 114]
        numericas = {0, 1}
        filas = [
            [
                _escapar(item.get("numero")),
                f"{item.get('score'):.4f}"
                if isinstance(item.get("score"), (int, float))
                and not isinstance(item.get("score"), bool)
                else "No disponible",
                _escapar(item.get("bloque") or "Sin bloque"),
                _recortar(item.get("texto_150chars"), 150),
            ]
            for item in articulos[:MAX_FILAS_ARTICULOS]
        ]
    else:
        cabeceras = ["Articulo", "Bloque", "Texto"]
        anchos = [16, 28, 130]
        numericas = {0}
        filas = [
            [
                _escapar(item.get("numero")),
                _escapar(item.get("bloque") or "Sin bloque"),
                _recortar(item.get("texto_150chars"), 150),
            ]
            for item in articulos[:MAX_FILAS_ARTICULOS]
        ]
    flujo = [*_seccion(titulo), _tabla(cabeceras, filas, anchos, numericas)]
    if len(articulos) > MAX_FILAS_ARTICULOS:
        flujo.append(
            Paragraph(
                _escapar(
                    f"Se listan {MAX_FILAS_ARTICULOS} de {len(articulos)}; "
                    "el resto queda en el JSON del vinculo."
                ),
                ESTILO_NOTA,
            )
        )
    return flujo


def _bloque_citas(link: dict[str, Any]) -> list:
    """Compone la tabla de sentencias citadas resueltas.

    Args:
        link: Respuesta de /v1/link.

    Returns:
        Lista de flowables (vacia si no hay citas resueltas).
    """
    citas = [
        cita
        for cita in (link.get("sentencias_citadas") or [])
        if isinstance(cita, dict) and cita.get("resuelta")
    ]
    if not citas:
        return []
    filas = [
        [
            _escapar(cita.get("canon")),
            _escapar(cita.get("id") or "No disponible"),
            _escapar(cita.get("ano") or "No disponible"),
            f"{cita.get('pagerank'):.6f}"
            if isinstance(cita.get("pagerank"), (int, float))
            and not isinstance(cita.get("pagerank"), bool)
            else "No disponible",
        ]
        for cita in citas[:MAX_FILAS_CITAS]
    ]
    flujo = [
        *_seccion("Sentencias citadas resueltas"),
        _tabla(
            ["Canon", "Id", "Ano", "PageRank"],
            filas,
            [50, 22, 18, 84],
            {1, 2, 3},
        ),
    ]
    if len(citas) > MAX_FILAS_CITAS:
        flujo.append(
            Paragraph(
                _escapar(f"Se listan {MAX_FILAS_CITAS} de {len(citas)}."),
                ESTILO_NOTA,
            )
        )
    return flujo


def construir_ficha(
    link: dict[str, Any],
    *,
    generado: datetime | None = None,
) -> bytes:
    """Genera la ficha PDF de una sentencia analizada.

    Args:
        link: Respuesta de /v1/link con canon, articulos, citas y resumen.
        generado: Fecha de generacion; por defecto, la actual.

    Returns:
        Bytes del PDF listo para descargar.

    Raises:
        ValueError: Si no hay ningun dato de la sentencia que mostrar.
    """
    if not isinstance(link, dict) or not link:
        raise ValueError("No hay datos de la sentencia para la ficha PDF.")
    canon = link.get("canon_detectado") or "Sin canon detectado"
    id_sentencia = link.get("id") or "Sin id"
    fecha = (generado or datetime.now()).strftime("%Y-%m-%d %H:%M")

    flujo: list = [
        Paragraph("Ficha de sentencia", ESTILO_TITULO),
        Paragraph(
            _escapar(
                f"Juris-PLN. {canon}. Id del corpus: {id_sentencia}. "
                f"Generada el {fecha}."
            ),
            ESTILO_SUBTITULO,
        ),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=0.8, color=AZUL, spaceAfter=4),
    ]
    flujo.extend(_bloque_area(link))
    flujo.extend(
        _bloque_articulos(
            link, "articulos_extraidos", "Articulos extraidos del texto"
        )
    )
    flujo.extend(
        _bloque_articulos(
            link, "articulos_recuperados", "Articulos recuperados (ruta v2.2)"
        )
    )
    flujo.extend(_bloque_citas(link))
    resumen = str(link.get("resumen") or "").strip()
    if resumen:
        flujo.extend(_seccion("Resumen"))
        flujo.append(
            Paragraph(_recortar(resumen, MAX_CARACTERES_RESUMEN), ESTILO_CUERPO)
        )
    flujo.extend([Spacer(1, 8), HRFlowable(width="100%", thickness=0.4, color=BORDE)])
    flujo.append(Paragraph(_escapar(AVISO_PROCEDENCIA), ESTILO_NOTA))

    buffer = BytesIO()
    documento = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
        title=f"Ficha {canon}",
        author="Juris-PLN",
    )
    documento.build(flujo)
    return buffer.getvalue()


def nombre_ficha(link: dict[str, Any]) -> str:
    """Arma el nombre del archivo de la ficha.

    Args:
        link: Respuesta de /v1/link.

    Returns:
        Nombre ficha_<canon>_<id>.pdf saneado para el sistema de archivos.
    """
    canon = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(link.get("canon_detectado") or "sentencia")
    )
    id_sentencia = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(link.get("id") or "sin_id")
    )
    return f"ficha_{canon}_{id_sentencia}.pdf"
