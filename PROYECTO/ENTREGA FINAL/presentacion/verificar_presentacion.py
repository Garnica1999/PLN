"""verificar_presentacion.py

Fecha: 2026-10-31
Descripcion: verifica el deck de la Entrega 3 reabriendo el PPTX con
python-pptx y el PDF con pypdf. Comprueba numero de diapositivas y paginas,
ausencia de cajas de texto vacias, formas dentro de la pagina y desbordes
estimados del texto con las metricas reales de las fuentes Liberation, que
son metricamente equivalentes a Times New Roman y Arial.

Uso:
    /home/carlos/venv/bin/python presentacion/verificar_presentacion.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

BASE_DIR = Path(__file__).resolve().parent
PPTX = BASE_DIR / "Entrega_3_Presentacion.pptx"
PDF = BASE_DIR / "Entrega_3_Presentacion.pdf"

FUENTES = {
    "Times New Roman": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "Arial": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
}
TOLERANCIA_PT = 3.0


def alto_requerido(tf, ancho_pt: float) -> float:
    """Estima el alto necesario para renderizar un text frame.

    Args:
        tf: Text frame de python-pptx.
        ancho_pt: Ancho util en puntos.

    Returns:
        Alto estimado en puntos.
    """
    alto = 0.0
    for p in tf.paragraphs:
        size = 16.0
        for r in p.runs:
            if r.font.size is not None:
                size = r.font.size.pt
                break
        fuente = "Arial"
        for r in p.runs:
            if r.font.name:
                fuente = r.font.name
                break
        ruta = FUENTES.get(fuente, FUENTES["Arial"])
        font = ImageFont.truetype(ruta, round(size * 4))
        escala = size / (size * 4)

        texto = "".join(r.text for r in p.runs)
        lineas = 0
        for bloque in texto.split("\n") or [""]:
            palabras = bloque.split()
            if not palabras:
                lineas += 1
                continue
            actual = ""
            for palabra in palabras:
                prueba = f"{actual} {palabra}".strip()
                if font.getlength(prueba) * escala <= ancho_pt or not actual:
                    actual = prueba
                else:
                    lineas += 1
                    actual = palabra
            lineas += 1

        interlineado = p.line_spacing if isinstance(p.line_spacing, float) else 1.12
        alto += lineas * size * 1.2 * interlineado
        if p.space_after is not None:
            alto += p.space_after.pt
        if p.space_before is not None:
            alto += p.space_before.pt
    return alto


def revisar_pptx() -> list[str]:
    """Revisa estructura, cajas vacias, limites y desbordes del PPTX.

    Returns:
        Lista de problemas encontrados; vacia si todo esta correcto.
    """
    prs = Presentation(str(PPTX))
    problemas: list[str] = []
    ancho = prs.slide_width
    alto = prs.slide_height
    if (ancho, alto) != (Emu(int(13.333 * 914400)), Emu(int(7.5 * 914400))):
        problemas.append(f"tamano de diapositiva inesperado: {ancho} x {alto}")

    if len(prs.slides) != 12:
        problemas.append(f"diapositivas: {len(prs.slides)}, se esperaban 12")

    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.left is None or shape.top is None:
                continue
            if shape.left < 0 or shape.top < 0:
                problemas.append(f"slide {i}: forma fuera de la pagina ({shape.name})")
            if shape.left + shape.width > ancho + 1000:
                problemas.append(f"slide {i}: forma excede el ancho ({shape.name})")
            if shape.top + shape.height > alto + 1000:
                problemas.append(f"slide {i}: forma excede el alto ({shape.name})")

            if not shape.has_text_frame:
                continue
            texto = shape.text_frame.text.strip()
            if shape.shape_type == 17 and not texto:  # 17 = TEXT_BOX
                problemas.append(f"slide {i}: caja de texto vacia ({shape.name})")
            if not texto:
                continue

            tf = shape.text_frame
            ancho_util = (shape.width - tf.margin_left - tf.margin_right) / 12700.0
            alto_util = (shape.height - tf.margin_top - tf.margin_bottom) / 12700.0
            necesario = alto_requerido(tf, ancho_util)
            if necesario > alto_util + TOLERANCIA_PT:
                problemas.append(
                    f"slide {i}: posible desborde en '{texto[:42]}' "
                    f"({necesario:.0f} pt necesarios, {alto_util:.0f} pt disponibles)"
                )
    return problemas


def revisar_pdf() -> list[str]:
    """Revisa numero y tamano de las paginas del PDF.

    Returns:
        Lista de problemas encontrados; vacia si todo esta correcto.
    """
    from pypdf import PdfReader

    problemas: list[str] = []
    lector = PdfReader(str(PDF))
    if len(lector.pages) != 12:
        problemas.append(f"paginas del PDF: {len(lector.pages)}, se esperaban 12")
    for i, pagina in enumerate(lector.pages, 1):
        ancho = float(pagina.mediabox.width)
        alto = float(pagina.mediabox.height)
        if abs(ancho - 960) > 1 or abs(alto - 540) > 1:
            problemas.append(f"pagina {i}: tamano {ancho:.0f} x {alto:.0f} pt")
    return problemas


def main() -> None:
    """Ejecuta ambas revisiones e imprime el resumen."""
    problemas = revisar_pptx() + revisar_pdf()
    if problemas:
        print(f"PROBLEMAS ({len(problemas)}):")
        for problema in problemas:
            print(" -", problema)
        sys.exit(1)
    print("OK: 12 diapositivas, 12 paginas, sin cajas vacias ni desbordes estimados.")
    print(f"PPTX: {PPTX.stat().st_size / 1024:.0f} KB, PDF: {PDF.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - main reporta cualquier fallo de verificacion
        traceback.print_exc()
        sys.exit(1)
