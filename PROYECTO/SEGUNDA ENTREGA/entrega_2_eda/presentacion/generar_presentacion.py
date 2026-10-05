"""generar_presentacion.py

Fecha: 2026-10-04
Descripcion: genera la presentacion de la Entrega 2 del proyecto PLN
(EDA avanzado, representacion de textos y propuesta metodologica) EDITANDO
una COPIA del deck de la Entrega 1 (PLN_PROYECTO.pptx), de modo que el
resultado conserva exactamente su plantilla: fuentes, tamanos, colores,
geometrias, vinetas y fondos. El deck original de la Entrega 1 NUNCA se
modifica (solo se lee).

Estructura final (15 diapositivas):
  01 Portada (Entrega 2)                         -> slide original editada
  02 Contenido (agenda)                          -> slide original editada
  03-04 Resumen de la Entrega 1 (2)              -> clones de "Depuracion" y
                                                    "Dimension del corpus"
  05-06 Estado del arte (2)                      -> clones de "Estado del arte"
  07-14 Trabajo de la Entrega 2 (8)              -> clones editados + figuras
  15 Conclusiones y trabajo futuro               -> clon de "Conclusiones"

Tecnica (misma que scripts/fusionar_deck.py de la Entrega 1):
- clone_slide: deepcopy del spTree + remapeo de relaciones de imagen.
- Edicion de texto reutilizando el XML existente (rPr) para conservar
  formato; parrafos nuevos con el patron de vinetas del deck.
- Figuras nuevas insertadas con add_picture y ajustadas a las cajas de las
  figuras originales (sin deformar), con notas y barra de "hallazgo" del
  propio deck.

Contenido: solo cifras verificadas del informe de la Entrega 2
(informe/main.tex; resultados/metricas_entrega_ii.json); alcance 1992-2021;
sin resultados de modelos; sin emojis; sin simbolo de seccion.

Uso:
    /home/carlos/venv/bin/python presentacion/generar_presentacion.py
"""

from __future__ import annotations

import copy
import math
import shutil
import sys
import traceback
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.oxml.ns import qn
from lxml import etree

# =========================================================================
# Rutas y constantes
# =========================================================================
BASE_DIR = Path(__file__).resolve().parents[1]          # entrega_2_eda/
DECK_E1 = Path("/home/carlos/Git/PLN/PROYECTO/PRIMERA ENTREGA/PLN_PROYECTO.pptx")
FIG_DIR = BASE_DIR / "figuras"
OUT_DIR = BASE_DIR / "presentacion"
OUT_FILE = OUT_DIR / "Entrega_2_Presentacion.pptx"

EMU_IN = 914400.0
AZUL = "2F75B5"
GRIS_TEXTO = "425466"
GRIS_TARJETA = "162536"

# Figuras de la Entrega 2 usadas en el deck
FIG_NGRAMAS = FIG_DIR / "fig_ngramas_bi_tri.png"
FIG_COLOCACIONES = FIG_DIR / "fig_colocaciones.png"
FIG_MTLD = FIG_DIR / "fig_mtld_distribucion.png"
FIG_LEMMA_STEM = FIG_DIR / "fig_lemma_vs_stem.png"
FIG_LOGODDS = FIG_DIR / "fig_logodds_periodos.png"
FIG_TFIDF = FIG_DIR / "fig_representacion_tfidf.png"
FIG_EMBEDDINGS = FIG_DIR / "fig_embeddings_ejemplo.png"

# Indices (0-based) de las slides del deck E1 usadas como plantilla
IDX_PORTADA = 0
IDX_AGENDA = 1
IDX_TEXTO = 3          # "Lenguaje juridico como reto de PLN" (cuerpo CuadroTexto 10)
IDX_ARTE = 7           # "Estado del arte" (cuerpo Fusion-D2-Cuerpo con vinetas)
IDX_DEPURACION = 10    # "Depuracion y enriquecimiento del corpus" (flujo de tarjetas)
IDX_DIMENSION = 14     # "Dimension del corpus" (tarjetas de estadisticas + hallazgo)
IDX_FIG2 = 15          # "¿Que palabras dominan...?" (2 figuras lado a lado)
IDX_FIG1 = 16          # "Evolucion del corpus en el tiempo" (1 figura ancha)
IDX_FINAL = 20         # "Conclusiones" (tarjetas + panel siguiente etapa)


# =========================================================================
# Helpers XML / slides (portados de fusionar_deck.py)
# =========================================================================
def max_shape_id(prs):
    mx = 0
    for s in prs.slides:
        for cnv in s.shapes._spTree.iter(qn("p:cNvPr")):
            try:
                mx = max(mx, int(cnv.get("id")))
            except (TypeError, ValueError):
                pass
    return mx


def clone_slide(prs, src_idx, shape_id_start):
    """Clona la slide src_idx (deepcopy del spTree + remapeo de imagenes)."""
    src = prs.slides[src_idx]
    dest = prs.slides.add_slide(src.slide_layout)
    for shp in list(dest.shapes):  # quitar placeholders instanciados por add_slide
        shp._element.getparent().remove(shp._element)
    for shp in src.shapes:
        dest.shapes._spTree.append(copy.deepcopy(shp._element))
    next_id = shape_id_start
    for cnv in dest.shapes._spTree.iter(qn("p:cNvPr")):
        cnv.set("id", str(next_id))
        next_id += 1
    for blip in dest.shapes._spTree.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed"))
        if rid is None:
            continue
        rel = src.part.rels[rid]
        new_rid = dest.part.relate_to(rel.target_part, rel.reltype)
        blip.set(qn("r:embed"), new_rid)
    return dest


def delete_slide(prs, slide):
    """Elimina una slide del deck (quita el sldId y su relacion)."""
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        rid = sldId.get(qn("r:id"))
        if prs.part.related_part(rid) is slide.part:
            prs.part.drop_rel(rid)
            sldIdLst.remove(sldId)
            return
    raise ValueError("Slide no encontrada para eliminar: {}".format(slide.part.partname))


def reorder_slides(prs, orden):
    """Reordena las slides segun la lista `orden` (objetos slide)."""
    sldIdLst = prs.slides._sldIdLst
    pres_part = prs.part
    mapa = {}
    for sldId in sldIdLst:
        rid = sldId.get(qn("r:id"))
        part = pres_part.related_part(rid)
        mapa[str(part.partname)] = sldId
    for sldId in list(sldIdLst):
        sldIdLst.remove(sldId)
    for s in orden:
        sldIdLst.append(mapa[str(s.part.partname)])


def find_shapes(slide, name=None, name_prefix=None, shape_type=None,
                min_w_in=0.0, has_text=None):
    out = []
    for sh in slide.shapes:
        if name is not None and sh.name != name:
            continue
        if name_prefix is not None and not sh.name.startswith(name_prefix):
            continue
        if shape_type is not None and sh.shape_type != shape_type:
            continue
        if sh.width is not None and sh.width / EMU_IN < min_w_in:
            continue
        if has_text is True and not (sh.has_text_frame and sh.text_frame.text.strip()):
            continue
        out.append(sh)
    return out


def remove_shape(slide, shp):
    shp._element.getparent().remove(shp._element)


def set_run_text(run, text):
    run.text = text


def first_content_run(paragraph):
    for r in paragraph.runs:
        if r.text.strip():
            return r
    return paragraph.runs[0] if paragraph.runs else None


def replace_band_title(slide, title):
    """Titulo sobre la banda (patron del deck: 32 pt bold blanco)."""
    for sh in slide.shapes:
        if sh.shape_type != 1 or not sh.has_text_frame:
            continue
        if not sh.name.startswith("Google Shape;105"):
            continue
        p = sh.text_frame.paragraphs[0]
        run = first_content_run(p)
        if run is None:
            return
        rpr = run._r.find(qn("a:rPr"))
        if rpr is not None:
            rpr.set("sz", "3200")
            rpr.set("b", "1")
        set_run_text(run, title)
        txBody = sh._element.find(qn("p:txBody"))
        bodypr = txBody.find(qn("a:bodyPr"))
        if bodypr is not None:
            for tag in ("a:normAutofit", "a:spAutoFit"):
                child = bodypr.find(qn(tag))
                if child is not None:
                    bodypr.remove(child)
            etree.SubElement(bodypr, qn("a:noAutofit"))
        return


def replace_subtitle(slide, subtitle):
    """Subtitulo (TextBox 7) conservando su formato (28 pt bold)."""
    for sh in find_shapes(slide, name="TextBox 7", has_text=True):
        p = sh.text_frame.paragraphs[0]
        run = first_content_run(p)
        if run is not None:
            set_run_text(run, subtitle)
        return


def make_para(runs, marL=None, lvl=None, indent=None, bullet=False):
    """Crea un a:p con la estructura del deck. runs: dicts {text, bold, italic, sz, color}."""
    p = etree.Element(qn("a:p"))
    pPr = etree.SubElement(p, qn("a:pPr"))
    if lvl is not None:
        pPr.set("lvl", str(lvl))
    if marL is not None:
        pPr.set("marL", str(marL))
    if indent is not None:
        pPr.set("indent", str(indent))
    if bullet:
        buFont = etree.SubElement(pPr, qn("a:buFont"))
        buFont.set("typeface", "Courier New")
        buChar = etree.SubElement(pPr, qn("a:buChar"))
        buChar.set("char", "o")
    for spec in runs:
        r = etree.SubElement(p, qn("a:r"))
        rPr = etree.SubElement(r, qn("a:rPr"))
        rPr.set("lang", "es-ES")
        rPr.set("dirty", "0")
        rPr.set("sz", str(spec.get("sz", 1800)))
        if spec.get("bold"):
            rPr.set("b", "1")
        if spec.get("italic"):
            rPr.set("i", "1")
        color = spec.get("color")
        if color is not None:
            sf = etree.SubElement(rPr, qn("a:solidFill"))
            clr = etree.SubElement(sf, qn("a:srgbClr"))
            clr.set("val", color)
        t = etree.SubElement(r, qn("a:t"))
        t.text = spec["text"]
    return p


def clear_paragraphs(txBody_el):
    for p in txBody_el.findall(qn("a:p")):
        txBody_el.remove(p)


def replace_body_with_bullets(txBody_el, bullets):
    """Reemplaza el cuerpo por parrafos con vineta 'o' (patron del deck)."""
    clear_paragraphs(txBody_el)
    for runs in bullets:
        txBody_el.append(make_para(runs, marL=742950, lvl=1, indent=-285750, bullet=True))


def rewrite_paragraphs(shape, lineas):
    """Reescribe el texto del shape con `lineas` (lista de listas de runs).

    Reutiliza los formatos (rPr) de los parrafos y runs existentes; clona el
    ultimo run/parrafo cuando hace falta y elimina los sobrantes.
    """
    txBody = shape._element.find(qn("p:txBody"))
    paras = txBody.findall(qn("a:p"))
    while len(paras) < len(lineas):
        nuevo = copy.deepcopy(paras[-1])
        for r in nuevo.findall(qn("a:r")):
            nuevo.remove(r)
        txBody.append(nuevo)
        paras = txBody.findall(qn("a:p"))
    for p in paras[len(lineas):]:
        txBody.remove(p)
    paras = txBody.findall(qn("a:p"))
    for p, linea in zip(paras, lineas):
        runs = p.findall(qn("a:r"))
        if not runs:
            continue
        while len(runs) < len(linea):
            nuevo_r = copy.deepcopy(runs[-1])
            runs[-1].addnext(nuevo_r)
            runs = p.findall(qn("a:r"))
        for r, texto in zip(runs, linea):
            t = r.find(qn("a:t"))
            if t is None:
                t = etree.SubElement(r, qn("a:t"))
            t.text = texto
        for r in runs[len(linea):]:
            p.remove(r)


def rewrite_card(shape, textos):
    """Reescribe una tarjeta de un solo parrafo con saltos de linea a:br.

    `textos` es la lista de lineas visuales de la tarjeta (una cadena por
    linea; "" produce una linea en blanco). A diferencia de
    `rewrite_paragraphs`, trabaja sobre el unico parrafo de las tarjetas del
    deck (numero, etiqueta y cuerpo separados por a:br) y reutiliza los
    formatos (rPr) de los runs existentes por posicion, clonando el ultimo
    cuando hacen falta mas.
    """
    txBody = shape._element.find(qn("p:txBody"))
    p = txBody.findall(qn("a:p"))[0]
    plantillas = [copy.deepcopy(r) for r in p.findall(qn("a:r"))]
    for hijo in list(p):
        if hijo.tag in (qn("a:r"), qn("a:br")):
            p.remove(hijo)
    end = p.find(qn("a:endParaRPr"))

    def insertar(el):
        if end is not None:
            end.addprevious(el)
        else:
            p.append(el)

    pos = 0
    for i, texto in enumerate(textos):
        if i > 0:
            insertar(etree.Element(qn("a:br")))
        if texto == "":
            continue
        if pos < len(plantillas):
            run = plantillas[pos]
        elif plantillas:
            run = copy.deepcopy(plantillas[-1])
        else:
            run = etree.Element(qn("a:r"))
            etree.SubElement(run, qn("a:rPr"))
        t = run.find(qn("a:t"))
        if t is None:
            t = etree.SubElement(run, qn("a:t"))
        t.text = texto
        insertar(run)
        pos += 1


def set_shape_geom(shape, left_in, top_in, width_in, height_in):
    shape.left = Emu(int(left_in * EMU_IN))
    shape.top = Emu(int(top_in * EMU_IN))
    shape.width = Emu(int(width_in * EMU_IN))
    shape.height = Emu(int(height_in * EMU_IN))


def set_shape_geom_el(el, left_in, top_in, width_in, height_in):
    xfrm = el.find(qn("p:spPr")).find(qn("a:xfrm"))
    off = xfrm.find(qn("a:off"))
    ext = xfrm.find(qn("a:ext"))
    off.set("x", str(int(left_in * EMU_IN)))
    off.set("y", str(int(top_in * EMU_IN)))
    ext.set("cx", str(int(width_in * EMU_IN)))
    ext.set("cy", str(int(height_in * EMU_IN)))


def estimate_paragraph_lines(p_el, usable_w_in):
    """Estima la altura de un a:p (en pulgadas) segun sus runs."""
    pPr = p_el.find(qn("a:pPr"))
    mar = 0.0
    if pPr is not None and pPr.get("marL"):
        mar = int(pPr.get("marL")) / EMU_IN
    width = max(1.0, usable_w_in - mar)
    line_spacing = 1.2
    if pPr is not None:
        ln = pPr.find(qn("a:lnSpc"))
        if ln is not None:
            pct = ln.find(qn("a:spcPct"))
            if pct is not None and pct.get("val"):
                line_spacing = int(pct.get("val")) / 100000.0
    lineas = [[]]
    for r in p_el.iter(qn("a:r")):
        rPr = r.find(qn("a:rPr"))
        sz = 1800
        if rPr is not None and rPr.get("sz"):
            sz = int(rPr.get("sz"))
        t_el = r.find(qn("a:t"))
        texto = t_el.text or "" if t_el is not None else ""
        partes = texto.split("\n")
        for i, parte in enumerate(partes):
            if i > 0:
                lineas.append([])
            if parte:
                lineas[-1].append((parte, sz))
    while len(lineas) > 1 and not lineas[-1]:
        lineas.pop()
    if not any(lineas):
        return 1800 * 1.2 / 7200.0
    total_h = 0.0
    for linea in lineas:
        if not linea:
            total_h += 1800 * 1.2 / 7200.0
            continue
        sz_max = max(sz for _, sz in linea)
        largo = sum(len(t) for t, _ in linea)
        char_w = 0.5 * (sz_max / 100.0) / 72.0
        cpl = max(1, int(width / char_w))
        n_lines = max(1, math.ceil(largo / cpl))
        total_h += n_lines * (sz_max / 100.0) * line_spacing / 72.0
    return total_h


def estimate_text_height(txBody_el, usable_w_in):
    total = 0.0
    for p in txBody_el.findall(qn("a:p")):
        total += estimate_paragraph_lines(p, usable_w_in)
    return total


def fit_body(shape, left_in, top_in, width_in, max_bottom_in=7.2):
    """Ajusta la altura del cuerpo al texto estimado (patron de la fusion E1)."""
    est = estimate_text_height(shape._element.find(qn("p:txBody")), width_in)
    alto = min(max(2.6, est + 0.35), max_bottom_in - top_in)
    set_shape_geom(shape, left_in, top_in, width_in, alto)


# =========================================================================
# Helpers propios de esta entrega
# =========================================================================
def add_fitted_picture(slide, path, left_in, top_in, width_in, height_in,
                       name="E2-Figura"):
    """Inserta un PNG ajustado a la caja (sin deformar), centrado, y lo nombra."""
    with Image.open(path) as im:
        ratio = im.width / im.height
    w = width_in
    h = w / ratio
    if h > height_in:
        h = height_in
        w = h * ratio
    left = left_in + (width_in - w) / 2.0
    top = top_in + (height_in - h) / 2.0
    pic = slide.shapes.add_picture(
        str(path), Emu(int(left * EMU_IN)), Emu(int(top * EMU_IN)),
        Emu(int(w * EMU_IN)), Emu(int(h * EMU_IN)))
    pic.name = name
    return pic


def add_nota(slide, texto, left_in, top_in, width_in, size=11, align=2,
             italic=True, bold=False, name="E2-Nota"):
    """Nota al pie de figura/slide con el estilo de notas del deck (Arial)."""
    from pptx.dml.color import RGBColor
    tb = slide.shapes.add_textbox(
        Emu(int(left_in * EMU_IN)), Emu(int(top_in * EMU_IN)),
        Emu(int(width_in * EMU_IN)), Emu(int(0.32 * EMU_IN)))
    tb.name = name
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = texto
    r.font.name = "Arial"
    r.font.size = Pt(size)
    r.font.italic = italic
    r.font.bold = bold
    r.font.color.rgb = RGBColor.from_string(GRIS_TEXTO)
    return tb


def add_hallazgo(slide, src_el, texto, top_in=6.45, left_in=0.86, width_in=11.57):
    """Clona la barra HALLAZGO del deck y escribe `texto` tras el rotulo."""
    bar = copy.deepcopy(src_el)
    set_shape_geom_el(bar, left_in, top_in, width_in, 0.86)
    cnv = bar.find(qn("p:nvSpPr")).find(qn("p:cNvPr"))
    cnv.set("name", "E2-Hallazgo")
    txBody = bar.find(qn("p:txBody"))
    paras = txBody.findall(qn("a:p"))
    runs = paras[0].findall(qn("a:r"))
    t0 = runs[0].find(qn("a:t"))
    t0.text = "HALLAZGO"
    t1 = runs[1].find(qn("a:t"))
    t1.text = "  |  " + texto
    slide.shapes._spTree.append(bar)
    return bar


def set_slide_notes(slide, texto):
    """Notas del orador de la diapositiva."""
    slide.notes_slide.notes_text_frame.text = texto


# =========================================================================
# Agenda (patron de la fusion E1)
# =========================================================================
def agenda_rows(slide_agenda):
    """5 filas de la agenda; cada fila = [rect, num, nombre, rango]."""
    filas = []
    actual = []
    for shp in slide_agenda.shapes:
        is_row_rect = (shp.shape_type == 1
                       and "Rectángulo: esquinas red" in shp.name
                       and not (shp.has_text_frame and shp.text_frame.text.strip()))
        if is_row_rect:
            if actual:
                filas.append(actual)
            actual = [shp._element]
        elif actual:
            actual.append(shp._element)
    if actual:
        filas.append(actual)
    return filas


def set_fila_texto(fila, num, nombre, detalle, rango):
    """Escribe el contenido de una fila: [rect, num, nombre+detalle, rango].

    Conserva el formato del run del nombre (18 pt bold) y del detalle
    (12 pt gris) y normaliza el salto de linea (exactamente un a:br).
    """
    el_num, el_nombre, el_rango = fila[1], fila[2], fila[3]
    for t in el_num.iter(qn("a:t")):
        t.text = num
    p = el_nombre.find(qn("p:txBody")).find(qn("a:p"))
    runs = p.findall(qn("a:r"))
    if runs:
        t0 = runs[0].find(qn("a:t"))
        if t0 is not None:
            t0.text = nombre
        if len(runs) >= 2:
            r_det = runs[1]
        else:
            r_det = copy.deepcopy(runs[0])
            rpr = r_det.find(qn("a:rPr"))
            if rpr is not None:
                rpr.set("sz", "1200")
                rpr.set("b", "0")
                for c in rpr.iter(qn("a:srgbClr")):
                    c.set("val", GRIS_TEXTO)
            p.append(r_det)
        t1 = r_det.find(qn("a:t"))
        if t1 is None:
            t1 = etree.SubElement(r_det, qn("a:t"))
        t1.text = detalle
        for br in p.findall(qn("a:br")):
            p.remove(br)
        br = etree.Element(qn("a:br"))
        rpr_det = r_det.find(qn("a:rPr"))
        if rpr_det is not None:
            br.append(copy.deepcopy(rpr_det))
        r_det.addprevious(br)
    for t in el_rango.iter(qn("a:t")):
        t.text = rango


def update_agenda(slide_agenda):
    """Reescribe las 5 filas de la agenda con la ruta de la Entrega 2."""
    filas = agenda_rows(slide_agenda)
    assert len(filas) == 5, "Agenda con {} filas, se esperaban 5".format(len(filas))
    specs = [
        ("01", "Resumen de la Entrega 1", "Corpus, limpieza y textometría básica", "03–04"),
        ("02", "Estado del arte", "Antecedentes y representación", "05–06"),
        ("03", "Corpus medido y EDA avanzado", "Dataset, n-gramas, diversidad y diacronía", "07–12"),
        ("04", "Representación y metodología", "TF-IDF, embeddings y propuesta de modelado", "13–14"),
        ("05", "Conclusiones", "Hallazgos y trabajo futuro", "15"),
    ]
    for fila, (num, nombre, detalle, rango) in zip(filas, specs):
        set_fila_texto(fila, num, nombre, detalle, rango)


# =========================================================================
# Construccion del deck
# =========================================================================
def editar_portada(slide):
    """Portada: agrega la linea de la Entrega 2 y la universidad."""
    ph = find_shapes(slide, name_prefix="Google Shape;90")[0]
    txBody = ph._element.find(qn("p:txBody"))
    p0 = txBody.findall(qn("a:p"))[0]
    nuevo = copy.deepcopy(p0)
    r = nuevo.findall(qn("a:r"))[0]
    rpr = r.find(qn("a:rPr"))
    if rpr is not None:
        rpr.set("sz", "2000")
    t = r.find(qn("a:t"))
    t.text = "Entrega 2: EDA avanzado, representación de textos y propuesta metodológica"
    txBody.append(nuevo)

    autores = find_shapes(slide, name_prefix="Google Shape;91")[0]
    txA = autores._element.find(qn("p:txBody"))
    p_ult = txA.findall(qn("a:p"))[-1]
    nueva = copy.deepcopy(p_ult)
    t = nueva.findall(qn("a:r"))[0].find(qn("a:t"))
    t.text = "Pontificia Universidad Javeriana · Octubre de 2026"
    txA.append(nueva)


def construir():
    """Construye el deck completo y lo guarda; devuelve la lista final de slides."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DECK_E1, OUT_FILE)  # copia de trabajo; el original no se toca
    prs = Presentation(str(OUT_FILE))
    assert len(prs.slides) == 22, "El deck E1 debe tener 22 slides"

    # Referencias a las slides ORIGINALES (antes de clonar): solo estas se
    # eliminan o conservan; los clones son slides nuevas.
    originales = list(prs.slides)
    portada = originales[IDX_PORTADA]
    agenda = originales[IDX_AGENDA]

    # Barra HALLAZGO del deck (se clona en las slides de figuras)
    hallazgo_src = find_shapes(prs.slides[IDX_DIMENSION],
                               name="Rectángulo: esquinas redondeadas 12")[0]._element

    sid = max_shape_id(prs) + 1

    # --- clones de plantillas (antes de borrar originales) ---
    t_e1a = clone_slide(prs, IDX_DEPURACION, sid)            # Resumen E1: corpus
    t_e1b = clone_slide(prs, IDX_DIMENSION, sid + 60)        # Resumen E1: limpieza/textometria
    t_art1 = clone_slide(prs, IDX_ARTE, sid + 120)           # Estado del arte 1
    t_art2 = clone_slide(prs, IDX_ARTE, sid + 180)           # Estado del arte 2
    t_corr = clone_slide(prs, IDX_TEXTO, sid + 240)          # Correcciones vs E1
    t_cm = clone_slide(prs, IDX_DIMENSION, sid + 300)        # Corpus medido
    t_ngr = clone_slide(prs, IDX_FIG2, sid + 360)            # N-gramas y colocaciones
    t_mtl = clone_slide(prs, IDX_FIG2, sid + 420)            # MTLD y morfologia
    t_log = clone_slide(prs, IDX_FIG1, sid + 480)            # Log-odds
    t_rec = clone_slide(prs, IDX_ARTE, sid + 540)            # Recursos lexicos
    t_rep = clone_slide(prs, IDX_FIG2, sid + 600)            # Representacion
    t_pro = clone_slide(prs, IDX_ARTE, sid + 660)            # Propuesta metodologica
    t_fin = clone_slide(prs, IDX_FINAL, sid + 720)           # Conclusiones

    # ---------------------------------------------------------------------
    # 03. Resumen de la Entrega 1: corpus y depuracion (flujo de tarjetas)
    # ---------------------------------------------------------------------
    replace_band_title(t_e1a, "Resumen de la Entrega 1")
    replace_subtitle(t_e1a, "Corpus: del dato crudo al analizable")
    set_slide_notes(t_e1a,
                    "Sintesis de la Entrega 1: el corpus se depuro desde 23.750 "
                    "registros crudos; la extraccion por expresiones regulares logro "
                    "99,97 % de eficacia; se excluyeron 8 filas (6 autos y 2 HTML) y "
                    "quedaron 23.742 sentencias con una muestra estratificada de 25 "
                    "documentos para inspeccion.")

    # ---------------------------------------------------------------------
    # 04. Resumen de la Entrega 1: limpieza y textometria (tarjetas)
    # ---------------------------------------------------------------------
    replace_band_title(t_e1b, "Resumen de la Entrega 1")
    replace_subtitle(t_e1b, "Limpieza y textometría básica")
    set_slide_notes(t_e1b,
                    "La limpieza en ocho pasos normalizo el texto (minusculas, NFD, "
                    "preservacion de la ene, filtro de ruido y 306 stopwords "
                    "normalizadas). La textometria basica midio 260,99 millones de "
                    "palabras crudas, 250,01 millones de tokens normalizados y 128,81 "
                    "millones de tokens sin stopwords: las sentencias son documentos "
                    "muy extensos y formulaicos.")

    # ---------------------------------------------------------------------
    # 05-06. Estado del arte (2 slides)
    # ---------------------------------------------------------------------
    replace_band_title(t_art1, "Estado del arte")
    replace_subtitle(t_art1, "PLN legal en español y preprocesamiento")
    cuerpo_art1 = find_shapes(t_art1, name="Fusion-D2-Cuerpo")[0]
    cuerpo_art1.name = "E2-Arte1-Cuerpo"
    refs1 = [
        ("Anzola Cortés (2021): ",
         "minería de datos sobre más de 6.000 sentencias de constitucionalidad "
         "(1992-2020): el mismo universo documental del corpus."),
        ("Barbosa Becerra (2021): ",
         "análisis léxico-discursivo de sentencias de la Corte; evidencia fórmulas "
         "estables y léxico especializado."),
        ("HaCohen-Kerner et al. (2020): ",
         "el preprocesamiento cambia los resultados según el corpus y la tarea: "
         "fundamenta documentar cada paso de limpieza."),
        ("Soh et al. (2019): ",
         "corpus de sentencias largas: cautela con las frecuencias globales; "
         "comparación de clasificadores clásicos y neuronales."),
        ("Aletras et al. (2016) y González-González et al. (2023): ",
         "el poder predictivo varía por secciones de la sentencia (TEDH) y la "
         "clasificación de sentencias en español usa diccionarios de dominio."),
        ("Löffler et al. (2025): ",
         "escasez de recursos legales anotados en español para América Latina."),
    ]
    replace_body_with_bullets(
        cuerpo_art1._element.find(qn("p:txBody")),
        [[{"text": a, "bold": True}, {"text": b}] for a, b in refs1])
    txb = cuerpo_art1._element.find(qn("p:txBody"))
    txb.append(make_para(
        [{"text": "No hay trabajos indexados que usen el dataset exacto de Hugging "
                  "Face (verificado 2026-09-05); los antecedentes cubren el corpus "
                  "subyacente y la metodología.",
          "italic": True, "sz": 1200, "color": GRIS_TEXTO}], marL=457200, lvl=1))
    fit_body(cuerpo_art1, 0.75, 2.694, 11.42)
    set_slide_notes(t_art1,
                    "El estado del arte de esta entrega se apoya en el trabajo sobre "
                    "el mismo corpus, la evidencia de que el preprocesamiento importa "
                    "y los antecedentes de clasificacion de sentencias largas y en "
                    "espanol.")

    replace_band_title(t_art2, "Estado del arte")
    replace_subtitle(t_art2, "Representación, diversidad y diacronía")
    cuerpo_art2 = find_shapes(t_art2, name="Fusion-D2-Cuerpo")[0]
    cuerpo_art2.name = "E2-Arte2-Cuerpo"
    refs2 = [
        ("Manning et al. (2008): ",
         "fundamentos de recuperación de información y ponderación TF-IDF."),
        ("Dunning (1993): ",
         "razón de verosimilitud (LLR) para la estadística de colocaciones."),
        ("McCarthy y Jarvis (2010): ",
         "MTLD como medida de diversidad léxica robusta a la longitud del texto."),
        ("Hamilton et al. (2016): ",
         "log-odds con suavizado para comparar vocabularios entre periodos."),
        ("Chalkidis et al. (2020, 2021): ",
         "LEGAL-BERT y MultiEURLEX: vocabulario jurídico distinto del general y "
         "deriva temporal que exige particiones cronológicas."),
        ("Reimers y Gurevych (2019), Cañete et al. (2020) y Wang et al. (2022): ",
         "Sentence-BERT, BETO y e5 multilingüe como representaciones densas de "
         "comparación."),
    ]
    replace_body_with_bullets(
        cuerpo_art2._element.find(qn("p:txBody")),
        [[{"text": a, "bold": True}, {"text": b}] for a, b in refs2])
    txb2 = cuerpo_art2._element.find(qn("p:txBody"))
    txb2.append(make_para(
        [{"text": "Estas referencias fundamentan la representación (TF-IDF y "
                  "embeddings), la medida de colocaciones y el análisis diacrónico "
                  "de esta entrega.",
          "italic": True, "sz": 1200, "color": GRIS_TEXTO}], marL=457200, lvl=1))
    fit_body(cuerpo_art2, 0.75, 2.694, 11.42)
    set_slide_notes(t_art2,
                    "Para la representacion y la diacronia nos apoyamos en los "
                    "fundamentos de recuperacion de informacion, la razon de "
                    "verosimilitud, la MTLD, el log-odds y los modelos contextuales "
                    "en espanol y de dominio legal.")

    # ---------------------------------------------------------------------
    # 07. Correcciones y modificaciones respecto a la E1
    # ---------------------------------------------------------------------
    replace_band_title(t_corr, "Trabajo de la Entrega 2")
    replace_subtitle(t_corr, "Correcciones respecto a la Entrega 1")
    cuerpo_corr = find_shapes(t_corr, name="CuadroTexto 10")[0]
    cuerpo_corr.name = "E2-Correcciones-Cuerpo"
    cambios = [
        ("Dataset medido: ",
         "parquet de 23.742 filas × 12 columnas con texto, texto limpio y métricas "
         "por documento."),
        ("EDA extendido: ",
         "n-gramas, colocaciones, lemas/stems/POS, MTLD, granularidad y recursos "
         "léxicos."),
        ("Visualizaciones: ",
         "siete figuras nuevas; las figuras de la Entrega 1 se conservan."),
        ("Unidad validada: ",
         "sentencia 97,17 % vs subsentencia 4,64 % (criterio pre-registrado de "
         "+15 pp)."),
        ("Periodos: ",
         "contraste léxico 1992-2001 / 2002-2011 / 2012-2021 ejecutado (log-odds)."),
        ("Representación y metodología: ",
         "TF-IDF sobre lemas y propuesta de modelado definidas."),
    ]
    replace_body_with_bullets(
        cuerpo_corr._element.find(qn("p:txBody")),
        [[{"text": a, "bold": True}, {"text": b}] for a, b in cambios])
    txb_c = cuerpo_corr._element.find(qn("p:txBody"))
    txb_c.append(make_para(
        [{"text": "Las cifras canónicas de la Entrega 1 (23.742 documentos; "
                  "260.988.316 / 250.012.101 palabras) no requirieron corrección.",
          "italic": True, "sz": 1200, "color": GRIS_TEXTO}], marL=457200, lvl=1))
    fit_body(cuerpo_corr, 0.75, 2.694, 11.42)
    set_slide_notes(t_corr,
                    "Respecto a la Entrega 1 no hay rectificaciones de cifras: los "
                    "cambios son extensiones verificadas, del dataset medido al EDA "
                    "extendido, la unidad validada, la diacronia y la representacion "
                    "con la propuesta metodologica.")

    # ---------------------------------------------------------------------
    # 08. Corpus medido (tarjetas de estadisticas)
    # ---------------------------------------------------------------------
    replace_band_title(t_cm, "Corpus medido")
    replace_subtitle(t_cm, "Conjunto de datos y verificación")
    cards_cm = {
        "Rectángulo: esquinas redondeadas 1": ["23.742", "FILAS VERIFICADAS"],
        "Rectángulo: esquinas redondeadas 2": ["12", "COLUMNAS"],
        "Rectángulo: esquinas redondeadas 5": ["1.544,7 MB", "TAMAÑO DEL PARQUET"],
        "Rectángulo: esquinas redondeadas 7": [
            "260.988.316", "PALABRAS CRUDAS",
            "Suma verificada sobre las 23.742 filas"],
        "Rectángulo: esquinas redondeadas 8": [
            "250.012.101", "PALABRAS LIMPIAS",
            "Tras limpieza y filtro de ruido"],
        "Rectángulo: esquinas redondeadas 9": [
            "248.533.344", "LEMAS TOTALES",
            "Join verificado; 1.640.749.780 caracteres"],
    }
    for nombre, textos in cards_cm.items():
        sh = find_shapes(t_cm, name=nombre)[0]
        sh.name = "E2-CM-" + nombre.split()[-1]
        rewrite_card(sh, textos)
    encabezado = find_shapes(t_cm, name="CuadroTexto 6")[0]
    rewrite_paragraphs(encabezado, [["ESQUEMA DEL CONJUNTO MEDIDO"]])
    # Barra de hallazgo: verificacion
    hallazgo_cm = find_shapes(t_cm, name="Rectángulo: esquinas redondeadas 12")[0]
    rewrite_paragraphs(hallazgo_cm, [[
        "VERIFICACIÓN",
        "  |  23.742 ids únicos, sin nulos en claves; sumas reconciliadas; "
        "muestra de 100 filas + diccionario de datos."]])
    hallazgo_cm.name = "E2-CM-Hallazgo"
    set_slide_notes(t_cm,
                    "El requisito 3 se cumple con un parquet de 23.742 filas por 12 "
                    "columnas, verificado con aserciones duras: sin nulos, ids unicos "
                    "y sumas reconciliadas con las cifras canonicas. Se adjuntan "
                    "muestra estratificada, diccionario de datos y lemas por documento.")

    # ---------------------------------------------------------------------
    # 09. N-gramas y colocaciones (2 figuras)
    # ---------------------------------------------------------------------
    replace_band_title(t_ngr, "EDA avanzado")
    replace_subtitle(t_ngr, "N-gramas y colocaciones")
    for shp in list(t_ngr.shapes):
        if shp.shape_type == 13 and shp.width / EMU_IN > 4.0 and shp.height / EMU_IN > 3.0:
            remove_shape(t_ngr, shp)
    add_fitted_picture(t_ngr, FIG_NGRAMAS, 0.79, 2.9, 5.51, 2.45, "E2-Fig-Ngramas")
    add_fitted_picture(t_ngr, FIG_COLOCACIONES, 7.09, 2.9, 5.51, 2.45, "E2-Fig-Colocaciones")
    add_nota(t_ngr, "Top-15 de bigramas y trigramas más frecuentes.",
             0.79, 5.42, 5.51)
    add_nota(t_ngr, "Colocaciones por LLR (top-10): LLR 75 % vs frecuencia 55 % y PMI 0 %.",
             7.09, 5.42, 5.51)
    add_hallazgo(t_ngr, hallazgo_src,
                 "accion de tutela (344.893) es el trigrama más frecuente; el 82,7 % "
                 "de los trigramas guardados responde al patrón con palabra intermedia "
                 "vacía.", top_in=6.35)
    set_slide_notes(t_ngr,
                    "El corpus es fuertemente formulaico: accion de tutela domina y la "
                    "validacion manual de 20 colocaciones da 75 por ciento de formulas "
                    "validas con la razon de verosimilitud, frente a 55 por ciento de "
                    "la frecuencia y 0 por ciento del PMI.")

    # ---------------------------------------------------------------------
    # 10. Diversidad lexica y morfologia (2 figuras)
    # ---------------------------------------------------------------------
    replace_band_title(t_mtl, "EDA avanzado")
    replace_subtitle(t_mtl, "Diversidad léxica y morfología")
    for shp in list(t_mtl.shapes):
        if shp.shape_type == 13 and shp.width / EMU_IN > 4.0 and shp.height / EMU_IN > 3.0:
            remove_shape(t_mtl, shp)
    add_fitted_picture(t_mtl, FIG_MTLD, 0.79, 2.95, 5.51, 2.35, "E2-Fig-MTLD")
    add_fitted_picture(t_mtl, FIG_LEMMA_STEM, 7.09, 2.95, 5.51, 2.35, "E2-Fig-LemmaStem")
    add_nota(t_mtl, "MTLD por tipo de providencia y evolución anual.",
             0.79, 5.42, 5.51)
    add_nota(t_mtl, "Compresión del vocabulario: formas, lemas y stems.",
             7.09, 5.42, 5.51)
    add_hallazgo(t_mtl, hallazgo_src,
                 "MTLD 61,03 (T 61,63 > C 59,54 > SU 58,88); descenso sostenido de "
                 "64,32 a 58,53 entre periodos; lemas 87,3 % y stems 54,2 % del "
                 "vocabulario de formas.", top_in=6.35)
    set_slide_notes(t_mtl,
                    "La diversidad lexica desciende de forma sostenida en las tres "
                    "decadas y la lematizacion reduce el vocabulario conservando la "
                    "legibilidad: es la base de la representacion TF-IDF.")

    # ---------------------------------------------------------------------
    # 11. Evolucion lexica por periodos (figura ancha)
    # ---------------------------------------------------------------------
    replace_band_title(t_log, "EDA avanzado")
    replace_subtitle(t_log, "Evolución léxica por periodos")
    for shp in list(t_log.shapes):
        if shp.shape_type == 13 and shp.width / EMU_IN > 4.0 and shp.height / EMU_IN > 3.0:
            remove_shape(t_log, shp)
    add_fitted_picture(t_log, FIG_LOGODDS, 0.93, 2.55, 11.47, 3.6, "E2-Fig-Logodds")
    add_nota(t_log, "Log-odds con suavizado add-0,5 (2002-2011 → 2012-2021): "
                    "top-10 que sube y top-10 que baja.",
             0.93, 6.18, 11.47)
    add_hallazgo(t_log, hallazgo_src,
                 "Suben uariv, jep y coronavirus; bajan nilser y fomin. El método "
                 "replica la referencia interna (solapamiento del top-10: 7/10 a "
                 "9/10); siglas y ruido OCR se conservan como límite documentado.",
                 top_in=6.5)
    set_slide_notes(t_log,
                    "El contraste por periodos cumple la promesa de la Entrega 1: en "
                    "la ultima decada gana peso el vocabulario del conflicto armado y "
                    "la pandemia. Los extremos incluyen siglas y ruido de OCR; el "
                    "metodo es fiel a la referencia interna y se reporta el limite.")

    # ---------------------------------------------------------------------
    # 12. Recursos lexicos y ontologias (bullets)
    # ---------------------------------------------------------------------
    replace_band_title(t_rec, "EDA avanzado")
    replace_subtitle(t_rec, "Recursos léxicos y ontologías")
    cuerpo_rec = find_shapes(t_rec, name="Fusion-D2-Cuerpo")[0]
    cuerpo_rec.name = "E2-Recursos-Cuerpo"
    recs = [
        ("EuroVoc / MultiEURLEX: ",
         "tesauro multilingüe (español) de derecho de la UE; multietiqueta; "
         "aplicabilidad parcial (no mapea la Constitución ni la tutela)."),
        ("LKIF: ",
         "ontología formal OWL de conceptos jurídicos; aplicabilidad baja; útil como "
         "patrón de modelado."),
        ("ELI / ECLI: ",
         "esquemas de identificación y metadatos; no son vocabularios temáticos."),
        ("LegalRuleML / Akoma Ntoso: ",
         "marcado de reglas y estructura; baja a corto plazo (anotación manual "
         "costosa)."),
        ("Decisión: ",
         "taxonomía ligera desde la Constitución Política: títulos y capítulos como "
         "bloques temáticos, artículos 1-380 como etiquetas finas y respaldo "
         "\u201cderecho a X\u201d para tutelas."),
    ]
    replace_body_with_bullets(
        cuerpo_rec._element.find(qn("p:txBody")),
        [[{"text": a, "bold": True}, {"text": b}] for a, b in recs])
    txb_r = cuerpo_rec._element.find(qn("p:txBody"))
    txb_r.append(make_para(
        [{"text": "EuroVoc en español se conserva solo como referencia de "
                  "comparación; la decisión condiciona el diseño de etiquetas de la "
                  "propuesta metodológica.",
          "italic": True, "sz": 1200, "color": GRIS_TEXTO}], marL=457200, lvl=1))
    fit_body(cuerpo_rec, 0.75, 2.694, 11.42)
    set_slide_notes(t_rec,
                    "Ninguna ontologia internacional cubre el espanol juridico "
                    "colombiano: se evaluaron cuatro candidatas y se decidio derivar "
                    "una taxonomia ligera de la propia Constitucion Politica.")

    # ---------------------------------------------------------------------
    # 13. Representacion: TF-IDF y embeddings (2 figuras)
    # ---------------------------------------------------------------------
    replace_band_title(t_rep, "Representación")
    replace_subtitle(t_rep, "TF-IDF sobre lemas y embeddings")
    for shp in list(t_rep.shapes):
        if shp.shape_type == 13 and shp.width / EMU_IN > 4.0 and shp.height / EMU_IN > 3.0:
            remove_shape(t_rep, shp)
    add_fitted_picture(t_rep, FIG_TFIDF, 0.79, 2.9, 5.51, 2.5, "E2-Fig-TFIDF")
    add_fitted_picture(t_rep, FIG_EMBEDDINGS, 7.09, 2.9, 5.51, 2.5, "E2-Fig-Embeddings")
    add_nota(t_rep, "TF-IDF: cola larga del vocabulario y top-15 rasgos de "
                    "T-001/2021.", 0.79, 5.5, 5.51)
    add_nota(t_rep, "e5-base: similitud coseno entre cinco documentos (768 "
                    "dimensiones).", 7.09, 5.5, 5.51)
    add_hallazgo(t_rep, hallazgo_src,
                 "TF-IDF sobre lemas: 23.742 × 100.000, 95,66 % de esparsidad, ajuste "
                 "en ≈ 212 s; embeddings e5: similitud media 0,899 (0,91-0,92 entre "
                 "los documentos representativos).", top_in=6.35)
    set_slide_notes(t_rep,
                    "La representacion principal es TF-IDF sobre lemas: "
                    "interpretable, de bajo costo y con 95,66 por ciento de "
                    "esparsidad. Los embeddings contextuales se evaluaron como "
                    "comparacion: capturan cercania semantica con mayor costo.")

    # ---------------------------------------------------------------------
    # 14. Propuesta metodologica (bullets)
    # ---------------------------------------------------------------------
    replace_band_title(t_pro, "Propuesta metodológica")
    replace_subtitle(t_pro, "Modelos y criterios de evaluación")
    cuerpo_pro = find_shapes(t_pro, name="Fusion-D2-Cuerpo")[0]
    cuerpo_pro.name = "E2-Propuesta-Cuerpo"
    prop = [
        ("Tarea: ",
         "clasificación supervisada multi-clase; tipo de providencia (T/C/SU) como "
         "categoría inicial y área temática derivada de la Constitución como "
         "objetivo principal (etiquetas a validar)."),
        ("Particiones: ",
         "70/15/15 estratificado (semilla 42, prueba intocable); vectorizador "
         "ajustado solo con entrenamiento; evaluación cronológica complementaria."),
        ("Modelos a comparar: ",
         "línea base mayoritaria; TF-IDF sobre lemas con modelos lineales; "
         "transformer en español con segmentación y agregación."),
        ("Criterios: ",
         "macro-F1 (principal, por el desbalance T 72,02 %); F1 por clase; matriz de "
         "confusión; análisis de errores; robustez por periodo; costo (tiempo, "
         "memoria, GPU)."),
    ]
    replace_body_with_bullets(
        cuerpo_pro._element.find(qn("p:txBody")),
        [[{"text": a, "bold": True}, {"text": b}] for a, b in prop])
    txb_p = cuerpo_pro._element.find(qn("p:txBody"))
    txb_p.append(make_para(
        [{"text": "El protocolo fija desde ya las reglas del juego: particiones "
                  "selladas, anti-leakage y métricas robustas al desbalance.",
          "italic": True, "sz": 1200, "color": GRIS_TEXTO}], marL=457200, lvl=1))
    fit_body(cuerpo_pro, 0.75, 2.694, 11.42)
    set_slide_notes(t_pro,
                    "La propuesta define la tarea, las particiones con prueba "
                    "intocable, los modelos a comparar y los criterios: macro-F1, "
                    "analisis de errores y costo. Sin resultados de modelos: es una "
                    "propuesta metodologica.")

    # ---------------------------------------------------------------------
    # 15. Conclusiones y trabajo futuro (tarjetas)
    # ---------------------------------------------------------------------
    rotulo_fin = find_shapes(t_fin, name="CuadroTexto 1")[0]
    rewrite_paragraphs(rotulo_fin, [["RESULTADOS DE LA SEGUNDA ENTREGA"]])
    cards_fin = {
        "Rectángulo: esquinas redondeadas 2": [
            "23.742", "SENTENCIAS",
            "Dataset medido verificado; cifras de la E1 sin corrección."],
        "Rectángulo: esquinas redondeadas 5": [
            "95,66 %", "ESPARSIDAD TF-IDF",
            "Representación calculada sobre el corpus completo."],
        "Rectángulo: esquinas redondeadas 6": [
            "64,32→58,53", "MTLD (1992-2021)",
            "Diversidad léxica en descenso sostenido."],
        "Rectángulo: esquinas redondeadas 7": [
            "CONCLUSIÓN PRINCIPAL",
            "El género es formulaico (accion de tutela domina el vocabulario) y su "
            "léxico evoluciona; el corpus medido y la representación TF-IDF dejan "
            "lista la fase de modelado."],
        "Rectángulo: esquinas redondeadas 8": [
            "SIGUIENTE ETAPA",
            "",
            "01  Validar etiquetas temáticas con expertos",
            "02  Comparar modelos con macro-F1",
            "03  Análisis de errores por periodo",
            "04  Embeddings de dominio y recuperación",
            "05  Resúmenes para sentencias extensas",
            "06  Documentar y publicar el corpus medido"],
        "Rectángulo: esquinas redondeadas 10": [
            "DEL CORPUS MEDIDO  →  A LA FASE DE MODELADO"],
    }
    for nombre, textos in cards_fin.items():
        sh = find_shapes(t_fin, name=nombre)[0]
        sh.name = "E2-FIN-" + nombre.split()[-1]
        rewrite_card(sh, textos)
    set_slide_notes(t_fin,
                    "Cierre: corpus y dataset consolidados, genero formulaico, "
                    "diversidad decreciente con recambio tematico, unidad validada y "
                    "representacion definida. El trabajo inmediato es validar "
                    "etiquetas, comparar modelos y analizar errores por periodo.")

    # ---------------------------------------------------------------------
    # Portada y agenda (slides originales conservadas)
    # ---------------------------------------------------------------------
    editar_portada(portada)
    update_agenda(agenda)
    set_slide_notes(portada,
                    "Presentamos la Entrega 2: EDA avanzado, representacion de "
                    "textos y propuesta metodologica del proyecto de PLN sobre "
                    "jurisprudencia constitucional colombiana.")
    set_slide_notes(agenda,
                    "La ruta de la presentacion: resumen de la Entrega 1, estado del "
                    "arte, el trabajo de la Entrega 2 y las conclusiones con el "
                    "trabajo futuro.")

    # ---------------------------------------------------------------------
    # Borrar las slides originales que no se conservan y reordenar
    # ---------------------------------------------------------------------
    conservadas = [portada, agenda]
    for s in originales:
        if s not in conservadas:
            delete_slide(prs, s)
    orden = [portada, agenda, t_e1a, t_e1b, t_art1, t_art2,
             t_corr, t_cm, t_ngr, t_mtl, t_log, t_rec, t_rep, t_pro, t_fin]
    assert len(orden) == 15, "Orden final con {} slides".format(len(orden))
    reorder_slides(prs, orden)

    prs.save(str(OUT_FILE))
    print("OK: deck guardado en {}".format(OUT_FILE))
    return orden


# =========================================================================
# Verificacion
# =========================================================================
def verificar():
    """Reabre el deck: conteos, imagenes y avisos de desborde de cajas E2-*."""
    prs = Presentation(str(OUT_FILE))
    n = len(prs.slides)
    assert n == 15, "Se esperaban 15 slides, hay {}".format(n)
    print("\nVERIFICACION: {} slides".format(n))
    total_pics = 0
    problemas = []
    for i, s in enumerate(prs.slides, 1):
        pics = [sh for sh in s.shapes if sh.shape_type == 13]
        total_pics += len(pics)
        titulos = [sh.text_frame.text.strip().split("\n")[0]
                   for sh in s.shapes
                   if sh.has_text_frame and sh.text_frame.text.strip()]
        print("  slide {:2d} | pics={:2d} | {}".format(
            i, len(pics), titulos[0][:60] if titulos else "(sin texto)"))
        for sh in s.shapes:
            if not (sh.name.startswith("E2-") and sh.has_text_frame
                    and sh.text_frame.text.strip()):
                continue
            left, top = sh.left / EMU_IN, sh.top / EMU_IN
            width, height = sh.width / EMU_IN, sh.height / EMU_IN
            if (left < -0.02 or top < -0.02 or left + width > 13.35
                    or top + height > 7.45):
                problemas.append("slide {} '{}': geometria L={:.2f} T={:.2f} "
                                 "R={:.2f} B={:.2f}".format(
                                     i, sh.name, left, top, left + width, top + height))
                continue
            est = estimate_text_height(sh._element.find(qn("p:txBody")), width)
            if est > height + 0.12:
                problemas.append("slide {} '{}': texto {:.2f}in > caja {:.2f}in".format(
                    i, sh.name, est, height))
    print("  imagenes embebidas:", total_pics)
    if problemas:
        print("  AVISOS:")
        for p in problemas:
            print("   -", p)
    else:
        print("  Sin avisos de desborde en cajas nuevas/editadas.")

    # Regresion: los cuerpos de las tarjetas deben llegar al deck (bug de
    # parrafos sin runs en rewrite_paragraphs, corregido con rewrite_card).
    todo = "\n".join(sh.text_frame.text for s in prs.slides for sh in s.shapes
                     if sh.has_text_frame)
    for clave in ["Dataset medido verificado",
                  "Suma verificada sobre las 23.742 filas",
                  "01  Validar etiquetas temáticas con expertos",
                  "06  Documentar y publicar el corpus medido"]:
        assert clave in todo, "Falta texto esperado en el deck: {!r}".format(clave)
    print("  aserciones de contenido: OK")
    return n, total_pics, problemas


def main():
    try:
        construir()
        verificar()
        print("\nFin de la ejecucion.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
