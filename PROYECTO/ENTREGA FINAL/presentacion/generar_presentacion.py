"""generar_presentacion.py

Fecha: 2026-10-31
Descripcion: genera el deck de la Entrega 3 de Juris-PLN (12 diapositivas,
16:9) con python-pptx desde cero, replicando la identidad visual del deck de
la Entrega 2: fondo blanco, superficie de tarjeta #F5F7FA, tinta #162536,
gris de texto #425466 y un unico acento azul #2F75B5.

Estructura (una diapositiva por hallazgo):
  01 Portada
  02 El problema
  03 La pregunta y el producto
  04 Los datos
  05 Como se mide
  06 Resultado de recuperacion
  07 Lo que no funciono
  08 El producto
  09 La demo en cuatro pantallas
  10 El agente, medido
  11 Limites y etica
  12 Cierre

Reglas de diseno: franja azul de 0,08 in en el borde superior, pie con
"Juris-PLN . Entrega 3 . 2026" y numero de pagina, titulos en serif 28-32 pt,
cuerpo en sans 14-18 pt, cifras clave en 36-44 pt, maximo cuatro vinetas de
catorce palabras, sin degradados, sin sombras, sin emojis y sin guiones largos.

Uso:
    /home/carlos/venv/bin/python presentacion/generar_presentacion.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# =========================================================================
# SECCION 0: rutas, paleta y tipografia
# =========================================================================
BASE_DIR = Path(__file__).resolve().parent
FIG_MTLD = BASE_DIR / "fig_mtld_por_ano.png"
OUT_DIR = BASE_DIR
OUT_PPTX = OUT_DIR / "Entrega_3_Presentacion.pptx"
OUT_GUION = OUT_DIR / "guion_presentacion.md"

BLANCO = "FFFFFF"
TINTA = "162536"
GRIS_TEXTO = "425466"
TARJETA = "F5F7FA"
AZUL = "2F75B5"

SERIF = "Times New Roman"
SANS = "Arial"

SLIDE_W = 13.333
SLIDE_H = 7.5
MARGEN = 0.85
ANCHO = SLIDE_W - 2 * MARGEN          # 11,633 in de retícula util
PIE_Y = 7.02
TITULO_Y = 0.56


# =========================================================================
# SECCION 1: utilidades de dibujo
# =========================================================================
def rgb(hex_color: str) -> RGBColor:
    """Convierte un color hexadecimal en RGBColor.

    Args:
        hex_color: Color de seis digitos sin almohadilla, por ejemplo "2F75B5".

    Returns:
        Color listo para asignar a rellenos, lineas y fuentes.
    """
    return RGBColor.from_string(hex_color)


def _plano(shape) -> None:
    """Deja una forma sin efectos del tema: sin sombra y sin estilo heredado.

    Elimina el nodo p:style, porque LibreOffice sigue aplicando su effectRef
    aunque el efecto explicito este vacio, y vacia la lista de efectos.

    Args:
        shape: Forma o conector de python-pptx.
    """
    estilo = shape._element.find(qn("p:style"))
    if estilo is not None:
        shape._element.remove(estilo)
    shape.shadow.inherit = False


def add_texto(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    texto: str,
    size: float = 16,
    color: str = GRIS_TEXTO,
    bold: bool = False,
    italic: bool = False,
    font: str = SANS,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    line_spacing: float = 1.12,
    space_after: float = 0,
    espaciado: int | None = None,
):
    """Inserta una caja de texto sin margenes internos.

    Args:
        slide: Diapositiva destino.
        x: Distancia al borde izquierdo en pulgadas.
        y: Distancia al borde superior en pulgadas.
        w: Ancho de la caja en pulgadas.
        h: Alto de la caja en pulgadas.
        texto: Contenido; los saltos de linea crean parrafos.
        size: Tamano en puntos.
        color: Color hexadecimal del texto.
        bold: Si el texto va en negrita.
        italic: Si el texto va en cursiva.
        font: Familia tipografica.
        align: Alineacion de parrafo.
        anchor: Anclaje vertical del texto.
        line_spacing: Interlineado como multiplo del tamano.
        space_after: Separacion posterior entre parrafos en puntos.
        espaciado: Tracking en centesimas de punto, o None.

    Returns:
        La caja de texto creada.
    """
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor

    for i, linea in enumerate(str(texto).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        if space_after:
            p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = linea
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        r.font.color.rgb = rgb(color)
        if espaciado is not None:
            r.font._rPr.set("spc", str(espaciado))
    return tb


def add_titulo(slide, texto: str, size: float = 30, y: float = TITULO_Y, h: float = 1.35):
    """Escribe el titulo de la diapositiva en serif.

    Args:
        slide: Diapositiva destino.
        texto: Titulo, formulado como hallazgo.
        size: Tamano en puntos (28 a 32 en el cuerpo del deck).
        y: Distancia al borde superior en pulgadas.
        h: Alto reservado en pulgadas.

    Returns:
        La caja de texto del titulo.
    """
    return add_texto(
        slide,
        MARGEN,
        y,
        ANCHO,
        h,
        texto,
        size=size,
        color=TINTA,
        bold=True,
        font=SERIF,
        line_spacing=1.0,
    )


def add_kicker(slide, texto: str, y: float, color: str = AZUL):
    """Escribe una etiqueta superior en versalitas simuladas.

    Args:
        slide: Diapositiva destino.
        texto: Etiqueta breve en mayusculas.
        y: Distancia al borde superior en pulgadas.
        color: Color hexadecimal de la etiqueta.

    Returns:
        La caja de texto creada.
    """
    return add_texto(
        slide,
        MARGEN,
        y,
        ANCHO,
        0.3,
        texto.upper(),
        size=11,
        color=color,
        bold=True,
        espaciado=140,
    )


def add_pie(slide, numero: int) -> None:
    """Dibuja el pie comun: autoria del deck y numero de pagina.

    Args:
        slide: Diapositiva destino.
        numero: Numero de pagina impreso a la derecha.
    """
    add_texto(slide, MARGEN, PIE_Y, 6.5, 0.28, "Juris-PLN · Entrega 3 · 2026", size=11, color=GRIS_TEXTO)
    add_texto(
        slide,
        SLIDE_W - MARGEN - 1.5,
        PIE_Y,
        1.5,
        0.28,
        str(numero),
        size=11,
        color=GRIS_TEXTO,
        align=PP_ALIGN.RIGHT,
    )


def add_franja(slide) -> None:
    """Dibuja la franja azul de 0,08 in del borde superior.

    Args:
        slide: Diapositiva destino.
    """
    barra = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SLIDE_W), Inches(0.08))
    barra.fill.solid()
    barra.fill.fore_color.rgb = rgb(AZUL)
    barra.line.fill.background()
    _plano(barra)


def add_tarjeta(slide, x: float, y: float, w: float, h: float, relleno: str = TARJETA):
    """Dibuja una tarjeta rectangular de superficie con esquinas suaves.

    Args:
        slide: Diapositiva destino.
        x: Distancia al borde izquierdo en pulgadas.
        y: Distancia al borde superior en pulgadas.
        w: Ancho en pulgadas.
        h: Alto en pulgadas.
        relleno: Color hexadecimal de la superficie.

    Returns:
        La forma creada, lista para recibir texto en su text_frame.
    """
    forma = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    forma.adjustments[0] = 0.05
    forma.fill.solid()
    forma.fill.fore_color.rgb = rgb(relleno)
    forma.line.fill.background()
    _plano(forma)
    return forma


def texto_en_tarjeta(
    forma,
    etiqueta: str | None,
    cuerpo: str,
    cuerpo_size: float = 16,
    etiqueta_size: float = 11,
    color_cuerpo: str = GRIS_TEXTO,
    align=PP_ALIGN.LEFT,
    margen: float = 0.24,
    anchor=MSO_ANCHOR.TOP,
) -> None:
    """Rellena una tarjeta con una etiqueta pequena y un cuerpo.

    Args:
        forma: Forma devuelta por add_tarjeta.
        etiqueta: Texto superior en azul, o None para omitirlo.
        cuerpo: Texto principal.
        cuerpo_size: Tamano del cuerpo en puntos.
        etiqueta_size: Tamano de la etiqueta en puntos.
        color_cuerpo: Color hexadecimal del cuerpo.
        align: Alineacion de parrafo.
        margen: Margen interno en pulgadas.
        anchor: Anclaje vertical del texto.
    """
    tf = forma.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = Inches(margen)
    tf.margin_right = Inches(margen)
    tf.margin_top = Inches(0.18)
    tf.margin_bottom = Inches(0.12)
    tf.vertical_anchor = anchor

    primero = True
    if etiqueta:
        p = tf.paragraphs[0]
        primero = False
        p.alignment = align
        p.line_spacing = 1.0
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = etiqueta.upper()
        r.font.size = Pt(etiqueta_size)
        r.font.bold = True
        r.font.name = SANS
        r.font.color.rgb = rgb(AZUL)
        r.font._rPr.set("spc", "120")

    for linea in str(cuerpo).split("\n"):
        p = tf.paragraphs[0] if primero else tf.add_paragraph()
        primero = False
        p.alignment = align
        p.line_spacing = 1.1
        r = p.add_run()
        r.text = linea
        r.font.size = Pt(cuerpo_size)
        r.font.name = SANS
        r.font.color.rgb = rgb(color_cuerpo)


def add_stat(slide, x: float, y: float, w: float, valor: str, etiqueta: str,
             size: float = 40, alto_valor: float = 0.78, color_valor: str = TINTA):
    """Dibuja una cifra clave con su etiqueta explicativa.

    Args:
        slide: Diapositiva destino.
        x: Distancia al borde izquierdo en pulgadas.
        y: Distancia al borde superior en pulgadas.
        w: Ancho en pulgadas.
        valor: Cifra que se destaca.
        etiqueta: Texto que la explica.
        size: Tamano de la cifra en puntos.
        alto_valor: Alto reservado para la cifra en pulgadas.
        color_valor: Color hexadecimal de la cifra.
    """
    add_texto(slide, x, y, w, alto_valor, valor, size=size, color=color_valor,
              bold=True, font=SANS, line_spacing=0.98)
    add_texto(slide, x, y + alto_valor, w, 0.62, etiqueta, size=13, color=GRIS_TEXTO,
              line_spacing=1.1)


def add_vinetas(slide, x: float, y: float, w: float, h: float, items: list[str],
                size: float = 17, space_after: float = 14, color: str = GRIS_TEXTO):
    """Escribe una lista con vinetas cuadradas azules.

    Args:
        slide: Diapositiva destino.
        x: Distancia al borde izquierdo en pulgadas.
        y: Distancia al borde superior en pulgadas.
        w: Ancho en pulgadas.
        h: Alto en pulgadas.
        items: Vinetas, maximo cuatro y catorce palabras cada una.
        size: Tamano del texto en puntos.
        space_after: Separacion entre vinetas en puntos.
        color: Color hexadecimal del texto.
    """
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0

    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.15
        p.space_after = Pt(space_after)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.34)))
        pPr.set("indent", str(-Inches(0.34)))
        bu_clr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(bu_clr, qn("a:srgbClr")).set("val", AZUL)
        etree.SubElement(pPr, qn("a:buSzPct")).set("val", "80000")
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "▪")
        r = p.add_run()
        r.text = item
        r.font.size = Pt(size)
        r.font.name = SANS
        r.font.color.rgb = rgb(color)


def add_flecha(slide, x1: float, y1: float, x2: float, y2: float) -> None:
    """Dibuja una flecha fina azul entre dos puntos.

    Args:
        slide: Diapositiva destino.
        x1: Origen horizontal en pulgadas.
        y1: Origen vertical en pulgadas.
        x2: Destino horizontal en pulgadas.
        y2: Destino vertical en pulgadas.
    """
    flecha = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    flecha.line.color.rgb = rgb(AZUL)
    flecha.line.width = Pt(1.5)
    ln = flecha.line._get_or_add_ln()
    etree.SubElement(ln, qn("a:tailEnd")).attrib.update(
        {"type": "triangle", "w": "med", "len": "med"}
    )
    _plano(flecha)


def add_notas(slide, texto: str) -> None:
    """Escribe las notas del orador de la diapositiva.

    Args:
        slide: Diapositiva destino.
        texto: Guion de 60 a 90 segundos.
    """
    slide.notes_slide.notes_text_frame.text = texto


def nueva_slide(prs, numero: int):
    """Crea una diapositiva en blanco con franja y pie ya dibujados.

    Args:
        prs: Presentacion destino.
        numero: Numero de pagina.

    Returns:
        La diapositiva creada.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_franja(slide)
    add_pie(slide, numero)
    return slide


# =========================================================================
# SECCION 2: guion del orador (fuente unica para notas y guion_presentacion.md)
# =========================================================================
GUION = [
    {
        "titulo": "Portada",
        "voz": "23.742 sentencias",
        "texto": (
            "Buenas tardes. Soy Carlos Andrés Garnica Salazar y presento la Entrega 3 de "
            "Juris-PLN, el proyecto de Procesamiento de Lenguaje Natural de la Maestría en "
            "Ingeniería de Sistemas y Computación de la Javeriana. El título resume el "
            "recorrido completo: partimos de 23.742 sentencias de la Corte Constitucional y "
            "llegamos a un sistema que devuelve los artículos de la Constitución que "
            "fundamentan cada providencia, con la evidencia textual que lo respalda. En los "
            "próximos quince minutos voy a presentar el problema, la forma de medirlo, los "
            "resultados, lo que no funcionó y los límites. Todos los resultados están "
            "congelados y construidos sobre datos públicos."
        ),
    },
    {
        "titulo": "El problema",
        "voz": "48 por ciento de los tokens limpios",
        "texto": (
            "El punto de partida es un corpus de 23.742 sentencias publicadas entre 1992 y "
            "2021, con 261 millones de palabras. Cuando un jurista necesita saber en qué se "
            "apoya una decisión, no tiene un índice: la cita que sostiene la ratio vive "
            "dentro del texto. Por eso lee el documento completo y lo repite caso por caso. "
            "La tarea es repetitiva, costosa y difícil de auditar a escala. Hay un dato que "
            "explica la dificultad: el 48 por ciento de los tokens limpios son fórmulas "
            "institucionales, no conceptos jurídicos. Casi la mitad del texto no aporta "
            "señal semántica para recuperar."
        ),
    },
    {
        "titulo": "La pregunta y el producto",
        "voz": "evidencia textual verificable",
        "texto": (
            "La pregunta de trabajo es concreta: dada una providencia, qué artículos de la "
            "Constitución la fundamentan y con qué evidencia textual. La respuesta no es una "
            "opinión: cada artículo llega con la frase que lo respalda, para que cualquiera "
            "pueda verificarla. Sobre esa base el producto añade dos vistas: las citas "
            "resueltas y la jurisprudencia asociada a cada artículo. El público son juristas, "
            "investigadores y litigantes. Conviene delimitar la utilidad desde el inicio: el "
            "sistema ordena la búsqueda y la hace auditable, no dictamina ni sustituye el "
            "criterio jurídico."
        ),
    },
    {
        "titulo": "Los datos",
        "voz": "10.530 palabras limpias de media",
        "texto": (
            "El corpus tiene 23.742 sentencias: 17.099 de tutela, 6.246 de constitucionalidad "
            "y 397 de sentencia unificada. Cada documento promedia 10.530 palabras limpias y "
            "una mediana de 7.158, es decir, documentos largos con una cola importante. La "
            "figura muestra dos cosas: la diversidad léxica medida con MTLD por tipo de "
            "providencia, y su evolución anual. El dato importante está en la línea: el MTLD "
            "cae de 64,88 a 56,77 entre 1992 y 2021. El vocabulario se empobrece y se "
            "institucionaliza, lo que refuerza la necesidad de un recuperador que no dependa "
            "solo de coincidencia de palabras."
        ),
    },
    {
        "titulo": "Cómo se mide",
        "voz": "90 casos y 820 pares",
        "texto": (
            "La evaluación tiene tres piezas. Primero, un benchmark interno de 90 casos con "
            "etiquetas generadas por un modelo de lenguaje, que sirve para iterar rápido. "
            "Segundo, un gold humano B3 de 50 casos y 820 pares artículo sentencia, que es la "
            "referencia más exigente. Tercero, C3, un corpus de 2022 a 2026 que no participó "
            "en el desarrollo y mide deriva temporal. Las métricas son nDCG arroba 10, F1 "
            "arroba 5 y MRR, siempre con intervalos de confianza al 95 por ciento por "
            "bootstrap pareado, porque comparar promedios sueltos enmascara diferencias "
            "pequeñas. El gate de adopción es el ADR-16: la mejora debe ser significativa y "
            "consistente entre conjuntos."
        ),
    },
    {
        "titulo": "Resultado de recuperación",
        "voz": "0,6907 frente a 0,6013",
        "texto": (
            "Aquí está el resultado central. La línea base BM25 con expansión por hubs obtiene "
            "0,6013 de nDCG arroba 10. La versión 2.2, que combina el retriever B2 con el "
            "reranker B1, sube a 0,6907. Sobre el gold humano la misma configuración alcanza "
            "0,7852, porque el gold es más limpio que las etiquetas automáticas. La mejora "
            "pareada frente a la línea base es de 0,067 puntos, con intervalo de confianza al "
            "95 por ciento entre 0,042 y 0,089: el cero queda fuera, así que la mejora no es "
            "ruido. En el corpus posterior, C3, el sistema mantiene 0,7472, sin deriva "
            "significativa."
        ),
    },
    {
        "titulo": "Lo que no funcionó",
        "voz": "31,81 por ciento de ruido de etiqueta",
        "texto": (
            "La transparencia es parte del resultado. Probamos un ajuste LoRA sobre el modelo "
            "C2 y perdió 0,198 puntos en consultas: no generalizó. Una segunda época del "
            "retriever B2 sumó 0,0013, una diferencia dentro del ruido y sin significancia. "
            "Con 200 casos zero shot generados por el modelo de lenguaje tampoco superamos la "
            "línea base. Y el clasificador M4 se queda en 0,2870 de macro F1, con un 31,81 por "
            "ciento de ruido de etiqueta detectado con cleanlab y documentado en el ADR-19. "
            "Por eso el área de clasificación queda como exploratoria y no se ofrece en el "
            "producto: preferimos declararlo antes que presentar una cifra que no se sostiene."
        ),
    },
    {
        "titulo": "El producto",
        "voz": "tres capas de guardrails",
        "texto": (
            "El producto corre entero en local, en tres procesos. Streamlit en el puerto 8502 "
            "sirve la interfaz. FastAPI en el 8123 expone los contratos y aplica los "
            "guardrails. El núcleo validado, con recuperación, reranking y el modelo local vía "
            "ollama, hace el trabajo pesado. Los endpoints principales son link, evidencia, "
            "jurisprudencia por artículo y ask. Los guardrails están en tres capas: un router "
            "determinista que clasifica la intención antes del modelo, un agente con tool "
            "calling que solo consulta herramientas del núcleo, y una verificación de grounding "
            "que exige cita en el texto. Sin cita, no hay respuesta."
        ),
    },
    {
        "titulo": "La demo en cuatro pantallas",
        "voz": "id 2195, T-293 de 1997",
        "texto": (
            "La demo se recorre en cuatro pantallas. La primera va de la sentencia a la "
            "Constitución, con la cita textual al lado de cada artículo. La segunda va del "
            "artículo a la jurisprudencia: el artículo 86 aparece en 16.405 sentencias del "
            "corpus, un número que muestra la escala del problema. La tercera muestra la "
            "evaluación con la procedencia de cada resultado, para que el usuario sepa de dónde "
            "sale. La cuarta entrega la ficha en PDF y las sentencias similares. El ejemplo que "
            "usamos siempre es el mismo: la sentencia con identificador 2195, que corresponde a "
            "la T-293 de 1997, con 6 artículos extraídos y 10 recuperados."
        ),
    },
    {
        "titulo": "El agente, medido",
        "voz": "13,4 segundos de mediana",
        "texto": (
            "El agente también se mide. Sobre un gold set de 100 prompts, el ruteo exacto "
            "acierta el 90 por ciento. La ruta jurídica, los saludos no rechazados, el rechazo "
            "de consultas fuera de alcance, el uso correcto de herramientas y el grounding "
            "llegan al 100 por ciento. Es decir, cuando el sistema responde, la respuesta está "
            "anclada en evidencia recuperada. La latencia es el costo visible: 13,4 segundos de "
            "mediana y 60,8 en el percentil 95. Ese p95 define el presupuesto de espera de la "
            "demo y explica por qué la interfaz muestra el avance por etapas."
        ),
    },
    {
        "titulo": "Límites y ética",
        "voz": "corpus hasta 2021",
        "texto": (
            "Los límites están declarados. El corpus termina en 2021; el periodo posterior solo "
            "se usa para medir deriva. El clasificador M4 no es fiable y no se ofrece como "
            "resultado. El gold humano se anotó con apoyo de un modelo de lenguaje, no con "
            "validación de expertos legales, así que hay que leerlo como proxy y no como "
            "verdad de referencia. Los datos son públicos y no contienen información personal. "
            "Queda un riesgo técnico abierto: el modelo local puede degradar su salida, y para "
            "eso existe el fallback determinista, que devuelve igualmente la evidencia "
            "recuperada."
        ),
    },
    {
        "titulo": "Cierre",
        "voz": "C1 con 150 casos",
        "texto": (
            "Cierro con lo esencial: el producto funciona y es auditable. Cada respuesta se "
            "puede rastrear hasta la frase que la sostiene, y los resultados negativos están "
            "documentados igual que los positivos. El trabajo inmediato tiene tres frentes: C1 "
            "con 150 casos y anotación controlada, el despliegue externo con límites de uso "
            "declarados, y la validación con expertos legales, que es el único camino para "
            "convertir el gold proxy en referencia. La frase que resume la postura del "
            "proyecto es la que aparece al pie: recuperar la evidencia, no reemplazar al "
            "jurista. Gracias."
        ),
    },
]


# =========================================================================
# SECCION 3: diapositivas
# =========================================================================
def slide_01_portada(prs):
    """Construye la portada del deck.

    Args:
        prs: Presentacion destino.

    Returns:
        La diapositiva creada.
    """
    s = nueva_slide(prs, 1)
    add_kicker(s, "Procesamiento de Lenguaje Natural", 1.55)
    add_titulo(
        s,
        "Juris-PLN. De 23.742 sentencias\na los artículos que las fundamentan",
        size=32,
        y=1.95,
        h=1.5,
    )
    regla = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGEN), Inches(3.62),
                               Inches(1.2), Inches(0.04))
    regla.fill.solid()
    regla.fill.fore_color.rgb = rgb(AZUL)
    regla.line.fill.background()
    _plano(regla)

    add_texto(
        s,
        MARGEN,
        3.92,
        9.6,
        1.0,
        "Entrega 3, Procesamiento de Lenguaje Natural, Maestría en Ingeniería "
        "de Sistemas y Computación, Javeriana",
        size=17,
        color=GRIS_TEXTO,
        line_spacing=1.2,
    )
    add_texto(s, MARGEN, 5.35, 9.0, 0.35, "Carlos Andrés Garnica Salazar", size=17, color=TINTA)
    add_texto(s, MARGEN, 5.78, 9.0, 0.3, "noviembre de 2026", size=14, color=GRIS_TEXTO)
    add_notas(s, GUION[0]["texto"])


def slide_02_problema(prs):
    """Construye la diapositiva del problema y su costo.

    Args:
        prs: Presentacion destino.

    Returns:
        La diapositiva creada.
    """
    s = nueva_slide(prs, 2)
    add_titulo(s, "Ubicar la cita que sostiene la ratio exige leer el documento completo")
    add_vinetas(
        s,
        MARGEN,
        2.12,
        7.35,
        3.2,
        [
            (
                "23.742 sentencias de la Corte Constitucional entre 1992 y 2021: "
                "261 millones de palabras."
            ),
            "La cita que sostiene la ratio vive dentro del texto, sin índice.",
            "Un jurista lee el documento completo para encontrarla, caso por caso.",
        ],
        size=17,
        space_after=18,
    )
    add_stat(s, 8.62, 2.05, 3.9, "23.742", "sentencias en el corpus", size=40)
    add_stat(s, 8.62, 3.55, 3.9, "261 M", "palabras acumuladas", size=40)

    add_tarjeta(s, MARGEN, 5.72, ANCHO, 0.98)
    add_texto(s, MARGEN + 0.28, 5.9, 1.6, 0.72, "48 %", size=40, color=AZUL, bold=True)
    add_texto(
        s,
        MARGEN + 2.0,
        5.99,
        8.9,
        0.62,
        "de los tokens limpios son fórmulas institucionales, no conceptos jurídicos.",
        size=17,
        color=TINTA,
        line_spacing=1.1,
    )
    add_notas(s, GUION[1]["texto"])


def slide_03_pregunta(prs):
    """Construye la diapositiva de la pregunta de trabajo y el producto.

    Args:
        prs: Presentacion destino.

    Returns:
        La diapositiva creada.
    """
    s = nueva_slide(prs, 3)
    add_titulo(s, "La respuesta útil no es un dictamen: es evidencia trazable")
    add_tarjeta(s, MARGEN, 2.1, ANCHO, 1.22)
    add_texto(
        s,
        MARGEN + 0.32,
        2.32,
        ANCHO - 0.64,
        0.8,
        "Dada una providencia, ¿qué artículos de la Constitución la fundamentan "
        "y con qué evidencia textual?",
        size=21,
        color=TINTA,
        italic=True,
        font=SERIF,
        line_spacing=1.15,
    )
    add_vinetas(
        s,
        MARGEN,
        3.75,
        10.6,
        2.0,
        [
            "Devuelve los artículos citados y la frase que los respalda.",
            "Añade citas resueltas y jurisprudencia por artículo.",
            "Lo usan juristas, investigadores y litigantes.",
        ],
        size=17,
        space_after=16,
    )
    barra = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGEN), Inches(5.92),
                               Inches(0.05), Inches(0.5))
    barra.fill.solid()
    barra.fill.fore_color.rgb = rgb(AZUL)
    barra.line.fill.background()
    _plano(barra)
    add_texto(
        s,
        MARGEN + 0.22,
        5.98,
        11.0,
        0.4,
        "No dictamina: ordena la búsqueda y la hace auditable.",
        size=18,
        color=TINTA,
        bold=True,
    )
    add_notas(s, GUION[2]["texto"])


def slide_04_datos(prs):
    """Construye la diapositiva de caracterizacion del corpus con la figura MTLD.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 4)
    add_titulo(s, "El corpus promedia 10.530 palabras limpias y pierde diversidad léxica")
    add_stat(s, MARGEN, 1.98, 3.3, "23.742", "sentencias: T 17.099, C 6.246, SU 397", size=40)
    add_stat(s, MARGEN, 3.2, 3.3, "10.530", "palabras limpias de media; mediana 7.158", size=40)
    add_stat(s, MARGEN, 4.62, 3.3, "56,77", "MTLD en 2021, desde 64,88 en 1992", size=40)

    if FIG_MTLD.exists():
        s.shapes.add_picture(str(FIG_MTLD), Inches(4.42), Inches(2.1), width=Inches(8.05))
        add_texto(
            s,
            4.42,
            5.2,
            8.05,
            0.62,
            "MTLD medio por año: la caída sostenida indica menor riqueza "
            "léxica en el periodo. Fuente: resultados/diacronia.json.",
            size=12,
            color=GRIS_TEXTO,
            line_spacing=1.15,
        )
    else:
        print(f"AVISO: no existe la figura {FIG_MTLD}; se omite en la diapositiva 4.")
        add_texto(s, 4.85, 2.4, 7.4, 1.2, "Figura no disponible.", size=16, color=GRIS_TEXTO)

    add_texto(s, MARGEN, 6.42, 7.0, 0.3, "Fuente: Entrega 2, EDA, 23.742 sentencias.",
              size=11, color=GRIS_TEXTO)
    add_notas(s, GUION[3]["texto"])


def slide_05_medicion(prs):
    """Construye la diapositiva del protocolo de evaluacion.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 5)
    add_titulo(s, "La medición combina 90 casos internos, gold humano y corpus posterior")

    ancho_tarjeta = 3.72
    for i, (etiqueta, cuerpo) in enumerate(
        [
            ("Benchmark interno", "90 casos con etiquetas\ngeneradas por LLM"),
            ("Gold humano B3", "50 casos y 820 pares\nartículo y sentencia"),
            ("C3", "Corpus 2022 a 2026,\nno visto en desarrollo"),
        ]
    ):
        x = MARGEN + i * (ancho_tarjeta + 0.235)
        forma = add_tarjeta(s, x, 2.1, ancho_tarjeta, 1.42)
        texto_en_tarjeta(forma, etiqueta, cuerpo, cuerpo_size=17, margen=0.26)

    add_texto(s, MARGEN, 3.92, ANCHO, 0.3, "MÉTRICAS", size=11, color=AZUL, bold=True,
              espaciado=140)
    add_texto(
        s,
        MARGEN,
        4.25,
        ANCHO,
        0.5,
        "nDCG@10, F1@5 y MRR, con IC 95 % por bootstrap pareado.",
        size=19,
        color=TINTA,
        line_spacing=1.1,
    )
    add_tarjeta(s, MARGEN, 5.3, ANCHO, 1.0)
    add_texto(s, MARGEN + 0.28, 5.5, 11.0, 0.62,
              "Gate de adopción ADR-16: mejora significativa frente a la línea base "
              "y consistencia entre conjuntos.",
              size=15, color=GRIS_TEXTO, line_spacing=1.15)
    add_notas(s, GUION[4]["texto"])


def slide_06_resultado(prs):
    """Construye la diapositiva de resultados de recuperacion con barras.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 6)
    add_titulo(s, "El sistema sube de 0,6013 a 0,6907 y llega a 0,7852 con gold humano")

    filas = [
        ("BM25 + hubs", 0.6013, "0,6013"),
        ("v2.2 (B2 + B1)", 0.6907, "0,6907"),
        ("v2.2 + gold humano", 0.7852, "0,7852"),
    ]
    x_pista = 3.62
    ancho_pista = 5.9
    for i, (etiqueta, valor, texto) in enumerate(filas):
        y = 2.16 + i * 0.78
        add_texto(s, MARGEN, y + 0.03, 2.6, 0.34, etiqueta, size=15, color=GRIS_TEXTO)
        pista = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x_pista), Inches(y),
                                   Inches(ancho_pista), Inches(0.34))
        pista.fill.solid()
        pista.fill.fore_color.rgb = rgb(TARJETA)
        pista.line.fill.background()
        _plano(pista)
        barra = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x_pista), Inches(y),
                                   Inches(ancho_pista * valor), Inches(0.34))
        barra.fill.solid()
        barra.fill.fore_color.rgb = rgb(AZUL)
        barra.line.fill.background()
        _plano(barra)
        add_texto(s, 9.72, y - 0.02, 1.5, 0.4, texto, size=18, color=TINTA, bold=True)

    linea = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGEN), Inches(4.62),
                               Inches(ANCHO), Inches(0.012))
    linea.fill.solid()
    linea.fill.fore_color.rgb = rgb("DDE3EA")
    linea.line.fill.background()
    _plano(linea)

    add_texto(s, MARGEN, 4.85, 2.3, 0.8, "+0,067", size=42, color=AZUL, bold=True)
    add_texto(
        s,
        2.95,
        4.98,
        5.3,
        0.85,
        "puntos de nDCG@10 pareado frente a la línea base; IC 95 % [0,042; 0,089].",
        size=15,
        color=GRIS_TEXTO,
        line_spacing=1.2,
    )
    add_texto(s, 8.75, 4.85, 3.7, 0.3, "C3 · 2022 A 2026", size=11, color=AZUL,
              bold=True, espaciado=140)
    add_texto(s, 8.75, 5.12, 3.7, 0.74, "0,7472", size=36, color=TINTA, bold=True)
    add_texto(s, 8.75, 5.82, 3.7, 0.3, "sin deriva temporal significativa", size=14,
              color=GRIS_TEXTO)
    add_notas(s, GUION[5]["texto"])


def slide_07_negativos(prs):
    """Construye la diapositiva de resultados negativos.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 7)
    add_titulo(s, "Cuatro vías se descartaron con datos, no por intuición")

    tarjetas = [
        ("C2-LoRA", "Perdió 0,198 puntos de nDCG@10 en consultas: el ajuste no generalizó."),
        ("B2, segunda época", "Sumó 0,0013: dentro del ruido, sin significancia."),
        ("Zero-shot, 200 casos", "Con etiquetas de LLM no superó la línea base."),
        ("M4, clasificación", "Macro-F1 0,2870 con 31,81 % de ruido de etiqueta."),
    ]
    for i, (etiqueta, cuerpo) in enumerate(tarjetas):
        x = MARGEN + (i % 2) * 5.83
        y = 2.08 + (i // 2) * 1.45
        forma = add_tarjeta(s, x, y, 5.65, 1.25)
        texto_en_tarjeta(forma, etiqueta, cuerpo, cuerpo_size=16, margen=0.26)

    add_texto(
        s,
        MARGEN,
        5.3,
        11.2,
        0.4,
        "Por eso el área de clasificación queda como exploratoria (ADR-19).",
        size=16,
        color=TINTA,
        bold=True,
        line_spacing=1.15,
    )
    add_notas(s, GUION[6]["texto"])


def slide_08_producto(prs):
    """Construye la diapositiva de arquitectura del producto y guardrails.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 8)
    add_titulo(s, "Tres procesos locales sostienen el producto y sus guardrails")

    cajas = [
        (MARGEN, 3.35, "Streamlit", "8502, interfaz"),
        (4.92, 3.35, "FastAPI", "8123, contratos y guardrails"),
        (8.99, 3.49, "Núcleo validado", "recuperación, rerank y ollama"),
    ]
    for x, w, titulo, sub in cajas:
        forma = add_tarjeta(s, x, 2.05, w, 1.15)
        texto_en_tarjeta(forma, None, titulo, cuerpo_size=18, margen=0.26)
        add_texto(s, x + 0.26, 2.78, w - 0.4, 0.3, sub, size=13, color=GRIS_TEXTO)

    add_flecha(s, MARGEN + 3.42, 2.62, 4.88, 2.62)
    add_flecha(s, 8.3, 2.62, 8.95, 2.62)

    add_texto(s, MARGEN, 3.6, 2.0, 0.3, "ENDPOINTS", size=11, color=AZUL, bold=True,
              espaciado=140)
    add_texto(
        s,
        MARGEN,
        3.92,
        ANCHO,
        0.4,
        "/v1/link   ·   /v1/evidencia   ·   /v1/article/{n}/jurisprudencia   ·   /v1/ask",
        size=14,
        color=GRIS_TEXTO,
    )

    add_texto(s, MARGEN, 4.55, 6.0, 0.3, "GUARDRAILS EN TRES CAPAS", size=11, color=AZUL,
              bold=True, espaciado=140)
    capas = [
        ("01", "Router determinista: clasifica la intención antes del modelo."),
        ("02", "Agente con tool calling: solo consulta herramientas del núcleo."),
        ("03", "Verificación de grounding: sin cita en el texto, no hay respuesta."),
    ]
    for i, (num, texto) in enumerate(capas):
        y = 4.92 + i * 0.55
        add_texto(s, MARGEN, y, 0.55, 0.35, num, size=13, color=AZUL, bold=True)
        add_texto(s, MARGEN + 0.58, y - 0.02, 10.9, 0.4, texto, size=15, color=GRIS_TEXTO)
    add_notas(s, GUION[7]["texto"])


def slide_09_demo(prs):
    """Construye la diapositiva del recorrido de la demo.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 9)
    add_titulo(s, "Cuatro pantallas recorren el flujo completo sobre una sentencia real")

    pantallas = [
        ("P1 · Sentencia a Constitución", "Artículos fundamentados con la cita textual de respaldo."),
        ("P2 · Artículo a jurisprudencia", "El artículo 86 aparece en 16.405 sentencias."),
        ("P3 · Evaluación con procedencia", "Cada resultado muestra su origen y su puntaje."),
        ("P4 · Ficha PDF y similares", "Descarga de la ficha y de sentencias cercanas."),
    ]
    for i, (etiqueta, cuerpo) in enumerate(pantallas):
        x = MARGEN + (i % 2) * 5.83
        y = 2.05 + (i // 2) * 1.45
        forma = add_tarjeta(s, x, y, 5.65, 1.25)
        texto_en_tarjeta(forma, etiqueta, cuerpo, cuerpo_size=16, margen=0.26)

    add_tarjeta(s, MARGEN, 5.42, ANCHO, 0.95)
    add_texto(s, MARGEN + 0.28, 5.6, 1.5, 0.28, "CASO REAL", size=11, color=AZUL,
              bold=True, espaciado=140)
    add_texto(
        s,
        MARGEN + 1.55,
        5.62,
        9.6,
        0.5,
        "id 2195, T-293 de 1997: 6 artículos extraídos y 10 recuperados.",
        size=16,
        color=TINTA,
        line_spacing=1.1,
    )
    add_notas(s, GUION[8]["texto"])


def slide_10_agente(prs):
    """Construye la diapositiva de metricas del agente sobre el gold set.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 10)
    add_titulo(s, "Con 100 prompts el ruteo acierta el 90 % y el grounding llega al 100 %")

    metricas = [
        ("90 %", "ruteo exacto"),
        ("100 %", "ruta jurídica"),
        ("100 %", "saludos no rechazados"),
        ("100 %", "rechazo fuera de alcance"),
        ("100 %", "tool-call correcto"),
        ("100 %", "grounding verificado"),
        ("13,4 s", "latencia p50"),
        ("60,8 s", "latencia p95"),
    ]
    ancho = 2.78
    for i, (valor, etiqueta) in enumerate(metricas):
        x = MARGEN + (i % 4) * (ancho + 0.17)
        y = 2.35 + (i // 4) * 1.52
        forma = add_tarjeta(s, x, y, ancho, 1.33)
        tf = forma.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = Inches(0.22)
        tf.margin_right = Inches(0.16)
        tf.margin_top = Inches(0.16)
        tf.margin_bottom = Inches(0.08)
        p = tf.paragraphs[0]
        p.line_spacing = 1.0
        r = p.add_run()
        r.text = valor
        r.font.size = Pt(36)
        r.font.bold = True
        r.font.name = SANS
        r.font.color.rgb = rgb(TINTA)
        p2 = tf.add_paragraph()
        p2.line_spacing = 1.1
        p2.space_before = Pt(4)
        r2 = p2.add_run()
        r2.text = etiqueta
        r2.font.size = Pt(13)
        r2.font.name = SANS
        r2.font.color.rgb = rgb(GRIS_TEXTO)

    add_texto(
        s,
        MARGEN,
        5.62,
        ANCHO,
        0.6,
        "Gold set de 100 prompts. El p95 de 60,8 s define el presupuesto de espera de la demo.",
        size=15,
        color=GRIS_TEXTO,
    )
    add_notas(s, GUION[9]["texto"])


def slide_11_limites(prs):
    """Construye la diapositiva de limites y etica.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 11)
    add_titulo(s, "Los límites están declarados y el clasificador queda fuera del producto")
    add_vinetas(
        s,
        MARGEN,
        2.12,
        11.2,
        3.1,
        [
            "El corpus llega a 2021; después solo se mide la deriva.",
            "El clasificador M4 no es fiable y no se ofrece al usuario.",
            "El gold humano se anotó con apoyo de LLM, sin expertos legales.",
            "Datos públicos y sin PII; el modelo local puede degradar.",
        ],
        size=18,
        space_after=17,
    )
    add_tarjeta(s, MARGEN, 5.35, ANCHO, 0.95)
    add_texto(
        s,
        MARGEN + 0.28,
        5.55,
        10.9,
        0.6,
        "Fallback determinista: si el modelo falla, la respuesta conserva la cita.",
        size=16,
        color=TINTA,
        line_spacing=1.1,
    )
    add_notas(s, GUION[10]["texto"])


def slide_12_cierre(prs):
    """Construye la diapositiva de cierre con el trabajo inmediato.

    Args:
        prs: Presentacion destino.
    """
    s = nueva_slide(prs, 12)
    add_kicker(s, "Cierre", 0.62)
    add_titulo(s, "El producto funciona y es auditable; falta la validación experta", y=0.95)

    pasos = [
        ("01", "C1: 150 casos con anotación controlada."),
        ("02", "Despliegue externo con límites de uso declarados."),
        ("03", "Validación con expertos legales."),
    ]
    for i, (num, texto) in enumerate(pasos):
        y = 2.45 + i * 0.62
        add_texto(s, MARGEN, y, 0.6, 0.35, num, size=13, color=AZUL, bold=True)
        add_texto(s, MARGEN + 0.62, y - 0.03, 9.5, 0.4, texto, size=18, color=TINTA)

    add_texto(
        s,
        MARGEN,
        4.55,
        ANCHO,
        0.35,
        "23.742 sentencias · 0,6907 de nDCG@10 · grounding 100 %",
        size=13,
        color=GRIS_TEXTO,
        espaciado=40,
    )
    add_texto(
        s,
        MARGEN,
        5.15,
        ANCHO,
        0.9,
        "Recuperar la evidencia, no reemplazar al jurista.",
        size=28,
        color=AZUL,
        italic=True,
        font=SERIF,
        line_spacing=1.05,
    )
    add_notas(s, GUION[11]["texto"])


# =========================================================================
# SECCION 4: generacion y guion
# =========================================================================
def escribir_guion(ruta: Path) -> None:
    """Escribe el guion de la presentacion en Markdown.

    Args:
        ruta: Archivo de salida.
    """
    lineas = [
        "# Guion de la presentación (Entrega 3)",
        "",
        "Duración estimada: 60 a 90 segundos por diapositiva.",
        "Cada bloque indica la cifra que debe decirse en voz alta.",
        "",
    ]
    for i, bloque in enumerate(GUION, 1):
        lineas.append(f"## {i}. {bloque['titulo']}")
        lineas.append("")
        lineas.append(f"*Cifra en voz alta: {bloque['voz']}.*")
        lineas.append("")
        lineas.append(bloque["texto"])
        lineas.append("")
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    """Genera el PPTX, el guion en Markdown y comprueba el resultado."""
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    prs.core_properties.title = "Juris-PLN, Entrega 3"
    prs.core_properties.author = "Carlos Andrés Garnica Salazar"
    prs.core_properties.subject = "Procesamiento de Lenguaje Natural, Javeriana"

    constructores = [
        slide_01_portada,
        slide_02_problema,
        slide_03_pregunta,
        slide_04_datos,
        slide_05_medicion,
        slide_06_resultado,
        slide_07_negativos,
        slide_08_producto,
        slide_09_demo,
        slide_10_agente,
        slide_11_limites,
        slide_12_cierre,
    ]
    for constructor in constructores:
        constructor(prs)

    prs.save(str(OUT_PPTX))
    escribir_guion(OUT_GUION)

    print(f"Diapositivas: {len(prs.slides)}")
    print(f"PPTX: {OUT_PPTX} ({OUT_PPTX.stat().st_size / 1024:.0f} KB)")
    print(f"Guion: {OUT_GUION} ({OUT_GUION.stat().st_size / 1024:.0f} KB)")
    print(f"Figura incluida: {FIG_MTLD.exists()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - main reporta cualquier fallo de generacion
        traceback.print_exc()
        sys.exit(1)
