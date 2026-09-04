# %% [markdown]
"""# Sec 01 — El corpus: morfologia del documento y normalizacion

## 1. Que es un documento en cada corpus

**Libros** (Informe Final CEV, 10 tomos PDF, ~9.000 paginas): el documento es
el **segmento**, construido con la regla ADR-001:

- Limite de segmento = seccion del TOC de nivel 1-2 de cada tomo; el front y
  back matter (portadas, glosario, bibliografia) se descartan.
- **Fusion**: las secciones de menos de **300 palabras** se fusionan con la
  siguiente (o con la anterior si es la ultima), para evitar segmentos
  minusculos.
- **Techo**: las secciones de mas de **4.000 palabras** se parten en
  fragmentos de hasta ~3.500 palabras, cortando en limites de pagina.
- **Id**: `TOMO-p<pagina_inicio>-<titulo-slug>`
  (ej. `HALLAZGOS-p129-las-masacres`).
- Resultado: **917 segmentos / 2.271.493 palabras** (extraccion sin solapes,
  sin huecos y con cobertura de pagina del 100 %; mapeo TOC 91-100 % por tomo,
  verificado en `resultados/extraccion_libros_report.json`).

**Entrevistas** (CEV, archivo unico JSON): el documento es la
**transcripcion completa** de la entrevista (id = nombre del PDF original,
ej. `5eb4e7bfda6da3b502da3589.pdf`; hasta ~850.000 caracteres). Son 2.486
transcripciones de victimas, testigos y excombatientes, con las marcas de
sistema de la transcripcion (etiquetas de turno TEST/ENT, codigos de pagina,
anotaciones entre corchetes y tokens de anonimizacion).

La celda siguiente carga la tabla por tomo desde
`resultados/extraccion_libros_report.json` (reporte de la Fase 3).
"""


# %%
"""Imports, rutas (respaldo standalone) y tabla por tomo de la Fase 3."""

import json

import pandas as pd

# Rutas de sec00. En el notebook ya estan definidas por sec00_setup y este
# bloque no las pisa; el respaldo solo aplica si se ejecuta la seccion como
# script standalone (misma convencion de autocontencion que el TALLER-2).
if "RUTA_RAIZ" not in globals():
    import os as _os
    from pathlib import Path as _Path

    _candidato = _Path(_os.getcwd()).resolve()
    while _candidato != _candidato.parent:
        if (_candidato / "data" / "corpus" / "segmentos_libros.jsonl").exists():
            break
        _candidato = _candidato.parent
    RUTA_RAIZ = _candidato
    RUTA_CORPUS = RUTA_RAIZ / "data" / "corpus"
    RUTA_RESULTADOS = RUTA_RAIZ / "resultados"
    RUTA_SCRIPTS = RUTA_RAIZ / "scripts"

RUTA_RAW = RUTA_RAIZ / "data" / "raw"

# Orden canonico de los 10 tomos (plan/PLAN.md, Fase 3).
ORDEN_TOMOS = (
    "HALLAZGOS",
    "CONVOCATORIA",
    "NARRATIVA",
    "SUFRIR",
    "TESTIMONIAL",
    "CUERPO",
    "NNA",
    "VIOLACIONES",
    "ETNICO",
    "FUERA",
)
reporte_extraccion = json.loads(
    (RUTA_RESULTADOS / "extraccion_libros_report.json").read_text(encoding="utf-8")
)

filas_tomos = []
for tomo in ORDEN_TOMOS:
    datos = reporte_extraccion["tomos"][tomo]
    filas_tomos.append(
        {
            "tomo": tomo,
            "segmentos": datos["n_segmentos"],
            "palabras": f"{datos['palabras_totales']:,}",
            "media_palabras": f"{datos['palabras_media']:,.0f}",
            "min_palabras": datos["palabras_min"],
            "max_palabras": datos["palabras_max"],
        }
    )
tabla_tomos = pd.DataFrame(filas_tomos)
print("Segmentos de libro por tomo (Fase 3, extraccion desde PDF):")
print(tabla_tomos.to_string(index=False))
n_segmentos_total = sum(
    reporte_extraccion["tomos"][tomo]["n_segmentos"] for tomo in ORDEN_TOMOS
)
print(
    f"TOTAL: {n_segmentos_total:,} segmentos | "
    f"entrevistas (coleccion de queries): 2.486 transcripciones completas"
)


# %%
"""Normalizacion de entrevistas: ejemplo real antes/despues (1 registro).

Se lee UN solo registro: la primera entrevista del archivo crudo (JSON array
de ~230 MB; se decodifica solo su primer elemento) y su version normalizada
(primera linea del JSONL). Ambos comparten id_doc.
"""


def decodificar_primer_registro_lista_json(ruta: str) -> dict:
    """Decodifica solo el primer elemento de un JSON array de gran tamano.

    Lee el archivo por fragmentos con crecimiento exponencial y usa
    `json.JSONDecoder.raw_decode`; el costo es O(tamano del primer registro)
    y no O(archivo completo): aqui el crudo pesa ~230 MB y solo interesa una
    entrevista de ejemplo.

    Args:
        ruta: Ruta al archivo JSON que contiene un array de objetos.

    Returns:
        Primer elemento del array (dict de la entrevista cruda).

    Raises:
        ValueError: si el archivo no contiene un primer registro completo.
    """
    decodificador = json.JSONDecoder()
    with open(ruta, encoding="utf-8") as fh:
        fragmento = ""
        tamano = 1_000_000
        while True:
            trozo = fh.read(tamano)
            if not trozo:
                raise ValueError(f"JSON sin primer registro completo: {ruta}")
            fragmento += trozo
            inicio = fragmento.find("[")
            if inicio >= 0:
                idx = inicio + 1
                while idx < len(fragmento) and fragmento[idx] in " \t\n\r":
                    idx += 1  # raw_decode no omite espacios en blanco
                try:
                    registro, _ = decodificador.raw_decode(fragmento, idx=idx)
                    return registro
                except json.JSONDecodeError:
                    pass
            tamano *= 2


ruta_json_crudo = RUTA_RAW / "entrevistas_all_2023-03-21_14-24_05.json"
ruta_normalizadas = RUTA_CORPUS / "entrevistas_normalizadas.jsonl"
if ruta_json_crudo.exists() and ruta_normalizadas.exists():
    registro_crudo = decodificar_primer_registro_lista_json(str(ruta_json_crudo))
    with ruta_normalizadas.open(encoding="utf-8") as fh:
        registro_normalizado = json.loads(fh.readline())
else:
    # Respaldo estatico del mismo ejemplo real (doc 5eb4e7bfda6da3b502da3589.pdf):
    # el JSON crudo (~230 MB) no se versiona en git, asi que en un entorno sin
    # data/raw la celda sigue siendo demostrativa.
    print(
        "[aviso] JSON crudo de entrevistas no disponible en este entorno; "
        "se muestra un respaldo estatico del ejemplo real (mismo id_doc)."
    )
    registro_crudo = {
        "id_doc": "5eb4e7bfda6da3b502da3589.pdf",
        "text": (
            "TEST: Bajo la apariencia de una democracia formal, en Colombia se "
            "vivia un regimen de terror: servicios de inteligencia estatales "
            "de la ORGANIZACION PUBLICA 1 ---- se entrelazaban con grupos "
            "paramilitares de extrema derecha."
        ),
    }
    registro_normalizado = {
        "id_doc": "5eb4e7bfda6da3b502da3589.pdf",
        "text_normalizado": (
            "Bajo la apariencia de una democracia formal, en Colombia se vivia "
            "un regimen de terror: servicios de inteligencia estatales de la "
            "organizacion_publica se entrelazaban con grupos paramilitares de "
            "extrema derecha."
        ),
    }

id_ejemplo = registro_crudo["id_doc"]
print(
    f"id_doc del ejemplo: {id_ejemplo} | crudo <-> normalizado alineados: "
    f"{registro_normalizado['id_doc'] == id_ejemplo}"
)

lineas_crudas = [linea for linea in registro_crudo["text"].split("\n") if linea.strip()]
print("\n--- ANTES (texto crudo: 3 lineas reales con marcas de sistema) ---")
for linea in lineas_crudas[:3]:
    print(repr(linea[:170]))
texto_normalizado = registro_normalizado["text_normalizado"]
print("\n--- DESPUES (texto normalizado, inicio real del mismo testimonio) ---")
print(texto_normalizado[:400])
print("...")

print("\n--- Un token de anonimizacion, antes y despues ---")
i = registro_crudo["text"].find("ORGANIZACI")
if i >= 0:
    print("crudo      : ...", " ".join(registro_crudo["text"][i - 120 : i + 60].split()), "...")
j = texto_normalizado.find("organizacion_publica")
if j >= 0:
    print("normalizado: ...", " ".join(texto_normalizado[j - 100 : j + 40].split()), "...")


# %%
"""Categorias de anonimizacion y volumenes globales de la normalizacion."""

inventario = json.loads(
    (RUTA_RESULTADOS / "preprocesado" / "inventario_patrones_entrevistas.json")
    .read_text(encoding="utf-8")
)
totales = inventario["totales"]
categorias = pd.DataFrame(
    [
        {"categoria": nombre, "ocurrencias": conteo}
        for nombre, conteo in sorted(
            totales["anonimizados"].items(), key=lambda par: -par[1]
        )
    ]
)
print("Tokens anonimizados mapeados a categorias canonicas (ADR-002):")
print(categorias.to_string(index=False))
n_variantes = sum(len(v) for v in totales["variantes_anonimizados"].values())
print(
    f"Total mapeado: {sum(totales['anonimizados'].values()):,} tokens "
    f"({n_variantes} variantes ortograficas distintas normalizadas)"
)
print(
    f"Chars: {totales['chars_antes']:,} -> {totales['chars_despues']:,} "
    f"({inventario['reduccion_pct']:.2f} %)"
)
print(
    f"Etiquetas de hablante eliminadas: {totales['n_hablantes_linea']:,} | "
    f"guiones: {totales['n_guiones']:,} | anotaciones entre corchetes: "
    f"{totales['n_brackets']:,}"
)


# %% [markdown]
"""## 2. El pipeline de preprocesado (codigo esencial, sin ejecutar spaCy)

El preprocesado completo tiene dos partes (scripts/02_preprocesar.py):

1. **Normalizacion estructural** (regex, sin spaCy): elimina codigos de
   sistema (`001-VI-00008`, lineas `AUDIO`), etiquetas de hablante al inicio
   de linea, guiones largos y anotaciones entre corchetes (inventariadas por
   tipo: interrupciones, emociones, pausas, cortes...), y mapea los tokens de
   anonimizacion a 8 categorias canonicas en minusculas (ADR-002:
   `organizacion_publica`, `organizacion_social`, `lugar`, `familiar`,
   `conocido`, `institucion_educativa`, `partido_politico` y `anonimo`).
   El inventario completo de patrones aplicados (30+ reglas) se documento en
   `resultados/preprocesado/inventario_patrones_entrevistas.json` (versionado)
   y la celda de abajo lee la tabla de categorias desde ese archivo.
2. **Preprocesado linguistico** (spaCy `es_core_news_md`, CPU): tokeniza,
   pasa a minusculas y lematiza con un pipeline unico para libros y
   entrevistas; elimina stopwords de spaCy y las muletillas del habla
   (config/stopwords_habla.txt), conserva los anos 1900-2025 y las categorias
   canonicas como un solo token (ADR-003; `n_process=8`, ~4.1x vs 1 hilo).

**El corpus ya viene procesado** (data/corpus/preprocesado/: 917 + 2.486
documentos en `*_lemas.txt`/`*_tokens.txt` con sus `*_docs.json` alineados),
asi que el notebook NO vuelve a ejecutar spaCy ni reimporta el script de
produccion. La celda siguiente define una **funcion local `normalizar_demo`**,
un subset DEMOSTRATIVO y DOCUMENTADO de la parte 1 (las reglas mas visibles:
etiquetas de hablante al inicio de linea, anotaciones entre corchetes,
guiones >= 3 y el mapeo de anonimizados del ADR-002), y la aplica a las 3
lineas crudas del ejemplo. El corpus real ya fue normalizado con el
normalizador completo de scripts/02_preprocesar.py; esta funcion no es la
produccion, solo muestra el mecanismo sobre un fragmento.
"""


# %%
"""Normalizacion estructural (subset demostrativo) aplicada al ejemplo.

`normalizar_demo` replica el criterio del ADR-002 sobre un SUBSET de reglas
(no el normalizador completo de scripts/02_preprocesar.py, que ademas cubre
codigos de sistema, lineas AUDIO, etiquetas aisladas y colapso de turnos):
elimina etiquetas de hablante al inicio de linea, anotaciones entre
corchetes, guiones separadores (>= 3) y mapea los anonimizados a las 8
categorias canonicas (especificas primero y generico al final), dejando el
resto en minusculas con sus tildes. Se aplica a las 3 primeras lineas crudas
del ejemplo y a una linea dirigida que ejercita las categorias canonicas.
"""
import re

_PATRONES_ANONIMIZADOS: list[tuple[str, re.Pattern[str]]] = []
_e = r"[ \t\n]"
_num = rf"(?:{_e}*[0-9]+)"
_no_minuscula = r"(?![ \t]*[a-záéíóúñ])"
_especificos = [
    ("organizacion_publica", rf"ORGANIZACI[ÓO]N{_e}+P[ÚU]BLICA(?:{_num}|{_no_minuscula})"),
    ("organizacion_social", rf"ORGANIZACI[ÓO]N{_e}+SOCIAL(?:{_num}|{_no_minuscula})"),
    ("institucion_educativa", rf"INSTITUCI[ÓO]N{_e}+EDUCATIVA(?:{_num}|{_no_minuscula})"),
    ("partido_politico", rf"PARTIDO{_e}+POLITICO(?:{_num}|{_no_minuscula})"),
    ("lugar", r"\bLUGAR[ \t\n]*[0-9]+\b"),
    ("familiar", r"\bFAMILIAR[ \t\n]*[0-9]+\b"),
    ("conocido", r"\bCONOCIDO[ \t\n]*[0-9]+\b"),
]
_generico = (
    r"(?<![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])[A-ZÑÁÉÍÓÚ]{2,}[0-9]+"
    r"(?![A-ZÑÁÉÍÓÚa-zñáéíóú0-9])"
)
_PATRONES_ANONIMIZADOS = [
    (categoria, re.compile(patron)) for categoria, patron in _especificos
] + [("anonimo", re.compile(_generico))]


def normalizar_demo(texto: str) -> str:
    """Subset demostrativo de la normalizacion estructural (ADR-002).

    Aplica en orden: anotaciones entre corchetes -> etiquetas de hablante al
    inicio de linea (`TEST:`, `ENT1:`...) -> anonimizados a categorias
    canonicas (especificas primero: organizacion_publica, organizacion_social,
    institucion_educativa, partido_politico, lugar, familiar, conocido; y al
    final el generico `anonimo` para el resto de MAYUSCULAS+digitos) ->
    guiones separadores (>= 3) -> colapso de espacios. Cierra en minusculas
    conservando las tildes. Es un subset DOCUMENTADO con fines didacticos; el
    corpus real se normalizo con scripts/02_preprocesar.py (30+ patrones en
    resultados/preprocesado/inventario_patrones_entrevistas.json).

    Args:
        texto: transcripcion cruda (una o varias lineas).

    Returns:
        Texto normalizado del subset (una linea, minusculas con tildes).
    """
    t = texto.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    t = re.sub(r"\[[^\[\]]*\]", "", t)  # anotaciones entre corchetes
    t = re.sub(r"(?m)^[ \t]*[A-ZÑ]{1,16}[0-9]{0,3}:[ \t]*", "", t)  # hablante
    for categoria, patron in _PATRONES_ANONIMIZADOS:
        t = patron.sub(categoria, t)
    t = re.sub(r"[ \t]*[\-–—]{3,}[ \t]*", " ", t)  # guiones separadores
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


texto_prueba = "\n".join(lineas_crudas[:3])
normalizado_demo = normalizar_demo(texto_prueba)
print("=== normalizar_demo(): subset demostrativo de la parte 1 (regex) ===")
print("Reglas del subset: corchetes, etiquetas de hablante en inicio de linea,")
print("anonimizados ADR-002 (8 categorias), guiones >=3, minusculas con tildes.")
print("(el normalizador completo de scripts/02_preprocesar.py usa 30+ patrones;")
print(" ver inventario_patrones_entrevistas.json en resultados/preprocesado/)")
print("\n=== Subset aplicado a las 3 primeras lineas crudas del ejemplo ===")
print("ANTES :", " | ".join(texto_prueba.splitlines())[:220])
print("DESPUES:", normalizado_demo[:220])

print("\n=== Linea dirigida: las marcas del subset y las categorias canonicas ===")
oracion_demo = (
    "TEST: la ORGANIZACIÓN PÚBLICA 1 ---- y la ORGANIZACION SOCIAL 2 "
    "llegaron a LUGAR7 [emocion] donde vivía FAMILIAR3 con CONOCIDO4; "
    "la INSTITUCION EDUCATIVA5 y el PARTIDO POLITICO6, y luego ANONIMO9."
)
# Las categorias canonicas del ADR-002 quedan SIN el numero del token de
# anonimizacion original (el patron consume 'LUGAR7' -> 'lugar'), igual que
# en el corpus normalizado de produccion.
salida_esperada = (
    "la organizacion_publica y la organizacion_social llegaron a lugar donde "
    "vivía familiar con conocido; la institucion_educativa y el "
    "partido_politico, y luego anonimo."
)
normalizado_verificacion = normalizar_demo(oracion_demo)
print("ANTES :", oracion_demo)
print("DESPUES:", normalizado_verificacion)
print("Mapeo correcto de ORGANIZACIÓN PÚBLICA 1 -> organizacion_publica:",
      "organizacion_publica" in normalizado_verificacion)
print("Coincide con la salida esperada del subset:", normalizado_verificacion == salida_esperada)


# %% [markdown]
"""## 3. Regeneracion opcional (solo si se quiere reconstruir el corpus)

Si alguien quisiera regenerar el corpus desde el material crudo, el camino
recomendado es el CLI del script de produccion (NO reejecutar spaCy dentro
del notebook):

```bash
# Parte 1 + parte 2, libros y entrevistas (todo el pipeline):
/home/carlos/venv/bin/python scripts/02_preprocesar.py
# Equivalente por partes:
/home/carlos/venv/bin/python scripts/02_preprocesar.py normalizar
/home/carlos/venv/bin/python scripts/02_preprocesar.py spacy --corpus libros
/home/carlos/venv/bin/python scripts/02_preprocesar.py spacy --corpus entrevistas
```

La celda siguiente define (con try/except) la instalacion de spaCy y del
modelo `es_core_news_md`, por si en Colab se quisiera activar la regeneracion
desde el propio notebook. El flag `REGENERAR_PREPROCESADO_SPACY` permanece en
False: el corpus ya procesado es la fuente de las secciones 02 y 03.
"""


# %%
"""Rutina de instalacion de spaCy + modelo (solo si se activa el flag)."""
import subprocess
import sys

REGENERAR_PREPROCESADO_SPACY = False  # cambiar a True solo para regenerar


def instalar_spacy_con_modelo() -> None:
    """Instala spaCy y descarga es_core_news_md si no estan disponibles."""
    try:
        import spacy
    except ImportError:
        print("Instalando spaCy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "spacy"])
    import spacy

    try:
        spacy.load("es_core_news_md")
        print("Modelo es_core_news_md disponible.")
    except OSError:
        print("Descargando es_core_news_md...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "es_core_news_md"]
        )


if REGENERAR_PREPROCESADO_SPACY:
    instalar_spacy_con_modelo()
else:
    print(
        "Regeneracion desactivada (flag en False). El corpus preprocesado ya "
        "existe en data/corpus/preprocesado/ y es el insumo de las secciones "
        "02 (EDA) y 03 (IR)."
    )

