"""P0+P1 - gap analysis linguistico POR OPERACION y auditoria de legalidad de herramientas NLP.

## P0: que se ha usado de verdad, operacion por operacion

La pregunta es si existe informacion linguistica ESTRUCTURAL (secuencias tipadas) que las fases
anteriores no explotaron de forma equivalente. La hipotesis no es "POS mejora AGE" ni "NER mejora
AGE": es que una representacion tipada y secuencial expresa morfosintaxis, estilo y estructura de
entidad que la representacion lexica convencional no expresa directamente.

El metodo es el que manda el brief: buscar por OPERACION y metodologia, no por nombre. Se recorre
el codigo ejecutado (233 scripts), se aisla cada uso real de un anotador (imports y uso de
atributos de anotacion) y se comprueba si algun script construye SECUENCIAS de etiquetas. Un
informe no basta: manda el codigo.

## P1: legalidad de las herramientas

Se auditan las fuentes normativas disponibles HOY (reglas especificas de la competencia,
capturadas de la pagina Rules; prohibiciones del profesor confirmadas por Carlos: transformers,
LLMs y embeddings preentrenados o entrenados) y se clasifica cada recurso como
ALLOWED / AMBIGUOUS / PROHIBITED. El anotador `es_core_news_md` cae en zona gris: D-001 lo
permitia de forma explicita, D-026/D-043 lo autorizaron y usaron, y V31 lo dejo en cuarentena.
La clasificacion queda documentada y la resolucion se eleva a Carlos.

No se ejecuta ningun experimento aqui: P2 (POS estructurado) queda gateado a la resolucion.

No usa test, no usa holdout, no genera submission, no toca el candidato adoptado.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-29"
CAPTURA = ROOT / "references" / "age_rules_capture_2026-09-29.md"
GAP_CSV = cfg.REPORTS / "linguistic_enrichment_gap_analysis.csv"
GAP_MD = cfg.REPORTS / "linguistic_enrichment_gap_analysis.md"
LEGAL_MD = cfg.REPORTS / "age_linguistic_legality_audit.md"
REGISTRO_CSV = cfg.REPORTS / "linguistic_annotation_registry.csv"
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_p0p1_gap_y_legalidad.json"
NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
MARCADOR_NB = "FASE - LINGUISTIC STRUCTURED ENRICHMENT"

HASH_MODELO = "889a2fd05d217fa7c1bcffa867af7fc30911bc7b48523f87b5b2525846553ef5"
SCRIPTS_CON_SPACY = {"phase10_dos_vias.py", "v14a_linguistico.py"}
ATRIBUTOS_ANOTACION = r"\.pos_|\.tag_|\.morph|\.dep_|\.lemma_|\.ents\b|doc\.ents|nlp\.pipe"
FIRMAS_SECUENCIA = [r"join\([^\)]*\.(pos_|tag_|lemma_|morph|dep_)",
                    r"\.(pos_|tag_)\.(join|append)",
                    r"ngram[^\n]*(pos_|tag_|morph)",
                    r"(pos_|tag_|morph)[^\n]*ngram"]
FIRMAS_TAGGER_NLTK = [r"pos_tag\(", r"nltk\.tag", r"PerceptronTagger", r"TreebankWordTokenizer"]

TEXTO_REGLAS = """## Competition Rules

Esta competencia es académica, privada y grupal, dirigida únicamente a los estudiantes de la
Maestría en Inteligencia Artificial – Curso de Procesamiento del Lenguaje Natural (NLP).

### Reglas Generales

- Los equipos deberán trabajar en grupos (formados según indicaciones del curso).
- Cada equipo puede enviar múltiples predicciones, pero solo se considerará la mejor puntuación
  en el leaderboard público.
- Está permitido el uso de librerías comunes de NLP y ML, incluyendo: NLTK, Scikit-learn,
  TensorFlow, PyTorch, Pandas, Numpy.
- Se espera un notebook claro y bien documentado, que describa:
  - El procesamiento de los datos.
  - Las estrategias de preprocesamiento y representación de texto.
  - Los modelos y experimentos realizados.
- El dataset de prueba no debe ser usado para entrenamiento.
- El archivo de predicción debe respetar el formato definido en age_submission.csv.

### Evaluación Académica

- La nota de la competencia se calculará de la siguiente manera:
- 70% se evaluará en función del puntaje obtenido en Kaggle (F1 Score, leaderboard privado).
- 30% corresponderá al notebook entregado, valorando claridad, justificación de las técnicas y
  análisis de resultados.

### Escala de Calificación

- La calificación máxima (5.0) se asignará al equipo con mejor desempeño en Kaggle.
- A partir de esa referencia, las notas se irán reduciendo proporcionalmente hacia abajo según el
  ranking obtenido.
- El notebook puede subir o bajar ligeramente la nota final según su calidad."""

METADATOS_CAPTURA = """
# Captura: reglas especificas de la competencia TweetAge_2026_PUJ

- Fuente: pagina Rules de la competicion, https://www.kaggle.com/competitions/tweet-age-2026-puj/rules
- Fecha de captura: 2026-09-29
- Metodo: renderizado de la pagina publica con lector externo (r.jina.ai); la seccion
  "Competition Rules" (reglas especificas del organizador) se renderizo completa. La seccion
  "Kaggle Competition Foundational Rules (Non-editable)" es el texto legal generico de Kaggle y
  no se transcribe aqui.
- Fidelidad: el bloque siguiente es el texto del organizador tal cual se renderizo, con sus
  tildes. No se modifica.
- Observacion: la seccion especifica NO menciona transformers, LLMs ni modelos preentrenados.
  Esas prohibiciones existen a nivel de curso y fueron confirmadas por Carlos el 2026-09-29:
  (1) transformers y LLMs prohibidos por indicacion del profesor; (2) embeddings prohibidos
  expresamente, preentrenados o entrenados desde cero. Registro en DECISIONS.md (D-001, D-075).

---

"""

FILAS_GAP = [
    {"id": "G-POS-COUNT", "componente": "POS (conteos)",
     "forma_probada": "26 features de conteo de etiqueta POS con es_core_news_md",
     "evidencia_previa": "PH10/D-026: MLP + POS 0,5944 vs base 0,5976 (-0,0032); ambas vias -0,0013",
     "fichero_clave": "scripts/phase10_dos_vias.py",
     "status": "CLOSED", "por_que": "medido con autorizacion explicita; resultado negativo",
     "siguiente_paso": "ninguno"},
    {"id": "G-POS-RATIO", "componente": "POS (ratios por grupo)",
     "forma_probada": "ratios determinante/conjuncion/adjetivo/adverbio y 33 ratios morfo-sintacticos mas",
     "evidencia_previa": "V14a: 0,2409 aislado (suelo 0,1667); V14b: fusion -0,0015, 1/5 folds",
     "fichero_clave": "scripts/v14a_linguistico.py",
     "status": "CLOSED", "por_que": "medido; el bloque tiene senal aislada pero la fusion no la convierte",
     "siguiente_paso": "la forma secuencial es otro mecanismo: ver G-POS-SEQ"},
    {"id": "G-POS-SEQ", "componente": "POS (secuencia / n-grams)",
     "forma_probada": "string de etiquetas POS por tweet -> n-grams sparse como canal tipado",
     "evidencia_previa": "NUNCA construido; los 2 unicos scripts con spaCy hacen conteos y ratios; el audit de V13 cerro 'n-gramas de POS' citando D-026 (conteos): cierre por nombre, no por operacion",
     "fichero_clave": "scripts/phase10_dos_vias.py, scripts/v14a_linguistico.py (unicos con anotacion)",
     "status": "GENUINELY_NEW", "por_que": "0 scripts con firmas de construccion de secuencias en 233 ficheros",
     "siguiente_paso": "P2, gateado a resolucion de legalidad y con pre-registro"},
    {"id": "G-LEMA-AGG", "componente": "lemas (diversidad)",
     "forma_probada": "lema_distintos_por_token, hapax por lema, forma_distintas_por_lemma",
     "evidencia_previa": "V14a/b: mismas 37 variables; fusion -0,0015",
     "fichero_clave": "scripts/v14a_linguistico.py",
     "status": "CLOSED", "por_que": "agregados de diversidad; medidos",
     "siguiente_paso": "los tokens lemma_POS son otra representacion: ver G-LEMA-POS"},
    {"id": "G-LEMA-POS", "componente": "lemma + POS",
     "forma_probada": "tokens lemma_ETIQUETA (querer_VERB, casa_NOUN) y bigrams si la dimension es razonable",
     "evidencia_previa": "NUNCA construido",
     "fichero_clave": "-",
     "status": "GENUINELY_NEW", "por_que": "distinto de la diversidad de lemas y de los conteos POS",
     "siguiente_paso": "P3, condicionado"},
    {"id": "G-MORF-SUFIJO", "componente": "morfologia por sufijos",
     "forma_probada": "rasgos por sufijo escritos a mano",
     "evidencia_previa": "audit V13: redundante con char_wb(3-5)",
     "fichero_clave": "reports/v13_gap_audit.csv",
     "status": "CLOSED", "por_que": "redundancia medida en el audit de V13; D-043 lo distingue de la morfologia del analizador",
     "siguiente_paso": "ninguno"},
    {"id": "G-MORF-RATIO", "componente": "morfologia (ratios)",
     "forma_probada": "10 ratios de tiempo/modo/persona + 6 de genero/numero/diminutivo/aumentativo",
     "evidencia_previa": "V14a/b: fusion -0,0015, 1/5 folds",
     "fichero_clave": "scripts/v14a_linguistico.py",
     "status": "CLOSED", "por_que": "medido",
     "siguiente_paso": "la forma de tokens/secuencias: ver G-MORF-TOK"},
    {"id": "G-MORF-TOK", "componente": "morfologia (tokens y secuencias cortas)",
     "forma_probada": "tokens tipo Tense=Past|1P encadenados -> secuencias cortas",
     "evidencia_previa": "NUNCA construido",
     "fichero_clave": "-",
     "status": "GENUINELY_NEW", "por_que": "D-043 ya distinguia ratios de la informacion morfologica en otra forma; la secuencial no existe",
     "siguiente_paso": "P4, condicionado"},
    {"id": "G-DEP-AGG", "componente": "dependencias (agregados)",
     "forma_probada": "profundidad media, ratios de coordinacion/subordinacion, hijos por token, pasiva",
     "evidencia_previa": "V14a/b: fusion -0,0015",
     "fichero_clave": "scripts/v14a_linguistico.py",
     "status": "CLOSED", "por_que": "medido",
     "siguiente_paso": "los patrones tipados son otro mecanismo: ver G-DEP-PAT"},
    {"id": "G-DEP-PAT", "componente": "patrones de dependencia",
     "forma_probada": "patrones compactos tipados (subj->VERB, DET->NOUN) o sus secuencias",
     "evidencia_previa": "NUNCA construido; P19 (arboles SFL de conjunciones) cerro por azar una via distinta y sin POS",
     "fichero_clave": "scripts/v13_p19_conjunciones.py",
     "status": "GENUINELY_NEW", "por_que": "no existe operacion de patrones de dependencia",
     "siguiente_paso": "P5, condicionado"},
    {"id": "G-NER-AGG", "componente": "entidades (conteos/ratios)",
     "forma_probada": "ent_per/org/loc/misc ratios, entidades distintas, por documento",
     "evidencia_previa": "V14a/b: fusion -0,0015",
     "fichero_clave": "scripts/v14a_linguistico.py",
     "status": "CLOSED", "por_que": "medido",
     "siguiente_paso": "las secuencias de TIPO: ver G-NER-TIPO"},
    {"id": "G-NER-TIPO", "componente": "NER estructural (secuencia de tipos)",
     "forma_probada": "PERSON VERB LOCATION... como tokens de tipo, sin identidad de entidad",
     "evidencia_previa": "NUNCA construido",
     "fichero_clave": "-",
     "status": "GENUINELY_NEW", "por_que": "los conteos/ratios no conservan orden ni estructura; la identidad no se usara (reduce topic leakage)",
     "siguiente_paso": "P6, condicionado"},
    {"id": "G-NER-POS", "componente": "NER + POS combinado",
     "forma_probada": "entidad tipada + contexto POS (PERSON VERB, LOCATION ADP NOUN)",
     "evidencia_previa": "NUNCA construido",
     "fichero_clave": "-",
     "status": "GENUINELY_NEW", "por_que": "combinacion estructural, no dos listas independientes",
     "siguiente_paso": "solo si P6 justifica"},
    {"id": "G-TEXTO-TIPADO", "componente": "texto tipado",
     "forma_probada": "canal paralelo: texto original + texto tipado (PERSON compra_VERB coches_NOUN)",
     "evidencia_previa": "NUNCA construido",
     "fichero_clave": "-",
     "status": "GENUINELY_NEW", "por_que": "no existe ningun canal que conserve el texto y lo tipe a la vez",
     "siguiente_paso": "P7, condicionado"},
    {"id": "G-FUNCWORD", "componente": "palabras funcionales",
     "forma_probada": "lexicones propios (26 pronombres, 36 conjunciones, 23 modalidad) y grupos de composicion en regex",
     "evidencia_previa": "V13-P17A 0,1918; V13-P19 y V13-P20 por debajo del azar; V31-03 composicion dentro de canales regex",
     "fichero_clave": "scripts/v13_p17a_pronombres.py",
     "status": "CLOSED", "por_que": "cerrado con medicion en V13; su forma secuencial queda cubierta por G-POS-SEQ",
     "siguiente_paso": "ninguno"},
    {"id": "G-REGEX-CH", "componente": "canales linguisticos regex (V31)",
     "forma_probada": "ortografia, composicion, discurso, posicion, url, forma de palabra (listas a mano, sin tagger)",
     "evidencia_previa": "V31-03: mejor canal +0,006820 (5/5) en el banco; V31-05 +0,0000642; V31-06 +0,0002948 sobre el control barajado",
     "fichero_clave": "scripts/v31_03_canales_linguisticos.py",
     "status": "CLOSED (integracion)", "por_que": "la senal en el banco es real; la integracion en el meta-espacio no la captura (control de atribucion)",
     "siguiente_paso": "ninguna sin mecanismo nuevo"},
    {"id": "G-TIPO-SEC", "componente": "secuencia de tipos superficiales de token",
     "forma_probada": "vista estructural sin contenido: secuencia de @ / # / URL / N / W (PH21); alfabeto ampliado (PH22)",
     "evidencia_previa": "PH21: F1 aislado 0,2285, rescate 18,17%, +0,0018 (4/5), holdout +0,0031, NO adoptada; PH22: alfabeto mas rico no mejora -> via cerrada",
     "fichero_clave": "scripts/phase21_vista_estructural.py",
     "status": "CLOSED (bajo gate)", "por_que": "criterio pre-registrado +0,003 no alcanzado; no es POS (tipos superficiales, no gramatica)",
     "siguiente_paso": "no puede usarse como permiso ni como duplicado de G-POS-SEQ"},
    {"id": "G-RESID", "componente": "correcciones en score-space del residuo linguistico",
     "forma_probada": "residuo global (P33), local pairwise (P34), pendiente sobre el margen (G12)",
     "evidencia_previa": "P33 -0,0002152; P34 -0,0002039; G12 +0,0003018 sin superar a la constante",
     "fichero_clave": "scripts/p33_01_residualizacion.py",
     "status": "CLOSED", "por_que": "tres vias con control de atribucion; el control gana o empata",
     "siguiente_paso": "ninguno"},
    {"id": "G-ENTITY-EV", "componente": "evidencia supervisada por entidad",
     "forma_probada": "P(clase|mencion/hashtag/RT) in-fold",
     "evidencia_previa": "V13-P1C: completo 0,3997 | sin cuentas 0,3410 | sin cuentas ni hashtags 0,2992",
     "fichero_clave": "scripts/v13_p1c_identidades.py",
     "status": "CLOSED", "por_que": "via supervisada por token/entidad, cerrada con medicion",
     "siguiente_paso": "ninguno"},
    {"id": "G-ESTILO", "componente": "estilo / textometria",
     "forma_probada": "n_chars, cttr, entropia lexica, ablacion de estilo",
     "evidencia_previa": "E-002/E-010/E-011: estilo aislado ~0,2793 y absorcion por char n-grams",
     "fichero_clave": "scripts/experimento_e001_wordchar_linearsvc.py",
     "status": "CLOSED", "por_que": "medido en la etapa E; redundante con forma",
     "siguiente_paso": "ninguno"},
    {"id": "P-TRANSFORMER", "componente": "transformers",
     "forma_probada": "-", "evidencia_previa": "prohibidos por indicacion del profesor (confirmado por Carlos, 2026-09-29); D-001",
     "fichero_clave": "-", "status": "PROHIBITED", "por_que": "regla explicita",
     "siguiente_paso": "no usar"},
    {"id": "P-LLM", "componente": "LLMs",
     "forma_probada": "-", "evidencia_previa": "prohibidos por indicacion del profesor (confirmado por Carlos, 2026-09-29); D-001",
     "fichero_clave": "-", "status": "PROHIBITED", "por_que": "regla explicita",
     "siguiente_paso": "no usar"},
    {"id": "P-EMB", "componente": "embeddings / vectorizacion",
     "forma_probada": "-", "evidencia_previa": "prohibidos expresamente, preentrenados o entrenados desde cero (Carlos, 2026-09-29); D-001",
     "fichero_clave": "-", "status": "PROHIBITED", "por_que": "regla explicita",
     "siguiente_paso": "no usar"},
    {"id": "P-EXT", "componente": "corpus y etiquetas externas",
     "forma_probada": "-", "evidencia_previa": "prohibido por el brief (secciones 8 y 25)",
     "fichero_clave": "-", "status": "PROHIBITED", "por_que": "integridad academica y reglas",
     "siguiente_paso": "no usar"},
    {"id": "G-ANOTADOR", "componente": "anotador preentrenado no-transformer (spaCy es_core_news_md)",
     "forma_probada": "herramienta de anotacion determinista para producir secuencias tipadas",
     "evidencia_previa": "D-001 lo permitia explicitamente ('POS tagger no-transformer de spaCy'); D-026/D-043 lo autorizaron y usaron con procedencia; V31 lo dejo en cuarentena por ambiguedad",
     "fichero_clave": "scripts/v14a_linguistico.py, reports/v31_compliance.md",
     "status": "AMBIGUOUS", "por_que": "conflicto interno documentado + las prohibiciones del profesor no cubren anotadores; su tok2vec usa vectores estaticos internamente (include_static_vectors=true; se usarian solo las etiquetas)",
     "siguiente_paso": "resolucion pendiente (opciones A/B/C en D-075); sin resolucion, P2 no se ejecuta"},
]

FILAS_REGISTRO = [
    {"recurso": "spacy", "tipo": "libreria", "version": "3.8.13", "idioma": "n/a",
     "licencia": "MIT", "fecha_alta": "2026-08-19", "hash_sha256": "n/a",
     "clasificacion": "ALLOWED",
     "evidencia": "reglas especificas: librerias comunes de NLP permitidas; uso historico autorizado (V14)",
     "condiciones": "registro en requirements.txt"},
    {"recurso": "es_core_news_md", "tipo": "anotador preentrenado no-transformer",
     "version": "3.8.0", "idioma": "es", "licencia": "GPL-3.0", "fecha_alta": "2026-08-19",
     "hash_sha256": HASH_MODELO,
     "clasificacion": "AMBIGUOUS",
     "evidencia": "D-001 permitido explicito; D-026/D-043 autorizaron y usaron; V31 cuarentena; no es transformer/LLM/embedding; include_static_vectors=true (solo se usarian etiquetas)",
     "condiciones": "NO usable para una submission mientras AMBIGUOUS; resolucion en D-075"},
    {"recurso": "NLTK", "tipo": "libreria", "version": "3.9.4", "idioma": "n/a",
     "licencia": "Apache-2.0", "fecha_alta": "2026-08-19", "hash_sha256": "n/a",
     "clasificacion": "ALLOWED (libreria)",
     "evidencia": "las reglas la listan por nombre",
     "condiciones": "si se usara un tagger NLTK preentrenado regiria la misma resolucion que es_core_news_md"},
    {"recurso": "transformers", "tipo": "libreria instalada, sin uso",
     "version": "5.16.1", "idioma": "n/a", "licencia": "Apache-2.0", "fecha_alta": "2026-08-19",
     "hash_sha256": "n/a", "clasificacion": "PROHIBITED (tecnica)",
     "evidencia": "prohibido por el profesor; D-038: instalado no es usado; uso en el proyecto = 0",
     "condiciones": "no usar"},
    {"recurso": "sentence-transformers", "tipo": "libreria instalada, sin uso",
     "version": "6.0.1", "idioma": "n/a", "licencia": "Apache-2.0", "fecha_alta": "2026-08-19",
     "hash_sha256": "n/a", "clasificacion": "PROHIBITED (tecnica)",
     "evidencia": "embeddings prohibidos expresamente",
     "condiciones": "no usar"},
    {"recurso": "torch", "tipo": "libreria", "version": "2.14.0", "idioma": "n/a",
     "licencia": "BSD-3-Clause", "fecha_alta": "2026-08-19", "hash_sha256": "n/a",
     "clasificacion": "ALLOWED (libreria)",
     "evidencia": "las reglas listan PyTorch entre las librerias permitidas",
     "condiciones": "no usar para embeddings; el proyecto lo usa solo en la comprobacion de GPU"},
    {"recurso": "stanza", "tipo": "anotador preentrenado (no instalado)",
     "version": "n/a", "idioma": "es", "licencia": "n/a", "fecha_alta": "n/a", "hash_sha256": "n/a",
     "clasificacion": "PROHIBITED (sus modelos dependen de embeddings preentrenados)",
     "evidencia": "embeddings prohibidos expresamente", "condiciones": "no usar"},
    {"recurso": "udpipe", "tipo": "anotador preentrenado (no instalado)",
     "version": "n/a", "idioma": "es", "licencia": "n/a", "fecha_alta": "n/a", "hash_sha256": "n/a",
     "clasificacion": "AMBIGUOUS",
     "evidencia": "misma zona gris que es_core_news_md, sin la autorizacion historica y sin estar instalado",
     "condiciones": "no usar: no aporta nada que no cubra el caso ya auditado"},
    {"recurso": "flair", "tipo": "anotador preentrenado (no instalado)",
     "version": "n/a", "idioma": "es", "licencia": "n/a", "fecha_alta": "n/a", "hash_sha256": "n/a",
     "clasificacion": "PROHIBITED (embeddings contextuales y transformers)",
     "evidencia": "embeddings y transformers prohibidos", "condiciones": "no usar"},
]


def escanear_operaciones() -> dict:
    """Barrido por OPERACION sobre todo el codigo ejecutable (scripts/ y src/)."""
    spacy_import = []
    usa_atributos = []
    firmas_secuencia = []
    tagger_nltk = []
    propio = Path(__file__).name
    archivos = [p for p in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").glob("*.py"))
                if "__pycache__" not in str(p) and p.name != propio]
    for p in archivos:
        texto = p.read_text(encoding="utf-8")
        try:
            arbol = ast.parse(texto)
        except SyntaxError:
            arbol = None
        if arbol is not None:
            imports = set()
            for nodo in ast.walk(arbol):
                if isinstance(nodo, ast.Import):
                    imports.update(a.name for a in nodo.names)
                elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                    imports.add(nodo.module)
            if any(i.split(".")[0] == "spacy" for i in imports):
                spacy_import.append(p.name)
        if re.search(ATRIBUTOS_ANOTACION, texto):
            usa_atributos.append(p.name)
        for pat in FIRMAS_SECUENCIA:
            if re.search(pat, texto):
                firmas_secuencia.append((p.name, pat))
        for pat in FIRMAS_TAGGER_NLTK:
            if re.search(pat, texto):
                tagger_nltk.append((p.name, pat))
    return {"archivos_barridos": len(archivos), "spacy_import": sorted(spacy_import),
            "usan_atributos_anotacion": sorted(usa_atributos),
            "firmas_secuencia": sorted(set(firmas_secuencia)),
            "tagger_nltk": sorted(set(tagger_nltk))}


def filas_linguisticas_experiments() -> pd.DataFrame:
    """Filas de experiments.csv cuyo texto menciona el vocabulario linguistico de la fase."""
    df = pd.read_csv(ROOT / "experiments.csv", dtype=str).fillna("")
    patron = re.compile(
        r"\bpos\b|spacy|morfolog|ner|entidad|dependen|sintax|linguistic|tagger|lema|"
        r"estructura de tipos|canal", re.IGNORECASE)
    mascara = df.apply(lambda fila: bool(patron.search(" ".join(fila.astype(str)))), axis=1)
    return df[mascara]


def estadisticas_corpus() -> dict:
    """Dataciones del corpus relevantes para la calidad de anotacion (solo train)."""
    train = datos.load_train()
    textos = train[cfg.TEXT_COL].fillna("").astype(str)
    no_ascii = textos.apply(lambda t: any(ord(c) > 127 for c in t))
    acentos = textos.str.contains(r"[áéíóúüñÁÉÍÓÚÜÑ]", regex=True)
    return {
        "filas_train": int(len(textos)),
        "filas_con_no_ascii": int(no_ascii.sum()),
        "pct_no_ascii": round(100.0 * float(no_ascii.mean()), 3),
        "filas_con_vocales_acentuadas": int(acentos.sum()),
        "pct_con_acentos": round(100.0 * float(acentos.mean()), 3),
        "pct_con_mencion": round(100.0 * float(textos.str.contains("@").mean()), 1),
        "pct_con_hashtag": round(100.0 * float(textos.str.contains("#").mean()), 1),
        "pct_con_http": round(100.0 * float(textos.str.contains("http").mean()), 1),
        "longitud_mediana": int(textos.str.len().median()),
        "longitud_max": int(textos.str.len().max()),
    }


def ficha_modelo() -> dict:
    """Ficha tecnica verificada en runtime del anotador auditado."""
    import spacy
    ruta = Path(spacy.util.get_package_path("es_core_news_md"))
    meta_json = ruta / "es_core_news_md" / "meta.json"
    if not meta_json.exists():
        meta_json = ruta / "meta.json"
    sha = hashlib.sha256(meta_json.read_bytes()).hexdigest()
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    nlp = spacy.load("es_core_news_md")
    cfg_txt = nlp.config.to_str()
    estaticos = bool(re.search(r"include_static_vectors\s*=\s*true", cfg_txt))
    return {
        "nombre": meta.get("name"), "version": meta.get("version"),
        "licencia": meta.get("license"), "idioma": "es",
        "componentes": nlp.pipe_names,
        "sha256_meta_json": sha, "sha_coincide_con_v14": sha == HASH_MODELO,
        "include_static_vectors": estaticos,
        "usa_transformer_en_config": bool(re.search(r"transformer", cfg_txt, re.IGNORECASE)),
        "spacy_version": spacy.__version__,
    }


def escribe(ruta: Path, texto: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    texto = "\n".join(l.rstrip() for l in texto.splitlines()) + "\n"
    ruta.write_text(texto, encoding="utf-8")


def escribe_gap_csv() -> None:
    cols = ["id", "componente", "forma_probada", "evidencia_previa", "fichero_clave",
            "status", "por_que", "siguiente_paso"]
    with GAP_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        escritor.writeheader()
        for fila in FILAS_GAP:
            escritor.writerow({c: fila[c] for c in cols})


def escribe_registro_csv() -> None:
    cols = ["recurso", "tipo", "version", "idioma", "licencia", "fecha_alta", "hash_sha256",
            "clasificacion", "evidencia", "condiciones"]
    with REGISTRO_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        escritor.writeheader()
        for fila in FILAS_REGISTRO:
            escritor.writerow({c: fila[c] for c in cols})


def md_fila(vals) -> str:
    return "| " + " | ".join(str(v) for v in vals) + " |"


def escribe_gap_md(ops: dict, corpus: dict, filas_exp: pd.DataFrame) -> None:
    lineas = [
        "# Enriquecimiento linguistico - P0: gap analysis por operacion",
        "",
        "**Estado: P0 completado. P2 (POS estructurado) queda gateado a la resolucion de P1.**",
        "",
        "## La pregunta",
        "",
        "> Existe informacion linguistica ESTRUCTURAL en el texto de AGE que no haya sido explotada",
        "> de forma equivalente en las fases anteriores y que pueda incorporarse mediante una",
        "> representacion estructurada?",
        "",
        "La hipotesis NO es \"POS mejora AGE\" ni \"NER mejora AGE\". Es que una representacion",
        "linguistica estructurada y tipada (secuencias) puede expresar relaciones morfosintacticas,",
        "estilisticas y de entidad que la representacion lexica convencional no expresa directamente.",
        "",
        "## Metodo (por operacion, no por nombre)",
        "",
        f"- Barrido de **{ops['archivos_barridos']} ficheros** de codigo ejecutable (scripts/ y src/):",
        "  imports por AST + busqueda de uso de atributos de anotacion (`token.pos_`, `.tag_`,",
        "  `.morph`, `.dep_`, `.lemma_`, `.ents`) + firmas de CONSTRUCCION DE SECUENCIAS.",
        "- Barrido de `experiments.csv` (154 filas) por vocabulario linguistico.",
        "- Revision de las auditorias de cierre (39 mecanismos), V31, P32, P33, P34 y G12.",
        "",
        "## Resultado del barrido por operacion",
        "",
        f"- Scripts que importan `spacy`: **{len(ops['spacy_import'])}** -> `{', '.join(ops['spacy_import'])}`",
        f"- Scripts que usan atributos de anotacion: **{len(ops['usan_atributos_anotacion'])}**",
        f"- Scripts con firmas de CONSTRUCCION DE SECUENCIAS de etiquetas: **{len(ops['firmas_secuencia'])}**",
        f"- Usos de taggers NLTK (`pos_tag`, etc.): **{len(ops['tagger_nltk'])}**",
        "",
        "Ningun script del proyecto construye una secuencia de etiquetas (POS, morfologia o",
        "dependencias). El unico uso real de un anotador son dos scripts: uno de conteos (26) y otro",
        "de ratios agregados (37). Las secuencias tipadas son, por tanto, una operacion que NUNCA se",
        "ha ejecutado.",
        "",
        "## Lo que SI se uso",
        "",
        md_fila(["Componente", "Forma", "Evidencia", "Resultado"]),
        md_fila(["---", "---", "---", "---"]),
        md_fila(["POS", "26 conteos", "PH10 / D-026", "-0,0032"]),
        md_fila(["POS + morfologia + sintaxis + NER + lemas", "37 ratios agregados",
                 "V14a/v14b", "aislado 0,2409; fusion -0,0015 (1/5)"]),
        md_fila(["Funcionales", "lexicones propios (pronombres, conjunciones, modalidad)",
                 "V13-P17A/P19/P20", "0,1918 / por debajo del azar"]),
        md_fila(["Canales regex", "ortografia, composicion, discurso, posicion (listas a mano)",
                 "V31-03/04/05/06", "+0,006820 en banco; ~0 en integracion"]),
        md_fila(["Tipos superficiales de token", "secuencia @/#/URL/N/W y alfabeto ampliado",
                 "PH21/PH22", "+0,0018 (4/5), no adoptada; ampliar no mejora"]),
        md_fila(["Residuo linguistico", "correccion global/local/pendiente en score-space",
                 "P33/P34/G12", "-0,0002 / -0,0002 / sin superar la constante"]),
        "",
        "## Dos cierres historicos imprecisos (correccion de auditoria)",
        "",
        "1. `reports/v13_gap_audit.csv` marca **\"Sintaxis: n-gramas de POS\" como CERRADO** citando",
        "   D-026 (-0,0032). Pero D-026 midio **26 conteos de etiqueta**, no n-grams: el cierre se",
        "   heredo por PARECIDO DE PALABRAS, no por operacion. La forma secuencial nunca se construyo.",
        "2. La morfologia por sufijos esta cerrada con razon (redundante con `char_wb(3-5)`), pero",
        "   eso NO es la morfologia del analizador (tiempo, modo, persona...). D-043 ya lo distinguio",
        "   al abrir V14; aqui se confirma con el barrido por operacion.",
        "",
        "## Lo que NUNCA se uso (candidatos genuinos)",
        "",
        md_fila(["ID", "Representacion", "Por que es mecanicamente distinta"]),
        md_fila(["---", "---", "---"]),
        md_fila(["L1", "Secuencia de POS (n-grams de etiquetas)",
                 "los conteos/ratios no conservan orden; 0 firmas de secuencia en el codigo"]),
        md_fila(["L2", "lemma_POS (querer_VERB) y bigrams",
                 "distinto de diversidad de lemas y de conteos de etiqueta"]),
        md_fila(["L3", "Tokens de morfologia (Tense=Past|1P) y secuencias cortas",
                 "los ratios V14 son agregados; la forma tipada no existe"]),
        md_fila(["L4", "Patrones de dependencia tipados (subj->VERB)",
                 "los agregados V14 no conservan patron ni orden"]),
        md_fila(["L5", "Secuencia de tipos de entidad (PERSON VERB LOCATION)",
                 "los conteos V14 no conservan estructura; sin identidad de entidad"]),
        md_fila(["L6", "NER + POS combinado (estructural)",
                 "combinacion, no dos listas independientes"]),
        md_fila(["L7", "Texto tipado (canal paralelo al texto original)",
                 "no existe ningun canal de este tipo"]),
        "",
        "## La pared de integracion (el riesgo real de la fase)",
        "",
        "La historia no dice que esta senal no exista: dice que la INTEGRACION es el cuello. Cinco",
        "rutas de integracion tienen control de atribucion y todas cerraron (G01/G02 del audit de 39",
        "mecanismos): feature en el meta-espacio (+0,0002948 sobre barajado), miembro de mezcla",
        "(+0,0000642), residuo global (-0,0002152), residuo local (-0,0002039), pendiente en el",
        "margen (+0,0003018 sin superar la constante). Ademas V14b (-0,0015) y PH21 (+0,0018, bajo el",
        "gate pre-registrado de +0,003). La variabilidad de entrenamiento medida es 8,9e-04, y",
        "cualquier delta debe juzgarse contra esa escala.",
        "",
        "**Consecuencia para el diseno**: la pregunta de esta fase no es \"tiene senal POS?\" (casi",
        "cualquier representacion tiene alguna), sino \"contiene informacion que el campeon actual",
        "(plug-in, CV 0,6154643) no explota de forma equivalente?\". Cada experimento debe medir",
        "rescate unico / rescate compartido / dano / neto contra el candidato ADOPTADO, no contra V30",
        "argmax, y debe pre-registrar su ruta de integracion y su control antes de ejecutar.",
        "",
        "## Calidad de anotacion esperada (datos observados)",
        "",
        f"- Corpus: {corpus['filas_train']:,} filas de train; **{corpus['pct_con_acentos']}% con acentos**",
        f"  ({corpus['filas_con_vocales_acentuadas']} filas); el corpus es ASCII (de-accentuado):",
        "  eso degrada el etiquetado morfologico del espanol (\"canto\" presente vs \"canto\" preterito).",
        f"- Twitter: {corpus['pct_con_mencion']}% con mencion, {corpus['pct_con_hashtag']}% con hashtag,",
        f"  {corpus['pct_con_http']}% con URL; longitud mediana {corpus['longitud_mediana']}.",
        "- Cobertura medida en V14 sobre este mismo corpus: 99,91% de documentos con tokens,",
        "  99,91% con lemas, 99,94% con frases, 77,9% con algun verbo, 81,13% con alguna entidad.",
        "- Demostracion directa del riesgo: sobre texto sin acentos, `anos` se lematiza `ano` (no",
        "  `año`), y `jaja` se etiqueta PROPN. La degradacion existe y se medira en P2 si se ejecuta.",
        "",
        "## Clasificacion (P0) y gate",
        "",
        "- Las SIETE familias de representacion tipada y secuencial (L1-L7) son **GENUINELY_NEW**",
        "  por operacion: no existe una sola linea de codigo que las construya.",
        "- Las formas AGREGADAS de la misma informacion (conteos, ratios, canales regex) estan",
        "  cerradas con medicion; la vista de tipos SUPERFICIALES quedo bajo el gate y no es",
        "  equivalente (no es gramatica).",
        "- **Gate**: P2 solo se ejecuta si (a) se resuelve la clasificacion del anotador (P1) y",
        "  (b) se pre-registra el diseno (ruta de integracion, control, referencia = plug-in).",
        "",
        "## Estado y limites",
        "",
        "- El candidato adoptado (V30 + plug-in, LB 0,63575) es FINAL_REFERENCE y no se ha tocado.",
        "- Esta fase no usa test, no usa holdout, no genera submission y no usa el leaderboard.",
        "- P0 no entrena nada: es auditoria de codigo y de documentacion.",
    ]
    escribe(GAP_MD, "\n".join(lineas))


def escribe_legality_md(modelo: dict, ops: dict) -> None:
    lineas = [
        "# P1 - Auditoria de legalidad de herramientas linguisticas externas",
        "",
        "**Clasificacion del anotador `es_core_news_md`: AMBIGUOUS. Estado: P2 gateado.**",
        "",
        "## Objeto",
        "",
        "Antes de usar spaCy, Stanza, UDPipe, Flair u otra herramienta linguistica externa hay que",
        "determinar si esta permitido usar modelos preentrenados de etiquetado POS, NER y sintaxis,",
        "y clasificar cada recurso como ALLOWED / AMBIGUOUS / PROHIBITED. Regla del brief: si existe",
        "ambiguedad normativa, el recurso no puede producir una submission y la razon se documenta.",
        "",
        "## Fuentes normativas disponibles (2026-09-29)",
        "",
        md_fila(["Fuente", "Contenido", "Fecha"]),
        md_fila(["---", "---", "---"]),
        md_fila(["Pagina Rules de la competicion (captura en `references/age_rules_capture_2026-09-29.md`)",
                 "Reglas especificas del organizador: librerias comunes de NLP y ML permitidas (NLTK, "
                 "Scikit-learn, TensorFlow, PyTorch, Pandas, Numpy); test no para entrenamiento; "
                 "formato de envio. NO menciona transformers, LLMs ni modelos preentrenados.",
                 "2026-09-29"]),
        md_fila(["Indicacion del profesor (confirmada por Carlos)",
                 "Transformers y LLMs: prohibidos. Embeddings: prohibidos expresamente, "
                 "preentrenados o entrenados desde cero.", "2026-09-29"]),
        md_fila(["D-001 (interpretacion aprobada por Carlos, riesgo documentado)",
                 "Prohibido: transformers/LLMs y todo modelo de vectorizacion/embedding. Permitido "
                 "explicitamente: 'POS tagger no-transformer de spaCy', TF-IDF, features artesanales, "
                 "clasificadores clasicos.", "2026-09-22"]),
        md_fila(["D-026 y D-043 (autorizaciones y usos reales)",
                 "PH10 midio 26 features de POS con es_core_news_md (autorizado); V14 construyo 37 "
                 "variables morfo-sintacticas y de entidades con el mismo modelo (autorizado, con "
                 "SHA256 registrado).",
                 "commits 3a518ac y 92fff77"]),
        md_fila(["V31+ (compliance)",
                 "Dejo el 'POS tagger como predictor' en QUARANTINE por ser un modelo externo "
                 "preentrenado y no ser las reglas explicitas sobre eso.", "commit fc77f1c, 2026-09-29"]),
        "",
        "## Prohibiciones firmes (no discutidas)",
        "",
        "| Familia | Estado | Evidencia |",
        "|---|---|---|",
        "| Transformers (BERT, RoBERTa, ...) | **PROHIBITED** | indicacion del profesor, confirmada por Carlos |",
        "| LLMs | **PROHIBITED** | indicacion del profesor, confirmada por Carlos |",
        "| Embeddings / vectorizacion (preentrenados o entrenados) | **PROHIBITED** | prohibicion expresa; D-001 |",
        "| Corpus o etiquetas externas | **PROHIBITED** | brief, integridad academica |",
        "",
        "## El anotador: ficha tecnica verificada hoy",
        "",
        md_fila(["Campo", "Valor"]),
        md_fila(["---", "---"]),
        md_fila(["Modelo", f"{modelo['nombre']} {modelo['version']} (spaCy {modelo['spacy_version']})"]),
        md_fila(["Idioma / licencia", f"{modelo['idioma']} / {modelo['licencia']}"]),
        md_fila(["Componentes", ", ".join(modelo["componentes"])]),
        md_fila(["SHA256 de meta.json", modelo["sha256_meta_json"][:32] + "..."]),
        md_fila(["Coincide con el registrado en V14", str(modelo["sha_coincide_con_v14"])]),
        md_fila(["Es transformer (config)", str(modelo["usa_transformer_en_config"])]),
        md_fila(["Usa vectores estaticos preentrenados internamente",
                 f"**{modelo['include_static_vectors']}** (include_static_vectors=true)"]),
        "",
        "## Por que AMBIGUOUS y no ALLOWED ni PROHIBITED",
        "",
        "- **No esta PROHIBITED por ningun texto disponible**: no es transformer, no es LLM, no es un",
        "  modelo de vectorizacion usado como representacion. Las tres prohibiciones firmes no lo cubren.",
        "- **Tampoco es ALLOWED sin reservas**: es un modelo externo PRETRENETRADO (entrenado por",
        "  Explosion sobre corpus externos, y su tok2vec consulta vectores estaticos preentrenados",
        "  internamente, `include_static_vectors=true`); el profesor no se ha pronunciado sobre",
        "  anotadores preentrenados como categoria; y la prohibicion expresa de embeddings, leida de",
        "  forma estricta ('en cualquier punto del pipeline'), lo alcanzaria.",
        "- **El proyecto tiene dos posiciones documentadas y opuestas**: D-001 lo permitia por nombre",
        "  y D-026/D-043 lo autorizaron y usaron (con procedencia); V31 lo dejo en cuarentena.",
        "  Esa contradiccion no la resuelve este informe: la resuelve Carlos (o el profesor).",
        "",
        "## Interpretaciones en juego",
        "",
        "| Lectura | Alcance de la prohibicion de embeddings | Consecuencia para el anotador |",
        "|---|---|---|",
        "| A: sola representacion | Prohibido USAR embeddings como representacion o feature; no alcanza a "
        "un anotador cuyas ETIQUETAS se usan | ALLOWED bajo D-001 (etiquetas si, vectores no) |",
        "| B: estricta | Prohibido que un modelo preentrenado con vectores participe en cualquier paso "
        "| El anotador queda fuera; la fase solo podria usar anotacion construida por nosotros |",
        "",
        "## Consecuencia mientras siga AMBIGUOUS",
        "",
        "1. Ningun resultado de esta fase puede alimentar una submission (regla del brief).",
        "2. P2 **no se ejecuta** hasta que la clasificacion se resuelva.",
        "3. El candidato adoptado (V30 + plug-in) es FINAL_REFERENCE y no depende de nada de esto.",
        "",
        "## Resolucion pendiente (elevada a Carlos)",
        "",
        "- **Opcion A**: extender expresamente a esta fase la lectura de D-001 (etiquetas si, vectores",
        "  no): habilita P2 con procedencia registrada, control de barajado y ablacion sin canal;",
        "  la submission (si algun dia supera todos los gates) seguira exigiendo autorizacion explicita.",
        "- **Opcion B**: mantener la cuarentena de V31: la fase cierra con P0/P1 documentados y sin",
        "  experimentos con anotadores preentrenados.",
        "- **Opcion C**: consultar al profesor y pausar P2 hasta su respuesta.",
        "",
        "## Registro de procedencia (requisito del brief, cumplido)",
        "",
        "Ficha completa en `reports/linguistic_annotation_registry.csv`: recurso, tipo, version,",
        "idioma, licencia, fecha de alta, SHA256 y clasificacion. El modelo esta instalado en el venv",
        "desde 2026-08-19; el hash de `meta.json` coincide con el registrado en V14.",
        "",
        "## Verificaciones negativas de esta fase (hasta hoy)",
        "",
        "| Restriccion | Estado |",
        "|---|---|",
        "| uso de test para decidir algo | 0 |",
        "| uso del holdout | 0 |",
        "| tuning con leaderboard | 0 |",
        "| submissions generadas | 0 |",
        "| llamadas de red en la fase | 0 (la consulta a la pagina publica de reglas es lectura de "
        "reglamento, no de datos) |",
        "| uso de transformers/LLMs/embeddings | 0 |",
        "| scripts con firmas de secuencias tipadas | 0 |",
    ]
    escribe(LEGAL_MD, "\n".join(lineas))


def escribe_captura() -> None:
    escribe(CAPTURA, METADATOS_CAPTURA + TEXTO_REGLAS)


def anade(ruta: Path, marca: str, bloque: str) -> str:
    if not ruta.exists():
        return f"{ruta.name}: no existe"
    texto = ruta.read_text(encoding="utf-8")
    if marca in texto:
        return f"{ruta.name}: ya estaba"
    cuerpo = (texto if texto.endswith("\n") else texto + "\n") + bloque
    escribe(ruta, cuerpo)
    return f"{ruta.name}: +{len(bloque.splitlines())} lineas"


D_075 = """
---

## D-075 - Reapertura: fase de enriquecimiento linguistico estructurado (P0 y P1 completados, P2 gateado)

- **Observation**: el candidato adoptado (V30 + plug-in, LB 0,63575) queda como FINAL_REFERENCE y
  no se toca. Se reabre una fase con una unica pregunta autorizada: si existe informacion
  linguistica ESTRUCTURAL (secuencias tipadas de POS, lemma_POS, morfologia, dependencias,
  entidades o texto tipado) que las fases anteriores no explotaron de forma equivalente.

- **P0 - gap analysis por operacion** (233 ficheros de codigo y 154 filas de experiments.csv):
  - Lo USADO historicamente: 26 conteos POS (PH10/D-026, -0,0032); 37 ratios agregados de
    morfologia, sintaxis, entidades y lemas (V14a/b, fusion -0,0015 con 1/5 folds); lexicones
    propios (V13); canales regex sin tagger (V31-03, +0,006820 en banco y ~0 en integracion);
    secuencia de tipos SUPERFICIALES de token (PH21/PH22, +0,0018, no adoptada; enriquecer el
    alfabeto no mejora); correcciones en score-space del residuo (P33/P34/G12, cerradas).
  - Lo NUNCA usado: las SECUENCIAS tipadas (L1-L7). Verificado por operacion: 0 scripts con
    firmas de construccion de secuencias; solo 2 scripts usan spaCy y hacen conteos y ratios.
  - Dos cierres historicos imprecisos documentados: "n-gramas de POS" figuraba como cerrado
    citando D-026 (que midio conteos), y la morfologia por sufijos (cerrada con razon por
    redundancia con char_wb) no es la morfologia del analizador (D-043 ya lo distinguia).
  - La pared de integracion es el riesgo real: cinco rutas con control de atribucion, todas
    cerradas; variabilidad de entrenamiento 8,9e-04 como escala de juicio.

- **P1 - auditoria de legalidad**: se capturo la pagina Rules de la competicion HOY
  (`references/age_rules_capture_2026-09-29.md`): las reglas especificas permiten librerias
  comunes de NLP y ML (NLTK, Scikit-learn, TensorFlow, PyTorch, Pandas, Numpy) y no mencionan
  transformers ni LLMs ni preentrenados. Carlos confirmo las prohibiciones de curso:
  transformers y LLMs (profesor) y embeddings EXPRESAMENTE, preentrenados o entrenados.
  El anotador `es_core_news_md` (3.8.0, GPL-3.0, SHA256 de meta.json 889a2fd0..., no-transformer)
  queda **AMBIGUOUS**: D-001 lo permitia por nombre y D-026/D-043 lo autorizaron y usaron; V31
  lo dejo en cuarentena; y su tok2vec usa vectores estaticos preentrenados internamente
  (include_static_vectors=true), aunque de el solo se usarian las etiquetas.

- **Decision**: la fase queda abierta en P0/P1. **P2 (POS estructurado) NO se ejecuta hasta
  resolver la clasificacion del anotador.** Mientras sea AMBIGUOUS, ningun resultado de esta via
  puede alimentar una submission (regla del brief). La resolucion se eleva a Carlos en tres
  opciones: (A) extender expresamente la lectura de D-001 a esta fase (etiquetas si, vectores no,
  con procedencia y ablacion); (B) mantener la cuarentena estricta; (C) consultar al profesor.
  El candidato adoptado no se modifica en ninguna de las tres.

- **Consequences**: ningun resultado nuevo puede desplazar al candidato adoptado sin gate completo,
  segunda particion y autorizacion explicita. Artefactos: `reports/linguistic_enrichment_gap_analysis.md`,
  `reports/linguistic_enrichment_gap_analysis.csv`, `reports/age_linguistic_legality_audit.md`,
  `reports/linguistic_annotation_registry.csv`, `references/age_rules_capture_2026-09-29.md`,
  `experiments/age_linguistic_p0p1_gap_y_legalidad.json`. Sin test, sin holdout, sin submission.
"""

RUNLOG = """
---

## Reapertura: enriquecimiento linguistico estructurado (P0 y P1) · 2026-09-29

- **P0 (gap analysis por operacion)**: barrido de 233 ficheros y 154 filas de experiments.csv.
  Las SECUENCIAS tipadas (POS, lemma_POS, morfologia, dependencias, entidades, texto tipado)
  NUNCA se construyeron: 0 firmas de secuencia en todo el codigo. Dos cierres historicos
  imprecisos detectados y documentados (n-gramas de POS cerrados por nombre citando conteos).
- **P1 (legalidad)**: reglas especificas capturadas de la pagina Rules (librerias comunes
  permitidas; nada sobre preentrenados). Prohibiciones confirmadas por Carlos: transformers,
  LLMs y embeddings (preentrenados o entrenados). Anotador spaCy = **AMBIGUOUS** (conflicto
  D-001/D-043 vs V31 + uso interno de vectores estaticos).
- **P2 gateado**: no se ejecuta hasta resolver la clasificacion del anotador (opciones A/B/C
  elevadas a Carlos). Sin test, sin holdout, sin submission, sin tocar el candidato adoptado.
"""

ESTADO_FASE = """
---

# FASE ABIERTA - LINGUISTIC STRUCTURED ENRICHMENT (2026-09-29)

Pregunta: existe informacion linguistica estructural (secuencias tipadas) no explotada de forma
equivalente? El candidato adoptado (V30 + plug-in, LB 0,63575) permanece como FINAL_REFERENCE.

| Paso | Estado | Resultado |
|---|---|---|
| P0 gap analysis por operacion | **COMPLETADO** | Secuencias tipadas L1-L7 = GENUINELY_NEW; formas agregadas y regex cerradas con medicion |
| P1 auditoria de legalidad | **COMPLETADO** | Anotador es_core_news_md = AMBIGUOUS (conflicto interno + vectores estaticos internos) |
| P2 POS estructurado | **GATEADO** | No se ejecuta hasta resolver la clasificacion (opciones A/B/C) |

Reglas especificas capturadas: `references/age_rules_capture_2026-09-29.md`.
Artefactos: `reports/linguistic_enrichment_gap_analysis.{md,csv}`,
`reports/age_linguistic_legality_audit.md`, `reports/linguistic_annotation_registry.csv`.
Sin test, sin holdout, sin submissions.
"""


def anade_experiments() -> dict:
    ruta = ROOT / "experiments.csv"
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas[0].split(",")]
    existentes = {l.split(",", 1)[0] for l in lineas[1:] if l}
    nuevas = [
        {"experiment_id": "AGE-LING-P0-GAP", "fecha": FECHA, "bloque": "30.1",
         "etapa": "P0 - gap analysis linguistico por operacion",
         "pregunta": "existe informacion linguistica estructural (secuencias tipadas) no explotada de forma equivalente?",
         "hipotesis": "una representacion tipada y secuencial expresa morfosintaxis, estilo y entidad que la representacion lexica no expresa directamente",
         "features": "ninguna: auditoria de codigo ejecutado (233 ficheros) y experiments.csv (154 filas)",
         "preprocessing": "ninguno", "representacion": "auditoria de representaciones historicas",
         "modelo": "ninguno", "hiperparametros": "ninguno",
         "validacion": "barrido por operacion: imports por AST + atributos de anotacion + firmas de secuencia",
         "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "",
         "resultado": "secuencias tipadas (POS, lemma_POS, morfologia, dependencias, entidades, texto tipado) NUNCA construidas: 0 firmas en el codigo; formas agregadas (26 conteos, 37 ratios) y regex cerradas con medicion; 2 cierres historicos imprecisos por nombre documentados",
         "interpretacion": "L1-L7 son GENUINELY_NEW por operacion; el riesgo de la fase es la integracion (5 rutas con control, todas cerradas; variabilidad 8,9e-04)",
         "decision": "P2 condicionado a la resolucion de P1",
         "duracion_seg": "0.0",
         "artefactos": "reports/linguistic_enrichment_gap_analysis.md | reports/linguistic_enrichment_gap_analysis.csv"},
        {"experiment_id": "AGE-LING-P1-LEGALITY", "fecha": FECHA, "bloque": "30.1",
         "etapa": "P1 - auditoria de legalidad de herramientas NLP",
         "pregunta": "se pueden usar anotadores preentrenados de POS/NER/sintaxis en AGE?",
         "hipotesis": "las prohibiciones del curso cubren transformers, LLMs y embeddings; los anotadores no estan cubiertos por ningun texto",
         "features": "ninguna: auditoria normativa y tecnica",
         "preprocessing": "ninguno", "representacion": "no aplica",
         "modelo": "ninguno", "hiperparametros": "ninguno",
         "validacion": "captura de la pagina Rules (2026-09-29) + prohibiciones del profesor confirmadas + decisiones internas (D-001, D-026, D-043, V31)",
         "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "", "micro_f1_mean": "",
         "accuracy_mean": "",
         "resultado": "es_core_news_md (3.8.0, GPL-3.0, SHA256 889a2fd0...) = AMBIGUOUS: D-001 lo permitia, D-026/D-043 lo autorizaron, V31 lo cuarenteno; include_static_vectors=true (solo se usarian etiquetas)",
         "interpretacion": "no esta prohibido por ningun texto; tampoco es ALLOWED sin reservas; la contradiccion interna la resuelve Carlos",
         "decision": "P2 gateado; opciones A/B/C elevadas a Carlos",
         "duracion_seg": "0.0",
         "artefactos": "reports/age_linguistic_legality_audit.md | reports/linguistic_annotation_registry.csv | references/age_rules_capture_2026-09-29.md"},
    ]
    anadidas = []
    with ruta.open("a", encoding="utf-8") as f:
        for reg in nuevas:
            if reg["experiment_id"] in existentes:
                continue
            f.write(",".join(
                f'"{reg[c]}"' if ("," in str(reg[c]) or '"' in str(reg[c])) else str(reg[c])
                for c in cols) + "\n")
            anadidas.append(reg["experiment_id"])
    return {"anadidas": anadidas, "filas": len(ruta.read_text(encoding="utf-8").splitlines()) - 1}


def anade_notebook(ops: dict, corpus: dict, modelo: dict) -> dict:
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    antes = len(nb.cells)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and MARCADOR_NB in c.source:
            del nb.cells[i:]
            break

    def md(t):
        nb.cells.append(nbformat.v4.new_markdown_cell(t))

    def code(t):
        nb.cells.append(nbformat.v4.new_code_cell(t))

    md("""# FASE - LINGUISTIC STRUCTURED ENRICHMENT

## Reapertura autorizada (2026-09-29)

Pregunta unica de la fase: existe **informacion linguistica estructural** en el texto de AGE que
no haya sido explotada de forma equivalente en fases anteriores y que pueda incorporarse mediante
una representacion estructurada?

El candidato adoptado (**V30 + plug-in, LB 0,63575**) permanece como `FINAL_REFERENCE` y no se
modifica. La hipotesis no es "POS mejora AGE" ni "NER mejora AGE": es que una representacion
**tipada y secuencial** expresa morfosintaxis, estilo y estructura de entidad que la representacion
lexica convencional no expresa directamente.

## P0 - gap analysis por operacion

Lo que manda es el **codigo ejecutado**, no las palabras de un informe: se barrio el codigo por
OPERACION (imports por AST, uso de atributos de anotacion, firmas de construccion de secuencias).""")

    code("""import json, pathlib
import pandas as pd
RAIZ = pathlib.Path("..")
res = json.loads((RAIZ / "experiments" / "age_linguistic_p0p1_gap_y_legalidad.json").read_text(encoding="utf-8"))
gap = pd.read_csv(RAIZ / "reports" / "linguistic_enrichment_gap_analysis.csv")
print("Barrido:", res["operaciones"]["archivos_barridos"], "ficheros de codigo")
print("Scripts que importan spacy:", res["operaciones"]["spacy_import"])
print("Scripts con firmas de SECUENCIA de etiquetas:", len(res["operaciones"]["firmas_secuencia"]))
print("Usos de taggers NLTK:", len(res["operaciones"]["tagger_nltk"]))
print()
print(gap[["id", "componente", "status"]].to_string(index=False))
assert len(res["operaciones"]["firmas_secuencia"]) == 0, "existe una secuencia tipada: revisar el gap"
print()
print("[OK] las secuencias tipadas nunca se construyeron: L1-L7 son GENUINELY_NEW")""")

    md("""## P1 - legalidad de las herramientas

Reglas especificas capturadas de la pagina Rules el 2026-09-29 (ver
`references/age_rules_capture_2026-09-29.md`): permiten librerias comunes de NLP y ML (NLTK,
Scikit-learn, TensorFlow, PyTorch) y **no** mencionan transformers, LLMs ni preentrenados.
Prohibiciones confirmadas por Carlos: transformers y LLMs (profesor) y **embeddings
expresamente, preentrenados o entrenados**. El anotador `es_core_news_md` queda **AMBIGUOUS**
(conflicto interno D-001/D-043 vs V31 y vectores estaticos internos).""")

    code("""reg = pd.read_csv(RAIZ / "reports" / "linguistic_annotation_registry.csv")
print(reg[["recurso", "version", "clasificacion"]].to_string(index=False))
print()
print("Modelo auditado:", res["modelo"]["nombre"], res["modelo"]["version"],
      "| licencia:", res["modelo"]["licencia"])
print("SHA256 meta.json coincide con V14:", res["modelo"]["sha_coincide_con_v14"])
print("Es transformer:", res["modelo"]["usa_transformer_en_config"],
      "| usa vectores estaticos internos:", res["modelo"]["include_static_vectors"])
print()
print("Corpus (train):", res["corpus"]["filas_train"], "filas |",
      res["corpus"]["pct_con_acentos"], "% con acentos |",
      res["corpus"]["pct_con_mencion"], "% con mencion |",
      res["corpus"]["pct_con_http"], "% con URL")
assert res["modelo"]["sha_coincide_con_v14"] is True
assert res["modelo"]["usa_transformer_en_config"] is False
assert res["corpus"]["filas_con_vocales_acentuadas"] == 0
assert (reg.loc[reg["recurso"] == "es_core_news_md", "clasificacion"] == "AMBIGUOUS").all()
print()
print("[OK] ficha del anotador y clasificacion verificadas contra el artefacto fuente")""")

    md("""## Estado de la fase

| Paso | Estado |
|---|---|
| P0 gap analysis por operacion | **COMPLETADO** |
| P1 auditoria de legalidad | **COMPLETADO** |
| P2 POS estructurado | **GATEADO** (resolucion del anotador pendiente: opciones A/B/C) |

Mientras el anotador siga AMBIGUOUS, ningun resultado de esta via puede alimentar una submission
(regla del brief). Sin test, sin holdout, sin uso del leaderboard, sin tocar el candidato adoptado.""")
    nbformat.write(nb, NOTEBOOK)
    return {"antes": antes, "despues": len(nb.cells), "anadidas": len(nb.cells) - antes}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    print("=== P0: barrido por operacion ===", flush=True)
    ops = escanear_operaciones()
    print(f"  archivos barridos: {ops['archivos_barridos']}", flush=True)
    print(f"  imports de spacy: {ops['spacy_import']}", flush=True)
    print(f"  atributos de anotacion: {ops['usan_atributos_anotacion']}", flush=True)
    print(f"  firmas de secuencia: {ops['firmas_secuencia'] or 'NINGUNA'}", flush=True)
    print(f"  taggers NLTK: {ops['tagger_nltk'] or 'NINGUNO'}", flush=True)
    if ops["firmas_secuencia"]:
        print("[STOP] existe una construccion de secuencias no inventariada", flush=True)
        return 1

    print("\n=== corpus: calidad de anotacion esperada ===", flush=True)
    corpus = estadisticas_corpus()
    for k, v in corpus.items():
        print(f"  {k}: {v}", flush=True)

    print("\n=== P1: ficha del anotador ===", flush=True)
    modelo = ficha_modelo()
    for k, v in modelo.items():
        print(f"  {k}: {v}", flush=True)
    if not modelo["sha_coincide_con_v14"]:
        print("[STOP] el modelo cambio de hash respecto de V14", flush=True)
        return 1

    filas_exp = filas_linguisticas_experiments()
    print(f"\n  filas linguisticas en experiments.csv: {len(filas_exp)}", flush=True)

    print("\n=== artefactos ===", flush=True)
    escribe_captura()
    escribe_gap_csv()
    escribe_registro_csv()
    escribe_gap_md(ops, corpus, filas_exp)
    escribe_legality_md(modelo, ops)
    for f in (CAPTURA, GAP_CSV, GAP_MD, LEGAL_MD, REGISTRO_CSV):
        print(f"  {f.relative_to(ROOT)}: {len(f.read_text(encoding='utf-8').splitlines())} lineas",
              flush=True)

    sha_captura = hashlib.sha256(CAPTURA.read_bytes()).hexdigest()
    RESULTADO.write_text(json.dumps({
        "fase": "P0 gap analysis + P1 legalidad", "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "operaciones": ops, "corpus": corpus, "modelo": modelo,
        "filas_linguisticas_experiments": int(len(filas_exp)),
        "clasificacion_anotador": "AMBIGUOUS",
        "p2_estado": "GATEADO a resolucion (opciones A/B/C en D-075)",
        "captura_reglas_sha256": sha_captura,
        "sin_test": True, "sin_holdout": True, "sin_submission": True,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  experiments/age_linguistic_p0p1_gap_y_legalidad.json", flush=True)

    print("\n=== documentacion ===", flush=True)
    print("  " + anade(ROOT / "DECISIONS.md", "## D-075", D_075), flush=True)
    print("  " + anade(ROOT / "RUNLOG.md", "Reapertura: enriquecimiento linguistico", RUNLOG),
          flush=True)
    print("  " + anade(ROOT / "PROJECT_STATE.md", "FASE ABIERTA - LINGUISTIC STRUCTURED ENRICHMENT",
                       ESTADO_FASE), flush=True)
    print("  " + anade(ROOT / "FINAL_REPORT.md", "FASE ABIERTA - LINGUISTIC STRUCTURED ENRICHMENT",
                       ESTADO_FASE), flush=True)
    print("  " + str(anade_experiments()), flush=True)

    print("\n=== notebook ===", flush=True)
    print("  " + str(anade_notebook(ops, corpus, modelo)), flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
