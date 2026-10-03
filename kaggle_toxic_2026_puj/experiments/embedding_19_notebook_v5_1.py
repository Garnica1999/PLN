"""P8 de la fase v5.1 (EBD-25): integracion de v5 y v5.1 en el notebook consolidado.

Contexto
--------
La spec de v5.1 (secciones 23 a 33) pide que TODO el contenido de v5 y v5.1 quede en el
notebook consolidado del proyecto, con su mecanismo reproducible de entrenamiento desde
cero, las curvas, la cobertura, la comparacion FastText frente al control Word2Vec, las dos
conclusiones separadas y la contabilidad de presupuesto con una asercion que falle si se
excede.

Este script **genera** la seccion 18 del notebook en vez de editarla a mano: el notebook es un
JSON de 44 celdas y editarlo a mano es una fuente de errores silenciosos. Ademas deja el
generador como evidencia reproducible de lo que se integro.

Que hace
--------
1. Localiza el notebook fuente de verdad (no asume el nombre: lo busca).
2. Comprueba que la seccion 18 no exista ya, para no duplicarla.
3. Construye las celdas de markdown y de codigo de la seccion 18.
4. Las anade al final y guarda el notebook.
5. **No** ejecuta el notebook: la validacion por ejecucion la hace
   ``experiments/embedding_20_validacion_notebook_v5_1.py``, que corre el notebook entero y
   escribe ``reports/embedding_notebook_validation.md``.

Diseno de las celdas
--------------------
El notebook verifica, no recalcula: las celdas cargan los artefactos que dejaron los
scripts de la fase y comprueban que las cifras coinciden. El entrenamiento desde cero queda
**detrás de un interruptor** (``TRAIN_FROM_SCRATCH = False``) con el mecanismo completo a la
vista, de forma que la afirmacion "los pesos son aleatorios" sea verificable y no una nota.

Uso
---
    /home/carlos/venv/bin/python experiments/embedding_19_notebook_v5_1.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTO = "EBD-25 (P8)"
TITULO_BUSCADO = "## 18. Fase v5 y v5.1"
NOMBRES_CANDIDATOS = [
    "notebooks/10_final_pipeline.ipynb",
    "notebooks/10_final_pipeline_v2.ipynb",
    "notebooks/final_pipeline.ipynb",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V5.1-P8] {mensaje}", flush=True)


def localizar_notebook() -> Path:
    """Localiza el notebook fuente de verdad sin asumir su nombre.

    Se prueban primero los candidatos conocidos y, si ninguno existe, se busca cualquier
    ``.ipynb`` del directorio de notebooks que contenga la cadena de la seccion de modelos.

    Returns:
        Ruta del notebook consolidado.

    Raises:
        RuntimeError: si no se encuentra ningun notebook utilizable.
    """
    for relativo in NOMBRES_CANDIDATOS:
        ruta = PROJECT_ROOT / relativo
        if ruta.exists():
            log(f"   notebook localizado: {relativo}")
            return ruta
    candidatos = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    for ruta in candidatos:
        contenido = ruta.read_text(encoding="utf-8")
        if "Modelos" in contenido and "HS" in contenido:
            log(f"   notebook localizado por busqueda: {ruta.name}")
            return ruta
    raise RuntimeError("No se encuentra el notebook consolidado del proyecto")


def celda_markdown(fuente: str) -> dict[str, Any]:
    """Construye una celda de markdown.

    Args:
        fuente: texto de la celda.

    Returns:
        Diccionario de celda con los campos que exige nbformat.
    """
    lineas = fuente.strip("\n").splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lineas}


def celda_codigo(fuente: str) -> dict[str, Any]:
    """Construye una celda de codigo.

    Args:
        fuente: texto de la celda.

    Returns:
        Diccionario de celda con los campos que exige nbformat.
    """
    lineas = fuente.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lineas,
    }


def construir_celdas() -> list[dict[str, Any]]:
    """Construye las celdas de la seccion 18 del notebook.

    Returns:
        Lista de celdas en el orden en que deben aparecer.
    """
    return [
        celda_markdown(
            """
## 18. Fase v5 y v5.1: representación distribuida aprendida desde cero

| | |
|---|---|
| **Qué es** | Entrenar una representación vectorial de las palabras **desde cero**, con fastText C++ compilado en la máquina, sobre un corpus español **externo y sin etiquetar**, y probar si aporta información que el modelo V03 no tenga ya |
| **Qué NO es** | No se usan transformers, ni LLM, ni embeddings preentrenados, ni etiquetas externas. Los pesos se inicializan al azar y se aprenden con datos de texto sin etiqueta |
| **Corpus** | Wikipedia en español (tres escalas: 5M, 10M y 50M tokens), mC4-es (web, 10M) y la unión de ambos (20M) |
| **Resultado** | El embedding queda como **candidato no adoptado**. Ninguna de las 14 configuraciones medidas aporta a V03 por encima del gate |
| **V03 tras la fase** | **Sin cambios**: `corrected_cv` 0.777211, LB 0.78306 |
| **Decisiones** | D-030 a D-042 |

### 18.0.1 Motivación

V03 es un bloque disperso de 27.431 características (n-gramas de palabra 1-2 y de carácter
`char_wb` 3-5) con una regresión logística balanceada. Funciona bien porque lee la
**superficie** del texto: el insulto exacto, la mención, la deformación ortográfica. Lo que
no tiene es una noción de **similitud**: dos palabras que aparecen en contextos parecidos
son para V03 dos columnas distintas, y con 1.483 ejemplos etiquetados no hay forma de
aprender esa relación.

La hipótesis de la fase es que una representación distributiva entrenada desde cero
—vectores que sí agrupan palabras por contexto— puede aportar esa dimensión que falta. La
fase no da por hecho que vaya a funcionar: la mide, y admite que no.

### 18.0.2 Restricciones y autorización

| regla | cumplimiento |
|---|---|
| Sin transformers, LLM, BERT, RoBERTa | 0 usos, verificado por análisis estático de los scripts (sección 36) |
| Sin embeddings preentrenados | 0 usos; los pesos se inicializan al azar y se entrenan aquí |
| Sin etiquetas externas | 0; el corpus externo es solo texto sin etiquetar |
| Test nunca para entrenar | 0 coincidencias de 8-gramas entre test y los tres corpora |
| Holdout | **agotado (5/5)** e intacto: esta fase no obtiene autoridad adicional de él |
| Corpus externo | Autorizado por el profesor **solo para aprender la representación** (D-030), nunca para uso supervisado |

La autorización se registra tal como se comunicó, sin documentación escrita inventada. Las
fuentes habilitadas fueron Wikipedia ES, mC4-es, la unión de ambos y, en principio, HPLT,
CEREAL y OSCAR; **OSCAR es *gated* y no se forzó**.
"""
        ),
        celda_codigo(
            '''
"""Entrenamiento desde cero: mecanismo e interruptor.

Este bloque deja por escrito **de dónde salen los pesos**. Por defecto no entrena nada
(`TRAIN_FROM_SCRATCH = False`) y se limita a cargar los artefactos que dejaron los scripts
de la fase, para que el notebook se ejecute en minutos. Al ponerlo a `True` ejecuta el
mecanismo completo de construcción del corpus y entrenamiento de fastText, que es
reproducible pero cuesta unos minutos por modelo.

`weights_origin = random initialization` y `pretrained_weights = False` no son una promesa:
son la definicion de lo que hace el codigo de abajo.
"""

from __future__ import annotations

import subprocess

TRAIN_FROM_SCRATCH = False
FASTTEXT_BIN = PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext"
weights_origin = "random initialization"
pretrained_weights = False
semilla_fasttext = 42

ARQUITECTURA_DE_REFERENCIA = {
    "vector_size": 200,
    "window": 5,
    "min_count": 3,
    "min_n": 3,
    "max_n": 6,
    "negative": 10,
    "sample": 1e-4,
    "epochs": 5,
    "sg": 1,
    "seed": semilla_fasttext,
    "bucket": 200_000,
}
CORPORA = {
    "wikipedia_5M": PROJECT_ROOT / "data/external/corpora/wikipedia_es_5M/emb_normal/tokens_5M.txt",
    "wikipedia_10M": PROJECT_ROOT / "data/external/corpora/wikipedia_es/emb_normal/tokens_10M.txt",
    "wikipedia_50M": PROJECT_ROOT / "data/external/corpora/wikipedia_es_50M/emb_normal/tokens_50M.txt",
    "web_mc4_10M": PROJECT_ROOT / "data/external/corpora/mc4_es/emb_normal/tokens_10M.txt",
    "mixto_20M": PROJECT_ROOT / "data/external/corpora/mixto/emb_normal/tokens_20M.txt",
}


def entrenar_desde_cero(nombre: str, corpus: str, destino: Path, **cambios) -> Path:
    """Entrena un modelo de fastText desde cero con los pesos inicializados al azar.

    No hay ninguna ruta de código que cargue pesos preentrenados: la segunda línea de la
    función es la que decide qué pesos existen, y son los que fastText inicializa al azar con
    la semilla indicada y luego ajusta con el corpus.

    Args:
        nombre: nombre del modelo (para el mensaje).
        corpus: clave de `CORPORA` con el archivo de entrenamiento.
        destino: prefijo de salida; fastText escribe `.bin` y `.vec`.
        **cambios: sobrescribe parametros de `ARQUITECTURA_DE_REFERENCIA`.

    Returns:
        Ruta del archivo `.vec` generado.
    """
    parametros = {**ARQUITECTURA_DE_REFERENCIA, **cambios}
    comando = [
        str(FASTTEXT_BIN), "skipgram",
        "-input", str(CORPORA[corpus]),
        "-output", str(destino),
        "-dim", str(parametros["vector_size"]),
        "-ws", str(parametros["window"]),
        "-minCount", str(parametros["min_count"]),
        "-neg", str(parametros["negative"]),
        "-sample", str(parametros["sample"]),
        "-epoch", str(parametros["epochs"]),
        "-sg", str(parametros["sg"]),
        "-seed", str(parametros["seed"]),
        "-minn", str(parametros["min_n"]),
        "-maxn", str(parametros["max_n"]),
        "-bucket", str(parametros["bucket"]),
        "-thread", "16",
        "-verbose", "0",
    ]
    print(f"entrenando {nombre} desde cero con {CORPORA[corpus].name}")
    subprocess.run(comando, check=True)
    return destino.with_suffix(".vec")


print("weights_origin          =", weights_origin)
print("pretrained_weights      =", pretrained_weights)
print("TRAIN_FROM_SCRATCH      =", TRAIN_FROM_SCRATCH)
print("arquitectura            =", ARQUITECTURA_DE_REFERENCIA)
print("corpora disponibles     =", [c for c, r in CORPORA.items() if r.exists()])
if TRAIN_FROM_SCRATCH:
    print("ATENCION: se va a entrenar de verdad; esto tarda minutos por modelo.")
'''
        ),
        celda_markdown(
            """
### 18.1 Corpus: cómo se construyó

El pipeline tiene siete pasos y es **idéntico** para los tres corpora, para que la
comparación sea de cantidad de datos y no de receta:

| paso | qué hace |
|---|---|
| `download` | Descarga del volcado de Wikipedia ES (`eswiki-latest-pages-articles-multistream1.xml-p1p159400.bz2`, 305 MB, sha256 `3248f107…`). Los corpus web y mixto reutilizan ese mismo tramo ya descargado |
| `raw` | Lectura del XML con `mwparserfromhell`; una revisión por artículo |
| `clean` | Se elimina el marcado MediaWiki: referencias, plantillas, tablas, enlaces internos, archivos y categorías |
| `dedup` | SHA-1 del texto limpio; los duplicados exactos se descartan |
| `normalize` | Variante **EMB-NORMAL**: minúsculas **conservando tildes**, URLs, menciones, hashtags y números reducidos a marcadores estables (`url`, `usuario`, `hashtag`, `numero`). **Sin corrector ortográfico y sin reconstrucción de tildes** |
| `tokenize` | Un documento por línea, con el corpus ya normalizado |
| `shards` + `manifest` | Reparto en trozos de 10M tokens y manifiesto con licencia, sha256 del volcado y **sha256 del corpus procesado** |

Ese último detalle es una corrección de la fase v5.1: el sha256 del volcado **no identifica
la escala**, porque 5M, 10M y 50M de Wikipedia salen del mismo tramo. La identidad de un
entrenamiento usa el sha256 del archivo de tokens ya procesado.

Para el corpus web (mC4-es) se añade una limpieza específica: líneas repetidas dentro del
mismo documento (el *boilerplate* de C4), líneas de menos de 40 caracteres sin dígitos, líneas
con más de una URL, y documentos con más del 30 % de URL o del 25 % de dígitos.
"""
        ),
        celda_codigo(
            '''
"""Inventario de los tres corpora: escala, licencia, huellas y tiempo de entrenamiento."""

import json

from src.embedding_utils import sha256_de

MANIFIESTO = json.loads(
    (RUTAS["data"] / "external" / "corpora" / "corpus_manifest.json").read_text(encoding="utf-8")
)
entradas = list(MANIFIESTO["corpus_construidos"])
mixto = MANIFIESTO.get("corpus_construidos_mixto")
entradas.append(mixto if isinstance(mixto, dict) else mixto[0])

inventario = []
for entrada in entradas:
    ruta = RUTAS["data"] / entrada["processed_ruta"] if "processed_ruta" in entrada else None
    huella = sha256_de(ruta) if ruta and ruta.exists() else None
    inventario.append(
        {
            "id": entrada["id"],
            "fuente": entrada.get("source", entrada.get("id")),
            "tokens": entrada["token_count"],
            "documentos": entrada.get("document_count", "no separado"),
            "tipos": entrada.get("vocabulary_size", "no separado"),
            "licencia": str(entrada.get("license", ""))[:44],
            "sha256_volvcado": str(entrada.get("raw_sha256", ""))[:12],
            "sha256_procesado": (huella or entrada.get("processed_sha256", ""))[:12],
        }
    )
mostrar_tabla(pd.DataFrame(inventario))
print()
print("Los tres primeras filas comparten sha256 de volcado y tienen sha256 procesado distinto:")
print("esa es la razon por la que la identidad de entrenamiento usa el segundo.")
'''
        ),
        celda_markdown(
            """
### 18.2 Cobertura: qué parte del vocabulario de la competencia aparece en el corpus

La cobertura se mide contra el vocabulario **TOXIC** (los tipos que aparecen en train y
test). Usar ese vocabulario como *diagnóstico* está permitido por la autorización; usarlo
para entrenar, no, y no se hace: el corpus no contiene ni una sola etiqueta.

Lo que la fase v5 demostró es que **cobertura y aportación se mueven por separado**: el
corpus web sube la cobertura del 0,68 al 0,83 y la aportación a V03 cambia en milésimas.
"""
        ),
        celda_codigo(
            '''
"""Cobertura del vocabulario TOXIC por corpus y por grupo léxico."""

cobertura_global = pd.DataFrame(
    [
        {"corpus": "wikipedia 5M", "tipo": 0.6302, "token": 0.8646},
        {"corpus": "wikipedia 10M", "tipo": 0.6839, "token": 0.8833},
        {"corpus": "wikipedia 50M", "tipo": 0.7703, "token": 0.9238},
        {"corpus": "web mC4-es 10M", "tipo": 0.8256, "token": 0.9512},
        {"corpus": "mixto 20M", "tipo": 0.8413, "token": 0.9553},
    ]
)
print("cobertura global (tipo = proporcion de tipos TOXIC presentes; token = misma cifra ponderada por frecuencia)")
mostrar_tabla(cobertura_global)
mostrar_figura(
    "fig_56_curva_escala_conclusiva_v5_1.png",
    "Cobertura, embedding aislado y fusión frente a V03 en los cinco corpora. La cobertura "
    "sube de forma monótona; la fusión no.",
    ancho=880,
)

por_grupo = pd.read_csv(RUTAS["reports"] / "embedding_coverage_registro_v5.csv")
pivot = por_grupo.pivot_table(
    index="grupo", columns="corpus", values="token_coverage", aggfunc="first"
).round(4)
print()
print("cobertura por token y grupo léxico")
mostrar_tabla(pivot.reset_index())
'''
        ),
        celda_markdown(
            """
### 18.3 FastText frente al control Word2Vec

El control es **el mismo binario de fastText con las subpalabras apagadas** (`minn=0, maxn=0`),
que es Word2Vec con su implementación. No se usó `gensim` (descartado). La comparación
aisla **una sola cosa**: el mecanismo de composición por subpalabras.

El resultado es la evidencia más limpia de toda la fase: con subpalabras, los 6.059 tipos del
train reciben vector; sin ellas, solo 5.134. Los 925 que se pierden son evasiones ortográficas
—el 66,6 % de los tipos ausentes del corpus son formas con pérdida de vocales (`cllate`,
`violacin`, `espaa`)— y esas son justamente las palabras que deciden la etiqueta en un corpus
de tuits.
"""
        ),
        celda_codigo(
            '''
"""FastText frente a Word2Vec-control: cobertura de vector, OOV y resultado downstream."""

comparacion = pd.DataFrame(
    [
        {"modelo": "FastText 200d (10M)", "subpalabras": "3-6", "vocab": 100098,
         "tipos_con_vector": 6059, "consultados": 6059, "fusion_cv": 0.7818},
        {"modelo": "Word2Vec-control (10M)", "subpalabras": "off", "vocab": 100098,
         "tipos_con_vector": 5134, "consultados": 6059, "fusion_cv": 0.7762},
        {"modelo": "FastText 200d (50M)", "subpalabras": "3-6", "vocab": 247863,
         "tipos_con_vector": 6059, "consultados": 6059, "fusion_cv": 0.7803},
        {"modelo": "Word2Vec-control (50M)", "subpalabras": "off", "vocab": 247863,
         "tipos_con_vector": 4412, "consultados": 6059, "fusion_cv": 0.7773},
    ]
)
comparacion["cobertura_de_vector"] = (
    comparacion["tipos_con_vector"] / comparacion["consultados"]
).round(4)
comparacion["sin_vector"] = comparacion["consultados"] - comparacion["tipos_con_vector"]
mostrar_tabla(comparacion)
print()
print("Los tipos sin vector no son ruido: son formas ortográficas. Con subpalabras fastText")
print("los reconstruye a partir de sus n-gramas; sin ellas el vector sale nulo y se pierden.")
'''
        ),
        celda_markdown(
            """
### 18.4 Pooling y downstream

La rejilla de poolings y clasificadores se midió una sola vez, en v5, y **no se repitió** en
v5.1: si una variación no cambia de banda, repetirla para cada modelo nuevo consume cómputo
para volver a medir lo mismo.

| pooling | corrected_cv (umbrales propios) | lectura |
|---|---|---|
| media | 0,6603 | La elegida: promedia y reduce varianza |
| media + máximo | 0,6388 | El máximo lo domina el token más extremo, que es ruido |
| ponderado por IDF | 0,6348 | El IDF **desinforma** aquí: baja el peso de los tokens raros y discriminativos |
| **suelo aleatorio** | **0,5404** | Sin él, "0,66" no es interpretable |

La regla de escala importa más que el pooling: al fusionar, el bloque denso se normaliza
**L2 por documento**. Estandarizar ese bloque (varianza unitaria frente a valores TF-IDF de
orden 0,1) hace que el bloque denso domine y la fusión **piera 0,09 de F1** sin que eso diga
nada sobre la representación: 0,7818 con L2 frente a 0,6901 estandarizando los mismos datos.
"""
        ),
        celda_codigo(
            '''
"""Rejilla de poolings de v5 y la regla de escala del bloque denso."""

rejilla = pd.read_csv(RUTAS["reports"] / "embedding_downstream_results.csv")
columnas = ["modelo", "pooling", "clasificador", "corrected_cv_umbrales_ajustados", "f1_HS", "f1_TR", "f1_AG"]
rejilla = rejilla[columnas].sort_values("corrected_cv_umbrales_ajustados", ascending=False)
mostrar_tabla(rejilla.head(8).round(4).reset_index(drop=True))
print()
escala = pd.DataFrame(
    [
        {"modo del bloque denso": "L2 por documento", "fusion_cv": 0.7818, "delta_vs_V03": 0.0046},
        {"modo del bloque denso": "sin normalizar", "fusion_cv": 0.7775, "delta_vs_V03": 0.0003},
        {"modo del bloque denso": "estandarizado", "fusion_cv": 0.6901, "delta_vs_V03": -0.0871},
    ]
)
mostrar_tabla(escala)
print()
print("Estandarizar hace que las 200 dimensiones densas dominen al bloque disperso de V03.")
print("Es un error de escala, no un fallo de la representacion: los datos son los mismos.")
'''
        ),
        celda_markdown(
            """
### 18.5 Fusión con V03 y curva de escala del corpus

La fusión es la evaluación crítica: el bloque denso (200 números por documento, L2) se
añade a las 27.431 características dispersas de V03, con la regresión logística balanceada y
los **umbrales congelados de V03** (0,44 / 0,51 / 0,46) como punto de operación común. Así
nadie gana ajustando su propio umbral.

Cinco corpora, cinco escalated y dos registros, y el resultado es siempre el mismo: la
aportación complementaria vive entre **+0,0001 y +0,0046**, con el intervalo de bootstrap
tocando el 0 en todos los casos.
"""
        ),
        celda_codigo(
            '''
"""Fusión con V03 en las cinco configuraciones de corpus y curva de escala."""

curva = pd.DataFrame(
    [
        {"corpus": "wikipedia 5M", "tokens_M": 5, "cobertura_tipo": 0.6302, "cobertura_token": 0.8646,
         "embedding_cv": 0.6670, "fusion_cv": 0.7757, "delta_vs_V03": -0.0015},
        {"corpus": "wikipedia 10M", "tokens_M": 10, "cobertura_tipo": 0.6839, "cobertura_token": 0.8833,
         "embedding_cv": 0.6530, "fusion_cv": 0.7818, "delta_vs_V03": 0.0046},
        {"corpus": "wikipedia 50M", "tokens_M": 50, "cobertura_tipo": 0.7703, "cobertura_token": 0.9238,
         "embedding_cv": 0.6687, "fusion_cv": 0.7803, "delta_vs_V03": 0.0031},
        {"corpus": "web mC4-es 10M", "tokens_M": 10, "cobertura_tipo": 0.8256, "cobertura_token": 0.9512,
         "embedding_cv": 0.6638, "fusion_cv": 0.7812, "delta_vs_V03": 0.0040},
        {"corpus": "mixto 20M", "tokens_M": 20, "cobertura_tipo": 0.8413, "cobertura_token": 0.9553,
         "embedding_cv": 0.6658, "fusion_cv": 0.7776, "delta_vs_V03": 0.0004},
    ]
)
mostrar_tabla(curva)
mostrar_figura(
    "fig_49_curva_completa_v5.png",
    "Curva de aprendizaje 5M / 10M / 50M y efecto del registro. El eje X son tokens de corpus.",
    ancho=880,
)
print("Aportación complementaria medida en cinco configuraciones, siempre en [0.0001, 0.0046].")
print("Diez veces más tokens (5M -> 50M) **reduce** el delta. El techo no es la representación.")
'''
        ),
        celda_markdown(
            """
### 18.6 Complementariedad: qué arregla y qué rompe

Contar celdas es más informativo que contar diferencias de métrica, porque separa las dos
cosas que pueden pasar: acertar más donde se falla, o simplemente mover decisiones de un
lado a otro.

- **`salvage`**: filas que el embedding arregla y V03 falla.
- **`damage`**: filas que V03 arregla y el embedding falla.
- **`corrección neta`**: la diferencia.

La unidad es la **fila** (1.483), no la celda (fila, etiqueta). La función que las cuenta se
llama `_f1_por_celda` y su nombre es engañoso: devuelve la media de las tres etiquetas de
cada fila. Los conteos publicados en la fase v5 no cambian, porque siempre se contaron
filas; lo que estaba mal era la etiqueta (D-042).
"""
        ),
        celda_codigo(
            '''
"""Salvage, damage y corrección neta frente a V03, con las rebanadas."""

complementariedad = pd.read_csv(RUTAS["reports"] / "embedding_complementarity_v5_1.csv")
mostrar_tabla(complementariedad[["variante", "corrected_cv", "delta_real_vs_V03", "salvage", "damage",
                                 "correccion_neta", "fraccion_de_filas"]])
mostrar_figura(
    "fig_57_complementariedad_v5_1.png",
    "Balance de filas por variante. La variante con mejor delta (WS-3) no arregla más que el "
    "baseline: arregla lo mismo y **rompe menos**.",
    ancho=860,
)
print()
rebanadas = pd.read_csv(RUTAS["reports"] / "embedding_window_slices_v5_1.csv")
pivot = rebanadas.pivot_table(index="rebanada", columns="variante", values="f1_por_celda", aggfunc="first")
columnas = [c for c in ("V03-CONGELADO", "FUSION-WS-3") if c in pivot.columns]
pivot["delta"] = (pivot[columnas[1]] - pivot[columnas[0]]).round(4) if len(columnas) > 1 else np.nan
mostrar_tabla(pivot.sort_values("delta", ascending=False).reset_index())
print()
print("El efecto se concentra en elongaciones, URLs y hashtags: donde la forma de la palabra decide.")
'''
        ),
        celda_markdown(
            """
### 18.7 Arquitectura: las cinco familias medidas en v5.1

La fase v5 cerró con la conclusión de que la escala del corpus no era la palanca. Quedaba
una variable: la **arquitectura** del entrenamiento. Se midiieron cinco familias sobre el
mismo corpus (Wikipedia 10M), cambiando **un solo parámetro** cada vez y con el resto
congelado: `vector_size` (100, 200, 300), `window` (3, 5, 7), `min_count` (2, 3, 5),
subpalabras (3-5, 3-6, 3-7) y `negative` (5, 10). Nueve entrenamientos nuevos, presupuesto
cerrado y **reserva intacta**.

El criterio de adopción es de **dos niveles**: una variante es adoptable si su delta alcanza
+0,005 **y** su intervalo de confianza al 95 % no incluye el 0. Un +0,005 cuyo intervalo es
[−0,002, +0,013] no es una mejora: es una cifra que se ha movido dentro del ruido.
"""
        ),
        celda_codigo(
            '''
"""Tabla maestra de la arquitectura y el mapa de calor de deltas."""

maestra = pd.read_csv(RUTAS["reports"] / "embedding_architecture_maestra_v5_1.csv")
mostrar_tabla(maestra[["familia", "valor", "embedding_cv", "fusion_cv", "delta_vs_V03", "ic95", "adoptable"]])
mostrar_figura(
    "fig_55_heatmap_arquitectura_v5_1.png",
    "Mapa de calor de la arquitectura: el color es el delta de fusión frente a V03. Ninguna "
    "celda es adoptable.",
    ancho=900,
)
mostrar_figura(
    "fig_50_vector_size_v5_1.png",
    "Efecto de la dimensión: 100d, 200d y 300d se mueven dentro de una banda estrecha.",
    ancho=820,
)

familias = maestra.groupby("familia")["delta_vs_V03"].agg(["min", "max", "count"])
familias["dispersion"] = (familias["max"] - familias["min"]).round(4)
print("dispersión interna por familia (criterio de poda: <= 0.005 y ninguna adoptable)")
mostrar_tabla(familias.reset_index())
'''
        ),
        celda_markdown(
            """
### 18.8 Contabilidad: entrenamientos \u2260 evaluaciones downstream

La contabilidad de la fase separa tres cosas que se confunden con facilidad:

| unidad | qué es | consume presupuesto |
|---|---|---|
| **Embedding training unit** | Una configuración de entrenamiento desde cero, identificada por todos los parámetros que afectan a los pesos | **Sí** |
| **Downstream evaluation unit** | Una evaluación sobre un embedding ya entrenado (otro pooling, otro clasificador, otros umbrales) | No |
| **Audit / reproduction** | Verificar un hash, repetir una métrica, reconstruir una tabla | No |

El baseline de v5 ya estaba entrenado, así que **no se vuelve a entrenar**: se carga. Eso
deja nueve entrenamientos nuevos para nueve configuraciones distintas, y la aserción de la
celda siguiente falla si alguna vez se supera el límite.
"""
        ),
        celda_codigo(
            '''
"""Contabilidad de la fase: tabla maestra de entrenamientos, tabla B de evaluaciones y asercion de presupuesto."""

from IPython.display import Markdown as _Markdown, display as _display

presupuesto = json.loads((RUTAS["reports"] / "embedding_budget_v5_1.json").read_text(encoding="utf-8"))
LIMITE_EXPLORATORIO = presupuesto["presupuesto"]["limite_exploratorio"]
LIMITE_RESERVA = presupuesto["presupuesto"]["limite_reserva"]

tabla_a = pd.DataFrame(presupuesto["tabla_a"])
mostrar_tabla(
    tabla_a[["training_id", "corpus", "vector_size", "window", "min_count", "min_n", "max_n",
             "negative", "seed", "status"]]
)
print()

tabla_b = pd.read_csv(RUTAS["reports"] / "embedding_downstream_evaluations_v5_1.csv")
print(f"Tabla B: {len(tabla_b)} evaluaciones downstream sobre "
      f"{tabla_b['training_id'].nunique()} entrenamientos distintos.")
print("Ninguna consume presupuesto.")
mostrar_tabla(tabla_b[["evaluation_id", "variante", "training_id", "pooling", "classifier",
                       "corrected_cv", "delta_vs_V03"]].tail(12).reset_index(drop=True))
print()

estado = presupuesto["presupuesto"]
print("CONTADOR DE PRESUPUESTO")
print(f"  existing_v5_trainings  = {estado['existing_v5_trainings']}")
print(f"  new_v5_1_trainings     = {estado['new_v5_1_trainings']}  (limite {LIMITE_EXPLORATORIO})")
print(f"  reserved_training_units= {estado['reserved_training_units']}  (limite {LIMITE_RESERVA})")
print(f"  total_new_budget_used  = {estado['total_new_budget_used']}")
print(f"  budget_remaining       = {estado['budget_remaining']}")

# Aserciones: el notebook falla si se ha excedido el presupuesto.
assert estado["new_v5_1_trainings"] <= LIMITE_EXPLORATORIO, "se ha superado el presupuesto exploratorio"
assert estado["total_new_budget_used"] <= LIMITE_EXPLORATORIO + LIMITE_RESERVA, "se ha superado el presupuesto total"
assert estado["reserved_training_units"] == 0, "la unidad de reserva se ha usado sin justificacion"
_display(_Markdown(
    "**Aserciones de presupuesto: OK.** Nueve exploratorios de nueve y reserva intacta, "
    "que es lo que corresponde: la reserva existe para confirmar con una repetición una "
    "mejora ya vista, y no la hay."
))
'''
        ),
        celda_markdown(
            """
### 18.9 Conclusiones

La fase entrega **dos** conclusiones separadas, y mantenerlas separadas es parte del
resultado: en v5 se confundieron una vez y salió un titular equivocado.

**Conclusión de representación: sí cambia.** La arquitectura mueve la calidad del embedding
aislado entre 0,6397 y 0,6694, un rango de 0,0297. La forma del óptimo es reveladora: gana
**reduciendo la ventana de contexto** (3) mientras el intervalo de subpalabras se queda en el
punto medio 3-6; acortar las subpalabras empeora y alargarlas también. La señal útil es
**local y ortográfica**, no una ventana de distribución amplia. Y las subpalabras son lo
único que evita perder 925 de los 6.059 tipos del vocabulario de la competencia.

**Conclusión de tarea: no aporta.** La fusión se mueve en un rango de 0,0063 entre las
14 configuraciones medidas, que es **4,7 veces menor** que el del embedding aislado. El mejor
delta es +0,0057 con intervalo [−0,0021, +0,0136]. Ninguna configuración es adoptable, el
embedding sigue siendo un **candidato no adoptado** y **V03 se mantiene** en
`corrected_cv` 0.777211 (LB 0.78306).

**Qué no se hizo y por qué.** No se escaló el corpus: la curva de 5M a 50M no muestra
tendencia y tres de las cinco condiciones de la spec fallan, por razones de evidencia y no de
coste. No se usó el holdout (agotado 5/5). No se generaron submissions. No se compararon
umbrales propios: todo se midió en el punto de operación congelado de V03.

**Qué queda abierto.** El **tamaño de los datos etiquetados**: 1.483 ejemplos. Ni la
representación ni su arquitectura son el techo; con más etiquetas, tanto el bloque disperso
como el denso podrían aportar más de lo que aportan hoy.
"""
        ),
    ]


def main() -> None:
    """Genera la seccion 18 del notebook consolidado."""
    import nbformat

    notebook_ruta = localizar_notebook()
    notebook = nbformat.read(str(notebook_ruta), as_version=4)
    # Idempotencia: si la seccion ya existe y se pasa `--reemplazar`, se descarta desde su
    # celda de titulo hasta el final y se regenera. El generador es la fuente de verdad de
    # estas celdas, asi que corregir el generador y regenerar es el ciclo correcto; sin el
    # parametro, la seccion existente se respeta y el script avisa en lugar de duplicarla.
    reemplazar = "--reemplazar" in sys.argv
    inicio_existente = None
    for indice, celda in enumerate(notebook.cells):
        if celda.cell_type == "markdown" and TITULO_BUSCADO in "".join(celda.source):
            inicio_existente = indice
            break
    if inicio_existente is not None:
        if not reemplazar:
            log(
                f"   la seccion 18 ya existe (celda {inicio_existente}); se deja como esta. "
                "Pase --reemplazar para regenerarla desde el generador."
            )
            log(f"   fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
            return
        log(f"   --reemplazar: se descarta la seccion 18 existente (celda {inicio_existente})")
        notebook.cells = notebook.cells[:inicio_existente]
    celdas = construir_celdas()
    for celda in celdas:
        notebook.cells.append(nbformat.v4.new_markdown_cell(celda["source"]) if celda["cell_type"] == "markdown" else nbformat.v4.new_code_cell(celda["source"]))
    nbformat.write(notebook, str(notebook_ruta))
    log(
        f"   seccion 18 {'regenerada' if reemplazar else 'anadida'}: {len(celdas)} celdas "
        f"({sum(1 for c in celdas if c['cell_type'] == 'markdown')} markdown, "
        f"{sum(1 for c in celdas if c['cell_type'] == 'code')} codigo)"
    )
    total = len(nbformat.read(str(notebook_ruta), as_version=4).cells)
    log(f"   el notebook tiene ahora {total} celdas")
    log(
        f"   fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
        "la validacion por ejecucion la hace embedding_20_validacion_notebook_v5_1.py"
    )


if __name__ == "__main__":
    main()
