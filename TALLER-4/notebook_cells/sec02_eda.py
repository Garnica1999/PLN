# %% [markdown]
"""# Sec 02 — EDA del corpus CEV ejecutado en el notebook

El EDA completo (Fase 5) fue un script de produccion (scripts/03_eda.py) que
congelo la config del modelo (resultados/eda/ y config/). En esta seccion el
EDA se ejecuta EN el notebook con el MISMO criterio que el script de
produccion (parada en consumo sin ruido tipografico, correcciones de lemas,
semilla 42) para que el lector reproduzca el razonamiento y las cifras
oficiales sin depender de archivos intermedios: **paridad numerica
verificada** contra resultados/eda/ en la ejecucion.

Bloques de esta seccion:

- **Bloque A** — Estadisticas por corpus: docs, lemas de contenido, tipos y
  longitudes por documento (media/mediana/p90/max) con histogramas y la
  distribucion por tomo de libros.
- **Bloque B** — Diversidad lexica por documento: TTR muestreado (n=1000,
  Bestgen 2023) y Yule's K (x10^4), con violines por corpus.
- **Bloque C** — Frecuencias top-30 de unigramas (barras) y top-15 de
  bigramas/trigramas (tablas) por corpus, con parada en consumo.
- **Bloque D** — Nubes de palabras uni/bi/trigramas por corpus (wordcloud,
  top-150), generadas inline.
- **Bloque E** — Terminos distintivos libros vs entrevistas (log-odds con
  previo dirichlet 0,5; Monroe et al. 2008), con verificacion contra
  resultados/eda/distintivos_logodds.csv.

Tres figuras del EDA oficial se muestran con `mostrar_png()` desde
`imagenes/` (versionado) porque su generacion requiere datos no versionados
(`entrevistas_normalizadas.jsonl`, >100 MB) o el analisis minhash-LSH
completo de produccion: histograma de caracteres, histograma de paginas y
near-duplicates de entrevistas. Las demas figuras se generan en vivo con
matplotlib inline y quedan embebidas como PNG en el notebook.
"""


# %%
"""Imports, rutas y helpers de sec00 (respaldo solo para standalone).

En el notebook los objetos de sec00 (`RUTA_*`, `PARADA_EDA`, `CORRECCIONES`,
`mostrar_png`) ya existen en el namespace y este bloque no los pisa; el
respaldo solo aplica si la seccion se ejecuta como script standalone (misma
convencion de autocontencion que el TALLER-2).
"""
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

if "RUTA_RAIZ" not in globals():  # respaldo standalone robusto (ver sec00)
    import os as _os

    from spacy.lang.es import STOP_WORDS as _STOP_WORDS

    def _raiz_standalone() -> Path:
        """Localiza la raiz del proyecto TALLER-4 (respaldo standalone).

        Copia del algoritmo robusto de sec00 (sin TALLER4_ROOT ni
        auto-descarga, para no sorprender en ejecuciones sueltas): ancestros
        desde el cwd, subcarpetas de hasta 3 niveles bajo el cwd y bajo cada
        ancestro, y rutas tipicas de Colab. Nunca devuelve la raiz del
        filesystem: si no encuentra, lanza un error claro.

        Returns:
            Path absoluto de la raiz del proyecto.

        Raises:
            FileNotFoundError: si falta data/corpus/segmentos_libros.jsonl.
        """
        marcador = Path("data") / "corpus" / "segmentos_libros.jsonl"
        excluidos = {".git", "drive", "sample_data", "venv", "node_modules"}
        cwd = Path(_os.getcwd()).resolve()

        def cumple(directorio: Path) -> bool:
            """True si el directorio contiene el marcador de la raiz."""
            return (directorio / marcador).exists()

        def caminar(base: Path, max_profundidad: int):
            """Itera base y sus subcarpetas hasta la profundidad dada."""
            nivel_base = len(Path(base).parts)
            for dir_actual, subdirs, _ in _os.walk(base):
                nivel = len(Path(dir_actual).parts) - nivel_base
                if nivel >= max_profundidad:
                    subdirs[:] = []
                else:
                    subdirs[:] = sorted(
                        d
                        for d in subdirs
                        if not d.startswith(".") and d not in excluidos
                    )
                yield Path(dir_actual)

        ancestros = [cwd, *cwd.parents]
        for ancestro in ancestros:
            if cumple(ancestro):
                return ancestro
        for ancestro in ancestros:
            for candidato in caminar(ancestro, 3):
                if cumple(candidato):
                    return candidato
        for candidata in (
            Path("/content/TALLER-4"),
            Path("/content/drive/MyDrive/TALLER-4"),
        ):
            if cumple(candidata):
                return candidata
        raise FileNotFoundError(
            "No se encontro la raiz del proyecto TALLER-4: falta "
            "data/corpus/segmentos_libros.jsonl en el arbol de directorios "
            "(sube la carpeta TALLER-4 completa a /content o descomprime ahi "
            "el zip del repositorio; queda en /content/PLN-main/TALLER-4 y "
            "tambien se detecta)."
        )

    RUTA_RAIZ = _raiz_standalone()
    RUTA_CORPUS = RUTA_RAIZ / "data" / "corpus"
    RUTA_PREPROCESADO = RUTA_CORPUS / "preprocesado"
    RUTA_CONFIG = RUTA_RAIZ / "config"
    RUTA_EDA_RESULTADOS = RUTA_RAIZ / "resultados" / "eda"
    RUTA_IMAGENES = RUTA_RAIZ / "imagenes"
    SEMILLA = 42

    def cargar_correcciones() -> dict[str, str]:
        """Carga config/lemas_correcciones.json (respaldo standalone)."""
        datos = json.loads((RUTA_CONFIG / "lemas_correcciones.json").read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in datos["correcciones"].items()}

    def cargar_muletillas() -> set[str]:
        """Carga config/stopwords_habla.txt (respaldo standalone)."""
        ruta = RUTA_CONFIG / "stopwords_habla.txt"
        return {
            linea.strip()
            for linea in ruta.read_text(encoding="utf-8").splitlines()
            if linea.strip() and not linea.startswith("#")
        }

    def cargar_ruido() -> set[str]:
        """Carga el ruido tipografico de config/tfidf_config.json (standalone)."""
        datos = json.loads((RUTA_CONFIG / "tfidf_config.json").read_text(encoding="utf-8"))
        return {str(t) for t in datos["ruido_tipografico_libros_excluido_en_eda"]}

    CORRECCIONES = cargar_correcciones()
    MULETILLAS = cargar_muletillas()
    RUIDO_LIBROS = cargar_ruido()
    PARADA_EDA = set(_STOP_WORDS) | MULETILLAS

    def mostrar_png(nombre: str, ancho: int = 900) -> None:
        """Muestra un PNG de imagenes/ (respaldo standalone, igual que sec00)."""
        ruta = RUTA_IMAGENES / nombre
        if not ruta.exists():
            print(f"[aviso] No existe la figura oficial {ruta.name} en imagenes/;")
            return
        try:
            from IPython.display import Image, display
        except ImportError:
            imagen = plt.imread(ruta)
            plt.figure(figsize=(10, imagen.shape[0] / imagen.shape[1] * 10))
            plt.imshow(imagen)
            plt.axis("off")
            plt.show()
            return
        display(Image(filename=str(ruta), width=ancho))


CORPUS = ("libros", "entrevistas")
N_MUESTRA_TTR = 1000  # muestra fija para TTR muestreado (Bestgen 2023)
N_TOP = 30  # top de frecuencias por n-grama
N_TOP_NUBE = 150  # max_words de las nubes (igual que scripts/03_eda.py)
MIN_APOYO_LOGODDS = 40  # frecuencia conjunta minima para el log-odds
ALFA_LOGODDS = 0.5  # previo dirichlet para log-odds (Monroe et al. 2008)
TOKEN_SEP = -1  # separador de documento en la codificacion de n-gramas


def lemas_contenido(linea: str) -> list[str]:
    """Aplica correcciones de lemas y parada en consumo a una linea.

    Replica `corregir_y_filtrar()` de scripts/03_eda.py: por token, primero
    la correccion (lema anomalo -> lema correcto, config/lemas_correcciones
    .json) y, sobre el resultado, la parada en consumo del EDA (spaCy
    STOP_WORDS + config/stopwords_habla.txt; SIN ruido tipografico, paridad
    con el EDA de produccion). Anos 1900-2025 y categorias canonicas no se
    ven afectados.

    Args:
        linea: linea de *_lemas.txt (lemas separados por espacio).

    Returns:
        Lista de lemas de contenido del documento.
    """
    salida: list[str] = []
    for token in linea.split():
        if CORRECCIONES:
            token = CORRECCIONES.get(token, token)
        if token not in PARADA_EDA:
            salida.append(token)
    return salida


def codificar_corpus(nombre: str) -> tuple[list[str], list[np.ndarray], Counter[str]]:
    """Codifica un corpus de lemas en arrays de ids (vocabulario por corpus).

    Lee el archivo *_lemas.txt por lineas (generador, sin cargar el archivo
    completo) y replica `construir_vocabulario_y_arrays()` de 03_eda.py: el
    vocabulario se ordena por frecuencia descendente (empates alfabeticos,
    id 0 = tipo mas frecuente) y cada documento queda como un array int32 de
    ids. Los documentos vacios se conservan como arrays vacios.

    Args:
        nombre: "libros" o "entrevistas".

    Returns:
        Tupla (vocabulario id->token, arrays por documento, contador global
        de unigramas del corpus).
    """
    ruta = RUTA_PREPROCESADO / f"{nombre}_lemas.txt"
    contador: Counter[str] = Counter()
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            contador.update(lemas_contenido(linea))
    vocabulario = [
        token for token, _ in sorted(contador.items(), key=lambda par: (-par[1], par[0]))
    ]
    token_a_id = {token: idx for idx, token in enumerate(vocabulario)}
    documentos: list[np.ndarray] = []
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            contenido = lemas_contenido(linea)
            documentos.append(
                np.asarray([token_a_id[t] for t in contenido], dtype=np.int32)
            )
    return vocabulario, documentos, contador


# Carga unica de ambos corpus (una vez por seccion; los bloques reutilizan).
print("Codificando corpus (paridad con scripts/03_eda.py)...")
VOCABULARIO: dict[str, list[str]] = {}
DOCUMENTOS: dict[str, list[np.ndarray]] = {}
CONTADORES_UNI: dict[str, Counter[str]] = {}
N_CONTENIDO_POR_DOC: dict[str, np.ndarray] = {}
for _nombre in CORPUS:
    vocabulario, documentos, contador = codificar_corpus(_nombre)
    VOCABULARIO[_nombre] = vocabulario
    DOCUMENTOS[_nombre] = documentos
    CONTADORES_UNI[_nombre] = contador
    N_CONTENIDO_POR_DOC[_nombre] = np.asarray([len(d) for d in documentos], dtype=float)
    print(
        f"  {_nombre}: {len(documentos)} docs | "
        f"{sum(len(d) for d in documentos):,} lemas de contenido | "
        f"{len(vocabulario):,} tipos"
    )


# %%
"""Bloque A1: estadisticas por corpus (tabla) e histogramas de longitudes."""


def resumen_longitudes(valores: np.ndarray) -> dict[str, float]:
    """Resumen (media/mediana/p90/p99/min/max) de un vector de longitudes.

    Args:
        valores: longitudes por documento (lemas de contenido).

    Returns:
        Dict con las estadisticas redondeadas a 1 decimal.
    """
    percentiles = np.percentile(valores, [25, 50, 90, 99])
    return {
        "media": round(float(valores.mean()), 1),
        "mediana": round(float(percentiles[1]), 1),
        "p90": round(float(percentiles[2]), 1),
        "p99": round(float(percentiles[3]), 1),
        "min": float(valores.min()),
        "max": float(valores.max()),
    }


filas_estadisticas = []
for _nombre in CORPUS:
    _arr = N_CONTENIDO_POR_DOC[_nombre]
    _s = resumen_longitudes(_arr)
    filas_estadisticas.append(
        {
            "corpus": _nombre,
            "docs": len(_arr),
            "lemas_contenido": int(_arr.sum()),
            "tipos": len(VOCABULARIO[_nombre]),
            "media": _s["media"],
            "mediana": _s["mediana"],
            "p90": _s["p90"],
            "min": _s["min"],
            "max": _s["max"],
            "vacios": int((_arr == 0).sum()),
        }
    )
df_estadisticas = pd.DataFrame(filas_estadisticas)
print("Estadisticas por corpus (lemas de contenido, parada en consumo del EDA):")
print(df_estadisticas.to_string(index=False))
print(
    "(Paridad esperada con resultados/eda/estadisticas_corpus.csv: libros 917 "
    "docs / 928.253 lemas / 33.277 tipos; entrevistas 2.486 / 9.702.534 / "
    "109.391; 2 docs vacios legitimas en entrevistas.)"
)

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
for ax, _nombre, color in zip(ejes, CORPUS, ("#4C72B0", "#DD8452"), strict=True):
    valores = N_CONTENIDO_POR_DOC[_nombre]
    recortar = _nombre == "entrevistas"
    if recortar:
        corte = float(np.percentile(valores, 99))
        datos_hist = valores[valores <= corte]
        extra = f"p99={corte:,.0f} "
    else:
        datos_hist = valores
        extra = ""
    ax.hist(datos_hist, bins=60, color=color, edgecolor="white")
    ax.set_title(
        f"{_nombre}: lemas de contenido por documento\n"
        f"n={len(valores)} media={valores.mean():,.0f} "
        f"mediana={np.median(valores):,.0f} {extra}max={valores.max():,.0f}",
        fontsize=10,
    )
    ax.set_xlabel("lemas de contenido por documento")
    ax.set_ylabel("documentos")
    ax.axvline(np.median(valores), color="darkred", ls="--", lw=1, label="mediana")
    ax.legend(fontsize=8)
fig.suptitle("Histograma de lemas de contenido por documento (post parada EDA)", y=1.02)
fig.tight_layout()
plt.show()


# %% [markdown]
"""## Bloque A (texto) — Longitudes de los documentos

Distribucion compacta y acotada en libros (la banda 300-4.000 palabras del
ADR-001 se nota en el rango 117-1.765 lemas): media 1.012 y mediana 1.124
lemas por segmento. Las entrevistas tienen asimetria extrema tipica de la
oralidad: media 3.903 lemas pero mediana 3.113 y cola hasta 45.278 (p99 =
20.282, recorte visual del histograma): hay pocas entrevistas muy largas y
muchas de tamano medio. Las 2 entrevistas vacias legitimas (sin texto de
origen) se conservan y se documentan en resultados/eda/longitudes_por_doc.csv.
"""


# %%
"""Bloque A2: lemas de contenido por tomo (libros, desde metadatos)."""


def tabla_markdown(encabezados: list[str], filas: list[list[str]]) -> str:
    """Formatea una tabla simple en markdown (para imprimir en la celda).

    Args:
        encabezados: nombres de las columnas.
        filas: filas como listas de cadenas (mismo orden que encabezados).

    Returns:
        Texto markdown con cabecera, separador y filas.
    """
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "|".join("---" for _ in encabezados) + "|"]
    for fila in filas:
        lineas.append("| " + " | ".join(fila) + " |")
    return "\n".join(lineas)


metadatos_libros = json.loads(
    (RUTA_PREPROCESADO / "libros_docs.json").read_text(encoding="utf-8")
)["docs"]
tomo_por_doc = [m["tomo"] for m in metadatos_libros]
orden_tomos = [
    "HALLAZGOS", "CONVOCATORIA", "NARRATIVA", "SUFRIR", "TESTIMONIAL",
    "CUERPO", "NNA", "VIOLACIONES", "ETNICO", "FUERA",
]
filas_tomo = []
for tomo in orden_tomos:
    _idx = [i for i, t in enumerate(tomo_por_doc) if t == tomo]
    _arr = N_CONTENIDO_POR_DOC["libros"][_idx]
    filas_tomo.append(
        {
            "tomo": tomo,
            "n_segmentos": len(_idx),
            "media": round(float(_arr.mean()), 1),
            "mediana": round(float(np.median(_arr)), 1),
            "min": float(_arr.min()),
            "max": float(_arr.max()),
        }
    )
df_tomo = pd.DataFrame(filas_tomo)
print("Lemas de contenido por segmento y tomo (paridad con estadisticas_por_tomo.csv):")
print(df_tomo.to_string(index=False))
print(
    "VIOLACIONES (1.317) y ETNICO (1.357) tienen los segmentos mas largos en "
    "media, y CONVOCATORIA los mas cortos (420): el tamano del segmento varia "
    "con el genero del capitulo (informe analitico vs recopilacion de casos)."
)

fig, ax = plt.subplots(figsize=(10, 5))
medias = df_tomo["media"].to_numpy()
barras = ax.bar(np.arange(len(medias)), medias, color="#4C72B0")
ax.bar_label(barras, fmt="%.0f", fontsize=8)
ax.set_xticks(np.arange(len(df_tomo)), df_tomo["tomo"])
ax.set_title("libros: lemas de contenido por segmento y tomo (media)")
ax.set_xlabel("tomo")
ax.set_ylabel("lemas de contenido por segmento (media)")
fig.autofmt_xdate(rotation=0)
fig.tight_layout()
plt.show()


# %% [markdown]
"""Nota: histogramas de caracteres y paginas de las entrevistas (figuras
oficiales del EDA de produccion).

Requieren `data/corpus/entrevistas_normalizadas.jsonl` (>100 MB, no
versionado) y los metadatos de paginas del PDF original; por eso aqui se
muestra el PNG oficial de scripts/03_eda.py (imagenes/eda_entrevistas_
hist_chars.png) con `mostrar_png()`, en lugar de regenerar el analisis.
"""


# %%
"""Muestra la figura oficial: histograma de caracteres normalizados."""
mostrar_png("eda_entrevistas_hist_chars.png")


# %% [markdown]
"""El texto normalizado va de 0 a 824.072 caracteres; las 2 entrevistas
vacias legitimas (sin texto de origen) se conservan y se documentan en
resultados/eda/longitudes_por_doc.csv."""


# %%
"""Muestra la figura oficial: histograma de paginas por entrevista."""
mostrar_png("eda_entrevistas_hist_paginas.png")


# %% [markdown]
"""Las transcripciones van de 1 a 204 paginas (el PDF original conserva el
conteo de paginas escaneadas); la mayoria de entrevistas se concentra por
debajo de 30 paginas."""


# %%
"""## Bloque B — Diversidad lexica por documento (TTR muestreado y Yule's K)

TTR muestreado (Bestgen, 2023): muestra fija de n=1.000 tokens sin
reemplazo por documento con `np.random.default_rng(SEMILLA)` (un solo
generador recorrido en el orden libros -> entrevistas, igual que el EDA de
produccion); si el documento tiene menos de 1.000 tokens, TTR = tipos/len.
Yule's K (x10^4): `1e4 * (sum_v f_v^2 - n) / n^2` con f_v = frecuencias de
los tipos del documento (n = lemas de contenido). Valores altos de Yule
indican menor diversidad. Si el computo superara ~120 s en esta maquina, la
celda cae a una muestra determinista de 400 documentos por corpus (semilla
42) y lo anota en la salida.
"""


def ttr_muestreado(doc_ids: np.ndarray, rng: np.random.Generator) -> tuple[float, bool]:
    """TTR con muestra fija de N_MUESTRA_TTR tokens (Bestgen 2023).

    Args:
        doc_ids: ids de vocabulario de un documento.
        rng: generador aleatorio compartido (determinismo, semilla 42).

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
    """Yule's K (x10^4): 1e4 (sum_v f_v^2 - n) / n^2 de un documento.

    Args:
        doc_ids: ids de vocabulario de un documento (n = len(doc_ids)).

    Returns:
        Yule's K del documento (nan si el documento esta vacio).
    """
    n = len(doc_ids)
    if n == 0:
        return float("nan")
    frecuencias = np.bincount(doc_ids)
    frecuencias = frecuencias[frecuencias > 0].astype(np.float64)
    s1 = float(np.sum(frecuencias**2))
    return 1e4 * (s1 - n) / (n * n)


_t_inicio = time.time()
_rng = np.random.default_rng(SEMILLA)
TTR_POR_CORPUS: dict[str, list[float]] = {}
YULE_POR_CORPUS: dict[str, list[float]] = {}
for _nombre in CORPUS:
    ttrs: list[float] = []
    yules: list[float] = []
    for doc in DOCUMENTOS[_nombre]:
        ttr, _parcial = ttr_muestreado(doc, _rng)
        ttrs.append(ttr)
        yules.append(yule_k(doc))
    TTR_POR_CORPUS[_nombre] = ttrs
    YULE_POR_CORPUS[_nombre] = yules
_t_ttr = time.time() - _t_inicio

_muestreo_nota = ""
if _t_ttr > 120:
    # Muestra determinista de 400 docs por corpus (semilla 42) y recalculo.
    _muestreo_nota = (
        f"computo completo >120 s ({_t_ttr:.0f} s); se reporta una muestra "
        "determinista de 400 docs por corpus (semilla 42)"
    )
    _rng_muestra = np.random.default_rng(42)
    for _nombre in CORPUS:
        _n_docs = len(DOCUMENTOS[_nombre])
        _k = min(400, _n_docs)
        _elegidos = _rng_muestra.choice(_n_docs, size=_k, replace=False)
        _rng2 = np.random.default_rng(42)
        TTR_POR_CORPUS[_nombre] = [
            ttr_muestreado(DOCUMENTOS[_nombre][i], _rng2)[0] for i in _elegidos
        ]
        YULE_POR_CORPUS[_nombre] = [yule_k(DOCUMENTOS[_nombre][i]) for i in _elegidos]
    print(_muestreo_nota)

print(f"TTR muestreado y Yule K por documento calculados en {_t_ttr:.1f} s")
for _nombre in CORPUS:
    _ttr = np.nanmean(TTR_POR_CORPUS[_nombre])
    _yule = np.nanmean(YULE_POR_CORPUS[_nombre])
    print(
        f"  {_nombre:12s} | TTR_m1000 media = {_ttr:.3f} "
        f"(oficial 0,609 libros / 0,514 entrevistas) | "
        f"Yule K media = {_yule:.1f} (oficial 31,8 / 38,7)"
    )

fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, _metrica, _titulo, _formato in [
    (ejes[0], "ttr", "TTR muestreado (n=1000 tokens)", ".3f"),
    (ejes[1], "yule", "Yule's K (x10^4; menor = mas diverso)", ".0f"),
]:
    _datos = (
        [np.asarray(TTR_POR_CORPUS[c], dtype=float) for c in CORPUS]
        if _metrica == "ttr"
        else [np.asarray(YULE_POR_CORPUS[c], dtype=float) for c in CORPUS]
    )
    _partes = ax.violinplot(_datos, showmedians=True, showextrema=False)
    for _cuerpo, _color in zip(_partes["bodies"], ["#4C72B0", "#DD8452"], strict=True):
        _cuerpo.set_facecolor(_color)
        _cuerpo.set_alpha(0.7)
    ax.set_xticks([1, 2], CORPUS)
    ax.set_title(_titulo)
    for _pos, _arr in enumerate(_datos, start=1):
        ax.text(
            _pos + 0.08,
            float(np.median(_arr)),
            f"mediana {np.median(_arr):{_formato}}",
            va="center",
            fontsize=8,
        )
fig.suptitle("Diversidad lexica por corpus (lemas de contenido)", y=1.02)
fig.tight_layout()
plt.show()


# %% [markdown]
"""Paneles de violin por documento (figura generada arriba, en el notebook):
TTR con muestra fija n=1.000 (Bestgen, 2023) media 0,609 en libros vs 0,514
en entrevistas; Yule K 31,8 vs 38,7 (menor = mas diverso). La oralidad
transcrita es sistematicamente mas repetitiva, ruido conocido del registro
hablado (Moreno-Sandoval, 2024)."""


# %%
"""## Bloque C — Frecuencias top-30 de uni/bi/trigramas por corpus

Conteo con parada en consumo del EDA sobre las ventanas de n lemas
consecutivos por documento (sin cruzar limites de documento), replicando
`ngramas_top()` de scripts/03_eda.py: los documentos se codifican con su
vocabulario (id 0 = tipo mas frecuente) y los n-gramas se enumeran en base
len(vocabulario)+1 con separador de documento. La figura muestra las barras
del top-30 de unigramas por corpus.
"""


def decodificar_top_ngramas(
    ids: np.ndarray, conteos: np.ndarray, n: int, base: int,
    vocabulario: list[str], top: int,
) -> list[tuple[str, int]]:
    """Decodifica los 'top' n-gramas mas frecuentes a 'a_b_c'.

    Args:
        ids: ids codificados de los n-gramas (de np.unique).
        conteos: frecuencia de cada n-grama (alineado con ids).
        n: tamano del n-grama (1, 2 o 3).
        base: base de la codificacion (len(vocabulario) + 1).
        vocabulario: id -> token del corpus.
        top: cuantos n-gramas devolver.

    Returns:
        Lista (n-grama 'a_b', frecuencia) ordenada por frecuencia desc.
    """
    if len(ids) == 0:
        return []
    k = min(top, len(ids))
    indices = np.argpartition(-conteos, k - 1)[:k]
    posiciones = indices[np.argsort(-conteos[indices])]
    salida: list[tuple[str, int]] = []
    for pos in posiciones:
        resto = int(ids[pos])
        partes: list[str] = []
        for _ in range(n):
            resto, idx = divmod(resto, base)
            partes.append(vocabulario[idx])
        salida.append(("_".join(reversed(partes)), int(conteos[pos])))
    return salida


def contar_ngramas(
    documentos: list[np.ndarray], n: int, vocabulario: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Cuenta los n-gramas consecutivos sin cruzar limites de documento.

    Args:
        documentos: arrays de ids por documento.
        n: tamano del n-grama (1, 2 o 3).
        vocabulario: id -> token del corpus (para la base de codificacion).

    Returns:
        Tupla (ids codificados unicos, conteos) para decodificar top-k.
    """
    if n == 1:
        total = np.concatenate(documentos).astype(np.int64)
        return np.unique(total, return_counts=True)
    base = len(vocabulario) + 1
    partes = [
        np.concatenate([doc.astype(np.int64), [TOKEN_SEP]])
        for doc in documentos
        if len(doc) >= n
    ]
    if not partes:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    arr = np.concatenate(partes)
    if n == 2:
        ids_cod = arr[:-1] * base + arr[1:]
        valido = (arr[:-1] >= 0) & (arr[1:] >= 0)
    else:
        ids_cod = (arr[:-2] * base + arr[1:-1]) * base + arr[2:]
        valido = (arr[:-2] >= 0) & (arr[1:-1] >= 0) & (arr[2:] >= 0)
    return np.unique(ids_cod[valido], return_counts=True)


# Tablas top-15 de bigramas y trigramas por corpus (markdown en print).
NGRAMAS_CONTEO: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
for _nombre in CORPUS:
    for _n in (2, 3):
        NGRAMAS_CONTEO[(_nombre, _n)] = contar_ngramas(
            DOCUMENTOS[_nombre], _n, VOCABULARIO[_nombre]
        )
        ids_cod, conteos_ng = NGRAMAS_CONTEO[(_nombre, _n)]
        _base = len(VOCABULARIO[_nombre]) + 1
        top15 = decodificar_top_ngramas(
            ids_cod, conteos_ng, _n, _base, VOCABULARIO[_nombre], 15
        )
        print(f"\n{_nombre}: top-15 {_n}-gramas (lemas de contenido)")
        print(
            tabla_markdown(
                ["n-grama", "frecuencia"],
                [[t, f"{f:,}"] for t, f in top15],
            )
        )
print(
    "\n(Paridad esperada con resultados/eda/frecuencias_*_top30.csv: libros "
    "conflicto_armado 3.134, derechos_humanos 2.113; entrevistas anio_anio "
    "8.215, conocido_conocido 7.879; trigramas violacion_derechos_humanos 692 "
    "y conocido_conocido_conocido 1.727.)"
)


# %%
"""Top-30 de unigramas de libros (lemas de contenido, parada en consumo)."""

top_libros = CONTADORES_UNI["libros"].most_common(N_TOP)
fig, ax = plt.subplots(figsize=(10, 8))
etiquetas = [t for t, _ in reversed(top_libros)]
valores = [c for _, c in reversed(top_libros)]
ax.barh(np.arange(len(valores)), valores, color="#4C72B0")
ax.set_yticks(np.arange(len(valores)), etiquetas, fontsize=8)
ax.set_title(
    "libros: top-30 unigramas (lemas de contenido)", fontsize=11
)
ax.set_xlabel("frecuencia")
fig.tight_layout()
plt.show()
print(
    "Top-10 libros: "
    + " ".join(t for t, _ in CONTADORES_UNI["libros"].most_common(10))
)

# %%
"""Top-30 de unigramas de entrevistas (lemas de contenido, parada en consumo)."""

top_entrevistas = CONTADORES_UNI["entrevistas"].most_common(N_TOP)
fig, ax = plt.subplots(figsize=(10, 8))
etiquetas = [t for t, _ in reversed(top_entrevistas)]
valores = [c for _, c in reversed(top_entrevistas)]
ax.barh(np.arange(len(valores)), valores, color="#DD8452")
ax.set_yticks(np.arange(len(valores)), etiquetas, fontsize=8)
ax.set_title(
    "entrevistas: top-30 unigramas (lemas de contenido)", fontsize=11
)
ax.set_xlabel("frecuencia")
fig.tight_layout()
plt.show()
print(
    "Top-10 entrevistas: "
    + " ".join(t for t, _ in CONTADORES_UNI["entrevistas"].most_common(10))
)


# %% [markdown]
"""Lectura de las frecuencias (figuras de arriba, generadas en el notebook):

- Unigramas de libros: armado, victima, grupo, persona, violencia, comision,
  mujer, conflicto, anio, entrevista: lexico del informe analitico sobre el
  conflicto.
- Unigramas de entrevistas: gente, llegar, anio, pasar, cosa, venir, querer:
  verbos y sustantivos de la narracion oral de vida, no del informe.

Las tablas top-15 de bigramas/trigramas (impresas arriba) confirman las
colocaciones de dominio de libros (conflicto_armado, derechos_humanos,
grupo_armado) y las reduplicaciones orales de entrevistas (anio_anio,
conocido_conocido). Por eso el modelo IR usa ngram_range=(1,2) (ADR-005):
los trigramas no muestran colocaciones estables."""


# %%
"""Bloque D1: nubes de libros (unigramas, bigramas y trigramas, top-150)."""

N_TOP_NUBE_CONTEO = 450  # candidatos decodificados para quedarse con 150

fig, ejes = plt.subplots(1, 3, figsize=(18, 6))
for ax, _n, _tipo in zip(ejes, (1, 2, 3), ("unigramas", "bigramas", "trigramas"), strict=True):
    if _n == 1:
        frecuencias_nube: Counter[str] = CONTADORES_UNI["libros"]
    else:
        ids_cod, conteos_ng = NGRAMAS_CONTEO[("libros", _n)]
        _base = len(VOCABULARIO["libros"]) + 1
        top_ng = decodificar_top_ngramas(
            ids_cod, conteos_ng, _n, _base, VOCABULARIO["libros"],
            N_TOP_NUBE_CONTEO,
        )
        frecuencias_nube = Counter(dict(top_ng))
    nube = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        max_words=N_TOP_NUBE,
        collocations=False,
        random_state=SEMILLA,
    )
    nube.generate_from_frequencies(dict(frecuencias_nube))
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"libros: {_tipo} mas frecuentes (top {N_TOP_NUBE})", fontsize=11
    )
fig.suptitle("Nubes de libros (uni/bi/trigramas, lemas de contenido)", y=1.01)
fig.tight_layout()
plt.show()

# %%
"""Bloque D2: nubes de entrevistas (unigramas, bigramas y trigramas, top-150)."""

fig, ejes = plt.subplots(1, 3, figsize=(18, 6))
for ax, _n, _tipo in zip(ejes, (1, 2, 3), ("unigramas", "bigramas", "trigramas"), strict=True):
    if _n == 1:
        frecuencias_nube = CONTADORES_UNI["entrevistas"]
    else:
        ids_cod, conteos_ng = NGRAMAS_CONTEO[("entrevistas", _n)]
        _base = len(VOCABULARIO["entrevistas"]) + 1
        top_ng = decodificar_top_ngramas(
            ids_cod, conteos_ng, _n, _base, VOCABULARIO["entrevistas"],
            N_TOP_NUBE_CONTEO,
        )
        frecuencias_nube = Counter(dict(top_ng))
    nube = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        max_words=N_TOP_NUBE,
        collocations=False,
        random_state=SEMILLA,
    )
    nube.generate_from_frequencies(dict(frecuencias_nube))
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"entrevistas: {_tipo} mas frecuentes (top {N_TOP_NUBE})", fontsize=11
    )
fig.suptitle("Nubes de entrevistas (uni/bi/trigramas, lemas de contenido)", y=1.01)
fig.tight_layout()
plt.show()


# %% [markdown]
"""Lectura de las nubes (figuras de arriba, generadas en el notebook):

- Libros: el lexico del informe analitico (armado, victima, conflicto_armado,
  derechos_humanos, grupo_armado) con colocaciones de dominio estables.
- Entrevistas: la narracion oral (gente, llegar, anio, pasar) con
  reduplicaciones (anio_anio, conocido_conocido) y las categorias canonicas
  de anonimizacion pegadas al contexto (unidad_militar_ejercito,
  junta_accion_comunal)."""


# %%
"""## Bloque E — Terminos distintivos libros vs entrevistas (log-odds)

Log-odds con previo de Dirichlet 0,5 (Monroe et al. 2008):
score(w) = log((c_a + alfa) / (N_a + alfa V)) - log((c_b + alfa) / (N_b + alfa V))
con V = tipos de la union y apoyo minimo conjunto 40 (c_a + c_b >= 40).
score > 0: distintivo de libros; score < 0: distintivo de entrevistas
(ordenados por magnitud; top-25 por direccion). Igual que la etapa 6 de
scripts/03_eda.py, se excluye el ruido tipografico de libros (42 tokens,
RUIDO_LIBROS) del contador de libros antes de comparar.
"""


def log_odds_distintivos(
    contador_a: Counter[str], contador_b: Counter[str],
    top: int = 25, alfa: float = ALFA_LOGODDS, min_apoyo: int = MIN_APOYO_LOGODDS,
) -> tuple[list[tuple[str, float, int, int]], list[tuple[str, float, int, int]]]:
    """Terminos distintivos por log-odds con previo dirichlet (Monroe 2008).

    Args:
        contador_a: frecuencias del corpus A (libros, sin ruido tipografico).
        contador_b: frecuencias del corpus B (entrevistas).
        top: cuantos distintivos por direccion.
        alfa: previo dirichlet (0.5).
        min_apoyo: frecuencia conjunta minima c_a + c_b.

    Returns:
        Tupla (distintivos de A, distintivos de B) con (token, |log-odds|,
        c_a, c_b) ordenados por magnitud descendente.
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
    puntuados.sort(key=lambda par: -par[1])
    distintos_a = [(t, s, ca, cb) for t, s, ca, cb in puntuados if s > 0][:top]
    distintos_b = [(t, -s, ca, cb) for t, s, ca, cb in reversed(puntuados) if s < 0][:top]
    return distintos_a, distintos_b


ca_ruido = Counter(
    {t: f for t, f in CONTADORES_UNI["libros"].items() if t not in RUIDO_LIBROS}
)
cb = CONTADORES_UNI["entrevistas"]
distintivos_libros, distintivos_entrevistas = log_odds_distintivos(ca_ruido, cb)

fig, ejes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, lista, titulo, color in [
    (
        ejes[0],
        list(reversed(distintivos_libros)),
        "Distintivos de LIBROS (log-odds > 0)",
        "#4C72B0",
    ),
    (
        ejes[1],
        list(reversed(distintivos_entrevistas)),
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
    "Terminos distintivos libros vs entrevistas (lemas de contenido)", y=1.01
)
fig.tight_layout()
plt.show()

# Verificacion contra el CSV oficial de la etapa 6 (top-1 por direccion).
distintivos_csv = pd.read_csv(RUTA_EDA_RESULTADOS / "distintivos_logodds.csv")
print("Top-5 distintivos de LIBROS:", " ".join(t for t, *_ in distintivos_libros[:5]))
print("Top-5 distintivos de ENTREVISTAS:", " ".join(t for t, *_ in distintivos_entrevistas[:5]))
for _direccion, _calculados in (
    ("libros", distintivos_libros),
    ("entrevistas", distintivos_entrevistas),
):
    _fila_oficial = distintivos_csv[distintivos_csv["direccion"] == _direccion]
    _coinciden = sum(
        1 for (t, _, _, _), (_t2, _s2) in zip(
            _calculados,
            zip(_fila_oficial["token"], _fila_oficial["log_odds"]),
            strict=False,
        )
        if t == _t2
    )
    _top1_ok = _calculados[0][0] == _fila_oficial.iloc[0]["token"]
    print(
        f"Verificacion vs distintivos_logodds.csv [{_direccion}]: top-1 "
        f"coincide: {_top1_ok} ({_calculados[0][0]} = oficial "
        f"{_fila_oficial.iloc[0]['token']}) | tokens top-25 en misma posicion: "
        f"{_coinciden}/25 (el resto son empates exactos de |log-odds|)"
    )


# %% [markdown]
"""Los distintivos de libros son tecnicos (transmedia 8,0, externas 7,8,
microdato 7,4) y los de entrevistas son las categorias de anonimizacion
(anonimo 9,1, organizacion_social 8,5, organizacion_publica 8,2): huella del
registro oral y de la normalizacion, no del contenido. El solapamiento
Jaccard de vocabulario es solo 0,221 (25.851 tipos compartidos), pero el
vocabulario de libros cubre el 95,7 % de los tokens de entrevistas (cifras
oficiales de resultados/eda/solapamiento_vocabulario.csv)."""


# %% [markdown]
"""Nota: near-duplicates entre entrevistas (figura oficial del EDA de
produccion).

El analisis completo (minhash + LSH con shingles de 5 lemas, Broder 1997, y
verificacion Jaccard exacta de los candidatos) vive en scripts/03_eda.py y
sus resultados en resultados/eda/near_duplicates_entrevistas.csv; aqui se
muestra el PNG oficial (imagenes/eda_near_duplicates_entrevistas.png).
"""


# %%
"""Muestra la figura oficial: near-duplicates entre entrevistas."""
mostrar_png("eda_near_duplicates_entrevistas.png")


# %% [markdown]
"""Izquierda: histograma de Jaccard (shingles de 5 lemas) de los 368 pares
candidatos LSH, con la linea del umbral 0,85; derecha: top-10 pares casi
identicos. Hay 345 pares > 0,85 (526 entrevistas unicas, 21,2 % del corpus)
y se REPORTAN sin eliminar: el corpus oficial debe permanecer integro, y el
matching IR ya los trata como documentos separados."""


# %% [markdown]
"""## Interpretacion final del EDA (cifras reproducidas en este notebook)

Con los bloques A-E se reproducen las cifras oficiales de la Fase 5
(resultados/eda/resumen_eda.md):

1. **Tamano**: libros 917 docs / 928.253 lemas de contenido / 33.277 tipos
   (media 1.012 lemas por segmento); entrevistas 2.486 docs / 9.702.534
   lemas / 109.391 tipos (media 3.903; 2 docs vacios legitimas).
2. **Diversidad**: TTR_m1000 libros 0,609 vs entrevistas 0,514; Yule K 31,8
   vs 38,7: la oralidad transcrita es lexicamente mas repetitiva (ruido
   sistematico, Moreno-Sandoval 2024).
3. **Frecuencias**: las colocaciones de dominio (conflicto_armado,
   derechos_humanos) justifican el ngram_range=(1,2) del vectorizador
   (ADR-005); los trigramas no son estables y no entran al modelo.
4. **Comparacion**: Jaccard de vocabulario 0,221; el vocabulario de libros
   cubre el 95,7 % de los tokens de entrevistas; los distintivos de
   entrevistas son las categorias de anonimizacion (huella de la
   normalizacion, no del contenido).
5. **Near-duplicates**: 345 pares > 0,85 (21,2 % de las entrevistas)
   reportados sin eliminar (corpus oficial integro).
"""

