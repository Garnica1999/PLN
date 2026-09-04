# %% [markdown]
"""# Sec 02 — EDA ligero en el notebook

El EDA completo (Fase 5) fue un script de produccion (scripts/03_eda.py) que
congelo la config del modelo; sus figuras pesadas (nubes, violines,
near-duplicates) viven en `imagenes/` y se EMBEBEN mas abajo como markdown
con rutas relativas al notebook (`../imagenes/eda_*.png`).

En esta seccion se ejecutan EN el notebook analisis ligeros sobre los mismos
insumos (lemas ya procesados), para que el lector pueda reproducir el
razonamiento sin depender de archivos intermedios:

1. Histograma del numero de lemas por documento, por corpus (con la cola de
   entrevistas recortada al p99, igual que el EDA).
2. Top-20 de unigramas por corpus con la parada EN CONSUMO de la Fase 6
   (mismo criterio del modelo IR: spacy STOP_WORDS + muletillas + ruido +
   62 correcciones).
3. Dos figuras adicionales desde CSVs baratos de resultados/eda/:
   longitudes por documento (lemas de contenido) y diversidad lexica (TTR
   muestreado y Yule K).
4. Embebido de las 14 figuras pesadas del EDA con su interpretacion y la
   tabla de metricas por fase de resultados/metricas.json.
"""

# %%
"""Imports, rutas y helpers de sec00 (respaldo solo para standalone)."""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "RUTA_RAIZ" not in globals():  # respaldo standalone (ver sec00_setup)
    import os as _os

    _candidato = Path(_os.getcwd()).resolve()
    while _candidato != _candidato.parent:
        if (_candidato / "data" / "corpus" / "segmentos_libros.jsonl").exists():
            break
        _candidato = _candidato.parent
    RUTA_RAIZ = _candidato
    RUTA_CORPUS = RUTA_RAIZ / "data" / "corpus"
    RUTA_PREPROCESADO = RUTA_CORPUS / "preprocesado"
    RUTA_CONFIG = RUTA_RAIZ / "config"
    RUTA_EXPERIMENTOS = RUTA_RAIZ / "experimentos"
    RUTA_RESULTADOS = RUTA_RAIZ / "resultados"
    RUTA_EDA_RESULTADOS = RUTA_RESULTADOS / "eda"

if "cargar_lemas" not in globals():  # respaldo standalone (igual logica)
    def cargar_lemas(corpus: str) -> list[str]:
        ruta = RUTA_PREPROCESADO / f"{corpus}_lemas.txt"
        return [linea.rstrip("\n") for linea in ruta.open(encoding="utf-8")]

if "aplicar_parada" not in globals():
    import importlib
    import sys as _sys

    def aplicar_parada(linea_lemas: str) -> str:
        if "experimento_1_tfidf_base" not in _sys.modules:
            _sys.path.insert(0, str(RUTA_EXPERIMENTOS))
            importlib.import_module("experimento_1_tfidf_base")
        exp = _sys.modules["experimento_1_tfidf_base"]
        parada = exp.construir_parada(incluir_muletillas=True)
        correcciones = exp.cargar_correcciones()
        return exp.limpiar_linea(linea_lemas, parada, correcciones)

# %%
"""Histograma del numero de lemas por documento (por corpus)."""

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
for ax, corpus, color in zip(
    ejes, ("libros", "entrevistas"), ("#4C72B0", "#DD8452"), strict=True
):
    n_por_doc = [len(linea.split()) for linea in cargar_lemas(corpus)]
    valores = np.asarray(n_por_doc, dtype=float)
    recortar = corpus == "entrevistas"
    if recortar:
        corte = float(np.percentile(valores, 99))
        datos_hist = valores[valores <= corte]
        extra = f"p99={corte:,.0f} "
    else:
        datos_hist = valores
        extra = ""
    ax.hist(datos_hist, bins=60, color=color, edgecolor="white")
    ax.set_title(
        f"{corpus}: lemas por documento (salida fase 4)\n"
        f"n={len(valores)} media={valores.mean():,.0f} "
        f"mediana={np.median(valores):,.0f} {extra}max={valores.max():,.0f}",
        fontsize=10,
    )
    ax.set_xlabel("lemas por documento")
    ax.set_ylabel("documentos")
    ax.axvline(np.median(valores), color="darkred", ls="--", lw=1, label="mediana")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %%
"""Top-20 de unigramas por corpus (lemas de contenido, parada en consumo)."""

fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
for ax, corpus, color in zip(
    ejes, ("libros", "entrevistas"), ("#4C72B0", "#DD8452"), strict=True
):
    contador: Counter[str] = Counter()
    for linea in cargar_lemas(corpus):
        limpia = aplicar_parada(linea)
        if limpia:
            contador.update(limpia.split())
    top = contador.most_common(20)
    etiquetas = [t for t, _ in reversed(top)]
    valores = [c for _, c in reversed(top)]
    ax.barh(np.arange(len(valores)), valores, color=color)
    ax.set_yticks(np.arange(len(valores)), etiquetas, fontsize=8)
    ax.set_title(f"{corpus}: top-20 unigramas (lemas de contenido)", fontsize=10)
    ax.set_xlabel("frecuencia")
fig.tight_layout()
plt.show()

print("Top-10 libros      :", " ".join(t for t, _ in Counter(
    t
    for linea in cargar_lemas("libros")
    for t in aplicar_parada(linea).split()
).most_common(10)))
print("Top-10 entrevistas :", " ".join(t for t, _ in Counter(
    t
    for linea in cargar_lemas("entrevistas")
    for t in aplicar_parada(linea).split()
).most_common(10)))

# %%
"""Dos figuras adicionales desde CSVs baratos de resultados/eda/."""

longitudes = pd.read_csv(RUTA_EDA_RESULTADOS / "longitudes_por_doc.csv")
diversidad = pd.read_csv(RUTA_EDA_RESULTADOS / "diversidad_lexica.csv")

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
for ax, corpus, color in zip(
    ejes, ("libros", "entrevistas"), ("#4C72B0", "#DD8452"), strict=True
):
    sub = longitudes[longitudes["corpus"] == corpus]["n_lemas_contenido_eda"]
    ax.hist(sub, bins=60, color=color, edgecolor="white")
    ax.set_title(
        f"{corpus}: lemas de contenido (EDA)\nn={len(sub)} "
        f"media={sub.mean():,.0f} mediana={sub.median():,.0f}",
        fontsize=10,
    )
    ax.set_xlabel("lemas de contenido por documento")
fig.tight_layout()
plt.show()

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metrica, titulo, formato in [
    (ejes[0], "ttr_m1000", "TTR muestreado (n=1000 tokens)", ".3f"),
    (ejes[1], "yule_k", "Yule K (menor = mas diverso)", ".0f"),
]:
    datos = [
        diversidad.loc[diversidad["corpus"] == c, metrica].dropna().to_numpy()
        for c in ("libros", "entrevistas")
    ]
    violines = ax.violinplot(datos, showmedians=True, showextrema=False)
    for cuerpo, color in zip(
        violines["bodies"], ["#4C72B0", "#DD8452"], strict=True
    ):
        cuerpo.set_facecolor(color)
        cuerpo.set_alpha(0.7)
    ax.set_xticks([1, 2], ("libros", "entrevistas"))
    ax.set_title(titulo)
    for pos, arr in enumerate(datos, start=1):
        ax.text(
            pos + 0.08,
            float(np.median(arr)),
            f"mediana {np.median(arr):{formato}}",
            va="center",
            fontsize=8,
        )
fig.suptitle("Diversidad lexica por corpus (lemas de contenido)", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
"""## Figuras pesadas del EDA (Fase 5) — I. Estadisticas y longitudes

Figuras generadas por scripts/03_eda.py y embebidas desde `../imagenes/`
(ruta relativa al notebook, que vive en `notebooks/`).

### Longitudes de los documentos

**Figura 1 — Libros: histograma de lemas de contenido por segmento**

![Histograma de lemas por segmento de libro](../imagenes/eda_libros_hist_lemas.png)

Distribucion compacta y acotada (la banda 300-4.000 palabras del ADR-001 se
nota en el rango 117-1.765 lemas): media 1.012 y mediana 1.124 lemas por
segmento; los histogramas de la Fase 5 usan los lemas de contenido (post
parada), por eso la media difiere de la de la seccion anterior (salida cruda
de la fase 4).

**Figura 2 — Libros: lemas de contenido por segmento y tomo**

![Boxplot de lemas por tomo](../imagenes/eda_libros_lemas_por_tomo.png)

VIOLACIONES (1.317) y ETNICO (1.357) tienen los segmentos mas largos en media,
y CONVOCATORIA los mas cortos (420): el tamano del segmento varia con el
genero del capitulo (informe analitico vs recopilacion de casos).

**Figura 3 — Entrevistas: histograma de lemas de contenido**

![Histograma de lemas por entrevista](../imagenes/eda_entrevistas_hist_lemas.png)

Asimetria extrema tipica de la oralidad: media 3.903 lemas pero mediana 3.113
y cola hasta 45.278 (p99 = 20.282, recorte visual): hay pocas entrevistas
muy largas y muchas de tamano medio.

**Figura 4 — Entrevistas: histograma de caracteres normalizados**

![Histograma de caracteres por entrevista](../imagenes/eda_entrevistas_hist_chars.png)

El texto normalizado va de 0 a 824.072 caracteres; las 2 entrevistas vacias
legitimas (sin texto de origen) se conservan y se documentan en
`resultados/eda/longitudes_por_doc.csv`.

**Figura 5 — Entrevistas: histograma de paginas por entrevista**

![Histograma de paginas por entrevista](../imagenes/eda_entrevistas_hist_paginas.png)

Las transcripciones van de 1 a 204 paginas (el PDF original conserva el
conteo de paginas escaneadas); la mayoria de entrevistas se concentra por
debajo de 30 paginas.
"""

# %% [markdown]
"""## II. Diversidad lexica

**Figura 6 — Violines de TTR muestreado y Yule K por corpus**

![Diversidad lexica por corpus](../imagenes/eda_diversidad_lexica.png)

Paneles de violin por documento: TTR con muestra fija n=1.000 (Bestgen,
2023): media 0,609 en libros vs 0,514 en entrevistas; Yule K 31,8 vs 38,7
(menor = mas diverso). La oralidad transcrita es sistematicamente mas
repetitiva, ruido conocido del registro hablado (Moreno-Sandoval, 2024).
"""

# %% [markdown]
"""## III. Nubes uni/bi/trigramas (lemas de contenido, top-150)

**Figuras 7-9 — Libros: nubes de unigramas, bigramas y trigramas**

![Nube de unigramas de libros](../imagenes/eda_libros_nube_unigramas.png)

Unigramas de libros: armado, victima, grupo, persona, violencia, comision:
lexico del informe analitico sobre el conflicto.

![Nube de bigramas de libros](../imagenes/eda_libros_nube_bigramas.png)

Bigramas de libros: conflicto armado, derechos humanos, grupo armado,
fuerza publica: colocaciones de dominio estables (por eso el modelo IR usa
ngram_range=(1,2), ADR-005).

![Nube de trigramas de libros](../imagenes/eda_libros_nube_trigramas.png)

Trigramas de libros: violacion de derechos humanos, nina nino adolescente:
colocaciones mas largas, dispersas (el EDA no encontro trigramas estables y
por eso no entran al vectorizador).

**Figuras 10-12 — Entrevistas: nubes de unigramas, bigramas y trigramas**

![Nube de unigramas de entrevistas](../imagenes/eda_entrevistas_nube_unigramas.png)

Unigramas de entrevistas: gente, llegar, ano, pasar, cosa, venir, querer:
verbos y sustantivos de la narracion oral de vida, no del informe.

![Nube de bigramas de entrevistas](../imagenes/eda_entrevistas_nube_bigramas.png)

Bigramas de entrevistas: ano_ano y conocido_conocido (reduplicaciones orales)
y grupo_armado / conflicto_armado: el lexico de dominio aparece, pero
contaminado por la repeticion del habla.

![Nube de trigramas de entrevistas](../imagenes/eda_entrevistas_nube_trigramas.png)

Trigramas de entrevistas: conocido_conocido_conocido, junta_accion_comunal y
unidad_militar_ejercito (categorias de anonimizacion pegadas al contexto).
"""

# %% [markdown]
"""## IV. Comparacion lexica y near-duplicates

**Figura 13 — Terminos distintivos libros vs entrevistas (log-odds)**

![Distintivos por log-odds](../imagenes/eda_comparacion_distintivos_logodds.png)

Log-odds con previo de Dirichlet 0,5: los distintivos de libros son tecnicos
(transmedia 8,0, externas 7,8, microdato 7,4) y los de entrevistas son las
categorias de anonimizacion (anonimo 9,1, organizacion_social 8,5,
organizacion_publica 8,2): huella del registro oral y de la normalizacion,
no del contenido. El solapamiento Jaccard de vocabulario es solo 0,221
(25.851 tipos compartidos), pero el vocabulario de libros cubre el 95,7 % de
los tokens de entrevistas.

**Figura 14 — Near-duplicates entre entrevistas**

![Near duplicates de entrevistas](../imagenes/eda_near_duplicates_entrevistas.png)

Izquierda: histograma de Jaccard (shingles de 5 lemas) de los 368 pares
candidatos LSH, con la linea del umbral 0,85; derecha: top-10 pares casi
identicos. Hay 345 pares > 0,85 (526 entrevistas unicas, 21,2 % del corpus)
y se REPORTAN sin eliminar: el corpus oficial debe permanecer integro, y el
matching IR ya los trata como documentos separados.
"""

# %% [markdown]
"""## V. Metricas por fase del proyecto (resultados/metricas.json)

Resumen acumulativo del proyecto por fase (fuente:
`resultados/metricas.json`, Fases 3 a 6):

| Fase | Script | Alcance | Resultado clave |
|---|---|---|---|
| 3 | `scripts/01_extraer_libros.py` | Extraccion de los 10 tomos PDF | 917 segmentos / 2.271.493 palabras; mapeo TOC 91-100 %, cobertura 100 %, 0 solapes |
| 4 | `scripts/02_preprocesar.py` | Normalizacion + spaCy (2.486 entrevistas) | 200,5M -> 188,8M chars (-5,8 %); 197.630 anonimizados a 8 categorias; 40,3M tokens -> 10,9M lemas (27,1 %) |
| 5 | `scripts/03_eda.py` | EDA y congelacion de config | TTR 0,609/0,514; Yule K 31,8/38,7; Jaccard 0,221; 345 near-dups; config tf-idf congelada |
| 6 | `experimentos/experimento_1_tfidf_base.py` + `scripts/04_evaluar_muestra.py` | IR TF-IDF + evaluacion de muestra | 118.748 features; 2.484/2.486 queries con match; P@1 estricto 0,533 / aceptable 1,0 (muestra 30) |

La Fase 6 se ejecuta y discute en la seccion siguiente (Sec 03).
"""
