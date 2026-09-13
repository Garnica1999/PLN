# %% [markdown]
"""# Sec 03c — Comparacion de corpus: heatmap entrevistas <-> libros

El punto 5 del taller pide describir que entrevistas se vinculan con que
libros. Esta seccion lo hace con dos salidas complementarias:

## 1. Heatmap 10 x 10 (grupo de entrevistas x tomo de libro)

- **Filas (grupos)**: las 2.486 entrevistas agrupadas por el tomo del
  top-1 de su ranking **BM25** (Sec 03b). Es una asignacion no supervisada
  de cada entrevista a un tomo del informe.
- **Columnas**: los 10 tomos de los libros.
- **Valor de la celda (g, T)**: similitud coseno TF-IDF **media** entre
  todas las entrevistas del grupo g y **todos** los segmentos del tomo T,
  calculada sobre la matriz del modelo base de la Sec 03 (misma que genera
  el ranking TF-IDF; no se re-entrena nada).
- **Normalizacion**: NO se normaliza por el numero de segmentos del tomo
  porque la media por par (entrevista, segmento) ya es intensiva respecto
  al tamanio: la celda mide afinidad lexica media, no la masa total de
  similitud. La masa (suma) si creceria con el numero de segmentos y
  confundiria afinidad con volumen; el efecto del tamanio del tomo ya se
  ve en la distribucion del top-1 (Sec 03). El CSV guarda ademas
  `n_entrevistas` por grupo para leer el peso de cada fila.
- La **diagonal** (grupo g contra el tomo g) mide la coherencia entre la
  asignacion BM25 y la similitud TF-IDF; las celdas **fuera de la
  diagonal** muestran las afinidades cruzadas (que grupos se parecen a
  otros tomos).

## 2. Cuadro de vinculos entrevista -> segmento (por tomo)

Para cada tomo, las 3 mejores parejas entrevista -> segmento segun el
score BM25 (id_doc, id_segmento, score, titulo del segmento) mas los
fragmentos lematizados de la pareja: la evidencia concreta detras del
heatmap, que responde "que entrevistas se vinculan a textos de cada
libro". Esquema del CSV: tomo, puesto, id_doc, id_segmento, score_bm25,
titulo_segmento, fragmento_entrevista, fragmento_libro (los fragmentos son
los primeros 200 caracteres de los textos lematizados limpios de esa
pareja; aporte de Erika Rivera, Taller_4_IR_CEV_V2).

## 3. Heatmap top-25 entrevistas x 10 tomos (maximo BM25)

Vista por DOCUMENTO adaptada del aporte de Erika Rivera
(Taller_4_IR_CEV_V2): filas = las 25 entrevistas con mayor suma de scores
BM25, columnas = los 10 tomos en orden oficial, valor de la celda = score
BM25 MAXIMO de esa entrevista con cualquier segmento del tomo. A
diferencia del heatmap 10 x 10 (filas = grupos agregados, celda = coseno
TF-IDF medio), aqui se conserva la intensidad individual de cada
entrevista; usa `contexto_base["matriz_bm25"]` (float32) de la Sec 03b y
se omite con aviso si esa matriz no esta disponible.

Salidas: `resultados/heatmap_tomos.csv`,
`imagenes/heatmap_entrevistas_libros.png`,
`resultados/vinculos_entrevistas_libros.csv`,
`resultados/heatmap_top25.csv` e
`imagenes/heatmap_top25_entrevistas_tomos.png` (todas son salidas del
notebook; ningun insumo). La interpretacion esta al final de la seccion.
"""


# %%
"""Heatmap entrevistas (grupo = tomo del top-1 BM25) x tomo de libro."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

contexto_base = globals().get("contexto_base")
ranking_bm25 = globals().get("ranking_bm25")
RUTA_RESULTADOS = globals().get("RUTA_RESULTADOS")
RUTA_IMAGENES = globals().get("RUTA_IMAGENES")
if any(
    valor is None
    for valor in (contexto_base, ranking_bm25, RUTA_RESULTADOS, RUTA_IMAGENES)
):
    raise RuntimeError(
        "La Sec 03c reutiliza el contexto de la Sec 03 y el ranking BM25 de "
        "la Sec 03b (mas las rutas de la Sec 00): ejecuta esas secciones "
        "antes de esta."
    )

TOMOS_ORDEN = [
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
]


def construir_matriz_heatmap(
    similitudes: np.ndarray,
    ids_queries: list[str],
    grupo_por_query: dict[str, str],
    tomo_por_segmento: list[str],
    tomos_orden: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz 10 x 10 de coseno TF-IDF medio entre grupos de entrevistas y tomos.

    Para cada grupo g (entrevistas cuyo top-1 BM25 cae en el tomo g) y cada
    tomo T se promedia la similitud coseno TF-IDF (matriz de la Sec 03)
    entre todas las entrevistas del grupo y todos los segmentos de T. La
    media por par (entrevista, segmento) es intensiva respecto al tamanio
    del tomo, por lo que no se normaliza por el numero de segmentos
    (documentado en el encabezado): la celda mide afinidad lexica media,
    no masa total de similitud. Un grupo sin entrevistas produce NaN.

    Args:
        similitudes: matriz (n_queries x n_docs) del TF-IDF base.
        ids_queries: id_doc de cada fila de la matriz, en orden.
        grupo_por_query: id_doc -> tomo del top-1 BM25 (asignacion).
        tomo_por_segmento: tomo de cada columna (segmento), en orden.
        tomos_orden: orden canonico de los 10 tomos.

    Returns:
        Tupla (matriz (10 x 10) de medias, n_entrevistas por grupo (10,)).
    """
    indices_tomo = {
        tomo: [j for j, etiqueta in enumerate(tomo_por_segmento) if etiqueta == tomo]
        for tomo in tomos_orden
    }
    filas: list[list[float]] = []
    n_entrevistas: list[int] = []
    for grupo in tomos_orden:
        indices_grupo = [
            i for i, id_query in enumerate(ids_queries)
            if grupo_por_query.get(id_query) == grupo
        ]
        n_entrevistas.append(len(indices_grupo))
        valores: list[float] = []
        for tomo in tomos_orden:
            if not indices_grupo:
                valores.append(float("nan"))
                continue
            bloque = similitudes[np.ix_(indices_grupo, indices_tomo[tomo])]
            valores.append(float(bloque.mean()))
        filas.append(valores)
    return np.asarray(filas, dtype=np.float64), np.asarray(n_entrevistas, dtype=int)


top1_bm25 = ranking_bm25[ranking_bm25["rango"] == 1]
grupo_por_query = dict(zip(top1_bm25["id_doc"], top1_bm25["tomo"], strict=True))
tomo_por_segmento = [registro["tomo"] for registro in contexto_base["docs_libros"]]
matriz_heatmap, n_entrevistas_grupo = construir_matriz_heatmap(
    contexto_base["similitudes"],
    contexto_base["ids_queries"],
    grupo_por_query,
    tomo_por_segmento,
    TOMOS_ORDEN,
)

print("== Heatmap entrevistas x libros (coseno TF-IDF medio) ==")
print(f"Grupos (tomo del top-1 BM25): {dict(zip(TOMOS_ORDEN, n_entrevistas_grupo.tolist()))}")
print("\nMatriz 10 x 10 (filas = grupo de entrevistas; columnas = tomo del libro):")
heatmap_df = pd.DataFrame(matriz_heatmap, index=TOMOS_ORDEN, columns=TOMOS_ORDEN)
print(heatmap_df.round(4).to_string())

diagonal = np.diag(matriz_heatmap)
media_fuera = (
    float(np.nansum(matriz_heatmap) - np.nansum(diagonal))
    / (matriz_heatmap.size - len(TOMOS_ORDEN))
)
# Robustez: np.nanargmax lanzaria ValueError con una fila toda NaN (grupo
# sin entrevistas, hoy no ocurre); el bucle solo evalua filas con datos.
maximos_fila = np.full(len(TOMOS_ORDEN), -1, dtype=int)
for i, fila in enumerate(matriz_heatmap):
    if np.any(np.isfinite(fila)):
        maximos_fila[i] = int(np.nanargmax(fila))
n_grupos_no_vacios = int((maximos_fila >= 0).sum())
n_max_en_diagonal = int(
    np.sum(maximos_fila == np.arange(len(TOMOS_ORDEN)))
)
print(f"\nDiagonal (grupo g vs tomo g): {dict(zip(TOMOS_ORDEN, np.round(diagonal, 4).tolist()))}")
print(
    f"Media fuera de la diagonal: {media_fuera:.4f} | "
    f"media diagonal: {float(np.nanmean(diagonal)):.4f}"
)
print(
    "El maximo de cada fila cae en la diagonal en "
    f"{n_max_en_diagonal}/{n_grupos_no_vacios} grupos no vacios"
)

# --- CSV (matriz ancha + n_entrevistas por grupo) ---
heatmap_salida = heatmap_df.copy()
heatmap_salida.insert(0, "n_entrevistas", n_entrevistas_grupo)
ruta_heatmap_csv = RUTA_RESULTADOS / "heatmap_tomos.csv"
try:
    ruta_heatmap_csv.parent.mkdir(parents=True, exist_ok=True)
    heatmap_salida.to_csv(ruta_heatmap_csv, index_label="grupo_entrevistas")
    print(f"\n[ok] resultados/{ruta_heatmap_csv.name}: matriz 10 x 10 (salida del notebook)")
except OSError as exc:
    print(f"[aviso] no se pudo escribir resultados/{ruta_heatmap_csv.name}: {exc}")

# --- Figura anotada ---
fig, ax = plt.subplots(figsize=(11, 7))
imagen = ax.imshow(matriz_heatmap, cmap="viridis", aspect="auto")
ax.set_xticks(np.arange(len(TOMOS_ORDEN)), TOMOS_ORDEN, rotation=45, ha="right")
ax.set_yticks(np.arange(len(TOMOS_ORDEN)), TOMOS_ORDEN)
umbral_texto = float(np.nanmax(matriz_heatmap)) * 0.55
for i in range(len(TOMOS_ORDEN)):
    for j in range(len(TOMOS_ORDEN)):
        ax.text(
            j, i, f"{matriz_heatmap[i, j]:.3f}",
            ha="center", va="center", fontsize=7,
            color="white" if matriz_heatmap[i, j] < umbral_texto else "black",
        )
fig.colorbar(imagen, ax=ax, label="coseno TF-IDF medio (entrevista, segmento)")
ax.set_xlabel("tomo del segmento de libro")
ax.set_ylabel("grupo de entrevistas (tomo del top-1 BM25)")
ax.set_title(
    "Heatmap entrevistas -> libros: afinidad TF-IDF media por grupo y tomo\n"
    "(filas = 2.486 entrevistas agrupadas por el tomo de su top-1 BM25)",
    fontsize=11,
)
plt.tight_layout()
ruta_heatmap_png = RUTA_IMAGENES / "heatmap_entrevistas_libros.png"
try:
    RUTA_IMAGENES.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_heatmap_png, dpi=150, bbox_inches="tight")
    print(
        f"[ok] imagenes/{ruta_heatmap_png.name}: "
        f"{ruta_heatmap_png.stat().st_size / 1024:.1f} KB "
        "(salida del notebook)"
    )
except OSError as exc:
    print(f"[aviso] no se pudo guardar imagenes/{ruta_heatmap_png.name}: {exc}")
plt.show()


# %%
"""Heatmap top-25 entrevistas x 10 tomos con el MAXIMO score BM25.

Adaptado del aporte de Erika Rivera (Taller_4_IR_CEV_V2): filas = las 25
entrevistas con mayor SUMA de scores BM25; columnas = los 10 tomos en
orden oficial; celda = MAXIMO score BM25 de la entrevista contra cualquier
segmento del tomo. La vista por documento contrasta con el heatmap 10 x 10
de arriba (filas = grupos agregados, celda = coseno TF-IDF medio). La
matriz completa `contexto_base["matriz_bm25"]` (float32) la expone la
Sec 03b en cada corrida; si no esta disponible, la celda avisa y se omite.
"""
import seaborn as sns

contexto_base = globals().get("contexto_base")
RUTA_RESULTADOS = globals().get("RUTA_RESULTADOS")
RUTA_IMAGENES = globals().get("RUTA_IMAGENES")
matriz_bm25 = (
    None if contexto_base is None else contexto_base.get("matriz_bm25")
)
if matriz_bm25 is None or RUTA_RESULTADOS is None or RUTA_IMAGENES is None:
    print(
        "[aviso] contexto_base['matriz_bm25'] no disponible (o rutas de la "
        "Sec 00 ausentes): se omite el heatmap top-25. La matriz la expone "
        "la Sec 03b en cada corrida."
    )
else:
    docs_entrevistas = contexto_base["docs_entrevistas"]
    docs_libros = contexto_base["docs_libros"]
    tomos_por_segmento = np.array(
        [registro["tomo"] for registro in docs_libros]
    )
    matriz_max_tomo = np.zeros(
        (matriz_bm25.shape[0], len(TOMOS_ORDEN)), dtype=np.float32
    )
    for indice_tomo, tomo in enumerate(TOMOS_ORDEN):
        indices_tomo = np.where(tomos_por_segmento == tomo)[0]
        if indices_tomo.size:
            matriz_max_tomo[:, indice_tomo] = matriz_bm25[
                :, indices_tomo
            ].max(axis=1)

    suma_por_entrevista = matriz_max_tomo.sum(axis=1)
    indices_top25 = np.argsort(-suma_por_entrevista, kind="stable")[:25]
    ids_top25 = [docs_entrevistas[i]["id"] for i in indices_top25]
    etiquetas_top25 = [id_doc[:15] + "..." for id_doc in ids_top25]
    df_top25 = pd.DataFrame(
        matriz_max_tomo[indices_top25],
        index=etiquetas_top25,
        columns=TOMOS_ORDEN,
    )
    print("\n== Heatmap top-25 entrevistas x 10 tomos (maximo BM25 por tomo) ==")
    print(
        f"Dimension: {df_top25.shape} | max BM25 = "
        f"{float(df_top25.to_numpy().max()):.3f} | suma top-25 = "
        f"{float(np.sort(suma_por_entrevista)[::-1][:25].sum()):,.1f}"
    )

    ruta_heatmap_top25 = RUTA_RESULTADOS / "heatmap_top25.csv"
    try:
        ruta_heatmap_top25.parent.mkdir(parents=True, exist_ok=True)
        df_top25_csv = df_top25.copy()
        df_top25_csv.index = ids_top25  # el CSV guarda los id_doc completos
        df_top25_csv.to_csv(ruta_heatmap_top25, index_label="id_doc")
        print(
            f"[ok] resultados/{ruta_heatmap_top25.name}: "
            f"{df_top25.shape[0]} x {df_top25.shape[1]} (salida del notebook)"
        )
    except OSError as exc:
        print(
            f"[aviso] no se pudo escribir resultados/"
            f"{ruta_heatmap_top25.name}: {exc}"
        )

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        df_top25, cmap="YlOrRd", annot=False, linewidths=0.5,
        cbar_kws={"label": "Nivel de relevancia (BM25)"},
    )
    plt.title(
        "Relacionamiento tematico: top-25 entrevistas vs. tomos de la CEV "
        "(maximo BM25)",
        fontsize=14, pad=15,
    )
    plt.xlabel("Tomos oficiales (ejes tematicos)", fontsize=12, labelpad=10)
    plt.ylabel("ID del documento (entrevista)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ruta_heatmap_top25_png = (
        RUTA_IMAGENES / "heatmap_top25_entrevistas_tomos.png"
    )
    try:
        RUTA_IMAGENES.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_heatmap_top25_png, dpi=150, bbox_inches="tight")
        print(
            f"[ok] imagenes/{ruta_heatmap_top25_png.name}: "
            f"{ruta_heatmap_top25_png.stat().st_size / 1024:.1f} KB "
            "(salida del notebook)"
        )
    except OSError as exc:
        print(
            f"[aviso] no se pudo guardar imagenes/"
            f"{ruta_heatmap_top25_png.name}: {exc}"
        )
    plt.show()


# %% [markdown]
"""## Lectura del heatmap top-25 (documento vs. agregado)

Vista adaptada del aporte de Erika Rivera (`Taller_4_IR_CEV_V2`).
Diferencia con el heatmap 10 x 10: alli cada fila es un GRUPO de
entrevistas (agregado por el tomo del top-1 BM25) y la celda promedia el
coseno TF-IDF grupo-tomo; aqui cada fila es UNA entrevista (las 25 con
mayor suma de BM25) y la celda es el MAXIMO score BM25 de esa entrevista
con cualquier segmento del tomo. La vista por documento conserva la
intensidad individual (un testimonio puede saturar un eje tematico
especifico) que el promedio por grupo disuelve; el agregado, en cambio,
resume la afinidad media del corpus.
"""


# %%
"""Cuadro de vinculos: 3 mejores parejas entrevista -> segmento por tomo (BM25).

Esquema del CSV: tomo, puesto, id_doc, id_segmento, score_bm25,
titulo_segmento, fragmento_entrevista, fragmento_libro. Los fragmentos son
los primeros 200 caracteres de los textos lematizados limpios de la pareja
(contexto_base["entrevistas_limpios"]/["libros_limpios"], indices de los
metadatos); aporte de Erika Rivera (Taller_4_IR_CEV_V2).
"""


def tabla_markdown_simple(encabezados: list[str], filas: list[list[str]]) -> str:
    """Formatea una tabla markdown simple (para mostrar en la celda).

    Args:
        encabezados: nombres de las columnas.
        filas: filas como listas de cadenas (mismo orden que encabezados).

    Returns:
        Texto markdown con cabecera, separador y filas.
    """
    lineas = [
        "| " + " | ".join(encabezados) + " |",
        "|" + "|".join("---" for _ in encabezados) + "|",
    ]
    lineas.extend("| " + " | ".join(fila) + " |" for fila in filas)
    return "\n".join(lineas)


def mostrar_markdown(texto: str) -> None:
    """Muestra texto markdown en el notebook (fallback: print sin IPython).

    Args:
        texto: contenido markdown a renderizar.
    """
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(texto)
        return
    display(Markdown(texto))


indice_entrevistas = {
    registro["id"]: i
    for i, registro in enumerate(contexto_base["docs_entrevistas"])
}
indice_libros = {
    registro["id"]: i
    for i, registro in enumerate(contexto_base["docs_libros"])
}


def fragmento_lematizado(
    textos: list[str], indice: int, longitud: int = 200
) -> str:
    """Primeros `longitud` caracteres de un texto lematizado limpio.

    Args:
        textos: lista de textos de contexto_base (entrevistas_limpios o
            libros_limpios).
        indice: posicion del documento en esa lista.
        longitud: caracteres a conservar (200).

    Returns:
        Fragmento sin saltos de linea ni espacios repetidos; agrega "..." si
        el texto original es mas largo que `longitud`.
    """
    texto = " ".join(textos[indice].split())
    return texto[:longitud] + ("..." if len(texto) > longitud else "")


filas_vinculos: list[dict] = []
for tomo in TOMOS_ORDEN:
    mejores = ranking_bm25[ranking_bm25["tomo"] == tomo].nlargest(3, "score")
    for puesto, (_, fila) in enumerate(mejores.iterrows(), start=1):
        filas_vinculos.append(
            {
                "tomo": tomo,
                "puesto": puesto,
                "id_doc": fila["id_doc"],
                "id_segmento": fila["id_segmento"],
                "score_bm25": round(float(fila["score"]), 6),
                "titulo_segmento": str(fila["titulo_segmento"]),
                "fragmento_entrevista": fragmento_lematizado(
                    contexto_base["entrevistas_limpios"],
                    indice_entrevistas[fila["id_doc"]],
                ),
                "fragmento_libro": fragmento_lematizado(
                    contexto_base["libros_limpios"],
                    indice_libros[fila["id_segmento"]],
                ),
            }
        )
vinculos = pd.DataFrame(
    filas_vinculos,
    columns=[
        "tomo",
        "puesto",
        "id_doc",
        "id_segmento",
        "score_bm25",
        "titulo_segmento",
        "fragmento_entrevista",
        "fragmento_libro",
    ],
)
print("== Cuadro de vinculos (top-3 entrevista -> segmento por tomo, BM25) ==")
print(f"{len(vinculos)} parejas (3 por tomo) sobre el ranking BM25 de la Sec 03b")

ruta_vinculos = RUTA_RESULTADOS / "vinculos_entrevistas_libros.csv"
try:
    ruta_vinculos.parent.mkdir(parents=True, exist_ok=True)
    vinculos.to_csv(ruta_vinculos, index=False)
    print(f"[ok] resultados/{ruta_vinculos.name}: {len(vinculos)} filas (salida del notebook)")
except OSError as exc:
    print(f"[aviso] no se pudo escribir resultados/{ruta_vinculos.name}: {exc}")

# Tabla markdown (id_doc truncado y titulo recortado para que quepa; el CSV
# lleva los valores completos). El pipe se reemplaza para no romper markdown.
filas_tabla_md = []
for _, fila in vinculos.iterrows():
    filas_tabla_md.append(
        [
            fila["tomo"],
            str(fila["puesto"]),
            fila["id_doc"][:18] + "..",
            fila["id_segmento"][:46] + (".." if len(fila["id_segmento"]) > 46 else ""),
            f"{fila['score_bm25']:.4f}",
            str(fila["titulo_segmento"]).replace("|", "/")[:52],
            fila["fragmento_entrevista"].replace("|", "/"),
            fila["fragmento_libro"].replace("|", "/"),
        ]
    )
mostrar_markdown(
    "**Vinculos entrevista -> segmento por tomo (BM25, top-3)**\n\n"
    + tabla_markdown_simple(
        [
            "tomo",
            "p.",
            "id_doc",
            "id_segmento",
            "score",
            "titulo_segmento",
            "frag. entrevista (200 car.)",
            "frag. libro (200 car.)",
        ],
        filas_tabla_md,
    )
)


# %% [markdown]
"""## Interpretacion del heatmap y de los vinculos

**La diagonal domina: validacion cruzada entre BM25 (asigna) y TF-IDF
(mide).** La media de la diagonal (0,0725) supera a la media fuera de la
diagonal (0,0608) y, en los 10 grupos, la diagonal esta por encima de la
media de su propia fila: las entrevistas asignadas por BM25 a un tomo son
las mas afines a ese tomo tambien con la similitud TF-IDF. La asignacion
por top-1 no es ruido.

**La columna TESTIMONIAL captura el sesgo de registro oral.** El grupo
TESTIMONIAL concentra 1.863 de las 2.486 entrevistas (74,9 %; 75,0 % de
las 2.484 con match) y su celda
diagonal es la mayor de la matriz (0,0904). Su fila es la mas contrastada:
0,0904 en TESTIMONIAL frente a 0,037-0,060 en el resto de columnas. En 6
de los 7 grupos restantes el maximo de la fila cae en la columna
TESTIMONIAL (no en su diagonal) y en el septimo (HALLAZGOS) cae en
VIOLACIONES (0,0711, muy cerca de su diagonal 0,0666): el maximo de la
fila coincide con la diagonal solo en TESTIMONIAL, ETNICO y FUERA. Es la
lectura visual del sesgo de dominio de la Sec 03: todas las entrevistas
hablan como testimonios y el tomo TESTIMONIAL esta escrito como hablan
las entrevistas.

**Afinidades cruzadas visibles.** Fuera de la diagonal, las columnas
TESTIMONIAL (0,066-0,079) y NARRATIVA (0,045-0,078) son las mas altas
para casi todos los grupos: todo el corpus comparte el lexico del
conflicto y los tomos analiticos reutilizan relatos de victimas. Las
celdas mas bajas estan en CONVOCATORIA (0,037-0,052): su lexico
institucional ("acoger la realidad de las victimas") apenas solapa con
las entrevistas.

**Advertencias de lectura.** (1) El grupo CONVOCATORIA tiene n=2: su
diagonal (0,0767) parece alta pero un promedio con 2 entrevistas es
estadisticamente fragil y no debe citarse como evidencia; (2) la
agrupacion usa el top-1 BM25, que hereda el sesgo TESTIMONIAL: el heatmap
mide coherencia, no correccion; (3) sin gold estandar no se puede afirmar
que una afinidad alta sea relevancia real (el lexico compartido del
conflicto infla todas las celdas); (4) la media de cosenos favorece
terminos comunes del dominio y no distingue hechos, poblacion ni periodo.
La mejora documentada en la Sec 04 es un gold estandar y embeddings
semanticos que midan sentido, no solo solape lexico.
"""
