# %% [markdown]
"""# TALLER-2 — EDA del Tesauro de la Comision de la Verdad

**Objetivo**: caracterizar la estructura del tesauro (12.341 terminos, 41 grupos)
para seleccionar 5 grupos base de la ontologia WordNet. El tesauro codifica la
jerarquia dentro del propio termino con el separador `" - "`
(ej. `Armas - Instrumentos - Armas blancas - Bayoneta`).

**Preguntas que responde este EDA**:
1. Distribucion de terminos por grupo y tamano de los grupos.
2. Profundidad jerarquica (niveles) por grupo: min / mediana / max.
3. Tipos de termino: simples (1 segmento) vs compuestos (2+ segmentos).
4. Patrones anomalos: duplicados, vacios, menciones, acronimos.
5. Cobertura preliminar en WordNet espanol (NLTK + omw-1.4) por grupo candidato.
"""

# %%
"""Configuracion del EDA: imports, constantes y estilos graficos.

En el notebook, `sec00_setup` define todas las rutas y umbrales globales y
`sec00_clases` el nucleo POO (Repository). Este bloque solo anade el respaldo
para ejecutar la seccion como script standalone (mismas constantes, apuntando
a las rutas reales del proyecto), de modo que el archivo es autocontenido.
"""
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from nltk.corpus import wordnet as wn

try:  # nucleo POO (sec00_clases): modulo local de notebook_cells/
    import sec00_clases as sc
except ImportError:
    try:
        ruta_celdas = Path(__file__).resolve().parent  # script standalone
    except NameError:
        ruta_celdas = Path.cwd()  # notebook: kernel en notebooks/ o en la raiz
        if ruta_celdas.name == "notebooks":
            ruta_celdas = ruta_celdas.parent / "notebook_cells"
    sys.path.insert(0, str(ruta_celdas))
    try:
        import sec00_clases as sc
    except ImportError:
        import types  # notebook: las celdas de sec00_clases ya definieron las clases

        sc = types.ModuleType("sec00_clases")
        sc.TesauroRepositorio = globals()["TesauroRepositorio"]
        sys.modules["sec00_clases"] = sc

# Constantes globales de sec00 (respaldo SOLO para ejecucion standalone)
if "RUTA_EXCEL" not in globals():
    RUTA_BASE = Path(__file__).resolve().parent.parent
    RUTA_DATOS = RUTA_BASE / "data"
    RUTA_IMAGENES = RUTA_BASE / "imagenes"
    RUTA_EXCEL = RUTA_DATOS / "tesauro_orginal_cev.xlsx"
    RUTA_JSON_GRUPOS = RUTA_DATOS / "grupos_seleccionados.json"
    RUTA_CSV_COBERTURA = RUTA_DATOS / "cobertura_wordnet_grupos.csv"
    FIGURAS = {
        "top_grupos": RUTA_IMAGENES / "barras_top_grupos.png",
        "profundidad": RUTA_IMAGENES / "profundidad_grupos.png",
        "tipos_termino": RUTA_IMAGENES / "distribucion_tipos_termino.png",
        "heatmap_nivel": RUTA_IMAGENES / "heatmap_terminos_grupos_nivel.png",
        "arbol_grupo": RUTA_IMAGENES / "arbol_jerarquico_armas.png",
        "cobertura_grupos": RUTA_IMAGENES / "cobertura_wordnet_grupos.png",
    }
    UMBRAL_COBERTURA = 80
    TOP_GRUPOS_FIGURA = 15

RUTA_IMAGENES.mkdir(exist_ok=True)

# Configuracion grafica de la seccion (los mismos rcParams que ya usaba el
# EDA; el tema de seaborn lo fija sec00_setup en el notebook, no se altera).
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

# %%
"""Carga del tesauro y limpieza basica via el repositorio (con cache)."""

repositorio = sc.TesauroRepositorio(RUTA_EXCEL, RUTA_JSON_GRUPOS)
tesauro = repositorio.cargar_tesauro()
tesauro = sc.TesauroRepositorio.con_estructura(tesauro)
tesauro = sc.TesauroRepositorio.marcar_hojas(tesauro)

print("=== EDA TALLER-2 (Tesauro CEV) ===")
print(f"Filas utiles: {len(tesauro)} | Grupos: {tesauro['grupo'].nunique()}")
print(f"Terminos unicos: {tesauro['termino'].nunique()} | "
      f"Duplicados: {tesauro['termino'].duplicated(keep=False).sum()} filas")
print(f"Niveles globales: min={tesauro['niveles'].min()} "
      f"mediana={tesauro['niveles'].median():.0f} max={tesauro['niveles'].max()}")

anomalias = sc.TesauroRepositorio.detectar_anomalias(tesauro)
totales = anomalias.sum()
print("Patrones anomalos (filas): "
      f"parentesis={totales['con_parentesis']}, comillas={totales['con_comillas']}, "
      f"acronimos={totales['acronimo']}, separador_final={totales['separador_final']}")

# %%
"""Clases POO de la seccion: analisis (EdaTesauro) y figuras (GraficadorEda).

SOLID: el acceso a datos vive en `sc.TesauroRepositorio` (sec00); aqui solo se
combinan metricas y se genera la visualizacion. Las funciones sueltas
`resumen_por_grupo`, `cobertura_wordnet_por_grupo` y las figuras del codigo
anterior pasan a ser metodos de estas dos clases (DRY dentro de la seccion).
"""


class EdaTesauro:
    """Metricas del EDA: resumen por grupo y cobertura WordNet-espanol.

    Responsabilidad unica: calculo (sin graficas). Recibe el DataFrame ya
    cargado y enriquecido por el repositorio (estructura y hojas).

    Attributes:
        df: Tesauro con columnas derivadas (`niveles`, `hoja`, `es_hoja`).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @staticmethod
    def normalizar_sin_acentos(texto: str) -> str:
        """Normaliza un termino a minusculas sin diacriticos (NFKD + ascii).

        Nota: se conserva esta logica local en lugar de usar
        `sc.NormalizadorAcentos` porque esa clase ademas elimina parentesis y
        simbolos, lo que alteraria las coberturas calculadas.
        """
        return (
            unicodedata.normalize("NFKD", texto)
            .encode("ascii", "ignore")
            .decode()
            .lower()
            .strip()
        )

    @staticmethod
    def tiene_synset_espanol(termino: str) -> bool:
        """Verifica si un termino tiene >=1 synset en WordNet espanol.

        Prueba el termino tal cual (con acentos) y, si no hay resultados, la
        version normalizada sin acentos.

        Args:
            termino: Segmento final del path (hoja del termino).

        Returns:
            True si alguna de las dos variantes tiene synsets en lang='spa'.
        """
        if not termino:
            return False
        if wn.synsets(termino, lang="spa"):
            return True
        normalizado = EdaTesauro.normalizar_sin_acentos(termino)
        return bool(
            normalizado and normalizado != termino and wn.synsets(normalizado, lang="spa")
        )

    def resumen_por_grupo(self) -> pd.DataFrame:
        """Construye la tabla resumen por grupo.

        Returns:
            DataFrame indexado por grupo con columnas: total_terminos,
            niveles_min, niveles_mediana, niveles_max, raices y hojas.
        """
        grupo = self.df.groupby("grupo")
        return pd.DataFrame({
            "total_terminos": grupo["termino"].size(),
            "niveles_min": grupo["niveles"].min(),
            "niveles_mediana": grupo["niveles"].median(),
            "niveles_max": grupo["niveles"].max(),
            "raices": grupo["niveles"].apply(lambda s: (s == 1).sum()),
            "hojas": grupo["es_hoja"].sum(),
        }).sort_values("total_terminos", ascending=False)

    def cobertura_wordnet(self, umbral: int) -> pd.DataFrame:
        """Cobertura WordNet-espanol por grupo: % de hojas con >=1 synset.

        Args:
            umbral: Minimo de terminos para considerar un grupo candidato.

        Returns:
            DataFrame con total_terminos, cubiertos, porcentaje_cobertura,
            niveles_mediana y niveles_max.
        """
        candidatos = self.df.groupby("grupo")["termino"].size()
        candidatos = candidatos[candidatos >= umbral].index
        filas = self.df[self.df["grupo"].isin(candidatos)].copy()
        filas["cubierto"] = filas["hoja"].apply(self.tiene_synset_espanol)

        cobertura = filas.groupby("grupo").agg(
            total_terminos=("termino", "size"),
            cubiertos=("cubierto", "sum"),
            niveles_mediana=("niveles", "median"),
            niveles_max=("niveles", "max"),
        )
        cobertura["porcentaje_cobertura"] = (
            100 * cobertura["cubiertos"] / cobertura["total_terminos"]
        ).round(2)
        return cobertura.sort_values("porcentaje_cobertura", ascending=False)


class GraficadorEda:
    """Figuras del EDA: genera cada grafica y la guarda en imagenes/.

    Responsabilidad unica: visualizacion. Las rutas de salida provienen del
    diccionario global FIGURAS (sec00) y el calculo de las metricas vive en
    EdaTesauro.
    """

    def guardar(self, fig: plt.Figure, ruta: Path) -> None:
        """Guarda la figura (150 dpi), la cierra y confirma en consola.

        Args:
            fig: Figura de matplotlib a guardar.
            ruta: Ruta completa del PNG de salida.
        """
        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Figura guardada: {ruta.name}")

    def top_grupos(self, resumen_df: pd.DataFrame, top: int = TOP_GRUPOS_FIGURA) -> None:
        """Figura 1: barras horizontales con los top grupos por numero de terminos."""
        datos = resumen_df.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(datos.index, datos["total_terminos"], color="#4C72B0")
        for i, v in enumerate(datos["total_terminos"]):
            ax.text(v + 20, i, f"{v:,}".replace(",", "."), va="center", fontsize=8)
        ax.set_xlabel("Numero de terminos")
        ax.set_title(f"Top {top} grupos del tesauro por numero de terminos")
        ax.set_xlim(0, datos["total_terminos"].max() * 1.12)
        fig.tight_layout()
        self.guardar(fig, FIGURAS["top_grupos"])

    def profundidad(
        self, df: pd.DataFrame, resumen_df: pd.DataFrame, top: int = TOP_GRUPOS_FIGURA
    ) -> None:
        """Figura 2: violin de la distribucion de niveles por grupo (top tamano)."""
        top_grupos = resumen_df.head(top).index.tolist()
        datos = df[df["grupo"].isin(top_grupos)].copy()
        orden = sorted(
            top_grupos, key=lambda g: df.loc[df["grupo"] == g, "niveles"].median()
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.violinplot(data=datos, x="grupo", y="niveles", order=orden, ax=ax,
                       palette="muted", inner="quartile", cut=0)
        ax.set_xticklabels(orden, rotation=45, ha="right")
        ax.set_ylabel("Niveles jerarquicos")
        ax.set_xlabel("Grupo")
        ax.set_title("Distribucion de profundidad jerarquica por grupo (top 15)")
        fig.tight_layout()
        self.guardar(fig, FIGURAS["profundidad"])

    def tipos_termino(self, df: pd.DataFrame) -> None:
        """Figura 3: terminos segun el numero de segmentos del path."""
        conteo = df["niveles"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        colores = ["#55A868" if n == 1 else "#4C72B0" for n in conteo.index]
        ax.bar(conteo.index.astype(str), conteo.values, color=colores)
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 30, f"{v:,}".replace(",", "."), ha="center", fontsize=8)
        ax.set_xlabel("Numero de segmentos del path (nivel)")
        ax.set_ylabel("Numero de terminos")
        ax.set_title("Terminos simples (1 segmento) vs compuestos (2+ segmentos)")
        fig.tight_layout()
        self.guardar(fig, FIGURAS["tipos_termino"])

    def heatmap_grupo_nivel(self, df: pd.DataFrame, top: int = TOP_GRUPOS_FIGURA) -> None:
        """Figura 4: heatmap de conteo de terminos por grupo (top) y nivel."""
        top_grupos = df.groupby("grupo")["termino"].size().nlargest(top).index
        tabla = df[df["grupo"].isin(top_grupos)].pivot_table(
            index="grupo", columns="niveles", values="termino", aggfunc="size",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(tabla, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
        ax.set_xlabel("Nivel jerarquico")
        ax.set_ylabel("Grupo")
        ax.set_title("Conteo de terminos por grupo y nivel (top 15)")
        fig.tight_layout()
        self.guardar(fig, FIGURAS["heatmap_nivel"])

    def arbol_grupo(self, df: pd.DataFrame, grupo: str, max_nivel: int = 6) -> None:
        """Figura 5: arbol jerarquico (networkx) de un grupo de ejemplo.

        Cada nodo es un segmento del path; las aristas unen padre-hijo. Layout
        jerarquico: y = nivel del nodo, x = posicion de aparicion en el nivel.

        Args:
            df: Tesauro completo (con columnas derivadas).
            grupo: Nombre del grupo a graficar.
            max_nivel: Profundidad maxima a dibujar (evita figuras saturadas).
        """
        sub = df[(df["grupo"] == grupo) & (df["niveles"] <= max_nivel)].copy()
        grafo = nx.DiGraph()
        for termino in sub["termino"]:
            partes = termino.split(" - ")
            for i in range(1, len(partes)):
                grafo.add_edge(" - ".join(partes[:i]), " - ".join(partes[: i + 1]))

        nivel_nodo = {nodo: len(nodo.split(" - ")) for nodo in grafo.nodes}
        nodos_por_nivel: dict[int, list[str]] = {}
        for nodo in grafo.nodes:
            nodos_por_nivel.setdefault(nivel_nodo[nodo], []).append(nodo)
        pos = {}
        for nivel, nodos in nodos_por_nivel.items():
            for j, nodo in enumerate(nodos):
                pos[nodo] = (j / max(len(nodos) - 1, 1), -nivel)

        fig, ax = plt.subplots(figsize=(20, 12))
        nx.draw_networkx_edges(grafo, pos, ax=ax, arrows=False,
                               edge_color="#BBBBBB", width=0.8)
        nx.draw_networkx_nodes(grafo, pos, ax=ax, node_size=150,
                               node_color="#4C72B0", alpha=0.9)
        etiquetas = {nodo: nodo.split(" - ")[-1] for nodo in grafo.nodes}
        # Etiquetas legibles: nivel <=2 horizontales grandes; nivel >=3 rotadas
        # a 90 grados para evitar el solapamiento entre nodos adyacentes.
        for nodo, (x, y) in pos.items():
            if nivel_nodo[nodo] <= 2:
                ax.text(x, y + 0.08, etiquetas[nodo], ha="center", va="bottom",
                        fontsize=10, color="#2b2b2b")
            elif nivel_nodo[nodo] <= 4:
                ax.text(x, y - 0.10, etiquetas[nodo], ha="center", va="top",
                        fontsize=8, color="#444444")
            else:  # nivel 5+: rotadas para evitar solapamiento
                ax.text(x, y - 0.12, etiquetas[nodo], ha="center", va="top",
                        fontsize=7, color="#555555", rotation=90)
        ax.set_title(f"Arbol jerarquico del grupo '{grupo}' (hasta nivel {max_nivel}, "
                     f"{grafo.number_of_nodes()} nodos)", fontsize=14)
        ax.axis("off")
        fig.tight_layout()
        self.guardar(fig, FIGURAS["arbol_grupo"])

    def cobertura(self, cobertura_df: pd.DataFrame) -> None:
        """Figura 6: barras del % de cobertura WordNet-espanol por grupo."""
        datos = cobertura_df.sort_values("porcentaje_cobertura").iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(datos.index, datos["porcentaje_cobertura"], color="#C44E52")
        for i, v in enumerate(datos["porcentaje_cobertura"]):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
        ax.axvline(1.0, color="grey", ls="--", lw=1)
        ax.text(1.1, len(datos) - 0.4, "umbral 1%", fontsize=8, color="grey")
        ax.set_xlabel("% de hojas con >=1 synset en WordNet espanol")
        ax.set_title(
            "Cobertura preliminar WordNet-espanol por grupo candidato (>=80 terminos)"
        )
        fig.tight_layout()
        self.guardar(fig, FIGURAS["cobertura_grupos"])

# %%
"""Resumen por grupo: tamano, profundidad, raices y hojas + figura de top."""

analizador = EdaTesauro(tesauro)
resumen = analizador.resumen_por_grupo()
print("=== Resumen por grupo (5 mayores y 5 menores) ===")
columnas = ["total_terminos", "niveles_mediana", "niveles_max", "raices", "hojas"]
print(pd.concat([resumen.head(5), resumen.tail(5)])[columnas].to_string())

graficador = GraficadorEda()
graficador.top_grupos(resumen)

# %% [markdown]
"""![Top 15 grupos](../imagenes/barras_top_grupos.png)

**Interpretacion**: el tesauro esta dominado por 4 grupos gigantes: Organizaciones
(6.405 terminos), Estructuras armadas (1.416), Medios de comunicacion (1.354) y
Profesiones y oficios (942). Esos grupos, por su tamano, son inadecuados como
punto de partida de una ontologia construida a mano (y en el caso de
Organizaciones suponen mas de la mitad del tesauro). El resto de grupos (37)
tienen tamanos entre 5 y 223 terminos, rango mucho mas manejable.
"""

# %%
"""Profundidad jerarquica por grupo: violin de niveles, tipos y heatmap."""

graficador.profundidad(tesauro, resumen)
graficador.tipos_termino(tesauro)
graficador.heatmap_grupo_nivel(tesauro)

print("\n=== Profundidad jerarquica: grupos con mayor riqueza (top 10) ===")
columnas_niveles = ["niveles_mediana", "niveles_max", "total_terminos"]
print(resumen[columnas_niveles].head(10).to_string())

# %% [markdown]
"""![Distribucion de profundidad](../imagenes/profundidad_grupos.png)

![Tipos de termino](../imagenes/distribucion_tipos_termino.png)

![Heatmap grupo por nivel](../imagenes/heatmap_terminos_grupos_nivel.png)

**Interpretacion**: la profundidad global va de 1 a 10 niveles (mediana 4).
Los grupos con jerarquia mas rica son Estructuras armadas (mediana 6, maximo 10),
Armas (mediana 6, maximo 6) y Profesiones y oficios (mediana 4, maximo 6).
El 98% de los terminos son compuestos (2+ segmentos) y el 74% tiene entre 3 y 5
niveles, lo que confirma una taxonomia real y no una lista plana. El heatmap
muestra que casi todos los grupos concentran sus terminos en los niveles 2-4.
"""

# %%
"""Arbol jerarquico del grupo 'Armas' (grupo de ejemplo)."""

graficador.arbol_grupo(tesauro, "Armas")

# %% [markdown]
"""![Arbol jerarquico de Armas](../imagenes/arbol_jerarquico_armas.png)

**Interpretacion**: el grupo Armas (94 terminos) tiene una estructura de arbol
profundo y limpia: la raiz `Armas` se bifurca en `Instrumentos` y `Medios`, y a
partir de ahi cadenas de hasta 6 niveles (ej. `Armas - Instrumentos - Armas
blancas - Bayoneta`). Es un grupo ideal para construir una ontologia: tamano
manejable, taxonomia con profundidad real y vocabulario con correspondencia en
WordNet (Cuchillo, Espada, Pistola, Revólver...).
"""

# %%
"""Cobertura preliminar en WordNet espanol (NLTK + omw-1.4).

Se evalua el segmento final (hoja) de cada termino: si la hoja tiene >=1 synset
en lang='spa'. Se prueba primero con acentos y, si no hay resultados, se
reintenta la version normalizada sin acentos. Solo se evaluan grupos con
>= UMBRAL_COBERTURA terminos (candidatos a la ontologia).
"""

cobertura = analizador.cobertura_wordnet(UMBRAL_COBERTURA)
graficador.cobertura(cobertura)
cobertura.to_csv(RUTA_CSV_COBERTURA)

print(f"\n=== Cobertura WordNet espanol (grupos >= {UMBRAL_COBERTURA} terminos) ===")
print(cobertura.to_string())
print(f"Cobertura guardada en: data/{RUTA_CSV_COBERTURA.name}")

# %% [markdown]
"""![Cobertura WordNet espanol](../imagenes/cobertura_wordnet_grupos.png)

**Interpretacion**: la cobertura de WordNet espanol es baja en todo el tesauro
(vocabulario muy especializado del conflicto colombiano). Los grupos con mejor
relacion senal/ruido son Profesiones y oficios (17.3%), Actores armados y otros
participantes en las dinamicas de la guerra (14.6%), Impactos (11.2%) y Armas
(7.4%). Grupos centrales para la Comision de la Verdad como Hechos victimizantes
(2.2%) y Poblaciones (1.4%) tienen cobertura escasa pero no nula, y aportan el
vocabulario de mayor relevancia tematica. La linea punteada marca el umbral de
1% usado en la seleccion (seccion 02).
"""
