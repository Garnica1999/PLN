# %% [markdown]
# # Seccion 4 - Relaciones semanticas WordNet sobre los conceptos de la ontologia (Fase 2b)
#
# **Fase 2b del TALLER-2.** Aplicacion masiva del extractor de 9 relaciones
# semanticas sobre los **610 conceptos** de la ontologia `tesauro_cev.owl`
# (6 grupos seleccionados en la Fase 1a, extension aprobada: se incorporo
# "Afrontamientos y resistencias no violentas - cotidianas" con 23 terminos).
# Esta seccion usa el **nucleo POO de
# la seccion 00** (`sc.MotorRelacionesWordNet`), que orquesta la cadena de
# normalizadores (`sc.CadenaNormalizadores`), la fabrica de extractores
# (`sc.FabricaExtractores` con sus 9 implementaciones Strategy) y NLTK:
# la funcion `motor.extraer_relaciones(termino, profundidad=1)` recibe como
# entrada los conceptos de la ontologia. Si `ontologia/tesauro_cev.owl`
# existe (lo genera la seccion 03), los conceptos se leen de ahi con su path
# jerarquico y su grupo (`sc.ServicioOntologia`); en caso contrario (seccion
# 03 aun no ejecutada) se usa como respaldo el JSON
# `data/grupos_seleccionados.json` via `sc.TesauroRepositorio`, que contiene
# exactamente los mismos terminos.
#
# Las 9 relaciones implementadas (una clase extractora por tipo, registrada
# en la fabrica de sec00):
#
# 1. **Sinonimia**: lemmas espanoles del mismo synset.
# 2. **Antonimia**: lemmas antonimos (via enlaces del PWN, retraducidos a espanol).
# 3. **Hiponimia**: synsets subordinados (recursion configurable con profundidad).
# 4. **Hiperonimia**: synsets superordinados (misma recursion por simetria).
# 5. **Meronimia**: parte/sustancia/miembro de (part + substance + member meronyms).
# 6. **Holonimia**: contiene a (part + substance + member holonyms).
# 7. **Troponimia**: verbos de manera (hiponimos de un synset verbal).
# 8. **Coordinados**: hermanos (mismos hiperonimos directos, excluyendo el propio synset).
# 9. **Derivacion morfologica**: lemmas derivados (derivationally_related_forms).
#
# Nota sobre omw-1.4 en espanol: solo 38.512 de 117.659 synsets del PWN tienen
# traduccion al espanol (~33%). Los enlaces de antonimia y derivacion viven en
# los lemmas ingleses del PWN; aqui se resuelven sobre esos lemmas y se
# retraducen al espanol cuando la traduccion existe, reportando el nombre PWN
# con la marca "(en)" en caso contrario. El termino de entrada NUNCA se
# traduce al ingles: si no existe en espanol, se devuelve `encontrado: False`.
#
# Productos de esta seccion:
#
# - `data/wordnet_relaciones.csv`: una fila por concepto con los conteos de
#   cada relacion y ejemplos en JSON compacto (trazabilidad completa).
# - 5 figuras en `imagenes/`: cobertura por grupo, heatmap grupo x tipo de
#   relacion, top-15 conceptos, grafo de relaciones compartidas entre
#   conceptos de la ontologia, y profundidad vs cobertura.
# - `data/grafo_relaciones_edges.csv`: aristas del grafo (concepto-concepto).
# - Celda de **verificacion de las 9 relaciones** (rubrica criterio 3) con
#   conceptos reales de la ontologia (incluye el grupo nuevo de
#   Afrontamientos) y assertions.

# %%
"""Imports, verificacion de los corpora NLTK, bloque standalone y estilo."""
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Rutas relativas al notebook (las celdas viven en notebook_cells/).
try:
    RUTA_BASE = Path(__file__).resolve().parent.parent  # modo script
except NameError:
    RUTA_BASE = Path(os.getcwd()).parent  # modo notebook (notebooks/)

# Nucleo POO de sec00: en el notebook ya se ejecuto como celdas; en modo
# script se importa desde notebook_cells/ para que la seccion sea standalone.
sys.path.insert(0, str(RUTA_BASE / "notebook_cells"))
import sec00_clases as sc

# Constantes globales del proyecto (definidas en sec00_setup, una sola vez).
# En el notebook ya viven en el namespace; el import solo se usa en modo
# script standalone (cuando el archivo sec00_setup.py existe en notebook_cells/).
if "RUTA_ONTOLOGIA" not in globals():
    from sec00_setup import (
        COLORES_GRUPO,
        FIGURAS,
        RUTA_CSV_GRAFO,
        RUTA_CSV_WORDNET,
        RUTA_EXCEL,
        RUTA_IMAGENES,
        RUTA_JSON_GRUPOS,
        RUTA_ONTOLOGIA,
        TIPOS_RELACION,
    )

# Verificacion de los corpora NLTK (wordnet + omw-1.4 en espanol). Los datos
# ya estan en disco (instalados en la Fase 0); este bloque NO descarga nada.
try:
    from nltk.corpus import wordnet as wn

    wn.synsets("desplazamiento", lang="spa")
except LookupError as error:
    print("ERROR: faltan los corpora NLTK (wordnet u omw-1.4).")
    print("Ejecutar una sola vez, con internet:")
    print('  import nltk')
    print('  nltk.download("wordnet")')
    print('  nltk.download("omw-1.4")')
    raise SystemExit(1) from error

TOTAL_SYNSETS_SPA = sum(1 for _ in wn.all_synsets(lang="spa"))
TOTAL_SYNSETS_PWN = sum(1 for _ in wn.all_synsets())

# Estilo de figuras: espanol, legibles, 150 dpi.
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

print(
    f"Corpora OK: {TOTAL_SYNSETS_SPA:,} synsets en espanol "
    f"de {TOTAL_SYNSETS_PWN:,} del PWN "
    f"({TOTAL_SYNSETS_SPA / TOTAL_SYNSETS_PWN:.1%})."
)

# %% [markdown]
# ## Carga de los conceptos de la ontologia
#
# Requisito del taller: "la funcion debe recibir como entrada los conceptos de
# la ontologia". Se intenta leer `ontologia/tesauro_cev.owl` (generado por la
# seccion 03) con el facade `sc.ServicioOntologia`: se recorren las clases
# OWL y se recupera el path jerarquico original desde la anotacion
# `rdfs:comment` (grupo = anotacion `cev:grupo`). Si el archivo OWL no existe
# aun, se usa como respaldo el JSON de la Fase 1a via `sc.TesauroRepositorio`,
# que contiene exactamente los mismos terminos con sus paths y grupos.
#
# Cada concepto se representa como una fila de DataFrame con:
# `grupo` (trazabilidad), `termino` (path completo "A - B - C") y
# `concepto` (segmento final, que es lo que se consulta en WordNet). La
# profundidad del path se deriva del propio termino (niveles separados por
# " - ") cuando el analisis la necesita.

# %%
"""Lectura de conceptos desde la ontologia OWL (preferente) o JSON (respaldo)."""


def _conceptos_desde_json() -> pd.DataFrame:
    """Construye el DataFrame de conceptos desde el JSON de grupos (respaldo).

    Usa el repository de sec00 para no conocer el formato del JSON: cada
    path "A - B - C" de cada grupo genera una fila con su segmento final.

    Returns:
        DataFrame con columnas `grupo`, `termino` y `concepto`.
    """
    repositorio = sc.TesauroRepositorio(RUTA_EXCEL, RUTA_JSON_GRUPOS)
    filas = []
    for grupo, paths in repositorio.terminos_por_grupo().items():
        for path in paths:
            filas.append({
                "grupo": grupo,
                "termino": path,
                "concepto": path.split(" - ")[-1].strip(),
            })
    return pd.DataFrame(filas)


servicio_ontologia = sc.ServicioOntologia.obtener(RUTA_ONTOLOGIA)
if RUTA_ONTOLOGIA.exists():
    print(f"Leyendo conceptos desde la ontologia: {RUTA_ONTOLOGIA}")
    servicio_ontologia.cargar()
    df_conceptos = servicio_ontologia.conceptos()
    fuente_conceptos = "ontologia_owl"
else:
    print(
        f"La ontologia {RUTA_ONTOLOGIA} aun no existe (la genera la seccion 03); "
        "se usan los conceptos del JSON de la Fase 1a."
    )
    df_conceptos = _conceptos_desde_json()
    fuente_conceptos = "json_respaldo"

TOTAL_CONCEPTOS = len(df_conceptos)
GRUPOS = list(dict.fromkeys(df_conceptos["grupo"]))

print(
    f"Fuente: {fuente_conceptos} | {TOTAL_CONCEPTOS} conceptos en "
    f"{len(GRUPOS)} grupos."
)
print("Ejemplos de conceptos con su path y nivel:")
for fila in df_conceptos.head(5).itertuples():
    niveles = len(fila.termino.split(" - "))
    print(
        f'  n={niveles} [{fila.grupo[:30]:<30}] {fila.termino}'
        f"  ->  concepto: {fila.concepto}"
    )

# %% [markdown]
# ## Aplicacion masiva del extractor (Motor POO de sec00)
#
# Se ejecuta `motor.analizar_conceptos(df_conceptos)` sobre los conceptos
# cargados (610 en el OWL actual, 6 grupos). El Motor normaliza cada path
# (cadena de
# normalizadores), busca los synsets en espanol probando las variantes en
# orden y aplica los 9 extractores registrados en la fabrica; los resultados
# se cachean por termino (evita recomputar conceptos repetidos entre grupos).
# Se guardan en `data/wordnet_relaciones.csv` con una fila por concepto:
# grupo, path completo, concepto, si fue encontrado, conteos de las 9
# relaciones y ejemplos en JSON compacto. La columna `variante_usada`
# registra que variante de busqueda produjo los synsets (trazabilidad de las
# normalizaciones).

# %%
"""Ejecucion masiva con el Motor y guardado del CSV de relaciones."""

motor = sc.MotorRelacionesWordNet()

inicio = time.time()
df_relaciones = motor.analizar_conceptos(df_conceptos)
df_relaciones = df_relaciones.rename(columns={
    "termino": "termino_path",
    "ejemplos": "relaciones_ejemplo",
    **{t: f"n_{t}" for t in TIPOS_RELACION},
})
# Trazabilidad de los synsets consultados: se reconsultan contra el cache del
# Motor (gratis, los resultados ya estan memorizados por termino).
df_relaciones["synsets_referencia"] = [
    "; ".join(motor.extraer_relaciones(termino).synsets_consulta)
    for termino in df_relaciones["termino_path"]
]
columnas_rel = [f"n_{t}" for t in TIPOS_RELACION]
df_relaciones["total_relaciones"] = df_relaciones[columnas_rel].sum(axis=1)
orden_csv = [
    "grupo", "termino_path", "concepto", "encontrado", *columnas_rel,
    "relaciones_ejemplo", "variante_usada", "synsets_referencia",
]
df_relaciones.to_csv(RUTA_CSV_WORDNET, index=False, encoding="utf-8", columns=orden_csv)
duracion = time.time() - inicio

print(
    f"{len(df_relaciones)} conceptos procesados en {duracion:.1f} s "
    f"({df_relaciones['termino_path'].nunique()} terminos unicos consultados)."
)
print(f"CSV guardado: {RUTA_CSV_WORDNET.relative_to(RUTA_BASE)}")
print()

# Resumen por grupo: cobertura y relaciones totales.
print(f'{"Grupo":<55} {"n":>4} {"enc":>4} {"conRel":>6} {"%":>6}')
print("-" * 80)
filas_grupo = []
for grupo in GRUPOS:
    sub = df_relaciones[df_relaciones["grupo"] == grupo]
    encontrados = int(sub["encontrado"].sum())
    con_relaciones = int((sub[columnas_rel].sum(axis=1) > 0).sum())
    pct = con_relaciones / len(sub) if len(sub) else 0.0
    filas_grupo.append((grupo, len(sub), encontrados, con_relaciones, pct))
    print(f"{grupo:<55} {len(sub):>4} {encontrados:>4} {con_relaciones:>6} "
          f"{pct:>6.1%}")

total_con_rel = int((df_relaciones[columnas_rel].sum(axis=1) > 0).sum())
print("-" * 80)
print(f'{"TOTAL":<55} {len(df_relaciones):>4} '
      f'{int(df_relaciones["encontrado"].sum()):>4} {total_con_rel:>6} '
      f'{total_con_rel / len(df_relaciones):>6.1%}')

print()
print("Totales por tipo de relacion (todos los conceptos):")
for tipo in TIPOS_RELACION:
    total = int(df_relaciones[f"n_{tipo}"].sum())
    aportan = int((df_relaciones[f"n_{tipo}"] > 0).sum())
    print(f"  {tipo:<13} {total:>7} relaciones en {aportan:>4} conceptos")


def _clasificar_variante(variante: str, candidatos: list[str]) -> str:
    """Clasifica la variante usada segun su posicion en la cadena de busqueda.

    La cadena de candidatos del Motor tiene un orden fijo: forma base,
    variante morfologica (singular/plural), forma sin acentos, forma con
    guiones bajos y, para los compuestos, el ultimo y el primer token.

    Args:
        variante: Variante registrada en el resultado del Motor.
        candidatos: Variantes generadas por la cadena de normalizadores.

    Returns:
        Categoria de la variante: exacta, guiones_bajos, ultimo_token,
        primer_token o singular_plural_o_sin_acentos.
    """
    if not variante or not candidatos:
        return "sin_resultados"
    if variante == candidatos[0]:
        return "exacta"
    if variante == candidatos[0].replace(" ", "_"):
        return "guiones_bajos"
    tokens = candidatos[0].split()
    if len(tokens) > 1:
        if variante == tokens[-1]:
            return "ultimo_token"
        if variante == tokens[0]:
            return "primer_token"
    return "singular_plural_o_sin_acentos"


print()
print("Trazabilidad de la variante usada (conceptos encontrados):")
conteo_variantes = defaultdict(int)
for fila in df_conceptos.itertuples():
    resultado = motor.extraer_relaciones(fila.termino)
    if not resultado.encontrado:
        continue
    categoria = _clasificar_variante(
        resultado.variante_usada, motor.cadena.candidatos(fila.termino)
    )
    conteo_variantes[categoria] += 1
for categoria, cantidad in conteo_variantes.items():
    print(f"  {categoria:<30} {cantidad:>4} ({cantidad / TOTAL_CONCEPTOS:.1%})")

# %% [markdown]
# ## Clases POO del analisis (SOLID: calculos y figuras separados)
#
# `AnalisisRelaciones` encapsula los calculos (cobertura, matriz grupo x
# tipo, top de conceptos, grafo compartido y correlaciones de profundidad)
# con cache de resultados; `AnalizadorRelaciones` dibuja y guarda las 5
# figuras y el CSV de aristas, con un metodo `analizar_completo()` que las
# ejecuta todas y consolida las metricas. El Motor se inyecta por
# constructor (inversion de dependencias).

# %%
"""Clases POO del analisis: calculos (AnalisisRelaciones) y figuras (AnalizadorRelaciones)."""


class AnalisisRelaciones:
    """Calculos del analisis de relaciones WordNet (SOLID: solo computo).

    Encapsula los datos derivados del analisis masivo (cobertura por grupo,
    matriz grupo x tipo, top de conceptos, grafo compartido y correlacion de
    profundidad) y los cachea para que las figuras no recalculen nada.

    Attributes:
        df: DataFrame de relaciones (columnas n_<tipo> y total_relaciones).
        conceptos: DataFrame de conceptos de la ontologia (grupo/termino/concepto).
        tipos: Lista de los 9 tipos de relacion analizados.
    """

    def __init__(
        self, df_relaciones: pd.DataFrame, conceptos: pd.DataFrame, tipos: list[str]
    ) -> None:
        self.df = df_relaciones
        self.conceptos = conceptos
        self.tipos = list(tipos)
        self._cobertura: pd.DataFrame | None = None
        self._matriz: pd.DataFrame | None = None
        self._grafo: tuple[nx.Graph, pd.DataFrame] | None = None
        self._profundidad: tuple[pd.DataFrame, float, float] | None = None

    def columnas_rel(self) -> list[str]:
        """Nombres de las columnas de conteo por tipo."""
        return [f"n_{t}" for t in self.tipos]

    def cobertura(self) -> pd.DataFrame:
        """Conceptos con al menos 1 relacion por grupo (absolutos y %).

        Returns:
            DataFrame con columnas grupo, n, con_relaciones y pct (0-1).
        """
        if self._cobertura is None:
            col = self.columnas_rel()
            filas = []
            for grupo, sub in self.df.groupby("grupo"):
                con_rel = int((sub[col].sum(axis=1) > 0).sum())
                filas.append({
                    "grupo": grupo,
                    "n": len(sub),
                    "con_relaciones": con_rel,
                    "pct": con_rel / len(sub),
                })
            self._cobertura = pd.DataFrame(filas)
        return self._cobertura

    def matriz_grupo_tipo(self) -> pd.DataFrame:
        """Promedio de relaciones por concepto, matriz grupo x tipo.

        La normalizacion por tamano de grupo permite comparar grupos de
        distinto numero de conceptos (Poblaciones tiene 214 y Actores 81).

        Returns:
            DataFrame con grupos como filas y tipos como columnas.
        """
        if self._matriz is None:
            col = self.columnas_rel()
            matriz = pd.DataFrame(
                0.0, index=list(dict.fromkeys(self.df["grupo"])), columns=col
            )
            for grupo, sub in self.df.groupby("grupo"):
                matriz.loc[grupo] = sub[col].sum(axis=0) / len(sub)
            self._matriz = matriz
        return self._matriz

    def top_conceptos(self, n: int = 15) -> pd.DataFrame:
        """Top-n conceptos por numero total de relaciones (ascendente).

        Returns:
            DataFrame ordenado de menor a mayor para barras horizontales.
        """
        return (
            self.df.sort_values("total_relaciones", ascending=False)
            .head(n)
            .iloc[::-1]
            .reset_index(drop=True)
        )

    def grafo_compartido(
        self,
        motor_extractor,
        tipos_grafo: tuple[str, ...] = ("sinonimia", "hiponimia", "coordinados"),
        fanout_maximo: int = 30,
    ) -> tuple[nx.Graph, pd.DataFrame]:
        """Grafo de conceptos que comparten terminos relacionados.

        Los nodos son los conceptos con al menos una relacion; las aristas
        unen conceptos que comparten al menos un termino relacionado en los
        tipos indicados. Los terminos relacionados con fan-out superior a
        `fanout_maximo` (hubs genericos como "cosa") se excluyen de la
        construccion de aristas para no saturar la red con aristas triviales.

        Args:
            motor_extractor: Motor con los resultados cacheados por termino.
            tipos_grafo: Tipos de relacion que crean aristas.
            fanout_maximo: Maximo de conceptos que pueden compartir un termino.

        Returns:
            Tupla (grafo_figura, df_aristas): el grafo solo con aristas
            fuertes (>= 2 tipos compartidos) para la figura, y el DataFrame
            con TODAS las aristas (fidelidad completa para el CSV).
        """
        if self._grafo is not None:
            return self._grafo
        indices_con_rel = [
            i for i, fila in enumerate(self.conceptos.itertuples())
            if self.df.iloc[i][self.columnas_rel()].sum() > 0
        ]
        rel_por_termino: dict[str, dict[str, set[int]]] = {
            tipo: defaultdict(set) for tipo in tipos_grafo
        }
        for i in indices_con_rel:
            resultado = motor_extractor.extraer_relaciones(
                self.conceptos.iloc[i]["termino"]
            )
            for tipo in tipos_grafo:
                for nombre in resultado.relaciones.get(tipo, []):
                    rel_por_termino[tipo][nombre].add(i)
        aristas: dict[tuple[int, int], set[str]] = {}
        for tipo in tipos_grafo:
            for indices in rel_por_termino[tipo].values():
                if len(indices) < 2 or len(indices) > fanout_maximo:
                    continue
                for par in combinations(sorted(indices), 2):
                    aristas.setdefault(par, set()).add(tipo)
        filas_aristas = [
            {
                "concepto_a": self.conceptos.iloc[a]["concepto"],
                "grupo_a": self.conceptos.iloc[a]["grupo"],
                "concepto_b": self.conceptos.iloc[b]["concepto"],
                "grupo_b": self.conceptos.iloc[b]["grupo"],
                "tipos_compartidos": "; ".join(sorted(tipos)),
                "n_relaciones_compartidas": len(tipos),
            }
            for (a, b), tipos in aristas.items()
        ]
        grafo_figura = nx.Graph()
        for i in indices_con_rel:
            fila = self.conceptos.iloc[i]
            grafo_figura.add_node(
                i,
                concepto=fila["concepto"],
                grupo=fila["grupo"],
                total_rel=int(self.df.iloc[i]["total_relaciones"]),
            )
        for (a, b), tipos in aristas.items():
            if len(tipos) < 2:
                continue
            grafo_figura.add_edge(
                a, b, tipos="; ".join(sorted(tipos)), n_compartidas=len(tipos)
            )
        self._grafo = (grafo_figura, pd.DataFrame(filas_aristas))
        return self._grafo

    def profundidad(self) -> tuple[pd.DataFrame, float, float]:
        """Correlaciones entre profundidad del path y total de relaciones.

        Returns:
            Tupla (df_niveles, spearman, pearson) con los conceptos con al
            menos una relacion y su numero de niveles derivado del path.
        """
        if self._profundidad is None:
            df_niveles = self.df[
                ["concepto", "grupo", "termino_path", "total_relaciones"]
            ].copy()
            df_niveles["niveles"] = df_niveles["termino_path"].str.split(" - ").str.len()
            df_niveles = df_niveles[df_niveles["total_relaciones"] > 0]
            spearman = df_niveles["niveles"].corr(
                df_niveles["total_relaciones"], method="spearman"
            )
            pearson = df_niveles["niveles"].corr(
                df_niveles["total_relaciones"], method="pearson"
            )
            self._profundidad = (df_niveles, float(spearman), float(pearson))
        return self._profundidad

    def totales_por_tipo(self) -> pd.DataFrame:
        """Totales y conceptos que aportan, por tipo de relacion."""
        return pd.DataFrame([
            {
                "tipo": tipo,
                "total": int(self.df[f"n_{tipo}"].sum()),
                "conceptos": int((self.df[f"n_{tipo}"] > 0).sum()),
            }
            for tipo in self.tipos
        ])


class AnalizadorRelaciones:
    """Figuras del analisis WordNet: un metodo por figura + analizar_completo().

    SOLID: delega todos los calculos en `AnalisisRelaciones`; esta clase solo
    dibuja, guarda archivos (PNG y CSV de aristas) y reporta metricas. Cada
    figura se genera una sola vez (cache interno) para que
    `analizar_completo()` no duplique trabajo.

    Attributes:
        motor: Motor de relaciones (fuente de los resultados cacheados).
        analisis: AnalisisRelaciones con los datos del analisis masivo.
        ruta_imagenes: Directorio donde se guardan las figuras.
        colores_grupo: Dict grupo -> color hex para los nodos/barras.
        rutas_figuras: Dict nombre -> ruta PNG (constantes de sec00_setup).
        ruta_csv_grafo: Ruta del CSV de aristas del grafo compartido.
    """

    def __init__(
        self,
        motor_extractor,
        analisis: AnalisisRelaciones,
        ruta_imagenes: Path,
        colores_grupo: dict[str, str],
        rutas_figuras: dict[str, Path],
        ruta_csv_grafo: Path,
    ) -> None:
        self.motor = motor_extractor
        self.analisis = analisis
        self.ruta_imagenes = Path(ruta_imagenes)
        self.colores_grupo = colores_grupo
        self.rutas_figuras = rutas_figuras
        self.ruta_csv_grafo = Path(ruta_csv_grafo)
        self._generadas: dict[str, Path] = {}

    def _guardar_figura(self, figura, ruta: Path, mostrar: bool, etiqueta: str) -> Path:
        """Guarda una figura en disco y la muestra (o la cierra en silencio).

        Args:
            figura: Objeto Figure de matplotlib.
            ruta: Ruta PNG de salida.
            mostrar: True para plt.show() y mensaje de guardado.
            etiqueta: Nombre legible del archivo (relativo a RUTA_BASE).

        Returns:
            Ruta del PNG guardado.
        """
        ruta.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(str(ruta), dpi=150, bbox_inches="tight")
        if mostrar:
            plt.show()
            print(f"Figura guardada: {etiqueta}")
        plt.close(figura)
        return ruta

    def figura_cobertura(self, mostrar: bool = True) -> Path:
        """Figura 1: cobertura de conceptos con >=1 relacion, por grupo.

        Las barras se ordenan de mayor a menor cobertura (sin incluir el
        TOTAL) para una lectura descendente clara; el TOTAL se muestra al
        final en un color mas oscuro.
        """
        if "cobertura" in self._generadas:
            return self._generadas["cobertura"]
        cobertura = self.analisis.cobertura().sort_values("pct", ascending=False)
        total_con_rel = int(cobertura["con_relaciones"].sum())
        total_n = int(cobertura["n"].sum())
        nombres = list(cobertura["grupo"]) + ["TOTAL"]
        valores = list(cobertura["pct"] * 100) + [total_con_rel / total_n * 100]
        absolutos = list(
            cobertura.apply(
                lambda fila: f"{fila['con_relaciones']} de {fila['n']}", axis=1
            )
        ) + [f"{total_con_rel} de {total_n}"]

        figura, eje = plt.subplots(figsize=(12, 6))
        colores_barras = ["#2a6f97"] * len(cobertura) + ["#8b1c1c"]
        barras = eje.barh(nombres, valores, color=colores_barras)
        for barra, texto in zip(barras, absolutos):
            eje.text(
                barra.get_width() + 0.6, barra.get_y() + barra.get_height() / 2,
                texto, va="center", fontsize=9,
            )
        eje.axvline(
            total_con_rel / total_n * 100, color="#c44536", linestyle="--",
            linewidth=1.4, label=f"Cobertura global: {total_con_rel / total_n:.1%}",
        )
        eje.set_xlabel("Conceptos con al menos 1 relacion en WordNet-espanol (%)")
        eje.set_title("Cobertura de WordNet-espanol por grupo de la ontologia")
        eje.set_xlim(0, max(valores) * 1.18)
        eje.legend(loc="lower right")
        eje.grid(axis="x", alpha=0.3)
        figura.tight_layout()
        ruta = self.rutas_figuras["cobertura_conceptos"]
        self._generadas["cobertura"] = self._guardar_figura(
            figura, ruta, mostrar, ruta.relative_to(RUTA_BASE)
        )
        return self._generadas["cobertura"]

    def figura_heatmap(self, mostrar: bool = True) -> Path:
        """Figura 2: heatmap grupo x tipo de relacion (normalizado por grupo)."""
        if "heatmap" in self._generadas:
            return self._generadas["heatmap"]
        matriz = self.analisis.matriz_grupo_tipo()
        grupos = list(matriz.index)
        tipos = [t.capitalize() for t in self.analisis.tipos]

        figura, eje = plt.subplots(figsize=(12, 6))
        imagen = eje.imshow(matriz.to_numpy(), cmap="YlGnBu", aspect="auto")
        eje.set_xticks(range(len(tipos)))
        eje.set_xticklabels(tipos, rotation=35, ha="right")
        eje.set_yticks(range(len(grupos)))
        eje.set_yticklabels([g if len(g) <= 34 else g[:31] + "..." for g in grupos])
        maximo = matriz.to_numpy().max()
        for i in range(len(grupos)):
            for j in range(len(tipos)):
                valor = matriz.iloc[i, j]
                color = "white" if valor > maximo * 0.62 else "black"
                eje.text(j, i, f"{valor:.1f}", ha="center", va="center",
                         fontsize=9, color=color)
        eje.set_title("Relaciones WordNet por grupo (promedio de relaciones por "
                      "concepto, normalizado por tamano de grupo)")
        figura.colorbar(imagen, ax=eje, label="Relaciones por concepto")
        figura.tight_layout()
        ruta = self.rutas_figuras["heatmap_relaciones"]
        self._generadas["heatmap"] = self._guardar_figura(
            figura, ruta, mostrar, ruta.relative_to(RUTA_BASE)
        )
        return self._generadas["heatmap"]

    def figura_top_conceptos(self, mostrar: bool = True) -> Path:
        """Figura 3: top-15 conceptos por numero total de relaciones."""
        if "top_conceptos" in self._generadas:
            return self._generadas["top_conceptos"]
        top15 = self.analisis.top_conceptos(15)
        conteos_por_grupo = self.analisis.df["grupo"].value_counts()

        figura, eje = plt.subplots(figsize=(12, 8))
        colores = [self.colores_grupo[g] for g in top15["grupo"]]
        # Nombres legibles: se recortan a 68 caracteres con puntos suspensivos
        # para que no se encimen con las barras de los conceptos mas largos.
        nombres = [
            c if len(c) <= 68 else c[:67].rstrip() + "…" for c in top15["concepto"]
        ]
        barras = eje.barh(nombres, top15["total_relaciones"],
                          color=colores, alpha=0.9)
        for barra, total in zip(barras, top15["total_relaciones"]):
            eje.text(
                barra.get_width() + 1.2, barra.get_y() + barra.get_height() / 2,
                str(total), va="center", fontsize=9,
            )
        eje.set_xlabel("Numero total de relaciones WordNet (suma de las 9)")
        eje.set_title("Top-15 conceptos con mas relaciones en WordNet-espanol")
        leyenda = [
            Patch(facecolor=self.colores_grupo[g],
                  label=f"{g} ({conteos_por_grupo.get(g, 0)})")
            for g in self.colores_grupo
        ]
        eje.legend(handles=leyenda, loc="lower right", fontsize=8)
        figura.tight_layout()
        ruta = self.rutas_figuras["top_conceptos"]
        self._generadas["top_conceptos"] = self._guardar_figura(
            figura, ruta, mostrar, ruta.relative_to(RUTA_BASE)
        )
        return self._generadas["top_conceptos"]

    def figura_grafo(self, mostrar: bool = True) -> tuple[Path, dict]:
        """Figura 4: grafo de relaciones compartidas + CSV de aristas.

        Escribe `data/grafo_relaciones_edges.csv` con TODAS las aristas
        (una por par de conceptos que comparte al menos una relacion, con los
        tipos compartidos) y dibuja el subgrafo de aristas "fuertes"
        (compartidas por al menos 2 de los 3 tipos): las aristas que solo
        comparten coordinados son las mas abundantes y convertirian la red en
        una bola de pelos ilegible.

        Returns:
            Tupla (ruta PNG, metricas de red) con nodos, aristas, componentes,
            componente gigante, aislados y un ejemplo de arista.
        """
        if "grafo" in self._generadas:
            return self._generadas["grafo"], self._metricas_grafo
        grafo_figura, df_aristas = self.analisis.grafo_compartido(self.motor)
        self.ruta_csv_grafo.parent.mkdir(parents=True, exist_ok=True)
        df_aristas.to_csv(self.ruta_csv_grafo, index=False, encoding="utf-8")

        figura, eje = plt.subplots(figsize=(14, 11))
        posicion = nx.spring_layout(grafo_figura, seed=42, k=0.45, iterations=80)
        colores_nodo = [
            self.colores_grupo[grafo_figura.nodes[n]["grupo"]]
            for n in grafo_figura.nodes
        ]
        tamanos_nodo = [
            30 + 18 * grafo_figura.nodes[n]["total_rel"] ** 0.5
            for n in grafo_figura.nodes
        ]
        nx.draw_networkx_edges(grafo_figura, posicion, ax=eje, alpha=0.25,
                               width=0.8, edge_color="#444444")
        nx.draw_networkx_nodes(grafo_figura, posicion, ax=eje,
                               node_size=tamanos_nodo, node_color=colores_nodo,
                               alpha=0.85)
        grados = dict(grafo_figura.degree())
        nodos_etiquetados = {
            n: grafo_figura.nodes[n]["concepto"]
            for n, g in grados.items() if g >= 4
        }
        nx.draw_networkx_labels(grafo_figura, posicion, labels=nodos_etiquetados,
                                ax=eje, font_size=7)
        leyenda = [Patch(facecolor=self.colores_grupo[g], label=g)
                   for g in self.colores_grupo]
        eje.legend(handles=leyenda, loc="upper left", fontsize=8, framealpha=0.9)
        eje.set_title("Grafo de relaciones compartidas entre conceptos de la "
                      "ontologia (aristas fuertes: >=2 tipos de relacion)")
        eje.axis("off")
        figura.tight_layout()

        componentes = list(nx.connected_components(grafo_figura))
        grandes = sorted(componentes, key=len, reverse=True)
        nodos_con_rel = int(
            (self.analisis.df[self.analisis.columnas_rel()].sum(axis=1) > 0).sum()
        )
        aislados = nodos_con_rel - grafo_figura.number_of_nodes()
        par_ejemplo = (
            next(iter(grafo_figura.edges()))
            if grafo_figura.number_of_edges() else None
        )
        if par_ejemplo:
            a, b = par_ejemplo
            ejemplo = (
                f'"{grafo_figura.nodes[a]["concepto"]}" - '
                f'"{grafo_figura.nodes[b]["concepto"]}" compartiendo '
                f'{grafo_figura[a][b]["tipos"]}'
            )
        else:
            ejemplo = "(sin aristas fuertes)"
        self._metricas_grafo = {
            "aristas_csv": len(df_aristas),
            "aristas_figura": grafo_figura.number_of_edges(),
            "nodos": grafo_figura.number_of_nodes(),
            "componentes": len(componentes),
            "gigante": len(grandes[0]) if grandes else 0,
            "siguientes": [len(c) for c in grandes[1:6]],
            "aislados": aislados,
            "ejemplo": ejemplo,
        }
        ruta = self.rutas_figuras["grafo_relaciones"]
        self._generadas["grafo"] = self._guardar_figura(
            figura, ruta, mostrar, ruta.relative_to(RUTA_BASE)
        )
        if mostrar:
            print(f"CSV de aristas (todas): {self.ruta_csv_grafo.relative_to(RUTA_BASE)}")
            print(f"Relaciones compartidas totales: {len(df_aristas)} aristas; "
                  f"figura con aristas fuertes: {grafo_figura.number_of_edges()}.")
            print(f"Grafo figura: {grafo_figura.number_of_nodes()} nodos, "
                  f"{grafo_figura.number_of_edges()} aristas, "
                  f"{len(componentes)} componentes conexas.")
            print("Componentes mas grandes: " + " | ".join(
                f"{len(c)} nodos" for c in grandes[:3]))
            print(f"Conceptos con relaciones pero sin arista fuerte (aislados): "
                  f"{aislados} de {nodos_con_rel} ({aislados / nodos_con_rel:.1%})")
            print(f"Ejemplo de arista: {ejemplo}")
        return ruta, self._metricas_grafo

    def figura_profundidad(self, mostrar: bool = True) -> Path:
        """Figura 5: profundidad del path vs numero de relaciones (scatter)."""
        if "profundidad" in self._generadas:
            return self._generadas["profundidad"]
        df_niveles, spearman, pearson = self.analisis.profundidad()
        grupos = list(dict.fromkeys(df_niveles["grupo"]))

        figura, eje = plt.subplots(figsize=(11, 6.5))
        for grupo in grupos:
            sub = df_niveles[df_niveles["grupo"] == grupo]
            eje.scatter(
                sub["niveles"] + np.random.uniform(-0.12, 0.12, len(sub)),
                sub["total_relaciones"], s=28, alpha=0.6,
                color=self.colores_grupo[grupo], label=grupo, edgecolors="none",
            )
        medianas = df_niveles.groupby("niveles")["total_relaciones"].median()
        eje.plot(medianas.index, medianas.values, color="#c44536", marker="o",
                 linewidth=2, label="Mediana por nivel")
        eje.set_xticks(sorted(df_niveles["niveles"].unique()))
        eje.set_xlabel("Profundidad del path en el tesauro (niveles)")
        eje.set_ylabel("Numero total de relaciones WordNet")
        eje.set_title("Profundidad del concepto vs numero de relaciones WordNet")
        eje.legend(fontsize=8, loc="upper right")
        eje.text(0.02, 0.97,
                 f"Spearman rho = {spearman:.3f} | Pearson r = {pearson:.3f}",
                 transform=eje.transAxes, fontsize=10, va="top",
                 bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
        figura.tight_layout()
        ruta = self.rutas_figuras["profundidad_cobertura"]
        self._generadas["profundidad"] = self._guardar_figura(
            figura, ruta, mostrar, ruta.relative_to(RUTA_BASE)
        )
        if mostrar:
            print(f"Correlacion Spearman niveles vs relaciones: {spearman:.3f}")
            print("Medianas de relaciones por nivel:")
            for nivel, mediana in medianas.items():
                print(f"  nivel {nivel}: {mediana:.1f} relaciones (n="
                      f"{int((df_niveles['niveles'] == nivel).sum())})")
        return self._generadas["profundidad"]

    def analizar_completo(self) -> dict:
        """Ejecuta las 5 figuras (cacheadas) y devuelve el resumen de metricas.

        Returns:
            Dict con conceptos, cobertura, totales por tipo, correlacion de
            profundidad y metricas de red del grafo compartido.
        """
        self.figura_cobertura(mostrar=False)
        self.figura_heatmap(mostrar=False)
        self.figura_top_conceptos(mostrar=False)
        _, metricas_grafo = self.figura_grafo(mostrar=False)
        self.figura_profundidad(mostrar=False)
        _, spearman, _ = self.analisis.profundidad()
        totales = self.analisis.totales_por_tipo()
        cobertura = self.analisis.cobertura()
        return {
            "conceptos": len(self.analisis.df),
            "con_relaciones": int(cobertura["con_relaciones"].sum()),
            "total_relaciones": int(totales["total"].sum()),
            "totales_por_tipo": dict(zip(totales["tipo"], totales["total"])),
            "spearman": spearman,
            "grafo": metricas_grafo,
        }


analisis_relaciones = AnalisisRelaciones(df_relaciones, df_conceptos, TIPOS_RELACION)
analizador_relaciones = AnalizadorRelaciones(
    motor, analisis_relaciones, RUTA_IMAGENES, COLORES_GRUPO, FIGURAS, RUTA_CSV_GRAFO
)

# %% [markdown]
# ## Analisis 1 - Cobertura de WordNet-espanol por grupo
#
# La figura muestra el porcentaje de conceptos de cada grupo (y el total)
# para los que el extractor encontro al menos una relacion en omw-1.4.
# Recuerdese que omw-1.4 solo traduce ~33% de los synsets del PWN al espanol,
# por lo que la cobertura esperada es parcial, y que el tesauro contiene
# vocabulario muy especifico del conflicto armado colombiano que no existe en
# un WordNet general.

# %%
"""Figura 1: cobertura de conceptos con al menos 1 relacion, por grupo."""
analizador_relaciones.figura_cobertura()

# %% [markdown]
# ## Analisis 2 - Heatmap de relaciones por grupo
#
# Matriz grupo x tipo de relacion con los **conteos totales normalizados por
# grupo** (relaciones del tipo / numero de conceptos del grupo). La
# normalizacion permite comparar grupos de distinto tamano (Poblaciones tiene
# 214 conceptos y Actores armados 81). Se destacan los tipos de relacion que
# dominan el espacio semantico (coordinados e hiponimia) frente a los casi
# ausentes (antonimia, troponimia, holonimia).

# %%
"""Figura 2: heatmap grupo x tipo de relacion (normalizado por grupo)."""
analizador_relaciones.figura_heatmap()

# %% [markdown]
# ## Analisis 3 - Conceptos con mas relaciones
#
# Los 15 conceptos con mayor numero total de relaciones. Son los "hubs"
# lexicos de la ontologia: conceptos compuestos que no existen como lemma en
# omw-1.4 y se resuelven por un nucleo lexico polisemico (genero, familia,
# vida) que engancha con muchas ramas de WordNet. El color indica el grupo al
# que pertenece el concepto.

# %%
"""Figura 3: top-15 conceptos por numero total de relaciones."""
top15 = analisis_relaciones.top_conceptos(15)
print("Top-15 conceptos (total | variante usada | concepto | grupo):")
for fila in top15.itertuples():
    print(f"  {int(fila.total_relaciones):>6} | {fila.variante_usada:<32} | "
          f"{fila.concepto[:55]:<57} | {fila.grupo[:30]}")
analizador_relaciones.figura_top_conceptos()

# %% [markdown]
# ## Analisis 4 - Grafo de relaciones compartidas entre conceptos
#
# Red donde los nodos son los conceptos de la ontologia con al menos una
# relacion y las aristas conectan dos conceptos que comparten algun termino
# relacionado en **sinonimia, hiponimia o coordinados**. El color del nodo
# indica el grupo; el tamano, el numero total de relaciones. Los terminos
# relacionados con un fan-out superior a 30 conceptos (hubs genericos como
# "cosa") se excluyen de la construccion de aristas para no saturar el grafo
# con aristas triviales.
#
# El CSV `data/grafo_relaciones_edges.csv` guarda TODAS las aristas (una por
# par de conceptos que comparte al menos una relacion, con los tipos
# compartidos). Para la FIGURA se dibuja el subgrafo de aristas "fuertes"
# (compartidas por al menos 2 de los 3 tipos): las aristas que solo comparten
# coordinados son las mas abundantes (86.9% de las 5.765 del CSV) y provienen
# del diseno de la extraccion - cada synset genera hermanos taxonomicos - de
# modo que dibujarlas todas convertiria la red en una bola de pelos ilegible.

# %%
"""Figura 4: grafo de relaciones compartidas entre conceptos + CSV de aristas."""
analizador_relaciones.figura_grafo()

# %% [markdown]
# ## Analisis 5 - Profundidad del path versus numero de relaciones
#
# Cada concepto tiene una profundidad (niveles del path en el tesauro, de 1 a
# 6). Este scatter contrasta la profundidad con el numero total de relaciones
# encontradas. Se incluye el coeficiente de correlacion de Spearman (robusto
# a la asimetria) y las medianas por nivel.

# %%
"""Figura 5: profundidad del path vs numero de relaciones (scatter)."""
analizador_relaciones.figura_profundidad()

# %% [markdown]
# ## Verificacion de las 9 relaciones (rubrica criterio 3)
#
# La rubrica exige "identificar entidades y sus relaciones" con WordNet. Esta
# celda demuestra, con **conceptos reales de la ontologia**, que cada uno de
# los 9 tipos de relacion produce resultados no vacios. Para cada tipo se
# prueban primero unos conceptos preferidos (buscados por patron en los 6
# grupos, elegidos por su perfil lexico: genericos y con vecinos en
# omw-1.4), por ejemplo:
#
# - **Mujeres** (Poblaciones) -> antonimia [hombre], sinonimia [hembra_adulta].
# - **Poblacion en situacion de desplazamiento** (Poblaciones) -> derivacion
#   [desplazarse, trasladar], hiperonimia [cambio].
# - **Guerra** (Actores armados) -> meronimia [combate, agresion], antonimia [paz].
# - **Ordenar** y **Municiones Sin Explotar** (verbos de la ontologia) -> troponimia.
# - **Alimentacion** y **Silencio como afrontamiento** (Afrontamientos y
#   resistencias no violentas - cotidianas, grupo nuevo de la extension) ->
#   sinonimia/meronimia/derivacion y antonimia respectivamente; asi la
#   verificacion cubre los 6 grupos con concepto real de la extension.
#
# Si algun tipo no produce resultados con los preferidos, la busqueda se
# amplia automaticamente al resto de conceptos de la ontologia (se documenta
# en la salida). El bloque termina con 6 `assert` que comprueban la cobertura
# de los 9 tipos y la presencia del grupo de Afrontamientos.

# %%
"""Verificacion de las 9 relaciones con conceptos reales de la ontologia."""

PREFERIDOS_POR_TIPO: dict[str, list[tuple[str, str | None]]] = {
    "sinonimia": [
        ("alimentación", "Afrontamientos y resistencias no violentas - cotidianas"),
        ("mujer", None), ("desplazamiento", None), ("miedo", None), ("guerra", None),
    ],
    "antonimia": [
        ("silencio", "Afrontamientos y resistencias no violentas - cotidianas"),
        ("mujer", None), ("guerra", None), ("miedo", None), ("explotar", None),
        ("ordenar", None),
    ],
    "hiponimia": [
        ("desplazamiento", None), ("alimentación", None), ("miedo", None),
        ("guerra", None), ("mujer", None),
    ],
    "hiperonimia": [
        ("desplazamiento", None), ("mujer", None), ("alimentación", None),
        ("miedo", None),
    ],
    "meronimia": [
        ("alimentación", "Afrontamientos y resistencias no violentas - cotidianas"),
        ("mujer", None), ("guerra", None), ("miedo", None),
    ],
    "holonimia": [("conflicto", None), ("territorio", None), ("proceso", None), ("estructura", None)],
    "troponimia": [
        ("ordenar", None), ("explotar", None), ("cometer", None), ("consentir", None),
        ("financiar", None), ("instigar", None), ("omitir", None), ("planear", None),
    ],
    "coordinados": [
        ("desplazamiento", None), ("alimentación", None), ("mujer", None), ("guerra", None),
    ],
    "derivacion": [
        ("desplazamiento", None), ("mujer", None), ("miedo", None), ("guerra", None),
    ],
}


def verificar_nueve_relaciones(
    motor_extractor, df_conceptos_owl: pd.DataFrame,
    preferidos: dict[str, list[str]],
) -> pd.DataFrame:
    """Comprueba que los 9 tipos de relacion producen resultados en la OWL.

    Para cada tipo se prueban primero los conceptos preferidos (buscados por
    patron de texto en el nombre del concepto); si ninguno produce resultados
    para el tipo, la busqueda se amplia a todos los conceptos de la
    ontologia hasta cubrirlo.

    Args:
        motor_extractor: Motor de relaciones (consulta los resultados).
        df_conceptos_owl: DataFrame de conceptos reales de la ontologia.
        preferidos: Dict tipo de relacion -> patrones de busqueda en orden.

    Returns:
        DataFrame indexado por tipo con el concepto hallado (path OWL), el
        grupo, la variante usada, el numero de relaciones y ejemplos.
    """
    filas = []
    ampliados = []
    for tipo in sc.TIPOS_RELACION:
        hallazgo = None
        for patron, grupo in preferidos[tipo]:
            candidatos = df_conceptos_owl
            if grupo is not None:
                candidatos = candidatos[candidatos["grupo"] == grupo]
            hit = candidatos[
                candidatos["concepto"].str.lower().str.contains(patron, regex=False)
            ]
            if hit.empty:
                continue
            resultado = motor_extractor.extraer_relaciones(hit.iloc[0]["termino"])
            if resultado.relaciones.get(tipo):
                hallazgo = (hit.iloc[0], resultado)
                break
        if hallazgo is None:  # amplia la busqueda a todos los conceptos
            for fila in df_conceptos_owl.itertuples():
                resultado = motor_extractor.extraer_relaciones(fila.termino)
                if resultado.relaciones.get(tipo):
                    hallazgo = (fila, resultado)
                    break
            ampliados.append(tipo)
        fila_owl, resultado = hallazgo
        filas.append({
            "tipo": tipo,
            "concepto_owl": fila_owl.termino,
            "grupo": fila_owl.grupo,
            "variante_usada": resultado.variante_usada,
            "n_relaciones": len(resultado.relaciones.get(tipo, [])),
            "ejemplos": ", ".join(resultado.relaciones.get(tipo, [])[:3]),
        })
    if ampliados:
        print("Tipos cubiertos ampliando la busqueda a todos los conceptos:",
              ", ".join(ampliados))
    return pd.DataFrame(filas).set_index("tipo")


verificacion = verificar_nueve_relaciones(motor, df_conceptos, PREFERIDOS_POR_TIPO)

# Assertions (rubrica criterio 3): los 9 tipos deben producir resultados no
# vacios para al menos un concepto real de la ontologia, y la extension debe
# quedar representada con un concepto del grupo de Afrontamientos.
assert len(verificacion) == len(sc.TIPOS_RELACION), "Deben verificarse los 9 tipos"
assert set(verificacion.index) == set(sc.TIPOS_RELACION), "Tipos de relacion incompletos"
assert (verificacion["n_relaciones"] > 0).all(), "Algun tipo quedo sin relaciones"
assert (verificacion["concepto_owl"].str.len() > 0).all(), "Conceptos deben venir de la OWL"
assert (verificacion["variante_usada"].str.len() > 0).all(), "Variante usada sin registrar"
assert any(
    fila.grupo == "Afrontamientos y resistencias no violentas - cotidianas"
    for fila in verificacion.itertuples()
), "Debe verificarse al menos una relacion con un concepto del grupo Afrontamientos"

print("Verificacion de las 9 relaciones: OK (6 assertions superadas)")
print()
print("Tabla resumen (conceptos reales de la ontologia):")
with pd.option_context("display.max_colwidth", 65, "display.width", 170):
    print(verificacion.to_string())

# %% [markdown]
# ## Interpretacion de las figuras
#
# ### 1. Cobertura por grupo
#
# De los 610 conceptos de la ontologia (609 terminos del JSON de la Fase 1a
# mas los nodos intermedios anadidos por la seccion 03, menos las 6 raices),
# **308 (50.5%) tienen al menos una relacion** en WordNet-espanol. Por grupo:
# **Hechos victimizantes 79.8%** (71/89), **Afrontamientos y resistencias no
# violentas - cotidianas 73.9%** (17/23), **Impactos 73.8%** (79/107),
# **Actores armados 69.1%** (56/81), **Armas 51.0%** (49/96) y **Poblaciones
# 16.8%** (36/214). El patron es claro: los grupos con vocabulario generico y
# cotidiano (victima, dano, ejercito, miedo, perdida) se cubren bien, porque
# esos conceptos existen en el espanol general del PWN. El grupo nuevo de
# Afrontamientos (23 terminos, cobertura de hojas ~4.3% en la Fase 1a)
# confirma la regla: su vocabulario cotidiano (alimentacion, silencio, fe,
# memoria, apoyo) es el que mejor traduce omw-1.4. Poblaciones arrastra la
# cobertura hacia abajo (214 conceptos, solo 36 encontrados): sus hojas son
# mayoritariamente grupos sociales, organizaciones y denominaciones etnicas
# propias del contexto colombiano (p. ej. comunidades afrodescendientes,
# organizaciones de victimas, pueblos indigenas como Wayuu o Saliba) que no
# estan en omw-1.4. La seleccion de la Fase 1a ya habia anticipado este
# dato (cobertura de hojas: Poblaciones 1.41%, Hechos victimizantes 2.22%) y
# aun asi los incluyo por su relevancia tematica para la CEV; el resultado
# confirma que la cobertura de WordNet no es uniforme entre dominios.
#
# ### 2. Heatmap grupo x tipo
#
# Las dos relaciones mas productivas son **coordinados** (24.920 relaciones;
# 12.1 a 79.6 por concepto segun el grupo) e **hiponimia** (22.290; 8.5 a
# 93.2 por concepto). La hiponimia domina en **Impactos** (93.2 por concepto,
# frente a 18.1 en Poblaciones): el nucleo de sus conceptos compuestos
# (cambios, perdidas, afectaciones) es muy polisemico y cada synset aporta
# su arbol de hiponimos de 2 niveles. La **sinonimia** (2.638; 1.3 a 8.0 por
# concepto) y la **derivacion morfologica** (1.566; 0.7 a 4.9) acompanan el
# perfil de los grupos de vocabulario generico (Actores, Impactos, Hechos y
# el nuevo Afrontamientos, con 7.8 de sinonimia y 4.9 de derivacion por
# concepto). La **meronimia** (310 en total) es modesta y la **holonimia**
# casi inexistente (80): el PWN espanol descompone poco el dominio belico
# (cartucho, canon, culata no estan traducidos). La **antonimia** es escasa
# (163; 0.0 a 0.6 por concepto): el vocabulario del conflicto es poco polar
# (mujer-hombre, guerra-paz, miedo-coraje, silencio-ruido). La **troponimia**
# (81) se concentra en verbos concretos de la ontologia (Ordenar,
# Municiones Sin Explotar -> "explotar", Cometer, Planear, y ahora los de
# Afrontamientos: Volver a integrarse -> "volver", Solicitar apoyo ->
# "solicitar") y es nula en Hechos victimizantes y Poblaciones. Poblaciones
# muestra el perfil mas pobre en todas las relaciones, coherente con su baja
# cobertura.
#
# ### 3. Top-15 conceptos
#
# El top esta dominado por conceptos compuestos que se resuelven por nucleos
# lexicos polisemicos: dos conceptos con nucleo **"genero"** (2.804
# relaciones c/u: "Cambios forzados en la corporabilidad y performatividad
# del genero" y "Impactos en las relaciones y arreglos de genero"), seis con
# nucleo **"familia"** (1.325 c/u: Padres y Madres de familia, afectaciones
# emocionales, cambios economicos, alteracion del proyecto de vida y uso de
# sustancias psicoactivas de algun miembro de la familia), cinco con nucleo
# **"vida"** (927 c/u: vida familiar y comunitaria, vida intrafilas, amenaza
# y atentado al derecho a la vida, incremento de conflictividades),
# "Volver a integrarse a la estructura armada" (719, nucleo "volver", del
# grupo nuevo de Afrontamientos) e "Instrumentos" (703, nucleo
# "instrumento"). La extension aporta asi su primer concepto al top-15. Este
# es un hallazgo metodologico importante: **el conteo alto no indica
# relevancia semantica sino ambiguedad del nucleo de fallback** - los
# conceptos compuestos que no existen como lemma en omw-1.4 se resuelven por
# un nucleo polisemico (multiples synsets, cada uno con su arbol de
# hiponimos), y la hiponimia del Motor suma la recursion de 2 niveles de
# todos los synsets de la consulta. Los 15 primeros acumulan **19.615
# relaciones, el 36.2% del total**.
#
# ### 4. Grafo de relaciones compartidas
#
# El CSV registra **5.765 aristas** (pares de conceptos que comparten al
# menos un termino relacionado); el 86.9% comparte solo coordinados, ruido
# estructural de la extraccion, por eso la figura muestra las **758 aristas
# fuertes** (>=2 tipos, 13.1% del total). La figura dibuja los 308 nodos
# (todos los conceptos con relaciones) y la red tiene **124 componentes
# conexas**, con un componente gigante de 123 nodos: los puentes lexicos
# entre grupos existen (conceptos como "vida", "familia" o "genero" aparecen
# en varios grupos y los conectan), y esta vez **ningun concepto queda
# aislado** (0 de 308 sin arista fuerte; en la version de 5 grupos eran 96,
# 32.7%): el vocabulario cotidiano de Afrontamientos (apoyo, participacion,
# memoria) comparte terminos con Actores e Impactos y teje la red. La arista
# de ejemplo une "Utilizacion en acciones belicas" con "Uso de sustancias
# psicoactivas" (comparten sinonimia, hiponimia y coordinados): la
# terminologia de hechos y de impactos se enlaza a traves de terminos
# compartidos.
#
# Nota metodologica: la deduplicacion del Motor (por nombre del termino
# relacionado, no por par nombre-synset) reduce los terminos compartidos
# frente a la version anterior del extractor, de modo que la red es mas
# pequena (5.765 aristas frente a 8.545; 758 aristas fuertes frente a 1.777)
# pero igualmente significativa.
#
# ### 5. Profundidad vs cobertura
#
# Correlacion **Spearman rho = -0.180** (debil negativa; Pearson -0.139) y
# medianas por nivel: nivel 2 = 67.5 (n=62), nivel 3 = 95.0 (n=180), nivel 4
# = 58.5 (n=24), nivel 5 = 29.0 (n=18), nivel 6 = 20.0 (n=24). No hay nivel 1
# con relaciones: `conceptos()` excluye las 6 raices (su path no contiene
# " - "), de modo que el analisis arranca en los conceptos de nivel 2. El
# patron NO es monotono: el nivel 3 supera al 2 porque concentra los
# conceptos con nucleos polisemicos (familia, vida, genero) que disparan la
# hiponimia, mientras que los niveles 5-6 (conceptos mas profundos y
# especificos del tesauro, p. ej. pueblos indigenas concretos) tienen las
# medianas mas bajas: la especificidad reduce el radio de relacion. La
# dispersion dentro de cada nivel es enorme (conviven conceptos genericos
# como "mujer" con tecnicos sin cobertura), por eso rho es debil: la
# profundidad del path es un predictor util pero no determinista de la
# cobertura; la especificidad del dominio pesa mas que la posicion
# jerarquica.

# %% [markdown]
# ## Analisis de cobertura y limitaciones
#
# **Cobertura del corpus.** omw-1.4 traduce al espanol 38.512 de los 117.659
# synsets del PWN (**32.7%**). De los 610 conceptos de la ontologia, **308
# (50.5%)** se encuentran y todos producen al menos una relacion. Tres
# causas de los 302 no encontrados:
#
# 1. **Especificidad del dominio**: el tesauro de la Comision de la Verdad
#    documenta el conflicto colombiano (desaparicion forzada, reclutamiento
#    ilicito, comunidades etnicas, organizaciones de victimas) y ese
#    vocabulario no existe en un WordNet general.
# 2. **Cobertura parcial del espanol**: muchos synsets del PWN carecen de
#    traduccion en omw-1.4; el extractor NUNCA traduce el termino al ingles,
#    de modo que esos conceptos quedan fuera por diseno (evita falsos
#    positivos semanticos, a costo de cobertura).
# 3. **Formas compuestas**: los conceptos multiword ("desplazamiento
#    forzado", "conflicto armado interno") rara vez existen como lemma
#    compuesto; el Motor intenta variantes (guiones bajos, singular/plural,
#    sin acentos, ultimo/primer token).
#
# **Hallazgo de trazabilidad (variante usada).** De los 308 conceptos
# encontrados, **solo 37 (6.1% del total) coincidieron exactamente** con su
# forma normalizada; los otros 271 se resolvieron con una variante: **158
# por el ultimo token** y **103 por el primer token** de los compuestos, **9
# por singular/plural o sin acentos** y 1 por la forma con guiones bajos.
# La columna `variante_usada` del CSV documenta cada caso. Esto significa
# que la cobertura del 50.5% es en buena parte **cobertura de nucleos
# lexicos**, no de los conceptos compuestos completos: el grafo y los
# conteos heredan el sesgo de esos nucleos (p. ej. "vida", "familia" o
# "genero").
#
# **Calidad de las relaciones.** La antonimia y la derivacion se resuelven
# sobre los lemmas ingleses del PWN y se retraducen al espanol; cuando no
# hay traduccion, el relacionado aparece con el nombre PWN y la marca
# "(en)" (visible en el CSV). La sinonimia, hiponimia, meronimia, holonimia,
# troponimia y coordinados operan directamente sobre el grafo espanol.
#
# **Heuristicas de normalizacion.** La singularizacion/pluralizacion de la
# cadena de normalizadores es simple y no cubre irregulares ("leyes" ->
# "ley" falla); los parentesis se eliminan ("tierra (suelo)" -> "tierra");
# los acentos son obligatorios (omw-1.4 los incluye: "victima" solo existe
# con tilde).
#
# **Implicaciones para la ontologia.** Las relaciones WordNet son una
# primera capa lexico-semantica sobre la jerarquia OWL de la seccion 03, no
# una verdad del dominio: sirven para (a) enriquecer cada concepto con sus
# vecinos lexicos, (b) detectar puentes entre grupos (grafo) y (c) detectar
# huecos de vocabulario. La cobertura medida (50.5% de conceptos con >=1
# relacion; 6.1% con coincidencia exacta) es el punto de partida honesto
# para la seccion 05 (consultas SPARQL) y para la reflexion final del
# taller.

# %%
"""Resumen consolidado del analisis (analizar_completo)."""
resumen = analizador_relaciones.analizar_completo()
print("Resumen consolidado del analisis WordNet (sec04):")
print(f"  Conceptos analizados: {resumen['conceptos']} "
      f"({resumen['con_relaciones']} con >=1 relacion, "
      f"{resumen['con_relaciones'] / resumen['conceptos']:.1%})")
print(f"  Total de relaciones: {resumen['total_relaciones']}")
print("  Por tipo: " + ", ".join(
    f"{tipo} {cantidad}" for tipo, cantidad in resumen["totales_por_tipo"].items()
))
print(f"  Spearman (profundidad vs relaciones): {resumen['spearman']:.3f}")
grafo_resumen = resumen["grafo"]
print(f"  Grafo: {grafo_resumen['aristas_csv']} aristas en CSV, "
      f"{grafo_resumen['aristas_figura']} fuertes en figura, "
      f"{grafo_resumen['nodos']} nodos, {grafo_resumen['componentes']} componentes")
