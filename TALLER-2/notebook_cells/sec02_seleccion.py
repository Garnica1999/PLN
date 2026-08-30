# %% [markdown]
"""# TALLER-2 — Seleccion de los 6 grupos base de la ontologia

Se eligen 6 grupos del tesauro como base de la ontologia WordNet aplicando
criterios objetivos y cuantificables: 5 mediante seleccion greedy y 1 grupo
adicional incorporado de la revision del taller resuelto ajeno
(GRUPOS_ADICIONALES), que cierra el ciclo del conflicto:

1. **Tamano manejable**: entre 50 y 450 terminos por grupo (se excluyen
   Organizaciones 6.405, Estructuras armadas 1.416, Medios de comunicacion
   1.354 y Profesiones y oficios 942); total de los 6 grupos < 1200.
2. **Relevancia para la Comision de la Verdad**: solo grupos nucleo del
   conflicto armado colombiano (lista explicita de dominios tematicos).
3. **Cobertura WordNet-espanol**: porcentaje de hojas con >=1 synset en
   lang='spa' mayor o igual a 1% (la linea punteada de la figura de cobertura).
4. **Riqueza jerarquica**: mediana y maximo de niveles del path.
5. **Diversidad tematica**: un grupo por eje conceptual (agentes, eventos,
   estados/efectos, instrumentos, poblaciones, procesos) mediante seleccion
   greedy con penalizacion de ejes repetidos; tras el greedy se incorpora el
   grupo del eje afrontamientos_resistencias para completar el ciclo
   Actor -> Hecho -> Poblacion -> Impacto -> Afrontamiento.

El resultado se guarda en `data/grupos_seleccionados.json` con la lista de
los 6 grupos, sus estadisticas, justificacion y TODOS los terminos (paths
completos) que seran la base de la ontologia (609 terminos).
"""

# %%
"""Configuracion de la seleccion: imports y constantes.

En el notebook, `sec00_setup` define las rutas y umbrales globales; aqui se
anade el respaldo para ejecutar la seccion como script standalone. Son
locales de esta seccion los mapas de dominio que no viven en sec00:
GRUPOS_NUCLEO, EJES_GRUPO y TOTAL_MAX_TERMINOS.
"""
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
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
    RUTA_EXCEL = RUTA_DATOS / "tesauro_orginal_cev.xlsx"
    RUTA_JSON_GRUPOS = RUTA_DATOS / "grupos_seleccionados.json"
    RUTA_CSV_COBERTURA = RUTA_DATOS / "cobertura_wordnet_grupos.csv"
    RANGO_TAMANO_SELECCION = (50, 450)
    UMBRAL_MIN_COBERTURA = 1.0
    NUM_GRUPOS_SELECCION = 5
    GRUPOS_ADICIONALES: tuple[str, ...] = (
        "Afrontamientos y resistencias no violentas - cotidianas",
    )  # union con el taller resuelto ajeno (cierra el ciclo del conflicto)
    EJES_POR_GRUPO: dict[str, str] = {
        "Actores armados y otros participantes en las dinámicas de la guerra": "agentes",
        "Impactos": "estados_efectos",
        "Armas": "instrumentos",
        "Poblaciones": "poblaciones",
        "Hechos victimizantes": "eventos",
        "Afrontamientos y resistencias no violentas - cotidianas": (
            "afrontamientos_resistencias"
        ),
    }  # eje de los grupos finales (sec00_setup); EJES_GRUPO mapea el adicional
    # a estados_efectos para el greedy, aqui se conserva el eje propio.

# Umbrales de seleccion (criterios 1 y 3): derivados de las constantes de sec00
TAMANO_MIN, TAMANO_MAX = RANGO_TAMANO_SELECCION
TOTAL_MAX_TERMINOS = 1200  # tope global de terminos de los 6 grupos (criterio 1)

# Eje conceptual de cada grupo (criterio 5). Un grupo por eje en la seleccion.
# Mapa especifico de esta seccion: sec00 solo expone EJES_POR_GRUPO para los
# 5 grupos finales, insuficiente para el greedy (los ejes de los descartados
# deben colisionar con los elegidos para garantizar la diversidad).
EJES_GRUPO = {
    "Organizaciones": "agentes",
    "Estructuras armadas": "agentes",
    "Actores armados y otros participantes en las dinámicas de la guerra": "agentes",
    "Alias de los actores armados": "agentes",
    "Partidos y movimientos políticos": "agentes",
    "El Estado y sus responsabilidades en el conflicto armado": "agentes",
    "Direcciones de la Comisión de la Verdad": "agentes",
    "Profesiones y oficios": "agentes",
    "Hechos victimizantes": "eventos",
    "Hitos": "eventos",
    "Despojo de tierras y desplazamiento forzado": "eventos",
    "Exilio como consecuencia del conflicto armado": "eventos",
    "Procesos de paz impulsados por el Estado": "eventos",
    "Dinámicas transfronterizas": "eventos",
    "Dimensiones internacionales del conflicto": "eventos",
    "Impactos": "estados_efectos",
    "Causas, dinámicas e impactos del conflicto armado sobre los pueblos y territorios étnicos": "estados_efectos",
    "Afrontamientos y resistencias no violentas - cotidianas": "estados_efectos",
    "Resistencias organizadas": "estados_efectos",
    "Estereotipos y estigmatización": "estados_efectos",
    "Voces en el conflicto armado": "estados_efectos",
    "Armas": "instrumentos",
    "Medios de comunicación": "instrumentos",
    "Cartografía": "instrumentos",
    "Narcotráfico y conflicto armado": "instrumentos",
    "Poblaciones": "poblaciones",
    "Lenguas nativas": "poblaciones",
    "Género": "poblaciones",
    "Niveles educativos": "poblaciones",
    "Mecanismos de participación ciudadana": "procesos",
    "Programas e iniciativas sociales": "procesos",
    "Aliados culturales": "procesos",
    "Salud y misión médica": "procesos",
    "Mandatos, enfoques y objetivos de la Comisión de la Verdad": "procesos",
    "Recomendaciones para la no repetición": "procesos",
    "Desarrollos institucionales y políticas públicas": "procesos",
    "Democracia y conflicto armado": "procesos",
    "Conflicto armado, sociedad y cultura": "procesos",
    "Economía y modelos de desarrollo y conflicto armado interno": "procesos",
    "Sectores económicos": "procesos",
    "Políticas internacionales": "procesos",
}

# Grupos nucleo del conflicto armado colombiano (criterio 2)
GRUPOS_NUCLEO = {
    "Armas",
    "Hechos victimizantes",
    "Impactos",
    "Actores armados y otros participantes en las dinámicas de la guerra",
    "Poblaciones",
    "Estructuras armadas",
    "Alias de los actores armados",
    "Hitos",
    "Mecanismos de participación ciudadana",
    "Partidos y movimientos políticos",
    "Despojo de tierras y desplazamiento forzado",
    "Exilio como consecuencia del conflicto armado",
    "Narcotráfico y conflicto armado",
    "Dinámicas transfronterizas",
    "Dimensiones internacionales del conflicto",
    "Procesos de paz impulsados por el Estado",
    "Causas, dinámicas e impactos del conflicto armado sobre los pueblos y territorios étnicos",
    "Direcciones de la Comisión de la Verdad",
    "Mandatos, enfoques y objetivos de la Comisión de la Verdad",
    "Recomendaciones para la no repetición",
    "El Estado y sus responsabilidades en el conflicto armado",
    "Desarrollos institucionales y políticas públicas",
    "Democracia y conflicto armado",
    "Género",
    "Lenguas nativas",
    "Afrontamientos y resistencias no violentas - cotidianas",
    "Resistencias organizadas",
    "Estereotipos y estigmatización",
    "Voces en el conflicto armado",
    "Salud y misión médica",
}

# %%
"""Carga del tesauro (repositorio) y seleccion de los 6 grupos base."""


class SelectorGrupos:
    """Selecciona los 6 grupos base de la ontologia con criterios objetivos.

    Responsabilidad unica: aplicar los filtros (tamano, relevancia CEV,
    cobertura WordNet), el greedy por eje conceptual (NUM_GRUPOS_SELECCION)
    y la extension con GRUPOS_ADICIONALES, y documentar la decision en el
    JSON de salida. La carga de datos la cubre `sc.TesauroRepositorio`
    (sec00).

    Attributes:
        tesauro: DataFrame con grupo, termino, niveles y hoja.
    """

    def __init__(self, tesauro: pd.DataFrame) -> None:
        self.tesauro = tesauro

    @staticmethod
    def normalizar_sin_acentos(texto: str) -> str:
        """Normaliza un termino a minusculas sin diacriticos (NFKD + ascii).

        Misma logica que en sec01; se conserva local para que la seccion sea
        autocontenida (NormalizadorAcentos filtraria parentesis y simbolos).
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
        """
        if not termino:
            return False
        if wn.synsets(termino, lang="spa"):
            return True
        normalizado = SelectorGrupos.normalizar_sin_acentos(termino)
        return bool(
            normalizado and normalizado != termino and wn.synsets(normalizado, lang="spa")
        )

    def calcular_cobertura(self) -> pd.DataFrame:
        """Carga la cobertura WordNet del CSV (seccion 01) y completa faltantes.

        Los grupos elegibles por tamano que no estan en el CSV (por tener
        menos de 80 terminos) se calculan aqui con `tiene_synset_espanol`.

        Returns:
            DataFrame con columnas: total_terminos, cubiertos,
            porcentaje_cobertura, niveles_mediana y niveles_max.
        """
        cobertura = (
            pd.read_csv(RUTA_CSV_COBERTURA, index_col=0)
            if RUTA_CSV_COBERTURA.exists()
            else pd.DataFrame()
        )
        elegibles_tamano = self.tesauro.groupby("grupo")["termino"].size()
        elegibles_tamano = elegibles_tamano[
            (elegibles_tamano >= TAMANO_MIN) & (elegibles_tamano <= TAMANO_MAX)
        ]
        faltantes = elegibles_tamano.index.difference(cobertura.index)
        if len(faltantes) > 0:
            filas = self.tesauro[self.tesauro["grupo"].isin(faltantes)].copy()
            filas["cubierto"] = filas["hoja"].apply(self.tiene_synset_espanol)
            extra = filas.groupby("grupo").agg(
                total_terminos=("termino", "size"),
                cubiertos=("cubierto", "sum"),
                niveles_mediana=("niveles", "median"),
                niveles_max=("niveles", "max"),
            )
            extra["porcentaje_cobertura"] = (
                100 * extra["cubiertos"] / extra["total_terminos"]
            ).round(2)
            cobertura = pd.concat([cobertura, extra])
        return cobertura

    def calcular_score(self, candidatos: pd.DataFrame) -> pd.Series:
        """Score compuesto de seleccion (0-1) para ordenar grupos elegibles.

        Componentes: cobertura WordNet (45%), riqueza jerarquica (35%) y
        tamano cercano al optimo de 300 terminos (20%).

        Returns:
            Series con el score por grupo, indexada por nombre de grupo.
        """
        cobertura_max = candidatos["porcentaje_cobertura"].max()
        z_cov = candidatos["porcentaje_cobertura"] / cobertura_max
        z_riqueza = 0.5 * (candidatos["niveles_mediana"] / 6) + 0.5 * (
            candidatos["niveles_max"] / 10
        )
        n = candidatos["total_terminos"]
        z_tam = np.where(n <= 300, n / 300, (1 - (n - 300) / 150).clip(0, 1))
        return (0.45 * z_cov + 0.35 * z_riqueza + 0.20 * z_tam).round(4)

    def seleccionar_grupos(self, elegibles: pd.DataFrame) -> list:
        """Selecciona 5 grupos con greedy por eje conceptual (diversidad).

        En cada paso se elige el grupo de mayor score cuyo eje no este
        representado aun; si todos los ejes estan representados y faltan
        grupos, se completa con el mayor score global.

        Args:
            elegibles: DataFrame con score y eje por grupo.

        Returns:
            Lista de nombres de los 5 grupos seleccionados, en orden de score.
        """
        elegibles = elegibles.copy()
        elegibles["score"] = self.calcular_score(elegibles)
        elegibles["eje"] = elegibles.index.map(EJES_GRUPO).fillna("otros")
        elegibles = elegibles.sort_values("score", ascending=False)

        elegidos, ejes_usados = [], set()
        for _ in range(NUM_GRUPOS_SELECCION):
            candidato = elegibles[~elegibles.index.isin(elegidos)]
            candidato = candidato[~candidato["eje"].isin(ejes_usados)]
            if candidato.empty:
                candidato = elegibles[~elegibles.index.isin(elegidos)]
            mejor = candidato.index[0]
            elegidos.append(mejor)
            ejes_usados.add(elegibles.loc[mejor, "eje"])
        return elegidos

    def verificar_diversidad(self, elegidos: list) -> None:
        """Imprime el eje conceptual de cada grupo (control de diversidad)."""
        ejes = {g: EJES_GRUPO.get(g, "otros") for g in elegidos}
        print("\nEjes conceptuales de los grupos seleccionados:")
        for g, e in ejes.items():
            print(f"  {e:16s} <- {g}")
        print(f"Ejes distintos: {len(set(ejes.values()))} de {len(elegidos)}")

    def justificar_grupo(self, grupo: str, fila: pd.Series, eje: str, indice: int) -> str:
        """Redacta la justificacion de un grupo con sus estadisticas reales."""
        return (
            f"{indice}. {grupo}: {int(fila['total_terminos'])} terminos, "
            f"profundidad mediana {fila['niveles_mediana']:.0f} niveles "
            f"(maximo {fila['niveles_max']:.0f}), cobertura WordNet-espanol "
            f"{fila['porcentaje_cobertura']:.2f}%. Eje conceptual: {eje}. "
            "Cumple los 3 filtros (tamano 50-450, nucleo tematico de la CEV, "
            "cobertura >= 1%) y aporta diversidad al conjunto."
        )

    def calcular_cobertura_grupo(self, grupo: str) -> float:
        """Cobertura WordNet de un grupo arbitrario (ultimo segmento del path).

        Misma logica que `calcular_cobertura`: la columna hoja (ultimo
        segmento) se prueba tal cual y en variante sin acentos; WordNet
        maneja las frases multiword (OMW convierte espacios en guiones
        bajos). Se usa para los grupos adicionales que no superan el filtro
        de tamano y por eso no estan en el CSV de cobertura.
        """
        filas = self.tesauro[self.tesauro["grupo"] == grupo]
        if filas.empty:
            return 0.0
        cubiertos = filas["hoja"].apply(self.tiene_synset_espanol).sum()
        return round(100 * cubiertos / len(filas), 2)

    def estadisticas_grupo(self, grupo: str) -> pd.Series:
        """Estadisticas de seleccion de un grupo tomadas del tesauro.

        Args:
            grupo: Nombre del grupo.

        Returns:
            Series con total_terminos, niveles_mediana, niveles_max y
            porcentaje_cobertura (calculada con WordNet para este grupo).
        """
        filas = self.tesauro[self.tesauro["grupo"] == grupo]
        return pd.Series({
            "total_terminos": len(filas),
            "niveles_mediana": float(filas["niveles"].median()),
            "niveles_max": int(filas["niveles"].max()),
            "porcentaje_cobertura": self.calcular_cobertura_grupo(grupo),
        })

    def justificar_grupo_adicional(
        self, grupo: str, fila: pd.Series, eje: str, indice: int
    ) -> str:
        """Redacta la justificacion de un grupo incorporado tras el greedy."""
        return (
            f"{indice}. {grupo}: {int(fila['total_terminos'])} terminos, "
            f"profundidad mediana {fila['niveles_mediana']:.0f} niveles "
            f"(maximo {fila['niveles_max']:.0f}), cobertura WordNet-espanol "
            f"{fila['porcentaje_cobertura']:.2f}%. Eje conceptual: {eje}. "
            "Incorporado de la revision del taller resuelto ajeno: completa "
            "el ciclo del conflicto Actor -> Hecho -> Poblacion -> Impacto -> "
            "Afrontamiento, documentando como las comunidades responden a la "
            "guerra con resistencia no violenta. Cobertura baja esperada "
            "(vocabulario especializado)."
        )

    def generar_json(self, elegibles: pd.DataFrame, elegidos: list) -> dict:
        """Construye el diccionario completo para grupos_seleccionados.json.

        Incluye criterios, tabla de elegibles, estadisticas y justificacion
        de los grupos elegidos (greedy + GRUPOS_ADICIONALES) y la lista de
        TODOS sus terminos (paths completos).

        Returns:
            Dict anidado listo para json.dump.
        """
        grupos_finales = list(elegidos) + list(GRUPOS_ADICIONALES)
        terminos_por_grupo = {
            grupo: self.tesauro.loc[self.tesauro["grupo"] == grupo, "termino"].tolist()
            for grupo in grupos_finales
        }
        grupos_detalle = []
        for i, grupo in enumerate(grupos_finales, 1):
            if grupo in elegidos:
                fila = elegibles.loc[grupo]
                eje = EJES_GRUPO.get(grupo, "otros")
                justificacion = self.justificar_grupo(grupo, fila, eje, i)
            else:
                fila = self.estadisticas_grupo(grupo)
                eje = EJES_POR_GRUPO.get(grupo, "otros")
                justificacion = self.justificar_grupo_adicional(grupo, fila, eje, i)
            grupos_detalle.append({
                "grupo": grupo,
                "eje_conceptual": eje,
                "total_terminos": int(fila["total_terminos"]),
                "niveles_mediana": float(fila["niveles_mediana"]),
                "niveles_max": int(fila["niveles_max"]),
                "cobertura_wordnet_pct": float(fila["porcentaje_cobertura"]),
                "justificacion": justificacion,
            })
        return {
            "fecha": "2026-08-30",
            "fuente": "data/tesauro_orginal_cev.xlsx (hoja Tesauro)",
            "criterios": {
                "tamano_min": TAMANO_MIN,
                "tamano_max": TAMANO_MAX,
                "cobertura_wordnet_min_pct": UMBRAL_MIN_COBERTURA,
                "grupos_nucleo_cev": len(GRUPOS_NUCLEO),
                "total_max_terminos": TOTAL_MAX_TERMINOS,
                "metodo_seleccion": (
                    "filtros (tamano, relevancia CEV, cobertura) "
                    "+ greedy por eje conceptual"
                ),
            },
            "elegibles_tabla": elegibles.to_dict(orient="index"),
            "grupos_seleccionados": grupos_detalle,
            "total_terminos": sum(int(g["total_terminos"]) for g in grupos_detalle),
            "terminos_por_grupo": terminos_por_grupo,
        }

    def verificar_salida(self, payload: dict) -> None:
        """Valida el JSON generado: total <1200, sin duplicados ni vacios."""
        total = payload["total_terminos"]
        todos = [t for ts in payload["terminos_por_grupo"].values() for t in ts]
        assert total < TOTAL_MAX_TERMINOS, (
            f"Total {total} excede el limite {TOTAL_MAX_TERMINOS}"
        )
        assert len(todos) == len(set(todos)), "Hay terminos duplicados entre los grupos"
        assert all(t.strip() for t in todos), "Hay terminos vacios"
        total_grupos = NUM_GRUPOS_SELECCION + len(GRUPOS_ADICIONALES)
        assert len(payload["grupos_seleccionados"]) == total_grupos, (
            f"Deben ser exactamente {total_grupos} grupos"
        )
        print(f"\nVerificacion OK: {total} terminos unicos y no vacios "
              f"({TOTAL_MAX_TERMINOS} max), {total_grupos} grupos, "
              "sin duplicados.")


repositorio = sc.TesauroRepositorio(RUTA_EXCEL, RUTA_JSON_GRUPOS)
tesauro = repositorio.cargar_tesauro()
tesauro = sc.TesauroRepositorio.con_estructura(tesauro)
selector = SelectorGrupos(tesauro)
cobertura = selector.calcular_cobertura()

# Criterio 1: tamano manejable
cobertura["total_terminos"] = cobertura["total_terminos"].astype(int)
elegibles = cobertura[
    (cobertura["total_terminos"] >= TAMANO_MIN)
    & (cobertura["total_terminos"] <= TAMANO_MAX)
].copy()

# Criterio 2: relevancia para la CEV
elegibles = elegibles[elegibles.index.isin(GRUPOS_NUCLEO)]

# Criterio 3: cobertura WordNet-espanol minima
elegibles = elegibles[elegibles["porcentaje_cobertura"] >= UMBRAL_MIN_COBERTURA]

print("=== Grupos elegibles tras filtros (tamano 50-450, nucleo CEV, cobertura >=1%) ===")
print(elegibles.to_string())

elegidos = selector.seleccionar_grupos(elegibles)
print("\n=== Grupos seleccionados (greedy por eje conceptual) ===")
for i, grupo in enumerate(elegidos, 1):
    fila = elegibles.loc[grupo]
    print(f"{i}. {grupo} | n={fila['total_terminos']} | "
          f"cobertura={fila['porcentaje_cobertura']:.2f}% "
          f"| mediana_niveles={fila['niveles_mediana']:.0f} "
          f"| max_niveles={fila['niveles_max']:.0f}")

# %%
"""Extension aprobada: incorporacion de los grupos adicionales al JSON.

Tras el greedy se anaden los GRUPOS_ADICIONALES (constante de sec00_setup):
su cobertura WordNet se calcula aqui mismo (hoja del path, con/sin acentos,
multiword) porque su tamano no supera el filtro minimo y no figura en el CSV
de cobertura de la seccion 01.
"""
print("\n=== Grupos adicionales (cierre del ciclo del conflicto) ===")
for grupo in GRUPOS_ADICIONALES:
    fila = selector.estadisticas_grupo(grupo)
    print(f"{grupo} | n={fila['total_terminos']} | "
          f"cobertura={fila['porcentaje_cobertura']:.2f}% "
          f"| mediana_niveles={fila['niveles_mediana']:.0f} "
          f"| max_niveles={fila['niveles_max']:.0f}")

# %%
"""Verificacion de diversidad y generacion del JSON de salida."""

selector.verificar_diversidad(elegidos)

payload = selector.generar_json(elegibles, elegidos)
selector.verificar_salida(payload)

with open(RUTA_JSON_GRUPOS, "w", encoding="utf-8") as archivo:
    json.dump(payload, archivo, ensure_ascii=False, indent=2)
print(f"JSON guardado en: data/{RUTA_JSON_GRUPOS.name}")

# %% [markdown]
"""## Justificacion de los 6 grupos seleccionados

La seleccion aplica tres filtros secuenciales, un criterio de diversidad y
una extension del ciclo del conflicto:

1. **Tamano manejable** (50-450 terminos): se descartan los cuatro grupos
   gigantes (Organizaciones 6.405, Estructuras armadas 1.416, Medios de
   comunicacion 1.354, Profesiones y oficios 942) que harian inmanejable la
   construccion manual de la ontologia y el mapeo a WordNet.
2. **Relevancia para la Comision de la Verdad**: solo grupos nucleo del
   conflicto armado colombiano (30 de los 41 grupos segun el mapeo explicito).
3. **Cobertura WordNet-espanol >= 1%**: garantiza que una fraccion del
   vocabulario tenga correspondencia lexical en WordNet espanol (omw-1.4).
4. **Diversidad**: seleccion greedy con un maximo de un grupo por eje
   conceptual (agentes, eventos, estados/efectos, instrumentos, poblaciones).
5. **Cierre del ciclo**: tras el greedy se incorpora el grupo del eje
   afrontamientos_resistencias (GRUPOS_ADICIONALES), tomado de la revision
   del taller resuelto ajeno, para completar el ciclo
   Actor -> Hecho -> Poblacion -> Impacto -> Afrontamiento.

| # | Grupo | Terminos | Cobertura WN | Eje conceptual | Rol en el ciclo |
|---|-------|----------|--------------|----------------|-----------------|
| 1 | Actores armados y otros participantes en las dinamicas de la guerra | 82 | 14.63% | agentes | perpetran los hechos |
| 2 | Hechos victimizantes | 90 | 2.22% | eventos | nucleo del ciclo: hechos del conflicto |
| 3 | Armas | 94 | 7.45% | instrumentos | instrumentos empleados por los actores |
| 4 | Poblaciones | 213 | 1.41% | poblaciones | sujetos afectados por los hechos |
| 5 | Impactos | 107 | 11.21% | estados_efectos | consecuencias de los hechos |
| 6 | Afrontamientos y resistencias no violentas - cotidianas | 23 | 4.35% | afrontamientos_resistencias | respuestas no violentas que cierran el ciclo |

Los 5 primeros salen del greedy (uno por eje conceptual distinto); el sexto
se incorpora con justificacion propia aunque su tamano (23) no supere el
filtro minimo de 50 terminos, porque su rol es insustituible: documenta como
las comunidades responden a la guerra con resistencia no violenta. La
decision queda registrada con todos sus numeros en
`data/grupos_seleccionados.json`, junto con los paths completos de los 609
terminos que seran la base de la ontologia.
"""
# %% [markdown]
"""## Relevancia para el sector: por qué estos 6 grupos y esta ontología

La Comisión para el Esclarecimiento de la Verdad (CEV) tuvo como mandato
esclarecer los patrones y causas del conflicto armado colombiano a partir de
una cantidad masiva de información: testimonios, sentencias, informes y bases
de datos. Su propio Tesauro (2022, Equipo de Lenguaje Controlado del SIM) fue
la herramienta que normalizó esa terminología para catalogar y recuperar
conocimiento. Esta ontología continúa ese propósito con un medio formal:

1. **Los 6 grupos elegidos cubren el ciclo completo del hecho victimizante**:
   los *actores armados* (agentes) perpetran los *hechos victimizantes*
   (eventos) con *armas* (instrumentos); esos hechos afectan a las
   *poblaciones* (sujetos) y generan *impactos* (estados/efectos); las
   poblaciones responden con *afrontamientos y resistencias no violentas*
   (cierre del ciclo). Juntos permiten responder preguntas de negocio del
   sector: qué actores perpetraron qué hechos, con qué instrumentos, sobre
   qué poblaciones, con qué impactos y qué resistencias desarrollaron.
2. **Utilidad para la memoria histórica**: una ontología sobre este
   vocabulario habilita consultas formales (SPARQL) sobre la información
   documental de la CEV: recuperar todos los términos del subárbol de un
   hecho, relacionar hechos con actores (perpetradoPor), o asociar impactos
   a poblaciones (relacionadoCon), con semántica explícita y compartida.
3. **La elección está respaldada por datos** (criterios cuantitativos) y por
   el modelo conceptual de la CEV (las 7 relaciones raíz fijas de la
   ontología: hechos-perpetrados-por-actores, armas-como-instrumentos,
   hechos-que-afectan-poblaciones, impactos-generados-por-hechos,
   impactos-sobre-poblaciones, poblaciones-que-desarrollan-afrontamientos y
   afrontamientos-que-responden-a-hechos).

Esta justificación conecta la selección con el propósito institucional de la
CEV: representar el conocimiento del conflicto de forma reutilizable,
consultable e interoperable, más allá de un análisis plano de texto.
"""
