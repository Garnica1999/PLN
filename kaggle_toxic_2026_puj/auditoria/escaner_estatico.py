"""auditoria/escaner_estatico.py
Fecha: 2026-10-02
Descripcion: Escaner estatico basado en AST sobre los scripts de `experiments/`
del proyecto kaggle_toxic_2026_puj. Busca el fallo de clase dominante del
proyecto: codigo escrito y probado por el humo pero desconectado del camino
real de ejecucion (P82 y P83), ademas de otros patrones de riesgo (afirmaciones
vacias, ramas muertas, umbrales congelados, menciones al test o al holdout,
aleatoriedad sin semilla y excepciones silenciadas).

Detectores implementados (nombres exactos en la columna `detector`):
  D1_funciones_probadas_sin_llamador: funcion definida cuyo unico consumidor
      esta en funciones de humo (nombre con humo|casos|verificar|_smoke|test_)
      o que no tiene ningun consumidor. El detalle informa cuantos llamadores
      reales hay y si el unico es de humo.
  D2_asercion_vacia: `assert True`, `assert not False`, comparacion de dos
      literales iguales entre si o de un nombre consigo mismo.
  D3_rama_muerta: `if False`, `if 0`, `while False` u otro test constante falso
      a nivel de modulo, de funcion o en una expresion condicional (ternario).
  D4_break_en_medicion: `break` incondicional (nivel superior del cuerpo, no
      dentro de if/try) en un `for` sobre lista, tupla, dict o range, cuando el
      cuerpo modifica estado (llamadas o asignaciones) antes del break.
  D5_umbral_congelado_hardcodeado: literales 0.44, 0.51 o 0.46 usados en
      comparaciones o asignaciones. Si el mismo archivo los lee de un artefacto
      (contiene `leer_umbrales` o un `np.load` de umbrales), el hallazgo pasa a
      severidad INFO y el detalle lo declara informativo.
  D6_toca_test_o_holdout: menciones de `toxic_test.csv`, `holdout_v2` o la
      palabra `holdout`. Severidad INFO salvo que la misma funcion contenga
      ademas una llamada de ajuste (`fit` o `train`), caso en el que sube a
      MEDIA para visibilizar un posible entrenamiento sobre test.
  D7_semilla_ausente: el archivo usa `np.random`, `random.`, `torch.rand` o
      `random_state` y no contiene ninguna llamada a `seed`.
  D8_except_desnudo: `except:` o `except Exception:` con cuerpo `pass`.

Decisiones de implementacion (orientadas a pocos falsos positivos):
  - Un consumidor cuenta tanto si llama al nombre (`mi_funcion()`) como si lo
    referencia como valor (callback, `partial`, tabla de despacho): ambos usos
    conectan la funcion al camino real. Las referencias recursivas dentro de la
    propia funcion no cuentan como llamador externo.
  - D1 excluye funciones de entrada (`main`, `principal`), dunder, decoradas con
    algo que contenga `fixture`, auxiliares de humo declarados (nombre con
    `humo`, `caso`, `smoke` o `test_`), porque esos auxiliares existen
    precisamente para ser invocados desde el modo de humo, y ganchos de
    framework definidos como metodo (`forward` de torch.nn.Module y
    `setUp`/`tearDown` de unittest), que el framework invoca sin llamada
    explicita en el archivo.
  - La clasificacion de humo es insensible a mayusculas y examina toda la cadena
    de ambitos que contiene la referencia (clase o funciones anidadas).
  - Los hallazgos se deduplican por (archivo, linea, detector) conservando la
    severidad mas alta; en D6 la escalada por fit/train usa el nombre final de
    la llamada, de modo que `train_test_split` no escala el hallazgo.
  - La fecha del resumen se fija en CONFIG para que dos ejecuciones produzcan
    exactamente el mismo markdown y el mismo md5.

Salidas nuevas, ambas dentro de `auditoria/`:
  - escaner_hallazgos.csv: columnas archivo,linea,detector,severidad,detalle.
  - escaner_resumen.md: conteos por detector, top 10 de archivos por severidad
    ALTA o MEDIA y lista de funciones D1 sin llamadores reales.

Uso:
  /home/carlos/venv/bin/python auditoria/escaner_estatico.py
  /home/carlos/venv/bin/python auditoria/escaner_estatico.py --directorio RUTA
      --csv RUTA.csv --resumen RUTA.md

El codigo de este script es ASCII puro; el markdown de salida lleva acentos y
enie construidos con escapes \\uXXXX. No modifica ningun fichero existente.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import re
import sys
import tokenize
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# ===== SECCION 1: CONFIGURACION =====

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_AUDITORIA = Path(__file__).resolve().parent

CONFIG = {
    "fecha_escaneo": "2026-10-02",
    "directorio_experimentos": RAIZ_PROYECTO / "experiments",
    "ruta_csv": DIR_AUDITORIA / "escaner_hallazgos.csv",
    "ruta_resumen": DIR_AUDITORIA / "escaner_resumen.md",
    "umbrales_congelados": (0.44, 0.51, 0.46),
    "max_invocadores_en_detalle": 4,
    "max_usos_en_detalle": 5,
}

D1 = "D1_funciones_probadas_sin_llamador"
D2 = "D2_asercion_vacia"
D3 = "D3_rama_muerta"
D4 = "D4_break_en_medicion"
D5 = "D5_umbral_congelado_hardcodeado"
D6 = "D6_toca_test_o_holdout"
D7 = "D7_semilla_ausente"
D8 = "D8_except_desnudo"

DETECTORES = (D1, D2, D3, D4, D5, D6, D7, D8)
SEVERIDADES = ("ALTA", "MEDIA", "BAJA", "INFO")
ORDEN_SEVERIDAD = {severidad: indice for indice, severidad in enumerate(SEVERIDADES)}
UMBRALES_D5 = tuple(CONFIG["umbrales_congelados"])

PATRON_HUMO = re.compile(r"humo|caso|verificar|_smoke|test_", re.IGNORECASE)
PATRON_AUXILIAR_HUMO = re.compile(r"humo|caso|smoke|test_", re.IGNORECASE)
NOMBRES_ENTRADA = frozenset({"main", "principal"})
GANCHOS_DE_METODO = frozenset(
    {"forward", "setUp", "tearDown", "setUpClass", "tearDownClass"}
)
# El patron visitante de ast.NodeVisitor despacha visit_* y generic_visit por
# nombre en tiempo de ejecucion: su llamador es el framework y no una llamada
# sintactica. Sin esta exclusion el escaner se denuncia a si mismo y marca como
# desconectados los visitantes que el propio codigo usa.
PATRON_VISITANTE = re.compile(r"^(visit_\w+|generic_visit)$")
PATRON_MENCION_TEST = re.compile(r"toxic_test\.csv|holdout_v2|holdout", re.IGNORECASE)
PATRON_ARTEFACTO_UMBRAL = re.compile(r"leer_umbrales|np\.load\s*\([^\n)]*umbral", re.IGNORECASE)
NOMBRES_FIT_O_TRAIN = frozenset(
    {
        "fit",
        "train",
        "entrenar",
        "ajustar",
        "fit_transform",
        "partial_fit",
        "entrenar_modelo",
        "train_model",
    }
)
TIPOS_TOKEN_MENCION = (
    tokenize.NAME,
    tokenize.STRING,
    tokenize.COMMENT,
    getattr(tokenize, "FSTRING_MIDDLE", -1),
)


# ===== SECCION 2: MODELO DE DATOS Y UTILIDADES =====

@dataclass(frozen=True)
class Hallazgo:
    """Un hallazgo del escaner, listo para el CSV y el resumen.

    Args:
        archivo: ruta del script relativa a la raiz del proyecto.
        linea: linea 1-based donde se detecto el patron.
        detector: nombre exacto del detector que lo emitio.
        severidad: ALTA, MEDIA, BAJA o INFO.
        detalle: descripcion textual del hallazgo (ASCII).
        nombre: nombre de la funcion implicada cuando aplica (solo D1).
    """

    archivo: str
    linea: int
    detector: str
    severidad: str
    detalle: str
    nombre: str = ""


@dataclass(frozen=True)
class Funcion:
    """Una funcion o metodo definido en un modulo.

    Args:
        nombre: nombre simple de la funcion.
        linea: linea de la definicion.
        fin: ultima linea del cuerpo.
        nodo: nodo AST de la definicion.
        decoradores: textos fuente de los decoradores.
        en_clase: True si la definicion cuelga de una clase (es metodo).
    """

    nombre: str
    linea: int
    fin: int
    nodo: ast.AST
    decoradores: tuple[str, ...]
    en_clase: bool = False


@dataclass(frozen=True)
class Referencia:
    """Uso de un nombre en el modulo (llamada o referencia como valor).

    Args:
        nombre: nombre referenciado.
        linea: linea del uso.
        invocadores: cadena de ambitos que contiene el uso.
    """

    nombre: str
    linea: int
    invocadores: tuple[str, ...]


def a_ascii(texto: str) -> str:
    """Devuelve `texto` con todo caracter no ASCII sustituido por '?'.

    Args:
        texto: cadena de entrada.

    Returns:
        Cadena ASCII segura para el CSV.
    """
    return texto.encode("ascii", "replace").decode("ascii")


def texto_linea(codigo: str, numero: int) -> str:
    """Devuelve la linea `numero` (1-based) de `codigo`, sin espacios extremos.

    Args:
        codigo: fuente completa del modulo.
        numero: numero de linea 1-based.

    Returns:
        Texto de la linea en ASCII, o cadena vacia si la linea no existe.
    """
    lineas = codigo.splitlines()
    if 1 <= numero <= len(lineas):
        return a_ascii(lineas[numero - 1].strip())
    return ""


def texto_nodo(codigo: str, nodo: ast.AST) -> str:
    """Devuelve el texto fuente de `nodo`, con respaldo en `ast.unparse`.

    Args:
        codigo: fuente completa del modulo.
        nodo: nodo AST del que se quiere el texto.

    Returns:
        Texto del nodo en una sola linea, en ASCII.
    """
    segmento = None
    try:
        segmento = ast.get_source_segment(codigo, nodo)
    except (ValueError, TypeError):
        segmento = None
    if segmento:
        return a_ascii(" ".join(segmento.split()))
    return a_ascii(" ".join(ast.unparse(nodo).split()))


def es_humo(nombre: str) -> bool:
    """Indica si un nombre de ambito pertenece al modo de humo o a pruebas.

    Args:
        nombre: nombre simple de funcion, metodo o clase.

    Returns:
        True si el nombre coincide con el patron de humo del proyecto.
    """
    return bool(PATRON_HUMO.search(nombre))


def cadena_es_humo(cadena: str) -> bool:
    """Indica si algun tramo de una cadena de invocadores es de humo.

    Args:
        cadena: ambitos unidos por punto, por ejemplo "Clase.metodo".

    Returns:
        True si al menos un ambito coincide con el patron de humo.
    """
    return any(es_humo(tramo) for tramo in cadena.split(".") if tramo)


def listar_scripts(directorio: Path) -> list[Path]:
    """Lista los scripts .py de `directorio` ignorando `__pycache__`.

    Args:
        directorio: raiz del barrido.

    Returns:
        Rutas ordenadas de forma estable por su representacion POSIX.
    """
    return sorted(
        (ruta for ruta in directorio.rglob("*.py") if "__pycache__" not in ruta.parts),
        key=lambda ruta: ruta.as_posix(),
    )


def etiqueta_archivo(ruta: Path, base: Path) -> str:
    """Devuelve la ruta que se muestra en los informes.

    Args:
        ruta: ruta del archivo escaneado.
        base: raiz contra la que se intenta relativizar.

    Returns:
        Ruta relativa a `base` si es posible; en caso contrario, absoluta.
    """
    try:
        return ruta.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return ruta.resolve().as_posix()


# ===== SECCION 3: RECOLECCION DE DEFINICIONES Y REFERENCIAS =====

def recolectar_definiciones(arbol: ast.AST) -> list[Funcion]:
    """Recolecta todas las funciones y metodos definidos en el modulo.

    Args:
        arbol: AST del modulo.

    Returns:
        Lista de Funcion ordenada por linea y nombre.
    """

    class _Recolector(ast.NodeVisitor):
        """Visitante que distingue metodos (definidos dentro de una clase)."""

        def __init__(self) -> None:
            self.funciones: list[Funcion] = []
            self._clases: int = 0

        def visit_ClassDef(self, nodo: ast.ClassDef) -> None:
            """Aumenta la profundidad de clase y recorre su cuerpo."""
            self._clases += 1
            self.generic_visit(nodo)
            self._clases -= 1

        def _agregar(self, nodo: ast.AST) -> None:
            """Registra una definicion de funcion y recorre su cuerpo."""
            decoradores = tuple(ast.unparse(decorador) for decorador in nodo.decorator_list)
            fin = getattr(nodo, "end_lineno", None) or nodo.lineno
            self.funciones.append(
                Funcion(
                    nodo.name,
                    nodo.lineno,
                    fin,
                    nodo,
                    decoradores,
                    en_clase=self._clases > 0,
                )
            )
            self.generic_visit(nodo)

        def visit_FunctionDef(self, nodo: ast.FunctionDef) -> None:
            """Registra una funcion normal."""
            self._agregar(nodo)

        def visit_AsyncFunctionDef(self, nodo: ast.AsyncFunctionDef) -> None:
            """Registra una funcion asincrona."""
            self._agregar(nodo)

    recolector = _Recolector()
    recolector.visit(arbol)
    return sorted(
        recolector.funciones, key=lambda funcion: (funcion.linea, funcion.nombre)
    )


def nombre_llamado(nodo: ast.AST) -> str | None:
    """Devuelve el nombre final de una expresion llamada.

    Args:
        nodo: nodo correspondiente al campo `func` de un ast.Call.

    Returns:
        Nombre para `f()` o `objeto.metodo()`; None en otros casos.
    """
    if isinstance(nodo, ast.Name):
        return nodo.id
    if isinstance(nodo, ast.Attribute):
        return nodo.attr
    return None


class ColectorReferencias(ast.NodeVisitor):
    """Recolecta usos de nombres con la cadena de ambitos que los contiene."""

    def __init__(self) -> None:
        self.referencias: list[Referencia] = []
        self._pila: list[str] = []

    def _cadena(self) -> tuple[str, ...]:
        """Devuelve la pila de ambitos actual como tupla."""
        return tuple(self._pila)

    def visit_FunctionDef(self, nodo: ast.FunctionDef) -> None:
        """Visita una funcion, separando decoradores y cuerpo."""
        self._visitar_funcion(nodo)

    def visit_AsyncFunctionDef(self, nodo: ast.AsyncFunctionDef) -> None:
        """Visita una funcion asincrona, separando decoradores y cuerpo."""
        self._visitar_funcion(nodo)

    def _visitar_funcion(self, nodo: ast.AST) -> None:
        """Visita decoradores y valores por defecto fuera del ambito propio.

        Args:
            nodo: definicion de funcion (normal o asincrona).
        """
        for decorador in nodo.decorator_list:
            self.visit(decorador)
        valores = list(nodo.args.defaults) + [
            valor for valor in nodo.args.kw_defaults if valor is not None
        ]
        for valor in valores:
            self.visit(valor)
        self._pila.append(nodo.name)
        for sentencia in nodo.body:
            self.visit(sentencia)
        self._pila.pop()

    def visit_ClassDef(self, nodo: ast.ClassDef) -> None:
        """Visita una clase empujando su nombre al ambito de los metodos."""
        for decorador in nodo.decorator_list:
            self.visit(decorador)
        for base in list(nodo.bases) + [clave.value for clave in nodo.keywords]:
            self.visit(base)
        self._pila.append(nodo.name)
        for sentencia in nodo.body:
            self.visit(sentencia)
        self._pila.pop()

    def visit_Call(self, nodo: ast.Call) -> None:
        """Registra el nombre llamado y continua el recorrido de argumentos."""
        nombre = nombre_llamado(nodo.func)
        if nombre:
            self.referencias.append(Referencia(nombre, nodo.lineno, self._cadena()))
        self.generic_visit(nodo)

    def visit_Name(self, nodo: ast.Name) -> None:
        """Registra referencias de nombres en contexto de lectura."""
        if isinstance(nodo.ctx, ast.Load):
            self.referencias.append(Referencia(nodo.id, nodo.lineno, self._cadena()))
        self.generic_visit(nodo)


# ===== SECCION 4: DETECTORES =====

def excluir_de_d1(funcion: Funcion) -> bool:
    """Indica si una funcion no debe auditarse con D1.

    Args:
        funcion: definicion a evaluar.

    Returns:
        True para entrada (main/principal), dunder, decoradas con algo que
        contenga `fixture`, ganchos de metodo de framework y auxiliares de
        humo declarados.
    """
    nombre = funcion.nombre
    if nombre in NOMBRES_ENTRADA:
        return True
    if nombre.startswith("__") and nombre.endswith("__"):
        return True
    if any("fixture" in decorador.lower() for decorador in funcion.decoradores):
        return True
    if funcion.en_clase and nombre in GANCHOS_DE_METODO:
        return True
    if funcion.en_clase and PATRON_VISITANTE.match(nombre):
        return True
    return bool(PATRON_AUXILIAR_HUMO.search(nombre))


def _referencia_es_recursiva(referencia: Referencia, nombre: str) -> bool:
    """Indica si una referencia se produce dentro de la propia funcion."""
    return bool(referencia.invocadores) and referencia.invocadores[-1] == nombre


def detectar_d1(
    funciones: list[Funcion],
    referencias: list[Referencia],
    archivo: str,
) -> list[Hallazgo]:
    """Detecta funciones cuyo unico consumo esta en humo o no existe.

    Args:
        funciones: definiciones del modulo.
        referencias: usos de nombres del modulo.
        archivo: etiqueta del archivo para el hallazgo.

    Returns:
        Hallazgos D1 (ALTA sin consumidores, MEDIA con consumidores solo de
        humo).
    """
    por_nombre: dict[str, list[Referencia]] = defaultdict(list)
    for referencia in referencias:
        por_nombre[referencia.nombre].append(referencia)
    hallazgos = []
    for funcion in funciones:
        if excluir_de_d1(funcion):
            continue
        consumidores: set[str] = set()
        for referencia in por_nombre.get(funcion.nombre, []):
            if _referencia_es_recursiva(referencia, funcion.nombre):
                continue
            consumidores.add(".".join(referencia.invocadores) if referencia.invocadores else "modulo")
        reales = sorted(cadena for cadena in consumidores if not cadena_es_humo(cadena))
        humo = sorted(cadena for cadena in consumidores if cadena_es_humo(cadena))
        if reales:
            continue
        if not consumidores:
            severidad = "ALTA"
            detalle = (
                f"funcion={funcion.nombre}; llamadores_reales=0; "
                "llamadores_de_humo=0; unico_de_humo=no; invocadores=[]"
            )
        else:
            severidad = "MEDIA"
            muestra = humo[: CONFIG["max_invocadores_en_detalle"]]
            detalle = (
                f"funcion={funcion.nombre}; llamadores_reales=0; "
                f"llamadores_de_humo={len(humo)}; "
                f"unico_de_humo={'si' if len(humo) == 1 else 'no'}; "
                f"invocadores={muestra}"
            )
        hallazgos.append(
            Hallazgo(archivo, funcion.linea, D1, severidad, a_ascii(detalle), funcion.nombre)
        )
    return hallazgos


def _literal(nodo: ast.AST) -> tuple[bool, object]:
    """Extrae el valor de un literal simple, incluidos signos unarios.

    Args:
        nodo: expresion a evaluar.

    Returns:
        Tupla (es_literal, valor). El valor es None cuando no es literal.
    """
    if isinstance(nodo, ast.Constant):
        return True, nodo.value
    if (
        isinstance(nodo, ast.UnaryOp)
        and isinstance(nodo.op, (ast.USub, ast.UAdd))
        and isinstance(nodo.operand, ast.Constant)
        and isinstance(nodo.operand.value, (int, float))
        and not isinstance(nodo.operand.value, bool)
    ):
        valor = -nodo.operand.value if isinstance(nodo.op, ast.USub) else nodo.operand.value
        return True, valor
    return False, None


def _asercion_vacia(test: ast.AST) -> bool:
    """Indica si la condicion de un assert es tautologica.

    Args:
        test: expresion del assert.

    Returns:
        True para `True`, `not False`, literales iguales o nombre contra si
        mismo.
    """
    if isinstance(test, ast.Constant) and test.value is True:
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        operando = test.operand
        if isinstance(operando, ast.Constant) and operando.value is False:
            return True
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        izquierda, derecha = test.left, test.comparators[0]
        if isinstance(izquierda, ast.Name) and isinstance(derecha, ast.Name):
            return izquierda.id == derecha.id
        izquierda_es, izquierda_valor = _literal(izquierda)
        derecha_es, derecha_valor = _literal(derecha)
        if izquierda_es and derecha_es:
            try:
                return bool(izquierda_valor == derecha_valor)
            except (TypeError, ValueError):
                return False
    return False


def detectar_d2(arbol: ast.AST, codigo: str, archivo: str) -> list[Hallazgo]:
    """Detecta aserciones tautologicas (`assert True`, `1 == 1`, `x == x`)."""
    hallazgos = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assert) and _asercion_vacia(nodo.test):
            detalle = texto_nodo(codigo, nodo)
            hallazgos.append(Hallazgo(archivo, nodo.lineno, D2, "ALTA", detalle))
    return hallazgos


def _test_muerto(test: ast.AST) -> bool:
    """Indica si un test es una constante falsa (`False`, `0`, `""`, None)."""
    if not isinstance(test, ast.Constant):
        return False
    valor = test.value
    if valor is None or valor is False:
        return True
    if isinstance(valor, (int, float, complex)) and not isinstance(valor, bool) and valor == 0:
        return True
    return isinstance(valor, (str, bytes)) and not valor


def detectar_d3(arbol: ast.AST, codigo: str, archivo: str) -> list[Hallazgo]:
    """Detecta ramas muertas por test constante falso.

    Cubre `if`, `while` y expresiones condicionales (`x if False else y`).
    """
    hallazgos = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.If, ast.While)) and _test_muerto(nodo.test):
            palabra = "if" if isinstance(nodo, ast.If) else "while"
            detalle = f"{palabra} {ast.unparse(nodo.test)}:"
            hallazgos.append(Hallazgo(archivo, nodo.lineno, D3, "MEDIA", a_ascii(detalle)))
        elif isinstance(nodo, ast.IfExp) and _test_muerto(nodo.test):
            detalle = f"ternario con if {ast.unparse(nodo.test)}: {texto_linea(codigo, nodo.lineno)}"
            hallazgos.append(Hallazgo(archivo, nodo.lineno, D3, "MEDIA", a_ascii(detalle)))
    return hallazgos


def _es_iterable_de_medicion(nodo: ast.AST) -> bool:
    """Indica si el iterable de un for es un literal pequeno o un range."""
    if isinstance(nodo, (ast.List, ast.Tuple, ast.Dict)):
        return True
    return (
        isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == "range"
    )


def _recorrer_sin_definiciones(nodo: ast.AST):
    """Recorre un subarbol sin entrar en funciones, lambdas ni clases.

    Args:
        nodo: raiz del subarbol.

    Yields:
        Nodos del subarbol en orden de pila.
    """
    pila = [nodo]
    while pila:
        actual = pila.pop()
        yield actual
        for hijo in ast.iter_child_nodes(actual):
            if isinstance(hijo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            pila.append(hijo)


def _modifica_estado(sentencias: list[ast.stmt]) -> bool:
    """Indica si alguna sentencia contiene llamadas o asignaciones."""
    for sentencia in sentencias:
        for nodo in _recorrer_sin_definiciones(sentencia):
            if isinstance(nodo, ast.Call):
                return True
            if isinstance(nodo, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                return True
    return False


def detectar_d4(arbol: ast.AST, codigo: str, archivo: str) -> list[Hallazgo]:
    """Detecta breaks incondicionales que cortan una medicion sobre literal."""
    hallazgos = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.For) or not _es_iterable_de_medicion(nodo.iter):
            continue
        for indice, sentencia in enumerate(nodo.body):
            if not isinstance(sentencia, ast.Break):
                continue
            if _modifica_estado(nodo.body[:indice]):
                detalle = f"break incondicional en for de medicion: {texto_linea(codigo, nodo.lineno)}"
                hallazgos.append(Hallazgo(archivo, nodo.lineno, D4, "MEDIA", detalle))
            break
    return hallazgos


def _mapa_de_padres(arbol: ast.AST) -> dict[ast.AST, ast.AST]:
    """Construye el mapa hijo -> padre del arbol completo."""
    padres: dict[ast.AST, ast.AST] = {}
    pila = [arbol]
    while pila:
        nodo = pila.pop()
        for hijo in ast.iter_child_nodes(nodo):
            padres[hijo] = nodo
            pila.append(hijo)
    return padres


def _contexto_umbral(nodo: ast.AST, padres: dict[ast.AST, ast.AST]) -> str | None:
    """Clasifica el contexto de un literal: comparacion, asignacion o None."""
    actual = padres.get(nodo)
    while actual is not None:
        if isinstance(actual, ast.Compare):
            return "comparacion"
        if isinstance(actual, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            return "asignacion"
        actual = padres.get(actual)
    return None


def detectar_d5(arbol: ast.AST, codigo: str, archivo: str, fuente: str) -> list[Hallazgo]:
    """Detecta los umbrales congelados 0.44, 0.51 y 0.46 hardcodeados.

    Args:
        arbol: AST del modulo.
        codigo: fuente del modulo.
        archivo: etiqueta del archivo.
        fuente: fuente usada para decidir si el archivo lee de un artefacto.

    Returns:
        Hallazgos BAJA (o INFO si el archivo lee umbrales de un artefacto).
    """
    lee_artefacto = bool(PATRON_ARTEFACTO_UMBRAL.search(fuente))
    padres = _mapa_de_padres(arbol)
    hallazgos = []
    lineas_vistas: set[int] = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Constant) or isinstance(nodo.value, bool):
            continue
        if not isinstance(nodo.value, (int, float)):
            continue
        if not any(abs(float(nodo.value) - umbral) < 1e-12 for umbral in UMBRALES_D5):
            continue
        contexto = _contexto_umbral(nodo, padres)
        if contexto is None or nodo.lineno in lineas_vistas:
            continue
        lineas_vistas.add(nodo.lineno)
        severidad = "INFO" if lee_artefacto else "BAJA"
        nota = (
            "; informativo: el archivo lee umbrales de un artefacto"
            if lee_artefacto
            else ""
        )
        detalle = f"umbral {nodo.value!r} en {contexto}: {texto_linea(codigo, nodo.lineno)}{nota}"
        hallazgos.append(Hallazgo(archivo, nodo.lineno, D5, severidad, detalle))
    return hallazgos


def _lineas_mencion_test(codigo: str) -> list[int]:
    """Devuelve las lineas que mencionan el test o el holdout.

    Args:
        codigo: fuente del modulo.

    Returns:
        Lineas ordenadas con menciones de `toxic_test.csv`, `holdout_v2` o
        `holdout`, incluyendo comentarios y cadenas.
    """
    lineas: set[int] = set()
    try:
        flujo = io.StringIO(codigo)
        for token in tokenize.generate_tokens(flujo.readline):
            if token.type not in TIPOS_TOKEN_MENCION:
                continue
            if PATRON_MENCION_TEST.search(token.string):
                lineas.add(token.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return sorted(lineas)
    return sorted(lineas)


def _funcion_contenedora(funciones: list[Funcion], linea: int) -> Funcion | None:
    """Devuelve la funcion mas interna que contiene una linea."""
    candidatas = [
        funcion for funcion in funciones if funcion.linea <= linea <= funcion.fin
    ]
    if not candidatas:
        return None
    return max(candidatas, key=lambda funcion: funcion.linea)


def _tiene_ajuste(funcion: Funcion) -> bool:
    """Indica si la funcion contiene una llamada de ajuste (fit o train)."""
    for nodo in ast.walk(funcion.nodo):
        if not isinstance(nodo, ast.Call):
            continue
        nombre = nombre_llamado(nodo.func)
        if not nombre:
            continue
        if (
            nombre in NOMBRES_FIT_O_TRAIN
            or nombre.startswith("entrenar")
            or nombre.startswith("train_")
        ):
            return True
    return False


def detectar_d6(codigo: str, archivo: str, funciones: list[Funcion]) -> list[Hallazgo]:
    """Detecta menciones al test o al holdout en el codigo.

    Args:
        codigo: fuente del modulo.
        archivo: etiqueta del archivo.
        funciones: definiciones del modulo, para ubicar la funcion contenedora.

    Returns:
        Hallazgos INFO, salvo que la misma funcion contenga fit o train, en
        cuyo caso la severidad es MEDIA.
    """
    hallazgos = []
    for linea in _lineas_mencion_test(codigo):
        contenedora = _funcion_contenedora(funciones, linea)
        con_ajuste = contenedora is not None and _tiene_ajuste(contenedora)
        severidad = "MEDIA" if con_ajuste else "INFO"
        detalle = f"mencion: {texto_linea(codigo, linea)}"
        if contenedora is not None:
            detalle += f"; funcion={contenedora.nombre}"
        if con_ajuste:
            detalle += "; fit_o_train=si"
        hallazgos.append(Hallazgo(archivo, linea, D6, severidad, a_ascii(detalle)))
    return hallazgos


def _usos_de_aleatoriedad(arbol: ast.AST) -> list[int]:
    """Devuelve las lineas que usan aleatoriedad o `random_state`."""
    usos: set[int] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Attribute):
            if (
                nodo.attr == "random"
                and isinstance(nodo.value, ast.Name)
                and nodo.value.id in {"np", "numpy"}
            ):
                usos.add(nodo.lineno)
            elif isinstance(nodo.value, ast.Name) and nodo.value.id == "random":
                usos.add(nodo.lineno)
            elif nodo.attr.startswith("rand") and "torch" in ast.unparse(nodo.value):
                usos.add(nodo.lineno)
        elif isinstance(nodo, ast.Call) and any(
            clave.arg == "random_state" for clave in nodo.keywords
        ):
            usos.add(nodo.lineno)
        elif isinstance(nodo, ast.Name) and nodo.id == "random_state":
            usos.add(nodo.lineno)
    return sorted(usos)


def _hay_semilla(arbol: ast.AST) -> bool:
    """Indica si el modulo contiene alguna llamada a seed."""
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call):
            nombre = nombre_llamado(nodo.func)
            if nombre and "seed" in nombre:
                return True
    return False


def detectar_d7(arbol: ast.AST, archivo: str) -> list[Hallazgo]:
    """Detecta aleatoriedad o `random_state` sin ninguna llamada a seed.

    Args:
        arbol: AST del modulo.
        archivo: etiqueta del archivo.

    Returns:
        Un unico hallazgo BAJA por archivo, en la primera linea de uso.
    """
    usos = _usos_de_aleatoriedad(arbol)
    if not usos or _hay_semilla(arbol):
        return []
    primeras = usos[: CONFIG["max_usos_en_detalle"]]
    detalle = (
        f"sin llamada a seed en el archivo; usos en lineas {primeras} "
        f"(total {len(usos)})"
    )
    return [Hallazgo(archivo, usos[0], D7, "BAJA", detalle)]


def detectar_d8(arbol: ast.AST, codigo: str, archivo: str) -> list[Hallazgo]:
    """Detecta `except:` o `except Exception:` con cuerpo `pass`."""
    hallazgos = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.ExceptHandler):
            continue
        tipo = nodo.type
        desnudo = tipo is None or (
            isinstance(tipo, ast.Name) and tipo.id in {"Exception", "BaseException"}
        )
        if not desnudo:
            continue
        if len(nodo.body) == 1 and isinstance(nodo.body[0], ast.Pass):
            texto = "except:" if tipo is None else f"except {ast.unparse(tipo)}:"
            hallazgos.append(Hallazgo(archivo, nodo.lineno, D8, "BAJA", f"{texto} pass"))
    return hallazgos


# ===== SECCION 5: ESCANEO =====

def escanear_archivo(ruta: Path, base: Path) -> list[Hallazgo]:
    """Ejecuta los ocho detectores sobre un archivo.

    Args:
        ruta: ruta del script.
        base: raiz para etiquetar el archivo.

    Returns:
        Hallazgos sin deduplicar de ese archivo.
    """
    codigo = ruta.read_text(encoding="utf-8")
    arbol = ast.parse(codigo)
    archivo = etiqueta_archivo(ruta, base)
    funciones = recolectar_definiciones(arbol)
    colector = ColectorReferencias()
    colector.visit(arbol)
    hallazgos = []
    hallazgos.extend(detectar_d1(funciones, colector.referencias, archivo))
    hallazgos.extend(detectar_d2(arbol, codigo, archivo))
    hallazgos.extend(detectar_d3(arbol, codigo, archivo))
    hallazgos.extend(detectar_d4(arbol, codigo, archivo))
    hallazgos.extend(detectar_d5(arbol, codigo, archivo, codigo))
    hallazgos.extend(detectar_d6(codigo, archivo, funciones))
    hallazgos.extend(detectar_d7(arbol, archivo))
    hallazgos.extend(detectar_d8(arbol, codigo, archivo))
    return hallazgos


def deduplicar(hallazgos: list[Hallazgo]) -> list[Hallazgo]:
    """Deduplica por (archivo, linea, detector) y ordena la salida.

    Args:
        hallazgos: hallazgos crudos.

    Returns:
        Hallazgos unicos ordenados por severidad, archivo, linea y detector.
    """
    mejores: dict[tuple[str, int, str], Hallazgo] = {}
    for hallazgo in hallazgos:
        clave = (hallazgo.archivo, hallazgo.linea, hallazgo.detector)
        previo = mejores.get(clave)
        if previo is None or ORDEN_SEVERIDAD[hallazgo.severidad] < ORDEN_SEVERIDAD[previo.severidad]:
            mejores[clave] = hallazgo
    return sorted(
        mejores.values(),
        key=lambda hallazgo: (
            ORDEN_SEVERIDAD[hallazgo.severidad],
            hallazgo.archivo,
            hallazgo.linea,
            hallazgo.detector,
        ),
    )


def escanear(directorio: Path, base: Path) -> tuple[list[Hallazgo], dict[str, object]]:
    """Barre todos los scripts de `directorio` y agrega estadisticas.

    Args:
        directorio: raiz del barrido.
        base: raiz para etiquetar archivos.

    Returns:
        Tupla (hallazgos deduplicados, estadisticas del barrido).
    """
    archivos = listar_scripts(directorio)
    hallazgos: list[Hallazgo] = []
    errores: list[tuple[str, int | None]] = []
    for ruta in archivos:
        try:
            hallazgos.extend(escanear_archivo(ruta, base))
        except SyntaxError as error:
            errores.append((etiqueta_archivo(ruta, base), error.lineno))
    limpios = deduplicar(hallazgos)
    estadisticas: dict[str, object] = {
        "archivos_barridos": len(archivos),
        "errores_de_sintaxis": errores,
        "total_hallazgos": len(limpios),
        "por_detector": Counter(hallazgo.detector for hallazgo in limpios),
        "por_severidad": Counter(hallazgo.severidad for hallazgo in limpios),
    }
    return limpios, estadisticas


# ===== SECCION 6: SALIDAS =====

def escribir_csv(ruta: Path, hallazgos: list[Hallazgo]) -> None:
    """Escribe el CSV de hallazgos con encabezado y orden estable.

    Args:
        ruta: destino del CSV.
        hallazgos: hallazgos ya deduplicados.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="ascii", errors="replace", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow(["archivo", "linea", "detector", "severidad", "detalle"])
        for hallazgo in hallazgos:
            escritor.writerow(
                [
                    a_ascii(hallazgo.archivo),
                    hallazgo.linea,
                    hallazgo.detector,
                    hallazgo.severidad,
                    a_ascii(hallazgo.detalle),
                ]
            )


def _tabla_conteo_por_detector(hallazgos: list[Hallazgo]) -> list[str]:
    """Construye la tabla markdown de conteos por detector y severidad."""
    conteo = Counter((hallazgo.detector, hallazgo.severidad) for hallazgo in hallazgos)
    lineas = [
        "| Detector | ALTA | MEDIA | BAJA | INFO | Total |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for detector in DETECTORES:
        celdas = [conteo.get((detector, severidad), 0) for severidad in SEVERIDADES]
        total = sum(celdas)
        lineas.append(
            f"| {detector} | {celdas[0]} | {celdas[1]} | {celdas[2]} | {celdas[3]} | {total} |"
        )
    return lineas


def _tabla_top_archivos(hallazgos: list[Hallazgo]) -> list[str]:
    """Construye la tabla markdown del top 10 por hallazgos ALTA o MEDIA."""
    prioridad = [
        hallazgo for hallazgo in hallazgos if hallazgo.severidad in ("ALTA", "MEDIA")
    ]
    if not prioridad:
        return ["Sin hallazgos de severidad ALTA o MEDIA."]
    por_archivo = Counter(hallazgo.archivo for hallazgo in prioridad)
    altas = Counter(
        hallazgo.archivo for hallazgo in prioridad if hallazgo.severidad == "ALTA"
    )
    orden = sorted(
        por_archivo,
        key=lambda archivo: (-por_archivo[archivo], -altas[archivo], archivo),
    )
    lineas = [
        "| Archivo | ALTA | MEDIA | Total |",
        "| --- | --- | --- | --- |",
    ]
    for archivo in orden[:10]:
        medias = por_archivo[archivo] - altas[archivo]
        lineas.append(f"| {archivo} | {altas[archivo]} | {medias} | {por_archivo[archivo]} |")
    return lineas


def _lista_d1(hallazgos: list[Hallazgo], severidad: str) -> list[str]:
    """Construye la lista `archivo:linea:nombre` de hallazgos D1."""
    seleccion = [
        hallazgo
        for hallazgo in hallazgos
        if hallazgo.detector == D1 and hallazgo.severidad == severidad
    ]
    if not seleccion:
        return ["- Ninguna."]
    return [
        f"- `{hallazgo.archivo}:{hallazgo.linea}:{hallazgo.nombre or '?'}`"
        for hallazgo in seleccion
    ]


def escribir_resumen(
    ruta: Path,
    hallazgos: list[Hallazgo],
    estadisticas: dict[str, object],
) -> None:
    """Escribe el resumen markdown en espanol con acentos.

    Args:
        ruta: destino del markdown.
        hallazgos: hallazgos deduplicados.
        estadisticas: estadisticas devueltas por `escanear`.
    """
    errores = estadisticas["errores_de_sintaxis"]
    lineas = [
        "# Resumen del escaneo est\u00e1tico de experimentos",
        "",
        f"Fecha de referencia: {CONFIG['fecha_escaneo']} (fijada en CONFIG para que "
        "la salida sea determinista).",
        "",
        "## Alcance",
        "",
        f"- Archivos `.py` barridos: **{estadisticas['archivos_barridos']}** "
        "(`experiments/**/*.py`, sin `__pycache__`).",
        f"- Archivos con error de sintaxis: **{len(errores)}**.",
        f"- Hallazgos totales: **{estadisticas['total_hallazgos']}**.",
        "",
        "## Conteo por detector",
        "",
    ]
    lineas.extend(_tabla_conteo_por_detector(hallazgos))
    lineas.extend(
        [
            "",
            "## Top 10 de archivos por hallazgos de severidad ALTA o MEDIA",
            "",
        ]
    )
    lineas.extend(_tabla_top_archivos(hallazgos))
    lineas.extend(
        [
            "",
            "## Funciones D1 sin llamadores reales",
            "",
            "Formato `archivo:l\u00ednea:nombre`. Incluye las funciones sin ning\u00fan "
            "consumidor y las que solo son consumidas desde el modo de humo.",
            "",
            "### Sin ning\u00fan llamador (severidad ALTA)",
            "",
        ]
    )
    lineas.extend(_lista_d1(hallazgos, "ALTA"))
    lineas.extend(
        [
            "",
            "### Solo llamadores de humo (severidad MEDIA)",
            "",
        ]
    )
    lineas.extend(_lista_d1(hallazgos, "MEDIA"))
    lineas.extend(
        [
            "",
            "## Notas de interpretaci\u00f3n",
            "",
            "- D1 excluye `main`, `principal`, dunder, funciones decoradas con "
            "`fixture`, auxiliares de humo (`humo`, `caso`, `smoke`, `test_`) y "
            "ganchos de m\u00e9todo de framework (`forward`, `setUp`, `tearDown`).",
            "- D5 marca como informativo (severidad INFO) el archivo que lee los "
            "umbrales de un artefacto (`leer_umbrales` o un `np.load` de umbrales).",
            "- D6 sube a MEDIA cuando la misma funci\u00f3n menciona el test o el "
            "holdout y adem\u00e1s contiene una llamada de ajuste (`fit` o `train`).",
            "- D7 emite un solo hallazgo por archivo, en la primera l\u00ednea de uso.",
            "- El CSV `escaner_hallazgos.csv` es la fuente de este resumen y est\u00e1 "
            "ordenado por severidad y archivo.",
        ]
    )
    if errores:
        lineas.extend(["", "## Errores de sintaxis", ""])
        lineas.extend(f"- `{archivo}` (l\u00ednea {linea})" for archivo, linea in errores)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="\n") as manejador:
        manejador.write("\n".join(lineas) + "\n")


def _imprimir_consola(estadisticas: dict[str, object]) -> None:
    """Imprime un resumen corto de la corrida en consola."""
    por_detector = estadisticas["por_detector"]
    por_severidad = estadisticas["por_severidad"]
    print(f"Archivos barridos: {estadisticas['archivos_barridos']}")
    print(f"Errores de sintaxis: {len(estadisticas['errores_de_sintaxis'])}")
    print(f"Hallazgos: {estadisticas['total_hallazgos']}")
    print("Por severidad: " + ", ".join(f"{s}={por_severidad.get(s, 0)}" for s in SEVERIDADES))
    for detector in DETECTORES:
        print(f"  {detector}: {por_detector.get(detector, 0)}")


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Escaner estatico (AST) de los scripts de experiments/.",
    )
    parser.add_argument(
        "--directorio",
        type=Path,
        default=CONFIG["directorio_experimentos"],
        help="Directorio a barrer (por defecto experiments/).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CONFIG["ruta_csv"],
        dest="ruta_csv",
        help="Ruta del CSV de hallazgos.",
    )
    parser.add_argument(
        "--resumen",
        type=Path,
        default=CONFIG["ruta_resumen"],
        dest="ruta_resumen",
        help="Ruta del resumen markdown.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del escaner.

    Args:
        argv: argumentos de linea de comandos (None usa sys.argv).

    Returns:
        0 si el escaneo termino, 1 ante cualquier error.
    """
    argumentos = construir_parser().parse_args(argv)
    try:
        hallazgos, estadisticas = escanear(argumentos.directorio, RAIZ_PROYECTO)
        escribir_csv(argumentos.ruta_csv, hallazgos)
        escribir_resumen(argumentos.ruta_resumen, hallazgos, estadisticas)
        _imprimir_consola(estadisticas)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
