"""Ensambla el notebook entregable del TALLER-4 a partir de las celdas fuente.

Cada archivo en notebook_cells/secXX_*.py contiene celdas separadas por lineas
que contienen exactamente "# %%" (celdas de codigo) o "# %% [markdown]" (celdas
de markdown cuyo contenido es un docstring triple-comillado, o comentarios
"# "). Las celdas markdown admiten markdown normal: titulos, tablas y figuras
con rutas relativas al notebook (../imagenes/eda_*.png), porque el contenido
se incrusta tal cual en la celda.

Marcadores de embebido (celdas de codigo de sec0b_regen.py):
    Una linea de la forma
        CODIGO_EMBED_XX = None  # [embed: scripts/01_extraer_libros.py]
    pide que el codigo canonico del script se incruste EN EL ENSAMBLADO como
    un literal de cadena (fuente unica en build: si cambia el script, se
    re-ensambla el notebook y la celda queda al dia). El ensamblador
    reemplaza la linea completa por
        CODIGO_EMBED_XX = r'''<contenido del script>'''
    (literal r-triple si el contenido no contiene comillas simples triples;
    repr() como respaldo seguro). El literal se define con el MISMO nombre de
    variable de la linea fuente y el contenido NUNCA se ejecuta aqui.

Uso (con el venv de Carlos):
    /home/carlos/venv/bin/python scripts/ensamblar_notebook_t4.py
"""
from pathlib import Path

import nbformat as nbf

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CELDAS = RUTA_PROYECTO / "notebook_cells"
RUTA_SCRIPTS = RUTA_PROYECTO / "scripts"
RUTA_SALIDA = RUTA_PROYECTO / "notebooks" / "Taller_4_IR_CEV.ipynb"

MARCA_CODIGO = "# %%"
MARCA_MARKDOWN = "# %% [markdown]"

PORTADA = """<img src="https://www.javeriana.edu.co/recursosdb/20125/5571273/escudo-circular.png" alt="Escudo" style="float:left;width:70px;">

<div align="center">

# Pontificia Universidad Javeriana

## Facultad de Ingeniería

### Curso: Procesamiento de Lenguaje Natural

# TALLER-4 — Recuperación de Información aplicada a entrevistas y libros de la Comisión de la Verdad (CEV)

**Integrantes**: Erika Valeria Rivera Jiménez — Julián Guillermo Carrillo Meneses — Carlos Andrés Garnica Salazar — Oscar Andrés Prieto Moreno

**Septiembre de 2026**

---

</div>
"""

SECUENCIA = [
    "sec00_setup.py",
    "sec0b_regen.py",
    "sec01_corpus.py",
    "sec02_eda.py",
    "sec03_ir.py",
    "sec04_observaciones.py",
    "sec05_entrega.py",
]

# Patron de linea marcador:  NOMBRE_VAR = None  # [embed: ruta/relativa]
PREFIJO_MARCADOR = "# [embed:"
_REGEX_MARCADOR = __import__("re").compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*None\s*#\s*\[embed:\s*([^\]]+)\]\s*$"
)


def _literal_embebido(contenido: str) -> str:
    """Convierte el contenido de un script en un literal Python valido.

    Se prefiere el literal crudo de comillas simples triples (legible, sin
    re-escape de \\n); si el contenido contiene comillas simples triples o
    termina en backslash se usa repr() (cadena de una linea, sin ambiguedad).

    Args:
        contenido: Texto completo del script (UTF-8).

    Returns:
        Literal Python que reconstruye exactamente el texto del script.
    """
    if "'''" not in contenido and not contenido.endswith("\\"):
        return "r'''" + contenido + "'''"
    return repr(contenido)


def _resolver_marcadores(celda: dict, nombre_archivo: str) -> int:
    """Resuelve los marcadores de embebido de una celda de codigo, in situ.

    Por cada linea `VAR = None  # [embed: <ruta>]` lee el archivo indicado
    (relativo a la raiz del proyecto) y reemplaza la linea por
    `VAR = <literal con el contenido>`.

    Args:
        celda: Dict {"tipo", "fuente"} de la celda (se modifica "fuente").
        nombre_archivo: Nombre del archivo de celdas (solo para mensajes).

    Returns:
        Numero de marcadores resueltos.

    Raises:
        ValueError: Si la ruta del marcador no existe o el archivo no se lee.
    """
    if celda["tipo"] != "code":
        return 0
    lineas = celda["fuente"].splitlines()
    salida: list[str] = []
    n_resueltos = 0
    for linea in lineas:
        coincidencia = _REGEX_MARCADOR.match(linea)
        if coincidencia is None:
            salida.append(linea)
            continue
        nombre_var, ruta_script = coincidencia.group(1), coincidencia.group(2).strip()
        ruta_script = ruta_script.strip()
        ruta = RUTA_PROYECTO / ruta_script
        if not ruta.exists():
            raise ValueError(
                f"[embed] en {nombre_archivo}: no existe {ruta_script} "
                f"(buscado en {ruta})"
            )
        contenido = ruta.read_text(encoding="utf-8")
        salida.append(f"{nombre_var} = {_literal_embebido(contenido)}")
        n_resueltos += 1
    celda["fuente"] = "\n".join(salida)
    return n_resueltos


def extraer_celdas(archivo: Path) -> list[dict]:
    """Extrae las celdas de un archivo fuente y resuelve sus marcadores.

    Args:
        archivo: Ruta al archivo .py con las celdas separadas por "# %%".

    Returns:
        Lista de dicts {"tipo": "code"|"markdown", "fuente": [lineas]}.

    Raises:
        ValueError: Si el archivo no define ninguna celda, el markdown no es
            un docstring valido o queda un marcador sin resolver.
    """
    lineas = archivo.read_text(encoding="utf-8").splitlines()
    celdas: list[dict] = []
    actual: list[str] = []
    tipo_actual: str | None = None

    def guardar() -> None:
        nonlocal actual, tipo_actual
        if tipo_actual is None:
            return
        contenido = "\n".join(actual).strip()
        if tipo_actual == "markdown":
            if contenido.startswith('"""'):
                # Formato docstring triple-comillado.
                if contenido.endswith('"""'):
                    contenido = contenido[3:-3].strip()
                elif '"""' in contenido[3:]:
                    cierre = contenido[3:].index('"""') + 3
                    contenido = contenido[3:cierre].strip()
                else:
                    raise ValueError(
                        f"Celda markdown invalida en {archivo.name}: "
                        "docstring sin cierre"
                    )
            elif all(linea.startswith("#") for linea in contenido.splitlines()):
                # Formato comentarios "# " (linea a linea).
                lineas_md = []
                for linea in contenido.splitlines():
                    linea = linea.removeprefix("#").removeprefix(" ")
                    lineas_md.append(linea)
                contenido = "\n".join(lineas_md).strip()
            else:
                raise ValueError(
                    f"Celda markdown invalida en {archivo.name}: "
                    "ni docstring ni comentarios '# '"
                )
        if contenido:
            celdas.append({"tipo": tipo_actual, "fuente": contenido})
        actual, tipo_actual = [], None

    for linea in lineas:
        if linea.strip() == MARCA_MARKDOWN:
            guardar()
            tipo_actual = "markdown"
            continue
        if linea.strip() == MARCA_CODIGO:
            guardar()
            tipo_actual = "code"
            continue
        if tipo_actual is not None:
            actual.append(linea)

    guardar()
    if not celdas:
        raise ValueError(f"{archivo.name} no contiene celdas validas")

    n_marcadores = 0
    for celda in celdas:
        n_marcadores += _resolver_marcadores(celda, archivo.name)
    if n_marcadores:
        print(f"[embed] {archivo.name}: {n_marcadores} marcador(es) resueltos")
    return celdas


def verificar_sin_marcadores_pendientes(nb: nbf.NotebookNode) -> None:
    """Comprueba que ninguna celda del notebook conserve un marcador sin
    resolver (error de ensamblado si el patron aparece en alguna linea).

    Args:
        nb: Notebook ya construido.

    Raises:
        ValueError: Si alguna celda contiene el texto de un marcador.
    """
    for celda in nb.cells:
        if celda.cell_type != "code":
            continue
        for linea in celda.source.splitlines():
            if PREFIJO_MARCADOR in linea:
                raise ValueError(
                    "Marcador de embebido sin resolver en el notebook: "
                    + linea.strip()
                )


def ensamblar() -> None:
    """Construye el notebook entregable y lo guarda en notebooks/."""
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
        "colab": {"provenance": []},
    }
    nb.cells.append(nbf.v4.new_markdown_cell(PORTADA))
    for nombre in SECUENCIA:
        archivo = RUTA_CELDAS / nombre
        if not archivo.exists():
            print(f"[AVISO] No existe {archivo.name}; se omite.")
            continue
        celdas = extraer_celdas(archivo)
        for celda in celdas:
            if celda["tipo"] == "markdown":
                nb.cells.append(nbf.v4.new_markdown_cell(celda["fuente"]))
            else:
                nb.cells.append(nbf.v4.new_code_cell(celda["fuente"]))
        print(f"[OK] {nombre}: {len(celdas)} celdas anadidas")

    verificar_sin_marcadores_pendientes(nb)

    RUTA_SALIDA.parent.mkdir(exist_ok=True)
    nbf.write(nb, RUTA_SALIDA)
    total = len(nb.cells)
    md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"\nNotebook guardado: {RUTA_SALIDA}")
    print(f"Total celdas: {total} ({md} markdown, {total - md} codigo)")


if __name__ == "__main__":
    ensamblar()
