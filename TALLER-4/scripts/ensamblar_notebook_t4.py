r"""Ensambla el notebook entregable del TALLER-4 a partir de las celdas fuente.

Cada archivo en notebook_cells/secXX_*.py contiene celdas separadas por lineas
que contienen exactamente "# %%" (celdas de codigo) o "# %% [markdown]" (celdas
de markdown cuyo contenido es un docstring triple-comillado, o comentarios
"# "). Las celdas markdown admiten markdown normal: titulos, tablas y figuras
con rutas relativas al notebook (../imagenes/eda_*.png), porque el contenido
se incrusta tal cual en la celda.

Marcadores de expansion (celdas de codigo de sec0b_regen.py):
    Una celda code cuya unica linea es
        # %% [expand: scripts/01_extraer_libros.py]
    pide que el codigo canonico del script se EXPANDA EN EL ENSAMBLADO como
    celdas de codigo normales (fuente unica en build: si cambia el script, se
    re-ensambla el notebook y las celdas quedan al dia). El ensamblador
    sustituye la celda marcador por las celdas generadas y NUNCA deja
    codigo-como-string ejecutado con exec():

    1. Preambulo en la primera celda del bloque: un comentario [auto] y un
       '__file__' sintetico (str(RUTA_RAIZ / "scripts" / <nombre>)) para que
       'parents[1]' del script resuelva a la raiz del proyecto detectada por
       las celdas de sec00/sec0b. Si la celda empieza con un import
       'from __future__', la asignacion de '__file__' se inserta DESPUES de
       el (un 'from __future__' debe ser la primera sentencia del archivo o
       celda: solo admiten delante docstring y comentarios).
    2. Se elimina el bloque 'if __name__ == "__main__":' desde su primera
       linea hasta el final del archivo: en el kernel de Jupyter
       '__name__' == "__main__" en las celdas y el guard ejecutaria el
       pipeline completo en Run all (los scripts terminan todos con el guard).
    3. Particion en celdas por banners de seccion: lineas completas cuyo
       primer caracter no-espaciado tras '#' son >=3 guiones (patron
        '^#\s*-{3,}'), detectadas con tokenize (el escaner de estado del
        propio Python) para NO partir dentro de docstrings triples (de
        comillas simples o dobles); una celda nueva EMPIEZA en la linea del
        banner. Los scripts sin banners quedan en una sola celda.
    4. Celda final de alias tras el bloque de cada script: 'main_extraccion',
       'main_preprocesar' o 'main_eda' (evita que la definicion de 'main' de
       un script pise a la del anterior); para 03_eda.py ademas restaura el
       backend inline de matplotlib (el script fija "Agg" a nivel de modulo).

Uso (con el venv de Carlos):
    /home/carlos/venv/bin/python scripts/ensamblar_notebook_t4.py
"""
from __future__ import annotations

import io
import re
import tokenize
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
    "sec03b_metricas.py",
    "sec03c_comparacion.py",
    "sec04_observaciones.py",
    "sec05_entrega.py",
]

# Marcador de expansion: celda code cuya unica linea es
#   # %% [expand: ruta/relativa/al/script.py]
PREFIJO_EXPAND = "# %% [expand:"
_REGEX_MARCA_EXPAND = re.compile(r"^# %% \[expand:\s*([^\]]+?)\s*\]\s*$")

# Marcador antiguo de embebido (v3): prohibido dejar restos en el notebook.
PREFIJO_EMBED = "# [embed:"
PREFIJOS_MARCADOR = (PREFIJO_EXPAND, PREFIJO_EMBED)

# Bloque guard 'if __name__ == "__main__":' desde su primera linea hasta EOF.
_REGEX_GUARD_MAIN = re.compile(
    r'^if __name__ == ["\']__main__["\']:.*', re.MULTILINE | re.DOTALL
)

# Banner de seccion: comentario cuyo primer caracter tras '#' y espacios son
# >=3 guiones seguidos (los scripts 01 y 03 separan asi sus secciones).
_REGEX_BANNER = re.compile(r"^#\s*-{3,}")

# Celdas de cierre (alias [auto]) por script expandido. 'main' colisiona
# entre scripts en el mismo kernel: cada bloque termina con su alias.
_CIERRES_SCRIPT: dict[str, str] = {
    "01_extraer_libros.py": (
        "main_extraccion = main  # [auto] alias (evita colision de 'main' entre scripts)"
    ),
    "02_preprocesar.py": "main_preprocesar = main  # [auto] alias",
    "03_eda.py": (
        "main_eda = main  # [auto] alias\n"
        "# [auto] restaurar backend inline tras la expansion de 03_eda.py\n"
        "try:\n"
        "    import matplotlib\n"
        '    matplotlib.use("module://matplotlib_inline.backend_inline")\n'
        "except Exception:\n"
        "    pass"
    ),
}


def _lineas_banner(contenido: str) -> list[int]:
    r"""Detecta las lineas de banner de seccion fuera de strings/docstrings.

    Delega el estado de docstrings triples (de comillas simples o dobles, con
    escapes) a tokenize, el escaner de estado del propio Python: solo los
    tokens COMMENT
    pueden ser banners, asi que una linea '#' dentro de un docstring jamas se
    cuenta como banner. Una linea es banner si el comentario completo matchea
    '^#\s*-{3,}' (comienza tras '#' y espacios con >=3 guiones seguidos).

    Args:
        contenido: Modulo Python completo (docstring y todo).

    Returns:
        Numeros de linea 1-based donde empieza un banner (orden de lectura).

    Raises:
        ValueError: Si tokenize no puede escanear el contenido.
    """
    numeros: list[int] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(contenido).readline):
            if token.type == tokenize.COMMENT and _REGEX_BANNER.match(token.string):
                numeros.append(token.start[0])
    except (tokenize.TokenError, IndentationError) as exc:
        raise ValueError(f"No se pudieron detectar los banners (tokenize): {exc}") from exc
    return numeros


def _aplicar_preambulo(texto: str, nombre_script: str) -> str:
    """Antepone a la primera celda expandida el comentario [auto] y el
    '__file__' sintetico.

    El '__file__' apunta a RUTA_RAIZ / "scripts" / <nombre_script>: RUTA_RAIZ
    es la raiz del proyecto detectada por las celdas de sec00/sec0b, de modo
    que Path(__file__).resolve().parents[1] del script resuelve a esa raiz.

    Args:
        texto: Contenido de la primera celda expandida (bloque inicial).
        nombre_script: Nombre del script expandido (ej. 01_extraer_libros.py).

    Returns:
        La celda con el preambulo antepuesto.
    """
    comentario = (
        "# [auto] Codigo expandido desde "
        f"scripts/{nombre_script} (ensamblador; fuente unica en build)."
    )
    asignacion = (
        f'__file__ = str(RUTA_RAIZ / "scripts" / "{nombre_script}")  '
        "# sintetico: parents[1] == raiz del proyecto"
    )
    resto = texto.splitlines()
    # Un 'from __future__' debe ser la primera sentencia (solo docstring y
    # comentarios pueden precederlo): si existe, la asignacion va despues de el.
    indice_future = next(
        (i for i, linea in enumerate(resto) if linea.startswith("from __future__ import")),
        None,
    )
    if indice_future is None:
        return "\n".join([comentario, asignacion, *resto])
    return "\n".join([comentario, *resto[: indice_future + 1], asignacion, *resto[indice_future + 1 :]])


def _celda_cierre_script(nombre_script: str) -> str:
    """Devuelve la celda code de cierre (alias [auto]) del script expandido.

    Args:
        nombre_script: Nombre del script expandido.

    Returns:
        Texto de la celda de cierre.

    Raises:
        ValueError: Si no hay cierre definido para el script (no expandible).
    """
    try:
        return _CIERRES_SCRIPT[nombre_script]
    except KeyError as exc:
        raise ValueError(f"Sin celda de cierre definida para expandir {nombre_script}") from exc


def _celdas_desde_script(ruta_relativa: str, nombre_archivo: str) -> list[dict]:
    """Expande un script en celdas code normales (transformaciones 1-4).

    Lee el script, elimina su guard '__main__', lo parte por banners de
    seccion (tokenize, nunca dentro de docstrings) y antepone el preambulo
    con '__file__' sintetico en la primera celda; cierra el bloque con la
    celda de alias del script.

    Args:
        ruta_relativa: Ruta del script relativa a la raiz del proyecto.
        nombre_archivo: Nombre del archivo de celdas (solo para mensajes).

    Returns:
        Celdas {"tipo": "code", "fuente": texto} del bloque expandido.

    Raises:
        ValueError: Si la ruta no existe, no se puede escanear o el script
            no tiene celda de cierre definida.
    """
    ruta = RUTA_PROYECTO / ruta_relativa
    if not ruta.exists():
        raise ValueError(
            f"[expand] en {nombre_archivo}: no existe {ruta_relativa} "
            f"(buscado en {ruta})"
        )
    nombre_script = ruta.name
    sin_guard = _REGEX_GUARD_MAIN.sub("", ruta.read_text(encoding="utf-8"))
    banners = _lineas_banner(sin_guard)
    lineas = sin_guard.splitlines()

    bloques: list[list[str]] = []
    inicio = 0
    for numero_linea in banners:
        indice = numero_linea - 1
        bloques.append(lineas[inicio:indice])
        inicio = indice
    bloques.append(lineas[inicio:])

    celdas: list[dict] = []
    primera_emitida = False
    for bloque in bloques:
        texto = "\n".join(bloque).strip()
        if not texto:
            continue
        if not primera_emitida:
            texto = _aplicar_preambulo(texto, nombre_script)
            primera_emitida = True
        celdas.append({"tipo": "code", "fuente": texto})
    celdas.append({"tipo": "code", "fuente": _celda_cierre_script(nombre_script)})
    return celdas


def _resolver_expansiones(celdas: list[dict], nombre_archivo: str) -> list[dict]:
    """Expande las celdas marcadas '# %% [expand: <ruta>]', in situ.

    Args:
        celdas: Celdas {"tipo", "fuente"} extraidas del archivo.
        nombre_archivo: Nombre del archivo de celdas (solo para mensajes).

    Returns:
        Lista con las celdas originales salvo las marcadas, reemplazadas por
        las celdas code expandidas del script indicado.

    Raises:
        ValueError: Si la ruta del marcador no existe (via _celdas_desde_script).
    """
    salida: list[dict] = []
    n_expandidas = 0
    for celda in celdas:
        if celda["tipo"] != "code":
            salida.append(celda)
            continue
        coincidencia = _REGEX_MARCA_EXPAND.match(celda["fuente"].strip())
        if coincidencia is None:
            salida.append(celda)
            continue
        salida.extend(
            _celdas_desde_script(coincidencia.group(1).strip(), nombre_archivo)
        )
        n_expandidas += 1
    if n_expandidas:
        print(f"[expand] {nombre_archivo}: {n_expandidas} marcador(es) expandido(s)")
    return salida


def extraer_celdas(archivo: Path) -> list[dict]:
    """Extrae las celdas de un archivo fuente y resuelve sus expansiones.

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

    return _resolver_expansiones(celdas, archivo.name)


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
            if any(prefijo in linea for prefijo in PREFIJOS_MARCADOR):
                raise ValueError(
                    "Marcador sin resolver en el notebook: " + linea.strip()
                )


def verificar_sin_codigo_como_string(nb: nbf.NotebookNode) -> None:
    """Comprueba que el notebook no conserve codigo-como-string con exec().

    La v4 expande los scripts en celdas de codigo normales: no debe quedar
    ninguna variable CODIGO_EMBED_* ni llamadas 'exec(' en el ensamblado.

    Args:
        nb: Notebook ya construido.

    Raises:
        ValueError: Si alguna celda contiene CODIGO_EMBED_ o 'exec('.
    """
    for celda in nb.cells:
        fuente = celda.source if isinstance(celda.source, str) else "".join(celda.source)
        if "CODIGO_EMBED_" in fuente or "exec(" in fuente:
            raise ValueError(
                f"Queda codigo-como-string o exec( en la celda "
                f"{celda.cell_type}: {fuente[:80]!r}"
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
    verificar_sin_codigo_como_string(nb)

    RUTA_SALIDA.parent.mkdir(exist_ok=True)
    nbf.write(nb, RUTA_SALIDA)
    total = len(nb.cells)
    md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"\nNotebook guardado: {RUTA_SALIDA}")
    print(f"Total celdas: {total} ({md} markdown, {total - md} codigo)")


if __name__ == "__main__":
    ensamblar()
