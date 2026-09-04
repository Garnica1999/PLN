"""Ensambla el notebook entregable del TALLER-4 a partir de las celdas fuente.

Cada archivo en notebook_cells/secXX_*.py contiene celdas separadas por lineas
que contienen exactamente "# %%" (celdas de codigo) o "# %% [markdown]" (celdas
de markdown cuyo contenido es un docstring triple-comillado, o comentarios
"# "). Las celdas markdown admiten markdown normal: titulos, tablas y figuras
con rutas relativas al notebook (../imagenes/eda_*.png), porque el contenido
se incrusta tal cual en la celda.

Uso (con el venv de Carlos):
    /home/carlos/venv/bin/python scripts/ensamblar_notebook_t4.py
"""
from pathlib import Path

import nbformat as nbf

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CELDAS = RUTA_PROYECTO / "notebook_cells"
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
    "sec01_corpus.py",
    "sec02_eda.py",
    "sec03_ir.py",
    "sec04_observaciones.py",
    "sec05_entrega.py",
]


def extraer_celdas(archivo: Path) -> list[dict]:
    """Extrae las celdas de un archivo fuente.

    Args:
        archivo: Ruta al archivo .py con las celdas separadas por "# %%".

    Returns:
        Lista de dicts {"tipo": "code"|"markdown", "fuente": [lineas]}.

    Raises:
        ValueError: Si el archivo no define ninguna celda o el markdown
            no es un docstring valido.
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
    return celdas


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

    RUTA_SALIDA.parent.mkdir(exist_ok=True)
    nbf.write(nb, RUTA_SALIDA)
    total = len(nb.cells)
    md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"\nNotebook guardado: {RUTA_SALIDA}")
    print(f"Total celdas: {total} ({md} markdown, {total - md} codigo)")


if __name__ == "__main__":
    ensamblar()
