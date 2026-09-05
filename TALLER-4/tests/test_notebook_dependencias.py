"""test_notebook_dependencias.py

Fecha: 2026-09-05
Descripcion: Tests de independencia y robustez del notebook entregable
    (notebooks/Taller_4_IR_CEV.ipynb, ensamblado desde notebook_cells/):

    (a) ninguna celda code importa scripts/ ni experimentos/ (ni usa
        importlib / sys.path+append hacia ellos); los docstrings pueden
        MENCIONAR la independencia ("no usa importlib..."), por eso el test
        detecta uso real (import_module/spec_from_file/append) y no la
        mera palabra.
    (b) ninguna celda markdown referencia ../imagenes/ (no renderiza en
        GitHub/Colab; las figuras van inline o via mostrar_png).
    (c) todo literal de archivo (.jsonl/.txt/.csv/.json/.png) citado entre
        comillas en celdas code es un archivo versionado en git o esta en la
        lista blanca de archivos >100 MB con guard .exists() en la misma
        celda. Lista blanca (crece documentada): el JSON crudo de entrevistas
        y las normalizadas/tokens (generados por scripts/02; >100 MB, no
        versionables). Se ignoran literales con interpolacion (llaves) y con
        espacios (nombres descriptivos).
    (d) las celdas que resuelven la raiz (def buscar_raiz / _raiz_standalone)
        lanzan FileNotFoundError con el marcador segmentos_libros.jsonl y no
        contienen la asignacion silenciosa del ultimo candidato.
    (e) en el ipynb no queda ningun marcador de embebido sin resolver y si
        aparece el codigo embebido (CODIGO_EMBED) con REGEN_AUTOMATICO=False.
    (f) el ipynb contiene la celda de diagnostico de prerrequisitos.

El notebook debe estar ensamblado (scripts/ensamblar_notebook_t4.py); si no
existe, el modulo se omite con pytest.skip.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RUTA_PROYECTO / "notebooks" / "Taller_4_IR_CEV.ipynb"

# Archivos >100 MB no versionados con respaldo estatico o regenerables;
# cualquier literal de estos exige guard .exists() en la misma celda.
LISTA_BLANCA = {
    "entrevistas_all_2023-03-21_14-24_05.json",  # JSON crudo (Drive del taller)
    "entrevistas_normalizadas.jsonl",  # genera scripts/02 normalizar
    "entrevistas_tokens.txt",  # genera scripts/02 spacy (191 MB)
    "paginas.jsonl",  # intermedio de scripts/01 en data/corpus/paginas/ (.gitignore)
}

_PATRON_LITERAL = re.compile(r"""(['"])([^'"\n]*?\.(?:jsonl|txt|csv|json|png))\1""")
_PATRON_IMPORT_RELATIVO = re.compile(
    r"^\s*(?:import|from)\s+(?:scripts|experimentos)(?:\s|\.|$)", re.MULTILINE
)

EXTENSIONES = (".jsonl", ".txt", ".csv", ".json", ".png")


def _celdas_code() -> list[str]:
    """Devuelve el source de todas las celdas code del notebook.

    Returns:
        Lista con el texto de cada celda code.

    Raises:
        pytest.skip: si el notebook no existe (falta ensamblar).
    """
    if not RUTA_NOTEBOOK.exists():
        pytest.skip(
            f"No existe {RUTA_NOTEBOOK}: ejecuta primero "
            "scripts/ensamblar_notebook_t4.py"
        )
    with RUTA_NOTEBOOK.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    return [
        "".join(celda["source"])
        for celda in nb["cells"]
        if celda["cell_type"] == "code"
    ]


def _celdas_markdown() -> list[str]:
    """Devuelve el source de todas las celdas markdown del notebook."""
    with RUTA_NOTEBOOK.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    return [
        "".join(celda["source"])
        for celda in nb["cells"]
        if celda["cell_type"] == "markdown"
    ]


def _archivos_versionados() -> tuple[set[str], set[str]]:
    """Consulta git ls-files y devuelve basenames y rutas relativas a TALLER-4.

    Returns:
        Tupla (basenames, rutas sin el prefijo TALLER-4/) de archivos
        versionados en el repositorio.
    """
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=RUTA_PROYECTO,
        capture_output=True,
        text=True,
        check=False,
    )
    rutas = [linea for linea in proc.stdout.splitlines() if linea.strip()]
    basenames = {Path(ruta).name for ruta in rutas}
    relativas = {ruta.removeprefix("TALLER-4/") for ruta in rutas}
    return basenames, relativas


def test_a_sin_imports_de_scripts_ni_experimentos() -> None:
    """(a) Ninguna celda code importa scripts/, experimentos/, importlib o
    sys.path con append hacia esas carpetas (uso real, no menciones)."""
    for celda in _celdas_code():
        assert not _PATRON_IMPORT_RELATIVO.search(celda), (
            "import relativo de scripts/experimentos en una celda code"
        )
        if "sys.path" in celda:
            for mutacion in (
                "sys.path.append",
                "sys.path.insert",
                "sys.path.extend",
                "sys.path[0:0]",
            ):
                assert mutacion not in celda, f"{mutacion} presente en una celda code"
        if "importlib" in celda:
            for uso in ("import_module", "spec_from_file", "exec_module"):
                assert uso not in celda, f"importlib.{uso} presente en una celda code"


def test_b_sin_referencias_imagenes_relativas() -> None:
    """(b) Ninguna celda markdown referencia ../imagenes/."""
    for celda in _celdas_markdown():
        assert "../imagenes/" not in celda, "referencia ../imagenes/ en markdown"


def test_c_literales_de_archivo_versionados_o_lista_blanca() -> None:
    """(c) Literales de archivo en celdas code: versionados o lista blanca
    con guard .exists() en la misma celda."""
    basenames, rutas_relativas = _archivos_versionados()
    for celda in _celdas_code():
        for literal in _PATRON_LITERAL.findall(celda):
            candidato = literal[1]
            if (
                not candidato
                or "{" in candidato
                or "}" in candidato
                or "$" in candidato
                or " " in candidato
                or candidato.startswith(".")
                or not candidato.endswith(EXTENSIONES)
            ):
                continue  # interpolacion o no es un archivo
            nombre = Path(candidato).name
            versionado = nombre in basenames or candidato in rutas_relativas
            if versionado:
                continue
            assert nombre in LISTA_BLANCA, (
                f"literal de archivo no versionado y fuera de la lista blanca: "
                f"{candidato!r}"
            )
            assert ".exists()" in celda, (
                f"literal en lista blanca {candidato!r} sin guard .exists() "
                "en la misma celda"
            )


def test_d_resolucion_de_raiz_con_error_claro() -> None:
    """(d) Las celdas que resuelven la raiz lanzan FileNotFoundError claro y
    no asignan silenciosamente el ultimo candidato sin verificar."""
    celdas_resolutoras = [
        celda
        for celda in _celdas_code()
        if "def buscar_raiz" in celda or "def _raiz_standalone" in celda
    ]
    assert celdas_resolutoras, "no se encontro ninguna celda que defina buscar_raiz"
    for celda in celdas_resolutoras:
        assert "FileNotFoundError" in celda, "falta FileNotFoundError en la celda"
        assert "segmentos_libros.jsonl" in celda, "falta el marcador en la celda"
        assert "raise FileNotFoundError" in celda, "la celda debe lanzar el error"
    for celda in _celdas_code():
        assert "RUTA_RAIZ = _candidato" not in celda, (
            "asignacion silenciosa del ultimo candidato presente en una celda"
        )


def test_e_marcadores_resueltos_y_flags_apagados() -> None:
    """(e) Sin marcadores [embed: sin resolver; CODIGO_EMBED presente y
    REGEN_AUTOMATICO en False."""
    todas = _celdas_code() + _celdas_markdown()
    assert not any("[embed:" in celda for celda in todas), (
        "quedo un marcador de embebido sin resolver en el notebook"
    )
    assert any("CODIGO_EMBED" in celda for celda in todas), (
        "no aparece el codigo embebido (CODIGO_EMBED) en el notebook"
    )
    assert any(
        re.search(r"REGEN_AUTOMATICO\s*=\s*False", celda) for celda in todas
    ), "REGEN_AUTOMATICO debe estar definido en False"


def test_f_celda_de_diagnostico_de_prerrequisitos() -> None:
    """(f) El notebook contiene la celda de diagnostico de prerrequisitos."""
    assert any("prerrequisitos" in celda for celda in _celdas_code()), (
        "no aparece la celda de diagnostico de prerrequisitos"
    )
